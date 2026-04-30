"""Direct loader for ``IWFMModel.from_preprocessor``.

In v1.x this body lived as a classmethod in ``core/model.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pyiwfm.core.loaders._common import (
    _COMPONENT_LOAD_EXCEPTIONS,
    StrictMode,
    _finalize_collected_errors,
    _record_component_failure,
)

if TYPE_CHECKING:
    from pyiwfm.core.model import IWFMModel


def load_from_preprocessor(
    pp_file: Path | str,
    load_streams: bool = True,
    load_lakes: bool = True,
    *,
    strict: StrictMode = False,
) -> IWFMModel:
    """
    Load a model from PreProcessor input files.

    This loads the model structure (mesh, stratigraphy) and optionally
    the stream and lake geometry from the preprocessor input file and
    all files referenced by it.

    Note: This creates a "partial" model with only the static geometry
    defined in the preprocessor. It does not include dynamic components
    like groundwater parameters, pumping, or root zone data which are
    defined in the simulation input files.

    Args:
        pp_file: Path to the main PreProcessor input file
        load_streams: If True, load stream network geometry
        load_lakes: If True, load lake geometry
        strict: Behaviour on component-load failure. Three values:

            - ``False`` (default): log a warning, record the error in
              ``model.metadata`` (introspect via
              :attr:`IWFMModel.load_errors`), and continue. Returns a
              partially-loaded model.
            - ``True``: raise
              :class:`~pyiwfm.core.exceptions.ComponentLoadError` on
              the first component that fails. Best for calibration /
              analysis pipelines that need a complete model and want
              fail-fast.
            - ``"collect"``: load every component, then raise a single
              :class:`~pyiwfm.core.exceptions.ValidationError` at the
              end if any failed. Best for user-facing surfaces (CLI,
              web API) — users see all problems in one report.

    Returns:
        IWFMModel instance with mesh, stratigraphy, and optionally
        streams/lakes geometry loaded

    Example:
        >>> model = load_from_preprocessor("Preprocessor/Preprocessor.in")
        >>> print(f"Loaded {model.n_nodes} nodes, {model.n_elements} elements")
    """
    from pyiwfm.core.mesh import AppGrid, Subregion
    from pyiwfm.core.model import IWFMModel
    from pyiwfm.core.model_factory import (
        build_reaches_from_node_reach_ids as _build_reaches_from_node_reach_ids,
    )
    from pyiwfm.core.model_factory import (
        resolve_stream_node_coordinates as _resolve_stream_node_coordinates,
    )
    from pyiwfm.io.preprocessor import (
        read_preprocessor_main,
        read_subregions_file,
    )
    from pyiwfm.io.preprocessor.mesh import read_elements, read_nodes, read_stratigraphy

    pp_file = Path(pp_file)
    config = read_preprocessor_main(pp_file)

    # Read nodes
    if config.nodes_file is None:
        from pyiwfm.core.exceptions import FileFormatError

        raise FileFormatError("Nodes file not specified in PreProcessor file")
    nodes = read_nodes(config.nodes_file)

    # Read elements
    if config.elements_file is None:
        from pyiwfm.core.exceptions import FileFormatError

        raise FileFormatError("Elements file not specified in PreProcessor file")
    elements, n_subregions, subregion_names = read_elements(config.elements_file)

    # Read subregions: prefer separate file, fall back to names from element file
    subregions: dict[int, Subregion] = {}
    if config.subregions_file and config.subregions_file.exists():
        subregions = read_subregions_file(config.subregions_file)
    elif subregion_names:
        subregions = {
            sr_id: Subregion(id=sr_id, name=name) for sr_id, name in subregion_names.items()
        }

    # Create mesh
    mesh = AppGrid(nodes=nodes, elements=elements, subregions=subregions)
    mesh.compute_areas()
    mesh.compute_connectivity()

    # Read stratigraphy
    stratigraphy = None
    if config.stratigraphy_file and config.stratigraphy_file.exists():
        stratigraphy = read_stratigraphy(config.stratigraphy_file)

    # Create model
    model = IWFMModel(
        name=config.model_name or pp_file.stem,
        mesh=mesh,
        stratigraphy=stratigraphy,
        metadata={
            "source": "preprocessor",
            "preprocessor_file": str(pp_file),
            "length_unit": config.length_unit,
            "area_unit": config.area_unit,
            "volume_unit": config.volume_unit,
        },
    )

    # Load stream geometry if requested
    if load_streams and config.streams_file and config.streams_file.exists():
        try:
            from pyiwfm.components.stream import AppStream, StrmNode, StrmReach
            from pyiwfm.io.streams import StreamReader, StreamSpecReader

            stream = AppStream()
            first_error: Exception | None = None

            # Try simple stream-nodes format first
            try:
                reader = StreamReader()
                nodes_dict = reader.read_stream_nodes(config.streams_file)
                for node in nodes_dict.values():
                    stream.add_node(node)
            except Exception as exc:
                first_error = exc

            # Fallback: parse as StreamsSpec (reach-based format)
            if not stream.nodes:
                try:
                    spec_reader = StreamSpecReader()
                    n_reaches, _n_rt, reach_specs = spec_reader.read(config.streams_file)
                    for rs in reach_specs:
                        for sn_id in rs.node_ids:
                            if sn_id not in stream.nodes:
                                gw = rs.node_to_gw_node.get(sn_id)
                                node = StrmNode(
                                    id=sn_id,
                                    x=0.0,
                                    y=0.0,
                                    reach_id=rs.id,
                                    gw_node=gw if gw and gw > 0 else None,
                                )
                                # Transfer bottom elevation from rating table section
                                if sn_id in rs.node_bottom_elevations:
                                    node.bottom_elev = rs.node_bottom_elevations[sn_id]
                                # Transfer rating table
                                if sn_id in rs.node_rating_tables:
                                    import numpy as np

                                    from pyiwfm.components.stream import StreamRating

                                    stages, flows = rs.node_rating_tables[sn_id]
                                    node.rating = StreamRating(
                                        stages=np.array(stages, dtype=np.float64),
                                        flows=np.array(flows, dtype=np.float64),
                                    )
                                stream.add_node(node)
                        stream.add_reach(
                            StrmReach(
                                id=rs.id,
                                upstream_node=rs.node_ids[0] if rs.node_ids else 0,
                                downstream_node=rs.node_ids[-1] if rs.node_ids else 0,
                                nodes=list(rs.node_ids),
                                name=rs.name,
                            )
                        )
                except Exception:
                    # Both paths failed — re-raise original error
                    if first_error is not None:
                        raise first_error from None

            # Safety net: build reaches from node reach_ids
            _build_reaches_from_node_reach_ids(stream)

            model.streams = stream
        except _COMPONENT_LOAD_EXCEPTIONS as e:
            _record_component_failure(model, "streams", config.streams_file, e, strict=strict)

    # Load lake geometry if requested
    if load_lakes and config.lakes_file and config.lakes_file.exists():
        try:
            from pyiwfm.components.lake import AppLake, LakeElement
            from pyiwfm.io.lakes import LakeReader

            lake_reader = LakeReader()
            lakes_dict = lake_reader.read_lake_definitions(config.lakes_file)

            lakes = AppLake()
            for lake in lakes_dict.values():
                lakes.add_lake(lake)
                for elem_id in lake.elements:
                    lakes.add_lake_element(LakeElement(element_id=elem_id, lake_id=lake.id))

            model.lakes = lakes
        except _COMPONENT_LOAD_EXCEPTIONS as e:
            _record_component_failure(model, "lakes", config.lakes_file, e, strict=strict)

    model.metadata["source"] = "preprocessor"
    _resolve_stream_node_coordinates(model)
    _finalize_collected_errors(model, strict)
    return model
