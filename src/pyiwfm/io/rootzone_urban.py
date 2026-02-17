"""
Urban land-use sub-file reader for IWFM RootZone component.

This module reads the IWFM urban land-use file (URBFL), which is
referenced by the RootZone component main file.  The file contains
urban area data references, root depth, curve numbers, water demand
and water-use specification references, per-subregion management data,
per-element surface-flow destinations, and initial soil-moisture
conditions.

Format matches ``Class_UrbanLandUse_v50::New()`` in the Fortran source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.io.iwfm_reader import (
    COMMENT_CHARS,
    LineBuffer as _LineBuffer,
    is_comment_line as _is_comment_line,
    strip_inline_comment as _strip_comment,
)


# ── Data classes ──────────────────────────────────────────────────────


@dataclass
class UrbanCurveNumberRow:
    """Curve-number data for one subregion (urban).

    Attributes:
        subregion_id: Subregion identifier (1-based).
        cn_values: Curve-number values, one per soil type.
    """

    subregion_id: int
    cn_values: list[float] = field(default_factory=list)


@dataclass
class UrbanManagementRow:
    """Per-subregion urban water management data.

    Attributes:
        subregion_id: Subregion identifier (1-based).
        pervious_fraction: Fraction of pervious area (0–1).
        demand_column: Column index into water-demand data file.
        water_use_column: Column index into water-use specs file.
        etc_column: Column index into ET data file.
        return_flow_column: Column index into return-flow fractions.
        reuse_column: Column index into reuse fractions.
    """

    subregion_id: int
    pervious_fraction: float = 1.0
    demand_column: int = 0
    water_use_column: int = 0
    etc_column: int = 0
    return_flow_column: int = 0
    reuse_column: int = 0


@dataclass
class SurfaceFlowDestRow:
    """Per-element surface-flow destination.

    Attributes:
        element_id: Element identifier (1-based).
        dest_type: Destination type code:
            1 = outside model domain,
            2 = stream node,
            3 = lake,
            4 = subregion,
            5 = groundwater element.
        dest_id: Destination identifier (node / lake / subregion / element ID).
    """

    element_id: int
    dest_type: int = 1
    dest_id: int = 0


@dataclass
class UrbanInitialConditionRow:
    """Initial soil-moisture conditions for one subregion.

    Attributes:
        subregion_id: Subregion identifier (1-based).
        precip_fractions: Precipitation fractions, one per soil type.
        moisture_contents: Soil-moisture contents, one per soil type.
    """

    subregion_id: int
    precip_fractions: list[float] = field(default_factory=list)
    moisture_contents: list[float] = field(default_factory=list)


@dataclass
class UrbanLandUseConfig:
    """Configuration parsed from an urban land-use sub-file.

    Attributes:
        area_data_file: Path to urban-area time-series data file.
        root_depth_factor: Conversion factor for root depth.
        root_depth: Urban root depth (already multiplied by factor).
        curve_numbers: Curve-number rows (one per subregion).
        demand_file: Path to urban water-demand data file.
        water_use_specs_file: Path to urban water-use specifications file.
        management: Per-subregion management / pointer data.
        surface_flow_destinations: Per-element surface-flow destinations.
        initial_conditions: Initial soil-moisture condition rows.
    """

    area_data_file: Path | None = None
    root_depth_factor: float = 1.0
    root_depth: float = 0.0
    curve_numbers: list[UrbanCurveNumberRow] = field(default_factory=list)
    demand_file: Path | None = None
    water_use_specs_file: Path | None = None
    management: list[UrbanManagementRow] = field(default_factory=list)
    surface_flow_destinations: list[SurfaceFlowDestRow] = field(
        default_factory=list
    )
    initial_conditions: list[UrbanInitialConditionRow] = field(
        default_factory=list
    )


# ── Reader ────────────────────────────────────────────────────────────


class UrbanLandUseReader:
    """Reader for IWFM urban land-use sub-file.

    Parses the positional-sequential file referenced as *URBFL* in
    the RootZone component main file.

    Args:
        n_subregions: Number of subregions (for exact row counts).
        n_elements: Number of elements (for destination row counts).
    """

    def __init__(
        self,
        n_subregions: int | None = None,
        n_elements: int | None = None,
    ) -> None:
        self._n_sub = n_subregions
        self._n_elem = n_elements

    def read(
        self,
        filepath: Path | str,
        base_dir: Path | None = None,
    ) -> UrbanLandUseConfig:
        """Parse the urban land-use file.

        Args:
            filepath: Path to the urban land-use file.
            base_dir: Base directory for resolving relative sub-file
                paths.  Defaults to the parent of *filepath*.

        Returns:
            Populated :class:`UrbanLandUseConfig`.
        """
        filepath = Path(filepath)
        if base_dir is None:
            base_dir = filepath.parent

        with open(filepath, "r") as f:
            lines = f.readlines()
        buf = _LineBuffer(lines)
        config = UrbanLandUseConfig()

        # 1. Land-use area data file
        val = buf.next_data_or_empty()
        if val:
            config.area_data_file = self._resolve(base_dir, val)

        # 2. Root-depth conversion factor
        val = buf.next_data()
        try:
            config.root_depth_factor = float(val)
        except ValueError as exc:
            raise FileFormatError(
                f"Invalid root depth factor: '{val}'",
                line_number=buf.line_num,
            ) from exc

        # 3. Root-depth value
        val = buf.next_data()
        try:
            config.root_depth = float(val) * config.root_depth_factor
        except ValueError as exc:
            raise FileFormatError(
                f"Invalid root depth: '{val}'",
                line_number=buf.line_num,
            ) from exc

        # 4. Curve-number data (per subregion)
        config.curve_numbers = self._read_cn_rows(buf)

        # 5. Urban water-demand data file
        val = buf.next_data_or_empty()
        if val:
            config.demand_file = self._resolve(base_dir, val)

        # 6. Urban water-use specifications file
        val = buf.next_data_or_empty()
        if val:
            config.water_use_specs_file = self._resolve(base_dir, val)

        # 7. Per-subregion management data (7 columns)
        config.management = self._read_management_rows(buf)

        # 8. Per-element surface-flow destinations (3 columns)
        config.surface_flow_destinations = self._read_dest_rows(buf)

        # 9. Initial conditions
        config.initial_conditions = self._read_initial_conditions(buf)

        return config

    # ── internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _resolve(base_dir: Path, filepath: str) -> Path:
        p = Path(filepath.strip())
        if p.is_absolute():
            return p
        return base_dir / p

    def _read_rows(
        self, buf: _LineBuffer, min_cols: int, n_expected: int | None = None,
    ) -> list[list[str]]:
        """Read tabular data rows with comment-based section boundaries."""
        rows: list[list[str]] = []
        started = False

        while True:
            raw = buf.next_line()
            if raw is None:
                break

            if _is_comment_line(raw):
                if n_expected is not None:
                    continue
                if started:
                    buf.pushback()
                    break
                continue

            value, _ = _strip_comment(raw)
            parts = value.split()

            if len(parts) < min_cols:
                buf.pushback()
                break

            rows.append(parts)
            started = True

            if n_expected is not None and len(rows) >= n_expected:
                break

        return rows

    def _read_cn_rows(self, buf: _LineBuffer) -> list[UrbanCurveNumberRow]:
        raw_rows = self._read_rows(buf, min_cols=2, n_expected=self._n_sub)
        result: list[UrbanCurveNumberRow] = []
        for parts in raw_rows:
            try:
                sub_id = int(parts[0])
                vals = [float(v) for v in parts[1:]]
                result.append(
                    UrbanCurveNumberRow(subregion_id=sub_id, cn_values=vals)
                )
            except ValueError:
                break
        return result

    def _read_management_rows(
        self, buf: _LineBuffer
    ) -> list[UrbanManagementRow]:
        raw_rows = self._read_rows(buf, min_cols=7, n_expected=self._n_sub)
        result: list[UrbanManagementRow] = []
        for parts in raw_rows:
            try:
                sub_id = int(parts[0])
                result.append(
                    UrbanManagementRow(
                        subregion_id=sub_id,
                        pervious_fraction=float(parts[1]),
                        demand_column=int(float(parts[2])),
                        water_use_column=int(float(parts[3])),
                        etc_column=int(float(parts[4])),
                        return_flow_column=int(float(parts[5])),
                        reuse_column=int(float(parts[6])),
                    )
                )
            except ValueError:
                break
        return result

    def _read_dest_rows(
        self, buf: _LineBuffer
    ) -> list[SurfaceFlowDestRow]:
        raw_rows = self._read_rows(buf, min_cols=3, n_expected=self._n_elem)
        result: list[SurfaceFlowDestRow] = []
        for parts in raw_rows:
            try:
                elem_id = int(parts[0])
                dtype = int(float(parts[1]))
                did = int(float(parts[2]))
                result.append(
                    SurfaceFlowDestRow(
                        element_id=elem_id,
                        dest_type=dtype,
                        dest_id=did,
                    )
                )
            except ValueError:
                break
        return result

    def _read_initial_conditions(
        self, buf: _LineBuffer
    ) -> list[UrbanInitialConditionRow]:
        raw_rows = self._read_rows(buf, min_cols=3, n_expected=self._n_sub)
        result: list[UrbanInitialConditionRow] = []
        for parts in raw_rows:
            try:
                sub_id = int(parts[0])
                data = [float(v) for v in parts[1:]]
                pf = data[0::2]
                mc = data[1::2]
                result.append(
                    UrbanInitialConditionRow(
                        subregion_id=sub_id,
                        precip_fractions=pf,
                        moisture_contents=mc,
                    )
                )
            except ValueError:
                break
        return result


# ── convenience function ──────────────────────────────────────────────


def read_urban_landuse(
    filepath: Path | str,
    base_dir: Path | None = None,
    n_subregions: int | None = None,
    n_elements: int | None = None,
) -> UrbanLandUseConfig:
    """Read an IWFM urban land-use sub-file.

    Args:
        filepath: Path to the urban land-use file.
        base_dir: Base directory for resolving relative paths.
        n_subregions: Number of subregions (for exact row counts).
        n_elements: Number of elements (for destination row counts).

    Returns:
        :class:`UrbanLandUseConfig` with parsed values.
    """
    reader = UrbanLandUseReader(
        n_subregions=n_subregions, n_elements=n_elements
    )
    return reader.read(filepath, base_dir)
