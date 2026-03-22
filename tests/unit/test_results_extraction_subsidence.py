"""Comprehensive tests for subsidence extraction via LazyHeadDataLoader and ResultsExtractor.

Test groups:
1. LazyHeadDataLoader — synthetic HDF5 (no external data dependency)
2. ResultsExtractor — synthetic data with layer aggregation
3. Integration with real C2VSimFG data (skipped if file not found)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

# Skip all tests if h5py is not available
h5py = pytest.importorskip("h5py")

from pyiwfm.io.head_loader import LazyHeadDataLoader
from pyiwfm.calibration.results_extraction import ExtractionResult, ExtractionSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_iwfm_native_hdf5(
    path: Path,
    dataset_name: str,
    n_nodes: int,
    n_layers: int,
    n_timesteps: int,
    *,
    data_fn=None,
    write_nlayers_attr: bool = False,
) -> None:
    """Create a synthetic IWFM-native HDF5 file.

    Data layout: (n_timesteps, n_nodes * n_layers) with layer-major ordering
    [all_nodes_layer1, all_nodes_layer2, ...].
    """
    total_cols = n_nodes * n_layers
    if data_fn is None:
        # Default: value = layer * 1000 + node (1-based)
        def data_fn(ts, layer, node):
            return float(layer * 1000 + node + ts * 0.1)

    data = np.zeros((n_timesteps, total_cols), dtype=np.float64)
    for ts in range(n_timesteps):
        for layer in range(n_layers):
            for node in range(n_nodes):
                flat_idx = layer * n_nodes + node
                data[ts, flat_idx] = data_fn(ts, layer + 1, node + 1)

    base = datetime(2020, 1, 31)
    times = [(base + timedelta(days=30 * i)).isoformat() for i in range(n_timesteps)]

    with h5py.File(path, "w") as f:
        f.create_dataset(dataset_name, data=data)
        ts_ds = f.create_dataset("times", data=np.array(times, dtype="S32"))
        if write_nlayers_attr:
            f[dataset_name].attrs["NLayers"] = n_layers


# ============================================================================
# Group 1: LazyHeadDataLoader — synthetic HDF5
# ============================================================================


class TestLoaderHeadDetection:
    """Verify loader auto-detects head vs subsidence datasets."""

    def test_load_iwfm_native_head_hdf5(self, tmp_path: Path) -> None:
        fp = tmp_path / "head.hdf"
        _make_iwfm_native_hdf5(fp, "GWHeadAtAllNodes", n_nodes=4, n_layers=2, n_timesteps=3)
        loader = LazyHeadDataLoader(fp, n_layers=2)
        assert loader.data_type == "head"
        assert loader.n_nodes == 4
        assert loader.n_layers == 2
        assert loader.n_frames == 3

    def test_load_iwfm_native_subsidence_hdf5(self, tmp_path: Path) -> None:
        fp = tmp_path / "subs.hdf"
        _make_iwfm_native_hdf5(fp, "SubsidenceAtAllNodes", n_nodes=4, n_layers=2, n_timesteps=3)
        loader = LazyHeadDataLoader(fp, n_layers=2)
        assert loader.data_type == "subsidence"
        assert loader.n_nodes == 4
        assert loader.n_layers == 2


class TestLoaderReshape:
    """Verify the flat → (n_nodes, n_layers) reshape is correct."""

    def test_reshape_correctness(self, tmp_path: Path) -> None:
        """Each frame[node, layer] must equal the known value."""
        n_nodes, n_layers = 3, 2
        fp = tmp_path / "reshape.hdf"
        _make_iwfm_native_hdf5(
            fp, "SubsidenceAtAllNodes", n_nodes=n_nodes, n_layers=n_layers, n_timesteps=2,
        )
        loader = LazyHeadDataLoader(fp, n_layers=n_layers)
        frame = loader.get_frame(0)
        assert frame.shape == (n_nodes, n_layers)

        # data_fn default: layer*1000 + node + ts*0.1
        # ts=0: frame[node-1, layer-1] = layer*1000 + node + 0
        for layer in range(1, n_layers + 1):
            for node in range(1, n_nodes + 1):
                expected = float(layer * 1000 + node)
                assert_allclose(
                    frame[node - 1, layer - 1],
                    expected,
                    err_msg=f"Mismatch at node={node}, layer={layer}",
                )

    def test_multilayer_reshape_large(self, tmp_path: Path) -> None:
        """Verify reshape for larger model (100 nodes, 4 layers)."""
        n_nodes, n_layers = 100, 4
        fp = tmp_path / "large.hdf"
        _make_iwfm_native_hdf5(
            fp, "GWHeadAtAllNodes", n_nodes=n_nodes, n_layers=n_layers, n_timesteps=2,
        )
        loader = LazyHeadDataLoader(fp, n_layers=n_layers)
        frame = loader.get_frame(1)
        assert frame.shape == (n_nodes, n_layers)

        # Spot check corners
        assert_allclose(frame[0, 0], 1 * 1000 + 1 + 0.1)  # node 1, layer 1, ts=1
        assert_allclose(frame[99, 3], 4 * 1000 + 100 + 0.1)  # node 100, layer 4, ts=1


class TestLoaderNLayers:
    """Verify n_layers resolution: explicit > attribute > fallback."""

    def test_nlayers_from_parameter(self, tmp_path: Path) -> None:
        fp = tmp_path / "param.hdf"
        _make_iwfm_native_hdf5(
            fp, "GWHeadAtAllNodes", n_nodes=6, n_layers=3, n_timesteps=1,
            write_nlayers_attr=True,
        )
        # Pass n_layers=2 which differs from the attribute (3)
        # Explicit should win
        loader = LazyHeadDataLoader(fp, n_layers=2)
        assert loader.n_layers == 2
        assert loader.n_nodes == 9  # 18 / 2

    def test_nlayers_from_attribute(self, tmp_path: Path) -> None:
        fp = tmp_path / "attr.hdf"
        _make_iwfm_native_hdf5(
            fp, "GWHeadAtAllNodes", n_nodes=6, n_layers=3, n_timesteps=1,
            write_nlayers_attr=True,
        )
        loader = LazyHeadDataLoader(fp)
        assert loader.n_layers == 3
        assert loader.n_nodes == 6

    def test_nlayers_fallback_warning(self, tmp_path: Path, caplog) -> None:
        fp = tmp_path / "noattr.hdf"
        _make_iwfm_native_hdf5(
            fp, "GWHeadAtAllNodes", n_nodes=6, n_layers=3, n_timesteps=1,
            write_nlayers_attr=False,
        )
        import logging
        with caplog.at_level(logging.WARNING, logger="pyiwfm.io.head_loader"):
            loader = LazyHeadDataLoader(fp)
        assert loader.n_layers == 1  # fallback
        assert loader.n_nodes == 18  # 6*3 columns / 1 layer
        assert "NLayers not provided" in caplog.text


class TestLoaderCache:
    """Verify cache returns consistent results."""

    def test_cache_consistency(self, tmp_path: Path) -> None:
        fp = tmp_path / "cache.hdf"
        _make_iwfm_native_hdf5(
            fp, "GWHeadAtAllNodes", n_nodes=4, n_layers=2, n_timesteps=5,
        )
        loader = LazyHeadDataLoader(fp, n_layers=2)
        frame_a = loader.get_frame(2)
        frame_b = loader.get_frame(2)
        assert_allclose(frame_a, frame_b)


# ============================================================================
# Group 2: Layer Aggregation (unit-level, no full ResultsExtractor)
# ============================================================================


class TestLayerAggregation:
    """Test _aggregate_layers logic via ResultsExtractor internals."""

    def _make_per_layer(self, n_times: int, n_layers: int, values_fn) -> np.ndarray:
        """Create per-layer array for testing aggregation."""
        arr = np.zeros((n_times, n_layers), dtype=np.float64)
        for t in range(n_times):
            for l in range(n_layers):
                arr[t, l] = values_fn(t, l)
        return arr

    def test_subsidence_sums_layers(self) -> None:
        """For SUBSIDENCE layer=0, values should be summed across layers."""
        from pyiwfm.calibration.results_extraction import ResultsExtractor

        # Manually test the aggregation logic
        # Layer 1: [1, 2, 3], Layer 2: [10, 20, 30]
        per_layer = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        n_times = 3

        # Create a minimal extractor to test _aggregate_layers
        extractor = object.__new__(ResultsExtractor)
        extractor._data_type = "SUBSIDENCE"
        extractor._incremental = False
        extractor._t_weights = {}

        spec = ExtractionSpec(name="test", x=0.0, y=0.0, layer=0)
        result = extractor._aggregate_layers(per_layer, spec, 2)

        assert_allclose(result, [11.0, 22.0, 33.0])

    def test_head_averages_layers(self) -> None:
        """For HEAD layer=0, values should be averaged across layers."""
        from pyiwfm.calibration.results_extraction import ResultsExtractor

        per_layer = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

        extractor = object.__new__(ResultsExtractor)
        extractor._data_type = "HEAD"
        extractor._incremental = False
        extractor._t_weights = {}

        spec = ExtractionSpec(name="test", x=0.0, y=0.0, layer=0)
        result = extractor._aggregate_layers(per_layer, spec, 2)

        assert_allclose(result, [5.5, 11.0, 16.5])

    def test_single_layer_extraction(self) -> None:
        """layer=1 should return only that layer's values."""
        from pyiwfm.calibration.results_extraction import ResultsExtractor

        per_layer = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

        extractor = object.__new__(ResultsExtractor)
        extractor._data_type = "SUBSIDENCE"
        extractor._incremental = False
        extractor._t_weights = {}

        spec = ExtractionSpec(name="test", x=0.0, y=0.0, layer=1)
        result = extractor._aggregate_layers(per_layer, spec, 2)

        assert_allclose(result, [1.0, 2.0, 3.0])

    def test_incremental_conversion(self) -> None:
        """Incremental: output[0]=0, output[i]=agg[i]-agg[i-1]."""
        from pyiwfm.calibration.results_extraction import ResultsExtractor

        # Cumulative subsidence: [0, 1, 3, 6] (single layer)
        per_layer = np.array([[0.0], [1.0], [3.0], [6.0]])

        extractor = object.__new__(ResultsExtractor)
        extractor._data_type = "SUBSIDENCE"
        extractor._incremental = True
        extractor._t_weights = {}

        spec = ExtractionSpec(name="test", x=0.0, y=0.0, layer=0)
        aggregated = extractor._aggregate_layers(per_layer, spec, 1)

        # Apply incremental logic (mirrors extract() method)
        n_times = len(aggregated)
        incr = np.full(n_times, np.nan, dtype=np.float64)
        incr[0] = 0.0
        incr[1:] = aggregated[1:] - aggregated[:-1]

        assert_allclose(incr, [0.0, 1.0, 2.0, 3.0])

    def test_cumulative_mode(self) -> None:
        """Non-incremental subsidence returns raw cumulative values."""
        from pyiwfm.calibration.results_extraction import ResultsExtractor

        per_layer = np.array([[0.0], [1.0], [3.0], [6.0]])

        extractor = object.__new__(ResultsExtractor)
        extractor._data_type = "SUBSIDENCE"
        extractor._incremental = False
        extractor._t_weights = {}

        spec = ExtractionSpec(name="test", x=0.0, y=0.0, layer=0)
        result = extractor._aggregate_layers(per_layer, spec, 1)

        assert_allclose(result, [0.0, 1.0, 3.0, 6.0])


# ============================================================================
# Group 3: Integration with real C2VSimFG data
# ============================================================================

_C2VSIMFG_SUBS_HDF = Path(__file__).resolve().parents[4] / "c2vsimfg" / "Results" / "C2VSimFG_SubsidenceAll.hdf"
_has_c2vsimfg_subs = _C2VSIMFG_SUBS_HDF.exists()


@pytest.mark.skipif(not _has_c2vsimfg_subs, reason="C2VSimFG subsidence HDF5 not found")
class TestC2VSimFGSubsidence:
    """Integration tests using real C2VSimFG subsidence data."""

    def test_metadata(self) -> None:
        """Verify HDF5 dataset exists and has expected structure."""
        with h5py.File(_C2VSIMFG_SUBS_HDF, "r") as f:
            assert "SubsidenceAtAllNodes" in f
            ds = f["SubsidenceAtAllNodes"]
            assert ds.ndim == 2
            n_timesteps, total_cols = ds.shape
            assert n_timesteps > 1, "Should have multiple timesteps"
            assert total_cols > 0, "Should have data columns"
            # Verify not all zeros (beyond initial condition)
            last_row = ds[-1, :]
            assert not np.all(last_row == 0.0), "Last timestep should have non-zero data"

    def test_load_frame(self) -> None:
        """Load one frame and verify shape and value range."""
        # C2VSimFG has 4 layers; detect from total_cols / expected n_nodes
        with h5py.File(_C2VSIMFG_SUBS_HDF, "r") as f:
            total_cols = f["SubsidenceAtAllNodes"].shape[1]

        # C2VSimFG: 30179 nodes, 4 layers → 120716 columns
        # Try n_layers=4 first
        n_layers = 4
        expected_nodes = total_cols // n_layers
        assert expected_nodes * n_layers == total_cols, (
            f"total_cols={total_cols} not divisible by n_layers={n_layers}"
        )

        loader = LazyHeadDataLoader(_C2VSIMFG_SUBS_HDF, n_layers=n_layers)
        assert loader.n_nodes == expected_nodes
        assert loader.n_layers == n_layers

        # Load last frame (should have accumulated subsidence)
        frame = loader.get_frame(loader.n_frames - 1)
        assert frame.shape == (expected_nodes, n_layers)
        # Subsidence values should be finite
        assert np.all(np.isfinite(frame)), "All values should be finite"

    def test_data_not_all_zeros(self) -> None:
        """Last timestep should have non-zero subsidence somewhere."""
        n_layers = 4
        loader = LazyHeadDataLoader(_C2VSIMFG_SUBS_HDF, n_layers=n_layers)
        frame = loader.get_frame(loader.n_frames - 1)
        assert np.any(frame != 0.0), "Subsidence data should not be all zeros"
