"""Unit tests for the generic IWFM time series data writer.

Tests IWFMTimeSeriesDataWriter, TimeSeriesDataConfig, DSSPathItem,
and all factory helper functions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyiwfm.io.timeseries_writer import (
    DSSPathItem,
    IWFMTimeSeriesDataWriter,
    TimeSeriesDataConfig,
    make_ag_water_demand_ts_config,
    make_crop_coeff_ts_config,
    make_diversion_ts_config,
    make_et_ts_config,
    make_irig_period_ts_config,
    make_max_lake_elev_ts_config,
    make_precip_ts_config,
    make_pumping_ts_config,
    make_return_flow_ts_config,
    make_reuse_ts_config,
    make_stream_inflow_ts_config,
    make_stream_surface_area_ts_config,
)

# =============================================================================
# DSSPathItem tests
# =============================================================================


class TestDSSPathItem:
    """Tests for the DSSPathItem dataclass."""

    def test_creation(self) -> None:
        item = DSSPathItem(index=1, path="/A/B/C//1DAY/F/")
        assert item.index == 1
        assert item.path == "/A/B/C//1DAY/F/"


# =============================================================================
# TimeSeriesDataConfig tests
# =============================================================================


class TestTimeSeriesDataConfig:
    """Tests for the TimeSeriesDataConfig dataclass."""

    def test_defaults(self) -> None:
        config = TimeSeriesDataConfig()
        assert config.title == ""
        assert config.ncol == 0
        assert config.factor == 1.0
        assert config.time_unit == ""
        assert config.has_time_unit is False
        assert config.nsp == 1
        assert config.nfq == 0
        assert config.dss_file == ""
        assert config.ncol_tag == "NCOL"
        assert config.factor_tag == "FACT"
        assert config.nsp_tag == "NSP"
        assert config.nfq_tag == "NFQ"
        assert config.use_dss is False

    def test_use_dss_property_false_when_no_paths(self) -> None:
        config = TimeSeriesDataConfig()
        assert config.use_dss is False

    def test_use_dss_property_true_when_paths_set(self) -> None:
        config = TimeSeriesDataConfig(dss_paths=[DSSPathItem(index=1, path="/A/B/C//1DAY/F/")])
        assert config.use_dss is True

    def test_custom_tags(self) -> None:
        config = TimeSeriesDataConfig(
            ncol_tag="NCOLPUMP",
            factor_tag="FACTPUMP",
            nsp_tag="NSPPUMP",
            nfq_tag="NFQPUMP",
        )
        assert config.ncol_tag == "NCOLPUMP"
        assert config.factor_tag == "FACTPUMP"
        assert config.nsp_tag == "NSPPUMP"
        assert config.nfq_tag == "NFQPUMP"


# =============================================================================
# IWFMTimeSeriesDataWriter tests
# =============================================================================


class TestIWFMTimeSeriesDataWriter:
    """Tests for the IWFMTimeSeriesDataWriter class."""

    def test_write_inline_data(self, tmp_path: Path) -> None:
        """Test writing a TS file with inline data."""
        config = TimeSeriesDataConfig(
            title="Test Pumping TS",
            ncol=3,
            factor=1.0,
            nsp=1,
            nfq=0,
            ncol_tag="NCOLPUMP",
            factor_tag="FACTPUMP",
            nsp_tag="NSPPUMP",
            nfq_tag="NFQPUMP",
            dates=["10/01/1990_24:00", "10/02/1990_24:00"],
            data=np.array(
                [
                    [100.0, 200.0, 300.0],
                    [110.0, 210.0, 310.0],
                ]
            ),
        )

        writer = IWFMTimeSeriesDataWriter()
        outfile = tmp_path / "TSPumping.dat"
        result = writer.write(config, outfile)

        assert result == outfile
        assert outfile.exists()

        content = outfile.read_text()
        # Check header elements
        assert "Test Pumping TS" in content
        assert "NCOLPUMP" in content
        assert "FACTPUMP" in content
        assert "NSPPUMP" in content
        assert "NFQPUMP" in content
        assert "DSSFL" in content

        # Check data rows present
        assert "10/01/1990_24:00" in content
        assert "10/02/1990_24:00" in content
        assert "100.000000" in content
        assert "310.000000" in content

    def test_write_1d_data(self, tmp_path: Path) -> None:
        """Test writing with 1D data array (single column)."""
        config = TimeSeriesDataConfig(
            title="Single Column TS",
            ncol=1,
            factor=2.5,
            nsp=1,
            nfq=0,
            dates=["01/01/2000_24:00"],
            data=np.array([42.0]),
        )

        writer = IWFMTimeSeriesDataWriter()
        outfile = tmp_path / "single.dat"
        writer.write(config, outfile)

        content = outfile.read_text()
        assert "01/01/2000_24:00" in content
        assert "42.000000" in content

    def test_write_with_time_unit(self, tmp_path: Path) -> None:
        """Test writing a TS file with TUNIT line."""
        config = TimeSeriesDataConfig(
            title="Surface Area TS",
            ncol=2,
            factor=1.0,
            has_time_unit=True,
            time_unit="1DAY",
            nsp=1,
            nfq=0,
            ncol_tag="NCOLSA",
            factor_tag="FACTSA",
            time_unit_tag="TUNITSA",
            nsp_tag="NSPSA",
            nfq_tag="NFQSA",
            dates=["10/01/1990_24:00"],
            data=np.array([[500.0, 600.0]]),
        )

        writer = IWFMTimeSeriesDataWriter()
        outfile = tmp_path / "SurfaceArea.dat"
        writer.write(config, outfile)

        content = outfile.read_text()
        assert "TUNITSA" in content
        assert "1DAY" in content

    def test_write_without_time_unit(self, tmp_path: Path) -> None:
        """Test that TUNIT line is omitted when has_time_unit is False."""
        config = TimeSeriesDataConfig(
            title="No Time Unit TS",
            ncol=1,
            factor=1.0,
            has_time_unit=False,
            nsp=1,
            nfq=0,
            dates=["10/01/1990_24:00"],
            data=np.array([1.0]),
        )

        writer = IWFMTimeSeriesDataWriter()
        outfile = tmp_path / "notime.dat"
        writer.write(config, outfile)

        content = outfile.read_text()
        # The tag "/ TUNIT" should not appear on any parameter line
        assert "/ TUNIT" not in content

    def test_write_with_column_mapping(self, tmp_path: Path) -> None:
        """Test writing with column mapping section."""
        config = TimeSeriesDataConfig(
            title="Inflow TS with Mapping",
            ncol=2,
            factor=1.0,
            nsp=1,
            nfq=0,
            ncol_tag="NCOLSTRM",
            factor_tag="FACTSTRM",
            nsp_tag="NSPSTRM",
            nfq_tag="NFQSTRM",
            column_header="ID   IRST",
            column_mapping=[
                "    1      1",
                "    2      3",
            ],
            dates=["10/01/1990_24:00"],
            data=np.array([[10.0, 20.0]]),
        )

        writer = IWFMTimeSeriesDataWriter()
        outfile = tmp_path / "Inflow.dat"
        writer.write(config, outfile)

        content = outfile.read_text()
        assert "ID   IRST" in content
        assert "1      1" in content
        assert "2      3" in content

    def test_write_dss_mode(self, tmp_path: Path) -> None:
        """Test writing in DSS pathname mode."""
        config = TimeSeriesDataConfig(
            title="DSS Mode TS",
            ncol=2,
            factor=1.0,
            nsp=1,
            nfq=0,
            dss_file="model.dss",
            dss_paths=[
                DSSPathItem(index=1, path="/BASIN/LOC1/FLOW//1DAY/PYIWFM/"),
                DSSPathItem(index=2, path="/BASIN/LOC2/FLOW//1DAY/PYIWFM/"),
            ],
        )

        writer = IWFMTimeSeriesDataWriter()
        outfile = tmp_path / "dss_mode.dat"
        writer.write(config, outfile)

        content = outfile.read_text()
        assert "DSS Pathnames" in content
        assert "/BASIN/LOC1/FLOW//1DAY/PYIWFM/" in content
        assert "/BASIN/LOC2/FLOW//1DAY/PYIWFM/" in content
        # Data header should NOT appear in DSS mode
        assert "Time Series Data" not in content

    def test_write_dss_mode_convenience(self, tmp_path: Path) -> None:
        """Test write_dss_mode convenience method."""
        config = TimeSeriesDataConfig(
            title="DSS Convenience",
            ncol=1,
            dss_paths=[DSSPathItem(index=1, path="/A/B/C//1DAY/F/")],
        )

        writer = IWFMTimeSeriesDataWriter()
        outfile = tmp_path / "dss_conv.dat"
        result = writer.write_dss_mode(config, outfile)
        assert result == outfile
        assert outfile.exists()

    def test_write_dss_mode_raises_without_paths(self, tmp_path: Path) -> None:
        """Test that write_dss_mode raises if dss_paths is empty."""
        config = TimeSeriesDataConfig(title="No DSS paths")
        writer = IWFMTimeSeriesDataWriter()
        with pytest.raises(ValueError, match="dss_paths must be populated"):
            writer.write_dss_mode(config, tmp_path / "bad.dat")

    def test_write_no_data(self, tmp_path: Path) -> None:
        """Test writing header-only (no data, no DSS)."""
        config = TimeSeriesDataConfig(
            title="Header Only",
            ncol=5,
            factor=1.0,
            nsp=1,
            nfq=0,
        )

        writer = IWFMTimeSeriesDataWriter()
        outfile = tmp_path / "header_only.dat"
        writer.write(config, outfile)

        content = outfile.read_text()
        assert "Header Only" in content
        # No data rows, just header
        lines = content.strip().split("\n")
        # All lines should be comment or parameter lines
        for line in lines:
            assert not line.strip().startswith("10/")  # No date rows

    def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test that parent directories are created automatically."""
        config = TimeSeriesDataConfig(
            title="Nested File",
            ncol=1,
            dates=["01/01/2000_24:00"],
            data=np.array([1.0]),
        )

        writer = IWFMTimeSeriesDataWriter()
        outfile = tmp_path / "a" / "b" / "c" / "data.dat"
        writer.write(config, outfile)
        assert outfile.exists()

    def test_write_custom_data_fmt(self, tmp_path: Path) -> None:
        """Test writing with a custom data format string."""
        config = TimeSeriesDataConfig(
            title="Custom Fmt",
            ncol=1,
            dates=["01/01/2000_24:00"],
            data=np.array([3.14159]),
            data_fmt="%10.2f",
        )

        writer = IWFMTimeSeriesDataWriter()
        outfile = tmp_path / "custom_fmt.dat"
        writer.write(config, outfile)

        content = outfile.read_text()
        assert "3.14" in content

    def test_write_description_lines(self, tmp_path: Path) -> None:
        """Test custom description lines in header."""
        config = TimeSeriesDataConfig(
            title="Custom Desc",
            ncol=1,
            description_lines=["Line one", "Line two", "Line three"],
            dates=["01/01/2000_24:00"],
            data=np.array([1.0]),
        )

        writer = IWFMTimeSeriesDataWriter()
        outfile = tmp_path / "desc.dat"
        writer.write(config, outfile)

        content = outfile.read_text()
        assert "Line one" in content
        assert "Line two" in content
        assert "Line three" in content

    def test_round_trip_data_integrity(self, tmp_path: Path) -> None:
        """Test that data values are preserved through write + read-back."""
        rng = np.random.default_rng(42)
        ncol = 5
        n_times = 10
        data = rng.uniform(0, 1000, (n_times, ncol))
        dates = [f"10/{i + 1:02d}/1990_24:00" for i in range(n_times)]

        config = TimeSeriesDataConfig(
            title="Round Trip",
            ncol=ncol,
            factor=1.0,
            nsp=1,
            nfq=0,
            dates=dates,
            data=data,
        )

        writer = IWFMTimeSeriesDataWriter()
        outfile = tmp_path / "roundtrip.dat"
        writer.write(config, outfile)

        # Read back and verify data values
        content = outfile.read_text()
        lines = content.strip().split("\n")
        [
            line
            for line in lines
            if line.strip()
            and not line.strip().startswith("C")
            and "/" not in line.split()[0]
            and line.strip()[0].isdigit() is False
        ]

        # Filter to lines that start with date patterns
        date_lines = [
            line
            for line in lines
            if line.strip() and line.strip()[:2].isdigit() and "/" in line.strip()[:10]
        ]

        assert len(date_lines) == n_times
        for i, line in enumerate(date_lines):
            parts = line.split()
            # First part is date, rest are values
            values = [float(v) for v in parts[1:]]
            np.testing.assert_allclose(values, data[i], atol=1e-4)


# =============================================================================
# Factory helper tests
# =============================================================================


class TestFactoryHelpers:
    """Tests for all factory helper functions."""

    def test_make_pumping_ts_config(self) -> None:
        config = make_pumping_ts_config(ncol=3, factor=2.0, nsp=2, nfq=1)
        assert config.ncol == 3
        assert config.factor == 2.0
        assert config.nsp == 2
        assert config.nfq == 1
        assert config.ncol_tag == "NCOLPUMP"
        assert config.factor_tag == "FACTPUMP"
        assert config.nsp_tag == "NSPPUMP"
        assert config.nfq_tag == "NFQPUMP"
        assert "Pumping" in config.title

    def test_make_stream_inflow_ts_config(self) -> None:
        config = make_stream_inflow_ts_config(ncol=5)
        assert config.ncol == 5
        assert config.ncol_tag == "NCOLSTRM"
        assert config.factor_tag == "FACTSTRM"
        assert config.nsp_tag == "NSPSTRM"
        assert config.nfq_tag == "NFQSTRM"
        assert "Inflow" in config.title

    def test_make_diversion_ts_config(self) -> None:
        config = make_diversion_ts_config(ncol=4)
        assert config.ncol == 4
        assert config.ncol_tag == "NCOLDV"
        assert config.factor_tag == "FACTDV"
        assert config.nsp_tag == "NSPDV"
        assert config.nfq_tag == "NFQDV"
        assert "Diversion" in config.title

    def test_make_precip_ts_config(self) -> None:
        config = make_precip_ts_config(ncol=10)
        assert config.ncol == 10
        assert config.ncol_tag == "NRAIN"
        assert config.factor_tag == "FACTRN"
        assert config.nsp_tag == "NSPRN"
        assert config.nfq_tag == "NFQRN"
        assert "Precip" in config.title

    def test_make_et_ts_config(self) -> None:
        config = make_et_ts_config(ncol=7)
        assert config.ncol == 7
        assert config.ncol_tag == "NCOLET"
        assert config.factor_tag == "FACTET"
        assert config.nsp_tag == "NSPET"
        assert config.nfq_tag == "NFQET"
        assert "Evapotranspiration" in config.title

    def test_make_crop_coeff_ts_config(self) -> None:
        config = make_crop_coeff_ts_config(ncol=12)
        assert config.ncol == 12
        assert config.factor == 1.0  # No factor for crop coeff
        assert config.ncol_tag == "NCFF"
        assert config.nsp_tag == "NSPCFF"
        assert config.nfq_tag == "NFQCFF"
        assert "Crop" in config.title

    def test_make_return_flow_ts_config(self) -> None:
        config = make_return_flow_ts_config(ncol=6)
        assert config.ncol == 6
        assert config.factor == 1.0
        assert config.ncol_tag == "NCOLRT"
        assert config.nsp_tag == "NSPRT"
        assert config.nfq_tag == "NFQRT"
        assert "Return" in config.title

    def test_make_reuse_ts_config(self) -> None:
        config = make_reuse_ts_config(ncol=4)
        assert config.ncol == 4
        assert config.ncol_tag == "NCOLRUF"
        assert config.nsp_tag == "NSPRUF"
        assert config.nfq_tag == "NFQRUF"
        assert "Reuse" in config.title

    def test_make_irig_period_ts_config(self) -> None:
        config = make_irig_period_ts_config(ncol=3)
        assert config.ncol == 3
        assert config.ncol_tag == "NCOLIP"
        assert config.nsp_tag == "NSPIP"
        assert config.nfq_tag == "NFQIP"
        assert "Irrigation" in config.title

    def test_make_ag_water_demand_ts_config(self) -> None:
        config = make_ag_water_demand_ts_config(ncol=8, factor=0.5)
        assert config.ncol == 8
        assert config.factor == 0.5
        assert config.ncol_tag == "NDMAG"
        assert config.factor_tag == "FACTDMAG"
        assert config.nsp_tag == "NSPDMAG"
        assert config.nfq_tag == "NFQDMAG"
        assert "Ag" in config.title or "Agricultural" in config.title

    def test_make_max_lake_elev_ts_config(self) -> None:
        config = make_max_lake_elev_ts_config(ncol=2)
        assert config.ncol == 2
        assert config.ncol_tag == "NCOLLK"
        assert config.factor_tag == "FACTLK"
        assert config.nsp_tag == "NSPLK"
        assert config.nfq_tag == "NFQLK"
        assert "Lake" in config.title

    def test_make_stream_surface_area_ts_config(self) -> None:
        config = make_stream_surface_area_ts_config(ncol=5)
        assert config.ncol == 5
        assert config.has_time_unit is True
        assert config.time_unit == "1DAY"
        assert config.ncol_tag == "NCOLSA"
        assert config.factor_tag == "FACTSA"
        assert config.time_unit_tag == "TUNITSA"
        assert config.nsp_tag == "NSPSA"
        assert config.nfq_tag == "NFQSA"
        assert "Surface" in config.title

    def test_factory_kwargs_pass_through(self) -> None:
        """Test that extra kwargs are passed to TimeSeriesDataConfig."""
        config = make_pumping_ts_config(
            ncol=1,
            dss_file="test.dss",
            time_unit="1MON",
        )
        assert config.dss_file == "test.dss"
        assert config.time_unit == "1MON"


# =============================================================================
# Factory + Writer integration tests
# =============================================================================


class TestFactoryWriterIntegration:
    """Test that factory-created configs work end-to-end with the writer."""

    @pytest.fixture
    def writer(self) -> IWFMTimeSeriesDataWriter:
        return IWFMTimeSeriesDataWriter()

    @pytest.fixture
    def sample_dates(self) -> list[str]:
        return [f"10/{i + 1:02d}/1990_24:00" for i in range(3)]

    @pytest.fixture
    def sample_data_3col(self) -> np.ndarray:
        return np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )

    def test_pumping_round_trip(
        self,
        tmp_path: Path,
        writer: IWFMTimeSeriesDataWriter,
        sample_dates: list[str],
        sample_data_3col: np.ndarray,
    ) -> None:
        config = make_pumping_ts_config(
            ncol=3,
            dates=sample_dates,
            data=sample_data_3col,
        )
        outfile = tmp_path / "TSPumping.dat"
        writer.write(config, outfile)

        content = outfile.read_text()
        assert "NCOLPUMP" in content
        assert "FACTPUMP" in content
        assert "10/01/1990_24:00" in content

    def test_precip_round_trip(
        self,
        tmp_path: Path,
        writer: IWFMTimeSeriesDataWriter,
        sample_dates: list[str],
        sample_data_3col: np.ndarray,
    ) -> None:
        config = make_precip_ts_config(
            ncol=3,
            dates=sample_dates,
            data=sample_data_3col,
        )
        outfile = tmp_path / "Precip.dat"
        writer.write(config, outfile)

        content = outfile.read_text()
        assert "NRAIN" in content
        assert "FACTRN" in content

    def test_et_round_trip(
        self,
        tmp_path: Path,
        writer: IWFMTimeSeriesDataWriter,
        sample_dates: list[str],
        sample_data_3col: np.ndarray,
    ) -> None:
        config = make_et_ts_config(
            ncol=3,
            dates=sample_dates,
            data=sample_data_3col,
        )
        outfile = tmp_path / "ET.dat"
        writer.write(config, outfile)

        content = outfile.read_text()
        assert "NCOLET" in content
        assert "FACTET" in content

    def test_stream_inflow_round_trip(
        self,
        tmp_path: Path,
        writer: IWFMTimeSeriesDataWriter,
        sample_dates: list[str],
        sample_data_3col: np.ndarray,
    ) -> None:
        config = make_stream_inflow_ts_config(
            ncol=3,
            dates=sample_dates,
            data=sample_data_3col,
            column_header="ID   IRST",
            column_mapping=["    1      1", "    2      2", "    3      3"],
        )
        outfile = tmp_path / "StreamInflow.dat"
        writer.write(config, outfile)

        content = outfile.read_text()
        assert "NCOLSTRM" in content
        assert "ID   IRST" in content

    def test_surface_area_has_tunit(
        self,
        tmp_path: Path,
        writer: IWFMTimeSeriesDataWriter,
        sample_dates: list[str],
        sample_data_3col: np.ndarray,
    ) -> None:
        config = make_stream_surface_area_ts_config(
            ncol=3,
            dates=sample_dates,
            data=sample_data_3col,
        )
        outfile = tmp_path / "SurfArea.dat"
        writer.write(config, outfile)

        content = outfile.read_text()
        assert "TUNITSA" in content
        assert "1DAY" in content


# =============================================================================
# Import via io package tests
# =============================================================================


class TestImports:
    """Test that all exports are importable from pyiwfm.io."""

    def test_import_writer_class(self) -> None:
        from pyiwfm.io import IWFMTimeSeriesDataWriter as W

        assert W is not None

    def test_import_config_class(self) -> None:
        from pyiwfm.io import TimeSeriesDataConfig as C

        assert C is not None

    def test_import_dss_path_item(self) -> None:
        from pyiwfm.io import DSSPathItem as D

        assert D is not None

    def test_import_factory_helpers(self) -> None:
        from pyiwfm.io import (
            make_ag_water_demand_ts_config,
            make_crop_coeff_ts_config,
            make_diversion_ts_config,
            make_et_ts_config,
            make_irig_period_ts_config,
            make_max_lake_elev_ts_config,
            make_precip_ts_config,
            make_pumping_ts_config,
            make_return_flow_ts_config,
            make_reuse_ts_config,
            make_stream_inflow_ts_config,
            make_stream_surface_area_ts_config,
        )

        # All should be callable
        assert callable(make_pumping_ts_config)
        assert callable(make_stream_inflow_ts_config)
        assert callable(make_diversion_ts_config)
        assert callable(make_precip_ts_config)
        assert callable(make_et_ts_config)
        assert callable(make_crop_coeff_ts_config)
        assert callable(make_return_flow_ts_config)
        assert callable(make_reuse_ts_config)
        assert callable(make_irig_period_ts_config)
        assert callable(make_ag_water_demand_ts_config)
        assert callable(make_max_lake_elev_ts_config)
        assert callable(make_stream_surface_area_ts_config)
