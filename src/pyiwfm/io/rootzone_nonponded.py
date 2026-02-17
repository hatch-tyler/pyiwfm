"""
Non-ponded agricultural crop sub-file reader for IWFM RootZone component.

This module reads the IWFM non-ponded agricultural crop file (AGNPFL),
which is referenced by the RootZone component main file.  The file
contains crop definitions, curve numbers, ET-column pointers,
irrigation period references, soil moisture targets, return-flow /
reuse fractions, and initial conditions.

The same binary format (``Class_AgLandUse_v50``) is used by both the
non-ponded and ponded crop files in IWFM v5.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.io.iwfm_reader import (
    LineBuffer as _LineBuffer,
)
from pyiwfm.io.iwfm_reader import (
    is_comment_line as _is_comment_line,
)
from pyiwfm.io.iwfm_reader import (
    strip_inline_comment as _strip_comment,
)

# ── Data classes ──────────────────────────────────────────────────────


@dataclass
class CurveNumberRow:
    """Curve-number data for one subregion.

    Attributes:
        subregion_id: Subregion identifier (1-based).
        cn_values: Curve-number values, one per soil type.
    """

    subregion_id: int
    cn_values: list[float] = field(default_factory=list)


@dataclass
class EtcPointerRow:
    """ET-column pointers for one subregion.

    Attributes:
        subregion_id: Subregion identifier (1-based).
        etc_columns: Column indices into the ET data file, one per crop.
    """

    subregion_id: int
    etc_columns: list[int] = field(default_factory=list)


@dataclass
class IrrigationPointerRow:
    """Irrigation-period column pointers for one subregion.

    Attributes:
        subregion_id: Subregion identifier (1-based).
        irrig_columns: Column indices, one per crop.
    """

    subregion_id: int
    irrig_columns: list[int] = field(default_factory=list)


@dataclass
class SoilMoisturePointerRow:
    """Soil-moisture column pointers for one subregion.

    Attributes:
        subregion_id: Subregion identifier (1-based).
        columns: Column indices, one per crop.
    """

    subregion_id: int
    columns: list[int] = field(default_factory=list)


@dataclass
class SupplyReturnReuseRow:
    """Water-supply, return-flow and reuse column pointers for one subregion.

    Attributes:
        subregion_id: Subregion identifier (1-based).
        supply_column: Water-supply column pointer.
        return_flow_column: Return-flow fraction column pointer.
        reuse_column: Reuse fraction column pointer.
    """

    subregion_id: int
    supply_column: int = 0
    return_flow_column: int = 0
    reuse_column: int = 0


@dataclass
class InitialConditionRow:
    """Initial soil-moisture conditions for one subregion.

    For each soil type there are two values:
    - fraction of initial soil moisture due to precipitation (0–1)
    - initial soil-moisture content (0–1)

    Attributes:
        subregion_id: Subregion identifier (1-based).
        precip_fractions: Precipitation fractions, one per soil type.
        moisture_contents: Soil-moisture contents, one per soil type.
    """

    subregion_id: int
    precip_fractions: list[float] = field(default_factory=list)
    moisture_contents: list[float] = field(default_factory=list)


@dataclass
class NonPondedCropConfig:
    """Configuration parsed from a non-ponded agricultural crop sub-file.

    Attributes:
        n_crops: Number of non-ponded crops.
        subregional_area_file: Path to subregional crop-area data file.
        elemental_area_file: Path to elemental agricultural area file.
        area_output_factor: Length / area output conversion factor.
        area_output_unit: Output unit label (e.g. "ACRES").
        area_output_file: Path to average-crop output file.
        root_depth_factor: Conversion factor for rooting depths.
        root_depths: Rooting depth for each crop (length ``n_crops``).
        curve_numbers: Curve-number rows (one per subregion).
        etc_pointers: ET-column pointers (one row per subregion).
        irrigation_period_file: Path to irrigation-period data file.
        irrigation_pointers: Irrigation-period column pointers.
        min_soil_moisture_file: Path to minimum soil-moisture data file.
        min_moisture_pointers: Minimum soil-moisture column pointers.
        target_soil_moisture_file: Path to target soil-moisture data file (optional).
        target_moisture_pointers: Target soil-moisture column pointers.
        water_demand_file: Path to agricultural water-demand data file (optional).
        demand_from_moisture_flag: Demand-computation flag (1 = begin, 2 = end).
        supply_return_reuse: Supply / return-flow / reuse rows.
        initial_conditions: Initial soil-moisture condition rows.
    """

    n_crops: int = 0
    subregional_area_file: Path | None = None
    elemental_area_file: Path | None = None
    area_output_factor: float = 1.0
    area_output_unit: str = ""
    area_output_file: Path | None = None
    root_depth_factor: float = 1.0
    root_depths: list[float] = field(default_factory=list)
    curve_numbers: list[CurveNumberRow] = field(default_factory=list)
    etc_pointers: list[EtcPointerRow] = field(default_factory=list)
    irrigation_period_file: Path | None = None
    irrigation_pointers: list[IrrigationPointerRow] = field(default_factory=list)
    min_soil_moisture_file: Path | None = None
    min_moisture_pointers: list[SoilMoisturePointerRow] = field(default_factory=list)
    target_soil_moisture_file: Path | None = None
    target_moisture_pointers: list[SoilMoisturePointerRow] = field(default_factory=list)
    water_demand_file: Path | None = None
    demand_from_moisture_flag: int = 1
    supply_return_reuse: list[SupplyReturnReuseRow] = field(default_factory=list)
    initial_conditions: list[InitialConditionRow] = field(default_factory=list)


# ── Reader ────────────────────────────────────────────────────────────


class NonPondedCropReader:
    """Reader for IWFM non-ponded agricultural crop sub-file.

    This parses the positional-sequential file referenced as *AGNPFL* in
    the RootZone component main file.  The format matches
    ``Class_AgLandUse_v50::New()`` in the Fortran source.

    Args:
        n_subregions: Number of subregions in the model.  If provided,
            tabular sections read exactly this many rows (skipping
            comments within).  If ``None``, sections are delimited by
            comment lines (i.e. a comment line after data rows ends
            the section).
    """

    def __init__(self, n_subregions: int | None = None) -> None:
        self._n_sub = n_subregions

    def read(
        self,
        filepath: Path | str,
        base_dir: Path | None = None,
    ) -> NonPondedCropConfig:
        """Parse the non-ponded crop file.

        Args:
            filepath: Path to the non-ponded crop file.
            base_dir: Base directory for resolving relative sub-file
                paths.  Defaults to the parent of *filepath*.

        Returns:
            Populated :class:`NonPondedCropConfig`.
        """
        filepath = Path(filepath)
        if base_dir is None:
            base_dir = filepath.parent

        with open(filepath) as f:
            lines = f.readlines()
        buf = _LineBuffer(lines)
        config = NonPondedCropConfig()

        # 1. NCrops
        val = buf.next_data()
        try:
            config.n_crops = int(val)
        except ValueError as exc:
            raise FileFormatError(
                f"Invalid NCrops value: '{val}'",
                line_number=buf.line_num,
            ) from exc

        # 2. Subregional crop-area data file
        val = buf.next_data_or_empty()
        if val:
            config.subregional_area_file = self._resolve(base_dir, val)

        # 3. Elemental agricultural area file
        val = buf.next_data_or_empty()
        if val:
            config.elemental_area_file = self._resolve(base_dir, val)

        # 4. Output conversion factor
        val = buf.next_data_or_empty()
        if val:
            try:
                config.area_output_factor = float(val)
            except ValueError:
                pass

        # 5. Output unit
        val = buf.next_data_or_empty()
        if val:
            config.area_output_unit = val

        # 6. Output file
        val = buf.next_data_or_empty()
        if val:
            config.area_output_file = self._resolve(base_dir, val)

        # 7. Root-depth conversion factor
        val = buf.next_data_or_empty()
        if val:
            try:
                config.root_depth_factor = float(val)
            except ValueError:
                pass

        # 8–7+NCrops.  One root depth per crop.
        for _ in range(config.n_crops):
            val = buf.next_data()
            try:
                config.root_depths.append(float(val) * config.root_depth_factor)
            except ValueError as exc:
                raise FileFormatError(
                    f"Invalid root depth: '{val}'",
                    line_number=buf.line_num,
                ) from exc

        # 9. Curve-number data (per subregion).
        config.curve_numbers = self._read_cn_rows(buf)

        # 10. ETc column pointers (per subregion).
        config.etc_pointers = self._read_pointer_rows(buf)

        # 11. Irrigation-period data file + pointer table.
        val = buf.next_data_or_empty()
        if val:
            config.irrigation_period_file = self._resolve(base_dir, val)
        config.irrigation_pointers = self._read_pointer_rows(buf)  # type: ignore[assignment]

        # 12. Minimum soil-moisture data file + pointer table.
        val = buf.next_data_or_empty()
        if val:
            config.min_soil_moisture_file = self._resolve(base_dir, val)
        config.min_moisture_pointers = self._read_pointer_rows(buf)  # type: ignore[assignment]

        # 13. Target soil-moisture data file (optional) + pointer table.
        val = buf.next_data_or_empty()
        if val:
            config.target_soil_moisture_file = self._resolve(base_dir, val)
            config.target_moisture_pointers = self._read_pointer_rows(buf)  # type: ignore[assignment]

        # 14. Water-demand data file (optional).
        val = buf.next_data_or_empty()
        if val:
            config.water_demand_file = self._resolve(base_dir, val)

        # 15. Demand-from-moisture flag.
        val = buf.next_data_or_empty()
        if val:
            try:
                config.demand_from_moisture_flag = int(val)
            except ValueError:
                pass

        # 16. Supply / return-flow / reuse pointers.
        config.supply_return_reuse = self._read_supply_rows(buf)

        # 17. Initial conditions.
        config.initial_conditions = self._read_initial_conditions(buf)

        return config

    # ── internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _resolve(base_dir: Path, filepath: str) -> Path:
        p = Path(filepath.strip())
        if p.is_absolute():
            return p
        return base_dir / p

    # ── tabular-data helpers ──────────────────────────────────────────

    def _read_rows(self, buf: _LineBuffer, min_cols: int) -> list[list[str]]:
        """Read tabular data rows from the buffer.

        If ``n_subregions`` was provided, reads exactly that many data
        rows (skipping comments within).  Otherwise, reads data rows
        until a comment line or a line with fewer than *min_cols*
        columns is encountered, pushing that line back.

        Returns:
            List of split-value lists (one per row).
        """
        rows: list[list[str]] = []
        n_expected = self._n_sub
        started = False

        while True:
            raw = buf.next_line()
            if raw is None:
                break

            if _is_comment_line(raw):
                if n_expected is not None:
                    # Known count: skip comments within the section
                    continue
                if started:
                    # Unknown count: comment after data = section end
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

    def _read_cn_rows(self, buf: _LineBuffer) -> list[CurveNumberRow]:
        raw_rows = self._read_rows(buf, min_cols=2)
        result: list[CurveNumberRow] = []
        for parts in raw_rows:
            try:
                sub_id = int(parts[0])
                vals = [float(v) for v in parts[1:]]
                result.append(CurveNumberRow(subregion_id=sub_id, cn_values=vals))
            except ValueError:
                break
        return result

    def _read_pointer_rows(self, buf: _LineBuffer) -> list[EtcPointerRow]:
        raw_rows = self._read_rows(buf, min_cols=2)
        result: list[EtcPointerRow] = []
        for parts in raw_rows:
            try:
                sub_id = int(parts[0])
                cols = [int(float(v)) for v in parts[1:]]
                result.append(EtcPointerRow(subregion_id=sub_id, etc_columns=cols))
            except ValueError:
                break
        return result

    def _read_supply_rows(self, buf: _LineBuffer) -> list[SupplyReturnReuseRow]:
        raw_rows = self._read_rows(buf, min_cols=4)
        result: list[SupplyReturnReuseRow] = []
        for parts in raw_rows:
            try:
                sub_id = int(parts[0])
                supply = int(float(parts[1]))
                ret = int(float(parts[2]))
                reuse = int(float(parts[3]))
                result.append(
                    SupplyReturnReuseRow(
                        subregion_id=sub_id,
                        supply_column=supply,
                        return_flow_column=ret,
                        reuse_column=reuse,
                    )
                )
            except ValueError:
                break
        return result

    def _read_initial_conditions(self, buf: _LineBuffer) -> list[InitialConditionRow]:
        raw_rows = self._read_rows(buf, min_cols=3)
        result: list[InitialConditionRow] = []
        for parts in raw_rows:
            try:
                sub_id = int(parts[0])
                data = [float(v) for v in parts[1:]]
                pf = data[0::2]
                mc = data[1::2]
                result.append(
                    InitialConditionRow(
                        subregion_id=sub_id,
                        precip_fractions=pf,
                        moisture_contents=mc,
                    )
                )
            except ValueError:
                break
        return result


# ── convenience function ──────────────────────────────────────────────


def read_nonponded_crop(
    filepath: Path | str,
    base_dir: Path | None = None,
    n_subregions: int | None = None,
) -> NonPondedCropConfig:
    """Read an IWFM non-ponded agricultural crop sub-file.

    Args:
        filepath: Path to the non-ponded crop file.
        base_dir: Base directory for resolving relative paths.
        n_subregions: Number of subregions (for exact row counts).

    Returns:
        :class:`NonPondedCropConfig` with parsed values.
    """
    reader = NonPondedCropReader(n_subregions=n_subregions)
    return reader.read(filepath, base_dir)
