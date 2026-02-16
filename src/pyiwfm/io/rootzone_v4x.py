"""
v4.x sub-file readers and writers for IWFM RootZone component.

This module handles the per-element, v4.0-v4.13 Fortran file formats
for the four land-use sub-files (non-ponded ag, ponded ag, urban,
native/riparian).  These formats differ from the v5.0/subregion-based
readers in ``rootzone_nonponded.py`` etc.

The existing v5.0 readers and their 1100+ passing tests are left
untouched; this module provides parallel v4.x-specific I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.io.iwfm_reader import (
    COMMENT_CHARS,
    LineBuffer as _LineBuffer,
    is_comment_line as _is_comment_line,
    strip_inline_comment as _parse_value_line,
)


# =====================================================================
# Helper data classes
# =====================================================================


@dataclass
class RootDepthRow:
    """Root depth entry for one crop."""

    crop_index: int
    max_root_depth: float
    fractions_column: int


@dataclass
class ElementCropRow:
    """Per-element row with one value per crop (or per category)."""

    element_id: int
    values: list[float] = field(default_factory=list)


@dataclass
class AgInitialConditionRow:
    """Initial condition row: element_id, precip_fraction, MC per crop."""

    element_id: int
    precip_fraction: float
    moisture_contents: list[float] = field(default_factory=list)


# =====================================================================
# Non-ponded agricultural config (v4.x)
# =====================================================================


@dataclass
class NonPondedCropConfigV4x:
    """Configuration parsed from a v4.x non-ponded agricultural sub-file."""

    n_crops: int = 0
    demand_from_moisture_flag: int = 1
    crop_codes: list[str] = field(default_factory=list)
    area_data_file: Path | None = None
    n_budget_crops: int = 0
    budget_crop_codes: list[str] = field(default_factory=list)
    lwu_budget_file: Path | None = None
    rz_budget_file: Path | None = None
    root_depth_fractions_file: Path | None = None
    root_depth_factor: float = 1.0
    root_depth_data: list[RootDepthRow] = field(default_factory=list)
    curve_numbers: list[ElementCropRow] = field(default_factory=list)
    etc_pointers: list[ElementCropRow] = field(default_factory=list)
    supply_req_pointers: list[ElementCropRow] = field(default_factory=list)
    irrigation_pointers: list[ElementCropRow] = field(default_factory=list)
    min_soil_moisture_file: Path | None = None
    min_moisture_pointers: list[ElementCropRow] = field(default_factory=list)
    target_soil_moisture_file: Path | None = None
    target_moisture_pointers: list[ElementCropRow] = field(default_factory=list)
    return_flow_pointers: list[ElementCropRow] = field(default_factory=list)
    reuse_pointers: list[ElementCropRow] = field(default_factory=list)
    leaching_factors_file: Path | None = None
    leaching_pointers: list[ElementCropRow] = field(default_factory=list)
    initial_conditions: list[AgInitialConditionRow] = field(default_factory=list)


# =====================================================================
# Ponded agricultural config (v4.x)
# =====================================================================


@dataclass
class PondedCropConfigV4x:
    """Configuration parsed from a v4.x ponded agricultural sub-file.

    Fixed 5 crop types: rice (3 decomp variants) + refuge (2 types).
    """

    area_data_file: Path | None = None
    n_budget_crops: int = 0
    budget_crop_codes: list[str] = field(default_factory=list)
    lwu_budget_file: Path | None = None
    rz_budget_file: Path | None = None
    root_depth_factor: float = 1.0
    root_depths: list[float] = field(default_factory=list)  # 5 values
    curve_numbers: list[ElementCropRow] = field(default_factory=list)
    etc_pointers: list[ElementCropRow] = field(default_factory=list)
    supply_req_pointers: list[ElementCropRow] = field(default_factory=list)
    irrigation_pointers: list[ElementCropRow] = field(default_factory=list)
    ponding_depth_file: Path | None = None
    operations_flow_file: Path | None = None
    ponding_depth_pointers: list[ElementCropRow] = field(default_factory=list)
    decomp_water_pointers: list[ElementCropRow] = field(default_factory=list)
    return_flow_pointers: list[ElementCropRow] = field(default_factory=list)
    reuse_pointers: list[ElementCropRow] = field(default_factory=list)
    initial_conditions: list[AgInitialConditionRow] = field(default_factory=list)


# =====================================================================
# Urban config (v4.x)
# =====================================================================


@dataclass
class UrbanElementRowV4x:
    """Per-element urban data (10-column row)."""

    element_id: int
    pervious_fraction: float
    curve_number: float
    pop_column: int
    per_cap_column: int
    demand_fraction: float
    etc_column: int
    return_flow_column: int
    reuse_column: int
    water_use_column: int


@dataclass
class UrbanInitialRowV4x:
    """Urban initial conditions: element_id, precip_frac, moisture."""

    element_id: int
    precip_fraction: float
    moisture_content: float


@dataclass
class UrbanConfigV4x:
    """Configuration parsed from a v4.x urban land-use sub-file."""

    area_data_file: Path | None = None
    root_depth_factor: float = 1.0
    root_depth: float = 0.0
    population_file: Path | None = None
    per_capita_water_use_file: Path | None = None
    water_use_specs_file: Path | None = None
    element_data: list[UrbanElementRowV4x] = field(default_factory=list)
    initial_conditions: list[UrbanInitialRowV4x] = field(default_factory=list)


# =====================================================================
# Native/riparian config (v4.x)
# =====================================================================


@dataclass
class NativeRiparianElementRowV4x:
    """Per-element native/riparian data (5 or 6-column row).

    The 6th column (``riparian_stream_node``) is the stream node index
    for riparian ET from stream (ISTRMRV), present in v4.1+ files.
    """

    element_id: int
    native_cn: float
    riparian_cn: float
    native_etc_column: int
    riparian_etc_column: int
    riparian_stream_node: int = 0


@dataclass
class NativeRiparianInitialRowV4x:
    """Native/riparian initial conditions (3-column row)."""

    element_id: int
    native_moisture: float
    riparian_moisture: float


@dataclass
class NativeRiparianConfigV4x:
    """Configuration parsed from a v4.x native/riparian sub-file."""

    area_data_file: Path | None = None
    root_depth_factor: float = 1.0
    native_root_depth: float = 0.0
    riparian_root_depth: float = 0.0
    element_data: list[NativeRiparianElementRowV4x] = field(default_factory=list)
    initial_conditions: list[NativeRiparianInitialRowV4x] = field(
        default_factory=list
    )


# =====================================================================
# Base reader helper
# =====================================================================


class _V4xReaderBase:
    """Shared helpers for v4.x sub-file readers."""

    def __init__(self, n_elements: int = 0) -> None:
        self.n_elements = n_elements

    @staticmethod
    def _make_buffer(filepath: Path | str) -> _LineBuffer:
        filepath = Path(filepath)
        with open(filepath, "r") as fh:
            lines = fh.readlines()
        return _LineBuffer(lines)

    @staticmethod
    def _resolve_path(base_dir: Path | None, raw: str) -> Path | None:
        raw = raw.strip()
        if not raw:
            return None
        p = Path(raw)
        if p.is_absolute() or base_dir is None:
            return p
        return base_dir / p

    def _read_element_table(
        self, buf: _LineBuffer, n_values: int
    ) -> list[ElementCropRow]:
        """Read *n_elements* rows each with *element_id + n_values* cols.

        Handles the IWFM ``IE=0`` shorthand: when the first row has
        ``element_id == 0``, a single row of values applies to every
        element and only one data line is present in the file.
        """
        rows: list[ElementCropRow] = []

        # Read first row to check for IE=0 shorthand
        line = buf.next_data()
        parts = line.split()
        elem_id = int(parts[0])
        vals = [float(v) for v in parts[1 : 1 + n_values]]

        if elem_id == 0:
            # IE=0: single row applies to all elements
            for eid in range(1, self.n_elements + 1):
                rows.append(ElementCropRow(element_id=eid, values=list(vals)))
            return rows

        # Normal path: first row is element 1, read remaining
        rows.append(ElementCropRow(element_id=elem_id, values=vals))
        for _ in range(self.n_elements - 1):
            line = buf.next_data()
            parts = line.split()
            elem_id = int(parts[0])
            vals = [float(v) for v in parts[1 : 1 + n_values]]
            rows.append(ElementCropRow(element_id=elem_id, values=vals))
        return rows

    def _read_ag_initial_conditions(
        self, buf: _LineBuffer, n_crops: int
    ) -> list[AgInitialConditionRow]:
        """Read NElements rows: elem_id, precip_frac, MC_crop1..N."""
        rows: list[AgInitialConditionRow] = []
        for _ in range(self.n_elements):
            line = buf.next_data()
            parts = line.split()
            elem_id = int(parts[0])
            precip_frac = float(parts[1])
            mcs = [float(v) for v in parts[2 : 2 + n_crops]]
            rows.append(
                AgInitialConditionRow(
                    element_id=elem_id,
                    precip_fraction=precip_frac,
                    moisture_contents=mcs,
                )
            )
        return rows


# =====================================================================
# Non-ponded crop reader (v4.x)
# =====================================================================


class NonPondedCropReaderV4x(_V4xReaderBase):
    """Reader for IWFM v4.x non-ponded agricultural crop sub-file.

    Reads the 24-section file produced by
    ``Class_NonPondedAgLandUse_v40::New()``.
    """

    def read(
        self,
        filepath: Path | str,
        base_dir: Path | None = None,
    ) -> NonPondedCropConfigV4x:
        filepath = Path(filepath)
        if base_dir is None:
            base_dir = filepath.parent
        buf = self._make_buffer(filepath)
        cfg = NonPondedCropConfigV4x()

        # 1. NCrops
        cfg.n_crops = int(buf.next_data())

        # 2. Demand-from-moisture flag
        cfg.demand_from_moisture_flag = int(buf.next_data())

        # 3. Crop codes (NCrops lines)
        for _ in range(cfg.n_crops):
            cfg.crop_codes.append(buf.next_data().strip())

        # 4. Area data file
        cfg.area_data_file = self._resolve_path(base_dir, buf.next_data())

        # 5. NBudgetCrops
        cfg.n_budget_crops = int(buf.next_data())

        # 6-8. Budget crop codes and files
        if cfg.n_budget_crops > 0:
            for _ in range(cfg.n_budget_crops):
                cfg.budget_crop_codes.append(buf.next_data().strip())
            cfg.lwu_budget_file = self._resolve_path(base_dir, buf.next_data())
            cfg.rz_budget_file = self._resolve_path(base_dir, buf.next_data())

        # 9. Root depth fractions file
        cfg.root_depth_fractions_file = self._resolve_path(
            base_dir, buf.next_data()
        )

        # 10. Root depth factor
        cfg.root_depth_factor = float(buf.next_data())

        # 11. Root depth data (NCrops rows)
        for _ in range(cfg.n_crops):
            line = buf.next_data()
            parts = line.split()
            cfg.root_depth_data.append(
                RootDepthRow(
                    crop_index=int(parts[0]),
                    max_root_depth=float(parts[1]),
                    fractions_column=int(parts[2]),
                )
            )

        # 12. Curve numbers (NElements rows x NCrops+1 cols)
        cfg.curve_numbers = self._read_element_table(buf, cfg.n_crops)

        # 13. ETc pointers
        cfg.etc_pointers = self._read_element_table(buf, cfg.n_crops)

        # 14. Water supply req pointers
        cfg.supply_req_pointers = self._read_element_table(buf, cfg.n_crops)

        # 15. Irrigation period pointers
        cfg.irrigation_pointers = self._read_element_table(buf, cfg.n_crops)

        # 16. Min soil moisture file
        cfg.min_soil_moisture_file = self._resolve_path(
            base_dir, buf.next_data()
        )

        # 17. Min moisture pointers
        cfg.min_moisture_pointers = self._read_element_table(buf, cfg.n_crops)

        # 18. Target soil moisture file (optional)
        target_raw = buf.next_data_or_empty()
        cfg.target_soil_moisture_file = self._resolve_path(base_dir, target_raw)

        # 19. Target moisture pointers (if target file given)
        if cfg.target_soil_moisture_file is not None:
            cfg.target_moisture_pointers = self._read_element_table(
                buf, cfg.n_crops
            )

        # 20. Return flow fraction pointers
        cfg.return_flow_pointers = self._read_element_table(buf, cfg.n_crops)

        # 21. Reuse fraction pointers
        cfg.reuse_pointers = self._read_element_table(buf, cfg.n_crops)

        # 22. Leaching factors file (optional)
        leach_raw = buf.next_data_or_empty()
        cfg.leaching_factors_file = self._resolve_path(base_dir, leach_raw)

        # 23. Leaching factor pointers (if file given)
        if cfg.leaching_factors_file is not None:
            cfg.leaching_pointers = self._read_element_table(buf, cfg.n_crops)

        # 24. Initial conditions
        cfg.initial_conditions = self._read_ag_initial_conditions(
            buf, cfg.n_crops
        )

        return cfg


# =====================================================================
# Ponded crop reader (v4.x)
# =====================================================================


class PondedCropReaderV4x(_V4xReaderBase):
    """Reader for IWFM v4.x ponded agricultural crop sub-file.

    Fixed 5 crop types (3 rice + 2 refuge).
    """

    N_PONDED_CROPS = 5

    def read(
        self,
        filepath: Path | str,
        base_dir: Path | None = None,
    ) -> PondedCropConfigV4x:
        filepath = Path(filepath)
        if base_dir is None:
            base_dir = filepath.parent
        buf = self._make_buffer(filepath)
        cfg = PondedCropConfigV4x()
        nc = self.N_PONDED_CROPS

        # Area data file
        cfg.area_data_file = self._resolve_path(base_dir, buf.next_data())

        # NBudgetCrops
        cfg.n_budget_crops = int(buf.next_data())

        # Budget crop codes and files
        if cfg.n_budget_crops > 0:
            for _ in range(cfg.n_budget_crops):
                cfg.budget_crop_codes.append(buf.next_data().strip())
            cfg.lwu_budget_file = self._resolve_path(base_dir, buf.next_data())
            cfg.rz_budget_file = self._resolve_path(base_dir, buf.next_data())

        # Root depth factor
        cfg.root_depth_factor = float(buf.next_data())

        # 5 root depths
        for _ in range(nc):
            cfg.root_depths.append(float(buf.next_data()))

        # Curve numbers
        cfg.curve_numbers = self._read_element_table(buf, nc)

        # ETc pointers
        cfg.etc_pointers = self._read_element_table(buf, nc)

        # Supply req pointers
        cfg.supply_req_pointers = self._read_element_table(buf, nc)

        # Irrigation period pointers
        cfg.irrigation_pointers = self._read_element_table(buf, nc)

        # Ponding depth file
        cfg.ponding_depth_file = self._resolve_path(base_dir, buf.next_data())

        # Operations flow file
        cfg.operations_flow_file = self._resolve_path(base_dir, buf.next_data())

        # Ponding depth pointers
        cfg.ponding_depth_pointers = self._read_element_table(buf, nc)

        # Decomp water pointers — Fortran reads 2 cols (elem_id + 1 value)
        # when lReadNCrops=.FALSE. (the v4.1 default)
        cfg.decomp_water_pointers = self._read_element_table(buf, 1)

        # Return flow pointers
        cfg.return_flow_pointers = self._read_element_table(buf, nc)

        # Reuse pointers
        cfg.reuse_pointers = self._read_element_table(buf, nc)

        # Initial conditions
        cfg.initial_conditions = self._read_ag_initial_conditions(buf, nc)

        return cfg


# =====================================================================
# Urban reader (v4.x)
# =====================================================================


class UrbanReaderV4x(_V4xReaderBase):
    """Reader for IWFM v4.x urban land-use sub-file."""

    def read(
        self,
        filepath: Path | str,
        base_dir: Path | None = None,
    ) -> UrbanConfigV4x:
        filepath = Path(filepath)
        if base_dir is None:
            base_dir = filepath.parent
        buf = self._make_buffer(filepath)
        cfg = UrbanConfigV4x()

        # Area data file
        cfg.area_data_file = self._resolve_path(base_dir, buf.next_data())

        # Root depth factor
        cfg.root_depth_factor = float(buf.next_data())

        # Root depth
        cfg.root_depth = float(buf.next_data())

        # Population file
        cfg.population_file = self._resolve_path(base_dir, buf.next_data())

        # Per-capita water use file
        cfg.per_capita_water_use_file = self._resolve_path(
            base_dir, buf.next_data()
        )

        # Water use specs file
        cfg.water_use_specs_file = self._resolve_path(
            base_dir, buf.next_data()
        )

        # Element data (NElements rows x 10 cols)
        for _ in range(self.n_elements):
            line = buf.next_data()
            parts = line.split()
            cfg.element_data.append(
                UrbanElementRowV4x(
                    element_id=int(parts[0]),
                    pervious_fraction=float(parts[1]),
                    curve_number=float(parts[2]),
                    pop_column=int(parts[3]),
                    per_cap_column=int(parts[4]),
                    demand_fraction=float(parts[5]),
                    etc_column=int(parts[6]),
                    return_flow_column=int(parts[7]),
                    reuse_column=int(parts[8]),
                    water_use_column=int(parts[9]),
                )
            )

        # Initial conditions (NElements rows x 3 cols)
        for _ in range(self.n_elements):
            line = buf.next_data()
            parts = line.split()
            cfg.initial_conditions.append(
                UrbanInitialRowV4x(
                    element_id=int(parts[0]),
                    precip_fraction=float(parts[1]),
                    moisture_content=float(parts[2]),
                )
            )

        return cfg


# =====================================================================
# Native/riparian reader (v4.x)
# =====================================================================


class NativeRiparianReaderV4x(_V4xReaderBase):
    """Reader for IWFM v4.x native/riparian vegetation sub-file."""

    def read(
        self,
        filepath: Path | str,
        base_dir: Path | None = None,
    ) -> NativeRiparianConfigV4x:
        filepath = Path(filepath)
        if base_dir is None:
            base_dir = filepath.parent
        buf = self._make_buffer(filepath)
        cfg = NativeRiparianConfigV4x()

        # Area data file
        cfg.area_data_file = self._resolve_path(base_dir, buf.next_data())

        # Root depth factor
        cfg.root_depth_factor = float(buf.next_data())

        # Native root depth
        cfg.native_root_depth = float(buf.next_data())

        # Riparian root depth
        cfg.riparian_root_depth = float(buf.next_data())

        # Element data (NElements rows x 5 or 6 cols)
        for _ in range(self.n_elements):
            line = buf.next_data()
            parts = line.split()
            cfg.element_data.append(
                NativeRiparianElementRowV4x(
                    element_id=int(parts[0]),
                    native_cn=float(parts[1]),
                    riparian_cn=float(parts[2]),
                    native_etc_column=int(parts[3]),
                    riparian_etc_column=int(parts[4]),
                    riparian_stream_node=(
                        int(parts[5]) if len(parts) > 5 else 0
                    ),
                )
            )

        # Initial conditions (NElements rows x 3 cols)
        for _ in range(self.n_elements):
            line = buf.next_data()
            parts = line.split()
            cfg.initial_conditions.append(
                NativeRiparianInitialRowV4x(
                    element_id=int(parts[0]),
                    native_moisture=float(parts[1]),
                    riparian_moisture=float(parts[2]),
                )
            )

        return cfg


# =====================================================================
# Writers (v4.x)
# =====================================================================


class _V4xWriterBase:
    """Shared helpers for v4.x sub-file writers."""

    @staticmethod
    def _write_comment(f: TextIO, text: str) -> None:
        f.write(f"C  {text}\n")

    @staticmethod
    def _write_data(f: TextIO, value: str, desc: str = "") -> None:
        if desc:
            f.write(f"   {value:<40} / {desc}\n")
        else:
            f.write(f"   {value}\n")

    @staticmethod
    def _write_path(f: TextIO, path: Path | None, desc: str = "") -> None:
        val = str(path) if path else ""
        if desc:
            f.write(f"   {val:<40} / {desc}\n")
        else:
            f.write(f"   {val}\n")

    @staticmethod
    def _write_element_table(
        f: TextIO, rows: list[ElementCropRow]
    ) -> None:
        for row in rows:
            vals = " ".join(f"{v:>10.4f}" for v in row.values)
            f.write(f"   {row.element_id:<6} {vals}\n")

    @staticmethod
    def _write_ag_initial_conditions(
        f: TextIO, rows: list[AgInitialConditionRow]
    ) -> None:
        for row in rows:
            mcs = " ".join(f"{v:>10.6f}" for v in row.moisture_contents)
            f.write(
                f"   {row.element_id:<6} {row.precip_fraction:>10.4f} {mcs}\n"
            )


class NonPondedCropWriterV4x(_V4xWriterBase):
    """Writer for IWFM v4.x non-ponded agricultural crop sub-file."""

    def write(self, cfg: NonPondedCropConfigV4x, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            self._write_comment(f, "Non-ponded agricultural crop file (v4.x)")
            self._write_comment(f, "Generated by pyiwfm")
            self._write_comment(f, "")

            # 1. NCrops
            self._write_data(f, str(cfg.n_crops), "NCROPS")

            # 2. Demand-from-moisture flag
            self._write_data(
                f, str(cfg.demand_from_moisture_flag), "DEMAND_FROM_MOISTURE"
            )

            # 3. Crop codes
            self._write_comment(f, "Crop codes")
            for code in cfg.crop_codes:
                self._write_data(f, code)

            # 4. Area data file
            self._write_path(f, cfg.area_data_file, "AREA_FILE")

            # 5. NBudgetCrops
            self._write_data(f, str(cfg.n_budget_crops), "NBUDGETCROPS")

            # 6-8. Budget crop codes and files
            if cfg.n_budget_crops > 0:
                for code in cfg.budget_crop_codes:
                    self._write_data(f, code)
                self._write_path(f, cfg.lwu_budget_file, "LWU_BUDGET")
                self._write_path(f, cfg.rz_budget_file, "RZ_BUDGET")

            # 9. Root depth fractions file
            self._write_path(
                f, cfg.root_depth_fractions_file, "ROOT_DEPTH_FRAC_FILE"
            )

            # 10. Root depth factor
            self._write_data(f, f"{cfg.root_depth_factor:.4f}", "ROOT_DEPTH_FACTOR")

            # 11. Root depth data
            self._write_comment(f, "Root depth data: index  max_depth  frac_col")
            for rd in cfg.root_depth_data:
                f.write(
                    f"   {rd.crop_index:<6} {rd.max_root_depth:>10.4f}"
                    f" {rd.fractions_column:>6}\n"
                )

            # 12. Curve numbers
            self._write_comment(f, "Curve numbers")
            self._write_element_table(f, cfg.curve_numbers)

            # 13. ETc pointers
            self._write_comment(f, "ETc pointers")
            self._write_element_table(f, cfg.etc_pointers)

            # 14. Supply req pointers
            self._write_comment(f, "Water supply requirement pointers")
            self._write_element_table(f, cfg.supply_req_pointers)

            # 15. Irrigation period pointers
            self._write_comment(f, "Irrigation period pointers")
            self._write_element_table(f, cfg.irrigation_pointers)

            # 16. Min soil moisture file
            self._write_path(
                f, cfg.min_soil_moisture_file, "MIN_SOIL_MOISTURE_FILE"
            )

            # 17. Min moisture pointers
            self._write_comment(f, "Min moisture pointers")
            self._write_element_table(f, cfg.min_moisture_pointers)

            # 18. Target soil moisture file
            self._write_path(
                f, cfg.target_soil_moisture_file, "TARGET_SOIL_MOISTURE_FILE"
            )

            # 19. Target moisture pointers
            if cfg.target_soil_moisture_file is not None:
                self._write_comment(f, "Target moisture pointers")
                self._write_element_table(f, cfg.target_moisture_pointers)

            # 20. Return flow pointers
            self._write_comment(f, "Return flow fraction pointers")
            self._write_element_table(f, cfg.return_flow_pointers)

            # 21. Reuse pointers
            self._write_comment(f, "Reuse fraction pointers")
            self._write_element_table(f, cfg.reuse_pointers)

            # 22. Leaching factors file
            self._write_path(
                f, cfg.leaching_factors_file, "LEACHING_FACTORS_FILE"
            )

            # 23. Leaching factor pointers
            if cfg.leaching_factors_file is not None:
                self._write_comment(f, "Leaching factor pointers")
                self._write_element_table(f, cfg.leaching_pointers)

            # 24. Initial conditions
            self._write_comment(f, "Initial conditions")
            self._write_ag_initial_conditions(f, cfg.initial_conditions)


class PondedCropWriterV4x(_V4xWriterBase):
    """Writer for IWFM v4.x ponded agricultural crop sub-file."""

    def write(self, cfg: PondedCropConfigV4x, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        nc = 5
        with open(filepath, "w") as f:
            self._write_comment(f, "Ponded agricultural crop file (v4.x)")
            self._write_comment(f, "Generated by pyiwfm")
            self._write_comment(f, "")

            # Area data file
            self._write_path(f, cfg.area_data_file, "AREA_FILE")

            # NBudgetCrops
            self._write_data(f, str(cfg.n_budget_crops), "NBUDGETCROPS")

            if cfg.n_budget_crops > 0:
                for code in cfg.budget_crop_codes:
                    self._write_data(f, code)
                self._write_path(f, cfg.lwu_budget_file, "LWU_BUDGET")
                self._write_path(f, cfg.rz_budget_file, "RZ_BUDGET")

            # Root depth factor
            self._write_data(f, f"{cfg.root_depth_factor:.4f}", "ROOT_DEPTH_FACTOR")

            # 5 root depths
            self._write_comment(f, "Root depths (5 ponded crop types)")
            for rd in cfg.root_depths:
                self._write_data(f, f"{rd:.4f}")

            # Tables
            self._write_comment(f, "Curve numbers")
            self._write_element_table(f, cfg.curve_numbers)

            self._write_comment(f, "ETc pointers")
            self._write_element_table(f, cfg.etc_pointers)

            self._write_comment(f, "Supply req pointers")
            self._write_element_table(f, cfg.supply_req_pointers)

            self._write_comment(f, "Irrigation period pointers")
            self._write_element_table(f, cfg.irrigation_pointers)

            # Ponding depth file
            self._write_path(f, cfg.ponding_depth_file, "PONDING_DEPTH_FILE")

            # Operations flow file
            self._write_path(f, cfg.operations_flow_file, "OPERATIONS_FLOW_FILE")

            # Ponding depth pointers
            self._write_comment(f, "Ponding depth pointers")
            self._write_element_table(f, cfg.ponding_depth_pointers)

            # Decomposition water pointers
            self._write_comment(f, "Decomposition water pointers")
            self._write_element_table(f, cfg.decomp_water_pointers)

            # Return flow pointers
            self._write_comment(f, "Return flow fraction pointers")
            self._write_element_table(f, cfg.return_flow_pointers)

            # Reuse pointers
            self._write_comment(f, "Reuse fraction pointers")
            self._write_element_table(f, cfg.reuse_pointers)

            # Initial conditions
            self._write_comment(f, "Initial conditions")
            self._write_ag_initial_conditions(f, cfg.initial_conditions)


class UrbanWriterV4x(_V4xWriterBase):
    """Writer for IWFM v4.x urban land-use sub-file."""

    def write(self, cfg: UrbanConfigV4x, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            self._write_comment(f, "Urban land use file (v4.x)")
            self._write_comment(f, "Generated by pyiwfm")
            self._write_comment(f, "")

            self._write_path(f, cfg.area_data_file, "AREA_FILE")
            self._write_data(f, f"{cfg.root_depth_factor:.4f}", "ROOT_DEPTH_FACTOR")
            self._write_data(f, f"{cfg.root_depth:.4f}", "ROOT_DEPTH")
            self._write_path(f, cfg.population_file, "POPULATION_FILE")
            self._write_path(
                f, cfg.per_capita_water_use_file, "PER_CAPITA_WATER_USE_FILE"
            )
            self._write_path(f, cfg.water_use_specs_file, "WATER_USE_SPECS_FILE")

            # Element data
            self._write_comment(
                f,
                "IE  PervFrac  CN  PopCol  PerCapCol  DemFrac"
                "  EtcCol  RetFlowCol  ReuseCol  WaterUseCol",
            )
            for row in cfg.element_data:
                f.write(
                    f"   {row.element_id:<6}"
                    f" {row.pervious_fraction:>8.4f}"
                    f" {row.curve_number:>8.2f}"
                    f" {row.pop_column:>6}"
                    f" {row.per_cap_column:>6}"
                    f" {row.demand_fraction:>8.4f}"
                    f" {row.etc_column:>6}"
                    f" {row.return_flow_column:>6}"
                    f" {row.reuse_column:>6}"
                    f" {row.water_use_column:>6}\n"
                )

            # Initial conditions
            self._write_comment(f, "Initial conditions: IE  PrecipFrac  MC")
            for row in cfg.initial_conditions:
                f.write(
                    f"   {row.element_id:<6}"
                    f" {row.precip_fraction:>10.4f}"
                    f" {row.moisture_content:>10.6f}\n"
                )


class NativeRiparianWriterV4x(_V4xWriterBase):
    """Writer for IWFM v4.x native/riparian vegetation sub-file."""

    def write(self, cfg: NativeRiparianConfigV4x, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            self._write_comment(f, "Native/riparian vegetation file (v4.x)")
            self._write_comment(f, "Generated by pyiwfm")
            self._write_comment(f, "")

            self._write_path(f, cfg.area_data_file, "AREA_FILE")
            self._write_data(
                f, f"{cfg.root_depth_factor:.4f}", "ROOT_DEPTH_FACTOR"
            )
            self._write_data(
                f, f"{cfg.native_root_depth:.4f}", "NATIVE_ROOT_DEPTH"
            )
            self._write_data(
                f, f"{cfg.riparian_root_depth:.4f}", "RIPARIAN_ROOT_DEPTH"
            )

            # Element data — include ISTRMRV column if any row has it
            has_stream_node = any(
                row.riparian_stream_node != 0 for row in cfg.element_data
            )
            if has_stream_node:
                self._write_comment(
                    f,
                    "IE  NativeCN  RiparianCN  NativeEtcCol"
                    "  RiparianEtcCol  ISTRMRV",
                )
                for row in cfg.element_data:
                    f.write(
                        f"   {row.element_id:<6}"
                        f" {row.native_cn:>8.2f}"
                        f" {row.riparian_cn:>8.2f}"
                        f" {row.native_etc_column:>6}"
                        f" {row.riparian_etc_column:>6}"
                        f" {row.riparian_stream_node:>6}\n"
                    )
            else:
                self._write_comment(
                    f,
                    "IE  NativeCN  RiparianCN  NativeEtcCol  RiparianEtcCol",
                )
                for row in cfg.element_data:
                    f.write(
                        f"   {row.element_id:<6}"
                        f" {row.native_cn:>8.2f}"
                        f" {row.riparian_cn:>8.2f}"
                        f" {row.native_etc_column:>6}"
                        f" {row.riparian_etc_column:>6}\n"
                    )

            # Initial conditions
            self._write_comment(
                f, "Initial conditions: IE  NativeMC  RiparianMC"
            )
            for row in cfg.initial_conditions:
                f.write(
                    f"   {row.element_id:<6}"
                    f" {row.native_moisture:>10.6f}"
                    f" {row.riparian_moisture:>10.6f}\n"
                )


# =====================================================================
# Convenience functions
# =====================================================================


def read_nonponded_v4x(
    filepath: Path | str,
    base_dir: Path | None = None,
    n_elements: int = 0,
) -> NonPondedCropConfigV4x:
    """Read a v4.x non-ponded agricultural crop sub-file."""
    reader = NonPondedCropReaderV4x(n_elements=n_elements)
    return reader.read(filepath, base_dir)


def read_ponded_v4x(
    filepath: Path | str,
    base_dir: Path | None = None,
    n_elements: int = 0,
) -> PondedCropConfigV4x:
    """Read a v4.x ponded agricultural crop sub-file."""
    reader = PondedCropReaderV4x(n_elements=n_elements)
    return reader.read(filepath, base_dir)


def read_urban_v4x(
    filepath: Path | str,
    base_dir: Path | None = None,
    n_elements: int = 0,
) -> UrbanConfigV4x:
    """Read a v4.x urban land-use sub-file."""
    reader = UrbanReaderV4x(n_elements=n_elements)
    return reader.read(filepath, base_dir)


def read_native_riparian_v4x(
    filepath: Path | str,
    base_dir: Path | None = None,
    n_elements: int = 0,
) -> NativeRiparianConfigV4x:
    """Read a v4.x native/riparian vegetation sub-file."""
    reader = NativeRiparianReaderV4x(n_elements=n_elements)
    return reader.read(filepath, base_dir)
