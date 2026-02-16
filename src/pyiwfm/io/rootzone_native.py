"""
Native / riparian vegetation sub-file reader for IWFM RootZone component.

This module reads the IWFM native and riparian vegetation file (NVRVFL),
which is referenced by the RootZone component main file.  The file
contains land-use area references, root depths for native and riparian
vegetation, curve numbers for both vegetation types, ET-column pointers,
and initial soil-moisture conditions.

Format matches ``Class_NativeRiparianLandUse_v50::New()`` in the
Fortran source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.io.iwfm_reader import (
    COMMENT_CHARS,
    LineBuffer as _LineBuffer,
    is_comment_line as _is_comment_line,
    strip_inline_comment as _parse_value_line,
)


# ── Data classes ──────────────────────────────────────────────────────


@dataclass
class NativeRiparianCNRow:
    """Curve-number data for one subregion (native + riparian).

    Attributes:
        subregion_id: Subregion identifier (1-based).
        native_cn: Curve-number values for native vegetation, one per soil.
        riparian_cn: Curve-number values for riparian vegetation, one per soil.
    """

    subregion_id: int
    native_cn: list[float] = field(default_factory=list)
    riparian_cn: list[float] = field(default_factory=list)


@dataclass
class NativeRiparianEtcRow:
    """ET-column pointers for one subregion.

    Attributes:
        subregion_id: Subregion identifier (1-based).
        native_etc_column: Column index for native vegetation ETc.
        riparian_etc_column: Column index for riparian vegetation ETc.
    """

    subregion_id: int
    native_etc_column: int = 0
    riparian_etc_column: int = 0


@dataclass
class NativeRiparianInitialRow:
    """Initial soil-moisture conditions for one subregion.

    Values alternate between native and riparian per soil type:
    ``native_soil1  riparian_soil1  native_soil2  riparian_soil2 …``

    Attributes:
        subregion_id: Subregion identifier (1-based).
        native_moisture: Initial moisture content for native, one per soil.
        riparian_moisture: Initial moisture content for riparian, one per soil.
    """

    subregion_id: int
    native_moisture: list[float] = field(default_factory=list)
    riparian_moisture: list[float] = field(default_factory=list)


@dataclass
class NativeRiparianConfig:
    """Configuration parsed from a native/riparian vegetation sub-file.

    Attributes:
        area_data_file: Path to vegetation-area time-series data file.
        root_depth_factor: Conversion factor for root depths.
        native_root_depth: Native vegetation root depth (with factor applied).
        riparian_root_depth: Riparian vegetation root depth (with factor applied).
        curve_numbers: Curve-number rows (one per subregion).
        etc_pointers: ET-column pointer rows (one per subregion).
        initial_conditions: Initial soil-moisture condition rows.
    """

    area_data_file: Path | None = None
    root_depth_factor: float = 1.0
    native_root_depth: float = 0.0
    riparian_root_depth: float = 0.0
    curve_numbers: list[NativeRiparianCNRow] = field(default_factory=list)
    etc_pointers: list[NativeRiparianEtcRow] = field(default_factory=list)
    initial_conditions: list[NativeRiparianInitialRow] = field(
        default_factory=list
    )


# ── Reader ────────────────────────────────────────────────────────────


class NativeRiparianReader:
    """Reader for IWFM native/riparian vegetation sub-file.

    Parses the positional-sequential file referenced as *NVRVFL* in
    the RootZone component main file.

    Args:
        n_subregions: Number of subregions (for exact row counts).
    """

    def __init__(self, n_subregions: int | None = None) -> None:
        self._n_sub = n_subregions

    def read(
        self,
        filepath: Path | str,
        base_dir: Path | None = None,
    ) -> NativeRiparianConfig:
        """Parse the native/riparian vegetation file.

        Args:
            filepath: Path to the native/riparian vegetation file.
            base_dir: Base directory for resolving relative sub-file
                paths.  Defaults to the parent of *filepath*.

        Returns:
            Populated :class:`NativeRiparianConfig`.
        """
        filepath = Path(filepath)
        if base_dir is None:
            base_dir = filepath.parent

        with open(filepath, "r") as f:
            lines = f.readlines()
        buf = _LineBuffer(lines)
        config = NativeRiparianConfig()

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

        # 3. Native vegetation root depth
        val = buf.next_data()
        try:
            config.native_root_depth = float(val) * config.root_depth_factor
        except ValueError as exc:
            raise FileFormatError(
                f"Invalid native root depth: '{val}'",
                line_number=buf.line_num,
            ) from exc

        # 4. Riparian vegetation root depth
        val = buf.next_data()
        try:
            config.riparian_root_depth = float(val) * config.root_depth_factor
        except ValueError as exc:
            raise FileFormatError(
                f"Invalid riparian root depth: '{val}'",
                line_number=buf.line_num,
            ) from exc

        # 5. Curve-number data (per subregion: native + riparian)
        config.curve_numbers = self._read_cn_rows(buf)

        # 6. ETc column pointers (per subregion: native + riparian)
        config.etc_pointers = self._read_etc_rows(buf)

        # 7. Initial conditions (per subregion: native + riparian)
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
        self, buf: _LineBuffer, min_cols: int
    ) -> list[list[str]]:
        """Read tabular data rows with comment-based section boundaries."""
        rows: list[list[str]] = []
        n_expected = self._n_sub
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

            value, _ = _parse_value_line(raw)
            parts = value.split()

            if len(parts) < min_cols:
                buf.pushback()
                break

            rows.append(parts)
            started = True

            if n_expected is not None and len(rows) >= n_expected:
                break

        return rows

    def _read_cn_rows(self, buf: _LineBuffer) -> list[NativeRiparianCNRow]:
        """Read curve-number rows.

        Each row: ``subregion_id  native_cn1 … native_cnN  riparian_cn1 … riparian_cnN``
        """
        raw_rows = self._read_rows(buf, min_cols=3)
        result: list[NativeRiparianCNRow] = []
        for parts in raw_rows:
            try:
                sub_id = int(parts[0])
                vals = [float(v) for v in parts[1:]]
                n_soils = len(vals) // 2
                result.append(
                    NativeRiparianCNRow(
                        subregion_id=sub_id,
                        native_cn=vals[:n_soils],
                        riparian_cn=vals[n_soils:],
                    )
                )
            except ValueError:
                break
        return result

    def _read_etc_rows(self, buf: _LineBuffer) -> list[NativeRiparianEtcRow]:
        """Read ETc pointer rows: ``subregion_id  native_col  riparian_col``."""
        raw_rows = self._read_rows(buf, min_cols=3)
        result: list[NativeRiparianEtcRow] = []
        for parts in raw_rows:
            try:
                sub_id = int(parts[0])
                result.append(
                    NativeRiparianEtcRow(
                        subregion_id=sub_id,
                        native_etc_column=int(float(parts[1])),
                        riparian_etc_column=int(float(parts[2])),
                    )
                )
            except ValueError:
                break
        return result

    def _read_initial_conditions(
        self, buf: _LineBuffer
    ) -> list[NativeRiparianInitialRow]:
        """Read initial soil-moisture conditions.

        Each row: ``subregion_id  native1 riparian1 native2 riparian2 …``
        """
        raw_rows = self._read_rows(buf, min_cols=3)
        result: list[NativeRiparianInitialRow] = []
        for parts in raw_rows:
            try:
                sub_id = int(parts[0])
                data = [float(v) for v in parts[1:]]
                native = data[0::2]
                riparian = data[1::2]
                result.append(
                    NativeRiparianInitialRow(
                        subregion_id=sub_id,
                        native_moisture=native,
                        riparian_moisture=riparian,
                    )
                )
            except ValueError:
                break
        return result


# ── convenience function ──────────────────────────────────────────────


def read_native_riparian(
    filepath: Path | str,
    base_dir: Path | None = None,
    n_subregions: int | None = None,
) -> NativeRiparianConfig:
    """Read an IWFM native/riparian vegetation sub-file.

    Args:
        filepath: Path to the native/riparian vegetation file.
        base_dir: Base directory for resolving relative paths.
        n_subregions: Number of subregions (for exact row counts).

    Returns:
        :class:`NativeRiparianConfig` with parsed values.
    """
    reader = NativeRiparianReader(n_subregions=n_subregions)
    return reader.read(filepath, base_dir)
