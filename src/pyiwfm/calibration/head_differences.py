"""
Head difference computation for paired well observations.

Computes simulated head differences between paired wells screened at
different layers. Used for vertical gradient calibration targets.

Example
-------
>>> pairs = [HeadDiffPair("WELL_A%1", "WELL_A%3", "HD_WELL_A")]
>>> diffs = compute_head_differences(pairs, sim_data)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyiwfm.io.smp import SMPTimeSeries


@dataclass
class HeadDiffPair:
    """Specification for a head difference pair.

    Attributes
    ----------
    shallow_well : str
        Bore ID of the shallow well (e.g., ``"WELL_A%1"``).
    deep_well : str
        Bore ID of the deep well (e.g., ``"WELL_A%3"``).
    pair_name : str
        Name for the resulting difference time series.
    """

    shallow_well: str
    deep_well: str
    pair_name: str


def compute_head_differences(
    pairs: list[HeadDiffPair],
    sim_data: dict[str, SMPTimeSeries],
) -> dict[str, SMPTimeSeries]:
    """Compute simulated head differences between paired layer observations.

    For each pair, finds timestamps common to both wells and computes
    ``shallow_head - deep_head``.

    Parameters
    ----------
    pairs : list[HeadDiffPair]
        Well pairs to difference.
    sim_data : dict[str, SMPTimeSeries]
        Simulated head time series keyed by bore ID (e.g., from SMPReader).

    Returns
    -------
    dict[str, SMPTimeSeries]
        Difference time series keyed by pair name.
    """
    import numpy as np

    from pyiwfm.io.smp import SMPTimeSeries as _SMPTimeSeries

    results: dict[str, _SMPTimeSeries] = {}

    for pair in pairs:
        ts1 = sim_data.get(pair.shallow_well)
        ts2 = sim_data.get(pair.deep_well)
        if ts1 is None or ts2 is None:
            continue

        # Build lookup for deep well: date_key -> (index, value)
        deep_by_date: dict[str, tuple[int, float]] = {}
        for i in range(ts2.n_records):
            key = str(ts2.times[i])
            if key not in deep_by_date:
                deep_by_date[key] = (i, float(ts2.values[i]))

        # Match and compute differences
        diff_times = []
        diff_values = []
        for i in range(ts1.n_records):
            key = str(ts1.times[i])
            if key in deep_by_date:
                v1 = float(ts1.values[i])
                v2 = deep_by_date[key][1]
                if not (np.isnan(v1) or np.isnan(v2)):
                    diff_times.append(ts1.times[i])
                    diff_values.append(v1 - v2)

        if not diff_times:
            continue

        results[pair.pair_name] = _SMPTimeSeries(
            bore_id=pair.pair_name,
            times=np.array(diff_times, dtype="datetime64[s]"),
            values=np.array(diff_values, dtype=np.float64),
            excluded=np.zeros(len(diff_times), dtype=np.bool_),
        )

    return results


def read_pairs_file(pairs_path: Path) -> list[HeadDiffPair]:
    """Read head difference pairs from a pairs file.

    Expected format: two columns per line (``well1%%layer1  well2%%layer2``),
    optionally with a third column for the pair name.

    Parameters
    ----------
    pairs_path : Path
        Path to the pairs file (e.g., ``HeadDiffPairs.dat``).

    Returns
    -------
    list[HeadDiffPair]
    """
    pairs: list[HeadDiffPair] = []

    with open(pairs_path, encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f):
            parts = line.split()
            if len(parts) < 2:
                continue
            name = parts[2] if len(parts) >= 3 else f"HD{idx:05d}"
            pairs.append(
                HeadDiffPair(
                    shallow_well=parts[0].strip(),
                    deep_well=parts[1].strip(),
                    pair_name=name,
                )
            )

    return pairs
