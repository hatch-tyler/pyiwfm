"""Mixin providing budget and zone-budget reader methods for ModelState."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyiwfm.core.model import IWFMModel
    from pyiwfm.core.zones import ZoneDefinition
    from pyiwfm.io.budget import BudgetReader
    from pyiwfm.io.zbudget import ZBudgetReader

logger = logging.getLogger(__name__)


class BudgetStateMixin:
    """Mixin providing budget and zone-budget reader methods for ModelState."""

    # -- Attributes set by ModelState.__init__ (declared for type checkers) --
    _model: IWFMModel | None
    _results_dir: Path | None
    _budget_readers: dict[str, BudgetReader]
    _zbudget_readers: dict[str, ZBudgetReader]
    _active_zone_def: ZoneDefinition | None

    # ------------------------------------------------------------------
    # Budget readers
    # ------------------------------------------------------------------

    def get_available_budgets(self) -> list[str]:
        """Return list of budget types that have files available."""
        if self._model is None:
            return []

        budget_keys = {
            "gw": "gw_budget_file",
            "stream": "stream_budget_file",
            "stream_node": "stream_node_budget_file",
            "lwu": "rootzone_lwu_budget_file",
            "rootzone": "rootzone_rz_budget_file",
            "unsaturated": "unsat_zone_budget_file",
            "diversion": "stream_diversion_budget_file",
            "lake": "lake_budget_file",
            "small_watershed": "small_watershed_budget_file",
        }

        available: list[str] = []
        for btype, meta_key in budget_keys.items():
            fpath = self._model.metadata.get(meta_key)
            if fpath:
                p = Path(fpath)
                if not p.is_absolute() and self._results_dir:
                    p = self._results_dir / p
                if p.exists():
                    available.append(btype)

        return available

    def get_budget_reader(self, budget_type: str) -> BudgetReader | None:
        """Get or create a BudgetReader for the given budget type."""
        if budget_type in self._budget_readers:
            return self._budget_readers[budget_type]

        if self._model is None:
            return None

        budget_keys = {
            "gw": "gw_budget_file",
            "stream": "stream_budget_file",
            "stream_node": "stream_node_budget_file",
            "lwu": "rootzone_lwu_budget_file",
            "rootzone": "rootzone_rz_budget_file",
            "unsaturated": "unsat_zone_budget_file",
            "diversion": "stream_diversion_budget_file",
            "lake": "lake_budget_file",
            "small_watershed": "small_watershed_budget_file",
        }

        meta_key = budget_keys.get(budget_type)
        if not meta_key:
            return None

        fpath = self._model.metadata.get(meta_key)
        if not fpath:
            return None

        p = Path(fpath)
        if not p.is_absolute() and self._results_dir:
            p = self._results_dir / p
        if not p.exists():
            logger.warning("Budget file not found: %s", p)
            return None

        try:
            from pyiwfm.io.budget import BudgetReader

            reader = BudgetReader(p)
            self._budget_readers[budget_type] = reader
            logger.info("Budget reader for '%s': %s", budget_type, reader.descriptor)
            return reader
        except Exception as e:
            logger.error("Failed to load budget file %s: %s", p, e)
            return None

    # ------------------------------------------------------------------
    # ZBudget readers
    # ------------------------------------------------------------------

    def get_available_zbudgets(self) -> list[str]:
        """Return list of zbudget types that have HDF5 files available."""
        if self._model is None:
            return []

        zbudget_keys = {
            "gw": "gw_zbudget_file",
            "lwu": "rootzone_lwu_zbudget_file",
            "rootzone": "rootzone_rz_zbudget_file",
            "unsaturated": "unsat_zone_zbudget_file",
        }

        available: list[str] = []
        for ztype, meta_key in zbudget_keys.items():
            fpath = self._model.metadata.get(meta_key)
            if fpath:
                p = Path(fpath)
                if not p.is_absolute() and self._results_dir:
                    p = self._results_dir / p
                if p.exists():
                    available.append(ztype)

        return available

    def get_zbudget_reader(self, zbudget_type: str) -> ZBudgetReader | None:
        """Get or create a ZBudgetReader for the given type."""
        if zbudget_type in self._zbudget_readers:
            return self._zbudget_readers[zbudget_type]

        if self._model is None:
            return None

        zbudget_keys = {
            "gw": "gw_zbudget_file",
            "lwu": "rootzone_lwu_zbudget_file",
            "rootzone": "rootzone_rz_zbudget_file",
            "unsaturated": "unsat_zone_zbudget_file",
        }

        meta_key = zbudget_keys.get(zbudget_type)
        if not meta_key:
            return None

        fpath = self._model.metadata.get(meta_key)
        if not fpath:
            return None

        p = Path(fpath)
        if not p.is_absolute() and self._results_dir:
            p = self._results_dir / p
        if not p.exists():
            return None

        try:
            from pyiwfm.io.zbudget import ZBudgetReader

            reader = ZBudgetReader(p)
            self._zbudget_readers[zbudget_type] = reader
            logger.info("ZBudget reader for '%s': %s", zbudget_type, reader.descriptor)
            return reader
        except Exception as e:
            logger.error("Failed to load zbudget file %s: %s", p, e)
            return None

    def set_zone_definition(self, zone_def: ZoneDefinition) -> None:
        """Set the active zone definition for ZBudget analysis."""
        self._active_zone_def = zone_def

    def get_zone_definition(self) -> ZoneDefinition | None:
        """Get the current active zone definition."""
        return self._active_zone_def
