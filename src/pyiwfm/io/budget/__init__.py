"""Budget and zone-budget readers, writers, and helpers.

The v1.x flat modules ``pyiwfm.io.budget``, ``pyiwfm.io.zbudget``,
``pyiwfm.io.budget_checks``, ``pyiwfm.io.budget_control``,
``pyiwfm.io.zbudget_control``, ``pyiwfm.io.budget_excel``,
``pyiwfm.io.zbudget_excel``, ``pyiwfm.io.budget_pest``, and
``pyiwfm.io.budget_utils`` are now collapsed into one subpackage:

- :mod:`pyiwfm.io.budget.reader` — was ``budget.py``
  (``BudgetReader`` and friends).
- :mod:`pyiwfm.io.budget.zone_reader` — was ``zbudget.py``
  (``ZBudgetReader``, ``ZBudgetHeader``, ``ZoneInfo``).
- :mod:`pyiwfm.io.budget.checks` — was ``budget_checks.py``
  (mass-balance / sanity checks).
- :mod:`pyiwfm.io.budget.control` — was ``budget_control.py``.
- :mod:`pyiwfm.io.budget.zone_control` — was ``zbudget_control.py``.
- :mod:`pyiwfm.io.budget.excel` — was ``budget_excel.py``.
- :mod:`pyiwfm.io.budget.zone_excel` — was ``zbudget_excel.py``.
- :mod:`pyiwfm.io.budget.pest` — was ``budget_pest.py``.
- :mod:`pyiwfm.io.budget._utils` — was ``budget_utils.py`` (private
  helpers, name-prefixed since the module is implementation detail).

The package re-exports every public symbol. The eight v1.x flat
sibling paths are gone in v2.0; use ``from pyiwfm.io.budget import X``
instead. See ``docs/MIGRATION_v1_to_v2.md`` § 10.
"""

from __future__ import annotations

from pyiwfm.io.budget._utils import (
    apply_unit_conversion,
    filter_time_range,
    format_title_lines,
)
from pyiwfm.io.budget.checks import (
    BalanceCheckResult,
    BudgetSanityReport,
    check_all_budgets,
    check_budget_balance,
)
from pyiwfm.io.budget.control import (
    BudgetControlConfig,
    BudgetOutputSpec,
    read_budget_control,
)
from pyiwfm.io.budget.excel import (
    budget_control_to_excel,
    budget_to_excel,
)
from pyiwfm.io.budget.pest import (
    budget_to_pest_instruction,
    budget_to_pest_text,
)
from pyiwfm.io.budget.reader import (
    BUDGET_DATA_TYPES,
    DSS_DATA_TYPES,
    UNIT_MARKERS,
    ASCIIOutputInfo,
    BudgetHeader,
    BudgetReader,
    LocationData,
    TimeStepInfo,
    excel_julian_to_datetime,
    iwfm_date_to_iso,
    julian_to_datetime,
    parse_iwfm_datetime,
)
from pyiwfm.io.budget.zone_control import (
    ZBudgetControlConfig,
    ZBudgetOutputSpec,
    read_zbudget_control,
)
from pyiwfm.io.budget.zone_excel import (
    zbudget_control_to_excel,
    zbudget_to_excel,
)
from pyiwfm.io.budget.zone_reader import (
    ZBUDGET_DATA_TYPES,
    ZBudgetHeader,
    ZBudgetReader,
    ZoneInfo,
)

__all__ = [
    # reader.py
    "ASCIIOutputInfo",
    "BUDGET_DATA_TYPES",
    "DSS_DATA_TYPES",
    "UNIT_MARKERS",
    "BudgetHeader",
    "BudgetReader",
    "LocationData",
    "TimeStepInfo",
    "excel_julian_to_datetime",
    "iwfm_date_to_iso",
    "julian_to_datetime",
    "parse_iwfm_datetime",
    # zone_reader.py
    "ZBUDGET_DATA_TYPES",
    "ZBudgetHeader",
    "ZBudgetReader",
    "ZoneInfo",
    # checks.py
    "BalanceCheckResult",
    "BudgetSanityReport",
    "check_all_budgets",
    "check_budget_balance",
    # control.py
    "BudgetControlConfig",
    "BudgetOutputSpec",
    "read_budget_control",
    # zone_control.py
    "ZBudgetControlConfig",
    "ZBudgetOutputSpec",
    "read_zbudget_control",
    # excel.py
    "budget_control_to_excel",
    "budget_to_excel",
    # zone_excel.py
    "zbudget_control_to_excel",
    "zbudget_to_excel",
    # pest.py
    "budget_to_pest_instruction",
    "budget_to_pest_text",
    # _utils.py
    "apply_unit_conversion",
    "filter_time_range",
    "format_title_lines",
]
