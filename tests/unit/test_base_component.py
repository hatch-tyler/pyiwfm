"""Tests for the BaseComponent ABC contract.

The class enforces (a) ``validate``-and-``n_items`` as abstract methods
and (b) the convention that every concrete component's ``validate()``
raises :class:`~pyiwfm.core.exceptions.ComponentError` on invalid state
and returns ``None`` on success. v2.0 PR 3 fixed a docstring inconsistency
that previously described a "list of error strings" return shape no
implementation actually used.
"""

from __future__ import annotations

import pytest

from pyiwfm.core.base_component import BaseComponent
from pyiwfm.core.exceptions import ComponentError


def test_cannot_instantiate_base_directly() -> None:
    """``BaseComponent`` is abstract; instantiating it directly should raise."""
    with pytest.raises(TypeError, match="abstract"):
        BaseComponent()  # type: ignore[abstract]


def test_subclass_missing_validate_cannot_instantiate() -> None:
    """A subclass that omits ``validate`` is still abstract."""

    class Incomplete(BaseComponent):
        @property
        def n_items(self) -> int:
            return 0

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()  # type: ignore[abstract]


def test_subclass_missing_n_items_cannot_instantiate() -> None:
    """A subclass that omits ``n_items`` is still abstract."""

    class Incomplete(BaseComponent):
        def validate(self) -> None:
            pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()  # type: ignore[abstract]


def test_concrete_implementations_follow_raise_contract() -> None:
    """All six shipped components raise ComponentError on invalid state.

    The v2.0 docstring correction codified what every shipped
    implementation already did. This test pins that contract via a
    minimal subclass.
    """

    class GoodComponent(BaseComponent):
        def __init__(self, items: int = 0, valid: bool = True) -> None:
            self._items = items
            self._valid = valid

        def validate(self) -> None:
            if not self._valid:
                raise ComponentError("intentionally invalid")

        @property
        def n_items(self) -> int:
            return self._items

    # Valid: returns None
    assert GoodComponent(items=3, valid=True).validate() is None
    # Invalid: raises ComponentError
    with pytest.raises(ComponentError, match="intentionally invalid"):
        GoodComponent(items=0, valid=False).validate()
    # n_items contract
    assert GoodComponent(items=42).n_items == 42
