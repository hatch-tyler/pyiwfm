"""Abstract base class for IWFM model components.

All major IWFM simulation components (groundwater, streams, lakes,
root zone, small watersheds, unsaturated zone) share a common
interface defined by :class:`BaseComponent`. This enables generic
model validation and iteration over a heterogeneous component set.

v2.0 design notes
-----------------
The original v2.0 roadmap proposed adding ``to_dict`` / ``from_dict``
/ ``clone`` / ``validate_against(grid)`` as abstract methods on this
class, on the assumption that future serialization (HDF5/JSON
scenario diffs) and polymorphic mutation pipelines would need them.
A concrete-call audit in v2.0 PR 3 found no current caller invokes
those methods on any ``BaseComponent`` subclass — the reports and
metrics that *do* expose ``to_dict()`` are unrelated dataclasses.
Adding 4 new abstract methods × 6 component implementations for code
that nothing calls would violate the project's "don't design for
hypothetical future requirements" guidance (CLAUDE.md), so these
additions are deferred until a real caller needs them. When that
caller appears, adding the methods will be a single PR — not a
breaking change for downstream subclassers, since the addition would
ship alongside default implementations on this base.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseComponent(ABC):
    """Abstract base class for IWFM model components.

    Every component must implement:

    * :meth:`validate` -- raise :class:`~pyiwfm.core.exceptions.ComponentError`
      if the component state is invalid (the v2.0 contract; the v1.x
      class docstring incorrectly described a "list of error strings"
      return shape that no implementation actually used).
    * :attr:`n_items` -- number of primary entities managed by the
      component (e.g., number of wells, stream nodes, lakes).
    """

    @abstractmethod
    def validate(self) -> None:
        """Validate the component state.

        Raises
        ------
        pyiwfm.core.exceptions.ComponentError
            If the component state is invalid. All six current component
            implementations follow this contract; subclasses MUST raise
            :class:`~pyiwfm.core.exceptions.ComponentError` (or a subclass)
            on invalid state and return ``None`` on success.
        """
        ...

    @property
    @abstractmethod
    def n_items(self) -> int:
        """Return the number of primary entities in this component."""
        ...
