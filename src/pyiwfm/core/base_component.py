"""Abstract base class for IWFM model components.

All major IWFM simulation components (groundwater, streams, lakes,
root zone, small watersheds, unsaturated zone) share a common
interface defined by :class:`BaseComponent`.  This enables generic
model validation, iteration, and serialization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseComponent(ABC):
    """Abstract base class for IWFM model components.

    Every component must implement:

    * :meth:`validate` -- return a list of validation error strings
      (empty list means valid).
    * :attr:`n_items` -- number of primary entities managed by the
      component (e.g., number of wells, stream nodes, lakes).
    """

    @abstractmethod
    def validate(self) -> None:
        """Validate the component state.

        Raises
        ------
        ValidationError
            If the component state is invalid.
        """
        ...

    @property
    @abstractmethod
    def n_items(self) -> int:
        """Return the number of primary entities in this component."""
        ...
