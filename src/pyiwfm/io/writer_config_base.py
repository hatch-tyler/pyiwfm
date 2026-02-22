"""Base configuration for IWFM component file writers.

Provides :class:`BaseComponentWriterConfig`, a dataclass base that
encapsulates the fields common to every component writer config
(groundwater, streams, lakes, root zone, small watersheds, and
unsaturated zone).

Subclasses override :meth:`_get_subdir` and :meth:`_get_main_file`
to provide the component-specific defaults, while inheriting the
``output_dir``, ``version``, ``component_dir``, and ``main_path``
interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BaseComponentWriterConfig:
    """Base configuration shared by all IWFM component writers.

    Attributes
    ----------
    output_dir : Path
        Base output directory (typically ``<model>/Simulation``).
    version : str
        IWFM component version string (e.g. ``"4.0"``).
    """

    output_dir: Path
    version: str = "4.0"

    # -- subclass hooks ----------------------------------------------------

    def _get_subdir(self) -> str:
        """Return the subdirectory name for this component.

        Override in subclasses to return the component-specific
        subdirectory name (e.g. ``"GW"``, ``"Stream"``).
        """
        return ""

    def _get_main_file(self) -> str:
        """Return the main file name for this component.

        Override in subclasses to return the component-specific
        main file name (e.g. ``"GW_MAIN.dat"``).
        """
        return ""

    # -- derived paths -----------------------------------------------------

    @property
    def component_dir(self) -> Path:
        """Get the component subdirectory path."""
        subdir = self._get_subdir()
        if subdir:
            return self.output_dir / subdir
        return self.output_dir

    @property
    def main_path(self) -> Path:
        """Get the main file path."""
        return self.component_dir / self._get_main_file()
