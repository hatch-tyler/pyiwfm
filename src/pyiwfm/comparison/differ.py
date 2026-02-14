"""
Model differ for comparing IWFM models.

This module provides classes for comparing IWFM model components
and generating structured diffs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np

if TYPE_CHECKING:
    from pyiwfm.core.mesh import AppGrid
    from pyiwfm.core.stratigraphy import Stratigraphy


class DiffType(Enum):
    """Type of difference detected."""

    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"


@dataclass
class DiffItem:
    """
    A single difference item.

    Attributes:
        path: Path to the differing item (e.g., 'mesh.nodes.5.x')
        diff_type: Type of difference (added, removed, modified)
        old_value: Original value (None if added)
        new_value: New value (None if removed)
    """

    path: str
    diff_type: DiffType
    old_value: Any = None
    new_value: Any = None

    def __repr__(self) -> str:
        if self.diff_type == DiffType.ADDED:
            return f"+ {self.path}: {self.new_value}"
        elif self.diff_type == DiffType.REMOVED:
            return f"- {self.path}: {self.old_value}"
        else:
            return f"~ {self.path}: {self.old_value} -> {self.new_value}"


@dataclass
class MeshDiff:
    """
    Difference between two meshes.

    Attributes:
        items: List of difference items
        nodes_added: Number of nodes added
        nodes_removed: Number of nodes removed
        nodes_modified: Number of nodes modified
        elements_added: Number of elements added
        elements_removed: Number of elements removed
        elements_modified: Number of elements modified
    """

    items: list[DiffItem] = field(default_factory=list)
    nodes_added: int = 0
    nodes_removed: int = 0
    nodes_modified: int = 0
    elements_added: int = 0
    elements_removed: int = 0
    elements_modified: int = 0

    @property
    def is_identical(self) -> bool:
        """Check if meshes are identical."""
        return len(self.items) == 0

    @classmethod
    def compare(cls, mesh1: "AppGrid", mesh2: "AppGrid") -> "MeshDiff":
        """
        Compare two meshes and return their differences.

        Args:
            mesh1: First mesh (original)
            mesh2: Second mesh (modified)

        Returns:
            MeshDiff containing all differences
        """
        diff = cls()

        # Compare nodes
        node_ids1 = set(mesh1.nodes.keys())
        node_ids2 = set(mesh2.nodes.keys())

        # Added nodes
        for nid in node_ids2 - node_ids1:
            node = mesh2.nodes[nid]
            diff.items.append(DiffItem(
                path=f"mesh.nodes.{nid}",
                diff_type=DiffType.ADDED,
                new_value={"x": node.x, "y": node.y, "is_boundary": node.is_boundary},
            ))
            diff.nodes_added += 1

        # Removed nodes
        for nid in node_ids1 - node_ids2:
            node = mesh1.nodes[nid]
            diff.items.append(DiffItem(
                path=f"mesh.nodes.{nid}",
                diff_type=DiffType.REMOVED,
                old_value={"x": node.x, "y": node.y, "is_boundary": node.is_boundary},
            ))
            diff.nodes_removed += 1

        # Modified nodes
        for nid in node_ids1 & node_ids2:
            node1 = mesh1.nodes[nid]
            node2 = mesh2.nodes[nid]

            if not np.isclose(node1.x, node2.x):
                diff.items.append(DiffItem(
                    path=f"mesh.nodes.{nid}.x",
                    diff_type=DiffType.MODIFIED,
                    old_value=node1.x,
                    new_value=node2.x,
                ))
                diff.nodes_modified += 1

            if not np.isclose(node1.y, node2.y):
                diff.items.append(DiffItem(
                    path=f"mesh.nodes.{nid}.y",
                    diff_type=DiffType.MODIFIED,
                    old_value=node1.y,
                    new_value=node2.y,
                ))

            if node1.is_boundary != node2.is_boundary:
                diff.items.append(DiffItem(
                    path=f"mesh.nodes.{nid}.is_boundary",
                    diff_type=DiffType.MODIFIED,
                    old_value=node1.is_boundary,
                    new_value=node2.is_boundary,
                ))

        # Compare elements
        elem_ids1 = set(mesh1.elements.keys())
        elem_ids2 = set(mesh2.elements.keys())

        # Added elements
        for eid in elem_ids2 - elem_ids1:
            elem = mesh2.elements[eid]
            diff.items.append(DiffItem(
                path=f"mesh.elements.{eid}",
                diff_type=DiffType.ADDED,
                new_value={
                    "vertices": elem.vertices,
                    "subregion": elem.subregion,
                },
            ))
            diff.elements_added += 1

        # Removed elements
        for eid in elem_ids1 - elem_ids2:
            elem = mesh1.elements[eid]
            diff.items.append(DiffItem(
                path=f"mesh.elements.{eid}",
                diff_type=DiffType.REMOVED,
                old_value={
                    "vertices": elem.vertices,
                    "subregion": elem.subregion,
                },
            ))
            diff.elements_removed += 1

        # Modified elements
        for eid in elem_ids1 & elem_ids2:
            elem1 = mesh1.elements[eid]
            elem2 = mesh2.elements[eid]

            if elem1.vertices != elem2.vertices:
                diff.items.append(DiffItem(
                    path=f"mesh.elements.{eid}.vertices",
                    diff_type=DiffType.MODIFIED,
                    old_value=elem1.vertices,
                    new_value=elem2.vertices,
                ))
                diff.elements_modified += 1

            if elem1.subregion != elem2.subregion:
                diff.items.append(DiffItem(
                    path=f"mesh.elements.{eid}.subregion",
                    diff_type=DiffType.MODIFIED,
                    old_value=elem1.subregion,
                    new_value=elem2.subregion,
                ))

        return diff


@dataclass
class StratigraphyDiff:
    """
    Difference between two stratigraphy definitions.

    Attributes:
        items: List of difference items
    """

    items: list[DiffItem] = field(default_factory=list)

    @property
    def is_identical(self) -> bool:
        """Check if stratigraphy is identical."""
        return len(self.items) == 0

    @classmethod
    def compare(
        cls,
        strat1: "Stratigraphy",
        strat2: "Stratigraphy",
        tolerance: float = 1e-6,
    ) -> "StratigraphyDiff":
        """
        Compare two stratigraphy definitions.

        Args:
            strat1: First stratigraphy (original)
            strat2: Second stratigraphy (modified)
            tolerance: Tolerance for floating point comparisons

        Returns:
            StratigraphyDiff containing all differences
        """
        diff = cls()

        # Check layer count
        if strat1.n_layers != strat2.n_layers:
            diff.items.append(DiffItem(
                path="stratigraphy.n_layers",
                diff_type=DiffType.MODIFIED,
                old_value=strat1.n_layers,
                new_value=strat2.n_layers,
            ))

        # Check node count
        if strat1.n_nodes != strat2.n_nodes:
            diff.items.append(DiffItem(
                path="stratigraphy.n_nodes",
                diff_type=DiffType.MODIFIED,
                old_value=strat1.n_nodes,
                new_value=strat2.n_nodes,
            ))
            return diff  # Can't compare arrays if sizes differ

        # Compare ground surface elevations
        gs_diff = np.abs(strat1.gs_elev - strat2.gs_elev)
        modified_gs = np.where(gs_diff > tolerance)[0]
        for idx in modified_gs:
            diff.items.append(DiffItem(
                path=f"stratigraphy.gs_elev[{idx}]",
                diff_type=DiffType.MODIFIED,
                old_value=float(strat1.gs_elev[idx]),
                new_value=float(strat2.gs_elev[idx]),
            ))

        # Compare top elevations
        if strat1.n_layers == strat2.n_layers:
            top_diff = np.abs(strat1.top_elev - strat2.top_elev)
            modified_top = np.argwhere(top_diff > tolerance)
            for idx in modified_top:
                node_idx, layer_idx = idx
                diff.items.append(DiffItem(
                    path=f"stratigraphy.top_elev[{node_idx},{layer_idx}]",
                    diff_type=DiffType.MODIFIED,
                    old_value=float(strat1.top_elev[node_idx, layer_idx]),
                    new_value=float(strat2.top_elev[node_idx, layer_idx]),
                ))

            # Compare bottom elevations
            bot_diff = np.abs(strat1.bottom_elev - strat2.bottom_elev)
            modified_bot = np.argwhere(bot_diff > tolerance)
            for idx in modified_bot:
                node_idx, layer_idx = idx
                diff.items.append(DiffItem(
                    path=f"stratigraphy.bottom_elev[{node_idx},{layer_idx}]",
                    diff_type=DiffType.MODIFIED,
                    old_value=float(strat1.bottom_elev[node_idx, layer_idx]),
                    new_value=float(strat2.bottom_elev[node_idx, layer_idx]),
                ))

            # Compare active node flags
            active_diff = strat1.active_node != strat2.active_node
            modified_active = np.argwhere(active_diff)
            for idx in modified_active:
                node_idx, layer_idx = idx
                diff.items.append(DiffItem(
                    path=f"stratigraphy.active_node[{node_idx},{layer_idx}]",
                    diff_type=DiffType.MODIFIED,
                    old_value=bool(strat1.active_node[node_idx, layer_idx]),
                    new_value=bool(strat2.active_node[node_idx, layer_idx]),
                ))

        return diff


@dataclass
class ModelDiff:
    """
    Container for all model differences.

    Attributes:
        mesh_diff: Mesh differences
        stratigraphy_diff: Stratigraphy differences
    """

    mesh_diff: MeshDiff | None = None
    stratigraphy_diff: StratigraphyDiff | None = None

    @property
    def items(self) -> list[DiffItem]:
        """Get all difference items."""
        all_items = []
        if self.mesh_diff:
            all_items.extend(self.mesh_diff.items)
        if self.stratigraphy_diff:
            all_items.extend(self.stratigraphy_diff.items)
        return all_items

    @property
    def is_identical(self) -> bool:
        """Check if models are identical."""
        mesh_identical = self.mesh_diff is None or self.mesh_diff.is_identical
        strat_identical = (
            self.stratigraphy_diff is None or self.stratigraphy_diff.is_identical
        )
        return mesh_identical and strat_identical

    def summary(self) -> str:
        """
        Generate a human-readable summary of differences.

        Returns:
            Summary string
        """
        lines = ["Model Difference Summary", "=" * 40]

        if self.is_identical:
            lines.append("Models are identical.")
            return "\n".join(lines)

        stats = self.statistics()
        lines.append(f"Total changes: {stats['total_changes']}")
        lines.append(f"  Added: {stats['added']}")
        lines.append(f"  Removed: {stats['removed']}")
        lines.append(f"  Modified: {stats['modified']}")
        lines.append("")

        if self.mesh_diff and not self.mesh_diff.is_identical:
            lines.append("Mesh Changes:")
            lines.append(f"  Nodes: +{self.mesh_diff.nodes_added} "
                        f"-{self.mesh_diff.nodes_removed} "
                        f"~{self.mesh_diff.nodes_modified}")
            lines.append(f"  Elements: +{self.mesh_diff.elements_added} "
                        f"-{self.mesh_diff.elements_removed} "
                        f"~{self.mesh_diff.elements_modified}")

        if self.stratigraphy_diff and not self.stratigraphy_diff.is_identical:
            lines.append("Stratigraphy Changes:")
            lines.append(f"  {len(self.stratigraphy_diff.items)} modifications")

        return "\n".join(lines)

    def filter_by_path(self, prefix: str) -> "ModelDiff":
        """
        Filter diff items by path prefix.

        Args:
            prefix: Path prefix to filter by

        Returns:
            New ModelDiff with filtered items
        """
        filtered_mesh = None
        filtered_strat = None

        if self.mesh_diff:
            filtered_mesh = MeshDiff(
                items=[i for i in self.mesh_diff.items if prefix in i.path]
            )

        if self.stratigraphy_diff:
            filtered_strat = StratigraphyDiff(
                items=[i for i in self.stratigraphy_diff.items if prefix in i.path]
            )

        return ModelDiff(mesh_diff=filtered_mesh, stratigraphy_diff=filtered_strat)

    def filter_by_type(self, diff_type: DiffType) -> "ModelDiff":
        """
        Filter diff items by diff type.

        Args:
            diff_type: Type of diff to filter by

        Returns:
            New ModelDiff with filtered items
        """
        filtered_mesh = None
        filtered_strat = None

        if self.mesh_diff:
            filtered_mesh = MeshDiff(
                items=[i for i in self.mesh_diff.items if i.diff_type == diff_type]
            )

        if self.stratigraphy_diff:
            filtered_strat = StratigraphyDiff(
                items=[i for i in self.stratigraphy_diff.items if i.diff_type == diff_type]
            )

        return ModelDiff(mesh_diff=filtered_mesh, stratigraphy_diff=filtered_strat)

    def statistics(self) -> dict[str, int]:
        """
        Calculate diff statistics.

        Returns:
            Dictionary with statistics
        """
        items = self.items
        return {
            "total_changes": len(items),
            "added": sum(1 for i in items if i.diff_type == DiffType.ADDED),
            "removed": sum(1 for i in items if i.diff_type == DiffType.REMOVED),
            "modified": sum(1 for i in items if i.diff_type == DiffType.MODIFIED),
        }

    def to_dict(self) -> dict[str, Any]:
        """
        Convert diff to dictionary representation.

        Returns:
            Dictionary representation of diff
        """
        result = {}

        if self.mesh_diff:
            result["mesh"] = {
                "is_identical": self.mesh_diff.is_identical,
                "nodes_added": self.mesh_diff.nodes_added,
                "nodes_removed": self.mesh_diff.nodes_removed,
                "nodes_modified": self.mesh_diff.nodes_modified,
                "elements_added": self.mesh_diff.elements_added,
                "elements_removed": self.mesh_diff.elements_removed,
                "elements_modified": self.mesh_diff.elements_modified,
                "items": [
                    {
                        "path": item.path,
                        "type": item.diff_type.value,
                        "old_value": item.old_value,
                        "new_value": item.new_value,
                    }
                    for item in self.mesh_diff.items
                ],
            }

        if self.stratigraphy_diff:
            result["stratigraphy"] = {
                "is_identical": self.stratigraphy_diff.is_identical,
                "items": [
                    {
                        "path": item.path,
                        "type": item.diff_type.value,
                        "old_value": item.old_value,
                        "new_value": item.new_value,
                    }
                    for item in self.stratigraphy_diff.items
                ],
            }

        return result


class ModelDiffer:
    """
    Compare two IWFM models and generate differences.

    This class provides methods to compare individual model components
    or entire models.
    """

    def __init__(self, tolerance: float = 1e-6) -> None:
        """
        Initialize the model differ.

        Args:
            tolerance: Tolerance for floating point comparisons
        """
        self.tolerance = tolerance

    def diff_meshes(
        self,
        mesh1: "AppGrid",
        mesh2: "AppGrid",
    ) -> MeshDiff:
        """
        Compare two meshes.

        Args:
            mesh1: First mesh (original)
            mesh2: Second mesh (modified)

        Returns:
            MeshDiff containing differences
        """
        return MeshDiff.compare(mesh1, mesh2)

    def diff_stratigraphy(
        self,
        strat1: "Stratigraphy",
        strat2: "Stratigraphy",
    ) -> StratigraphyDiff:
        """
        Compare two stratigraphy definitions.

        Args:
            strat1: First stratigraphy (original)
            strat2: Second stratigraphy (modified)

        Returns:
            StratigraphyDiff containing differences
        """
        return StratigraphyDiff.compare(strat1, strat2, self.tolerance)

    def diff(
        self,
        mesh1: "AppGrid | None" = None,
        mesh2: "AppGrid | None" = None,
        strat1: "Stratigraphy | None" = None,
        strat2: "Stratigraphy | None" = None,
    ) -> ModelDiff:
        """
        Compare model components.

        Args:
            mesh1: First mesh (original)
            mesh2: Second mesh (modified)
            strat1: First stratigraphy (original)
            strat2: Second stratigraphy (modified)

        Returns:
            ModelDiff containing all differences
        """
        mesh_diff = None
        strat_diff = None

        if mesh1 is not None and mesh2 is not None:
            mesh_diff = self.diff_meshes(mesh1, mesh2)

        if strat1 is not None and strat2 is not None:
            strat_diff = self.diff_stratigraphy(strat1, strat2)

        return ModelDiff(mesh_diff=mesh_diff, stratigraphy_diff=strat_diff)
