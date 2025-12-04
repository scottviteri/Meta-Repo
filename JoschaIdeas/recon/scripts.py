"""
Convenience functions for building ReCoN scripts and hierarchies.

This module provides a fluent API for constructing ReCoN networks
that represent hierarchical scripts with ordered and parallel substeps.
"""

from typing import List, Optional, Union, Callable, Dict, Any
import numpy as np

from .core import (
    ReCoNNetwork,
    ScriptUnit,
    TerminalUnit,
    LinkType,
)


class ScriptBuilder:
    """
    Builder for constructing ReCoN script hierarchies.

    Provides a fluent API for defining:
    - Hierarchical decomposition (parent-child relationships)
    - Sequential ordering (predecessor-successor relationships)
    - Terminal measurements/actions

    Example usage:
        builder = ScriptBuilder()
        builder.script("recognize_scene") \\
            .children([
                builder.script("check_feature_A"),
                builder.script("check_feature_B"),
            ]) \\
            .as_sequence()  # Children run in order

        network = builder.build()
    """

    def __init__(self, dim_activation: int = 1):
        """
        Initialize the builder.

        Args:
            dim_activation: Default activation dimension for units
        """
        self.dim_activation = dim_activation
        self._network = ReCoNNetwork()
        self._unit_counter = 0
        self._current_unit: Optional[str] = None
        self._pending_children: Dict[str, List[str]] = {}
        self._pending_sequences: Dict[str, List[str]] = {}

    def _generate_id(self, prefix: str = "unit") -> str:
        """Generate a unique unit ID."""
        self._unit_counter += 1
        return f"{prefix}_{self._unit_counter}"

    def script(self, name: Optional[str] = None,
               aggregation: str = "all",
               dim: Optional[int] = None) -> "ScriptBuilder":
        """
        Create a new script unit.

        Args:
            name: Unique name for the unit (auto-generated if None)
            aggregation: "all" or "any" for child aggregation policy
            dim: Activation dimension (uses default if None)

        Returns:
            self for method chaining
        """
        uid = name if name else self._generate_id("script")
        dim = dim if dim is not None else self.dim_activation

        unit = ScriptUnit(uid, dim, aggregation_policy=aggregation)
        self._network.add_unit(unit)
        self._current_unit = uid

        return self

    def terminal(self, name: Optional[str] = None,
                 measurement_fn: Optional[Callable[[], np.ndarray]] = None,
                 threshold: float = 0.5,
                 dim: Optional[int] = None) -> "ScriptBuilder":
        """
        Create a new terminal unit.

        Args:
            name: Unique name for the unit
            measurement_fn: Function that returns measurement values
            threshold: Threshold for confirmation
            dim: Activation dimension

        Returns:
            self for method chaining
        """
        uid = name if name else self._generate_id("terminal")
        dim = dim if dim is not None else self.dim_activation

        unit = TerminalUnit(uid, dim, measurement_fn, threshold)
        self._network.add_unit(unit)
        self._current_unit = uid

        return self

    def children(self, child_ids: List[str]) -> "ScriptBuilder":
        """
        Specify children for the current script unit.

        Args:
            child_ids: List of child unit IDs

        Returns:
            self for method chaining
        """
        if self._current_unit is None:
            raise ValueError("No current unit selected")

        self._pending_children[self._current_unit] = child_ids
        return self

    def as_sequence(self) -> "ScriptBuilder":
        """
        Make the current unit's children execute in sequence.

        Children will have POR/RET links between them.

        Returns:
            self for method chaining
        """
        if self._current_unit is None:
            raise ValueError("No current unit selected")

        if self._current_unit in self._pending_children:
            children = self._pending_children[self._current_unit]
            self._pending_sequences[self._current_unit] = children

        return self

    def as_parallel(self) -> "ScriptBuilder":
        """
        Make the current unit's children execute in parallel.

        This is the default behavior; children have no POR/RET links.

        Returns:
            self for method chaining
        """
        # Remove from sequences if present
        if self._current_unit in self._pending_sequences:
            del self._pending_sequences[self._current_unit]

        return self

    def set_root(self, unit_id: Optional[str] = None) -> "ScriptBuilder":
        """
        Mark a unit as a root that can receive external requests.

        Args:
            unit_id: ID of the unit (uses current if None)

        Returns:
            self for method chaining
        """
        uid = unit_id if unit_id is not None else self._current_unit
        if uid is None:
            raise ValueError("No unit specified")

        self._network.set_root(uid)
        return self

    def select(self, unit_id: str) -> "ScriptBuilder":
        """
        Select an existing unit as the current unit.

        Args:
            unit_id: ID of the unit to select

        Returns:
            self for method chaining
        """
        if unit_id not in self._network.units:
            raise ValueError(f"Unit '{unit_id}' not found")

        self._current_unit = unit_id
        return self

    def _finalize_links(self):
        """Create all pending links."""
        # Create parent-child links
        for parent_id, child_ids in self._pending_children.items():
            for child_id in child_ids:
                self._network.add_link_pair(parent_id, child_id, "partonomic")

        # Create sequence links
        for parent_id, child_ids in self._pending_sequences.items():
            for i in range(len(child_ids) - 1):
                pred_id = child_ids[i]
                succ_id = child_ids[i + 1]
                self._network.add_link_pair(pred_id, succ_id, "temporal")

    def build(self) -> ReCoNNetwork:
        """
        Build and return the network.

        Returns:
            The constructed ReCoNNetwork
        """
        self._finalize_links()
        return self._network

    def get_network(self) -> ReCoNNetwork:
        """Get the network being built (without finalizing)."""
        return self._network


def build_recognition_script(
    name: str,
    feature_names: List[str],
    feature_fns: List[Callable[[], np.ndarray]],
    sequential: bool = False,
    aggregation: str = "all",
    thresholds: Optional[List[float]] = None,
) -> ReCoNNetwork:
    """
    Build a simple recognition script with terminal feature detectors.

    Args:
        name: Name of the root script unit
        feature_names: Names for the terminal units
        feature_fns: Measurement functions for each feature
        sequential: Whether features should be checked in sequence
        aggregation: "all" or "any" for the root unit's aggregation
        thresholds: Thresholds for each terminal (default: 0.5)

    Returns:
        A ReCoNNetwork representing the recognition script
    """
    if thresholds is None:
        thresholds = [0.5] * len(feature_names)

    builder = ScriptBuilder()

    # Create terminal units for features
    for fname, fn, thresh in zip(feature_names, feature_fns, thresholds):
        builder.terminal(fname, measurement_fn=fn, threshold=thresh)

    # Create root script unit
    builder.script(name, aggregation=aggregation)
    builder.children(feature_names)

    if sequential:
        builder.as_sequence()

    builder.set_root()

    return builder.build()


def build_hierarchical_script(
    structure: Dict[str, Any],
    measurement_fns: Optional[Dict[str, Callable[[], np.ndarray]]] = None,
) -> ReCoNNetwork:
    """
    Build a hierarchical script from a nested dictionary structure.

    The structure dict format:
        {
            "name": "root_name",
            "type": "script" | "terminal",
            "sequential": True | False,  # optional, default False
            "aggregation": "all" | "any",  # optional, default "all"
            "children": [  # for script units
                {"name": "child1", "type": "terminal"},
                {"name": "child2", "type": "script", "children": [...]},
            ]
        }

    Args:
        structure: Nested dict describing the hierarchy
        measurement_fns: Dict mapping terminal names to measurement functions

    Returns:
        A ReCoNNetwork representing the script hierarchy
    """
    if measurement_fns is None:
        measurement_fns = {}

    builder = ScriptBuilder()

    def process_node(node: Dict[str, Any]) -> str:
        """Process a node in the structure recursively."""
        name = node.get("name")
        node_type = node.get("type", "script")

        if node_type == "terminal":
            fn = measurement_fns.get(name)
            threshold = node.get("threshold", 0.5)
            builder.terminal(name, measurement_fn=fn, threshold=threshold)
            return name

        # Script unit
        aggregation = node.get("aggregation", "all")
        builder.script(name, aggregation=aggregation)

        # Process children
        children = node.get("children", [])
        child_ids = []
        for child in children:
            child_id = process_node(child)
            child_ids.append(child_id)

        # Set up children and ordering
        if child_ids:
            builder.select(name)
            builder.children(child_ids)
            if node.get("sequential", False):
                builder.as_sequence()

        return name

    root_name = process_node(structure)
    builder.select(root_name)
    builder.set_root()

    return builder.build()


def build_active_perception_script(
    hypothesis_name: str,
    fovea_positions: List[tuple],
    feature_detector: Callable[[tuple], Callable[[], np.ndarray]],
    sequential: bool = True,
) -> ReCoNNetwork:
    """
    Build an active perception script that tests hypotheses by
    moving a fovea to different positions and detecting features.

    Args:
        hypothesis_name: Name of the hypothesis to test
        fovea_positions: List of (x, y) positions to check
        feature_detector: Function that takes a position and returns a
                          measurement function for that position
        sequential: Whether to check positions in sequence

    Returns:
        A ReCoNNetwork for active perception
    """
    builder = ScriptBuilder()

    # Create terminal units for each fovea position
    position_names = []
    for i, pos in enumerate(fovea_positions):
        name = f"{hypothesis_name}_pos_{i}"
        position_names.append(name)
        fn = feature_detector(pos)
        builder.terminal(name, measurement_fn=fn)

    # Create root hypothesis script
    builder.script(hypothesis_name, aggregation="all")
    builder.children(position_names)

    if sequential:
        builder.as_sequence()

    builder.set_root()

    return builder.build()
