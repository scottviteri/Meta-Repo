"""
Simple grid world environment for testing ReCoN active perception.

This module provides a basic 2D grid environment with:
- Configurable object patterns
- Fovea-based perception
- Simple feature detectors
"""

from typing import Dict, List, Tuple, Optional, Callable
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from recon.env import GridWorldEnvironment, Fovea
from recon.core import ReCoNNetwork, ScriptUnit, TerminalUnit
from recon.scripts import ScriptBuilder
from recon.engine import ReCoNEngine, ExecutionStatus


def create_object_patterns() -> Dict[str, np.ndarray]:
    """
    Create a set of simple object patterns for testing.

    Returns:
        Dict mapping pattern names to 2D numpy arrays
    """
    patterns = {}

    # Simple shapes (8x8)
    # Square
    square = np.zeros((8, 8))
    square[1:7, 1:7] = 1.0
    patterns["square"] = square

    # Circle (approximate)
    circle = np.zeros((8, 8))
    center = 3.5
    for i in range(8):
        for j in range(8):
            if (i - center) ** 2 + (j - center) ** 2 <= 9:
                circle[i, j] = 1.0
    patterns["circle"] = circle

    # Triangle
    triangle = np.zeros((8, 8))
    for i in range(7):
        width = i + 1
        start = (8 - width) // 2
        triangle[i + 1, start:start + width] = 1.0
    patterns["triangle"] = triangle

    # Cross
    cross = np.zeros((8, 8))
    cross[3:5, :] = 1.0
    cross[:, 3:5] = 1.0
    patterns["cross"] = cross

    # L-shape
    l_shape = np.zeros((8, 8))
    l_shape[1:7, 1:3] = 1.0
    l_shape[5:7, 1:7] = 1.0
    patterns["l_shape"] = l_shape

    return patterns


class SimpleGridWorld(GridWorldEnvironment):
    """
    A simple grid world with placeable objects and fovea-based perception.

    Designed for testing ReCoN active perception mechanisms.
    """

    def __init__(
        self,
        width: int = 64,
        height: int = 64,
        fovea_size: Tuple[int, int] = (16, 16),
    ):
        """
        Initialize the grid world.

        Args:
            width: Grid width
            height: Grid height
            fovea_size: Size of the fovea (width, height)
        """
        super().__init__(width, height, num_layers=1)
        self.fovea = Fovea(0, 0, fovea_size[0], fovea_size[1])
        self.objects: Dict[str, Tuple[int, int, np.ndarray]] = {}
        self.patterns = create_object_patterns()

    def place_object(
        self,
        name: str,
        x: int,
        y: int,
        pattern: Optional[np.ndarray] = None,
        pattern_name: Optional[str] = None,
    ):
        """
        Place an object in the world.

        Args:
            name: Unique name for this object instance
            x: X position
            y: Y position
            pattern: Pattern array (if not using named pattern)
            pattern_name: Name of a predefined pattern
        """
        if pattern is None:
            if pattern_name is None:
                raise ValueError("Must provide either pattern or pattern_name")
            pattern = self.patterns[pattern_name]

        self.objects[name] = (x, y, pattern.copy())
        self.add_object(x, y, pattern, layer=0)

    def clear_objects(self):
        """Remove all objects from the world."""
        self.grid.fill(0)
        self.objects.clear()

    def move_fovea_to(self, x: int, y: int):
        """Move the fovea to a position."""
        self.fovea.move_to(x, y)

    def get_fovea_view(self) -> np.ndarray:
        """Get the current fovea view."""
        return self.get_fovea_observation(self.fovea)

    def check_object_at_fovea(self, object_name: str) -> float:
        """
        Check if a specific object is visible in the current fovea view.

        Args:
            object_name: Name of the object to check

        Returns:
            Confidence score (0-1) that the object is present
        """
        if object_name not in self.objects:
            return 0.0

        obj_x, obj_y, pattern = self.objects[object_name]

        # Check if object overlaps with fovea
        fov_x1, fov_y1, fov_x2, fov_y2 = self.fovea.get_bounds()
        obj_x2 = obj_x + pattern.shape[1]
        obj_y2 = obj_y + pattern.shape[0]

        # Check for overlap
        overlap_x1 = max(fov_x1, obj_x)
        overlap_y1 = max(fov_y1, obj_y)
        overlap_x2 = min(fov_x2, obj_x2)
        overlap_y2 = min(fov_y2, obj_y2)

        if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
            return 0.0

        # Compute overlap ratio
        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
        object_area = pattern.shape[0] * pattern.shape[1]

        return overlap_area / object_area

    def create_object_detector(
        self,
        pattern_name: str,
        threshold: float = 0.8,
    ) -> Callable[[], np.ndarray]:
        """
        Create a measurement function that detects a pattern in the fovea.

        Args:
            pattern_name: Name of the pattern to detect
            threshold: Correlation threshold

        Returns:
            Measurement function for a terminal unit
        """
        pattern = self.patterns[pattern_name]
        detector = self.create_pattern_detector(pattern, threshold)

        def measurement_fn() -> np.ndarray:
            obs = self.get_fovea_view()
            return detector(obs)

        return measurement_fn

    def create_position_checker(
        self,
        x: int,
        y: int,
        expected_value: float = 1.0,
        tolerance: float = 0.3,
    ) -> Callable[[], np.ndarray]:
        """
        Create a measurement function that checks a specific position.

        Args:
            x: X position to check
            y: Y position to check
            expected_value: Expected value at the position
            tolerance: Allowed deviation

        Returns:
            Measurement function for a terminal unit
        """
        def measurement_fn() -> np.ndarray:
            # Move fovea to position
            self.fovea.move_to(x, y)
            obs = self.get_fovea_view()

            # Check if center region matches expected value
            center_region = obs[0,
                               obs.shape[1]//4:3*obs.shape[1]//4,
                               obs.shape[2]//4:3*obs.shape[2]//4]

            if center_region.size == 0:
                return np.array([0.0])

            mean_val = np.mean(center_region)
            match_score = 1.0 - min(1.0, abs(mean_val - expected_value) / tolerance)

            return np.array([match_score])

        return measurement_fn


def build_object_recognition_network(
    world: SimpleGridWorld,
    object_name: str,
    check_positions: List[Tuple[int, int]],
    pattern_name: str,
) -> Tuple[ReCoNNetwork, ReCoNEngine]:
    """
    Build a ReCoN network for recognizing an object at multiple positions.

    The network tests the hypothesis that the object is present by
    sequentially checking each position with a pattern detector.

    Args:
        world: The grid world environment
        object_name: Name of the hypothesis (e.g., "is_square_present")
        check_positions: List of (x, y) positions to check
        pattern_name: Name of the pattern to detect

    Returns:
        Tuple of (network, engine)
    """
    builder = ScriptBuilder()

    # Create terminal units for each check position
    terminal_names = []
    for i, (x, y) in enumerate(check_positions):
        name = f"check_pos_{i}"
        terminal_names.append(name)

        # Create measurement function that moves fovea and checks
        def make_measurement_fn(px, py):
            def measurement_fn():
                world.move_fovea_to(px, py)
                detector = world.create_object_detector(pattern_name)
                return detector()
            return measurement_fn

        builder.terminal(name, measurement_fn=make_measurement_fn(x, y))

    # Create root script unit
    builder.script(object_name, aggregation="any")  # Object found if ANY position matches
    builder.children(terminal_names)
    builder.as_sequence()  # Check positions in order
    builder.set_root()

    network = builder.build()
    engine = ReCoNEngine(network)

    return network, engine


def demo_basic_recognition():
    """
    Demonstrate basic object recognition with ReCoN.
    """
    print("=== Basic Object Recognition Demo ===\n")

    # Create world and place a square
    world = SimpleGridWorld(64, 64, fovea_size=(16, 16))
    world.place_object("my_square", x=20, y=20, pattern_name="square")

    print("World created with square at position (20, 20)")
    print(f"Grid shape: {world.grid.shape}")

    # Define positions to check
    positions = [
        (10, 10),   # Empty area
        (20, 20),   # Where the square is!
        (40, 40),   # Empty area
    ]

    print(f"\nChecking positions: {positions}")

    # Build recognition network
    network, engine = build_object_recognition_network(
        world,
        "find_square",
        positions,
        "square"
    )

    print(f"\nNetwork structure:")
    print(f"  Units: {len(network.units)}")
    print(f"  Links: {len(network.links)}")

    # Run recognition
    print("\n--- Running Recognition ---")
    engine.request("find_square")

    for step in range(10):
        result = engine.step()
        print(f"Step {result.step_number}: {result.status.name}")
        print(f"  Active: {result.active_units}")
        print(f"  Confirmed: {result.confirmed_units}")

        if result.status != ExecutionStatus.RUNNING:
            break

    print(f"\nFinal status: {result.status.name}")

    return result.status == ExecutionStatus.CONFIRMED


def demo_scene_recognition():
    """
    Demonstrate scene recognition with multiple objects.
    """
    print("\n=== Scene Recognition Demo ===\n")

    # Create world with multiple objects
    world = SimpleGridWorld(64, 64, fovea_size=(16, 16))
    world.place_object("square1", x=5, y=5, pattern_name="square")
    world.place_object("circle1", x=30, y=10, pattern_name="circle")
    world.place_object("triangle1", x=15, y=40, pattern_name="triangle")

    print("Scene created with:")
    print("  - Square at (5, 5)")
    print("  - Circle at (30, 10)")
    print("  - Triangle at (15, 40)")

    # Build a hierarchical recognition network
    builder = ScriptBuilder()

    # Terminal detectors at known positions
    builder.terminal("detect_square",
                    measurement_fn=world.create_position_checker(5, 5, 1.0))
    builder.terminal("detect_circle",
                    measurement_fn=world.create_position_checker(30, 10, 1.0))
    builder.terminal("detect_triangle",
                    measurement_fn=world.create_position_checker(15, 40, 1.0))

    # Scene recognition script (all objects must be present)
    builder.script("recognize_scene", aggregation="all")
    builder.children(["detect_square", "detect_circle", "detect_triangle"])
    builder.as_parallel()  # Check all in parallel
    builder.set_root()

    network = builder.build()
    engine = ReCoNEngine(network)

    print("\n--- Running Scene Recognition ---")
    engine.request("recognize_scene")

    for step in range(10):
        result = engine.step()
        print(f"Step {result.step_number}: {result.status.name}")

        if result.status != ExecutionStatus.RUNNING:
            break

    print(f"\nScene recognized: {result.status == ExecutionStatus.CONFIRMED}")

    return result.status == ExecutionStatus.CONFIRMED


if __name__ == "__main__":
    # Run demos
    success1 = demo_basic_recognition()
    success2 = demo_scene_recognition()

    print("\n=== Demo Results ===")
    print(f"Basic recognition: {'PASSED' if success1 else 'FAILED'}")
    print(f"Scene recognition: {'PASSED' if success2 else 'FAILED'}")
