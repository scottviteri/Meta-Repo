"""
Minecraft-like voxel world environment for ReCoN active perception.

This module provides a simplified 3D voxel environment with:
- Multiple block types
- 2D projections for perception
- Location hypothesis testing
"""

from enum import IntEnum
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from recon.env import Environment, Fovea
from recon.core import ReCoNNetwork
from recon.scripts import ScriptBuilder
from recon.engine import ReCoNEngine, ExecutionStatus


class BlockType(IntEnum):
    """Types of blocks in the Minecraft-like world."""
    AIR = 0
    STONE = 1
    DIRT = 2
    GRASS = 3
    WOOD = 4
    LEAVES = 5
    WATER = 6
    SAND = 7
    BRICK = 8


# Visual properties for each block type (for rendering)
BLOCK_COLORS = {
    BlockType.AIR: (0.7, 0.85, 1.0),      # Sky blue
    BlockType.STONE: (0.5, 0.5, 0.5),      # Gray
    BlockType.DIRT: (0.6, 0.4, 0.2),       # Brown
    BlockType.GRASS: (0.3, 0.7, 0.3),      # Green
    BlockType.WOOD: (0.5, 0.3, 0.1),       # Dark brown
    BlockType.LEAVES: (0.2, 0.6, 0.2),     # Dark green
    BlockType.WATER: (0.2, 0.4, 0.8),      # Blue
    BlockType.SAND: (0.9, 0.8, 0.5),       # Yellow
    BlockType.BRICK: (0.7, 0.3, 0.2),      # Reddish
}


@dataclass
class Location:
    """A named location in the world with characteristic features."""
    name: str
    x: int
    y: int
    z: int
    feature_blocks: List[Tuple[Tuple[int, int, int], BlockType]]
    description: str = ""


class MinecraftLikeWorld(Environment):
    """
    A simplified Minecraft-like voxel world.

    The world is represented as a 3D grid of blocks. For perception,
    we project views onto 2D planes that can be processed by the
    ReCoN system.
    """

    def __init__(
        self,
        width: int = 64,
        height: int = 32,
        depth: int = 64,
        fovea_size: Tuple[int, int] = (16, 16),
    ):
        """
        Initialize the voxel world.

        Args:
            width: X dimension
            height: Y dimension (vertical)
            depth: Z dimension
            fovea_size: Size of the perceptual fovea
        """
        self.width = width
        self.height = height
        self.depth = depth

        # 3D voxel grid (y, z, x) for easier slicing
        self.voxels = np.zeros((height, depth, width), dtype=np.int8)

        # Perceptual state
        self.fovea = Fovea(0, 0, fovea_size[0], fovea_size[1])
        self.camera_pos = (width // 2, height // 2, depth // 2)
        self.camera_dir = (0, 0, 1)  # Looking in +Z direction

        # Named locations
        self.locations: Dict[str, Location] = {}

        # Generate basic terrain
        self._generate_terrain()

    def _generate_terrain(self):
        """Generate simple terrain with hills."""
        for x in range(self.width):
            for z in range(self.depth):
                # Simple height variation
                base_height = 8
                height_var = int(3 * np.sin(x * 0.1) * np.cos(z * 0.1))
                ground_level = base_height + height_var

                # Fill below ground
                for y in range(ground_level):
                    if y < ground_level - 3:
                        self.voxels[y, z, x] = BlockType.STONE
                    else:
                        self.voxels[y, z, x] = BlockType.DIRT

                # Top layer is grass
                if ground_level < self.height:
                    self.voxels[ground_level, z, x] = BlockType.GRASS

    def set_block(self, x: int, y: int, z: int, block_type: BlockType):
        """Set a block at the given position."""
        if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth:
            self.voxels[y, z, x] = block_type

    def get_block(self, x: int, y: int, z: int) -> BlockType:
        """Get the block type at a position."""
        if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth:
            return BlockType(self.voxels[y, z, x])
        return BlockType.AIR

    def build_structure(
        self,
        x: int,
        y: int,
        z: int,
        structure: List[Tuple[Tuple[int, int, int], BlockType]],
    ):
        """
        Build a structure from a list of relative block positions.

        Args:
            x, y, z: Base position
            structure: List of ((dx, dy, dz), block_type) tuples
        """
        for (dx, dy, dz), block_type in structure:
            self.set_block(x + dx, y + dy, z + dz, block_type)

    def build_tree(self, x: int, y: int, z: int, height: int = 5):
        """Build a simple tree at the given position."""
        # Trunk
        for dy in range(height):
            self.set_block(x, y + dy, z, BlockType.WOOD)

        # Leaves (simple cube on top)
        for dx in range(-2, 3):
            for dy in range(-1, 2):
                for dz in range(-2, 3):
                    if abs(dx) + abs(dz) <= 3:
                        self.set_block(x + dx, y + height + dy, z + dz, BlockType.LEAVES)

    def build_house(self, x: int, y: int, z: int, width: int = 5, depth: int = 6, height: int = 4):
        """Build a simple house structure."""
        # Walls
        for dx in range(width):
            for dy in range(height):
                for dz in range(depth):
                    # Only build walls (not interior)
                    if dx == 0 or dx == width - 1 or dz == 0 or dz == depth - 1:
                        self.set_block(x + dx, y + dy, z + dz, BlockType.BRICK)
                    elif dy == height - 1:
                        # Roof
                        self.set_block(x + dx, y + dy, z + dz, BlockType.WOOD)

        # Door opening
        self.set_block(x + width // 2, y, z, BlockType.AIR)
        self.set_block(x + width // 2, y + 1, z, BlockType.AIR)

    def add_location(self, location: Location):
        """Add a named location to the world."""
        self.locations[location.name] = location

    def reset(self):
        """Reset the world."""
        self.voxels.fill(0)
        self._generate_terrain()

    def get_observation(self) -> np.ndarray:
        """Get a 2D projection of the world (top-down view)."""
        # Top-down view: for each (x, z), find highest non-air block
        projection = np.zeros((self.depth, self.width, 3), dtype=float)

        for x in range(self.width):
            for z in range(self.depth):
                # Find highest block
                for y in range(self.height - 1, -1, -1):
                    block = BlockType(self.voxels[y, z, x])
                    if block != BlockType.AIR:
                        color = BLOCK_COLORS[block]
                        projection[z, x] = color
                        break

        return projection

    def get_front_view(self, x: int, z: int, width: int = 16) -> np.ndarray:
        """
        Get a front view (x-y plane) from a position looking in +z.

        Args:
            x: X position of viewer
            z: Z position (looking towards +z)
            width: View width

        Returns:
            2D array of block colors
        """
        half_w = width // 2
        view = np.zeros((self.height, width, 3), dtype=float)

        for dx in range(-half_w, half_w):
            vx = x + dx
            if 0 <= vx < self.width:
                for y in range(self.height):
                    # Ray cast in +z direction
                    for vz in range(z, min(z + 32, self.depth)):
                        block = self.get_block(vx, y, vz)
                        if block != BlockType.AIR:
                            view[self.height - 1 - y, dx + half_w] = BLOCK_COLORS[block]
                            break

        return view

    def get_fovea_observation(self, fovea: Fovea) -> np.ndarray:
        """Get a region from the top-down projection."""
        full_obs = self.get_observation()
        x1, y1, x2, y2 = fovea.get_bounds()

        x1 = max(0, min(x1, self.width))
        x2 = max(0, min(x2, self.width))
        y1 = max(0, min(y1, self.depth))
        y2 = max(0, min(y2, self.depth))

        return full_obs[y1:y2, x1:x2, :]

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute an action (fovea movement, camera movement)."""
        action_type, params = action
        info = {}

        if action_type == "move_fovea":
            x, z = params
            self.fovea.move_to(x, z)
            info["fovea_pos"] = (x, z)

        elif action_type == "move_camera":
            dx, dy, dz = params
            cx, cy, cz = self.camera_pos
            self.camera_pos = (cx + dx, cy + dy, cz + dz)
            info["camera_pos"] = self.camera_pos

        return self.get_observation(), 0.0, False, info

    def create_location_detector(
        self,
        location_name: str,
        check_radius: int = 2,
    ) -> Callable[[], np.ndarray]:
        """
        Create a measurement function that checks if we're at a location.

        Args:
            location_name: Name of the location
            check_radius: Radius to check around the location center

        Returns:
            Measurement function for a terminal unit
        """
        location = self.locations.get(location_name)
        if location is None:
            raise ValueError(f"Unknown location: {location_name}")

        def measurement_fn() -> np.ndarray:
            # Check if the characteristic blocks are present
            matches = 0
            total = len(location.feature_blocks)

            if total == 0:
                return np.array([0.0])

            for (dx, dy, dz), expected_block in location.feature_blocks:
                actual = self.get_block(
                    location.x + dx,
                    location.y + dy,
                    location.z + dz
                )
                if actual == expected_block:
                    matches += 1

            score = matches / total
            return np.array([score])

        return measurement_fn

    def create_block_detector(
        self,
        x: int,
        y: int,
        z: int,
        expected_block: BlockType,
    ) -> Callable[[], np.ndarray]:
        """
        Create a measurement function that checks for a specific block.

        Args:
            x, y, z: Position to check
            expected_block: Expected block type

        Returns:
            Measurement function for a terminal unit
        """
        def measurement_fn() -> np.ndarray:
            actual = self.get_block(x, y, z)
            match = 1.0 if actual == expected_block else 0.0
            return np.array([match])

        return measurement_fn


def build_location_recognition_network(
    world: MinecraftLikeWorld,
    location_names: List[str],
) -> Tuple[ReCoNNetwork, ReCoNEngine]:
    """
    Build a ReCoN network for recognizing which location we're at.

    Creates a network with one hypothesis per location, where each
    hypothesis checks for characteristic features.

    Args:
        world: The Minecraft-like world
        location_names: Names of locations to recognize

    Returns:
        Tuple of (network, engine)
    """
    builder = ScriptBuilder()

    # Create a detector for each location
    for loc_name in location_names:
        location = world.locations[loc_name]

        # Create terminal units for each feature block
        feature_terminals = []
        for i, ((dx, dy, dz), expected_block) in enumerate(location.feature_blocks):
            terminal_name = f"{loc_name}_feature_{i}"
            feature_terminals.append(terminal_name)

            detector = world.create_block_detector(
                location.x + dx,
                location.y + dy,
                location.z + dz,
                expected_block,
            )
            builder.terminal(terminal_name, measurement_fn=detector)

        # Create hypothesis script for this location
        builder.script(f"hypothesis_{loc_name}", aggregation="all")
        builder.children(feature_terminals)
        builder.as_parallel()

    # Create top-level "where am I" script
    hypothesis_names = [f"hypothesis_{name}" for name in location_names]
    builder.script("identify_location", aggregation="any")
    builder.children(hypothesis_names)
    builder.as_parallel()
    builder.set_root()

    network = builder.build()
    engine = ReCoNEngine(network)

    return network, engine


def demo_location_recognition():
    """
    Demonstrate location recognition in a Minecraft-like world.
    """
    print("=== Minecraft-like World Location Recognition Demo ===\n")

    # Create world
    world = MinecraftLikeWorld(64, 32, 64)

    # Build some structures
    world.build_tree(20, 10, 20)
    world.build_house(40, 10, 30)

    # Define locations based on their features
    forest_location = Location(
        name="forest",
        x=20, y=10, z=20,
        feature_blocks=[
            ((0, 0, 0), BlockType.WOOD),   # Tree trunk
            ((0, 5, 0), BlockType.LEAVES), # Tree leaves
            ((1, 5, 0), BlockType.LEAVES),
        ],
        description="A forested area with trees"
    )

    house_location = Location(
        name="house",
        x=40, y=10, z=30,
        feature_blocks=[
            ((0, 0, 0), BlockType.BRICK),  # Wall
            ((4, 0, 0), BlockType.BRICK),  # Opposite wall
            ((2, 3, 2), BlockType.WOOD),   # Roof
        ],
        description="A brick house"
    )

    world.add_location(forest_location)
    world.add_location(house_location)

    print("World created with:")
    print("  - Tree at (20, 10, 20)")
    print("  - House at (40, 10, 30)")

    # Build recognition network
    network, engine = build_location_recognition_network(
        world,
        ["forest", "house"]
    )

    print(f"\nNetwork structure:")
    print(f"  Units: {len(network.units)}")
    print(f"  Links: {len(network.links)}")

    # Run recognition
    print("\n--- Running Location Recognition ---")
    engine.request("identify_location")

    for step in range(15):
        result = engine.step()
        print(f"Step {result.step_number}: {result.status.name}")
        if result.confirmed_units:
            print(f"  Confirmed: {result.confirmed_units}")

        if result.status != ExecutionStatus.RUNNING:
            break

    # Check which location was recognized
    forest_unit = network.get_unit("hypothesis_forest")
    house_unit = network.get_unit("hypothesis_house")

    print("\n--- Recognition Results ---")
    if hasattr(forest_unit, 'state'):
        print(f"Forest hypothesis: {forest_unit.state.name}")
    if hasattr(house_unit, 'state'):
        print(f"House hypothesis: {house_unit.state.name}")

    print(f"\nFinal status: {result.status.name}")

    return result.status == ExecutionStatus.CONFIRMED


if __name__ == "__main__":
    success = demo_location_recognition()
    print(f"\n=== Demo {'PASSED' if success else 'FAILED'} ===")
