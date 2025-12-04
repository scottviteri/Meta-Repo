#!/usr/bin/env python3
"""
Active Perception Demo for Request Confirmation Networks.

This demo shows how ReCoNs can be used for active perception:
- The agent has a limited fovea that can only see part of the scene
- Recognition requires actively moving the fovea to test hypotheses
- Scripts encode where to look and what features to expect

This demonstrates the key insight from Bach & Herger (2015):
"A perceptual representation amounts to a hierarchical script to test
for the presence of the object in the environment."
"""

import sys
import os
from typing import List, Tuple, Dict

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from recon.env import GridWorldEnvironment, Fovea
from recon.core import (
    ReCoNNetwork,
    ScriptUnit,
    TerminalUnit,
    ScriptState,
    TerminalState,
)
from recon.scripts import ScriptBuilder, build_hierarchical_script
from recon.engine import ReCoNEngine, ExecutionStatus
from recon.autoencoder import SimpleAutoencoder, PerceptualFrontEnd


class ActivePerceptionDemo:
    """
    Demonstrates active perception with ReCoNs.

    The demo creates a simple world with objects at known locations,
    then tests whether a ReCoN network can recognize which scene
    is present by actively sampling the environment.
    """

    def __init__(
        self,
        world_size: Tuple[int, int] = (64, 64),
        fovea_size: Tuple[int, int] = (16, 16),
    ):
        """
        Initialize the demo.

        Args:
            world_size: Size of the world (width, height)
            fovea_size: Size of the fovea (width, height)
        """
        self.world_size = world_size
        self.fovea_size = fovea_size
        self.world = GridWorldEnvironment(
            world_size[0], world_size[1],
            num_layers=1,
        )
        self.world.fovea = Fovea(0, 0, fovea_size[0], fovea_size[1])

        # Track fovea movements for visualization
        self.fovea_history: List[Tuple[int, int]] = []

    def create_scene_a(self):
        """
        Create Scene A: Three objects in a diagonal pattern.

        - Square at (10, 10)
        - Circle at (30, 30)
        - Triangle at (50, 50)
        """
        self.world.grid.fill(0)

        # Square (8x8)
        self.world.grid[0, 10:18, 10:18] = 1.0

        # Circle at (30, 30)
        for i in range(8):
            for j in range(8):
                if (i - 3.5) ** 2 + (j - 3.5) ** 2 <= 9:
                    self.world.grid[0, 30 + i, 30 + j] = 1.0

        # Triangle at (50, 50)
        for i in range(8):
            width = i + 1
            start = (8 - width) // 2
            for j in range(width):
                if 50 + i < 64 and 50 + start + j < 64:
                    self.world.grid[0, 50 + i, 50 + start + j] = 1.0

    def create_scene_b(self):
        """
        Create Scene B: Objects in different positions.

        - Circle at (10, 10)
        - Square at (45, 20)
        """
        self.world.grid.fill(0)

        # Circle at (10, 10)
        for i in range(8):
            for j in range(8):
                if (i - 3.5) ** 2 + (j - 3.5) ** 2 <= 9:
                    self.world.grid[0, 10 + i, 10 + j] = 1.0

        # Square at (45, 20)
        self.world.grid[0, 20:28, 45:53] = 1.0

    def create_feature_detector(
        self,
        position: Tuple[int, int],
        expected_mean: float = 0.5,
        tolerance: float = 0.3,
    ):
        """
        Create a feature detector that checks a position.

        Args:
            position: (x, y) position to check
            expected_mean: Expected mean value at the position
            tolerance: Allowed deviation

        Returns:
            Measurement function for a terminal unit
        """
        x, y = position

        def measurement_fn() -> np.ndarray:
            # Move fovea to position
            self.world.fovea.move_to(x, y)
            self.fovea_history.append((x, y))

            # Get observation
            obs = self.world.get_fovea_observation(self.world.fovea)

            # Compute mean value
            mean_val = np.mean(obs)

            # Score based on distance from expected
            if expected_mean > 0:
                # Looking for object presence
                score = min(1.0, mean_val / expected_mean)
            else:
                # Looking for absence
                score = max(0.0, 1.0 - mean_val)

            return np.array([score])

        return measurement_fn

    def build_scene_a_recognizer(self) -> Tuple[ReCoNNetwork, ReCoNEngine]:
        """
        Build a ReCoN network to recognize Scene A.

        The network tests three positions in sequence:
        1. Check for object at (10, 10)
        2. Check for object at (30, 30)
        3. Check for object at (50, 50)

        All three must be present for Scene A to be confirmed.
        """
        builder = ScriptBuilder()

        # Create terminal detectors for each expected object position
        builder.terminal(
            "check_pos_1",
            measurement_fn=self.create_feature_detector((10, 10), 0.5),
            threshold=0.3,
        )
        builder.terminal(
            "check_pos_2",
            measurement_fn=self.create_feature_detector((30, 30), 0.5),
            threshold=0.3,
        )
        builder.terminal(
            "check_pos_3",
            measurement_fn=self.create_feature_detector((50, 50), 0.5),
            threshold=0.3,
        )

        # Create recognition script
        builder.script("recognize_scene_a", aggregation="all")
        builder.children(["check_pos_1", "check_pos_2", "check_pos_3"])
        builder.as_sequence()  # Check positions in order
        builder.set_root()

        network = builder.build()
        engine = ReCoNEngine(network)

        return network, engine

    def build_scene_b_recognizer(self) -> Tuple[ReCoNNetwork, ReCoNEngine]:
        """
        Build a ReCoN network to recognize Scene B.

        Tests for:
        1. Object at (10, 10)
        2. Object at (45, 20)
        3. NO object at (30, 30)
        """
        builder = ScriptBuilder()

        builder.terminal(
            "check_pos_1",
            measurement_fn=self.create_feature_detector((10, 10), 0.5),
            threshold=0.3,
        )
        builder.terminal(
            "check_pos_2",
            measurement_fn=self.create_feature_detector((45, 20), 0.5),
            threshold=0.3,
        )
        builder.terminal(
            "check_pos_3",
            measurement_fn=self.create_feature_detector((30, 30), 0.0),  # Expect empty
            threshold=0.7,
        )

        builder.script("recognize_scene_b", aggregation="all")
        builder.children(["check_pos_1", "check_pos_2", "check_pos_3"])
        builder.as_sequence()
        builder.set_root()

        network = builder.build()
        engine = ReCoNEngine(network)

        return network, engine

    def build_scene_discriminator(self) -> Tuple[ReCoNNetwork, ReCoNEngine]:
        """
        Build a network that discriminates between Scene A and Scene B.

        Uses a hierarchical structure:
        - Root: "identify_scene" (any child succeeds)
          - "hypothesis_scene_a" (all features match)
            - check positions for scene A
          - "hypothesis_scene_b" (all features match)
            - check positions for scene B
        """
        structure = {
            "name": "identify_scene",
            "type": "script",
            "aggregation": "any",  # Either hypothesis can succeed
            "sequential": False,   # Test hypotheses in parallel
            "children": [
                {
                    "name": "hypothesis_scene_a",
                    "type": "script",
                    "aggregation": "all",
                    "sequential": True,
                    "children": [
                        {"name": "a_check_1", "type": "terminal", "threshold": 0.3},
                        {"name": "a_check_2", "type": "terminal", "threshold": 0.3},
                        {"name": "a_check_3", "type": "terminal", "threshold": 0.3},
                    ],
                },
                {
                    "name": "hypothesis_scene_b",
                    "type": "script",
                    "aggregation": "all",
                    "sequential": True,
                    "children": [
                        {"name": "b_check_1", "type": "terminal", "threshold": 0.3},
                        {"name": "b_check_2", "type": "terminal", "threshold": 0.3},
                    ],
                },
            ],
        }

        # Define measurement functions
        measurement_fns = {
            "a_check_1": self.create_feature_detector((10, 10), 0.5),
            "a_check_2": self.create_feature_detector((30, 30), 0.5),
            "a_check_3": self.create_feature_detector((50, 50), 0.5),
            "b_check_1": self.create_feature_detector((10, 10), 0.5),
            "b_check_2": self.create_feature_detector((45, 20), 0.5),
        }

        network = build_hierarchical_script(structure, measurement_fns)
        engine = ReCoNEngine(network)

        return network, engine

    def run_recognition(
        self,
        network: ReCoNNetwork,
        engine: ReCoNEngine,
        root_name: str,
        max_steps: int = 20,
        verbose: bool = True,
    ) -> ExecutionStatus:
        """
        Run the recognition process.

        Args:
            network: The ReCoN network
            engine: The simulation engine
            root_name: Name of the root unit to request
            max_steps: Maximum simulation steps
            verbose: Whether to print progress

        Returns:
            Final execution status
        """
        self.fovea_history.clear()
        engine.reset()

        if verbose:
            print(f"\n--- Starting recognition for '{root_name}' ---")

        engine.request(root_name)

        for step in range(max_steps):
            result = engine.step()

            if verbose:
                print(f"Step {result.step_number}: {result.status.name}")
                if result.active_units:
                    print(f"  Active: {result.active_units}")
                if result.confirmed_units:
                    print(f"  Confirmed: {result.confirmed_units}")
                if result.failed_units:
                    print(f"  Failed: {result.failed_units}")

            if result.status != ExecutionStatus.RUNNING:
                break

        if verbose:
            print(f"\nFinal status: {result.status.name}")
            print(f"Fovea positions visited: {self.fovea_history}")

        return result.status


def main():
    """Run the active perception demo."""
    print("=" * 60)
    print("Active Perception Demo with Request Confirmation Networks")
    print("=" * 60)

    demo = ActivePerceptionDemo()

    # Test 1: Recognize Scene A when Scene A is present
    print("\n\n### Test 1: Recognize Scene A (Scene A present) ###")
    demo.create_scene_a()
    network, engine = demo.build_scene_a_recognizer()
    status = demo.run_recognition(network, engine, "recognize_scene_a")
    test1_passed = status == ExecutionStatus.CONFIRMED
    print(f"\nTest 1 {'PASSED' if test1_passed else 'FAILED'}: "
          f"Scene A {'correctly' if test1_passed else 'incorrectly'} "
          f"{'recognized' if test1_passed else 'not recognized'}")

    # Test 2: Recognize Scene A when Scene B is present (should fail)
    print("\n\n### Test 2: Recognize Scene A (Scene B present) ###")
    demo.create_scene_b()
    engine.reset()
    demo.fovea_history.clear()
    status = demo.run_recognition(network, engine, "recognize_scene_a")
    test2_passed = status == ExecutionStatus.FAILED
    print(f"\nTest 2 {'PASSED' if test2_passed else 'FAILED'}: "
          f"Scene A hypothesis {'correctly' if test2_passed else 'incorrectly'} "
          f"{'rejected' if test2_passed else 'confirmed'}")

    # Test 3: Scene discriminator on Scene A
    print("\n\n### Test 3: Discriminate Scene (Scene A present) ###")
    demo.create_scene_a()
    network, engine = demo.build_scene_discriminator()
    status = demo.run_recognition(network, engine, "identify_scene")

    # Check which hypothesis was confirmed
    hyp_a = network.get_unit("hypothesis_scene_a")
    hyp_b = network.get_unit("hypothesis_scene_b")
    scene_a_confirmed = (hasattr(hyp_a, 'state') and
                         hyp_a.state == ScriptState.CONFIRMED)
    scene_b_confirmed = (hasattr(hyp_b, 'state') and
                         hyp_b.state == ScriptState.CONFIRMED)

    test3_passed = scene_a_confirmed and not scene_b_confirmed
    print(f"\nHypothesis A state: {hyp_a.state.name if hasattr(hyp_a, 'state') else 'N/A'}")
    print(f"Hypothesis B state: {hyp_b.state.name if hasattr(hyp_b, 'state') else 'N/A'}")
    print(f"\nTest 3 {'PASSED' if test3_passed else 'FAILED'}: "
          f"{'Scene A correctly identified' if test3_passed else 'Incorrect identification'}")

    # Test 4: Scene discriminator on Scene B
    print("\n\n### Test 4: Discriminate Scene (Scene B present) ###")
    demo.create_scene_b()
    engine.reset()
    demo.fovea_history.clear()
    status = demo.run_recognition(network, engine, "identify_scene")

    hyp_a = network.get_unit("hypothesis_scene_a")
    hyp_b = network.get_unit("hypothesis_scene_b")
    scene_a_confirmed = (hasattr(hyp_a, 'state') and
                         hyp_a.state == ScriptState.CONFIRMED)
    scene_b_confirmed = (hasattr(hyp_b, 'state') and
                         hyp_b.state == ScriptState.CONFIRMED)

    test4_passed = scene_b_confirmed and not scene_a_confirmed
    print(f"\nHypothesis A state: {hyp_a.state.name if hasattr(hyp_a, 'state') else 'N/A'}")
    print(f"Hypothesis B state: {hyp_b.state.name if hasattr(hyp_b, 'state') else 'N/A'}")
    print(f"\nTest 4 {'PASSED' if test4_passed else 'FAILED'}: "
          f"{'Scene B correctly identified' if test4_passed else 'Incorrect identification'}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    all_passed = test1_passed and test2_passed and test3_passed and test4_passed
    print(f"Test 1 (Recognize A when A present): {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Test 2 (Reject A when B present):    {'PASSED' if test2_passed else 'FAILED'}")
    print(f"Test 3 (Identify scene as A):        {'PASSED' if test3_passed else 'FAILED'}")
    print(f"Test 4 (Identify scene as B):        {'PASSED' if test4_passed else 'FAILED'}")
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
