"""
Environment interface for ReCoN terminal nodes.

This module provides:
- Abstract Environment base class
- GridWorld implementation for simple 2D environments
- Fovea/sensor abstractions for active perception
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Dict, Any, Callable
from dataclasses import dataclass
import numpy as np


@dataclass
class Fovea:
    """
    A foveal sensor that can be positioned in the environment.

    Represents a rectangular region of attention that can be moved
    to different positions in the visual field.
    """
    x: int = 0
    y: int = 0
    width: int = 16
    height: int = 16

    def get_bounds(self) -> Tuple[int, int, int, int]:
        """Get (x_min, y_min, x_max, y_max) bounds."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def move_to(self, x: int, y: int):
        """Move the fovea to a new position."""
        self.x = x
        self.y = y

    def move_by(self, dx: int, dy: int):
        """Move the fovea by a relative offset."""
        self.x += dx
        self.y += dy


class Environment(ABC):
    """
    Abstract base class for environments that ReCoN terminal units
    can interact with.

    Environments provide:
    - State representation (e.g., images, feature maps)
    - Measurement functions for sensors
    - Action execution for actuators
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset the environment to initial state."""
        pass

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        """Get the full observation (e.g., image)."""
        pass

    @abstractmethod
    def get_fovea_observation(self, fovea: Fovea) -> np.ndarray:
        """Get observation within fovea bounds."""
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute an action in the environment.

        Args:
            action: The action to execute

        Returns:
            Tuple of (observation, reward, done, info)
        """
        pass

    def create_feature_detector(
        self,
        feature_fn: Callable[[np.ndarray], np.ndarray],
        fovea: Optional[Fovea] = None,
    ) -> Callable[[], np.ndarray]:
        """
        Create a measurement function for a terminal unit.

        Args:
            feature_fn: Function that extracts features from an observation
            fovea: Optional fovea to use (uses full observation if None)

        Returns:
            A callable that returns feature values when called
        """
        def measurement_fn() -> np.ndarray:
            if fovea is not None:
                obs = self.get_fovea_observation(fovea)
            else:
                obs = self.get_observation()
            return feature_fn(obs)

        return measurement_fn

    def create_position_detector(
        self,
        position: Tuple[int, int],
        feature_fn: Callable[[np.ndarray], np.ndarray],
        fovea_size: Tuple[int, int] = (16, 16),
    ) -> Callable[[], np.ndarray]:
        """
        Create a measurement function that checks a specific position.

        The fovea is moved to the position before measuring.

        Args:
            position: (x, y) position to check
            feature_fn: Function to extract features
            fovea_size: Size of the fovea (width, height)

        Returns:
            A callable that returns feature values at the position
        """
        fovea = Fovea(position[0], position[1], fovea_size[0], fovea_size[1])

        def measurement_fn() -> np.ndarray:
            obs = self.get_fovea_observation(fovea)
            return feature_fn(obs)

        return measurement_fn


class GridWorldEnvironment(Environment):
    """
    A simple 2D grid world environment.

    The world is represented as a 2D array of cell values.
    Supports multiple layers (e.g., terrain, objects, agent).
    """

    def __init__(
        self,
        width: int = 64,
        height: int = 64,
        num_layers: int = 1,
        cell_values: Optional[np.ndarray] = None,
    ):
        """
        Initialize the grid world.

        Args:
            width: Width of the grid
            height: Height of the grid
            num_layers: Number of layers (channels)
            cell_values: Initial cell values (shape: layers x height x width)
        """
        self.width = width
        self.height = height
        self.num_layers = num_layers

        if cell_values is not None:
            self.grid = cell_values.copy()
        else:
            self.grid = np.zeros((num_layers, height, width), dtype=float)

        self._initial_grid = self.grid.copy()

        # Agent position (optional)
        self.agent_pos: Optional[Tuple[int, int]] = None

        # Fovea for active perception
        self.fovea = Fovea(0, 0, 16, 16)

    def reset(self) -> None:
        """Reset to initial state."""
        self.grid = self._initial_grid.copy()
        self.agent_pos = None
        self.fovea.move_to(0, 0)

    def get_observation(self) -> np.ndarray:
        """Get the full grid."""
        return self.grid.copy()

    def get_fovea_observation(self, fovea: Fovea) -> np.ndarray:
        """Get grid values within fovea bounds."""
        x1, y1, x2, y2 = fovea.get_bounds()

        # Clip to grid bounds
        x1 = max(0, min(x1, self.width))
        x2 = max(0, min(x2, self.width))
        y1 = max(0, min(y1, self.height))
        y2 = max(0, min(y2, self.height))

        return self.grid[:, y1:y2, x1:x2].copy()

    def set_cell(self, x: int, y: int, value: float, layer: int = 0):
        """Set a cell value."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[layer, y, x] = value

    def get_cell(self, x: int, y: int, layer: int = 0) -> float:
        """Get a cell value."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[layer, y, x]
        return 0.0

    def set_region(
        self,
        x: int,
        y: int,
        values: np.ndarray,
        layer: int = 0,
    ):
        """Set a rectangular region of cells."""
        h, w = values.shape
        x2 = min(x + w, self.width)
        y2 = min(y + h, self.height)
        self.grid[layer, y:y2, x:x2] = values[:y2-y, :x2-x]

    def add_object(
        self,
        x: int,
        y: int,
        pattern: np.ndarray,
        layer: int = 0,
    ):
        """Add an object (pattern) to the grid."""
        self.set_region(x, y, pattern, layer)

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute an action.

        Supported actions:
        - "move_fovea": (x, y) - move fovea to position
        - "move_agent": (dx, dy) - move agent by offset

        Args:
            action: Tuple of (action_type, parameters)

        Returns:
            (observation, reward, done, info)
        """
        action_type, params = action
        reward = 0.0
        done = False
        info = {}

        if action_type == "move_fovea":
            x, y = params
            self.fovea.move_to(x, y)
            info["fovea_pos"] = (x, y)

        elif action_type == "move_agent":
            if self.agent_pos is not None:
                dx, dy = params
                new_x = self.agent_pos[0] + dx
                new_y = self.agent_pos[1] + dy

                # Bounds checking
                if 0 <= new_x < self.width and 0 <= new_y < self.height:
                    self.agent_pos = (new_x, new_y)
                    info["agent_pos"] = self.agent_pos

        return self.get_observation(), reward, done, info

    def create_pattern_detector(
        self,
        pattern: np.ndarray,
        threshold: float = 0.8,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Create a feature function that detects a pattern.

        Args:
            pattern: The pattern to detect (2D array)
            threshold: Correlation threshold for detection

        Returns:
            Feature function that returns match score
        """
        pattern_flat = pattern.flatten()
        pattern_norm = np.linalg.norm(pattern_flat)

        def feature_fn(obs: np.ndarray) -> np.ndarray:
            # Use first layer if multi-layer
            if obs.ndim == 3:
                obs = obs[0]

            # Resize observation to match pattern if needed
            if obs.shape != pattern.shape:
                # Simple case: compute normalized correlation
                obs_flat = obs.flatten()
                if len(obs_flat) != len(pattern_flat):
                    # Different sizes - use simple mean comparison
                    return np.array([np.mean(obs)])

            obs_flat = obs.flatten()
            obs_norm = np.linalg.norm(obs_flat)

            if obs_norm == 0 or pattern_norm == 0:
                return np.array([0.0])

            # Normalized correlation
            correlation = np.dot(obs_flat, pattern_flat) / (obs_norm * pattern_norm)

            return np.array([max(0, correlation)])

        return feature_fn

    def create_value_detector(
        self,
        target_value: float,
        tolerance: float = 0.1,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Create a feature function that detects a specific value.

        Args:
            target_value: The value to detect
            tolerance: Allowed deviation from target

        Returns:
            Feature function that returns match score
        """
        def feature_fn(obs: np.ndarray) -> np.ndarray:
            mean_value = np.mean(obs)
            distance = abs(mean_value - target_value)
            score = max(0, 1.0 - distance / (tolerance + 1e-6))
            return np.array([score])

        return feature_fn


class VisualEnvironment(Environment):
    """
    Environment that uses images as observations.

    Can load images from files or generate synthetic scenes.
    """

    def __init__(
        self,
        image: Optional[np.ndarray] = None,
        width: int = 256,
        height: int = 256,
        channels: int = 3,
    ):
        """
        Initialize the visual environment.

        Args:
            image: Initial image (H x W x C or H x W)
            width: Image width if no image provided
            height: Image height if no image provided
            channels: Number of channels if no image provided
        """
        if image is not None:
            self.image = image.astype(float)
            if self.image.ndim == 2:
                self.image = self.image[:, :, np.newaxis]
            self.height, self.width, self.channels = self.image.shape
        else:
            self.width = width
            self.height = height
            self.channels = channels
            self.image = np.zeros((height, width, channels), dtype=float)

        self._initial_image = self.image.copy()
        self.fovea = Fovea(0, 0, 16, 16)

    def reset(self) -> None:
        """Reset to initial image."""
        self.image = self._initial_image.copy()
        self.fovea.move_to(0, 0)

    def set_image(self, image: np.ndarray):
        """Set the current image."""
        self.image = image.astype(float)
        if self.image.ndim == 2:
            self.image = self.image[:, :, np.newaxis]

    def get_observation(self) -> np.ndarray:
        """Get the full image."""
        return self.image.copy()

    def get_fovea_observation(self, fovea: Fovea) -> np.ndarray:
        """Get image region within fovea bounds."""
        x1, y1, x2, y2 = fovea.get_bounds()

        x1 = max(0, min(x1, self.width))
        x2 = max(0, min(x2, self.width))
        y1 = max(0, min(y1, self.height))
        y2 = max(0, min(y2, self.height))

        return self.image[y1:y2, x1:x2, :].copy()

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute an action (mainly fovea movement)."""
        action_type, params = action
        info = {}

        if action_type == "move_fovea":
            x, y = params
            self.fovea.move_to(x, y)
            info["fovea_pos"] = (x, y)

        return self.get_observation(), 0.0, False, info

    def create_edge_detector(self) -> Callable[[np.ndarray], np.ndarray]:
        """Create a simple edge detection feature function."""
        def feature_fn(obs: np.ndarray) -> np.ndarray:
            if obs.ndim == 3:
                obs = np.mean(obs, axis=2)

            # Simple gradient-based edge detection
            dx = np.diff(obs, axis=1)
            dy = np.diff(obs, axis=0)

            edge_strength = np.mean(np.abs(dx)) + np.mean(np.abs(dy))
            return np.array([edge_strength])

        return feature_fn

    def create_color_detector(
        self,
        target_color: np.ndarray,
        tolerance: float = 0.2,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Create a color detection feature function."""
        target_color = np.array(target_color, dtype=float)

        def feature_fn(obs: np.ndarray) -> np.ndarray:
            if obs.ndim == 2:
                obs = obs[:, :, np.newaxis]

            mean_color = np.mean(obs, axis=(0, 1))

            # Ensure shapes match
            if len(mean_color) != len(target_color):
                return np.array([0.0])

            distance = np.linalg.norm(mean_color - target_color)
            max_distance = np.linalg.norm(target_color) + tolerance
            score = max(0, 1.0 - distance / max_distance)

            return np.array([score])

        return feature_fn
