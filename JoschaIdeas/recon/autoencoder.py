"""
Perceptual front-end using denoising autoencoders.

This module provides neural network components for learning
compressed perceptual representations that can be used by
ReCoN terminal units for feature detection.

The implementation uses PyTorch if available, with a NumPy fallback.
"""

from typing import Optional, Tuple, List, Callable
from abc import ABC, abstractmethod
import numpy as np

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    @abstractmethod
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input to latent representation."""
        pass

    @abstractmethod
    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent representation to reconstruction."""
        pass

    @abstractmethod
    def train_step(self, x: np.ndarray, noise_level: float = 0.1) -> float:
        """Perform one training step on a batch."""
        pass

    def create_feature_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        """Create a feature function for ReCoN terminal units."""
        def feature_fn(obs: np.ndarray) -> np.ndarray:
            return self.encode(obs)
        return feature_fn


class SimpleAutoencoder(FeatureExtractor):
    """
    A simple NumPy-based autoencoder for environments where
    PyTorch is not available.

    Uses a single hidden layer with sigmoid activations.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: Optional[int] = None,
        learning_rate: float = 0.01,
    ):
        """
        Initialize the autoencoder.

        Args:
            input_dim: Dimension of input (flattened)
            latent_dim: Dimension of latent representation
            hidden_dim: Hidden layer dimension (default: average of input and latent)
            learning_rate: Learning rate for gradient descent
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim or (input_dim + latent_dim) // 2
        self.lr = learning_rate

        # Initialize weights with Xavier initialization
        scale1 = np.sqrt(2.0 / (input_dim + self.hidden_dim))
        scale2 = np.sqrt(2.0 / (self.hidden_dim + latent_dim))

        self.W1 = np.random.randn(input_dim, self.hidden_dim) * scale1
        self.b1 = np.zeros(self.hidden_dim)

        self.W2 = np.random.randn(self.hidden_dim, latent_dim) * scale2
        self.b2 = np.zeros(latent_dim)

        self.W3 = np.random.randn(latent_dim, self.hidden_dim) * scale2
        self.b3 = np.zeros(self.hidden_dim)

        self.W4 = np.random.randn(self.hidden_dim, input_dim) * scale1
        self.b4 = np.zeros(input_dim)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation with numerical stability."""
        return np.where(x >= 0,
                       1 / (1 + np.exp(-x)),
                       np.exp(x) / (1 + np.exp(x)))

    def _sigmoid_grad(self, s: np.ndarray) -> np.ndarray:
        """Gradient of sigmoid: s * (1 - s)."""
        return s * (1 - s)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input to latent representation."""
        x_flat = x.flatten()
        if len(x_flat) != self.input_dim:
            # Resize if needed
            x_flat = np.resize(x_flat, self.input_dim)

        h1 = self._sigmoid(x_flat @ self.W1 + self.b1)
        z = self._sigmoid(h1 @ self.W2 + self.b2)
        return z

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent representation."""
        h3 = self._sigmoid(z @ self.W3 + self.b3)
        x_recon = self._sigmoid(h3 @ self.W4 + self.b4)
        return x_recon

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Forward pass with intermediate values for backprop."""
        x_flat = x.flatten()
        if len(x_flat) != self.input_dim:
            x_flat = np.resize(x_flat, self.input_dim)

        # Encoder
        z1 = x_flat @ self.W1 + self.b1
        h1 = self._sigmoid(z1)

        z2 = h1 @ self.W2 + self.b2
        z = self._sigmoid(z2)

        # Decoder
        z3 = z @ self.W3 + self.b3
        h3 = self._sigmoid(z3)

        z4 = h3 @ self.W4 + self.b4
        x_recon = self._sigmoid(z4)

        cache = {
            'x': x_flat, 'z1': z1, 'h1': h1, 'z2': z2, 'z': z,
            'z3': z3, 'h3': h3, 'z4': z4, 'x_recon': x_recon
        }
        return x_recon, cache

    def train_step(self, x: np.ndarray, noise_level: float = 0.1) -> float:
        """
        Perform one training step with denoising.

        Args:
            x: Input data (will be flattened)
            noise_level: Standard deviation of Gaussian noise

        Returns:
            Reconstruction loss
        """
        x_flat = x.flatten()
        if len(x_flat) != self.input_dim:
            x_flat = np.resize(x_flat, self.input_dim)

        # Add noise for denoising autoencoder
        x_noisy = x_flat + noise_level * np.random.randn(self.input_dim)
        x_noisy = np.clip(x_noisy, 0, 1)

        # Forward pass
        x_recon, cache = self.forward(x_noisy)

        # Compute loss (MSE)
        loss = np.mean((x_recon - x_flat) ** 2)

        # Backward pass
        batch_size = 1  # Single sample
        d_x_recon = 2 * (x_recon - x_flat) / self.input_dim

        # Layer 4
        d_z4 = d_x_recon * self._sigmoid_grad(cache['x_recon'])
        d_W4 = np.outer(cache['h3'], d_z4)
        d_b4 = d_z4
        d_h3 = d_z4 @ self.W4.T

        # Layer 3
        d_z3 = d_h3 * self._sigmoid_grad(cache['h3'])
        d_W3 = np.outer(cache['z'], d_z3)
        d_b3 = d_z3
        d_z = d_z3 @ self.W3.T

        # Layer 2
        d_z2 = d_z * self._sigmoid_grad(cache['z'])
        d_W2 = np.outer(cache['h1'], d_z2)
        d_b2 = d_z2
        d_h1 = d_z2 @ self.W2.T

        # Layer 1
        d_z1 = d_h1 * self._sigmoid_grad(cache['h1'])
        d_W1 = np.outer(cache['x'], d_z1)
        d_b1 = d_z1

        # Update weights
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2
        self.W3 -= self.lr * d_W3
        self.b3 -= self.lr * d_b3
        self.W4 -= self.lr * d_W4
        self.b4 -= self.lr * d_b4

        return loss

    def train(
        self,
        data: np.ndarray,
        epochs: int = 100,
        noise_level: float = 0.1,
        verbose: bool = True,
    ) -> List[float]:
        """
        Train on a dataset.

        Args:
            data: Training data (N x input_dim or N x H x W x C)
            epochs: Number of training epochs
            noise_level: Noise level for denoising
            verbose: Whether to print progress

        Returns:
            List of losses per epoch
        """
        losses = []
        n_samples = len(data)

        for epoch in range(epochs):
            epoch_loss = 0.0
            indices = np.random.permutation(n_samples)

            for idx in indices:
                loss = self.train_step(data[idx], noise_level)
                epoch_loss += loss

            epoch_loss /= n_samples
            losses.append(epoch_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")

        return losses


if TORCH_AVAILABLE:
    class ConvAutoencoder(nn.Module, FeatureExtractor):
        """
        Convolutional autoencoder for image-based perception.

        Uses PyTorch for GPU acceleration and automatic differentiation.
        """

        def __init__(
            self,
            input_channels: int = 1,
            latent_dim: int = 32,
            input_size: Tuple[int, int] = (16, 16),
            learning_rate: float = 0.001,
            device: Optional[str] = None,
        ):
            """
            Initialize the convolutional autoencoder.

            Args:
                input_channels: Number of input channels
                latent_dim: Dimension of latent representation
                input_size: (height, width) of input images
                learning_rate: Learning rate for Adam optimizer
                device: Device to use ('cuda', 'cpu', or None for auto)
            """
            nn.Module.__init__(self)

            self.input_channels = input_channels
            self.latent_dim = latent_dim
            self.input_size = input_size

            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)

            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(input_channels, 16, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )

            # Calculate flattened size
            with torch.no_grad():
                dummy = torch.zeros(1, input_channels, *input_size)
                flat_size = self.encoder(dummy).shape[1]

            self.fc_encode = nn.Linear(flat_size, latent_dim)

            # Decoder
            self.fc_decode = nn.Linear(latent_dim, flat_size)

            h_out = (input_size[0] + 3) // 4
            w_out = (input_size[1] + 3) // 4
            self.decoder_reshape = (32, h_out, w_out)

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(16, input_channels, 3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid(),
            )

            self.to(self.device)
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        def encode(self, x: np.ndarray) -> np.ndarray:
            """Encode input to latent representation."""
            self.eval()
            with torch.no_grad():
                x_tensor = self._prepare_input(x)
                h = self.encoder(x_tensor)
                z = self.fc_encode(h)
                return z.cpu().numpy().flatten()

        def decode(self, z: np.ndarray) -> np.ndarray:
            """Decode latent representation."""
            self.eval()
            with torch.no_grad():
                z_tensor = torch.tensor(z, dtype=torch.float32).to(self.device)
                if z_tensor.dim() == 1:
                    z_tensor = z_tensor.unsqueeze(0)
                h = self.fc_decode(z_tensor)
                h = h.view(-1, *self.decoder_reshape)
                x_recon = self.decoder(h)
                return x_recon.cpu().numpy().squeeze()

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass returning reconstruction and latent."""
            h = self.encoder(x)
            z = self.fc_encode(h)
            h_dec = self.fc_decode(z)
            h_dec = h_dec.view(-1, *self.decoder_reshape)
            x_recon = self.decoder(h_dec)

            # Crop to input size if needed
            x_recon = x_recon[:, :, :x.shape[2], :x.shape[3]]

            return x_recon, z

        def _prepare_input(self, x: np.ndarray) -> torch.Tensor:
            """Prepare numpy input for the network."""
            if x.ndim == 2:
                x = x[np.newaxis, np.newaxis, :, :]
            elif x.ndim == 3:
                x = x[np.newaxis, :, :, :]
                if x.shape[1] > x.shape[3]:
                    x = np.transpose(x, (0, 3, 1, 2))

            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            return x_tensor

        def train_step(self, x: np.ndarray, noise_level: float = 0.1) -> float:
            """Perform one training step with denoising."""
            self.train()
            x_tensor = self._prepare_input(x)

            # Add noise
            noise = torch.randn_like(x_tensor) * noise_level
            x_noisy = torch.clamp(x_tensor + noise, 0, 1)

            self.optimizer.zero_grad()
            x_recon, _ = self.forward(x_noisy)

            loss = F.mse_loss(x_recon, x_tensor)
            loss.backward()
            self.optimizer.step()

            return loss.item()

        def train_batch(
            self,
            data: np.ndarray,
            epochs: int = 100,
            batch_size: int = 32,
            noise_level: float = 0.1,
            verbose: bool = True,
        ) -> List[float]:
            """
            Train on a dataset.

            Args:
                data: Training data (N x C x H x W or N x H x W)
                epochs: Number of training epochs
                batch_size: Batch size
                noise_level: Noise level for denoising
                verbose: Whether to print progress

            Returns:
                List of losses per epoch
            """
            if data.ndim == 3:
                data = data[:, np.newaxis, :, :]

            dataset = torch.tensor(data, dtype=torch.float32)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )

            losses = []

            for epoch in range(epochs):
                epoch_loss = 0.0
                n_batches = 0

                for batch in dataloader:
                    batch = batch.to(self.device)

                    # Add noise
                    noise = torch.randn_like(batch) * noise_level
                    batch_noisy = torch.clamp(batch + noise, 0, 1)

                    self.optimizer.zero_grad()
                    recon, _ = self.forward(batch_noisy)
                    loss = F.mse_loss(recon, batch)
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                epoch_loss /= n_batches
                losses.append(epoch_loss)

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")

            return losses


def create_autoencoder(
    input_dim: int = 256,
    latent_dim: int = 16,
    use_conv: bool = False,
    **kwargs,
) -> FeatureExtractor:
    """
    Factory function to create an appropriate autoencoder.

    Args:
        input_dim: Input dimension (for MLP) or ignored (for conv)
        latent_dim: Latent dimension
        use_conv: Whether to use convolutional autoencoder
        **kwargs: Additional arguments passed to the constructor

    Returns:
        A FeatureExtractor instance
    """
    if use_conv and TORCH_AVAILABLE:
        return ConvAutoencoder(latent_dim=latent_dim, **kwargs)
    else:
        return SimpleAutoencoder(input_dim, latent_dim, **kwargs)


class PerceptualFrontEnd:
    """
    Complete perceptual front-end that combines an autoencoder
    with ReCoN integration.

    Provides methods to:
    - Train on environment observations
    - Create feature detectors for terminal units
    - Extract learned features from observations
    """

    def __init__(
        self,
        autoencoder: FeatureExtractor,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize the perceptual front-end.

        Args:
            autoencoder: The autoencoder to use for feature extraction
            feature_names: Optional names for latent dimensions
        """
        self.autoencoder = autoencoder
        self.feature_names = feature_names

    def extract_features(self, observation: np.ndarray) -> np.ndarray:
        """Extract latent features from an observation."""
        return self.autoencoder.encode(observation)

    def create_feature_detector(
        self,
        feature_index: int,
        threshold: float = 0.5,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Create a feature detector for a specific latent dimension.

        Args:
            feature_index: Index of the latent dimension
            threshold: Threshold for detection (not used in output,
                      but can inform the terminal unit)

        Returns:
            Feature function for a terminal unit
        """
        def feature_fn(obs: np.ndarray) -> np.ndarray:
            z = self.autoencoder.encode(obs)
            return np.array([z[feature_index]])

        return feature_fn

    def create_similarity_detector(
        self,
        reference: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Create a detector that measures similarity to a reference.

        Args:
            reference: Reference observation

        Returns:
            Feature function that returns similarity score
        """
        z_ref = self.autoencoder.encode(reference)
        z_ref_norm = np.linalg.norm(z_ref)

        def feature_fn(obs: np.ndarray) -> np.ndarray:
            z = self.autoencoder.encode(obs)
            z_norm = np.linalg.norm(z)

            if z_norm == 0 or z_ref_norm == 0:
                return np.array([0.0])

            similarity = np.dot(z, z_ref) / (z_norm * z_ref_norm)
            return np.array([max(0, similarity)])

        return feature_fn

    def train_on_environment(
        self,
        env,
        n_samples: int = 1000,
        epochs: int = 50,
        noise_level: float = 0.1,
        random_positions: bool = True,
    ) -> List[float]:
        """
        Train the autoencoder on observations from an environment.

        Args:
            env: Environment to sample from
            n_samples: Number of samples to collect
            epochs: Training epochs
            noise_level: Denoising noise level
            random_positions: Whether to sample from random fovea positions

        Returns:
            Training losses
        """
        observations = []

        for _ in range(n_samples):
            if random_positions and hasattr(env, 'fovea'):
                # Random fovea position
                x = np.random.randint(0, env.width - env.fovea.width)
                y = np.random.randint(0, env.height - env.fovea.height)
                env.fovea.move_to(x, y)
                obs = env.get_fovea_observation(env.fovea)
            else:
                obs = env.get_observation()

            observations.append(obs.flatten())

        data = np.array(observations)

        # Normalize to [0, 1]
        data_min = data.min()
        data_max = data.max()
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)

        # Train
        if hasattr(self.autoencoder, 'train'):
            return self.autoencoder.train(data, epochs, noise_level)
        else:
            losses = []
            for epoch in range(epochs):
                epoch_loss = 0
                for sample in data:
                    loss = self.autoencoder.train_step(sample, noise_level)
                    epoch_loss += loss
                losses.append(epoch_loss / len(data))
            return losses
