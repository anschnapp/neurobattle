"""Neural network brain for robots and turrets."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class Brain:
    """Small feedforward neural network driven by NumPy.

    Architecture: input -> hidden (tanh) -> output (tanh)
    Weights are plain arrays — this IS the brain.
    """

    input_size: int
    hidden_size: int
    output_size: int
    weights1: np.ndarray = field(repr=False, default=None)
    bias1: np.ndarray = field(repr=False, default=None)
    weights2: np.ndarray = field(repr=False, default=None)
    bias2: np.ndarray = field(repr=False, default=None)

    def __post_init__(self):
        if self.weights1 is None:
            self.randomize()

    def randomize(self):
        """Initialize with small random weights."""
        scale = 1.0 / np.sqrt(self.input_size)
        self.weights1 = np.random.randn(self.input_size, self.hidden_size).astype(np.float32) * scale
        self.bias1 = np.zeros(self.hidden_size, dtype=np.float32)
        scale2 = 1.0 / np.sqrt(self.hidden_size)
        self.weights2 = np.random.randn(self.hidden_size, self.output_size).astype(np.float32) * scale2
        self.bias2 = np.zeros(self.output_size, dtype=np.float32)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run a forward pass. inputs shape: (input_size,) -> outputs shape: (output_size,)"""
        hidden = np.tanh(inputs @ self.weights1 + self.bias1)
        outputs = np.tanh(hidden @ self.weights2 + self.bias2)
        return outputs

    def copy(self) -> Brain:
        """Deep copy of this brain."""
        return Brain(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            weights1=self.weights1.copy(),
            bias1=self.bias1.copy(),
            weights2=self.weights2.copy(),
            bias2=self.bias2.copy(),
        )

    def get_flat_weights(self) -> np.ndarray:
        """Flatten all parameters into a single 1D array."""
        return np.concatenate([
            self.weights1.ravel(),
            self.bias1.ravel(),
            self.weights2.ravel(),
            self.bias2.ravel(),
        ])

    def set_flat_weights(self, flat: np.ndarray):
        """Load parameters from a flat 1D array."""
        idx = 0
        size = self.input_size * self.hidden_size
        self.weights1 = flat[idx:idx + size].reshape(self.input_size, self.hidden_size).copy()
        idx += size
        size = self.hidden_size
        self.bias1 = flat[idx:idx + size].copy()
        idx += size
        size = self.hidden_size * self.output_size
        self.weights2 = flat[idx:idx + size].reshape(self.hidden_size, self.output_size).copy()
        idx += size
        size = self.output_size
        self.bias2 = flat[idx:idx + size].copy()

    def param_count(self) -> int:
        return (self.input_size * self.hidden_size + self.hidden_size +
                self.hidden_size * self.output_size + self.output_size)

    def save(self, path: str):
        np.savez(path,
                 input_size=self.input_size,
                 hidden_size=self.hidden_size,
                 output_size=self.output_size,
                 weights1=self.weights1,
                 bias1=self.bias1,
                 weights2=self.weights2,
                 bias2=self.bias2)

    @staticmethod
    def load(path: str) -> Brain:
        data = np.load(path)
        return Brain(
            input_size=int(data['input_size']),
            hidden_size=int(data['hidden_size']),
            output_size=int(data['output_size']),
            weights1=data['weights1'],
            bias1=data['bias1'],
            weights2=data['weights2'],
            bias2=data['bias2'],
        )
