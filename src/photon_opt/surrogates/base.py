from typing import Protocol

import torch
from botorch.models.model import Model


class Surrogate(Protocol):
    """What the engine needs from a surrogate model (inputs in [0, 1]^d)."""

    last_mll: float | None

    @property
    def model(self) -> Model:
        """The fitted model, handed to the acquisition function."""
        ...

    def fit(self, x_unit: torch.Tensor, y: torch.Tensor) -> "Surrogate":
        """Fit to inputs (n, d) and scores (n,)."""
        ...

    def hyperparameters(self, names: list[str]) -> dict[str, float]:
        """Numbers worth logging (return {} if there are none)."""
        ...