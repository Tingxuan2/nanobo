from abc import ABC, abstractmethod

import torch

from photon_opt.space import ParameterSpace


class Driver(ABC):
    """Contract: physical parameters (n, d) in, scores (n,) out. Higher is better."""
    max_parallel: int = 1  # how many designs this driver can simulate at the same time
    

    def __init__(self, space: ParameterSpace) -> None:
        self._space = space

    @property
    def space(self) -> ParameterSpace:
        return self._space

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Validate the input, run the simulation, validate the output."""
        if x.ndim != 2 or x.shape[-1] != self._space.dim:
            raise ValueError(
                f"Expected x of shape (n, {self._space.dim}), got {tuple(x.shape)}"
            )
        y = self._evaluate(x)
        if y.shape != (x.shape[0],):
            raise RuntimeError(
                f"Driver returned shape {tuple(y.shape)}, expected ({x.shape[0]},)"
            )
        return y

    @abstractmethod
    def _evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Subclasses implement the actual simulation here."""