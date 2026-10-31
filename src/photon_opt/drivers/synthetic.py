import torch

from photon_opt.drivers.base import Driver
from photon_opt.space import ParameterSpace


class SyntheticDriver(Driver):  # inherit from Driver Class
    """Cheap fake simulator: a resonance-like peak plus a mild ripple, optional noise."""

    def __init__(
        self,
        space: ParameterSpace,
        center: torch.Tensor | None = None,
        noise_std: float = 0.0,
        seed: int = 0,
    ) -> None:
        super().__init__(space)
        if center is None:
            center = torch.full((space.dim,), 0.3, dtype=torch.float64)
        self._center = center  # peak location in normalized [0, 1] coordinates
        self._noise_std = noise_std
        self._gen = torch.Generator().manual_seed(seed)

    def _evaluate(self, x: torch.Tensor) -> torch.Tensor:
        u = self.space.to_unit(x)
        r2 = ((u - self._center) ** 2).sum(dim=-1)
        y = 1.0 / (1.0 + 25.0 * r2) + 0.05 * torch.cos(10.0 * u.sum(dim=-1))
        if self._noise_std > 0:
            noise = torch.randn(y.shape, generator=self._gen, dtype=y.dtype)
            y = y + self._noise_std * noise
        return y