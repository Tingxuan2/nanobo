import torch
from torch.quasirandom import SobolEngine

from photon_opt.acquisition import make_acquisition
from photon_opt.candidate_samplers import optimize_acquisition
from photon_opt.space import ParameterSpace
from photon_opt.surrogates import GPSurrogate


class BOEngine:
    """Sequential Bayesian optimization with ask/tell. Physical units in and out."""

    def __init__(self, space: ParameterSpace, n_init: int = 8, seed: int = 0) -> None:
        self.space = space
        self.n_init = n_init
        self._sobol = SobolEngine(space.dim, scramble=True, seed=seed)
        self._surrogate = GPSurrogate()
        self._x_unit = torch.empty(0, space.dim, dtype=torch.float64)
        self._y = torch.empty(0, dtype=torch.float64)

    @property
    def n_observations(self) -> int:
        return self._y.shape[0]

    def ask(self) -> torch.Tensor:
        """Next design to simulate, physical units, shape (1, d)."""
        if self.n_observations < self.n_init:
            u = self._sobol.draw(1, dtype=torch.float64)
        else:
            self._surrogate.fit(self._x_unit, self._y)
            acqf = make_acquisition(self._surrogate.model, best_f=self._y.max())
            u = optimize_acquisition(acqf, self.space.dim)
        return self.space.from_unit(u)

    def tell(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Record simulated designs x (n, d), physical, with scores y (n,)."""
        u = self.space.to_unit(x.to(torch.float64))
        self._x_unit = torch.cat([self._x_unit, u])
        self._y = torch.cat([self._y, y.to(torch.float64)])

    @property
    def best(self) -> tuple[torch.Tensor, float]:
        """Best design so far (physical units) and its score."""
        if self.n_observations == 0:
            raise RuntimeError("No observations yet")
        i = int(self._y.argmax())
        return self.space.from_unit(self._x_unit[i]), self._y[i].item()