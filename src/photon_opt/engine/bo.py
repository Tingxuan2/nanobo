import csv
import os
from pathlib import Path

import torch
from torch.quasirandom import SobolEngine

from photon_opt.acquisition import make_acquisition
from photon_opt.candidate_samplers import optimize_acquisition
from photon_opt.space import ParameterSpace
from photon_opt.surrogates import GPSurrogate
from photon_opt.telemetry import Telemetry


class BOEngine:
    """Batch Bayesian optimization with ask/tell. Physical units in and out.

    If checkpoint_path is given, every tell() saves all data to that CSV file,
    and an existing file is loaded on creation (resume).
    If telemetry is given, scores and GP diagnostics are logged to it.
    """

    def __init__(
        self,
        space: ParameterSpace,
        n_init: int = 8,
        seed: int = 0,
        checkpoint_path: str | Path | None = None,
        telemetry: Telemetry | None = None,
    ) -> None:
        self.space = space
        self.n_init = n_init
        self._sobol = SobolEngine(space.dim, scramble=True, seed=seed)
        self._surrogate = GPSurrogate()
        self._telemetry = telemetry if telemetry is not None else Telemetry()
        self._x_unit = torch.empty(0, space.dim, dtype=torch.float64)
        self._y = torch.empty(0, dtype=torch.float64)
        self._checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
        if self._checkpoint is not None and self._checkpoint.exists():
            self._load()

    @property
    def n_observations(self) -> int:
        return self._y.shape[0]

    def ask(self, q: int = 1) -> torch.Tensor:
        """Next q designs to simulate, physical units, shape (q, d)."""
        if q < 1:
            raise ValueError(f"q must be >= 1, got {q}")
        if self.n_observations < self.n_init:
            u = self._sobol.draw(q, dtype=torch.float64)
        else:
            self._surrogate.fit(self._x_unit, self._y)
            self._telemetry.log_surrogate(
                self.n_observations,
                self._surrogate.last_mll,
                self._surrogate.hyperparameters(self.space.names),
            )
            acqf = make_acquisition(self._surrogate.model, best_f=self._y.max(), q=q)
            u = optimize_acquisition(acqf, self.space.dim, q=q)
        return self.space.from_unit(u)

    def tell(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Record simulated designs x (n, d), physical, with scores y (n,)."""
        start = self.n_observations
        self._append(x, y)
        self._save()
        self._telemetry.log_round(
            start, self.space.names, x, y.to(torch.float64), self._y.max().item()
        )

    @property
    def best(self) -> tuple[torch.Tensor, float]:
        """Best design so far (physical units) and its score."""
        if self.n_observations == 0:
            raise RuntimeError("No observations yet")
        i = int(self._y.argmax())
        return self.space.from_unit(self._x_unit[i]), self._y[i].item()

    def _append(self, x: torch.Tensor, y: torch.Tensor) -> None:
        u = self.space.to_unit(x.to(torch.float64))
        self._x_unit = torch.cat([self._x_unit, u])
        self._y = torch.cat([self._y, y.to(torch.float64)])

    def _save(self) -> None:
        if self._checkpoint is None:
            return
        self._checkpoint.parent.mkdir(parents=True, exist_ok=True)
        x = self.space.from_unit(self._x_unit)
        tmp = self._checkpoint.with_suffix(self._checkpoint.suffix + ".tmp")
        with tmp.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([*self.space.names, "score"])
            for xi, yi in zip(x.tolist(), self._y.tolist()):
                writer.writerow([*xi, yi])
        os.replace(tmp, self._checkpoint)  # atomic: never leaves a half-written file

    def _load(self) -> None:
        assert self._checkpoint is not None
        with self._checkpoint.open(newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            if header != [*self.space.names, "score"]:
                raise ValueError(
                    f"Checkpoint columns {header} do not match space {self.space.names}"
                )
            rows = [[float(v) for v in row] for row in reader]
        if rows:
            data = torch.tensor(rows, dtype=torch.float64)
            self._append(data[:, :-1], data[:, -1])
            self._sobol.fast_forward(len(rows))  # do not reuse already-used Sobol points