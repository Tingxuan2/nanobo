from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from photon_opt.telemetry.base import Telemetry


class TensorBoardTelemetry(Telemetry):
    """Writes scores, designs and GP diagnostics to a TensorBoard log directory."""

    def __init__(self, log_dir: str | Path) -> None:
        self._writer = SummaryWriter(log_dir=str(log_dir))

    def log_round(
        self,
        step: int,
        names: list[str],
        x: torch.Tensor,
        y: torch.Tensor,
        best_y: float,
    ) -> None:
        for i, (xi, yi) in enumerate(zip(x.tolist(), y.tolist())):
            idx = step + i
            self._writer.add_scalar("objective/score", yi, idx)
            for name, value in zip(names, xi):
                self._writer.add_scalar(f"design/{name}", value, idx)
        self._writer.add_scalar("objective/best_so_far", best_y, step + len(y) - 1)
        self._writer.flush()

    def log_surrogate(
        self, step: int, mll: float | None, hyperparams: dict[str, float]
    ) -> None:
        if mll is not None:
            self._writer.add_scalar("gp/mll", mll, step)
        for key, value in hyperparams.items():
            self._writer.add_scalar(f"gp/{key}", value, step)
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()