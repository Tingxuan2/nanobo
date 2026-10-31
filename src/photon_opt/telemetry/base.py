import torch


class Telemetry:
    """No-op telemetry. Subclasses override the methods they want to record."""

    def log_round(
        self,
        step: int,
        names: list[str],
        x: torch.Tensor,
        y: torch.Tensor,
        best_y: float,
    ) -> None:
        """Called after each tell(). step = index of the first new observation."""

    def log_surrogate(
        self, step: int, mll: float | None, hyperparams: dict[str, float]
    ) -> None:
        """Called after each GP fit. step = number of observations used."""

    def close(self) -> None:
        """Flush and release resources."""