import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


class GPSurrogate:
    """Gaussian-process surrogate on normalized inputs. Scores: higher is better."""

    def __init__(self) -> None:
        self._model: SingleTaskGP | None = None

    @property
    def model(self) -> SingleTaskGP:
        """The fitted BoTorch model (needed later by the acquisition function)."""
        if self._model is None:
            raise RuntimeError("Call fit() before using the model")
        return self._model

    def fit(self, x_unit: torch.Tensor, y: torch.Tensor) -> "GPSurrogate":
        """Fit to inputs (n, d) in [0, 1] and scores (n,)."""
        x_unit = x_unit.to(torch.float64)
        y = y.to(torch.float64)
        if x_unit.ndim != 2 or y.shape != (x_unit.shape[0],):
            raise ValueError(
                f"Expected x (n, d) and y (n,), got {tuple(x_unit.shape)} and {tuple(y.shape)}"
            )
        model = SingleTaskGP(x_unit, y.unsqueeze(-1), outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.eval()
        self._model = model
        return self

    @torch.no_grad()
    def predict(self, x_unit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, std), each of shape (n,), for inputs (n, d) in [0, 1]."""
        posterior = self.model.posterior(x_unit.to(torch.float64))
        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.squeeze(-1).sqrt()
        return mean, std