import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.model import Model


def make_acquisition(
    model: Model, best_f: float | torch.Tensor, q: int = 1
) -> AcquisitionFunction:
    """Log Expected Improvement; the Monte-Carlo batch version when q > 1."""
    if q == 1:
        return LogExpectedImprovement(model=model, best_f=best_f)
    return qLogExpectedImprovement(model=model, best_f=best_f)