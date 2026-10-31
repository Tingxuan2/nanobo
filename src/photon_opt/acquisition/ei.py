import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.models.model import Model


def make_acquisition(model: Model, best_f: float | torch.Tensor) -> LogExpectedImprovement:
    """Log Expected Improvement over the best raw score seen so far."""
    return LogExpectedImprovement(model=model, best_f=best_f)