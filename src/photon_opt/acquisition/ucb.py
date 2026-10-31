import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.models.model import Model


def make_ucb(
    model: Model, best_f: torch.Tensor, q: int = 1, beta: float = 4.0
) -> AcquisitionFunction:
    """Upper Confidence Bound: mean + sqrt(beta) * std. Larger beta explores more.

    best_f is unused (UCB does not need it) but kept so the signature matches
    AcquisitionFactory.
    """
    if q == 1:
        return UpperConfidenceBound(model=model, beta=beta)
    return qUpperConfidenceBound(model=model, beta=beta)