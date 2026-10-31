import torch
from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf


def optimize_acquisition(
    acqf: AcquisitionFunction,
    dim: int,
    q: int = 1,
    num_restarts: int = 10,
    raw_samples: int = 256,
) -> torch.Tensor:
    """Find the best batch of q points in [0, 1]^dim. Returns shape (q, dim)."""
    bounds = torch.stack(
        [
            torch.zeros(dim, dtype=torch.float64),
            torch.ones(dim, dtype=torch.float64),
        ]
    )
    candidate, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        sequential=q > 1,  # pick the batch one point at a time (greedy)
    )
    return candidate.detach()