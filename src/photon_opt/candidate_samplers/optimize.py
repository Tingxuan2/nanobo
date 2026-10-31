import torch
from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf


def optimize_acquisition(
    acqf: AcquisitionFunction,
    dim: int,
    num_restarts: int = 10,
    raw_samples: int = 256,
) -> torch.Tensor:
    """Find the best point in [0, 1]^dim. Returns shape (1, dim)."""
    bounds = torch.stack(
        [
            torch.zeros(dim, dtype=torch.float64),
            torch.ones(dim, dtype=torch.float64),
        ]
    )
    candidate, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    return candidate.detach()