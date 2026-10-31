from collections.abc import Callable

import torch
from botorch.acquisition import AcquisitionFunction

# (acquisition function, dim, batch size q) -> best q points in [0, 1]^dim, shape (q, dim)
CandidateOptimizer = Callable[[AcquisitionFunction, int, int], torch.Tensor]