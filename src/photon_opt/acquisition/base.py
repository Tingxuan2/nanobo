from collections.abc import Callable

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.models.model import Model

# (model, best score so far, batch size q) -> acquisition function
AcquisitionFactory = Callable[[Model, torch.Tensor, int], AcquisitionFunction]