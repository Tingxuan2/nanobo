import torch

from photon_opt.drivers import SyntheticDriver
from photon_opt.engine import BOEngine, run
from photon_opt.space import Parameter, ParameterSpace

s = ParameterSpace((Parameter("radius", 50, 150), Parameter("period", 300, 600)))
d = SyntheticDriver(s)

engine = BOEngine(s, n_init=6, seed=0)
x_best, y_best = run(engine, d, n_evals=30)
print("BO best:", x_best, y_best)

torch.manual_seed(0)
u = torch.rand(30, 2, dtype=torch.float64)
print("random best:", d.evaluate(s.from_unit(u)).max().item())