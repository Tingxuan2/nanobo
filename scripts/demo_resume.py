import sys

from photon_opt.drivers import SyntheticDriver
from photon_opt.engine import BOEngine, run
from photon_opt.space import Parameter, ParameterSpace

n_evals = int(sys.argv[1])

s = ParameterSpace((Parameter("radius", 50, 150), Parameter("period", 300, 600)))
d = SyntheticDriver(s)

engine = BOEngine(s, n_init=6, seed=0, checkpoint_path="runs/demo.csv")
print("already stored:", engine.n_observations)
x_best, y_best = run(engine, d, n_evals=n_evals, q=1)
print("best:", x_best, y_best)