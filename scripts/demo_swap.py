from functools import partial

from photon_opt.acquisition import make_ucb
from photon_opt.drivers import SyntheticDriver
from photon_opt.engine import BOEngine, run
from photon_opt.space import Parameter, ParameterSpace

s = ParameterSpace((Parameter("radius", 50, 150), Parameter("period", 300, 600)))
d = SyntheticDriver(s)

variants = {
    "EI (default)": BOEngine(s, n_init=8, seed=0),
    "UCB beta=4": BOEngine(s, n_init=8, seed=0, acquisition=partial(make_ucb, beta=4.0)),
    "UCB beta=0.5": BOEngine(s, n_init=8, seed=0, acquisition=partial(make_ucb, beta=0.5)),
}
for name, engine in variants.items():
    x_best, y_best = run(engine, d, n_evals=30, q=1)
    print(f"{name:14s} best score = {y_best:.4f}   design = {x_best.tolist()}")