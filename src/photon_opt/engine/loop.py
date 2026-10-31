import torch

from photon_opt.drivers import Driver
from photon_opt.engine.bo import BOEngine


def run(
    engine: BOEngine,
    driver: Driver,
    n_evals: int,
    q: int | None = None,
) -> tuple[torch.Tensor, float]:
    """Run rounds of ask(q) -> evaluate -> tell until n_evals simulations are done.

    q defaults to driver.max_parallel; the last round is shortened to hit n_evals.
    """
    q = driver.max_parallel if q is None else q
    while engine.n_observations < n_evals:
        q_now = min(q, n_evals - engine.n_observations)
        x = engine.ask(q_now)
        y = driver.evaluate(x)
        engine.tell(x, y)
    return engine.best