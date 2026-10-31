import torch

from photon_opt.drivers import Driver
from photon_opt.engine.bo import BOEngine


def run(engine: BOEngine, driver: Driver, n_evals: int) -> tuple[torch.Tensor, float]:
    """Run ask -> evaluate -> tell until n_evals simulations are done."""
    while engine.n_observations < n_evals:
        x = engine.ask()
        y = driver.evaluate(x)
        engine.tell(x, y)
    return engine.best