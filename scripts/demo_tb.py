from photon_opt.drivers import SyntheticDriver
from photon_opt.engine import BOEngine, run
from photon_opt.space import Parameter, ParameterSpace
from photon_opt.telemetry.tensorboard import TensorBoardTelemetry

s = ParameterSpace((Parameter("radius", 50, 150), Parameter("period", 300, 600)))
d = SyntheticDriver(s)

telemetry = TensorBoardTelemetry("runs/tb/demo1")
engine = BOEngine(s, n_init=8, seed=0, telemetry=telemetry)
x_best, y_best = run(engine, d, n_evals=40, q=1)
telemetry.close()
print("best:", x_best, y_best)