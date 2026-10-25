# photon_opt 🔬⚡

`photon_opt` is a modular Python framework for the inverse design of nanophotonic metasurfaces (such as gold nanodisks). It bridges electromagnetic Maxwell solvers (**Ansys Lumerical** & **COMSOL Multiphysics**) with frontier Bayesian Optimization (**BoTorch**) and live experiment tracking.

---

## 🏗️ Architecture & Component Overview

The package is structured around strict separation of concerns—decoupling mathematical optimization, physical solvers, data persistence, and seat management: