# crossbridge

**crossbridge** is a highly efficient, vectorized Python library for simulating cardiac myofilament activation and crossbridge dynamics.

## The Mathematical Models
`crossbridge` implements several reduced-order models of cardiac myofilament activation
([RDQ18](docs/models/rdq18.md), [RDQ20-MF](docs/models/rdq20mf.md), [Land2017](docs/models/land2017.md),
[Lewalle2024](docs/models/lewalle2024.md)), all sharing a common interface (see "Choosing a Model"
below) so that a coupled electromechanics simulation can swap between them with minimal code
changes. See [docs/models](docs/models/index.md) for a description and reference for each model.

## Installation

The package requires Python 3.11+. You can install the base package and its dependencies using `pip`.

To install the library from the source code:
```bash
git clone https://github.com/ComputationalPhysiology/crossbridge.git
cd crossbridge
pip install .
```

To install with optional dependencies (for running demos, tests, or building docs):
```bash
pip install ".[demos]"  # Installs scipy, matplotlib, gotranx, numba, zero-mech, etc.
pip install ".[test]"   # Installs pytest and coverage tools
pip install ".[docs]"   # Installs jupyter-book and sphinx plugins
pip install ".[all]"    # Installs everything
```

## Basic Usage
To most basic usage its to solve for a single cell. The `RDQ18` class provides an `advance_ODE` method that takes in the time step, calcium concentration, and sarcomere length to update the internal state of the model. This can then be used to compute an active tension based on the fraction of permissive crossbridges.

```python
import numpy as np
from crossbridge import RDQ18, calcium_trace, sl_trace

# Initialize a model with 1000 cells/integration points
num_cells = 1
dt = 0.01
dt_sarc = 2.5e-5
sarcomere = RDQ18(num_cells=num_cells, Ta_max=60.0, params={"dt": dt_sarc})

# Inputs
t = np.arange(0, 1, dt)  # Time array [s]
Ca = calcium_trace(t)  # Calcium transient [uM]
SL = sl_trace(t)  # Sarcomere length transient [um]


Ta = np.zeros(len(t))
# Advance the ODEs by one time step
for i, (Cai, SLi) in enumerate(zip(Ca, SL)):
    sarcomere.advance_ODE(dt, Cai, np.array([SLi]))

    # Compute the fraction of permissive crossbridges (proxy for active tension)
    permissivity = sarcomere.compute_permissivity()[0]
    active_tension = sarcomere.Ta_max * permissivity
    Ta[i] = active_tension

import matplotlib.pyplot as plt

fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
ax[0].plot(t, Ca)
ax[0].set_ylabel("Calcium [uM]")
ax[1].plot(t, SL)
ax[1].set_ylabel("Sarcomere Length [um]")
ax[2].plot(t, Ta)
ax[2].set_ylabel("Active Tension [kPa]")
ax[2].set_xlabel("Time [s]")
fig.tight_layout()
plt.show()
```
![Example Output](https://github.com/user-attachments/assets/4d07bea6-9f5d-4aae-a1df-2debe6199999)

## Choosing a Model

All models subclass the same `CardiacActivationModel` abstract base class and share one
constructor and stepping interface:

```python
ModelClass(num_cells, Ta_max=..., params={...})
model.default_parameters()      # classmethod: dict of physiological defaults
model.advance_step(dt, Ca_val, SL_vals, dSL_vals=None)   # integrate one time step
model.get_active_tension()      # -> np.ndarray, shape (num_cells,), kPa
model.reset()                   # restore the model's initial state
```

This means a coupled simulation loop written against `advance_step`/`get_active_tension` works
unchanged if the model class is swapped out. `RDQ18.advance_ODE` (used in the example above) is
that model's original, more detailed entry point; `advance_step` is the portable one to use when
you want to be able to swap models.

| Model         | State representation                          | `Ta_max` semantics                                             | Typical `dt`   |
|---------------|------------------------------------------------|------------------------------------------------------------------|---------------|
| `RDQ18`       | RU triplet joint-probability tensor            | Used directly: `Ta = Ta_max * compute_permissivity()`            | 2.5e-5 s      |
| `RDQ20MF`     | RU triplet tensor + explicit crossbridge states| **Unused** — tension is computed from `params["a_XB"]` instead   | 2.5e-5 s      |
| `Land2017`    | Troponin/crossbridge state populations         | **Unused** — tension is computed from `params["Tref"]` instead   | up to ~1e-3 s (adaptive internal sub-stepping) |
| `Lewalle2024` | Land2017 state populations + OFF-state feedback| **Unused** — tension is computed from `params["Tref"]` instead   | up to ~1e-3 s (adaptive internal sub-stepping) |

Only `RDQ18` scales tension via the constructor's `Ta_max` argument; `RDQ20MF`, `Land2017`, and
`Lewalle2024` compute tension intrinsically from their own parameter set (`a_XB`, `Tref`) and
accept `Ta_max` purely for interface compatibility. Check which case applies before relying on
`Ta_max` when swapping models.

Since all models share the same constructor signature, a model can be selected by name at
runtime via the small registry in `crossbridge`:

```python
from crossbridge import get_model

ModelClass = get_model("RDQ20MF")  # or "RDQ18", "Land2017", "Lewalle2024"
sarcomere = ModelClass(num_cells=100, params={"SL0": 2.0})
```

## Coupling to Electrophysiology and Mechanics
Most cellular and tissue-level simulations will require coupling the `RDQ18` model to electrophysiology and mechanics. The `advance_ODE` method is designed to be called at every time step of a larger simulation loop, allowing the sarcomere dynamics to evolve in response to changing calcium and length conditions.

The calcium concentration can be either a scalar (if all cells/integration points are assumed to have the same calcium transient) or an array with one entry per cell/integration point. The sarcomere length can similarly be a scalar or an array. Below is a pseudo-code example of how this coupling might look in a larger simulation loop:

```python
...

SL = np.full(num_cells, 2.2)  # Initial sarcomere length [um]
for t in time_steps:
    # Compute calcium from electrophysiology model
    Cai = compute_calcium(t)
    sarcomere.advance_ODE(dt, Cai, SL)
    # Compute active tension and update mechanics
    permissivity = sarcomere.compute_permissivity()[0]
    active_tension = sarcomere.Ta_max * permissivity
    # Update mechanics model with new active tension
    # and compute new sarcomere length (SL) based on
    # the mechanical response of the tissue
    SL = compute_new_length(active_tension, SL)
...
```
## Examples & Demos

The `demo/` folder contains several scripts demonstrating how to couple the `crossbridge` model to different physics scales:

### 1. Reproducing the Papers (`reproduce_figures.py`, `reproduce_figures_rdq20mf.py`, `reproduce_figures_land2017.py`, `reproduce_figures_lewalle2024.py`)
Run each standalone model to recreate the original validation figures/results from its paper.
* **RDQ18 / RDQ20MF — Steady State:** Computes force-calcium relationships, length-dependent activation, and Hill curves.
* **RDQ18 / RDQ20MF — Dynamic Twitches:** Simulates twitches under varying calcium transients and fixed sarcomere lengths.
* **RDQ18 — Tension Redevelopment ($k_{tr}$):** Simulates sudden crossbridge detachment and subsequent exponential recovery.
* **Land2017:** Passive viscoelastic step response, the steady-state force-calcium relationship at three sarcomere lengths (its ad hoc `beta0`/`beta1` length-dependent activation), the biphasic quick-stretch response driven by the distortion-decay crossbridge model, and isometric twitches at different SL using the paper's "whole organ model" recalibration (see the script's docstring for what is/isn't a literal figure reproduction, since the paper's own whole-organ finite-element figure is out of scope for this package).
* **Lewalle2024:** Steady-state force-pCa curves at two sarcomere lengths, the length dependence of active tension (Frank-Starling), and isometric twitches at different SL — showing that myosin OFF-state feedback on total force alone reproduces length-dependent activation (see the script's docstring for what is/isn't a literal figure reproduction, since some of the paper's figures compare against an unimplemented baseline model).

### 2. Model Comparison (`compare_models.py`)
Drives all three models (`RDQ18`, `RDQ20MF`, `Lewalle2024`) through the same calcium transient
and sarcomere-length protocol via the `get_model()` registry, to demonstrate that a coupling loop
written against the shared `advance_step`/`get_active_tension` interface works unchanged when the
model class is swapped. Compares twitch kinetics (raw and peak-normalized) and length-dependent
activation (normalized to a common reference SL) across the three models.

### 3. Coupled 0D Electromechanics (`holzapfel_torord_*.py`)
These demos couple the `RDQ18` model to a cell-level electrophysiology model (ToRORd) and a macro-scale tissue mechanics model (`zero_mech` with Holzapfel-Ogden materials).
* **`holzapfel_torord_isometric.py`**: Simulates an isometric contraction where the macroscopic tissue length is clamped ($\lambda=1.0$) and the resulting internal fiber stress is measured over time.
* **`holzapfel_torord_isotonic.py`**: Simulates a fully unloaded free contraction where the active tension forces the tissue to shorten ($\lambda < 1.0$) against zero external stress.

### 4. Finite Element Coupling (`fem.py`)
Demonstrates how to couple the point-wise `RDQ18` model to a full 3D Finite Element Method (FEM) mesh using `dolfinx` and `pulse`. It evaluates the sarcomere model over the integration points of a unit cube.

### 5. Interactive Plotting (`interactive_plot.py`)
An interactive `matplotlib` script that runs a single cell simulation in real-time, displaying the transient states, calcium inputs, and fraction of permissive states visually. This produce a similar plot as the one provided in the Matlab code of the original paper.

## Testing and Development
We use `pytest` for unit testing. To run the test suite and check code coverage:
```bash
pytest
```
To run the pre-commit linters (Ruff and MyPy):
```bash
pre-commit run --all-files
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
