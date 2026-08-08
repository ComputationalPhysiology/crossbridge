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

To install the library you can use pip:
```bash
python3 -m pip install crossbridge
```

To install with optional dependencies (for running demos, tests, or building docs):
```bash
python3 -m pip install "crossbridge[demos]"  # Installs scipy, matplotlib, gotranx, numba, zero-mech, etc.
python3 -m pip install "crossbridge[test]"   # Installs pytest and coverage tools
python3 -m pip install "crossbridge[docs]"   # Installs jupyter-book and sphinx plugins
python3 -m pip install "crossbridge[all]"    # Installs everything
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

The `demo/` folder contains several scripts demonstrating how to couple the `crossbridge` model to different physics scales. See [demos](demo/index.md) for a description of each demo and how to run them.


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
