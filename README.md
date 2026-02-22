# crossbridge

**crossbridge** is a highly efficient, vectorized Python library for simulating cardiac myofilament activation and crossbridge dynamics.

## The Mathematical Model
This library implements the reduced-order Ordinary Differential Equation (ODE) model of sarcomere dynamics proposed by Regazzoni, Dedè, and Quarteroni (2018).

Spatially explicit Markov Chain models of the sarcomere accurately capture length-dependent activation and nearest-neighbor cooperative interactions (such as attached crossbridges increasing the affinity of troponin C to calcium). However, these full models involve an intractable number of degrees of freedom (on the order of $10^{21}$) and require slow Monte Carlo simulations.

The RDQ18 model overcomes this by using a physically motivated assumption of conditional independence to track joint probabilities of triplets of consecutive units. This condenses the system to roughly 2,200 variables, resulting in a system of ODEs that solves 10,000 times faster than the original Monte Carlo method while maintaining high accuracy.

**Reference:**
> Regazzoni, F., Dedè, L., & Quarteroni, A. (2018). *Active contraction of cardiac cells: a reduced model for sarcomere dynamics with cooperative interactions.* Biomechanics and Modeling in Mechanobiology, 17(6), 1663-1686.

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

### 1. Reproducing the Paper (`reproduce_figures.py`)
Runs the standalone `RDQ18` model to recreate the original validation figures from the 2018 paper.
* **Steady State:** Computes force-calcium relationships, length-dependent activation, and Hill curves.
* **Dynamic Twitches:** Simulates twitches under varying calcium transients and fixed sarcomere lengths.
* **Tension Redevelopment ($k_{tr}$):** Simulates sudden crossbridge detachment and subsequent exponential recovery.

### 2. Coupled 0D Electromechanics (`holzapfel_torord_*.py`)
These demos couple the `RDQ18` model to a cell-level electrophysiology model (ToRORd) and a macro-scale tissue mechanics model (`zero_mech` with Holzapfel-Ogden materials).
* **`holzapfel_torord_isometric.py`**: Simulates an isometric contraction where the macroscopic tissue length is clamped ($\lambda=1.0$) and the resulting internal fiber stress is measured over time.
* **`holzapfel_torord_isotonic.py`**: Simulates a fully unloaded free contraction where the active tension forces the tissue to shorten ($\lambda < 1.0$) against zero external stress.

### 3. Finite Element Coupling (`fem.py`)
Demonstrates how to couple the point-wise `RDQ18` model to a full 3D Finite Element Method (FEM) mesh using `dolfinx` and `pulse`. It evaluates the sarcomere model over the integration points of a unit cube.

### 4. Interactive Plotting (`plot.py`)
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
