# Examples & Demos

The `demo/` folder contains several scripts demonstrating how to couple the `crossbridge` model to different physics scales:

## 1. Reproducing the Papers (`reproduce_figures.py`, `reproduce_figures_rdq20mf.py`, `reproduce_figures_land2017.py`, `reproduce_figures_lewalle2024.py`, `reproduce_figures_tewari2016.py`, `reproduce_figures_marzban2020.py`, `reproduce_figures_musgrave2025.py`)
Run each standalone model to recreate the original validation figures/results from its paper.
* **RDQ18 / RDQ20MF — Steady State:** Computes force-calcium relationships, length-dependent activation, and Hill curves.
* **RDQ18 / RDQ20MF — Dynamic Twitches:** Simulates twitches under varying calcium transients and fixed sarcomere lengths.
* **RDQ18 — Tension Redevelopment ($k_{tr}$):** Simulates sudden crossbridge detachment and subsequent exponential recovery.
* **Land2017:** Passive viscoelastic step response, the steady-state force-calcium relationship at three sarcomere lengths (its ad hoc `beta0`/`beta1` length-dependent activation), the biphasic quick-stretch response driven by the distortion-decay crossbridge model, and isometric twitches at different SL using the paper's "whole organ model" recalibration (see the script's docstring for what is/isn't a literal figure reproduction, since the paper's own whole-organ finite-element figure is out of scope for this package).
* **Tewari2016:** Twitches driven by the Janssen et al. (2002) calcium transients at 2-10 Hz and 1.9-2.2 um (the paper's Figs. 2, 3 and F.1, against its model and data), the steady-state force-calcium relation at five sarcomere lengths (Fig. E.1, Table E.1), metabolite and calcium effects on force development (Fig. G.1), tension redevelopment after a slack-restretch (Figs. 4 and H.1), and the reading of the paper's reference temperature this implementation depends on. The script's docstring lists what does and does not reproduce; the paper's heart and circulation figures are out of scope.
* **Marzban2020:** The paper's fit to the Janssen et al. data (its Fig. 3) and the steady-state pCa50 of its caption, force-dependent recruitment from the myosin OFF state and its share in length-dependent activation, twitches and crossbridge ATP turnover at the metabolite levels of sham and failing rat hearts, and a side-by-side comparison with Tewari2016.
* **Musgrave2025:** The human atrial models against their data: the passive model (Fig. 2), and the crossbridge models against the active complex modulus and steady-state stress under five levels of ATP and Pi (Fig. 4, Table 2), linearized from the model's `.ode` file. Then the muscle models driven by the paper's calcium transients (Fig. 3): isometric twitches per muscle and per myofilament area (Fig. 5), work loops at ten afterloads with their work, shortening, velocity, power, crossbridge energy and efficiency (Figs. 6 and 7), and twitches at high Pi, at low ATP and with calcium handling and crossbridges crossed over between the groups (Figs. 8A, 9A and 10A). The script's docstring lists the two places where it differs from the paper.
* **Lewalle2024:** Steady-state force-pCa curves at two sarcomere lengths, the length dependence of active tension (Frank-Starling), and isometric twitches at different SL — showing that myosin OFF-state feedback on total force alone reproduces length-dependent activation (see the script's docstring for what is/isn't a literal figure reproduction, since some of the paper's figures compare against an unimplemented baseline model).

## 2. Model Comparison (`compare_models.py`)
Drives all three models (`RDQ18`, `RDQ20MF`, `Lewalle2024`) through the same calcium transient
and sarcomere-length protocol via the `get_model()` registry, to demonstrate that a coupling loop
written against the shared `advance_step`/`get_active_tension` interface works unchanged when the
model class is swapped. Compares twitch kinetics (raw and peak-normalized) and length-dependent
activation (normalized to a common reference SL) across the three models.

## 3. Coupled 0D Electromechanics (`holzapfel_torord_*.py`)
These demos couple the `RDQ18` model to a cell-level electrophysiology model (ToRORd) and a macro-scale tissue mechanics model (`zero_mech` with Holzapfel-Ogden materials).
* **`holzapfel_torord_isometric.py`**: Simulates an isometric contraction where the macroscopic tissue length is clamped ($\lambda=1.0$) and the resulting internal fiber stress is measured over time.
* **`holzapfel_torord_isotonic.py`**: Simulates a fully unloaded free contraction where the active tension forces the tissue to shorten ($\lambda < 1.0$) against zero external stress.

## 4. Finite Element Coupling (`fem.py`)
Demonstrates how to couple the point-wise `RDQ18` model to a full 3D Finite Element Method (FEM) mesh using `dolfinx` and `pulse`. It evaluates the sarcomere model over the integration points of a unit cube.

## 5. Interactive Plotting (`interactive_plot.py`)
An interactive `matplotlib` script that runs a single cell simulation in real-time, displaying the transient states, calcium inputs, and fraction of permissive states visually. This produce a similar plot as the one provided in the Matlab code of the original paper.
