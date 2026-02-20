# # Coupling a Neo-Hookean material model to ToRORd_dynCl_endo

from pathlib import Path
from tqdm import tqdm
import urllib.request
import gotranx.cli.cellml2ode
import gotranx.cli.gotran2py

# from scipy.integrate import solve_ivp
from scipy.optimize import root
import numpy as np
import numba
import matplotlib.pyplot as plt
import sympy  # noqa: F401
import zero_mech

from crossbridge import RDQ18


experiment = zero_mech.experiments.uniaxial_tension()
mat = zero_mech.material.HolzapfelOgden()
comp = zero_mech.compressibility.Incompressible()
act = zero_mech.active.ActiveStress()
mech_model = zero_mech.Model(material=mat, compressibility=comp, active=act)

# First Piola-Kirchhoff stress
P = mech_model.first_piola_kirchhoff(experiment.F)


# Generate code and save it to a file if it does not exist
module_path = Path("ToRORd_dynCl_endo.py")
if not module_path.is_file():
    print("Generating ToRORd_dynCl_endo.py from CellML...")
    url = "https://raw.githubusercontent.com/jtmff/torord/refs/heads/master/cellml/ToRORd_dynCl_endo.cellml"
    cellml_path = module_path.with_suffix(".cellml")
    urllib.request.urlretrieve(url, cellml_path)
    ode_path = module_path.with_suffix(".ode")
    gotranx.cli.cellml2ode.main(cellml_path, outname=ode_path)
    gotranx.cli.gotran2py.main(
        fname=ode_path,
        scheme=[gotranx.schemes.Scheme.generalized_rush_larsen],
        outname=module_path,
    )


import ToRORd_dynCl_endo as ep_model


# Set time step to 0.1 ms
dt = 0.05
# Simulate model for 1000 ms
BCL = 400
num_beats = 3
t = np.arange(0, num_beats * BCL, dt)
t_one_beat = np.arange(0, BCL, dt)
assert len(t) == len(t_one_beat) * num_beats, "Time array length mismatch"

y = ep_model.init_state_values()
# Get initial parameter values
params = ep_model.init_parameter_values(i_Stim_Period=BCL)

# Get the index of the membrane potential
V_index = ep_model.state_index("v")
Ca_index = ep_model.state_index("cai")
# Get the index of the active tension from the land model

Istim_index = ep_model.monitor_index("Istim")
fgr = numba.njit(ep_model.generalized_rush_larsen)
# rhs = ep_model.rhs
mon = ep_model.monitor_values

# First run EP model to steady state
print("Running EP model to steady state...")
num_beats_pre = 100
Vs = np.zeros(len(t_one_beat) * num_beats_pre)
Cas = np.zeros(len(t_one_beat) * num_beats_pre)

i = 0
for beat_nr in tqdm(range(num_beats_pre)):
    for ti in t_one_beat:
        y = fgr(y, ti, dt, params)
        Vs[i] = y[V_index]
        Cas[i] = y[Ca_index]
        i += 1


fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(Vs)
ax[0].set_ylabel("V (mV)")
ax[1].plot(Cas)
ax[1].set_ylabel("Ca (mM)")
ax[1].set_xlabel("Time (ms)")
fig.tight_layout()
fig.savefig("ep_steady_state.png")


def func(x, Ta):
    lmbda, p = x

    replace = {
        mech_model["p"]: p,
        mech_model["Ta"]: Ta,
        experiment["λ"]: lmbda,
        **mat.default_parameters(),
    }

    P11 = P[0, 0].xreplace(replace)
    P22 = P[1, 1].xreplace(replace)
    return np.array([P11, P22], dtype=np.float64)


lmbda_value = 1.0
prev_lmbda = lmbda_value

p_value = 0.0

dt_mech = dt * 1e-3  # Convert ms to seconds for sarcomere model
dt_sarc = 2.5e-5

sarcomere_parms = RDQ18.default_parameters()
sarcomere_parms["dt"] = dt_sarc
sarcomere_parms["T"] = BCL * 1e-3  # Total simulation time in seconds
sl0 = sarcomere_parms["l0"]
sarcomere = RDQ18(num_cells=1, Ta_max=150.0, params=sarcomere_parms)

# Let us simulate the model

V = np.zeros(len(t))
Ca = np.zeros(len(t))
Ta = np.zeros(len(t))
Istim = np.zeros(len(t))
lmbdas = np.zeros(len(t))
dLambdas = np.zeros(len(t))
ps = np.zeros(len(t))

i = 0
for beat_nr in tqdm(range(num_beats), desc="Beat number"):
    for ti in tqdm(t_one_beat, total=len(t_one_beat), desc="Time within beat", leave=False):
        y = fgr(y, ti, dt, params)
        V[i] = y[V_index]
        Ca[i] = y[Ca_index]

        sarcomere.advance_ODE(dt_mech, Ca[i] * 1e3, np.array([sl0 * lmbda_value]))

        permissivity = sarcomere.compute_permissivity()[0]
        Ta[i] = sarcomere.Ta_max * permissivity
        res = root(
            func,
            np.array([lmbda_value, p_value]),
            args=(Ta[i],),
            method="hybr",
        )
        lmbda_value, p_value = res.x
        lmbdas[i] = lmbda_value
        ps[i] = p_value

        dLambda = (lmbda_value - prev_lmbda) / dt
        dLambdas[i] = dLambda

        prev_lmbda = lmbda_value
        # Istim[i] = monitor[Istim_index]
        i += 1


# And plot the results
fig, ax = plt.subplots(2, 3, sharex=True)
ax[0, 0].plot(t, V)
ax[1, 0].plot(t, Ta)
ax[0, 1].plot(t, Ca)
ax[1, 1].plot(t, dLambdas)
ax[0, 2].plot(t, lmbdas)
ax[1, 2].plot(t, ps)
ax[1, 0].set_xlabel("Time (ms)")
ax[1, 1].set_xlabel("Time (ms)")
ax[0, 0].set_ylabel("V (mV)")
ax[1, 0].set_ylabel("Ta (kPa)")
ax[0, 1].set_ylabel("Ca (mM)")
ax[1, 1].set_ylabel("dLambda")
ax[0, 2].set_ylabel("Lambda")
ax[1, 2].set_ylabel("p")
for axi in ax.flatten():
    axi.grid()
fig.tight_layout()
fig.savefig("neohookean_torord.png")
