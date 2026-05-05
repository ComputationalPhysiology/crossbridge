# %% [markdown]
# # Multiscale Cardiac Modeling: Zenker + ToR-ORd + RDQ18 + Regazzoni2020
#
# This script represents a fully coupled Tri-Scale Digital Twin:
# 1. 0D: Zenker Baroreflex (computes shock compensation factors)
# 2. 0D: Regazzoni2020 Closed-Loop Circulation (enforces mass conservation)
# 3. 1D: ToR-ORd Electrophysiology (driven by compensated Heart Rate)
# 4. 3D: RDQ18 Sarcomere Mechanics (driven by compensated Contractility)

# %%
import logging
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from tqdm import tqdm
from pathlib import Path
import urllib.request
import gotranx.cli.cellml2ode
import gotranx.cli.gotran2py
import numba

import zero_mech
from crossbridge import RDQ18
from circulation.zenker import Zenker
from circulation.regazzoni2020 import Regazzoni2020

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
# Suppress the verbose circulation base logging to keep output clean
logging.getLogger("circulation.base").setLevel(logging.WARNING)

# %% [markdown]
# ## 1. ZENKER PRE-PROCESSING (Baroreflex Response)

# %%
logger.info("--- Running Zenker Model Baseline ---")
zenker_normal = Zenker()
zenker_normal.solve(T=100.0, dt=1e-3, dt_eval=0.1)
hz_norm = zenker_normal.history

HR_normal = hz_norm["fHR"][-1]
R_TPR_normal = hz_norm["R_TPR"][-1]
C_PRSW_normal = hz_norm["C_PRSW"][-1]

logger.info("--- Running Zenker Model Hemorrhage ---")
# Simulate a 1000 mL blood loss (100 mL/s for 10 seconds)
# Note: We omit Vv0 overrides so the structural anatomy remains constant.
bleed_params = {
    "start_withdrawal": 10.0,
    "end_withdrawal": 20.0,
    "flow_withdrawal": -100.0,
    "flow_infusion": 0.0,
}
zenker_bleed = Zenker(parameters=bleed_params)
zenker_bleed.solve(T=200.0, dt=1e-3, dt_eval=0.1, initial_state=zenker_normal.state)
hz_bleed = zenker_bleed.history

HR_bleed = hz_bleed["fHR"][-1]
R_TPR_bleed = hz_bleed["R_TPR"][-1]
C_PRSW_bleed = hz_bleed["C_PRSW"][-1]

# Extract the scaling factors computed by the simulated nervous system
HR_factor = HR_bleed / HR_normal
R_TPR_factor = R_TPR_bleed / R_TPR_normal
C_PRSW_factor = C_PRSW_bleed / C_PRSW_normal

logger.info(
    f"Baroreflex Response -> HR: {HR_factor:.2f}x, R_TPR: {R_TPR_factor:.2f}x, Contractility: {C_PRSW_factor:.2f}x"
)

# Convert Zenker output to Hz for the Regazzoni baseline
HR_normal_hz = 1.25
HR_bleed_hz = 1.25 * HR_factor

# %% [markdown]
# ## 2. Electrophysiology Model Setup (ToR-ORd)

# %%
module_path = Path("ToRORd_dynCl_endo.py")
if not module_path.is_file():
    logger.info("Generating ToRORd_dynCl_endo.py from CellML...")
    url = "https://raw.githubusercontent.com/jtmff/torord/refs/heads/master/cellml/ToRORd_dynCl_endo.cellml"
    cellml_path = module_path.with_suffix(".cellml")
    urllib.request.urlretrieve(url, cellml_path)
    ode_path = module_path.with_suffix(".ode")
    gotranx.cli.cellml2ode.main(cellml_path, outname=ode_path)
    gotranx.cli.gotran2py.main(
        fname=ode_path, scheme=[gotranx.schemes.Scheme.generalized_rush_larsen], outname=module_path
    )

import ToRORd_dynCl_endo as ep_model

fgr = numba.njit(ep_model.generalized_rush_larsen)
V_index = ep_model.state_index("v")
Ca_index = ep_model.state_index("cai")


def prepace_ep(HR_hz):
    BCL_MS = 1000.0 / HR_hz
    DT_EP_MS = 0.05
    NUM_BEATS = 100

    y_init = ep_model.init_state_values()
    params_ep = ep_model.init_parameter_values(i_Stim_Period=BCL_MS)
    stim_idx = ep_model.parameter_index("i_Stim_Amplitude")
    params_ep[stim_idx] = -150.0

    t_one_beat = np.arange(0, BCL_MS, DT_EP_MS)
    ca_last_beat = []

    logger.info(f"Pre-pacing ToR-ORd at {HR_hz * 60:.0f} BPM for {NUM_BEATS} beats...")
    for beat in tqdm(range(NUM_BEATS)):
        for ti in t_one_beat:
            t_global = beat * BCL_MS + ti
            y_init = fgr(y_init, t_global, DT_EP_MS, params_ep)
            if beat == NUM_BEATS - 1:
                ca_last_beat.append(y_init[Ca_index] * 1000.0)

    ca_min, ca_max = np.min(ca_last_beat), np.max(ca_last_beat)
    logger.info(f"Ca range for {HR_hz * 60:.0f} BPM: min={ca_min:.3f}, max={ca_max:.3f}")
    return y_init, params_ep, ca_min, ca_max


y_steady_norm, p_ep_norm, ca_min_norm, ca_max_norm = prepace_ep(HR_normal_hz)
y_steady_bleed, p_ep_bleed, ca_min_bleed, ca_max_bleed = prepace_ep(HR_bleed_hz)

# %% [markdown]
# ## 3. Mechanics & Geometric Scaling

# %%
WALL_THICKNESS_CM = 1.0
CHAMBER_RADIUS_CM = 3.0
KPA_TO_MMHG = 7.5
GEO_SCALE_FACTOR = ((2.0 * WALL_THICKNESS_CM) / CHAMBER_RADIUS_CM) * KPA_TO_MMHG

UNLOADED_VOLUME_ML = 120.0
RESTING_SARCOMERE_LENGTH_UM = 2.1

experiment = zero_mech.experiments.uniaxial_tension()
mech_model = zero_mech.Model(
    material=zero_mech.material.HolzapfelOgden(),
    compressibility=zero_mech.compressibility.Incompressible(),
    active=zero_mech.active.ActiveStress(),
)

P_ho = mech_model.first_piola_kirchhoff(experiment.F)
mat_params = zero_mech.material.HolzapfelOgden().default_parameters()
P11_sym, P22_sym = P_ho[0, 0].xreplace(mat_params), P_ho[1, 1].xreplace(mat_params)

lam_sym, Ta_sym, p_sym = experiment["λ"], mech_model["Ta"], mech_model["p"]
P11_fast = sp.lambdify((lam_sym, Ta_sym, p_sym), P11_sym, "numpy")
P22_fast = sp.lambdify((lam_sym, Ta_sym, p_sym), P22_sym, "numpy")


def solve_fiber_stress(lmbda, Ta, p_guess):
    try:
        res = root_scalar(
            lambda p_val: P22_fast(lmbda, Ta, p_val),
            x0=p_guess,
            x1=p_guess + 0.05,
            method="secant",
            maxiter=50,
        )
        return P11_fast(lmbda, Ta, res.root), res.root
    except Exception:
        return np.nan, p_guess


# %% [markdown]
# ## 4. The Multiscale Regazzoni Wrapper


# %%
def run_multiscale_closed_loop(
    name, HR_hz, R_TPR_fact, C_PRSW_fact, volume_offset_ml, ep_state_init, ep_params, ca_min, ca_max
):
    logger.info(f"--- Setting up {name} Closed-Loop System ---")

    # 1. Base Regazzoni Parameters
    base_model = Regazzoni2020(add_units=False)
    params = base_model.parameters.copy()
    init_state = base_model._initial_state.copy()

    params["HR"] = HR_hz

    # Apply Baroreflex Systemic Resistance
    params["circulation"]["SYS"]["R_AR"] *= R_TPR_fact
    params["circulation"]["SYS"]["R_VEN"] *= R_TPR_fact

    # Apply Baroreflex Contractility to unmodeled chambers (LA, RA, RV)
    for chamber in ["LA", "RA", "RV"]:
        params["chambers"][chamber]["EA"] *= C_PRSW_fact

    # Apply Blood Loss to the Venous Reservoir
    C_ven = params["circulation"]["SYS"]["C_VEN"]
    init_state["p_VEN_SYS"] += volume_offset_ml / C_ven

    # 2. Encapsulate ToR-ORd and RDQ18 States
    ep_state = ep_state_init.copy()
    sarcomere = RDQ18(num_cells=1, Ta_max=250.0 * C_PRSW_fact, params=RDQ18.default_parameters())
    mech_state = {"Ta": 0.0, "p_val": 0.0, "P_LV": 0.0}

    # Setup diagnostic logging interval (~4 times per beat)
    log_step = int((1.0 / HR_hz / 1e-3) / 4)

    # 3. Create Callbacks injected into Regazzoni2020
    def callback(model, step, t, **kwargs):
        # Regazzoni dt is 1e-3. ToR-ORd is stiff and needs 0.05ms steps.
        dt_s = 1e-3
        sub_steps = 20
        dt_ep = (dt_s * 1000.0) / sub_steps
        t_start_ms = (t - dt_s) * 1000.0

        for k in range(sub_steps):
            t_sub = t_start_ms + k * dt_ep
            ep_state[:] = fgr(ep_state, t_sub, dt_ep, ep_params)

        Ca_uM = ep_state[Ca_index] * 1000.0

        # Read the current LV Volume tracked by the closed loop
        idx_lv = model.states_names.index("V_LV")
        V_LV = model.state[idx_lv]

        safe_volume = max(5.0, V_LV)
        lmbda = (safe_volume / UNLOADED_VOLUME_ML) ** (1.0 / 3.0)

        # Map dynamic EP calcium to the mechanics model
        normalized_ca = np.clip((Ca_uM - ca_min) / (ca_max - ca_min), 0.0, 1.0)
        Ca_input = 0.08 + (normalized_ca**2) * (0.90 - 0.08)

        sarcomere.advance_ODE(
            dt_s, np.array([Ca_input]), np.array([RESTING_SARCOMERE_LENGTH_UM * lmbda])
        )
        mech_state["Ta"] = sarcomere.Ta_max * sarcomere.compute_permissivity()[0]

        # DIAGNOSTIC LOGGING
        if step < 5 or step % log_step == 0:
            idx_ao = model.states_names.index("p_AR_SYS")
            P_ao = model.state[idx_ao]
            P_LA = model.var[0]  # p_LA is the first variable in the var array
            logger.info(
                f"t={t:.2f}s | Vm={ep_state[V_index]:.1f} | Ca={Ca_uM:.2f} | Ta={mech_state['Ta']:.1f} | V_LV={V_LV:.1f} | P_LV={mech_state['P_LV']:.1f} | P_ao={P_ao:.1f} | P_LA={P_LA:.1f}"
            )

    def p_LV(V_LV, t):
        # Called by Regazzoni to get the Left Ventricle pressure at the current volume
        safe_volume = max(5.0, V_LV)
        lmbda = (safe_volume / UNLOADED_VOLUME_ML) ** (1.0 / 3.0)
        P11, mech_state["p_val"] = solve_fiber_stress(lmbda, mech_state["Ta"], mech_state["p_val"])

        # Cache the current pressure for the diagnostic logger
        mech_state["P_LV"] = max(0.0, P11 * GEO_SCALE_FACTOR)
        return mech_state["P_LV"]

    # 4. Initialize and Solve
    model = Regazzoni2020(
        parameters=params,
        initial_state=init_state,
        add_units=False,
        p_LV=p_LV,
        callback=callback,
        verbose=False,
    )

    logger.info(f"Solving {name} Closed-Loop Model for 20 beats...")
    history = model.solve(num_beats=20, dt=1e-3, dt_eval=1e-3)
    return history


# %% [markdown]
# ## 5. Execution and Visualization

# %%
if __name__ == "__main__":
    # 1. Simulate the Normal Closed-Loop System
    hist_norm = run_multiscale_closed_loop(
        name="Normal",
        HR_hz=HR_normal_hz,
        R_TPR_fact=1.0,
        C_PRSW_fact=1.0,
        volume_offset_ml=0.0,
        ep_state_init=y_steady_norm,
        ep_params=p_ep_norm,
        ca_min=ca_min_norm,
        ca_max=ca_max_norm,
    )

    # 2. Simulate Hemorrhagic Shock (1000 mL lost)
    hist_bleed = run_multiscale_closed_loop(
        name="Hemorrhage",
        HR_hz=HR_bleed_hz,
        R_TPR_fact=R_TPR_factor,
        C_PRSW_fact=C_PRSW_factor,
        volume_offset_ml=-1000.0,
        ep_state_init=y_steady_bleed,
        ep_params=p_ep_bleed,
        ca_min=ca_min_bleed,
        ca_max=ca_max_bleed,
    )

    # 3. Plot the final physiological response
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    # Extract only the final stable beat
    samples_norm = int(1.0 / HR_normal_hz / 1e-3)
    slc_norm = slice(-samples_norm, None)

    ax.plot(hist_norm["V_LV"], hist_norm["p_LV"], color="blue", alpha=0.2)
    ax.plot(
        hist_norm["V_LV"][slc_norm],
        hist_norm["p_LV"][slc_norm],
        color="blue",
        lw=2.5,
        label=f"Normal (HR: {HR_normal_hz * 60:.0f} BPM)",
    )

    samples_bleed = int(1.0 / HR_bleed_hz / 1e-3)
    slc_bleed = slice(-samples_bleed, None)

    ax.plot(hist_bleed["V_LV"], hist_bleed["p_LV"], color="red", alpha=0.2)
    ax.plot(
        hist_bleed["V_LV"][slc_bleed],
        hist_bleed["p_LV"][slc_bleed],
        color="red",
        lw=2.5,
        label=f"Hemorrhage (HR: {HR_bleed_hz * 60:.0f} BPM)",
    )

    ax.set_xlabel("Left Ventricular Volume (mL)")
    ax.set_ylabel("Left Ventricular Pressure (mmHg)")
    ax.set_title("Closed-Loop Multiscale Circulation: Hemorrhagic Shock", weight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plt.savefig("closed_loop_shock.png", dpi=300)
    logger.info("Saved final PV loop comparison to closed_loop_shock.png")
