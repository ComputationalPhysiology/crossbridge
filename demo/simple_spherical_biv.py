# %% [markdown]
# # Tutorial: Biventricular Multiscale Cardiac Modeling
#
# This script demonstrates how to couple a 1D electrophysiology cell model (ToR-ORd)
# and a 3D sarcomere mechanics model (RDQ18) into a 0D closed-loop circulation
# model (Regazzoni2020) to simulate a healthy, resting biventricular human heart.

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
from circulation.regazzoni2020 import Regazzoni2020


def main():
    # --- Configure Logging ---
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    # Suppress the verbose circulation base logging to keep output clean
    logging.getLogger("circulation.base").setLevel(logging.WARNING)
    # =========================================================================
    # 1. ELECTROPHYSIOLOGY SETUP (ToR-ORd)
    # =========================================================================
    logger.info("--- Step 1: Setting up Electrophysiology ---")
    HR_hz = 1.25  # 75 Beats Per Minute

    module_path = Path("ToRORd_dynCl_endo.py")
    if not module_path.is_file():
        logger.info("Generating ToRORd_dynCl_endo.py from CellML...")
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

    fgr = numba.njit(ep_model.generalized_rush_larsen)
    Ca_index = ep_model.state_index("cai")

    # Pre-pace the cell model to a steady state
    BCL_MS = 1000.0 / HR_hz
    DT_EP_MS = 0.05
    NUM_BEATS = 100

    y_init = ep_model.init_state_values()
    params_ep = ep_model.init_parameter_values(i_Stim_Period=BCL_MS)
    params_ep[ep_model.parameter_index("i_Stim_Amplitude")] = -150.0

    t_one_beat = np.arange(0, BCL_MS, DT_EP_MS)
    ca_last_beat = []

    logger.info(f"Pre-pacing ToR-ORd at {HR_hz * 60:.0f} BPM for {NUM_BEATS} beats...")
    for beat in tqdm(range(NUM_BEATS)):
        for ti in t_one_beat:
            t_global = beat * BCL_MS + ti
            y_init = fgr(y_init, t_global, DT_EP_MS, params_ep)
            if beat == NUM_BEATS - 1:
                ca_last_beat.append(y_init[Ca_index] * 1000.0)

    ep_state_init = y_init
    ca_min, ca_max = np.min(ca_last_beat), np.max(ca_last_beat)

    # =========================================================================
    # 2. MECHANICS & GEOMETRY SETUP
    # =========================================================================
    logger.info("--- Step 2: Setting up Mechanics and Geometry ---")
    KPA_TO_MMHG = 7.5
    RESTING_SARCOMERE_LENGTH_UM = 2.1

    # Left Ventricle (Thick Wall -> High Pressure Systemic Circulation)
    WALL_THICKNESS_LV_CM = 1.0
    CHAMBER_RADIUS_LV_CM = 3.0
    GEO_SCALE_LV = ((2.0 * WALL_THICKNESS_LV_CM) / CHAMBER_RADIUS_LV_CM) * KPA_TO_MMHG
    UNLOADED_VOLUME_LV_ML = 120.0

    # Right Ventricle (Thin Wall -> Low Pressure Pulmonary Circulation)
    WALL_THICKNESS_RV_CM = 0.3
    CHAMBER_RADIUS_RV_CM = 3.5
    GEO_SCALE_RV = ((2.0 * WALL_THICKNESS_RV_CM) / CHAMBER_RADIUS_RV_CM) * KPA_TO_MMHG
    UNLOADED_VOLUME_RV_ML = 120.0

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

    # =========================================================================
    # 3. CLOSED-LOOP COUPLING (Regazzoni 2020)
    # =========================================================================
    logger.info("--- Step 3: Coupling to Closed-Loop Circulation ---")

    base_model = Regazzoni2020(add_units=False)
    params = base_model.parameters.copy()
    init_state = base_model._initial_state.copy()

    params["HR"] = HR_hz

    # Add a volume buffer to ensure the large resting ventricles have sufficient preload
    C_ven = params["circulation"]["SYS"]["C_VEN"]
    init_state["p_VEN_SYS"] += 250.0 / C_ven

    # Encapsulate EP and Sarcomere states for the callback
    ep_state = ep_state_init.copy()
    sarcomere_lv = RDQ18(num_cells=1, Ta_max=250.0, params=RDQ18.default_parameters())
    sarcomere_rv = RDQ18(num_cells=1, Ta_max=250.0, params=RDQ18.default_parameters())

    mech_state = {
        "Ta_LV": 0.0,
        "p_val_LV": 0.0,
        "P_LV": 0.0,
        "Ta_RV": 0.0,
        "p_val_RV": 0.0,
        "P_RV": 0.0,
    }

    # --- NEW: Data collectors for plotting ---
    data_ep = []
    data_mech = {"Ca_input": [], "Ta_LV": [], "Ta_RV": [], "lmbda_LV": [], "lmbda_RV": []}

    log_step = int((1.0 / HR_hz / 1e-3) / 4)

    def callback(model, step, t, **kwargs):
        """Advances the microscopic EP and Mechanics models at every macroscopic step."""
        dt_s = 1e-3
        sub_steps = 20
        dt_ep = (dt_s * 1000.0) / sub_steps
        t_start_ms = (t - dt_s) * 1000.0

        # 1. Advance the Electrophysiology (Sub-stepped for stiffness)
        for k in range(sub_steps):
            t_sub = t_start_ms + k * dt_ep
            ep_state[:] = fgr(ep_state, t_sub, dt_ep, params_ep)

        Ca_uM = ep_state[Ca_index] * 1000.0

        # 2. Extract macroscopic volumes and map to 1D fiber stretch
        V_LV = model.state[model.states_names.index("V_LV")]
        V_RV = model.state[model.states_names.index("V_RV")]

        lmbda_lv = (max(5.0, V_LV) / UNLOADED_VOLUME_LV_ML) ** (1.0 / 3.0)
        lmbda_rv = (max(5.0, V_RV) / UNLOADED_VOLUME_RV_ML) ** (1.0 / 3.0)

        # 3. Normalize calcium and run crossbridge mechanics
        normalized_ca = np.clip((Ca_uM - ca_min) / (ca_max - ca_min), 0.0, 1.0)
        Ca_input = 0.08 + (normalized_ca**2) * (0.90 - 0.08)

        sarcomere_lv.advance_ODE(
            dt_s, np.array([Ca_input]), np.array([RESTING_SARCOMERE_LENGTH_UM * lmbda_lv])
        )
        sarcomere_rv.advance_ODE(
            dt_s, np.array([Ca_input]), np.array([RESTING_SARCOMERE_LENGTH_UM * lmbda_rv])
        )

        mech_state["Ta_LV"] = sarcomere_lv.Ta_max * sarcomere_lv.compute_permissivity()[0]
        mech_state["Ta_RV"] = sarcomere_rv.Ta_max * sarcomere_rv.compute_permissivity()[0]

        # --- NEW: Save data for plotting ---
        data_ep.append(ep_state.copy())
        data_mech["Ca_input"].append(Ca_input)
        data_mech["Ta_LV"].append(mech_state["Ta_LV"])
        data_mech["Ta_RV"].append(mech_state["Ta_RV"])
        data_mech["lmbda_LV"].append(lmbda_lv)
        data_mech["lmbda_RV"].append(lmbda_rv)

        # Diagnostic logging
        if step < 5 or step % log_step == 0:
            logger.info(
                f"t={t:.2f}s | Ca={Ca_uM:.2f} | Ta={mech_state['Ta_LV']:.1f} | "
                f"V_LV={V_LV:.1f} | P_LV={mech_state['P_LV']:.1f} | V_RV={V_RV:.1f} | P_RV={mech_state['P_RV']:.1f}"
            )

    def p_BiV(V_LV, V_RV, t):
        """Converts active sarcomere tension into macroscopic chamber pressure using the Law of Laplace."""
        lmbda_lv = (max(5.0, V_LV) / UNLOADED_VOLUME_LV_ML) ** (1.0 / 3.0)
        lmbda_rv = (max(5.0, V_RV) / UNLOADED_VOLUME_RV_ML) ** (1.0 / 3.0)

        P11_lv, mech_state["p_val_LV"] = solve_fiber_stress(
            lmbda_lv, mech_state["Ta_LV"], mech_state["p_val_LV"]
        )
        P11_rv, mech_state["p_val_RV"] = solve_fiber_stress(
            lmbda_rv, mech_state["Ta_RV"], mech_state["p_val_RV"]
        )

        # Apply geometrical scaling (Law of Laplace)
        mech_state["P_LV"] = max(0.0, P11_lv * GEO_SCALE_LV)
        mech_state["P_RV"] = max(0.0, P11_rv * GEO_SCALE_RV)

        return mech_state["P_LV"], mech_state["P_RV"]

    # Initialize the wrapper
    model = Regazzoni2020(
        parameters=params,
        initial_state=init_state,
        add_units=False,
        p_BiV=p_BiV,
        callback=callback,
        verbose=False,
    )

    logger.info("Solving Biventricular Closed-Loop Model for 15 beats...")
    history = model.solve(num_beats=15, dt=1e-3, dt_eval=1e-3)

    # --- NEW: Bundle collected data into history ---
    history["ep_state"] = np.array(data_ep).T
    for key, val in data_mech.items():
        history[key] = np.array(val)

    # =========================================================================
    # 4. PLOTTING RESULTS
    # =========================================================================
    logger.info("--- Step 4: Generating Plots ---")

    # Extract only the final stable beat
    samples_per_beat = int(1.0 / HR_hz / 1e-3)
    stable_slice = slice(-samples_per_beat, None)
    t_plot = np.arange(samples_per_beat) * 1e-3

    # --- PLOT 1: PV Loops ---
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Left Ventricle
    axs[0].plot(history["V_LV"], history["p_LV"], color="blue", alpha=0.2)
    axs[0].plot(
        history["V_LV"][stable_slice],
        history["p_LV"][stable_slice],
        color="blue",
        lw=2.5,
        label="Stable Beat",
    )
    axs[0].set_xlabel("Volume (mL)", weight="bold")
    axs[0].set_ylabel("Pressure (mmHg)", weight="bold")
    axs[0].set_title("Left Ventricle (Thick Wall)", weight="bold")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    # Right Ventricle
    axs[1].plot(history["V_RV"], history["p_RV"], color="blue", alpha=0.2)
    axs[1].plot(
        history["V_RV"][stable_slice],
        history["p_RV"][stable_slice],
        color="blue",
        lw=2.5,
        label="Stable Beat",
    )
    axs[1].set_xlabel("Volume (mL)", weight="bold")
    axs[1].set_ylabel("Pressure (mmHg)", weight="bold")
    axs[1].set_title("Right Ventricle (Thin Wall)", weight="bold")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    fig.tight_layout()
    plt.savefig("tutorial_biventricular_pv_loops.png", dpi=300)

    # --- PLOT 2: Wiggers & Cellular Mechanics Dashboard ---
    fig_w, axs_col = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

    # 1. Pressures
    axs_col[0].plot(t_plot, history["p_LV"][stable_slice], "r-", lw=2, label="LV")
    axs_col[0].plot(t_plot, history["p_AR_SYS"][stable_slice], "r--", lw=1.5, label="Aorta")
    axs_col[0].plot(t_plot, history["p_LA"][stable_slice], "r:", lw=1.5, label="LA")
    axs_col[0].plot(t_plot, history["p_RV"][stable_slice], "b-", lw=2, label="RV")
    axs_col[0].plot(
        t_plot, history["p_AR_PUL"][stable_slice], "b--", lw=1.5, label="Pulmonary Artery"
    )
    axs_col[0].set_title("Wiggers Diagram (Pressures)", weight="bold")
    axs_col[0].set_ylabel("Pressure (mmHg)")
    axs_col[0].grid(True, alpha=0.3)
    axs_col[0].legend(loc="upper right", fontsize="small", ncol=2)

    # 2. Volumes
    axs_col[1].plot(t_plot, history["V_LV"][stable_slice], "r-", lw=2, label="LV Volume")
    axs_col[1].plot(t_plot, history["V_RV"][stable_slice], "b-", lw=2, label="RV Volume")
    axs_col[1].set_title("Chamber Volumes", weight="bold")
    axs_col[1].set_ylabel("Volume (mL)")
    axs_col[1].grid(True, alpha=0.3)
    axs_col[1].legend(loc="upper right")

    # 3. EP (Voltage & Calcium)
    V_index = ep_model.state_index("v")
    l1 = axs_col[2].plot(
        t_plot, history["ep_state"][V_index, stable_slice], "k-", lw=2, label="Voltage (Vm)"
    )
    axs_col[2].set_ylabel("Voltage (mV)", weight="bold")
    ax_ca = axs_col[2].twinx()
    l2 = ax_ca.plot(t_plot, history["Ca_input"][stable_slice], "g-", lw=2, label="Calcium (Mapped)")
    ax_ca.set_ylabel("Ca Concentration (uM)", color="g", weight="bold")
    axs_col[2].set_title("1D Electrophysiology", weight="bold")
    axs_col[2].grid(True, alpha=0.3)
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    axs_col[2].legend(lns, labs, loc="center right")

    # 4. Mechanics (Active Tension & Stretch)
    l3 = axs_col[3].plot(
        t_plot, history["Ta_LV"][stable_slice], "r-", lw=2, label="Active Tension (LV)"
    )
    l4 = axs_col[3].plot(
        t_plot, history["Ta_RV"][stable_slice], "b-", lw=2, label="Active Tension (RV)"
    )
    axs_col[3].set_ylabel("Active Tension (kPa)", weight="bold")
    ax_sl = axs_col[3].twinx()
    l5 = ax_sl.plot(
        t_plot,
        history["lmbda_LV"][stable_slice] * RESTING_SARCOMERE_LENGTH_UM,
        "r--",
        lw=1.5,
        label="Sarcomere Length (LV)",
    )
    l6 = ax_sl.plot(
        t_plot,
        history["lmbda_RV"][stable_slice] * RESTING_SARCOMERE_LENGTH_UM,
        "b--",
        lw=1.5,
        label="Sarcomere Length (RV)",
    )
    ax_sl.set_ylabel("Sarcomere Length (um)", color="gray", weight="bold")
    axs_col[3].set_title("3D Sarcomere Mechanics", weight="bold")
    axs_col[3].set_xlabel("Time (s)", weight="bold")
    axs_col[3].grid(True, alpha=0.3)
    lns2 = l3 + l4 + l5 + l6
    labs2 = [l.get_label() for l in lns2]
    axs_col[3].legend(lns2, labs2, loc="center right", fontsize="small", ncol=2)

    fig_w.tight_layout()
    plt.savefig("tutorial_biventricular_dashboard.png", dpi=300)

    logger.info(
        "Simulation Complete! Saved plots to tutorial_biventricular_pv_loops.png and tutorial_biventricular_dashboard.png"
    )


if __name__ == "__main__":
    main()
