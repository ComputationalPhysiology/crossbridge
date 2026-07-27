# %% [markdown]
# # The Frank-Starling Mechanism: WK3 Circulation + zero_mech + crossbridge
#
# In this tutorial, we construct a closed-loop cardiovascular model using a
# 3-Element Windkessel (WK3) model.
#
# Instead of prescribing a macroscopic Time-Varying Elastance, we drive the mechanics
# using the `RDQ18` sarcomere model and `zero_mech` to compute the Law of Laplace.
# We demonstrate this by simulating two different filling pressures (Low vs. High Preload).

import time
import logging
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

import zero_mech
from crossbridge import RDQ18, calcium_trace
from circulation.windkessel import ThreeElementWindkessel
from circulation.units import ureg

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("circulation.base").setLevel(logging.WARNING)


def setup_mechanics():
    # 1. Physiological Anchor for Sarcomere Length
    V_ED_target = 140.0
    SL_ED_target = 2.2
    V0 = 120.0
    SL_rest = SL_ED_target * (V0 / V_ED_target) ** (1.0 / 3.0)

    # 2. Law of Laplace
    wall_thickness = 0.4
    chamber_radius = 5.0
    laplace_multiplier = (2.0 * wall_thickness) / chamber_radius
    kpa_to_mmhg = 7.5
    geo_scale = laplace_multiplier * kpa_to_mmhg

    def volume_to_stretch(V_LV):
        V_safe = max(10.0, V_LV)
        return (V_safe / V0) ** (1.0 / 3.0)

    # 3. zero_mech Model
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
            if res.converged:
                return P11_fast(lmbda, Ta, res.root), res.root
        except Exception:
            pass
        return P11_fast(lmbda, Ta, p_guess), p_guess

    return solve_fiber_stress, geo_scale, volume_to_stretch, SL_rest


def simulate_pv_loop(P_LA, heart_rate=75, num_beats=6):
    logger.info(f"--- Starting Simulation for Preload (P_LA) = {P_LA} mmHg ---")

    solve_fiber_stress, geo_scale, volume_to_stretch, SL_rest = setup_mechanics()

    heart_rate_hz = heart_rate / 60.0

    # Base parameters with correct units
    params = ThreeElementWindkessel.default_parameters()
    params["HR"] = heart_rate_hz * ureg("Hz")
    params["P_venous"] = P_LA * ureg("mmHg")

    # Internal state for the mechanics coupling
    sarcomere = RDQ18(num_cells=1, Ta_max=120.0)
    mech_state = {"Ta": 0.0, "p_guess": 0.0, "P_LV": 0.0}

    # Data collectors for plotting
    data = {"Ca": [], "Ta": [], "lmbda": []}

    log_interval = int((1.0 / heart_rate_hz / 1e-3) / 4)

    def callback(model, step, t, **kwargs):
        dt_s = 1e-3
        t_beat = t % (1.0 / heart_rate_hz)

        # Drive the Ca transient
        Ca = calcium_trace(t_beat, cmax=1.2, tau1=0.02, tau2=0.22, t0=0.05)

        # Map volume to stretch
        V_LV = model.state[model.states_names.index("V_LV")]
        lmbda = volume_to_stretch(V_LV)

        # Advance crossbridges
        sarcomere.advance_ODE(dt_s, np.array([Ca]), np.array([SL_rest * lmbda]))
        mech_state["Ta"] = sarcomere.Ta_max * sarcomere.compute_permissivity()[0]

        data["Ca"].append(Ca)
        data["Ta"].append(mech_state["Ta"])
        data["lmbda"].append(lmbda)

        if step > 0 and step % log_interval == 0:
            P_ao = model.var[1]
            logger.info(
                f"t={t:.2f}s | V_LV={V_LV:.1f} | P_LV={mech_state['P_LV']:.1f} | P_ao={P_ao:.1f}"
            )

    def p_LV_func(V_LV, t):
        """Couples the active 1D stress into the 0D chamber pressure."""
        lmbda = volume_to_stretch(V_LV)
        P11_stress, mech_state["p_guess"] = solve_fiber_stress(
            lmbda, mech_state["Ta"], mech_state["p_guess"]
        )
        mech_state["P_LV"] = max(0.0, P11_stress * geo_scale)
        return mech_state["P_LV"]

    # Initialize and solve the Windkessel
    model = ThreeElementWindkessel(parameters=params, p_LV_func=p_LV_func, callback=callback)
    history = model.solve(num_beats=num_beats, dt=1e-3, dt_eval=1e-3)

    # Inject custom data into history
    for key, val in data.items():
        history[key] = np.array(val)

    # Extract only the final stable beat
    steps_per_beat = int(1.0 / heart_rate_hz / 1e-3)
    stable_idx = slice(-steps_per_beat, None)

    results = {
        "t": np.arange(steps_per_beat) * 1e-3,
        "V_LV": history["V_LV"][stable_idx],
        "P_LV": history["p_LV"][stable_idx],
        "P_ao": history["p_ao"][stable_idx],
        "P_LA": np.full(steps_per_beat, P_LA),
        "Ca": history["Ca"][stable_idx],
        "lmbda": history["lmbda"][stable_idx],
        "Ta": history["Ta"][stable_idx],
    }

    return results


if __name__ == "__main__":
    start_time = time.time()

    res_low = simulate_pv_loop(P_LA=6.0)
    res_high = simulate_pv_loop(P_LA=12.0)

    logger.info(f"Total computation time: {time.time() - start_time:.2f} seconds")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.2, 1], hspace=0.3)

    # LEFT PANEL: PV Loop
    ax_pv = fig.add_subplot(gs[:, 0])
    ax_pv.plot(res_low["V_LV"], res_low["P_LV"], color="blue", lw=2.5, label="Low Preload (6 mmHg)")
    ax_pv.plot(
        res_high["V_LV"], res_high["P_LV"], color="red", lw=2.5, label="High Preload (12 mmHg)"
    )

    ax_pv.set_title("Frank-Starling Mechanism: Pressure-Volume Loops", fontsize=16, weight="bold")
    ax_pv.set_xlabel("Left Ventricular Volume (mL)", fontsize=14)
    ax_pv.set_ylabel("Left Ventricular Pressure (mmHg)", fontsize=14)
    ax_pv.grid(True, linestyle="--", alpha=0.6)
    ax_pv.legend(loc="upper left", fontsize=12)

    # RIGHT PANELS: Wiggers Dashboards
    # 1. Pressures
    ax_press = fig.add_subplot(gs[0, 1])
    for res, color in [(res_low, "blue"), (res_high, "red")]:
        ax_press.plot(res["t"], res["P_LV"], color=color, lw=2)
        ax_press.plot(res["t"], res["P_ao"], color=color, linestyle="--", lw=1.5, alpha=0.8)
        ax_press.plot(res["t"], res["P_LA"], color=color, linestyle=":", lw=1.5, alpha=0.8)

    ax_press.set_title("Hemodynamics (Wiggers Diagram)", fontsize=14, weight="bold")
    ax_press.set_ylabel("Pressure (mmHg)", fontsize=12)
    ax_press.grid(True, linestyle="--", alpha=0.5)

    # 2. Volume
    ax_vol = fig.add_subplot(gs[1, 1], sharex=ax_press)
    ax_vol.plot(res_low["t"], res_low["V_LV"], color="blue", lw=2)
    ax_vol.plot(res_high["t"], res_high["V_LV"], color="red", lw=2)
    ax_vol.set_ylabel("Volume (mL)", fontsize=12)
    ax_vol.grid(True, linestyle="--", alpha=0.5)

    # 3. Calcium Transient
    ax_ca = fig.add_subplot(gs[2, 1], sharex=ax_press)
    ax_ca.plot(res_low["t"], res_low["Ca"], color="blue", lw=2)
    ax_ca.plot(res_high["t"], res_high["Ca"], color="red", lw=2)
    ax_ca.set_ylabel(r"$Ca^{2+}$ ($\mu$M)", fontsize=12)
    ax_ca.grid(True, linestyle="--", alpha=0.5)

    # 4. Mechanics
    ax_mech = fig.add_subplot(gs[3, 1], sharex=ax_press)
    ax_mech_twin = ax_mech.twinx()

    for res, color in [(res_low, "blue"), (res_high, "red")]:
        ax_mech.plot(res["t"], res["Ta"], color=color, lw=2)
        ax_mech_twin.plot(res["t"], res["lmbda"], color=color, linestyle="-.", lw=1.5, alpha=0.7)

    ax_mech.set_xlabel("Time (s)", fontsize=12)
    ax_mech.set_ylabel("Active Tension (kPa)", fontsize=12)
    ax_mech_twin.set_ylabel(r"Fiber Stretch ($\lambda$)", fontsize=12)
    ax_mech.grid(True, linestyle="--", alpha=0.5)

    plt.setp(ax_press.get_xticklabels(), visible=False)
    plt.setp(ax_vol.get_xticklabels(), visible=False)
    plt.setp(ax_ca.get_xticklabels(), visible=False)

    fig.tight_layout()
    fig.savefig("frank_starling_windkessel.png", dpi=300, bbox_inches="tight")
    logger.info("Saved Frank-Starling Dashboard to frank_starling_windkessel.png")
    plt.show()
