"""
Reproduction of key figures from the RDQ20-MF paper:

  F. Regazzoni, L. Dede', A. Quarteroni, "Biophysically detailed mathematical
  models of multiscale cardiac active mechanics", PLOS Computational Biology (2020)
  https://doi.org/10.1371/journal.pcbi.1008294

Figures reproduced:
  Fig 12  — Steady-state force-calcium curves (human, body temperature)
  Fig 14  — Steady-state force-length curves  (human, body temperature)
  Fig 15  — Isometric twitches + phase loops  (rat, room temperature)
  Fig 16  — Isometric twitches                (human, body temperature)

Notes on parameterisation:
  * Figures 11-15 (rat) use params_RDQ20-MF_rat_room-temperature
  * Figures 12, 14, 16 (human) use params_RDQ20-MF_human_body-temperature
  * The calcium transient for rat twitches is a bi-exponential fit to the
    experimental data of Janssen & de Tombe (1997) (Table S2 in the paper).
  * The calcium transient for human twitches is from the ToR-ORd ionic model.
    For convenience we use a bi-exponential approximation that matches the
    qualitative shape used in the paper (peak ≈ 0.8 µM, longer duration).

Run:
    python demo/reproduce_figures_rdq20mf.py [--show]
"""

import numpy as np
import matplotlib.pyplot as plt

from crossbridge import RDQ20MF


# ---------------------------------------------------------------------------
# Parameter sets (from Table 3 of the paper)
# ---------------------------------------------------------------------------

PARAMS_RAT = {
    "LA": 1.25, "LM": 1.65, "LB": 0.18, "SL0": 2.2,
    "mu": 10.0, "gamma": 12.0, "Q": 2.0,
    "Kd0": 0.835, "alphaKd": -1.258,
    "Koff": 120.0, "Kbasic": 24.0,
    "r0": 134.31, "alpha": 25.184,
    "mu0_fP": 32.708, "mu1_fP": 0.779,
    "a_XB": 22.894e3,
}

PARAMS_HUMAN = {
    "LA": 1.25, "LM": 1.65, "LB": 0.18, "SL0": 2.2,
    "mu": 10.0, "gamma": 12.0, "Q": 2.0,
    "Kd0": 0.381, "alphaKd": -0.571,
    "Koff": 100.0, "Kbasic": 13.0,
    "r0": 134.31, "alpha": 25.184,
    "mu0_fP": 32.653, "mu1_fP": 0.778,
    "a_XB": 22.894e3,
}


# ---------------------------------------------------------------------------
# Calcium transients
# ---------------------------------------------------------------------------

def _calcium_rat(t: np.ndarray, c0: float = 0.1) -> np.ndarray:
    """
    Bi-exponential calcium transient for rat at room temperature.

    Approximates the synthetic transient fitted from Janssen & de Tombe (1997)
    as described in Supporting Information S2 of the paper.
    Parameters: peak ≈ 1.2 µM, tau_rise = 0.020 s, tau_decay = 0.110 s,
    t_peak = 0.050 s relative to stimulus.
    """
    t_start = 0.0   # [s]
    t_peak = 0.05   # [s]
    cmax = 1.2      # [µM]
    tau1 = 0.020    # [s]  rise
    tau2 = 0.110    # [s]  decay

    beta = (tau1 / tau2) ** (-1 / (tau1 / tau2 - 1)) - \
           (tau1 / tau2) ** (-1 / (1 - tau2 / tau1))
    Ca = np.full_like(t, c0, dtype=float)
    mask = t >= (t_start + t_peak)
    dt = t[mask] - (t_start + t_peak)
    Ca[mask] = c0 + (cmax - c0) / beta * (np.exp(-dt / tau1) - np.exp(-dt / tau2))
    Ca = np.maximum(Ca, c0)
    return Ca


def _calcium_human(t: np.ndarray, c0: float = 0.1) -> np.ndarray:
    """
    Bi-exponential approximation of the ToR-ORd calcium transient for human
    body-temperature cells (used in Fig 16 of the paper).
    Peak ≈ 0.80 µM, slower kinetics consistent with human heart rate ~70 bpm.
    """
    t_start = 0.0
    t_peak = 0.065   # [s]
    cmax = 0.80      # [µM]
    tau1 = 0.030     # [s]
    tau2 = 0.180     # [s]

    beta = (tau1 / tau2) ** (-1 / (tau1 / tau2 - 1)) - \
           (tau1 / tau2) ** (-1 / (1 - tau2 / tau1))
    Ca = np.full_like(t, c0, dtype=float)
    mask = t >= (t_start + t_peak)
    dt = t[mask] - (t_start + t_peak)
    Ca[mask] = c0 + (cmax - c0) / beta * (np.exp(-dt / tau1) - np.exp(-dt / tau2))
    Ca = np.maximum(Ca, c0)
    return Ca


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_steady_state(sl_values, ca_values, params, duration=1.0):
    """
    Run a (SL, Ca) grid to steady state. Returns Ta [kPa] matrix (n_SL, n_Ca).
    """
    SL_grid, Ca_grid = np.meshgrid(sl_values, ca_values, indexing="ij")
    SL_flat = SL_grid.flatten()
    Ca_flat = Ca_grid.flatten()
    n = len(SL_flat)

    model = RDQ20MF(num_cells=n, params=params)
    dt = model.dt
    dSL = np.zeros(n)
    for _ in range(int(duration / dt)):
        model.advance_step(dt, Ca_flat, SL_flat, dSL)

    ta = model.get_active_tension().reshape(SL_grid.shape)
    return ta


def _run_twitches(sl_values, ca_fn, params, T_beat=0.8, n_warmup=3):
    """
    Run isometric twitches over multiple beats to reach a periodic steady state.

    Parameters
    ----------
    sl_values : array (n_cells,)
    ca_fn     : callable(t_array) → Ca array (same shape)
    params    : dict
    T_beat    : float, beat period [s]
    n_warmup  : int, warm-up beats (not stored)

    Returns
    -------
    t_arr     : (n_steps,) time within one beat
    Ta_hist   : (n_steps, n_cells)
    Ca_hist   : (n_steps, n_cells)
    """
    n_cells = len(sl_values)
    model = RDQ20MF(num_cells=n_cells, params=params)
    dt = model.dt
    n_steps = int(T_beat / dt)
    t_arr = np.arange(n_steps) * dt
    Ca_beat = ca_fn(t_arr)                         # (n_steps,) — same for all SL
    SL_arr = np.asarray(sl_values, dtype=float)
    dSL_arr = np.zeros(n_cells)

    # Warm-up beats (use same Ca transient, repeated)
    for _ in range(n_warmup):
        for i in range(n_steps):
            Ca_curr = np.full(n_cells, Ca_beat[i])
            model.advance_step(dt, Ca_curr, SL_arr, dSL_arr)

    # Record one beat
    Ta_hist = np.zeros((n_steps, n_cells))
    Ca_hist = np.zeros((n_steps, n_cells))
    for i in range(n_steps):
        Ca_curr = np.full(n_cells, Ca_beat[i])
        Ca_hist[i] = Ca_curr
        model.advance_step(dt, Ca_curr, SL_arr, dSL_arr)
        Ta_hist[i] = model.get_active_tension()

    return t_arr, Ta_hist, Ca_hist


# ---------------------------------------------------------------------------
# Figure 12 — Steady-state force-calcium (human, body temperature)
# ---------------------------------------------------------------------------

def plot_fig12_steady_state_ca(save=True):
    """
    Reproduces Fig 12 (right panel): Steady-state Ta vs [Ca²⁺] for the
    MF-ODE model calibrated for body-temperature human cardiomyocytes.
    SL ∈ {1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3} µm.
    """
    print("Generating Fig 12 — Steady-state force-calcium (human)...")

    sl_levels = np.array([1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3])
    ca_levels = np.logspace(np.log10(0.05), np.log10(15.0), 60)

    ta_map = _run_steady_state(sl_levels, ca_levels, PARAMS_HUMAN, duration=1.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(sl_levels)))
    for i, sl in enumerate(sl_levels):
        ax.semilogx(ca_levels, ta_map[i], color=colors[i], label=f"{sl:.1f}")

    ax.set_title("Fig 12 (MF-ODE): Steady-state Ta vs [Ca²⁺]\n(human, body temperature)")
    ax.set_xlabel(r"$[Ca^{2+}]_i$ [$\mu$M]")
    ax.set_ylabel(r"$T_a$ [kPa]")
    ax.set_xlim(0.05, 15)
    ax.set_ylim(0, None)
    ax.legend(title="SL [µm]", fontsize="small", loc="upper left")
    ax.grid(True, which="both", alpha=0.2)
    plt.tight_layout()

    fname = "rdq20mf_fig12_steady_state_ca.png"
    if save:
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"  Saved: {fname}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 14 — Steady-state force-length (human, body temperature)
# ---------------------------------------------------------------------------

def plot_fig14_steady_state_sl(save=True):
    """
    Reproduces Fig 14 (right panel): Steady-state Ta vs SL for the
    MF-ODE model calibrated for body-temperature human cardiomyocytes.
    [Ca²⁺] levels from the paper legend.
    """
    print("Generating Fig 14 — Steady-state force-length (human)...")

    sl_fine = np.linspace(1.55, 2.3, 50)
    # Ca levels matching the paper Fig 14 legend
    ca_levels = np.array([0.13, 0.16, 0.20, 0.25, 0.32, 0.40, 0.50,
                          0.63, 0.79, 1.00, 1.26, 2.51, 5.01])

    ta_map = _run_steady_state(sl_fine, ca_levels, PARAMS_HUMAN, duration=1.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(ca_levels)))
    for i, ca in enumerate(ca_levels):
        ax.plot(sl_fine, ta_map[:, i], color=colors[i], label=f"{ca:.2f}")

    ax.set_title("Fig 14 (MF-ODE): Steady-state Ta vs SL\n(human, body temperature)")
    ax.set_xlabel(r"SL [$\mu$m]")
    ax.set_ylabel(r"$T_a$ [kPa]")
    ax.set_xlim(1.55, 2.3)
    ax.set_ylim(0, None)
    ax.legend(title=r"$[Ca^{2+}]_i$ [µM]", fontsize="x-small", ncol=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    fname = "rdq20mf_fig14_steady_state_sl.png"
    if save:
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"  Saved: {fname}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 15 — Isometric twitches + phase loops (rat, room temperature)
# ---------------------------------------------------------------------------

def plot_fig15_rat_twitches(save=True):
    """
    Reproduces Fig 15: Force transients (top) and phase loops Ta vs Ca (bottom)
    for rat, room-temperature cardiomyocytes at SL ∈ {1.90, 2.05, 2.20} µm.

    The calcium transient is a bi-exponential fit matching Janssen & de Tombe 1997.
    Three warm-up beats are run first so the crossbridge state is in periodic
    steady state before recording.
    """
    print("Generating Fig 15 — Isometric twitches + phase loops (rat)...")

    sl_values = np.array([1.90, 2.05, 2.20])
    T_beat = 0.8   # [s]

    t_arr, Ta_hist, Ca_hist = _run_twitches(
        sl_values, _calcium_rat, PARAMS_RAT,
        T_beat=T_beat, n_warmup=3,
    )

    fig, axes = plt.subplots(2, 1, figsize=(7, 9))
    labels = ["SL = 1.90 µm", "SL = 2.05 µm", "SL = 2.20 µm"]
    colors = ["tab:blue", "tab:orange", "tab:red"]

    # Top: Ta vs time
    ax = axes[0]
    for i, lbl in enumerate(labels):
        ax.plot(t_arr, Ta_hist[:, i], color=colors[i], label=lbl)
    ax.set_title("Fig 15 (MF-ODE): Isometric twitches — rat, room temperature")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$T_a$ [kPa]")
    ax.set_xlim(0, T_beat)
    ax.set_ylim(0, None)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.2)

    # Bottom: Ta vs Ca (phase loop)
    ax = axes[1]
    for i, lbl in enumerate(labels):
        ax.plot(Ca_hist[:, i], Ta_hist[:, i], color=colors[i], label=lbl)
    ax.set_title("Fig 15 (MF-ODE): Phase loops — rat, room temperature")
    ax.set_xlabel(r"$[Ca^{2+}]_i$ [$\mu$M]")
    ax.set_ylabel(r"$T_a$ [kPa]")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fname = "rdq20mf_fig15_rat_twitches.png"
    if save:
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"  Saved: {fname}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 16 — Isometric twitches (human, body temperature)
# ---------------------------------------------------------------------------

def plot_fig16_human_twitches(save=True):
    """
    Reproduces Fig 16: Tension transients during isometric twitches at different
    SL for body-temperature human cardiomyocytes (MF-ODE model).
    SL ∈ {1.7, 1.8, 1.9, 2.0} µm as in the paper.
    """
    print("Generating Fig 16 — Isometric twitches (human)...")

    sl_values = np.array([1.7, 1.8, 1.9, 2.0])
    T_beat = 0.8   # [s]

    t_arr, Ta_hist, _ = _run_twitches(
        sl_values, _calcium_human, PARAMS_HUMAN,
        T_beat=T_beat, n_warmup=3,
    )
    # Normalised traces
    Ta_max = Ta_hist.max(axis=0, keepdims=True)
    Ta_norm = np.where(Ta_max > 0, Ta_hist / Ta_max, Ta_hist)

    fig, axes = plt.subplots(2, 1, figsize=(7, 8))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    labels = [f"SL = {sl:.1f} µm" for sl in sl_values]

    ax = axes[0]
    for i, lbl in enumerate(labels):
        ax.plot(t_arr, Ta_hist[:, i], color=colors[i], label=lbl)
    ax.set_title("Fig 16 (MF-ODE): Isometric twitches — human, body temperature")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$T_a$ [kPa]")
    ax.set_xlim(0, T_beat)
    ax.set_ylim(0, None)
    ax.legend(title="SL [µm]", fontsize="small")
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    for i, lbl in enumerate(labels):
        ax.plot(t_arr, Ta_norm[:, i], color=colors[i], label=lbl)
    ax.set_title("Normalised tension transients")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$T_a / T_a^{max}$ [-]")
    ax.set_xlim(0, T_beat)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fname = "rdq20mf_fig16_human_twitches.png"
    if save:
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"  Saved: {fname}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reproduce RDQ20-MF paper figures.")
    parser.add_argument("--show", action="store_true",
                        help="Display figures interactively instead of saving.")
    args = parser.parse_args()
    save = not args.show

    print("=" * 60)
    print("RDQ20-MF Figure Reproduction")
    print("=" * 60)

    plot_fig12_steady_state_ca(save=save)
    plot_fig14_steady_state_sl(save=save)
    plot_fig15_rat_twitches(save=save)
    plot_fig16_human_twitches(save=save)

    print("\nDone.")
