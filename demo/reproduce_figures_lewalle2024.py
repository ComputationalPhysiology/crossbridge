# %% [markdown]
# # Reproducing the Lewalle2024 (myosin OFF-state) paper
#
# Reproduction of key results from the Lewalle et al. (2024) myosin OFF-state paper:
#
# > A. Lewalle, G. Milburn, K. S. Campbell, S. A. Niederer, "Cardiac length-dependent
# > activation driven by force-dependent thick-filament dynamics", Biophysical
# > Journal, 123(18), 2996-3009 (2024). https://doi.org/10.1016/j.bpj.2024.07.021
#
# This reproduces the paper's central claim -- that myosin OFF-state feedback on
# *total* force (the paper's "paradigm A", `which_dep="totalforce"`) alone
# reproduces the length-dependent activation (LDA) of active tension -- using
# this package's `Lewalle2024` model at its default calibration.
#
# This model's `default_parameters()` (a, k, pCa50ref, kuw, kws, gs, gw, phi,
# Aeff, Tref, k1, k2, koffon) are, verbatim, the parameters the paper reports
# in its Fig. 6 caption as "recalibrated to approximate the Awinda et al.
# control measurements" at SL = 1.9 and 2.3 um -- *not* the Tref = 109 kPa /
# [Ca2+]50 = 1.17 uM test parameterization used for the separate
# consistency-check in Fig. 3e (whose fitted k1/k2 pair the paper does not
# fully report, only the ratio K_OFF). Figs. A-B below therefore compare
# against Fig. 6a (the F-pCa curve at these two calibrated lengths), not
# Fig. 3e.
#
# It does *not* reproduce Figs. 3-4's parameter-space fitting maps, which compare
# the amended model against the *original* (non-OFF-state) Land et al. (2017)
# model to back out equivalent ad hoc LDA parameters; that original model is not
# implemented in this package. Instead, this notebook evaluates the amended
# (OFF-state) model directly:
#
# - **Fig A** -- Steady-state total-tension-pCa curves at SL = 1.9 and 2.3 um,
#   the two lengths and parameterization used in the paper's Fig. 6a: longer
#   SL should show both higher maximum force and higher calcium sensitivity
#   (left-shifted pCa50).
# - **Fig B** -- Length dependence of active tension across the same SL range
#   (1.8-2.3 um) used throughout the paper's SL-dependence figures (a
#   Frank-Starling curve), summarizing the LDA effect the paper attributes to
#   OFF-state force feedback.
# - **Fig C** -- Isometric twitches at different SL under a physiological
#   calcium transient -- not a specific paper figure (the paper's own dynamic
#   results are frequency-domain stiffness measurements, not transient-driven
#   twitches), but it demonstrates that the LDA effect persists under this
#   package's per-timestep coupling API, not just in the closed-form steady
#   state.
#
# This can also be run as a standalone script, saving each figure to a PNG
# file instead of displaying it inline:
# ```bash
# python demo/reproduce_figures_lewalle2024.py --save
# ```

# %%
import numpy as np
import matplotlib.pyplot as plt

from crossbridge import Lewalle2024, calcium_trace


# %% [markdown]
# ## Helpers


# %%
def _run_steady_state(sl_values, pca_values, duration=2.0):
    """
    Run a (SL, pCa) grid to steady state.

    Returns (Ta, Ttotal), each shape (n_SL, n_pCa), in kPa.
    """
    SL_grid, pCa_grid = np.meshgrid(sl_values, pca_values, indexing="ij")
    SL_flat = SL_grid.flatten()
    Ca_flat = 10.0 ** (-pCa_grid.flatten()) * 1e6  # [uM]
    n = len(SL_flat)

    model = Lewalle2024(num_cells=n)
    dt = model.dt
    for _ in range(int(duration / dt)):
        model.advance_step(dt, Ca_flat, SL_flat)

    Ta = model.get_active_tension().reshape(SL_grid.shape)
    Ttotal = model.get_total_tension().reshape(SL_grid.shape)
    return Ta, Ttotal


def _run_twitch(sl_values, ca_fn, duration=1.0):
    """
    Run a single isometric twitch per SL under a prescribed calcium transient.

    Returns (t_arr, Ta_hist), Ta_hist shape (n_steps, n_cells), kPa.
    """
    sl_values = np.asarray(sl_values, dtype=float)
    n = len(sl_values)
    model = Lewalle2024(num_cells=n)
    dt = model.dt
    n_steps = int(duration / dt)
    t_arr = np.arange(n_steps) * dt
    Ca_t = ca_fn(t_arr)  # (n_steps,) -- same transient for every SL

    Ta_hist = np.zeros((n_steps, n))
    for i in range(n_steps):
        model.advance_step(dt, float(Ca_t[i]), sl_values)
        Ta_hist[i] = model.get_active_tension()

    return t_arr, Ta_hist


# %% [markdown]
# ## Fig A -- Steady-state force-pCa at two sarcomere lengths
#
# Longer SL should show both higher maximum force and higher calcium
# sensitivity (a left-shifted pCa50), matching the paper's Fig. 6a. SL = 1.9
# and 2.3 um are the two lengths used in that figure -- at this model's
# *default* parameters, which are exactly its "recalibrated to Awinda et al."
# ones (see module docstring). SL = 1.8/2.0 um (used by an earlier version of
# this demo) don't correspond to any paper figure and, combined with a pCa
# window (4.3-6.3) that cropped off the low-Ca plateau, made the LDA effect
# barely visible -- not because the model lacks it, but because neither the
# lengths nor the pCa range matched what the paper actually plots.


# %%
def plot_figA_force_pca(save=True):
    print("Generating Fig A -- Steady-state force-pCa (SL = 1.9, 2.3 um)...")

    sl_levels = np.array([1.9, 2.3])
    pca_levels = np.linspace(3.5, 6.5, 60)

    _, ttotal = _run_steady_state(sl_levels, pca_levels, duration=2.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["tab:blue", "tab:red"]
    for i, sl in enumerate(sl_levels):
        ax.plot(pca_levels, ttotal[i], color=colors[i], label=f"SL = {sl:.1f} um")

    ax.invert_xaxis()  # pCa convention: high Ca (activation) on the left
    ax.set_title("Fig A: Steady-state force-pCa (Lewalle2024, paradigm A)\ncf. paper Fig. 6a")
    ax.set_xlabel("pCa")
    ax.set_ylabel(r"$T_\mathrm{total}$ [kPa]")
    ax.set_ylim(0, None)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    fname = "lewalle2024_figA_force_pca.png"
    if save:
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"  Saved: {fname}")
    else:
        plt.show()
    plt.close(fig)



# %% [markdown]
# ## Fig B -- Length dependence of active tension (Frank-Starling)
#
# Steady-state active tension at submaximal calcium (near the half-activation
# point of Fig A, pCa ~ 5.3) across a range of SL: the Frank-Starling curve
# produced purely by OFF-state force feedback. Submaximal rather than
# saturating Ca is used deliberately -- as in Fig A, the OFF-state feedback
# loop's effect on tension is largest on the rising part of the F-pCa curve
# and largely vanishes once every RU is already saturated, matching how real
# (submaximally activated) myocardium exhibits Frank-Starling behavior.


# %%
def plot_figB_length_tension(save=True):
    print("Generating Fig B -- Length dependence of active tension...")

    sl_fine = np.linspace(1.8, 2.3, 20)
    pca_submaximal = np.array([5.3])

    ta, _ = _run_steady_state(sl_fine, pca_submaximal, duration=2.0)
    ta = ta[:, 0]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(sl_fine, ta, color="tab:purple", marker="o", markersize=3)
    ax.set_title(
        "Fig B: Length dependence of active tension (Frank-Starling)\nat submaximal pCa = 5.3"
    )
    ax.set_xlabel("SL [um]")
    ax.set_ylabel(r"$T_a$ [kPa]")
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    fname = "lewalle2024_figB_length_tension.png"
    if save:
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"  Saved: {fname}")
    else:
        plt.show()
    plt.close(fig)


# %% [markdown]
# ## Fig C -- Isometric twitches at different SL
#
# Isometric twitch tension transients at three sarcomere lengths, under a
# shared calcium transient. Demonstrates the LDA effect in the dynamic
# (per-timestep) coupling regime, not just the closed-form steady state.


# %%
def plot_figC_twitches(save=True):
    print("Generating Fig C -- Isometric twitches at different SL...")

    sl_values = np.array([1.85, 2.0, 2.15])
    duration = 1.5  # [s]

    def ca_fn(t):
        return calcium_trace(t, c0=0.1, cmax=5.0, tau1=0.02, tau2=0.15, t0=0.05)

    t_arr, ta_hist = _run_twitch(sl_values, ca_fn, duration=duration)

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    colors = ["tab:blue", "tab:orange", "tab:red"]
    labels = [f"SL = {sl:.2f} um" for sl in sl_values]

    ax = axes[0]
    ax.plot(t_arr, ca_fn(t_arr), color="black")
    ax.set_ylabel(r"$[Ca^{2+}]_i$ [$\mu$M]")
    ax.set_title("Fig C: Isometric twitches at different SL (Lewalle2024)")
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    for i, lbl in enumerate(labels):
        ax.plot(t_arr, ta_hist[:, i], color=colors[i], label=lbl)
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$T_a$ [kPa]")
    ax.set_xlim(0, duration)
    ax.set_ylim(0, None)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fname = "lewalle2024_figC_twitches.png"
    if save:
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"  Saved: {fname}")
    else:
        plt.show()
    plt.close(fig)


# %% [markdown]
# ## Generate the figures
#
# `__name__ == "__main__"` is true both when this file is run directly as a
# script and when it is executed as this notebook, so the same block serves
# both: inline display here, or `--save` to write PNG files when run from
# the command line. `parse_known_args` (rather than `parse_args`) makes this
# robust to being executed under a Jupyter kernel, whose own launcher
# arguments would otherwise be rejected as unrecognized.

# %%
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reproduce Lewalle2024 paper results.")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save figures to PNG files instead of displaying them interactively.",
    )
    args, _ = parser.parse_known_args()

    print("=" * 60)
    print("Lewalle2024 (myosin OFF-state) Figure Reproduction")
    print("=" * 60)

    plot_figA_force_pca(save=args.save)
    plot_figB_length_tension(save=args.save)
    plot_figC_twitches(save=args.save)

    print("\nDone.")
