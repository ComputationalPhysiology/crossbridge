# %% [markdown]
# # Reproducing the Land et al. (2017) human contraction paper
#
# Reproduction of key results from the paper this package's `Land2017` model
# implements:
#
# > S. Land, S.J. Park-Holohan, N.P. Smith, C.G. dos Remedios, J.C. Kentish,
# > S.A. Niederer, "A model of cardiac contraction based on novel
# > measurements of tension development in human cardiomyocytes", Journal of
# > Molecular and Cellular Cardiology, 106, 68-83 (2017).
# > https://doi.org/10.1016/j.yjmcc.2017.03.008
#
# `Land2017` is the *un-amended* base model: length-dependent activation
# (LDA) here is the paper's ad hoc `beta0`/`beta1` mechanism, later replaced
# by explicit myosin OFF-state feedback in this package's `Lewalle2024`
# model (see `demo/reproduce_figures_lewalle2024.py`).
#
# This does *not* reproduce Fig. 7's whole-organ biventricular finite-element
# simulation (Sec. 3.6) -- that requires a 3D mechanics solver outside this
# package's scope. Instead, this notebook evaluates the single-cell model
# directly against the paper's cellular-scale figures:
#
# - **Fig A** -- Passive viscoelastic step response, cf. paper Fig. 2: a
#   sequence of stretches shows the characteristic fast elastic jump
#   followed by a slower viscous decay (the dashpot element `Cd`).
# - **Fig B** -- Steady-state force-calcium relationship at SL = 1.8, 2.0,
#   2.2 um, cf. paper Fig. 4: the model's default ("skinned model", Table B)
#   parameters, driven purely by `beta0`/`beta1` (no OFF-state feedback
#   here), reproducing both the increase in maximum force and the
#   left-shifted pCa50 with increasing SL.
# - **Fig C** -- Quick-stretch response, cf. paper Fig. 5A: a fast 1% length
#   step produces the paper's characteristic *biphasic* response -- an
#   instantaneous force jump, a rapid decay dropping tension *below* the
#   eventual new steady state, then a slow monotonic recovery -- driven by
#   the distortion-decay terms (`Zs`, `Zw`, `gs`, `gw`).
# - **Fig D** -- Isometric twitch at different SL, cf. paper Fig. 6: using
#   the paper's "whole organ model" recalibration (Table B's alternate
#   column: higher `ca50_ref` sensitivity via a lower reference level,
#   higher `nTm`, faster `kuw`/`kws`, higher `Tref`), driven by a
#   physiological calcium transient, showing the Frank-Starling effect in
#   the dynamic (twitch) regime.
#
# This can also be run as a standalone script, saving each figure to a PNG
# file instead of displaying it inline:
# ```bash
# python demo/reproduce_figures_land2017.py --save
# ```

# %%
import numpy as np
import matplotlib.pyplot as plt

from crossbridge import Land2017, calcium_trace

#: Table B's "Whole organ model value" column -- the parameters the paper
#: changes from the skinned-myocyte calibration to represent intact muscle
#: (Sec. 3.5), used only for Fig D below.
WHOLE_ORGAN_PARAMS = {
    "ca50_ref": 0.805,  # [uM]
    "nTm": 5.0,
    "kuw": 182.0,  # [s^-1] (0.182/ms)
    "kws": 12.0,  # [s^-1] (0.012/ms)
    "Tref": 120000.0,  # [Pa] (120 kPa)
}


# %% [markdown]
# ## Fig A -- Passive viscoelastic step response
#
# A sequence of five increasing step stretches (2-10% of SL0), each held for
# ~1 s, with zero calcium throughout (`Ta = 0`, isolating the passive
# element). Each step shows a fast elastic jump (the parallel spring `F1`)
# followed by a slower viscous decay towards a lower steady value (the
# series spring-dashpot `F2`/`Cd`), matching the shape of paper Fig. 2B.


# %%
def plot_figA_passive_step_response(save=True):
    print("Generating Fig A -- Passive viscoelastic step response...")

    model = Land2017(num_cells=1)
    dt = 1e-3
    SL0 = model.p["SL0"]
    step_fractions = [0.02, 0.04, 0.06, 0.08, 0.10]
    hold_steps = 1000  # 1 s per step

    t_arr = []
    SL_arr = []
    Tp_arr = []
    t = 0.0
    for frac in step_fractions:
        SL = SL0 * (1.0 + frac)
        for _ in range(hold_steps):
            model.advance_step(dt, 0.0, np.array([SL]))
            t += dt
            t_arr.append(t)
            SL_arr.append(SL)
            Tp_arr.append(model.get_passive_tension()[0])

    t_arr = np.array(t_arr)
    SL_arr = np.array(SL_arr)
    Tp_arr = np.array(Tp_arr)

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    axes[0].plot(t_arr, SL_arr, color="black")
    axes[0].set_ylabel("SL [um]")
    axes[0].set_title("Fig A: Passive viscoelastic step response (Land2017)\ncf. paper Fig. 2B")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(t_arr, Tp_arr, color="tab:blue")
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel(r"$T_p$ [kPa]")
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    fname = "land2017_figA_passive_step.png"
    if save:
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"  Saved: {fname}")
    else:
        plt.show()
    plt.close(fig)


# %% [markdown]
# ## Fig B -- Steady-state force-pCa at three sarcomere lengths
#
# Longer SL should show both higher maximum force (beta0 > 0) and higher
# calcium sensitivity, i.e. a left-shifted pCa50 (beta1 < 0), matching the
# paper's Fig. 4.


# %%
def _run_steady_state(sl_values, ca_values, duration=3.0, params=None):
    """Run a (SL, Ca) grid to steady state. Returns Ta, shape (n_SL, n_Ca), kPa."""
    SL_grid, Ca_grid = np.meshgrid(sl_values, ca_values, indexing="ij")
    SL_flat = SL_grid.flatten()
    Ca_flat = Ca_grid.flatten()
    n = len(SL_flat)

    model = Land2017(num_cells=n, params=params)
    dt = model.dt
    for _ in range(int(duration / dt)):
        model.advance_step(dt, Ca_flat, SL_flat)

    return model.get_active_tension().reshape(SL_grid.shape)


def plot_figB_force_pca(save=True):
    print("Generating Fig B -- Steady-state force-pCa (SL = 1.8, 2.0, 2.2 um)...")

    sl_levels = np.array([1.8, 2.0, 2.2])
    ca_levels = np.logspace(-1, 1.3, 50)  # 0.1-20 uM

    ta = _run_steady_state(sl_levels, ca_levels, duration=3.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["tab:green", "tab:blue", "tab:red"]
    for i, sl in enumerate(sl_levels):
        ax.semilogx(ca_levels, ta[i], color=colors[i], label=f"SL = {sl:.1f} um")

    ax.set_title("Fig B: Steady-state force-calcium relationship (Land2017)\ncf. paper Fig. 4")
    ax.set_xlabel(r"$[Ca^{2+}]$ [$\mu$M]")
    ax.set_ylabel(r"$T_a$ [kPa]")
    ax.set_ylim(0, None)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    fname = "land2017_figB_force_pca.png"
    if save:
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"  Saved: {fname}")
    else:
        plt.show()
    plt.close(fig)


# %% [markdown]
# ## Fig C -- Quick-stretch response (velocity dependence)
#
# A cell held at steady-state active tension is given a fast 1% length step
# (applied over 10 ms) and then held at the new length. The paper's Fig. 5A
# shows this produces a *biphasic* response: tension jumps up instantly
# (crossbridge distortion from the powerstroke), then decays rapidly,
# dropping *below* the eventual new steady-state tension, before recovering
# slowly and monotonically to the higher steady state set by LDA.


# %%
def plot_figC_quick_stretch(save=True):
    print("Generating Fig C -- Quick-stretch response...")

    model = Land2017(num_cells=1)
    dt_fast = 1e-4
    SL0 = 1.9
    Ca = 3.0  # uM, held fixed throughout (isolates the length response)

    # Settle to steady state at SL0
    for _ in range(int(3.0 / dt_fast)):
        model.advance_step(dt_fast, Ca, np.array([SL0]))

    stretch_frac = 0.01
    SL1 = SL0 * (1.0 + stretch_frac)
    n_ramp = 100  # 10 ms ramp
    n_hold = 8000  # 0.8 s hold

    t_arr = []
    Ta_arr = []
    t = 0.0
    for i in range(n_ramp):
        sl = SL0 + (SL1 - SL0) * (i + 1) / n_ramp
        model.advance_step(dt_fast, Ca, np.array([sl]))
        t += dt_fast
        t_arr.append(t)
        Ta_arr.append(model.get_active_tension()[0])
    for _ in range(n_hold):
        model.advance_step(dt_fast, Ca, np.array([SL1]))
        t += dt_fast
        t_arr.append(t)
        Ta_arr.append(model.get_active_tension()[0])

    t_arr = np.array(t_arr) - t_arr[0]
    Ta_arr = np.array(Ta_arr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t_arr * 1000, Ta_arr, color="tab:red")
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(
        f"Fig C: Quick-stretch response ({stretch_frac * 100:.0f}% step, Land2017)\ncf. paper Fig. 5A"
    )
    ax.set_xlabel("t [ms]")
    ax.set_ylabel(r"$T_a$ [kPa]")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    fname = "land2017_figC_quick_stretch.png"
    if save:
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"  Saved: {fname}")
    else:
        plt.show()
    plt.close(fig)


# %% [markdown]
# ## Fig D -- Isometric twitch at different SL (whole-organ recalibration)
#
# Using Table B's "whole organ model" parameters (higher calcium
# sensitivity, higher tropomyosin cooperativity `nTm`, faster crossbridge
# cycling, higher `Tref`), driven by a physiological calcium transient, at a
# range of extension ratios `Lambda = SL/SL0` -- reproducing the
# Frank-Starling effect in the dynamic twitch regime shown in paper Fig. 6.


# %%
def plot_figD_isometric_twitch(save=True):
    print("Generating Fig D -- Isometric twitch at different SL (whole-organ params)...")

    lambdas = np.array([0.9, 0.95, 1.0, 1.05, 1.1])
    SL0 = 1.8
    sl_values = SL0 * lambdas
    duration = 1.0  # s

    model = Land2017(num_cells=len(sl_values), params=WHOLE_ORGAN_PARAMS)
    dt = model.dt
    n_steps = int(duration / dt)
    t_arr = np.arange(n_steps) * dt

    def ca_fn(t):
        return calcium_trace(t, c0=0.15, cmax=0.6, tau1=0.02, tau2=0.11, t0=0.05)

    Ca_t = ca_fn(t_arr)
    Ta_hist = np.zeros((n_steps, len(sl_values)))
    for i in range(n_steps):
        model.advance_step(dt, float(Ca_t[i]), sl_values)
        Ta_hist[i] = model.get_active_tension()

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    axes[0].plot(t_arr, Ca_t, color="black")
    axes[0].set_ylabel(r"$[Ca^{2+}]_i$ [$\mu$M]")
    axes[0].set_title(
        "Fig D: Isometric twitch at different SL (Land2017, whole-organ params)\ncf. paper Fig. 6"
    )
    axes[0].grid(True, alpha=0.2)

    colors = plt.cm.viridis(np.linspace(0, 1, len(lambdas)))
    for i, lam in enumerate(lambdas):
        axes[1].plot(t_arr, Ta_hist[:, i], color=colors[i], label=rf"$\lambda$ = {lam:.2f}")
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel(r"$T_a$ [kPa]")
    axes[1].set_xlim(0, duration)
    axes[1].set_ylim(0, None)
    axes[1].legend(fontsize="small")
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    fname = "land2017_figD_isometric_twitch.png"
    if save:
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"  Saved: {fname}")
    else:
        plt.show()
    plt.close(fig)


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
    print("Land2017 (human cardiac contraction) Figure Reproduction")
    print("=" * 60)

    plot_figA_passive_step_response(save=args.save)
    plot_figB_force_pca(save=args.save)
    plot_figC_quick_stretch(save=args.save)
    plot_figD_isometric_twitch(save=args.save)

    print("\nDone.")
