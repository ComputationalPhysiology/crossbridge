# %% [markdown]
# # Reproducing the Tran et al. (2010) metabolite-sensitive model
#
# Reproduction of the model figures of the paper this package's `Tran2010`
# implements:
#
# > K. Tran, N.P. Smith, D.S. Loiselle, E.J. Crampin, "A metabolite-sensitive,
# > thermodynamically constrained model of cardiac cross-bridge cycling:
# > implications for force development during ischemia", Biophysical Journal,
# > 98(2), 267-276 (2010). https://doi.org/10.1016/j.bpj.2009.10.011
#
# The model is Rice et al.'s (2008) (`Rice2008`) with a crossbridge cycle made
# reversible and dependent on MgATP, MgADP, Pi and H+, parameterized for
# guinea pig. All of the paper's model results are at saturating calcium
# (200 uM) unless a calcium axis is shown, and its forces are Rice et al.'s
# normalized force, the class's `get_active_tension()` at `Ta_max = 1`.
#
# - **Fig. 2** -- force against Pi (A), Ca50 against pH (B), maximal force
#   against pH (C) and force-calcium curves at four pH (D).
# - **Fig. 3** -- force against MgATP at three MgADP levels (A), and tension
#   redevelopment after the strongly bound bridges are detached, at three
#   MgATP levels (B, and normalized in C).
# - **Fig. 4** -- complex stiffness (dynamic modulus and phase) against
#   frequency at three Pi levels.
# - **Fig. 5** -- maximal force through 15 min of ischemia, and its
#   sensitivity to each metabolite, pH and calcium at four time points.
#
# Not covered: the schematic (Fig. 1); the alternative placements of the Pi
# and H+ binding steps that Figs. 2A and 2C compare against (the paper
# rejects them, and the class implements only the chosen one); and the
# experimental data the paper overlays (Figs. 3D and 4C-D are data alone),
# which this package does not ship. Where the figures below show the paper's
# curves, the values are read off its figures, to about 2% of the axis.
#
# **What reproduces.** Every model number here is printed by this notebook,
# and `tests/test_tran2010.py` pins those of Figs. 2, 3A-B and 5.
#
# - Fig. 2A: force at 5-30 mM Pi relative to Pi = 0 is 73.5, 58.1, 48.1, 41.0
#   and 31.6% (paper about 72, 57, 49, 42 and 33).
# - Fig. 2B: Ca50 at pH 7.5, 7.2, 6.8 and 6.4 relative to pH 7 is 42, 68, 151
#   and 363% (paper about 48, 70, 140 and 355). This is with the true [H+];
#   the CellML's [H+], ten times larger, gives 32, 63, 160 and 409% (see the
#   class docstring).
# - Fig. 2D: maximal force relative to pH 7.2 is 1, 0.90, 0.78 and 0.50 at
#   pH 7.2, 7.0, 6.8 and 6.4 (paper 1, 0.89, 0.77, 0.49), with Ca50 0.71,
#   1.03, 1.56 and 3.74 uM (paper about 0.75, 1.0, 1.5 and 4).
# - Fig. 3A: the force-MgATP curves at 1 nM and 10 uM MgADP cross at
#   0.82 mM MgATP (paper 0.9 mM): below it more MgADP lowers force, above it
#   raises it.
# - Fig. 3B: the steady force after redevelopment at 3 and 10 mM MgATP is
#   0.518 and 0.239 of that at 0.1 mM (paper 0.52 and 0.25), and the force at
#   0.1 mM is still rising at 1.5 s, as in the paper.
# - Fig. 3C: redevelopment is faster at higher MgATP, in the paper's order.
# - Fig. 4: the dynamic modulus rises with frequency to a plateau and falls
#   with Pi, and the phase shift peaks at a few Hz, the peak moving to higher
#   frequency with Pi, as in the paper and Kawai et al.'s data.
# - Fig. 5: maximal force at 0, 2.5, 5, 10 and 15 min of ischemia is 0.851,
#   0.334, 0.193, 0.090 and 0.059 (paper about 0.87, 0.34, 0.20, 0.08 and
#   0.05), and each sensitivity curve has the paper's shape: force falls
#   with Pi and acidosis at every time point, and rises with MgADP at T = 0
#   but barely depends on it from 2.5 min on.
#
# **What does not.**
#
# - Fig. 2C: maximal force against pH is below the paper's curve at its
#   ends, by 11 and 8% of the force at pH 7 (131 and 30 at pH 8 and 6,
#   against about 142 and 38).
# - Fig. 2D: the paper's absolute forces are 1.44 times the model's (1.18
#   against 0.82 at pH 7.2), at every pH alike.
# - Fig. 4: the shape reproduces, the numbers less so. At the paper's
#   amplitude the phase peaks are lower and at higher frequency (29, 33 and
#   37 degrees at 8.0, 5.9 and 3.7 Hz for 16, 8 and 0 mM Pi, against the
#   paper's about 47, 48 and 53 degrees at 5, 3 and 2 Hz; the small-amplitude
#   limit is 33, 37 and 43 degrees); the dynamic modulus is about
#   0.078 at low frequency (paper about 0.04) and plateaus at 0.22, 0.26 and
#   0.33 at 300 Hz (paper about 0.21, 0.24 and 0.29, still rising there).
# - The text's 13% rise of isometric force from 0 to 3 mM MgADP, by which
#   the MgADP dissociation constant was fitted: the model more than doubles
#   (x2.28, printed below), as does the paper's own Fig. 5 (row C, T = 0),
#   whose force falls steeply as MgADP goes to zero.
#
# **How the steady states are computed.** The figures other than 3B-C and 4
# are isometric steady states at fixed metabolite levels, and the class takes
# its metabolites as parameters, so each level is its own model. Running
# every one to steady state would take minutes, so `steady_state` below sets
# the state to its steady state directly: troponin and the crossbridge
# occupancies at the steady state of their (linear) equations, the strains at
# 0 and `x_0`, which is their isometric steady state (and the regulatory units
# outside the overlap, which feed no output, at theirs). This uses the class's
# own rate functions (private methods, which is why it lives here and not in
# the class), and the notebook checks it against the class run to steady
# state with `advance_step`; Fig. 2D is also computed that way.

# %%
import functools

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from crossbridge import Tran2010

#: The paper's maximal activation [uM].
CA_MAX = 200.0


def _save_or_show(fig, fname, save):
    plt.tight_layout()
    if save:
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"  Saved: {fname}")
    else:
        plt.show()
    plt.close(fig)


def _family(n, cmap="viridis"):
    """n colours of one sequential colormap, for traces ordered by a parameter."""
    return plt.colormaps[cmap](np.linspace(0.0, 0.85, n))


def params(**changes):
    """`Tran2010.default_parameters()` with `changes`."""
    return {**Tran2010.default_parameters(), **changes}


# %% [markdown]
# ## Isometric steady states


# %%
def steady_state(model, Ca, SL):
    """
    Set `model` (one cell per entry of `Ca`) to its isometric steady state at
    calcium `Ca` [uM] and sarcomere length `SL` [um].
    """
    Ca = np.broadcast_to(np.asarray(Ca, dtype=float), (model.num_cells,))
    SL = np.full(model.num_cells, float(SL))
    x = model._get_x()
    konCa = model._konT * Ca
    x[0] = konCa / (konCa + model._koffHT)  # TRPNCaH
    x[1] = konCa / (konCa + model._koffLT)  # TRPNCaL
    x[7], x[8] = 0.0, model.p["x_0"]  # xXBprer, xXBpostr
    rates = model._rates(x, SL)  # kn_pT depends on troponin, set above
    x[4:7] = model._blocks(Ca, np.zeros_like(Ca), rates).occ_ss.T  # P, XBprer, XBpostr
    kn, kp = rates["kn_pT"], rates["kp_nT"]  # the units outside the overlap, which feed nothing
    x[2], x[3] = kp / (kn + kp), kn / (kn + kp)  # N_NoXB, P_NoXB
    model._set_x(x)
    model._SL_curr = SL
    return model


def steady_force(Ca=CA_MAX, SL=2.2, **changes):
    """Isometric steady active force, one value per entry of `Ca`."""
    Ca = np.atleast_1d(np.asarray(Ca, dtype=float))
    model = steady_state(Tran2010(Ca.size, params=params(**changes)), Ca, SL)
    return model.get_active_tension()


def fmax(**changes):
    """Maximal (200 uM calcium) isometric force."""
    return float(steady_force(CA_MAX, **changes)[0])


def ca50(**changes):
    """Calcium [uM] at half the force at 10 mM calcium."""
    half = 0.5 * float(steady_force(1e4, **changes)[0])
    return 10 ** brentq(
        lambda e: float(steady_force(10**e, **changes)[0]) - half, -2.0, 3.0, xtol=1e-12
    )


def check_steady_state():
    """The steady state above against the class run 3 s at 1 ms calls."""
    print("Checking the steady state against advance_step...")
    Ca = np.array([0.5, 1.0, 2.0, CA_MAX])
    c = {"TmpC": 30.0, "pH": 6.8, "Pi": 10.0}
    model = Tran2010(Ca.size, params=params(**c))
    for _ in range(3000):
        model.advance_step(1e-3, Ca, 2.1, 0.0)
    exact = steady_force(Ca, SL=2.1, **c)
    print(
        f"  max |force difference| at Ca {Ca} uM: {np.abs(model.get_active_tension() - exact).max():.1e}"
    )


# %% [markdown]
# ## Fig. 2 -- Pi and pH
#
# A: force against Pi at 22 degC, SL 2.2 um and the reference MgATP, MgADP
# and pH. B-D at 30 degC and SL 2.1 um, metabolites at their reference
# levels. B normalizes Ca50 to its value at pH 7, C maximal force to its
# value at pH 7 (the curve passes 100 there, although the caption says pH
# 7.2). D is run with `advance_step`, 2 s at 1 ms calls, one cell per
# calcium level.

# %%
FIG2 = {"TmpC": 30.0, "SL": 2.1}

#: The paper's curves, read off.
PAPER_2A = ([5.0, 10.0, 15.0, 20.0, 30.0], [72, 57, 49, 42, 33])
PAPER_2B = ([7.5, 7.2, 7.0, 6.8, 6.4], [48, 70, 100, 140, 355])
PAPER_2C = ([8.0, 7.0, 6.0], [142, 100, 38])
PAPER_2D = {7.2: 1.18, 7.0: 1.05, 6.8: 0.91, 6.4: 0.58}


def plot_fig2(save=True):
    print("Generating Fig. 2 -- Pi and pH...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # A
    Pi = np.linspace(0.0, 30.0, 61)
    F0 = fmax(TmpC=22.0, Pi=0.0)
    rel = np.array([100 * fmax(TmpC=22.0, Pi=p) / F0 for p in Pi])
    ax = axes[0, 0]
    ax.plot(Pi, rel, "k", label="Tran2010")
    ax.plot(*PAPER_2A, "o", mfc="none", color="tab:red", label="paper (read off)")
    ax.set(
        xlabel="[Pi] [mM]",
        ylabel="relative force [%]",
        ylim=(0, 100),
        title="A: force against Pi (22 degC, SL 2.2 um)",
    )
    print(
        "  A: force relative to Pi = 0 at "
        + ", ".join(f"{p:g} mM {100 * fmax(TmpC=22.0, Pi=p) / F0:.1f}%" for p in PAPER_2A[0])
    )

    # B
    pH = np.linspace(7.5, 6.0, 31)
    ref = ca50(pH=7.0, **FIG2)
    ax = axes[0, 1]
    ax.plot(pH, [100 * ca50(pH=h, **FIG2) / ref for h in pH], "k", label="Tran2010, [H+]")
    cellml = {**FIG2, "pH_ref": 6.15}  # the CellML's H, 10 [H+]: pH and pH_ref one unit lower
    ref_c = ca50(pH=6.0, **cellml)
    ax.plot(
        pH,
        [100 * ca50(pH=h - 1.0, **cellml) / ref_c for h in pH],
        ":",
        color="gray",
        label="with the CellML's H = 10 [H+]",
    )
    ax.plot(*PAPER_2B, "o", mfc="none", color="tab:red", label="paper (read off)")
    ax.set(
        xlabel="pH",
        ylabel="Ca50 relative to pH 7 [%]",
        xlim=(7.5, 6.0),
        ylim=(50, 400),
        title="B: calcium sensitivity against pH",
    )
    print(
        "  B: Ca50 relative to pH 7 at "
        + ", ".join(f"pH {h} {100 * ca50(pH=h, **FIG2) / ref:.0f}%" for h in (7.5, 7.2, 6.8, 6.4))
    )
    print(
        "     with the CellML's H: "
        + ", ".join(
            f"pH {h} {100 * ca50(pH=h - 1.0, **cellml) / ref_c:.0f}%" for h in (7.5, 7.2, 6.8, 6.4)
        )
    )

    # C
    pH = np.linspace(8.0, 4.0, 81)
    f7 = fmax(pH=7.0, **FIG2)
    ax = axes[1, 0]
    ax.plot(pH, [100 * fmax(pH=h, **FIG2) / f7 for h in pH], "k", label="Tran2010")
    ax.plot(*PAPER_2C, "o", mfc="none", color="tab:red", label="paper (read off)")
    ax.set(
        xlabel="pH",
        ylabel="maximal force relative to pH 7 [%]",
        xlim=(8.0, 4.0),
        ylim=(0, 150),
        title="C: maximal force against pH",
    )
    print(
        "  C: maximal force relative to pH 7 at "
        + ", ".join(f"pH {h:g} {100 * fmax(pH=h, **FIG2) / f7:.1f}%" for h in (8.0, 7.2, 6.0, 5.0))
    )

    # D, with advance_step
    Ca = np.logspace(-1.0, 2.0, 61)  # [uM]
    ax = axes[1, 1]
    f72 = None
    for color, (h, paper) in zip(_family(4, "plasma"), PAPER_2D.items()):
        model = Tran2010(Ca.size, params=params(pH=h, TmpC=FIG2["TmpC"]))
        for _ in range(2000):
            model.advance_step(1e-3, Ca, FIG2["SL"], 0.0)
        F = model.get_active_tension()
        f72 = F[-1] if f72 is None else f72
        ax.semilogx(Ca * 1e-6, F, color=color, label=f"pH {h}")
        ax.axhline(paper, color=color, ls=":", lw=1)
        print(
            f"  D: pH {h}: maximal force {F[-1]:.3f} ({F[-1] / f72:.3f} of pH 7.2; paper {paper}, {paper / PAPER_2D[7.2]:.2f}), Ca50 {ca50(pH=h, **FIG2):.3f} uM"
        )
    ax.plot([], [], ":", color="gray", label="paper's plateaus (read off)")
    ax.set(
        xlabel=r"[Ca$^{2+}$] [M]",
        ylabel="force",
        xlim=(1e-7, 1e-4),
        ylim=(0, 1.3),
        title="D: force-calcium curves (2 s of advance_step)",
    )

    for ax in axes.flat:
        ax.legend(fontsize="x-small")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Fig. 2: Pi and pH (Tran2010), cf. paper Fig. 2")
    _save_or_show(fig, "tran2010_fig2_pi_ph.png", save)


# %% [markdown]
# ## Fig. 3 -- MgATP, MgADP and tension redevelopment
#
# A: maximal force against MgATP at three MgADP levels (Pi 0, pH 7, SL 2.1
# um, 20 degC). The paper scales its curves to its data, which are not
# shipped here, so the curves below are relative to the force at 1 nM MgADP
# and 0.01 mM MgATP.
#
# B-C: the muscle is held at its steady state (here reached by 5 s of
# `advance_step`), and at 300 ms the strongly bound crossbridges are
# detached as the paper does it: for 5 ms the forward rates ``fappT`` and
# ``hfT`` are 1000 times slower and the reverse ones ``gappT`` and ``hbT``
# 1000 times faster. As in the Rice2008 demo's Ktr, `get_state` and
# `set_state` move the state into a second model with those rates for the
# 5 ms and back, and force is read through the first model, whose force
# normalization the changed rates would otherwise alter. MgADP 1 uM, Pi 2
# mM, pH 7, SL 2.1 um, 20 degC.

# %%
FIG3A = {"TmpC": 20.0, "SL": 2.1, "pH": 7.0, "Pi": 0.0}
FIG3B = {"TmpC": 20.0, "pH": 7.0, "MgADP": 0.001, "Pi": 2.0}
SWITCH, RELEASE = 5e-3, 0.3  # [s]


def redevelopment(MgATP, t_end=1.5):
    """Force [normalized] at 1 ms intervals from 0 to `t_end` s, with the detachment at 0.3 s."""
    p = params(MgATP=MgATP, **FIG3B)
    model = Tran2010(1, params=p)
    detached = Tran2010(
        1,
        params={
            **p,
            "fapp": p["fapp"] / 1e3,
            "hf": p["hf"] / 1e3,
            "gapp": p["gapp"] * 1e3,
            "hb": p["hb"] * 1e3,
        },
    )
    for _ in range(5000):  # to the steady state
        model.advance_step(1e-3, CA_MAX, 2.1, 0.0)
    # 1 ms calls, the 5 ms switch in 0.1 ms calls of the detached model, then 1 ms calls again
    calls = (
        [(model, 1e-3)] * round(RELEASE / 1e-3)
        + [(detached, 1e-4)] * round(SWITCH / 1e-4)
        + [(model, 1e-3)] * round((t_end - RELEASE - SWITCH) / 1e-3)
    )
    t, F = [0.0], [model.get_active_tension()[0]]
    for k, (m, dt) in enumerate(calls):
        if m is detached and calls[k - 1][0] is model:
            detached.set_state(model.get_state())
        m.advance_step(dt, CA_MAX, 2.1, 0.0)
        if m is detached:
            model.set_state(detached.get_state())  # back, and to read the force
        t.append(t[-1] + dt)
        F.append(model.get_active_tension()[0])
    return np.array(t), np.array(F)


def plot_fig3(save=True):
    print("Generating Fig. 3 -- MgATP, MgADP and tension redevelopment...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    # A
    atp = np.logspace(-2.0, 1.0, 61)
    adp_levels = {"1 nM": 1e-6, "1 uM": 1e-3, "10 uM": 1e-2}  # [mM]
    ref = fmax(MgATP=0.01, MgADP=1e-6, **FIG3A)
    for (label, adp), ls in zip(adp_levels.items(), ("-", "--", ":")):
        F = [fmax(MgATP=a, MgADP=adp, **FIG3A) / ref for a in atp]
        axes[0].semilogx(atp, F, "k", ls=ls, label=f"[MgADP] = {label}")

    def gap(log_atp):
        a = 10**log_atp
        return fmax(MgATP=a, MgADP=1e-6, **FIG3A) - fmax(MgATP=a, MgADP=1e-2, **FIG3A)

    cross = 10 ** brentq(gap, -1.5, 0.8, xtol=1e-12)
    axes[0].axvline(
        cross, color="tab:red", lw=1, alpha=0.6, label=f"1 nM and 10 uM cross: {cross:.2f} mM"
    )
    print(f"  A: the curves at 1 nM and 10 uM MgADP cross at {cross:.3f} mM MgATP (paper 0.9 mM)")
    print(
        f"     force from 0.01 to 10 mM MgATP at 1 nM MgADP falls to {fmax(MgATP=10.0, MgADP=1e-6, **FIG3A) / ref:.2f} of its value"
    )
    axes[0].set(
        xlabel="[MgATP] [mM]",
        ylabel="force relative to 1 nM MgADP, 0.01 mM MgATP",
        title="A: force against MgATP (Pi 0, pH 7, 20 degC)",
    )

    # B, C
    runs = {a: redevelopment(a) for a in (0.1, 3.0, 10.0)}
    f_ref = runs[0.1][1].max()
    for (a, (t, F)), ls in zip(runs.items(), ("-", "--", ":")):
        axes[1].plot(1e3 * t, F / f_ref, "k", ls=ls, label=f"[MgATP] = {a:g} mM")
        after = t >= RELEASE + SWITCH - 1e-9
        axes[2].plot(1e3 * t[after], F[after] / F.max(), "k", ls=ls, label=f"[MgATP] = {a:g} mM")
        recovered = F[after] / F[after][-1]
        t_half = 1e3 * (t[after][np.argmax(recovered >= 0.5)] - RELEASE)
        print(
            f"  B: {a:5g} mM MgATP: force before {F[round(RELEASE / 1e-3) - 1] / f_ref:.3f}, minimum {F[after].min() / f_ref:.4f}, at 1.5 s {F[-1] / f_ref:.3f}; half of it regained in {t_half:.0f} ms"
        )
    steady = [fmax(MgATP=a, SL=2.1, **FIG3B) for a in (0.1, 3.0, 10.0)]
    print(
        f"     steady force at 3 and 10 mM relative to 0.1 mM: {steady[1] / steady[0]:.3f}, {steady[2] / steady[0]:.3f} (paper 0.52, 0.25)"
    )
    for paper in (0.52, 0.25):
        axes[1].axhline(paper, color="tab:red", ls=":", lw=1)
    axes[1].plot([], [], ":", color="tab:red", label="paper's 3, 10 mM (read off)")
    axes[1].set(
        xlabel="time [ms]",
        ylabel="force relative to 0.1 mM MgATP",
        xlim=(0, 1500),
        ylim=(0, 1.1),
        title="B: redevelopment after detachment at 300 ms",
    )
    axes[2].set(
        xlabel="time [ms]",
        ylabel="force relative to its own maximum",
        xlim=(300, 1200),
        ylim=(0, 1.05),
        title="C: as B, each normalized",
    )

    for ax in axes:
        ax.legend(fontsize="x-small")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Fig. 3: MgATP, MgADP and tension redevelopment (Tran2010), cf. paper Fig. 3")
    _save_or_show(fig, "tran2010_fig3_mgatp.png", save)


# %% [markdown]
# ## Fig. 4 -- Complex stiffness
#
# The paper imposes a sinusoidal length change of 0.25% of L0 = 2.28 um and
# reports the amplitude of the force oscillation, active plus passive (the
# "dynamic modulus", in normalized force), and its phase lead over length.
# MgATP 5 mM, MgADP 0, pH 7.15, 20 degC, maximal activation.
#
# **This amplitude is not small.** It strains each half-sarcomere by 2.9 nm,
# 40% of the crossbridges' mean strain `x_0`, where the strain dependence of
# ``hfT`` (``exp(-5 (x / x_0)^2)``) is far from linear: the class's response
# at 20 Hz is 12% below its small-amplitude limit. So the black curves are
# the class itself, driven by the sinusoid with `advance_step` (100 calls a
# period, the first harmonic fitted over the last two of four periods), at
# 0.5-300 Hz. Below 0.5 Hz this costs minutes and the response is nearly
# linear (within 2% at 0.1-0.5 Hz), so it is not simulated there.
#
# The grey curves are that small-amplitude limit over the whole range: the
# model's `.ode` file (`Tran2010.ODE_FILE`, which `tests/test_ode_files.py`
# pins to the class) linearized about its steady state. With ``y`` the
# states and the length and its rate the inputs, the response to
# ``L0 (1 + eps e^{i w t})`` is ``eps L0 [c (iw - J)^-1 (b_L + iw b_v) +
# d_L]``. The class at 0.0025% amplitude agrees with it to 0.3%. This needs
# gotranx (the `demos` extra).

# %%
L0, AMPLITUDE = 2.28, 0.0025
FIG4 = {"TmpC": 20.0, "MgATP": 5.0, "MgADP": 0.0, "pH": 7.15}


@functools.cache
def _ode():
    """The generated Python module of `Tran2010.ODE_FILE`, and its state names."""
    import gotranx

    ode = gotranx.load_ode(Tran2010.ODE_FILE)
    ns: dict = {}
    exec(gotranx.cli.gotran2py.get_code(ode, format=gotranx.codegen.python.Format.none), ns)
    return ns, [state.name for state in ode.states]


def complex_stiffness(Pi, freqs):
    """Force response [normalized force] to a 0.25% sinusoid at frequencies `freqs` [Hz], linearized."""
    ns, states = _ode()
    p0 = params(Pi=Pi, **FIG4)
    model = steady_state(Tran2010(1, params=p0), CA_MAX, L0)
    y = ns["init_state_values"]()
    for name in states:
        y[ns["state_index"](name)] = getattr(model, name)[0]
    p = ns["init_parameter_values"](
        **{k: v for k, v in p0.items() if k not in ("dt", "Ca0")}, Ca=CA_MAX, SL=L0, dSL=0.0
    )
    assert np.abs(ns["rhs"](0.0, y, p)).max() < 1e-9  # a steady state of the file too
    iTa, iTp = ns["monitor_index"]("Ta"), ns["monitor_index"]("Tp")

    def rhs_and_force(y, p):
        m = ns["monitor_values"](0.0, y, p)
        return ns["rhs"](0.0, y, p), m[iTa] + m[iTp]

    def derivative(perturb, h):
        (f_plus, g_plus), (f_minus, g_minus) = perturb(h), perturb(-h)
        return (f_plus - f_minus) / (2 * h), (g_plus - g_minus) / (2 * h)

    n = len(y)
    J, c = np.zeros((n, n)), np.zeros(n)
    for j in range(n):
        e = np.eye(n)[j]
        J[:, j], c[j] = derivative(
            lambda h: rhs_and_force(y + h * e, p), 1e-7 * max(abs(y[j]), 1e-4)
        )

    def input_derivative(name):
        i = ns["parameter_index"](name)
        return derivative(lambda h: rhs_and_force(y, p + h * np.eye(len(p))[i]), 1e-7)

    (b_L, d_L), (b_v, d_v) = input_derivative("SL"), input_derivative("dSL")
    out = [
        c @ np.linalg.solve(1j * w * np.eye(n) - J, b_L + 1j * w * b_v) + d_L + 1j * w * d_v
        for w in 2 * np.pi * np.asarray(freqs)
    ]
    return AMPLITUDE * L0 * np.array(out)


def simulated_stiffness(Pi, freq, periods=4, per_period=100):
    """The response from the class at `freq` [Hz]: the sinusoid with advance_step, fitted over the last two periods."""
    model = steady_state(Tran2010(1, params=params(Pi=Pi, **FIG4)), CA_MAX, L0)
    dt = 1.0 / (freq * per_period)
    w = 2 * np.pi * freq
    t = np.arange(1, periods * per_period + 1) * dt
    F = []
    for tk in t:
        tm = tk - 0.5 * dt  # length and its rate held over the call at its midpoint
        model.advance_step(
            dt, CA_MAX, L0 * (1 + AMPLITUDE * np.sin(w * tm)), L0 * AMPLITUDE * w * np.cos(w * tm)
        )
        F.append(model.get_active_tension()[0] + model.get_passive_tension()[0])
    last = slice(-2 * per_period, None)
    A = np.column_stack([np.sin(w * t[last]), np.cos(w * t[last]), np.ones(2 * per_period)])
    s, co, _ = np.linalg.lstsq(A, np.array(F)[last], rcond=None)[0]
    return complex(s, co)  # sin-coefficient + i cos-coefficient: modulus and phase lead


def _peak(freqs, H):
    """Phase peak [deg] and its frequency [Hz]: a parabola in log f through the grid's maximum."""
    phase = np.degrees(np.angle(H))
    k = int(np.clip(np.argmax(phase), 1, len(freqs) - 2))
    x, y = np.log(freqs[k - 1 : k + 2]), phase[k - 1 : k + 2]
    a, b, c = np.polyfit(x, y, 2)
    return c - b * b / (4 * a), float(np.exp(-b / (2 * a)))


def plot_fig4(save=True):
    print("Generating Fig. 4 -- complex stiffness...")
    freqs = np.logspace(-2.0, np.log10(300.0), 200)
    simulated = np.logspace(np.log10(0.5), np.log10(300.0), 18)
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 7.5), sharex=True)
    for Pi, ls in {16.0: "-", 8.0: "--", 0.0: ":"}.items():
        H = np.array([simulated_stiffness(Pi, f) for f in simulated])
        lin = complex_stiffness(Pi, freqs)
        axes[0].semilogx(freqs, np.abs(lin), color="0.6", ls=ls, lw=1)
        axes[1].semilogx(freqs, np.degrees(np.angle(lin)), color="0.6", ls=ls, lw=1)
        axes[0].semilogx(simulated, np.abs(H), "k", ls=ls, label=f"[Pi] = {Pi:g} mM")
        axes[1].semilogx(simulated, np.degrees(np.angle(H)), "k", ls=ls, label=f"[Pi] = {Pi:g} mM")
        (p, fp), (pl, fpl) = _peak(simulated, H), _peak(freqs, lin)
        print(
            f"  Pi {Pi:4g} mM: modulus {np.abs(H[0]):.3f} at 0.5 Hz, {np.abs(H[-1]):.3f} at 300 Hz; phase peak {p:.1f} deg at {fp:.1f} Hz"
        )
        print(
            f"               small-amplitude limit: modulus {np.abs(lin[0]):.3f} at 0.01 Hz, {np.abs(lin[-1]):.3f} at 300 Hz; phase peak {pl:.1f} deg at {fpl:.1f} Hz"
        )
    axes[0].plot([], [], color="0.6", lw=1, label="small-amplitude limit (linearized .ode)")
    print("  (paper, read off: modulus about 0.04 below 1 Hz, plateaus about 0.21, 0.24, 0.29;")
    print("   phase peaks about 47 deg at 5 Hz, 48 at 3 Hz, 53 at 2 Hz for 16, 8, 0 mM)")
    axes[0].set(ylabel="dynamic modulus", ylim=(0, 0.4), title="A: dynamic modulus")
    axes[1].set(
        xlabel="frequency [Hz]",
        ylabel="phase shift [degrees]",
        ylim=(0, 75),
        title="B: phase shift",
    )
    for ax in axes:
        ax.legend(fontsize="x-small", loc="upper left")
        ax.grid(True, alpha=0.2)
    fig.suptitle(
        "Fig. 4: complex stiffness, 0.25% of 2.28 um (Tran2010), cf. paper Fig. 4A-B",
        fontsize="medium",
    )
    _save_or_show(fig, "tran2010_fig4_complex_stiffness.png", save)


# %% [markdown]
# ## Fig. 5 -- Force during ischemia
#
# The first column is the ischemic time course, the metabolite levels read
# off the paper's first column at T = 0, 2.5, 5, 10 and 15 min (linear in
# between), and the maximal force they give. The other columns vary one of
# MgATP, Pi, MgADP, pH or calcium at a time, the others held at their levels
# at that time point (the circles). Calcium is 200 uM except in the last
# row. SL 2.2 um, 25 degC.

# %%
TIMES = np.array([0.0, 2.5, 5.0, 10.0, 15.0])  # [min]
#: Levels read off the paper's Fig. 5, first column, at TIMES.
ISCHEMIA = {
    "MgATP": np.array([5.0, 4.5, 4.3, 3.2, 1.6]),  # [mM]
    "Pi": np.array([2.0, 13.0, 15.0, 22.0, 27.0]),  # [mM]
    "MgADP": np.array([0.036, 0.20, 0.33, 0.25, 0.16]),  # [mM]
    "pH": np.array([7.15, 6.75, 6.5, 6.27, 6.15]),
}
PAPER_5E = [0.87, 0.34, 0.20, 0.08, 0.05]
FIG5 = {"TmpC": 25.0, "SL": 2.2}


def ischemic_levels(T):
    return {k: float(np.interp(T, TIMES, v)) for k, v in ISCHEMIA.items()}


def plot_fig5(save=True):
    print("Generating Fig. 5 -- force during ischemia...")
    rows = {
        "MgATP": (np.linspace(0.05, 6.0, 60), "[MgATP] [mM]"),
        "Pi": (np.linspace(0.0, 30.0, 61), "[Pi] [mM]"),
        "MgADP": (np.linspace(0.0, 0.4, 81), "[MgADP] [mM]"),
        "pH": (np.linspace(7.5, 6.0, 61), "pH"),
    }
    fig, axes = plt.subplots(5, 5, figsize=(14, 12))

    # first column: the time courses and the force they give
    T = np.linspace(0.0, 15.0, 61)
    for ax, (name, (_, label)) in zip(axes[:4, 0], rows.items()):
        ax.plot(T, [ischemic_levels(t)[name] for t in T], "k")
        ax.plot(TIMES, ISCHEMIA[name], "o", color="k", ms=4)
        ax.set(ylabel=label)
    force = [fmax(**FIG5, **ischemic_levels(t)) for t in T]
    at_times = [fmax(**FIG5, **ischemic_levels(t)) for t in TIMES]
    axes[4, 0].plot(T, force, "k", label="Tran2010")
    axes[4, 0].plot(TIMES, at_times, "o", color="k", ms=4)
    axes[4, 0].plot(TIMES, PAPER_5E, "o", mfc="none", color="tab:red", label="paper (read off)")
    axes[4, 0].set(xlabel="time [min]", ylabel="normalized force", ylim=(0, 1))
    axes[4, 0].legend(fontsize="xx-small")
    print(
        "  E: maximal force at "
        + ", ".join(f"{t:g} min {f:.3f}" for t, f in zip(TIMES, at_times))
        + f" (paper about {', '.join(map(str, PAPER_5E))})"
    )

    # sensitivity columns
    for col, T0 in enumerate(TIMES[:4], start=1):
        base = ischemic_levels(T0)
        f0 = fmax(**FIG5, **base)
        axes[0, col].set_title(f"T = {T0:g} min")
        for row, (name, (values, label)) in enumerate(rows.items()):
            ax = axes[row, col]
            ax.plot(values, [fmax(**FIG5, **{**base, name: v}) for v in values], "k")
            ax.plot(base[name], f0, "o", color="k", ms=4)
            ax.set(xlabel=label, ylim=(0, 1))
            if name == "pH":
                ax.set_xlim(7.5, 6.0)
        Ca = np.logspace(-1.0, 3.0, 81)
        ax = axes[4, col]
        ax.semilogx(Ca, steady_force(Ca, **FIG5, **base), "k")
        ax.plot(CA_MAX, f0, "o", color="k", ms=4)
        ax.set(xlabel=r"[Ca$^{2+}$] [$\mu$M]", ylim=(0, 1))
        adp = rows["MgADP"][0]
        F_adp = [fmax(**FIG5, **{**base, "MgADP": v}) for v in (adp[0], adp[-1])]
        print(
            f"  T = {T0:4g} min: force {f0:.3f}; MgADP 0 -> 0.4 mM: {F_adp[0]:.3f} -> {F_adp[1]:.3f}; Ca50 {ca50(**FIG5, **base):.2f} uM"
        )
    for ax in axes.flat:
        ax.grid(True, alpha=0.2)
    fig.suptitle(
        "Fig. 5: maximal force during ischemia and its sensitivities (Tran2010), cf. paper Fig. 5"
    )
    _save_or_show(fig, "tran2010_fig5_ischemia.png", save)


# %% [markdown]
# ## The 13% MgADP claim
#
# The paper fitted the MgADP dissociation constant `kdADP` to a 13% rise of
# isometric force from 0 to 3 mM MgADP. With its published value the model
# more than doubles, at Fig. 2A's conditions.


# %%
def report_mgadp():
    ratio = fmax(TmpC=22.0, MgADP=3.0) / fmax(TmpC=22.0, MgADP=1e-9)
    print(f"MgADP 0 -> 3 mM at Fig. 2A's conditions: force x{ratio:.2f} (the text: a 13% rise)")


# %%
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reproduce the figures of Tran et al. (2010).")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save figures to PNG files instead of displaying them interactively.",
    )
    args, _ = parser.parse_known_args()

    print("=" * 60)
    print("Tran2010 (metabolite-sensitive guinea pig myofilament) Figure Reproduction")
    print("=" * 60)

    check_steady_state()
    plot_fig2(save=args.save)
    plot_fig3(save=args.save)
    plot_fig4(save=args.save)
    plot_fig5(save=args.save)
    report_mgadp()

    print("\nDone.")
