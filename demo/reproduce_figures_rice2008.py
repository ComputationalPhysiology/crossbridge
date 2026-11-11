# %% [markdown]
# # Reproducing the Rice et al. (2008) myofilament model
#
# Reproduction of the model figures of the paper this package's `Rice2008`
# implements:
#
# > J.J. Rice, F. Wang, D.M. Bers, P.P. de Tombe, "Approximate model of
# > cooperative activation and crossbridge cycling in cardiac muscle using
# > ordinary differential equations", Biophysical Journal, 95(5), 2368-2390
# > (2008). https://doi.org/10.1529/biophysj.107.119487
#
# Troponin's calcium binding switches regulatory units between
# non-permissive and permissive through a steep Hill function that stands in
# for their cooperativity, and permissive units recruit crossbridges that
# rotate, carry a mean strain and detach. With a muscle model around it
# (mass, viscosity, passive force, a series spring or an afterload), the
# paper shows the steady states, force-velocity relations, twitches and
# tension redevelopment of rat trabeculae and cells. All are rat, and all
# but Figs. 4 and 8 are at 22.5 degC.
#
# - **Fig. 1B-C** -- the single-overlap fractions and the passive force.
# - **Fig. 3** -- steady-state force-calcium relations: isosarcometric, at a
#   fixed muscle length with a series spring, and the length of an unloaded
#   cell.
# - **Fig. 4** -- quick releases against afterloads, and force-velocity
#   relations at 20, 25 and 30 degC.
# - **Fig. 5** -- isosarcometric twitches at six lengths and six calcium
#   levels.
# - **Fig. 6** -- twitches of an unloaded cell shortening against its passive
#   force.
# - **Fig. 7** -- twitches at a fixed muscle length with series springs of
#   seven stiffnesses.
# - **Fig. 8** -- tension redevelopment (Ktr) against calcium and temperature.
#
# Not covered: the schematics (Figs. 1A, 1D, 2 and 11), and Figs. 9 and 10,
# which couple the model to the Chicago rabbit cell model.
#
# **Protocols.** The paper does not give the sarcomere length, calcium or
# afterloads of Figs. 4 and 8; these come from its Data S1, the authors' own
# code (parameter sets `mshrel` and `isosktr`). Fig. 4A runs at 25 degC, as
# its caption says, not at Data S1's 30 degC. Fig. 5B runs at 2.2 um, as the
# text says, not at its caption's 2.3 um. Where the paper and its CellML
# transcription differ, the class follows the CellML for `kn_p` = 500 and
# `kp_n` = 50 /s and for the sign of Eq. 22, and the paper for the rest
# length of 1.9 um and the collagen force (Eq. 48). Its default temperature
# is 22.5 degC. All of these choices are also Data S1's, apart from the
# collagen force. See `Rice2008`'s docstring. The muscle model below uses
# the CellML's sign for the integral of force, and the paper's preload
# (Eq. 52).
#
# **What reproduces.** Every number below is printed by this notebook.
# "Read off" marks values read off the paper's figures, to about 0.005 force
# units or um, 0.1 um/s, or a few ms. The other paper values are printed in
# the paper.
#
# - Fig. 1B: thin-filament fraction 0.1667 at 1.4 um and 0.6458 at 2.4 um
#   (paper 0.17 and 0.64), thick-filament fraction 1 at 2.3-2.4 um.
# - Fig. 3A: maximal force 0.5319 (SL 1.8 um) to 0.9216 (2.3 um), with pCa50
#   5.8866 to 6.1492 and nH 6.719 to 7.594. The nH at 2.3 um is the paper's
#   dashed 7.6.
# - Fig. 3B: shortening at 10 uM 0.74, 3.41, 6.28, 9.56, 13.02, 16.70 and
#   20.58% for KSE 50 to 1 (paper 0.74, 3.4, 6.3, 9.6, 13, 17, 20). The
#   paper's "total force" is the force at the muscle's end, active plus
#   passive, ``KSE (2.2 - SL) + F_passive(2.2)``. Like its curves, it starts
#   from the preload (0.0382) and plateaus at 0.853 to 0.491 (read off 0.857
#   to 0.493). The Hill fits are to the force above the preload, as in
#   `tests/test_rice2008.py`: pCa50 6.097 and nH 7.16 for KSE 50 (paper 6.1,
#   7.6), and 6.004 and 4.11 for KSE 1 (paper 6.0, 4.0).
# - Fig. 3C: pCa50 5.980 and nH 2.77 (paper 5.9, 3.0), shortest SL 1.4362 um
#   (paper 1.4-1.5).
# - Fig. 4B at 20 and 25 degC: the shortening velocities match the paper's
#   plotted points to within 0.45 um/s (25 degC: 8.34, 4.55, 3.08 um/s at
#   afterloads 0, 0.1, 0.2, read off 8.1, 5.0, 3.2). At 25 degC the Eq. 41 fit
#   is Vmax 8.29 um/s, a 0.175, b 1.49 um/s and k 0.180 (paper 8.03, 0.19,
#   1.6, 0.20).
# - Fig. 5: peaks 0.1340 to 0.8256 (5A) and 0.6930 to 0.0996 (5B), each
#   within 0.004 of the paper's figure (read off 0.135 to 0.825 and 0.69 to
#   0.10). The 50%-50% durations are 0.1273 s (dagger), 0.1574 s (circle)
#   and 0.2150 s (star). They match the paper's plotted inset (about 0.13,
#   0.16 and 0.21 s), not its caption's 0.140, 0.187 and 0.223 s.
# - Fig. 6: shortest SL 1.6330 um at 1.45 uM to 1.8155 um at 0.85 uM (read
#   off 1.633 to 1.815). Peak total force 0.0242 (read off 0.024). From 50%
#   shortening to 50% relaxation takes 0.2477 s at 1.45 uM (paper 0.247) and
#   0.2683 s at 0.85 uM. The paper's inset curve takes about 0.266 s for the
#   latter; its caption's 0.229 s contradicts it.
# - Fig. 7: shortest SL 2.1867 um (KSE 50) to 1.9930 um (KSE 1), against the
#   read-offs 2.188 to 1.993. The caption calls panel B the total force, but
#   its curves are the active force: Data S1 plots that (`force_out`) for
#   this figure, and the model's active peaks, 0.6695 to 0.2413, match the
#   read-offs 0.668 to 0.242. Both forces are plotted here. The KSE 50 twitch
#   is close to the isometric twitch at 2.2 um, as the paper says: it peaks
#   at 0.6695 against 0.6930. The paper's own figures show the same gap
#   (0.668 against 0.69). The largest difference, 0.0570, is on the rise, at
#   0.085 s.
# - Fig. 8: force before the release 0.80933 at 10 uM and 20 degC, as Data
#   S1 gives it. Ktr rises with calcium and temperature. At 20 degC it runs
#   from 5.97 /s at force 0.1057 to 21.01 /s at 0.8093 (read off about 6.5
#   to 22). Over pCa 5.0-6.4 it runs from 2.8 to 8.5 /s at 15 degC (read off
#   about 3.5 to 9) and from 11.3 to 51.9 /s at 25 degC (12.5 to 52).
#
# **What does not.** Fig. 4 has four mismatches.
#
# - At 30 degC the 20-40 ms window misses the paper's fastest velocities: by
#   then the lightest loads have shortened far and are slowing down. At zero
#   afterload the velocity is 12.98 um/s in the window and 23.44 over 5-15 ms
#   (paper 22.7), so Vmax comes out 13.02 against 22.7. The paper's points
#   from afterload 0.3 up are reproduced (4.81, 3.56, 2.61 um/s against the
#   read-offs 5.0, 3.8, 2.9).
# - No single window reproduces all three of the paper's Vmax values. Over
#   5-15 ms, zero afterload gives 14.90 and 18.37 um/s at 20 and 25 degC
#   (paper 3.82 and 8.03).
# - At 20 degC the isometric force at the release (0.7666) is below the
#   afterloads 0.8 and 0.85, so those two lengthen slowly. The Eq. 41 fit
#   then gives b 0.60 against 0.89, a 0.159 against 0.19, and Vmax 3.49
#   against 3.82.
# - Fig. 4A at 25 degC: the zero-afterload trace matches the paper's (SL
#   1.444 against 1.445 um at 1 s). The loaded traces end shorter: 0.5 ends
#   at 1.911 against 1.995 um, and 0.3 at 1.653 against 1.715. Their
#   velocities in the window match Fig. 4B, so the paper's traces slow down
#   more afterwards. The lightly loaded traces also show a shoulder 10-20 ms
#   after the release, where the paper's look smooth.
#
# **Numerics.** The muscle model of Figs. 4, 6 and 7 is coupled to the class
# at 0.1 ms steps. Halving the step moves Fig. 6's shortest SL by at most
# 3.8e-05 um. Every mechanics run stays inside the range over which
# `Rice2008`'s integrator is verified (shortening to 300 um/s per 1 ms call,
# sustained lengthening to 25 um/s). The fastest shortening is 25.6 um/s,
# in the zero-afterload release at 30 degC, and the fastest lengthening is
# 4.96 um/s, in Fig. 7's KSE 1 relengthening.
#
# This can also be run as a standalone script, saving each figure to a PNG
# file instead of displaying it inline:
# ```bash
# python demo/reproduce_figures_rice2008.py --save
# ```

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from crossbridge import Rice2008
from crossbridge.utils import rice_calcium

#: Sarcomere lengths [um] outside which the muscle model holds SL still (the paper's
#: SLmin and SLmax): dSL/dt = 0 unless SL_MIN < SL <= SL_MAX.
SL_MIN, SL_MAX = 1.4, 2.4

#: The peak calcium levels [uM] of Figs. 5B and 6: Table 3's 1.45 uM, scaled down.
AMPLITUDES = np.array([1.45, 1.25, 1.15, 1.05, 0.95, 0.85])

#: Series-spring stiffnesses [normalized force / um] of Figs. 3B and 7.
KSE_VALUES = np.array([50.0, 10.0, 5.0, 3.0, 2.0, 1.4, 1.0])


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


def passive_force(SL, params=None):
    """
    The passive force F_passive(SL) of Eqs. 47-48 (titin, plus collagen above
    2.25 um), from the class: `get_passive_tension` depends on SL alone, so
    one call at that length gives it.
    """
    SL = np.atleast_1d(np.asarray(SL, dtype=float))
    model = Rice2008(SL.size, params=params)
    model.advance_step(1e-3, 0.0, SL, 0.0)
    return model.get_passive_tension()


def _hill(pCa, Fmax, pCa50, nH):
    return Fmax / (1.0 + 10.0 ** (nH * (pCa - pCa50)))


def _fit_hill(pCa, F):
    """Fmax, pCa50 and nH of a Hill curve through (pCa, F)."""
    return curve_fit(_hill, pCa, F, p0=[max(np.max(F), 1e-3), 6.0, 4.0])[0]


def _half_duration(t, y):
    """Time from y first rising through half its peak to its last falling through it, interpolated."""
    half = 0.5 * y.max()
    above = np.flatnonzero(y >= half)
    i, j = above[0], above[-1]
    t_up = np.interp(half, [y[i - 1], y[i]], [t[i - 1], t[i]])
    t_down = np.interp(half, [y[j + 1], y[j]], [t[j + 1], t[j]])
    return t_down - t_up


# %% [markdown]
# ## Fig. 1 -- Sarcomere geometry and passive force
#
# The fraction of the thick filament in the single-overlap zone, where myosin
# heads can bind, rises to 1 at 2.3 um. The thin filament's fraction never
# reaches 1 (Eqs. 42-46, `Rice2008._overlap`). Passive force is titin's,
# reflected about the rest length of 1.9 um so that it restores the rest
# length from either side (the cell). A trabecula adds collagen above
# 2.25 um, which limits it to about 2.3 um (Eqs. 47-49).

# %%
def plot_fig1(save=True):
    print("Generating Fig. 1 -- overlap fractions and passive force...")
    SL = np.linspace(1.4, 2.4, 201)
    thick, thin = Rice2008(1)._overlap(SL)
    trabecula = passive_force(SL)
    cell = passive_force(SL, {"PCon_c": 0.0})

    at = {s: np.argmin(np.abs(SL - s)) for s in (1.4, 2.3, 2.4)}
    print(f"    thin-filament fraction at 1.4 um: {thin[at[1.4]]:.4f} (paper 0.17), at 2.4 um: {thin[at[2.4]]:.4f} (paper 0.64)")
    print(f"    thick-filament fraction at 2.3 and 2.4 um: {thick[at[2.3]]:.4f}, {thick[at[2.4]]:.4f} (paper 1)")
    print(f"    cell passive force at 1.4 and 2.4 um: {cell[at[1.4]]:+.3f}, {cell[at[2.4]]:+.3f} (read off -0.3 and 0.3)")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(SL, thick, color="black", lw=2, label="thick filament")
    axes[0].plot(SL, thin, color="tab:blue", lw=2, label="thin filament")
    axes[0].set(xlabel=r"SL [$\mu$m]", ylabel="single-overlap fraction", ylim=(0, 1.05), title="B: single-overlap fractions")
    axes[1].plot(SL, trabecula, color="tab:blue", lw=2, label="trabecula (titin + collagen)")
    axes[1].plot(SL, cell, color="black", lw=2, label="cell (titin, PCon_c = 0)")
    axes[1].axvline(1.9, color="gray", lw=0.8, ls=":")
    axes[1].set(xlabel=r"SL [$\mu$m]", ylabel="normalized passive force", ylim=(-0.4, 0.8), title="C: passive force (rest length 1.9 um)")
    for ax in axes:
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Fig. 1B-C: sarcomere geometry and passive force (Rice2008), cf. paper Fig. 1")
    _save_or_show(fig, "rice2008_fig1_geometry.png", save)


# %% [markdown]
# ## Fig. 3 -- Steady states
#
# Each cell of one vectorized model is held at its own calcium and sarcomere
# length until steady (1 s at 1 ms calls). Fits are
# ``F = Fmax / (1 + 10 ** (nH (pCa - pCa50)))``.
#
# - **A**, isosarcometric: SL held. Longer sarcomeres raise maximal force
#   with the overlap. They also shift the curve left: near strongly bound
#   crossbridges, troponin binds calcium with higher affinity.
# - **B**, a muscle of fixed length, its sarcomeres held at 2.2 um through a
#   series spring of stiffness KSE. The sarcomeres shorten until
#   ``Ta + Tp(SL) - F_preload = KSE (2.2 - SL)``, where the preload
#   ``F_preload = F_passive(2.2)`` (Eq. 52) holds the muscle at rest. The
#   paper plots the total force at the muscle's end, ``Ta + Tp``. That is
#   ``KSE (2.2 - SL)`` on top of the preload (0.038), from which its curves
#   start. The Hill fits are to the force above the preload. A more
#   compliant spring gives more shortening, less force, and a less steep
#   curve.
# - **C**, an unloaded cell from its rest length of 1.9 um. It shortens until
#   ``Ta + Tp = 0``, and the passive force restores it.
#
# B and C are static force balances: roots in SL of the steady-state force.
# The roots are found in three rounds, as `tests/test_rice2008.py` does:
# cells held at a grid of lengths 0.1 um apart (0.3 s), then at three
# lengths 0.01 um apart around the interpolated root (0.5 s), then at three
# 0.005 um apart (1 s).

# %%
def _held(Ca, SL, params=None, hold=1.0):
    """Ta and Tp after `hold` s at constant Ca [uM] and SL [um], one cell per (broadcast) pair."""
    Ca, SL = np.broadcast_arrays(np.asarray(Ca, dtype=float), np.asarray(SL, dtype=float))
    model = Rice2008(Ca.size, params=params)
    for _ in range(round(hold / 1e-3)):
        model.advance_step(1e-3, Ca.ravel(), SL.ravel(), 0.0)
    return model.get_active_tension().reshape(Ca.shape), model.get_passive_tension().reshape(Ca.shape)


def steady_tension(SLs, Cas, params=None, hold=1.0):
    """Steady-state active tension, shape (len(SLs), len(Cas)): every SL with every Ca, held `hold` s."""
    SL, Ca = np.meshgrid(SLs, Cas, indexing="ij")
    return _held(Ca, SL, params, hold)[0]


def _zero(SL, g):
    """Per row, the zero of g in SL: linear interpolation across its first rise through zero."""
    out = np.empty(SL.shape[0])
    for r, (x, y) in enumerate(zip(SL, g)):
        up = np.flatnonzero((y[:-1] <= 0) & (y[1:] > 0))
        k = up[0] if up.size else int(np.argmin(np.abs(y[:-1]) + np.abs(y[1:])))
        out[r] = x[k] - y[k] * (x[k + 1] - x[k]) / (y[k + 1] - y[k])
    return out


def static_balance(pCa, residuals):
    """
    Per residual ``f(Ta, Tp, SL)``, increasing in SL, the steady-state SL at
    which it vanishes, for every pCa: an array of shape (len(residuals), len(pCa)).
    """
    Ca = 10.0 ** (6.0 - pCa)
    SL = np.broadcast_to(np.round(np.arange(1.4, 2.2 + 1e-9, 0.1), 6), (pCa.size, 9))
    Ta, Tp = _held(Ca[:, None], SL, hold=0.3)
    roots = np.array([_zero(SL, f(Ta, Tp, SL)) for f in residuals])
    for half_width, hold in ((0.01, 0.5), (0.005, 1.0)):
        SL = roots[:, :, None] + half_width * np.array([-1.0, 0.0, 1.0])
        Ta, Tp = _held(Ca[None, :, None], SL, hold=hold)
        roots = np.array([_zero(SL[i], f(Ta[i], Tp[i], SL[i])) for i, f in enumerate(residuals)])
    return roots


def plot_fig3(save=True):
    print("Generating Fig. 3 -- steady states (about a minute)...")
    fig, axes = plt.subplots(3, 1, figsize=(7, 12))
    curve = np.linspace(4.5, 7.0, 400)

    # A: isosarcometric
    sls = np.array([1.8, 1.9, 2.0, 2.1, 2.2, 2.3])
    pCa = np.linspace(4.5, 7.0, 26)
    Ca = 10.0 ** (6.0 - pCa)
    F = steady_tension(sls, Ca)
    print("    A: SL [um]   Fmax     pCa50    nH")
    for sl, Fi, c in zip(sls, F, _family(len(sls))):
        Fmax, pCa50, nH = _fit_hill(pCa, Fi)
        print(f"       {sl:.1f}       {Fmax:.4f}   {pCa50:.4f}   {nH:.3f}")
        axes[0].semilogx(Ca, Fi, "o-", color=c, ms=3, lw=1.2, label=f"SL {sl} um")
    for i in (0, -1):  # the paper's dashed comparison: true Hill curves of nH 7.6
        Fmax, pCa50 = curve_fit(lambda p, Fmax, pCa50: _hill(p, Fmax, pCa50, 7.6), pCa, F[i], p0=[F[i].max(), 6.0])[0]
        axes[0].semilogx(10.0 ** (6.0 - curve), _hill(curve, Fmax, pCa50, 7.6), "--", color="gray", lw=1.2, label="Hill, nH = 7.6" if i == 0 else None)
    print("       (paper: the dashed comparison curves have nH = 7.6)")
    axes[0].set(ylabel="normalized active force", title="A: isosarcometric F-Ca")

    # B and C: static force balances
    preload = passive_force(2.2)[0]
    residuals = [lambda Ta, Tp, SL, K=K: Ta + Tp - preload - K * (2.2 - SL) for K in KSE_VALUES]
    residuals.append(lambda Ta, Tp, SL: Ta + Tp)  # C: an unloaded cell
    pCa = np.linspace(5.0, 6.8, 19)
    Ca = 10.0 ** (6.0 - pCa)
    roots = static_balance(pCa, residuals)

    paper_shortening = [0.74, 3.4, 6.3, 9.6, 13, 17, 20]
    paper_plateau = [0.857, 0.791, 0.733, 0.671, 0.614, 0.555, 0.493]
    paper_fits = {50.0: (6.1, 7.6), 1.0: (6.0, 4.0)}
    print(f"    B: preload F_passive(2.2) = {preload:.4f}")
    print("       KSE     shortening at 10 uM [%] (paper)   total force at 10 uM (read off)   pCa50 (paper)   nH (paper)")
    for K, SL, short, plateau, c in zip(KSE_VALUES, roots[:-1], paper_shortening, paper_plateau, _family(len(KSE_VALUES))):
        force = K * (2.2 - SL)  # above the preload
        Fmax, pCa50, nH = _fit_hill(pCa, force)
        shortening = 100.0 * (2.2 - SL[0]) / 2.2
        line = f"       {K:<5g}   {shortening:5.2f} ({short})                    {force[0] + preload:.3f} ({plateau})"
        if K in paper_fits:
            p50, pnH = paper_fits[K]
            line += f"                    {pCa50:.3f} ({p50})     {nH:.2f} ({pnH})"
            axes[1].semilogx(10.0 ** (6.0 - curve), preload + _hill(curve, Fmax, pCa50, nH), "--", color="gray", lw=1.2)
        print(line)
        axes[1].semilogx(Ca, force + preload, "o-", color=c, ms=3, lw=1.2, label=f"KSE {K:g}: {shortening:.3g}% shortening (paper {short})")
    axes[1].set(ylabel="normalized total force", title="B: fixed muscle length, SL 2.2 um at rest (dashed: Hill fits)")

    SL = roots[-1]
    Fmax, pCa50, nH = _fit_hill(pCa, 1.9 - SL)
    print(f"    C: unloaded cell: pCa50 {pCa50:.3f} (paper 5.9), nH {nH:.2f} (paper 3.0), shortest SL {SL.min():.4f} um (paper 1.4-1.5)")
    axes[2].semilogx(Ca, SL, "o-", color="black", ms=3, lw=1.2, label="unloaded cell")
    axes[2].semilogx(10.0 ** (6.0 - curve), 1.9 - _hill(curve, Fmax, pCa50, nH), "--", color="gray", lw=1.2, label=f"Hill fit: pCa50 {pCa50:.2f}, nH {nH:.2f} (paper 5.9, 3.0)")
    axes[2].set(ylabel=r"SL [$\mu$m]", ylim=(1.4, 2.0), title="C: steady-state SL of an unloaded cell from 1.9 um")

    for ax in axes:
        ax.set(xlabel=r"[Ca$^{2+}$] [$\mu$M]", xlim=(0.3, 10))
        ax.legend(fontsize="x-small", loc="best")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Fig. 3: steady states, 22.5 degC (Rice2008), cf. paper Fig. 3")
    _save_or_show(fig, "rice2008_fig3_steady_states.png", save)


# %% [markdown]
# ## The muscle model
#
# The paper's Figs. 4, 6 and 7 put the myofilaments in a muscle (its Fig.
# 1D). Active and passive force act in parallel with a viscosity and a mass.
# The free end carries a series spring, an afterload, or nothing. The
# equations use the CellML's sign convention. The paper's Eqs. 38-39 and
# 50-51 have the opposite one, which would make active force lengthen the
# sarcomere.
#
# ```text
# dSL/dt   = (intf - viscosity (SL - SLset)) / mass      (0 unless 1.4 < SL <= 2.4)
# dintf/dt = -(F_active + F_passive) + F_afterload + F_preload
# ```
#
# ``F_preload = F_passive(SLset)`` (Eq. 52) holds the resting muscle at its
# set length SLset. ``F_afterload`` takes one of three forms:
# ``KSE (SLset - SL)`` for a series spring (Fig. 7); a constant after a
# release, with SL held before it (Fig. 4); or zero for an unloaded cell
# (Fig. 6). Mass (5e-5) and viscosity (0.003) are the paper's rat values, in
# normalized force times s^2/um and s/um. The muscle model lives here, not in
# the library: there, sarcomere length is an input, as for every model of
# the package.
#
# The coupling is staggered. Each step first advances ``intf`` and SL with
# the forces at its start (semi-implicit Euler: ``intf`` first, then SL with
# the new velocity). It then advances `Rice2008` over the step with the new
# SL and that velocity, taking calcium at the step's midpoint. The activated
# muscle's crossbridges are a stiff spring, about 85 force units per um, so
# the mass oscillates with a period of about 5 ms. The default 0.1 ms step
# resolves that, and the step check under Fig. 6 confirms it. ``F_total`` is
# the total force, ``F_active + F_passive``.

# %%
def simulate_muscle(Ca, t_end, SLset, *, KSE=None, afterload=None, release_time=None, params=None, mass=5e-5, viscosity=0.003, dt=1e-4):
    """
    The paper's muscle model around a `Rice2008`, one muscle per cell.

    Parameters
    ----------
    Ca : callable
        ``Ca(t)`` [uM], a scalar or shape (n,).
    t_end : float
        Duration [s], from t = 0.
    SLset : float or ndarray
        Set (initial) sarcomere length [um].
    KSE : ndarray, shape (n,), optional
        Series-spring stiffness: ``F_afterload = KSE (SLset - SL)``.
    afterload : ndarray, shape (n,), optional
        Constant afterload, applied from `release_time` (default 0); before
        it SL is held at SLset. With neither KSE nor afterload, the cell is
        unloaded.
    params : dict, optional
        `Rice2008` parameters.
    mass, viscosity : float
        The paper's rat values.
    dt : float
        Step [s].

    Returns
    -------
    dict
        ``"t"`` shape (steps + 1,), and ``"SL"``, ``"F_active"`` and
        ``"F_total"`` (active plus passive), shape (steps + 1, n).
    """
    KSE = None if KSE is None else np.asarray(KSE, dtype=float)
    afterload = None if afterload is None else np.asarray(afterload, dtype=float)
    shapes = [np.shape(Ca(0.0)), np.shape(SLset)]
    shapes += [np.shape(x) for x in (KSE, afterload) if x is not None]
    n = int(np.prod(np.broadcast_shapes(*shapes), dtype=int))
    SLset = np.broadcast_to(np.asarray(SLset, dtype=float), (n,))
    model = Rice2008(n, params=params)
    F_preload = passive_force(SLset, params)
    steps = round(t_end / dt)
    release = 0 if release_time is None else round(release_time / dt)

    SL, intf = SLset.copy(), np.zeros(n)
    F_active, F_passive = np.zeros(n), F_preload.copy()  # at rest, at SLset
    out = {key: np.zeros((steps + 1, n)) for key in ("SL", "F_active", "F_total")}
    out["SL"][0], out["F_total"][0] = SL, F_passive
    for k in range(steps):
        if k >= release:
            if KSE is not None:
                F_afterload = KSE * (SLset - SL)
            elif afterload is not None:
                F_afterload = afterload
            else:
                F_afterload = 0.0
            intf = intf + dt * (-(F_active + F_passive) + F_afterload + F_preload)
            v = (intf - viscosity * (SL - SLset)) / mass
            v = np.where((SL > SL_MIN) & (SL <= SL_MAX), v, 0.0)
        else:
            v = np.zeros(n)  # held at SLset until the release
        SL = SL + dt * v
        model.advance_step(dt, Ca((k + 0.5) * dt), SL, v)
        F_active, F_passive = model.get_active_tension(), model.get_passive_tension()
        out["SL"][k + 1] = SL
        out["F_active"][k + 1] = F_active
        out["F_total"][k + 1] = F_active + F_passive
    out["t"] = np.arange(steps + 1) * dt
    return out


def report_velocities(result, label):
    """
    Print a run's fastest shortening and lengthening against the range over
    which `Rice2008`'s integrator is verified (its module docstring:
    shortening to 300 um/s per 1 ms call, sustained lengthening to 25 um/s).
    """
    v = np.diff(result["SL"], axis=0) / np.diff(result["t"])[:, None]
    shortening, lengthening = max(-v.min(), 0.0), max(v.max(), 0.0)
    inside = shortening <= 300.0 and lengthening <= 25.0
    print(f"    {label}: fastest shortening {shortening:.1f} um/s, lengthening {lengthening:.2f} um/s ({'inside' if inside else 'OUTSIDE'} the verified 300 / 25 um/s)")


# %% [markdown]
# ## Fig. 4 -- Quick releases and force-velocity relations
#
# A muscle at 2.2 um is activated by a calcium step to 10 uM at 0.5 s and
# held isometric. At 0.65 s it is released against an afterload of 0 to 0.8
# (by 0.1), or 0.85 (about isometric). These values are Data S1's (`mshrel`);
# the paper gives none. Fig. 4A is at 25 degC, as its caption says. Data S1's
# set has 30 degC, but the paper's zero-afterload trace is the model's at
# 25 degC. Fig. 4B is at 20, 25 and 30 degC.
#
# Right after the release, the attached crossbridges are unloaded in a fast
# transient (the paper's arrow). The muscle then shortens at a roughly
# constant velocity. Velocity here is the mean over 20-40 ms after the
# release; at 20 and 25 degC the transient has died out by then at every
# afterload. The fit is the paper's Eq. 41, ``V = (a Vmax - b F) / (F + a)``,
# with F the afterload.

# %%
LOADS = np.r_[np.arange(0.0, 0.81, 0.1), 0.85]
VELOCITY_WINDOW = (0.020, 0.040)  # [s] after the release
#: Eq. 41's Vmax [um/s], a, b [um/s] and k = b / Vmax, as printed in the paper.
PAPER_FV = {20.0: (3.82, 0.19, 0.89, 0.23), 25.0: (8.03, 0.19, 1.6, 0.20), 30.0: (22.7, 0.16, 3.7, 0.16)}
#: Fig. 4B's points [um/s] at afterloads 0-0.7, read off the figure (to about 0.1 um/s).
PAPER_FV_POINTS = {
    20.0: [3.9, 1.9, 1.35, 1.03, 0.75, 0.52, 0.38, 0.25],
    25.0: [8.1, 5.0, 3.2, 2.27, 1.64, 1.17, 0.87, 0.52],
    30.0: [22.6, 12.6, 7.7, 5.0, 3.8, 2.9, 2.05, 1.3],
}
#: Fig. 4A's sarcomere lengths [um] at 1 s, per afterload, read off the figure.
PAPER_4A_END = [1.445, 1.505, 1.585, 1.715, 1.89, 1.995, 2.07, 2.145, 2.185, 2.2]


def _eq41(F, Vmax, a, b):
    return (a * Vmax - b * F) / (F + a)


def plot_fig4(save=True):
    print("Generating Fig. 4 -- quick releases and force-velocity...")
    release = 0.65

    def calcium(t):  # Data S1's step: 0.1 uM, then 10 uM rising in 8 ms from 0.5 s
        return rice_calcium(t, amplitude=10.0, diastolic=0.1, tau1=8000.0, tau2=0.008, t_start=0.5)

    def mean_velocity(t, SL, start, end):
        i, j = (np.searchsorted(t, release + w - 1e-9) for w in (start, end))
        return (SL[i] - SL[j]) / (t[j] - t[i])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    temperatures = {20.0: "tab:blue", 25.0: "black", 30.0: "tab:red"}
    fits = {}
    for T, color in temperatures.items():
        run = simulate_muscle(calcium, 1.0, 2.2, afterload=LOADS, release_time=release, params={"TmpC": T})
        t, SL = run["t"], run["SL"]
        report_velocities(run, f"{T:g} degC")
        V = mean_velocity(t, SL, *VELOCITY_WINDOW)
        fits[T] = curve_fit(_eq41, LOADS, V, p0=[V[0], 0.2, 1.0])[0]
        F_release = run["F_active"][np.searchsorted(t, release - 1e-9), 0]
        early = mean_velocity(t, SL, 0.005, 0.015)[0]
        print(f"       isometric force at the release {F_release:.4f}; zero-afterload velocity over 5-15 ms {early:.2f} um/s")
        print("       V [um/s] at afterloads 0-0.85: " + ", ".join(f"{v:.2f}" for v in V))
        print("       read off the paper, 0-0.7:      " + ", ".join(f"{v:.2f}" for v in PAPER_FV_POINTS[T]))
        if T == 25.0:
            print("       SL at 1 s [um]:                 " + ", ".join(f"{s:.3f}" for s in SL[-1]))
            print("       read off the paper's Fig. 4A:   " + ", ".join(f"{s:.3f}" for s in PAPER_4A_END))
            for SLi, F, c in zip(SL.T, LOADS, _family(len(LOADS), "viridis_r")):
                axes[0].plot(t, SLi, color=c, lw=1.2)
                axes[0].annotate(f"{F:g}", (1.0, SLi[-1]), xytext=(3, 0), textcoords="offset points", fontsize="x-small", va="center")
        F_curve = np.linspace(0.0, 0.9, 200)
        axes[1].plot(LOADS, V, "o", color=color, label=f"{T:g} degC")
        axes[1].plot(F_curve, _eq41(F_curve, *fits[T]), "--", color=color, lw=1.2, label=f"Eq. 41 fit: Vmax {fits[T][0]:.1f} (paper {PAPER_FV[T][0]})")
    print("    T [degC]   Vmax [um/s] (paper)   a (paper)      b [um/s] (paper)   k = b/Vmax (paper)")
    for T, (Vmax, a, b) in fits.items():
        pV, pa, pb, pk = PAPER_FV[T]
        print(f"    {T:4.0f}       {Vmax:5.2f} ({pV})          {a:.3f} ({pa})    {b:.2f} ({pb})         {b / Vmax:.3f} ({pk})")

    axes[0].axvspan(release + VELOCITY_WINDOW[0], release + VELOCITY_WINDOW[1], color="gray", alpha=0.15, lw=0)
    axes[0].set(xlabel="t [s]", ylabel=r"SL [$\mu$m]", xlim=(0.6, 1.0), ylim=(1.4, 2.3), title="A: release at 0.65 s against afterloads (labelled), 25 degC")
    axes[1].set(xlabel="normalized force (afterload)", ylabel=r"shortening velocity [$\mu$m/s]", xlim=(0, 1), ylim=(-1, 25), title="B: force-velocity, mean over 20-40 ms after release")
    axes[1].legend(fontsize="x-small")
    for ax in axes:
        ax.grid(True, alpha=0.2)
    fig.suptitle("Fig. 4: quick releases and force-velocity relations (Rice2008), cf. paper Fig. 4")
    _save_or_show(fig, "rice2008_fig4_force_velocity.png", save)


# %% [markdown]
# ## Fig. 5 -- Isosarcometric twitches
#
# Table 3's calcium transient (`crossbridge.utils.rice_calcium`, peak
# 1.45 uM at 42 ms) drives cells held at SL 1.8-2.3 um (A). Scaled to lower
# peaks, it drives cells held at 2.2 um (B). 5B is at 2.2 um, as the paper's
# text says: its caption says 2.3 um, but the six plotted peaks are the
# model's at 2.2 um. In both panels, more activation gives more force and a
# longer twitch. Length prolongs the twitch more than calcium does (the
# inset): crossbridges detach faster in shorter sarcomeres (``gappT``, Eq.
# 19). Durations run from 50% activation to 50% relaxation.

# %%
def plot_fig5(save=True):
    print("Generating Fig. 5 -- isosarcometric twitches...")
    sls = np.array([1.8, 1.9, 2.0, 2.1, 2.2, 2.3])
    SL = np.r_[sls, np.full(len(AMPLITUDES), 2.2)]  # A: SL at 1.45 uM; B: amplitude at 2.2 um
    amplitude = np.r_[np.full(len(sls), 1.45), AMPLITUDES]
    model = Rice2008(SL.size)
    t = np.arange(1001) * 1e-3
    F = np.zeros((t.size, SL.size))
    for k in range(t.size - 1):
        model.advance_step(1e-3, rice_calcium((k + 0.5) * 1e-3, amplitude=amplitude), SL, 0.0)
        F[k + 1] = model.get_active_tension()
    A, B = F[:, : len(sls)], F[:, len(sls) :]

    paper_A = [0.135, 0.25, 0.395, 0.545, 0.69, 0.825]
    paper_B = [0.69, 0.57, 0.46, 0.335, 0.20, 0.10]
    print("    A: peak at SL 1.8-2.3 um: " + ", ".join(f"{p:.4f} ({q})" for p, q in zip(A.max(0), paper_A)) + "  (read off)")
    print("    B: peak at 1.45-0.85 uM:  " + ", ".join(f"{p:.4f} ({q})" for p, q in zip(B.max(0), paper_B)) + "  (read off)")
    inset = {"dagger (1.8 um, 1.45 uM)": (A[:, 0], 0.140, 0.13), "circle (2.2 um, 0.85 uM)": (B[:, -1], 0.187, 0.16), "star (2.3 um, 1.45 uM)": (A[:, -1], 0.223, 0.21)}
    print("    50%-50% duration [s]       model    caption   paper's plotted curve")
    for name, (Fi, caption, plotted) in inset.items():
        print(f"    {name:<26} {_half_duration(t, Fi):.4f}   {caption:.3f}     about {plotted:.2f}")
    print("    -> the model matches the paper's plotted curves, not its caption")

    fig, axes = plt.subplots(2, 1, figsize=(7, 8))
    for Fi, sl, c in zip(A.T, sls, _family(len(sls))):
        axes[0].plot(t, Fi, color=c, lw=1.5, label=f"SL {sl} um")
    for Fi, amp, c in zip(B.T, AMPLITUDES, _family(len(AMPLITUDES), "viridis_r")):
        axes[1].plot(t, Fi, color=c, lw=1.5, label=f"{amp} uM")
    ins = axes[0].inset_axes([0.58, 0.45, 0.38, 0.45])
    ins.plot(t, rice_calcium(t), color="black")
    ins.set(xlabel="t [s]", ylabel=r"[Ca$^{2+}$] [$\mu$M]")
    ins = axes[1].inset_axes([0.58, 0.45, 0.38, 0.45])
    for (name, (Fi, _, _)), style in zip(inset.items(), (":", "--", "-")):
        ins.plot(t, Fi / Fi.max(), style, color="black", label=name.split(" (")[0])
    ins.set(xlabel="t [s]", ylabel="self-normalized", xlim=(0, 1))
    ins.legend(fontsize="xx-small")
    axes[0].set(title="A: isosarcometric twitches at SL 1.8-2.3 um, peak Ca 1.45 uM")
    axes[1].set(title="B: at SL 2.2 um, peak Ca 1.45-0.85 uM")
    for ax in axes:
        ax.set(xlabel="t [s]", ylabel="normalized force", xlim=(0, 1), ylim=(0, None))
        ax.legend(fontsize="x-small", loc="upper left")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Fig. 5: isosarcometric twitches, 22.5 degC (Rice2008), cf. paper Fig. 5")
    _save_or_show(fig, "rice2008_fig5_twitches.png", save)


# %% [markdown]
# ## Fig. 6 -- Unloaded cell shortening
#
# A cell starts at its rest length of 1.9 um, with no preload and no
# afterload. Driven by the transients of Fig. 5B, it shortens against its
# passive restoring force, and its total force (active plus passive) stays
# close to zero. The mass makes it ring at the shortest length. The inset
# self-normalizes the shortening (1 = rest length, 0 = shortest) of the
# strongest twitch (star, 1.45 uM) and the weakest (circle, 0.85 uM). The
# step check runs the protocol again at half the step.

# %%
def _fig6_run(dt):
    return simulate_muscle(lambda t: rice_calcium(t, amplitude=AMPLITUDES), 1.0, 1.9, dt=dt)


def plot_fig6(save=True):
    print("Generating Fig. 6 -- unloaded cell shortening...")
    run = _fig6_run(1e-4)
    t, SL = run["t"], run["SL"]
    report_velocities(run, "Fig. 6")
    half_step = _fig6_run(5e-5)["SL"].min(axis=0)
    print(f"    step check: halving dt (1e-4 -> 5e-5 s) moves the shortest SL by at most {np.abs(SL.min(axis=0) - half_step).max():.1e} um")

    paper_min = [1.633, 1.68, 1.71, 1.74, 1.775, 1.815]
    print("    shortest SL [um] at 1.45-0.85 uM: " + ", ".join(f"{s:.4f} ({p})" for s, p in zip(SL.min(axis=0), paper_min)) + "  (read off)")
    print(f"    peak total force at 1.45 uM: {run['F_total'][:, 0].max():.4f} (read off 0.024)")
    shortening = (1.9 - SL) / (1.9 - SL.min(axis=0))
    print("    50% shortening to 50% relaxation [s]   model    caption   paper's inset (read off)")
    for i, name, caption, plotted in ((0, "star (1.45 uM)", 0.247, 0.246), (-1, "circle (0.85 uM)", 0.229, 0.266)):
        print(f"    {name:<38} {_half_duration(t, shortening[:, i]):.4f}   {caption:.3f}     about {plotted:.3f}")
    print("    -> the model matches the paper's inset; the caption's 0.229 s contradicts it")

    fig, axes = plt.subplots(2, 1, figsize=(7, 8))
    for i, (amp, c) in enumerate(zip(AMPLITUDES, _family(len(AMPLITUDES), "viridis_r"))):
        axes[0].plot(t, SL[:, i], color=c, lw=1.5, label=f"{amp} uM")
        axes[1].plot(t, run["F_total"][:, i], color=c, lw=1.5, label=f"{amp} uM")
    ins = axes[0].inset_axes([0.56, 0.24, 0.4, 0.4])
    ins.plot(t, 1.0 - shortening[:, 0], "-", color="black", label="star, 1.45 uM")
    ins.plot(t, 1.0 - shortening[:, -1], "--", color="black", label="circle, 0.85 uM")
    ins.set(xlabel="t [s]", ylabel="SL (self-normalized)", xlim=(0, 1))
    ins.legend(fontsize="xx-small", loc="lower right")
    axes[0].set(ylabel=r"SL [$\mu$m]", ylim=(1.6, 1.95), title="A: sarcomere length of an unloaded cell from 1.9 um")
    axes[1].set(ylabel="normalized total force", ylim=(-0.005, 0.025), title="B: total force (active + passive)")
    for ax in axes:
        ax.set(xlabel="t [s]", xlim=(0, 1))
        ax.legend(fontsize="x-small", loc="upper left" if ax is axes[0] else "upper right")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Fig. 6: cell shortening twitches, 22.5 degC (Rice2008), cf. paper Fig. 6")
    _save_or_show(fig, "rice2008_fig6_cell_shortening.png", save)


# %% [markdown]
# ## Fig. 7 -- Fixed-length twitches with a series spring
#
# The muscle is held at a fixed length, its sarcomeres at 2.2 um at rest. A
# series spring of stiffness KSE lets them shorten as force develops. The
# more compliant the spring, the more they shorten, the lower the force, and
# the later its peak. Relaxation, by contrast, is faster. The spring
# re-lengthens the sarcomeres and strains the attached crossbridges. That
# slows their forward rotation (Eq. 22) and hastens the decline of force.
#
# The paper's caption calls panel B the total force, but its curves are the
# active force. Their peaks are the model's active ones (printed below), and
# Data S1 plots the active force (`force_out`) for this figure. So the solid
# lines here are the active force. The dashed lines are the total force,
# active plus passive. It is higher by the passive force: the preload (0.038)
# at rest, and less as the sarcomeres shorten.

# %%
def plot_fig7(save=True):
    print("Generating Fig. 7 -- fixed-length twitches with a series spring...")
    run = simulate_muscle(rice_calcium, 1.0, 2.2, KSE=KSE_VALUES)
    t, SL = run["t"], run["SL"]
    report_velocities(run, "Fig. 7")
    paper_min = [2.188, 2.145, 2.11, 2.077, 2.05, 2.022, 1.993]
    paper_peak = [0.668, 0.567, 0.475, 0.395, 0.335, 0.285, 0.242]
    print("    KSE    shortest SL [um] (read off)   peak active force (read off)   peak total force")
    for i, K in enumerate(KSE_VALUES):
        print(f"    {K:<5g}  {SL[:, i].min():.4f} ({paper_min[i]})             {run['F_active'][:, i].max():.4f} ({paper_peak[i]})               {run['F_total'][:, i].max():.4f}")

    isometric = Rice2008(1)
    F_iso = np.zeros_like(t)
    for k in range(t.size - 1):
        isometric.advance_step(t[1], rice_calcium((k + 0.5) * t[1]), 2.2, 0.0)
        F_iso[k + 1] = isometric.get_active_tension()[0]
    gap = np.abs(run["F_active"][:, 0] - F_iso)
    print(f"    KSE 50 against the isometric twitch at 2.2 um (Fig. 5B): peak {run['F_active'][:, 0].max():.4f} against {F_iso.max():.4f}; largest difference {gap.max():.4f}, at {t[np.argmax(gap)]:.3f} s")

    fig, axes = plt.subplots(2, 1, figsize=(7, 8))
    for i, (K, c) in enumerate(zip(KSE_VALUES, _family(len(KSE_VALUES)))):
        axes[0].plot(t, SL[:, i], color=c, lw=1.5, label=f"KSE {K:g}")
        axes[1].plot(t, run["F_active"][:, i], color=c, lw=1.5, label=f"KSE {K:g}")
        axes[1].plot(t, run["F_total"][:, i], "--", color=c, lw=0.8)
    axes[1].plot(t, F_iso, ":", color="tab:red", lw=1.5, label="isometric, 2.2 um")
    axes[0].set(ylabel=r"SL [$\mu$m]", ylim=(1.95, 2.25), title="A: sarcomere length")
    axes[1].set(ylabel="normalized force", ylim=(0, 0.8), title="B: active force (solid) and total force (dashed)")
    for ax in axes:
        ax.set(xlabel="t [s]", xlim=(0, 1))
        ax.legend(fontsize="x-small", loc="lower right" if ax is axes[0] else "upper right")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Fig. 7: fixed-length twitches, SL 2.2 um, 22.5 degC (Rice2008), cf. paper Fig. 7")
    _save_or_show(fig, "rice2008_fig7_series_spring.png", save)


# %% [markdown]
# ## Fig. 8 -- Tension redevelopment (Ktr)
#
# A muscle at 2.2 um (Data S1's set `isosktr`; the paper states no length)
# is activated by a calcium step at 0.3 s, and its force is steady by 2 s.
# The paper does not release and restretch it. It removes the strongly bound
# crossbridges directly: for 2 ms the forward rates ``fappT`` and ``hfT`` are
# 200 times slower, and the reverse ones ``gappT`` and ``hbT`` 200 times
# faster. Here `get_state` and `set_state` move the state into a second
# `Rice2008` with those rates for the 2 ms, and back.
#
# Force is read through the first model throughout. The second model
# normalizes force by the maximal occupancies at its own base rates (Eqs.
# 33-34), which the 200-fold changes alter, so its own `get_active_tension`
# would be wrong. That would affect only those 2 ms, which the fit leaves
# out. The fit runs from the end of the 2 ms, t1, to 4 s (the end of Data
# S1's run): ``F0 + (Fss - F0) (1 - exp(-Ktr (t - t1)))``. The paper plots
# Ktr against the force before the release, although its caption says
# against calcium; both are shown.

# %%
KTR_RELEASE, KTR_SWITCH, KTR_END = 2.0, 2e-3, 4.0  # [s]


def ktr(amplitudes, TmpC):
    """Ktr runs, one cell per calcium level [uM]: t, force, the force before the release, and (F0, Fss, Ktr) per cell."""
    amplitudes = np.asarray(amplitudes, dtype=float)
    params = {"TmpC": TmpC}
    model = Rice2008(amplitudes.size, params=params)
    p = model.p
    swapped = Rice2008(amplitudes.size, params={**params, "fapp": p["fapp"] / 200, "hf": p["hf"] / 200, "gapp": p["gapp"] * 200, "hb": p["hb"] * 200})

    def advance(m, t, dt):
        Ca = rice_calcium(t + 0.5 * dt, amplitude=amplitudes, diastolic=0.1, tau1=8000.0, tau2=0.008, t_start=0.3)
        m.advance_step(dt, Ca, 2.2, 0.0)

    # 1 ms calls, then the 2 ms switch in 0.1 ms calls, then 1 ms calls again
    calls = [(model, 1e-3)] * round(KTR_RELEASE / 1e-3) + [(swapped, 1e-4)] * round(KTR_SWITCH / 1e-4) + [(model, 1e-3)] * round((KTR_END - KTR_RELEASE - KTR_SWITCH) / 1e-3)
    t, F = [0.0], [model.get_active_tension()]
    for k, (m, dt) in enumerate(calls):
        if m is swapped and calls[k - 1][0] is model:
            swapped.set_state(model.get_state())
        advance(m, t[-1], dt)
        if m is swapped:
            model.set_state(swapped.get_state())  # back, and to read the force
        t.append(t[-1] + dt)
        F.append(model.get_active_tension())
    t, F = np.array(t), np.array(F)

    before = np.searchsorted(t, KTR_RELEASE - 1e-9)
    t1 = KTR_RELEASE + KTR_SWITCH
    rec = t >= t1 - 1e-9

    def recovery(t, F0, Fss, k):
        return F0 + (Fss - F0) * (1.0 - np.exp(-k * (t - t1)))

    fits = np.array([curve_fit(recovery, t[rec], F[rec, i], p0=[F[rec, i][0], F[-1, i], 10.0])[0] for i in range(amplitudes.size)])
    return t, F, F[before], fits


def plot_fig8(save=True):
    print("Generating Fig. 8 -- tension redevelopment (Ktr, about half a minute)...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    amplitudes = np.array([10.0, 1.0, 0.85, 0.75, 0.65])
    t, F, F_before, fits = ktr(amplitudes, 20.0)
    print(f"    A (20 degC): force before the release at 10 uM {F_before[0]:.5f} (Data S1 0.80933)")
    print("       Ca [uM]   force before   Ktr [1/s]")
    t1 = KTR_RELEASE + KTR_SWITCH
    tf = np.linspace(t1, 2.35, 200)
    for i, (amp, c) in enumerate(zip(amplitudes, _family(len(amplitudes)))):
        F0, Fss, k = fits[i]
        print(f"       {amp:<5g}     {F_before[i]:.4f}         {k:.2f}")
        axes[0].plot(t, F[:, i], color=c, lw=2.5, alpha=0.6)
        axes[0].plot(tf, F0 + (Fss - F0) * (1.0 - np.exp(-k * (tf - t1))), "--", color="black", lw=1)
        axes[0].annotate(f"{amp:g} uM: {k:.1f} /s", (2.35, F[np.searchsorted(t, 2.35), i]), xytext=(3, 0), textcoords="offset points", fontsize="x-small", va="center")
    axes[0].set(xlabel="t [s]", ylabel="normalized force", xlim=(1.95, 2.35), ylim=(0, 0.9), title="A: Ktr at 20 degC (dashed: fits; labels: Ktr)")

    pCa = np.linspace(5.0, 6.4, 15)
    print("    B: T [degC]   force before -> Ktr [1/s], pCa 5.0-6.4 by 0.1")
    for T, color in {15.0: "tab:blue", 20.0: "black", 25.0: "tab:red"}.items():
        _, _, F_before, fits = ktr(10.0 ** (6.0 - pCa), T)
        print(f"       {T:4.0f}       " + ", ".join(f"{f:.3f} -> {k:.1f}" for f, k in zip(F_before, fits[:, 2])))
        axes[1].plot(F_before, fits[:, 2], "o-", color=color, ms=4, label=f"{T:g} degC")
        axes[2].semilogx(10.0 ** (6.0 - pCa), fits[:, 2], "o-", color=color, ms=4, label=f"{T:g} degC")
    print("       (read off the paper's Fig. 8B: about 3.5-9 /s at 15 degC, 6.5-22 at 20, 12.5-52 at 25)")
    axes[1].set(xlabel="normalized force before the release", ylabel="Ktr [1/s]", xlim=(0, 1), ylim=(0, 60), title="B: Ktr against force (the paper's axis)")
    axes[2].set(xlabel=r"[Ca$^{2+}$] [$\mu$M]", ylabel="Ktr [1/s]", ylim=(0, 60), title="B: the same against calcium (its caption's)")
    for ax in axes[1:]:
        ax.legend(fontsize="x-small", loc="upper left")
    for ax in axes:
        ax.grid(True, alpha=0.2)
    fig.suptitle("Fig. 8: tension redevelopment at SL 2.2 um (Rice2008), cf. paper Fig. 8")
    _save_or_show(fig, "rice2008_fig8_ktr.png", save)


# %%
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reproduce the figures of Rice et al. (2008).")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save figures to PNG files instead of displaying them interactively.",
    )
    args, _ = parser.parse_known_args()

    print("=" * 60)
    print("Rice2008 (rat myofilament) Figure Reproduction")
    print("=" * 60)

    plot_fig1(save=args.save)
    plot_fig3(save=args.save)
    plot_fig4(save=args.save)
    plot_fig5(save=args.save)
    plot_fig6(save=args.save)
    plot_fig7(save=args.save)
    plot_fig8(save=args.save)

    print("\nDone.")
