# %% [markdown]
# # Reproducing the Tewari et al. (2016) rat myofilament paper
#
# Reproduction of the myofilament results of the paper this package's
# `Tewari2016` model implements:
#
# > S.G. Tewari, S.M. Bugenhagen, K.C. Vinnakota, J.J. Rice, P.M.L. Janssen,
# > D.A. Beard, "Influence of metabolic dysfunction on cardiac mechanics in
# > decompensated hypertrophy and heart failure", Journal of Molecular and
# > Cellular Cardiology, 94, 162-175 (2016).
# > https://doi.org/10.1016/j.yjmcc.2016.04.003
#
# The model couples the strain-dependent, metabolite-sensitive crossbridge
# cycle of Tewari et al. (2015) to Rice et al. (2008)'s calcium activation,
# with force slowing calcium unbinding from troponin. The paper then embeds it
# in a TriSeg heart and a closed-loop circulation (its Figs. 5-7); those
# organ-scale results need a heart model outside this package's scope, so this
# notebook covers the myofilament results:
#
# - **Fig. 2** -- calcium transients at 2-10 Hz and the isometric twitches
#   they drive at SL = 2.2 um.
# - **Fig. 3** -- relaxation time (RT50), developed tension (Tdev) and time to
#   peak (TTP) against sarcomere length at 4 Hz and against frequency at
#   2.2 um, with the paper's own model points and the Janssen et al. (2002)
#   data, both read off its Fig. 3.
# - **Fig. E.1** -- the steady-state force-calcium relation at five sarcomere
#   lengths, against the Ca50 and Hill coefficients of the paper's Table E.1.
# - **Fig. F.1** -- 4 Hz twitches at SL = 1.9-2.2 um (Frank-Starling).
# - **Fig. G.1** -- how MgATP, MgADP, Pi and calcium change the rate of force
#   development, at 20 degC.
# - **Figs. 4 and H.1** -- tension redevelopment (Ktr) after a slack-restretch,
#   at two muscle lengths and three temperatures.
# - **The reference temperature** -- why `T_ref_act` defaults to 17 degC.
#
# **Calcium.** The paper drives its twitches with the transients Janssen et
# al. measured; this notebook uses Marzban et al. (2020)'s fits to them
# (`crossbridge.utils.janssen_calcium`), the closest form available.
#
# **What does and does not reproduce.** Table E.1 is reproduced to within
# 0.16 uM in Ca50, and Fig. E.1's 135% rise of maximal force from 1.85 to
# 2.25 um exactly. At SL = 2.2 um twitch amplitudes match the paper's model to
# within 10% at 4-10 Hz, but come out low at 2 Hz (20 against 39 kPa) and at
# shorter lengths (41 against 46 kPa at 2.1 um, 8 against 15 at 1.9 um), and
# twitches peak and relax about 10 ms sooner. Ktr comes out up to three times
# the paper's. Its equations and Table B.1 are all there is
# to go on -- the code of this version of the model was not published -- so
# these are listed rather than tuned away; see the model's docstring for the
# one reading of the paper they depend on.
#
# This can also be run as a standalone script, saving each figure to a PNG
# file instead of displaying it inline:
# ```bash
# python demo/reproduce_figures_tewari2016.py --save
# ```

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from crossbridge import Tewari2016
from crossbridge.utils import janssen_calcium

#: Fig. 3 of the paper, read off the published figure (so to about +-1 ms or kPa):
#: frequency [Hz] or SL [um] -> (RT50 [ms], Tdev [kPa], TTP [ms]).
PAPER_MODEL_FREQ = {
    2: (30.6, 39.2, 53.2),
    3: (31.7, 51.4, 49.7),
    4: (32.3, 60.1, 44.9),
    5: (30.9, 63.6, 43.4),
    6: (29.0, 64.3, 43.4),
    7: (28.7, 65.4, 39.8),
    8: (25.5, 65.4, 37.7),
    9: (23.9, 64.7, 38.8),
    10: (21.8, 62.9, 36.1),
}
JANSSEN_DATA_FREQ = {
    2: (35.9, 37.4, 50.9),
    3: (33.6, 46.2, 48.0),
    4: (31.2, 55.6, 45.3),
    5: (29.2, 65.4, 43.4),
    6: (28.7, 71.3, 41.7),
    7: (27.4, 74.1, 38.8),
    8: (27.3, 74.5, 37.1),
    9: (26.0, 73.8, 36.5),
    10: (25.9, 67.8, 34.4),
}
PAPER_MODEL_SL = {1.9: (19.8, 15.4, 39.4), 2.0: (22.8, 28.8, 41.9), 2.1: (26.6, 46.2, 44.0), 2.2: (31.8, 61.3, 45.1)}
JANSSEN_DATA_SL = {1.9: (19.2, 8.7, 35.0), 2.0: (22.8, 16.5, 36.9), 2.1: (25.8, 33.3, 41.7), 2.2: (29.9, 54.5, 50.5)}

#: Table E.1: SL [um] -> (Ca50 [uM], Hill coefficient) of the paper's model.
TABLE_E1 = {1.85: (2.0, 7.0), 1.95: (1.8, 8.6), 2.05: (1.5, 10.4), 2.15: (1.2, 11.0), 2.25: (1.0, 11.0)}


def _save_or_show(fig, fname, save):
    plt.tight_layout()
    if save:
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"  Saved: {fname}")
    else:
        plt.show()
    plt.close(fig)


# %% [markdown]
# ## Isometric twitches
#
# Each cell of one vectorized model is a muscle held at its own length and
# paced at its own frequency. Every simulation runs for 1.5 s, at least three
# beats at 2 Hz, by which time the twitch repeats exactly, and the last
# complete beat is analysed: developed tension (peak minus diastolic), time to
# peak from the stimulus, and relaxation from peak to half of it.


# %%
def run_twitches(frequencies, sarcomere_lengths, params=None, duration=1.5, dt=5e-4):
    """Isometric twitches; returns (t within the last beat, Ca, Ta) per cell and the metrics."""
    frequencies = np.asarray(frequencies, dtype=float)
    SL = np.asarray(sarcomere_lengths, dtype=float)
    model = Tewari2016(len(SL), params=params)
    n_steps = round(duration / dt)
    t = (np.arange(n_steps) + 1) * dt
    Ca = np.array([janssen_calcium(t, f) for f in frequencies]).T  # (n_steps, cells)
    Ta = np.zeros_like(Ca)
    for k in range(n_steps):
        model.advance_step(dt, Ca[k], SL, 0.0)
        Ta[k] = model.get_active_tension()

    beats, metrics = [], []
    for i, f in enumerate(frequencies):
        per_beat = round(1.0 / f / dt)
        start = (n_steps // per_beat - 1) * per_beat
        tb, Cb, Tb = t[start : start + per_beat] - t[start] + dt, Ca[start : start + per_beat, i], Ta[start : start + per_beat, i]
        peak = np.argmax(Tb)
        Tdev = Tb[peak] - Tb.min()
        half = peak + np.argmax(Tb[peak:] <= Tb.min() + 0.5 * Tdev)
        beats.append((tb, Cb, Tb))
        metrics.append((1e3 * (tb[half] - tb[peak]), Tdev, 1e3 * tb[peak]))  # RT50, Tdev, TTP
    return beats, np.array(metrics)


# %% [markdown]
# ## Fig. 2 -- Calcium transients and the twitches they drive
#
# Faster pacing raises peak calcium and shortens the transient, so tension
# rises from 2 to 7 Hz and relaxes sooner (the paper's Fig. 2B: 40 kPa at
# 2 Hz, 60-65 kPa at 4-10 Hz).


# %%
def plot_fig2_transients(save=True):
    print("Generating Fig. 2 -- calcium transients and twitches at SL = 2.2 um...")
    freqs = [2, 4, 7, 10]
    beats, _ = run_twitches(freqs, [2.2] * len(freqs))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["tab:green", "tab:blue", "tab:red", "black"]
    for (tb, Cb, Tb), f, c in zip(beats, freqs, colors):
        axes[0].plot(1e3 * tb, Cb, color=c, label=f"{f} Hz")
        axes[1].plot(1e3 * tb, Tb, color=c, label=f"{f} Hz")
    axes[0].set(xlabel="t [ms]", ylabel=r"$[Ca^{2+}]$ [$\mu$M]", xlim=(0, 300), title="A: calcium (Janssen et al. 2002, fitted)")
    axes[1].set(xlabel="t [ms]", ylabel=r"$T_a$ [kPa]", xlim=(0, 150), title="B: isometric tension, SL = 2.2 um")
    for ax in axes:
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Fig. 2: calcium and tension transients (Tewari2016), cf. paper Fig. 2")
    _save_or_show(fig, "tewari2016_fig2_transients.png", save)


# %% [markdown]
# ## Fig. 3 -- Force-frequency-length relations
#
# Against sarcomere length (4 Hz) and pacing frequency (SL = 2.2 um). Lines are
# this implementation; open circles the paper's own model and filled circles
# Janssen et al.'s data, both read off the paper's Fig. 3. Developed tension
# follows the paper's model from 4 Hz up and at 2.2 um, and is low at 2 Hz and
# at shorter lengths; twitches peak and relax about 10 ms sooner.


# %%
def plot_fig3_force_frequency_length(save=True):
    print("Generating Fig. 3 -- RT50, Tdev and TTP against SL and frequency...")
    sls = np.array(sorted(PAPER_MODEL_SL))
    freqs = np.array(sorted(PAPER_MODEL_FREQ))
    _, by_sl = run_twitches([4.0] * len(sls), sls)
    _, by_freq = run_twitches(freqs, [2.2] * len(freqs))

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    labels = [r"RT$_{50}$ [ms]", r"T$_{dev}$ [kPa]", "TTP [ms]"]
    limits = [(0, 40), (0, 100), (0, 60)]
    for row, (x, ours, paper, data, xlabel) in enumerate(
        [
            (sls, by_sl, PAPER_MODEL_SL, JANSSEN_DATA_SL, r"SL [$\mu$m] (4 Hz)"),
            (freqs, by_freq, PAPER_MODEL_FREQ, JANSSEN_DATA_FREQ, "frequency [Hz] (SL = 2.2 um)"),
        ]
    ):
        for col in range(3):
            ax = axes[row, col]
            ax.plot(x, ours[:, col], "-", color="tab:blue", lw=2, label="Tewari2016 (this package)")
            ax.plot(list(paper), [v[col] for v in paper.values()], "o", mfc="white", color="black", label="paper's model")
            ax.plot(list(data), [v[col] for v in data.values()], "o", color="black", label="Janssen et al. data")
            ax.set(xlabel=xlabel, ylabel=labels[col], ylim=limits[col] if row == 0 or col != 1 else (0, 100))
            ax.grid(True, alpha=0.2)
    axes[0, 1].set_ylim(0, 80)
    axes[0, 0].legend(fontsize="x-small")
    fig.suptitle("Fig. 3: force-frequency-length relations (Tewari2016), cf. paper Fig. 3")
    _save_or_show(fig, "tewari2016_fig3_force_frequency_length.png", save)


# %% [markdown]
# ## Fig. E.1 -- Steady-state force-calcium-length relation
#
# Under the conditions of the paper's Fig. E.1 (MgATP 5 mM, no MgADP, no Pi),
# held at each calcium level until steady. Longer sarcomeres raise maximal
# force with thick-filament overlap, by 35% from 1.85 to 2.25 um, and shift
# the curve left, through the overlap-dependent calcium affinity of
# troponin. Each curve is fitted with a Hill function and compared with the
# paper's Table E.1.


# %%
def _hill(c, Fmax, c50, n):
    return Fmax * c**n / (c**n + c50**n)


def steady_force_pca(sls, cas, params, duration=0.5):
    SL, Ca = (a.ravel() for a in np.meshgrid(sls, cas, indexing="ij"))
    model = Tewari2016(len(SL), params=params)
    for _ in range(round(duration / model.dt)):
        model.advance_step(model.dt, Ca, SL, 0.0)
    return model.get_active_tension().reshape(len(sls), len(cas))


def plot_figE1_force_pca(save=True):
    print("Generating Fig. E.1 -- steady-state force-pCa at five sarcomere lengths...")
    sls = np.array(sorted(TABLE_E1))
    cas = np.geomspace(0.1, 10.0, 40)
    tension = steady_force_pca(sls, cas, {"MgATP": 5.0, "MgADP": 0.0, "Pi": 0.0})
    reference = tension[0].max()

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(sls)))
    print("    SL [um]   Ca50 [uM] (paper)   Hill n (paper)")
    for i, sl in enumerate(sls):
        Fmax, c50, n = curve_fit(_hill, cas, tension[i], p0=[tension[i].max(), 1.5, 8.0])[0]
        print(f"    {sl:.2f}      {c50:.2f} ({TABLE_E1[sl][0]})         {n:.1f} ({TABLE_E1[sl][1]})")
        ax.semilogx(cas, 100 * tension[i] / reference, color=colors[i], label=f"SL = {sl:.2f} um: Ca50 {c50:.2f} uM, n {n:.1f}")
    ax.set(xlabel=r"$[Ca^{2+}]$ [$\mu$M]", ylabel="relative force [%]", ylim=(0, 145))
    ax.set_title("Fig. E.1: force-calcium-length (Tewari2016)\ncf. paper Fig. E.1B and Table E.1")
    ax.legend(fontsize="x-small", loc="upper left")
    ax.grid(True, alpha=0.2)
    _save_or_show(fig, "tewari2016_figE1_force_pca.png", save)


# %% [markdown]
# ## Fig. F.1 -- Twitches at four sarcomere lengths
#
# At 4 Hz, longer sarcomeres develop more tension and peak later (the paper's
# Fig. F.1B: 15, 29, 46 and 61 kPa at 1.9-2.2 um).


# %%
def plot_figF1_length_dependence(save=True):
    print("Generating Fig. F.1 -- twitches at SL = 1.9-2.2 um, 4 Hz...")
    sls = [1.9, 2.0, 2.1, 2.2]
    beats, _ = run_twitches([4.0] * len(sls), sls)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for (tb, _, Tb), sl, c in zip(beats, sls, ["tab:green", "tab:blue", "tab:red", "black"]):
        ax.plot(1e3 * tb, Tb, color=c, label=f"SL = {sl} um")
    ax.set(xlabel="t [ms]", ylabel=r"$T_a$ [kPa]", xlim=(0, 150), ylim=(0, None))
    ax.set_title("Fig. F.1: length dependence of the 4 Hz twitch (Tewari2016)\ncf. paper Fig. F.1B")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.2)
    _save_or_show(fig, "tewari2016_figF1_length_dependence.png", save)


# %% [markdown]
# ## Fig. G.1 -- Metabolites, calcium and the rate of force development
#
# From rest, calcium is stepped up at t = 0 and tension develops at 20 degC
# and SL = 2.2 um, normalized to its final value. More MgATP speeds crossbridge
# detachment (k3) and so cycling; MgADP competes with it and slows it, as in
# the paper. Pi barely changes the time course from 0.5 to 20 mM: in the
# paper it speeds it slightly, here it slows it slightly. Low calcium develops
# force slowly, limited by the cooperative activation of the thin filament.


# %%
def force_development(conditions, Ca, duration=0.2, dt=5e-4):
    """Normalized tension rise after a calcium step; one cell per condition."""
    n = len(conditions)
    t = (np.arange(round(duration / dt)) + 1) * dt
    tension = np.zeros((len(t), n))
    for i, params in enumerate(conditions):
        model = Tewari2016(1, params={"T": 20.0, **params})
        for k in range(len(t)):
            model.advance_step(dt, Ca[i], 2.2, 0.0)
            tension[k, i] = model.get_active_tension()[0]
    return t, tension / tension[-1]


def plot_figG1_metabolites(save=True):
    print("Generating Fig. G.1 -- metabolites, calcium and force development at 20 degC...")
    panels = [
        ("A: ADP = Pi = 0, Ca 15 uM", "ATP", [{"MgATP": c, "MgADP": 0.0, "Pi": 0.0} for c in (0.1, 0.5, 5.0)], [15.0] * 3, (0.1, 0.5, 5.0), 0.1),
        ("B: ATP 5 mM, Pi = 0, Ca 15 uM", "ADP", [{"MgATP": 5.0, "MgADP": c, "Pi": 0.0} for c in (0.1, 0.5, 2.0)], [15.0] * 3, (0.1, 0.5, 2.0), 0.1),
        ("C: ATP 5 mM, ADP = 0, Ca 15 uM", "Pi", [{"MgATP": 5.0, "MgADP": 0.0, "Pi": c} for c in (0.5, 5.0, 20.0)], [15.0] * 3, (0.5, 5.0, 20.0), 0.1),
        ("D: ATP 5 mM, ADP = Pi = 0", "Ca", [{"MgATP": 5.0, "MgADP": 0.0, "Pi": 0.0}] * 3, [1.0, 2.5, 5.0], (1.0, 2.5, 5.0), 0.2),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, (title, name, conditions, Ca, values, duration) in zip(axes.flat, panels):
        t, tension = force_development(conditions, Ca, duration=duration)
        unit = r"$\mu$M" if name == "Ca" else "mM"
        for i, c in enumerate(["black", "tab:blue", "tab:red"]):
            ax.plot(1e3 * t, tension[:, i], color=c, label=f"{name} = {values[i]} {unit}")
        ax.set(xlabel="t [ms]", ylabel="relative tension", title=title, ylim=(0, 1.05))
        ax.legend(fontsize="small", loc="lower right")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Fig. G.1: rate of force development at 20 degC (Tewari2016), cf. paper Fig. G.1")
    _save_or_show(fig, "tewari2016_figG1_metabolites.png", save)


# %% [markdown]
# ## Figs. 4 and H.1 -- Tension redevelopment after a slack-restretch (Ktr)
#
# The muscle, held at length L0 under constant calcium (MgATP 5 mM, MgADP
# 16 uM, Pi 0.5 mM), is shortened to 0.8 L0 for 10 ms and restretched. As the
# paper describes, the rapid length changes disrupt the attached
# crossbridges; here that is done directly, by detaching them all -- their
# occupancy returned to P, the thin filament untouched -- at the release and at
# the restretch, through `get_state`/`set_state`. Imposing the 0.44 um steps
# themselves would strain attached crossbridges far beyond the range the
# second-order moment closure describes, and at the velocities a series
# element produces (about 1000 um/s) the truncated moment equations grow
# without bound. The rise after the restretch is fitted with
# F = F_max (1 - exp(-Ktr t)) + F_init.
#
# The paper holds calcium at 1 uM; here that gives 5 kPa, so the protocol runs
# at 1.5 uM, where the tension before the release is the paper's ~66 kPa. Ktr
# comes out 78 s^-1 at L0 = 2.2 um, about three times the paper's 27 s^-1, and
# 71 s^-1 at 0.9 L0 against the paper's 49 s^-1: about equal at the two
# lengths instead of faster at the shorter one. It is dominated by the
# cooperative reactivation of the thin filament, and grows from 27 to 37 degC
# with a Q10 of 3.7 rather than the paper's 2.75.

# %%
MOMENTS = ("p1_0", "p1_1", "p1_2", "p2_0", "p2_1", "p2_2", "p3_0", "p3_1", "p3_2")


def detach_all_crossbridges(model):
    state = model.get_state()
    for name in MOMENTS:
        state[name] = np.zeros_like(state[name])
    model.set_state(state)


def ktr_protocol(L0, T, Ca=1.5, dt=2e-4, hold=1.0, after=0.4):
    """Slack-restretch at each L0 (one cell each); returns t, tension and the fitted Ktr."""
    L0 = np.asarray(L0, dtype=float)
    model = Tewari2016(len(L0), params={"T": T, "MgATP": 5.0, "MgADP": 0.016, "Pi": 0.5})
    n_steps = round((hold + 0.010 + after) / dt)
    t = np.arange(n_steps) * dt
    tension = np.zeros((n_steps, len(L0)))
    release, restretch = round(hold / dt), round((hold + 0.010) / dt)
    for k in range(n_steps):
        if k in (release, restretch):
            detach_all_crossbridges(model)
        SL = 0.8 * L0 if release <= k < restretch else L0
        model.advance_step(dt, Ca, SL, 0.0)
        tension[k] = model.get_active_tension()

    def recovery(t, Fmax, k, Finit):
        return Fmax * (1 - np.exp(-k * t)) + Finit

    t_rise = t[restretch:] - t[restretch]
    ktr = [curve_fit(recovery, t_rise, tension[restretch:, i], p0=[tension[-1, i], 50.0, 0.0])[0][1] for i in range(len(L0))]
    return t - hold, tension, np.array(ktr)


def plot_fig4_ktr(save=True):
    print("Generating Figs. 4 and H.1 -- tension redevelopment after slack-restretch...")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    t, tension, ktr = ktr_protocol([2.2, 1.98], T=37.0)
    for i, (label, paper) in enumerate([("L0 = 2.2 um", 27.25), ("0.9 L0 = 1.98 um", 49.06)]):
        axes[0].plot(1e3 * t, tension[:, i], label=f"{label}: Ktr = {ktr[i]:.0f} /s (paper {paper})")
    axes[0].set(xlabel="t from release [ms]", ylabel=r"$T_a$ [kPa]", xlim=(-50, 400), title="Fig. 4B: slack-restretch at 37 degC")

    paper_ktr = {27.0: 10.0, 32.0: 14.0, 37.0: 27.5}
    for T, c in zip(paper_ktr, ["tab:blue", "tab:orange", "tab:red"]):
        t, tension, ktr = ktr_protocol([2.2], T=T)
        rise = t >= 0.010
        axes[1].plot(1e3 * (t[rise] - 0.010), tension[rise, 0] / tension[-1, 0], color=c, label=f"{T:.0f} degC: Ktr = {ktr[0]:.0f} /s (paper {paper_ktr[T]})")
    axes[1].set(xlabel="t from restretch [ms]", ylabel="relative tension", xlim=(0, 400), title="Fig. H.1: temperature, L0 = 2.2 um")
    for ax in axes:
        ax.legend(fontsize="small", loc="lower right")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Figs. 4 and H.1: tension redevelopment, Ca = 1.5 uM (Tewari2016)")
    _save_or_show(fig, "tewari2016_fig4_ktr.png", save)


# %% [markdown]
# ## The reference temperature of the activation rates
#
# Table B.1's crossbridge rates are the 17 degC values of Tewari et al.
# (2015), scaled to 37.5 degC by the table's Q10 values. The paper writes the
# calcium-activation rates with (T - 37.5)/10 instead, which would make them
# 37.5 degC values; read that way (dashed), the model needs about twice the
# calcium for the same force and the 4 Hz twitch nearly vanishes. Reading
# those rates at 17 degC too (solid, the default `T_ref_act`) reproduces the
# paper's Table E.1 and twitches.


# %%
def plot_reference_temperature(save=True):
    print("Generating the reference-temperature comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cas = np.geomspace(0.3, 10.0, 30)
    for T_ref_act, style in ((17.0, "-"), (37.5, "--")):
        params = {"T_ref_act": T_ref_act}
        tension = steady_force_pca(np.array([2.25]), cas, {"MgATP": 5.0, "MgADP": 0.0, "Pi": 0.0, **params})[0]
        axes[0].semilogx(cas, tension, style, color="black", label=f"T_ref_act = {T_ref_act} degC")
        beats, metrics = run_twitches([4.0], [2.2], params=params)
        tb, _, Tb = beats[0]
        axes[1].plot(1e3 * tb, Tb, style, color="black", label=f"T_ref_act = {T_ref_act} degC: {metrics[0, 1]:.0f} kPa")
    axes[0].axvline(TABLE_E1[2.25][0], color="tab:red", lw=0.8, label="paper's Ca50 (Table E.1)")
    axes[0].set(xlabel=r"$[Ca^{2+}]$ [$\mu$M]", ylabel=r"$T_a$ [kPa]", title="steady state, SL = 2.25 um")
    axes[1].axhline(PAPER_MODEL_FREQ[4][1], color="tab:red", lw=0.8, label="paper's Tdev (Fig. 3E)")
    axes[1].set(xlabel="t [ms]", ylabel=r"$T_a$ [kPa]", xlim=(0, 150), title="4 Hz twitch, SL = 2.2 um")
    for ax in axes:
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Reference temperature of Table B.1's activation rates (Tewari2016)")
    _save_or_show(fig, "tewari2016_reference_temperature.png", save)


# %%
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reproduce Tewari et al. (2016) myofilament results.")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save figures to PNG files instead of displaying them interactively.",
    )
    args, _ = parser.parse_known_args()

    print("=" * 60)
    print("Tewari2016 (rat myofilament) Figure Reproduction")
    print("=" * 60)

    plot_fig2_transients(save=args.save)
    plot_fig3_force_frequency_length(save=args.save)
    plot_figE1_force_pca(save=args.save)
    plot_figF1_length_dependence(save=args.save)
    plot_figG1_metabolites(save=args.save)
    plot_fig4_ktr(save=args.save)
    plot_reference_temperature(save=args.save)

    print("\nDone.")
