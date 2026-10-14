# %% [markdown]
# # Reproducing the Marzban, Lopez & Beard (2020) rat myofilament model
#
# Reproduction of the myofilament results of the curated model this package's
# `Marzban2020` implements:
#
# > B. Marzban, R. Lopez, D.A. Beard, "Computational modeling of coupled
# > energetics and mechanics in the rat ventricular myocardium", Physiome
# > (2020). https://doi.org/10.36903/physiome.12964970
#
# the cardiomyocyte model of R. Lopez et al., "Impaired myocardial energetics
# causes mechanical dysfunction in decompensated failing hearts", Function, 1
# (2020) zqaa018. It extends `Tewari2016` (see
# `demo/reproduce_figures_tewari2016.py`) with two mechanisms of Campbell,
# Janssen & Campbell (2018): a cooperative thin filament that calcium switches
# on, and a myosin OFF (super-relaxed) state from which force recruits heads.
#
# The papers embed the model in a TriSeg heart, a circulation and a model of
# mitochondrial energetics, fitted to individual rats; those need models
# outside this package. This notebook covers the myofilament:
#
# - **Fig. 3** -- the paper's fit to Janssen et al. (2002)'s rat trabeculae:
#   calcium transients, developed tension against sarcomere length at 4 Hz,
#   and relaxation time, developed tension and time to peak against
#   frequency, with the paper's model points and the data read off its Fig. 3.
# - **The steady state** -- force-calcium relations at SL = 1.9 and 2.3 um,
#   against the pCa50 and Hill coefficients in the caption of Fig. 3.
# - **The myosin OFF state** -- force recruiting myosin heads during a twitch,
#   and what that adds to length-dependent activation.
# - **Metabolites** -- why the energetic state of a failing heart weakens it:
#   twitches and crossbridge ATP turnover at the metabolite levels of sham
#   and TAC (failing) rat hearts.
# - **Tewari2016 and Marzban2020** -- the two models side by side.
#
# **What does and does not reproduce.** With Pi = 1.0 mM (the default; see
# the model's docstring for why) developed tension follows the paper's model
# to within 8% at 2-10 Hz and 4% at SL = 1.9-2.2 um, RT50 to 4 ms, and the
# steady-state pCa50 to 0.01. Twitches peak 7-9 ms later than the paper's
# model at every frequency.
#
# This can also be run as a standalone script, saving each figure to a PNG
# file instead of displaying it inline:
# ```bash
# python demo/reproduce_figures_marzban2020.py --save
# ```

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from crossbridge import Marzban2020, Tewari2016
from crossbridge.utils import janssen_calcium

#: Fig. 3B of the paper, read off the published figure (so to about +-1 ms or kPa):
#: frequency [Hz] -> (RT50 [ms], Tdev [kPa], TTP [ms]) at SL = 2.2 um, and
#: SL [um] -> Tdev [kPa] at 4 Hz.
PAPER_MODEL_FREQ = {
    2: (41.3, 33.5, 60.0),
    3: (34.6, 44.7, 54.1),
    4: (30.7, 53.8, 48.1),
    5: (25.9, 66.5, 44.8),
    6: (25.0, 67.6, 43.7),
    7: (24.1, 80.0, 39.0),
    8: (22.8, 77.1, 36.8),
    9: (21.8, 73.5, 35.9),
    10: (21.1, 66.5, 34.2),
}
JANSSEN_DATA_FREQ = {
    2: (35.6, 37.5, 51.1),
    3: (32.8, 46.2, 47.8),
    4: (30.7, 56.0, 45.5),
    5: (28.9, 62.9, 43.1),
    6: (28.3, 71.6, 41.6),
    7: (27.5, 74.2, 39.0),
    8: (27.3, 75.3, 36.8),
    9: (25.9, 72.7, 35.9),
    10: (25.2, 66.2, 34.2),
}
PAPER_MODEL_SL = {1.9: 19.1, 2.0: 27.2, 2.1: 38.8, 2.2: 53.9}
JANSSEN_DATA_SL = {1.9: 9.0, 2.0: 16.8, 2.1: 33.6, 2.2: 55.4}

#: Cytosolic metabolite levels [mM]: midpoints of the ranges the paper's Table 12 gives
#: for its sham and TAC (failing) rats.
SHAM = {"MgATP": 8.3, "MgADP": 0.054, "Pi": 1.47}
TAC = {"MgATP": 7.0, "MgADP": 0.041, "Pi": 2.36}


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
# Each cell of one vectorized model is a muscle held at its own length, paced
# at its own frequency by the Janssen et al. calcium transients as the paper
# fitted them (`crossbridge.utils.janssen_calcium`). The last complete beat of
# 1.5 s is analysed: developed tension (peak minus diastolic), time to peak
# from the stimulus, and relaxation from peak to half of it.


# %%
def run_twitches(model_class, frequencies, sarcomere_lengths, params=None, duration=1.5, dt=5e-4):
    """Isometric twitches; per cell the last beat as (t, Ca, Ta, U_NR or None) and the metrics."""
    frequencies = np.asarray(frequencies, dtype=float)
    SL = np.asarray(sarcomere_lengths, dtype=float)
    model = model_class(len(SL), params=params)
    n_steps = round(duration / dt)
    t = (np.arange(n_steps) + 1) * dt
    Ca = np.array([janssen_calcium(t, f) for f in frequencies]).T  # (n_steps, cells)
    Ta, U = np.zeros_like(Ca), np.zeros_like(Ca)
    for k in range(n_steps):
        model.advance_step(dt, Ca[k], SL, 0.0)
        Ta[k] = model.get_active_tension()
        U[k] = getattr(model, "U_NR", np.nan)

    beats, metrics = [], []
    for i, f in enumerate(frequencies):
        per_beat = round(1.0 / f / dt)
        beat = slice((n_steps // per_beat - 1) * per_beat, (n_steps // per_beat) * per_beat)
        tb, Tb = t[beat] - t[beat][0] + dt, Ta[beat, i]
        peak = np.argmax(Tb)
        Tdev = Tb[peak] - Tb.min()
        half = peak + np.argmax(Tb[peak:] <= Tb.min() + 0.5 * Tdev)
        beats.append((tb, Ca[beat, i], Tb, U[beat, i]))
        metrics.append((1e3 * (tb[half] - tb[peak]), Tdev, 1e3 * tb[peak]))  # RT50, Tdev, TTP
    return beats, np.array(metrics)


# %% [markdown]
# ## Fig. 3 -- The fit to Janssen et al.'s rat trabeculae
#
# Lines are this implementation; open circles the paper's model and filled
# circles the data, both read off the paper's Fig. 3B.


# %%
def plot_fig3_janssen_fit(save=True):
    print("Generating Fig. 3 -- calcium transients and the force-frequency-length fit...")
    freqs = np.array(sorted(PAPER_MODEL_FREQ))
    sls = np.array(sorted(PAPER_MODEL_SL))
    _, by_freq = run_twitches(Marzban2020, freqs, [2.2] * len(freqs))
    _, by_sl = run_twitches(Marzban2020, [4.0] * len(sls), sls)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(2, 3, (1, 4))
    t = np.linspace(0, 0.35, 1401)
    for f, c in zip([3, 5, 6, 7, 10], ["tab:green", "tab:blue", "tab:red", "black", "tab:cyan"]):
        beat = t < 1.0 / f
        ax.plot(1e3 * t[beat], janssen_calcium(t[beat], f), color=c, label=f"{f} Hz")
    ax.set(xlabel="t [ms]", ylabel=r"$[Ca^{2+}]$ [$\mu$M]", title="A: calcium transients (fitted)")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.2)

    def panel(position, x, ours, paper, data, xlabel, ylabel, ylim):
        ax = fig.add_subplot(2, 3, position)
        ax.plot(x, ours, "-", color="tab:blue", lw=2, label="Marzban2020 (this package)")
        ax.plot(list(paper), list(paper.values()), "o", mfc="white", color="black", label="paper's model")
        ax.plot(list(data), list(data.values()), "o", color="black", label="Janssen et al. data")
        ax.set(xlabel=xlabel, ylabel=ylabel, ylim=ylim)
        ax.grid(True, alpha=0.2)
        return ax

    def column(table, col):
        return {k: v[col] for k, v in table.items()}

    panel(2, sls, by_sl[:, 1], PAPER_MODEL_SL, JANSSEN_DATA_SL, r"SL [$\mu$m] (4 Hz)", r"T$_{dev}$ [kPa]", (0, 80)).legend(fontsize="x-small")
    panel(3, freqs, by_freq[:, 0], column(PAPER_MODEL_FREQ, 0), column(JANSSEN_DATA_FREQ, 0), "frequency [Hz]", r"RT$_{50}$ [ms]", (0, 45))
    panel(5, freqs, by_freq[:, 1], column(PAPER_MODEL_FREQ, 1), column(JANSSEN_DATA_FREQ, 1), "frequency [Hz]", r"T$_{dev}$ [kPa]", (0, 100))
    panel(6, freqs, by_freq[:, 2], column(PAPER_MODEL_FREQ, 2), column(JANSSEN_DATA_FREQ, 2), "frequency [Hz]", "TTP [ms]", (0, 75))
    fig.suptitle("Fig. 3: fit to Janssen et al. (2002) (Marzban2020, SL = 2.2 um unless given), cf. paper Fig. 3")
    _save_or_show(fig, "marzban2020_fig3_janssen_fit.png", save)


# %% [markdown]
# ## The steady state
#
# Held at each calcium level until steady. The caption of the paper's Fig. 3
# reports "Ca50" = 5.89 and 6.001 with Hill coefficients 4.63 and 4.47 at
# SL = 1.9 and 2.3 um; as calcium concentrations those would need micromolar
# thousands of times above the transients, so they are pCa50 values (Ca50 of
# 1.29 and 1.00 uM). Unlike `Tewari2016`, where longer sarcomeres raise
# troponin's affinity, the length dependence here is mostly in maximal force.


# %%
def plot_steady_state(save=True):
    print("Generating the steady-state force-pCa relation...")
    sls = np.array([1.9, 2.3])
    caption = {1.9: (5.89, 4.63), 2.3: (6.001, 4.47)}
    cas = np.geomspace(0.1, 20.0, 40)
    SL, Ca = (a.ravel() for a in np.meshgrid(sls, cas, indexing="ij"))
    model = Marzban2020(len(SL))
    for _ in range(600):
        model.advance_step(1e-3, Ca, SL, 0.0)
    tension = model.get_active_tension().reshape(len(sls), len(cas))

    fig, ax = plt.subplots(figsize=(6, 5))
    print("    SL [um]   pCa50 (caption)   Hill n (caption)")
    for i, (sl, c) in enumerate(zip(sls, ["tab:blue", "tab:red"])):
        Fmax, c50, n = curve_fit(lambda x, F, c50, n: F * x**n / (x**n + c50**n), cas, tension[i], p0=[tension[i].max(), 1.0, 4.0])[0]
        print(f"    {sl:.1f}       {6 - np.log10(c50):.3f} ({caption[sl][0]})     {n:.2f} ({caption[sl][1]})")
        ax.semilogx(cas, tension[i], color=c, label=f"SL = {sl} um: pCa50 {6 - np.log10(c50):.2f} (paper {caption[sl][0]}), n {n:.1f}")
    ax.set(xlabel=r"$[Ca^{2+}]$ [$\mu$M]", ylabel=r"$T_a$ [kPa]", ylim=(0, None))
    ax.set_title("Steady-state force-calcium relation (Marzban2020)\ncf. paper Fig. 3 caption")
    ax.legend(fontsize="x-small", loc="upper left")
    ax.grid(True, alpha=0.2)
    _save_or_show(fig, "marzban2020_steady_state.png", save)


# %% [markdown]
# ## The myosin OFF state
#
# At rest a fifth of the myosin heads are ON, k_SR / (k_SR + k_mSR); during a
# twitch the force recruits nearly all the rest (left), and more so at longer
# sarcomeres, where there is more force to do it. That feedback is Campbell et
# al. (2018)'s explanation of length-dependent activation: switching it off
# (k_force = 0, right, dashed) leaves the tension small, with heads held at
# their resting fraction, and makes it depend much less on length -- the
# twitch at 2.2 um is 1.7 times that at 1.9 um, against 3 times with it.


# %%
def plot_off_state(save=True):
    print("Generating the OFF-state figure...")
    sls = [1.9, 2.0, 2.1, 2.2]
    colors = ["tab:green", "tab:blue", "tab:red", "black"]
    with_force, m_with = run_twitches(Marzban2020, [4.0] * 4, sls)
    without, m_without = run_twitches(Marzban2020, [4.0] * 4, sls, params={"k_force": 0.0})

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for (tb, _, Tb, Ub), (_, _, Tb0, _), sl, c in zip(with_force, without, sls, colors):
        axes[0].plot(1e3 * tb, Ub, color=c, label=f"SL = {sl} um")
        axes[1].plot(1e3 * tb, Tb, color=c, label=f"SL = {sl} um")
        axes[1].plot(1e3 * tb, Tb0, "--", color=c)
    axes[0].set(xlabel="t [ms]", ylabel=r"$U_{NR}$ (myosin ON)", ylim=(0, 1), title="heads ON during a 4 Hz twitch")
    axes[1].set(xlabel="t [ms]", ylabel=r"$T_a$ [kPa]", title="tension; dashed: k_force = 0")
    axes[2].plot(sls, m_with[:, 1] / m_with[0, 1], "o-", color="black", label="force-dependent recruitment")
    axes[2].plot(sls, m_without[:, 1] / m_without[0, 1], "o--", color="gray", label="k_force = 0")
    axes[2].set(xlabel=r"SL [$\mu$m]", ylabel=r"T$_{dev}$ / T$_{dev}$(1.9 um)", title="length-dependent activation")
    for ax in axes:
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Force-dependent recruitment from the myosin OFF state (Marzban2020)")
    _save_or_show(fig, "marzban2020_off_state.png", save)


# %% [markdown]
# ## Metabolites and the failing heart
#
# Lopez et al.'s central result is that the energetic state of a failing
# heart alone impairs its contraction: TAC rats had less ATP and more Pi.
# At the midpoints of the paper's metabolite ranges for sham and TAC rats
# (Table 12), the 4 Hz twitch loses about half its tension and the
# crossbridges turn over ATP more slowly -- mostly through Pi, which slows
# the A1 -> A2 transition (k1) and speeds detachment back to P (kd).


# %%
def plot_metabolites(save=True):
    print("Generating the metabolite figure...")
    dt, duration, f = 5e-4, 1.0, 4.0
    t = (np.arange(round(duration / dt)) + 1) * dt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label, params, c in (("sham", SHAM, "black"), ("TAC", TAC, "tab:red")):
        model = Marzban2020(1, params=params)
        Ta, J = np.zeros_like(t), np.zeros_like(t)
        for k, tk in enumerate(t):
            model.advance_step(dt, janssen_calcium(tk, f), 2.2, 0.0)
            Ta[k], J[k] = model.get_active_tension()[0], model.get_atp_hydrolysis_rate()[0]
        beat = t >= duration - 1.0 / f
        tb = 1e3 * (t[beat] - t[beat][0])
        text = f"{label}: ATP {params['MgATP']}, ADP {params['MgADP']}, Pi {params['Pi']} mM"
        axes[0].plot(tb, Ta[beat], color=c, label=text)
        axes[1].plot(tb, J[beat], color=c, label=f"{label}: mean {J[beat].mean():.2f} /s")
    axes[0].set(xlabel="t [ms]", ylabel=r"$T_a$ [kPa]", title="4 Hz twitch, SL = 2.2 um")
    axes[1].set(xlabel="t [ms]", ylabel="ATP hydrolysis [1/s per head]", title="crossbridge ATP turnover")
    for ax in axes:
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Metabolite levels of sham and failing (TAC) rat hearts (Marzban2020)")
    _save_or_show(fig, "marzban2020_metabolites.png", save)


# %% [markdown]
# ## Tewari2016 and Marzban2020
#
# The same 4 Hz calcium transient at four sarcomere lengths. Both models
# reach similar tension at 2.2 um; Marzban2020 relaxes more slowly, after its
# refit to the relaxation data, and keeps more tension at short lengths, where
# Tewari2016's troponin affinity drops with filament overlap.


# %%
def plot_comparison(save=True):
    print("Generating the Tewari2016/Marzban2020 comparison...")
    sls = [1.9, 2.0, 2.1, 2.2]
    colors = ["tab:green", "tab:blue", "tab:red", "black"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, cls in zip(axes, (Tewari2016, Marzban2020)):
        beats, _ = run_twitches(cls, [4.0] * len(sls), sls)
        for (tb, _, Tb, _), sl, c in zip(beats, sls, colors):
            ax.plot(1e3 * tb, Tb, color=c, label=f"SL = {sl} um")
        ax.set(xlabel="t [ms]", title=cls.__name__, xlim=(0, 150))
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel(r"$T_a$ [kPa]")
    fig.suptitle("4 Hz twitches in the two models (default parameters)")
    _save_or_show(fig, "marzban2020_vs_tewari2016.png", save)


# %%
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reproduce Marzban et al. (2020) myofilament results.")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save figures to PNG files instead of displaying them interactively.",
    )
    args, _ = parser.parse_known_args()

    print("=" * 60)
    print("Marzban2020 (rat myofilament) Figure Reproduction")
    print("=" * 60)

    plot_fig3_janssen_fit(save=args.save)
    plot_steady_state(save=args.save)
    plot_off_state(save=args.save)
    plot_metabolites(save=args.save)
    plot_comparison(save=args.save)

    print("\nDone.")
