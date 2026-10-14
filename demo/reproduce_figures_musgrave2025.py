# %% [markdown]
# # Reproducing the Musgrave et al. (2025) human atrial model
#
# Reproduction of the results of the model this package's `Musgrave2025`
# implements:
#
# > J.H. Musgrave, J.-C. Han, M.-L. Ward, A.J. Taberner, K. Tran,
# > "Metabolite-sensitive cross-bridge models of human atria reveal the impact
# > of diabetes on muscle mechanoenergetics", The Journal of Physiology, 604
# > (2025) 5316-5341. https://doi.org/10.1113/JP288463
#
# The first crossbridge models fitted to human data: atrial trabeculae from
# patients with and without type 2 diabetes, measured permeabilized at five
# levels of ATP and Pi, and then put to work in isometric twitches and work
# loops, where the crossbridges' ATP use gives the muscle's efficiency.
#
# - **Fig. 2** -- the passive model against the passive complex modulus and
#   stress-length data.
# - **Fig. 3** -- the 1 Hz calcium transients that drive the muscle models.
# - **Fig. 4 and Table 2** -- the crossbridge models against the active
#   complex modulus and steady-state stress under the five metabolite
#   conditions.
# - **Fig. 5** -- isometric twitches, per muscle and per myofilament area.
# - **Figs. 6 and 7** -- work loops and their mechanoenergetics: work,
#   shortening, velocity, power, crossbridge energy and efficiency.
# - **Figs. 8A, 9A and 10A** -- twitches at high Pi and low ATP, and with
#   calcium handling and crossbridges crossed over between the groups.
#
# Not covered: the sensitivity analysis of Fig. 11, and the work loops of
# Figs. 8-10 (B-G), which run Figs. 6-7 at 21 metabolite levels and four
# model pairings.
#
# **What reproduces.** Everything below that the paper reports as a number,
# to its printed precision, except two: the maximum power of shortening falls
# by 73% in the diabetic model here and by 69% in the paper, whose work loops
# hold the afterload with a mass-dashpot controller where this notebook
# solves for it exactly; and at 10 mM Pi the twitch amplitude falls by 28% and
# 26% (non-diabetic, diabetic) where the paper's text has "26% vs. 28%".
#
# **Data.** The experimental points are the averaged human atrial data of
# Musgrave et al. (2025), Cardiovasc Diabetol 24:266, as published on figshare
# under CC BY 4.0 (https://doi.org/10.6084/m9.figshare.28180337), rounded
# here to four decimals.
#
# This can also be run as a standalone script, saving each figure to a PNG
# file instead of displaying it inline:
# ```bash
# python demo/reproduce_figures_musgrave2025.py --save
# ```

# %%
import functools

import numpy as np
import matplotlib.pyplot as plt

from crossbridge import Musgrave2025
from crossbridge.utils import musgrave_calcium

ND = Musgrave2025.default_parameters()
D = Musgrave2025.diabetic_parameters()
GROUPS = {"non-diabetic": (ND, False, "black"), "diabetic": (D, True, "tab:orange")}

#: Table 3's K [kPa/um], the maximally activated crossbridge model's: Fig. 4 and Table 2
#: are that model's, with no non-permissible state N (here km0 = 0).
K_CROSSBRIDGE = {"non-diabetic": 9029.917130064021, "diabetic": 5265.695318949247}

#: Perturbation frequencies [Hz] of the complex modulus (the fits leave out 100 Hz).
FREQS = np.array([0.1778, 0.3162, 0.5623, 1.0, 1.778, 3.162, 5.623, 10.0, 17.78, 31.62, 56.23, 100.0])

#: The metabolite conditions [mM] of the active data, in the order of the arrays below.
CONDITIONS = {
    "baseline (5 ATP, 1 Pi)": (5.0, 1.0),
    "0.1 mM ATP": (0.1, 1.0),
    "1 mM ATP": (1.0, 1.0),
    "0 mM Pi": (5.0, 0.0),
    "10 mM Pi": (5.0, 10.0),
}

#: Steady-state active stress [kPa] at pCa 4.5: data mean, its standard error, and the
#: paper's model (Table 2), per condition.
STRESS = {
    "non-diabetic": ([20.35, 24.92, 21.49, 20.91, 17.04], [1.83, 2.84, 2.70, 2.37, 1.62], [22.0, 27.2, 23.1, 22.6, 17.9]),
    "diabetic": ([16.21, 18.95, 18.97, 17.12, 12.63], [2.12, 2.33, 3.07, 2.25, 1.80], [16.9, 19.1, 18.4, 17.2, 14.4]),
}

#: Active complex modulus [MPa] per condition (rows) and frequency (columns).
ACTIVE_CM = {
    "non-diabetic": np.array([
        [0.1292+0.0015j, 0.1351+0.0096j, 0.1383+0.0057j, 0.1518+0.0092j, 0.1388-0.0108j, 0.1375-0.0118j, 0.1265-0.0083j, 0.1095+0.0223j, 0.1395+0.0886j, 0.2125+0.1150j, 0.2775+0.0934j, 0.3383+0.0887j],
        [0.1404+0.0155j, 0.1725+0.0195j, 0.1686+0.0156j, 0.1809+0.0041j, 0.1725-0.0090j, 0.1482-0.0123j, 0.1363+0.0263j, 0.1621+0.0902j, 0.2416+0.1242j, 0.3123+0.1057j, 0.3544+0.0743j, 0.4028+0.0869j],
        [0.1346+0.0105j, 0.1357+0.0219j, 0.1598-0.0126j, 0.1382-0.0000j, 0.1230+0.0060j, 0.1214-0.0029j, 0.1151+0.0009j, 0.1052+0.0303j, 0.1326+0.0917j, 0.2028+0.1182j, 0.2696+0.0975j, 0.3260+0.0945j],
        [0.1174+0.0248j, 0.1306+0.0204j, 0.1278-0.0026j, 0.1365-0.0044j, 0.1326-0.0099j, 0.1276-0.0117j, 0.1153-0.0048j, 0.1039+0.0282j, 0.1307+0.0848j, 0.1985+0.1175j, 0.2701+0.0965j, 0.3272+0.0998j],
        [0.0988+0.0046j, 0.1121+0.0248j, 0.1202+0.0051j, 0.1248-0.0025j, 0.1148+0.0001j, 0.1162-0.0130j, 0.1076-0.0159j, 0.0882-0.0086j, 0.0764+0.0314j, 0.1059+0.1009j, 0.1918+0.1184j, 0.2715+0.1193j],
    ]),
    "diabetic": np.array([
        [0.0851+0.0143j, 0.0855+0.0127j, 0.0983+0.0087j, 0.1093+0.0001j, 0.1048-0.0058j, 0.0942-0.0104j, 0.0771+0.0003j, 0.0738+0.0423j, 0.1104+0.0887j, 0.1750+0.0989j, 0.2225+0.0762j, 0.2647+0.0797j],
        [0.1051+0.0147j, 0.1112+0.0073j, 0.1279+0.0040j, 0.1198-0.0016j, 0.1104-0.0111j, 0.0959-0.0111j, 0.0711+0.0314j, 0.1090+0.1002j, 0.1955+0.1140j, 0.2467+0.0847j, 0.2684+0.0558j, 0.3128+0.0715j],
        [0.0941+0.0247j, 0.1061+0.0193j, 0.1291+0.0034j, 0.1298+0.0026j, 0.1241-0.0022j, 0.1129-0.0057j, 0.0926+0.0185j, 0.1066+0.0752j, 0.1769+0.1097j, 0.2392+0.0915j, 0.2693+0.0639j, 0.3118+0.0774j],
        [0.0853+0.0161j, 0.0966+0.0034j, 0.0932+0.0010j, 0.1050-0.0053j, 0.0989-0.0068j, 0.0936-0.0087j, 0.0744+0.0104j, 0.0758+0.0583j, 0.1349+0.1036j, 0.2023+0.0954j, 0.2386+0.0663j, 0.2800+0.0745j],
        [0.0805+0.0135j, 0.0909+0.0028j, 0.0951+0.0002j, 0.1001+0.0017j, 0.0924-0.0038j, 0.0948-0.0130j, 0.0782-0.0069j, 0.0694+0.0247j, 0.0939+0.0694j, 0.1434+0.0859j, 0.1855+0.0764j, 0.2340+0.0833j],
    ]),
}  # fmt: skip

#: Passive complex modulus [MPa] at pCa 9, its elastic and viscous standard errors, and
#: passive stress [kPa] (with its standard error) at SL / 2.2 um = 1, 0.95, 0.9, 0.85.
PASSIVE = {
    "non-diabetic": (
        np.array([0.0840+0.0026j, 0.0846+0.0072j, 0.0754-0.0060j, 0.0900+0.0019j, 0.0913+0.0027j, 0.0919+0.0028j, 0.0928+0.0036j, 0.0945+0.0059j, 0.0956+0.0093j, 0.1042+0.0127j, 0.1122+0.0147j, 0.1350+0.0274j]),
        [0.0067, 0.0113, 0.0078, 0.0075, 0.0095, 0.0099, 0.0097, 0.0101, 0.0103, 0.0113, 0.0118, 0.0132],
        [0.0090, 0.0050, 0.0063, 0.0016, 0.0011, 0.0009, 0.0010, 0.0011, 0.0013, 0.0014, 0.0017, 0.0032],
        [4.832, 2.166, 0.868, 0.0],
        [0.482, 0.250, 0.098, 0.0],
    ),
    "diabetic": (
        np.array([0.0632-0.0084j, 0.0637+0.0029j, 0.0545+0.0008j, 0.0624+0.0041j, 0.0632+0.0023j, 0.0609+0.0008j, 0.0630+0.0036j, 0.0626+0.0058j, 0.0654+0.0091j, 0.0724+0.0114j, 0.0800+0.0121j, 0.0964+0.0216j]),
        [0.0093, 0.0101, 0.0073, 0.0066, 0.0089, 0.0076, 0.0079, 0.0081, 0.0088, 0.0098, 0.0099, 0.0126],
        [0.0059, 0.0020, 0.0046, 0.0039, 0.0013, 0.0018, 0.0007, 0.0012, 0.0017, 0.0019, 0.0021, 0.0036],
        [3.423, 1.686, 0.700, 0.0],
        [0.445, 0.245, 0.094, 0.0],
    ),
}  # fmt: skip

R_GAS = 8.314e-3  # [kJ/(mol K)]
RHO_XB = 0.25  # [mol/m^3] crossbridge density (Tran et al. 2017)
L0 = 2.2  # [um] the starting and optimal sarcomere length


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
# A muscle held at SL = 2.2 um and driven by a group's 1 Hz calcium transient
# (`crossbridge.utils.musgrave_calcium`). The model has no state slower than a
# beat, so the second beat is periodic. Duration is the paper's: from the
# stress rising above 5% of the amplitude to its falling back below it.


# %%
def twitch(params, diabetic_calcium, dt=1e-3, beats=2):
    """The last beat of the total stress [kPa] and (diastolic, systolic, amplitude, duration [ms])."""
    model = Musgrave2025(1, params=params)
    n = round(1.0 / dt)
    for _ in range(beats):
        stress = np.zeros(n)
        for k in range(n):
            model.advance_step(dt, musgrave_calcium((k + 0.5) * dt, diabetic_calcium), L0, 0.0)
            stress[k] = (model.get_active_tension() + model.get_passive_tension())[0]
    amplitude = stress.max() - stress.min()
    above = np.flatnonzero(stress - stress.min() > 0.05 * amplitude)
    metrics = (stress.min(), stress.max(), amplitude, 1e3 * dt * (above[-1] - above[0]))
    return (np.arange(n) + 1) * dt, stress, metrics


# %% [markdown]
# ## The complex modulus
#
# The stiffness against small sinusoidal length changes, from the model's
# `.ode` file (`Musgrave2025.ODE_FILE`, which `tests/test_ode_files.py` pins to
# the class) linearized about its steady state: with ``y`` the states and the
# length ``L`` and its rate the inputs, the stress response to
# ``L0 (1 + eps e^{i w t})`` over ``eps`` is
# ``L0 [c (iw - J)^{-1} (b_L + iw b_v) + d_L]``, in MPa.


# %%
@functools.cache
def _ode():
    """The generated Python module of `Musgrave2025.ODE_FILE`, and its state names."""
    import gotranx

    ode = gotranx.load_ode(Musgrave2025.ODE_FILE)
    ns: dict = {}
    exec(gotranx.cli.gotran2py.get_code(ode, format=gotranx.codegen.python.Format.none), ns)
    return ns, [state.name for state in ode.states]


def complex_modulus(params, Ca, output="Ta", freqs=FREQS):
    """Complex modulus [MPa] of `output` ("Ta" or "Tp") at calcium `Ca` [uM], SL = 2.2 um."""
    ns, states = _ode()
    model = Musgrave2025(1, params=params)
    for _ in range(500):  # 0.5 s to the steady state
        model.advance_step(1e-3, Ca, L0, 0.0)
    y = ns["init_state_values"]()
    for name in states:
        y[ns["state_index"](name)] = getattr(model, name)[0]
    ode_params = {k: v for k, v in model.p.items() if k not in ("dt", "Ca0")}
    p = ns["init_parameter_values"](**ode_params, Ca=Ca, SL=L0, dSL=0.0)
    out = ns["monitor_index"](output)

    def rhs_and_output(y, p):
        return ns["rhs"](0.0, y, p), ns["monitor_values"](0.0, y, p)[out]

    def derivative(perturb, h):
        (f_plus, g_plus), (f_minus, g_minus) = perturb(h), perturb(-h)
        return (f_plus - f_minus) / (2 * h), (g_plus - g_minus) / (2 * h)

    n = len(y)
    J, c = np.zeros((n, n)), np.zeros(n)
    for j in range(n):
        e = np.eye(n)[j]
        J[:, j], c[j] = derivative(lambda h: rhs_and_output(y + h * e, p), 1e-7 * max(abs(y[j]), 1e-4))

    def input_derivative(name):
        i = ns["parameter_index"](name)
        return derivative(lambda h: rhs_and_output(y, p + h * np.eye(len(p))[i]), 1e-7)

    (b_L, d_L), (b_v, d_v) = input_derivative("SL"), input_derivative("dSL")
    modulus = []
    for w in 2 * np.pi * np.asarray(freqs):
        response = np.linalg.solve(1j * w * np.eye(n) - J, b_L + 1j * w * b_v)
        modulus.append(L0 / 1000 * (c @ response + d_L + 1j * w * d_v))
    return np.array(modulus)


def crossbridge_model(group, ATP=5.0, Pi=1.0):
    """The maximally activated crossbridge model: no N state (km0 = 0), Table 3's K."""
    params = GROUPS[group][0]
    return {**params, "MgATP": ATP, "Pi": Pi, "km0": 0.0, "K_myo": K_CROSSBRIDGE[group] / params["rho_myo"]}


# %% [markdown]
# ## Fig. 2 -- The passive model
#
# Relaxed (pCa 9), the muscle is a spring and dashpot in series, in parallel
# with an exponential spring: the complex modulus rises only with the
# dashpot, above ~1 Hz, and the steady-state stress-length relation is the
# exponential spring alone.


# %%
def plot_fig2_passive(save=True):
    print("Generating Fig. 2 -- the passive model...")
    fs = np.logspace(-1, 2, 100)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for group, (params, _, color) in GROUPS.items():
        data, em_se, vm_se, fl, fl_se = PASSIVE[group]
        model = complex_modulus(params, 1e-3, "Tp", fs)
        for ax, part, se in ((axes[0], np.real, em_se), (axes[1], np.imag, vm_se)):
            ax.errorbar(FREQS, part(data), se, fmt="o", ms=4, color=color)
            ax.semilogx(fs, part(model), color=color, label=group)
        ratio = np.array([1.0, 0.95, 0.9, 0.85])
        axes[2].errorbar(ratio, fl, fl_se, fmt="o", ms=4, color=color)
        lengths = np.linspace(0.85, 1.0, 50)
        stress = []
        for r in lengths:
            m = Musgrave2025(1, params={**params, "Ca0": 1e-3})
            m.advance_step(1e-3, 1e-3, r * L0, 0.0)
            stress.append(m.get_passive_tension()[0])
        axes[2].plot(lengths, stress, color=color, label=group)
    axes[0].set(xlabel="frequency [Hz]", ylabel="elastic modulus [MPa]", title="A: passive elastic modulus")
    axes[1].set(xlabel="frequency [Hz]", ylabel="viscous modulus [MPa]", title="B: passive viscous modulus")
    axes[2].set(xlabel=r"SL / 2.2 $\mu$m", ylabel="passive stress [kPa]", title="C: passive stress-length")
    for ax in axes:
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Fig. 2: the passive model (lines) and data (points, mean +- SE), cf. paper Fig. 2")
    _save_or_show(fig, "musgrave2025_fig2_passive.png", save)


# %% [markdown]
# ## Fig. 3 -- The calcium transients
#
# The diabetic transient rises and decays more slowly from a higher diastolic
# level, as Jones et al. (2023) measured; the diabetic myofilaments are also
# less sensitive to calcium (Ca50 0.408 against 0.330 uM).


# %%
def plot_fig3_calcium(save=True):
    print("Generating Fig. 3 -- the calcium transients...")
    t = np.linspace(0, 1, 1001)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for group, (params, diabetic, color) in GROUPS.items():
        ax.plot(t, musgrave_calcium(t, diabetic), color=color, lw=2, label=f"{group} (Ca50 {params['Ca50']} uM)")
    ax.set(xlabel="t [s]", ylabel=r"$[Ca^{2+}]$ [$\mu$M]", title="Fig. 3: 1 Hz calcium transients")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.2)
    _save_or_show(fig, "musgrave2025_fig3_calcium.png", save)


# %% [markdown]
# ## Fig. 4 and Table 2 -- The crossbridge models
#
# Maximally activated (pCa 4.5), under the five metabolite conditions. The
# paper fitted 11 parameters per group to these complex moduli (up to 56 Hz)
# and stresses, and reports the root-mean-square misfit of the moduli
# averaged over the conditions: 12.5 kPa (non-diabetic) and 10.6 kPa
# (diabetic). The diabetic moduli sit lower, and their dip and rise shift to
# lower frequencies: slower crossbridges.


# %%
def plot_fig4_crossbridges(save=True):
    print("Generating Fig. 4 and Table 2 -- the crossbridge models...")
    fs = np.logspace(-1, 2, 100)
    colors = ["black", "tab:blue", "tab:red", "tab:green", "tab:purple"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    print("    group          RMSE of the fit [kPa] (paper)")
    for row, group in enumerate(GROUPS):
        misfit = []
        for (label, (ATP, Pi)), data, color in zip(CONDITIONS.items(), ACTIVE_CM[group], colors):
            modulus = complex_modulus(crossbridge_model(group, ATP, Pi), 32.0, "Ta", np.r_[FREQS[:11], fs])
            fitted, curve = modulus[:11], modulus[11:]
            misfit.append(np.sqrt(np.mean(np.abs(data[:11] - fitted) ** 2) / 2))
            for ax, part in zip(axes[row], (np.real, np.imag)):
                ax.semilogx(FREQS, part(data), "o", ms=4, color=color)
                ax.semilogx(fs, part(curve), color=color, label=label)
        paper = {"non-diabetic": 12.5, "diabetic": 10.6}[group]
        print(f"    {group:<14} {1e3 * np.mean(misfit):.2f} ({paper})")
        axes[row, 0].set(ylabel=f"{group}\nelastic modulus [MPa]")
        axes[row, 1].set(ylabel="viscous modulus [MPa]")
    for ax in axes[1]:
        ax.set_xlabel("frequency [Hz]")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.2)
    axes[0, 0].legend(fontsize="small")
    fig.suptitle("Fig. 4: the crossbridge models (lines) against the active complex modulus (points)")
    _save_or_show(fig, "musgrave2025_fig4_complex_modulus.png", save)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    x = np.arange(len(CONDITIONS))
    print("    group          condition                 data   paper's model   this")
    for ax, group in zip(axes, GROUPS):
        mean, se, paper = STRESS[group]
        ours = []
        for i, (label, (ATP, Pi)) in enumerate(CONDITIONS.items()):
            model = Musgrave2025(1, params=crossbridge_model(group, ATP, Pi))
            for _ in range(500):
                model.advance_step(1e-3, 32.0, L0, 0.0)
            ours.append(model.get_active_tension()[0])
            print(f"    {group:<14} {label:<24} {mean[i]:5.1f}   {paper[i]:5.1f}          {ours[i]:5.2f}")
        ax.bar(x - 0.25, mean, 0.25, yerr=se, color="lightgray", label="data (mean +- SE)")
        ax.bar(x, paper, 0.25, color="gray", label="paper's model")
        ax.bar(x + 0.25, ours, 0.25, color=GROUPS[group][2], label="this package")
        ax.set_xticks(x, list(CONDITIONS), rotation=20, fontsize="small")
        ax.set_title(group)
        ax.grid(True, axis="y", alpha=0.2)
    axes[0].set_ylabel("active stress at pCa 4.5 [kPa]")
    axes[0].legend(fontsize="small")
    fig.suptitle("Table 2: steady-state active stress")
    _save_or_show(fig, "musgrave2025_table2_stress.png", save)


# %% [markdown]
# ## Fig. 5 -- Isometric twitches
#
# Per muscle cross-section (A), the diabetic twitch is 41% lower at its peak
# and 23% longer; per myofilament area (B, ``rho_myo = 1``), where the
# diabetic muscles' lower myocyte fraction (0.292 against 0.361) drops out,
# the amplitudes are closer (33% apart, against 45%).


# %%
def plot_fig5_twitches(save=True):
    print("Generating Fig. 5 -- isometric twitches...")
    paper = {"non-diabetic": (20.2, 14.5, 550, 40.0), "diabetic": (12.0, 7.9, 674, 26.9)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    print("    group          systolic [kPa]   amplitude [kPa]   duration [ms]   myocyte amplitude [kPa]   (paper)")
    for group, (params, diabetic, color) in GROUPS.items():
        t, stress, (_, systolic, amplitude, duration) = twitch(params, diabetic)
        _, stress_myo, (_, _, amplitude_myo, _) = twitch({**params, "rho_myo": 1.0}, diabetic)
        axes[0].plot(t, stress, color=color, lw=2, label=group)
        axes[1].plot(t, stress_myo, color=color, lw=2, label=group)
        p = paper[group]
        print(f"    {group:<14} {systolic:5.1f} ({p[0]})      {amplitude:5.1f} ({p[1]})       {duration:4.0f} ({p[2]})      {amplitude_myo:5.1f} ({p[3]})")
    axes[0].set(xlabel="t [s]", ylabel="stress [kPa]", title="A: muscle stress")
    axes[1].set(xlabel="t [s]", ylabel="stress [kPa]", title="B: myocyte stress (rho_myo = 1)")
    for ax in axes:
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Fig. 5: 1 Hz isometric twitches at SL = 2.2 um, cf. paper Fig. 5")
    _save_or_show(fig, "musgrave2025_fig5_twitches.png", save)


# %% [markdown]
# ## Figs. 6 and 7 -- Work loops
#
# Each beat starts isometric at 2.2 um; once the stress reaches the afterload
# the muscle shortens against it, holds its length when it can shorten no
# further, and is restretched to 2.2 um between 0.8 and 0.95 s. The ten
# afterloads span 1.05 times the diastolic stress to 1.01 times the isometric
# peak, and run as the ten cells of one vectorized model. While shortening,
# each step solves for the velocity at which the total stress at its end is
# the afterload, by Newton's method with `get_active_stiffness()` (plus the
# passive stiffnesses) as the derivative. Crossbridge energy per beat is
# ``-dG_ATP`` times the integrated ATP flux
# ``rho_xb rho_myo (k3 C - k-3 A)`` (`get_atp_hydrolysis_rate`), and
# efficiency is work over that energy.


# %%
def work_loops(params, diabetic, n_loads=10, beats=3, dt=1e-3):
    """Per afterload, the last beat's loops and the paper's six metrics."""
    _, isometric, _ = twitch(params, diabetic)
    afterload = np.linspace(1.05 * isometric.min(), isometric.max(), n_loads)
    afterload[-1] *= 1.01  # the isometric loop
    model = Musgrave2025(n_loads, params=params)
    p = model.p
    n = round(1.0 / dt)
    for _ in range(beats):
        SL, v = np.full(n_loads, L0), np.zeros(n_loads)
        phase = np.zeros(n_loads, dtype=int)  # 0 isometric, 1 shortening, 2 held, 3 restretched
        start_restretch = None
        L, F, J = (np.zeros((n, n_loads)) for _ in range(3))
        for k in range(n):
            t_end = (k + 1) * dt
            Ca = musgrave_calcium((k + 0.5) * dt, diabetic)
            if t_end > 0.8:
                if start_restretch is None:
                    start_restretch, phase = SL.copy(), np.where(SL < L0, 3, 2)
                s = np.clip((t_end - 0.8) / 0.15, 0.0, 1.0)
                target = start_restretch + (L0 - start_restretch) * 0.5 * (1.0 - np.cos(np.pi * s))
                v = np.where(phase == 3, (target - SL) / dt, 0.0)
            state = model.get_state()
            model.advance_step(dt, Ca, SL + v * dt, v)
            shortening = phase == 1
            for _ in range(10 if shortening.any() else 0):
                residual = np.where(shortening, model.get_active_tension() + model.get_passive_tension() - afterload, 0.0)
                if np.all(np.abs(residual) < 1e-9):
                    break
                slope = dt * (
                    model.get_active_stiffness() / p["SL0"]
                    + p["Kp1"]
                    + p["Kp2"] * p["phi_p"] * np.exp(p["phi_p"] * (SL + v * dt - p["L_r"]))
                )
                v = v - residual / slope
                model.set_state(state)
                model.advance_step(dt, Ca, SL + v * dt, v)
            done = shortening & (v > 0.0)  # can shorten no further: hold
            if done.any():
                v, phase = np.where(done, 0.0, v), np.where(done, 2, phase)
                model.set_state(state)
                model.advance_step(dt, Ca, SL + v * dt, v)
            SL = SL + v * dt
            stress = model.get_active_tension() + model.get_passive_tension()
            phase = np.where((phase == 0) & (stress >= afterload), 1, phase)
            if start_restretch is None:
                v = np.where(phase == 1, v, 0.0)
            L[k], F[k], J[k] = SL, stress, model.get_atp_hydrolysis_rate()

    t = (np.arange(n) + 1) * dt
    dG = p["dG_ATP0"] + R_GAS * p["T"] * np.log(p["MgADP"] * p["Pi"] / (1000.0 * p["MgATP"]))
    out = {
        "relative afterload": afterload / afterload[-1],
        "work [kJ/m^3]": np.abs(np.trapezoid(F, L / L0, axis=0)),
        "shortening [% of Lo]": 100.0 * (1.0 - L.min(axis=0) / L0),
        "XB energy [kJ/m^3]": -dG * np.abs(np.trapezoid(RHO_XB * p["rho_myo"] * J, t, axis=0)),
        "velocity [Lo/s]": -np.min(np.gradient(L, dt, axis=0), axis=0) / L0,
    }
    out["power [kW/m^3]"] = out["velocity [Lo/s]"] * afterload
    out["XB efficiency [%]"] = 100.0 * out["work [kJ/m^3]"] / out["XB energy [kJ/m^3]"]
    return out, (t, L, F)


def plot_fig6_7_work_loops(save=True):
    print("Generating Figs. 6 and 7 -- work loops (about 20 s)...")
    results = {}
    fig6, axes6 = plt.subplots(3, 2, figsize=(10, 9), sharex="row")
    for col, (group, (params, diabetic, color)) in enumerate(GROUPS.items()):
        metrics, (t, L, F) = work_loops(params, diabetic)
        results[group] = metrics
        for i in range(L.shape[1]):
            lw, alpha = (2.5, 1.0) if i == L.shape[1] - 1 else (1.0, 0.6)
            axes6[0, col].plot(t, F[:, i], color=color, lw=lw, alpha=alpha)
            axes6[1, col].plot(t, L[:, i] / L0, color=color, lw=lw, alpha=alpha)
            axes6[2, col].plot(L[:, i] / L0, F[:, i], color=color, lw=lw, alpha=alpha)
        axes6[0, col].set(title=group, ylabel="stress [kPa]")
        axes6[1, col].set(xlabel="t [s]", ylabel="L / Lo")
        axes6[2, col].set(xlabel="L / Lo", ylabel="stress [kPa]")
    for ax in axes6.ravel():
        ax.grid(True, alpha=0.2)
    fig6.suptitle("Fig. 6: work loops at ten afterloads (bold: isometric), cf. paper Fig. 6")
    _save_or_show(fig6, "musgrave2025_fig6_work_loops.png", save)

    names = [k for k in results["non-diabetic"] if k != "relative afterload"]
    fig7, axes7 = plt.subplots(2, 3, figsize=(13, 7))
    paper = {"work [kJ/m^3]": -44, "velocity [Lo/s]": -52, "power [kW/m^3]": -69, "XB energy [kJ/m^3]": -75}
    print("    maximum                  non-diabetic   diabetic   change (paper)")
    for ax, name in zip(axes7.ravel(), names):
        for group, (_, _, color) in GROUPS.items():
            ax.plot(results[group]["relative afterload"], results[group][name], "o:", color=color, label=group)
        nd, d = results["non-diabetic"][name].max(), results["diabetic"][name].max()
        note = f"{paper[name]:+d}%" if name in paper else ("similar" if "shortening" in name else "more than 2x")
        print(f"    {name:<24} {nd:8.2f}     {d:8.2f}   {100 * (d / nd - 1):+5.0f}% ({note})")
        ax.set(xlabel="relative afterload", ylabel=name)
        ax.grid(True, alpha=0.2)
    axes7[0, 0].legend(fontsize="small")
    fig7.suptitle("Fig. 7: work-loop mechanoenergetics against relative afterload, cf. paper Fig. 7")
    _save_or_show(fig7, "musgrave2025_fig7_mechanoenergetics.png", save)


# %% [markdown]
# ## Figs. 8A and 9A -- Pi and ATP
#
# Raised Pi (10 mM) speeds the reverse of attachment and lowers both
# twitches by about a quarter. Lowered ATP (1 mM) slows detachment: the
# twitches grow and lengthen, much more in the diabetic model, whose
# kd_ATP (5 mM, the bound of the fit) leaves ATP far from saturating, and its
# diastolic stress rises.


# %%
def plot_fig8_9_metabolites(save=True):
    print("Generating Figs. 8A and 9A -- Pi and ATP...")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    print("    group          amplitude change at 10 mM Pi   at 1 mM ATP (duration)")
    for group, (params, diabetic, color) in GROUPS.items():
        t, base, (_, _, amp, dur) = twitch(params, diabetic)
        _, high_pi, (_, _, amp_pi, _) = twitch({**params, "Pi": 10.0}, diabetic)
        _, low_atp, (_, _, amp_atp, dur_atp) = twitch({**params, "MgATP": 1.0}, diabetic)
        for ax, other, label in ((axes[0], high_pi, "10 mM Pi"), (axes[1], low_atp, "1 mM ATP")):
            ax.plot(t, base, color=color, lw=2, label=f"{group}, baseline")
            ax.plot(t, other, "--", color=color, lw=2, label=f"{group}, {label}")
        print(f"    {group:<14} {100 * (amp_pi / amp - 1):+5.0f}%                         {100 * (amp_atp / amp - 1):+5.0f}% ({100 * (dur_atp / dur - 1):+.0f}%)")
    print("    (paper: Pi 26% vs. 28% lower; ATP +9% and +29%, duration +2% and +25%)")
    axes[0].set(xlabel="t [s]", ylabel="stress [kPa]", title="Fig. 8A: raised Pi (ATP 5 mM)")
    axes[1].set(xlabel="t [s]", title="Fig. 9A: lowered ATP (Pi 1 mM)")
    for ax in axes:
        ax.legend(fontsize="x-small")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Effects of Pi and ATP on the isometric twitch, cf. paper Figs. 8A and 9A")
    _save_or_show(fig, "musgrave2025_fig8_9_metabolites.png", save)


# %% [markdown]
# ## Fig. 10A -- Calcium handling and crossbridges crossed over
#
# With the non-diabetic passive model throughout: diabetic calcium handling
# (its transient and Ca50) and diabetic crossbridges each lower the twitch
# by a quarter or more, while the longer twitch comes mostly from the
# crossbridges.


# %%
def plot_fig10_cross_over(save=True):
    print("Generating Fig. 10A -- the cross-over...")
    passive = {key: ND[key] for key in ("Kp1", "mu", "Kp2", "phi_p")}
    pairings = {  # crossbridges / calcium handling
        "ND/ND": (ND, False, "black", "-"),
        "D/ND": ({**D, **passive, "Ca50": ND["Ca50"]}, False, "tab:purple", ":"),
        "ND/D": ({**ND, "Ca50": D["Ca50"]}, True, "tab:green", "--"),
        "D/D": ({**D, **passive}, True, "tab:orange", "-"),
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    reference = None
    print("    XB/Ca   amplitude [kPa]   change   duration change")
    for label, (params, diabetic, color, style) in pairings.items():
        t, stress, (_, _, amp, dur) = twitch(params, diabetic)
        reference = reference or (amp, dur)
        ax.plot(t, stress, style, color=color, lw=2, label=label)
        print(f"    {label:<6}  {amp:6.2f}            {100 * (amp / reference[0] - 1):+4.0f}%   {100 * (dur / reference[1] - 1):+4.0f}%")
    print("    (paper: -25 to -30% for each alone, -45% for both; duration +7% (D Ca), +15% (D XB), +23%)")
    ax.set(xlabel="t [s]", ylabel="stress [kPa]", title="Fig. 10A: crossbridges / calcium handling")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.2)
    _save_or_show(fig, "musgrave2025_fig10_cross_over.png", save)


# %%
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reproduce Musgrave et al. (2025) human atrial results.")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save figures to PNG files instead of displaying them interactively.",
    )
    args, _ = parser.parse_known_args()

    print("=" * 60)
    print("Musgrave2025 (human atrial myofilament) Figure Reproduction")
    print("=" * 60)

    plot_fig2_passive(save=args.save)
    plot_fig3_calcium(save=args.save)
    plot_fig4_crossbridges(save=args.save)
    plot_fig5_twitches(save=args.save)
    plot_fig6_7_work_loops(save=args.save)
    plot_fig8_9_metabolites(save=args.save)
    plot_fig10_cross_over(save=args.save)
