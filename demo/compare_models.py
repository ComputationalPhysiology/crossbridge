# %% [markdown]
# # Comparing RDQ18, RDQ20MF, and Lewalle2024
#
# Side-by-side comparison of the three `crossbridge` models: `RDQ18`, `RDQ20MF`,
# and `Lewalle2024`.
#
# All three subclass `CardiacActivationModel` and share the same construction
# and stepping interface (`ModelClass(num_cells, Ta_max, params)`,
# `advance_step(dt, Ca_val, SL_vals, dSL_vals=None)`, `get_active_tension()`),
# even though their internal state representations, characteristic time steps,
# and `Ta_max` semantics differ substantially -- see the README's "Choosing a
# Model" section. This notebook drives all three through the *same* calcium
# transient and sarcomere-length protocol using the `crossbridge.get_model()`
# registry, to show that a coupling loop written against the shared interface
# works unchanged when the model class is swapped.
#
# Two comparisons are made:
#
# 1. Isometric twitch under a shared calcium transient at a common SL,
#    compared both in raw units and peak-normalized (to compare kinetics
#    independent of each model's own tension scale).
# 2. Steady-state length dependence (Frank-Starling), peak-normalized to
#    each model's own value at a reference SL -- all three reproduce
#    length-dependent activation, but via different mechanisms (spatial
#    filament overlap for RDQ18/RDQ20MF, myosin OFF-state force feedback
#    for Lewalle2024).
#
# Only RDQ18 uses `Ta_max` as a direct tension scale; RDQ20MF and Lewalle2024
# compute tension from their own internal parameters (`a_XB`, `Tref`) and
# ignore it, so the *raw* (non-normalized) magnitudes shown here are not a
# physical calibration comparison between models -- only the normalized
# panels should be read that way.
#
# This can also be run as a standalone script, saving each figure to a PNG
# file instead of displaying it inline:
# ```bash
# python demo/compare_models.py --save
# ```

# %%
import numpy as np
import matplotlib.pyplot as plt

from crossbridge import get_model, calcium_trace

# %% [markdown]
# ## Model configuration

# %%
# RDQ18 needs an explicit Ta_max to set its tension scale; RDQ20MF and
# Lewalle2024 ignore Ta_max (see intro above) and use their own defaults,
# so passing it there has no effect on the result -- kept only so every
# model can be constructed with the same kwargs dict below.
MODEL_KWARGS = {
    "RDQ18": {"Ta_max": 60.0},
    "RDQ20MF": {},
    "Lewalle2024": {},
}
COLORS = {"RDQ18": "tab:blue", "RDQ20MF": "tab:orange", "Lewalle2024": "tab:green"}

# A common operating range covered by all three models' calibrations.
REFERENCE_SL = 2.0  # [um]
SL_SWEEP = np.linspace(1.9, 2.1, 11)  # [um]


def _print_model_table():
    print(f"{'Model':<12} {'State repr.':<32} {'Ta_max used?':<14} {'dt':<10}")
    print("-" * 70)
    rows = [
        ("RDQ18", "RU triplet probability tensor", "yes"),
        ("RDQ20MF", "RU tensor + explicit XB states", "no (uses a_XB)"),
        ("Lewalle2024", "Land2017 + OFF-state populations", "no (uses Tref)"),
    ]
    for name, repr_str, ta_str in rows:
        model = get_model(name)(num_cells=1, **MODEL_KWARGS[name])
        print(f"{name:<12} {repr_str:<32} {ta_str:<14} {model.dt:.1e} s")


_print_model_table()

# %% [markdown]
# ## Twitch comparison


# %%
def _run_twitch(name, ca_fn, sl, duration):
    ModelClass = get_model(name)
    model = ModelClass(num_cells=1, **MODEL_KWARGS[name])
    dt = model.dt
    n_steps = int(duration / dt)
    t_arr = np.arange(n_steps) * dt
    Ca_t = ca_fn(t_arr)

    Ta_hist = np.zeros(n_steps)
    SL_arr = np.array([sl])
    for i in range(n_steps):
        model.advance_step(dt, float(Ca_t[i]), SL_arr)
        Ta_hist[i] = model.get_active_tension()[0]
    return t_arr, Ta_hist


def plot_twitch_comparison(save=True):
    print("Comparing isometric twitches across models...")

    duration = 1.2  # [s]

    def ca_fn(t):
        return calcium_trace(t, c0=0.1, cmax=3.0, tau1=0.02, tau2=0.13, t0=0.05)

    traces = {name: _run_twitch(name, ca_fn, REFERENCE_SL, duration) for name in MODEL_KWARGS}

    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

    t_ref = traces["RDQ18"][0]
    axes[0].plot(t_ref, ca_fn(t_ref), color="black")
    axes[0].set_ylabel(r"$[Ca^{2+}]_i$ [$\mu$M]")
    axes[0].set_title(f"Shared calcium transient (SL = {REFERENCE_SL:.1f} um)")
    axes[0].grid(True, alpha=0.2)

    ax = axes[1]
    for name, (t_arr, ta) in traces.items():
        ax.plot(t_arr, ta, color=COLORS[name], label=name)
    ax.set_ylabel(r"$T_a$ [kPa]")
    ax.set_title("Raw active tension (not cross-calibrated -- see intro above)")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    for name, (t_arr, ta) in traces.items():
        peak = ta.max()
        ta_norm = ta / peak if peak > 0 else ta
        ax.plot(t_arr, ta_norm, color=COLORS[name], label=name)
    ax.set_xlabel("t [s]")
    ax.set_ylabel(r"$T_a / T_a^{max}$ [-]")
    ax.set_title("Peak-normalized (kinetics comparison)")
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fname = "compare_models_twitch.png"
    if save:
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"  Saved: {fname}")
    else:
        plt.show()
    plt.close(fig)


# %% [markdown]
# ## Length-dependence comparison

# %%
# A moderately activating (submaximal) calcium level per model, chosen so
# each model sits on the steep part of its own force-calcium relationship
# (where length-dependent activation is most visible) rather than saturated.
CA_LEVEL = {"RDQ18": 0.5, "RDQ20MF": 0.5, "Lewalle2024": 6.0}


def _run_length_dependence(name, sl_values, duration=1.5):
    ModelClass = get_model(name)
    n = len(sl_values)
    model = ModelClass(num_cells=n, **MODEL_KWARGS[name])
    dt = model.dt
    Ca_arr = np.full(n, CA_LEVEL[name])
    SL_arr = np.asarray(sl_values, dtype=float)
    for _ in range(int(duration / dt)):
        model.advance_step(dt, Ca_arr, SL_arr)
    return model.get_active_tension()


def plot_length_dependence_comparison(save=True):
    print("Comparing length-dependent activation across models...")

    fig, ax = plt.subplots(figsize=(6, 5))
    for name in MODEL_KWARGS:
        ta = _run_length_dependence(name, SL_SWEEP)
        ref_idx = np.argmin(np.abs(SL_SWEEP - REFERENCE_SL))
        ref = ta[ref_idx]
        ta_norm = ta / ref if ref > 0 else ta
        ax.plot(SL_SWEEP, ta_norm, "o-", color=COLORS[name], label=name, markersize=4)

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("SL [um]")
    ax.set_ylabel(f"$T_a$ / $T_a$(SL={REFERENCE_SL:.1f} um)  [-]")
    ax.set_title("Length-dependent activation, normalized to a common reference SL")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    fname = "compare_models_length_dependence.png"
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

    parser = argparse.ArgumentParser(description="Compare RDQ18, RDQ20MF, and Lewalle2024.")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save figures to PNG files instead of displaying them interactively.",
    )
    args, _ = parser.parse_known_args()

    plot_twitch_comparison(save=args.save)
    plot_length_dependence_comparison(save=args.save)

    print("\nDone.")
