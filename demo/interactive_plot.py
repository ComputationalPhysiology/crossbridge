# %% [markdown]
# # 0D Active Strain Electromechanics and Visualization
#
# This tutorial demonstrates how to simulate a zero-dimensional (0D) cardiac contraction using the `RDQ18` sarcomere model from the `crossbridge` library.
#
# The script features two modes of operation:
# 1. **Imposed Sarcomere Length (`force_length_coupling = False`)**: The sarcomere is subjected to a predefined kinematic stretch protocol.
# 2. **Active Strain Coupling (`force_length_coupling = True`)**: The model couples the generated active force back to the tissue kinematics. The fraction of permissive crossbridges acts as a proxy for active tension, which drives a 0D active strain differential equation to compute tissue shortening.
#
# Additionally, this script breaks down the internal states of the spatially-explicit model, visualizing the distribution of myosin heads across the thick filament.

# %%
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from crossbridge import RDQ18, calcium_trace, sl_trace

# %% [markdown]
# ## 1. Setup and Initialization
#
# First, we initialize the `RDQ18` model for a single computational cell (`num_cells=1`). We extract the default parameters from the model and set up our simulation configurations.
#
# We define the macroscopic physical parameters used for the active strain coupling:
# * `l0`: The reference resting sarcomere length.
# * `muA`: A viscosity parameter dictating the rate of active stretch evolution.
# * `alpha`: A scaling factor that converts the normalized permissivity into a physical active force.

# %%
initial_time = time.time()

# --- 1. Setup ---
# We use num_cells=1 for a single 0D simulation trace
sarcomere = RDQ18(num_cells=1)
p = sarcomere.p

# Simulation Parameters
dt = p["dt"]
force_length_coupling = True  # Set to True to enable force-length coupling
interactive_plotting = True

outdir = Path("plot_output")
outdir.mkdir(exist_ok=True)
filename = outdir / "frame_{:03d}.png"

T = 1.0
plot_period = 1000
coupling_period = 10

# Macroscopic / Reference Lengths for the active strain formulation
l0 = 2.2  # Reference length (um)
muA = 0.05  # Viscosity parameter
alpha = 0.2  # Contractility / Active force scaling
gammaF0 = 0.0  # Initial active strain

# Calcium Transient Parameters
c0 = 0.1
cmax = 1.1
tau1 = 0.02
tau2 = 0.11
t0 = 0.1

maxiterMicro = int(np.ceil(T / dt))
maxiterPlot = int(np.ceil(maxiterMicro / plot_period))

# Storage arrays for plotting history
t_plot_arr = np.zeros(maxiterPlot)
Pfrac_plot_arr = np.zeros(maxiterPlot)
gammaF_plot_arr = np.zeros(maxiterPlot)
SL_plot_arr = np.zeros(maxiterPlot)

# Imposed SL function parameters for the non-coupled case
SL0 = l0
SL1 = l0 * 0.93
SLt0 = 0.15
SLt1 = 0.55
SLtau0 = 0.05
SLtau1 = 0.02

# Initial State
gammaF = gammaF0
SL_t = (
    l0 * (1 + gammaF)
    if force_length_coupling
    else sl_trace(0, SL0=SL0, SL1=SL1, SLt0=SLt0, SLt1=SLt1, SLtau0=SLtau0, SLtau1=SLtau1)
)

# Plotting Window Initialization
if interactive_plotting:
    plt.ion()
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
axs = axes.flatten()

time_ode = 0
iterPlot = 0

# %% [markdown]
# ## 2. The Simulation Loop and Visualization
#
# At every micro-timestep (`dt`), the script performs three main steps:
#
# 1. **Evaluate Calcium**: It queries the analytical bi-exponential function to find the current intracellular calcium concentration.
# 2. **Mechanics Update**: If `force_length_coupling` is enabled, the mechanics are updated at a slower macro-timestep (dictated by `coupling_period`). The active strain $\gamma_f$ evolves according to a simplified 0D version of the active strain model.
# 3. **Sarcomere ODE Advance**: The calculated calcium and updated sarcomere length are fed into the `RDQ18` model, which steps the transition probabilities forward in time.
#
# ### Visualizing Internal Spatial States
#
# To truly understand the RDQ18 model, it is helpful to look at its spatially explicit nature. The model divides the half-thick filament into `nu` (typically 36) discrete units.
#
# During the plotting block, we calculate the marginal probabilities of the four possible states (0N, 1N, 0P, 1P) for every individual node along the filament. Because nodes near the bare zone (H-zone) or outside the actin overlap region experience different physical conditions, their state probabilities will differ.
#
# The resulting 6-panel plot shows:
# 1. **Top-Left**: The calcium transient driving the contraction.
# 2. **Mid-Left**: The active strain ($\gamma_F$) or sarcomere length (SL) over time.
# 3. **Bottom-Left**: The global fraction of permissive states (proxy for macro-scale force).
# 4. **Top-Right**: The geometric overlap functions ($\chi_{LA}$ and $\chi_{RA}$) indicating which sections of the myosin filament are actively facing an actin filament.
# 5. **Mid-Right**: A stacked bar chart showing the probability distribution of the 4 states at each of the 36 spatial locations.
# 6. **Bottom-Right**: The cumulative spatial distribution of those states.

# %%
# --- 2. Simulation Loop ---
for iterMicro in range(1, maxiterMicro + 1):
    t = iterMicro * dt

    # 2a. Calcium Evaluation
    Ca_val = calcium_trace(
        t=t,
        c0=c0,
        cmax=cmax,
        tau1=tau1,
        tau2=tau2,
        t0=t0,
    )

    # 2b. Mechanics (Coupling or Imposed SL)
    if force_length_coupling and (iterMicro % coupling_period == 0):
        permissivity = sarcomere.compute_permissivity()
        Pfrac = permissivity[0]

        # Active strain approach: gammaF evolves based on the generated force
        gammaF += dt * coupling_period / muA * (-alpha * Pfrac + (1 / (1 + gammaF) - 1))
        SL_t = l0 * (1 + gammaF)

    elif not force_length_coupling:
        SL_t = sl_trace(t, SL0=SL0, SL1=SL1, SLt0=SLt0, SLt1=SLt1, SLtau0=SLtau0, SLtau1=SLtau1)

    # 2c. Advance Sarcomere Model
    start_time = time.time()
    # advance_ODE will update probabilities internally using SL_t and Ca_val
    sarcomere.advance_ODE(dt, Ca_val, np.array([SL_t]))
    time_ode += time.time() - start_time

    # 2d. Plotting
    if iterMicro % plot_period == 0:
        # Stats for plotting
        # Calculate the marginal probabilities for the first, middle, and last units
        m0 = np.sum(np.sum(sarcomere.xODE[0], axis=1), axis=1)
        m_mid = np.sum(np.sum(sarcomere.xODE, axis=1), axis=2)
        m_last = np.sum(np.sum(sarcomere.xODE[-1], axis=0), axis=0)

        # Flatten to (4,) for a single cell
        m0 = m0[:, 0]
        m_mid = m_mid[:, :, 0]
        m_last = m_last[:, 0]

        # Reconstruct the full spatial profile of states
        marginalsODE = np.vstack([m0, m_mid, m_last])
        PfracODE = np.mean(marginalsODE[:, 2] + marginalsODE[:, 3])

        # Sort states for plotting: [1P, 0P, 1N, 0N]
        m_sorted = marginalsODE[:, [2, 3, 1, 0]]
        m_cum = np.cumsum(m_sorted, axis=1)

        # Update History
        t_plot_arr[iterPlot] = t
        Pfrac_plot_arr[iterPlot] = PfracODE
        SL_plot_arr[iterPlot] = SL_t
        if force_length_coupling:
            gammaF_plot_arr[iterPlot] = gammaF

        # Draw Plots
        tt = np.linspace(0, T, 1000)

        # 1. Calcium Input
        axs[0].clear()
        axs[0].plot(
            tt,
            calcium_trace(
                t=tt,
                c0=c0,
                cmax=cmax,
                tau1=tau1,
                tau2=tau2,
                t0=t0,
            ),
        )
        axs[0].plot(t, Ca_val, "o")
        axs[0].set_ylabel(r"Ca [$\mu$M]")
        axs[0].set_title(f"t={t:.3f}")

        # 2. Geometric Overlap
        axs[1].clear()
        x_dom = np.arange(p["nu"])
        # Recompute Chi for viz using internal helper
        ChiLA_plot, ChiRA_plot = sarcomere._compute_Chi(np.array([SL_t]))
        axs[1].plot(x_dom, ChiLA_plot[:, 0], label="ChiLA")
        axs[1].plot(x_dom, ChiRA_plot[:, 0], label="ChiRA")
        axs[1].legend()

        # 3. Kinematics (Active Strain or SL)
        axs[2].clear()
        if force_length_coupling:
            axs[2].plot(t_plot_arr[: iterPlot + 1], gammaF_plot_arr[: iterPlot + 1])
            axs[2].set_ylabel(r"$\gamma_F$")
        else:
            axs[2].plot(
                tt,
                sl_trace(
                    tt,
                    SL0=SL0,
                    SL1=SL1,
                    SLt0=SLt0,
                    SLt1=SLt1,
                    SLtau0=SLtau0,
                    SLtau1=SLtau1,
                ),
            )
            axs[2].plot(t, SL_t, "o")
            axs[2].set_ylabel(r"SL [$\mu$m]")

        # 4. Spatially Explicit State Distribution
        axs[3].clear()
        colors = plt.cm.viridis([0.9, 0.6, 0.3, 0.0])
        labels = ["1P", "0P", "1N", "0N"]
        bottom = np.zeros(p["nu"])
        for k in range(4):
            axs[3].bar(
                x_dom,
                m_sorted[:, k],
                bottom=bottom,
                color=colors[k],
                label=labels[k],
                width=1.0,
            )
            bottom += m_sorted[:, k]
        axs[3].legend(loc="upper right", fontsize="small", ncol=4)
        axs[3].set_ylim(0, 1)

        # 5. Global Permissivity
        axs[4].clear()
        axs[4].plot(t_plot_arr[: iterPlot + 1], Pfrac_plot_arr[: iterPlot + 1], "b-")
        axs[4].plot(t, PfracODE, "bo")
        axs[4].set_ylim(0, 1)
        axs[4].set_ylabel("Fraction P")

        # 6. Cumulative State Curves
        axs[5].clear()
        axs[5].plot(m_cum[:, 0], "b-")
        axs[5].plot(m_cum[:, 1], "b-", linewidth=2)
        axs[5].plot(m_cum[:, 2], "b-")
        axs[5].set_ylim(0, 1)

        t_ode_avg = time_ode / iterMicro * 1000
        fig.suptitle(f"ODE={t_ode_avg:.2f}ms")

        if interactive_plotting:
            plt.pause(0.001)
        else:
            if filename is not None:
                fig.savefig(filename.as_posix().format(iterPlot), dpi=300)

        iterPlot += 1

print(f"Total Simulation Time: {time.time() - initial_time:.2f}s")
if interactive_plotting:
    plt.ioff()
plt.show()

# %% [markdown]
# ![_](https://github.com/user-attachments/assets/e0bb2dc4-81d4-4e18-b626-97b3480088f3)
