import numpy as np
import matplotlib.pyplot as plt
import time
from crossbridge import RDQ18, calcium_trace, sl_trace


def run_simulation_with_plots():
    initial_time = time.time()

    # --- 1. Setup ---
    # We use num_cells=1 for a single 0D simulation trace
    sarcomere = RDQ18(num_cells=1)
    p = sarcomere.p

    # Simulation Parameters
    dt = p["dt"]
    T = 1.0
    plot_period = 1000
    coupling_period = 10
    # Set to True to enable force-length coupling in the simulation
    force_length_coupling = True

    # Macroscopic / Reference Lengths
    l0 = 2.2
    muA = 0.05
    alpha = 0.2
    gammaF0 = 0.0

    c0 = 0.1
    cmax = 1.1
    tau1 = 0.02
    tau2 = 0.11
    t0 = 0.1

    maxiterMicro = int(np.ceil(T / dt))
    maxiterPlot = int(np.ceil(maxiterMicro / plot_period))

    t_plot_arr = np.zeros(maxiterPlot)
    Pfrac_plot_arr = np.zeros(maxiterPlot)
    gammaF_plot_arr = np.zeros(maxiterPlot)
    SL_plot_arr = np.zeros(maxiterPlot)

    # Imposed SL function for non-coupled case
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

    # Plotting Window
    plt.ion()
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axs = axes.flatten()

    time_ode = 0
    iterPlot = 0

    # --- 2. Simulation Loop ---
    for iterMicro in range(1, maxiterMicro + 1):
        t = iterMicro * dt

        # 2a. Calcium
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
            # Active strain approach
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
            m0 = np.sum(np.sum(sarcomere.xODE[0], axis=1), axis=1)
            m_mid = np.sum(np.sum(sarcomere.xODE, axis=1), axis=2)
            m_last = np.sum(np.sum(sarcomere.xODE[-1], axis=0), axis=0)

            # Flatten to (4,) for single cell
            m0 = m0[:, 0]
            m_mid = m_mid[:, :, 0]
            m_last = m_last[:, 0]

            marginalsODE = np.vstack([m0, m_mid, m_last])
            PfracODE = np.mean(marginalsODE[:, 2] + marginalsODE[:, 3])

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

            axs[1].clear()
            x_dom = np.arange(p["nu"])
            # Recompute Chi for viz using internal helper
            ChiLA_plot, ChiRA_plot = sarcomere._compute_Chi(np.array([SL_t]))
            axs[1].plot(x_dom, ChiLA_plot[:, 0], label="ChiLA")
            axs[1].plot(x_dom, ChiRA_plot[:, 0], label="ChiRA")
            axs[1].legend()

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

            axs[4].clear()
            axs[4].plot(t_plot_arr[: iterPlot + 1], Pfrac_plot_arr[: iterPlot + 1], "b-")
            axs[4].plot(t, PfracODE, "bo")
            axs[4].set_ylim(0, 1)
            axs[4].set_ylabel("Fraction P")

            axs[5].clear()
            axs[5].plot(m_cum[:, 0], "b-")
            axs[5].plot(m_cum[:, 1], "b-", linewidth=2)
            axs[5].plot(m_cum[:, 2], "b-")
            axs[5].set_ylim(0, 1)

            t_ode_avg = time_ode / iterMicro * 1000
            fig.suptitle(f"ODE={t_ode_avg:.2f}ms")

            plt.pause(0.001)
            iterPlot += 1

    print(f"Total Simulation Time: {time.time() - initial_time:.2f}s")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_simulation_with_plots()
