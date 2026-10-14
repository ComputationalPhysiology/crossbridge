import numpy as np


def calcium_trace(
    t: float | np.ndarray,
    t_start: float = 0.0,
    c0: float = 0.1,
    cmax: float = 1.1,
    tau1: float = 0.02,
    tau2: float = 0.11,
    t0: float = 0.1,
):
    """Generates a calcium transient trace based on a bi-exponential function.

    Parameters:
    -----------
    t : float or np.ndarray
        Time(s) at which to evaluate the calcium concentration.
    t_start : float, optional
        Time at which the calcium transient starts. Default is 0.0.
    c0 : float, optional
        Baseline calcium concentration. Default is 0.1.
    cmax : float, optional
        Peak calcium concentration. Default is 1.1.
    tau1 : float, optional
        Time constant for the rising phase of the transient. Default is 0.02.
    tau2 : float, optional
        Time constant for the falling phase of the transient. Default is 0.11.
    t0 : float, optional
        Time at which the calcium transient reaches its peak. Default is 0.1.
    """
    assert tau1 > 0 and tau2 > 0, "Time constants must be positive"
    assert t0 >= t_start, "Peak time must be greater than or equal to start time"

    # beta = (tau1 / tau2) ** (-1 / (tau1 / tau2 - 1)) - (tau1 / tau2) ** (-1 / (1 - tau2 / tau1))
    # val = c0 + (cmax - c0) / beta * (np.exp(-(t - t0) / tau1) - np.exp(-(t - t0) / tau2))
    # return val if t >= t0 else c0

    beta = (tau1 / tau2) ** (-1 / (tau1 / tau2 - 1)) - (tau1 / tau2) ** (-1 / (1 - tau2 / tau1))

    # Vectorized time handling
    val = np.full_like(t, c0)
    mask = t >= t0

    def val_expr(dt):
        return c0 + (cmax - c0) / beta * (np.exp(-dt / tau1) - np.exp(-dt / tau2))

    if isinstance(t, np.ScalarType):
        if mask:
            val = val_expr(t - t0)
    else:
        assert isinstance(t, np.ndarray), "Input time must be a float or numpy array"
        if np.any(mask):
            dt_trans = t[mask] - t0
            val[mask] = val_expr(dt_trans)
    return val


#: Fits of ``Ca(phi) = a / phi * exp(-b (ln(phi) + c)^2) + Ca0`` to the rat trabecula
#: calcium transients of Janssen et al. (2002), by Marzban et al. (2020) (`para_fitted_Ca`
#: in CardiovascularMechanics.m of github.com/beards-lab/Rat-Cardiac-Energetic). Rows:
#: stimulation frequency [Hz], a, b, c [-] and Ca0 [uM].
JANSSEN_CALCIUM_FITS = np.array(
    [
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        [0.0838, 0.1306, 0.1802, 0.2557, 0.3099, 0.3613, 0.408, 0.4539, 0.4602],
        [0.7513, 0.8756, 1.0274, 1.4988, 1.6107, 1.6741, 1.7902, 2.1398, 1.9832],
        [2.237, 2.0486, 1.948, 1.852, 1.6932, 1.6773, 1.5988, 1.4952, 1.4524],
        [0.1865, 0.1815, 0.1709, 0.1693, 0.161, 0.1661, 0.1425, 0.1458, 0.1222],
    ]
)


def janssen_calcium(t: float | np.ndarray, frequency: float) -> np.ndarray:
    """Rat cardiac calcium transient [uM] at 37 degC, paced at `frequency` [Hz].

    The transients Janssen et al. (2002) measured in rat trabeculae, which
    Tewari et al. (2016) and Marzban et al. (2020) used to identify their
    myofilament models, as Marzban et al. fitted them: a log-normal-shaped
    pulse each period, ``a / phi * exp(-b (ln(phi) + c)^2) + Ca0`` with ``phi``
    the phase within the beat (see `JANSSEN_CALCIUM_FITS`). Between the fitted
    frequencies (2-10 Hz) the coefficients are interpolated with pchip, as in
    that code. Periodic in `t` [s], with each beat starting at t = 0 mod 1/f
    (offset by 0.1 ms, as in the original, to keep phi > 0).

    These are fits to measured data, unlike the synthetic `calcium_trace`;
    their peaks are 1.3 uM (2 Hz) to 2.4 uM (7 Hz).
    """
    from scipy.interpolate import PchipInterpolator

    freqs, *rows = JANSSEN_CALCIUM_FITS
    if not freqs[0] <= frequency <= freqs[-1]:
        raise ValueError(f"frequency {frequency} Hz is outside the fitted 2-10 Hz")
    a, b, c, Ca0 = (float(PchipInterpolator(freqs, row)(frequency)) for row in rows)
    period = 1.0 / frequency
    phi = np.mod(np.asarray(t, dtype=float) + 1e-4, period) / period
    return a / phi * np.exp(-b * (np.log(phi) + c) ** 2) + Ca0


def sl_trace(
    t,
    SL0=2.2,
    SL1=2.046,
    SLt0=0.15,
    SLt1=0.55,
    SLtau0=0.05,
    SLtau1=0.02,
):
    """Generates a sarcomere length (SL) trace based on a bi-exponential function.

    Parameters:
    -----------
    t : float or np.ndarray
        Time(s) at which to evaluate the sarcomere length.
    SL0 : float, optional
        Baseline sarcomere length. Default is 2.2 um.
    SL1 : float, optional
        Minimum sarcomere length during contraction. Default is 2.046 um (93% of SL0).
    SLt0 : float, optional
        Time at which the sarcomere length starts to decrease. Default is 0.15 s.
    SLt1 : float, optional
        Time at which the sarcomere length starts to recover. Default is 0.55 s.
    SLtau0 : float, optional
        Time constant for the decreasing phase of the sarcomere length. Default is 0.05 s.
    SLtau1 : float, optional
        Time constant for the recovering phase of the sarcomere length. Default is 0.02 s.
    """

    assert SLtau0 > 0 and SLtau1 > 0, "Time constants must be positive"
    assert SLt0 >= 0 and SLt1 >= SLt0, (
        "Time points must be non-negative and SLt1 must be greater than or equal to SLt0"
    )

    term1 = np.maximum(0, 1 - np.exp((SLt0 - t) / SLtau0))
    term2 = np.maximum(0, 1 - np.exp((SLt1 - t) / SLtau1))
    return SL0 + (SL1 - SL0) * (term1 - term2)
