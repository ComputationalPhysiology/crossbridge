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


#: Fourier coefficients (a_j, j = 0..30; b_j, j = 1..30) [uM] of the 1 Hz human atrial
#: calcium transients that drive the muscle models of Musgrave et al. (2025), fitted here
#: to the 1 kHz traces of their code (github.com/JuliaMusgrave/AtrialModel_2025_Human).
MUSGRAVE_CALCIUM_FOURIER = {
    "nondiabetic": (
        np.array(
            [
                0.311769, 0.038564, -0.011919, -0.012598, -0.014442, -0.014061, -0.013204,
                -0.011557, -0.009969, -0.008459, -0.006845, -0.004857, -0.003575, -0.00253,
                -0.002342, -0.00147, -0.001037, -0.000691, -0.000257, 0.000063, 0.000292,
                0.000107, 0.000445, 0.000287, 0.000317, 0.000274, 0.000268, 0.000202,
                0.000157, 0.000138, 0.000101,
            ]
        ),
        np.array(
            [
                0.106232, 0.050523, 0.02958, 0.020381, 0.012978, 0.007301, 0.003594, 0.001295,
                -0.000484, -0.001687, -0.002146, -0.002351, -0.002579, -0.002141, -0.002253,
                -0.001907, -0.001721, -0.001574, -0.001343, -0.001171, -0.000817, -0.000637,
                -0.000548, -0.000276, -0.000302, -0.000247, -0.000073, -0.000021, -0.000022,
                -0.00001,
            ]
        ),
    ),
    "diabetic": (
        np.array(
            [
                0.345295, 0.01483, -0.016198, -0.016408, -0.016402, -0.014463, -0.011366,
                -0.008738, -0.006214, -0.003776, -0.002428, -0.001373, -0.001234, -0.000376,
                -0.00011, 0.000248, 0.000413, 0.00048, 0.000307, 0.000189, 0.000259, 0.00019,
                0.000169, 0.00003, 0.000021, 0.000021, 0.000036, -0.000027, -0.000008,
                -0.000022, -0.000032,
            ]
        ),
        np.array(
            [
                0.101007, 0.040144, 0.023486, 0.012368, 0.00515, 0.000952, -0.001348,
                -0.002756, -0.003195, -0.002811, -0.002508, -0.00229, -0.001792, -0.001515,
                -0.001246, -0.000839, -0.000516, -0.00052, -0.000138, -0.000044, -0.000117,
                -0.000011, 0.000013, -0.000002, 0.000021, 0.00003, 0.000053, 0.000033,
                -0.00001, 0.000019,
            ]
        ),
    ),
}  # fmt: skip


def musgrave_calcium(t: float | np.ndarray, diabetic: bool = False) -> np.ndarray:
    """Human atrial calcium transient [uM] at 37 degC, paced at 1 Hz.

    The transients that drive the non-diabetic and diabetic muscle models of
    Musgrave et al. (2025), their Fig. 3: a generic human atrial transient
    scaled to the diastolic level and amplitude Jones et al. (2023) measured
    in non-diabetic atrial trabeculae, and, for the diabetic one, also to the
    slower rise and decay and higher diastolic level they measured in
    diabetic ones. Diastolic 0.227 and 0.260 uM; peak 0.526 uM at 67 ms and
    0.517 uM at 92 ms.

    As a 30-harmonic Fourier series (`MUSGRAVE_CALCIUM_FOURIER`) fitted to the
    1 kHz traces of the authors' code: within 0.4% (non-diabetic) and 1%
    (diabetic) of the amplitude, the largest error where the traces' first
    and last samples differ. Periodic in `t` [s], each beat starting at
    t = 0 mod 1.
    """
    a, b = MUSGRAVE_CALCIUM_FOURIER["diabetic" if diabetic else "nondiabetic"]
    phase = 2.0 * np.pi * np.multiply.outer(np.asarray(t, dtype=float), np.arange(a.size))
    return np.cos(phase) @ a + np.sin(phase[..., 1:]) @ b


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
