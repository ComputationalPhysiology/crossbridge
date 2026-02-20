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
