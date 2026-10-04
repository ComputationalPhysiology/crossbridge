"""
Unit tests for the RDQ20MF model.

Test philosophy mirrors test_rdq18.py:
  - Structural: shape, initialization
  - Mathematical: probability conservation, state bounds
  - Physiological: Ca sensitivity, length dependence
  - Computational: vectorization independence
  - Numerical: stability, crossbridge dynamics
"""

import numpy as np
import pytest

from crossbridge import RDQ20MF


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model():
    """Fresh 5-cell RDQ20MF instance."""
    return RDQ20MF(num_cells=5)


@pytest.fixture
def dt(model):
    return model.dt


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


def test_initialization_shapes(model):
    """x_RU and x_XB must have the correct shapes after init."""
    # x_RU: (2, 2, 2, 2, num_cells)
    assert model.x_RU.shape == (2, 2, 2, 2, 5), f"x_RU shape mismatch: {model.x_RU.shape}"
    # x_XB: (2, 2, num_cells)
    assert model.x_XB.shape == (2, 2, 5), f"x_XB shape mismatch: {model.x_XB.shape}"


def test_initial_probability_sum(model):
    """RU probabilities must sum to 1 at initialization."""
    # Sum over all RU state axes (0..3), leaving the cell axis
    totals = model.x_RU.sum(axis=(0, 1, 2, 3))  # shape (5,)
    np.testing.assert_allclose(
        totals, 1.0, atol=1e-12, err_msg="Initial x_RU probabilities must sum to 1"
    )


def test_initial_state_is_non_permissive(model):
    """At init, all probability mass must be in the (0,0,0,0) state."""
    assert model.x_RU[0, 0, 0, 0, :].min() > 0.999, (
        "All probability should start in state (0,0,0,0)"
    )
    # All other states should be zero
    other = model.x_RU.copy()
    other[0, 0, 0, 0, :] = 0.0
    assert other.max() < 1e-12, "No probability should be in other states at init"


def test_default_parameters_keys():
    """default_parameters() must contain all required keys."""
    p = RDQ20MF.default_parameters()
    required = [
        "dt_RU",
        "LA",
        "LM",
        "LB",
        "SL0",
        "mu",
        "gamma",
        "Q",
        "Kd0",
        "alphaKd",
        "Koff",
        "Kbasic",
        "r0",
        "alpha",
        "mu0_fP",
        "mu1_fP",
        "a_XB",
    ]
    for key in required:
        assert key in p, f"Missing parameter key: {key}"


# ---------------------------------------------------------------------------
# Mathematical / conservation tests
# ---------------------------------------------------------------------------


def test_probability_conservation_after_stepping(model, dt):
    """
    RU probability must remain conserved (sum = 1) after many Euler steps.
    This is the most critical mathematical invariant of the RU ODE.
    """
    Ca = np.full(5, 0.5)
    SL = np.full(5, 2.2)

    for _ in range(200):
        model.advance_step(dt, Ca, SL)

    totals = model.x_RU.sum(axis=(0, 1, 2, 3))
    np.testing.assert_allclose(
        totals, 1.0, atol=1e-5, err_msg="x_RU probability conservation violated after 200 steps"
    )


def test_xru_values_non_negative(model, dt):
    """After stepping, x_RU values must remain >= 0."""
    Ca = np.full(5, 1.0)
    SL = np.full(5, 2.0)

    for _ in range(500):
        model.advance_step(dt, Ca, SL)

    assert model.x_RU.min() >= -1e-9, f"x_RU has negative values: min = {model.x_RU.min()}"


def test_xrU_values_bounded_above(model, dt):
    """After stepping, no x_RU value should exceed 1."""
    Ca = np.full(5, 2.0)
    SL = np.full(5, 2.2)

    for _ in range(500):
        model.advance_step(dt, Ca, SL)

    assert model.x_RU.max() <= 1.0 + 1e-9, f"x_RU has values > 1: max = {model.x_RU.max()}"


# ---------------------------------------------------------------------------
# Physiological tests
# ---------------------------------------------------------------------------


def test_calcium_sensitivity(dt):
    """
    Higher calcium concentration must lead to higher permissivity.
    This is a fundamental physiological requirement.
    """
    model_high = RDQ20MF(num_cells=1)
    model_low = RDQ20MF(num_cells=1)

    SL = np.array([2.2])
    Ca_high = np.array([2.0])  # saturating Ca
    Ca_low = np.array([0.1])  # diastolic Ca

    for _ in range(1000):
        model_high.advance_step(dt, Ca_high, SL)
        model_low.advance_step(dt, Ca_low, SL)

    P_high = model_high.compute_permissivity()[0]
    P_low = model_low.compute_permissivity()[0]

    assert P_high > P_low, f"Ca sensitivity failed: P_high={P_high:.4f}, P_low={P_low:.4f}"


def test_active_tension_calcium_sensitivity(dt):
    """
    Active tension should increase with calcium after enough crossbridge cycling.
    Tests that the XB sub-model also responds to Ca correctly.
    """
    model_high = RDQ20MF(num_cells=1)
    model_low = RDQ20MF(num_cells=1)

    SL = np.array([2.2])
    dSL = np.array([0.0])
    Ca_high = np.array([3.0])
    Ca_low = np.array([0.1])

    # Need enough steps to populate XBs (at least freqXB steps)
    n_steps = model_high.freqXB * 100  # ~100 ms
    for _ in range(n_steps):
        model_high.advance_step(dt, Ca_high, SL, dSL)
        model_low.advance_step(dt, Ca_low, SL, dSL)

    Ta_high = model_high.get_active_tension()[0]
    Ta_low = model_low.get_active_tension()[0]

    assert Ta_high > Ta_low, (
        f"Active tension Ca sensitivity failed: Ta_high={Ta_high:.2f}, Ta_low={Ta_low:.2f}"
    )


def test_length_dependence(dt):
    """
    Permissivity should be higher at optimal SL (2.2 µm) than at short SL (1.5 µm).
    This reflects the Frank-Starling mechanism via the frac_SO overlap function.
    """
    model_opt = RDQ20MF(num_cells=1)
    model_short = RDQ20MF(num_cells=1)

    Ca = np.array([1.0])
    SL_opt = np.array([2.2])
    SL_short = np.array([1.5])

    for _ in range(2000):
        model_opt.advance_step(dt, Ca, SL_opt)
        model_short.advance_step(dt, Ca, SL_short)

    P_opt = model_opt.compute_permissivity()[0]
    P_short = model_short.compute_permissivity()[0]

    assert P_opt > P_short, f"Length dependence failed: P_opt={P_opt:.4f}, P_short={P_short:.4f}"


def test_zero_calcium_stays_inactive(dt):
    """With no calcium, model should remain in (or near) non-permissive state."""
    model = RDQ20MF(num_cells=1)
    SL = np.array([2.2])
    Ca = np.array([0.0])

    for _ in range(5000):
        model.advance_step(dt, Ca, SL)

    P = model.compute_permissivity()[0]
    # Without Ca, RUs can still be permissive (spontaneous), but should be very low
    assert P < 0.15, f"Permissivity too high with zero Ca: P={P:.4f}"


def test_active_tension_non_negative(dt):
    """Active tension must never be negative."""
    model = RDQ20MF(num_cells=3)
    SL = np.array([1.8, 2.0, 2.2])
    Ca = np.array([0.5, 1.0, 2.0])
    dSL = np.zeros(3)

    n_steps = model.freqXB * 50
    for _ in range(n_steps):
        model.advance_step(dt, Ca, SL, dSL)

    Ta = model.get_active_tension()
    assert np.all(Ta >= 0.0), f"Negative active tension: {Ta}"


# ---------------------------------------------------------------------------
# Vectorization tests
# ---------------------------------------------------------------------------


def test_vectorization_independence(dt):
    """
    Cells in a vectorized batch must evolve independently.
    A zero-Ca cell must not be contaminated by a high-Ca neighbor.
    """
    model = RDQ20MF(num_cells=2)
    Ca = np.array([0.0, 5.0])
    SL = np.array([2.2, 2.2])

    for _ in range(10000):
        model.advance_step(dt, Ca, SL)

    P = model.compute_permissivity()

    # Cell 0 (Ca=0): should remain near zero
    assert P[0] < 0.2, f"Cell 0 (Ca=0) has spuriously high permissivity: {P[0]:.4f}"

    # Cell 1 (Ca=5): should be significantly activated
    assert P[1] > 0.1, f"Cell 1 (Ca=5) failed to activate: {P[1]:.4f}"


def test_single_vs_batch_consistency(dt):
    """
    A single-cell model and the same cell in a multi-cell batch must give
    identical results (within floating-point tolerance).
    """
    Ca_vals = np.array([0.8, 1.5, 0.3])
    SL_vals = np.array([2.1, 2.2, 1.9])
    dSL_vals = np.zeros(3)

    # Three separate single-cell models
    singles = [RDQ20MF(num_cells=1) for _ in range(3)]

    # One three-cell batch model
    batch = RDQ20MF(num_cells=3)

    n_steps = 500
    for _ in range(n_steps):
        for i, m in enumerate(singles):
            m.advance_step(dt, Ca_vals[i : i + 1], SL_vals[i : i + 1], dSL_vals[i : i + 1])
        batch.advance_step(dt, Ca_vals, SL_vals, dSL_vals)

    P_batch = batch.compute_permissivity()
    for i, m in enumerate(singles):
        P_single = m.compute_permissivity()[0]
        np.testing.assert_allclose(
            P_batch[i],
            P_single,
            rtol=1e-10,
            err_msg=f"Cell {i}: batch P={P_batch[i]:.6f} != single P={P_single:.6f}",
        )


# ---------------------------------------------------------------------------
# Overlap / geometry tests
# ---------------------------------------------------------------------------


def test_frac_SO_at_reference_length():
    """frac_SO should equal 1 at the plateau SL range [2*LA-LB, 2*LA+LB]."""
    model = RDQ20MF(num_cells=1)
    p = model.p
    # At exactly the plateau (2*LA), frac_SO should be 1
    SL_plateau = np.array([2 * p["LA"]])
    frac = model._frac_SO(SL_plateau)
    np.testing.assert_allclose(
        frac, 1.0, atol=1e-10, err_msg=f"frac_SO should be 1 at SL=2*LA, got {frac}"
    )


def test_frac_SO_zero_below_actin():
    """frac_SO should be 0 when SL <= LA (no single-overlap possible)."""
    model = RDQ20MF(num_cells=1)
    p = model.p
    SL_below = np.array([p["LA"] * 0.9])
    frac = model._frac_SO(SL_below)
    np.testing.assert_allclose(
        frac, 0.0, atol=1e-10, err_msg=f"frac_SO should be 0 below LA, got {frac}"
    )


def test_frac_SO_range():
    """frac_SO must be in [0, 1] across physiological SL range."""
    model = RDQ20MF(num_cells=20)
    SL_range = np.linspace(1.3, 2.6, 20)
    frac = model._frac_SO(SL_range)
    assert np.all(frac >= -1e-10), f"frac_SO below 0: min={frac.min()}"
    assert np.all(frac <= 1.0 + 1e-10), f"frac_SO above 1: max={frac.max()}"


# ---------------------------------------------------------------------------
# Crossbridge tests
# ---------------------------------------------------------------------------


def test_xb_advances_after_freqXB_steps(dt):
    """x_XB should be non-zero after enough steps to trigger XB advance."""
    model = RDQ20MF(num_cells=1)
    Ca = np.array([2.0])
    SL = np.array([2.2])
    dSL = np.array([0.0])

    # Run for well past one freqXB period
    for _ in range(model.freqXB * 3 + 1):
        model.advance_step(dt, Ca, SL, dSL)

    # XB should have populated
    assert model.x_XB.sum() > 1e-8, f"x_XB still zero after {model.freqXB * 3} steps: {model.x_XB}"


def test_xb_velocity_dependence(dt):
    """
    Positive shortening velocity (SL decreasing) should reduce XB attachment
    compared to zero velocity, due to velocity-dependent detachment.
    """
    model_static = RDQ20MF(num_cells=1)
    model_shortening = RDQ20MF(num_cells=1)

    Ca = np.array([2.0])
    SL = np.array([2.2])
    dSL_zero = np.array([0.0])
    dSL_shortening = np.array([-0.5])  # shortening at 0.5 µm/s

    n_steps = model_static.freqXB * 200
    for _ in range(n_steps):
        model_static.advance_step(dt, Ca, SL, dSL_zero)
        model_shortening.advance_step(dt, Ca, SL, dSL_shortening)

    Ta_static = model_static.get_active_tension()[0]
    Ta_shortening = model_shortening.get_active_tension()[0]

    # Shortening velocity increases detachment rate, so Ta should be lower
    assert Ta_static >= Ta_shortening, (
        f"Velocity dependence failed: Ta_static={Ta_static:.2f}, Ta_shortening={Ta_shortening:.2f}"
    )


# ---------------------------------------------------------------------------
# Steps longer than dt_RU
# ---------------------------------------------------------------------------


def _transient_tension(dt: float, duration: float = 0.06) -> np.ndarray:
    """Ta after each step of length ``dt`` [s] through a calcium transient.

    Calcium [µM] rises to 2 at 20 ms and decays, and is held at its end-of-step
    value over each step, as a coupled solver hands it over.
    """
    model = RDQ20MF(num_cells=1)
    Ta = []
    for i in range(round(duration / dt)):
        t = (i + 1) * dt
        Ca = 0.1 + 1.9 * (t / 0.02) * np.exp(1.0 - t / 0.02)
        model.advance_step(dt, Ca, 2.2, dSL_vals=0.0)
        Ta.append(model.get_active_tension()[0])
    return np.array(Ta)


def test_long_step_is_substeps_at_dt_RU():
    """A 1 ms step is 40 sub-steps of dt_RU: bit for bit, with constant inputs."""
    long, short = RDQ20MF(num_cells=3), RDQ20MF(num_cells=3)
    Ca, SL, dSL = np.array([0.5, 1.0, 2.0]), np.full(3, 2.2), np.zeros(3)
    n = round(1e-3 / short.dt)
    for _ in range(30):
        long.advance_step(1e-3, Ca, SL, dSL)
        for _ in range(n):
            short.advance_step(short.dt, Ca, SL, dSL)

    np.testing.assert_array_equal(long.x_RU, short.x_RU)
    np.testing.assert_array_equal(long.x_XB, short.x_XB)
    np.testing.assert_array_equal(long.get_active_tension(), short.get_active_tension())


def test_long_steps_follow_the_dt_RU_solution():
    """Through a calcium transient, 1 ms steps stay within 2 % of stepping at dt_RU,
    and the difference shrinks with the step.

    Before ``advance_step`` sub-stepped, a 1 ms call took one explicit Euler step
    of 1 ms and advanced the crossbridges by 40 ms on every 40th call, so Ta
    stayed 0 for the first 40 ms and then overshot.
    """

    def every_2_ms(Ta: np.ndarray, dt: float) -> np.ndarray:
        # A grid all the step sizes share
        every = round(2e-3 / dt)
        return Ta[every - 1 :: every]

    dt_RU = RDQ20MF(num_cells=1).dt
    ref = _transient_tension(dt_RU)
    errors = {
        dt: np.max(np.abs(every_2_ms(_transient_tension(dt), dt) - every_2_ms(ref, dt_RU)))
        for dt in (2e-3, 1e-3, 5e-4)
    }

    assert errors[1e-3] < 0.02 * ref.max(), (errors, ref.max())
    assert errors[2e-3] > errors[1e-3] > errors[5e-4], errors


def test_step_that_is_not_a_multiple_of_dt_RU():
    """0.61 ms is 25 equal sub-steps: probability is conserved, and Ta after 61 ms
    is within 2 % of stepping at dt_RU."""
    model, ref = RDQ20MF(num_cells=2), RDQ20MF(num_cells=2)
    Ca, SL, dSL = np.array([1.0, 2.0]), np.full(2, 2.2), np.zeros(2)
    for _ in range(100):
        model.advance_step(0.61e-3, Ca, SL, dSL)
    for _ in range(round(61e-3 / ref.dt)):
        ref.advance_step(ref.dt, Ca, SL, dSL)

    np.testing.assert_allclose(model.x_RU.sum(axis=(0, 1, 2, 3)), 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(model.get_active_tension(), ref.get_active_tension(), rtol=0.02)


# ---------------------------------------------------------------------------
# API / interface tests
# ---------------------------------------------------------------------------


def test_advance_step_scalar_inputs(dt):
    """advance_step must accept scalar Ca and SL without crashing."""
    model = RDQ20MF(num_cells=1)
    model.advance_step(dt, 1.0, 2.2)  # pure scalars
    assert model.x_RU.sum(axis=(0, 1, 2, 3))[0] > 0.9


def test_reset(dt):
    """reset() must restore the model to its initial state."""
    model = RDQ20MF(num_cells=3)
    Ca = np.full(3, 1.0)
    SL = np.full(3, 2.2)

    for _ in range(1000):
        model.advance_step(dt, Ca, SL)

    model.reset()

    # State should be back to initial
    np.testing.assert_allclose(
        model.x_RU[0, 0, 0, 0, :],
        1.0,
        atol=1e-12,
        err_msg="After reset, x_RU[0,0,0,0,:] should be 1",
    )
    np.testing.assert_allclose(
        model.x_XB, 0.0, atol=1e-12, err_msg="After reset, x_XB should be zero"
    )
    assert model._step_count == 0


def test_get_active_tension_shape(model, dt):
    """get_active_tension must return an array of shape (num_cells,)."""
    Ca = np.full(model.num_cells, 1.0)
    SL = np.full(model.num_cells, 2.2)
    dSL = np.zeros(model.num_cells)

    for _ in range(model.freqXB * 5):
        model.advance_step(dt, Ca, SL, dSL)

    Ta = model.get_active_tension()
    assert Ta.shape == (model.num_cells,), (
        f"get_active_tension shape {Ta.shape} != ({model.num_cells},)"
    )


def test_compute_permissivity_shape(model, dt):
    """compute_permissivity must return an array of shape (num_cells,)."""
    Ca = np.full(model.num_cells, 1.0)
    SL = np.full(model.num_cells, 2.2)
    model.advance_step(dt, Ca, SL)
    P = model.compute_permissivity()
    assert P.shape == (model.num_cells,), (
        f"compute_permissivity shape {P.shape} != ({model.num_cells},)"
    )


def test_param_override():
    """Custom params dict should override defaults."""
    model = RDQ20MF(num_cells=1, params={"Koff": 200.0})
    assert model.p["Koff"] == 200.0
    # Other keys should remain at defaults
    assert model.p["gamma"] == RDQ20MF.default_parameters()["gamma"]
