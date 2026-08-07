"""
Unit tests for the Land2017 model.

Test philosophy mirrors test_lewalle2024.py (this model is the un-amended
base Land et al. (2017) model that Lewalle2024 extends with OFF-state
dynamics in place of the beta0/beta1 ad hoc terms tested here):
  - Structural: shape, initialization
  - Mathematical: state conservation, state bounds
  - Physiological: Ca sensitivity, and the ad hoc length-dependent activation
    (LDA) via beta0 (max force) / beta1 (Ca sensitivity) -- this model's
    defining feature, later replaced by OFF-state feedback in Lewalle2024
  - Computational: vectorization independence
  - API: scalar inputs, reset, param overrides
"""

import numpy as np
import pytest

from crossbridge import Land2017


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model():
    """Fresh 5-cell Land2017 instance."""
    return Land2017(num_cells=5)


@pytest.fixture
def dt(model):
    return model.dt


def _conservation_residual(model):
    U = 1.0 - model.B - model.S - model.W
    return model.B + model.S + model.W + U


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


def test_initialization_shapes(model):
    for attr in ("CaTRPN", "B", "S", "W", "Zs", "Zw", "Cd"):
        arr = getattr(model, attr)
        assert arr.shape == (5,), f"{attr} shape mismatch: {arr.shape}"


def test_initial_state_is_quiescent(model):
    """At init, no crossbridges are formed and CaTRPN is at its own steady state."""
    for attr in ("B", "S", "W", "Zs", "Zw", "Cd"):
        arr = getattr(model, attr)
        np.testing.assert_allclose(arr, 0.0, atol=1e-12, err_msg=f"{attr} should start at 0")
    assert np.all(model.CaTRPN > 0.0), "CaTRPN should be strictly positive at init"
    assert np.all(model.CaTRPN < 1.0)


def test_default_parameters_keys():
    p = Land2017.default_parameters()
    required = [
        "dt",
        "SL0",
        "a",
        "b",
        "k",
        "eta_l",
        "eta_s",
        "k_trpn",
        "ntrpn",
        "ca50_ref",
        "ku",
        "nTm",
        "trpn50",
        "kuw",
        "kws",
        "rw",
        "rs",
        "gs",
        "gw",
        "phi",
        "Aeff",
        "beta0",
        "beta1",
        "Tref",
        "Ca0",
    ]
    for key in required:
        assert key in p, f"Missing parameter key: {key}"


def test_default_parameters_match_paper_skinned_table():
    """
    Spot-check a handful of values against Table B ("Skinned model value")
    of Land et al. (2017), after converting the paper's ms^-1 rate constants
    to this package's s^-1 convention (x1000).
    """
    p = Land2017.default_parameters()
    assert p["a"] == pytest.approx(2100.0)  # 2.1 kPa -> Pa
    assert p["b"] == pytest.approx(9.1)
    assert p["k"] == pytest.approx(7.0)
    assert p["ku"] == pytest.approx(1000.0)  # 1/ms -> 1000/s
    assert p["trpn50"] == pytest.approx(0.35)
    assert p["rw"] == pytest.approx(0.5)
    assert p["rs"] == pytest.approx(0.25)
    assert p["beta0"] == pytest.approx(2.3)
    assert p["beta1"] == pytest.approx(-2.4)
    assert p["Tref"] == pytest.approx(40500.0)  # 40.5 kPa -> Pa


# ---------------------------------------------------------------------------
# Mathematical / conservation tests
# ---------------------------------------------------------------------------


def test_state_conservation_after_stepping(model, dt):
    """B + S + W + U must remain 1 (up to the [0,1] clip)."""
    Ca = np.full(5, 3.0)
    SL = np.full(5, 1.9)
    for _ in range(200):
        model.advance_step(dt, Ca, SL)
    np.testing.assert_allclose(
        _conservation_residual(model), 1.0, atol=1e-8, err_msg="State conservation violated"
    )


def test_states_bounded(model, dt):
    """B, S, W must stay within [0, 1] after stepping."""
    Ca = np.full(5, 10.0)
    SL = np.full(5, 2.0)
    for _ in range(300):
        model.advance_step(dt, Ca, SL)
    for attr in ("B", "S", "W"):
        arr = getattr(model, attr)
        assert np.all(arr >= -1e-9), f"{attr} has negative values: {arr}"
        assert np.all(arr <= 1.0 + 1e-9), f"{attr} exceeds 1: {arr}"


# ---------------------------------------------------------------------------
# Physiological tests
# ---------------------------------------------------------------------------


def test_calcium_sensitivity(dt):
    """Higher calcium must lead to higher active tension."""
    model_high = Land2017(num_cells=1)
    model_low = Land2017(num_cells=1)

    SL = np.array([2.0])
    Ca_high = np.array([10.0])
    Ca_low = np.array([0.3])

    for _ in range(400):
        model_high.advance_step(dt, Ca_high, SL)
        model_low.advance_step(dt, Ca_low, SL)

    Ta_high = model_high.get_active_tension()[0]
    Ta_low = model_low.get_active_tension()[0]
    assert Ta_high > Ta_low, f"Ca sensitivity failed: Ta_high={Ta_high:.6f}, Ta_low={Ta_low:.6f}"


def test_length_dependent_max_force():
    """
    beta0 > 0 (the default): at saturating calcium, active tension must
    increase with sarcomere length (the paper's Fig. 4 finding: maximum
    force increases with SL).
    """
    Ca = np.array([10.0])
    Ta_by_SL = {}
    for SL in (1.8, 2.0, 2.2):
        m = Land2017(num_cells=1)
        dt = m.dt
        for _ in range(600):
            m.advance_step(dt, Ca, np.array([SL]))
        Ta_by_SL[SL] = m.get_active_tension()[0]

    assert Ta_by_SL[1.8] < Ta_by_SL[2.0] < Ta_by_SL[2.2], (
        f"Length-dependent max force failed: {Ta_by_SL}"
    )


def test_length_dependent_calcium_sensitivity():
    """
    beta1 < 0 (the default): at submaximal calcium, longer SL must produce
    higher active tension than shorter SL (a left-shifted pCa50, the
    paper's Fig. 4 calcium-sensitivity finding).
    """
    Ca = np.array([0.6])  # submaximal
    Ta_by_SL = {}
    for SL in (1.8, 2.2):
        m = Land2017(num_cells=1)
        dt = m.dt
        for _ in range(600):
            m.advance_step(dt, Ca, np.array([SL]))
        Ta_by_SL[SL] = m.get_active_tension()[0]

    assert Ta_by_SL[1.8] < Ta_by_SL[2.2], f"Length-dependent calcium sensitivity failed: {Ta_by_SL}"


def test_no_lda_when_beta_zero():
    """Sanity check: zeroing beta0/beta1 must remove the SL dependence entirely."""
    Ca = np.array([0.6])
    Ta_by_SL = {}
    for SL in (1.8, 2.2):
        m = Land2017(num_cells=1, params={"beta0": 0.0, "beta1": 0.0})
        dt = m.dt
        for _ in range(600):
            m.advance_step(dt, Ca, np.array([SL]))
        Ta_by_SL[SL] = m.get_active_tension()[0]

    np.testing.assert_allclose(
        Ta_by_SL[1.8],
        Ta_by_SL[2.2],
        rtol=1e-6,
        err_msg=f"Active tension should be SL-independent when beta0=beta1=0: {Ta_by_SL}",
    )


def test_active_tension_non_negative(model, dt):
    Ca = np.array([0.3, 1.0, 3.0, 0.5, 2.0])
    SL = np.array([1.8, 1.9, 2.0, 2.1, 1.95])
    for _ in range(200):
        model.advance_step(dt, Ca, SL)
    Ta = model.get_active_tension()
    assert np.all(Ta >= 0.0), f"Negative active tension: {Ta}"


def test_passive_tension_increases_with_stretch(dt):
    """F1 = a*(exp(b*(lambda-1))-1) must be monotonically increasing with SL."""
    Tp_by_SL = {}
    for SL in (1.8, 2.0, 2.2, 2.4):
        m = Land2017(num_cells=1)
        for _ in range(50):
            m.advance_step(dt, 0.0, np.array([SL]))
        Tp_by_SL[SL] = m.get_passive_tension()[0]

    sls = sorted(Tp_by_SL)
    values = [Tp_by_SL[sl] for sl in sls]
    assert all(x < y for x, y in zip(values, values[1:])), (
        f"Passive tension should increase monotonically with SL: {Tp_by_SL}"
    )


# ---------------------------------------------------------------------------
# Vectorization tests
# ---------------------------------------------------------------------------


def test_vectorization_independence(dt):
    """A resting (low Ca) cell must not be contaminated by an active neighbor."""
    model = Land2017(num_cells=2)
    Ca = np.array([0.1, 10.0])
    SL = np.array([2.0, 2.0])
    for _ in range(400):
        model.advance_step(dt, Ca, SL)

    Ta = model.get_active_tension()
    assert Ta[0] < 0.01, f"Cell 0 (low Ca) has spuriously high tension: {Ta[0]:.4f}"
    assert Ta[1] > 0.05, f"Cell 1 (high Ca) failed to activate: {Ta[1]:.4f}"


def test_single_vs_batch_consistency(dt):
    """A single-cell model and the same cell within a multi-cell batch must agree."""
    Ca_vals = np.array([0.1, 1.0, 10.0])
    SL_vals = np.full(3, 2.0)

    singles = [Land2017(num_cells=1) for _ in range(3)]
    batch = Land2017(num_cells=3)

    n_steps = 200
    for _ in range(n_steps):
        for i, m in enumerate(singles):
            m.advance_step(dt, Ca_vals[i : i + 1], SL_vals[i : i + 1])
        batch.advance_step(dt, Ca_vals, SL_vals)

    Ta_batch = batch.get_active_tension()
    for i, m in enumerate(singles):
        Ta_single = m.get_active_tension()[0]
        np.testing.assert_allclose(
            Ta_batch[i],
            Ta_single,
            rtol=1e-8,
            atol=1e-12,
            err_msg=f"Cell {i}: batch Ta={Ta_batch[i]:.8f} != single Ta={Ta_single:.8f}",
        )


# ---------------------------------------------------------------------------
# API / interface tests
# ---------------------------------------------------------------------------


def test_advance_step_scalar_inputs(dt):
    model = Land2017(num_cells=1)
    model.advance_step(dt, 1.0, 2.0)  # pure scalars, no crash
    assert model.CaTRPN[0] > 0.0


def test_no_spurious_velocity_on_first_call(dt):
    """
    The first advance_step call (no dSL_vals, no prior SL) must not
    finite-difference a velocity from the arbitrary post-reset SL0 default:
    Zw/Zs (driven purely by dLambda/dt) must stay exactly 0 after one step
    regardless of how far the first SL is from SL0.
    """
    model = Land2017(num_cells=1)
    model.advance_step(dt, 1.0, np.array([2.3]))  # far from SL0=1.9
    assert model.Zw[0] == 0.0
    assert model.Zs[0] == 0.0


def test_reset(dt):
    model = Land2017(num_cells=3)
    Ca = np.full(3, 2.0)
    SL = np.full(3, 2.0)
    for _ in range(200):
        model.advance_step(dt, Ca, SL)

    model.reset()

    for attr in ("B", "S", "W", "Zs", "Zw", "Cd"):
        np.testing.assert_allclose(
            getattr(model, attr), 0.0, atol=1e-12, err_msg=f"After reset, {attr} should be 0"
        )
    assert np.all(model.CaTRPN > 0.0)
    assert model._has_prev_step is False


def test_get_tension_shapes(model, dt):
    Ca = np.full(5, 1.0)
    SL = np.full(5, 2.0)
    model.advance_step(dt, Ca, SL)
    assert model.get_active_tension().shape == (5,)
    assert model.get_passive_tension().shape == (5,)
    assert model.get_total_tension().shape == (5,)
    assert model.compute_attached_fraction().shape == (5,)

    Ta = model.get_active_tension()
    Tp = model.get_passive_tension()
    Ttot = model.get_total_tension()
    np.testing.assert_allclose(Ttot, Ta + Tp, rtol=1e-10)


def test_param_override():
    model = Land2017(num_cells=1, params={"Tref": 120000.0})
    assert model.p["Tref"] == 120000.0
    assert model.p["kuw"] == Land2017.default_parameters()["kuw"]
