"""
Unit tests for the Lewalle2024 model.

Test philosophy mirrors test_rdq20mf.py:
  - Structural: shape, initialization
  - Mathematical: state conservation, state bounds
  - Physiological: Ca sensitivity, length-dependent activation (the paper's
    core claim for the default "totalforce" feedback paradigm)
  - Computational: vectorization independence, including specifically the
    mixed-CaTRPN-scale regression that once broke batched matrix
    exponentials for this model
  - API: scalar inputs, reset, param overrides, invalid config
"""

import numpy as np
import pytest

from crossbridge import Lewalle2024


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model():
    """Fresh 5-cell Lewalle2024 instance."""
    return Lewalle2024(num_cells=5)


@pytest.fixture
def dt(model):
    return model.dt


def _conservation_residual(model):
    U = 1.0 - model.B - model.S - model.W - model.BE - model.UE
    return model.B + model.S + model.W + model.BE + model.UE + U


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


def test_initialization_shapes(model):
    for attr in ("CaTRPN", "B", "S", "W", "Zs", "Zw", "Cd", "BE", "UE"):
        arr = getattr(model, attr)
        assert arr.shape == (5,), f"{attr} shape mismatch: {arr.shape}"


def test_initial_state_is_quiescent(model):
    """At init, no crossbridges are formed and CaTRPN is at its own steady state."""
    for attr in ("B", "S", "W", "Zs", "Zw", "Cd", "BE", "UE"):
        arr = getattr(model, attr)
        np.testing.assert_allclose(arr, 0.0, atol=1e-12, err_msg=f"{attr} should start at 0")
    assert np.all(model.CaTRPN > 0.0), "CaTRPN should be strictly positive at init"
    assert np.all(model.CaTRPN < 1.0)


def test_default_parameters_keys():
    p = Lewalle2024.default_parameters()
    required = [
        "dt",
        "SL0",
        "a",
        "b",
        "k",
        "eta_l",
        "eta_s",
        "k_trpn_on",
        "k_trpn_off",
        "ntrpn",
        "pCa50ref",
        "beta1",
        "ku",
        "nTm",
        "trpn50",
        "kuw",
        "kws",
        "rw",
        "rs",
        "gs",
        "es",
        "gw",
        "phi",
        "Aeff",
        "beta0",
        "Tref",
        "k1",
        "k2",
        "ra",
        "rb",
        "koffon",
        "which_dep",
        "dep_k1ork2",
        "Ca0",
    ]
    for key in required:
        assert key in p, f"Missing parameter key: {key}"


# ---------------------------------------------------------------------------
# Mathematical / conservation tests
# ---------------------------------------------------------------------------


def test_state_conservation_after_stepping(model, dt):
    """B + S + W + Boff + Uoff + U must remain 1 (up to the [0,1] clip)."""
    Ca = np.full(5, 1.0)
    SL = np.full(5, 1.9)
    for _ in range(100):
        model.advance_step(dt, Ca, SL)
    np.testing.assert_allclose(
        _conservation_residual(model), 1.0, atol=1e-8, err_msg="State conservation violated"
    )


def test_states_bounded(model, dt):
    """B, S, W, Boff, Uoff must stay within [0, 1] after stepping."""
    Ca = np.full(5, 3.0)
    SL = np.full(5, 2.0)
    for _ in range(300):
        model.advance_step(dt, Ca, SL)
    for attr in ("B", "S", "W", "BE", "UE"):
        arr = getattr(model, attr)
        assert np.all(arr >= -1e-9), f"{attr} has negative values: {arr}"
        assert np.all(arr <= 1.0 + 1e-9), f"{attr} exceeds 1: {arr}"


# ---------------------------------------------------------------------------
# Physiological tests
# ---------------------------------------------------------------------------


def test_calcium_sensitivity(dt):
    """Higher calcium must lead to higher active tension."""
    model_high = Lewalle2024(num_cells=1)
    model_low = Lewalle2024(num_cells=1)

    SL = np.array([1.9])
    Ca_high = np.array([3.0])
    Ca_low = np.array([0.3])

    for _ in range(200):
        model_high.advance_step(dt, Ca_high, SL)
        model_low.advance_step(dt, Ca_low, SL)

    Ta_high = model_high.get_active_tension()[0]
    Ta_low = model_low.get_active_tension()[0]
    assert Ta_high > Ta_low, f"Ca sensitivity failed: Ta_high={Ta_high:.6f}, Ta_low={Ta_low:.6f}"


def test_length_dependent_activation(dt):
    """
    Active tension must increase with sarcomere length for the default
    "totalforce" feedback paradigm (paradigm A) -- the paper's central
    finding: myosin OFF-state feedback on total force reproduces the
    Frank-Starling length dependence.
    """
    Ca = np.array([3.0])
    Ta_by_SL = {}
    for SL in (1.9, 2.0, 2.1):
        m = Lewalle2024(num_cells=1)
        for _ in range(1000):
            m.advance_step(dt, Ca, np.array([SL]))
        Ta_by_SL[SL] = m.get_active_tension()[0]

    assert Ta_by_SL[1.9] < Ta_by_SL[2.0] < Ta_by_SL[2.1], (
        f"Length-dependent activation failed: {Ta_by_SL}"
    )


def test_active_tension_non_negative(model, dt):
    Ca = np.array([0.3, 1.0, 3.0, 0.5, 2.0])
    SL = np.array([1.8, 1.9, 2.0, 2.1, 1.95])
    for _ in range(200):
        model.advance_step(dt, Ca, SL)
    Ta = model.get_active_tension()
    assert np.all(Ta >= 0.0), f"Negative active tension: {Ta}"


# ---------------------------------------------------------------------------
# Vectorization tests
# ---------------------------------------------------------------------------


def test_vectorization_independence(dt):
    """A resting (low Ca) cell must not be contaminated by an active neighbor."""
    model = Lewalle2024(num_cells=2)
    Ca = np.array([0.1, 3.0])
    SL = np.array([1.9, 1.9])
    for _ in range(300):
        model.advance_step(dt, Ca, SL)

    Ta = model.get_active_tension()
    assert Ta[0] < 0.01, f"Cell 0 (low Ca) has spuriously high tension: {Ta[0]:.4f}"
    assert Ta[1] > 0.05, f"Cell 1 (high Ca) failed to activate: {Ta[1]:.4f}"


def test_single_vs_batch_consistency_mixed_scale(dt):
    """
    A single-cell model and the same cell within a multi-cell batch spanning
    a wide range of Ca (and hence CaTRPN, which can differ across cells by
    orders of magnitude) must give identical results.

    Regression test: scipy.linalg.expm's batched mode was found to silently
    return wrong results for some matrices when a stack mixes very different
    norms, which is exactly what a mixed-Ca cell batch produces here.
    """
    Ca_vals = np.array([0.1, 1.0, 3.0])
    SL_vals = np.full(3, 1.9)

    singles = [Lewalle2024(num_cells=1) for _ in range(3)]
    batch = Lewalle2024(num_cells=3)

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
    model = Lewalle2024(num_cells=1)
    model.advance_step(dt, 1.0, 1.9)  # pure scalars, no crash
    assert model.CaTRPN[0] > 0.0


def test_no_spurious_velocity_on_first_call(dt):
    """
    The first advance_step call (no dSL_vals, no prior SL) must not
    finite-difference a velocity from the arbitrary post-reset SL0 default:
    Zw/Zs (driven purely by dLambda/dt) must stay exactly 0 after one step
    regardless of how far the first SL is from SL0.
    """
    model = Lewalle2024(num_cells=1)
    model.advance_step(dt, 1.0, np.array([2.1]))  # far from SL0=1.8
    assert model.Zw[0] == 0.0
    assert model.Zs[0] == 0.0


def test_reset(dt):
    model = Lewalle2024(num_cells=3)
    Ca = np.full(3, 2.0)
    SL = np.full(3, 2.0)
    for _ in range(200):
        model.advance_step(dt, Ca, SL)

    model.reset()

    for attr in ("B", "S", "W", "Zs", "Zw", "Cd", "BE", "UE"):
        np.testing.assert_allclose(
            getattr(model, attr), 0.0, atol=1e-12, err_msg=f"After reset, {attr} should be 0"
        )
    assert np.all(model.CaTRPN > 0.0)
    assert model._has_prev_step is False


def test_get_tension_shapes(model, dt):
    Ca = np.full(5, 1.0)
    SL = np.full(5, 1.9)
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
    model = Lewalle2024(num_cells=1, params={"k1": 2.0})
    assert model.p["k1"] == 2.0
    assert model.p["gw"] == Lewalle2024.default_parameters()["gw"]


def test_invalid_which_dep_raises():
    with pytest.raises(ValueError):
        Lewalle2024(num_cells=1, params={"which_dep": "not-a-real-paradigm"})


def test_invalid_dep_k1ork2_raises():
    with pytest.raises(ValueError):
        Lewalle2024(num_cells=1, params={"dep_k1ork2": "k3"})


def test_missing_koffon_raises_when_switching_paradigm():
    with pytest.raises(ValueError):
        Lewalle2024(num_cells=1, params={"which_dep": "force", "koffon": None})
