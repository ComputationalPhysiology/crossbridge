"""
Tests for `get_active_stiffness()` across all models.

Active stiffness Ka = d(dTa/dt)/d(dLambda/dt) is what lets a caller couple
these models to a tissue mechanics solver stably -- either as the
stabilization term of Regazzoni & Quarteroni (2020) Eq. (19), or as the
Newton tangent of a monolithic coupling. A Ka that is merely *plausible* is
worse than useless there: it will not crash, it will quietly degrade
convergence and bias the transient.

So the load-bearing test here is `test_matches_finite_difference`, which
never trusts the closed-form expression. It advances an activated model twice
from an identical state with two different shortening velocities and checks
that the resulting divergence in Ta is what Ka predicts, to first order in dt.
That catches sign errors, missing factors, unit slips, and an As/Aw mix-up --
and it keeps catching them if someone edits the ODEs and forgets Ka.

Test philosophy mirrors the per-model suites:
  - Mathematical: the finite-difference identity, and convergence order
  - Structural: shape, non-negativity, registry coverage
  - Physiological: Ka rises with activation; RDQ18 has none
  - API: scalar/array inputs, vectorization, reset
"""

import copy

import numpy as np
import pytest

from crossbridge import MODEL_REGISTRY, RDQ18, Land2017, Lewalle2024, RDQ20MF

# Models with genuine strain-rate feedback, for which R&Q Eq. (42) is exact.
VELOCITY_DEPENDENT = [Land2017, Lewalle2024]

N_CELLS = 4
CA_ACTIVE = 2.0  # [uM], well above the diastolic level -- drives S, W up
SL_TEST = 2.0  # [um]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _activate(model, dt=1e-3, n_steps=60, Ca=CA_ACTIVE, SL=SL_TEST):
    """Drive a model to a genuinely activated state.

    Without this, S = W = 0 and Ka = 0, so every assertion below would hold
    vacuously. Returns the model for chaining.
    """
    for _ in range(n_steps):
        model.advance_step(dt, Ca, SL, dSL_vals=0.0)
    return model


def _finite_difference_Ka(model, dt, dSL_a=-0.10, dSL_b=0.10, SL=SL_TEST, Ca=CA_ACTIVE):
    """Estimate Ka by advancing two copies of `model` at different velocities.

    Over one step of size dt, holding everything else fixed,

        Ta(dLambda_b) - Ta(dLambda_a) = dt * Ka * (dLambda_b - dLambda_a) + O(dt^2)

    so dividing through recovers Ka to first order. Velocities are supplied in
    um/s and converted to the model's dimensionless Lambda-rate by SL0.
    """
    a, b = copy.deepcopy(model), copy.deepcopy(model)
    a.advance_step(dt, Ca, SL, dSL_vals=dSL_a)
    b.advance_step(dt, Ca, SL, dSL_vals=dSL_b)

    dLambda_rate = (dSL_b - dSL_a) / model.p["SL0"]
    return (b.get_active_tension() - a.get_active_tension()) / (dt * dLambda_rate)


# ---------------------------------------------------------------------------
# The central test: Ka is the derivative it claims to be
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", VELOCITY_DEPENDENT)
def test_matches_finite_difference(ModelClass):
    """Ka must equal the measured sensitivity of dTa/dt to shortening velocity."""
    model = _activate(ModelClass(num_cells=N_CELLS))

    Ka = model.get_active_stiffness()
    assert np.all(Ka > 0.0), "test state is not activated; the check would be vacuous"

    Ka_fd = _finite_difference_Ka(model, dt=1e-5)

    rel_err = np.abs(Ka_fd - Ka) / np.abs(Ka)
    assert np.all(rel_err < 1e-2), (
        f"{ModelClass.__name__}: analytic Ka={Ka} disagrees with "
        f"finite-difference Ka={Ka_fd} (relative error {rel_err})"
    )


@pytest.mark.parametrize("ModelClass", VELOCITY_DEPENDENT)
def test_finite_difference_converges_first_order(ModelClass):
    """The finite-difference error must shrink as dt does.

    Agreement at one dt could be a coincidence of tolerance; a decreasing
    error sequence is evidence that Ka is the actual derivative rather than
    merely a nearby number.
    """
    model = _activate(ModelClass(num_cells=N_CELLS))
    Ka = model.get_active_stiffness()

    errors = [
        float(np.max(np.abs(_finite_difference_Ka(model, dt=dt) - Ka))) for dt in (4e-4, 2e-4, 1e-4)
    ]

    assert errors[0] > errors[1] > errors[2], (
        f"{ModelClass.__name__}: finite-difference error did not decrease under "
        f"refinement: {errors}"
    )


def test_rdq20mf_matches_finite_difference():
    """RDQ20MF's Ka (R&Q Eq. 52) verified against its crossbridge sub-step.

    Two deviations from the tests above, both deliberate:

    - `alpha=0` removes the velocity-dependent detachment rate, the one term
      Eq. (52) omits (see `get_active_stiffness`). With it, the formula is
      exact and can be checked tightly rather than to a fudged tolerance.
    - The finite difference is taken across `_XB_advance` rather than
      `advance_step`, because velocity only enters the model there, and
      `advance_step` reaches it just once every `freqXB` calls.
    """
    model = RDQ20MF(num_cells=N_CELLS, params={"alpha": 0.0})
    for _ in range(400):
        model.advance_step(model.dt, CA_ACTIVE, SL_TEST, dSL_vals=0.0)

    Ka = model.get_active_stiffness()
    assert np.all(Ka > 0.0), "test state has no attached crossbridges"

    dt_xb, dSL_a, dSL_b = 1e-5, -0.10, 0.10
    a, b = copy.deepcopy(model), copy.deepcopy(model)
    a._XB_advance(np.full(N_CELLS, dSL_a), dt_xb)
    b._XB_advance(np.full(N_CELLS, dSL_b), dt_xb)

    dLambda_rate = (dSL_b - dSL_a) / model.p["SL0"]
    Ka_fd = (b.get_active_tension() - a.get_active_tension()) / (dt_xb * dLambda_rate)

    rel_err = np.abs(Ka_fd - Ka) / np.abs(Ka)
    assert np.all(rel_err < 1e-2), (
        f"analytic Ka={Ka} disagrees with finite-difference Ka={Ka_fd} (relative error {rel_err})"
    )


# ---------------------------------------------------------------------------
# Structural / contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,ModelClass", sorted(MODEL_REGISTRY.items()))
def test_every_registered_model_implements_it(name, ModelClass):
    """The whole point of the shared interface is that a caller can rely on it."""
    model = ModelClass(num_cells=N_CELLS)
    Ka = model.get_active_stiffness()

    assert isinstance(Ka, np.ndarray), f"{name} returned {type(Ka)}, expected ndarray"
    assert Ka.shape == (N_CELLS,), f"{name} returned shape {Ka.shape}, expected ({N_CELLS},)"
    assert np.all(np.isfinite(Ka)), f"{name} returned non-finite values: {Ka}"


@pytest.mark.parametrize("name,ModelClass", sorted(MODEL_REGISTRY.items()))
def test_non_negative(name, ModelClass):
    """Attached crossbridges are springs; a negative stiffness would be an
    energy source and would break the stabilized scheme's stability proof."""
    model = _activate(ModelClass(num_cells=N_CELLS))
    Ka = model.get_active_stiffness()
    assert np.all(Ka >= 0.0), f"{name} produced negative active stiffness: {Ka}"


@pytest.mark.parametrize("name,ModelClass", sorted(MODEL_REGISTRY.items()))
def test_matches_active_tension_shape(name, ModelClass):
    """Ka is consumed alongside Ta in the same expression, so the two must be
    elementwise-compatible for every model."""
    model = _activate(ModelClass(num_cells=N_CELLS))
    assert model.get_active_stiffness().shape == model.get_active_tension().shape


def test_rdq18_is_exactly_zero():
    """RDQ18 has no strain-rate feedback at all -- see its docstring warning."""
    model = _activate(RDQ18(num_cells=N_CELLS))
    assert np.all(model.get_active_tension() > 0.0), "model failed to activate"
    np.testing.assert_array_equal(model.get_active_stiffness(), np.zeros(N_CELLS))


# ---------------------------------------------------------------------------
# Physiological behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", VELOCITY_DEPENDENT + [RDQ20MF])
def test_zero_at_rest_and_positive_when_activated(ModelClass):
    """A quiescent model has no attached crossbridges and hence no stiffness."""
    model = ModelClass(num_cells=N_CELLS)
    np.testing.assert_allclose(model.get_active_stiffness(), 0.0, atol=1e-12)

    _activate(model)
    assert np.all(model.get_active_stiffness() > 0.0)


@pytest.mark.parametrize("ModelClass", VELOCITY_DEPENDENT)
def test_increases_with_activation(ModelClass):
    """More calcium recruits more crossbridges, so more stiffness."""
    low = _activate(ModelClass(num_cells=N_CELLS), Ca=0.5)
    high = _activate(ModelClass(num_cells=N_CELLS), Ca=5.0)
    assert np.all(high.get_active_stiffness() > low.get_active_stiffness())


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", VELOCITY_DEPENDENT + [RDQ20MF])
def test_vectorization_independence(ModelClass):
    """Cells driven by different calcium must get independent stiffnesses,
    and each must match the same cell run on its own."""
    Ca = np.array([0.2, 1.0, 3.0, 6.0])
    batch = ModelClass(num_cells=4)
    for _ in range(60):
        batch.advance_step(1e-3, Ca, SL_TEST, dSL_vals=0.0)

    for i, ca in enumerate(Ca):
        single = ModelClass(num_cells=1)
        for _ in range(60):
            single.advance_step(1e-3, np.array([ca]), SL_TEST, dSL_vals=0.0)
        np.testing.assert_allclose(
            batch.get_active_stiffness()[i],
            single.get_active_stiffness()[0],
            rtol=1e-10,
            err_msg=f"cell {i} (Ca={ca}) differs between batch and single-cell runs",
        )


@pytest.mark.parametrize("name,ModelClass", sorted(MODEL_REGISTRY.items()))
def test_reset_restores_resting_stiffness(name, ModelClass):
    model = ModelClass(num_cells=N_CELLS)
    resting = model.get_active_stiffness().copy()
    _activate(model)
    model.reset()
    np.testing.assert_allclose(model.get_active_stiffness(), resting, atol=1e-12)
