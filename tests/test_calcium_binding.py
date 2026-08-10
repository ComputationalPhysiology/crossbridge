"""
Tests for `bound_calcium_fraction()` and `get_calcium_binding_rate()`.

These models bind cytosolic calcium. Coupling one to an electrophysiology
model means the EP side must have its own troponin buffer removed (or calcium
is buffered twice) and must instead receive the buffering flux back from here.
Get that flux wrong and nothing raises -- the calcium transient is simply the
wrong shape.

The load-bearing test is `test_reported_flux_conserves_calcium`: integrating
the reported rate over a beat must reproduce the total change in occupancy,
exactly, not approximately. That is the property a segregated coupling relies
on, and it is why the rate is defined as the mean over the step just taken
rather than an instantaneous derivative.

`test_matches_torord_troponin_kinetics` then checks the Land-family rate
against the ToR-ORd cell model's own `dCaTrpn_dt` expression, which is the
formula the coupling has to reproduce.
"""

import numpy as np
import pytest

from crossbridge import MODEL_REGISTRY, RDQ18, Land2017, Lewalle2024, RDQ20MF

N_CELLS = 3
SL_TEST = 2.0  # [um]


def _beat(model, duration=0.06, ca_peak=10.0, SL=SL_TEST):
    """Drive a model through a synthetic calcium transient.

    Steps at the model's own `dt`, which differs by ~40x across the registry
    (1e-3 s for the Land family, 2.5e-5 s for the RU-tensor models). Driving
    RDQ18/RDQ20MF at the Land-family step size takes their explicit RU
    integrator past its stability limit, and the probability tensor stops
    being normalized -- occupancy climbs above 1 and the reported rate is
    meaningless. That is a property of those integrators, not of the calcium
    bookkeeping under test here.

    `ca_peak` is deliberately supraphysiological so that every model, including
    Lewalle2024 with its Ca50 of ~5.6 uM, actually binds an appreciable amount.

    Yields (dt, Ca) per step so callers can accumulate whatever they need.
    """
    dt = model.dt
    for i in range(int(duration / dt)):
        # Rise then decay, so the binding rate changes sign during the beat
        t = i * dt
        Ca = 0.1 + (ca_peak - 0.1) * (t / 0.02) * np.exp(1.0 - t / 0.02)
        model.advance_step(dt, Ca, SL, dSL_vals=0.0)
        yield dt, Ca


def _hold(model, Ca, duration=0.06, SL=SL_TEST):
    """Hold a model at constant calcium, stepping at its own dt."""
    for _ in range(int(duration / model.dt)):
        model.advance_step(model.dt, Ca, SL, dSL_vals=0.0)


def _nudge(model, Ca, SL=SL_TEST):
    """Step just far enough for a change in calcium to register.

    RDQ20MF refreshes its Ca-dependent transition rates only every
    `freq_rates_update` steps (10, i.e. every 0.25 ms), so a single step after
    a change in calcium can still be integrating with the old rates. Stepping
    past that boundary keeps the test about the calcium bookkeeping rather
    than about that model's rate-refresh interval.
    """
    for _ in range(getattr(model, "freq_rates_update", 1) + 1):
        model.advance_step(model.dt, Ca, SL, dSL_vals=0.0)


# ---------------------------------------------------------------------------
# The property the coupling depends on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,ModelClass", sorted(MODEL_REGISTRY.items()))
def test_reported_flux_conserves_calcium(name, ModelClass):
    """Integral of the reported rate == total change in occupancy.

    If this drifts, a coupled simulation creates or destroys calcium every
    step. Because the rate is defined as the mean over the step, the identity
    is exact rather than merely first-order, so it is asserted tightly.
    """
    model = ModelClass(num_cells=N_CELLS)
    start = model.bound_calcium_fraction().copy()

    integrated = np.zeros(N_CELLS)
    for dt, _ in _beat(model):
        integrated += model.get_calcium_binding_rate() * dt

    change = model.bound_calcium_fraction() - start
    assert np.max(np.abs(change)) > 1e-3, f"{name}: beat did not move the occupancy"
    np.testing.assert_allclose(
        integrated,
        change,
        rtol=1e-10,
        atol=1e-12,
        err_msg=f"{name}: reported flux does not conserve calcium",
    )


def test_matches_torord_troponin_kinetics():
    """The Land-family rate must match ToR-ORd's own dCaTrpn_dt.

    ToR-ORd integrates

        dCaTrpn_dt = ktrpn * ((cai/cat50)**ntrpn * (1 - CaTrpn) - CaTrpn)

    (with cai converted to uM), and forms J_TRPN = dCaTrpn_dt * trpnmax. This
    is the expression the coupling has to reproduce, so check it directly
    rather than trusting that the internal solution is the same thing.

    Land2017 only: its `ca50_ref`, `k_trpn` and `ntrpn` *are* ToR-ORd's
    `cat50_ref`, `ktrpn` and `ntrpn`. Lewalle2024 reparameterizes the same
    kinetics in pCa with separate on/off rates, so there is no literal
    expression to compare against; it is covered by the conservation and
    dt-convergence tests instead.
    """
    model = Land2017(num_cells=N_CELLS)
    p = model.p
    Ca = 1.5

    # Advance a few steps so CaTRPN is away from its initial value
    for _ in range(20):
        model.advance_step(1e-3, Ca, SL_TEST, dSL_vals=0.0)

    CaTRPN = model.bound_calcium_fraction().copy()
    Lambda = SL_TEST / p["SL0"]

    Ca50 = max(p["ca50_ref"] + p["beta1"] * (min(Lambda, 1.2) - 1.0), 1e-6)
    kon = p["k_trpn"] * (Ca / Ca50) ** p["ntrpn"]
    koff = p["k_trpn"]
    expected = kon * (1.0 - CaTRPN) - koff * CaTRPN

    # Take one short step; the mean rate over it approaches the instantaneous
    # derivative as dt -> 0.
    dt = 1e-6
    model.advance_step(dt, Ca, SL_TEST, dSL_vals=0.0)
    rate = model.get_calcium_binding_rate()

    np.testing.assert_allclose(rate, expected, rtol=1e-4)


@pytest.mark.parametrize("ModelClass", [Land2017, Lewalle2024])
def test_rate_converges_to_instantaneous_derivative(ModelClass):
    """Mean-over-step and instantaneous derivative agree as dt -> 0.

    Confirms the mean-rate definition is a consistent approximation, not a
    different quantity.
    """
    import copy

    base = ModelClass(num_cells=N_CELLS)
    for _ in range(20):
        base.advance_step(1e-3, 1.5, SL_TEST, dSL_vals=0.0)

    rates = []
    for dt in (1e-4, 1e-5, 1e-6):
        m = copy.deepcopy(base)
        m.advance_step(dt, 1.5, SL_TEST, dSL_vals=0.0)
        rates.append(m.get_calcium_binding_rate())

    err = [float(np.max(np.abs(r - rates[-1]))) for r in rates[:-1]]
    assert err[0] > err[1], f"rate did not converge under dt refinement: {err}"


# ---------------------------------------------------------------------------
# Structural / contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,ModelClass", sorted(MODEL_REGISTRY.items()))
def test_occupancy_is_a_fraction(name, ModelClass):
    """Occupancy is dimensionless and bounded, for every model."""
    model = ModelClass(num_cells=N_CELLS)
    for _ in _beat(model):
        theta = model.bound_calcium_fraction()
        assert theta.shape == (N_CELLS,), f"{name}: shape {theta.shape}"
        assert np.all(theta >= -1e-12) and np.all(theta <= 1.0 + 1e-12), (
            f"{name}: occupancy left [0, 1]: min={theta.min()}, max={theta.max()}"
        )


@pytest.mark.parametrize("name,ModelClass", sorted(MODEL_REGISTRY.items()))
def test_zero_before_first_step_and_after_reset(name, ModelClass):
    """A rate reported before any step would be meaningless; it must be zero,
    and reset must not leave a stale one behind."""
    model = ModelClass(num_cells=N_CELLS)
    np.testing.assert_array_equal(model.get_calcium_binding_rate(), np.zeros(N_CELLS))

    for _ in _beat(model, duration=0.02):
        pass
    assert np.any(model.get_calcium_binding_rate() != 0.0)

    model.reset()
    np.testing.assert_array_equal(model.get_calcium_binding_rate(), np.zeros(N_CELLS))


@pytest.mark.parametrize("name,ModelClass", sorted(MODEL_REGISTRY.items()))
def test_binding_rate_follows_calcium(name, ModelClass):
    """Raising calcium must drive binding up; letting it fall must release."""
    model = ModelClass(num_cells=N_CELLS)

    # Equilibrate low, then step calcium up: the model must start binding.
    # Note the rate is checked right after the change, not after holding --
    # once the model re-equilibrates the rate correctly returns to zero.
    _hold(model, Ca=0.1)
    _nudge(model, Ca=10.0)
    assert np.all(model.get_calcium_binding_rate() > 0.0), f"{name}: not binding when Ca rises"

    # Equilibrate high, then drop calcium: the model must release.
    _hold(model, Ca=10.0)
    _nudge(model, Ca=0.01)
    assert np.all(model.get_calcium_binding_rate() < 0.0), f"{name}: not releasing when Ca falls"


@pytest.mark.parametrize("name,ModelClass", sorted(MODEL_REGISTRY.items()))
def test_vectorization_independence(name, ModelClass):
    """Cells at different calcium must bind independently, and each must match
    the same cell run alone."""
    Ca = np.array([0.2, 1.0, 4.0])
    batch = ModelClass(num_cells=3)
    _hold(batch, Ca)

    for i, ca in enumerate(Ca):
        single = ModelClass(num_cells=1)
        _hold(single, np.array([ca]))
        np.testing.assert_allclose(
            batch.get_calcium_binding_rate()[i],
            single.get_calcium_binding_rate()[0],
            rtol=1e-9,
            err_msg=f"{name}: cell {i} (Ca={ca}) differs between batch and single runs",
        )


def test_rdq18_advance_ode_also_records_the_step():
    """RDQ18's documented entry point is advance_ODE, which the README drives
    directly; the rate must work there and not only via advance_step."""
    model = RDQ18(num_cells=N_CELLS)
    start = model.bound_calcium_fraction().copy()
    integrated = np.zeros(N_CELLS)
    for _ in range(60):
        model.advance_ODE(1e-3, 2.0, SL_TEST)
        integrated += model.get_calcium_binding_rate() * 1e-3
    np.testing.assert_allclose(
        integrated, model.bound_calcium_fraction() - start, rtol=1e-10, atol=1e-12
    )


def test_rdq20mf_occupancy_is_a_marginal_of_the_ru_tensor():
    """Occupancy and permissivity are marginals of the same tensor over
    different axes, so both must be bounded by the total probability."""
    model = RDQ20MF(num_cells=N_CELLS)
    for _ in range(200):
        model.advance_step(model.dt, 2.0, 2.2, dSL_vals=0.0)

    total = model.x_RU.sum(axis=(0, 1, 2, 3))
    np.testing.assert_allclose(total, 1.0, atol=1e-8)
    assert np.all(model.bound_calcium_fraction() <= total + 1e-12)


@pytest.mark.parametrize("name,ModelClass", sorted(MODEL_REGISTRY.items()))
def test_occupancy_saturates_with_calcium(name, ModelClass):
    """Occupancy must span the full range as calcium does.

    This pins down *which* states count as calcium-bound, which for the
    RU-tensor models is a marginal over one axis of a 4-state tensor and easy
    to get subtly wrong. Calcium occupancy saturates at ~1 under saturating
    calcium, whereas the neighbouring quantity it could be confused with --
    permissivity -- plateaus well below that (0.92 for RDQ18, 0.77 for
    RDQ20MF), because tropomyosin kinetics limit it. So this separates them.
    """
    saturated = ModelClass(num_cells=N_CELLS)
    _hold(saturated, Ca=100.0, duration=0.2)
    assert np.all(saturated.bound_calcium_fraction() > 0.99), (
        f"{name}: occupancy failed to saturate at high calcium: "
        f"{saturated.bound_calcium_fraction()}"
    )

    depleted = ModelClass(num_cells=N_CELLS)
    _hold(depleted, Ca=1e-4, duration=0.2)
    assert np.all(depleted.bound_calcium_fraction() < 0.01), (
        f"{name}: occupancy failed to empty at negligible calcium: "
        f"{depleted.bound_calcium_fraction()}"
    )
