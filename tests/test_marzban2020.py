"""
Unit tests for the Marzban2020 model.

Test philosophy mirrors the other per-model suites:
  - Structural: shape, initialization
  - Mathematical: conservation, step-length independence, vectorization
  - Physiological: the paper's own results -- the steady-state pCa50 of its
    Fig. 3 caption and the 4 Hz twitch of its Fig. 3B -- the force-dependent
    recruitment from the myosin OFF state behind its length-dependent
    activation, and why the code's K_coop and k_off are used over Table 7's
  - API: scalar inputs, reset, parameter overrides
"""

import numpy as np
import pytest
from scipy.optimize import curve_fit

from crossbridge import Marzban2020
from crossbridge.marzban2020 import _MOMENTS
from crossbridge.utils import janssen_calcium


def _twitch(params=None, SL=2.2, frequency=4.0, beats=4, dt=5e-4):
    """(developed tension [kPa], diastolic tension [kPa], peak U_NR) of the last beat."""
    model = Marzban2020(1, params=params)
    n = round(1.0 / frequency / dt)
    tension, U = np.zeros(beats * n), np.zeros(beats * n)
    for k in range(beats * n):
        model.advance_step(dt, janssen_calcium((k + 1) * dt, frequency), SL, 0.0)
        tension[k], U[k] = model.get_active_tension()[0], model.U_NR[0]
    last = tension[-n:]
    return last.max() - last.min(), last.min(), U[-n:].max()


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------


def test_initialization_shapes():
    model = Marzban2020(num_cells=5)
    for name in ("N", "U_NR", *_MOMENTS):
        assert getattr(model, name).shape == (5,), name


def test_initial_state_is_quiescent():
    """Nothing attached, no calcium bound, myosin at its force-free OFF/ON balance."""
    model = Marzban2020(num_cells=3)
    p = model.p
    np.testing.assert_array_equal(model.N, 1.0)
    np.testing.assert_allclose(model.U_NR, p["k_SR"] / (p["k_SR"] + p["k_mSR"]))
    for name in _MOMENTS:
        np.testing.assert_array_equal(getattr(model, name), 0.0)
    np.testing.assert_array_equal(model.get_active_tension(), 0.0)
    np.testing.assert_array_equal(model.bound_calcium_fraction(), 0.0)


# ---------------------------------------------------------------------------
# Mathematical
# ---------------------------------------------------------------------------


def test_populations_stay_fractions_through_a_twitch():
    """N, U_NR, the occupancies and P = 1 - N - p1_0 - p2_0 - p3_0 stay in [0, 1]."""
    model = Marzban2020(num_cells=1)
    for k in range(500):
        model.advance_step(1e-3, janssen_calcium((k + 1) * 1e-3, 4.0), 2.2, 0.0)
        occupancies = np.array([model.N, model.p1_0, model.p2_0, model.p3_0])
        assert np.all(occupancies >= -1e-12) and np.all(occupancies.sum(axis=0) <= 1 + 1e-12)
        assert np.all((model.U_NR >= 0) & (model.U_NR <= 1))


def test_step_length_independence():
    """One 1 ms call is five 0.2 ms sub-steps, so it matches five 0.2 ms calls to rounding."""
    one, five = Marzban2020(num_cells=2), Marzban2020(num_cells=2)
    Ca, SL, dSL = np.array([1.5, 3.0]), np.array([2.0, 2.2]), np.array([-0.2, 0.1])
    for _ in range(20):
        one.advance_step(1e-3, Ca, SL, dSL)
        for _ in range(5):
            five.advance_step(2e-4, Ca, SL, dSL)
    np.testing.assert_allclose(one.get_active_tension(), five.get_active_tension(), rtol=1e-11)


def test_vectorization_independence():
    """
    Cells at different Ca and SL evolve exactly as they do alone, to the last bit:
    a cell differing by an ulp shows up as a 1e-9 relative error in a calcium
    binding rate near steady state, a difference of nearly equal fractions.
    """
    Ca, SL = np.array([0.5, 1.5, 3.0]), np.array([1.9, 2.1, 2.3])
    batch = Marzban2020(num_cells=3)
    for _ in range(50):
        batch.advance_step(1e-3, Ca, SL, 0.0)
    for i in range(3):
        single = Marzban2020(num_cells=1)
        for _ in range(50):
            single.advance_step(1e-3, Ca[i : i + 1], SL[i : i + 1], 0.0)
        batch_state = batch.get_state()
        for key, value in single.get_state().items():
            if np.ndim(value):
                np.testing.assert_array_equal(batch_state[key][i], value[0], err_msg=key)


# ---------------------------------------------------------------------------
# Physiological: the paper's results
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("SL", "pCa50"), [(1.9, 5.89), (2.3, 6.001)])
def test_reproduces_the_steady_state_pca50(SL, pCa50):
    """Fig. 3 caption: pCa50 5.89 at SL 1.9 um and 6.001 at 2.3 um (printed as "Ca50")."""
    cas = np.geomspace(0.2, 20.0, 24)
    model = Marzban2020(len(cas))
    for _ in range(600):
        model.advance_step(1e-3, cas, SL, 0.0)
    tension = model.get_active_tension()

    def hill(c, Fmax, c50, n):
        return Fmax * c**n / (c**n + c50**n)

    _, c50, n = curve_fit(hill, cas, tension, p0=[tension.max(), 1.0, 4.0])[0]
    assert 6.0 - np.log10(c50) == pytest.approx(pCa50, abs=0.03)
    assert 4.0 < n < 5.5  # 4.63 and 4.47 in the caption


def test_twitch_at_4_hz_reaches_the_papers_tension():
    """Fig. 3B: about 53 kPa developed at 4 Hz, SL 2.2 um (55 kPa here)."""
    developed, _, _ = _twitch()
    assert 50.0 < developed < 60.0


def test_force_recruits_myosin_heads():
    """Eq. (10): force switches heads ON, from 0.22 at rest to most of them at peak."""
    _, _, U_peak = _twitch()
    p = Marzban2020.default_parameters()
    assert U_peak > 0.9 > p["k_SR"] / (p["k_SR"] + p["k_mSR"])


def test_force_dependent_recruitment_steepens_length_dependence():
    """
    Campbell et al. (2018): recruitment from the OFF state by force contributes to
    length-dependent activation. Without it (k_force = 0) the 4 Hz twitch rises 1.7-fold
    from SL 1.9 to 2.2 um; with it, 3-fold.
    """
    ratios = []
    for k_force in (0.0, Marzban2020.default_parameters()["k_force"]):
        short, _, _ = _twitch({"k_force": k_force}, SL=1.9)
        long, _, _ = _twitch({"k_force": k_force}, SL=2.2)
        ratios.append(long / short)
    assert ratios[1] > ratios[0] + 0.8, ratios


def test_table_7_activation_parameters_do_not_relax():
    """Table 7's K_coop = 1.857 and k_off = 101.2 /s leave most of the tension between beats."""
    _, diastolic, _ = _twitch({"K_coop": 1.857, "k_off": 101.2})
    assert diastolic > 50.0
    _, diastolic, _ = _twitch()
    assert diastolic < 2.0


def test_failing_heart_metabolites_lower_tension():
    """Lopez et al. (2020): the lower MgATP and higher Pi of TAC hearts impair tension."""
    sham, _, _ = _twitch({"MgATP": 8.3, "MgADP": 0.054, "Pi": 1.47})
    tac, _, _ = _twitch({"MgATP": 7.0, "MgADP": 0.041, "Pi": 2.36})
    assert tac < 0.7 * sham


def test_atp_hydrolysis_rate():
    model = Marzban2020(num_cells=3)
    np.testing.assert_array_equal(model.get_atp_hydrolysis_rate(), 0.0)
    for _ in range(300):
        model.advance_step(1e-3, np.array([0.8, 1.2, 10.0]), 2.2, 0.0)
    rate = model.get_atp_hydrolysis_rate()
    assert 0.0 < rate[0] < rate[1] < rate[2]


def test_passive_tension():
    """Zero at SL_rest_pas; k_passive per half-sarcomere above it; collagen past 2.25 um."""
    model = Marzban2020(num_cells=3)
    model.advance_step(1e-3, 0.1, np.array([1.8, 2.0, 2.3]), 0.0)
    Tp = model.get_passive_tension()
    p = model.p
    assert Tp[0] == pytest.approx(0.0, abs=1e-15)
    assert Tp[1] == pytest.approx(p["k_passive"] * 0.1)
    assert Tp[2] > p["k_passive"] * 0.25


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_scalar_and_array_inputs_agree():
    a, b = Marzban2020(num_cells=2), Marzban2020(num_cells=2)
    for _ in range(20):
        a.advance_step(1e-3, 2.0, 2.1, 0.0)
        b.advance_step(1e-3, np.full(2, 2.0), np.full(2, 2.1), np.zeros(2))
    np.testing.assert_array_equal(a.get_active_tension(), b.get_active_tension())


def test_reset_restores_the_initial_state():
    model = Marzban2020(num_cells=2)
    initial = model.get_state()
    for _ in range(50):
        model.advance_step(1e-3, 3.0, 2.2, 0.0)
    model.reset()
    for key, value in initial.items():
        np.testing.assert_array_equal(model.get_state()[key], value, err_msg=key)


def test_parameter_overrides():
    model = Marzban2020(num_cells=1, params={"ka": 2 * 559.5568})
    assert model.p["ka"] == 2 * 559.5568
    assert model.p["k1"] == Marzban2020.default_parameters()["k1"]
