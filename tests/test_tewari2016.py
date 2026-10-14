"""
Unit tests for the Tewari2016 model.

Test philosophy mirrors the other per-model suites:
  - Structural: shape, initialization
  - Mathematical: conservation, step-length independence, vectorization
  - Physiological: the paper's own results -- Table E.1's steady-state
    force-calcium relation at five sarcomere lengths, and the 4 Hz twitch of
    Fig. 3 -- which also pin the reading of Table B.1's reference temperature
    (see the module docstring); metabolite and temperature effects
  - API: scalar inputs, reset, parameter overrides
"""

import numpy as np
import pytest
from scipy.optimize import curve_fit

from crossbridge import Tewari2016
from crossbridge.tewari2016 import _MOMENTS
from crossbridge.utils import janssen_calcium

#: Table E.1 of Tewari et al. (2016): SL [um] -> (Ca50 [uM], Hill coefficient).
TABLE_E1 = {
    1.85: (2.0, 7.0),
    1.95: (1.8, 8.6),
    2.05: (1.5, 10.4),
    2.15: (1.2, 11.0),
    2.25: (1.0, 11.0),
}
#: The conditions of Fig. E.1: MgATP 5 mM, no MgADP, no Pi.
SKINNED = {"MgATP": 5.0, "MgADP": 0.0, "Pi": 0.0}
CA_LEVELS = np.geomspace(0.5, 12.0, 20)


def _steady_tension(sls, cas, params=None, duration=0.5):
    """Steady active tension [kPa] on the (SL, Ca) grid, shape (len(sls), len(cas))."""
    SL, Ca = (a.ravel() for a in np.meshgrid(sls, cas, indexing="ij"))
    model = Tewari2016(len(SL), params=params)
    for _ in range(round(duration / model.dt)):
        model.advance_step(model.dt, Ca, SL, 0.0)
    return model.get_active_tension().reshape(len(sls), len(cas))


def _hill_fit(cas, tension):
    def hill(c, Fmax, c50, n):
        return Fmax * c**n / (c**n + c50**n)

    (Fmax, c50, n), _ = curve_fit(hill, cas, tension, p0=[tension.max(), 1.5, 8.0])
    return Fmax, c50, n


@pytest.fixture(scope="module")
def force_pca():
    """Fmax, Ca50 and Hill coefficient at each of Table E.1's lengths, at its conditions."""
    sls = np.array(sorted(TABLE_E1))
    tension = _steady_tension(sls, CA_LEVELS, params=SKINNED)
    return {sl: _hill_fit(CA_LEVELS, tension[i]) for i, sl in enumerate(sls)}


def _twitch(params=None, frequency=4.0, SL=2.2, beats=3):
    """Developed tension [kPa] of the last of `beats` isometric twitches."""
    model = Tewari2016(1, params=params)
    dt = model.dt
    n = round(1.0 / frequency / dt)
    tension = np.zeros(beats * n)
    for k in range(beats * n):
        model.advance_step(dt, janssen_calcium((k + 1) * dt, frequency), SL, 0.0)
        tension[k] = model.get_active_tension()[0]
    last = tension[-n:]
    return last.max() - last.min()


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------


def test_initialization_shapes():
    model = Tewari2016(num_cells=5)
    for name in ("CaL", "CaH", "N", *_MOMENTS):
        assert getattr(model, name).shape == (5,), name


def test_initial_state_is_quiescent():
    """Nothing attached, everything non-permissible, troponin at its diastolic steady state."""
    model = Tewari2016(num_cells=3)
    np.testing.assert_array_equal(model.N, 1.0)
    for name in _MOMENTS:
        np.testing.assert_array_equal(getattr(model, name), 0.0)
    assert np.all((model.CaL > 0) & (model.CaL < model.CaH) & (model.CaH < 1))
    np.testing.assert_array_equal(model.get_active_tension(), 0.0)


# ---------------------------------------------------------------------------
# Mathematical
# ---------------------------------------------------------------------------


def test_populations_stay_fractions_through_a_twitch():
    """N, the occupancies and the derived P = 1 - N - p1_0 - p2_0 - p3_0 stay in [0, 1]."""
    model = Tewari2016(num_cells=1)
    for k in range(500):
        model.advance_step(1e-3, janssen_calcium((k + 1) * 1e-3, 4.0), 2.2, 0.0)
        occupancies = np.array([model.N, model.p1_0, model.p2_0, model.p3_0])
        P = 1.0 - occupancies.sum(axis=0)
        assert np.all(occupancies >= -1e-12) and np.all(P >= -1e-12), k
        assert np.all((model.CaL >= 0) & (model.CaL <= 1) & (model.CaH >= 0) & (model.CaH <= 1))


def test_step_length_independence():
    """One 1 ms call is ten 0.1 ms sub-steps, so it matches ten 0.1 ms calls to rounding."""
    one, ten = Tewari2016(num_cells=2), Tewari2016(num_cells=2)
    Ca, SL, dSL = np.array([1.5, 3.0]), np.array([2.0, 2.2]), np.array([-0.2, 0.1])
    for _ in range(20):
        one.advance_step(1e-3, Ca, SL, dSL)
        for _ in range(10):
            ten.advance_step(1e-4, Ca, SL, dSL)
    np.testing.assert_allclose(one.get_active_tension(), ten.get_active_tension(), rtol=1e-11)


def test_vectorization_independence():
    """
    Cells at different Ca and SL evolve exactly as they do alone, to the last bit:
    a cell differing by an ulp shows up as a 1e-9 relative error in a calcium
    binding rate near steady state, a difference of nearly equal fractions.
    """
    Ca, SL = np.array([0.5, 1.5, 3.0]), np.array([1.9, 2.1, 2.3])
    batch = Tewari2016(num_cells=3)
    for _ in range(50):
        batch.advance_step(1e-3, Ca, SL, 0.0)
    for i in range(3):
        single = Tewari2016(num_cells=1)
        for _ in range(50):
            single.advance_step(1e-3, Ca[i : i + 1], SL[i : i + 1], 0.0)
        batch_state = batch.get_state()
        for key, value in single.get_state().items():
            if np.ndim(value):
                np.testing.assert_array_equal(batch_state[key][i], value[0], err_msg=key)


# ---------------------------------------------------------------------------
# Physiological: the paper's results
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("SL", sorted(TABLE_E1))
def test_reproduces_table_e1(force_pca, SL):
    """
    Ca50 within 0.2 uM and the Hill coefficient within 1.75 of Table E.1 (Fig. E.1). On a
    40-point grid from 0.3 to 30 uM they are within 0.16 uM and 1.53 (8.5 against 7.0 at
    1.85 um); the fitted Hill coefficient moves by up to 0.4 with the grid, Ca50 by 0.01.
    """
    _, c50, n = force_pca[SL]
    c50_paper, n_paper = TABLE_E1[SL]
    assert abs(c50 - c50_paper) < 0.2, (c50, c50_paper)
    assert abs(n - n_paper) < 1.75, (n, n_paper)


def test_maximal_force_follows_thick_filament_overlap(force_pca):
    """Fig. E.1: 135% of the 1.85 um maximum at 2.25 um, the ratio of OV_thick."""
    ratio = force_pca[2.25][0] / force_pca[1.85][0]
    OV_thick, _ = Tewari2016(1)._overlap(np.array([2.25, 1.85]))
    assert ratio == pytest.approx(OV_thick[0] / OV_thick[1], rel=0.01)
    assert ratio == pytest.approx(1.35, abs=0.02)


def test_twitch_at_4_hz_reaches_the_papers_tension():
    """Fig. 3E: about 57 kPa developed at 4 Hz, SL 2.2 um (isometric here: 61 kPa)."""
    assert 50.0 < _twitch() < 70.0


def test_activation_rates_read_at_37_5_degc_lose_the_twitch():
    """
    The reading of Eqs. (IV) and (VI) as printed -- activation rates at 37.5 degC -- is
    about half as sensitive to calcium and leaves the 4 Hz twitch at ~12 kPa, which is
    why `T_ref_act` defaults to 17 degC (module docstring).
    """
    printed = {"T_ref_act": 37.5}
    assert _twitch(printed) < 20.0
    sls, cas = np.array([2.25]), CA_LEVELS * 1.6
    _, c50_printed, _ = _hill_fit(cas, _steady_tension(sls, cas, {**SKINNED, **printed})[0])
    _, c50, _ = _hill_fit(CA_LEVELS, _steady_tension(sls, CA_LEVELS, SKINNED)[0])
    assert c50_printed / c50 > 1.3


def test_phosphate_lowers_force_and_atp_lowers_its_inhibition():
    """Tewari et al. (2015) Fig. 10A: isometric force falls as Pi rises."""
    forces = [
        _steady_tension(np.array([2.2]), np.array([10.0]), {"Pi": Pi})[0, 0]
        for Pi in (0.0, 2.0, 10.0)
    ]
    assert forces[0] > forces[1] > forces[2] > 0.0


def test_atp_hydrolysis_rate():
    """Zero at rest, positive and rising with activation (Eq. (13) of Tewari et al. 2015)."""
    model = Tewari2016(num_cells=3)
    np.testing.assert_array_equal(model.get_atp_hydrolysis_rate(), 0.0)
    for _ in range(300):
        model.advance_step(1e-3, np.array([1.0, 2.0, 10.0]), 2.2, 0.0)
    rate = model.get_atp_hydrolysis_rate()
    assert 0.0 < rate[0] < rate[1] < rate[2]


def test_warmer_muscle_develops_force_faster():
    """Fig. H.1: tension develops faster at 37 than at 27 degC."""
    halftimes = []
    for T in (27.0, 37.0):
        model = Tewari2016(num_cells=1, params={"T": T})
        tension = []
        for _ in range(400):
            model.advance_step(1e-3, 10.0, 2.2, 0.0)
            tension.append(model.get_active_tension()[0])
        halftimes.append(np.argmax(np.array(tension) >= 0.5 * tension[-1]))
    assert halftimes[1] < halftimes[0]


def test_force_slows_calcium_unbinding():
    """Appendix B: with zeta > 0, force keeps troponin occupied for longer."""
    occupancy = []
    for zeta in (0.0, 0.23):
        model = Tewari2016(num_cells=1, params={"zeta": zeta})
        for k in range(150):
            model.advance_step(1e-3, janssen_calcium((k + 1) * 1e-3, 4.0), 2.2, 0.0)
        occupancy.append(model.bound_calcium_fraction()[0])
    assert occupancy[1] > occupancy[0]


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_scalar_and_array_inputs_agree():
    a, b = Tewari2016(num_cells=2), Tewari2016(num_cells=2)
    for _ in range(20):
        a.advance_step(1e-3, 2.0, 2.1, 0.0)
        b.advance_step(1e-3, np.full(2, 2.0), np.full(2, 2.1), np.zeros(2))
    np.testing.assert_array_equal(a.get_active_tension(), b.get_active_tension())


def test_reset_restores_the_initial_state():
    model = Tewari2016(num_cells=2)
    initial = model.get_state()
    for _ in range(50):
        model.advance_step(1e-3, 3.0, 2.2, 0.0)
    model.reset()
    for key, value in initial.items():
        np.testing.assert_array_equal(model.get_state()[key], value, err_msg=key)


def test_parameter_overrides():
    model = Tewari2016(num_cells=1, params={"ka": 2 * 294.1})
    assert model.p["ka"] == 2 * 294.1
    assert model.p["k1"] == Tewari2016.default_parameters()["k1"]
