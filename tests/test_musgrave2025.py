"""
Unit tests for the Musgrave2025 model.

Test philosophy mirrors the other per-model suites:
  - Structural: shape, the quiescent reset, the two parameter sets
  - Mathematical: populations, step-length and vectorization independence, the
    limits the guards cover (no Pi, no calcium)
  - Physiological: the paper's own results -- the isometric twitches of its
    Fig. 5, the steady-state stresses of its Table 2, and the effects of ATP,
    Pi and the cross-over of calcium handling and crossbridges (Figs 8-10)
  - API: scalar inputs, reset, parameter overrides
"""

from functools import lru_cache

import numpy as np
import pytest

from crossbridge import Musgrave2025
from crossbridge.utils import musgrave_calcium

ND = Musgrave2025.default_parameters()
D = Musgrave2025.diabetic_parameters()
GROUPS = {"nondiabetic": ND, "diabetic": D}

#: Table 3's K [kPa/um], the maximally activated crossbridge model's (the authors' x_p).
K_CROSSBRIDGE = {"nondiabetic": 9029.917130064021, "diabetic": 5265.695318949247}


@lru_cache(maxsize=None)
def _cached_twitch(overrides: tuple, diabetic_calcium: bool, beats: int = 2):
    params = dict(overrides)
    model = Musgrave2025(1, params=params)
    dt, n = 1e-3, 1000
    for _ in range(beats):
        stress = np.zeros(n)
        for k in range(n):
            Ca = musgrave_calcium((k + 0.5) * dt, diabetic_calcium)
            model.advance_step(dt, Ca, 2.2, 0.0)
            stress[k] = (model.get_active_tension() + model.get_passive_tension())[0]
    amplitude = stress.max() - stress.min()
    above = np.flatnonzero(stress - stress.min() > 0.05 * amplitude)
    return stress.min(), stress.max(), amplitude, 1000 * dt * (above[-1] - above[0])


def _twitch(params: dict, diabetic_calcium: bool):
    """(diastolic, systolic, amplitude [kPa], duration [ms]) of the 1 Hz twitch at SL 2.2 um.

    The duration is the paper's: from the stress rising above 5% of the amplitude
    to its falling back below it. The model has no state slower than a beat, so
    the second beat is periodic.
    """
    return _cached_twitch(tuple(sorted(params.items())), diabetic_calcium)


def _steady_stress(params: dict, Ca: float = 32.0, SL: float = 2.2, duration: float = 0.5):
    model = Musgrave2025(1, params=params)
    for _ in range(round(duration / 1e-3)):
        model.advance_step(1e-3, Ca, SL, 0.0)
    return model.get_active_tension()[0]


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------


def test_initialization_shapes():
    model = Musgrave2025(num_cells=5)
    for name in ("CaTn", "N", "B", "C", "B_xB", "C_xC", "Fp1", "xB", "xC"):
        assert getattr(model, name).shape == (5,), name


def test_reset_is_quiescent():
    """No bridges attached; troponin, and tropomyosin over the heads in reach, in equilibrium."""
    model = Musgrave2025(num_cells=3)
    p = model.p
    for name in ("B", "C", "B_xB", "C_xC", "Fp1"):
        np.testing.assert_array_equal(getattr(model, name), 0.0)
    np.testing.assert_array_equal(model.get_active_tension(), 0.0)
    np.testing.assert_array_equal(model.get_active_stiffness(), 0.0)
    r = (p["Ca0"] / p["Ca50"]) ** p["n_tn"]
    np.testing.assert_allclose(model.CaTn, r / (1 + r), rtol=1e-15)
    A = model._z(np.array(p["SL0"])) - model.N
    a, b = model._tropomyosin_rates(model.CaTn)
    np.testing.assert_allclose(a * A, b * model.N, rtol=1e-12)


def test_diabetic_parameters_change_only_the_fitted_values():
    changed = {key for key in ND if D[key] != ND[key]}
    assert set(D) == set(ND)
    assert changed == {
        "Ca0", "Ca50", "k1", "km1", "k2", "km2", "k3", "phi_s1", "phi_s3", "kd_ATP",
        "phi_v", "phi_l", "rho_myo", "K_myo", "Kp1", "mu", "Kp2", "phi_p",
    }  # fmt: skip


# ---------------------------------------------------------------------------
# Mathematical
# ---------------------------------------------------------------------------


def test_populations_stay_fractions_through_a_twitch():
    """N, B, C and A = max(0, Z - N - B - C) stay in [0, 1], through a release too."""
    model = Musgrave2025(num_cells=2, params=D)
    SL = np.array([2.2, 2.2])
    for k in range(800):
        if 100 <= k < 150:
            SL = SL - np.array([0.0, 0.004])  # shorten the second cell by 0.2 um
        model.advance_step(1e-3, 1.5 * musgrave_calcium(k * 1e-3, True), SL, 0.0)
        for name in ("N", "B", "C", "CaTn"):
            value = getattr(model, name)
            assert np.all((value >= -1e-12) & (value <= 1 + 1e-12)), (k, name, value)
        assert np.all(model.N + model.B + model.C <= 1 + 1e-12)
        assert np.all(np.isfinite(model.xB) & np.isfinite(model.xC))


def test_step_length_independence():
    """One 1 ms call is ten 0.1 ms sub-steps, so it matches ten 0.1 ms calls to rounding."""
    one, ten = Musgrave2025(num_cells=2), Musgrave2025(num_cells=2)
    Ca, SL, dSL = np.array([0.5, 3.0]), np.array([2.0, 2.2]), np.array([-0.2, 0.1])
    for _ in range(30):
        one.advance_step(1e-3, Ca, SL, dSL)
        for _ in range(10):
            ten.advance_step(1e-4, Ca, SL, dSL)
    np.testing.assert_allclose(one.get_active_tension(), ten.get_active_tension(), rtol=1e-10)


def test_vectorization_independence():
    """
    Cells evolve exactly as they do alone, to the last bit, including a cell that
    comes off the clamp on A within a sub-step (split there) while its neighbours
    do not: from rest at SL 2.0 um, A starts clamped.
    """
    Ca, SL = np.array([0.4, 1.5, 3.0]), np.array([2.0, 2.1, 2.3])
    dSL = np.array([0.0, -0.1, 0.05])
    batch = Musgrave2025(num_cells=3)
    for _ in range(60):
        batch.advance_step(1e-3, Ca, SL, dSL)
    for i in range(3):
        single = Musgrave2025(num_cells=1)
        for _ in range(60):
            single.advance_step(1e-3, Ca[i : i + 1], SL[i : i + 1], dSL[i : i + 1])
        batch_state = batch.get_state()
        for key, value in single.get_state().items():
            if np.ndim(value):
                np.testing.assert_array_equal(batch_state[key][i], value[0], err_msg=key)


def test_no_pi_is_the_limit_of_little_pi():
    """With Pi and ATP cancelled from k-3, Pi = 0 is finite and continuous."""
    none = _steady_stress({**ND, "Pi": 0.0})
    little = _steady_stress({**ND, "Pi": 1e-9})
    assert np.isfinite(none)
    assert none == pytest.approx(little, rel=1e-8)


def test_survives_an_absence_of_calcium():
    """CaTn decays towards zero, below the floor on it in the tropomyosin rates."""
    model = Musgrave2025(num_cells=2)
    for _ in range(3):
        model.advance_step(0.1, 0.0, np.array([2.0, 2.2]), 0.0)
    assert np.all(model.CaTn < 1e-13)
    for name in ("N", "B", "C", "B_xB", "C_xC"):
        assert np.all(np.isfinite(getattr(model, name))), name
    np.testing.assert_allclose(model.get_active_tension(), 0.0, atol=1e-8)


# ---------------------------------------------------------------------------
# Physiological: the paper's results
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("group", "systolic", "amplitude", "duration"),
    [("nondiabetic", 20.2, 14.5, 550.0), ("diabetic", 12.0, 7.9, 674.0)],
)
def test_isometric_twitch_reproduces_fig_5a(group, systolic, amplitude, duration):
    """Fig. 5A: the 1 Hz twitch at SL 2.2 um; the paper rounds to 0.1 kPa and 1 ms."""
    _, sys_, amp, dur = _twitch(GROUPS[group], group == "diabetic")
    assert sys_ == pytest.approx(systolic, abs=0.05)
    assert amp == pytest.approx(amplitude, abs=0.05)
    assert dur == pytest.approx(duration, abs=3.0)


@pytest.mark.parametrize(("group", "amplitude"), [("nondiabetic", 40.0), ("diabetic", 26.9)])
def test_myocyte_twitch_reproduces_fig_5b(group, amplitude):
    """Fig. 5B: stress per myofilament area, K_myo in place of K."""
    _, _, amp, _ = _twitch({**GROUPS[group], "rho_myo": 1.0}, group == "diabetic")
    assert amp == pytest.approx(amplitude, abs=0.1)


TABLE_2 = {  # (MgATP, Pi) [mM] -> model stress [kPa], non-diabetic and diabetic
    (5.0, 1.0): (22.0, 16.9),
    (1.0, 1.0): (23.1, 18.4),
    (0.1, 1.0): (27.2, 19.1),
    (5.0, 0.0): (22.6, 17.2),
    (5.0, 10.0): (17.9, 14.4),
}


@pytest.mark.parametrize("group", sorted(GROUPS))
@pytest.mark.parametrize(("ATP", "Pi"), sorted(TABLE_2))
def test_crossbridge_model_reproduces_table_2(group, ATP, Pi):
    """
    Table 2 is the maximally activated crossbridge model's: no N state, Table 3's K.
    Switching off A -> N (km0 = 0) at 32 uM calcium leaves exactly that.
    """
    params = {**GROUPS[group], "MgATP": ATP, "Pi": Pi, "km0": 0.0}
    params["K_myo"] = K_CROSSBRIDGE[group] / params["rho_myo"]
    expected = TABLE_2[(ATP, Pi)][group == "diabetic"]
    assert _steady_stress(params) == pytest.approx(expected, abs=0.05)


@pytest.mark.parametrize("group", sorted(GROUPS))
def test_muscle_model_matches_the_crossbridge_model_at_baseline(group):
    """K is raised in the muscle model so that at 32 uM calcium it develops Table 2's stress."""
    expected = TABLE_2[(5.0, 1.0)][group == "diabetic"]
    assert _steady_stress(GROUPS[group]) == pytest.approx(expected, abs=0.05)


@pytest.mark.parametrize(
    ("group", "amplitude_rise", "duration_rise"),
    [("nondiabetic", 0.09, 0.02), ("diabetic", 0.29, 0.25)],
)
def test_low_atp_reproduces_fig_9a(group, amplitude_rise, duration_rise):
    """Fig. 9A: at 1 mM ATP, detachment slows: larger, longer twitches."""
    diabetic = group == "diabetic"
    base = _twitch(GROUPS[group], diabetic)
    low = _twitch({**GROUPS[group], "MgATP": 1.0}, diabetic)
    assert low[2] / base[2] - 1 == pytest.approx(amplitude_rise, abs=0.01)
    assert low[3] / base[3] - 1 == pytest.approx(duration_rise, abs=0.01)
    # ... and diastolic stress rises, by 1.1 kPa, only in the diabetic model
    assert (low[0] - base[0] > 1.0) == diabetic
    assert low[0] - base[0] > 0.0


@pytest.mark.parametrize(("group", "fall"), [("nondiabetic", 0.28), ("diabetic", 0.26)])
def test_high_pi_lowers_the_twitch_as_in_fig_8a(group, fall):
    """
    Fig. 8A: 10 mM Pi lowers the amplitude by 28% and 26% here. The paper's text says
    "26% vs. 28%" (non-diabetic first), the same numbers the other way round.
    """
    diabetic = group == "diabetic"
    base = _twitch(GROUPS[group], diabetic)
    high = _twitch({**GROUPS[group], "Pi": 10.0}, diabetic)
    assert 1 - high[2] / base[2] == pytest.approx(fall, abs=0.01)


def test_cross_over_reproduces_fig_10a():
    """
    Fig. 10A, all with the non-diabetic passive model: diabetic calcium handling (and
    Ca50) or diabetic crossbridges alone each lower the amplitude "between 25% and
    30%" (29.6% and 24.8% here), and lengthen the twitch by 7% and 15%; both
    together, by 45% and 23%.
    """
    passive = {key: ND[key] for key in ("Kp1", "mu", "Kp2", "phi_p")}
    reference = _twitch(ND, False)
    calcium = _twitch({**ND, "Ca50": D["Ca50"]}, True)
    crossbridges = _twitch({**D, **passive, "Ca50": ND["Ca50"]}, False)
    both = _twitch({**D, **passive}, True)

    def fall(twitch):
        return 1 - twitch[2] / reference[2]

    def lengthening(twitch):
        return twitch[3] / reference[3] - 1

    assert 0.245 <= fall(calcium) <= 0.30 and 0.245 <= fall(crossbridges) <= 0.30
    assert lengthening(calcium) == pytest.approx(0.07, abs=0.01)
    assert lengthening(crossbridges) == pytest.approx(0.15, abs=0.01)
    assert fall(both) == pytest.approx(0.45, abs=0.01)
    assert lengthening(both) == pytest.approx(0.23, abs=0.015)


def test_length_dependence():
    """Fewer heads in reach at shorter lengths, Eq. (6): less stress at SL 2.0 than 2.2 um."""
    assert _steady_stress(ND, Ca=1.0, SL=2.0) < 0.6 * _steady_stress(ND, Ca=1.0, SL=2.2)


def test_atp_hydrolysis_rate():
    """
    The net C -> A flux: at rest only the reverse k-3 A, which thermodynamics keeps
    to 1e-8 /s; rising with calcium; at maximal activation, lowering ATP from 5 to
    1 mM slows it more in the diabetic model, whose kd_ATP (5 mM) leaves ATP far
    from saturating.
    """
    model = Musgrave2025(num_cells=3)
    rest = model.get_atp_hydrolysis_rate()
    assert np.all((rest < 0.0) & (rest > -1e-7))
    for _ in range(300):
        model.advance_step(1e-3, np.array([0.4, 0.8, 32.0]), 2.2, 0.0)
    rate = model.get_atp_hydrolysis_rate()
    assert 0.0 < rate[0] < rate[1] < rate[2]

    def turnover(params):
        model = Musgrave2025(1, params=params)
        for _ in range(500):
            model.advance_step(1e-3, 32.0, 2.2, 0.0)
        return model.get_atp_hydrolysis_rate()[0]

    nd = turnover({**ND, "MgATP": 1.0}) / turnover(ND)
    d = turnover({**D, "MgATP": 1.0}) / turnover(D)
    assert d < nd < 1.0


def test_passive_stress():
    """Eqs. (19)-(20): the parallel spring is zero at L_r; the dashpot relaxes to mu v."""
    model = Musgrave2025(num_cells=2, params={"Ca0": 0.0})
    model.advance_step(1e-3, 0.0, np.array([ND["L_r"], 2.2]), 0.0)
    assert model.get_passive_tension()[0] == pytest.approx(0.0, abs=1e-12)
    for _ in range(100):
        model.advance_step(1e-3, 0.0, 2.2, 0.5)
    np.testing.assert_allclose(model.Fp1, ND["mu"] * 0.5, rtol=1e-6)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_scalar_and_array_inputs_agree():
    a, b = Musgrave2025(num_cells=2), Musgrave2025(num_cells=2)
    for _ in range(20):
        a.advance_step(1e-3, 2.0, 2.1, 0.0)
        b.advance_step(1e-3, np.full(2, 2.0), np.full(2, 2.1), np.zeros(2))
    np.testing.assert_array_equal(a.get_active_tension(), b.get_active_tension())


def test_reset_restores_the_initial_state():
    model = Musgrave2025(num_cells=2)
    initial = model.get_state()
    for _ in range(50):
        model.advance_step(1e-3, 3.0, 2.2, 0.0)
    model.reset()
    for key, value in initial.items():
        np.testing.assert_array_equal(model.get_state()[key], value, err_msg=key)


def test_parameter_overrides():
    model = Musgrave2025(num_cells=1, params={"k1": 2 * ND["k1"]})
    assert model.p["k1"] == 2 * ND["k1"]
    assert model.p["k3"] == ND["k3"]
