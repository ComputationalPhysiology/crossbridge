"""
Unit tests for the Rice2008 model.

Test philosophy mirrors the other per-model suites:
  - Structural: initialization, the overlap fractions of Fig. 1B, the rabbit parameters
  - Mathematical: normalized maximal force, steady strains, conservation of the
    occupancies, step-length independence
  - Review focus: no calcium, lengths outside the paper's range, fast shortening at body
    temperature, velocity estimated from successive lengths, a state moved between
    parameter sets

The reference values come from an independent Radau solution of the CellML equations in s.
At 22.5 degC that solution reproduces the authors' Data S1 sample outputs: the Fig. 5 twitch
within 7e-6, and the F-Ca ramp within 2.5e-5.
"""

import warnings

import numpy as np
import pytest

from crossbridge.rice2008 import _STATES, Rice2008
from crossbridge.utils import calcium_trace


def _calcium(t):
    """Table 3's transient: diastolic 0.09 uM, peak 1.45 uM, stimulus at t = 0."""
    return calcium_trace(t, c0=0.09, cmax=1.45, t0=0.0)


def _maximally_activated(Ta_max=1.0, duration=1.0, **params):
    """A cell held at Ca 1e4 uM and SL 2.3 um for `duration`, at 1 ms calls."""
    model = Rice2008(1, Ta_max=Ta_max, params=params)
    for _ in range(round(duration / 1e-3)):
        model.advance_step(1e-3, 1e4, 2.3, 0.0)
    return model


def Ta_at(**params):
    """Maximal active tension after 1 s, at Ta_max = 1."""
    return _maximally_activated(**params).get_active_tension()[0]


def _twitch_tension(dt, duration=0.4, SL=2.2):
    """Ta at the end of every 1 ms of an isometric twitch, with calls of `dt` [s]."""
    model = Rice2008(1)
    per_ms = round(1e-3 / dt)
    tension = []
    for k in range(round(duration / dt)):
        # calcium at the call's midpoint, so that both runs see the same transient
        # to second order and differ only by the integrator
        model.advance_step(dt, _calcium((k + 0.5) * dt), SL, 0.0)
        if (k + 1) % per_ms == 0:
            tension.append(model.get_active_tension()[0])
    return np.array(tension)


def _getters(model):
    return {
        "Ta": model.get_active_tension(),
        "Ka": model.get_active_stiffness(),
        "Tp": model.get_passive_tension(),
        "bound_ca": model.bound_calcium_fraction(),
        "binding_rate": model.get_calcium_binding_rate(),
    }


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------


def test_reset_is_quiescent_with_troponin_at_rest():
    m = Rice2008(3)
    p = m.p
    konT = p["kon"] * p["Qkon"] ** ((p["TmpC"] - 37) / 10)
    koffHT = p["koffH"] * p["koffmod"] * p["Qkoff"] ** ((p["TmpC"] - 37) / 10)
    koffLT = p["koffL"] * p["koffmod"] * p["Qkoff"] ** ((p["TmpC"] - 37) / 10)
    np.testing.assert_allclose(m.TRPNCaH, konT * 0.09 / (konT * 0.09 + koffHT), rtol=1e-15)
    np.testing.assert_allclose(m.TRPNCaL, konT * 0.09 / (konT * 0.09 + koffLT), rtol=1e-15)
    for name in ("N", "N_NoXB"):
        np.testing.assert_array_equal(getattr(m, name), 1.0, err_msg=name)
    for name in ("XBprer", "XBpostr", "P_NoXB", "xXBprer"):
        np.testing.assert_array_equal(getattr(m, name), 0.0, err_msg=name)
    np.testing.assert_array_equal(m.xXBpostr, 0.007)
    np.testing.assert_array_equal(m.get_active_tension(), 0.0)
    np.testing.assert_array_equal(m.get_active_stiffness(), 0.0)
    for name in _STATES:
        assert getattr(m, name).shape == (3,), name
    assert set(_STATES) <= set(m.get_state())


def test_overlap_fractions_fig1b():
    thick, thin = Rice2008(1)._overlap(np.array([1.1, 1.4, 1.65, 2.3, 2.4]))
    np.testing.assert_allclose(thick, [0.0, 8 / 31, 0.9 / 1.55, 1.0, 1.0], rtol=1e-12)
    np.testing.assert_allclose(thin, [0.0, 1 / 6, 0.45 / 1.2, 31 / 48, 31 / 48], rtol=1e-12)
    # the paper's rounded values: thin 0.17 at 1.4 um and 0.64 at 2.4 um
    assert thin[1] == pytest.approx(0.17, abs=0.005) and thin[4] == pytest.approx(0.64, abs=0.01)


def test_rabbit_parameters():
    rat, rabbit = Rice2008.default_parameters(), Rice2008.rabbit_parameters()
    assert rabbit.keys() == rat.keys()
    assert {k for k in rat if rat[k] != rabbit[k]} == {"xbmodsp", "koffmod", "TmpC"}
    assert (rabbit["xbmodsp"], rabbit["koffmod"], rabbit["TmpC"]) == (0.2, 0.9, 37.0)


# ---------------------------------------------------------------------------
# Mathematical
# ---------------------------------------------------------------------------


def test_maximal_force_is_normalized():
    """
    Eq. 35 divides by the maximal XBpostr occupancy at the base rates, so maximal force is
    about 1 at 37 degC, where every Q factor is 1. The CellML reference at Ca 1e4 uM and SL
    2.3 um gives 0.98787 at 37 degC and 0.92282 at 22.5 degC.
    """
    assert Ta_at(TmpC=37.0) == pytest.approx(0.98787, abs=2e-3)
    assert Ta_at(TmpC=22.5) == pytest.approx(0.92282, abs=2e-3)


def test_ta_max_scales_the_outputs_exactly():
    one = _maximally_activated(duration=0.1)
    fifty = _maximally_activated(Ta_max=50.0, duration=0.1)
    assert one.get_passive_tension()[0] > 0.0
    for getter in ("get_active_tension", "get_passive_tension", "get_active_stiffness"):
        np.testing.assert_array_equal(
            getattr(fifty, getter)(), 50.0 * getattr(one, getter)(), err_msg=getter
        )


def test_isometric_steady_strains():
    """Isometric, the strains relax to the Table 2 values whatever they start from."""
    model = Rice2008(2)
    model.xXBprer = np.array([0.004, -0.004])
    model.xXBpostr = np.array([0.0, 0.012])
    for _ in range(1000):
        model.advance_step(1e-3, 2.0, 2.2, 0.0)
    np.testing.assert_allclose(model.xXBprer, 0.0, atol=1e-9)
    np.testing.assert_allclose(model.xXBpostr, model.p["x_0"], rtol=1e-9)


def test_occupancies_are_conserved():
    """N_NoXB + P_NoXB stays 1, and N, XBprer, XBpostr and P = 1 - N - ... stay fractions."""
    model, peak = Rice2008(1), 0.0
    for k in range(500):
        model.advance_step(1e-3, _calcium((k + 1) * 1e-3), 2.2, 0.0)
        np.testing.assert_allclose(model.N_NoXB + model.P_NoXB, 1.0, rtol=0, atol=1e-12)
        P = 1.0 - model.N - model.XBprer - model.XBpostr
        for value in (model.N, model.XBprer, model.XBpostr, P):
            assert np.all((value >= -1e-12) & (value <= 1.0 + 1e-12)), k
        peak = max(peak, model.XBpostr[0])
    assert peak > 0.3, "the twitch did not activate the cell"


def test_step_length_independence():
    """A 0.4 s twitch at 1 ms calls stays within 1e-3 of the same twitch at 0.05 ms calls."""
    coarse, fine = _twitch_tension(1e-3), _twitch_tension(5e-5)
    assert fine.max() > 0.3, "the twitch did not activate the cell"
    assert np.max(np.abs(coarse - fine)) < 1e-3


# ---------------------------------------------------------------------------
# Review focus
# ---------------------------------------------------------------------------


def test_no_calcium_is_finite_and_silent():
    """No calcium, or an EP model's slight undershoot below zero, must not divide by zero."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = Rice2008(2)
        for _ in range(200):
            model.advance_step(1e-3, np.array([0.0, -0.1]), 2.2, 0.0)
        for name, value in _getters(model).items():
            assert np.all(np.isfinite(value)), name
        assert np.all(model.get_active_tension() >= 0.0)


def test_lengths_outside_the_paper_range():
    """A FEM solver can hand over 1.1 um, below any overlap, or 2.6 um, past Fig. 1B's range."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = Rice2008(2)
        for _ in range(300):
            model.advance_step(1e-3, 2.0, np.array([1.1, 2.6]), 0.0)
        for name, value in _getters(model).items():
            assert np.all(np.isfinite(value)), name
        assert model.get_active_tension()[0] == 0.0
        assert model.get_active_stiffness()[0] == 0.0
        assert model.get_active_tension()[1] > 0.1


def test_fast_shortening_at_body_temperature():
    """
    At 37 degC, compressed crossbridges isomerize faster as exp(5 (x/x_0)^2). A 20 ms ramp at
    -5 um/s from full activation, at 1 ms calls, stays within 2e-3 of the same ramp at 0.01 ms
    calls.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        coarse = _maximally_activated(duration=0.3, TmpC=37.0)
        fine = Rice2008(1, params={"TmpC": 37.0})
        fine.set_state(coarse.get_state())

        dSL, Ta_coarse, Ta_fine = -5.0, [], []
        for k in range(20):
            coarse.advance_step(1e-3, 1e4, 2.3 + dSL * (k + 1) * 1e-3, dSL)
            Ta_coarse.append(coarse.get_active_tension()[0])
            for j in range(100):
                fine.advance_step(1e-5, 1e4, 2.3 + dSL * (100 * k + j + 1) * 1e-5, dSL)
            Ta_fine.append(fine.get_active_tension()[0])

        assert np.all(np.isfinite(Ta_coarse)) and np.all(np.isfinite(Ta_fine))
        assert fine.xXBprer[0] < -0.05 * fine.p["x_0"], "the ramp did not compress the bridges"
        assert np.max(np.abs(np.array(Ta_coarse) - np.array(Ta_fine))) < 2e-3


def test_velocity_estimated_from_successive_lengths():
    """With dSL_vals=None, dSL is (SL_k - SL_{k-1}) / dt, and zero on the first call."""
    estimated, explicit = Rice2008(2), Rice2008(2)
    dt, SL_prev = 1e-3, None
    for k in range(30):
        SL = np.array([2.2 - 0.002 * k, 2.0 + 0.001 * k])
        estimated.advance_step(dt, 2.0, SL)
        dSL = np.zeros(2) if SL_prev is None else (SL - SL_prev) / dt
        explicit.advance_step(dt, 2.0, SL, dSL)
        SL_prev = SL
    explicit_state = explicit.get_state()
    for key, value in estimated.get_state().items():
        np.testing.assert_array_equal(value, explicit_state[key], err_msg=key)
    assert estimated.xXBprer[0] < 0.0 < estimated.xXBprer[1]


def test_state_moves_between_parameter_sets():
    """The Ktr protocol: a few ms with the rates of a second instance, via get_state/set_state."""
    a = Rice2008(2)
    for _ in range(200):
        a.advance_step(1e-3, np.array([2.0, 10.0]), 2.2, 0.0)
    b = Rice2008(2, params={"fapp": 2.5, "hf": 10.0, "gapp": 14000.0, "hb": 80000.0})
    b.set_state(a.get_state())
    for name in _STATES:
        np.testing.assert_array_equal(getattr(b, name), getattr(a, name), err_msg=name)

    attached = b.XBprer + b.XBpostr
    b.advance_step(1e-3, np.array([2.0, 10.0]), 2.2, 0.0)
    assert np.all(b.XBprer + b.XBpostr < attached)
    assert np.all(np.isfinite(b.get_active_tension()))
