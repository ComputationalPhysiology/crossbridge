"""
Unit tests for the Rice2008 model.

Test philosophy mirrors the other per-model suites:
  - Structural: initialization, the overlap fractions of Fig. 1B, the rabbit parameters
  - Mathematical: normalized maximal force, steady strains, conservation of the
    occupancies, step-length independence
  - Review focus: no calcium, lengths outside the paper's range, fast shortening at body
    temperature, a fast release within one call and its batch-mates, velocity estimated
    from successive lengths, a state moved between parameter sets

The reference values come from an independent Radau solution of the CellML equations in s.
At 22.5 degC that solution reproduces the authors' Data S1 sample outputs: the Fig. 5 twitch
within 7e-6, and the F-Ca ramp within 2.5e-5.
"""

import warnings

import numpy as np
import pytest
from scipy.optimize import curve_fit

from crossbridge.rice2008 import _STATES, Rice2008
from crossbridge.utils import calcium_trace, rice_calcium


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
    2.3 um gives 0.98787 at 37 degC and 0.92282 at 22.5 degC (where Data S1's F-Ca ramp
    ends): maximal force rises with temperature, as the paper says.
    """
    warm, cold = Ta_at(TmpC=37.0), Ta_at(TmpC=22.5)
    assert warm == pytest.approx(0.98787, abs=2e-3)
    assert cold == pytest.approx(0.92282, abs=2e-3)
    assert cold < warm


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


@pytest.mark.parametrize("TmpC, dSL", [(22.5, -100.0), (37.0, -150.0)])
def test_fast_release_in_one_call(TmpC, dSL):
    """
    A 1 ms call shortening at 100-150 um/s, where 0.1 ms sub-steps alone went to NaN: each
    sub-step moves SL by at most 0.35 x_0, which keeps Ta within 1% of the same call at
    1 us steps.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        activated = Rice2008(1, params={"TmpC": TmpC})
        for _ in range(300):
            activated.advance_step(1e-3, 2.0, 2.0, 0.0)
        coarse, fine = (Rice2008(1, params={"TmpC": TmpC}) for _ in range(2))
        for model in (coarse, fine):
            model.set_state(activated.get_state())

        SL = 2.0 + dSL * 1e-3
        coarse.advance_step(1e-3, 2.0, SL, dSL)
        for _ in range(1000):
            fine.advance_step(1e-6, 2.0, SL, dSL)
        Ta, Ta_ref = coarse.get_active_tension()[0], fine.get_active_tension()[0]

    assert np.isfinite(Ta)
    assert abs(Ta - Ta_ref) < 0.01 * max(abs(Ta_ref), 0.01), (Ta, Ta_ref)


def test_fast_cell_leaves_its_batch_mates_alone():
    """
    Each cell takes its own number of sub-steps, so a cell shortening at 150 um/s (62 of them
    per 1 ms call) next to an isometric one (10) changes nothing for either: both evolve
    bit-identically to running alone.
    """
    SL0, dSL = np.array([2.2, 2.2]), np.array([-150.0, 0.0])
    batch = Rice2008(2)
    singles = [Rice2008(1), Rice2008(1)]
    for k in range(53):
        moving = k >= 50  # activate, then release cell 0 for 3 ms, to SL 1.75 um
        SL = SL0 + dSL * (k - 49) * 1e-3 if moving else SL0
        v = dSL if moving else np.zeros(2)
        batch.advance_step(1e-3, 2.0, SL, v)
        for i, single in enumerate(singles):
            single.advance_step(1e-3, 2.0, SL[i : i + 1], v[i : i + 1])

    batch_state = batch.get_state()
    for i, single in enumerate(singles):
        for key, value in single.get_state().items():
            if np.ndim(value):
                np.testing.assert_array_equal(batch_state[key][i], value[0], err_msg=f"{key}[{i}]")
    assert batch.xXBprer[0] < 0.0 < batch.get_active_tension()[1]


def test_strain_coefficients_survive_hfT_underflow():
    """
    Past about 12 x_0 of strain hfT underflows to zero, and dutypostr with it. Eqs. 29-30 are
    written with the duty fractions multiplied out, so the system stays finite where the
    CellML's xPsi / dutypostr * hfT is 0/0, and equals that form where both are defined.
    """
    model = Rice2008(2)
    i, j = _STATES.index("xXBprer"), _STATES.index("xXBpostr")
    x = model._get_x()
    x[i], x[j] = [0.003, 13 * 0.007], [0.009, 14 * 0.007]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rates = model._rates(x, np.full(2, 2.2))
        M = model._system(np.full(2, 2.0), np.array([-1.0, 300.0]), rates)
    assert rates["hfT"][1] == 0.0 and rates["dutypostr"][1] == 0.0
    assert np.all(np.isfinite(M))

    xPsi, hbT, hfT = model.p["xPsi"], rates["hbT"][0], rates["hfT"][0]
    np.testing.assert_allclose(M[0, i, j], xPsi / rates["dutyprer"][0] * hbT, rtol=1e-13)
    np.testing.assert_allclose(M[0, j, i], xPsi / rates["dutypostr"][0] * hfT, rtol=1e-13)


@pytest.mark.parametrize("TmpC", [22.5, 37.0])
def test_extreme_strains_stay_finite_and_silent(TmpC):
    """
    Strains far outside anything the dynamics reach -- every combination of 5 and 20 x_0
    either side of their isometric values -- with SL held or stretched at 300 um/s: the cap
    on the exponents of hfmd and gxbmd keeps every getter finite, with no warning.
    """
    x_0 = Rice2008.default_parameters()["x_0"]
    offsets = np.array([-20.0, -5.0, 5.0, 20.0]) * x_0
    pre, post = (a.ravel() for a in np.meshgrid(offsets, x_0 + offsets, indexing="ij"))
    dSL = np.repeat([0.0, 300.0], pre.size)  # each combination held and stretched
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        activated = Rice2008(dSL.size, params={"TmpC": TmpC})
        for _ in range(300):
            activated.advance_step(1e-3, 2.0, 2.0, 0.0)
        model = Rice2008(dSL.size, params={"TmpC": TmpC})
        model.set_state(activated.get_state())
        model.xXBprer, model.xXBpostr = np.tile(pre, 2), np.tile(post, 2)
        model.advance_step(1e-3, 2.0, 2.0 + dSL * 1e-3, dSL)
        for name, value in _getters(model).items():
            assert np.all(np.isfinite(value)), name


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


def _sustained_stretch(v, dt_call, TmpC=22.5):
    """
    Ta, Ka and bound_ca after each 1 ms of: 60 ms isometric at 2 um with Ca rising at 20 ms,
    then 14 ms in which SL lengthens at `v` um/s for 4 ms and is held. The interval is split
    into calls of `dt_call`, so that a call of 1e-5 s is a sub-step of 1e-5 s.
    """
    model = Rice2008(1, params={"TmpC": TmpC})
    Ca = lambda t: calcium_trace(np.array([t]), c0=0.1, cmax=3.0, t0=0.02)[0]  # noqa: E731
    for k in range(60):
        model.advance_step(1e-3, Ca(k * 1e-3), 2.0, 0.0)
    out = []
    for k in range(14):
        SL = 2.0 + v * 0.004 * min((k + 1) / 4, 1.0)
        for _ in range(round(1e-3 / dt_call)):
            model.advance_step(dt_call, Ca(0.06 + k * 1e-3), SL, v if k < 4 else 0.0)
        out.append(
            [
                model.get_active_tension()[0],
                model.get_active_stiffness()[0],
                model.bound_calcium_fraction()[0],
            ]
        )
    return np.array(out)


@pytest.mark.parametrize(
    "TmpC, v, bound",
    [(22.5, 20.0, 1.3e-3), (22.5, 44.0, 2.2e-3), (22.5, 100.0, 2e-6), (37.0, 14.0, 2e-3)],
)
def test_sustained_lengthening_converges_to_the_model(TmpC, v, bound):
    """
    The class docstring's sustained lengthening (4 ms ramps from peak tension, then held), at
    the velocities of its largest errors and at 100 um/s. The reference is the class at
    1e-5 s calls, which is within 2.4e-5 (22.5 degC) and 4.9e-5 (37 degC) of the peak of a
    Radau solution of rice2008.ode (rtol 1e-11) at every velocity measured, 5-300 um/s;
    `tests/test_ode_files.py` compares a 50 um/s stretch with Radau itself. 1 ms calls are
    within `bound`, about 1.5 times the error measured: 8.3e-4 at 20 um/s (Ta) and 1.4e-3 at
    44 (Ka), the worst at 22.5 degC, 1.3e-6 at 100 um/s, and 1.3e-3 at 14 um/s (Ta), the worst
    at 37 degC. That error is the integrator's: 2e-5 s calls are within 1.0e-4 of the
    reference (asserted to 2e-4).
    """
    shipped = _sustained_stretch(v, 1e-3, TmpC)
    fine = _sustained_stretch(v, 2e-5, TmpC)
    finest = _sustained_stretch(v, 1e-5, TmpC)
    peak = np.abs(finest).max(axis=0)
    assert np.all(np.abs(fine - finest).max(axis=0) / peak < 2e-4)
    assert np.all(np.abs(shipped - finest).max(axis=0) / peak < bound)


# ---------------------------------------------------------------------------
# The paper's numbers (Rice et al. 2008, Figs. 3, 5 and 8)
#
# The pins are from an independent Radau solution of the CellML equations in s at 22.5 degC,
# which reproduces the authors' Data S1 outputs (Fig. 5 twitch within 7e-6, F-Ca ramp within
# 2.5e-5, Fig. 8's force before the release exactly). The values marked "Data S1" below are
# the authors' own.
# ---------------------------------------------------------------------------


def test_rice_calcium_table3():
    t_peak = np.log(0.11 / 0.02) * 0.02 * 0.11 / (0.11 - 0.02)  # 0.0416716 s
    Ca = rice_calcium(np.array([-0.1, 0.0, t_peak, 1.0]))
    assert Ca[0] == Ca[1] == 0.09
    assert Ca[2] == pytest.approx(1.45, rel=1e-12)  # the peak is exactly the amplitude
    assert rice_calcium(np.linspace(0, 1, 100001)).max() == pytest.approx(1.45, rel=1e-8)
    assert Ca[3] == pytest.approx(0.09, abs=1e-3)


def test_rice_calcium_rejects_equal_time_constants():
    with pytest.raises(ValueError, match="must differ"):
        rice_calcium(0.05, tau1=0.03, tau2=0.03)


def _steady_state(Ca, SL, hold=1.0, **params):
    """One cell per (Ca, SL) pair, held for `hold` s at 1 ms calls, in a single run."""
    Ca, SL = np.broadcast_arrays(np.asarray(Ca, dtype=float), np.asarray(SL, dtype=float))
    model = Rice2008(Ca.size, params=params)
    for _ in range(round(hold / 1e-3)):
        model.advance_step(1e-3, Ca.ravel(), SL.ravel(), 0.0)
    return model


def _hill(pCa, Fmax, pCa50, nH):
    return Fmax / (1.0 + 10.0 ** (nH * (pCa - pCa50)))


def _fit_hill(pCa, F):
    p0 = [max(np.max(F), 1e-3), 6.0, 4.0]
    return curve_fit(_hill, pCa, F, p0=p0)[0]


@pytest.mark.parametrize(
    "SL, Fmax, pCa50, nH",
    [(1.8, 0.5319, 5.8866, 6.719), (2.2, 0.8331, 6.0999, 7.345), (2.3, 0.9216, 6.1493, 7.591)],
)
def test_force_pca_fig3a(SL, Fmax, pCa50, nH):
    """Fig. 3A, isosarcometric F-pCa: the model's fits, and the paper's dashed nH of 7.6."""
    pCa = np.linspace(4.5, 7.0, 26)
    F = _steady_state(10.0 ** (6.0 - pCa), SL).get_active_tension()
    fit = _fit_hill(pCa, F)
    assert fit[0] == pytest.approx(Fmax, abs=0.005)
    assert fit[1] == pytest.approx(pCa50, abs=0.02)
    assert fit[2] == pytest.approx(nH, abs=0.25)
    if SL == 2.3:
        assert fit[2] == pytest.approx(7.6, abs=0.5)  # the paper's dashed curve


def _zero(SL, g):
    """Per row, the zero of the line through the pair of neighbours of the first sign change."""
    out = np.empty(SL.shape[0])
    for r, (x, y) in enumerate(zip(SL, g)):
        up = np.nonzero((y[:-1] <= 0) & (y[1:] > 0))[0]
        k = up[0] if up.size else int(np.argmin(np.abs(y[:-1]) + np.abs(y[1:])))
        out[r] = x[k] - y[k] * (x[k + 1] - x[k]) / (y[k + 1] - y[k])
    return out


def _static_balances(pCa, residuals):
    """
    Sarcomere length at which each of `residuals` (functions of Ta, Tp and SL, increasing in
    SL) vanishes, for every pCa, from steady states. Returns the lengths, and the passive
    tension at 2.2 um.

    The balance of Fig. 3B and 3C is a root in SL of the steady-state force, each evaluation
    of which holds a cell for 1 s, so the root is found in three rounds of one vectorized run
    each: a coarse grid (0.1 um) that brackets it, three lengths 0.01 um apart around the
    interpolated root, and three lengths 0.005 um apart around that. Only the last round holds
    1 s, the others 0.3 and 0.5 s, which place the next grid and nothing more. The root is the
    linear interpolation (or extrapolation) over the last 0.005 um.
    """
    Ca = 10.0 ** (6.0 - pCa)[:, None]
    coarse = np.round(np.arange(1.4, 2.2 + 1e-9, 0.1), 6)[None, :] * np.ones((pCa.size, 1))
    model = _steady_state(Ca, coarse, hold=0.3)
    Ta, Tp = model.get_active_tension(), model.get_passive_tension()
    Ta, Tp = Ta.reshape(coarse.shape), Tp.reshape(coarse.shape)
    preload = Tp[0, -1]  # at 2.2 um, titin only: collagen starts at 2.25
    guess = [_zero(coarse, f(Ta, Tp, coarse, preload)) for f in residuals]

    for half_width, hold in [(0.01, 0.5), (0.005, 1.0)]:
        SL = np.concatenate(
            [g[:, None] + half_width * np.array([-1.0, 0.0, 1.0]) for g in guess], axis=0
        )
        model = _steady_state(np.tile(Ca, (len(residuals), 1)), SL, hold=hold)
        Ta, Tp = model.get_active_tension(), model.get_passive_tension()
        Ta, Tp = Ta.reshape(SL.shape), Tp.reshape(SL.shape)
        guess = [
            _zero(
                SL[i * pCa.size : (i + 1) * pCa.size],
                f(
                    Ta[i * pCa.size : (i + 1) * pCa.size],
                    Tp[i * pCa.size : (i + 1) * pCa.size],
                    SL[i * pCa.size : (i + 1) * pCa.size],
                    preload,
                ),
            )
            for i, f in enumerate(residuals)
        ]
    return guess, preload


_PCA_3BC = np.linspace(5.0, 6.8, 19)


@pytest.fixture(scope="module")
def static_balances():
    """Sarcomere lengths of Fig. 3B (KSE 50 and 1) and Fig. 3C, for `_PCA_3BC`."""
    residuals = [
        # a cell at 2.2 um with a series spring: Ta + Tp(SL) - Tp_titin(2.2) = KSE (2.2 - SL)
        lambda Ta, Tp, SL, preload: Ta + Tp - preload - 50.0 * (2.2 - SL),
        lambda Ta, Tp, SL, preload: Ta + Tp - preload - 1.0 * (2.2 - SL),
        # an unloaded cell: Ta + Tp = 0
        lambda Ta, Tp, SL, preload: Ta + Tp,
    ]
    return _static_balances(_PCA_3BC, residuals)[0]


@pytest.mark.parametrize(
    "i, KSE, pCa50, nH, paper",
    [(0, 50.0, 6.097, 7.155, (6.1, 7.6)), (1, 1.0, 6.004, 4.107, (6.0, 4.0))],
)
def test_fixed_length_force_pca_fig3b(static_balances, i, KSE, pCa50, nH, paper):
    """
    Fig. 3B, a cell held at 2.2 um by a series spring of stiffness KSE: the sarcomere
    shortens until ``Ta + Tp(SL) - Tp_titin(2.2) = KSE (2.2 - SL)``, and the force is
    ``KSE (2.2 - SL)``. The preload is titin only, collagen being zero below 2.25 um.
    """
    fit = _fit_hill(_PCA_3BC, KSE * (2.2 - static_balances[i]))
    assert fit[1] == pytest.approx(pCa50, abs=0.02)
    assert fit[2] == pytest.approx(nH, abs=0.25)
    assert fit[1] == pytest.approx(paper[0], abs=0.1)
    assert fit[2] == pytest.approx(paper[1], abs=0.5)


def test_unloaded_cell_shortening_fig3c(static_balances):
    """
    Fig. 3C, an unloaded cell starting from 1.9 um: it shortens until ``Ta + Tp = 0``, and
    the Hill fit is to the shortening ``1.9 - SL``. The paper has "1.4-1.5 um" at full
    activation.
    """
    SL = static_balances[2]
    fit = _fit_hill(_PCA_3BC, 1.9 - SL)
    assert fit[1] == pytest.approx(5.980, abs=0.02)
    assert fit[2] == pytest.approx(2.772, abs=0.25)
    assert fit[1] == pytest.approx(5.9, abs=0.1)
    assert fit[2] == pytest.approx(3.0, abs=0.5)
    assert SL.min() == pytest.approx(1.4362, abs=0.005)
    assert 1.4 <= SL.min() <= 1.5


# (SL [um], peak calcium [uM]) of the Fig. 5 twitches
_TWITCHES = [(1.8, 1.45), (2.2, 0.85), (2.3, 1.45), (2.2, 1.45)]


@pytest.fixture(scope="module")
def twitches():
    """Isometric twitches of `_TWITCHES`, 0.6 s from reset() at 1 ms calls: t, force (cells)."""
    SL = np.array([s for s, _ in _TWITCHES])
    amp = np.array([a for _, a in _TWITCHES])
    model = Rice2008(len(_TWITCHES))
    t = np.arange(601) * 1e-3
    F = [model.get_active_tension()]
    for k in range(600):
        # calcium at the call's midpoint, as in _twitch_tension
        Ca = rice_calcium((k + 0.5) * 1e-3, amplitude=amp)
        model.advance_step(1e-3, Ca, SL, 0.0)
        F.append(model.get_active_tension())
    return t, np.array(F)


def _duration_peak(t, F):
    """50%-to-50% duration, peak and its time, interpolating the crossings linearly."""
    half = 0.5 * F.max()
    above = np.nonzero(F >= half)[0]
    i, j = above[0], above[-1]
    t_up = np.interp(half, [F[i - 1], F[i]], [t[i - 1], t[i]])
    t_down = np.interp(half, [F[j + 1], F[j]], [t[j + 1], t[j]])
    return t_down - t_up, F.max(), t[np.argmax(F)]


def test_twitch_durations_fig5(twitches):
    """
    Fig. 5's 50%-to-50% twitch durations.

    The caption's 0.140, 0.187 and 0.223 s are not used: the paper's own plotted curves
    cross 50% at about 0.13, 0.16 and 0.21 s, where the model does, and Data S1 gives
    0.2150 s for the last. 5B was run at 2.2 um, as the text says, not 2.3 um.
    """
    t, F = twitches
    d = {c: _duration_peak(t, F[:, c])[0] for c in range(3)}
    assert d[0] == pytest.approx(0.1273, abs=2e-3)  # SL 1.8, 1.45 uM
    assert d[1] == pytest.approx(0.1574, abs=2e-3)  # SL 2.2, 0.85 uM
    assert d[2] == pytest.approx(0.2150, abs=2e-3)  # SL 2.3, 1.45 uM (Data S1: 0.2150)
    assert d[0] < d[1] < d[2]


def test_twitch_peaks_fig5(twitches):
    """Fig. 5's peaks; SL 2.3 / 1.45 uM is Data S1's sample twitch (0.8256 at 0.141 s)."""
    t, F = twitches
    _, peak, t_peak = _duration_peak(t, F[:, 2])
    assert peak == pytest.approx(0.8256, abs=2e-3)
    assert t_peak == pytest.approx(0.1406, abs=1e-3)
    assert F[:, 3].max() == pytest.approx(0.6930, abs=2e-3)  # 5B, SL 2.2 / 1.45 uM: ~0.69
    assert F[:, 1].max() == pytest.approx(0.0996, abs=2e-3)  # 5B, SL 2.2 / 0.85 uM: ~0.10
    assert peak == pytest.approx(0.825, abs=0.01)  # 5A, read off the figure


def test_isometric_force_before_ktr_release():
    """Fig. 8, Ca 10 uM, SL 2.2 um, 20 degC: 0.80933 before the release (Data S1: 0.80933)."""
    F = _steady_state(10.0, 2.2, TmpC=20.0).get_active_tension()[0]
    assert F == pytest.approx(0.80933, abs=2e-3)
