"""
Tran2010 against the claims of its docstring: the figures of Tran et al. (2010) that the model
reproduces (model values pinned; the paper's, read off its figures, in the comments), the one
claim it does not, and the two departures from the CellML (the proton concentration, against
the CellML's; detachment to N, against the paper's Appendix).
"""

import numpy as np
import pytest

from crossbridge import Rice2008, Tran2010
from crossbridge.utils import rice_calcium

brentq = pytest.importorskip("scipy.optimize").brentq


def _model(**params) -> Tran2010:
    return Tran2010(1, params={**Tran2010.default_parameters(), **params})


def _steady_state(m: Tran2010, Ca: float, SL: float) -> np.ndarray:
    """The isometric steady state, exactly: troponin and the occupancies at their steady state
    for the rates at strains 0 and x_0, which are the strains' isometric steady state."""
    x = m._get_x()
    konCa = m._konT * Ca
    x[0] = konCa / (konCa + m._koffHT)
    x[1] = konCa / (konCa + m._koffLT)
    x[7], x[8] = 0.0, m.p["x_0"]
    rates = m._rates(x, np.array([SL]))
    x[4:7, 0] = m._blocks(np.array([Ca]), np.zeros(1), rates).occ_ss[0]
    return x


def steady_force(Ca: float = 200.0, SL: float = 2.2, **params) -> float:
    m = _model(**params)
    m._set_x(_steady_state(m, Ca, SL))
    m._SL_curr = np.array([SL])
    return float(m.get_active_tension()[0])


def ca50(pH: float, **params) -> float:
    fmax = steady_force(Ca=1e4, pH=pH, **params)
    return 10 ** brentq(
        lambda e: steady_force(Ca=10**e, pH=pH, **params) - 0.5 * fmax, -2.0, 3.0, xtol=1e-12
    )


#: Fig. 2B-D's conditions
FIG2 = {"TmpC": 30.0, "SL": 2.1}


def test_steady_state_is_where_the_class_settles():
    """The exact steady state of the tests is the class's, run 2 s at 1 ms calls."""
    m = _model(TmpC=30.0)
    for _ in range(2000):
        m.advance_step(1e-3, 200.0, 2.1)
    expected = _steady_state(_model(TmpC=30.0), 200.0, 2.1)
    states = [0, 1, 4, 5, 6, 7, 8]  # all but the units outside the overlap, which feed nothing
    np.testing.assert_allclose(m._get_x()[states], expected[states], atol=1e-12)


def test_fig2a_force_against_pi():
    # paper's curve: about 72, 57, 49, 42, 33
    F0 = steady_force(TmpC=22.0, Pi=1e-9)
    rel = [100 * steady_force(TmpC=22.0, Pi=P) / F0 for P in (5.0, 10.0, 15.0, 20.0, 30.0)]
    np.testing.assert_allclose(rel, [73.5, 58.1, 48.1, 41.0, 31.6], atol=0.05)


def test_fig2b_ca50_against_ph():
    # paper's curve: about 48, 70, 100, 140, 355
    ref = ca50(7.0, **FIG2)
    rel = [100 * ca50(pH, **FIG2) / ref for pH in (7.5, 7.2, 6.8, 6.4)]
    np.testing.assert_allclose(rel, [41.9, 68.4, 150.8, 363.0], atol=0.05)


def test_fig2b_with_the_cellmls_proton_concentration():
    """The CellML's H is ten times [H+]: pH and pH_ref one unit lower give it, kdHCa unchanged."""
    cellml = {**FIG2, "pH_ref": 6.15}
    ref = ca50(6.0, **cellml)
    rel = [100 * ca50(pH - 1.0, **cellml) / ref for pH in (7.5, 7.2, 6.8, 6.4)]
    np.testing.assert_allclose(rel, [32.1, 63.1, 159.5, 408.7], atol=0.05)


def test_cellmls_fxb_changes_a_twitch_by_less_than_1e_8():
    """
    With the CellML's H, fxbT is ten times larger; pH and pH_ref one unit lower and kdHCa ten
    times larger give that fxbT and the class's konT.
    """
    t = np.arange(0.0, 0.5, 1e-3)
    Ca = rice_calcium(t)
    sl = lambda s: 2.2 - 0.2 * np.clip((s - 0.15) / 0.02, 0.0, 1.0)  # noqa: E731
    out = []
    for params in ({}, {"pH": 6.15, "pH_ref": 6.15, "kdHCa": 0.2}):
        m, rows = _model(**params), []
        for k in range(len(t)):
            m.advance_step(1e-3, Ca[k], sl(t[k]), dSL_vals=(sl(t[k] + 1e-3) - sl(t[k])) / 1e-3)
            rows.append([m.get_active_tension()[0], m.get_active_stiffness()[0]])
        out.append(np.array(rows))
    assert np.all(np.abs(out[0] - out[1]).max(axis=0) / np.abs(out[0]).max(axis=0) < 1e-8)


def test_fig2d_maximal_force_and_ca50():
    # paper: relative force 1, 0.89, 0.77, 0.49; Ca50 about 0.75, 1.0, 1.5, 4 uM; force 1.18
    f72 = steady_force(pH=7.2, **FIG2)
    rel = [steady_force(pH=pH, **FIG2) / f72 for pH in (7.0, 6.8, 6.4)]
    np.testing.assert_allclose(rel, [0.902, 0.782, 0.505], atol=5e-4)
    assert f72 == pytest.approx(0.8209, abs=1e-4)
    np.testing.assert_allclose(
        [ca50(pH, **FIG2) for pH in (7.2, 7.0, 6.8, 6.4)], [0.706, 1.032, 1.555, 3.744], atol=5e-4
    )


def test_fig2c_is_below_the_papers_curve_at_the_ends():
    # paper's curve, normalized at pH 7: about 142 at pH 8 and 38 at pH 6
    f7 = steady_force(pH=7.0, **FIG2)
    rel = [100 * steady_force(pH=pH, **FIG2) / f7 for pH in (8.0, 6.0)]
    np.testing.assert_allclose(rel, [131.2, 29.6], atol=0.05)


def test_fig3a_mgadp_reversal_point():
    # paper: the curves cross at 0.9 mM MgATP
    c = {"TmpC": 20.0, "SL": 2.1, "pH": 7.0, "Pi": 1e-9}

    def gap(log_atp: float) -> float:
        atp = 10**log_atp
        return steady_force(MgADP=1e-6, MgATP=atp, **c) - steady_force(MgADP=0.01, MgATP=atp, **c)

    cross = 10 ** brentq(gap, -1.5, 0.8, xtol=1e-12)
    assert cross == pytest.approx(0.818, abs=1e-3)
    assert gap(-1.5) > 0.0 > gap(0.8)  # more MgADP: less force below, more above


def test_fig3b_steady_force_after_redevelopment():
    # paper: 0.52 and 0.25 of the force at 0.1 mM MgATP
    c = {"TmpC": 20.0, "SL": 2.1, "pH": 7.0, "MgADP": 0.001, "Pi": 2.0}
    f = [steady_force(MgATP=a, **c) for a in (0.1, 3.0, 10.0)]
    np.testing.assert_allclose([f[1] / f[0], f[2] / f[0]], [0.5177, 0.2392], atol=5e-4)


def test_fig5_maximal_force_through_ischemia():
    # paper: about 0.87, 0.34, 0.20, 0.08, 0.05 at 0, 2.5, 5, 10, 15 min
    points = [
        (5.0, 2.0, 0.036, 7.15),
        (4.5, 13.0, 0.20, 6.75),
        (4.3, 15.0, 0.33, 6.5),
        (3.2, 22.0, 0.25, 6.27),
        (1.6, 27.0, 0.16, 6.15),
    ]
    force = [steady_force(TmpC=25.0, MgATP=a, Pi=pi, MgADP=adp, pH=pH) for a, pi, adp, pH in points]
    np.testing.assert_allclose(force, [0.8505, 0.3343, 0.1925, 0.0898, 0.0588], atol=5e-4)


def test_mgadp_does_not_raise_force_by_13_percent():
    """The text's 13% from 0 to 3 mM MgADP, by which kdADP was fitted, does not reproduce."""
    ratio = steady_force(TmpC=22.0, MgADP=3.0) / steady_force(TmpC=22.0, MgADP=1e-9)
    assert ratio == pytest.approx(2.281, abs=1e-3)


def _paper_structure_force(Ca: float, SL: float, **params) -> float:
    """The isometric steady force with the paper's Appendix structure: detachment to P and fxbT
    from P, a Markov chain N <-> P <-> XBprer <-> XBpostr <-> P."""
    m = _model(**params)
    x = _steady_state(m, Ca, SL)
    r = {k: float(v[0]) for k, v in m._rates(x, np.array([SL])).items()}
    kn, kp = r["kn_pT"], r["kp_nT"]
    a1p, a1m, a2p, a2m, a3p, a3m = (r[k] for k in ("a1p", "a1m", "a2p", "a2m", "a3p", "a3m"))
    # generator Q (rows: from N, P, XBprer, XBpostr), steady state pi Q = 0, sum pi = 1
    Q = np.array(
        [
            [-kn, kn, 0.0, 0.0],
            [kp, -(kp + a1p + a3m), a1p, a3m],
            [0.0, a1m, -(a1m + a2p), a2p],
            [0.0, a3p, a2m, -(a3p + a2m)],
        ]
    )
    A = np.vstack([Q.T[:-1], np.ones(4)])
    pi = np.linalg.solve(A, [0.0, 0.0, 0.0, 1.0])
    SOVFthick, _ = m._overlap(np.array([SL]))
    return float(SOVFthick[0] * pi[3] / m._XBpostr_max)


def test_detachment_to_n_changes_the_figures_steady_forces_by_at_most_1_4_percent():
    cases = [({"TmpC": 22.0, "Pi": P}, 2.2) for P in (1e-9, 5.0, 10.0, 15.0, 20.0, 30.0)]
    cases += [
        ({"TmpC": 20.0, "pH": 7.0, "MgADP": 0.001, "Pi": 2.0, "MgATP": a}, 2.1)
        for a in (0.1, 3.0, 10.0)
    ]
    base_a, base_b = (
        steady_force(SL=2.2, **cases[0][0]),
        _paper_structure_force(200.0, 2.2, **cases[0][0]),
    )
    base_3b = [
        steady_force(SL=2.1, **cases[6][0]),
        _paper_structure_force(200.0, 2.1, **cases[6][0]),
    ]
    worst = 0.0
    for params, SL in cases:
        a, b = steady_force(SL=SL, **params), _paper_structure_force(200.0, SL, **params)
        # each figure normalized as the paper does: Fig. 2A by Pi = 0, Fig. 3B by 0.1 mM MgATP
        na, nb = (a / base_a, b / base_b) if SL == 2.2 else (a / base_3b[0], b / base_3b[1])
        worst = max(worst, abs(na - nb) / nb)
    assert 0.014 < worst < 0.0145


def test_reference_metabolites_give_rice2008s_cycle_rates():
    """At the reference levels the pseudo-first-order rates are Rice2008's (at Tran's species,
    temperature reference aside), and fxbT is negligible."""
    p = {**Tran2010.default_parameters(), "MgADP": 0.036}
    m = Tran2010(1, params=p)
    x = m._get_x()
    x[7], x[8] = 0.001, 0.008
    SL = np.array([2.1])
    r = m._rates(x, SL)
    assert r["a1m"][0] == pytest.approx(r["gappT"][0], rel=1e-14)
    assert r["a2m"][0] == pytest.approx(r["hbT"][0], rel=1e-14)
    assert r["a3p"][0] == pytest.approx(r["gxbT"][0], rel=1e-14)
    assert r["a3m"][0] < 1e-8 * r["a3p"][0]
    rice = Rice2008(1, params={k: v for k, v in p.items() if k in Rice2008.default_parameters()})
    assert m._TMPC_REF == 36.85 and rice._TMPC_REF == 37.0


def test_ph_slows_calcium_binding_competitively():
    """konT carries (H_ref^m + kdHCa^m) / (H^m + kdHCa^m), [H+] in uM."""
    base, acid = _model(), _model(pH=6.8)
    H, H_ref, kd = 10 ** (6 - 6.8), 10 ** (6 - 7.15), 0.02
    assert acid._konT / base._konT == pytest.approx((H_ref + kd) / (H + kd), rel=1e-14)


def test_atp_hydrolysis_rate_is_the_net_mgatp_step():
    m = Tran2010(2)
    t = np.arange(0.0, 0.3, 1e-3)
    for c in rice_calcium(t):
        m.advance_step(1e-3, c, np.array([2.2, 1.9]))
    r = m._rates(m._get_x(), m._SL_curr)
    SOVFthick, _ = m._overlap(m._SL_curr)
    np.testing.assert_allclose(
        m.get_atp_hydrolysis_rate(), SOVFthick * (r["a3p"] * m.XBpostr - r["a3m"] * m.P), rtol=1e-14
    )
    assert np.all(m.get_atp_hydrolysis_rate() > 0.0)


def test_n_is_derived_and_reset_starts_non_permissive():
    m = Tran2010(3)
    np.testing.assert_array_equal(m.N, 1.0)
    for _ in range(50):
        m.advance_step(1e-3, 5.0, 2.2)
    np.testing.assert_allclose(m.N, 1.0 - m.P - m.XBprer - m.XBpostr)
    assert np.all(m.P > 0.0)
    m.reset()
    np.testing.assert_array_equal(m.P, 0.0)
    np.testing.assert_array_equal(m.N, 1.0)


def test_rabbit_parameters_are_not_defined():
    with pytest.raises(NotImplementedError):
        Tran2010.rabbit_parameters()
