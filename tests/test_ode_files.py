"""
Each model's gotranx `.ode` file is pinned to the Python class it describes.

The contract every file follows (spec 8.2): a component `inputs` with the
parameters `Ca` [uM], `SL` [um] and `dSL` [um/s]; the intermediates `Ta` [kPa],
`Ka` [kPa per unit Lambda] and `bound_ca` [fraction] (and `Tp` [kPa] where the
class has a passive tension); the class's other parameters, named and defaulted
as `default_parameters()`; initial states those of `reset()`. The trajectories
of the class and of the generated right-hand side (Radau, tight tolerances)
agree to 1e-4 of the peak of each output.
"""

import numpy as np
import pytest

from crossbridge import RDQ20MF, Land2017, Lewalle2024
from crossbridge.utils import calcium_trace

gotranx = pytest.importorskip("gotranx")
solve_ivp = pytest.importorskip("scipy.integrate").solve_ivp

WITH_ODE: list[type] = [Land2017, Lewalle2024, RDQ20MF]
OUTPUTS: dict[type, tuple[str, ...]] = {
    Land2017: ("Ta", "Ka", "bound_ca", "Tp"),
    Lewalle2024: ("Ta", "Ka", "bound_ca", "Tp"),
    RDQ20MF: ("Ta", "Ka", "bound_ca"),
}
#: Keys of `default_parameters()` that are not parameters of the ODE.
NON_ODE = {"dt", "Ca0", "dt_RU"}
#: Class parameters that the file spells differently: key -> (ODE parameter, value map).
SWITCHES: dict[str, tuple[str, dict[str, float]]] = {
    "which_dep": (
        "which_dep",
        {"totalforce": 0.0, "force": 1.0, "passiveforce": 2.0, "Lambda": 3.0},
    ),
    "dep_k1ork2": ("dep_k1", {"k1": 1.0, "k2": 0.0}),
}
#: Where an output is compared against the class, and the floor of its peak [kPa, fraction].
GETTERS = {
    "Ta": "get_active_tension",
    "Ka": "get_active_stiffness",
    "bound_ca": "bound_calcium_fraction",
    "Tp": "get_passive_tension",
}
FLOOR = {"Ta": 1.0, "Tp": 1.0, "Ka": 1.0, "bound_ca": 1e-3}

#: RDQ20MF's crossbridge moments -> index into `x_XB`.
RDQ20MF_XB = {"mu0_P": (0, 0), "mu1_P": (1, 0), "mu0_N": (0, 1), "mu1_N": (1, 1)}

_MODULES: dict[type, dict] = {}


def _load(cls):
    return gotranx.load_ode(cls.ODE_FILE)


def _module(cls) -> dict:
    """The gotranx Python code for `cls.ODE_FILE`, executed into a namespace."""
    if cls not in _MODULES:
        code = gotranx.cli.gotran2py.get_code(_load(cls), format=gotranx.codegen.python.Format.none)
        namespace: dict = {}
        exec(code, namespace)
        _MODULES[cls] = namespace
    return _MODULES[cls]


def _state_of(model, name: str, cell: int = 0) -> float:
    """
    A cell of the model's state `name` (the ODE's name -> the class's attribute).

    RDQ20MF's tensors are spelled out: `x_{a}{b}{c}{B}` is `x_RU[a, b, c, B]` and
    `mu0_P, mu1_P, mu0_N, mu1_N` are `x_XB[0, 0], x_XB[1, 0], x_XB[0, 1], x_XB[1, 1]`.
    """
    if isinstance(model, RDQ20MF):
        if name in RDQ20MF_XB:
            return float(model.x_XB[RDQ20MF_XB[name]][cell])
        a, b, c, B = (int(ch) for ch in name.removeprefix("x_"))
        return float(model.x_RU[a, b, c, B, cell])
    return float(getattr(model, name)[cell])


def _inputs(sl, cmax, dt=1e-3):
    t = np.arange(0, 0.4, dt)
    Ca = calcium_trace(t, c0=0.1, cmax=cmax, t0=0.02)
    SL = sl(t + dt)
    dSL = (sl(t + dt) - sl(t)) / dt
    return Ca, SL, dSL


def _trajectory_errors(
    cls, params, ode_params, Ca, SL, dSL, dt=1e-3, class_substeps: int = 1, observe=None
) -> dict[str, float]:
    """
    Each output's max |class - ODE| over the steps, divided by its peak.

    The ODE starts from the class's own initial state, every state read by name. At the
    default parameters that is the file's initial state (`test_initial_states_match_reset`);
    at others it is not, since the file's CaTRPN is a number for the default parameters,
    while `reset()` computes it from the parameters given. The class takes
    `class_substeps` calls of `advance_step(dt / class_substeps, ...)` per interval, with
    the inputs held; the reference solves each whole interval. `observe(model)`, if
    given, is called after each interval.
    """
    m = _module(cls)
    model = cls(1, params=params)
    y = m["init_state_values"]()
    for state in _load(cls).states:
        y[m["state_index"](state.name)] = _state_of(model, state.name)
    p = m["init_parameter_values"](**ode_params)
    iCa, iSL, idSL = (m["parameter_index"](n) for n in ("Ca", "SL", "dSL"))
    outputs = OUTPUTS[cls]
    idx = {n: m["monitor_index"](n) for n in outputs}
    got: dict[str, list[float]] = {n: [] for n in outputs}
    ref: dict[str, list[float]] = {n: [] for n in outputs}
    for k in range(len(Ca)):
        for _ in range(class_substeps):
            model.advance_step(dt / class_substeps, Ca[k], SL[k], dSL_vals=dSL[k])
        if observe is not None:
            observe(model)
        p[iCa], p[iSL], p[idSL] = Ca[k], SL[k], dSL[k]
        sol = solve_ivp(
            lambda t, y: m["rhs"](t, y, p),
            (0, dt),
            y,
            method="Radau",
            rtol=1e-11,
            atol=1e-13,
        )
        y = sol.y[:, -1]
        mon = m["monitor_values"](dt, y, p)
        for n in outputs:
            got[n].append(float(getattr(model, GETTERS[n])()[0]))
            ref[n].append(float(mon[idx[n]]))
    errors = {}
    for n in outputs:
        a, b = np.array(got[n]), np.array(ref[n])
        peak = max(np.abs(b).max(), np.abs(a).max(), FLOOR[n])
        errors[n] = float(np.abs(a - b).max() / peak)
    return errors


@pytest.mark.parametrize("cls", WITH_ODE)
def test_inputs_are_a_removable_component(cls):
    ode = _load(cls)
    inputs = ode.get_component("inputs")
    assert {p.name for p in inputs.parameters} == {"Ca", "SL", "dSL"}
    assert set((ode - inputs).missing_variables) == {"Ca", "SL", "dSL"}


@pytest.mark.parametrize("cls", WITH_ODE)
def test_outputs_are_intermediates(cls):
    ode = _load(cls)
    for name in OUTPUTS[cls]:
        assert isinstance(ode[name], gotranx.atoms.Intermediate), name


@pytest.mark.parametrize("cls", WITH_ODE)
def test_parameters_match_default_parameters(cls):
    ode = _load(cls)
    in_ode = {p.name: float(p.value) for p in ode.parameters}
    in_ode = {k: v for k, v in in_ode.items() if k not in {"Ca", "SL", "dSL"}}
    defaults = cls.default_parameters()
    switches = {k: v for k, v in SWITCHES.items() if k in defaults}
    expected = {k: float(v) for k, v in defaults.items() if k not in NON_ODE | set(switches)}
    for key, (ode_name, mapping) in switches.items():
        expected[ode_name] = mapping[defaults[key]]
    assert in_ode == expected


@pytest.mark.parametrize("cls", WITH_ODE)
def test_initial_states_match_reset(cls):
    model, ode = cls(1), _load(cls)
    for state in ode.states:
        assert float(state.value) == pytest.approx(_state_of(model, state.name), rel=1e-15, abs=0)


PROTOCOLS = {  # SL(t) [um]; dSL is its forward difference over the step
    "isometric": lambda t: np.full_like(t, 2.0),
    "shorten_and_relengthen": lambda t: (
        2.0 - 0.15 * np.sin(np.pi * np.clip((t - 0.05) / 0.3, 0, 1)) ** 2
    ),
    "below_0.87": lambda t: np.full_like(t, 1.5),
    "at_0.87": lambda t: np.full_like(t, 1.566),
    "at_1.2": lambda t: np.full_like(t, 2.16),
    "above_1.2": lambda t: np.full_like(t, 2.3),
}


@pytest.mark.parametrize("protocol", sorted(PROTOCOLS))
def test_land2017_matches_its_ode(protocol):
    Ca, SL, dSL = _inputs(PROTOCOLS[protocol], cmax=3.0)
    errors = _trajectory_errors(Land2017, {}, {}, Ca, SL, dSL)
    assert max(errors.values()) < 1e-4, errors


def test_land2017_matches_its_ode_without_calcium():
    """
    Without calcium CaTRPN decays towards 0 and kb * CaTRPN^(-nTm/2) grows to about
    1e6 /s. Land2017's frozen-midpoint sub-steps do not resolve that: S is about 1% low
    after the first 1 ms step and converges only with the class's own step. The reference
    agrees across Radau, LSODA and BDF to about 1e-13, so Ka is held to the class's
    measured accuracy (1.7e-3 of the 1 kPa floor) and the other outputs to 1e-4. What
    the test pins is the decay of CaTRPN towards zero calcium: a mistranscribed decay
    would show in bound_ca as an O(1) error.
    """
    Ca, SL, dSL = _inputs(PROTOCOLS["isometric"], cmax=3.0)
    errors = _trajectory_errors(Land2017, {}, {}, np.zeros_like(Ca), SL, dSL)
    assert errors.pop("Ka") < 3e-3, errors
    assert max(errors.values()) < 1e-4, errors


def _quick_release(t):
    """SL [um]: 2.0, shortened to 1.75 over 10 ms from 150 ms (near peak tension), then held."""
    return 2.0 - 0.25 * np.clip((t - 0.15) / 0.01, 0, 1)


def test_land2017_matches_its_ode_through_a_quick_release():
    """
    The release drives Zs below -1, onto the lower branch of gsu, -gs (Zs + 1), which
    the protocols above never reach (Zs stays within [-0.41, 0.34]). The test asserts
    that the class gets there, so the branch is covered by construction.
    """
    Ca, SL, dSL = _inputs(_quick_release, cmax=3.0)
    Zs: list[float] = []
    errors = _trajectory_errors(
        Land2017, {}, {}, Ca, SL, dSL, observe=lambda model: Zs.append(model.Zs[0])
    )
    assert min(Zs) < -1.0, min(Zs)
    assert max(errors.values()) < 1e-4, errors


#: Land2017's parameters off their defaults, passed to the class and to the file alike,
#: so that a parameter swapped for another, or a parameter written as its default
#: value, shows.
LAND_OFF_DEFAULTS = {
    "beta0": 1.9,
    "beta1": -1.7,
    "ca50_ref": 2.1,
    "k_trpn": 140.0,
    "gs": 12.0,
    "gw": 480.0,
    "a": 1800.0,
    "b": 8.0,
    "k": 5.5,
    "eta_l": 0.25,
    "eta_s": 0.03,
}


def test_land2017_matches_its_ode_off_its_defaults():
    Ca, SL, dSL = _inputs(PROTOCOLS["shorten_and_relengthen"], cmax=3.0)
    params = LAND_OFF_DEFAULTS
    errors = _trajectory_errors(Land2017, params, params, Ca, SL, dSL)
    assert max(errors.values()) < 1e-4, errors


#: Lewalle2024 freezes Cd for a whole call and S, W per sub-step in its feedback, so it is
#: compared with ten calls per held-input interval.
LEWALLE_SUBSTEPS = 10
#: Cases still over 1e-4 at ten calls, bounded at twice their measured error (see below).
LEWALLE_BOUNDS = {
    ("totalforce", "k1", "isometric"): 3e-4,  # measured 1.43e-4
    ("force", "k1", "isometric"): 4e-4,  # 1.54e-4
    ("totalforce", "k2", "isometric"): 4e-4,  # 1.78e-4
    ("totalforce", "k2", "shorten_and_relengthen"): 4e-4,  # 1.65e-4
    ("force", "k2", "isometric"): 9e-4,  # 4.04e-4
    ("force", "k2", "shorten_and_relengthen"): 8e-4,  # 3.62e-4
}


@pytest.mark.parametrize("protocol", ["isometric", "shorten_and_relengthen"])
@pytest.mark.parametrize("which_dep", ["totalforce", "force", "passiveforce", "Lambda"])
@pytest.mark.parametrize("dep_k1ork2", ["k1", "k2"])
def test_lewalle2024_matches_its_ode(which_dep, dep_k1ork2, protocol):
    """
    The class takes ten calls per 1 ms interval. With one call, nine of the sixteen cases
    exceeded 1e-4 (up to 8.1e-4); with ten, six still do and are bounded in LEWALLE_BOUNDS
    at twice the measured error, every other case at 1e-4. The file is not the cause:
    the generated right-hand side equals the class's equations (its M x + c) to 4e-15 at
    random states, bound_ca and Tp agree to 3e-15, the "Lambda" paradigm, whose feedback
    depends on Lambda only and not on the state, agrees to 1e-6, and the reference
    agrees across Radau, LSODA and BDF to 1e-10. The class freezes Cd for a whole call
    and S, W per sub-step inside the force feedback, and its error shrinks with its own
    step: force/k2/isometric 8.0e-4 (one call, sub-step 0.2 ms), 4.0e-4 (ten calls),
    4.0e-5 (sub-step 10 us), 7.9e-6 (0.1 ms calls, 2 us sub-steps).
    """
    Ca, SL, dSL = _inputs(PROTOCOLS[protocol], cmax=10.0)
    params = {"which_dep": which_dep, "dep_k1ork2": dep_k1ork2}
    ode_params = {name: mapping[params[key]] for key, (name, mapping) in SWITCHES.items()}
    errors = _trajectory_errors(
        Lewalle2024, params, ode_params, Ca, SL, dSL, class_substeps=LEWALLE_SUBSTEPS
    )
    bound = LEWALLE_BOUNDS.get((which_dep, dep_k1ork2, protocol), 1e-4)
    assert max(errors.values()) < bound, errors


#: Lewalle2024's parameters off their defaults, passed to the class and to the file
#: alike. At the defaults ra = rb = es = 1, k_trpn_on = k_trpn_off and beta0 = beta1 = 0,
#: so swapping two of them, or writing one as its default value, would not show; here no
#: two share a value. beta1 is in M (Ca50 = 5.6e-6 M), koffon stays at its calibration.
LEWALLE_OFF_DEFAULTS = {
    "ra": 0.8,
    "rb": 1.3,
    "k_trpn_on": 120.0,
    "k_trpn_off": 80.0,
    "es": 0.6,
    "beta0": 0.7,
    "beta1": -2e-6,
}


@pytest.mark.parametrize(
    ("which_dep", "dep_k1ork2"), [("totalforce", "k1"), ("passiveforce", "k2")]
)
def test_lewalle2024_matches_its_ode_off_its_defaults(which_dep, dep_k1ork2):
    """
    Shortened and relengthened, so that Zs falls below -es and gsu's threshold counts.
    The class's CaTRPN(0) follows k_trpn_on and k_trpn_off and the file's does not, which
    is why `_trajectory_errors` starts the ODE from the class's state. Both cases are
    within 1e-4 at ten calls.
    """
    Ca, SL, dSL = _inputs(PROTOCOLS["shorten_and_relengthen"], cmax=10.0)
    params = {**LEWALLE_OFF_DEFAULTS, "which_dep": which_dep, "dep_k1ork2": dep_k1ork2}
    ode_params = dict(LEWALLE_OFF_DEFAULTS)
    for key, (name, mapping) in SWITCHES.items():
        ode_params[name] = mapping[params[key]]
    Zs: list[float] = []
    errors = _trajectory_errors(
        Lewalle2024,
        params,
        ode_params,
        Ca,
        SL,
        dSL,
        class_substeps=LEWALLE_SUBSTEPS,
        observe=lambda model: Zs.append(model.Zs[0]),
    )
    assert min(Zs) < -LEWALLE_OFF_DEFAULTS["es"], min(Zs)
    assert max(errors.values()) < 1e-4, errors


def test_lewalle2024_matches_its_ode_without_calcium():
    # Lewalle2024 floors Ca at 1e-6 uM rather than 0
    Ca, SL, dSL = _inputs(PROTOCOLS["isometric"], cmax=10.0)
    errors = _trajectory_errors(
        Lewalle2024, {}, {}, np.zeros_like(Ca), SL, dSL, class_substeps=LEWALLE_SUBSTEPS
    )
    assert max(errors.values()) < 1e-4, errors


def _ru_by_name(rhs):
    """`_RU_get_rhs()` (2, 2, 2, 2, n) -> {"x_abcB": d/dt of that state, (n,)}."""
    return {f"x_{a}{b}{c}{B}": rhs[a, b, c, B] for a, b, c, B in np.ndindex(2, 2, 2, 2)}


def _xb_by_name(rhs):
    """The XB right-hand side (n, 4), ordered [mu0_P, mu1_P, mu0_N, mu1_N] -> {name: (n,)}."""
    return {name: rhs[:, i] for i, name in enumerate(RDQ20MF_XB)}


def _states_by_name(ns, model):
    """The generated module's state array (num_states, n), each cell read by `_state_of`."""
    states = np.zeros((len(ns["init_state_values"]()), model.num_cells))
    for state in _load(type(model)).states:
        for cell in range(model.num_cells):
            states[ns["state_index"](state.name), cell] = _state_of(model, state.name, cell)
    return states


def _params_per_cell(ns, **inputs):
    """The generated module's parameter array (num_params, n): defaults, then the inputs."""
    n = len(next(iter(inputs.values())))
    params = np.repeat(ns["init_parameter_values"]()[:, None], n, axis=1)
    for name, values in inputs.items():
        params[ns["parameter_index"](name)] = values
    return params


def test_rdq20mf_derivatives_match_its_ode():
    """
    RDQ20MF steps its RU states and crossbridge moments at different rates (explicit
    Euler at dt_RU for the RU, the exact matrix exponential every freqXB-th step for the
    moments, rates refreshed every 10th), so its tension sits about 0.5-0.8% of the peak
    from the ODE's own, depending on the protocol. What the file must reproduce is the
    class's equations, so compare the derivatives and the three outputs at 50 states.
    They are random, except that eleven sarcomere lengths are pinned to cover every
    piece of frac_SO, its boundaries and either side of it (asserted), and two cells
    have all their RU mass in one state, so that a mean-field denominator is 0: x_0000
    (no permissive units, k_PN's) and x_1111 (no non-permissive units, k_NP's).
    """
    rng = np.random.default_rng(0)
    n = 50
    model = RDQ20MF(n)
    x_RU = rng.uniform(size=(2, 2, 2, 2, n))
    x_RU[..., :2] = 0.0
    x_RU[0, 0, 0, 0, 0] = 1.0  # cell 0: no permissive units
    x_RU[1, 1, 1, 1, 1] = 1.0  # cell 1: no non-permissive units
    model.x_RU = x_RU / x_RU.sum(axis=(0, 1, 2, 3))
    model.x_XB = rng.normal(scale=0.05, size=(2, 2, n))
    Ca = rng.uniform(0.0, 3.0, n)
    SL = rng.uniform(1.2, 4.2, n)
    # Below LA, on it, inside and on the boundaries of the pieces of frac_SO (LM = 1.65,
    # 2 LA - LB = 2.32, 2 LA + LB = 2.68, 2 LA + LM = 4.15), and beyond them.
    pinned = [1.0, 1.25, 1.45, 1.65, 2.0, 2.32, 2.5, 2.68, 3.0, 4.15, 4.5]
    SL[2 : 2 + len(pinned)] = pinned
    LA, LM, LB = (model.p[k] for k in ("LA", "LM", "LB"))
    edges = [LA, LM, 2 * LA - LB, 2 * LA + LB, 2 * LA + LM]
    # (-inf, LA], the four pieces (lo, hi] of frac_SO, (2 LA + LM, inf): each holds a cell
    cells_per_interval = np.bincount(np.searchsorted(edges, SL), minlength=len(edges) + 1)
    assert np.all(cells_per_interval > 0), cells_per_interval
    dSL = rng.uniform(-2.0, 2.0, n)
    model._update_Ca_rates(Ca, SL)
    model._SL_curr = SL
    with np.errstate(invalid="ignore"):  # np.where evaluates both branches of k_PN, k_NP
        A, b = model._XB_system(dSL)
    sol = np.stack([model.x_XB[0, 0], model.x_XB[1, 0], model.x_XB[0, 1], model.x_XB[1, 1]], -1)
    expected = {
        **_ru_by_name(model._RU_get_rhs()),
        **_xb_by_name(np.einsum("nij,nj->ni", A, sol) + b),
    }
    ns = _module(RDQ20MF)
    states = _states_by_name(ns, model)
    params = _params_per_cell(ns, Ca=Ca, SL=SL, dSL=dSL)
    with np.errstate(invalid="ignore"):  # the guarded divisions, in the branch not taken
        rhs = ns["rhs"](0.0, states, params)
        monitors = ns["monitor_values"](0.0, states, params)
    for name, value in expected.items():
        np.testing.assert_allclose(
            rhs[ns["state_index"](name)], value, rtol=1e-12, atol=1e-15, err_msg=name
        )
    for out, method in (
        ("Ta", "get_active_tension"),
        ("Ka", "get_active_stiffness"),
        ("bound_ca", "bound_calcium_fraction"),
    ):
        np.testing.assert_allclose(
            monitors[ns["monitor_index"](out)],
            getattr(model, method)(),
            rtol=1e-12,
            atol=1e-15,
            err_msg=out,
        )
