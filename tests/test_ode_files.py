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

from crossbridge import Land2017
from crossbridge.utils import calcium_trace

gotranx = pytest.importorskip("gotranx")
solve_ivp = pytest.importorskip("scipy.integrate").solve_ivp

WITH_ODE: list[type] = [Land2017]
OUTPUTS: dict[type, tuple[str, ...]] = {Land2017: ("Ta", "Ka", "bound_ca", "Tp")}
#: Keys of `default_parameters()` that are not parameters of the ODE.
NON_ODE = {"dt", "Ca0", "dt_RU"}
#: Class parameters that the file spells differently: key -> (ODE parameter, value map).
SWITCHES: dict[str, tuple[str, dict[str, float]]] = {}
#: Where an output is compared against the class, and the floor of its peak [kPa, fraction].
GETTERS = {
    "Ta": "get_active_tension",
    "Ka": "get_active_stiffness",
    "bound_ca": "bound_calcium_fraction",
    "Tp": "get_passive_tension",
}
FLOOR = {"Ta": 1.0, "Tp": 1.0, "Ka": 1.0, "bound_ca": 1e-3}

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


def _state_of(model, name: str) -> float:
    """Cell 0 of the model's state `name` (the ODE's name -> the class's attribute)."""
    return float(getattr(model, name)[0])


def _inputs(sl, cmax, dt=1e-3):
    t = np.arange(0, 0.4, dt)
    Ca = calcium_trace(t, c0=0.1, cmax=cmax, t0=0.02)
    SL = sl(t + dt)
    dSL = (sl(t + dt) - sl(t)) / dt
    return Ca, SL, dSL


def _trajectory_errors(cls, params, ode_params, Ca, SL, dSL, dt=1e-3) -> dict[str, float]:
    """Each output's max |class - ODE| over the steps, divided by its peak."""
    m = _module(cls)
    y = m["init_state_values"]()
    p = m["init_parameter_values"](**ode_params)
    iCa, iSL, idSL = (m["parameter_index"](n) for n in ("Ca", "SL", "dSL"))
    outputs = OUTPUTS[cls]
    idx = {n: m["monitor_index"](n) for n in outputs}
    model = cls(1, params=params)
    got: dict[str, list[float]] = {n: [] for n in outputs}
    ref: dict[str, list[float]] = {n: [] for n in outputs}
    for k in range(len(Ca)):
        model.advance_step(dt, Ca[k], SL[k], dSL_vals=dSL[k])
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
    expected = {k: float(v) for k, v in defaults.items() if k not in NON_ODE | set(SWITCHES)}
    for key, (ode_name, mapping) in SWITCHES.items():
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
