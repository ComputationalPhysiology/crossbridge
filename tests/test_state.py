import numpy as np
import pytest

import crossbridge

NAMES = sorted(crossbridge.MODEL_REGISTRY)
NUM_CELLS = 3
DT = 1e-3


def make(name):
    return crossbridge.get_model(name)(num_cells=NUM_CELLS)


def inputs(i):
    Ca = np.array([0.5, 1.0, 2.0]) * (1 + 0.1 * i)
    SL = np.array([2.0, 2.1, 2.2]) - 0.01 * i
    dSL = np.array([-0.1, 0.0, 0.1])
    return DT, Ca, SL, dSL


@pytest.mark.parametrize("name", NAMES)
def test_restored_model_advances_bit_identically(name):
    a = make(name)
    for i in range(3):
        a.advance_step(*inputs(i))
    b = make(name)
    b.set_state(a.get_state())
    for i in range(3, 6):
        a.advance_step(*inputs(i))
        b.advance_step(*inputs(i))
    assert np.array_equal(a.get_active_tension(), b.get_active_tension())
    assert np.array_equal(a.get_active_stiffness(), b.get_active_stiffness())
    assert np.array_equal(a.get_calcium_binding_rate(), b.get_calcium_binding_rate())
    state_b = b.get_state()
    for key, value in a.get_state().items():
        assert np.array_equal(value, state_b[key]), key


@pytest.mark.parametrize("name", NAMES)
def test_state_keys_cell_axis_and_scalar_types(name):
    model = make(name)
    model.advance_step(*inputs(0))
    state = model.get_state()
    assert list(state)[:2] == ["_prev_bound_ca", "_last_dt"]
    assert tuple(state)[2:] == tuple(type(model)._state_names)
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            assert value.shape[-1] == NUM_CELLS, key
        else:
            assert type(value) in (float, int, bool), key


@pytest.mark.parametrize("name", NAMES)
def test_get_state_returns_copies(name):
    model = make(name)
    model.advance_step(*inputs(0))
    state = model.get_state()
    snapshot = {k: np.copy(v) for k, v in state.items()}
    model.advance_step(*inputs(1))
    for key, value in snapshot.items():
        assert np.array_equal(state[key], value), key


@pytest.mark.parametrize("name", NAMES)
def test_set_state_refuses_missing_and_unknown_keys(name):
    model = make(name)
    state = model.get_state()
    missing = next(k for k in state if k not in ("_last_dt",))
    broken = {k: v for k, v in state.items() if k != missing}
    broken["bogus"] = 1.0
    with pytest.raises(KeyError, match=missing):
        model.set_state(broken)
    with pytest.raises(KeyError, match="bogus"):
        model.set_state(broken)


@pytest.mark.parametrize("name", NAMES)
def test_set_state_refuses_a_wrong_shape(name):
    model = make(name)
    state = model.get_state()
    key = next(k for k, v in state.items() if isinstance(v, np.ndarray))
    arr = state[key]
    state[key] = np.zeros(arr.shape[:-1] + (NUM_CELLS + 1,))
    with pytest.raises(ValueError, match=key):
        model.set_state(state)
