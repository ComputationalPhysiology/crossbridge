# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`crossbridge` is a vectorized Python library implementing several reduced-order models of
cardiac myofilament activation and crossbridge dynamics (RDQ18, RDQ20-MF, Lewalle2024), all
sharing a common interface so a coupled electromechanics simulation can swap between them.
See README.md for the physiological/mathematical background and references for each model.

## Commands

Install (editable, with test deps):
```bash
pip install -e ".[test]"
```

Run the full test suite (coverage is configured via `addopts` in `pyproject.toml`, so plain
`pytest` already produces coverage reports):
```bash
pytest
```

Run a single test file / test:
```bash
pytest tests/test_rdq18.py
pytest tests/test_rdq18.py::test_probability_conservation
```

Lint / format / type-check (same tools run in CI's `pre-commit.yml`):
```bash
pre-commit run --all-files
# or individually:
ruff check .
ruff format .
mypy --config-file pyproject.toml
```

Demos (require `pip install ".[demos]"`) live in `demo/` and are run directly, e.g.
`python demo/compare_models.py`, `python demo/reproduce_figures.py`. `demo/fem.py` additionally
needs `dolfinx` and `pulse` (3D FEM coupling) and is not part of the standard install extras.

## Architecture

All models live in `src/crossbridge/` and subclass the abstract `CardiacActivationModel`
(`src/crossbridge/base.py`), which fixes one constructor signature and stepping interface:

```python
ModelClass(num_cells, Ta_max, params=None)
model.default_parameters()   # classmethod: dict of physiological defaults
model.advance_step(dt, Ca_val, SL_vals, dSL_vals=None)   # integrate one time step
model.get_active_tension()   # -> np.ndarray, shape (num_cells,), kPa
model.reset()                # restore initial state without re-allocating precomputed constants
```

A model is looked up by name via the small registry in `src/crossbridge/__init__.py`
(`MODEL_REGISTRY` / `get_model(name)`) — add new models to both the registry and
`__all__` when introducing one.

Each model file (`rdq18.py`, `rdq20mf.py`, `land17.py`, `lewalle2024.py`) is self-contained: parameters are
merged from `default_parameters()` with a caller-supplied `params` dict in `__init__`, and all
state is stored as NumPy arrays with a trailing (or matching) `num_cells` dimension so a single
model instance vectorizes across many cells/integration points at once — there is no
per-cell Python loop in the hot path. `advance_step` is the portable, model-agnostic entry
point; model-specific historical entry points also exist (e.g. `RDQ18.advance_ODE`) and are
kept for backward compatibility / direct use — prefer `advance_step` in code meant to be
model-agnostic.

Only `RDQ18` scales tension via the constructor's `Ta_max` (`Ta_max * compute_permissivity()`).
`RDQ20MF`, `Land2017`, and `Lewalle2024` compute tension intrinsically from their own parameters
(`a_XB`, `Tref`, `Tref` respectively) and accept `Ta_max` purely for interface compatibility —
check which case applies before relying on `Ta_max` when adding code that swaps models. `Land2017`
is the un-amended base model that `Lewalle2024` extends (same troponin/crossbridge state
machinery, but `Lewalle2024` replaces `Land2017`'s ad hoc `beta0`/`beta1` length-dependent
activation with explicit myosin OFF-state feedback). See the "Choosing a
Model" table in README.md for the full parameter/state comparison.

`src/crossbridge/utils.py` provides synthetic `calcium_trace()` / `sl_trace()` generators used
throughout the tests and demos as standard inputs, not physiological measurements.

### Reference materials (not part of the package)

`new_models/` contains the original paper PDFs and reference implementations (Python/C++/MATLAB)
that the models in `src/crossbridge/` were ported/adapted from. `main.py` at the repo root is a
scratch/example script. Neither is part of the installable `crossbridge` package
(`tool.setuptools.packages.find` only picks up `src/`) — treat `new_models/` as read-only
ground truth when checking a port's numerical behavior against the original.

### Demos beyond single-cell usage

Several demos in `demo/` couple a model into larger simulations and are useful references for
architecture beyond the core library:
- `holzapfel_torord_isometric.py` / `holzapfel_torord_isotonic.py`: couple `RDQ18` to a
  ToRORd electrophysiology model and a `zero_mech` (Holzapfel-Ogden) tissue mechanics model.
- `fem.py`: couples `RDQ18` to a 3D FEM mesh via `dolfinx`/`pulse`, evaluated at integration
  points.
- `compare_models.py`: drives all three models through an identical Ca/SL protocol via
  `get_model()` to verify the shared-interface contract.
