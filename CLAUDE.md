# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

- `crossbridge`: vectorized Python library of reduced-order cardiac myofilament activation models (RDQ18, RDQ20-MF, Land2017, Lewalle2024), all sharing one interface so they're swappable.
- Published on PyPI as `crossbridge`.
- Model background/references live in `docs/models/`, not README.md (kept short by design).

## Commands

- Install (editable, dev): `pip install -e ".[test]"`
- Run all tests: `pytest`
- Run one test: `pytest tests/test_rdq18.py::test_probability_conservation`
- Lint/format/typecheck (matches CI's `pre-commit.yml`): `pre-commit run --all-files` (or individually `ruff check .`, `ruff format .`, `mypy --config-file pyproject.toml`)
- Demos need `pip install ".[demos]"`; run directly, e.g. `python demo/compare_models.py`. `demo/fem.py` additionally needs `dolfinx`/`pulse`, not covered by any extra.

## Architecture

- All models subclass `CardiacActivationModel` (`src/crossbridge/base.py`):
  `ModelClass(num_cells, Ta_max, params=None)`, `.default_parameters()`,
  `.advance_step(dt, Ca_val, SL_vals, dSL_vals=None)`, `.get_active_tension()`,
  `.get_active_stiffness()`, `.bound_calcium_fraction()`, `.reset()`. All are `@abstractmethod` —
  a subclass missing one fails loudly at instantiation rather than silently returning something
  wrong. `get_calcium_binding_rate()` is concrete on the base and needs no per-model work, but
  each model's stepping entry point **must** call `self._begin_step(dt)` before mutating state,
  and its `reset()` must clear `_last_dt`.
- **Adding a new model** — do all of: implement the class in `src/crossbridge/<name>.py`; register
  it in `MODEL_REGISTRY` and `__all__` in `src/crossbridge/__init__.py`; add `docs/models/<name>.md`
  and list it in `_toc.yml`; add an `automodule` block to `docs/api.rst`. The registry-parametrized
  tests in `tests/test_active_stiffness.py` pick the model up automatically.
- `get_active_stiffness()` returns `Ka = d(dTa/dt)/d(dLambda/dt)` in the same units as
  `get_active_tension()` (kPa), per unit dimensionless `Lambda = SL/SL0`. It is what lets a caller
  couple a model to a mechanics solver without the oscillatory instability of a naive staggered
  scheme (Regazzoni & Quarteroni 2020) — see `docs/models/index.md`. Formulas are per-family:
  distortion-decay models (`Land2017`, `Lewalle2024`) use `h*Tref/rs*(As*S + Aw*W)`; `RDQ20MF` uses
  `a_XB*frac_SO*(mu0_P + mu0_N)`; `RDQ18` is identically zero (no strain-rate feedback).
  **Never hand-check a new `Ka` by eye** — `tests/test_active_stiffness.py` verifies it by finite
  difference against the model's own dynamics, which is the only thing that keeps it honest when
  the ODEs change later.
- `bound_calcium_fraction()` / `get_calcium_binding_rate()` exist so a coupled EP model can get its
  troponin buffering flux back (`J_TRPN = rate * trpnmax`); without it calcium is either buffered
  twice or not at all, with nothing raised. The rate is the **mean over the last step**, not an
  instantaneous derivative, so that a segregated coupling conserves calcium exactly —
  `tests/test_calcium_binding.py::test_reported_flux_conserves_calcium` asserts that identity to
  1e-10 and is the test to keep working if you touch this.
- Model `dt` differs by ~40x across the registry (1e-3 s for the Land family, 2.5e-5 s for the
  RU-tensor models, whose RU integrator is explicit). Tests that drive models generically must step
  at `model.dt`; driving RDQ18/RDQ20MF at 1e-3 silently denormalizes their probability tensor.
  RDQ20MF additionally only refreshes Ca-dependent rates every `freq_rates_update` (10) steps.
- Each model file (`rdq18.py`, `rdq20mf.py`, `land17.py`, `lewalle2024.py`) is self-contained and
  fully vectorized — NumPy arrays with a `num_cells` dimension, no per-cell Python loop.
- `src/crossbridge/_expm.py`: batched matrix exponential used by `Land2017` (3x3) and
  `Lewalle2024` (5x5), which exponentiate one matrix per cell per sub-step — at organ scale
  (~32k cells/rank) that was ~90% of `advance_step`. It exists rather than calling
  `scipy.linalg.expm` on the stack because each matrix must keep **its own** scaling-and-squaring
  exponent: cell norms span decades (`CaTRPN ** (-nTm / 2)`), so one shared exponent loses the
  cells far from the dominant norm, and scipy's stacked mode is a Python loop internally anyway.
  Two implementations of one algorithm — a numba kernel with the 3x3/5x5 arithmetic hand-unrolled
  into scalars (70x / 25x over the scipy loop, taking `advance_step` at 31920 cells from 1.8 s to
  67 ms and from 2.7 s to 205 ms; the unrolling is what buys it, an `@njit` kernel using `@` on
  3x3 arrays allocates per cell and is no faster than scipy), and a pure-numpy fallback (6.5x / 3x) used when numba is absent, as it is in CI (numba is the optional `fast`
  extra, kept out of `test` so the suite still installs on Python versions numba lags behind).
  **The unrolled kernels are written by `tools/generate_expm.py` — change the algorithm there and
  re-run it rather than editing them by hand.**
  `tests/test_expm.py` pins the invariant that makes batching legitimate: a matrix alone and the
  same matrix inside a 5000-matrix mixed-norm batch come back bit-identical.
  `advance_step` is the portable cross-model entry point; some models also keep an original,
  more detailed entry point (e.g. `RDQ18.advance_ODE`).
- `Ta_max` semantics differ by model: `RDQ18` uses it directly
  (`Ta_max * compute_permissivity()`); `RDQ20MF`, `Land2017`, `Lewalle2024` compute tension
  intrinsically from their own params (`a_XB`/`Tref`) and only accept `Ta_max` for interface
  compatibility — check which applies before relying on it.
- `Land2017` is the un-amended base model; `Lewalle2024` extends it by replacing its ad hoc
  `beta0`/`beta1` length-dependent activation with explicit myosin OFF-state feedback.
- `src/crossbridge/utils.py`: synthetic `calcium_trace()`/`sl_trace()` generators for tests/demos,
  not physiological data.
- `new_models/` (paper PDFs + reference ports) and root `main.py` are reference/scratch material,
  not part of the installable package (`tool.setuptools.packages.find` only picks up `src/`) —
  treat `new_models/` as read-only ground truth when checking a port's numerics.

## Documentation

- Per-model background+references: `docs/models/*.md` (+ `index.md` overview). Per-demo
  descriptions: `demo/index.md`.
- Citations: add entries to `docs/refs.bib`, reference via `` {cite}`key` `` (MyST role; works
  both in `.md` files and inside jupytext-formatted demo `.py` files). Bibliography renders via a
  bare `` ```{bibliography}``` `` directive in `docs/models/index.md`.
- Docs are a jupyter-book site (`_config.yml`, `_toc.yml`) — new doc pages must be added to
  `_toc.yml` or they won't appear in the built site.

## Demos

- `holzapfel_torord_isometric.py` / `_isotonic.py`: couple `RDQ18` to a ToRORd EP model and
  `zero_mech` (Holzapfel-Ogden tissue mechanics). Running them generates
  `demo/ToRORd_dynCl_endo.{cellml,ode,py}` (downloaded/code-generated via `gotranx`, not
  hand-written) plus PNG outputs.
- `fem.py`: couples `RDQ18` to a 3D FEM mesh via `dolfinx`/`pulse`.
- `compare_models.py`: drives all models through an identical Ca/SL protocol via `get_model()`
  to verify the shared interface.
