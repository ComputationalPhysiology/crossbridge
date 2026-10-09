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
  and list it in `_toc.yml`; add an `automodule` block to `docs/api.rst`; write
  `src/crossbridge/<name>.ode` to the contract, set `ODE_FILE`, and pin it with an equivalence test
  in `tests/test_ode_files.py` (adding the class to `WITH_ODE` and `OUTPUTS` there runs the contract
  tests on it). A model that cannot be written as an `.ode` keeps `ODE_FILE = None`, says why on its
  docs page, and joins RDQ18 in `tests/test_ode_file_attribute.py`. The registry-parametrized
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
  RU-tensor models, whose RU integrator is explicit). `advance_step` takes a step of any length:
  RDQ18 (`advance_ODE`) and RDQ20MF split it into sub-steps of at most `model.dt`, with calcium and
  length held over the step, and the Land family sub-steps at its own `_TARGET_SUBSTEP`. Until
  RDQ20MF did, a longer call took one Euler step of that length and advanced the crossbridges on
  every 40th call whatever its length, so a coupled solver stepping at 1 ms got no tension for
  40 ms; the long-step tests in `tests/test_rdq20mf.py` pin the fix. RDQ20MF refreshes its
  Ca-dependent rates every `freq_rates_update` (10) sub-steps and advances its crossbridges every
  `freqXB` (40). Tests that check a model's own dynamics still step at `model.dt`.
- Each model file (`rdq18.py`, `rdq20mf.py`, `land17.py`, `lewalle2024.py`) is self-contained and
  fully vectorized — NumPy arrays with a `num_cells` dimension, no per-cell Python loop.
  `advance_step` is the portable cross-model entry point; some models also keep an original,
  more detailed entry point (e.g. `RDQ18.advance_ODE`).
- `src/crossbridge/_linalg.py`: batched `solve_batch` + `expm_batch` for the small dense systems of
  `Land2017` (3x3) and `Lewalle2024` (5x5), which solve for a steady state and exponentiate one
  matrix per cell per sub-step — at organ scale (~32k cells/rank) those two lines were essentially
  all of `advance_step`. `expm_batch` exists rather than calling `scipy.linalg.expm` on the stack
  because each matrix must keep **its own** scaling-and-squaring exponent: cell norms span decades
  (`CaTRPN ** (-nTm / 2)`), so one shared exponent loses the cells far from the dominant norm, and
  scipy's stacked mode is a Python loop internally anyway. `solve_batch` is just LAPACK's algorithm
  (Gaussian elimination with partial pivoting) without the per-matrix dispatch, and raises
  `LinAlgError` on an exact singularity as numpy does. Two implementations of each — unrolled numba
  kernels (70x / 25x for the exponential, 12x / 7.5x for the solve; the unrolling is what buys it,
  an `@njit` kernel using `@` on 3x3 arrays allocates per cell and is no faster than scipy) and
  numpy fallbacks used when numba is absent, as it is in CI (numba is the optional `fast` extra,
  kept out of `test` so the suite still installs on Python versions numba lags behind). Together
  they take `advance_step` at 31920 cells from ~1.8 s to ~47 ms (Land2017) and ~2.7 s to ~165 ms
  (Lewalle2024). **The unrolled kernels are written by `tools/generate_linalg.py` — change the
  algorithm there and re-run it rather than editing them by hand**; the emitter is the reviewable
  form of code nobody can check by eye, and `test_module_matches_its_generator` fails if the two
  ever diverge. `tests/test_linalg.py` also pins the
  invariant that makes batching legitimate: a matrix alone and the same matrix inside a mixed-norm
  batch of thousands come back bit-identical.
- **`.ode` files.** `src/crossbridge/land2017.ode`, `lewalle2024.ode` and `rdq20mf.ode` (package
  data, `*.ode`) are the models as gotranx files, so gotranx can generate them for other targets
  (C, Julia, UFL) and a caller can supply their inputs from another model. Each class points at
  its file through `ODE_FILE` (`ClassVar[Path | None]` on `CardiacActivationModel`: `None` there and
  on `RDQ18`). RDQ18 has none on purpose: its 2176 states are 64 families over 34 neighbouring
  triplets, which gotranx cannot write compactly without array states. A file holds the model's
  equations, not the class's integrator. The contract every file follows (`docs/models/index.md`,
  "The models as `.ode` files"): crossbridge's units (s, µM, µm; outputs in kPa); a component
  `inputs` holding the parameters `Ca` [µM], `SL` [µm] and `dSL` [µm/s], so
  `ode - ode.get_component("inputs")` leaves exactly those three missing; the intermediates `Ta`,
  `Ka` (per unit Λ) and `bound_ca`, plus `Tp` for Land2017 and Lewalle2024; every other parameter
  named and defaulted as in `default_parameters()`, minus the non-ODE keys `dt`, `Ca0`, `dt_RU`;
  initial states those of `reset()`, a computed one written as a number with its formula in a
  comment; `Min`/`Max` where the class uses `np.minimum`/`np.maximum`. Lewalle2024's string
  switches are numbers: `which_dep` 0 `"totalforce"` (default), 1 `"force"`, 2 `"passiveforce"`,
  3 `"Lambda"`; `dep_k1` 1 `"k1"` (default), 0 `"k2"`.
- `tests/test_ode_files.py` pins each file to its class. Contract tests run over `WITH_ODE` (inputs
  removable, outputs are intermediates, parameters equal to `default_parameters()` both ways,
  initial states equal to `reset()`'s). Land2017 and Lewalle2024 are compared by trajectory
  (`_trajectory_errors`: the class at 1 ms steps against `solve_ivp`, Radau at rtol 1e-11, on the
  generated rhs with the same held inputs; each output's max error over its peak, < 1e-4).
  RDQ20MF is compared by its derivatives at 50 random states (rtol 1e-12), because its multi-rate
  stepping sits ~0.55% from its own ODE. The looser bounds (Land2017's zero-calcium `Ka`, 3e-3;
  six Lewalle2024 force-feedback cases in `LEWALLE_BOUNDS`, 3e-4 to 9e-4, with the class at ten
  calls per 1 ms) are the **class's** integrator error, bounded from measurement after showing that
  the file is not the cause: the reference agrees across Radau, LSODA and BDF, and the class's
  error shrinks with its own step. Treat a new mismatch as a transcription error until it is shown
  the same way.
  The Lewalle2024 cases take ~85 s, most of the suite's ~150 s.
- **`rdq20mf.ode` is written by `tools/generate_rdq20mf_ode.py`** — change the generator and
  re-run it (`python3 tools/generate_rdq20mf_ode.py`), never edit the file;
  `tests/test_rdq20mf_ode_generator.py` fails if the two differ. States: `x_{a}{b}{c}{B}` is
  `x_RU[a, b, c, B]`; `mu0_P, mu1_P, mu0_N, mu1_N` are `x_XB[0, 0], [1, 0], [0, 1], [1, 1]`.
  `RDQ20MF._XB_system(dSL_dt) -> (A, b)` is the crossbridge system that `_XB_advance`
  exponentiates, split out so the derivative test can read it.
- gotranx is a test dependency only (the `test` extra, `gotranx>=2.4.0`, the first release with
  `Min`/`Max`). Every test that needs it `importorskip`s it, so the suite passes without it, with
  those tests skipped; `tests/test_ode_file_attribute.py` and the generator test need no gotranx.
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
