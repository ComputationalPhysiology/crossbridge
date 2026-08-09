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
  `.advance_step(dt, Ca_val, SL_vals, dSL_vals=None)`, `.get_active_tension()`, `.reset()`.
- **Adding a new model** — do all of: implement the class in `src/crossbridge/<name>.py`; register
  it in `MODEL_REGISTRY` and `__all__` in `src/crossbridge/__init__.py`; add `docs/models/<name>.md`
  and list it in `_toc.yml`; add an `automodule` block to `docs/api.rst`.
- Each model file (`rdq18.py`, `rdq20mf.py`, `land17.py`, `lewalle2024.py`) is self-contained and
  fully vectorized — NumPy arrays with a `num_cells` dimension, no per-cell Python loop.
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
