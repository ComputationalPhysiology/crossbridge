# The Mathematical Models

`crossbridge` implements several reduced-order models of cardiac myofilament activation, all
sharing a common interface (see ["Choosing a Model"](#choosing-a-model) in the
main README) so that a coupled electromechanics simulation can swap between them with minimal
code changes.

- **[RDQ18](rdq18.md)** — a spatially explicit Markov-chain-derived ODE model with
  nearest-neighbor cooperative interactions.
- **[RDQ20-MF](rdq20mf.md)** — a mean-field extension of RDQ18 adding an explicit
  crossbridge-cycling sub-system.
- **[Land2017](land2017.md)** — a human-ventricular contraction model fit directly to isometric
  tension measurements, with *ad hoc* length-dependent activation.
- **[Lewalle2024](lewalle2024.md)** — an amendment of Land2017 that replaces its ad hoc
  length-dependent activation with explicit myosin OFF-state force feedback.


(choosing-a-model)=
## Choosing a Model

All models subclass the same `CardiacActivationModel` abstract base class and share one
constructor and stepping interface:

```python
ModelClass(num_cells, Ta_max=..., params={...})
model.default_parameters()      # classmethod: dict of physiological defaults
model.advance_step(dt, Ca_val, SL_vals, dSL_vals=None)   # integrate one time step
model.get_active_tension()      # -> np.ndarray, shape (num_cells,), kPa
model.reset()                   # restore the model's initial state
```

This means a coupled simulation loop written against `advance_step`/`get_active_tension` works
unchanged if the model class is swapped out. `RDQ18.advance_ODE` (used in the example above) is
that model's original, more detailed entry point; `advance_step` is the portable one to use when
you want to be able to swap models.

| Model         | State representation                          | `Ta_max` semantics                                             | Typical `dt`   |
|---------------|------------------------------------------------|------------------------------------------------------------------|---------------|
| `RDQ18`       | RU triplet joint-probability tensor            | Used directly: `Ta = Ta_max * compute_permissivity()`            | 2.5e-5 s      |
| `RDQ20MF`     | RU triplet tensor + explicit crossbridge states| **Unused** — tension is computed from `params["a_XB"]` instead   | 2.5e-5 s      |
| `Land2017`    | Troponin/crossbridge state populations         | **Unused** — tension is computed from `params["Tref"]` instead   | up to ~1e-3 s (adaptive internal sub-stepping) |
| `Lewalle2024` | Land2017 state populations + OFF-state feedback| **Unused** — tension is computed from `params["Tref"]` instead   | up to ~1e-3 s (adaptive internal sub-stepping) |

Only `RDQ18` scales tension via the constructor's `Ta_max` argument; `RDQ20MF`, `Land2017`, and
`Lewalle2024` compute tension intrinsically from their own parameter set (`a_XB`, `Tref`) and
accept `Ta_max` purely for interface compatibility. Check which case applies before relying on
`Ta_max` when swapping models.

Since all models share the same constructor signature, a model can be selected by name at
runtime via the small registry in `crossbridge`:

```python
from crossbridge import get_model

ModelClass = get_model("RDQ20MF")  # or "RDQ18", "Land2017", "Lewalle2024"
sarcomere = ModelClass(num_cells=100, params={"SL0": 2.0})
```
