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

## Active stiffness and stable coupling

Every model also reports an **active stiffness** alongside its active tension:

```python
Ta = sarcomere.get_active_tension()    # kPa
Ka = sarcomere.get_active_stiffness()  # kPa per unit Lambda = SL / SL0
```

$K_a = \partial \dot{T_a} / \partial \dot{\lambda}$ measures how strongly the generated tension
responds to the *rate* of shortening. It exists because coupling any of these models to a tissue
mechanics solver is less innocent than it looks.

The usual approach is segregated (staggered): advance the activation model, then solve mechanics
with the resulting tension held fixed. {cite}`regazzoni2020oscillationfree` show that this scheme
develops non-physical oscillations, and is in fact *not convergent*, whenever the active stiffness
exceeds the passive stiffness of the tissue — routine in contracting myocardium. Crucially,
**reducing the time step makes it worse**, so the failure cannot be tuned away.

The fix is to stop treating active tension as a dead load during the mechanics solve and treat it
as what it physically is — a population of crossbridges acting as springs:

$$
\mathbf{P}_{act} = \left[T_a + K_a\left(\lambda^{k+1} - \lambda^{k}\right)\right]
                   \frac{\mathbf{F}\mathbf{f}_0 \otimes \mathbf{f}_0}{|\mathbf{F}\mathbf{f}_0|}
$$

The added term vanishes as $\Delta t \to 0$, so the scheme stays consistent, but it is
unconditionally stable. The same $K_a$ is also, to leading order in $\Delta t$, the derivative
$dT_a/d\lambda$ needed to couple a model *monolithically* through a Newton solve.

Two things to check when consuming `Ka`:

- **Stretch variable.** `Ka` is reported per unit of this package's dimensionless
  $\Lambda = SL/SL_0$. If your mechanics solver works in a stretch $\lambda$ with
  $SL = \lambda\,SL_{ref}$ and $SL_{ref} \neq SL_0$, rescale by the chain rule:
  $K_a^{solver} = K_a \cdot SL_{ref}/SL_0$.
- **`RDQ18` returns exactly zero**, because it has no strain-rate feedback at all. That means it
  needs no stabilization — but also that it exhibits no force-velocity (Hill) behaviour, which is
  a modelling limitation worth knowing before choosing it.

## References

```{bibliography}
```
