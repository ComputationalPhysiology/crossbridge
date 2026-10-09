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
ModelClass.ODE_FILE             # Path of the model's gotranx .ode file, or None
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

## Calcium buffering

Every model also reports how much cytosolic calcium it is currently holding, and how fast that
is changing:

```python
theta = sarcomere.bound_calcium_fraction()    # occupied fraction, [0, 1]
rate  = sarcomere.get_calcium_binding_rate()  # occupied fraction per second
```

This matters as soon as a model is coupled to an electrophysiology model, and it is easy to get
wrong in a way that produces no error at all.

These models bind calcium — that is what troponin does. So does the cell model on the other side.
Run both unchanged and calcium is buffered **twice**. The usual fix is to remove troponin from the
EP model and let the contraction model own it, which is exactly what the "Ca<sub>i</sub> split"
does. But then the EP model's calcium balance is missing the buffering term, and it must be
supplied from here. In ToR-ORd that term appears as

$$
J_{TRPN} = \frac{d\,CaTRPN}{dt}\,[TRPN]_{max}, \qquad
\frac{d\,ca_i}{dt} = B_{ca_i}\bigl(\ldots - J_{TRPN}\bigr)
$$

so multiply `get_calcium_binding_rate()` by your model's total troponin concentration
(`trpnmax`, 0.07 mM in ToR-ORd) to recover $J_{TRPN}$. Omit it and the calcium transient is
unbuffered: too large, too fast, and silently wrong.

Two details worth knowing:

- **It is the mean rate over the step just taken**, $(\theta^{n+1}-\theta^n)/\Delta t$, not the
  instantaneous derivative. That makes the reported flux exactly the calcium the model actually
  absorbed, so a segregated coupling conserves calcium rather than leaking it at
  $\mathcal{O}(\Delta t)$. The two agree as $\Delta t \to 0$.
- **It is a fraction, not a concentration.** None of these models knows your total troponin
  concentration, so the scaling is yours to apply. The rate is zero before the first step and
  after `reset()`.

(ode-files)=
## The models as `.ode` files

`Land2017`, `Lewalle2024` and `RDQ20MF` are also written as
[gotranx](https://github.com/finsberg/gotranx) `.ode` files, which ship with the package. gotranx
generates Python, C, Julia or UFL code from an `.ode` file, so a model can run outside NumPy, with
its inputs supplied by other code, such as a cell model that computes the calcium. Each class
gives the path to its file as `ODE_FILE`. `RDQ18` has no file (see [RDQ18](rdq18.md)), and its
`ODE_FILE` is `None`, as is the base class's, so `MODEL_REGISTRY[name].ODE_FILE` can be read for
every model.

```python
import gotranx
from crossbridge import Land2017

ode = gotranx.load_ode(Land2017.ODE_FILE)
code = gotranx.cli.gotran2py.get_code(ode)  # Ca, SL and dSL are parameters

coupled = ode - ode.get_component("inputs")
code = gotranx.cli.gotran2py.get_code(coupled)  # Ca, SL and dSL are missing variables
```

In the second module, `rhs(t, states, parameters, missing_variables)` takes the inputs in the
order of its `missing` dict: `Ca`, `SL`, `dSL`. The files use `Min` and `Max`, which gotranx
supports from version 2.4.0.

Every file follows the same contract:

- **Units** are the classes': time in s, calcium in µM, length in µm. The outputs are in kPa.
  Land2017 and Lewalle2024 keep their stress parameters (`a`, `Tref`) in Pa, as the classes do.
- **Inputs.** A component named `inputs` holds three parameters: `Ca` [µM], `SL` [µm] and `dSL`
  [µm/s]. In the file alone they are parameters, held over a step as `advance_step` holds them.
  Subtracting the component leaves exactly these three as missing variables, which a cell model
  or a tissue model then supplies.
- **Outputs** are intermediates: `Ta` [kPa], `Ka` [kPa per unit $\Lambda = SL/SL_0$] and
  `bound_ca` [fraction], the values of `get_active_tension()`, `get_active_stiffness()` and
  `bound_calcium_fraction()`. Land2017 and Lewalle2024 also have `Tp` [kPa], the value of
  `get_passive_tension()`.
- **Parameters** have the names and defaults of `default_parameters()`, without the keys that set
  up the class's integrator or its initial state: `dt`, `Ca0` and `dt_RU`.
- **Initial states** are those `reset()` sets. A value the class computes, such as the steady
  state of CaTRPN at `Ca0`, is written as a number, with its formula in a comment.

Lewalle2024's two string parameters are numbers in its file:

| `.ode` parameter | Value | Class parameter |
|------------------|-------|-----------------|
| `which_dep`      | 0 (default) | `which_dep = "totalforce"` |
|                  | 1     | `which_dep = "force"` |
|                  | 2     | `which_dep = "passiveforce"` |
|                  | 3     | `which_dep = "Lambda"` |
| `dep_k1`         | 1 (default) | `dep_k1ork2 = "k1"` |
|                  | 0     | `dep_k1ork2 = "k2"` |

A file holds the model's equations; the class integrates them with its own scheme.
`tests/test_ode_files.py` checks the contract for every file, and checks each file against its
class, as each model's page describes. gotranx's own schemes are first order in the time step, so
code generated from a file and stepped at the 1 ms step the Land family takes is less accurate
than the class.

## References

```{bibliography}
```
