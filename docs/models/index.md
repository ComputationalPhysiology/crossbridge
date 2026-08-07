# The Mathematical Models

`crossbridge` implements several reduced-order models of cardiac myofilament activation, all
sharing a common interface (see ["Choosing a Model"](../../README.md#choosing-a-model) in the
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
