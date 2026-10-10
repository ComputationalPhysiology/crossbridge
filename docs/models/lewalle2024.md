# Lewalle2024

A different modeling paradigm: an amendment of the Land et al. (2017) human-ventricular
contraction model ([Land2017](land2017.md) above) that replaces its ad hoc length-dependent-activation terms
with an explicit myosin "OFF" (super-relaxed) state whose OFF↔ON transition rate is modulated by
sarcomere force — a mechanosensitive feedback loop that reproduces the Frank-Starling length
dependence of active tension without any ad hoc SL-dependent terms.

**As an `.ode` file.**
[`lewalle2024.ode`](https://github.com/ComputationalPhysiology/crossbridge/blob/main/src/crossbridge/lewalle2024.ode),
reachable as `Lewalle2024.ODE_FILE`, is this model as a gotranx file, written to the contract in
{ref}`ode-files`, with `which_dep` and `dep_k1ork2` as the numeric parameters `which_dep` and
`dep_k1`. `tests/test_ode_files.py` pins it to the class for all eight combinations of the two,
over a calcium twitch with the sarcomere held or shortened and relengthened, and in two of them
with its parameters off their defaults: the class's outputs every 1 ms (taken as ten calls of
0.1 ms) agree with a tight `solve_ivp` solution of the file to 1e-4 of their peaks, except in
six cases with force feedback, where they differ by 1.4e-4 to 4.0e-4. In those the file computes
the feedback from the current state, while the class holds `Cd` fixed over a call and `S` and `W`
over a sub-step; the difference shrinks with the class's step.

**Reference:** {cite}`lewalle2024cardiac`
