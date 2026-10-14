# Tewari2016

A rat myofilament model built around a strain-dependent, metabolite-sensitive crossbridge cycle
{cite}`tewari2016dynamics`: myosin attaches loosely (A1), binds strongly (A2) and makes its
powerstroke (A3), and each attached state is a distribution over crossbridge strain, reduced to
ODEs through its first three moments. Strain feeds back on the transition rates, so the model
responds to the velocity of the sarcomere; MgATP, MgADP and Pi enter the rates as substrate and
products; temperature enters through Q10 factors. {cite}`tewari2016influence` couples this cycle to
the calcium activation of {cite}`rice2008approximate` -- troponin C with low- and high-affinity
sites in the non-overlap and overlap regions of the thin filament, setting the rate at which
crossbridges become permissible -- with force slowing calcium unbinding, and identifies it from
rat trabeculae at body temperature {cite}`janssen2002myofilament`. Active tension is the
crossbridge force scaled by thick-filament overlap.

The paper embeds the model in a TriSeg heart and a closed-loop circulation to study energetic
failure; the class is the myofilament, with sarcomere length an input as in every model here.
`get_passive_tension()` gives the paper's passive force and `get_atp_hydrolysis_rate()` the
crossbridge ATP turnover.

**Reading the paper.** The model's code was not published. Table B.1's crossbridge rates are the
17 °C rat values of {cite}`tewari2016dynamics`, scaled by its Q10 values; read with the
`(T - 37.5)/10` the paper prints for the activation rates, the model needs about twice the calcium
the paper's own results show. With the activation rates also at 17 °C (`T_ref_act`, the
default), the class reproduces the paper's steady-state force-calcium relation (its Table E.1) at
all five sarcomere lengths to within 0.16 µM in Ca50, and twitch tension to within 10% at 4–10 Hz.
Twitches peak and relax about 10 ms sooner than the paper's, twitch tension is low at 2 Hz and
below SL 2.2 µm, and tension redevelopment after a slack-restretch is up to three times faster. The model's docstring lists these and the two guards
that depart from the paper; `demo/reproduce_figures_tewari2016.py` shows each against the paper.

**As an `.ode` file.**
[`tewari2016.ode`](https://github.com/ComputationalPhysiology/crossbridge/blob/main/src/crossbridge/tewari2016.ode),
reachable as `Tewari2016.ODE_FILE`, is this model as a gotranx file, written to the contract in
{ref}`ode-files`, with the extra intermediate `J_ATP`. `tests/test_ode_files.py` pins it to the
class over a calcium twitch at sarcomere lengths below the resting length, at 2.2 µm and in the
collagen range, slowly shortened and relengthened, stretched and released at peak tension, and
with its parameters off their defaults: the class's `Ta`, `Ka`, `bound_ca` and `Tp` at 1 ms steps
agree with a tight `solve_ivp` solution of the file to 1e-4 of their peaks. The right-hand sides
also agree to 1e-12 at random states that reach the branches no trajectory does.

**References:** {cite}`tewari2016influence`, {cite}`tewari2016dynamics`
