# Marzban2020

An extension of [Tewari2016](tewari2016.md) used to show that the energetic state of a failing
heart alone impairs its contraction {cite}`lopez2020impaired`, in the curated form of
{cite}`marzban2020computational`. It keeps Tewari2016's strain-dependent, metabolite-sensitive
crossbridge moments and replaces its calcium activation with two mechanisms of
{cite}`campbell2018force`:

- a cooperative thin filament: calcium switches binding sites on at a rate that rises with the
  fraction already on, and only sites without an attached crossbridge switch off;
- a myosin OFF (super-relaxed) state: only ON heads attach, and force recruits OFF heads, so that
  the force of longer sarcomeres feeds back into their activation -- a mechanism of
  length-dependent activation.

Thick-filament overlap now scales attachment as well as force, and the rates were refitted at
37 °C to the rat trabeculae of {cite}`janssen2002myofilament`.

**Parameters** are those of the curated MATLAB code, which differs from the paper's Table 7 in
`K_coop` and `k_off`; with the table's values the muscle does not relax. The stiffnesses are the
table's in-vitro values (the code's include the in-vivo factor 1.4), and Pi defaults to 1.0 mM,
at which the class reproduces the paper's fit: developed tension within 8% of the paper's model
at 2–10 Hz and 4% at SL 1.9–2.2 µm, and the steady-state pCa50 of its Fig. 3 caption to 0.01.
Twitches peak 7–9 ms later than the paper's. `demo/reproduce_figures_marzban2020.py` shows the
fit, the OFF state at work and the effect of the metabolite levels of sham and failing hearts.

**As an `.ode` file.**
[`marzban2020.ode`](https://github.com/ComputationalPhysiology/crossbridge/blob/main/src/crossbridge/marzban2020.ode),
reachable as `Marzban2020.ODE_FILE`, is this model as a gotranx file, written to the contract in
{ref}`ode-files`, with the extra intermediate `J_ATP`. `tests/test_ode_files.py` pins it to the
class over the same protocols as Tewari2016's, a fast shortening that floors the force-dependent
recruitment, and its parameters off their defaults, to 1e-4 of each output's peak, and at random
states to 1e-11.

**References:** {cite}`marzban2020computational`, {cite}`lopez2020impaired`, {cite}`campbell2018force`
