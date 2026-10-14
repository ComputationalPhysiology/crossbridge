# Musgrave2025

The first crossbridge models fitted to human data {cite}`musgrave2025metabolite`: right atrial
trabeculae from patients undergoing bypass surgery, with and without type 2 diabetes, measured
permeabilized at maximal activation under five levels of ATP and Pi
{cite}`musgrave2025human`. There is one parameter set per group: `default_parameters()` is the
non-diabetic model and `diabetic_parameters()` the diabetic one.

- **Crossbridges.** A three-state stiffness-distortion model {cite}`razumova1999stiffness`: a
  detached but permissible state and two attached states, before and after the powerstroke.
  Each attached state carries the mean strain of its bridges, and the rates of attachment and
  detachment depend on it. Pi drives the reverse of attachment, and ATP binds in rapid
  equilibrium. The reverse of detachment follows from the free energy of ATP hydrolysis
  {cite}`tran2010metabolite`. Sarcomere length sets the fraction of heads within reach of a
  binding site.
- **Thin filament.** The troponin and tropomyosin of [Land2017](land2017.md), with each group's
  calcium sensitivity from {cite}`jones2023impaired`.
- **Passive stress.** A spring and dashpot in series, in parallel with an exponential spring.
- **Energetics.** Each crossbridge cycle hydrolyses one ATP. `get_atp_hydrolysis_rate()` gives
  the flux per head. Times the crossbridge density and myocyte fraction, it is the ATP flux per
  volume of muscle, and times the free energy of ATP hydrolysis, the crossbridges' energy use.

The model is atrial. Human atria hold more α-myosin than the ventricles, so its crossbridges
cycle faster than a human ventricle's would.

**What reproduces.**
- Driven by the paper's 1 Hz calcium transients (`crossbridge.utils.musgrave_calcium`), the
  class reproduces the isometric twitches of the paper's Fig. 5: systolic stress, amplitude and
  duration to the printed precision, except the diabetic duration, 672 ms against the paper's
  674 ms.
- It reproduces the steady-state stresses of Table 2, and the effects of ATP, Pi and the
  cross-over of calcium handling and crossbridges on the twitch (Figs. 8A, 9A and 10A).
  `tests/test_musgrave2025.py` pins each of these.
- `demo/reproduce_figures_musgrave2025.py` also reproduces:
  - the fit to the active complex modulus (Fig. 4), with the paper's root-mean-square misfits
    of 12.5 and 10.6 kPa;
  - the passive model (Fig. 2);
  - the work loops and their mechanoenergetics (Figs. 6 and 7): the diabetic model does 44%
    less work and shortens 52% more slowly, its crossbridges use 75% less energy, and it is
    more than twice as efficient.

**Two discrepancies with the paper:**
- Its maximum power of shortening falls by 73% here and by 69% in the paper. The paper's
  work loops hold the afterload with a mass-dashpot controller; the demo solves for it
  exactly.
- At 10 mM Pi the twitch amplitude falls by 28% and 26% (non-diabetic, diabetic). The paper's
  text gives the same two numbers in the other order.

**Parameters** are those of the authors' code, which Tables 1 and 3 round. The stiffness is the
muscle model's, which is higher than Table 3's: some heads stay non-permissible even at maximal
calcium. Setting `km0 = 0` and Table 3's stiffness recovers the crossbridge model of Table 2.
`rho_myo = 1` gives stress per myofilament area, the paper's myocyte stress.

**Departures from the paper.** None of these changes the model where the paper used it:
- **First moments as states.** The mean strains are carried as first moments, `B_xB = B xB`
  and `C_xC = C xC`. The paper's equations divide by the attached fractions, so they cannot
  start from rest; these can.
- **Thermodynamic reverse rate.** It is written with Pi and ATP cancelled, which keeps it
  finite at 0 mM Pi.
- **Troponin floor.** Troponin occupancy is floored at 1e-3 in the tropomyosin rates. The floor
  acts only below about 10 nM calcium.

**Numerical scheme.**
- Troponin and the dashpot are advanced exactly.
- The other states are advanced with an exponential midpoint rule at 0.1 ms sub-steps.
- The detached permissible fraction is clamped at zero when more heads are non-permissible or
  attached than are in reach. A sub-step in which it comes off that clamp is split where it
  does.

**As an `.ode` file.**
[`musgrave2025.ode`](https://github.com/ComputationalPhysiology/crossbridge/blob/main/src/crossbridge/musgrave2025.ode),
reachable as `Musgrave2025.ODE_FILE`, is this model with the non-diabetic parameters as a gotranx
file. It follows the contract in {ref}`ode-files`, with the extra intermediate `J_ATP`. Pass
`diabetic_parameters()` (without `dt` and `Ca0`) for the diabetic model.
`tests/test_ode_files.py` pins the file to the class:
- over the length protocols of the other strain-moment models, with both parameter sets, a
  release from the clamp, and the parameters off their defaults, to 1e-4 of each output's peak
  (2e-5 in practice);
- at random states, to 1e-12.

The mean strains divide by `Max(B, 1e-12)`, not `B`. gotranx 2.5 takes `B_xB / B` for a
removable singularity, since `B_xB` starts at zero, and by default replaces it with zero for
`|B| < 0.1`. A contract test checks that this singularity removal changes nothing in any
model's file.

**References:** {cite}`musgrave2025metabolite`, {cite}`musgrave2025human`,
{cite}`razumova1999stiffness`, {cite}`tran2010metabolite`, {cite}`jones2023impaired`,
{cite}`land2017model`
