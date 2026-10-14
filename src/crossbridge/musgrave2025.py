"""
Musgrave et al. (2025) human atrial myofilament model.

Based on: Musgrave, J.H., Han, J.-C., Ward, M.-L., Taberner, A.J., & Tran, K.
(2025). Metabolite-sensitive cross-bridge models of human atria reveal the
impact of diabetes on muscle mechanoenergetics. The Journal of Physiology,
604(13), 5316-5341. https://doi.org/10.1113/JP288463

The first crossbridge models fitted to human data: permeabilized right atrial
trabeculae from patients undergoing bypass surgery, with and without type 2
diabetes (Musgrave et al. 2025, Cardiovasc Diabetol 24:266), whose complex
modulus and active stress were measured at maximal activation under five
metabolite conditions (ATP 5, 1 and 0.1 mM; Pi 0, 1 and 10 mM). One parameter
set per group: `default_parameters()` is the non-diabetic model and
`diabetic_parameters()` the diabetic one.

* **Crossbridges.** A three-state stiffness-distortion model (Razumova et al.
  1999): detached but permissible ``A``, attached pre-powerstroke ``B`` and
  post-powerstroke ``C``, each attached state carrying the mean strain of its
  bridges (``xB``, ``xC``; new ``C`` bridges arrive at the powerstroke strain
  ``xC0``). Attachment ``k1`` slows with strain and detachment ``k3``
  speeds up with it. Pi drives the reverse attachment ``k-1``, ATP binds in
  rapid equilibrium (``k-2`` and ``k3``), and ``k-3`` follows from the free
  energy of ATP hydrolysis (Tran et al. 2010). Sarcomere length sets the
  fraction ``Z`` of heads within reach of a binding site.
* **Thin filament.** Land et al. (2017)'s troponin ``CaTn`` and tropomyosin
  ``N`` (heads with no permissible site), with each group's Ca50 from Jones
  et al. (2023).
* **Passive stress.** A spring and dashpot in series (``Fp1``), in parallel
  with an exponential spring above the resting length 1.87 um.
* **Energetics.** Each cycle hydrolyses one ATP, so the ATP flux per unit
  volume is ``rho_xb rho_myo (k3 C - k-3 A)`` with ``rho_xb = 0.25 mol/m^3``
  (`get_atp_hydrolysis_rate` gives the flux per head), and the energy per
  mole is ``-dG_ATP = -(dG_ATP0 + R T ln([ADP][Pi]/[ATP]))``.

The model is atrial, and human atria hold more alpha-myosin than the
ventricles, so its crossbridges cycle faster than a human ventricle's would.

**Parameters** are those of the authors' code (github.com/JuliaMusgrave/
AtrialModel_2025_Human, ``x_i`` in ``ND_xb_fit.mat`` / ``D_xb_fit.mat``),
which Tables 1 and 3 round. The stiffness is the muscle model's,
``K = rho_myo K_myo`` with Table A1's myocyte fraction and myofilament
stiffness: Table 3's K (9030 and 5270 GPa/m) is the maximally activated
crossbridge model's, and the muscle model raises it by 11% and 8% so that at
32 uM calcium, where 5-10% of heads stay in ``N``, it develops the same
stress. Set ``rho_myo = 1`` for stress per myofilament area (the paper's
myocyte stress, Fig. 5B). Metabolites default to the paper's baseline, 5 mM
ATP, 1 mM Pi and 36 uM ADP, at 37 degC.

**What reproduces**, pinned by ``tests/test_musgrave2025.py``, with the 1 Hz
transients of `crossbridge.utils.musgrave_calcium` at SL 2.2 um:

* Fig. 5A, the isometric twitches: systolic stress 20.2 and 12.0 kPa and
  amplitude 14.5 and 7.9 kPa (non-diabetic and diabetic), as in the paper;
  duration 550 and 672 ms (674 in the paper). With ``rho_myo = 1`` (Fig. 5B)
  the amplitudes are 40.0 and 27.0 kPa (26.9).
* Table 2, steady-state stress under the five metabolite conditions, to the
  printed 0.1 kPa, with ``N`` switched off (``km0 = 0``) and Table 3's K: the
  crossbridge model the table comes from. The muscle model, which keeps some
  heads in ``N`` even at 32 uM calcium, matches it at baseline only, by
  construction, and is up to 6% above it at 0.1 mM ATP.
* Fig. 9A, 1 mM ATP: the amplitude rises 9% and 29% and the duration 2% and
  25%, and diastolic stress rises (by 1.1 kPa) only in the diabetic model.
* Fig. 10A, the cross-over: diabetic calcium handling or diabetic
  crossbridges alone lower the amplitude by 25-30% and lengthen the twitch by
  7% and 15%.
* Fig. 8A, 10 mM Pi, with one discrepancy: the amplitude falls by 28% and
  26%, where the paper's text gives "26% vs. 28%", the same two numbers the
  other way round.

**Departures from the paper**, none of which changes it where it was used:

* The strains are carried as first moments, ``B_xB = B xB`` and
  ``C_xC = C xC`` (as in `Tewari2016`), whose equations are Eqs. (7)-(8)
  times ``B`` and ``C`` plus ``xB dB/dt`` and ``xC dC/dt``. The paper's form
  divides by ``B`` and ``C``, so it cannot start from rest (the authors' code
  starts from 1e-4); this one can, and `reset` does. Where ``B`` or ``C`` is
  below 1e-12 its mean strain is taken as that of new bridges, zero and
  ``xC0``.
* ``k-3`` is written with Pi and ATP cancelled: Eqs. (13)-(14) give
  ``k-3 = k1 k2 k3' e^{phi_s3 (xC - xC0)} [ADP] / (1000 k-1' k-2' kd_ATP
  e^{-dG_ATP0 / RT})``, the same for any Pi > 0 and finite at Pi = 0, where
  the paper's form is 0/0 (the authors' code avoids it with Pi = 1e-6 mM).
* ``CaTn`` is floored at 1e-3 in the tropomyosin rates, whose
  ``CaTn ** (-nTm / 2)`` diverges as calcium goes to zero. The floor acts
  only below about 10 nM calcium, where no head is permissible either way.

**Numerical scheme.** ``CaTn`` and ``Fp1`` are linear with constant inputs
and advanced exactly. ``N``, ``B``, ``C`` and the two moments form a linear
system once the strain-dependent ``k1`` and ``k3`` and the tropomyosin rates
are frozen, and where ``A`` is clamped at zero, or not, is fixed. Each
sub-step exponentiates it with those frozen at a midpoint state from a
half-step predictor (an exponential midpoint rule, second order). ``A`` can
only come off its clamp within a sub-step (by ``N`` decaying and bridges
detaching; it goes onto it only when ``Z`` falls, between steps), and a
sub-step in which it does is split where it does. At the default 0.1 ms
sub-step the class stays within 2e-5 of the peak of each output of its
``.ode`` file; at 0.2 ms, 8e-5, and 2e-4 without the split.

Examples
--------
>>> import numpy as np
>>> from crossbridge import Musgrave2025
>>>
>>> model = Musgrave2025(num_cells=10)
>>> model.advance_step(1e-3, Ca_val=0.5, SL_vals=np.full(10, 2.2))
>>> Ta = model.get_active_tension()  # shape (10,), kPa
>>> diabetic = Musgrave2025(10, params=Musgrave2025.diabetic_parameters())
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt

from ._linalg import expm_batch
from .base import CardiacActivationModel

#: Longest sub-step [s] over which the rates are frozen at one state.
_TARGET_SUBSTEP = 1e-4  # [s]

#: Gas constant [kJ/(mol K)].
_R = 8.314e-3

#: Floor on CaTn in the tropomyosin rates; see the module docstring.
_CATN_FLOOR = 1e-3

#: Below this occupancy a state's mean strain is taken as that of new bridges.
_TINY = 1e-12

# Rows/columns of the linear system; the last one carries the constant term.
_N, _B, _C, _MB, _MC = range(5)
_ONE = 5
_SIZE = 6


class Musgrave2025(CardiacActivationModel):
    """
    Vectorized Musgrave et al. (2025) human atrial myofilament model.

    Attributes
    ----------
    CaTn : ndarray, shape (num_cells,)
        Fraction of troponin C with calcium bound to its regulatory site.
    N : ndarray, shape (num_cells,)
        Fraction of heads without a permissible binding site.
    B, C : ndarray, shape (num_cells,)
        Fractions of heads attached before and after the powerstroke.
    B_xB, C_xC : ndarray, shape (num_cells,)
        First moments of their strain [um]: the fraction times the mean
        strain of its bridges, `xB` and `xC`.
    Fp1 : ndarray, shape (num_cells,)
        Stress in the passive spring-dashpot element [kPa].

    The detached permissible fraction ``A = max(0, Z(SL) - N - B - C)`` and
    the mean strains `xB` and `xC` are derived.
    """

    ODE_FILE = Path(__file__).with_name("musgrave2025.ode")

    CaTn: npt.NDArray[np.float64]
    N: npt.NDArray[np.float64]
    B: npt.NDArray[np.float64]
    C: npt.NDArray[np.float64]
    B_xB: npt.NDArray[np.float64]
    C_xC: npt.NDArray[np.float64]
    Fp1: npt.NDArray[np.float64]

    #: Attributes that evolve; see `CardiacActivationModel.get_state`.
    _state_names = (
        "CaTn",
        "N",
        "B",
        "C",
        "B_xB",
        "C_xC",
        "Fp1",
        "_SL_prev",
        "_SL_curr",
        "_has_prev_step",
    )

    def __init__(self, num_cells: int, Ta_max: float = 1.0, params: dict | None = None):
        """
        Initialize the Musgrave2025 model.

        Parameters
        ----------
        num_cells : int
            Number of independent cells / FEM integration points.
        Ta_max : float, optional
            Unused. Tension follows from the crossbridge stiffness
            (`K_myo`, `rho_myo`). Kept for API compatibility with
            `CardiacActivationModel`. Default is 1.0.
        params : dict, optional
            Parameter overrides. Keys should match those returned by
            `default_parameters()`; start from `diabetic_parameters()` for
            the diabetic model.
        """
        super().__init__(int(num_cells), Ta_max, params)

        self.p = type(self).default_parameters()
        if params:
            self.p.update(params)
        p = self.p
        self.dt = p["dt"]

        # Rates at the metabolite levels (Eqs. (10)-(12)), which are
        # parameters and so fixed for the model's lifetime.
        self._km1 = p["km1"] * p["Pi"]
        self._km2 = p["km2"] * p["kd_ATP"] / (p["kd_ATP"] + p["MgATP"])
        self._k3 = p["k3"] * p["MgATP"] / (p["kd_ATP"] + p["MgATP"])
        # Eqs. (13)-(14) with Pi and ATP cancelled: k-3 = k1 k2 k3 _c3 with k3
        # here the strain-scaled reference rate k3' e^{phi_s3 (xC - xC0)}
        self._c3 = p["MgADP"] / (
            1000.0 * p["km1"] * p["km2"] * p["kd_ATP"] * np.exp(-p["dG_ATP0"] / (_R * p["T"]))
        )

        self.reset()

    @classmethod
    def default_parameters(cls) -> dict:
        """
        Return the non-diabetic model's parameters (see the module docstring).

        Rates are in 1/s (``km1`` in 1/(s mM)), lengths and strains in um,
        stiffnesses and stresses in kPa (GPa/m = kPa/um), calcium in uM and
        the metabolites in mM.
        """
        p: dict = {}
        # Numerical: a *suggested* coupling dt; advance_step sub-steps
        # internally whatever the step it is given.
        p["dt"] = 1e-3  # [s]

        # SL0 sets the unit of Lambda = SL / SL0 in which Ka is reported, and
        # the length reset() starts from: the paper's optimal length Lo.
        p["SL0"] = 2.2  # [um]
        # Calcium reset() starts from, near the diastolic level of the
        # group's transient (`crossbridge.utils.musgrave_calcium`)
        p["Ca0"] = 0.23  # [uM]

        # Metabolites (the paper's baseline) and the thermodynamics of k-3
        p["MgATP"] = 5.0  # [mM]
        p["MgADP"] = 0.036  # [mM]
        p["Pi"] = 1.0  # [mM]
        p["dG_ATP0"] = -30.0  # [kJ/mol]
        p["T"] = 310.0  # [K]

        # Thin filament (Land et al. 2017; Ca50 from Jones et al. 2023)
        p["Ca50"] = 0.33  # [uM]
        p["k_tn"] = 100.0  # [1/s]
        p["n_tn"] = 2.0  # [-]
        p["k0"] = 1000.0  # [1/s] N -> A
        p["km0"] = 327.0  # [1/s] A -> N
        p["nTm"] = 5.0  # [-]

        # Crossbridge cycle: reference rates k1', k-1', k2, k-2', k3'
        p["k1"] = 88.97291545739041  # [1/s]
        p["km1"] = 6.680703481685643  # [1/(s mM)]
        p["k2"] = 88.83117612580399  # [1/s]
        p["km2"] = 112.58690705107963  # [1/s]
        p["k3"] = 96.5554076707849  # [1/s]
        p["phi_s1"] = 257.3553162449159  # [1/um]
        p["phi_s3"] = 44.872912839341836  # [1/um]
        p["kd_ATP"] = 0.28138787414484157  # [mM]
        p["phi_v"] = 0.029359047319968586  # [-]
        p["phi_l"] = 4.99850622220351  # [-]
        p["L_max"] = 2.3  # [um]
        p["xC0"] = 0.01  # [um] powerstroke
        # Stiffness K = rho_myo * K_myo, the code's 10052 kPa/um
        p["rho_myo"] = 0.361  # [-] myocyte fraction of the cross-section
        p["K_myo"] = 10052.452646625288 / 0.361  # [kPa/um]

        # Passive: spring Kp1 and dashpot mu in series, parallel spring Kp2
        p["Kp1"] = 13.299805644328579  # [kPa/um]
        p["mu"] = 0.06963919072646306  # [kPa s/um]
        p["Kp2"] = 0.5151627811248674  # [kPa]
        p["phi_p"] = 7.201953888948654  # [1/um]
        p["L_r"] = 1.87  # [um] resting length, 85% of 2.2 um
        return p

    @classmethod
    def diabetic_parameters(cls) -> dict:
        """
        Return the diabetic model's parameters: `default_parameters()` with
        the diabetic crossbridge, passive and Ca50 values.
        """
        p = cls.default_parameters()
        p.update(
            Ca0=0.26,
            Ca50=0.408,
            k1=44.25627532823596,
            km1=2.440127028999094,
            k2=86.2674332650524,
            km2=93.57045113490821,
            k3=50.00705708138363,
            phi_s1=7.129093270666559,
            phi_s3=198.80004414123349,
            kd_ATP=4.999998234601869,
            phi_v=0.03614104813752302,
            phi_l=5.0,
            rho_myo=0.292,
            K_myo=5696.585885603492 / 0.292,
            Kp1=11.539300103502612,
            mu=0.04876122054129823,
            Kp2=0.42007336011499036,
            phi_p=6.876281106387494,
        )
        return p

    def reset(self) -> None:
        """
        Reset to quiescent crossbridges: none attached, and troponin and
        tropomyosin in equilibrium at calcium `Ca0` and length `SL0`.
        """
        n = self.num_cells
        p = self.p
        r = (p["Ca0"] / p["Ca50"]) ** p["n_tn"]
        CaTn = r / (1.0 + r)
        # A <-> N in equilibrium, a A = b N, over the heads in reach, A + N = Z(SL0)
        a, b = self._tropomyosin_rates(np.array(CaTn))
        N = a / (a + b) * self._z(np.array(p["SL0"]))

        self.CaTn = np.full(n, CaTn)
        self.N = np.full(n, N)
        self.B = np.zeros(n)
        self.C = np.zeros(n)
        self.B_xB = np.zeros(n)
        self.C_xC = np.zeros(n)
        self.Fp1 = np.zeros(n)

        self._SL_prev = np.full(n, p["SL0"])
        self._SL_curr = np.full(n, p["SL0"])
        self._has_prev_step = False

        self._prev_bound_ca = np.zeros(n)
        self._last_dt = 0.0

    @property
    def xB(self) -> npt.NDArray[np.float64]:
        """Mean strain of the bridges in B [um]."""
        return self._strains(self._get_x())[0]

    @property
    def xC(self) -> npt.NDArray[np.float64]:
        """Mean strain of the bridges in C [um]."""
        return self._strains(self._get_x())[1]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _z(self, SL: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Eq. (6): fraction of heads within reach of a binding site."""
        p = self.p
        return np.where(SL < p["L_max"], 1.0 + p["phi_l"] * (SL / p["L_max"] - 1.0), 1.0)

    def _tropomyosin_rates(
        self, CaTn: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Eq. (22): the A -> N and N -> A rates at troponin occupancy `CaTn`."""
        p = self.p
        CaTn_safe = np.maximum(CaTn, _CATN_FLOOR)
        return p["km0"] * CaTn_safe ** (-p["nTm"] / 2.0), p["k0"] * CaTn_safe ** (p["nTm"] / 2.0)

    def _strains(
        self, x: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Mean strains xB, xC of state `x`, shape (5, n), from its moments. A
        state too sparse to define one takes that of its new bridges: zero for
        B, the powerstroke xC0 for C.
        """
        B, C = x[_B], x[_C]
        xB = np.where(B > _TINY, x[_MB] / np.maximum(B, _TINY), 0.0)
        xC = np.where(C > _TINY, x[_MC] / np.maximum(C, _TINY), self.p["xC0"])
        return xB, xC

    def _rates(
        self, x: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Eqs. (9), (12) and (13): k1, k3 and k-3 at the strains of state `x`."""
        p = self.p
        xB, xC = self._strains(x)
        k1 = p["k1"] * np.exp(-p["phi_s1"] * xB)
        e3 = np.exp(p["phi_s3"] * (xC - p["xC0"]))
        return k1, self._k3 * e3, k1 * p["k2"] * p["k3"] * e3 * self._c3

    @staticmethod
    def _free(x: npt.NDArray[np.float64], Z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Z - N - B - C of state `x`: A where positive; where not, A is clamped at zero."""
        return Z - x[_N] - x[_B] - x[_C]

    def _system(
        self,
        x: npt.NDArray[np.float64],
        CaTn: npt.NDArray[np.float64],
        Z: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
        reach: npt.NDArray[np.bool_] | None = None,
    ) -> npt.NDArray[np.float64]:
        """
        The (n, 6, 6) matrix of d[N, B, C, B xB, C xC, 1]/dt with the rates
        frozen at state `x` and troponin occupancy `CaTn`.

        In the first moments the strain equations (7)-(8) read
        d(B xB)/dt = -(k-1 + k2) B xB + phi_v v B and
        d(C xC)/dt = -(k-2 + k3) C xC + xC0 (k2 B + k-3 A) + phi_v v C.
        ``A = Z - N - B - C`` where `reach` (by default, where that is
        positive at `x`); elsewhere ``A`` is zero, and so are the terms in it.
        """
        p = self.p
        a, b = self._tropomyosin_rates(CaTn)
        k1, k3, km3 = self._rates(x)
        km1, k2, km2 = self._km1, p["k2"], self._km2
        xC0, phi_v = p["xC0"], p["phi_v"]
        if reach is None:
            reach = self._free(x, Z) > 0.0
        aA, k1A, km3A = (np.where(reach, k, 0.0) for k in (a, k1, km3))

        M = np.zeros((x.shape[1], _SIZE, _SIZE))
        # Eq. (22): dN/dt = a A - b N
        M[:, _N, _N] = -aA - b
        M[:, _N, _B] = M[:, _N, _C] = -aA
        M[:, _N, _ONE] = aA * Z
        # Eq. (3): dB/dt = k1 A - (k-1 + k2) B + k-2 C
        M[:, _B, _N] = -k1A
        M[:, _B, _B] = -k1A - km1 - k2
        M[:, _B, _C] = -k1A + km2
        M[:, _B, _ONE] = k1A * Z
        # Eq. (4): dC/dt = k2 B - (k-2 + k3) C + k-3 A
        M[:, _C, _N] = -km3A
        M[:, _C, _B] = k2 - km3A
        M[:, _C, _C] = -km2 - k3 - km3A
        M[:, _C, _ONE] = km3A * Z
        # Eq. (7) times B, plus xB times Eq. (3)
        M[:, _MB, _B] = phi_v * v
        M[:, _MB, _MB] = -km1 - k2
        # Eq. (8) times C, plus xC times Eq. (4)
        M[:, _MC, _N] = -xC0 * km3A
        M[:, _MC, _B] = xC0 * (k2 - km3A)
        M[:, _MC, _C] = phi_v * v - xC0 * km3A
        M[:, _MC, _MC] = -km2 - k3
        M[:, _MC, _ONE] = xC0 * km3A * Z
        return M

    def _get_x(self) -> npt.NDArray[np.float64]:
        return np.stack([self.N, self.B, self.C, self.B_xB, self.C_xC])

    def _set_x(self, x: npt.NDArray[np.float64]) -> None:
        self.N, self.B, self.C, self.B_xB, self.C_xC = x

    def _midpoint_step(
        self,
        x: npt.NDArray[np.float64],
        CaTn: npt.NDArray[np.float64],
        Z: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
        h: npt.NDArray[np.float64],
        reach: npt.NDArray[np.bool_] | None = None,
    ) -> npt.NDArray[np.float64]:
        """
        One exponential-midpoint step of length h[i] for cell i; x has shape
        (5, n). The rates are frozen at the state a half-step predictor
        reaches, and so is the branch of A unless `reach` fixes it.
        """
        # [x, 1] per cell, contiguous, so that a stacked matmul gives each cell
        # the same arithmetic whatever its batch-mates (see Tewari2016).
        x1 = np.empty((x.shape[1], _SIZE, 1))
        x1[:, :_ONE, 0] = x.T
        x1[:, _ONE, 0] = 1.0

        def solve(about, step):
            M = self._system(about, CaTn, Z, v, reach) * step[:, None, None]
            return np.ascontiguousarray((expm_batch(M) @ x1)[:, :_ONE, 0].T)

        return solve(solve(x, 0.5 * h), h)

    def _advance(
        self,
        x: npt.NDArray[np.float64],
        troponin: tuple[npt.NDArray[np.float64], ...],
        t0: float,
        Z: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
        h: float,
    ) -> npt.NDArray[np.float64]:
        """
        One sub-step of length h from time t0 within the step; x has shape (5, n).

        `troponin` is (CaTn at the start of the step, its limit, its rate):
        with calcium held, CaTn relaxes exponentially (Eq. (21)).

        The midpoint rule is second order only where A stays on one side of
        its clamp. Within a sub-step A can cross it upwards only -- at A = 0,
        N decays and B and C detach, so A grows; downwards it is pushed only
        by a fall in Z, which happens between steps -- so a cell that starts
        clamped and ends free is stepped again in two parts: clamped up to
        where Z - N - B - C reaches zero, interpolated linearly between the
        start and a clamped step to the end, and free after.
        """

        def CaTn(t, i=slice(None)):
            start, limit, rate = (part[i] for part in troponin)
            return limit + (start - limit) * np.exp(-rate * t)

        n = x.shape[1]
        x_new = self._midpoint_step(x, CaTn(t0 + 0.5 * h), Z, v, np.full(n, h))
        crossing = (self._free(x, Z) <= 0.0) & (self._free(x_new, Z) > 0.0)
        if not crossing.any():
            return x_new

        i = np.flatnonzero(crossing)
        xi, Zi, vi = x[:, i], Z[i], v[i]
        clamped = np.zeros(i.size, dtype=bool)
        full = np.full(i.size, h)
        g0 = self._free(xi, Zi)
        g1 = self._free(self._midpoint_step(xi, CaTn(t0 + 0.5 * h, i), Zi, vi, full, clamped), Zi)
        # A cell that stays clamped when stepped clamped throughout does so
        theta = np.where(g1 > 0.0, g0 / np.where(g1 > 0.0, g0 - g1, -1.0), 1.0)
        h1 = np.clip(theta, 0.0, 1.0) * h
        h2 = h - h1
        x_mid = self._midpoint_step(xi, CaTn(t0 + 0.5 * h1, i), Zi, vi, h1, clamped)
        x_new[:, i] = self._midpoint_step(x_mid, CaTn(t0 + h1 + 0.5 * h2, i), Zi, vi, h2, ~clamped)
        return x_new

    # ------------------------------------------------------------------
    # Public interface (CardiacActivationModel)
    # ------------------------------------------------------------------

    def advance_step(
        self,
        dt: float,
        Ca_val: float | npt.NDArray[np.float64],
        SL_vals: float | npt.NDArray[np.float64],
        dSL_vals: float | npt.NDArray[np.float64] | None = None,
    ) -> None:
        """
        Advance the model by one time step.

        Parameters
        ----------
        dt : float
            Time step [s]. Any length: it is split into sub-steps of at most
            `_TARGET_SUBSTEP`, with Ca_val, SL_vals and dSL_vals held fixed
            (zero-order hold) across it.
        Ca_val : float or np.ndarray, shape (num_cells,)
            Intracellular calcium concentration [uM].
        SL_vals : float or np.ndarray, shape (num_cells,)
            Sarcomere lengths [um].
        dSL_vals : float or np.ndarray, shape (num_cells,) or None
            Sarcomere length rate of change [um/s], which strains attached
            bridges and the passive dashpot. If None, estimated from the
            sarcomere length recorded on the previous call (zero on the
            first).
        """
        self._begin_step(dt)
        n = self.num_cells
        p = self.p

        SL = np.broadcast_to(np.asarray(SL_vals, dtype=float), (n,)).copy()
        Ca = np.maximum(np.broadcast_to(np.asarray(Ca_val, dtype=float), (n,)), 0.0)
        if dSL_vals is not None:
            v = np.broadcast_to(np.asarray(dSL_vals, dtype=float), (n,)).copy()
        elif self._has_prev_step:
            v = (SL - self._SL_prev) / dt
        else:
            v = np.zeros(n)

        # Eq. (21) with calcium held: CaTn relaxes exponentially to r / (1 + r)
        r = (Ca / p["Ca50"]) ** p["n_tn"]
        limit, rate = r / (1.0 + r), p["k_tn"] * (1.0 + r)
        troponin = (self.CaTn, limit, rate)

        Z = self._z(SL)
        n_sub = max(1, int(np.ceil(dt / _TARGET_SUBSTEP - 1e-9)))
        h = dt / n_sub
        x = self._get_x()
        for i in range(n_sub):
            x = self._advance(x, troponin, i * h, Z, v, h)
        self._set_x(x)
        self.CaTn = limit + (self.CaTn - limit) * np.exp(-rate * dt)

        # Eq. (19) with velocity held: Fp1 relaxes exponentially to mu v
        Fp1_inf = p["mu"] * v
        self.Fp1 = Fp1_inf + (self.Fp1 - Fp1_inf) * np.exp(-p["Kp1"] / p["mu"] * dt)

        self._SL_prev = SL
        self._SL_curr = SL
        self._has_prev_step = True

    def get_active_tension(self) -> npt.NDArray[np.float64]:
        """Active stress [kPa], Eq. (1): K (B xB + C xC), K = rho_myo K_myo."""
        p = self.p
        return p["rho_myo"] * p["K_myo"] * (self.B_xB + self.C_xC)

    def get_active_stiffness(self) -> npt.NDArray[np.float64]:
        r"""
        Active stiffness [kPa per unit Lambda = SL/SL0].

        .. math::
            K_a = SL_0 \, K \, \phi_v \left(B + C\right)

        Velocity enters only the strain equations, as ``phi_v dSL/dt`` in
        both, so it changes d(B xB + C xC)/dt by ``phi_v (B + C)`` per unit
        sarcomere velocity; the rates depend on strain, not on velocity.
        """
        p = self.p
        return p["SL0"] * p["rho_myo"] * p["K_myo"] * p["phi_v"] * (self.B + self.C)

    def bound_calcium_fraction(self) -> npt.NDArray[np.float64]:
        """Fraction of troponin C with calcium bound, ``CaTn`` (Eq. (21))."""
        return self.CaTn

    def get_passive_tension(self) -> npt.NDArray[np.float64]:
        """
        Passive stress [kPa], Eqs. (19)-(20): Fp1 + Kp2 (e^{phi_p (SL - L_r)} - 1).
        As in the authors' code, the exponential spring is not cut off below
        the resting length, where it goes slightly negative.
        """
        p = self.p
        return self.Fp1 + p["Kp2"] * (np.exp(p["phi_p"] * (self._SL_curr - p["L_r"])) - 1.0)

    def get_atp_hydrolysis_rate(self) -> npt.NDArray[np.float64]:
        """
        Crossbridge ATP hydrolysis rate [1/s per head], Eq. (24) without the
        densities: the net C -> A flux k3 C - k-3 A. Times
        ``rho_xb * rho_myo`` (``rho_xb = 0.25 mol/m^3``) it is the ATP flux
        per muscle volume of the paper's energetics.
        """
        x = self._get_x()
        _, k3, km3 = self._rates(x)
        A = np.maximum(self._z(self._SL_curr) - self.N - self.B - self.C, 0.0)
        return k3 * self.C - km3 * A
