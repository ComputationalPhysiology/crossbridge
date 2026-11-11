"""
Rice et al. (2008) cardiac myofilament model.

Based on: Rice, J.J., Wang, F., Bers, D.M., & de Tombe, P.P. (2008).
Approximate model of cooperative activation and crossbridge cycling in cardiac
muscle using ordinary differential equations. Biophysical Journal, 95(5),
2368-2390. https://doi.org/10.1529/biophysj.107.119487

Calcium binds the regulatory site of troponin C with low or high affinity
(``TRPNCaL``, ``TRPNCaH``: without and with strongly bound crossbridges
nearby), and the regulatory units switch between non-permissive ``N`` and
permissive ``P`` at rates set by their calcium occupancy through a steep Hill
function (``nperm = 15``), which stands in for the cooperativity of
neighbouring units. Permissive units recruit crossbridges into a weakly
force-bearing pre-rotated state ``XBprer``, which isomerizes into the
post-rotated ``XBpostr`` (the powerstroke) and detaches. Each attached state
carries the mean strain of its bridges (``xXBprer``, ``xXBpostr``), driven by
half the sarcomere velocity and relaxing to its isometric value, 0 and the
powerstroke ``x_0``; compressed pre-rotated bridges isomerize faster, and
post-rotated ones detach faster either side of ``x_0``. Active force is the
strain-weighted occupancy times the thick-filament single-overlap fraction,
normalized by the post-rotated occupancy at the base rates (Eqs. 33-35), so
that it is about 1 at full activation at 37 degC. Temperature enters through
Q10 factors, ``Q ** ((TmpC - 37) / 10)``; `rabbit_parameters()` is the
paper's rabbit variant.

**Where the class follows the CellML** (Physiome Model Repository,
rice_wang_bers_detombe_2008), which agrees with the authors' own code (the
paper's Data S1) on both points:

* ``kn_p = 500`` and ``kp_n = 50`` /s, which Table 1 prints the other way
  round. With the table's values the fully activated N -> P equilibrium is
  P/N = 0.1, so at most 9% of units would ever be permissive, while Eqs.
  33-35 normalize force on the assumption that ``kn_pT >> kp_nT``.
* the sign of Eq. 22, which the paper's text loses:
  ``hfmd = exp(-sign(xXBprer) hfmdc (xXBprer / x_0)^2)``. Compressed bridges
  (negative ``xXBprer``, during shortening) isomerize faster, as the paper's
  prose says.

**Where it follows the paper** (and Data S1) instead: titin's rest length
``SLrest`` is 1.9 um, where the text and Figs. 1C, 3C and 6 put zero passive
force (the CellML has 1.85). ``TmpC`` defaults to 22.5 degC, the temperature
of Figs. 5-7 (Table 3) and of Data S1 (the CellML has 24).

``N_NoXB`` and ``P_NoXB`` (Eqs. 7-8) are the regulatory units outside the
single-overlap region, which switch at the same rates as ``N`` but recruit no
crossbridges. No output and no other state depends on them; they are kept so
that the states are Table 2's and the CellML's, one for one.

**Departures from the paper** (and the CellML), three guards outside its
range: the overlap length is floored at zero, since the formula of Eqs.
42-46 turns negative below SL = ``len_thin`` (1.2 um, under the paper's
minimum of 1.4); calcium is clamped at zero on input, as an EP model's
undershoot can make it slightly negative; and the exponent arguments of
``hfmd`` and ``gxbmd`` (Eqs. 22 and 24) are capped at ``_EXP_ARG_MAX`` (40).
Without that cap, strains past about 10 ``x_0`` drive Eq. 30's coefficient
times the sub-step beyond the range of the matrix exponential (norms up to
1.7e38, where `expm_batch` stops squaring), and the result is NaN; with both
exponents at 40 it stays below 1e34 at the default rates. The cap binds only at strains 2-6
``x_0`` or more from their isometric values, and never inside the verified
range below, where the largest argument reached is 11.

Two expressions are evaluated in forms that are the same algebra without a
division that can be by zero: Eq. 10 as
``sqrt(Tropreg^nperm / (Tropreg^nperm + perm50^nperm))``, without dividing by
``Tropreg``, which is zero without calcium; and the strain coefficients of
Eqs. 29-30 with the duty fractions of Eqs. 31-32 multiplied out,
``xPsi / dutyprer = xPsi D / (fappT (hbT + gxbT))`` and
``xPsi hfT / dutypostr = xPsi D / fappT`` (``D`` their common denominator),
without dividing by ``dutypostr``, which is zero where ``hfT`` underflows.

**Numerical scheme.** Given calcium, SL, dSL and the rates, the nine states
obey a linear system (affine in the strains), which is solved exactly over a
sub-step with a matrix exponential of the system augmented by its constant
term. The rates -- the N <-> P switch through the troponin occupancy, the
isomerization and detachment through the strains, and the duty fractions
through both -- are taken at the sub-step's midpoint, from a half-step
predictor (the exponential midpoint rule), which keeps the error second order
in the sub-step. The system is stiff (the authors used CVODE): ``kp_nT``
reaches 2500 /s at 22.5 degC and 5000 /s at 37 degC when the filament is off,
and at 37 degC the strains relax at about 10^4 /s; the exponential takes
those exactly. What it cannot take exactly is the strain dependence of
``hfT`` and ``gxbT``, exponential in the squared strain: a sub-step that
moves the strain by a large fraction of ``x_0`` freezes them at a midpoint
the predictor misjudges, and with 0.1 ms sub-steps alone, shortening at
100 um/s for 1 ms gave NaN. So each cell takes sub-steps of at most
``_TARGET_SUBSTEP`` and, when it moves fast, more of them, short enough that
SL moves by at most ``_STRAIN_PER_SUBSTEP`` x_0 (0.35) in each -- a count of
its own, so that its result does not depend on its batch-mates. 0.35 is the
largest value that keeps one 1 ms call within 1% of the same call at 1 us
steps when shortening at 100 um/s (22.5 degC) and 150 um/s (37 degC).
Calcium, SL and dSL are held over the call.

**Verified range.** From an activated state (Ca 2 uM, SL 2 um), one 1 ms
call is within 0.9% (22.5 degC) and 0.25% (37 degC) of the same call at
1 us steps -- which agree with a Radau solution of the CellML equations to
3e-5 -- for shortening at up to 300 um/s (with SL held, up to the cap below)
and lengthening at up to 50 um/s, and 10 ms ramps at 1 ms calls stay within
2e-3 for shortening at up to 100 um/s. The velocity-driven count is capped at
``_MAX_STRAIN_SUBSTEPS`` (500) per call, which covers SL changing by
500 x 0.35 x_0 = 1.2 um in one call, more than a sarcomere's working range;
beyond it the sub-steps lengthen and the error grows without bound: the
results stay finite but are meaningless (Ta of order 1e11 for a dSL of
5000 um/s over one 1 ms call with SL held). Fast stretch is outside the
verified range for a reason of the model's own: past a strain of a few
``x_0`` (lengthening at 100 um/s for 1 ms) the post-rotated bridges detach at
1e20 /s and more, and Eq. 30's coefficient ``xPsi hfT / dutypostr`` reaches
1e21 /s, so that its right-hand side is a cancellation no solver resolves
(the class is then 1.3% from Radau at 22.5 degC). Further out the exponent
cap binds, and the results, though finite, are no longer the model's.

Examples
--------
>>> import numpy as np
>>> from crossbridge.rice2008 import Rice2008
>>>
>>> model = Rice2008(num_cells=10)
>>> model.advance_step(1e-3, Ca_val=1.0, SL_vals=np.full(10, 2.2))
>>> Ta = model.get_active_tension()  # shape (10,), normalized force times Ta_max
"""

import numpy as np
import numpy.typing as npt

from ._linalg import expm_batch
from .base import CardiacActivationModel

#: Longest sub-step [s] over which the rates are held.
_TARGET_SUBSTEP = 1e-4  # [s]

#: Most sarcomere-length change per sub-step, in units of x_0 (each strain is
#: driven by half of it): a cell moving fast takes more sub-steps than
#: `_TARGET_SUBSTEP` gives. Chosen by measurement; see the module docstring.
_STRAIN_PER_SUBSTEP = 0.35  # [-]

#: Cap on the sub-steps one call takes for velocity (module docstring).
_MAX_STRAIN_SUBSTEPS = 500

#: Cap on the exponent arguments of hfmd and gxbmd (Eqs. 22, 24), a guard that
#: keeps the system within the range of `expm_batch` (module docstring).
_EXP_ARG_MAX = 40.0  # [-]

#: The states, in the order of the linear system.
_STATES = (
    "TRPNCaH",
    "TRPNCaL",
    "N_NoXB",
    "P_NoXB",
    "N",
    "XBprer",
    "XBpostr",
    "xXBprer",
    "xXBpostr",
)
_TRPNCaH, _TRPNCaL, _N_NoXB, _P_NoXB, _N, _XBprer, _XBpostr, _xXBprer, _xXBpostr = range(9)
# The last row/column of the linear system carries the constant term.
_ONE = 9
_SIZE = 10


def _cycle_denominator(fapp, gapp, hf, hb, gxb):
    """
    King-Altman denominator of the cycle P -> XBprer -> XBpostr -> P, which
    sets the duty fractions (Eqs. 31-32) and, at the base rates, the maximal
    occupancies (Eqs. 33-34).
    """
    return fapp * gxb + gapp * hb + fapp * hb + gapp * gxb + fapp * hf + gxb * hf


class Rice2008(CardiacActivationModel):
    """
    Vectorized Rice et al. (2008) rat myofilament model.

    Attributes
    ----------
    TRPNCaH, TRPNCaL : ndarray, shape (num_cells,)
        Fraction of regulatory troponin C sites with Ca2+ bound, at high and
        low affinity.
    N_NoXB, P_NoXB : ndarray, shape (num_cells,)
        Non-permissive and permissive fractions of the regulatory units
        outside the single-overlap region. They feed nothing.
    N : ndarray, shape (num_cells,)
        Fraction of crossbridges in the non-permissive state.
    XBprer, XBpostr : ndarray, shape (num_cells,)
        Fraction of crossbridges attached, pre- and post-rotation.
    xXBprer, xXBpostr : ndarray, shape (num_cells,)
        Mean strain of the pre- and post-rotated crossbridges [um].

    The permissive fraction ``P = 1 - N - XBprer - XBpostr`` is derived.
    Sarcomere length is an input, as in every model of this package; the
    paper's muscle model (mass, viscosity, series element; Eqs. 38-40) belongs
    to the caller, and `get_passive_tension` supplies the passive force.
    Forces are the paper's normalized ones times `Ta_max`.
    """

    ODE_FILE = None

    TRPNCaH: npt.NDArray[np.float64]
    TRPNCaL: npt.NDArray[np.float64]
    N_NoXB: npt.NDArray[np.float64]
    P_NoXB: npt.NDArray[np.float64]
    N: npt.NDArray[np.float64]
    XBprer: npt.NDArray[np.float64]
    XBpostr: npt.NDArray[np.float64]
    xXBprer: npt.NDArray[np.float64]
    xXBpostr: npt.NDArray[np.float64]

    #: Attributes that evolve; see `CardiacActivationModel.get_state`.
    _state_names = (*_STATES, "_SL_prev", "_SL_curr", "_has_prev_step")

    def __init__(self, num_cells: int, Ta_max: float = 1.0, params: dict | None = None):
        """
        Initialize the Rice2008 model.

        Parameters
        ----------
        num_cells : int
            Number of independent cells / FEM integration points.
        Ta_max : float, optional
            Scale of the forces: `get_active_tension`, `get_passive_tension`
            and `get_active_stiffness` are the paper's normalized forces times
            `Ta_max`, so the default 1.0 gives them as the paper does. Unlike
            most models here, this one uses it.
        params : dict, optional
            Parameter overrides. Keys should match those returned by
            `default_parameters()`.
        """
        super().__init__(int(num_cells), Ta_max, params)

        self.p = type(self).default_parameters()
        if params:
            self.p.update(params)
        p = self.p
        self.dt = p["dt"]

        # Rate constants at temperature TmpC (Eq. 3), fixed for the model's
        # lifetime; those with a state-dependent factor are stored without it.
        def q(Q: float) -> float:
            return Q ** ((p["TmpC"] - 37.0) / 10.0)

        xb = p["xbmodsp"]
        self._konT = p["kon"] * q(p["Qkon"])
        self._koffHT = p["koffH"] * p["koffmod"] * q(p["Qkoff"])
        self._koffLT = p["koffL"] * p["koffmod"] * q(p["Qkoff"])
        self._kn_p = p["kn_p"] * q(p["Qkn_p"])  # times permtot
        self._kp_n = p["kp_n"] * q(p["Qkp_n"])  # times min(1 / permtot, 100)
        self._fappT = p["fapp"] * xb * q(p["Qfapp"])
        self._gapp = p["gapp"] * xb * q(p["Qgapp"])  # times 1 + (1 - SOVFthick) gslmod
        self._hf = p["hf"] * xb * q(p["Qhf"])  # times hfmd
        self._hbT = p["hb"] * xb * q(p["Qhb"])
        self._gxb = p["gxb"] * xb * q(p["Qgxb"])  # times gxbmd

        # Eqs. 33-34: the steady occupancies at the base rates, which
        # normalize force and the strongly bound fraction of Eq. 36.
        fapp, gapp, hf, hb, gxb = (p[k] for k in ("fapp", "gapp", "hf", "hb", "gxb"))
        D0 = _cycle_denominator(fapp, gapp, hf, hb, gxb)
        self._XBprer_max = (fapp * hb + fapp * gxb) / D0
        self._XBpostr_max = fapp * hf / D0

        self.reset()

    @classmethod
    def default_parameters(cls) -> dict:
        """
        Return default parameters: Table 1 of Rice et al. (2008), rat, with
        rates in /s, lengths in um and calcium in uM.

        ``kn_p`` and ``kp_n`` are the CellML's and Data S1's (the table has
        them swapped; see the module docstring), and ``TmpC`` is 22.5 degC,
        Table 3's. Rates are given at 37 degC and scaled to ``TmpC`` by
        ``Q ** ((TmpC - 37) / 10)``. Mechanics parameters (mass, viscosity,
        series elasticity, afterload) are not here: sarcomere length is an
        input.
        """
        p: dict = {}
        # Numerical: a *suggested* coupling dt; advance_step sub-steps
        # internally whatever the step it is given.
        p["dt"] = 1e-3  # [s]
        # Initialization helper (not a model parameter): the diastolic Ca at
        # which reset() puts troponin at its steady state (Table 3).
        p["Ca0"] = 0.09  # [uM]

        # Conditions. SL0 only sets the unit of Lambda = SL / SL0 in which Ka
        # is reported: the rest length, as in Tewari2016.
        p["SL0"] = 1.9  # [um]
        p["TmpC"] = 22.5  # [degC]

        # Sarcomere geometry (Appendix)
        p["len_thick"] = 1.65  # [um]
        p["len_hbare"] = 0.1  # [um]
        p["len_thin"] = 1.2  # [um]

        # Troponin and thin-filament regulation
        p["kon"] = 50.0  # [1/(uM s)]
        p["koffL"] = 250.0  # [1/s]
        p["koffH"] = 25.0  # [1/s]
        p["koffmod"] = 1.0  # [-] species modifier of koffL, koffH
        p["perm50"] = 0.5  # [-]
        p["nperm"] = 15.0  # [-]
        p["kn_p"] = 500.0  # [1/s]
        p["kp_n"] = 50.0  # [1/s]
        p["Qkon"] = 1.5  # [-]
        p["Qkoff"] = 1.3  # [-]
        p["Qkn_p"] = 1.6  # [-]
        p["Qkp_n"] = 1.6  # [-]

        # Crossbridge cycle
        p["fapp"] = 500.0  # [1/s]
        p["gapp"] = 70.0  # [1/s]
        p["gslmod"] = 6.0  # [-]
        p["hf"] = 2000.0  # [1/s]
        p["hfmdc"] = 5.0  # [-]
        p["hb"] = 400.0  # [1/s]
        p["gxb"] = 70.0  # [1/s]
        p["sigmap"] = 8.0  # [-]
        p["sigman"] = 1.0  # [-]
        p["xbmodsp"] = 1.0  # [-] species modifier of the crossbridge rates
        p["Qfapp"] = 6.25  # [-]
        p["Qgapp"] = 2.5  # [-]
        p["Qhf"] = 6.25  # [-]
        p["Qhb"] = 6.25  # [-]
        p["Qgxb"] = 6.25  # [-]

        # Mean strain
        p["x_0"] = 0.007  # [um] powerstroke
        p["xPsi"] = 2.0  # [-]

        # Passive force (Eqs. 47-48)
        p["SLrest"] = 1.9  # [um]
        p["PCon_t"] = 0.002  # [-]
        p["PExp_t"] = 10.0  # [1/um]
        p["SL_c"] = 2.25  # [um]
        p["PCon_c"] = 0.02  # [-]
        p["PExp_c"] = 70.0  # [1/um]
        return p

    @classmethod
    def rabbit_parameters(cls) -> dict:
        """
        The paper's rabbit variant: the defaults with the crossbridge rates
        slowed five-fold (``xbmodsp = 0.2``) and Ca2+ unbinding by 10%
        (``koffmod = 0.9``), at body temperature (``TmpC = 37``). The paper's
        larger rabbit mass is mechanics and stays with the caller.
        """
        return {**cls.default_parameters(), "xbmodsp": 0.2, "koffmod": 0.9, "TmpC": 37.0}

    def reset(self) -> None:
        """
        Reset to the quiescent state: no crossbridges attached, every
        regulatory unit non-permissive (N = N_NoXB = 1), the strains at their
        isometric values (Table 2: 0 and x_0), and troponin C at its steady
        state for `params["Ca0"]`.
        """
        n = self.num_cells
        Ca0 = max(self.p["Ca0"], 0.0)
        self.TRPNCaH = np.full(n, self._konT * Ca0 / (self._konT * Ca0 + self._koffHT))
        self.TRPNCaL = np.full(n, self._konT * Ca0 / (self._konT * Ca0 + self._koffLT))
        self.N_NoXB = np.ones(n)
        self.P_NoXB = np.zeros(n)
        self.N = np.ones(n)
        self.XBprer = np.zeros(n)
        self.XBpostr = np.zeros(n)
        self.xXBprer = np.zeros(n)
        self.xXBpostr = np.full(n, self.p["x_0"])

        self._SL_prev = np.full(n, self.p["SL0"])
        self._SL_curr = np.full(n, self.p["SL0"])
        self._has_prev_step = False

        self._prev_bound_ca = np.zeros(n)
        self._last_dt = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _overlap(
        self, SL: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Eqs. 42-46: single-overlap fractions (SOVFthick, SOVFthin) of Fig. 1B."""
        p = self.p
        sovr_ze = np.minimum(p["len_thick"] / 2.0, SL / 2.0)
        sovr_cle = np.maximum(p["len_thin"] - SL / 2.0, p["len_hbare"] / 2.0)
        L = np.maximum(sovr_ze - sovr_cle, 0.0)
        return 2.0 * L / (p["len_thick"] - p["len_hbare"]), L / p["len_thin"]

    def _rates(
        self, x: npt.NDArray[np.float64], SL: npt.NDArray[np.float64]
    ) -> dict[str, npt.NDArray[np.float64]]:
        """The state-dependent rates [1/s] and duty fractions at state `x`, shape (9, n)."""
        p = self.p
        SOVFthick, SOVFthin = self._overlap(SL)

        # Eqs. 9-13. Eq. 10 as sqrt(T^n / (T^n + perm50^n)): the printed form
        # divides by Tropreg, which is zero without calcium.
        Tropreg = (1.0 - SOVFthin) * x[_TRPNCaL] + SOVFthin * x[_TRPNCaH]
        T_n = np.maximum(Tropreg, 0.0) ** p["nperm"]
        permtot = np.sqrt(T_n / (T_n + p["perm50"] ** p["nperm"]))
        kn_pT = self._kn_p * permtot
        kp_nT = self._kp_n / np.maximum(permtot, 0.01)  # min(1 / permtot, 100)

        # Eqs. 18-25
        x_0, xpre, xpost = p["x_0"], x[_xXBprer], x[_xXBpostr]
        fappT = np.full_like(SL, self._fappT, dtype=float)
        gappT = self._gapp * (1.0 + (1.0 - SOVFthick) * p["gslmod"])
        # exp(-sign(x) hfmdc (x / x_0)^2), the CellML's sign (module docstring);
        # both exponents capped at _EXP_ARG_MAX, a guard far outside the
        # verified range
        hf_arg = -p["hfmdc"] * xpre * np.abs(xpre) / (x_0 * x_0)
        hfT = self._hf * np.exp(np.minimum(hf_arg, _EXP_ARG_MAX))
        hbT = np.full_like(SL, self._hbT, dtype=float)
        sigma = np.where(xpost < x_0, p["sigmap"], p["sigman"])
        gxb_arg = sigma * ((xpost - x_0) / x_0) ** 2
        gxbT = self._gxb * np.exp(np.minimum(gxb_arg, _EXP_ARG_MAX))

        # Eqs. 31-32: steady fractions of the cycle in each attached state,
        # which set how fast the strains relax.
        D = _cycle_denominator(fappT, gappT, hfT, hbT, gxbT)
        return {
            "kn_pT": kn_pT,
            "kp_nT": kp_nT,
            "fappT": fappT,
            "gappT": gappT,
            "hfT": hfT,
            "hbT": hbT,
            "gxbT": gxbT,
            "dutyprer": (fappT * hbT + fappT * gxbT) / D,
            "dutypostr": fappT * hfT / D,
        }

    def _system(
        self,
        Ca: npt.NDArray[np.float64],
        dSL: npt.NDArray[np.float64],
        rates: dict[str, npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        """
        The (n, 10, 10) matrix of d[x, 1]/dt for frozen rates: Eqs. 1-2, 7-8
        and 14-17 with P = 1 - N - XBprer - XBpostr, and Eqs. 29-30.
        """
        p = self.p
        kn_pT, kp_nT = rates["kn_pT"], rates["kp_nT"]
        fappT, gappT, hfT, hbT, gxbT = (rates[k] for k in ("fappT", "gappT", "hfT", "hbT", "gxbT"))
        x_0, xPsi = p["x_0"], p["xPsi"]
        konCa = self._konT * Ca

        M = np.zeros((len(kn_pT), _SIZE, _SIZE))
        # Eqs. 1-2: troponin C
        M[:, _TRPNCaH, _TRPNCaH] = -konCa - self._koffHT
        M[:, _TRPNCaH, _ONE] = konCa
        M[:, _TRPNCaL, _TRPNCaL] = -konCa - self._koffLT
        M[:, _TRPNCaL, _ONE] = konCa
        # Eqs. 7-8: regulatory units outside the single overlap
        M[:, _N_NoXB, _N_NoXB] = -kn_pT
        M[:, _N_NoXB, _P_NoXB] = kp_nT
        M[:, _P_NoXB, _N_NoXB] = kn_pT
        M[:, _P_NoXB, _P_NoXB] = -kp_nT
        # Eq. 14: dN/dt = -kn_pT N + kp_nT P
        M[:, _N, _N] = -kn_pT - kp_nT
        M[:, _N, _XBprer] = M[:, _N, _XBpostr] = -kp_nT
        M[:, _N, _ONE] = kp_nT
        # Eq. 16: dXBprer/dt = fappT P - (gappT + hfT) XBprer + hbT XBpostr
        M[:, _XBprer, _N] = -fappT
        M[:, _XBprer, _XBprer] = -fappT - gappT - hfT
        M[:, _XBprer, _XBpostr] = -fappT + hbT
        M[:, _XBprer, _ONE] = fappT
        # Eq. 17
        M[:, _XBpostr, _XBprer] = hfT
        M[:, _XBpostr, _XBpostr] = -hbT - gxbT
        # Eqs. 29-30 with the duty fractions of Eqs. 31-32 multiplied out, which
        # is the same algebra without dividing by a fraction that can underflow:
        # xPsi / dutyprer = xPsi D / (fappT (hbT + gxbT)) and
        # xPsi / dutypostr * hfT = xPsi D / fappT, finite where hfT underflows.
        D = _cycle_denominator(fappT, gappT, hfT, hbT, gxbT)
        # Eq. 29: d xXBprer/dt = dSL/2 + xPsi/dutyprer (-fappT xXBprer
        #                          + hbT (xXBpostr - x_0 - xXBprer))
        a = xPsi * D / (fappT * (hbT + gxbT))
        M[:, _xXBprer, _xXBprer] = -a * (fappT + hbT)
        M[:, _xXBprer, _xXBpostr] = a * hbT
        M[:, _xXBprer, _ONE] = 0.5 * dSL - a * hbT * x_0
        # Eq. 30: d xXBpostr/dt = dSL/2 + xPsi/dutypostr hfT (xXBprer + x_0 - xXBpostr)
        b = xPsi * D / fappT
        M[:, _xXBpostr, _xXBprer] = b
        M[:, _xXBpostr, _xXBpostr] = -b
        M[:, _xXBpostr, _ONE] = 0.5 * dSL + b * x_0
        return M

    def _get_x(self) -> npt.NDArray[np.float64]:
        return np.stack([getattr(self, name) for name in _STATES])

    def _set_x(self, x: npt.NDArray[np.float64]) -> None:
        for i, name in enumerate(_STATES):
            setattr(self, name, x[i])

    def _advance(
        self,
        x: npt.NDArray[np.float64],
        Ca: npt.NDArray[np.float64],
        SL: npt.NDArray[np.float64],
        dSL: npt.NDArray[np.float64],
        h: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        One exponential-midpoint sub-step; x has shape (9, n) and each cell
        its own length h, shape (n,).
        """
        # [x, 1] per cell, contiguous: a stacked matmul then gives each cell the
        # same arithmetic whatever its batch-mates (as in Tewari2016).
        x1 = np.empty((x.shape[1], _SIZE, 1))
        x1[:, :_ONE, 0] = x.T
        x1[:, _ONE, 0] = 1.0

        def solve(rates, step):
            M = self._system(Ca, dSL, rates) * step[:, None, None]
            return np.ascontiguousarray((expm_batch(M) @ x1)[:, :_ONE, 0].T)

        x_half = solve(self._rates(x, SL), 0.5 * h)
        return solve(self._rates(x_half, SL), h)

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
            Time step [s]. Any length: each cell splits it into sub-steps of
            at most `_TARGET_SUBSTEP`, and more when it shortens or lengthens
            fast (see the module docstring), with Ca_val, SL_vals and dSL_vals
            held fixed (zero-order hold) across it.
        Ca_val : float or np.ndarray, shape (num_cells,)
            Intracellular calcium concentration [uM]. Negative values are
            taken as zero.
        SL_vals : float or np.ndarray, shape (num_cells,)
            Sarcomere lengths [um].
        dSL_vals : float or np.ndarray, shape (num_cells,) or None
            Sarcomere length rate of change [um/s], which strains attached
            crossbridges. If None, estimated from the sarcomere length
            recorded on the previous call (zero on the first).
        """
        self._begin_step(dt)
        n = self.num_cells

        SL = np.broadcast_to(np.asarray(SL_vals, dtype=float), (n,)).copy()
        Ca = np.maximum(np.broadcast_to(np.asarray(Ca_val, dtype=float), (n,)), 0.0)
        if dSL_vals is not None:
            dSL = np.broadcast_to(np.asarray(dSL_vals, dtype=float), (n,)).copy()
        elif self._has_prev_step:
            dSL = (SL - self._SL_prev) / dt
        else:
            dSL = np.zeros(n)

        # Sub-steps per cell: at most _TARGET_SUBSTEP long, and short enough
        # that SL moves by at most _STRAIN_PER_SUBSTEP x_0 in each (capped).
        # Each cell takes its own, so its result does not depend on its
        # batch-mates; cells that have finished are left alone.
        n_time = max(1, int(np.ceil(dt / _TARGET_SUBSTEP - 1e-9)))
        n_strain = np.ceil(np.abs(dSL) * dt / (_STRAIN_PER_SUBSTEP * self.p["x_0"]))
        # (fmin: a NaN dSL takes the cap and shows in the state, not skipped)
        n_sub = np.maximum(n_time, np.fmin(n_strain, _MAX_STRAIN_SUBSTEPS)).astype(int)
        h = dt / n_sub
        x = self._get_x()
        for k in range(int(n_sub.max(initial=0))):
            active = n_sub > k
            if active.all():
                x = self._advance(x, Ca, SL, dSL, h)
            else:
                x[:, active] = self._advance(
                    x[:, active], Ca[active], SL[active], dSL[active], h[active]
                )
        self._set_x(x)

        self._SL_prev = SL
        self._SL_curr = SL
        self._has_prev_step = True

    def get_active_tension(self) -> npt.NDArray[np.float64]:
        """
        Active tension, Eq. 35: the normalized force times `Ta_max`,
        Ta = Ta_max SOVFthick (XBprer xXBprer + XBpostr xXBpostr) / (x_0 XBpostr_max),
        with XBpostr_max the post-rotated occupancy at the base rates (Eq. 34).
        """
        SOVFthick, _ = self._overlap(self._SL_curr)
        force = self.XBprer * self.xXBprer + self.XBpostr * self.xXBpostr
        return self.Ta_max * (SOVFthick * force / (self.p["x_0"] * self._XBpostr_max))

    def get_active_stiffness(self) -> npt.NDArray[np.float64]:
        r"""
        Active stiffness [`Ta_max` per unit Lambda = SL/SL0].

        .. math::
            K_a = \frac{SL_0\, T_{a,max}\, SOVF_{thick}\,(XB_{prer} + XB_{postr})}
                       {2\, x_0\, XB_{postr}^{max}}

        Velocity enters the model only through the strain equations (Eqs.
        29-30), each as :math:`\dot{SL}/2 = SL_0 \dot\Lambda / 2`, and
        tension is linear in the strains, with weights
        :math:`T_{a,max} SOVF_{thick} XB / (x_0 XB_{postr}^{max})`. The
        length dependence of the overlap itself, :math:`SOVF_{thick}'(SL)`,
        is absent: it is a dependence of :math:`T_a = g(r, \lambda)` on
        :math:`\lambda`, not of the states' right-hand side on
        :math:`\dot\lambda`, so it is not part of
        :math:`K_a = \nabla_r g \cdot \partial h / \partial \dot\lambda` as the
        base class defines it (Regazzoni & Quarteroni 2020), just as
        Land2017's :math:`h(\lambda)` is not. It is zero above SL = 2.3 um,
        where :math:`SOVF_{thick} = 1`. ``Ka >= 0`` as long as the
        occupancies are.
        """
        SOVFthick, _ = self._overlap(self._SL_curr)
        attached = self.XBprer + self.XBpostr
        return self.Ta_max * (
            self.p["SL0"] * SOVFthick * attached / (2.0 * self.p["x_0"] * self._XBpostr_max)
        )

    def bound_calcium_fraction(self) -> npt.NDArray[np.float64]:
        """
        Apparent troponin C occupancy, Eq. 37, which the paper hands back to
        the cell model as troponin buffering (Eq. 56):
        ``(1 - SOVFthin) TRPNCaL + SOVFthin (FrSBXB TRPNCaH + (1 - FrSBXB) TRPNCaL)``.
        Outside the single overlap the sites bind with low affinity; inside it
        with high affinity in proportion to the strongly bound fraction,
        ``FrSBXB = (XBprer + XBpostr) / (XBprer_max + XBpostr_max)`` (Eq. 36).
        """
        _, SOVFthin = self._overlap(self._SL_curr)
        FrSBXB = (self.XBprer + self.XBpostr) / (self._XBprer_max + self._XBpostr_max)
        high = FrSBXB * self.TRPNCaH + (1.0 - FrSBXB) * self.TRPNCaL
        return (1.0 - SOVFthin) * self.TRPNCaL + SOVFthin * high

    def get_passive_tension(self) -> npt.NDArray[np.float64]:
        """
        Passive tension, Eqs. 47-48, times `Ta_max`: titin, reflected about
        ``SLrest``, ``sign(SL - SLrest) PCon_t (exp(PExp_t |SL - SLrest|) - 1)``,
        and collagen above ``SL_c``, ``PCon_c (exp(PExp_c (SL - SL_c)) - 1)``.
        """
        p = self.p
        SL = self._SL_curr
        d = SL - p["SLrest"]
        titin = np.sign(d) * p["PCon_t"] * (np.exp(p["PExp_t"] * np.abs(d)) - 1.0)
        collagen = np.where(
            SL > p["SL_c"], p["PCon_c"] * (np.exp(p["PExp_c"] * (SL - p["SL_c"])) - 1.0), 0.0
        )
        return self.Ta_max * (titin + collagen)
