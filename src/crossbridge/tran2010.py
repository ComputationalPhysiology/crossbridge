"""
Tran et al. (2010) metabolite-sensitive cardiac myofilament model.

Based on: Tran, K., Smith, N.P., Loiselle, D.S., & Crampin, E.J. (2010). A
metabolite-sensitive, thermodynamically constrained model of cardiac
cross-bridge cycling: implications for force development during ischemia.
Biophysical Journal, 98(2), 267-276. https://doi.org/10.1016/j.bpj.2009.10.011

Rice et al. (2008) (`Rice2008`) with the crossbridge cycle made reversible and
dependent on MgATP, MgADP, Pi and H+, parameterized for guinea pig
(``xbmodsp = 0.2``). The rates of `Rice2008` become pseudo-first-order rates at
reference conditions (MgATP 5 mM, MgADP 36 uM, Pi 2 mM, pH 7.15), and

* Pi binds on the reverse of attachment: ``alpha1- = gappT Pi / Pi_ref``;
* the metabolic proton is released on the powerstroke, and MgADP binds between
  two force-bearing substates of ``XBpostr``:
  ``alpha2- = hbT (H / H_ref) (MgADP / MgADP_ref) (MgADP_ref + kdADP) / (MgADP + kdADP)``;
* MgATP detaches the post-rotated bridge (and is hydrolysed):
  ``alpha3+ = gxbT (MgATP / MgATP_ref) (MgADP_ref + kdADP) / (MgADP + kdADP)``;
* the reverse of that step, from ``P`` back to ``XBpostr``, closes the cycle
  thermodynamically: ``alpha3- = fxbT = kdADP fappT hfT gxbT Pi_ref H_ref /
  (MgATP_ref kMgATP hbT gappT)``, with ``kMgATP`` = 15400 mM^2. It is minute
  (1.3e-10 /s at the defaults), as the paper says;
* H+ competes with Ca2+ for troponin C, which slows calcium binding:
  ``konT = kon Qkon^... (H_ref^m + kdHCa^m) / (H^m + kdHCa^m)``.

The strain equations and duty fractions take the new rates (Appendix), and force
is normalized by the post-rotated occupancy at the base rates and reference
metabolites, so that it is the paper's normalized force. Forces are that times
`Ta_max` (default 1.0, not kPa), as in `Rice2008`. Metabolites are parameters,
in mM (``MgATP``, ``MgADP``, ``Pi``) and pH units (``pH``), fixed for the
model's lifetime; build a new model to change them.

**The class follows the Physiome CellML** (tran_2009, curated by C. Lloyd) in
its equations and parameters, including where the paper says otherwise, with
one exception below:

* bridges leave ``XBprer`` (``alpha1-``) and ``XBpostr`` (``alpha3+``) for the
  non-permissive ``N``, and ``fxbT`` brings them back from ``P`` while ``P``'s
  equation loses nothing for it: the CellML's ``dP/dt = kn_pT N - (alpha1+ +
  kp_nT) P``, with ``N = 1 - P - XBprer - XBpostr``. The paper's Appendix
  returns both detachments to ``P``, as `Rice2008` does, and takes ``fxbT``
  from ``P``. The steady forces of the figures differ by at most 1.4%
  between the two (Fig. 2A, Fig. 3B);
* the duty fractions' common denominator has the term ``alpha1- alpha3-``
  twice and no ``alpha1- alpha2-``, and the maximal pre-rotated occupancy has
  ``hbT`` (with its temperature and species factors) where the others have base
  rates; the paper's Appendix prints both the same way. The first scales how
  fast the strains relax, the second only the apparent troponin occupancy of
  `bound_calcium_fraction`;
* temperature is in degC, and the rates are given at 36.85 degC (the CellML's
  310 K, where `Rice2008` and the paper use 37 degC); ``TmpC`` defaults to the
  CellML's 299 K, 25.85 degC;
* titin's rest length ``SLrest`` is the CellML's 1.85 um (`Rice2008` takes the
  Rice paper's 1.9).

**The exception: the proton concentration.** The CellML computes it as
``H = 10^7 / 10^pH`` in uM, ten times [H+] (``10^(6 - pH)`` uM), against a
dissociation constant ``kdHCa`` of 0.02 uM (the paper's 2e-5 mM). The class
uses the true [H+]. Only the ratio ``H / H_ref`` enters the crossbridge rates,
so the difference shows in the pH dependence of calcium binding, where the
paper's Fig. 2B decides it: with [H+] the relative Ca50 at pH 7.5, 7.2, 6.8 and
6.4 (Fig. 2B's conditions: maximal activation, SL 2.1 um, 30 degC) is 42, 68,
151 and 363% of its value at pH 7, on the paper's curve (about 48, 70, 140 and
355); with the CellML's H, 32, 63, 160 and 409. It also makes ``fxbT`` ten
times smaller than the CellML's, which changes a twitch's outputs (with a
release or a stretch at its peak included) by less than 1e-8 of their peaks.

**Departures from the CellML's literal forms** are those of `Rice2008` (the
exponent cap ``_EXP_ARG_MAX`` on ``hfmd`` and ``gxbmd``, which binds only in
fast lengthening, the overlap floor, the calcium clamp, ``Max(Tropreg, 0)``)
and its division-safe evaluations, here of the strain coefficient
``xPsi alpha2+ / dutypostr``, written as ``xPsi S / (alpha1+ + alpha3- + alpha1- fxbT /
alpha2+)`` with ``fxbT / alpha2+`` computed without ``alpha2+`` (``fxbT`` is proportional to
it), which is the same algebra without a division by ``alpha2+``, zero where it underflows.

**Numerical scheme.** As `Rice2008`'s (its integrator, here with ``P`` in
place of ``N`` among the occupancies): each sub-step is the exact exponential
of the frozen-rate linear system, block by block, about the occupancies'
steady state, which with the CellML's structure is ``kn_pT C / det`` with
``C`` and ``det`` sums of products of positive rates (never singular).

**What reproduces** (``tests/test_tran2010.py`` pins each model value; paper
values read off its figures; isometric steady states at Ca 200 uM, computed
exactly: troponin and occupancies at their steady state, strains at 0 and x_0):

* Fig. 2A, force against Pi relative to Pi = 0 (MgATP 5 mM, MgADP 36.3 uM, pH
  7.15, SL 2.2 um, 22 degC): 73.5, 58.1, 48.1, 41.0 and 31.6% at 5, 10, 15,
  20 and 30 mM; the paper's curve about 72, 57, 49, 42 and 33.
* Fig. 2B, as above.
* Fig. 2D, pH 7.2, 7.0, 6.8 and 6.4 (SL 2.1 um, 30 degC): maximal force
  relative to pH 7.2 of 1, 0.90, 0.78 and 0.50 (paper 1, 0.89, 0.77, 0.49) and
  Ca50 0.71, 1.03, 1.56 and 3.74 uM (paper about 0.75, 1.0, 1.5 and 4). The
  paper's absolute forces are 1.44 times the model's (1.18 against 0.82 at pH
  7.2), at every pH alike.
* Fig. 3A, force against MgATP at MgADP 1 nM and 10 uM (Pi 0, pH 7, SL 2.1 um,
  20 degC): the curves cross at 0.82 mM (paper 0.9 mM); below it more MgADP
  lowers force, above it raises it.
* Fig. 3B, steady force after redevelopment at MgATP 3 and 10 mM relative to
  0.1 mM (MgADP 1 uM, Pi 2 mM, pH 7, SL 2.1 um, 20 degC): 0.518 and 0.239
  (paper 0.52 and 0.25).
* Fig. 5, maximal force through 15 min of ischemia (25 degC, SL 2.2 um), at
  the metabolite levels read off its first column: 0.851, 0.334, 0.193, 0.090
  and 0.059 at 0, 2.5, 5, 10 and 15 min (paper about 0.87, 0.34, 0.20, 0.08
  and 0.05).

**What does not:** the text's 13% rise of isometric force from 0 to 3 mM
MgADP, by which ``kdADP`` was fitted. The model, at Fig. 2A's conditions,
more than doubles (x2.28); so does the paper's own Fig. 5 (row C, T = 0),
whose force falls steeply as MgADP goes to zero. Fig. 2C (maximal force
against pH, normalized at pH 7) is lower than the paper's curve at its ends,
by 11 and 8% of the reference force (131 and 30 at pH 8 and 6, against about
142 and 38).

Examples
--------
>>> import numpy as np
>>> from crossbridge.tran2010 import Tran2010
>>>
>>> params = {**Tran2010.default_parameters(), "Pi": 10.0, "pH": 6.8}
>>> model = Tran2010(num_cells=10, params=params)
>>> model.advance_step(1e-3, Ca_val=1.0, SL_vals=np.full(10, 2.2))
>>> Ta = model.get_active_tension()  # shape (10,), normalized force times Ta_max
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt

from .rice2008 import Rice2008, _Blocks

#: The states, in the order of the linear system: `Rice2008`'s with P in place of N.
_STATES = (
    "TRPNCaH",
    "TRPNCaL",
    "N_NoXB",
    "P_NoXB",
    "P",
    "XBprer",
    "XBpostr",
    "xXBprer",
    "xXBpostr",
)


def _proton(pH: float) -> float:
    """[H+] [uM] at pH."""
    return 10.0 ** (6.0 - pH)


class Tran2010(Rice2008):
    """
    Vectorized Tran et al. (2010) myofilament model: `Rice2008` with a
    reversible, MgATP-, MgADP-, Pi- and H+-dependent crossbridge cycle.

    Attributes
    ----------
    TRPNCaH, TRPNCaL, N_NoXB, P_NoXB, XBprer, XBpostr, xXBprer, xXBpostr : ndarray
        As in `Rice2008`.
    P : ndarray, shape (num_cells,)
        Fraction of crossbridges in the permissive state; the non-permissive
        fraction ``N = 1 - P - XBprer - XBpostr`` is derived (the CellML's
        choice, the reverse of `Rice2008`'s).
    """

    ODE_FILE = Path(__file__).with_name("tran2010.ode")

    P: npt.NDArray[np.float64]

    _x_names = _STATES
    _state_names = (*_STATES, "_SL_prev", "_SL_curr", "_has_prev_step")
    #: The CellML's 310 K.
    _TMPC_REF = 36.85

    @classmethod
    def default_parameters(cls) -> dict:
        """
        Return default parameters: `Rice2008`'s with the CellML's values where
        they differ (guinea pig ``xbmodsp = 0.2``, ``TmpC`` 25.85 degC,
        ``SLrest`` 1.85 um), the metabolites and their reference levels, and
        the three parameters the paper adds (``kdADP``, ``kdHCa``, ``m``) and
        the equilibrium constant of MgATP hydrolysis. Concentrations of the
        metabolites in mM, of calcium and protons in uM.
        """
        p = super().default_parameters()
        p.update(TmpC=25.85, xbmodsp=0.2, SLrest=1.85)
        # Metabolites, and the reference levels at which the rates are Rice2008's
        p["MgATP"] = 5.0  # [mM]
        p["MgADP"] = 0.0363  # [mM]
        p["Pi"] = 2.0  # [mM]
        p["pH"] = 7.15  # [-]
        p["MgATP_ref"] = 5.0  # [mM]
        p["MgADP_ref"] = 0.036  # [mM]
        p["Pi_ref"] = 2.0  # [mM]
        p["pH_ref"] = 7.15  # [-]
        p["kdADP"] = 0.004  # [mM] MgADP dissociation constant
        p["kdHCa"] = 0.02  # [uM] H+ dissociation constant at the Ca2+ site of troponin C
        p["m"] = 1.0  # [-] H+ binding cooperativity
        p["kMgATP"] = 15400.0  # [mM^2] equilibrium constant of MgATP hydrolysis
        return p

    @classmethod
    def rabbit_parameters(cls) -> dict:
        """Not defined for Tran2010, whose species is set by ``xbmodsp`` (guinea pig)."""
        raise NotImplementedError(
            "Tran2010 is parameterized for guinea pig; see default_parameters()"
        )

    def _set_constants(self) -> None:
        super()._set_constants()
        p = self.p
        H, H_ref = _proton(p["pH"]), _proton(p["pH_ref"])
        m, kd = p["m"], p["kdHCa"]
        self._konT *= (H_ref**m + kd**m) / (H**m + kd**m)

        adp = (p["MgADP_ref"] + p["kdADP"]) / (p["MgADP"] + p["kdADP"])
        self._f_a1m = p["Pi"] / p["Pi_ref"]
        self._f_a2m = (H / H_ref) * (p["MgADP"] / p["MgADP_ref"]) * adp
        self._f_a3p = (p["MgATP"] / p["MgATP_ref"]) * adp
        # fxbT = f_a3m gxbT hfT fappT / (hbT gappT), concentrations in mM (H_ref too)
        self._f_a3m = p["kdADP"] * p["Pi_ref"] * 1e-3 * H_ref / (p["MgATP_ref"] * p["kMgATP"])

        # Force normalization (Appendix): the occupancies at the base rates and reference
        # metabolites, with the CellML's (and the Appendix's) hbT in XBprer_max.
        fapp, gapp, hf, hb, gxb = (p[k] for k in ("fapp", "gapp", "hf", "hb", "gxb"))
        fxb = self._f_a3m * gxb * hf * fapp / (hb * gapp)
        D0 = (
            fxb * hb
            + fapp * hf
            + fxb * gapp
            + fapp * hb
            + fapp * gxb
            + fxb * hb
            + gapp * gxb
            + gapp * hb
            + gxb * hf
        )
        self._XBprer_max = (fapp * self._hbT + fapp * gxb + fxb * hb) / D0
        self._XBpostr_max = (fxb * hb + fapp * hf + fxb * gapp) / D0

    def reset(self) -> None:
        """
        Reset to the quiescent state: as `Rice2008.reset`, with the permissive
        fraction P = 0 (so that N = 1).
        """
        super().reset()
        self.P = np.zeros(self.num_cells)

    @property  # type: ignore[override]
    def N(self) -> npt.NDArray[np.float64]:
        """Non-permissive fraction, ``1 - P - XBprer - XBpostr``."""
        return 1.0 - self.P - self.XBprer - self.XBpostr

    @N.setter
    def N(self, value: npt.NDArray[np.float64]) -> None:
        # Rice2008.reset assigns N; here it is derived, so that assignment is dropped.
        pass

    def _rates(
        self, x: npt.NDArray[np.float64], SL: npt.NDArray[np.float64]
    ) -> dict[str, npt.NDArray[np.float64]]:
        """
        `Rice2008`'s rates at state `x` and the cycle's metabolite-dependent ones,
        ``a1p, a1m, a2p, a2m, a3p, a3m`` (alpha1+ ... alpha3-), and the duty fractions.
        """
        r = super()._rates(x, SL)
        fappT, gappT, hfT, hbT, gxbT = (r[k] for k in ("fappT", "gappT", "hfT", "hbT", "gxbT"))
        a1p, a2p = fappT, hfT
        a1m = gappT * self._f_a1m
        a2m = hbT * self._f_a2m
        a3p = gxbT * self._f_a3p
        # fxbT = g3 hfT: alpha3- is proportional to alpha2+, which the strain coefficient uses
        g3 = self._f_a3m * gxbT * fappT / (hbT * gappT)
        a3m = g3 * hfT
        num_pre = a1p * a2m + a1p * a3p + a2m * a3m
        num_post = a2p * a3m + a1m * a3m + a1p * a2p
        S = (
            a1m * a3p
            + a1m * a3m
            + a2p * a3p
            + a1p * a2m
            + a1p * a3p
            + a2m * a3m
            + a2p * a3m
            + a1m * a3m
            + a1p * a2p
        )
        r.update(a1p=a1p, a1m=a1m, a2p=a2p, a2m=a2m, a3p=a3p, a3m=a3m, g3=g3, S=S)
        r.update(num_pre=num_pre, num_post=num_post, dutyprer=num_pre / S, dutypostr=num_post / S)
        return r

    def _blocks(
        self,
        Ca: npt.NDArray[np.float64],
        dSL: npt.NDArray[np.float64],
        rates: dict[str, npt.NDArray[np.float64]],
    ) -> _Blocks:
        """
        The frozen-rate system by block (the CellML's equations): occupancies (P, XBprer,
        XBpostr) with N = 1 - P - XBprer - XBpostr, and the strains of the Appendix.
        """
        p = self.p
        kn, kp = rates["kn_pT"], rates["kp_nT"]
        a1p, a1m, a2p, a2m, a3p, a3m = (
            rates[k] for k in ("a1p", "a1m", "a2p", "a2m", "a3p", "a3m")
        )
        x_0, xPsi = p["x_0"], p["xPsi"]
        konCa = self._konT * Ca
        n = len(kn)

        # dP/dt = kn_pT N - (alpha1+ + kp_nT) P
        occ = np.zeros((n, 3, 3))
        occ[:, 0, 0] = -kn - a1p - kp
        occ[:, 0, 1] = occ[:, 0, 2] = -kn
        # dXBprer/dt = alpha1+ P - (alpha1- + alpha2+) XBprer + alpha2- XBpostr
        occ[:, 1, 0] = a1p
        occ[:, 1, 1] = -a1m - a2p
        occ[:, 1, 2] = a2m
        # dXBpostr/dt = alpha3- P + alpha2+ XBprer - (alpha2- + alpha3+) XBpostr
        occ[:, 2, 0] = a3m
        occ[:, 2, 1] = a2p
        occ[:, 2, 2] = -a2m - a3p
        zero = np.zeros(n)
        occ_c = np.stack([kn, zero, zero], axis=-1)
        # The steady state (-A)^-1 c by cofactors: each a sum of products of positive rates.
        C_P = a1m * a2m + a1m * a3p + a2p * a3p
        C_pre = a1p * a2m + a1p * a3p + a2m * a3m
        C_post = a1p * a2p + a1m * a3m + a2p * a3m
        det = (kn + a1p + kp) * C_P + kn * (C_pre + C_post)
        occ_ss = np.stack([C_P, C_pre, C_post], axis=-1) * (kn / det)[:, None]

        # Strains: xPsi / dutyprer = xPsi S / num_pre, and xPsi alpha2+ / dutypostr =
        # xPsi alpha2+ S / num_post = xPsi S / (alpha1+ + alpha3- + alpha1- g3), since
        # num_post = alpha2+ (alpha1+ + alpha3- + alpha1- g3) with alpha3- = g3 alpha2+: finite
        # where alpha2+ underflows.
        S = rates["S"]
        a = xPsi * S / rates["num_pre"]
        b = xPsi * S / (a1p + a3m + a1m * rates["g3"])
        xb = np.empty((n, 2, 2))
        xb[:, 0, 0] = -a * (a1p + a2m)
        xb[:, 0, 1] = a * a2m
        xb[:, 1, 0] = b
        xb[:, 1, 1] = -b
        xb_c = np.stack([0.5 * dSL - a * a2m * x_0, 0.5 * dSL + b * x_0], axis=-1)

        return _Blocks(
            konCa=konCa,
            trpn_k=np.stack([konCa + self._koffHT, konCa + self._koffLT]),
            kn_pT=kn,
            kp_nT=kp,
            occ=occ,
            occ_c=occ_c,
            occ_ss=occ_ss,
            xb=xb,
            xb_c=xb_c,
            xb_det=a * b * a1p,
        )

    def get_atp_hydrolysis_rate(self) -> npt.NDArray[np.float64]:
        """
        Crossbridge ATP hydrolysis rate [1/s per myosin head]: the net flux of the
        MgATP step, ``alpha3+ XBpostr - alpha3- P``, times ``SOVFthick``, the fraction of
        heads in the single overlap whose states the occupancies are.
        """
        r = self._rates(self._get_x(), self._SL_curr)
        SOVFthick, _ = self._overlap(self._SL_curr)
        return SOVFthick * (r["a3p"] * self.XBpostr - r["a3m"] * self.P)
