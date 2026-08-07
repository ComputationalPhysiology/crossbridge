"""
Lewalle 2024 myosin OFF-state force-feedback model

Based on: Lewalle, A., Milburn, G., Campbell, K. S., & Niederer, S. A. (2024).
Cardiac length-dependent activation driven by force-dependent thick-filament
dynamics. Biophysical Journal, 123(18), 2996-3009.
https://doi.org/10.1016/j.bpj.2024.07.021

This model amends the Land et al. (2017) human ventricular contraction model
(J Mol Cell Cardiol 106:68-83) by replacing its ad hoc length-dependent
activation (LDA) terms (the beta0/beta1 parameters) with two additional
myosin "OFF" states, Boff and Uoff, mirroring the thin-filament-blocked (B)
and thin-filament-unblocked (U) states but with the myosin head parked in
the OFF (super-relaxed) configuration. The OFF<->ON transition rates k1, k2
are modulated by sarcomere force, producing a mechanosensitive feedback loop
that can by itself reproduce the Frank-Starling length dependence of active
tension, without any ad hoc SL-dependent terms.

The paper tests four feedback paradigms (its Eqs. 8-11), selected here via
`params["which_dep"]`:

- "totalforce": paradigm A, feedback on total force (active + passive). The
  *only* paradigm the paper finds able to reproduce the target length
  dependence -- this is the default.
- "force": paradigm B, feedback on active force only.
- "passiveforce": paradigm D, feedback on passive force only.
- "Lambda": paradigm C, feedback on sarcomere strain.

**Numerical scheme.** Unlike the RU/XB tensor models in this package (RDQ18,
RDQ20MF), this model's troponin-binding term scales as CaTRPN**(-nTm/2),
which diverges as CaTRPN -> 0, making a naive fixed-step explicit scheme
unstable. Rather than call an adaptive black-box stiff solver per step
(prohibitively slow when vectorized over many cells -- dense numerical
Jacobians scale with `9 * num_cells`), each `advance_step` call is
integrated using the fact that, once `Ca`, `SL`, `dSL` are held fixed over
the sub-interval (as they are throughout this package), the system
decomposes into pieces that are each either exactly linear or become linear
once the (slowly varying) force feedback is frozen per sub-step:

- `CaTRPN`, `Zw`, `Zs` are each *exactly* linear ODEs (given fixed Ca and
  dLambda/dt) and are solved in closed form for the whole step.
- `Cd` is piecewise-linear with a fixed sign for the whole step (it relaxes
  monotonically towards `Lambda - 1`) and is also solved exactly.
- `(B, S, W, Boff, Uoff)` are coupled through a *linear* system once the
  CaTRPN-dependent coefficients and the force-feedback rates k1/k2 are
  frozen at each sub-step's midpoint; this 5x5 system is solved exactly per
  sub-step via a matrix exponential (`scipy.linalg.expm`), the same
  technique RDQ20MF uses for its crossbridge sub-system. A handful of
  sub-steps (accuracy, not stability, is the only reason for more than one)
  keep the frozen coefficients tracking the true trajectory.

This makes every sub-step unconditionally stable (no eigenvalue-driven step
size restriction). The linear solve for the 5x5 system's steady state is
batched across all cells (`numpy.linalg.solve` on a stacked array), but the
matrix exponential itself is computed with a per-cell loop: cells can differ
in CaTRPN by orders of magnitude (e.g. a resting vs. an activated cell in
the same batch), and `scipy.linalg.expm`'s batched mode was found to return
silently incorrect results for some matrices when a stack mixes very
different norms. Each 5x5 exponential is cheap, so this is not the model's
performance bottleneck -- correctness took priority over avoiding this one
small loop.

Examples
--------
>>> import numpy as np
>>> from crossbridge import Lewalle2024
>>>
>>> model = Lewalle2024(num_cells=10)
>>> dt = 1e-3
>>> Ca = np.full(10, 1.0)    # 1.0 uM calcium
>>> SL = np.full(10, 1.9)    # 1.9 um sarcomere length
>>>
>>> model.advance_step(dt, Ca, SL)
>>> Ta = model.get_active_tension()  # shape (10,), kPa
"""

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm

from .base import CardiacActivationModel

_WHICH_DEP_CHOICES = ("totalforce", "force", "passiveforce", "Lambda")
_DEP_K1_OR_K2_CHOICES = ("k1", "k2")

#: Target sub-step size [s] used to refresh the frozen (CaTRPN-dependent and
#: force-feedback) coefficients of the (B, S, W, Boff, Uoff) linear system.
#: Sub-stepping here is an accuracy knob, not a stability requirement (each
#: sub-step is solved exactly via matrix exponential).
_TARGET_SUBSTEP = 2e-4  # [s]


class Lewalle2024(CardiacActivationModel):
    """
    Vectorized Land2017 + myosin OFF-state force-feedback model.

    Attributes
    ----------
    CaTRPN : ndarray, shape (num_cells,)
        Fraction of troponin C units with Ca2+ bound.
    B, S, W : ndarray, shape (num_cells,)
        Thin/thick-filament ON-state populations (blocked, strongly bound
        "post-stroke", weakly bound "pre-stroke").
    BE, UE : ndarray, shape (num_cells,)
        Mirror populations with the myosin head in the OFF state
        (blocked-OFF, unblocked-OFF).
    Zs, Zw : ndarray, shape (num_cells,)
        Cross-bridge distortions associated with the S and W states.
    Cd : ndarray, shape (num_cells,)
        Dashpot strain of the passive spring-dashpot element.

    `U = 1 - B - S - W - BE - UE` (thin-filament-unblocked, myosin-ON) is
    derived, not integrated, mirroring the conservation constraint in the
    reference model.

    Sarcomere length enters only algebraically (`Lambda = SL / SL0`, held
    fixed over each `advance_step` sub-interval) -- there is no separate
    "Lambda" ODE state, since length is imposed by the caller exactly like
    the other models in this package.
    """

    def __init__(self, num_cells: int, Ta_max: float = 1.0, params: dict | None = None):
        """
        Initialize the Lewalle2024 model.

        Parameters
        ----------
        num_cells : int
            Number of independent cells / FEM integration points.
        Ta_max : float, optional
            Unused. This model computes active tension intrinsically from
            `Tref` (like RDQ20MF's `a_XB`). Kept for API compatibility with
            `CardiacActivationModel`. Default is 1.0.
        params : dict, optional
            Parameter overrides. Keys should match those returned by
            `default_parameters()`, including the string-valued feedback
            configuration keys `"which_dep"` and `"dep_k1ork2"`.
        """
        super().__init__(int(num_cells), Ta_max, params)

        self.p = type(self).default_parameters()
        if params:
            self.p.update(params)

        p = self.p
        if p["which_dep"] not in _WHICH_DEP_CHOICES:
            raise ValueError(
                f"params['which_dep']={p['which_dep']!r} not supported; "
                f"choose one of {_WHICH_DEP_CHOICES} (the paper's paradigms A-D)."
            )
        if p["dep_k1ork2"] not in _DEP_K1_OR_K2_CHOICES:
            raise ValueError(
                f"params['dep_k1ork2']={p['dep_k1ork2']!r} must be one of {_DEP_K1_OR_K2_CHOICES}."
            )
        if p["koffon"] is None:
            raise ValueError(
                "params['koffon'] must be set explicitly when overriding 'which_dep' "
                "away from the calibrated default ('totalforce')."
            )

        self.dt = p["dt"]

        # Time-invariant rate constants (Land 2017 eqs. 23-28), computed once.
        self._kb = p["ku"] * p["trpn50"] ** p["nTm"] / (1 - p["rs"] - (1 - p["rs"]) * p["rw"])
        self._kwu = p["kuw"] * (1 / p["rw"] - 1) - p["kws"]
        self._ksu = p["kws"] * p["rw"] * (1 / p["rs"] - 1)
        self._Aw = p["Aeff"] * p["rs"] / ((1 - p["rs"]) * p["rw"] + p["rs"])
        self._As = self._Aw
        self._cw = p["phi"] * p["kuw"] * ((1 - p["rs"]) * (1 - p["rw"])) / ((1 - p["rs"]) * p["rw"])
        self._cs = p["phi"] * p["kws"] * ((1 - p["rs"]) * p["rw"]) / p["rs"]

        self.reset()

    def reset(self) -> None:
        """
        Reset model state: all populations to zero except CaTRPN, which is
        initialized at its own steady state for the diastolic calcium level
        `params["Ca0"]`. This avoids starting exactly at CaTRPN=0, where the
        CaTRPN**(-nTm/2) term in dB/dt is singular.
        """
        p = self.p
        n = self.num_cells

        Ca0_M = max(p["Ca0"], 1e-6) * 1e-6
        Ca50_0_M = 10.0 ** -p["pCa50ref"]  # beta1 term vanishes at Lambda=1
        CaTRPN0 = 1.0 / (
            1.0 + (p["k_trpn_off"] / p["k_trpn_on"]) * (Ca50_0_M / Ca0_M) ** p["ntrpn"]
        )

        self.CaTRPN = np.full(n, CaTRPN0)
        self.B = np.zeros(n)
        self.S = np.zeros(n)
        self.W = np.zeros(n)
        self.Zs = np.zeros(n)
        self.Zw = np.zeros(n)
        self.Cd = np.zeros(n)
        self.BE = np.zeros(n)
        self.UE = np.zeros(n)

        self._Lambda_prev = np.ones(n)
        self._Lambda_curr = np.ones(n)
        self._has_prev_step = False

    @classmethod
    def default_parameters(cls) -> dict:
        """
        Return default parameters, taken from the reference implementation
        (github.com/AlexLewalle/OFFstateFfeedback), i.e. the Land et al.
        (2017) human-ventricular calibration amended with OFF-state
        parameters k1, k2, koffon calibrated for `which_dep="totalforce"`.
        """
        p: dict = {}
        # Numerical: a *suggested* coupling dt. The ODE integration itself
        # sub-steps internally regardless of this value (see module docstring).
        p["dt"] = 1e-3  # [s]

        # Kinematics
        p["SL0"] = 1.8  # [um] resting sarcomere length

        # Passive tension (spring-dashpot)
        p["a"] = 241.0  # [Pa]
        p["b"] = 9.1  # [-]
        p["k"] = 8.86  # [-]
        p["eta_l"] = 0.2  # [s]
        p["eta_s"] = 20e-3  # [s]

        # Troponin / Ca2+ binding
        p["k_trpn_on"] = 0.1e3  # [s^-1]
        p["k_trpn_off"] = 0.1e3  # [s^-1]
        p["ntrpn"] = 2.58  # [-]
        p["pCa50ref"] = 5.25  # [-]
        p["beta1"] = 0.0  # ad hoc Ca50 LDA discarded; OFF-state replaces it

        # Thick/thin-filament cycling (Land 2017)
        p["ku"] = 1000.0  # [s^-1]
        p["nTm"] = 2.2  # [-]
        p["trpn50"] = 0.35  # [-]
        p["kuw"] = 4.98  # [s^-1]
        p["kws"] = 19.10  # [s^-1]
        p["rw"] = 0.5  # [-]
        p["rs"] = 0.25  # [-]
        p["gs"] = 42.1  # [s^-1]
        p["es"] = 1.0  # [-] asymmetry threshold in gsu
        p["gw"] = 28.3  # [s^-1]
        p["phi"] = 0.1498  # [-]
        p["Aeff"] = 125.0  # [-]
        p["beta0"] = 0.0  # ad hoc tension-magnitude LDA discarded
        p["Tref"] = 23.0e3  # [Pa]

        # OFF-state force feedback (Lewalle et al. 2024)
        p["k1"] = 0.877  # [s^-1]
        p["k2"] = 12.6  # [s^-1]
        p["ra"] = 1.0  # [-] residual-rate multiplier (e.g. for mavacamten)
        p["rb"] = 1.0  # [-] residual-rate multiplier (e.g. for mavacamten)
        p["koffon"] = 0.00144064  # [Pa^-1], calibrated for which_dep="totalforce"
        p["which_dep"] = "totalforce"  # paper's paradigm A (the only one validated)
        p["dep_k1ork2"] = "k1"

        # Initialization helper (not a physical model parameter): diastolic
        # Ca used only to set a numerically-safe CaTRPN(0) (see reset()).
        p["Ca0"] = 0.1  # [uM]

        return p

    # ------------------------------------------------------------------
    # Force / feedback helpers
    # ------------------------------------------------------------------

    def _forces(
        self,
        Lambda: npt.NDArray[np.float64],
        Cd: npt.NDArray[np.float64],
        S: npt.NDArray[np.float64],
        W: npt.NDArray[np.float64],
        Zs: npt.NDArray[np.float64],
        Zw: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return (Ta_active, Tp, Ttotal) in Pa for the given state."""
        p = self.p
        F1 = p["a"] * (np.exp(p["b"] * (Lambda - 1.0)) - 1.0)
        F2 = p["a"] * p["k"] * ((Lambda - 1.0) - Cd)
        Tp = F1 + F2
        Lambda_clamped = np.minimum(Lambda, 1.2)
        h = np.maximum(
            0.0, 1.0 + p["beta0"] * (Lambda_clamped + np.minimum(Lambda_clamped, 0.87) - 1.87)
        )
        Ta_active = h * p["Tref"] / p["rs"] * (S * (Zs + 1.0) + W * Zw)
        return Ta_active, Tp, Ta_active + Tp

    def _feedback_rates(
        self,
        Lambda: npt.NDArray[np.float64],
        Ta_active: npt.NDArray[np.float64],
        Tp: npt.NDArray[np.float64],
        Ttotal: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return (k1_fb, k2_fb) [s^-1], each shape (num_cells,)."""
        p = self.p
        n = self.num_cells
        if p["which_dep"] == "force":
            fb = np.maximum(Ta_active, 0.0)
        elif p["which_dep"] == "totalforce":
            fb = np.maximum(Ttotal, 0.0)
        elif p["which_dep"] == "passiveforce":
            fb = np.maximum(Tp, 0.0)
        else:  # "Lambda"
            fb = Lambda

        if p["dep_k1ork2"] == "k1":
            k1_fb = p["k1"] * (p["rb"] + p["koffon"] * fb)
            k2_fb = np.full(n, p["k2"] * p["ra"])
        else:
            denom = p["rb"] + p["koffon"] * fb
            k2_fb = np.divide(p["k2"], denom, out=np.full(n, np.inf), where=denom != 0)
            k1_fb = np.full(n, p["k1"] * p["ra"])
        return k1_fb, k2_fb

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
            Time step [s]. Ca_val, SL_vals, dSL_vals are held fixed
            (zero-order hold) across the sub-interval, matching the
            convention used by RDQ18/RDQ20MF.
        Ca_val : float or np.ndarray, shape (num_cells,)
            Intracellular calcium concentration [uM].
        SL_vals : float or np.ndarray, shape (num_cells,)
            Current sarcomere lengths [um].
        dSL_vals : float or np.ndarray, shape (num_cells,) or None
            Sarcomere length rate of change [um/s]. If None, estimated from
            the sarcomere length recorded on the previous call.
        """
        n = self.num_cells
        p = self.p

        if np.isscalar(SL_vals):
            SL_arr = np.full(n, float(SL_vals))
        else:
            SL_arr = np.asarray(SL_vals, dtype=float)
        assert SL_arr.shape == (n,), f"SL_vals shape {SL_arr.shape} must be ({n},)"

        if np.isscalar(Ca_val):
            Ca_arr = np.full(n, float(Ca_val))
        else:
            Ca_arr = np.asarray(Ca_val, dtype=float)

        Lambda = SL_arr / p["SL0"]

        if dSL_vals is not None:
            if np.isscalar(dSL_vals):
                dSL_arr = np.full(n, float(dSL_vals))
            else:
                dSL_arr = np.asarray(dSL_vals, dtype=float)
            dLambdadt = dSL_arr / p["SL0"]
        elif self._has_prev_step:
            dLambdadt = (Lambda - self._Lambda_prev) / dt
        else:
            # No prior call to estimate a velocity from, and no dSL_vals
            # given: assume zero rather than finite-differencing against
            # the arbitrary post-reset _Lambda_prev (which would otherwise
            # inject a spurious velocity "kick" on the very first call
            # whenever the caller's first SL differs from SL0).
            dLambdadt = np.zeros(n)

        # --- CaTRPN: exact solution of a linear ODE (Ca held fixed over dt) ---
        Ca_safe = np.maximum(Ca_arr, 1e-6)  # [uM] floor avoids division by 0
        pCa = -np.log10(Ca_safe * 1e-6)
        Ca50 = 10.0 ** -p["pCa50ref"] + p["beta1"] * (np.minimum(Lambda, 1.2) - 1.0)
        pCa50 = -np.log10(np.maximum(Ca50, 1e-12))
        kon = p["k_trpn_on"] * (10.0**-pCa / 10.0**-pCa50) ** p["ntrpn"]
        koff = p["k_trpn_off"]
        CaTRPN_ss = kon / (kon + koff)
        CaTRPN_rate = kon + koff

        def CaTRPN_at(t: npt.NDArray[np.float64] | float) -> npt.NDArray[np.float64]:
            return CaTRPN_ss + (self.CaTRPN - CaTRPN_ss) * np.exp(-CaTRPN_rate * t)

        # --- Zw, Zs: exact solutions of linear ODEs decoupled from everything
        # else (only driven by the externally imposed dLambdadt) ---
        def _relax(x0, rate, target, t):
            safe_rate = np.where(rate > 0, rate, 1.0)
            steady = np.where(rate > 0, target / safe_rate, x0 + target * t)
            return np.where(
                rate > 0, steady + (x0 - steady) * np.exp(-safe_rate * t), x0 + target * t
            )

        def Zw_at(t):
            return _relax(self.Zw, np.full(n, self._cw), self._Aw * dLambdadt, t)

        def Zs_at(t):
            return _relax(self.Zs, np.full(n, self._cs), self._As * dLambdadt, t)

        Zw_final = Zw_at(dt)
        Zs_final = Zs_at(dt)

        # --- Cd: piecewise-linear, but the sign of (Lambda - 1 - Cd) cannot
        # flip within the step (Cd relaxes monotonically towards Lambda - 1),
        # so a single regime, fixed from the start-of-step sign, is exact. ---
        target_Cd = Lambda - 1.0
        rate_Cd = np.where((target_Cd - self.Cd) > 0.0, p["k"] / p["eta_l"], p["k"] / p["eta_s"])
        Cd_final = target_Cd + (self.Cd - target_Cd) * np.exp(-rate_Cd * dt)

        # --- (B, S, W, Boff, Uoff): linear once CaTRPN-dependent and
        # force-feedback coefficients are frozen per sub-step; each sub-step
        # solved exactly via a batched matrix exponential. ---
        n_sub = max(1, int(np.ceil(dt / _TARGET_SUBSTEP)))
        h = dt / n_sub

        B, S, W, BE, UE = self.B, self.S, self.W, self.BE, self.UE
        t0 = 0.0
        for _ in range(n_sub):
            t_mid = t0 + 0.5 * h
            CaTRPN_mid = np.maximum(CaTRPN_at(t_mid), 1e-12)
            ca_pow_pos = CaTRPN_mid ** (p["nTm"] / 2.0)
            ca_pow_neg = CaTRPN_mid ** (-p["nTm"] / 2.0)

            Zw_mid = Zw_at(t_mid)
            Zs_mid = Zs_at(t_mid)
            gwu = p["gw"] * np.abs(Zw_mid)
            gsu = np.where(
                Zs_mid < -p["es"],
                -p["gs"] * (Zs_mid + p["es"]),
                np.where(Zs_mid > 0.0, p["gs"] * Zs_mid, 0.0),
            )

            Ta_active, Tp, Ttotal = self._forces(Lambda, self.Cd, S, W, Zs_mid, Zw_mid)
            k1_fb, k2_fb = self._feedback_rates(Lambda, Ta_active, Tp, Ttotal)

            M = np.zeros((n, 5, 5))
            c = np.zeros((n, 5))
            kb_neg = self._kb * ca_pow_neg
            ku_pos = p["ku"] * ca_pow_pos

            # Row 0: dB/dt
            M[:, 0, 0] = -kb_neg - ku_pos - k2_fb
            M[:, 0, 1] = -kb_neg
            M[:, 0, 2] = -kb_neg
            M[:, 0, 3] = -kb_neg + k1_fb
            M[:, 0, 4] = -kb_neg
            c[:, 0] = kb_neg
            # Row 1: dS/dt
            M[:, 1, 1] = -self._ksu - gsu
            M[:, 1, 2] = p["kws"]
            # Row 2: dW/dt
            M[:, 2, 0] = -p["kuw"]
            M[:, 2, 1] = -p["kuw"]
            M[:, 2, 2] = -p["kuw"] - self._kwu - p["kws"] - gwu
            M[:, 2, 3] = -p["kuw"]
            M[:, 2, 4] = -p["kuw"]
            c[:, 2] = p["kuw"]
            # Row 3: dBE/dt
            M[:, 3, 0] = k2_fb
            M[:, 3, 3] = -ku_pos - k1_fb
            M[:, 3, 4] = kb_neg
            # Row 4: dUE/dt
            M[:, 4, 0] = -k2_fb
            M[:, 4, 1] = -k2_fb
            M[:, 4, 2] = -k2_fb
            M[:, 4, 3] = ku_pos - k2_fb
            M[:, 4, 4] = -kb_neg - k2_fb - k1_fb
            c[:, 4] = k2_fb

            x0 = np.stack([B, S, W, BE, UE], axis=-1)  # (n, 5)
            try:
                x_ss = -np.linalg.solve(M, c[:, :, np.newaxis])[:, :, 0]
            except np.linalg.LinAlgError:
                x_ss = x0.copy()
            delta = x0 - x_ss
            # NOTE: scipy.linalg.expm's batched (stacked) mode silently
            # returns wrong results for some entries when the matrices in
            # the stack have very different norms (verified empirically) --
            # which happens here, since CaTRPN**(-nTm/2) can differ by
            # orders of magnitude across cells with different Ca. So this
            # is looped per cell rather than batched; each 5x5 exponential
            # is cheap and this is not the model's performance bottleneck.
            expM = np.stack([expm(M[i] * h) for i in range(n)])
            x_new = x_ss + np.einsum("nij,nj->ni", expM, delta)
            x_new = np.clip(x_new, 0.0, 1.0)

            B, S, W, BE, UE = (x_new[:, i] for i in range(5))
            t0 += h

        self.CaTRPN = np.maximum(CaTRPN_at(dt), 0.0)
        self.B, self.S, self.W, self.BE, self.UE = B, S, W, BE, UE
        self.Zw = Zw_final
        self.Zs = Zs_final
        self.Cd = Cd_final

        self._Lambda_prev = Lambda
        self._Lambda_curr = Lambda
        self._has_prev_step = True

    def get_active_tension(self) -> npt.NDArray[np.float64]:
        """
        Compute active tension (kPa) from the current state.

        Ta_active = h(Lambda) * Tref / rs * (S * (Zs + 1) + W * Zw)

        The reference implementation computes this in Pa; the result is
        converted to kPa here for consistency with the rest of this package.
        """
        Ta_active, _, _ = self._forces(self._Lambda_curr, self.Cd, self.S, self.W, self.Zs, self.Zw)
        return Ta_active / 1000.0

    def get_passive_tension(self) -> npt.NDArray[np.float64]:
        """Compute passive (spring-dashpot) tension (kPa) from the current state."""
        _, Tp, _ = self._forces(self._Lambda_curr, self.Cd, self.S, self.W, self.Zs, self.Zw)
        return Tp / 1000.0

    def get_total_tension(self) -> npt.NDArray[np.float64]:
        """Compute total (active + passive) tension (kPa)."""
        _, _, Ttotal = self._forces(self._Lambda_curr, self.Cd, self.S, self.W, self.Zs, self.Zw)
        return Ttotal / 1000.0

    def compute_attached_fraction(self) -> npt.NDArray[np.float64]:
        """
        Fraction of crossbridges in force-generating states (S + W).

        This model's analog of `compute_permissivity()` on the RDQ models.
        """
        return self.S + self.W
