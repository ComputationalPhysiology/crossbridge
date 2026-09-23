"""
Land et al. (2017) human cardiac contraction model.

Based on: Land, S., Park-Holohan, S.J., Smith, N.P., dos Remedios, C.G.,
Kentish, J.C., & Niederer, S.A. (2017). A model of cardiac contraction based
on novel measurements of tension development in human cardiomyocytes.
Journal of Molecular and Cellular Cardiology, 106, 68-83.
https://doi.org/10.1016/j.yjmcc.2017.03.008

This is the original human-ventricular contraction model: troponin C /
CaTRPN kinetics driving a tropomyosin-blocking state B, a three-state
crossbridge cycle (unbound U, pre-powerstroke W, post-powerstroke S) with a
distortion-decay model of crossbridge strain (Zw, Zs), a three-element
spring-dashpot passive element (Cd), and *ad hoc* phenomenological
length-dependent activation (LDA) via two parameters: beta0 shifts maximum
active tension with sarcomere length, and beta1 shifts calcium sensitivity
(pCa50) with sarcomere length.

`Lewalle2024` in this package amends this exact model by replacing beta0/
beta1 with an explicit myosin OFF-state feedback subsystem; the state
variables and equations shared between the two models (CaTRPN, B, S, W, Zs,
Zw, Cd, the passive spring-dashpot, h(lambda)/Ca50(lambda)) are intentionally
mirrored here.

**Numerical scheme.** Identical rationale and technique as `Lewalle2024`
(see that module's docstring for the full discussion of why a naive
fixed-step explicit scheme is unstable here, given the CaTRPN**(-nTm/2)
singularity as CaTRPN -> 0): `CaTRPN`, `Zw`, `Zs` are each solved exactly in
closed form for the whole step (given Ca, SL, dSL held fixed); `Cd` is
piecewise-linear with a fixed sign for the whole step and is also solved
exactly; `(B, S, W)` are coupled through a linear 3x3 system once the
CaTRPN-dependent coefficients are frozen at each sub-step's midpoint, solved
exactly per sub-step via a matrix exponential. This model has no OFF states
and no force-feedback rates, so its linear system is a 3x3 subset of
`Lewalle2024`'s 5x5. Both use `crossbridge._expm.expm_batch`, which scales
and squares each cell's matrix by its own norm -- necessary because CaTRPN,
and hence the norm, can differ by orders of magnitude across the cells of one
batch.

Examples
--------
>>> import numpy as np
>>> from crossbridge import Land2017
>>>
>>> model = Land2017(num_cells=10)
>>> dt = 1e-3
>>> Ca = np.full(10, 1.0)    # 1.0 uM calcium
>>> SL = np.full(10, 2.0)    # 2.0 um sarcomere length
>>>
>>> model.advance_step(dt, Ca, SL)
>>> Ta = model.get_active_tension()  # shape (10,), kPa
"""

import numpy as np
import numpy.typing as npt

from ._expm import expm_batch
from .base import CardiacActivationModel

#: Target sub-step size [s] used to refresh the frozen CaTRPN-dependent
#: coefficients of the (B, S, W) linear system. Sub-stepping here is an
#: accuracy knob, not a stability requirement (each sub-step is solved
#: exactly via matrix exponential).
_TARGET_SUBSTEP = 2e-4  # [s]


class Land2017(CardiacActivationModel):
    """
    Vectorized Land et al. (2017) human cardiac contraction model.

    Attributes
    ----------
    CaTRPN : ndarray, shape (num_cells,)
        Fraction of troponin C units with Ca2+ bound.
    B, S, W : ndarray, shape (num_cells,)
        Thin/thick-filament populations (blocked, strongly bound
        "post-stroke", weakly bound "pre-stroke").
    Zs, Zw : ndarray, shape (num_cells,)
        Cross-bridge distortions associated with the S and W states.
    Cd : ndarray, shape (num_cells,)
        Dashpot strain of the passive spring-dashpot element.

    `U = 1 - B - S - W` (thin-filament-unblocked) is derived, not
    integrated, mirroring the conservation constraint in the paper.

    Sarcomere length enters only algebraically (`Lambda = SL / SL0`, held
    fixed over each `advance_step` sub-interval) -- there is no separate
    "Lambda" ODE state, since length is imposed by the caller exactly like
    the other models in this package.
    """

    def __init__(self, num_cells: int, Ta_max: float = 1.0, params: dict | None = None):
        """
        Initialize the Land2017 model.

        Parameters
        ----------
        num_cells : int
            Number of independent cells / FEM integration points.
        Ta_max : float, optional
            Unused. This model computes active tension intrinsically from
            `Tref` (like RDQ20MF's `a_XB` and Lewalle2024's `Tref`). Kept
            for API compatibility with `CardiacActivationModel`. Default
            is 1.0.
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

        # Time-invariant rate constants (Land 2017 eqs. 23-28 / 59-63),
        # computed once. kb uses the fixed calibration constant TRPN50, not
        # the dynamic CaTRPN state -- Eq. 25/61 as literally printed reads
        # CaTRPN there, but this is a documented typo in the paper (the
        # steady-state derivation of kb, TRPN50 by definition is the CaTRPN
        # value at which B=0.5, so using dynamic CaTRPN would make kb
        # circular/state-dependent in a way inconsistent with the rest of
        # the derivation); the reference OFF-state extension of this model
        # (Lewalle et al. 2024) also uses the fixed TRPN50 constant.
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

        Ca0 = max(p["Ca0"], 0.0)
        CaTRPN0 = 1.0 / (1.0 + (p["ca50_ref"] / max(Ca0, 1e-6)) ** p["ntrpn"])

        self.CaTRPN = np.full(n, CaTRPN0)
        self.B = np.zeros(n)
        self.S = np.zeros(n)
        self.W = np.zeros(n)
        self.Zs = np.zeros(n)
        self.Zw = np.zeros(n)
        self.Cd = np.zeros(n)

        self._Lambda_prev = np.ones(n)
        self._Lambda_curr = np.ones(n)
        self._has_prev_step = False

        self._prev_bound_ca = np.zeros(n)
        self._last_dt = 0.0

    @classmethod
    def default_parameters(cls) -> dict:
        """
        Return default parameters, taken from Table B ("Skinned model
        value") of Land et al. (2017). Rate constants given in the paper as
        ms^-1 are converted to this package's s^-1 convention (x1000);
        [Ca2+] parameters are in uM, matching this package's convention for
        Ca_val elsewhere (RDQ18, RDQ20MF).

        The paper also reports a "Whole organ model value" column for
        `ca50_ref` (0.805 uM), `nTm` (5), `kuw` (0.182/ms), `kws`
        (0.012/ms), and `Tref` (120 kPa), used in Sec. 3.5-3.6 to represent
        intact rather than skinned myocytes; pass these as `params` to
        reproduce that calibration (see `demo/reproduce_figures_land2017.py`).
        """
        p: dict = {}
        # Numerical: a *suggested* coupling dt. The ODE integration itself
        # sub-steps internally regardless of this value (see module docstring).
        p["dt"] = 1e-3  # [s]

        # Kinematics. SL0 is not given as an explicit table parameter in
        # the paper; 1.8 um is used here for consistency with Lewalle et al.
        # (2024)'s OFF-state extension of this exact model, which inherited
        # it unchanged from the original Land 2017 code.
        p["SL0"] = 1.8  # [um]

        # Passive tension (spring-dashpot)
        p["a"] = 2100.0  # [Pa] (2.1 kPa)
        p["b"] = 9.1  # [-]
        p["k"] = 7.0  # [-]
        p["eta_l"] = 0.2  # [s] (200 ms)
        p["eta_s"] = 0.02  # [s] (20 ms)

        # Troponin / Ca2+ binding
        p["k_trpn"] = 100.0  # [s^-1] (0.1/ms)
        p["ntrpn"] = 2.0  # [-] (n_TRPN)
        p["ca50_ref"] = 2.5  # [uM] ([Ca2+]_T50^ref, skinned)

        # Thick/thin-filament cycling
        p["ku"] = 1000.0  # [s^-1] (1/ms)
        p["nTm"] = 2.2  # [-] (n_Tm, skinned)
        p["trpn50"] = 0.35  # [-]
        p["kuw"] = 26.0  # [s^-1] (0.026/ms, skinned)
        p["kws"] = 4.0  # [s^-1] (0.004/ms, skinned)
        p["rw"] = 0.5  # [-]
        p["rs"] = 0.25  # [-]
        p["gs"] = 8.5  # [s^-1 per unit distortion] (0.0085/ms)
        p["gw"] = 615.0  # [s^-1 per unit distortion] (0.615/ms)
        p["phi"] = 2.23  # [-]
        p["Aeff"] = 25.0  # [-]

        # Ad hoc length-dependent activation (this model's defining
        # feature -- Lewalle2024 replaces these with OFF-state feedback)
        p["beta0"] = 2.3  # [-] max-force LDA gradient
        p["beta1"] = -2.4  # [uM] calcium-sensitivity LDA gradient
        p["Tref"] = 40500.0  # [Pa] (40.5 kPa, skinned)

        # Initialization helper (not a physical model parameter): diastolic
        # Ca used only to set a numerically-safe CaTRPN(0) (see reset()).
        p["Ca0"] = 0.1  # [uM]

        return p

    # ------------------------------------------------------------------
    # Force / length-dependence helpers
    # ------------------------------------------------------------------

    def _h(self, Lambda: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Length-dependent activation factor h(Lambda), Land 2017 Eq. (20)-(21).

        Shared by :meth:`_forces` and :meth:`get_active_stiffness` so the two
        cannot drift apart -- both are proportional to the same h.
        """
        p = self.p
        Lambda_clamped = np.minimum(Lambda, 1.2)
        return np.maximum(
            0.0, 1.0 + p["beta0"] * (Lambda_clamped + np.minimum(Lambda_clamped, 0.87) - 1.87)
        )

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
        Ta_active = self._h(Lambda) * p["Tref"] / p["rs"] * (S * (Zs + 1.0) + W * Zw)
        return Ta_active, Tp, Ta_active + Tp

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
            convention used by the other models in this package.
        Ca_val : float or np.ndarray, shape (num_cells,)
            Intracellular calcium concentration [uM].
        SL_vals : float or np.ndarray, shape (num_cells,)
            Current sarcomere lengths [um].
        dSL_vals : float or np.ndarray, shape (num_cells,) or None
            Sarcomere length rate of change [um/s]. If None, estimated from
            the sarcomere length recorded on the previous call.
        """
        self._begin_step(dt)

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
        Ca_safe = np.maximum(Ca_arr, 0.0)
        Ca50 = np.maximum(p["ca50_ref"] + p["beta1"] * (np.minimum(Lambda, 1.2) - 1.0), 1e-6)
        kon = p["k_trpn"] * (Ca_safe / Ca50) ** p["ntrpn"]
        koff = p["k_trpn"]
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

        # --- (B, S, W): linear once CaTRPN-dependent coefficients are
        # frozen per sub-step; each sub-step solved exactly via a batched
        # (per-cell) matrix exponential. ---
        n_sub = max(1, int(np.ceil(dt / _TARGET_SUBSTEP)))
        h = dt / n_sub

        B, S, W = self.B, self.S, self.W
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
                Zs_mid < -1.0,
                -p["gs"] * (Zs_mid + 1.0),
                np.where(Zs_mid > 0.0, p["gs"] * Zs_mid, 0.0),
            )

            M = np.zeros((n, 3, 3))
            c = np.zeros((n, 3))
            kb_neg = self._kb * ca_pow_neg
            ku_pos = p["ku"] * ca_pow_pos

            # Row 0: dB/dt = kb_neg*(1-B-S-W) - ku_pos*B
            M[:, 0, 0] = -kb_neg - ku_pos
            M[:, 0, 1] = -kb_neg
            M[:, 0, 2] = -kb_neg
            c[:, 0] = kb_neg
            # Row 1: dS/dt = kws*W - (ksu+gsu)*S
            M[:, 1, 1] = -self._ksu - gsu
            M[:, 1, 2] = p["kws"]
            # Row 2: dW/dt = kuw*(1-B-S-W) - (kwu+kws+gwu)*W
            M[:, 2, 0] = -p["kuw"]
            M[:, 2, 1] = -p["kuw"]
            M[:, 2, 2] = -p["kuw"] - self._kwu - p["kws"] - gwu
            c[:, 2] = p["kuw"]

            x0 = np.stack([B, S, W], axis=-1)  # (n, 3)
            try:
                x_ss = -np.linalg.solve(M, c[:, :, np.newaxis])[:, :, 0]
            except np.linalg.LinAlgError:
                x_ss = x0.copy()
            delta = x0 - x_ss
            # NOTE: as in Lewalle2024, this stack mixes matrices whose
            # norms differ by orders of magnitude, since CaTRPN**(-nTm/2)
            # does across cells with different Ca. `expm_batch` scales and
            # squares each matrix by its own norm, so batching changes no
            # cell's result; a batched exponential that shares one exponent
            # across the stack does, which is why this was once a per-cell
            # scipy.linalg.expm loop.
            expM = expm_batch(M, h)
            x_new = x_ss + np.einsum("nij,nj->ni", expM, delta)
            x_new = np.clip(x_new, 0.0, 1.0)

            B, S, W = (x_new[:, i] for i in range(3))
            t0 += h

        self.CaTRPN = np.maximum(CaTRPN_at(dt), 0.0)
        self.B, self.S, self.W = B, S, W
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

        The reference formula is in Pa; the result is converted to kPa here
        for consistency with the rest of this package.
        """
        Ta_active, _, _ = self._forces(self._Lambda_curr, self.Cd, self.S, self.W, self.Zs, self.Zw)
        return Ta_active / 1000.0

    def get_active_stiffness(self) -> npt.NDArray[np.float64]:
        r"""
        Compute active stiffness (kPa per unit Lambda) from the current state.

        .. math::
            K_a = h(\Lambda) \frac{T_{ref}}{r_s} \left(A_s S + A_w W\right)

        This is Eq. (50) of Regazzoni & Quarteroni (2020) for the L17 model.
        It follows from the general definition
        :math:`K_a = \nabla_r g \cdot \partial h / \partial \dot\lambda` (see
        :meth:`~crossbridge.base.CardiacActivationModel.get_active_stiffness`)
        because ``Zs`` and ``Zw`` are the only states whose right-hand sides
        carry an explicit :math:`\dot\Lambda`,

        .. math::
            \dot{Z_s} = A_s \dot\Lambda - c_s Z_s, \qquad
            \dot{Z_w} = A_w \dot\Lambda - c_w Z_w,

        while :math:`\partial T_a/\partial Z_s = h T_{ref} S / r_s` and
        :math:`\partial T_a/\partial Z_w = h T_{ref} W / r_s`.

        As with :meth:`get_active_tension`, the reference formula is in Pa and
        the result is converted to kPa. ``Ka >= 0`` always, since ``h``, ``S``
        and ``W`` are all non-negative and ``As = Aw > 0``.
        """
        Ka = (
            self._h(self._Lambda_curr)
            * self.p["Tref"]
            / self.p["rs"]
            * (self._As * self.S + self._Aw * self.W)
        )
        return Ka / 1000.0

    def bound_calcium_fraction(self) -> npt.NDArray[np.float64]:
        """
        Troponin-C occupancy, i.e. the state variable ``CaTRPN`` itself.

        This is the same quantity ToR-ORd integrates as ``CaTrpn``, so
        multiplying :meth:`get_calcium_binding_rate` by that model's
        ``trpnmax`` recovers its ``J_TRPN`` directly.
        """
        return self.CaTRPN

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
