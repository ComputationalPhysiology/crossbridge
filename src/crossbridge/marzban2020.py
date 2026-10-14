"""
Marzban, Lopez & Beard (2020) rat cardiac myofilament model.

Based on: Marzban, B., Lopez, R., & Beard, D.A. (2020). Computational modeling
of coupled energetics and mechanics in the rat ventricular myocardium.
Physiome. https://doi.org/10.36903/physiome.12964970

the curated form of the cardiomyocyte model of: Lopez, R., Marzban, B., Gao,
X., et al., & Beard, D.A. (2020). Impaired myocardial energetics causes
mechanical dysfunction in decompensated failing hearts. Function, 1(2),
zqaa018. https://doi.org/10.1093/function/zqaa018

It extends `Tewari2016` -- the same strain-dependent, metabolite-sensitive
crossbridge moments -- with two mechanisms of Campbell, Janssen & Campbell
(2018), Biophys J 115:543-553:

* **Thin-filament activation.** Rice et al.'s troponin states (``CaL``,
  ``CaH``) and their force-dependent unbinding give way to a cooperative
  two-state thin filament: calcium switches binding sites from
  non-permissible ``N`` to permissible at a rate ``k_on [Ca] (1 + K_coop
  (1 - N))``, and only unattached permissible sites switch back, at
  ``k_off (1 + K_coop N)``.
* **The myosin OFF state.** Myosin heads are either super-relaxed (OFF,
  ``1 - U_NR``) or non-relaxed (ON, ``U_NR``), and only ON heads attach. The
  OFF -> ON rate rises with active force, ``k_SR (1 + k_force sigma_XB)``:
  force recruits heads, a source of length-dependent activation.

Thick-filament overlap now scales attachment as well as force, and the rates
were refitted at 37 degC (no temperature dependence).

**Parameters** are those of the curated MATLAB code
(`dXdT_cardiovascular_mechanics.m` in github.com/beards-lab/Rat-Cardiac-Energetic),
which the paper's Table 7 reproduces except in two places: it lists
``K_coop = 1.857`` and ``k_off = 101.2 /s`` where the code has 9.6846 and
723.852 /s, and with the table's values the model barely relaxes (RT50 above
90 ms at 4 Hz). The code multiplies both stiffnesses by ``lambda_XB = 1.4``,
the paper's in-vivo factor; the defaults here are Table 7's in-vitro values,
the code's divided by 1.4. ``k_SR`` and ``k_force`` are Table 7's in-vitro
fit (the code reads per-animal values from a spreadsheet).

**Metabolites.** The paper does not state the levels behind its in-vitro fit
(its Fig. 3); the code takes each rat's from the energetics model, Pi from
0.74 to 2.2 mM in the sham rats. With MgATP 8 mM and MgADP 18 uM (Tewari et
al. 2016's resting state) and Pi = 1.0 mM, the model reproduces that figure:
developed tension within 8% of the paper's model at 2-10 Hz and 4% at SL
1.9-2.2 um, and its caption's steady-state pCa50 of 5.89 and 6.00 at SL 1.9
and 2.3 um (5.90 and 6.01 here). Tension is steeply Pi-dependent -- 82 kPa at
4 Hz with Pi = 0.6 mM, 55 kPa at 1.0 -- so Pi defaults to 1.0 mM.

**Departures from the code**, both guards outside the physiological range:
the filament overlap is floored at zero, as in `Tewari2016`, and the
force-dependent OFF -> ON rate at zero, since active force below
``-1 / k_force`` (-0.86 kPa, during fast shortening) would make it negative
and take ``U_NR`` out of [0, 1].

**Numerical scheme.** The eleven states are linear apart from three terms:
the cooperative activation of the thin filament (quadratic in ``N``), the
attachment of ON heads to permissible sites (``U_NR P``) and the recruitment
of heads by force (``sigma_XB (1 - U_NR)``). Over each sub-step these are
linearized about a midpoint state from a half-step predictor, and the
resulting linear system is solved exactly with a matrix exponential (an
exponential Rosenbrock midpoint rule). Linearizing, rather than freezing the
coefficients as `Tewari2016` does, matters here: the thin filament relaxes in
about 0.1 ms, and freezing leaves 5e-4 of peak tension at 0.1 ms sub-steps,
where linearizing leaves 1.4e-6 (6e-6 at the default 0.2 ms).

Examples
--------
>>> import numpy as np
>>> from crossbridge import Marzban2020
>>>
>>> model = Marzban2020(num_cells=10)
>>> model.advance_step(1e-3, Ca_val=2.0, SL_vals=np.full(10, 2.2))
>>> Ta = model.get_active_tension()  # shape (10,), kPa
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt

from ._linalg import expm_batch
from .base import CardiacActivationModel

#: Longest sub-step [s] over which the nonlinear terms are linearized about one state.
_TARGET_SUBSTEP = 2e-4  # [s]

# Rows/columns of the linear system; the last one carries the constant term.
_N, _U = 0, 1
_P10, _P11, _P12, _P20, _P21, _P22, _P30, _P31, _P32 = range(2, 11)
_ONE = 11
_SIZE = 12

#: The moment states, in the order of the linear system.
_MOMENTS = ("p1_0", "p1_1", "p1_2", "p2_0", "p2_1", "p2_2", "p3_0", "p3_1", "p3_2")


class Marzban2020(CardiacActivationModel):
    """
    Vectorized Marzban, Lopez & Beard (2020) rat myofilament model.

    Attributes
    ----------
    N : ndarray, shape (num_cells,)
        Fraction of thin-filament binding sites in the non-permissible state
        (no calcium bound).
    U_NR : ndarray, shape (num_cells,)
        Fraction of myosin heads in the non-relaxed (ON) state.
    p1_0, p1_1, p1_2, p2_0, ..., p3_2 : ndarray, shape (num_cells,)
        Moments of the strain distributions of the attached states A1, A2, A3,
        as in `Tewari2016`: ``p{k}_0`` the fraction [-], ``p{k}_1`` [um] and
        ``p{k}_2`` [um^2], with strain in sarcomere length.

    The permissible fraction ``P = 1 - N - p1_0 - p2_0 - p3_0`` is derived.
    """

    ODE_FILE = Path(__file__).with_name("marzban2020.ode")

    N: npt.NDArray[np.float64]
    U_NR: npt.NDArray[np.float64]
    p1_0: npt.NDArray[np.float64]
    p1_1: npt.NDArray[np.float64]
    p1_2: npt.NDArray[np.float64]
    p2_0: npt.NDArray[np.float64]
    p2_1: npt.NDArray[np.float64]
    p2_2: npt.NDArray[np.float64]
    p3_0: npt.NDArray[np.float64]
    p3_1: npt.NDArray[np.float64]
    p3_2: npt.NDArray[np.float64]

    #: Attributes that evolve; see `CardiacActivationModel.get_state`.
    _state_names = ("N", "U_NR", *_MOMENTS, "_SL_prev", "_SL_curr", "_has_prev_step")

    def __init__(self, num_cells: int, Ta_max: float = 1.0, params: dict | None = None):
        """
        Initialize the Marzban2020 model.

        Parameters
        ----------
        num_cells : int
            Number of independent cells / FEM integration points.
        Ta_max : float, optional
            Unused. Tension follows from the crossbridge stiffnesses
            (`kstiff1`, `kstiff2`). Kept for API compatibility with
            `CardiacActivationModel`. Default is 1.0.
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

        # Rate constants at the metabolite levels (Eq. (6)), which are
        # parameters and so fixed for the model's lifetime.
        g_Pi = p["Pi"] / p["K_Pi"]
        g_ATP = p["MgATP"] / p["K_MgATP"]
        g_ADP = p["MgADP"] / p["K_MgADP"]
        self._kd = p["kd"] * g_Pi / (1.0 + g_Pi)
        self._k1 = p["k1"] / (1.0 + g_Pi)
        self._km2 = p["km2"] * g_ADP / (1.0 + g_ADP + g_ATP)
        self._k3 = p["k3"] * g_ATP / (1.0 + g_ATP + g_ADP)

        self.reset()

    @classmethod
    def default_parameters(cls) -> dict:
        """
        Return default parameters: those of the curated code, with Table 7's
        in-vitro stiffnesses and OFF-state rates (see the module docstring).

        All rates are at 37 degC. Lengths and strain are in um, stiffnesses
        and stresses in kPa, calcium in uM and the metabolites in mM. The
        passive parameters are the code's, converted from mmHg at its
        7.5 mmHg/kPa.
        """
        p: dict = {}
        # Numerical: a *suggested* coupling dt; advance_step sub-steps
        # internally whatever the step it is given.
        p["dt"] = 1e-3  # [s]

        # SL0 only sets the unit of Lambda = SL / SL0 in which Ka is reported:
        # the TriSeg reference length of the paper's heart model, where
        # SL = 1.9 exp(fibre strain).
        p["SL0"] = 1.9  # [um]

        # Metabolites: Tewari et al. (2016)'s resting MgATP and MgADP, and the
        # Pi at which the model reproduces the paper's Fig. 3 (module docstring)
        p["MgATP"] = 8.0  # [mM]
        p["MgADP"] = 0.018  # [mM]
        p["Pi"] = 1.0  # [mM]
        p["K_MgATP"] = 0.4897  # [mM]
        p["K_MgADP"] = 0.194  # [mM]
        p["K_Pi"] = 4.0  # [mM]

        # Sarcomere geometry
        p["L_thick"] = 1.67  # [um]
        p["L_thin"] = 1.2  # [um]
        p["L_bare"] = 0.1  # [um]

        # Thin-filament activation (Campbell et al. 2018)
        p["k_on"] = 101.1850  # [1/(uM s)]
        p["k_off"] = 723.8520  # [1/s]
        p["K_coop"] = 9.6846  # [-]

        # Myosin OFF <-> ON (super-relaxed <-> non-relaxed)
        p["k_SR"] = 14.44  # [1/s]
        p["k_force"] = 1.169  # [1/kPa] (Table 7: 1.169e-3 m^2/N)
        p["k_mSR"] = 50.032  # [1/s]

        # Crossbridge cycle
        p["ka"] = 559.5568  # [1/s]
        p["kd"] = 304.6708  # [1/s]
        p["k1"] = 112.3727  # [1/s]
        p["km1"] = 21.296  # [1/s]
        p["k2"] = 811.72  # [1/s]
        p["km2"] = 43.25  # [1/s]
        p["k3"] = 144.5586  # [1/s]
        p["alpha1"] = 10.0  # [1/um]
        p["alpha2"] = 9.1  # [1/um]
        p["alpha3"] = 5.93  # [1/um^2]
        p["s3"] = 9.9e-3  # [um]
        p["kstiff1"] = 1735.1  # [kPa/um]
        p["kstiff2"] = 45545.0  # [kPa/um]
        p["dr"] = 0.010  # [um] powerstroke

        # Passive force: linear (per half-sarcomere, as in the code) plus collagen
        p["k_passive"] = 25.0 / 7.5  # [kPa/um]
        p["SL_rest_pas"] = 1.8  # [um]
        p["PCon_collagen"] = 0.01 / 7.5  # [kPa]
        p["PExp_collagen"] = 70.0  # [1/um]
        p["SL_collagen"] = 2.25  # [um]
        return p

    def reset(self) -> None:
        """
        Reset to the quiescent state: no crossbridges attached, every
        binding site non-permissible (N = 1), and myosin at its force-free
        steady state, U_NR = k_SR / (k_SR + k_mSR).
        """
        n = self.num_cells
        p = self.p
        self.N = np.ones(n)
        self.U_NR = np.full(n, p["k_SR"] / (p["k_SR"] + p["k_mSR"]))
        for name in _MOMENTS:
            setattr(self, name, np.zeros(n))

        self._SL_prev = np.full(n, p["SL0"])
        self._SL_curr = np.full(n, p["SL0"])
        self._has_prev_step = False

        self._prev_bound_ca = np.zeros(n)
        self._last_dt = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _overlap(self, SL: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Eq. (11): fractional thick-filament overlap."""
        p = self.p
        ov_z = np.minimum(p["L_thick"] / 2.0, SL / 2.0)
        ov_m = np.maximum(SL / 2.0 - (SL - p["L_thin"]), p["L_bare"] / 2.0)
        return 2.0 * np.maximum(ov_z - ov_m, 0.0) / (p["L_thick"] - p["L_bare"])

    def _sigma_xb(
        self, x: npt.NDArray[np.float64], OV_thick: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Eq. (12): active stress [kPa] of state `x`, shape (11, n)."""
        p = self.p
        return OV_thick * (p["kstiff2"] * p["dr"] * x[_P30] + p["kstiff1"] * (x[_P21] + x[_P31]))

    def _system(
        self,
        x: npt.NDArray[np.float64],
        Ca: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
        OV_thick: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        The (n, 12, 12) matrix of d[x, 1]/dt, linearized about state `x`.

        Rows 2-10 are Eq. (5), the moments of `Tewari2016` with attachment
        scaled by U_NR and OV_thick; as in the code, the second moment of A3
        detaches at k3 without the alpha3 s3^2 term. The three nonlinear
        terms -- J_on and J_off (Eq. (7)), attachment ka U_NR P OV_thick, and
        the recruitment k_SR (1 + k_force sigma) (1 - U_NR) (Eq. (10)) -- are
        replaced by their tangents at `x`: each row holds the Jacobian, and
        its constant the value at `x` minus the Jacobian times `x`.
        """
        p = self.p
        a1, a2, a3, s3 = p["alpha1"], p["alpha2"], p["alpha3"], p["s3"]
        kd, k1, km2, k3 = self._kd, self._k1, self._km2, self._k3
        km1, k2 = p["km1"], p["k2"]
        k3s = k3 * (1.0 + a3 * s3 * s3)
        N, U = x[_N], x[_U]
        A = x[_P10] + x[_P20] + x[_P30]
        P = 1.0 - N - A

        M = np.zeros((self.num_cells, _SIZE, _SIZE))
        # Eqs. (7)-(8): dN/dt = -a N (1 + Kc (1 - N)) + b P (1 + Kc N), with
        # a = k_on [Ca], b = k_off and P = 1 - N - (p1_0 + p2_0 + p3_0).
        a, b, Kc = p["k_on"] * Ca, p["k_off"], p["K_coop"]
        dN = -a * N * (1.0 + Kc * (1.0 - N)) + b * P * (1.0 + Kc * N)
        dN_dN = -a * (1.0 + Kc - 2.0 * Kc * N) - b * (1.0 + Kc * N) + b * Kc * P
        dN_dA = -b * (1.0 + Kc * N)
        M[:, _N, _N] = dN_dN
        M[:, _N, _P10] = M[:, _N, _P20] = M[:, _N, _P30] = dN_dA
        M[:, _N, _ONE] = dN - dN_dN * N - dN_dA * A
        # Eq. (10): OFF -> ON recruited by force, its rate floored at zero
        sigma = self._sigma_xb(x, OV_thick)
        drive = 1.0 + p["k_force"] * sigma
        recruiting = drive > 0.0
        k_recruit = p["k_SR"] * np.where(recruiting, drive, 0.0)
        dU = k_recruit * (1.0 - U) - p["k_mSR"] * U
        dU_dU = -k_recruit - p["k_mSR"]
        dU_dsigma = np.where(recruiting, p["k_SR"] * p["k_force"] * (1.0 - U), 0.0)
        M[:, _U, _U] = dU_dU
        M[:, _U, _P30] = dU_dsigma * OV_thick * p["kstiff2"] * p["dr"]
        M[:, _U, _P21] = M[:, _U, _P31] = dU_dsigma * OV_thick * p["kstiff1"]
        # sigma is linear in the moments, so their columns times x sum to it
        M[:, _U, _ONE] = dU - dU_dU * U - dU_dsigma * sigma
        # Attachment ka U P OV_thick of ON heads to permissible sites, into A1
        # at zero strain: its tangent in U and in P = 1 - N - A.
        ka = p["ka"] * OV_thick
        M[:, _P10, _U] = ka * P
        M[:, _P10, _N] = -ka * U
        M[:, _P10, _P20] = M[:, _P10, _P30] = -ka * U
        M[:, _P10, _ONE] = ka * U * (1.0 - P)  # ka U P - (ka P U - ka U (N + A))
        # A1
        M[:, _P10, _P10] = -ka * U - kd - k1
        M[:, _P10, _P11] = k1 * a1
        M[:, _P10, _P12] = -0.5 * k1 * a1 * a1
        M[:, _P10, _P20] += km1
        M[:, _P10, _P21] = km1 * a1
        M[:, _P10, _P22] = 0.5 * km1 * a1 * a1
        M[:, _P11, _P10] = v
        M[:, _P11, _P11] = -kd - k1
        M[:, _P11, _P12] = k1 * a1
        M[:, _P11, _P21] = km1
        M[:, _P11, _P22] = km1 * a1
        M[:, _P12, _P11] = 2.0 * v
        M[:, _P12, _P12] = -kd - k1
        M[:, _P12, _P22] = km1
        # A2
        M[:, _P20, _P10] = k1
        M[:, _P20, _P11] = -k1 * a1
        M[:, _P20, _P12] = 0.5 * k1 * a1 * a1
        M[:, _P20, _P20] = -km1 - k2
        M[:, _P20, _P21] = -km1 * a1 + k2 * a2
        M[:, _P20, _P22] = -0.5 * km1 * a1 * a1 - 0.5 * k2 * a2 * a2
        M[:, _P20, _P30] = km2
        M[:, _P21, _P11] = k1
        M[:, _P21, _P12] = -k1 * a1
        M[:, _P21, _P20] = v
        M[:, _P21, _P21] = -km1 - k2
        M[:, _P21, _P22] = -km1 * a1 + k2 * a2
        M[:, _P21, _P31] = km2
        M[:, _P22, _P12] = k1
        M[:, _P22, _P21] = 2.0 * v
        M[:, _P22, _P22] = -km1 - k2
        M[:, _P22, _P32] = km2
        # A3
        M[:, _P30, _P20] = k2
        M[:, _P30, _P21] = -k2 * a2
        M[:, _P30, _P22] = 0.5 * k2 * a2 * a2
        M[:, _P30, _P30] = -km2 - k3s
        M[:, _P30, _P31] = -2.0 * k3 * a3 * s3
        M[:, _P30, _P32] = -k3 * a3
        M[:, _P31, _P21] = k2
        M[:, _P31, _P22] = -k2 * a2
        M[:, _P31, _P30] = v
        M[:, _P31, _P31] = -km2 - k3s
        M[:, _P31, _P32] = -2.0 * k3 * a3 * s3
        M[:, _P32, _P22] = k2
        M[:, _P32, _P31] = 2.0 * v
        M[:, _P32, _P32] = -km2 - k3
        return M

    def _get_x(self) -> npt.NDArray[np.float64]:
        return np.stack([self.N, self.U_NR, *(getattr(self, m) for m in _MOMENTS)])

    def _set_x(self, x: npt.NDArray[np.float64]) -> None:
        self.N, self.U_NR = x[_N], x[_U]
        for i, name in enumerate(_MOMENTS):
            setattr(self, name, x[_P10 + i])

    def _advance(
        self,
        x: npt.NDArray[np.float64],
        Ca: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
        OV_thick: npt.NDArray[np.float64],
        h: float,
    ) -> npt.NDArray[np.float64]:
        """One exponential-midpoint sub-step of length h; x has shape (11, n)."""
        # [x, 1] per cell, contiguous: a stacked matmul then gives each cell the
        # same arithmetic whatever its batch-mates, which einsum over the
        # transposed stack did not (cells differed from running alone by ~1 ulp).
        x1 = np.empty((self.num_cells, _SIZE, 1))
        x1[:, :_ONE, 0] = x.T
        x1[:, _ONE, 0] = 1.0

        def solve(about, step):
            M = self._system(about, Ca, v, OV_thick)
            return np.ascontiguousarray((expm_batch(M, step) @ x1)[:, :_ONE, 0].T)

        x_half = solve(x, 0.5 * h)
        return solve(x_half, h)

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
            Sarcomere length rate of change [um/s], the velocity that strains
            attached crossbridges. If None, estimated from the sarcomere
            length recorded on the previous call (zero on the first).
        """
        self._begin_step(dt)
        n = self.num_cells

        SL = np.broadcast_to(np.asarray(SL_vals, dtype=float), (n,)).copy()
        Ca = np.maximum(np.broadcast_to(np.asarray(Ca_val, dtype=float), (n,)), 0.0)
        if dSL_vals is not None:
            v = np.broadcast_to(np.asarray(dSL_vals, dtype=float), (n,)).copy()
        elif self._has_prev_step:
            v = (SL - self._SL_prev) / dt
        else:
            v = np.zeros(n)

        OV_thick = self._overlap(SL)
        n_sub = max(1, int(np.ceil(dt / _TARGET_SUBSTEP - 1e-9)))
        h = dt / n_sub
        x = self._get_x()
        for _ in range(n_sub):
            x = self._advance(x, Ca, v, OV_thick, h)
        self._set_x(x)

        self._SL_prev = SL
        self._SL_curr = SL
        self._has_prev_step = True

    def get_active_tension(self) -> npt.NDArray[np.float64]:
        """
        Active tension [kPa], Eq. (12): sigma_XB = OV_thick * (kstiff2 dr p3_0
        + kstiff1 (p2_1 + p3_1)).
        """
        return self._sigma_xb(self._get_x(), self._overlap(self._SL_curr))

    def get_active_stiffness(self) -> npt.NDArray[np.float64]:
        r"""
        Active stiffness [kPa per unit Lambda = SL/SL0].

        .. math::
            K_a = SL_0 \, OV_{thick}\, k_{stiff,1} \left(p_2^0 + p_3^0\right)

        As in `Tewari2016`: velocity enters only the moment equations, and
        tension depends on the first moments of A2 and A3 with weight
        :math:`OV_{thick}\,k_{stiff,1}`. The OFF state responds to force, not
        to velocity, so it adds nothing here.
        """
        OV_thick = self._overlap(self._SL_curr)
        return self.p["SL0"] * OV_thick * self.p["kstiff1"] * (self.p2_0 + self.p3_0)

    def bound_calcium_fraction(self) -> npt.NDArray[np.float64]:
        """
        Fraction of thin-filament binding sites with calcium bound, ``1 - N``:
        in Campbell et al.'s activation, calcium binding is the N -> P
        transition (Eq. (7)), and attached sites keep theirs.
        """
        return 1.0 - self.N

    def get_passive_tension(self) -> npt.NDArray[np.float64]:
        """
        Passive tension [kPa], Eqs. (15)-(16) as the code has them:
        k_passive (SL - SL_rest_pas) / 2 plus Rice et al.'s collagen.
        """
        p = self.p
        SL = self._SL_curr
        collagen = np.where(
            SL > p["SL_collagen"],
            p["PCon_collagen"] * (np.exp(p["PExp_collagen"] * (SL - p["SL_collagen"])) - 1.0),
            0.0,
        )
        return p["k_passive"] * (SL - p["SL_rest_pas"]) / 2.0 + collagen

    def get_atp_hydrolysis_rate(self) -> npt.NDArray[np.float64]:
        """
        Crossbridge ATP hydrolysis rate [1/s per myosin head]: the A3 -> P
        detachment flux, k3 (p3_0 + alpha3 (s3^2 p3_0 + 2 s3 p3_1 + p3_2)),
        which the code averages over a beat as the crossbridge turnover rate.
        """
        a3, s3 = self.p["alpha3"], self.p["s3"]
        strain = s3 * s3 * self.p3_0 + 2.0 * s3 * self.p3_1 + self.p3_2
        return self._k3 * (self.p3_0 + a3 * strain)
