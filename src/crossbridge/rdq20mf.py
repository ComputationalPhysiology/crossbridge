"""
RDQ20 Mean-Field ODE Model for Cardiomyocyte Force Generation (RDQ20-MF)

This module implements the vectorized, production version of the RDQ20 Mean-Field
ODE model as described in [1, 2]. Compared to the RDQ18 model, the RDQ20-MF
model adds explicit crossbridge (XB) cycling dynamics on top of the
regulatory unit (RU) kinetics, allowing it to produce an active tension (Ta)
that depends on both the permissive state fraction and the crossbridge
attachment state.

The key features of the model are:

- **Binary RU states**: Each regulatory unit (RU) has two binary state variables:
  the tropomyosin state T (0=non-permissive, 1=permissive) and the calcium
  binding state C (0=unbound, 1=bound). The joint distribution of a triplet
  (T_{i-1}, T_i, T_{i+1}, C_i) is tracked, giving a (2,2,2,2) state tensor.

- **Crossbridge cycling**: A 4-state (2x2) crossbridge model tracks the fraction
  of attached crossbridges in permissive (P) and non-permissive (N) states, with
  velocity-dependent detachment. The XB sub-step is solved exactly using the
  matrix exponential.

- **Length-dependent activation**: The overlap fraction `frac_SO(SL)` encodes
  the sarcomere length dependence of force, naturally reproducing the
  Frank-Starling mechanism.

- **Vectorization**: All computations are vectorized across `num_cells`,
  enabling efficient simulation of multiple cells or FEM integration points.

References:
-----------
[1] F. Regazzoni "Mathematical modeling and Machine Learning for the numerical
    simulation of cardiac electromechanics", PhD Thesis - Politecnico di Milano
    (2020) http://hdl.handle.net/10589/152617
[2] F. Regazzoni, L. Dede', A. Quarteroni "Biophysically detailed mathematical
    models of multiscale cardiac active mechanics", PLOS Computational Biology
    (2020) https://doi.org/10.1371/journal.pcbi.1008294

Example Usage:
--------------
>>> import numpy as np
>>> from crossbridge.rdq20mf import RDQ20MF
>>>
>>> model = RDQ20MF(num_cells=10)
>>> dt = model.dt
>>> Ca = np.full(10, 0.5)   # 0.5 µM calcium
>>> SL = np.full(10, 2.2)   # 2.2 µm sarcomere length
>>> dSL = np.zeros(10)
>>>
>>> model.advance_step(dt, Ca, SL, dSL)
>>> Ta = model.get_active_tension()  # shape (10,)
"""

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm

from .base import CardiacActivationModel


class RDQ20MF(CardiacActivationModel):
    """
    Vectorized RDQ20 Mean-Field ODE model for cardiomyocyte force generation.

    State variables:
    ----------------
    x_RU : np.ndarray, shape (2, 2, 2, 2, num_cells)
        Joint probability distribution of the RU triplet state
        (T_{i-1}, T_i, T_{i+1}, C_i) where T is tropomyosin permissivity
        (0=non-permissive, 1=permissive) and C is calcium binding (0/1).

    x_XB : np.ndarray, shape (2, 2, num_cells)
        Crossbridge state distribution. Axes: (permissive/non-permissive,
        first/second attached state, cell). x_XB[1, :, c] gives the fraction
        of attached crossbridges in cell c.
    """

    def __init__(self, num_cells: int, Ta_max: float = 1.0, params: dict | None = None):
        """
        Initialize the RDQ20-MF model.

        Parameters
        ----------
        num_cells : int
            Number of independent cells / FEM integration points.
        Ta_max : float, optional
            Scaling factor for active tension (kPa). Not used directly since
            the model computes Ta = a_XB * attached_fraction * frac_SO internally.
            Kept for API compatibility. Default is 1.0.
        params : dict, optional
            Parameter overrides. Keys should match those returned by
            `default_parameters()`.
        """
        super().__init__(int(num_cells), Ta_max, params)

        self.p = type(self).default_parameters()
        if params:
            self.p.update(params)

        self.dt = self.p["dt_RU"]
        # XB sub-step frequency: advance XB every freqXB RU steps (every 1 ms)
        self.freqXB: int = round(1e-3 / self.dt)
        # How often to recompute Ca-dependent rates (every 10 RU steps)
        self.freq_rates_update: int = round(2.5e-4 / self.dt)

        # Internal step counter (for XB and rate update scheduling)
        self._step_count: int = 0

        # Precompute constant part of kT (shape: (2,2,2,2))
        # kT[a,b,c,B] = rate of T_i going to state 1-b given context (a,b,c,B)
        # Axes: (T_{i-1}, T_i, T_{i+1}, C_i)
        self._init_rate_matrices()

        # --- State initialization ---
        # x_RU: shape (2, 2, 2, 2, num_cells)
        # Start with everything in state (0,0,0,0): T non-permissive, Ca unbound
        self.x_RU = np.zeros((2, 2, 2, 2, self.num_cells))
        self.x_RU[0, 0, 0, 0, :] = 1.0

        # x_XB: shape (2, 2, num_cells)
        # Axes: (permissive-index, sub-state, cell)
        self.x_XB = np.zeros((2, 2, self.num_cells))

        # Cache for SL-dependent quantities (updated each step)
        self._SL_prev = np.full(self.num_cells, self.p["SL0"])
        self._SL_curr = np.full(self.num_cells, self.p["SL0"])

        # kC: calcium transition rate matrix, shape (2, 2, num_cells)
        # kC[C_current, T_current, cell] = rate of LEAVING Ca state C_current given T_current
        # This matches reference: kC[A, a] = P(C_i leaving state A | T_i = a) / dt
        # Static (unbinding) entries — rate of leaving Ca-bound state (C=1):
        #   kC[1, 0, :] = Koff          (Ca unbinding given T non-permissive)
        #   kC[1, 1, :] = Koff / mu     (Ca unbinding given T permissive)
        # Dynamic (binding) entries — rate of leaving Ca-unbound state (C=0):
        #   kC[0, 0, :] = kC[0, 1, :] = Kon * Ca  (updated in _update_Ca_rates)
        self.kC = np.zeros((2, 2, self.num_cells))
        self.kC[1, 0, :] = self.p["Koff"]  # unbinding, T non-permissive
        self.kC[1, 1, :] = self.p["Koff"] / self.p["mu"]  # unbinding, T permissive

        # kT_base: time-invariant part of kT, shape (2, 2, 2, 2)
        # This is set in _init_rate_matrices and shared across cells.
        # The Q parameter modifies kT[:,0,:,:] dynamically per SL and cell.

    def _init_rate_matrices(self) -> None:
        """Precompute the time-invariant portions of the RU rate matrix kT."""
        p = self.p
        Q = p["Q"]
        gamma = p["gamma"]
        Kbasic = p["Kbasic"]
        mu = p["mu"]

        # expMat[a, c] = number of permissive neighbors = a + c  (a, c ∈ {0,1})
        expMat = np.array([[0, 1], [1, 2]])  # shape (2, 2)

        # kT_base shape: (2, 2, 2, 2)  → (a, b, c, B)
        # Only kT[:, b=1, :, :] (permissive → non-permissive) and
        # kT[:, b=0, :, :] (non-permissive → permissive) are non-zero.
        self.kT_base = np.zeros((2, 2, 2, 2))

        # Permissive → non-permissive (b=1 → 0):
        # Rate = Kbasic * gamma^(2 - n_perm_neighbors), independent of Ca and SL
        self.kT_base[:, 1, :, 0] = Kbasic * gamma ** (2 - expMat)
        self.kT_base[:, 1, :, 1] = Kbasic * gamma ** (2 - expMat)

        # Non-permissive → permissive (b=0 → 1):
        # Rate = Q * Kbasic * (1/mu if Ca unbound, 1 if Ca bound) * gamma^n_perm_neighbors
        # Q can be SL-dependent (here stored as static; vectorized update in _update_kT_NP)
        self.kT_base[:, 0, :, 0] = Q * Kbasic / mu * gamma**expMat
        self.kT_base[:, 0, :, 1] = Q * Kbasic * gamma**expMat

    @classmethod
    def default_parameters(cls) -> dict:
        """
        Return default parameters corresponding to the human body-temperature
        parameterization from the original paper (Table 2 in [2]).
        """
        p: dict = {}
        # Numerical
        p["dt_RU"] = 2.5e-5  # [s] RU integration timestep

        # Geometry
        p["LA"] = 1.25  # [µm] actin filament half-length
        p["LM"] = 1.65  # [µm] myosin filament half-length
        p["LB"] = 0.18  # [µm] bare zone half-length
        p["SL0"] = 2.2  # [µm] reference sarcomere length

        # RU steady-state / cooperativity
        p["mu"] = 10.0  # [-] Ca-binding cooperativity when permissive
        p["gamma"] = 12.0  # [-] nearest-neighbor cooperativity
        p["Q"] = 2.0  # [-] permissive transition cooperativity factor

        # Calcium binding (Kd-based; Kon computed dynamically as Koff/Kd)
        p["Kd0"] = 0.381  # [µM] dissociation constant at SL0=2.15
        p["alphaKd"] = -0.571  # [µM/µm] SL-dependence of Kd

        # RU kinetics
        p["Koff"] = 100.0  # [s^-1] Ca unbinding rate
        p["Kbasic"] = 13.0  # [s^-1] basic permissive ↔ non-permissive rate

        # XB cycling
        p["r0"] = 134.31  # [s^-1] baseline XB detachment rate
        p["alpha"] = 25.184  # [-] velocity-dependent detachment scaling
        p["mu0_fP"] = 32.653  # [s^-1] XB attachment rate (state 0)
        p["mu1_fP"] = 0.778  # [s^-1] XB attachment rate (state 1)

        # Upscaling
        p["a_XB"] = 22.894e3  # [kPa] active tension scaling

        return p

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _Kd(self, SL: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Sarcomere-length-dependent dissociation constant Kd [µM]."""
        return self.p["Kd0"] - self.p["alphaKd"] * (2.15 - SL)

    def _frac_SO(self, SL: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Overlap fraction between actin and myosin filaments (vectorized).

        Parameters
        ----------
        SL : np.ndarray, shape (num_cells,)
            Sarcomere lengths [µm].

        Returns
        -------
        np.ndarray, shape (num_cells,)
            Fraction of myosin half-length in single-overlap region [0, 1].
        """
        LA = self.p["LA"]
        LM = self.p["LM"]
        LB = self.p["LB"]
        LMh = (LM - LB) * 0.5  # half-length of cross-bridge bearing region

        frac = (
            (SL > LA) * (SL <= LM) * (SL - LA)
            + (SL > LM) * (SL <= 2 * LA - LB) * (SL + LM - 2 * LA) * 0.5
            + (SL > 2 * LA - LB) * (SL <= 2 * LA + LB) * LMh
            + (SL > 2 * LA + LB) * (SL <= 2 * LA + LM) * (LM + 2 * LA - SL) * 0.5
        ) / LMh
        return frac

    def _update_Ca_rates(
        self,
        Ca: float | npt.NDArray[np.float64],
        SL: npt.NDArray[np.float64],
    ) -> None:
        """
        Update the calcium-binding rate kC[0, :, :] based on current Ca and SL.

        kC[0, 0, c] = kC[0, 1, c] = Kon * Ca[c]  (rate of leaving Ca-unbound state)
        kC[1, :, c] = Koff (or Koff/mu) — already set in __init__
        """
        Kd = self._Kd(SL)  # shape (num_cells,)
        Kon = self.p["Koff"] / Kd  # [µM^-1 s^-1], shape (num_cells,)
        if np.isscalar(Ca):
            kon_ca = Kon * float(Ca)  # shape (num_cells,)
        else:
            kon_ca = Kon * np.asarray(Ca)  # shape (num_cells,)
        # kC[0, :, :] = Kon * Ca (rate of leaving Ca-unbound, regardless of T state)
        self.kC[0, 0, :] = kon_ca
        self.kC[0, 1, :] = kon_ca

    def _RU_get_rhs(self) -> npt.NDArray[np.float64]:
        """
        Compute the RHS of the mean-field RU ODEs (vectorized over num_cells).

        Returns
        -------
        np.ndarray, shape (2, 2, 2, 2, num_cells)
        """
        x = self.x_RU  # (2, 2, 2, 2, num_cells)

        # kT has shape (2, 2, 2, 2) — same for all cells (SL-dependence via Q
        # is captured in kT_base; if Q were SL-dependent per cell, we'd need
        # per-cell kT, but Q is a fixed parameter here).
        kT = self.kT_base  # (2, 2, 2, 2)

        # --- Tropomyosin fluxes ---
        # PhiT_C[a,b,c,B,cell] = x[a,b,c,B,cell] * kT[a,b,c,B]
        PhiT_C = x * kT[:, :, :, :, np.newaxis]
        PhiT_C = np.nan_to_num(PhiT_C, nan=0.0)

        # Marginal rate for left-neighbor transition, averaged over a (axis0) and B (axis3).
        # Matches reference exactly: kT_L = sum(PhiT_C,(0,3)) / sum(x,(0,3)), shape (b,c,cell).
        # Reference broadcast: kT_L[:,:,None,None]*x => (b,c,1,1) × (a,b,c,B)
        #   → first two dims of kT_L align with first two dims of x, i.e. (a,b) positions.
        # We replicate by appending cell dim and using the same [:,: ,None,None,:] broadcast.
        sum_PhiT_C_03 = PhiT_C.sum(axis=(0, 3))  # (b=2, c=2, num_cells)
        sum_x_03 = x.sum(axis=(0, 3))  # (b=2, c=2, num_cells)
        kT_L = np.divide(
            sum_PhiT_C_03, sum_x_03, out=np.zeros_like(sum_PhiT_C_03), where=sum_x_03 > 0
        )
        # kT_L shape (b,c,cell); broadcast as (b,c,1,1,cell) maps to dims (a,b,c,B,cell)
        PhiT_L = np.nan_to_num(
            kT_L[:, :, np.newaxis, np.newaxis, :] * x, nan=0.0
        )  # (2, 2, 2, 2, num_cells); PhiT_L[a,b,c,B,cell] = kT_L[a,b,cell]*x[a,b,c,B,cell]

        # Marginal rate for right-neighbor transition, averaged over c (axis2) and B (axis3).
        # kT_R = sum(PhiT_C,(2,3)) / sum(x,(2,3)), shape (a,b,cell).
        # Reference broadcast: kT_R[None,:,:,None]*x => (1,a,b,1) × (a,b,c,B)
        #   → last two non-trivial dims (a,b) align with dims 1,2 of x = (b,c) positions.
        sum_PhiT_C_23 = PhiT_C.sum(axis=(2, 3))  # (a=2, b=2, num_cells)
        sum_x_23 = x.sum(axis=(2, 3))  # (a=2, b=2, num_cells)
        kT_R = np.divide(
            sum_PhiT_C_23, sum_x_23, out=np.zeros_like(sum_PhiT_C_23), where=sum_x_23 > 0
        )
        # kT_R shape (a,b,cell); broadcast as (1,a,b,1,cell) maps to dims (a,b,c,B,cell)
        # but reference broadcasts [None,:,:,None] which places (a,b) at dims 1,2 → (b,c)
        PhiT_R = np.nan_to_num(
            kT_R[np.newaxis, :, :, np.newaxis, :] * x, nan=0.0
        )  # PhiT_R[a,b,c,B,cell] = kT_R[b,c,cell]*x[a,b,c,B,cell]

        # --- Calcium-binding fluxes ---
        # kC shape: (2, 2, num_cells) → (C_new, T_i, cell)
        # We need it indexed as kC[B_new, b_T, cell] and broadcast over (a, c) dims
        # PhiC_C[a,b,c,B,cell] = x[a,b,c,B,cell] * kC[B, b, cell]
        # kC.swapaxes(0,1) has shape (T_i, C_new, cell) = (2, 2, num_cells)
        # We want kC[B_new=dim3, T_i=dim1, cell=dim4]
        kC_transposed = self.kC.swapaxes(0, 1)  # (T_i=2, C_new=2, num_cells)
        PhiC_C = np.nan_to_num(
            x * kC_transposed[np.newaxis, :, np.newaxis, :, :], nan=0.0
        )  # (2, 2, 2, 2, num_cells)

        # --- Total RHS ---
        # For each flux Phi, the RHS contribution is:
        #   -Phi[..., current_state, ...] + Phi[..., flipped_state, ...]
        # which equals the difference between in-flux and out-flux.
        rhs = (
            -PhiT_L
            + np.flip(PhiT_L, axis=0)
            - PhiT_C
            + np.flip(PhiT_C, axis=1)
            - PhiT_R
            + np.flip(PhiT_R, axis=2)
            - PhiC_C
            + np.flip(PhiC_C, axis=3)
        )
        return rhs

    def _XB_advance(
        self,
        dSL_dt: npt.NDArray[np.float64],
        dt_xb: float,
    ) -> None:
        """
        Advance the crossbridge state x_XB over dt_xb using the matrix exponential.

        The XB sub-system for each cell is a linear 4-state ODE driven by the
        current permissive fraction and permissive↔non-permissive exchange rates
        from the RU state. The exact solution is computed via matrix exponential.

        Parameters
        ----------
        dSL_dt : np.ndarray, shape (num_cells,)
            Sarcomere shortening velocity [µm/s] for each cell.
        dt_xb : float
            Duration of the XB sub-step [s].
        """
        p = self.p
        x_RU = self.x_RU  # (2, 2, 2, 2, num_cells)

        # Normalized shortening velocity v = -dSLdt / SL0  [1/s]
        v = -dSL_dt / p["SL0"]  # shape (num_cells,)

        # Permissive fraction per cell: sum over (T_{i-1}, T_{i+1}, C_i) where T_i=1
        perm = x_RU[:, 1, :, :, :].sum(axis=(0, 1, 2))  # shape (num_cells,)

        # Mean permissive→non-permissive rate k_PN per cell
        x_perm = x_RU[:, 1, :, :, :]  # (2, 2, 2, num_cells)
        kT_perm = self.kT_base[:, 1, :, :]  # (2, 2, 2) — rate T_i goes 1→0
        num_kPN = (kT_perm[:, :, :, np.newaxis] * x_perm).sum(axis=(0, 1, 2))
        den_kPN = x_perm.sum(axis=(0, 1, 2))
        k_PN = np.where(den_kPN > 0, num_kPN / den_kPN, 0.0)  # shape (num_cells,)

        # Mean non-permissive→permissive rate k_NP per cell
        x_nonperm = x_RU[:, 0, :, :, :]  # (2, 2, 2, num_cells)
        kT_nonperm = self.kT_base[:, 0, :, :]  # (2, 2, 2)
        num_kNP = (kT_nonperm[:, :, :, np.newaxis] * x_nonperm).sum(axis=(0, 1, 2))
        den_kNP = x_nonperm.sum(axis=(0, 1, 2))
        k_NP = np.where(den_kNP > 0, num_kNP / den_kNP, 0.0)  # shape (num_cells,)

        # Velocity-dependent detachment rate
        r = p["r0"] + p["alpha"] * np.abs(v)  # shape (num_cells,)
        diag_P = r + k_PN  # shape (num_cells,)
        diag_N = r + k_NP  # shape (num_cells,)

        # Build and solve the 4x4 linear ODE for every cell at once (batched
        # over the leading axis): d/dt [xP0, xN0, xP1, xN1]^T = A @ [...] + rhs
        # where P=permissive, N=non-permissive, 0/1=XB sub-states. This mirrors
        # the per-cell system in the reference implementation without a
        # Python-level loop over cells (np.linalg.solve and scipy.linalg.expm
        # both operate on stacks of (num_cells, 4, 4) matrices).
        A = np.zeros((self.num_cells, 4, 4))
        A[:, 0, 0] = -diag_P
        A[:, 0, 2] = k_NP
        A[:, 1, 0] = -v
        A[:, 1, 1] = -diag_P
        A[:, 1, 3] = k_NP
        A[:, 2, 0] = k_PN
        A[:, 2, 2] = -diag_N
        A[:, 3, 1] = k_PN
        A[:, 3, 2] = -v
        A[:, 3, 3] = -diag_N

        zeros = np.zeros(self.num_cells)
        rhs_vec = np.stack(
            [perm * p["mu0_fP"], perm * p["mu1_fP"], zeros, zeros], axis=-1
        )  # (num_cells, 4), ordered to match the A/state layout above

        # sol: (num_cells, 4), Fortran-order flatten of x_XB[:, :, c] per cell
        sol = np.stack(
            [self.x_XB[0, 0, :], self.x_XB[1, 0, :], self.x_XB[0, 1, :], self.x_XB[1, 1, :]],
            axis=-1,
        )

        # Steady state: A @ sol_inf = -rhs_vec  →  sol_inf = -A^{-1} rhs_vec
        try:
            sol_inf = -np.linalg.solve(A, rhs_vec[:, :, np.newaxis])[:, :, 0]
        except np.linalg.LinAlgError:
            sol_inf = np.zeros((self.num_cells, 4))

        delta = sol - sol_inf  # (num_cells, 4)
        new_sol = sol_inf + np.einsum("nij,nj->ni", expm(dt_xb * A), delta)

        self.x_XB[0, 0, :] = new_sol[:, 0]
        self.x_XB[1, 0, :] = new_sol[:, 1]
        self.x_XB[0, 1, :] = new_sol[:, 2]
        self.x_XB[1, 1, :] = new_sol[:, 3]

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
        Advance the RDQ20-MF model by one time step.

        Parameters
        ----------
        dt : float
            Time step [s]. Should match `self.dt` (= 2.5e-5 s) for stability.
            If larger, the step is still taken but may be less accurate.
        Ca_val : float or np.ndarray, shape (num_cells,)
            Intracellular calcium concentration [µM].
        SL_vals : float or np.ndarray, shape (num_cells,)
            Current sarcomere lengths [µm].
        dSL_vals : float or np.ndarray, shape (num_cells,) or None
            Sarcomere shortening velocity [µm/s]. If None, estimated from
            stored previous SL and current dt.
        """
        self._begin_step(dt)

        # Broadcast scalars to arrays
        if np.isscalar(SL_vals):
            SL_vals = np.full(self.num_cells, float(SL_vals))
        else:
            SL_vals = np.asarray(SL_vals, dtype=float)

        if np.isscalar(Ca_val):
            Ca_arr: float | npt.NDArray[np.float64] = float(Ca_val)
        else:
            Ca_arr = np.asarray(Ca_val, dtype=float)

        assert SL_vals.shape == (self.num_cells,), (
            f"SL_vals shape {SL_vals.shape} must be ({self.num_cells},)"
        )

        # Update Ca-dependent rates (every freq_rates_update steps)
        if self._step_count % self.freq_rates_update == 0:
            self._update_Ca_rates(Ca_arr, SL_vals)

        # --- RU step (explicit Euler) ---
        rhs = self._RU_get_rhs()
        self.x_RU = self.x_RU + dt * rhs
        # Clip to avoid tiny negative values from floating-point drift
        np.clip(self.x_RU, 0.0, 1.0, out=self.x_RU)

        # --- XB step (every freqXB steps, i.e., every 1 ms) ---
        if self._step_count > 0 and self._step_count % self.freqXB == 0:
            if dSL_vals is not None:
                if np.isscalar(dSL_vals):
                    dSL_arr = np.full(self.num_cells, float(dSL_vals))
                else:
                    dSL_arr = np.asarray(dSL_vals, dtype=float)
            else:
                # Estimate from stored SL history
                dSL_arr = (SL_vals - self._SL_prev) / (dt * self.freqXB)
            self._XB_advance(dSL_arr, dt * self.freqXB)

        # Store SL for velocity estimation
        if self._step_count % self.freqXB == 0:
            self._SL_prev = SL_vals.copy()
        self._SL_curr = SL_vals.copy()

        self._step_count += 1

    def compute_permissivity(self) -> npt.NDArray[np.float64]:
        """
        Compute the permissive fraction for each cell (fraction of RUs in
        the permissive state T_i=1, weighted by the overlap fraction).

        Returns
        -------
        np.ndarray, shape (num_cells,)
        """
        # Sum over (T_{i-1}, T_{i+1}, C_i): permissive fraction = x_RU[:, 1, :, :, c].sum()
        perm = self.x_RU[:, 1, :, :, :].sum(axis=(0, 1, 2))  # (num_cells,)
        frac = self._frac_SO(self._SL_curr)
        return perm * frac

    def get_active_tension(self) -> npt.NDArray[np.float64]:
        """
        Compute macroscopic active tension (kPa) from the crossbridge state.

        Ta = a_XB * (fraction of attached XBs) * frac_SO(SL)

        Returns
        -------
        np.ndarray, shape (num_cells,)
        """
        # Attached XBs: x_XB[1, :, c].sum() for each cell c
        attached = self.x_XB[1, :, :].sum(axis=0)  # (num_cells,)
        frac = self._frac_SO(self._SL_curr)
        return self.p["a_XB"] * attached * frac

    def get_active_stiffness(self) -> npt.NDArray[np.float64]:
        r"""
        Compute active stiffness (kPa per unit Lambda) from the crossbridge state.

        .. math::
            K_a = a_{XB}\, \chi_{SO}(SL) \left[\mu_P^0 + \mu_N^0\right]

        This is Eq. (52) of Regazzoni & Quarteroni (2020). Where
        :meth:`get_active_tension` sums the *first*-order distribution moments
        (mean crossbridge elongation, hence a force), this sums the
        *zeroth*-order ones -- the fraction of binding sites actually carrying
        a crossbridge. Since each attached crossbridge is modelled as a linear
        spring, that fraction upscaled by ``a_XB`` is precisely the tissue-level
        stiffness of the attached population, which is why the same expression
        also follows from a purely physical derivation.

        .. note::
            The formal derivative :math:`\partial\dot T_a/\partial\dot\lambda`
            has one further contribution, through the velocity-dependent
            detachment rate :math:`r = r_0 + \alpha|v|`. It is deliberately
            omitted here, following R&Q: it is a detachment-rate effect rather
            than a stiffness, and being proportional to :math:`\mathrm{sign}(v)`
            it would make ``Ka`` discontinuous at zero shortening velocity --
            ruinous for the Newton tangent this quantity is meant to supply.
            Setting ``params={"alpha": 0.0}`` removes it from the model
            entirely, which is how the finite-difference test verifies this
            formula exactly.
        """
        # x_XB is indexed [moment_order, permissivity, cell]; moment order 0 is
        # the attached fraction, whereas get_active_tension() uses order 1.
        mu0 = self.x_XB[0, :, :].sum(axis=0)  # mu0_P + mu0_N, shape (num_cells,)
        frac = self._frac_SO(self._SL_curr)
        return self.p["a_XB"] * mu0 * frac

    def bound_calcium_fraction(self) -> npt.NDArray[np.float64]:
        """
        Marginal probability that a regulatory unit has calcium bound.

        ``x_RU`` is indexed ``[T_{i-1}, T_i, T_{i+1}, C_i, cell]``, so this
        fixes the unit's own calcium state ``C_i = 1`` and sums out the
        tropomyosin states -- the calcium analogue of what
        :meth:`compute_permissivity` does for ``T_i``.
        """
        return self.x_RU[:, :, :, 1, :].sum(axis=(0, 1, 2))

    def reset(self) -> None:
        """Reset model state to initial (fully non-permissive, no attached XBs)."""
        self._prev_bound_ca = np.zeros(self.num_cells)
        self._last_dt = 0.0
        self.x_RU[:] = 0.0
        self.x_RU[0, 0, 0, 0, :] = 1.0
        self.x_XB[:] = 0.0
        self._step_count = 0
        self._SL_prev = np.full(self.num_cells, self.p["SL0"])
        self._SL_curr = np.full(self.num_cells, self.p["SL0"])
