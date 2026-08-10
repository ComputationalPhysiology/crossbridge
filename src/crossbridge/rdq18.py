"""
Reduced-Order Model for Sarcomere Dynamics (RDQ18)

This module implements the reduced-order Ordinary Differential Equation (ODE) model
for the mechanical activation of cardiac myofilaments, as proposed by Regazzoni,
Dedè, and Quarteroni (2018).
The model derives from a spatially explicit continuous-time Markov Chain (CTMC)
that captures nearest-neighbor cooperative interactions along the myofilaments
(e.g., how the attachment of one crossbridge facilitates the attachment of neighbors).
By assuming conditional independence of specific sets of events, the original system
of ~10^21 degrees of freedom is reduced to a highly efficient system of ~2200 ODEs,
achieving a ~10,000x computational speedup without sacrificing the spatial fidelity
required to model length-dependent activation.

Key Features:
-------------
- **Vectorization**: Designed to simulate multiple independent cells or integration
  points simultaneously via the `num_cells` parameter, making it highly suitable
  for tissue-level finite element (FEM) or 0D coupled electromechanics simulations.
- **Length-Dependent Activation**: Explicitly models the overlap between actin and
  myosin filaments based on current Sarcomere Length (SL), naturally reproducing
  the macroscopic Frank-Starling mechanism.
- **Cooperativity**: Captures the steep, non-linear force-calcium relationship typical
  of cardiac muscle dynamics.

Reference:
----------
Regazzoni, F., Dedè, L., & Quarteroni, A. (2018). Active contraction of cardiac cells:
a reduced model for sarcomere dynamics with cooperative interactions.
Biomechanics and Modeling in Mechanobiology, 17(6), 1663-1686.
https://doi.org/10.1007/s10237-018-1049-0

Example Usage:
--------------
>>> import numpy as np
>>> from crossbridge.rdq18 import RDQ18
>>>
>>> # Initialize for 100 cells/integration points
>>> model = RDQ18(num_cells=100)
>>>
>>> # Define inputs for the current time step
>>> dt = 2.5e-5
>>> calcium_uM = np.full(100, 1.0)  # Intracellular calcium (1.0 uM)
>>> SL_um = np.full(100, 2.2)       # Sarcomere length (2.2 um)
>>>
>>> # Advance the model by one time step
>>> model.advance_ODE(dt, Ca_val=calcium_uM, SL_vals=SL_um)
>>>
>>> # Compute the fraction of permissive crossbridges (proxy for active tension)
>>> permissivity = model.compute_permissivity()
"""

import numpy as np
import numpy.typing as npt
from .base import CardiacActivationModel


class RDQ18(CardiacActivationModel):
    def __init__(self, num_cells: int, Ta_max: float = 100.0, params=None):
        """
        Vectorized implementation of the RDQ18 Sarcomere model.
        """
        # Call the abstract base class constructor
        super().__init__(int(num_cells), Ta_max, params)

        self.p = self.default_parameters()
        if params:
            self.p.update(params)

        self.nu = self.p["nu"]
        self.dt = self.p["dt"]

        # --- Precompute Spatial Functions & Matrices ---
        self.funcs = self._get_spatial_functions()

        # Constant broadcasting matrices
        # Shape: (nu, 4, 1, 4, 1, num_cells) for correct broadcasting
        _base = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 2, 2], [1, 1, 2, 2]])
        self.expMat = np.tile(
            _base[np.newaxis, :, np.newaxis, :, np.newaxis, np.newaxis],
            (self.nu, 1, 1, 1, 1, self.num_cells),
        )
        self.gammaPlusExp = self.p["gamma"] ** self.expMat
        self.gammaMinusExp = self.p["gamma"] ** (-self.expMat)

        # Index for spatial nodes (0..nu-1)
        self.j_indices = np.arange(self.nu)

        # --- Initialize State Variables ---
        # xODE shape: (nu-2, Left, Center, Right, Cell)
        self.xODE = np.zeros((self.nu - 2, 4, 4, 4, self.num_cells))

        # Initialize all units to state 0 (0N: index 0)
        self.xODE[:, 0, 0, 0, :] = 1.0

        # Transition Matrix PC: (nu, Left, Center, Right, Target, Cell)
        self.PC = np.zeros((self.nu, 4, 4, 4, 4, self.num_cells))

        # Pre-allocate Flux Matrices
        self.PhiC = np.zeros((self.nu - 2, 4, 4, 4, 4, self.num_cells))
        self.PhiL = np.zeros((self.nu - 2, 4, 4, 4, 4, self.num_cells))
        self.PhiR = np.zeros((self.nu - 2, 4, 4, 4, 4, self.num_cells))

    @classmethod
    def default_parameters(cls):
        p = {}
        # Simulation
        p["dt"] = 2.5e-5

        # Sarcomere Geometry & Rates
        p["LA"] = 1.2
        p["LM"] = 1.65
        p["LB"] = 0.1
        p["nu"] = 36
        p["Kon"] = 80.0
        p["Koff"] = 80.0
        p["Q0"] = 3.0
        p["SLQ"] = 2.2
        p["alphaQ"] = 1.4
        p["Kbasic"] = 10.0
        p["mu"] = 10.0
        p["gamma"] = 40.0
        p["aR"] = 0.1
        p["aL"] = 0.1
        # Calculated constants
        p["K1on"] = p["Kon"]
        p["K1off"] = p["Koff"] / p["mu"]
        return p

    def _get_spatial_functions(self):
        p = self.p
        xi = lambda i: (p["LM"] - p["LB"]) * 0.5 * (i + 1) / p["nu"]
        xAZ = lambda SL: (SL - p["LB"]) / 2
        xLA = lambda SL: p["LA"] - xAZ(SL) - p["LB"]
        xRA = lambda SL: xAZ(SL) - p["LA"]
        Q = lambda SL: p["Q0"] - p["alphaQ"] * (p["SLQ"] - SL) * (SL < p["SLQ"])
        return xi, xAZ, xLA, xRA, Q

    def _compute_Chi(self, SL: npt.NDArray[np.float64]):
        """Vectorized computation of Chi functions."""
        xi, xAZ, xLA, xRA, _ = self.funcs
        # xi_val: (nu, 1). SL: (1, num_cells)
        xi_val = xi(self.j_indices)[:, np.newaxis]
        SL_row = SL[np.newaxis, :]
        xRA_val = xRA(SL_row)
        xAZ_val = xAZ(SL_row)
        xLA_val = xLA(SL_row)

        # Use np.where for safe vectorized conditionals across arrays
        # ChiRA
        resRA = np.where(
            xi_val <= xRA_val,
            np.exp(-((xRA_val - xi_val) ** 2) / self.p["aR"] ** 2),
            np.where(xi_val < xAZ_val, 1.0, np.exp(-((xi_val - xAZ_val) ** 2) / self.p["aR"] ** 2)),
        )

        # ChiLA
        resLA = np.where(
            xi_val <= xLA_val, np.exp(-((xLA_val - xi_val) ** 2) / self.p["aR"] ** 2), 1.0
        )
        return resLA, resRA

    def update_probabilities(
        self, SL: npt.NDArray[np.float64], Ca_t: float | npt.NDArray[np.float64]
    ):
        """Update PC matrix based on current SL (vector) and Ca (scalar)."""
        Kpn0 = self.p["Kbasic"] * self.p["gamma"] ** 2
        Kpn1 = self.p["Kbasic"] * self.p["gamma"] ** 2

        # Reshape Chi for broadcasting: (nu, 1, 1, 1, 1, num_cells)
        ChiLA, ChiRA = self._compute_Chi(SL)
        ChiLA = ChiLA[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
        ChiRA = ChiRA[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]

        _, _, _, _, Q = self.funcs
        term_common = ChiRA * ChiLA * self.gammaPlusExp

        # Reshape Q(SL) to match PC shape (1, 1, 1, 1, 1, num_cells)
        Q_sl = Q(SL).reshape(1, 1, 1, 1, 1, self.num_cells)
        Knp0_val = Q_sl * self.p["Kbasic"] / self.p["mu"]
        Knp1_val = Q_sl * self.p["Kbasic"]

        # Reset Constant Rates
        self.PC[:] = 0.0
        self.PC[:, :, 1, :, 0, :] = self.p["Koff"]  # 1N -> 0N
        self.PC[:, :, 3, :, 0, :] = Kpn0 * self.gammaMinusExp[:, :, 0, :, 0, :]  # 0P -> 0N
        self.PC[:, :, 2, :, 1, :] = Kpn1 * self.gammaMinusExp[:, :, 0, :, 0, :]  # 1P -> 1N
        self.PC[:, :, 2, :, 3, :] = self.p["K1off"]  # 1P -> 0P

        # Dynamic Rates
        # 0N -> 1N
        self.PC[:, :, 0, :, 1, :] = self.p["Kon"] * ChiRA[:, :, 0, :, 0, :] * Ca_t
        # 1N -> 1P
        self.PC[:, :, 1, :, 2, :] = Knp1_val * term_common[:, :, 0, :, 0, :]
        # 0P -> 1P
        self.PC[:, :, 3, :, 2, :] = self.p["K1on"] * ChiRA[:, :, 0, :, 0, :] * Ca_t
        # 0N -> 0P
        self.PC[:, :, 0, :, 3, :] = Knp0_val * term_common[:, :, 0, :, 0, :]

    def advance_ODE(
        self,
        dt_step: float,
        Ca_val: float | npt.NDArray[np.float64],
        SL_vals: float | npt.NDArray[np.float64],
    ) -> None:
        """
        Advance the state of all cells by dt_step.

        This function internally updates probabilities based on current SL and Ca,
        and then integrates the ODEs using an adaptive time-stepping approach to ensure stability.

        Parameters:
        -----------
        dt_step : float
            The total time to advance the ODEs.
        Ca_val : float
            The current calcium concentration (scalar) in micro molar.
        SL_vals : np.ndarray
            Current sarcomere lengths for each cell (shape: (num_cells,)).

        """
        self._begin_step(dt_step)

        try:
            N_val = len(SL_vals)  # type: ignore[arg-type]
        except TypeError:
            SL_vals = np.full(self.num_cells, SL_vals)
            N_val = self.num_cells

        assert N_val == self.num_cells, "SL_vals length must match num_cells"

        try:
            N_val = len(Ca_val)  # type: ignore[arg-type]
        except TypeError:
            pass
        else:
            assert N_val == self.num_cells, (
                f"Ca_val length {N_val} must match num_cells {self.num_cells}"
            )

        self.update_probabilities(SL_vals, Ca_val)
        dt_acc = 0

        while dt_acc < dt_step:
            xODE2 = np.sum(self.xODE, axis=3)

            # Replicate xODE: (nu-2, L, C, R, T, Cell)
            xODErep = np.tile(self.xODE[:, :, :, :, np.newaxis, :], (1, 1, 1, 1, 4, 1))

            # --- PhiC ---
            self.PhiC[:] = self.PC[1 : self.nu - 1, ...] * xODErep

            # --- PhiL ---
            num_L = np.sum(self.PhiC[0 : self.nu - 3], axis=1)
            den_L = xODE2[1 : self.nu - 2, :, :, np.newaxis, :]
            ratio_L = np.divide(num_L, den_L, out=np.zeros_like(num_L), where=den_L != 0)
            ratio_L_exp = np.tile(ratio_L[:, :, :, np.newaxis, :, :], (1, 1, 1, 4, 1, 1))
            self.PhiL[1 : self.nu - 2] = ratio_L_exp

            # Boundary L
            bound_L = self.PC[0, 0, :, :, :, :]
            self.PhiL[0, :, :, :, :, :] = np.tile(bound_L[:, :, np.newaxis, :, :], (1, 1, 4, 1, 1))
            self.PhiL *= xODErep

            # --- PhiR ---
            num_R = np.sum(self.PhiC[1 : self.nu - 2], axis=3)
            den_R = xODE2[1 : self.nu - 2, :, :, np.newaxis, :]
            ratio_R = np.divide(num_R, den_R, out=np.zeros_like(num_R), where=den_R != 0)
            ratio_R_exp = np.tile(ratio_R[:, np.newaxis, ...], (1, 4, 1, 1, 1, 1))
            self.PhiR[0 : self.nu - 3] = ratio_R_exp

            # Boundary R
            bound_R = self.PC[self.nu - 1, :, :, 0, :, :]
            self.PhiR[self.nu - 3, :, :, :, :, :] = np.tile(
                bound_R[np.newaxis, :, :, :, :], (4, 1, 1, 1, 1)
            )
            self.PhiR *= xODErep

            # --- Update ---
            termC = np.swapaxes(self.PhiC, 2, 4) - self.PhiC
            termL = np.swapaxes(self.PhiL, 1, 4) - self.PhiL
            termR = np.swapaxes(self.PhiR, 3, 4) - self.PhiR
            flux = np.sum(termC + termL + termR, axis=4)

            xODEnew = self.xODE + self.dt * flux

            # Check for instability/bounds
            if np.min(xODEnew) < -1e-9 or np.max(xODEnew) > 1 + 1e-9:
                dt_curr = self.dt / 2.0
                xODEnew = self.xODE + dt_curr * flux
                xODEnew = np.clip(xODEnew, 0.0, 1.0)
            else:
                dt_curr = self.dt

            self.xODE = xODEnew
            dt_acc += dt_curr

    def _state_marginals(self) -> np.ndarray:
        """
        Per-node marginal distribution over the four RU states.

        ``xODE`` stores joint probabilities over neighbouring triplets, so a
        single node's marginal comes from a different axis depending on where
        it sits: the first node is read off the Left index of the first
        triplet, the last off the Right index of the last, and the interior
        nodes off the Center index of their own.

        The four states are ordered ``0N, 1N, 1P, 0P`` (see
        :meth:`update_probabilities`, where ``0N -> 1N`` carries ``Kon * Ca``):
        the leading digit is calcium bound, the trailing letter is
        permissive/non-permissive.

        Returns
        -------
        np.ndarray, shape (nu, 4, num_cells)
        """
        m0 = np.sum(np.sum(self.xODE[0], axis=1), axis=1)  # (Left, Cell)
        m_mid = np.sum(np.sum(self.xODE, axis=1), axis=2)  # (nu-2, Center, Cell)
        m_last = np.sum(np.sum(self.xODE[-1], axis=0), axis=0)  # (Right, Cell)
        return np.concatenate([m0[np.newaxis], m_mid, m_last[np.newaxis]], axis=0)

    def compute_permissivity(self) -> np.ndarray:
        """
        Compute the fraction of permissive states (1P + 0P) for each cell.
        """
        m = self._state_marginals()
        return np.mean(m[:, 2, :] + m[:, 3, :], axis=0)

    def bound_calcium_fraction(self) -> npt.NDArray[np.float64]:
        """
        Fraction of regulatory units with calcium bound, i.e. states 1N and 1P.

        The calcium counterpart of :meth:`compute_permissivity`, which selects
        the permissive states 1P and 0P from the same marginals.
        """
        m = self._state_marginals()
        return np.mean(m[:, 1, :] + m[:, 2, :], axis=0)

    def advance_step(
        self,
        dt: float,
        Ca_val: float | npt.NDArray[np.float64],
        SL_vals: float | npt.NDArray[np.float64],
        dSL_vals: float | npt.NDArray[np.float64] | None = None,
    ) -> None:
        """
        Integrate the model's internal ODEs forward by a single time step.
        (RDQ18 does not use dSL_vals).
        """
        self.advance_ODE(dt, Ca_val, SL_vals)

    def get_active_tension(self) -> npt.NDArray[np.float64]:
        """
        Compute and return the macroscopic active tension (Ta) generated.
        """
        return self.Ta_max * self.compute_permissivity()

    def get_active_stiffness(self) -> npt.NDArray[np.float64]:
        r"""
        Active stiffness, which is identically **zero** for this model.

        :math:`K_a = \nabla_r g \cdot \partial h/\partial\dot\lambda`, and no
        right-hand side in RDQ18 contains :math:`\dot\lambda` -- consistent
        with :meth:`advance_step` accepting ``dSL_vals`` purely for interface
        compatibility and ignoring it.

        .. warning::
            This is a real modelling property, not an implementation gap:
            **RDQ18 has no force-velocity (Hill) behaviour**. It reproduces
            the length dependence of force through ``Chi(SL)``, but a fibre
            shortening quickly generates exactly as much tension as an
            isometric one. Prefer :class:`~crossbridge.RDQ20MF`,
            :class:`~crossbridge.Land2017` or
            :class:`~crossbridge.Lewalle2024` when shortening velocity matters.

        The upside is that a segregated coupling to tissue mechanics is
        unconditionally stable for this model without any stabilization: the
        instability analysed by Regazzoni & Quarteroni is driven entirely by
        the strain-rate feedback that RDQ18 does not have.

        Returns
        -------
        np.ndarray, shape (num_cells,)
            An array of zeros.
        """
        return np.zeros(self.num_cells)

    def reset(self) -> None:
        """Reset model state to initial (fully non-permissive, unbound)."""
        self._prev_bound_ca = np.zeros(self.num_cells)
        self._last_dt = 0.0
        self.xODE[:] = 0.0
        self.xODE[:, 0, 0, 0, :] = 1.0
