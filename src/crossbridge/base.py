from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt


class CardiacActivationModel(ABC):
    """
    Abstract base class for all reduced-order cardiac activation models.
    Designed to support vectorized execution across multiple cells or
    integration points.
    """

    @abstractmethod
    def __init__(self, num_cells: int, Ta_max: float, params: dict | None = None):
        """
        Initialize the model state and precompute necessary constants.

        Parameters:
        -----------
        num_cells : int
            The number of independent spatial units to simulate simultaneously.
        Ta_max : float
            The maximum active tension scaling factor.
        params : dict, optional
            Model-specific parameters to override defaults.
        """
        self.num_cells = num_cells
        self.Ta_max = Ta_max

    @classmethod
    @abstractmethod
    def default_parameters(cls) -> dict:
        """
        Return a dictionary of the default physiological parameters for the model.
        """
        pass

    @abstractmethod
    def advance_step(
        self,
        dt: float,
        Ca_val: float | npt.NDArray[np.float64],
        SL_vals: float | npt.NDArray[np.float64],
        dSL_vals: float | npt.NDArray[np.float64] | None = None,
    ) -> None:
        """
        Integrate the model's internal states forward by a single time step.

        Parameters:
        -----------
        dt : float
            The time step size in seconds.
        Ca_val : float or np.ndarray
            Intracellular calcium concentration.
        SL_vals : float or np.ndarray
            Current sarcomere length(s).
        dSL_vals : float or np.ndarray, optional
            Current sarcomere shortening velocity. Defaults to 0 if not provided.
        """
        pass

    @abstractmethod
    def get_active_tension(self) -> npt.NDArray[np.float64]:
        """
        Compute and return the macroscopic active tension (Ta) generated.

        Returns:
        --------
        np.ndarray
            The active tension for each cell/integration point (shape: `num_cells`).
        """
        pass

    @abstractmethod
    def get_active_stiffness(self) -> npt.NDArray[np.float64]:
        r"""
        Compute the active stiffness :math:`K_a` of the current state.

        This is the sensitivity of the *rate* of active tension to the
        shortening velocity,

        .. math::
            K_a = \frac{\partial \dot{T_a}}{\partial \dot{\lambda}}
                = \nabla_r g \cdot \frac{\partial h}{\partial \dot{\lambda}}

        for a model written as :math:`\dot{r} = h(r, [Ca^{2+}]_i, \lambda,
        \dot\lambda)`, :math:`T_a = g(r, \lambda)` -- i.e. Eqs. (40)-(42) of
        Regazzoni & Quarteroni, *An oscillation-free fully partitioned scheme
        for the numerical modeling of cardiac active mechanics* (2020).

        Note that :math:`h` here is the **continuous** right-hand side, not
        the discrete update map implemented by `advance_step`; ``Ka`` is a
        property of the model, not of the integrator.

        Why this exists
        ---------------
        When a model in this package is coupled to a tissue mechanics solver,
        the two are usually advanced in a segregated (staggered) fashion. That
        scheme is unstable -- in fact not even convergent -- whenever the
        active stiffness exceeds the passive stiffness of the tissue, which is
        routine in contracting myocardium. Supplying ``Ka`` lets the caller
        add the consistent stabilization term of R&Q Eq. (19),

        .. math::
            P_{act} = \left[T_a + K_a (\lambda^{k+1} - \lambda^{k})\right]
                      \frac{F f_0 \otimes f_0}{|F f_0|},

        which is unconditionally stable. The same quantity is, to leading
        order in :math:`\Delta t`, the derivative :math:`dT_a/d\lambda` needed
        to couple the model *monolithically* through a Newton solve.

        Units and sign
        --------------
        Returned in the same tension units as :meth:`get_active_tension`
        (kPa), per unit of this model's dimensionless
        :math:`\Lambda = SL / SL_0`. Callers working in a different stretch
        variable must rescale by the chain rule: if :math:`SL = \lambda\,
        SL_{ref}` then :math:`K_a^{caller} = K_a \cdot SL_{ref}/SL_0`.

        ``Ka`` is non-negative for every model in this package -- attached
        crossbridges act as springs, never as an energy source.

        Returns
        --------
        np.ndarray
            The active stiffness for each cell/integration point
            (shape: `num_cells`).
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the model's internal state to its initial condition (as set by
        `__init__`), without re-allocating precomputed constants.
        """
        pass
