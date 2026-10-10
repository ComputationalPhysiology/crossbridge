from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar
import numpy as np
import numpy.typing as npt


class CardiacActivationModel(ABC):
    """
    Abstract base class for all reduced-order cardiac activation models.
    Designed to support vectorized execution across multiple cells or
    integration points.
    """

    #: The model's parameters: `default_parameters()` updated with `params`. Set by
    #: each model's `__init__`.
    p: dict

    #: This model as a gotranx `.ode` file, or None.
    ODE_FILE: ClassVar[Path | None] = None

    #: Names of the attributes, beyond the base ones (`_prev_bound_ca`,
    #: `_last_dt`), that make up the model's evolving state. Each model sets its
    #: own; `get_state` and `set_state` work over them.
    _state_names: tuple[str, ...] = ()

    @abstractmethod
    def __init__(self, num_cells: int, Ta_max: float = 1.0, params: dict | None = None):
        """
        Initialize the model state and precompute necessary constants.

        Parameters:
        -----------
        num_cells : int
            The number of independent spatial units to simulate simultaneously.
        Ta_max : float, optional
            The maximum active tension scaling factor. Each model sets its own
            default (e.g. 100 for RDQ18); the one here only marks it optional.
        params : dict, optional
            Model-specific parameters to override defaults.
        """
        self.num_cells = num_cells
        self.Ta_max = Ta_max
        self._prev_bound_ca = np.zeros(int(num_cells))
        self._last_dt = 0.0

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
    def bound_calcium_fraction(self) -> npt.NDArray[np.float64]:
        """
        Fraction of this model's calcium binding sites currently occupied.

        Dimensionless, in [0, 1], one value per cell. For the Land-family
        models this is the troponin-C occupancy `CaTRPN`; for the RU-tensor
        models (RDQ18, RDQ20MF) it is the marginal probability that a
        regulatory unit has calcium bound.

        Returns
        -------
        np.ndarray
            Occupied fraction for each cell/integration point
            (shape: `num_cells`).
        """
        pass

    def _begin_step(self, dt: float) -> None:
        """
        Record the pre-step calcium occupancy. Call at the top of the model's
        stepping entry point, before any state is modified; this is what makes
        :meth:`get_calcium_binding_rate` work.
        """
        self._prev_bound_ca = np.asarray(self.bound_calcium_fraction(), dtype=float).copy()
        self._last_dt = float(dt)

    def get_calcium_binding_rate(self) -> npt.NDArray[np.float64]:
        r"""
        Rate at which this model sequestered calcium over the last step, as a
        fraction of its binding sites per second.

        Why this exists
        ---------------
        These models bind cytosolic calcium. When one is coupled to an
        electrophysiology model that has had its own troponin buffer removed --
        the usual arrangement, since otherwise calcium is buffered twice -- the
        EP side needs that buffering flux back, or its calcium transient is
        unbuffered and comes out too large and too fast. Nothing raises; the
        answer is simply wrong.

        In the ToR-ORd cell model this term appears as

        .. math::
            J_{TRPN} = \frac{d\,CaTRPN}{dt}\, [TRPN]_{max}, \qquad
            \frac{d\,ca_i}{dt} = B_{ca_i}\left(\ldots - J_{TRPN}\right)

        so multiply this rate by your total troponin concentration to obtain
        :math:`J_{TRPN}` in concentration per second.

        What is returned
        ----------------
        The **mean rate over the step just taken**,
        :math:`(\theta^{n+1} - \theta^{n})/\Delta t`, not the instantaneous
        derivative. That is deliberate: it makes the reported flux exactly the
        calcium the model actually absorbed over the step, so a segregated
        coupling conserves calcium instead of leaking it at
        :math:`\mathcal{O}(\Delta t)`. The two agree as
        :math:`\Delta t \to 0`.

        Zero before the first step, and after :meth:`reset`.

        Returns
        -------
        np.ndarray, shape (num_cells,)
            Binding rate in units of occupied fraction per second.
        """
        if self._last_dt <= 0.0:
            return np.zeros(self.num_cells)
        return (self.bound_calcium_fraction() - self._prev_bound_ca) / self._last_dt

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the model's internal state to its initial condition (as set by
        `__init__`), without re-allocating precomputed constants.
        """
        pass

    def get_state(self) -> dict[str, npt.NDArray[np.float64] | float | int | bool]:
        """
        Return a copy of everything that evolves in the model.

        The keys are `_prev_bound_ca` and `_last_dt`, followed by the model's
        `_state_names`. Arrays are copies with the cell axis **last**; scalars
        are plain Python ``float``, ``int`` or ``bool``. Restoring the result
        with :meth:`set_state` on a model of the same class, `num_cells` and
        parameters makes it advance bit-identically to this one. Parameters
        and precomputed constants are not state.

        Returns
        -------
        dict
            State name to a copy of its value.
        """
        state: dict[str, npt.NDArray[np.float64] | float | int | bool] = {}
        for name in ("_prev_bound_ca", "_last_dt", *self._state_names):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)):
                state[name] = bool(value)
            elif isinstance(value, (int, np.integer)):
                state[name] = int(value)
            elif np.ndim(value) == 0:
                state[name] = float(value)
            else:
                state[name] = np.array(value, dtype=float, copy=True)
        return state

    def set_state(self, state: Mapping[str, npt.NDArray[np.float64] | float | int | bool]) -> None:
        """
        Restore a state taken by :meth:`get_state`.

        Parameters
        ----------
        state : Mapping
            Must have exactly the keys `get_state` returns.

        Raises
        ------
        KeyError
            Naming every missing and every unexpected key.
        ValueError
            Naming any key whose array shape differs from the model's.
        """
        names = ("_prev_bound_ca", "_last_dt", *self._state_names)
        missing = [n for n in names if n not in state]
        unexpected = [k for k in state if k not in names]
        if missing or unexpected:
            raise KeyError(f"set_state: missing keys {missing}, unexpected keys {unexpected}")
        new: dict[str, npt.NDArray[np.float64] | float | int | bool] = {}
        for name in names:
            current = getattr(self, name)
            value = state[name]
            if isinstance(current, (bool, np.bool_)):
                new[name] = bool(value)
            elif isinstance(current, (int, np.integer)):
                new[name] = int(value)  # type: ignore[arg-type]
            elif np.ndim(current) == 0:
                new[name] = float(value)  # type: ignore[arg-type]
            else:
                arr = np.array(value, dtype=float, copy=True)
                if arr.shape != np.shape(current):
                    raise ValueError(
                        f"set_state: {name!r} has shape {arr.shape}, expected {np.shape(current)}"
                    )
                new[name] = arr
        for name, value in new.items():
            setattr(self, name, value)
