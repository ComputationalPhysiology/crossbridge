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
