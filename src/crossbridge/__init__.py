from .base import CardiacActivationModel
from .rdq18 import RDQ18
from .rdq20mf import RDQ20MF
from .lewalle2024 import Lewalle2024
from .land17 import Land2017
from . import utils
from .utils import calcium_trace, sl_trace

#: Maps a short model name to its class, so a model can be selected by name
#: (e.g. from a config file) instead of importing the class directly. Every
#: model here is a `CardiacActivationModel` and can be constructed uniformly
#: as `ModelClass(num_cells, Ta_max, params)`, so switching between them
#: requires no other code changes beyond the name/class used.
MODEL_REGISTRY: dict[str, type[CardiacActivationModel]] = {
    "RDQ18": RDQ18,
    "RDQ20MF": RDQ20MF,
    "Lewalle2024": Lewalle2024,
    "Land2017": Land2017,
}


def get_model(name: str) -> type[CardiacActivationModel]:
    """
    Look up a `CardiacActivationModel` subclass by name.

    Parameters
    ----------
    name : str
        One of the keys in `MODEL_REGISTRY` (e.g. "RDQ18", "RDQ20MF",
        "Lewalle2024").

    Returns
    -------
    type[CardiacActivationModel]
        The model class, ready to be instantiated as
        `get_model(name)(num_cells, Ta_max, params)`.
    """
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model {name!r}. Available models: {available}") from None


__all__ = [
    "CardiacActivationModel",
    "RDQ18",
    "RDQ20MF",
    "Lewalle2024",
    "Land2017",
    "MODEL_REGISTRY",
    "get_model",
    "utils",
    "calcium_trace",
    "sl_trace",
]
