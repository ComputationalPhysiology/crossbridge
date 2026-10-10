"""`ODE_FILE` on every model, checked without gotranx."""

from crossbridge import MODEL_REGISTRY, CardiacActivationModel


def test_only_rdq18_has_no_ode_file():
    without = {name for name, cls in MODEL_REGISTRY.items() if cls.ODE_FILE is None}
    assert without == {"RDQ18"}
    assert CardiacActivationModel.ODE_FILE is None
    for cls in MODEL_REGISTRY.values():
        assert cls.ODE_FILE is None or cls.ODE_FILE.is_file()
