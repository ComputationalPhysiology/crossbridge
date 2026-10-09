"""`src/crossbridge/rdq20mf.ode` is generated, so it is checked against its generator."""

import importlib.util
import pathlib

_REPO = pathlib.Path(__file__).resolve().parents[1]
_GENERATOR = _REPO / "tools" / "generate_rdq20mf_ode.py"
_GENERATED = _REPO / "src" / "crossbridge" / "rdq20mf.ode"


def test_rdq20mf_ode_matches_its_generator():
    """If this fails, src/crossbridge/rdq20mf.ode was edited by hand: change the generator and
    re-run `python3 tools/generate_rdq20mf_ode.py`."""
    spec = importlib.util.spec_from_file_location("generate_rdq20mf_ode", _GENERATOR)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    assert generator.main() == _GENERATED.read_text()
