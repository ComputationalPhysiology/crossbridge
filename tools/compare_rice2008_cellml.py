"""
Check `src/crossbridge/rice2008.ode` against the published CellML of Rice et al. (2008).

Downloads the model from the Physiome Model Repository into a temporary directory, converts it
with gotranx, generates Python for both models and compares, at random states, the derivatives
of the nine states they share and the outputs `Ta`, `Tp` and `bound_ca` (CellML: `active`,
`ppforce`, `TropTot / Trop_conc`). The CellML has no `Ka`; `tests/test_active_stiffness.py`
checks that one by finite difference. Exit status: 0 if every quantity agrees to rtol 1e-12
(atol 1e-12), 1 if not, 2 if the download fails. Run as `python3 tools/compare_rice2008_cellml.py`.

The CellML is in ms and its sarcomere length is a state with its own mechanics, so it is aligned
to the file:

- rates in /ms (`kon`, `koffL`, `koffH`, `kn_p`, `kp_n`, `fapp`, `gapp`, `hf`, `hb`, `gxb`) are
  multiplied by 1000, and its `dSL` is fed in um/s, so its state derivatives come out in /s
  (um/s) with no further scaling. (Scaling the rates and also the derivatives would count the
  factor twice, and rates in /s beside a `dSL` in um/ms in `dxXBprer` would be inconsistent; this
  is the brief's "rates x 1000, derivatives x 1000, `dSL` / 1000" with the time unit chosen once);
- `SLrest` (1.85 in the CellML) and `TmpC` (24) are set to the file's (1.9, 22.5), where the
  paper's and the authors' code's values differ from the CellML's; `hbmdc` must be 0, as the
  file has no such term;
- `N_` is `N`; the other shared states keep their names;
- the CellML's `Cai` is `Ca_diastolic` while `time < start_time`, so `Ca_diastolic = Ca` and
  `time = 0` feed it the file's calcium;
- its `dSL` is `(intf - visc (SL - SLset)) / massf` , so `intf` is set to make that equal
  the file's `dSL`.

Where the file departs from the CellML's literal forms, the samples keep clear of the
difference, and the tool asserts that they do:

- the strain coefficients `xPsi D / (fappT (hbT + gxbT))` and `xPsi D / fappT` are algebraically
  the CellML's `xPsi / dutyprer` and `xPsi hfT / dutypostr`, equal to rounding (no mapping);
- the file caps the exponents of `hfmd` and `gxbmd` at `Rice2008`'s `_EXP_ARG_MAX` (20), which
  the CellML does not (`tests/test_ode_files.py::test_rice2008_derivatives_match_its_ode` holds
  the file's cap to the class's). The strains are sampled where the arguments, computed from the
  file's `hfmdc`, `sigmap` and `sigman`, stay below it: `xXBprer >= -1.9 x_0` (18.05 at most),
  `x_0 - xXBpostr <= 1.5 x_0` (the `sigmap` branch, 18) and `xXBpostr - x_0 <= 4 x_0` (the
  `sigman` branch, 16), each branch and both signs of `xXBprer` sampled;
- the file's `permtot` and `kp_nT` (`Max(Tropreg, 0)`, `Max(permtot, 0.01)`) equal the CellML's
  for `Tropreg > 0`, which the troponin samples guarantee;
- the CellML's titin term divides by `|SL - SLrest|`, NaN at `SL == SLrest`, and its strains are
  divided by `|xXBprer|` and `|xXBpostr - x_0|`; the samples avoid those points, and `SL_c`.
"""

import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import gotranx
import numpy as np
from gotranx.cli import gotran2py
from gotranx.codegen.python import Format
from gotranx.myokit import cellml_to_gotran

from crossbridge.rice2008 import _EXP_ARG_MAX

URL = (
    "https://models.physiomeproject.org/workspace/rice_wang_bers_detombe_2008/"
    "rawfile/HEAD/rice_wang_bers_detombe_2008.cellml"
)
ODE_FILE = Path(__file__).resolve().parents[1] / "src" / "crossbridge" / "rice2008.ode"

RTOL = 1e-12
ATOL = 1e-12
NUM = 200
SEED = 20081201

#: The CellML's rates in /ms, multiplied by 1000 into /s.
RATES = ("kon", "koffL", "koffH", "kn_p", "kp_n", "fapp", "gapp", "hf", "hb", "gxb")
#: The states of the two models (the file's name -> the CellML's).
SHARED = {
    "N": "N_",
    "N_NoXB": "N_NoXB",
    "P_NoXB": "P_NoXB",
    "XBprer": "XBprer",
    "XBpostr": "XBpostr",
    "TRPNCaH": "TRPNCaH",
    "TRPNCaL": "TRPNCaL",
    "xXBprer": "xXBprer",
    "xXBpostr": "xXBpostr",
}


def _module(ode) -> dict:
    """The gotranx Python code for `ode`, executed into a namespace (as tests/test_ode_files.py)."""
    code = gotran2py.get_code(ode, format=Format.none)
    namespace: dict = {}
    exec(code, namespace)
    return namespace


def _download(directory: Path) -> Path:
    path = directory / "rice2008.cellml"
    with urllib.request.urlopen(URL, timeout=60) as response:
        path.write_bytes(response.read())
    return path


def _states(m: dict, values: dict[str, np.ndarray]) -> np.ndarray:
    y = np.zeros((len(m["init_state_values"]()), NUM))
    for name, v in values.items():
        y[m["state_index"](name)] = v
    return y


def _parameters(m: dict, values: dict[str, float | np.ndarray]) -> np.ndarray:
    base = m["init_parameter_values"]()
    p = np.repeat(np.asarray(base, dtype=float)[:, None], NUM, axis=1)
    for name, v in values.items():
        p[m["parameter_index"](name)] = v
    return p


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        try:
            cellml_path = _download(Path(tmp))
        except (urllib.error.URLError, OSError) as error:
            print(f"Could not download the CellML from {URL}: {error}")
            return 2
        # Saved and loaded again, as gotranx needs for a converted file; no singularity removal,
        # so that the CellML's `x / |x|` stay what they are.
        cellml_to_gotran(cellml_path).save(Path(tmp) / "rice2008.ode")
        cell = gotranx.load_ode(Path(tmp) / "rice2008.ode", remove_singularities=False)

    ref = _module(cell)
    mine_ode = gotranx.load_ode(ODE_FILE, remove_singularities=False)
    mine = _module(mine_ode)

    # The file's own values for what the CellML disagrees on.
    base = mine["init_parameter_values"]()
    names = ("SLrest", "TmpC", "x_0", "hfmdc", "sigmap", "sigman")
    value = {n: base[mine["parameter_index"](n)] for n in names}

    rng = np.random.default_rng(SEED)
    x_0 = value["x_0"]
    SL = rng.uniform(1.45, 2.35, NUM)
    dSL = rng.uniform(-400.0, 400.0, NUM)  # um/s
    Ca = rng.uniform(0.02, 2.0, NUM)  # uM
    # SL exactly at SLrest or SL_c is a measure-zero event, but say so.
    assert np.all(np.abs(SL - value["SLrest"]) > 1e-6) and np.all(np.abs(SL - 2.25) > 1e-6)

    # Two crossbridge states and N in the simplex, troponin in (0, 1): Tropreg > 0.
    frac = rng.dirichlet([1.0, 1.0, 1.0, 1.0], NUM)
    states = {
        "N": frac[:, 0],
        "XBprer": 0.3 * frac[:, 1],
        "XBpostr": 0.3 * frac[:, 2],
        "TRPNCaH": rng.uniform(0.01, 0.99, NUM),
        "TRPNCaL": rng.uniform(0.01, 0.99, NUM),
        "N_NoXB": rng.uniform(0.0, 1.0, NUM),
        "xXBprer": rng.uniform(-1.9 * x_0, 4.0 * x_0, NUM),
        "xXBpostr": x_0 + rng.uniform(-1.5 * x_0, 4.0 * x_0, NUM),
    }
    states["P_NoXB"] = 1.0 - states["N_NoXB"]

    # The caps of the file's exponents cannot bind, on either branch of gxbmd, and the CellML's
    # divisions are defined.
    xp, xq = states["xXBprer"], states["xXBpostr"]
    hf_arg = -value["hfmdc"] * xp * np.abs(xp) / x_0**2
    gxb_arg = ((xq - x_0) / x_0) ** 2
    below = xq < x_0
    assert np.any(xp < 0.0) and np.any(xp > 0.0), "xXBprer is not sampled on both sides of 0"
    assert np.any(below) and np.any(~below), "xXBpostr is not sampled on both sides of x_0"
    assert np.all(hf_arg < _EXP_ARG_MAX), "hfmd argument reaches the cap"
    assert np.all(value["sigmap"] * gxb_arg[below] < _EXP_ARG_MAX), "gxbmd sigmap reaches the cap"
    assert np.all(value["sigman"] * gxb_arg[~below] < _EXP_ARG_MAX), "gxbmd sigman reaches the cap"
    assert np.all(xp != 0.0) and np.all(xq != x_0)

    # The file: inputs are parameters.
    y_mine = _states(mine, states)
    p_mine = _parameters(mine, {"Ca": Ca, "SL": SL, "dSL": dSL})

    # The CellML, aligned.
    cell_base = ref["init_parameter_values"]()
    cell_params = {n: cell_base[ref["parameter_index"](n)] * 1000.0 for n in RATES}
    assert cell_base[ref["parameter_index"]("hbmdc")] == 0.0
    visc, massf, SLset, trop_conc = (
        cell_base[ref["parameter_index"](n)] for n in ("visc", "massf", "SLset", "Trop_conc")
    )
    cell_params.update(SLrest=value["SLrest"], TmpC=value["TmpC"], Ca_diastolic=Ca, start_time=5.0)
    p_ref = _parameters(ref, cell_params)
    cell_states = {SHARED[n]: states[n] for n in SHARED}
    cell_states["SL"] = SL
    cell_states["intf"] = dSL * massf + visc * (SL - SLset)
    y_ref = _states(ref, cell_states)

    with np.errstate(all="raise"):
        d_mine = np.asarray(mine["rhs"](0.0, y_mine, p_mine))
        out_mine = mine["monitor_values"](0.0, y_mine, p_mine)
    # The CellML's `preload` is 0/0 once SLrest == SLset (both 1.9); it feeds only `dintf_dt`,
    # which is not compared, and the finiteness of what is compared is checked below.
    with np.errstate(all="raise", invalid="ignore"):
        d_ref = np.asarray(ref["rhs"](0.0, y_ref, p_ref))
        out_ref = ref["monitor_values"](0.0, y_ref, p_ref)

    # The CellML's dSL, with the aligned intf, is the file's.
    dsl = np.asarray(out_ref[ref["monitor_index"]("dSL")])
    assert np.allclose(dsl, dSL, rtol=1e-12, atol=1e-9), "intf does not give the file's dSL"

    rows = []
    for name, cname in SHARED.items():
        a = d_mine[mine["state_index"](name)]
        b = d_ref[ref["state_index"](cname)]
        rows.append((f"d{name}/dt", a, b))
    m_ = lambda k: np.asarray(out_mine[mine["monitor_index"](k)])  # noqa: E731
    c_ = lambda k: np.asarray(out_ref[ref["monitor_index"](k)])  # noqa: E731
    rows.append(("Ta", m_("Ta"), c_("active")))
    rows.append(("Tp", m_("Tp"), c_("ppforce")))
    rows.append(("bound_ca", m_("bound_ca"), c_("TropTot") / trop_conc))

    failed = []
    print(f"{NUM} random states, rtol {RTOL:g}, atol {ATOL:g}")
    for name, a, b in rows:
        a, b = np.broadcast_to(a, (NUM,)), np.broadcast_to(b, (NUM,))
        scale = np.maximum(np.abs(a), np.abs(b))
        rel = np.abs(a - b) / np.maximum(scale, ATOL)
        ok = np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.allclose(a, b, RTOL, ATOL)
        print(
            f"{name:12s} max rel diff {np.max(rel):9.3e}   max |value| {np.max(scale):9.3e}   "
            f"{'ok' if ok else 'MISMATCH'}"
        )
        if not ok:
            failed.append(name)
    if failed:
        print("Mismatches: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
