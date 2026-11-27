"""
Check `src/crossbridge/tran2010.ode` against the published CellML of Tran et al. (2010).

Downloads the model from the Physiome Model Repository into a temporary directory, converts it
with gotranx, generates Python for both models and compares, at random states, conditions and
metabolite levels, the derivatives of the nine states they share and the outputs `Ta`, `Tp` and
`bound_ca` (CellML: `F_active`, `ppforce`, `TropTot / Trop_conc`). The CellML has no `Ka`;
`tests/test_active_stiffness.py` checks that one by finite difference. Exit status: 0 if every
quantity agrees to rtol 1e-12 (atol 1e-12), 1 if not, 2 if the download fails. Run as
`python3 tools/compare_tran2010_cellml.py`.

**The one correction.** The CellML computes the proton concentration as `H = 10^7 / 10^pH` uM
(and `H_cons` likewise at pH 7.15), ten times [H+]; the file uses `10^(6 - pH)` uM (the class
docstring gives the evidence, the paper's Fig. 2B). The tool rewrites those two expressions in
the converted CellML before comparing, and asserts that each was there to rewrite once. Without
the rewrite, the derivatives of the troponin states (through the pH factor of `konT`) and of
`XBpostr` and the strains (through `fxbT`) differ, and `Ta` and `bound_ca` by about 1e-10
(through `fxb` in their normalization); the other derivatives and `Tp` agree.

The CellML is in ms and its sarcomere length is a state, so it is aligned to the file:

- rates in /ms (`kon`, `koffL`, `koffH`, `kn_p`, `kp_n`, `fapp`, `gapp`, `hf`, `hb`, `gxb`) are
  multiplied by 1000, and its `dSL` (a parameter, feeding `dSL_dt` and the strains) is fed in
  um/s, so its state derivatives come out in /s (um/s) with no further scaling;
- its metabolites are in uM (`MgATP`, `MgADP`, `xPi` and their `_cons`, `kdADP`) and its
  `kMgATP` in uM^2: the file's mM values times 1000 (10^6 for `kMgATP`); `kdHCa` is in uM in
  both;
- its temperature is in K, with rates given at 310 K: the file's `TmpC` + 273.15, and the file's
  rates are given at 36.85 degC;
- its `MgATP_cons`, `MgADP_cons`, `xPi_cons` are the file's `*_ref`; its reference pH, 7.15, is
  a number in `H_cons`, so the file's `pH_ref` is held at 7.15; its `hbmd` must be 1 and `hbmdc`
  unused, as the file has no such term; `SL` is a state of the CellML, set to the file's input.

Where the file departs from the CellML's literal forms, the samples keep clear of the
difference, and the tool asserts that they do: the exponent caps of `hfmd` and `gxbmd`
(`Rice2008._EXP_ARG_MAX`) do not bind; `Tropreg > 0`, so the file's `Max(Tropreg, 0)` and
`Max(permtot, 0.01)` are the CellML's forms; and the CellML's divisions by `|SL - SLrest|`,
`|xXBprer|` are defined. The file's strain coefficient `xPsi S / (a1p + a3m + a1m g3)` is the
CellML's `alpha2_plus xPsi / dutypostr` with `a2p` cancelled (`a3m = g3 a2p`), equal to
rounding.
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

URL = "https://models.physiomeproject.org/e/4b6/tran_2009.cellml"
ODE_FILE = Path(__file__).resolve().parents[1] / "src" / "crossbridge" / "tran2010.ode"

RTOL = 1e-12
ATOL = 1e-12
NUM = 200
SEED = 20100120

#: The CellML's rates in /ms, multiplied by 1000 into /s.
RATES = ("kon", "koffL", "koffH", "kn_p", "kp_n", "fapp", "gapp", "hf", "hb", "gxb")
#: The file's metabolites in mM -> the CellML's in uM.
METABOLITES = {
    "MgATP": "MgATP",
    "MgADP": "MgADP",
    "Pi": "xPi",
    "MgATP_ref": "MgATP_cons",
    "MgADP_ref": "MgADP_cons",
    "Pi_ref": "xPi_cons",
    "kdADP": "kdADP",
}
#: The states of the two models (the same names).
SHARED = ("P", "N_NoXB", "P_NoXB", "XBprer", "XBpostr", "TRPNCaH", "TRPNCaL", "xXBprer", "xXBpostr")
#: The CellML's proton concentration, and the correction.
H_FIX = {
    "H = 10000000.0/10.0**pH": "H = 1000000.0/10.0**pH",
    "H_cons = 10000000.0/10.0**7.15": "H_cons = 1000000.0/10.0**7.15",
}


def _module(ode) -> dict:
    """The gotranx Python code for `ode`, executed into a namespace (as tests/test_ode_files.py)."""
    code = gotran2py.get_code(ode, format=Format.none)
    namespace: dict = {}
    exec(code, namespace)
    return namespace


def _download(directory: Path) -> Path:
    path = directory / "tran2010.cellml"
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
        converted = Path(tmp) / "tran2010.ode"
        cellml_to_gotran(cellml_path).save(converted)
        text = converted.read_text()
        for old, new in H_FIX.items():
            assert text.count(old) == 1, f"expected the CellML's {old!r} once"
            text = text.replace(old, new)
        converted.write_text(text)
        # No singularity removal, so that the CellML's `x / |x|` stay what they are.
        cell = gotranx.load_ode(converted, remove_singularities=False)

    ref = _module(cell)
    mine = _module(gotranx.load_ode(ODE_FILE, remove_singularities=False))
    base = mine["init_parameter_values"]()
    value = {n: base[mine["parameter_index"](n)] for n in ("x_0", "hfmdc", "sigmap", "sigman")}
    assert base[mine["parameter_index"]("pH_ref")] == 7.15

    rng = np.random.default_rng(SEED)
    x_0 = value["x_0"]
    SL = rng.uniform(1.45, 2.35, NUM)
    dSL = rng.uniform(-400.0, 400.0, NUM)  # um/s
    Ca = rng.uniform(0.02, 20.0, NUM)  # uM
    TmpC = rng.uniform(15.0, 37.0, NUM)  # degC
    metabolites = {
        "MgATP": rng.uniform(0.1, 10.0, NUM),
        "MgADP": rng.uniform(1e-3, 1.0, NUM),
        "Pi": rng.uniform(0.1, 30.0, NUM),
        "MgATP_ref": rng.uniform(3.0, 7.0, NUM),
        "MgADP_ref": rng.uniform(0.02, 0.05, NUM),
        "Pi_ref": rng.uniform(1.0, 3.0, NUM),
        "kdADP": rng.uniform(0.002, 0.01, NUM),
    }
    pH = rng.uniform(6.0, 7.6, NUM)
    kdHCa = rng.uniform(0.01, 0.05, NUM)  # uM
    assert np.all(np.abs(SL - 1.85) > 1e-6) and np.all(np.abs(SL - 2.25) > 1e-6)

    frac = rng.dirichlet([1.0, 1.0, 1.0, 1.0], NUM)
    states = {
        "P": frac[:, 0],
        "XBprer": 0.3 * frac[:, 1],
        "XBpostr": 0.3 * frac[:, 2],
        "TRPNCaH": rng.uniform(0.01, 0.99, NUM),
        "TRPNCaL": rng.uniform(0.01, 0.99, NUM),
        "N_NoXB": rng.uniform(0.0, 1.0, NUM),
        "xXBprer": rng.uniform(-1.9 * x_0, 4.0 * x_0, NUM),
        "xXBpostr": x_0 + rng.uniform(-1.5 * x_0, 4.0 * x_0, NUM),
    }
    states["P_NoXB"] = 1.0 - states["N_NoXB"]

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

    y_mine = _states(mine, states)
    p_mine = _parameters(
        mine,
        {"Ca": Ca, "SL": SL, "dSL": dSL, "TmpC": TmpC, "pH": pH, "kdHCa": kdHCa, **metabolites},
    )

    cell_base = ref["init_parameter_values"]()
    assert cell_base[ref["parameter_index"]("hbmd")] == 1.0
    trop_conc = cell_base[ref["parameter_index"]("Trop_conc")]
    cell_params: dict = {n: cell_base[ref["parameter_index"](n)] * 1000.0 for n in RATES}
    cell_params.update({METABOLITES[k]: 1000.0 * v for k, v in metabolites.items()})
    kMgATP = base[mine["parameter_index"]("kMgATP")]
    cell_params.update(
        kMgATP=1e6 * kMgATP, TmpC=TmpC + 273.15, Cai=Ca, dSL=dSL, pH=pH, kdHCa=kdHCa, SEon=0.0
    )
    p_ref = _parameters(ref, cell_params)
    cell_states = {n: states[n] for n in SHARED}
    cell_states["SL"] = SL
    y_ref = _states(ref, cell_states)

    with np.errstate(all="raise"):
        d_mine = np.asarray(mine["rhs"](0.0, y_mine, p_mine))
        out_mine = mine["monitor_values"](0.0, y_mine, p_mine)
    # The CellML's `preload` is |x| / x of SLrest - SLset, finite here; `afterload` and `intf`
    # are not compared.
    with np.errstate(all="raise", invalid="ignore"):
        d_ref = np.asarray(ref["rhs"](0.0, y_ref, p_ref))
        out_ref = ref["monitor_values"](0.0, y_ref, p_ref)

    rows = []
    for name in SHARED:
        rows.append(
            (f"d{name}/dt", d_mine[mine["state_index"](name)], d_ref[ref["state_index"](name)])
        )
    m_ = lambda k: np.asarray(out_mine[mine["monitor_index"](k)])  # noqa: E731
    c_ = lambda k: np.asarray(out_ref[ref["monitor_index"](k)])  # noqa: E731
    rows.append(("Ta", m_("Ta"), c_("F_active")))
    rows.append(("Tp", m_("Tp"), c_("ppforce")))
    rows.append(("bound_ca", m_("bound_ca"), c_("TropTot") / trop_conc))

    failed = []
    print(f"{NUM} random states, rtol {RTOL:g}, atol {ATOL:g} (CellML's H corrected)")
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
