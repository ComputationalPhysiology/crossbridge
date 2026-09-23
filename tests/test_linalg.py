"""
Tests for `crossbridge._linalg`, the batched solve and matrix exponential that
`Land2017` and `Lewalle2024` advance their linear sub-system with.

The property this module exists for is *per-matrix* scaling-and-squaring. A
single cell batch mixes a resting cell with an activated one, whose matrices
differ in norm by decades (the troponin term goes as ``CaTRPN ** (-nTm / 2)``),
and an exponential that picks one scaling exponent for the whole stack loses
the matrices far from the dominant norm -- which is why the models used to
loop over cells with `scipy.linalg.expm` instead of handing it the stack.

So the load-bearing tests here are:

  - `test_matches_scipy_mixed_scale`: agreement with the per-matrix reference
    on a batch whose norms span ten decades, i.e. the regime that breaks a
    naive batched call.
  - `test_matrix_is_independent_of_its_batch`: the same matrix, alone and
    buried in a 5000-matrix stack of wildly different norms, must come back
    *bit-identical*. This is what lets a caller batch at all, and it is the
    invariant `test_single_vs_batch_consistency_mixed_scale` leans on in the
    model suites.

`solve_batch` is the plainer half -- it has to agree with
`numpy.linalg.solve`, and it has to pivot: the models' matrices have a zero in
the leading position of a row often enough that an elimination without row
swaps would divide by zero on real input rather than in some contrived corner.
`test_solve_needs_pivoting` is the one that notices.

Both the numba kernels and the pure-numpy fallbacks are held to all of it:
the fixtures parametrize over whichever are available, so a machine without
numba still tests the path it will actually run.
"""

import ast
import importlib.util
import pathlib

import numpy as np
import pytest
from scipy.linalg import expm as scipy_expm

from crossbridge._linalg import (
    HAVE_NUMBA,
    expm_batch,
    expm_batch_numpy,
    solve_batch,
    solve_batch_numpy,
)

# The sizes the two models actually use; 4 is here to pin the generic path.
SIZES = [3, 5]


def _impls():
    impls = [pytest.param(expm_batch_numpy, id="numpy")]
    if HAVE_NUMBA:
        impls.append(pytest.param(expm_batch, id="numba"))
    return impls


def _solve_impls():
    impls = [pytest.param(solve_batch_numpy, id="numpy")]
    if HAVE_NUMBA:
        impls.append(pytest.param(solve_batch, id="numba"))
    return impls


@pytest.fixture(params=_impls())
def batched_expm(request):
    """Every available implementation of the same contract."""
    return request.param


@pytest.fixture(params=_solve_impls())
def batched_solve(request):
    """Every available implementation of the same contract."""
    return request.param


def _reference(A, h=1.0):
    """Per-matrix scipy exponential -- the loop the models used to run."""
    return np.stack([scipy_expm(A[i] * h) for i in range(A.shape[0])])


def _mixed_scale_batch(n, m, seed=0, decades=(-4, 6)):
    """Stable-ish random matrices whose norms span many orders of magnitude."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, m, m)) - 3.0 * np.eye(m)
    return A * (10.0 ** rng.uniform(*decades, size=n))[:, None, None]


def _max_rel_error(got, ref):
    scale = np.maximum(np.abs(ref).max(axis=(1, 2)), 1e-300)
    return float((np.abs(got - ref).max(axis=(1, 2)) / scale).max())


# ---------------------------------------------------------------------------
# Mathematical correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", SIZES)
def test_matches_scipy_mixed_scale(batched_expm, m):
    """The whole point: decade-spanning norms in one batch, all correct."""
    A = _mixed_scale_batch(400, m)
    got = batched_expm(A, 2e-4)
    assert _max_rel_error(got, _reference(A, 2e-4)) < 1e-9


@pytest.mark.parametrize("m", SIZES + [4])
def test_matches_scipy_benign(batched_expm, m):
    """Ordinary small-norm matrices, including a size no kernel covers."""
    A = _mixed_scale_batch(50, m, seed=3, decades=(-2, 1))
    assert _max_rel_error(batched_expm(A, 1.0), _reference(A)) < 1e-12


def test_zero_matrix_gives_identity(batched_expm):
    got = batched_expm(np.zeros((4, 3, 3)), 1e-3)
    np.testing.assert_allclose(got, np.broadcast_to(np.eye(3), (4, 3, 3)), atol=0.0)


def test_diagonal_matrix_is_exponential_of_diagonal(batched_expm):
    d = np.array([[-1.0, 0.0, 300.0], [-1e5, 1.0, 0.0]])
    A = np.stack([np.diag(row) for row in d])
    got = batched_expm(A, 1e-3)
    for i in range(2):
        np.testing.assert_allclose(got[i], np.diag(np.exp(d[i] * 1e-3)), rtol=1e-13)


def test_nilpotent_matrix_terminates_series(batched_expm):
    """exp(N) = I + N + N^2/2 exactly for a 3x3 strictly upper triangular N."""
    N = np.array([[[0.0, 2.0, 3.0], [0.0, 0.0, 5.0], [0.0, 0.0, 0.0]]])
    expected = np.eye(3) + N[0] + 0.5 * (N[0] @ N[0])
    np.testing.assert_allclose(batched_expm(N, 1.0)[0], expected, rtol=1e-13)


def test_h_is_folded_into_the_matrix(batched_expm):
    """expm_batch(A, h) must equal expm_batch(A * h)."""
    A = _mixed_scale_batch(64, 5, seed=7)
    np.testing.assert_allclose(batched_expm(A, 2e-4), batched_expm(A * 2e-4, 1.0), rtol=1e-14)


# ---------------------------------------------------------------------------
# Per-matrix independence -- the reason this module exists
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", SIZES)
def test_matrix_is_independent_of_its_batch(batched_expm, m):
    """A matrix alone and the same matrix inside a huge mixed batch agree exactly.

    Not `allclose`: exactly. Each matrix carries its own scaling exponent, so
    nothing about its neighbours can reach it. A batched implementation that
    shares one exponent fails this by many digits for the small-norm entries.
    """
    A = _mixed_scale_batch(5000, m, seed=11)
    batched = batched_expm(A, 2e-4)
    probes = [0, 1, 17, 2500, 4999]
    singles = np.concatenate([batched_expm(A[i : i + 1], 2e-4) for i in probes])
    np.testing.assert_array_equal(singles, batched[probes])


def test_squaring_count_really_varies_across_the_batch():
    """Guard the premise: the fixture batch spans many squaring counts.

    If this ever collapses to a single count, the test above stops testing
    anything and the module's reason for existing would have evaporated.
    """
    A = _mixed_scale_batch(5000, 3, seed=11) * 2e-4
    nrm = np.abs(A).sum(axis=2).max(axis=1)
    s = np.where(nrm > 0.5, np.ceil(np.log2(nrm / 0.5)), 0.0)
    assert s.min() == 0
    assert s.max() > 10


# ---------------------------------------------------------------------------
# Structural / API
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", SIZES)
def test_shape_and_dtype_preserved(batched_expm, m):
    out = batched_expm(np.zeros((7, m, m)), 1.0)
    assert out.shape == (7, m, m)
    assert out.dtype == np.float64


def test_empty_batch(batched_expm):
    assert batched_expm(np.zeros((0, 3, 3)), 1e-3).shape == (0, 3, 3)


def test_non_contiguous_input_is_accepted():
    """Callers may hand over a slice; the kernels need C-contiguous input."""
    A = np.asfortranarray(_mixed_scale_batch(16, 3, seed=5))
    np.testing.assert_allclose(expm_batch(A, 2e-4), _reference(A, 2e-4), rtol=1e-10)


def test_both_implementations_agree():
    """The fallback is a fallback, not a different answer."""
    if not HAVE_NUMBA:
        pytest.skip("numba not installed; only one implementation to compare")
    for m in SIZES:
        A = _mixed_scale_batch(500, m, seed=13)
        np.testing.assert_allclose(expm_batch(A, 2e-4), expm_batch_numpy(A, 2e-4), rtol=1e-9)


# ---------------------------------------------------------------------------
# solve_batch
# ---------------------------------------------------------------------------


def _model_like_matrices(n, m, seed):
    """Matrices shaped like the models': one row (and its RHS) orders of magnitude
    above the rest, which is what the fast troponin equilibrium looks like."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, m, m)) + m * np.eye(m)
    b = rng.normal(size=(n, m))
    big = (10.0 ** rng.uniform(0, 12, size=n))[:, None]
    A[:, 0, :] *= big
    b[:, 0] *= big[:, 0]
    return A, b


def _backward_error(A, x, b):
    """||Ax - b|| / (||A|| ||x||), the quantity an LU solve actually bounds.

    Not ||Ax - b|| / ||b||: these matrices have one row many orders of
    magnitude above the others, so that ratio is ~1e-4 for LAPACK itself and
    says nothing about the solver.
    """
    r = np.abs(np.einsum("nij,nj->ni", A, x) - b).max(axis=1)
    scale = np.abs(A).sum(axis=2).max(axis=1) * np.abs(x).max(axis=1)
    return float((r / np.maximum(scale, 1e-300)).max())


@pytest.mark.parametrize("m", SIZES + [4])
def test_solve_matches_numpy(batched_solve, m):
    """Agreement with LAPACK on matrices shaped like the models' own."""
    A, b = _model_like_matrices(300, m, seed=2)
    x = batched_solve(A, b)
    ref = np.linalg.solve(A, b[:, :, np.newaxis])[:, :, 0]
    assert (np.abs(x - ref).max(axis=1) / np.abs(ref).max(axis=1)).max() < 1e-12


@pytest.mark.parametrize("m", SIZES + [4])
def test_solve_is_backward_stable(batched_solve, m):
    """On matrices ill-conditioned enough that LAPACK's own answer is not the
    yardstick, the kernel must still be as good a solver as LAPACK is.

    Here only the row is scaled up, not its right-hand side, so the solution
    is genuinely ill-determined and two backward-stable solvers differ in the
    6th digit. Both are "right"; what is checkable is the backward error.
    """
    rng = np.random.default_rng(2)
    A = rng.normal(size=(300, m, m)) + m * np.eye(m)
    A[:, 0, :] *= 1e12
    b = rng.normal(size=(300, m))
    x = batched_solve(A, b)
    ref = np.linalg.solve(A, b[:, :, np.newaxis])[:, :, 0]
    assert _backward_error(A, x, b) < 1e-14
    assert _backward_error(A, x, b) < 20 * max(_backward_error(A, ref, b), 1e-18)


def test_solve_needs_pivoting(batched_solve):
    """A zero in the leading position: elimination without a row swap divides by it.

    The first matrix has a zero pivot only in column 0, the second only in
    column 1, so both swap positions are exercised.
    """
    A = np.array(
        [
            [[0.0, 2.0, 1.0], [3.0, 1.0, 0.0], [1.0, 1.0, 4.0]],
            [[2.0, 0.0, 1.0], [4.0, 0.0, 3.0], [1.0, 5.0, 1.0]],
        ]
    )
    b = np.array([[5.0, 4.0, 9.0], [3.0, 7.0, 8.0]])
    x = batched_solve(A, b)
    np.testing.assert_allclose(np.einsum("nij,nj->ni", A, x), b, rtol=1e-13)


def test_solve_raises_on_singular(batched_solve):
    """Same failure as `numpy.linalg.solve`, so the models' `except` still fires."""
    A = np.stack([np.eye(3), np.zeros((3, 3))])
    with pytest.raises(np.linalg.LinAlgError):
        batched_solve(A, np.ones((2, 3)))


def test_solve_is_independent_of_its_batch(batched_solve):
    """As with the exponential: no matrix may be influenced by its neighbours."""
    rng = np.random.default_rng(4)
    A = rng.normal(size=(2000, 5, 5)) + 5.0 * np.eye(5)
    A *= (10.0 ** rng.uniform(-6, 6, size=2000))[:, None, None]
    b = rng.normal(size=(2000, 5))
    batched = batched_solve(A, b)
    probes = [0, 3, 999, 1999]
    singles = np.concatenate([batched_solve(A[i : i + 1], b[i : i + 1]) for i in probes])
    np.testing.assert_array_equal(singles, batched[probes])


def test_solve_does_not_mutate_its_input(batched_solve):
    """The models reuse M for the exponential right after solving with it."""
    rng = np.random.default_rng(6)
    A = rng.normal(size=(32, 5, 5)) + 5.0 * np.eye(5)
    b = rng.normal(size=(32, 5))
    A_before, b_before = A.copy(), b.copy()
    batched_solve(A, b)
    np.testing.assert_array_equal(A, A_before)
    np.testing.assert_array_equal(b, b_before)


def test_solve_shape_and_empty(batched_solve):
    assert batched_solve(np.zeros((0, 3, 3)), np.zeros((0, 3))).shape == (0, 3)
    out = batched_solve(np.broadcast_to(np.eye(5), (9, 5, 5)).copy(), np.ones((9, 5)))
    assert out.shape == (9, 5)
    np.testing.assert_allclose(out, 1.0)


def test_solve_implementations_agree():
    if not HAVE_NUMBA:
        pytest.skip("numba not installed; only one implementation to compare")
    rng = np.random.default_rng(8)
    for m in SIZES:
        A = rng.normal(size=(400, m, m)) + m * np.eye(m)
        b = rng.normal(size=(400, m))
        ref = solve_batch_numpy(A, b)
        got = solve_batch(A, b)
        assert (np.abs(got - ref).max(axis=1) / np.abs(ref).max(axis=1)).max() < 1e-12


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

_REPO = pathlib.Path(__file__).resolve().parents[1]
_GENERATOR = _REPO / "tools" / "generate_linalg.py"
_GENERATED = _REPO / "src" / "crossbridge" / "_linalg.py"


@pytest.mark.skipif(
    not (_GENERATOR.exists() and _GENERATED.exists()),
    reason="running against an installed copy, without the repo's tools/",
)
def test_module_matches_its_generator():
    """The checked-in kernels must be exactly what `tools/generate_linalg.py` emits.

    The unrolled arithmetic is not reviewable by reading it -- nobody verifies
    125 multiply-adds by eye -- so what actually gets reviewed is the six-line
    emitter that writes them. This is what keeps the reviewed thing and the
    executed thing the same thing, and what makes it safe to regenerate.

    Parse trees are compared rather than text, so `ruff format` over the
    generated file is not a difference; docstrings are still compared, since
    they are AST constants.

    If this fails, `src/crossbridge/_linalg.py` was edited by hand: move the
    change into the generator and re-run `python3 tools/generate_linalg.py`.
    """
    spec = importlib.util.spec_from_file_location("generate_linalg", _GENERATOR)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    emitted = ast.dump(ast.parse(generator.main()))
    checked_in = ast.dump(ast.parse(_GENERATED.read_text()))
    if emitted != checked_in:
        # Not a bare assert: these are megabyte-scale strings, and pytest would
        # print both of them at you instead of the one useful sentence.
        pytest.fail(
            "src/crossbridge/_linalg.py is out of sync with tools/generate_linalg.py. "
            "Move the change into the generator and re-run `python3 tools/generate_linalg.py`."
        )
