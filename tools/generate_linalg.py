"""Generate `src/crossbridge/_linalg.py`.

The numba kernels in that module have their 3x3 / 5x5 matrix arithmetic
unrolled into scalars, which is what makes them fast (a kernel that keeps the
matrices as arrays and uses `@` allocates a temporary per cell and is no
faster than scipy) and what makes them far too repetitive to maintain by hand.
So they are written by this script instead:

    python3 tools/generate_linalg.py

which rewrites the module and runs `ruff format` over it. Change the algorithm
here -- both sizes then stay in step by construction -- never in the generated
file. `tests/test_linalg.py` covers the result.
"""

from math import factorial

TOL = 1e-17  # truncation-error budget for the Taylor series
THETA = 0.5  # scaled inf-norm target
MAX_ORDER = 16


def order_table():
    """(threshold, order) pairs: smallest order whose bound nrm**(k+1)/(k+1)! <= TOL."""
    rows = []
    for k in range(2, MAX_ORDER + 1):
        thr = (TOL * factorial(k + 1)) ** (1.0 / (k + 1))
        rows.append((thr, k))
        if thr > THETA:
            break
    return rows


def idx(m):
    return [(j, k) for j in range(m) for k in range(m)]


def matmul(m, out, lhs, rhs, suffix=""):
    """Unrolled out[j,k] = sum_l lhs[j,l]*rhs[l,k] (+ optional scaling suffix)."""
    lines = []
    for j, k in idx(m):
        terms = " + ".join(f"{lhs}{j}{l} * {rhs}{l}{k}" for l in range(m))
        lines.append(f"{out}{j}{k} = ({terms}){suffix}" if suffix else f"{out}{j}{k} = {terms}")
    return lines


def solve_kernel(m):
    """Unrolled Gaussian elimination with partial pivoting, one RHS per matrix.

    Same algorithm LAPACK's `gesv` runs (and so `numpy.linalg.solve`), just
    without the per-matrix call overhead: the augmented row `a{r}{m}` is the
    right-hand side, carried through elimination and back substitution.
    """
    ind = " " * 8
    L = []
    # error_model="numpy": a zero pivot must produce inf and be caught by the
    # `singular` counter, not raise ZeroDivisionError out of the middle of a batch.
    L.append('@_njit(cache=True, nogil=True, error_model="numpy")')
    L.append(f"def _solve{m}_kernel(A, b, out):  # pragma: no cover - numba-compiled")
    L.append(f'    """Hand-unrolled batched solve of {m}x{m} A[i] x = b[i]."""')
    L.append("    singular = 0")
    L.append("    for i in range(A.shape[0]):")
    L.append(ind + "# --- load the augmented matrix [A | b] ---")
    for r in range(m):
        for c in range(m):
            L.append(ind + f"a{r}{c} = A[i, {r}, {c}]")
        L.append(ind + f"a{r}{m} = b[i, {r}]")
    for j in range(m - 1):
        L.append("")
        L.append(ind + f"# --- column {j}: pivot, then eliminate below ---")
        for r in range(j + 1, m):
            L.append(ind + f"if abs(a{r}{j}) > abs(a{j}{j}):")
            for c in range(j, m + 1):
                L.append(ind + f"    a{j}{c}, a{r}{c} = a{r}{c}, a{j}{c}")
        L.append(ind + f"if a{j}{j} == 0.0:")
        L.append(ind + "    singular += 1")
        for r in range(j + 1, m):
            L.append(ind + f"f{r} = a{r}{j} / a{j}{j}")
            for c in range(j + 1, m + 1):
                L.append(ind + f"a{r}{c} -= f{r} * a{j}{c}")
    L.append("")
    L.append(ind + "# --- back substitution ---")
    L.append(ind + f"if a{m - 1}{m - 1} == 0.0:")
    L.append(ind + "    singular += 1")
    for r in range(m - 1, -1, -1):
        rhs = f"a{r}{m}"
        for c in range(r + 1, m):
            rhs += f" - a{r}{c} * x{c}"
        expr = f"({rhs})" if r < m - 1 else rhs
        L.append(ind + f"x{r} = {expr} / a{r}{r}")
    for r in range(m):
        L.append(ind + f"out[i, {r}] = x{r}")
    L.append("    return singular")
    return "\n".join(L)


def kernel(m):
    ind = " " * 8
    L = []
    a = lambda j, k: f"a{j}{k}"  # noqa: E731

    L.append("@_njit(cache=True, fastmath=True, nogil=True)")
    L.append(f"def _expm{m}_kernel(A, h, out):  # pragma: no cover - numba-compiled")
    L.append(f'    """Hand-unrolled batched exp(A[i] * h) for {m}x{m} matrices."""')
    L.append("    for i in range(A.shape[0]):")
    L.append(ind + "# --- load and scale by the sub-step h ---")
    for j, k in idx(m):
        L.append(ind + f"{a(j, k)} = A[i, {j}, {k}] * h")
    L.append("")
    L.append(ind + "# --- per-matrix squaring count from this matrix's own inf-norm ---")
    for j in range(m):
        L.append(ind + f"r{j} = " + " + ".join(f"abs({a(j, k)})" for k in range(m)))
    L.append(ind + "nrm = " + "max(" * (m - 1) + "r0" + "".join(f", r{j})" for j in range(1, m)))
    L.append(ind + "s = 0")
    L.append(ind + "while nrm > _THETA and s < _S_MAX:")
    L.append(ind + "    nrm *= 0.5")
    L.append(ind + "    s += 1")
    L.append(ind + "if s > 0:")
    L.append(ind + "    f = 2.0 ** (-s)")
    for j, k in idx(m):
        L.append(ind + f"    {a(j, k)} *= f")
    L.append("")
    L.append(ind + "# --- Taylor series: E = I + A + A^2/2! + ... , P the running term ---")
    for j, k in idx(m):
        one = "1.0 + " if j == k else ""
        L.append(ind + f"e{j}{k} = {one}{a(j, k)}")
    for j, k in idx(m):
        L.append(ind + f"p{j}{k} = {a(j, k)}")
    L.append(ind + "for k in range(2, _taylor_order(nrm) + 1):")
    L.append(ind + "    inv = 1.0 / k")
    for line in matmul(m, "t", "p", "a", suffix=" * inv"):
        L.append(ind + "    " + line)
    for j, k in idx(m):
        L.append(ind + f"    p{j}{k} = t{j}{k}")
    for j, k in idx(m):
        L.append(ind + f"    e{j}{k} += p{j}{k}")
    L.append("")
    L.append(ind + "# --- undo the scaling: s repeated squarings ---")
    L.append(ind + "for _ in range(s):")
    for line in matmul(m, "t", "e", "e"):
        L.append(ind + "    " + line)
    for j, k in idx(m):
        L.append(ind + f"    e{j}{k} = t{j}{k}")
    L.append("")
    for j, k in idx(m):
        L.append(ind + f"out[i, {j}, {k}] = e{j}{k}")
    return "\n".join(L)


HEADER = '''"""
Fast batched linear algebra for the small dense systems of the Land-family
models (`Land2017`: 3x3, `Lewalle2024`: 5x5).

Both models advance their (B, S, W, ...) sub-system one sub-step at a time by
solving for its frozen-coefficient steady state and then applying an exact
matrix exponential -- `solve_batch` and `expm_batch` here, one matrix per cell.
Together they were the whole cost of `advance_step` at organ scale.

`expm_batch` is the interesting one. At whole-organ scale (tens of thousands
of cells per MPI rank, ten sub-steps per 2 ms caller step), exponentiating
31920 matrices one at a time through `scipy.linalg.expm` cost ~170 ms per
sub-step and was ~90% of `advance_step`.

Handing the whole stack to `scipy.linalg.expm` at once is not the fix. What
these models need from a batched exponential is that **each matrix keeps its
own scaling-and-squaring exponent**: the norms in one cell batch span decades
(the troponin term scales as ``CaTRPN ** (-nTm / 2)``, so a resting and an
activated cell differ by orders of magnitude), and squaring them all to the
batch maximum is both inaccurate and prone to overflow. Older scipy releases
were found to return silently wrong entries for such a stack -- the reason the
models looped per cell in the first place. Current scipy (1.17) is accurate,
because its stacked mode is itself a Python loop over the stack, so it is only
~1.5x faster than the explicit loop and does not solve the cost problem
either.

So this module implements scaling-and-squaring directly, keeping the per-matrix
exponent. Each matrix is scaled to inf-norm <= `_THETA`, exponentiated by a
Taylor series whose order comes from that matrix's own norm, then squared back
up on its own. A cell therefore gets bit-identical treatment whether it is
passed alone or inside a batch of 30000 -- see `tests/test_linalg.py`, and
`test_single_vs_batch_consistency_mixed_scale` in the model suites.

Two implementations of that one algorithm:

* a numba kernel with the 3x3 / 5x5 arithmetic **unrolled into scalars**
  (hence the wall of straight-line code below, which is written by
  `tools/generate_linalg.py` -- change the algorithm there and regenerate,
  do
  not hand-edit the kernels). The unrolling is the
  point, not the `@njit`: a kernel that keeps the matrices as arrays and uses
  `@` allocates a temporary per cell and is no faster than scipy (1.14x
  measured), while the unrolled form keeps every entry in a register.
* a pure-numpy batched fallback for when numba is not installed. It does the
  squarings on a stack sorted by exponent, so the count still varies per
  matrix.

`solve_batch` has no such subtlety -- it is `numpy.linalg.solve` restricted to
one small matrix and one right-hand side per cell: the same Gaussian
elimination with partial pivoting, minus the per-matrix LAPACK dispatch that
dominates at this size (12x for 3x3, 7.5x for 5x5). It exists only because
that dispatch became the next bottleneck once the exponential stopped being
one. It raises `numpy.linalg.LinAlgError` on an exactly singular matrix just
as numpy does, so callers that catch it keep working; on the merely
ill-conditioned matrices these models produce it is backward stable, like any
LU solve, which is all either implementation promises.

Measured on the real 31920-cell batches the models produce (inf-norms spanning
0.4 to 2e4, 0 to 16 squarings), against the per-matrix `scipy.linalg.expm`
loop: 70x for `Land2017`'s 3x3 and 25x for `Lewalle2024`'s 5x5 with numba,
6.5x and 3x with the numpy fallback, all agreeing with scipy to <= 2e-11
relative. Together with `solve_batch`, `advance_step` at 31920 cells goes from
~1.8 s to ~47 ms (`Land2017`) and from ~2.7 s to ~165 ms (`Lewalle2024`); over a
1 s twitch the resulting Ta and Ka track the old per-cell code to 1e-11
relative.
"""

from __future__ import annotations

import os

import numpy as np
import numpy.typing as npt

__all__ = ["expm_batch", "expm_batch_numpy", "solve_batch", "solve_batch_numpy", "HAVE_NUMBA"]

#: Matrices are scaled down until their inf-norm is at most this, so the
#: Taylor series below converges in a handful of terms.
_THETA = 0.5

#: Hard cap on the squaring count, so a non-finite norm cannot spin forever.
_S_MAX = 128

#: Taylor order that covers the worst case allowed by `_THETA`; see
#: `_taylor_order`, which picks a cheaper one per matrix in the kernels.
_MAX_ORDER = __MAX_ORDER__

#: Set to 1 to force the pure-numpy path (used by the tests to exercise it).
_FORCE_NUMPY = os.environ.get("CROSSBRIDGE_NO_NUMBA", "") not in ("", "0", "false", "False")

try:  # pragma: no cover - depends on the environment, both paths are tested
    from numba import njit as _njit

    HAVE_NUMBA = not _FORCE_NUMPY
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def _njit(*args, **kwargs):  # type: ignore[misc]
        """No-op stand-in so the kernels below stay importable without numba."""

        def wrap(func):
            return func

        return wrap


@_njit(cache=True, fastmath=True, inline="always")
def _taylor_order(nrm: float) -> int:  # pragma: no cover - numba-compiled
    """Smallest Taylor order whose truncation bound nrm**(k+1)/(k+1)! is negligible.

    `nrm` is the *scaled* inf-norm, so it never exceeds `_THETA` and the last
    branch below is an upper bound, not a fallback.
    """
'''


def main():
    parts = [HEADER]
    rows = order_table()
    for thr, k in rows[:-1]:
        parts[-1] += f"    if nrm <= {thr:.6e}:\n        return {k}\n"
    parts[-1] += f"    return {rows[-1][1]}\n"
    parts[-1] = parts[-1].replace("__MAX_ORDER__", str(rows[-1][1]))
    parts.append(kernel(3))
    parts.append(kernel(5))
    parts.append(solve_kernel(3))
    parts.append(solve_kernel(5))
    parts.append(FOOTER)
    return "\n\n\n".join(parts)


FOOTER = '''def expm_batch_numpy(
    A: npt.NDArray[np.float64], h: float = 1.0
) -> npt.NDArray[np.float64]:
    """Batched exp(A[i] * h) in pure numpy, with a per-matrix squaring count.

    Same algorithm as the numba kernels. The squarings are applied through a
    mask rather than uniformly, so a matrix whose norm needs 2 squarings gets
    2 and its neighbour needing 40 gets 40 -- squaring everything to the
    batch maximum is what loses accuracy for the small-norm matrices.
    """
    A = np.asarray(A, dtype=np.float64) * h
    n, m = A.shape[0], A.shape[1]

    nrm = np.abs(A).sum(axis=2).max(axis=1)
    s = np.zeros(n, dtype=np.int64)
    scale = np.isfinite(nrm) & (nrm > _THETA)
    if scale.any():
        s[scale] = np.minimum(np.ceil(np.log2(nrm[scale] / _THETA)), _S_MAX).astype(np.int64)
        A = A * np.exp2(-s.astype(np.float64))[:, None, None]

    # Unlike the kernels, this path runs one Taylor order for the whole batch,
    # so it uses the worst case allowed by the scaling above rather than the
    # batch maximum -- a matrix's result must not depend on its batch-mates.
    E = np.eye(m) + A
    P = A
    for k in range(2, _MAX_ORDER + 1):
        P = (P @ A) * (1.0 / k)
        E += P

    n_rounds = int(s.max(initial=0))
    if n_rounds:
        # Sorting by squaring count turns "the matrices still needing another
        # squaring" into a contiguous suffix, so each round is a slice rather
        # than a fancy-indexed copy in and out of the whole stack.
        perm = np.argsort(s, kind="stable")
        E = E[perm]
        s_sorted = s[perm]
        for k in range(n_rounds):
            start = int(np.searchsorted(s_sorted, k + 1, side="left"))
            view = E[start:]
            E[start:] = view @ view
        inverse = np.empty_like(perm)
        inverse[perm] = np.arange(n)
        E = E[inverse]
    return E


_EXPM_KERNELS = {3: _expm3_kernel, 5: _expm5_kernel}
_SOLVE_KERNELS = {3: _solve3_kernel, 5: _solve5_kernel}


def expm_batch(A: npt.NDArray[np.float64], h: float = 1.0) -> npt.NDArray[np.float64]:
    """Matrix exponential of each matrix in a stack: ``out[i] = exp(A[i] * h)``.

    Parameters
    ----------
    A
        Stack of square matrices, shape ``(n, m, m)``.
    h
        Scalar the whole stack is multiplied by first (the sub-step), folded
        into the kernel so no scaled copy of `A` is materialized.

    Returns
    -------
    Stack of the same shape. Each matrix is scaled, exponentiated and squared
    back up on its own, so mixing wildly different norms in one call is safe.
    """
    A = np.ascontiguousarray(A, dtype=np.float64)
    kernel = _EXPM_KERNELS.get(A.shape[-1]) if HAVE_NUMBA else None
    if kernel is None:
        return expm_batch_numpy(A, h)
    out = np.empty_like(A)
    kernel(A, float(h), out)
    return out


def solve_batch_numpy(A: npt.NDArray[np.float64], b: npt.NDArray[np.float64]):
    """Batched solve of ``A[i] x[i] = b[i]`` via numpy, one RHS per matrix."""
    return np.linalg.solve(A, b[:, :, np.newaxis])[:, :, 0]


def solve_batch(A: npt.NDArray[np.float64], b: npt.NDArray[np.float64]):
    """Solve ``A[i] x[i] = b[i]`` for a stack of small matrices.

    Parameters
    ----------
    A
        Stack of square matrices, shape ``(n, m, m)``.
    b
        Right-hand sides, shape ``(n, m)`` -- one per matrix.

    Returns
    -------
    Stack of solutions, shape ``(n, m)``.

    Raises
    ------
    numpy.linalg.LinAlgError
        If any matrix in the stack is exactly singular, matching
        `numpy.linalg.solve`. Note the whole call fails, as it does there;
        the near-singular case is not an error in either.
    """
    A = np.ascontiguousarray(A, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    kernel = _SOLVE_KERNELS.get(A.shape[-1]) if HAVE_NUMBA else None
    if kernel is None:
        return solve_batch_numpy(A, b)
    out = np.empty_like(b)
    if kernel(A, b, out):
        raise np.linalg.LinAlgError("Singular matrix")
    return out
'''

if __name__ == "__main__":
    import pathlib
    import subprocess

    dest = pathlib.Path(__file__).resolve().parents[1] / "src" / "crossbridge" / "_linalg.py"
    dest.write_text(main())
    subprocess.run(["ruff", "format", str(dest)], check=False)
    print(f"wrote {dest}")
