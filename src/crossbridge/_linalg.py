"""
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
_MAX_ORDER = 15

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
    if nrm <= 3.914868e-06:
        return 2
    if nrm <= 1.244666e-04:
        return 3
    if nrm <= 1.037137e-03:
        return 4
    if nrm <= 4.394290e-03:
        return 5
    if nrm <= 1.259932e-02:
        return 6
    if nrm <= 2.822865e-02:
        return 7
    if nrm <= 5.356271e-02:
        return 8
    if nrm <= 9.036002e-02:
        return 9
    if nrm <= 1.398169e-01:
        return 10
    if nrm <= 2.026258e-01:
        return 11
    if nrm <= 2.790705e-01:
        return 12
    if nrm <= 3.691238e-01:
        return 13
    if nrm <= 4.725343e-01:
        return 14
    return 15


@_njit(cache=True, fastmath=True, nogil=True)
def _expm3_kernel(A, h, out):  # pragma: no cover - numba-compiled
    """Hand-unrolled batched exp(A[i] * h) for 3x3 matrices."""
    for i in range(A.shape[0]):
        # --- load and scale by the sub-step h ---
        a00 = A[i, 0, 0] * h
        a01 = A[i, 0, 1] * h
        a02 = A[i, 0, 2] * h
        a10 = A[i, 1, 0] * h
        a11 = A[i, 1, 1] * h
        a12 = A[i, 1, 2] * h
        a20 = A[i, 2, 0] * h
        a21 = A[i, 2, 1] * h
        a22 = A[i, 2, 2] * h

        # --- per-matrix squaring count from this matrix's own inf-norm ---
        r0 = abs(a00) + abs(a01) + abs(a02)
        r1 = abs(a10) + abs(a11) + abs(a12)
        r2 = abs(a20) + abs(a21) + abs(a22)
        nrm = max(max(r0, r1), r2)
        s = 0
        while nrm > _THETA and s < _S_MAX:
            nrm *= 0.5
            s += 1
        if s > 0:
            f = 2.0 ** (-s)
            a00 *= f
            a01 *= f
            a02 *= f
            a10 *= f
            a11 *= f
            a12 *= f
            a20 *= f
            a21 *= f
            a22 *= f

        # --- Taylor series: E = I + A + A^2/2! + ... , P the running term ---
        e00 = 1.0 + a00
        e01 = a01
        e02 = a02
        e10 = a10
        e11 = 1.0 + a11
        e12 = a12
        e20 = a20
        e21 = a21
        e22 = 1.0 + a22
        p00 = a00
        p01 = a01
        p02 = a02
        p10 = a10
        p11 = a11
        p12 = a12
        p20 = a20
        p21 = a21
        p22 = a22
        for k in range(2, _taylor_order(nrm) + 1):
            inv = 1.0 / k
            t00 = (p00 * a00 + p01 * a10 + p02 * a20) * inv
            t01 = (p00 * a01 + p01 * a11 + p02 * a21) * inv
            t02 = (p00 * a02 + p01 * a12 + p02 * a22) * inv
            t10 = (p10 * a00 + p11 * a10 + p12 * a20) * inv
            t11 = (p10 * a01 + p11 * a11 + p12 * a21) * inv
            t12 = (p10 * a02 + p11 * a12 + p12 * a22) * inv
            t20 = (p20 * a00 + p21 * a10 + p22 * a20) * inv
            t21 = (p20 * a01 + p21 * a11 + p22 * a21) * inv
            t22 = (p20 * a02 + p21 * a12 + p22 * a22) * inv
            p00 = t00
            p01 = t01
            p02 = t02
            p10 = t10
            p11 = t11
            p12 = t12
            p20 = t20
            p21 = t21
            p22 = t22
            e00 += p00
            e01 += p01
            e02 += p02
            e10 += p10
            e11 += p11
            e12 += p12
            e20 += p20
            e21 += p21
            e22 += p22

        # --- undo the scaling: s repeated squarings ---
        for _ in range(s):
            t00 = e00 * e00 + e01 * e10 + e02 * e20
            t01 = e00 * e01 + e01 * e11 + e02 * e21
            t02 = e00 * e02 + e01 * e12 + e02 * e22
            t10 = e10 * e00 + e11 * e10 + e12 * e20
            t11 = e10 * e01 + e11 * e11 + e12 * e21
            t12 = e10 * e02 + e11 * e12 + e12 * e22
            t20 = e20 * e00 + e21 * e10 + e22 * e20
            t21 = e20 * e01 + e21 * e11 + e22 * e21
            t22 = e20 * e02 + e21 * e12 + e22 * e22
            e00 = t00
            e01 = t01
            e02 = t02
            e10 = t10
            e11 = t11
            e12 = t12
            e20 = t20
            e21 = t21
            e22 = t22

        out[i, 0, 0] = e00
        out[i, 0, 1] = e01
        out[i, 0, 2] = e02
        out[i, 1, 0] = e10
        out[i, 1, 1] = e11
        out[i, 1, 2] = e12
        out[i, 2, 0] = e20
        out[i, 2, 1] = e21
        out[i, 2, 2] = e22


@_njit(cache=True, fastmath=True, nogil=True)
def _expm5_kernel(A, h, out):  # pragma: no cover - numba-compiled
    """Hand-unrolled batched exp(A[i] * h) for 5x5 matrices."""
    for i in range(A.shape[0]):
        # --- load and scale by the sub-step h ---
        a00 = A[i, 0, 0] * h
        a01 = A[i, 0, 1] * h
        a02 = A[i, 0, 2] * h
        a03 = A[i, 0, 3] * h
        a04 = A[i, 0, 4] * h
        a10 = A[i, 1, 0] * h
        a11 = A[i, 1, 1] * h
        a12 = A[i, 1, 2] * h
        a13 = A[i, 1, 3] * h
        a14 = A[i, 1, 4] * h
        a20 = A[i, 2, 0] * h
        a21 = A[i, 2, 1] * h
        a22 = A[i, 2, 2] * h
        a23 = A[i, 2, 3] * h
        a24 = A[i, 2, 4] * h
        a30 = A[i, 3, 0] * h
        a31 = A[i, 3, 1] * h
        a32 = A[i, 3, 2] * h
        a33 = A[i, 3, 3] * h
        a34 = A[i, 3, 4] * h
        a40 = A[i, 4, 0] * h
        a41 = A[i, 4, 1] * h
        a42 = A[i, 4, 2] * h
        a43 = A[i, 4, 3] * h
        a44 = A[i, 4, 4] * h

        # --- per-matrix squaring count from this matrix's own inf-norm ---
        r0 = abs(a00) + abs(a01) + abs(a02) + abs(a03) + abs(a04)
        r1 = abs(a10) + abs(a11) + abs(a12) + abs(a13) + abs(a14)
        r2 = abs(a20) + abs(a21) + abs(a22) + abs(a23) + abs(a24)
        r3 = abs(a30) + abs(a31) + abs(a32) + abs(a33) + abs(a34)
        r4 = abs(a40) + abs(a41) + abs(a42) + abs(a43) + abs(a44)
        nrm = max(max(max(max(r0, r1), r2), r3), r4)
        s = 0
        while nrm > _THETA and s < _S_MAX:
            nrm *= 0.5
            s += 1
        if s > 0:
            f = 2.0 ** (-s)
            a00 *= f
            a01 *= f
            a02 *= f
            a03 *= f
            a04 *= f
            a10 *= f
            a11 *= f
            a12 *= f
            a13 *= f
            a14 *= f
            a20 *= f
            a21 *= f
            a22 *= f
            a23 *= f
            a24 *= f
            a30 *= f
            a31 *= f
            a32 *= f
            a33 *= f
            a34 *= f
            a40 *= f
            a41 *= f
            a42 *= f
            a43 *= f
            a44 *= f

        # --- Taylor series: E = I + A + A^2/2! + ... , P the running term ---
        e00 = 1.0 + a00
        e01 = a01
        e02 = a02
        e03 = a03
        e04 = a04
        e10 = a10
        e11 = 1.0 + a11
        e12 = a12
        e13 = a13
        e14 = a14
        e20 = a20
        e21 = a21
        e22 = 1.0 + a22
        e23 = a23
        e24 = a24
        e30 = a30
        e31 = a31
        e32 = a32
        e33 = 1.0 + a33
        e34 = a34
        e40 = a40
        e41 = a41
        e42 = a42
        e43 = a43
        e44 = 1.0 + a44
        p00 = a00
        p01 = a01
        p02 = a02
        p03 = a03
        p04 = a04
        p10 = a10
        p11 = a11
        p12 = a12
        p13 = a13
        p14 = a14
        p20 = a20
        p21 = a21
        p22 = a22
        p23 = a23
        p24 = a24
        p30 = a30
        p31 = a31
        p32 = a32
        p33 = a33
        p34 = a34
        p40 = a40
        p41 = a41
        p42 = a42
        p43 = a43
        p44 = a44
        for k in range(2, _taylor_order(nrm) + 1):
            inv = 1.0 / k
            t00 = (p00 * a00 + p01 * a10 + p02 * a20 + p03 * a30 + p04 * a40) * inv
            t01 = (p00 * a01 + p01 * a11 + p02 * a21 + p03 * a31 + p04 * a41) * inv
            t02 = (p00 * a02 + p01 * a12 + p02 * a22 + p03 * a32 + p04 * a42) * inv
            t03 = (p00 * a03 + p01 * a13 + p02 * a23 + p03 * a33 + p04 * a43) * inv
            t04 = (p00 * a04 + p01 * a14 + p02 * a24 + p03 * a34 + p04 * a44) * inv
            t10 = (p10 * a00 + p11 * a10 + p12 * a20 + p13 * a30 + p14 * a40) * inv
            t11 = (p10 * a01 + p11 * a11 + p12 * a21 + p13 * a31 + p14 * a41) * inv
            t12 = (p10 * a02 + p11 * a12 + p12 * a22 + p13 * a32 + p14 * a42) * inv
            t13 = (p10 * a03 + p11 * a13 + p12 * a23 + p13 * a33 + p14 * a43) * inv
            t14 = (p10 * a04 + p11 * a14 + p12 * a24 + p13 * a34 + p14 * a44) * inv
            t20 = (p20 * a00 + p21 * a10 + p22 * a20 + p23 * a30 + p24 * a40) * inv
            t21 = (p20 * a01 + p21 * a11 + p22 * a21 + p23 * a31 + p24 * a41) * inv
            t22 = (p20 * a02 + p21 * a12 + p22 * a22 + p23 * a32 + p24 * a42) * inv
            t23 = (p20 * a03 + p21 * a13 + p22 * a23 + p23 * a33 + p24 * a43) * inv
            t24 = (p20 * a04 + p21 * a14 + p22 * a24 + p23 * a34 + p24 * a44) * inv
            t30 = (p30 * a00 + p31 * a10 + p32 * a20 + p33 * a30 + p34 * a40) * inv
            t31 = (p30 * a01 + p31 * a11 + p32 * a21 + p33 * a31 + p34 * a41) * inv
            t32 = (p30 * a02 + p31 * a12 + p32 * a22 + p33 * a32 + p34 * a42) * inv
            t33 = (p30 * a03 + p31 * a13 + p32 * a23 + p33 * a33 + p34 * a43) * inv
            t34 = (p30 * a04 + p31 * a14 + p32 * a24 + p33 * a34 + p34 * a44) * inv
            t40 = (p40 * a00 + p41 * a10 + p42 * a20 + p43 * a30 + p44 * a40) * inv
            t41 = (p40 * a01 + p41 * a11 + p42 * a21 + p43 * a31 + p44 * a41) * inv
            t42 = (p40 * a02 + p41 * a12 + p42 * a22 + p43 * a32 + p44 * a42) * inv
            t43 = (p40 * a03 + p41 * a13 + p42 * a23 + p43 * a33 + p44 * a43) * inv
            t44 = (p40 * a04 + p41 * a14 + p42 * a24 + p43 * a34 + p44 * a44) * inv
            p00 = t00
            p01 = t01
            p02 = t02
            p03 = t03
            p04 = t04
            p10 = t10
            p11 = t11
            p12 = t12
            p13 = t13
            p14 = t14
            p20 = t20
            p21 = t21
            p22 = t22
            p23 = t23
            p24 = t24
            p30 = t30
            p31 = t31
            p32 = t32
            p33 = t33
            p34 = t34
            p40 = t40
            p41 = t41
            p42 = t42
            p43 = t43
            p44 = t44
            e00 += p00
            e01 += p01
            e02 += p02
            e03 += p03
            e04 += p04
            e10 += p10
            e11 += p11
            e12 += p12
            e13 += p13
            e14 += p14
            e20 += p20
            e21 += p21
            e22 += p22
            e23 += p23
            e24 += p24
            e30 += p30
            e31 += p31
            e32 += p32
            e33 += p33
            e34 += p34
            e40 += p40
            e41 += p41
            e42 += p42
            e43 += p43
            e44 += p44

        # --- undo the scaling: s repeated squarings ---
        for _ in range(s):
            t00 = e00 * e00 + e01 * e10 + e02 * e20 + e03 * e30 + e04 * e40
            t01 = e00 * e01 + e01 * e11 + e02 * e21 + e03 * e31 + e04 * e41
            t02 = e00 * e02 + e01 * e12 + e02 * e22 + e03 * e32 + e04 * e42
            t03 = e00 * e03 + e01 * e13 + e02 * e23 + e03 * e33 + e04 * e43
            t04 = e00 * e04 + e01 * e14 + e02 * e24 + e03 * e34 + e04 * e44
            t10 = e10 * e00 + e11 * e10 + e12 * e20 + e13 * e30 + e14 * e40
            t11 = e10 * e01 + e11 * e11 + e12 * e21 + e13 * e31 + e14 * e41
            t12 = e10 * e02 + e11 * e12 + e12 * e22 + e13 * e32 + e14 * e42
            t13 = e10 * e03 + e11 * e13 + e12 * e23 + e13 * e33 + e14 * e43
            t14 = e10 * e04 + e11 * e14 + e12 * e24 + e13 * e34 + e14 * e44
            t20 = e20 * e00 + e21 * e10 + e22 * e20 + e23 * e30 + e24 * e40
            t21 = e20 * e01 + e21 * e11 + e22 * e21 + e23 * e31 + e24 * e41
            t22 = e20 * e02 + e21 * e12 + e22 * e22 + e23 * e32 + e24 * e42
            t23 = e20 * e03 + e21 * e13 + e22 * e23 + e23 * e33 + e24 * e43
            t24 = e20 * e04 + e21 * e14 + e22 * e24 + e23 * e34 + e24 * e44
            t30 = e30 * e00 + e31 * e10 + e32 * e20 + e33 * e30 + e34 * e40
            t31 = e30 * e01 + e31 * e11 + e32 * e21 + e33 * e31 + e34 * e41
            t32 = e30 * e02 + e31 * e12 + e32 * e22 + e33 * e32 + e34 * e42
            t33 = e30 * e03 + e31 * e13 + e32 * e23 + e33 * e33 + e34 * e43
            t34 = e30 * e04 + e31 * e14 + e32 * e24 + e33 * e34 + e34 * e44
            t40 = e40 * e00 + e41 * e10 + e42 * e20 + e43 * e30 + e44 * e40
            t41 = e40 * e01 + e41 * e11 + e42 * e21 + e43 * e31 + e44 * e41
            t42 = e40 * e02 + e41 * e12 + e42 * e22 + e43 * e32 + e44 * e42
            t43 = e40 * e03 + e41 * e13 + e42 * e23 + e43 * e33 + e44 * e43
            t44 = e40 * e04 + e41 * e14 + e42 * e24 + e43 * e34 + e44 * e44
            e00 = t00
            e01 = t01
            e02 = t02
            e03 = t03
            e04 = t04
            e10 = t10
            e11 = t11
            e12 = t12
            e13 = t13
            e14 = t14
            e20 = t20
            e21 = t21
            e22 = t22
            e23 = t23
            e24 = t24
            e30 = t30
            e31 = t31
            e32 = t32
            e33 = t33
            e34 = t34
            e40 = t40
            e41 = t41
            e42 = t42
            e43 = t43
            e44 = t44

        out[i, 0, 0] = e00
        out[i, 0, 1] = e01
        out[i, 0, 2] = e02
        out[i, 0, 3] = e03
        out[i, 0, 4] = e04
        out[i, 1, 0] = e10
        out[i, 1, 1] = e11
        out[i, 1, 2] = e12
        out[i, 1, 3] = e13
        out[i, 1, 4] = e14
        out[i, 2, 0] = e20
        out[i, 2, 1] = e21
        out[i, 2, 2] = e22
        out[i, 2, 3] = e23
        out[i, 2, 4] = e24
        out[i, 3, 0] = e30
        out[i, 3, 1] = e31
        out[i, 3, 2] = e32
        out[i, 3, 3] = e33
        out[i, 3, 4] = e34
        out[i, 4, 0] = e40
        out[i, 4, 1] = e41
        out[i, 4, 2] = e42
        out[i, 4, 3] = e43
        out[i, 4, 4] = e44


@_njit(cache=True, nogil=True, error_model="numpy")
def _solve3_kernel(A, b, out):  # pragma: no cover - numba-compiled
    """Hand-unrolled batched solve of 3x3 A[i] x = b[i]."""
    singular = 0
    for i in range(A.shape[0]):
        # --- load the augmented matrix [A | b] ---
        a00 = A[i, 0, 0]
        a01 = A[i, 0, 1]
        a02 = A[i, 0, 2]
        a03 = b[i, 0]
        a10 = A[i, 1, 0]
        a11 = A[i, 1, 1]
        a12 = A[i, 1, 2]
        a13 = b[i, 1]
        a20 = A[i, 2, 0]
        a21 = A[i, 2, 1]
        a22 = A[i, 2, 2]
        a23 = b[i, 2]

        # --- column 0: pivot, then eliminate below ---
        if abs(a10) > abs(a00):
            a00, a10 = a10, a00
            a01, a11 = a11, a01
            a02, a12 = a12, a02
            a03, a13 = a13, a03
        if abs(a20) > abs(a00):
            a00, a20 = a20, a00
            a01, a21 = a21, a01
            a02, a22 = a22, a02
            a03, a23 = a23, a03
        if a00 == 0.0:
            singular += 1
        f1 = a10 / a00
        a11 -= f1 * a01
        a12 -= f1 * a02
        a13 -= f1 * a03
        f2 = a20 / a00
        a21 -= f2 * a01
        a22 -= f2 * a02
        a23 -= f2 * a03

        # --- column 1: pivot, then eliminate below ---
        if abs(a21) > abs(a11):
            a11, a21 = a21, a11
            a12, a22 = a22, a12
            a13, a23 = a23, a13
        if a11 == 0.0:
            singular += 1
        f2 = a21 / a11
        a22 -= f2 * a12
        a23 -= f2 * a13

        # --- back substitution ---
        if a22 == 0.0:
            singular += 1
        x2 = a23 / a22
        x1 = (a13 - a12 * x2) / a11
        x0 = (a03 - a01 * x1 - a02 * x2) / a00
        out[i, 0] = x0
        out[i, 1] = x1
        out[i, 2] = x2
    return singular


@_njit(cache=True, nogil=True, error_model="numpy")
def _solve5_kernel(A, b, out):  # pragma: no cover - numba-compiled
    """Hand-unrolled batched solve of 5x5 A[i] x = b[i]."""
    singular = 0
    for i in range(A.shape[0]):
        # --- load the augmented matrix [A | b] ---
        a00 = A[i, 0, 0]
        a01 = A[i, 0, 1]
        a02 = A[i, 0, 2]
        a03 = A[i, 0, 3]
        a04 = A[i, 0, 4]
        a05 = b[i, 0]
        a10 = A[i, 1, 0]
        a11 = A[i, 1, 1]
        a12 = A[i, 1, 2]
        a13 = A[i, 1, 3]
        a14 = A[i, 1, 4]
        a15 = b[i, 1]
        a20 = A[i, 2, 0]
        a21 = A[i, 2, 1]
        a22 = A[i, 2, 2]
        a23 = A[i, 2, 3]
        a24 = A[i, 2, 4]
        a25 = b[i, 2]
        a30 = A[i, 3, 0]
        a31 = A[i, 3, 1]
        a32 = A[i, 3, 2]
        a33 = A[i, 3, 3]
        a34 = A[i, 3, 4]
        a35 = b[i, 3]
        a40 = A[i, 4, 0]
        a41 = A[i, 4, 1]
        a42 = A[i, 4, 2]
        a43 = A[i, 4, 3]
        a44 = A[i, 4, 4]
        a45 = b[i, 4]

        # --- column 0: pivot, then eliminate below ---
        if abs(a10) > abs(a00):
            a00, a10 = a10, a00
            a01, a11 = a11, a01
            a02, a12 = a12, a02
            a03, a13 = a13, a03
            a04, a14 = a14, a04
            a05, a15 = a15, a05
        if abs(a20) > abs(a00):
            a00, a20 = a20, a00
            a01, a21 = a21, a01
            a02, a22 = a22, a02
            a03, a23 = a23, a03
            a04, a24 = a24, a04
            a05, a25 = a25, a05
        if abs(a30) > abs(a00):
            a00, a30 = a30, a00
            a01, a31 = a31, a01
            a02, a32 = a32, a02
            a03, a33 = a33, a03
            a04, a34 = a34, a04
            a05, a35 = a35, a05
        if abs(a40) > abs(a00):
            a00, a40 = a40, a00
            a01, a41 = a41, a01
            a02, a42 = a42, a02
            a03, a43 = a43, a03
            a04, a44 = a44, a04
            a05, a45 = a45, a05
        if a00 == 0.0:
            singular += 1
        f1 = a10 / a00
        a11 -= f1 * a01
        a12 -= f1 * a02
        a13 -= f1 * a03
        a14 -= f1 * a04
        a15 -= f1 * a05
        f2 = a20 / a00
        a21 -= f2 * a01
        a22 -= f2 * a02
        a23 -= f2 * a03
        a24 -= f2 * a04
        a25 -= f2 * a05
        f3 = a30 / a00
        a31 -= f3 * a01
        a32 -= f3 * a02
        a33 -= f3 * a03
        a34 -= f3 * a04
        a35 -= f3 * a05
        f4 = a40 / a00
        a41 -= f4 * a01
        a42 -= f4 * a02
        a43 -= f4 * a03
        a44 -= f4 * a04
        a45 -= f4 * a05

        # --- column 1: pivot, then eliminate below ---
        if abs(a21) > abs(a11):
            a11, a21 = a21, a11
            a12, a22 = a22, a12
            a13, a23 = a23, a13
            a14, a24 = a24, a14
            a15, a25 = a25, a15
        if abs(a31) > abs(a11):
            a11, a31 = a31, a11
            a12, a32 = a32, a12
            a13, a33 = a33, a13
            a14, a34 = a34, a14
            a15, a35 = a35, a15
        if abs(a41) > abs(a11):
            a11, a41 = a41, a11
            a12, a42 = a42, a12
            a13, a43 = a43, a13
            a14, a44 = a44, a14
            a15, a45 = a45, a15
        if a11 == 0.0:
            singular += 1
        f2 = a21 / a11
        a22 -= f2 * a12
        a23 -= f2 * a13
        a24 -= f2 * a14
        a25 -= f2 * a15
        f3 = a31 / a11
        a32 -= f3 * a12
        a33 -= f3 * a13
        a34 -= f3 * a14
        a35 -= f3 * a15
        f4 = a41 / a11
        a42 -= f4 * a12
        a43 -= f4 * a13
        a44 -= f4 * a14
        a45 -= f4 * a15

        # --- column 2: pivot, then eliminate below ---
        if abs(a32) > abs(a22):
            a22, a32 = a32, a22
            a23, a33 = a33, a23
            a24, a34 = a34, a24
            a25, a35 = a35, a25
        if abs(a42) > abs(a22):
            a22, a42 = a42, a22
            a23, a43 = a43, a23
            a24, a44 = a44, a24
            a25, a45 = a45, a25
        if a22 == 0.0:
            singular += 1
        f3 = a32 / a22
        a33 -= f3 * a23
        a34 -= f3 * a24
        a35 -= f3 * a25
        f4 = a42 / a22
        a43 -= f4 * a23
        a44 -= f4 * a24
        a45 -= f4 * a25

        # --- column 3: pivot, then eliminate below ---
        if abs(a43) > abs(a33):
            a33, a43 = a43, a33
            a34, a44 = a44, a34
            a35, a45 = a45, a35
        if a33 == 0.0:
            singular += 1
        f4 = a43 / a33
        a44 -= f4 * a34
        a45 -= f4 * a35

        # --- back substitution ---
        if a44 == 0.0:
            singular += 1
        x4 = a45 / a44
        x3 = (a35 - a34 * x4) / a33
        x2 = (a25 - a23 * x3 - a24 * x4) / a22
        x1 = (a15 - a12 * x2 - a13 * x3 - a14 * x4) / a11
        x0 = (a05 - a01 * x1 - a02 * x2 - a03 * x3 - a04 * x4) / a00
        out[i, 0] = x0
        out[i, 1] = x1
        out[i, 2] = x2
        out[i, 3] = x3
        out[i, 4] = x4
    return singular


def expm_batch_numpy(A: npt.NDArray[np.float64], h: float = 1.0) -> npt.NDArray[np.float64]:
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
