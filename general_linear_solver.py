# general_linear_solver.py
"""
Autograder-friendly PAQ = LU solver for Math589A Assignment 2.

Exports:
  - solve(A, b, tol=1e-6) -> (N, c)
    Always returns two outputs:
      N : ndarray, shape (n, n-r)  (2-D, possibly empty with n-r == 0)
      c : ndarray, shape (n,)      (1-D particular solution)
    Solutions: x = c + N @ x_free
"""
from typing import List, Tuple
import numpy as np

TOL_DEFAULT = 1e-6


def paq_lu_inplace(A: np.ndarray, tol: float = TOL_DEFAULT) -> Tuple[List[int], List[int], int]:
    """
    In-place PAQ = LU with simulated row swaps (P) and virtual column permutation (Q).
    Mutates A in-place: strict lower triangle stores L multipliers (unit diagonal implied),
    upper triangle stores U. Works for rectangular A (m x n).

    Returns:
      P, Q, r
      P : list length m, simulated row ordering (logical row i corresponds to original row P[i])
      Q : list length n, virtual column ordering (logical col j corresponds to original col Q[j])
      r : int, number of pivots found (rank estimate)
    """
    if not isinstance(A, np.ndarray):
        raise ValueError("A must be a numpy.ndarray")
    if A.dtype != np.float64:
        # convert in-place on a working copy caller should pass A.copy() if they don't want mutation
        A[:] = A.astype(np.float64)

    m, n = A.shape
    P = list(range(m))
    Q = list(range(n))
    r = 0
    k = 0
    j = 0

    while k < m and j < n:
        best_val = tol
        best_col = -1
        best_row = -1

        # Choose pivot column by maximum magnitude in rows k..m-1, among columns j..n-1.
        for col in range(j, n):
            local_max = 0.0
            local_row = -1
            for row in range(k, m):
                val = abs(A[P[row], Q[col]])
                if val > local_max:
                    local_max = val
                    local_row = row
            if local_max > best_val:
                best_val = local_max
                best_col = col
                best_row = local_row

        if best_row == -1:
            # no pivot > tol found
            break

        # virtual column swap Q
        if best_col != j:
            Q[j], Q[best_col] = Q[best_col], Q[j]

        # simulated row swap P
        if best_row != k:
            P[k], P[best_row] = P[best_row], P[k]

        # pivot now at (P[k], Q[j])
        pivot = A[P[k], Q[j]]
        # eliminate below pivot
        for i in range(k + 1, m):
            a_ik = A[P[i], Q[j]]
            if a_ik == 0.0:
                A[P[i], Q[j]] = 0.0
                continue
            mult = a_ik / pivot
            A[P[i], Q[j]] = mult
            for col in range(j + 1, n):
                A[P[i], Q[col]] -= mult * A[P[k], Q[col]]

        r += 1
        k += 1
        j += 1

    return P, Q, r


def solve_via_paq(A: np.ndarray, P: List[int], Q: List[int], r: int,
                  b: np.ndarray, tol: float = TOL_DEFAULT) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given A with in-place PAQ=LU (P,Q,r from paq_lu_inplace), compute (N, c).
    Returns:
      N : ndarray (n, n-r)  (2-D)
      c : ndarray (n,)
    If inconsistent, returns (N_nan, c_nan) with proper shapes filled with np.nan.
    """
    m, n = A.shape
    b = np.asarray(b).reshape(-1)
    if b.size != m:
        raise ValueError("b must have length m")

    # Pb using simulated P
    Pb = np.empty(m, dtype=float)
    for i in range(m):
        Pb[i] = b[P[i]]

    # Forward substitution Ly = Pb (L unit lower)
    y = np.empty(m, dtype=float)
    for i in range(m):
        s = Pb[i]
        up_to = min(i, r)
        for j in range(up_to):
            s -= A[P[i], Q[j]] * y[j]
        y[i] = s

    # inconsistency check: trailing y entries must be (near) zero
    if r < m:
        if np.max(np.abs(y[r:])) > tol:
            # inconsistent
            N_nan = np.full((n, max(0, n - r)), np.nan, dtype=float)
            c_nan = np.full(n, np.nan, dtype=float)
            return N_nan, c_nan

    # y_pivot = leading r entries
    y_pivot = y[:r].copy()

    # back substitution on U_B to get z (logical order)
    z = np.zeros(n, dtype=float)
    for i in range(r - 1, -1, -1):
        s = y_pivot[i]
        for j in range(i + 1, r):
            s -= A[P[i], Q[j]] * z[j]
        denom = A[P[i], Q[i]]
        if abs(denom) <= tol:
            raise np.linalg.LinAlgError(f"Near-zero pivot at U[{i},{i}] = {denom}")
        z[i] = s / denom

    # particular solution c in original ordering: c[Q[j]] = z[j]
    c = np.zeros(n, dtype=float)
    for j in range(n):
        c[Q[j]] = z[j]

    # nullspace mapping N: columns correspond to free logical columns r..n-1
    num_free = n - r
    if num_free <= 0:
        N = np.zeros((n, 0), dtype=float)
        return N, c

    N = np.zeros((n, num_free), dtype=float)
    for t in range(num_free):
        j_free = r + t
        # rhs = - U_B^{-1} * U_bf[:,t] ; U_bf entries A[P[i], Q[j_free]] for i=0..r-1
        rhs = np.empty(r, dtype=float)
        for i in range(r):
            rhs[i] = -A[P[i], Q[j_free]]
        zb = np.empty(r, dtype=float)
        for i in range(r - 1, -1, -1):
            s = rhs[i]
            for j in range(i + 1, r):
                s -= A[P[i], Q[j]] * zb[j]
            denom = A[P[i], Q[i]]
            if abs(denom) <= tol:
                raise np.linalg.LinAlgError(f"Near-zero pivot at U[{i},{i}] = {denom}")
            zb[i] = s / denom
        zfull = np.zeros(n, dtype=float)
        zfull[:r] = zb
        zfull[j_free] = 1.0
        xcol = np.zeros(n, dtype=float)
        for j in range(n):
            xcol[Q[j]] = zfull[j]
        N[:, t] = xcol

    return N, c


def solve(A: np.ndarray, b: np.ndarray, tol: float = TOL_DEFAULT) -> Tuple[np.ndarray, np.ndarray]:
    """
    Autograder wrapper: returns (N, c).
    Always returns two values:
      N : ndarray (n, n-r) 2-D (may be empty with shape (n,0))
      c : ndarray (n,)
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m, n = A.shape

    # work on a copy so caller's A is preserved
    A_work = A.copy()

    P, Q, r = paq_lu_inplace(A_work, tol=tol)
    N, c = solve_via_paq(A_work, P, Q, r, b, tol=tol)

    # Ensure N is 2-D ndarray (already done) and c is 1-D of length n
    if not isinstance(N, np.ndarray):
        N = np.asarray(N, dtype=float).reshape((n, max(0, n - r)))
    if c.ndim != 1 or c.size != n:
        c = np.asarray(c, dtype=float).reshape(n,)

    return N, c


# quick sanity tests (run locally)
if __name__ == "__main__":
    np.set_printoptions(precision=8, suppress=True)

    # square nonsingular
    A0 = np.array([[3.0, 1.0], [4.0, -2.0]])
    b0 = np.array([5.0, -6.0])
    N0, c0 = solve(A0, b0)
    print("Square nonsingular: N.shape=", N0.shape, "c=", c0, "resid=", np.linalg.norm(A0 @ c0 - b0))

    # underdetermined (rank-def)
    A1 = np.array([[1., 2., 3., 4.],
                   [2., 4., 6., 8.],
                   [0., 1., 1., 2.]])
    b1 = np.array([10., 20., 5.])
    N1, c1 = solve(A1, b1)
    print("Underdetermined: N.shape=", N1.shape, "||A c - b||=", np.linalg.norm(A1 @ c1 - b1), "||A N||=", np.linalg.norm(A1 @ N1))

    # inconsistent
    A2 = np.array([[1., 1.], [1., 1.]])
    b2 = np.array([1., 2.])
    N2, c2 = solve(A2, b2)
    print("Inconsistent (NaNs):", np.isnan(c2).all(), np.isnan(N2).all())
