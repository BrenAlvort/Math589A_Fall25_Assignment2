# general_linear_solver.py
"""
PAQ = LU factorization and solver following derivation.tex

Implements:
  - paq_lu_inplace(A, tol=1e-12)
      In-place PAQ = LU for rectangular A. Returns (P, Q, r).
      P: simulated row permutation (list of original row indices in logical order).
      Q: virtual column permutation (list of original column indices in logical order).
      r: number of pivots found (rank estimate).

  - solve_via_paq(A, P, Q, r, b, tol=1e-12)
      Given the overwritten A and permutations, returns:
        c: particular solution (n,)
        N: nullspace mapping (n, n-r)
        parameterize: callable(x_free) -> x
        pivot_cols, free_cols: lists of original column indices

  - solve(A, b, tol=1e-12)
      Autograder wrapper. If the system has a unique solution returns x (n,).
      Otherwise returns (c, N) where c is particular solution (free vars = 0).
"""
from typing import List, Tuple, Callable
import numpy as np


def paq_lu_inplace(A: np.ndarray, tol: float = 1e-12) -> Tuple[List[int], List[int], int]:
    """
    In-place PAQ = LU decomposition with simulated row exchanges (P) and
    virtual column exchanges (Q). Works for rectangular A (m x n).

    The input A is overwritten (L in strict lower triangle, U in upper triangle).
    The unit diagonal of L is implicit.

    Returns:
        P (list): simulated row-permutation vector (logical -> original row idx)
        Q (list): virtual column-permutation vector (logical -> original col idx)
        r (int) : number of pivots found (rank estimate)
    """
    if not isinstance(A, np.ndarray):
        A = np.array(A, dtype=float)
    else:
        if not np.issubdtype(A.dtype, np.floating):
            A = A.astype(float, copy=False)

    m, n = A.shape
    P = list(range(m))
    Q = list(range(n))
    r = 0
    k = 0  # logical pivot row index
    j = 0  # logical pivot column index

    # Loop until out of rows or columns
    while k < m and j < n:
        # Choose pivot column: among j..n-1 choose the column whose max |entry|
        # in rows k..m-1 is largest (this moves pivot columns to the front via Q).
        best_col = -1
        best_row = -1
        best_val = 0.0
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

        # Stop if no pivot larger than tolerance
        if best_val <= tol:
            break

        # Virtual column swap: move best_col to logical position j
        if best_col != j:
            Q[j], Q[best_col] = Q[best_col], Q[j]

        # Simulate row swap: move best_row to logical position k
        if best_row != k:
            P[k], P[best_row] = P[best_row], P[k]

        # Pivot is now at (P[k], Q[j])
        pivot = A[P[k], Q[j]]
        if abs(pivot) <= tol:
            # numerical safety; skip column
            j += 1
            continue

        # Eliminate below pivot, store multipliers in strict lower triangle
        for i in range(k + 1, m):
            a_ik = A[P[i], Q[j]]
            if a_ik == 0.0:
                A[P[i], Q[j]] = 0.0
                continue
            m_ik = a_ik / pivot
            A[P[i], Q[j]] = m_ik
            # update trailing columns (logical columns j+1..n-1)
            for col in range(j + 1, n):
                A[P[i], Q[col]] -= m_ik * A[P[k], Q[col]]

        r += 1
        k += 1
        j += 1

    return P, Q, r


def solve_via_paq(A: np.ndarray, P: List[int], Q: List[int], r: int,
                  b: np.ndarray, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, Callable, List[int], List[int]]:
    """
    Construct the general solution for A x = b given in-place PAQ = LU in A and permutations.

    Returns:
      c : particular solution vector in original column order (n,)
      N : nullspace mapping matrix (n, n-r) such that x = c + N @ x_free
      parameterize : function(x_free) -> x
      pivot_cols : list of original pivot column indices (length r)
      free_cols  : list of original free column indices (length n-r)
    """
    m, n = A.shape
    b = np.asarray(b).reshape(-1)
    if b.size != m:
        raise ValueError("b must have length m")

    # Permute b by P: Pb (logical rows order)
    Pb = np.empty(m, dtype=float)
    for i in range(m):
        Pb[i] = b[P[i]]

    # Forward substitution Ly = Pb.
    # L has unit diagonal and multipliers stored at A[P[i], Q[j]] for logical j < r.
    y = np.empty(m, dtype=float)
    for i in range(m):
        s = Pb[i]
        up_to = min(i, r)  # only multipliers from columns 0..r-1 are part of L used for pivot rows
        for j in range(up_to):
            s -= A[P[i], Q[j]] * y[j]
        y[i] = s

    # Back substitution on U to get z in permuted variable order (logical)
    # Only the first r logical rows/cols correspond to U_B (invertible r x r)
    z = np.zeros(n, dtype=float)
    if r > 0:
        for i in range(r - 1, -1, -1):
            s = y[i]
            for j in range(i + 1, r):
                s -= A[P[i], Q[j]] * z[j]
            denom = A[P[i], Q[i]]
            if abs(denom) <= tol:
                raise np.linalg.LinAlgError(f"Near-zero pivot encountered at U[{i},{i}] = {denom}")
            z[i] = s / denom

    # Particular solution in original ordering: c[Q[j]] = z[j]
    c = np.zeros(n, dtype=float)
    for j in range(n):
        c[Q[j]] = z[j]

    # Build nullspace mapping N (n x (n-r)).
    num_free = n - r
    if num_free == 0:
        N = np.zeros((n, 0), dtype=float)
    else:
        N = np.zeros((n, num_free), dtype=float)
        # For each free logical column t (logical index r+t), find z_b = -U_B^{-1} U_bf[:,t]
        for t in range(num_free):
            # Build rhs = -U_bf[:, t] where U_bf entries are A[P[i], Q[r+t]] for i = 0..r-1
            rhs = np.empty(r, dtype=float)
            for i in range(r):
                rhs[i] = -A[P[i], Q[r + t]]
            # Solve U_B * zb = rhs by back substitution
            zb = np.empty(r, dtype=float)
            for i in range(r - 1, -1, -1):
                s = rhs[i]
                for j in range(i + 1, r):
                    s -= A[P[i], Q[j]] * zb[j]
                denom = A[P[i], Q[i]]
                if abs(denom) <= tol:
                    raise np.linalg.LinAlgError(f"Near-zero pivot encountered at U[{i},{i}] = {denom}")
                zb[i] = s / denom
            # zfull (logical order): [zb; e_t] where e_t is 1 at position r+t
            zfull = np.zeros(n, dtype=float)
            zfull[:r] = zb
            zfull[r + t] = 1.0
            # map to original ordering: xcol[Q[j]] = zfull[j]
            xcol = np.zeros(n, dtype=float)
            for j in range(n):
                xcol[Q[j]] = zfull[j]
            N[:, t] = xcol

    pivot_cols = Q[:r].copy()
    free_cols = Q[r:].copy()

    def parameterize(x_free: np.ndarray) -> np.ndarray:
        x_free = np.asarray(x_free).reshape(-1)
        if x_free.size != (n - r):
            raise ValueError(f"x_free must have length {n - r}")
        return c + N @ x_free

    return c, N, parameterize, pivot_cols, free_cols


def solve(A: np.ndarray, b: np.ndarray, tol: float = 1e-12):
    """
    Autograder-friendly wrapper.

    If no free variables (n == r) returns the unique solution vector x (n,).
    Otherwise returns (c, N) where c is a particular solution (free vars = 0)
    and N is an (n, n-r) matrix mapping free variables to full solutions.

    NOTE: This function works on a copy of A (does not mutate caller-provided A).
    """
    A = np.asarray(A)
    b = np.asarray(b)
    A_work = A.copy().astype(float, copy=True)

    P, Q, r = paq_lu_inplace(A_work, tol=tol)
    c, N, _, _, _ = solve_via_paq(A_work, P, Q, r, b, tol=tol)
    if N.shape[1] == 0:
        # unique solution
        return c
    else:
        return c, N


# Quick local sanity test when run directly
if __name__ == "__main__":
    # square nonsingular example
    A0 = np.array([[3.0, 1.0, 2.0],
                   [6.0, 3.0, 4.0],
                   [3.0, 1.0, 5.0]])
    b0 = np.array([1.0, 2.0, 3.0])
    result = solve(A0, b0)
    print("solve(A0,b0) ->", result)
    # rectangular rank-deficient example
    A1 = np.array([[1., 2., 3., 4.],
                   [2., 4., 6., 8.],
                   [0., 1., 1., 2.]])
    b1 = np.array([10., 20., 5.])
    c, N = solve(A1, b1)
    print("particular c:", c)
    print("N shape:", N.shape)
    # verify Ax = b for particular solution
    print("residual ||A c - b||:", np.linalg.norm(A1 @ c - b1))
