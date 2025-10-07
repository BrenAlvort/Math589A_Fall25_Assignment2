# general_linear_solver.py
"""
general_linear_solver.py
PAQ = LU factorization (in-place) and solver returning (N, c)
where x = c + N @ x_free are all solutions of A x = b.

Author: ChatGPT (adapted for Math589A assignment)
Default pivot tolerance: 1e-6 (professor's TOL)
"""
from typing import List, Tuple
import numpy as np

TOL_DEFAULT = 1e-6  # use professor's recommended tolerance


def paq_lu_inplace(A: np.ndarray, tol: float = TOL_DEFAULT) -> Tuple[List[int], List[int], int]:
    """
    In-place PAQ = LU factorization for rectangular A (m x n).

    A is mutated in-place to hold multipliers (L) in strict lower triangle
    and U in the upper triangle. Unit diagonal of L is implicit.

    Pivot strategy:
      - At step (k,j) choose the column among j..n-1 whose maximum absolute entry
        in rows k..m-1 is largest; then pick the corresponding row (this moves
        pivot columns to the front via virtual column permutation Q).
      - Simulated row swaps are tracked by P (no physical row swaps).
    This is the same column-selection approach used earlier in the conversation
    and consistent with derivation.tex (we keep pivot columns at the front via Q).

    Parameters:
      A : ndarray (m,n) - will be modified in-place (must be float dtype)
      tol : float - pivot tolerance (use 1e-6 by default)

    Returns:
      P : list length m, P[i] = original row index at logical row i
      Q : list length n, Q[j] = original col index at logical col j
      r : int, number of pivots found (rank estimate)
    """
    if not isinstance(A, np.ndarray):
        raise ValueError("paq_lu_inplace expects a numpy.ndarray and performs in-place edits")
    if not np.issubdtype(A.dtype, np.floating):
        # convert to float in place by creating a float copy (caller should pass a float copy if they want originals preserved)
        A[:] = A.astype(np.float64)

    m, n = A.shape
    P = list(range(m))
    Q = list(range(n))
    r = 0
    k = 0  # logical pivot row
    j = 0  # logical pivot column

    while k < m and j < n:
        # Find best pivot column and row in the submatrix rows k..m-1, cols j..n-1
        best_val = tol
        best_col = -1
        best_row = -1
        for col in range(j, n):
            # find max in this column among logical rows k..m-1
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
            # no pivot found above tolerance -> done
            break

        # Virtual column swap: put best_col into logical column j
        if best_col != j:
            Q[j], Q[best_col] = Q[best_col], Q[j]

        # Simulate row swap: put best_row into logical row k
        if best_row != k:
            P[k], P[best_row] = P[best_row], P[k]

        # pivot is at A[P[k], Q[j]]
        pivot = A[P[k], Q[j]]
        # pivot should be > tol by construction
        # store multipliers and eliminate below pivot
        for i in range(k + 1, m):
            a_ik = A[P[i], Q[j]]
            if a_ik == 0.0:
                A[P[i], Q[j]] = 0.0
                continue
            mult = a_ik / pivot
            A[P[i], Q[j]] = mult  # store multiplier (L)
            # update trailing columns
            for col in range(j + 1, n):
                A[P[i], Q[col]] -= mult * A[P[k], Q[col]]

        r += 1
        k += 1
        j += 1

    return P, Q, r


def solve_via_paq(A: np.ndarray, P: List[int], Q: List[int], r: int,
                  b: np.ndarray, tol: float = TOL_DEFAULT) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given A overwritten with in-place PAQ=LU (as produced by paq_lu_inplace),
    and permutations P,Q with pivot count r, compute (N, c) such that
        x = c + N @ x_free
    are all solutions of A x = b.

    Returns:
      N : ndarray (n, n-r) nullspace mapping (columns are basis vectors)
      c : ndarray (n,) particular solution (free vars set to 0)
    If the system is inconsistent, returns (N_nan, c_nan) where c_nan is all NaNs
    and N_nan is all NaNs (shapes preserved).
    """
    m, n = A.shape
    b = np.asarray(b).reshape(-1)
    if b.size != m:
        raise ValueError("b must have length m")

    # Permute b by P (simulated row ordering)
    Pb = np.empty(m, dtype=float)
    for i in range(m):
        Pb[i] = b[P[i]]

    # Forward substitution Ly = Pb (L has unit diagonal)
    y = np.empty(m, dtype=float)
    for i in range(m):
        s = Pb[i]
        # L multipliers exist only in logical columns j < r
        up_to = min(i, r)
        for j in range(up_to):
            s -= A[P[i], Q[j]] * y[j]
        y[i] = s

    # Inconsistency check: entries of y corresponding to zero rows of U must be zero
    if r < m:
        if np.max(np.abs(y[r:])) > tol:
            # inconsistent -> return NaNs with proper shapes
            N_nan = np.full((n, max(0, n - r)), np.nan)
            c_nan = np.full(n, np.nan)
            return N_nan, c_nan

    # y_pivot = leading r entries of y
    y_pivot = y[:r].copy()

    # Back substitution on U_B to compute z (permuted variable vector)
    z = np.zeros(n, dtype=float)  # this is x' in logical (Q) ordering
    if r > 0:
        for i in range(r - 1, -1, -1):
            s = y_pivot[i]
            for j in range(i + 1, r):
                s -= A[P[i], Q[j]] * z[j]
            denom = A[P[i], Q[i]]
            if abs(denom) <= tol:
                # near-zero pivot encountered
                raise np.linalg.LinAlgError(f"Near-zero pivot at U[{i},{i}] = {denom}")
            z[i] = s / denom
    # z[r:] remain zeros (free vars = 0) for particular solution

    # Map particular solution to original ordering: c[Q[j]] = z[j]
    c = np.zeros(n, dtype=float)
    for j in range(n):
        c[Q[j]] = z[j]

    # Build nullspace mapping N (n x (n-r))
    num_free = n - r
    if num_free <= 0:
        N = np.zeros((n, 0), dtype=float)
        return N, c

    N = np.zeros((n, num_free), dtype=float)
    for t in range(num_free):
        j_free = r + t  # logical column index of t-th free variable
        # Build rhs = -U_B^{-1} * U_bf[:,t] (U_bf entries are A[P[i], Q[j_free]] for i=0..r-1)
        rhs = np.empty(r, dtype=float)
        for i in range(r):
            rhs[i] = -A[P[i], Q[j_free]]
        # Solve U_B * zb = rhs by back substitution
        zb = np.empty(r, dtype=float)
        for i in range(r - 1, -1, -1):
            s = rhs[i]
            for j in range(i + 1, r):
                s -= A[P[i], Q[j]] * zb[j]
            denom = A[P[i], Q[i]]
            if abs(denom) <= tol:
                raise np.linalg.LinAlgError(f"Near-zero pivot at U[{i},{i}] = {denom}")
            zb[i] = s / denom
        # zfull (logical order): [zb; e_t]
        zfull = np.zeros(n, dtype=float)
        zfull[:r] = zb
        zfull[j_free] = 1.0
        # map to original ordering xcol[Q[j]] = zfull[j]
        xcol = np.zeros(n, dtype=float)
        for j in range(n):
            xcol[Q[j]] = zfull[j]
        N[:, t] = xcol

    return N, c


def solve(A: np.ndarray, b: np.ndarray, tol: float = TOL_DEFAULT) -> Tuple[np.ndarray, np.ndarray]:
    """
    Autograder-friendly wrapper: returns (N, c) for A x = b.

    - Always returns two values (N, c).
    - N has shape (n, n-r). If n == r, N has shape (n, 0).
    - c has shape (n,).
    - If inconsistent, both are returned filled with np.nan (shapes preserved).

    Note: This function works on a copy of A (does not mutate the provided A).
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m, n = A.shape
    # Work on a copy so we don't modify caller's matrix
    A_work = A.copy()

    P, Q, r = paq_lu_inplace(A_work, tol=tol)
    N, c = solve_via_paq(A_work, P, Q, r, b, tol=tol)

    # Ensure N shape is (n, n-r)
    if N is None:
        N = np.zeros((n, max(0, n - r)), dtype=float)

    # Always return (N, c)
    return N, c


# quick manual sanity tests when run directly
if __name__ == "__main__":
    np.set_printoptions(precision=8, suppress=True)

    # square nonsingular
    A0 = np.array([[3.0, 1.0], [4.0, -2.0]], dtype=float)
    b0 = np.array([5.0, -6.0], dtype=float)
    N0, c0 = solve(A0, b0)
    print("Square nonsingular -> N shape:", N0.shape, "c:", c0, "resid:", np.linalg.norm(A0 @ c0 - b0))

    # rectangular underdetermined
    A1 = np.array([[1., 2., 3., 4.],
                   [2., 4., 6., 8.],
                   [0., 1., 1., 2.]], dtype=float)
    b1 = np.array([10., 20., 5.], dtype=float)
    N1, c1 = solve(A1, b1)
    print("Rectangular underdetermined -> N shape:", N1.shape)
    print("||A c - b||:", np.linalg.norm(A1 @ c1 - b1))
    print("||A N|| (should be near 0):", np.linalg.norm(A1 @ N1))

    # inconsistent
    A2 = np.array([[1., 1.], [1., 1.]], dtype=float)
    b2 = np.array([1., 2.], dtype=float)
    N2, c2 = solve(A2, b2)
    print("Inconsistent -> c has NaNs:", np.isnan(c2).all(), "N has NaNs:", np.isnan(N2).all())
