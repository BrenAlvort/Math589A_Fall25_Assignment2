from typing import List, Tuple, Callable
import numpy as np


def paq_lu_inplace(A: np.ndarray, tol: float = 1e-12) -> Tuple[List[int], List[int], int]:
    """
    In-place PAQ = LU decomposition with partial/full pivoting and column permutation.
    Works for rectangular A (m x n). P and Q track the original indices.

    NOTE: This function now operates IN-PLACE on the array A passed in and
    does NOT make an internal copy. Caller should pass a working copy if
    they don't want the original mutated.
    """
    # Validate input and ensure float dtype; do NOT make a copy here so the caller's A
    # (e.g. A_work) is actually overwritten with the LU factors.
    if not isinstance(A, np.ndarray):
        raise ValueError("paq_lu_inplace expects a numpy.ndarray and performs in-place edits")
    if not np.issubdtype(A.dtype, np.floating):
        # convert in place by view/copy: we must replace contents with float copy
        # but we do that by copying into the same variable, caller passed an array reference
        # so they must pass a float array to truly avoid external copying. We still proceed.
        A[:] = A.astype(float)

    m, n = A.shape
    P = list(range(m))  # P maps logical row index -> original row index
    Q = list(range(n))  # Q maps logical column index -> original column index
    r = 0  # Rank (number of pivots found)
    k = 0  # logical pivot row index
    j = 0  # logical pivot column index

    # Loop until out of logical pivot positions
    while k < m and j < n:
        # --- 1. Find Optimal Pivot (Full Pivoting over submatrix) ---
        best_row_idx = -1
        best_col_idx = -1
        # initialize best_val to tol so we only accept pivots > tol
        best_val = tol

        # Search the submatrix A[P[k:m], Q[j:n]] for the largest abs value
        for col_idx in range(j, n):
            for row_idx in range(k, m):
                val = abs(A[P[row_idx], Q[col_idx]])
                if val > best_val:
                    best_val = val
                    best_col_idx = col_idx
                    best_row_idx = row_idx

        # Stop if no sufficiently large pivot is found (rank is r)
        if best_row_idx == -1:  # No pivot > tol found
            break

        # --- 2. Perform Permutations (Virtual Swaps) ---
        # Column swap: move the best column (logical index best_col_idx) to position j
        if best_col_idx != j:
            Q[j], Q[best_col_idx] = Q[best_col_idx], Q[j]

        # Row swap: move the row containing the pivot (logical index best_row_idx) to position k
        if best_row_idx != k:
            P[k], P[best_row_idx] = P[best_row_idx], P[k]

        # Now the pivot is guaranteed to be at A[P[k], Q[j]] (original indices)

        # --- 3. Elimination ---
        pivot = A[P[k], Q[j]]
        r += 1  # We found a pivot, increment rank

        # Eliminate entries below the pivot (rows k+1 to m-1)
        for i in range(k + 1, m):
            # If pivot is zero this division will produce inf/nan; pivot>tol by construction
            m_ik = A[P[i], Q[j]] / pivot
            A[P[i], Q[j]] = m_ik  # Store multiplier in L

            # Update trailing submatrix entries (columns j+1 .. n-1)
            for col_idx in range(j + 1, n):
                A[P[i], Q[col_idx]] -= m_ik * A[P[k], Q[col_idx]]

        # Advance to the next logical pivot position
        k += 1
        j += 1

    # After the loop, the first r columns of Q are the pivot columns (B),
    # and the remaining n-r columns are the free columns (F).
    return P, Q, r


# -- keep solve_via_paq and solve mostly as you wrote them, minor style cleanups --


def solve_via_paq(A: np.ndarray, P: List[int], Q: List[int], r: int,
                  b: np.ndarray, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, Callable, List[int], List[int]]:
    """
    Construct the general solution x = c + N * x_free using the PAQ=LU factorization.
    This implementation follows the steps derived in derivation.tex.
    Assumes A already holds the LU factors (multipliers in strict lower, U in upper).
    """
    m, n = A.shape
    b = np.asarray(b).reshape(-1)
    if b.size != m:
        raise ValueError("b must have length m")

    # --- 1. Compute y = L^-1 P b ---
    Pb = b[np.asarray(P[:m])]   # Pb[i] = b[P[i]]
    y = np.zeros(m, dtype=float)

    # Forward substitution Ly = Pb. L is unit lower triangular.
    for i in range(m):
        s = Pb[i]
        # Only logical pivot columns j < r contribute multipliers
        for j in range(min(i, r)):
            s -= A[P[i], Q[j]] * y[j]
        y[i] = s

    # --- 2. Check for Inconsistency ---
    if r < m:
        if np.linalg.norm(y[r:]) > tol:
            # inconsistent
            c_nan = np.full(n, np.nan)
            N_nan = np.full((n, n - r), np.nan)
            return c_nan, N_nan, (lambda xf: np.full(n, np.nan)), Q[:r].copy(), Q[r:].copy()

    # y_pivot is the leading r entries of y
    y_pivot = y[:r]

    # --- 3. Compute Particular Solution c (back-sub U_B x'_B = y_pivot with x'_F = 0) ---
    z = np.zeros(n, dtype=float)  # z is in permuted logical order

    for i in range(r - 1, -1, -1):
        s = y_pivot[i]
        for j in range(i + 1, r):
            s -= A[P[i], Q[j]] * z[j]
        denom = A[P[i], Q[i]]
        if abs(denom) <= tol:
            raise np.linalg.LinAlgError(f"Near-zero pivot at U[{i},{i}] = {denom}")
        z[i] = s / denom

    # Map z back to original ordering c[Q[j]] = z[j]
    c = np.zeros(n, dtype=float)
    for j in range(n):
        c[Q[j]] = z[j]

    # --- 4. Nullspace mapping N ---
    num_free = n - r
    if num_free == 0:
        N = np.zeros((n, 0), dtype=float)
    else:
        N = np.zeros((n, num_free), dtype=float)
        for t in range(num_free):
            j_free = r + t
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

    If unique solution (n == r) returns x (n,).
    Otherwise returns (c, N) where c is particular solution and N is nullspace basis.
    """
    A = np.asarray(A)
    b = np.asarray(b)

    # make a working copy so caller's A is preserved (caller can pass A.copy() if they want in-place)
    A_work = A.copy().astype(float, copy=True)

    P, Q, r = paq_lu_inplace(A_work, tol=tol)
    c, N, _, _, _ = solve_via_paq(A_work, P, Q, r, b, tol=tol)

    if np.isnan(c).any():
        # inconsistent
        if N.shape[1] == 0:
            return np.full(A.shape[1], np.nan)
        else:
            return c, N

    if N.shape[1] == 0:
        return c  # unique solution
    else:
        return c, N
