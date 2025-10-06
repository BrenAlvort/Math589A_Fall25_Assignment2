# general_linear_solver.py
"""
PAQ = LU (following Algorithm 4.2 partial row pivoting)
Author: (replace with your name if desired)

Exports:
 - paq_lu_inplace(A, tol=1e-12)
 - solve_via_paq(A, P, r, b, tol=1e-12)
 - solve(A, b, tol=1e-12)  # autograder entry point

This implementation follows Algorithm 4.2 exactly (partial pivoting by rows).
No column permutations are performed (Q = identity).
The matrix A is overwritten in-place with L multipliers (strict lower part)
and U in the upper part; unit diagonal of L is implicit.
"""
import numpy as np


def paq_lu_inplace(A, tol=1e-12):
    """
    In-place LU with simulated row exchanges following Algorithm 4.2 (partial row pivoting).
    Works for rectangular A (m x n). Does NOT perform column pivots (Q = identity).

    Parameters
    ----------
    A : ndarray (m, n)
        Matrix to factor. Will be mutated in-place (float dtype recommended).
    tol : float
        Pivot tolerance.

    Returns
    -------
    P : list of int, length m
        Simulated row-permutation vector (logical -> original row index).
    r : int
        Number of pivots found (rank estimate).
    Notes
    -----
    - No column permutation is used: virtual column permutation Q is identity.
    - After the routine A contains L multipliers in strict lower triangle
      and U in the upper triangle. Unit diagonal of L is implicit.
    """
    A = np.asarray(A)
    if not np.issubdtype(A.dtype, np.floating):
        # ensure float to allow divisions
        A = A.astype(float, copy=False)

    m, n = A.shape
    P = list(range(m))  # simulated row ordering
    k = 0               # current logical pivot row index

    # Loop over columns in the Algorithm 4.2 style
    for j in range(n):
        if k >= m:
            break

        # Find pivot row p among rows k..m-1 maximizing abs(A[P[row], j])
        p = k
        max_val = abs(A[P[k], j])
        max_row = k
        for row in range(k + 1, m):
            val = abs(A[P[row], j])
            if val > max_val:
                max_val = val
                max_row = row
        p = max_row

        # If pivot is (near) zero, skip this column (no pivot here)
        if max_val <= tol:
            continue

        # Simulate row swap P[k] <-> P[p]
        if p != k:
            P[k], P[p] = P[p], P[k]

        pivot = A[P[k], j]
        # Perform elimination for rows i = k+1 .. m-1
        for i in range(k + 1, m):
            a_ik = A[P[i], j]
            if a_ik == 0.0:
                A[P[i], j] = 0.0
                continue
            m_ik = a_ik / pivot
            A[P[i], j] = m_ik  # store multiplier in L
            # Update remaining columns j+1 .. n-1
            for col in range(j + 1, n):
                A[P[i], col] -= m_ik * A[P[k], col]

        k += 1

    r = k
    return P, r


def solve_via_paq(A, P, r, b, tol=1e-12):
    """
    Solve A x = b using the in-place LU stored in A and simulated row-permutation P,
    following the decomposition produced by paq_lu_inplace (Q = identity).

    Returns:
      c : particular solution (n,) with free variables set to zero
      N : (n, n-r) nullspace mapping matrix (columns map free vars to full x)
      parameterize : callable(x_free) -> x
      pivot_cols : list of pivot column indices (logical/original)
      free_cols : list of free column indices
    """
    A = np.asarray(A)
    m, n = A.shape
    b = np.asarray(b).reshape(-1)
    if b.size != m:
        raise ValueError("b length must equal number of rows m")

    # Permute b by P (simulated row ordering): Pb[i] = b[P[i]]
    Pb = np.empty(m, dtype=float)
    for i in range(m):
        Pb[i] = b[P[i]]

    # Forward substitution Ly = Pb
    y = np.empty(m, dtype=float)
    for i in range(m):
        s = Pb[i]
        # L has multipliers stored in A[P[i], j] for j < r (but only columns up to n-1)
        # subtract contributions from previous rows j = 0..min(i-1, r-1)
        up_to = min(i, r)
        for j in range(up_to):
            s -= A[P[i], j] * y[j]
        y[i] = s

    # Back substitution on U (first r logical rows and first r columns are pivots)
    z = np.zeros(n, dtype=float)  # z is the vector of variables in column order (Q = identity)
    if r > 0:
        for i in range(r - 1, -1, -1):
            s = y[i]
            for j in range(i + 1, r):
                s -= A[P[i], j] * z[j]
            denom = A[P[i], i]
            if abs(denom) <= tol:
                raise np.linalg.LinAlgError(f"Near-zero pivot at U[{i},{i}] = {denom}")
            z[i] = s / denom

    # Particular solution: free vars (columns r..n-1) set to zero
    c = z.copy()  # since Q is identity, mapping is direct

    # Build N mapping free variables to full x
    num_free = n - r
    if num_free == 0:
        N = np.zeros((n, 0), dtype=float)
    else:
        N = np.zeros((n, num_free), dtype=float)
        for t in range(num_free):
            # logical free column index = r + t
            # rhs = -U_bb^{-1} * U_bf[:,t], where U_bf entries are A[P[i], r+t] for i=0..r-1
            rhs = np.empty(r, dtype=float)
            for i in range(r):
                rhs[i] = -A[P[i], r + t]
            zb = np.empty(r, dtype=float)
            for i in range(r - 1, -1, -1):
                s = rhs[i]
                for j in range(i + 1, r):
                    s -= A[P[i], j] * zb[j]
                denom = A[P[i], i]
                if abs(denom) <= tol:
                    raise np.linalg.LinAlgError(f"Near-zero pivot at U[{i},{i}] = {denom}")
                zb[i] = s / denom
            zfull = np.zeros(n, dtype=float)
            zfull[:r] = zb
            zfull[r + t] = 1.0
            N[:, t] = zfull

    pivot_cols = list(range(r))
    free_cols = list(range(r, n))

    def parameterize(x_free):
        x_free = np.asarray(x_free).reshape(-1)
        if x_free.size != num_free:
            raise ValueError(f"x_free must have length {num_free}")
        return c + N @ x_free

    return c, N, parameterize, pivot_cols, free_cols


def solve(A, b, tol=1e-12):
    """
    Autograder-friendly wrapper. Replaces/overwrites A internally (works on a copy).

    If there are no free variables (n == r) returns x (vector).
    Otherwise returns (c, N) where c is particular solution (free vars = 0)
    and N maps free variables to full x: x = c + N @ x_free.
    """
    A = np.asarray(A)
    b = np.asarray(b)
    # Work on a copy so we don't mutate the user's A
    A_work = A.copy().astype(float, copy=True)

    P, r = paq_lu_inplace(A_work, tol=tol)
    c, N, _, pivots, frees = solve_via_paq(A_work, P, r, b, tol=tol)

    # If no free variables, return single solution vector x
    if N.shape[1] == 0:
        return c  # unique solution vector
    else:
        return c, N


# Quick self-test when run directly:
if __name__ == "__main__":
    # Square, nonsingular example
    A0 = np.array([[2.0, 1.0], [4.0, -6.0]])
    b0 = np.array([5.0, -2.0])
    x_unique = solve(A0, b0)
    print("unique x:", x_unique, "residual:", np.linalg.norm(A0 @ x_unique - b0))

    # Underdetermined example
    A1 = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    b1 = np.array([6.0, 12.0])
    c, N = solve(A1, b1)
    print("particular c:", c)
    print("N shape:", N.shape)
    # verify Ax = b for particular solution c
    print("residual (particular):", np.linalg.norm(A1 @ c - b1))
