# general_linear_solver.py
import numpy as np

def paq_lu_inplace(A, tol=1e-12):
    """
    In-place PAQ = LU decomposition (rectangular A allowed).
    Uses simulated row swaps (P) and virtual column swaps (Q).
    The algorithm follows the structure of Algorithm 4.2 (partial pivoting),
    extended to choose pivot columns among remaining columns and record them
    in Q (virtual swaps). A is overwritten in-place: strict lower part = L multipliers,
    upper part = U. Unit diagonal of L is implicit.

    Parameters
    ----------
    A : (m, n) numpy array (will be mutated in-place; dtype should be float)
    tol : pivot tolerance

    Returns
    -------
    P : list length m, simulated row ordering (logical -> original row index)
    Q : list length n, virtual column ordering (logical -> original col index)
    r : int, number of pivots found (rank estimate)
    """
    if not isinstance(A, np.ndarray):
        A = np.array(A, dtype=float)
    else:
        # ensure float dtype for divisions
        if not np.issubdtype(A.dtype, np.floating):
            A = A.astype(float, copy=False)

    m, n = A.shape
    P = list(range(m))   # logical row i corresponds to original row P[i]
    Q = list(range(n))   # logical col j corresponds to original col Q[j]
    r = 0

    # We iterate k (logical pivot row) and j (logical pivot column). We'll keep them the same:
    k = 0  # logical row index (0-based)
    j = 0  # logical column index (0-based)
    while k < m and j < n:
        # --- Choose pivot column and pivot row in Algorithm-4.2 style ---
        # For each candidate column col in j..n-1, find the largest |A[P[i], Q[col]]|
        # for rows i = k..m-1. Pick the column with the largest such maximum.
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

        # If no sufficiently large pivot found, stop
        if best_val <= tol:
            break

        # Virtual column swap: put best_col into logical position j
        if best_col != j:
            Q[j], Q[best_col] = Q[best_col], Q[j]

        # Simulate row swap: swap P[k] and P[best_row]
        if best_row != k:
            P[k], P[best_row] = P[best_row], P[k]

        # Pivot is at (logical row k, logical col j) -> physical (P[k], Q[j])
        pivot = A[P[k], Q[j]]
        if abs(pivot) <= tol:
            # Numerical safety (shouldn't happen because best_val > tol)
            j += 1
            continue

        # Elimination: for i = k+1 .. m-1
        for i in range(k+1, m):
            # multiplier m_ik = A[i,k] / A[k,k] where indices are logical -> physical via P,Q
            a_ik = A[P[i], Q[j]]
            if a_ik == 0.0:
                A[P[i], Q[j]] = 0.0
                continue
            m_ik = a_ik / pivot
            A[P[i], Q[j]] = m_ik  # store multiplier in strict lower triangle
            # Update row i for logical columns col = j+1 .. n-1
            for col in range(j+1, n):
                A[P[i], Q[col]] -= m_ik * A[P[k], Q[col]]

        # Move to next pivot position
        r += 1
        k += 1
        j += 1

    return P, Q, r


def solve_via_paq(A, P, Q, r, b, tol=1e-12):
    """
    Solve A x = b using the in-place PAQ = LU stored in A with P, Q, and pivot count r.

    Returns:
      c : particular solution (free variables = 0), shape (n,)
      N : nullspace mapping matrix, shape (n, n-r)
      parameterize : function(x_free) -> x (length n)
      pivot_cols : list of original column indices that are pivots (length r)
      free_cols : list of original column indices that are free (length n-r)
    """
    A = A  # mutated in-place by paq_lu_inplace already
    m, n = A.shape
    b = np.asarray(b).reshape(-1)
    if b.size != m:
        raise ValueError("b must have length m")

    # Permute b by simulated row ordering P: compute Pb (logical order)
    Pb = np.empty(m, dtype=float)
    for i in range(m):
        Pb[i] = b[P[i]]

    # Forward substitution Ly = Pb
    # L has implicit ones on diagonal for first r rows, and multipliers in A[P[i], Q[j]] for j<r
    y = np.empty(m, dtype=float)
    for i in range(m):
        s = Pb[i]
        # subtract contributions from earlier rows: j = 0 .. min(i-1, r-1)
        up_to = min(i, r)
        for j in range(up_to):
            s -= A[P[i], Q[j]] * y[j]
        # diagonal of L is 1 for i < r (unit lower triangular part)
        # For i >= r, there is no diagonal in the LU block, but the forward formula is same
        y[i] = s

    # Back substitution on U (first r logical rows/cols)
    z = np.zeros(n, dtype=float)  # z is the variable vector in permuted (logical Q) order
    if r > 0:
        for i in range(r-1, -1, -1):
            s = y[i]
            for j in range(i+1, r):
                s -= A[P[i], Q[j]] * z[j]
            denom = A[P[i], Q[i]]
            if abs(denom) <= tol:
                raise np.linalg.LinAlgError(f"Near-zero pivot at U[{i},{i}] = {denom}")
            z[i] = s / denom
    # z[r:] remain zero for the particular solution (free vars set to 0)

    # Build particular solution c in original column order
    c = np.zeros(n, dtype=float)
    for j in range(n):
        c[Q[j]] = z[j]

    # Build N mapping free variables (logical columns r..n-1) to full x
    num_free = n - r
    if num_free == 0:
        N = np.zeros((n, 0), dtype=float)
    else:
        N = np.zeros((n, num_free), dtype=float)
        # For each free logical column t (0..num_free-1) corresponding to logical col r+t:
        for t in range(num_free):
            # Build rhs = -U_bf[:, t] where U_bf column entries are A[P[i], Q[r+t]] for i = 0..r-1
            rhs = np.empty(r, dtype=float)
            for i in range(r):
                rhs[i] = - A[P[i], Q[r + t]]
            # Solve U_bb * zb = rhs by back-substitution
            zb = np.empty(r, dtype=float)
            for i in range(r-1, -1, -1):
                s = rhs[i]
                for j in range(i+1, r):
                    s -= A[P[i], Q[j]] * zb[j]
                denom = A[P[i], Q[i]]
                if abs(denom) <= tol:
                    raise np.linalg.LinAlgError(f"Near-zero pivot at U[{i},{i}] = {denom}")
                zb[i] = s / denom
            # Compose full z in logical order: first r entries = zb, entry r+t = 1, other free entries 0
            zfull = np.zeros(n, dtype=float)
            zfull[:r] = zb
            zfull[r + t] = 1.0
            # Map to original order: x[Q[j]] = zfull[j]
            xcol = np.zeros(n, dtype=float)
            for j in range(n):
                xcol[Q[j]] = zfull[j]
            N[:, t] = xcol

    pivot_cols = Q[:r].copy()
    free_cols = Q[r:].copy()

    def parameterize(x_free):
        x_free = np.asarray(x_free).reshape(-1)
        if x_free.size != (n - r):
            raise ValueError(f"x_free must have length {n - r}")
        return c + N @ x_free

    return c, N, parameterize, pivot_cols, free_cols


# Quick self-test
if __name__ == "__main__":
    # Example: rectangular matrix
    A0 = np.array([
        [2., 4., 1., 0., 3.],
        [4., 8., 2., 0., 6.],
        [1., 0., 0., 5., 2.],
        [0., 3., 0., 6., 1.]
    ], dtype=float)

    A = A0.copy()
    P, Q, r = paq_lu_inplace(A, tol=1e-14)
    print("P:", P)
    print("Q:", Q)
    print("r:", r)
    print("A after PAQ=LU:\n", A)

    b = np.array([1., 2., 3., 4.], dtype=float)
    c, N, param, pivots, frees = solve_via_paq(A, P, Q, r, b)
    print("pivots:", pivots)
    print("free cols:", frees)
    print("particular c:", c)
    print("N shape:", N.shape)
    if N.shape[1] > 0:
        xf = np.ones(N.shape[1])
        x = param(xf)
    else:
        x = c
    print("Check Ax ~= b, residual:", np.linalg.norm(A0 @ x - b))
