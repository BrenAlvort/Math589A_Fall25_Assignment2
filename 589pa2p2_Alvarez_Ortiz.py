# 589pa2p2_Alvarez_Ortiz.py
import numpy as np
from copy import deepcopy

# -------------------------
# Problem 1: PAQ = LU (in-place)
# -------------------------
def paq_lu_inplace(A, tol=1e-12):
    """
    In-place PAQ = LU decomposition with simulated row exchanges and virtual column exchanges.
    Mutates A in-place and returns (A_over, P, Q, pivot_count, pivot_cols)
    """
    m, n = A.shape
    P = np.arange(m, dtype=int)
    Q = np.arange(n, dtype=int)
    min_dim = min(m, n)
    pivot_count = 0

    for k in range(min_dim):
        best_val = 0.0
        best_row = -1
        best_col = -1
        for col_log in range(k, n):
            col_phys = Q[col_log]
            col_segment = np.abs(A[P[k:], col_phys])
            if col_segment.size == 0:
                continue
            local_max_idx = int(np.argmax(col_segment))
            local_max_val = float(col_segment[local_max_idx])
            if local_max_val > best_val:
                best_val = local_max_val
                best_col = col_log
                best_row = k + local_max_idx

        if best_val <= tol:
            break

        # virtual column swap
        if best_col != k:
            Q[k], Q[best_col] = Q[best_col], Q[k]

        # simulated row swap
        if best_row != k and best_row != -1:
            P[k], P[best_row] = P[best_row], P[k]

        piv_r = P[k]
        piv_c = Q[k]
        piv = A[piv_r, piv_c]
        if abs(piv) <= tol:
            break

        # elimination
        for i_log in range(k+1, m):
            i_phys = P[i_log]
            m_ik = A[i_phys, piv_c] / piv
            A[i_phys, piv_c] = m_ik
            for j_log in range(k+1, n):
                j_phys = Q[j_log]
                A[i_phys, j_phys] -= m_ik * A[piv_r, j_phys]

        pivot_count += 1

    pivot_cols = Q[:pivot_count].copy()
    return A, P, Q, pivot_count, pivot_cols


# -------------------------
# Utility substitutions
# -------------------------
def forward_sub_unit(L, b):
    """Forward substitution for L with unit diagonal. L is (m x m)."""
    m = L.shape[0]
    y = np.zeros(m, dtype=float)
    for i in range(m):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y

def back_sub_upper(U, b, tol=1e-12):
    """Back substitution for upper-triangular square U."""
    p = U.shape[0]
    x = np.zeros(p, dtype=float)
    for i in range(p - 1, -1, -1):
        diag = U[i, i]
        if abs(diag) <= tol:
            raise np.linalg.LinAlgError("Singular triangular matrix in back substitution")
        x[i] = (b[i] - np.dot(U[i, i+1:], x[i+1:])) / diag
    return x


# -------------------------
# Problem 2: solver using PAQ = LU
# -------------------------
def solve_paq_lu(A_input, b_input, tol=1e-12):
    """
    Full-featured solver returning a dict with keys:
      - status: dict (has_solution, inconsistent, pivot_count, pivot_cols)
      - c : particular solution (n,) (free vars = 0) or None if inconsistent
      - N : (n x n_free) matrix mapping free vars -> full solution (or None if inconsistent)
      - param : function(x_free)->x
      - extra : diagnostic info (P,Q,A_over,L_full,U_full,z)
    """
    A = deepcopy(A_input).astype(float)
    b = np.asarray(b_input, dtype=float).reshape(-1)
    m, n = A.shape
    if b.shape[0] != m:
        raise ValueError("b has incompatible length")

    A_over, P, Q, p, pivot_cols = paq_lu_inplace(A, tol=tol)

    # reorder stored A into logical order used during elimination:
    # A_reordered = A_over[physical_row_for_logical_i, physical_col_for_logical_j]
    A_reordered = A_over[P, :][:, Q]   # shape (m,n)

    # Build logical L (m x m with unit diagonal) and logical U (m x n)
    L_logical = np.eye(m, dtype=float)
    for i in range(m):
        for j in range(min(i, n)):
            L_logical[i, j] = A_reordered[i, j]

    U_logical = np.zeros((m, n), dtype=float)
    for i in range(min(m, n)):
        for j in range(i, n):
            U_logical[i, j] = A_reordered[i, j]

    # Solve L y = P b (apply simulated row perm to b)
    Pb = b[P]
    y = forward_sub_unit(L_logical, Pb)

    # Check consistency: y[p:] should be ~0 for a solution to exist
    inconsistent = False
    if p < m:
        tail_norm = np.linalg.norm(y[p:])
        if tail_norm > max(1e-12, tol * np.linalg.norm(Pb)):
            inconsistent = True

    if inconsistent:
        return {
            'status': {'has_solution': False, 'inconsistent': True, 'pivot_count': p, 'pivot_cols': pivot_cols},
            'c': None, 'N': None, 'param': None,
            'extra': {'P': P, 'Q': Q, 'A_over': A_over, 'L_logical': L_logical, 'U_logical': U_logical, 'y': y}
        }

    # Partition variables: first p logical columns are basic, remaining are free
    n_free = n - p
    # U_top (p x n) relates to basic/free variables
    U_top = U_logical[:p, :]        # shape (p, n)
    y_top = y[:p]                   # shape (p,)

    # Partition U_top = [U_B | U_F] with U_B (p x p), U_F (p x (n-p))
    if p > 0:
        U_B = U_top[:, :p]
    else:
        U_B = np.zeros((0, 0), dtype=float)
    U_F = U_top[:, p:] if n_free > 0 else np.zeros((p, 0), dtype=float)

    # Particular solution: set free variables to zero, solve U_B x_B = y_top
    if p > 0:
        xB_part = back_sub_upper(U_B, y_top, tol=tol)
    else:
        xB_part = np.zeros(0, dtype=float)

    # Build N_B = - U_B^{-1} U_F (p x n_free)
    if n_free > 0 and p > 0:
        # Solve U_B * X = U_F column-by-column
        X = np.zeros_like(U_F)
        for j in range(U_F.shape[1]):
            X[:, j] = back_sub_upper(U_B, U_F[:, j], tol=tol)
        N_B = -X
    else:
        N_B = np.zeros((p, n_free), dtype=float)

    # Build logical c and N (logical ordering)
    c_logical = np.zeros(n, dtype=float)
    c_logical[:p] = xB_part
    if n_free > 0:
        N_logical = np.zeros((n, n_free), dtype=float)
        N_logical[:p, :] = N_B
        N_logical[p:, :] = np.eye(n_free, dtype=float)
    else:
        N_logical = np.zeros((n, 0), dtype=float)

    # Map back to original column order using Q (logical -> original)
    Qmat = np.eye(n)[:, Q]   # multiply logical vector on left yields original order
    c = Qmat @ c_logical
    N = Qmat @ N_logical

    # parameterization function
    def param(x_free):
        x_free = np.asarray(x_free, dtype=float).reshape(-1)
        if x_free.size != n_free:
            raise ValueError(f"x_free must have length {n_free}")
        return c + N @ x_free

    return {
        'status': {'has_solution': True, 'inconsistent': False, 'pivot_count': p, 'pivot_cols': pivot_cols},
        'c': c, 'N': N, 'param': param,
        'extra': {'P': P, 'Q': Q, 'A_over': A_over, 'L_logical': L_logical, 'U_logical': U_logical, 'y': y}
    }


# -------------------------
# Compatibility function expected by grader: solve(A,b) -> (c, N)
# -------------------------
def solve(A, b):
    """
    Compatibility function for autograder: returns (c, N) where
    c is a particular solution (free vars = 0) and N is the matrix with
    columns giving independent directions for free variables.

    Additionally attach the parameterization function at `solve.param`.
    """
    out = solve_paq_lu(A, b)
    if not out['status']['has_solution']:
        # For an inconsistent system, follow previous behavior: raise or return (None, None).
        # We'll return (None, None) and keep param = None
        solve.param = None
        return None, None
    c = out['c']
    N = out['N']
    solve.param = out['param']
    return c, N


# -------------------------
# Demo / quick self-test when run directly
# -------------------------
if __name__ == "__main__":
    A0 = np.array([
        [2.,  1.,  1.],
        [4., -6.,  0.],
        [-2., 7.,  2.],
        [1.,  5.,  3.]
    ], dtype=float)
    b0 = np.array([1., 2., 3., 4.], dtype=float)

    print("Running demo on example A (4x3) and b.")
    out = solve_paq_lu(A0, b0)
    print("status:", out['status'])
    if out['status']['has_solution']:
        c = out['c']
        N = out['N']
        print("Particular solution c:", c)
        print("N shape:", N.shape)
        if N.shape[1] > 0:
            print("Example param with x_free = ones:", out['param'](np.ones(N.shape[1])))
