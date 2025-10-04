import numpy as np
from copy import deepcopy

def solve_paq_lu(A_in, b, tol=1e-12):
    """
    Solve A x = b using PAQ = LU decomposition (from Problem 1).
    Returns a dict with:
        'c' : one particular solution (free vars = 0)
        'N' : nullspace matrix mapping free vars -> full solution
        'param': function(x_free) -> full x
        'status': diagnostic info
    """
    from 589pa2p1_Alvarez_Ortiz import paq_lu_inplace  # or import from same file if merged

    A = deepcopy(A_in).astype(float)
    m, n = A.shape
    b = np.array(b, dtype=float).reshape(m)

    # Step 1: PAQ = LU
    A_over, P, Q, pivot_count, pivot_cols = paq_lu_inplace(A, tol)
    rank = pivot_count
    p = rank

    # Apply row permutation to b: Pb
    Pb = b[P]

    # Step 2: Forward substitution Ly = Pb (unit diagonal)
    y = np.zeros_like(Pb)
    for i in range(m):
        sum_Ly = 0.0
        for j in range(i):
            sum_Ly += A_over[P[i], Q[j]] * y[j]
        y[i] = Pb[i] - sum_Ly

    # Step 3: Extract U (top p rows)
    U = np.zeros((p, n))
    for i in range(p):
        for j in range(n):
            if j >= i:
                U[i, j] = A_over[P[i], Q[j]]

    # Step 4: Check for inconsistency
    # If any zero row in U and corresponding y != 0
    if p < m:
        for i in range(p, m):
            if abs(y[i]) > tol:
                return {
                    'status': {'has_solution': False, 'inconsistent': True, 'pivot_count': p}
                }

    # Step 5: Partition variables
    basic_idx = Q[:p]
    free_idx = Q[p:]
    n_free = n - p

    # U_basic is p×p; U_free is p×n_free
    U_basic = U[:, :p]
    U_free = U[:, p:]

    # Step 6: Back substitution for particular solution (x_free = 0)
    x_basic = np.zeros(p)
    for i in reversed(range(p)):
        rhs = y[i]
        rhs -= np.dot(U_basic[i, i+1:], x_basic[i+1:])
        x_basic[i] = rhs / U_basic[i, i]

    c_full = np.zeros(n)
    c_full[basic_idx] = x_basic

    # Step 7: Build nullspace matrix N (for free vars)
    if n_free > 0:
        N = np.zeros((n, n_free))
        # for each free variable e_j (one at a time)
        for j in range(n_free):
            x_basic_j = np.zeros(p)
            col = -U_free[:, j]
            for i in reversed(range(p)):
                rhs = col[i] - np.dot(U_basic[i, i+1:], x_basic_j[i+1:])
                x_basic_j[i] = rhs / U_basic[i, i]
            x_full = np.zeros(n)
            x_full[basic_idx] = x_basic_j
            x_full[free_idx[j]] = 1.0
            N[:, j] = x_full
    else:
        N = np.zeros((n, 0))

    # Step 8: Parameterization function
    def param(x_free):
        x_free = np.array(x_free, dtype=float).reshape(-1)
        return c_full + (N @ x_free)

    return {
        'c': c_full,
        'N': N,
        'param': param,
        'status': {
            'has_solution': True,
            'inconsistent': False,
            'pivot_count': p
        }
    }
