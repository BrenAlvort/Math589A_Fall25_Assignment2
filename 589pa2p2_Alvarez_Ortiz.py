import numpy as np

def paq_lu_inplace(A, tol=1e-12):
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
            local_max_idx = np.argmax(col_segment)
            local_max_val = col_segment[local_max_idx]
            if local_max_val > best_val:
                best_val = local_max_val
                best_col = col_log
                best_row = k + int(local_max_idx)
        if best_val <= tol:
            break
        if best_col != k:
            Q[k], Q[best_col] = Q[best_col], Q[k]
        if best_row != k:
            P[k], P[best_row] = P[best_row], P[k]
        piv_r = P[k]
        piv_c = Q[k]
        piv = A[piv_r, piv_c]
        if abs(piv) <= tol:
            break
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

def forward_sub(L, b):
    m = L.shape[0]
    y = np.zeros(m, dtype=float)
    for i in range(m):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y

def back_sub_upper(U, b, tol=1e-12):
    p = U.shape[0]
    x = np.zeros(p, dtype=float)
    for i in range(p - 1, -1, -1):
        diag = U[i, i]
        if abs(diag) <= tol:
            raise np.linalg.LinAlgError("Singular triangular matrix in back substitution")
        x[i] = (b[i] - np.dot(U[i, i+1:], x[i+1:])) / diag
    return x

def solve_paq_lu(A_input, b_input, tol=1e-12):
    A = A_input.copy().astype(float)
    b = b_input.copy().astype(float)
    m, n = A.shape
    A_over, P, Q, p, pivot_cols = paq_lu_inplace(A, tol=tol)
    A_reordered = A_over[P, :][:, Q]
    L_full = np.eye(m, dtype=float)
    for i in range(m):
        for j in range(min(i, n)):
            L_full[i, j] = A_reordered[i, j]
    U_full = np.zeros((m, n), dtype=float)
    for i in range(min(m, n)):
        for j in range(i, n):
            U_full[i, j] = A_reordered[i, j]
    Pb = b[P]
    z = forward_sub(L_full, Pb)
    inconsistent = False
    if p < m:
        residual_tail_norm = np.linalg.norm(z[p:])
        if residual_tail_norm > max(1e-12, tol * np.linalg.norm(Pb)):
            inconsistent = True
    if inconsistent:
        return {
            'status': {'has_solution': False, 'inconsistent': True, 'pivot_count': p, 'pivot_cols': pivot_cols},
            'c': None, 'N': None, 'param': None,
            'extra': {'P': P, 'Q': Q, 'A_over': A_over, 'L_full': L_full, 'U_full': U_full, 'z': z}
        }
    U_top = U_full[:p, :]
    z_top = z[:p]
    U_B = U_top[:, :p] if p > 0 else np.zeros((0,0))
    U_F = U_top[:, p:] if n > p else np.zeros((p, 0), dtype=float)
    if p > 0:
        xB_part = back_sub_upper(U_B, z_top, tol=tol)
    else:
        xB_part = np.zeros(0, dtype=float)
    if n - p > 0 and p > 0:
        X = np.zeros_like(U_F)
        for j in range(U_F.shape[1]):
            X[:, j] = back_sub_upper(U_B, U_F[:, j], tol=tol)
        N_B = -X
    else:
        N_B = np.zeros((p, n-p), dtype=float)
    c_logical = np.zeros(n, dtype=float)
    c_logical[:p] = xB_part
    n_free = n - p
    if n_free > 0:
        N_logical = np.zeros((n, n_free), dtype=float)
        N_logical[:p, :] = N_B
        N_logical[p:, :] = np.eye(n_free, dtype=float)
    else:
        N_logical = np.zeros((n, 0), dtype=float)
    Qmat = np.eye(n)[:, Q]
    c = Qmat @ c_logical
    N = Qmat @ N_logical
    def param(x_free):
        x_free = np.asarray(x_free, dtype=float).reshape(-1)
        if x_free.size != n_free:
            raise ValueError(f"x_free should have length {n_free}, got {x_free.size}")
        return c + N @ x_free
    return {
        'status': {'has_solution': True, 'inconsistent': False, 'pivot_count': p, 'pivot_cols': pivot_cols},
        'c': c, 'N': N, 'param': param,
        'extra': {'P': P, 'Q': Q, 'A_over': A_over, 'L_full': L_full, 'U_full': U_full, 'z': z}
    }

if __name__ == "__main__":
    A0 = np.array([
        [2.,  1.,  1.],
        [4., -6.,  0.],
        [-2., 7.,  2.],
        [1.,  5.,  3.]
    ], dtype=float)
    b = np.array([1., 2., 3., 4.], dtype=float)
    out = solve_paq_lu(A0, b)
    print(out['status'])
