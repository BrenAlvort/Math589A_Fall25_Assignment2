import numpy as np

def paq_lu_inplace(A, tol=1e-12):
    """
    In-place PAQ = LU decomposition with simulated row exchanges and virtual column exchanges.

    Returns:
        A (mutated): contains L (strict lower, unit diag implicit) and U (upper)
        P: row-permutation vector (logical -> physical)
        Q: column-permutation vector (logical -> physical)
        pivot_count: number of pivots found
        pivot_cols: original indices of pivot columns
    """
    m, n = A.shape
    P = np.arange(m, dtype=int)
    Q = np.arange(n, dtype=int)

    min_dim = min(m, n)
    pivot_count = 0

    for k in range(min_dim):
        # find best pivot in submatrix rows k..m-1, cols k..n-1
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

        # virtual column swap
        if best_col != k:
            Q[k], Q[best_col] = Q[best_col], Q[k]

        # simulated row swap
        if best_row != k:
            P[k], P[best_row] = P[best_row], P[k]

        piv_r = P[k]
        piv_c = Q[k]
        piv = A[piv_r, piv_c]
        if abs(piv) <= tol:
            break

        # elimination: store multipliers in strict lower part (physical positions)
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


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    # Example rectangular matrix (4x3)
    A0 = np.array([
        [2.,  1.,  1.],
        [4., -6.,  0.],
        [-2., 7.,  2.],
        [1.,  5.,  3.]
    ], dtype=float)

    # Work on a copy so we keep the original for verification
    A = A0.copy()

    # Do PAQ = LU in-place
    A_over, P, Q, pivot_count, pivot_cols = paq_lu_inplace(A)

    print("=== PAQ = LU Decomposition ===")
    print("Original A:\n", A0)
    print("\nOverwritten A (contains L and U in physical storage):\n", A_over)
    print("Row permutation P (logical -> physical):", P)
    print("Column permutation Q (logical -> physical):", Q)
    print("Pivot count:", pivot_count)
    print("Pivot columns (original indices):", pivot_cols)

    # -------------------------
    # Rectangular-safe verify (corrected)
    # -------------------------
    m, n = A0.shape
    Pmat = np.eye(m)[P, :]    # permute rows
    Qmat = np.eye(n)[:, Q]    # permute cols

    # IMPORTANT: reorder the overwritten matrix into logical order (apply P and Q to stored A)
    # A_reordered[i,j] = A_over[P[i], Q[j]]
    A_reordered = A_over[P, :][:, Q]   # shape (m,n)

    # Extract L (unit diag implicit) and U from the reordered matrix
    L_logical = np.tril(A_reordered, -1) + np.eye(m, n)  # shape (m,n)
    U_logical = np.triu(A_reordered)                    # shape (m,n)

    p = pivot_count
    if p == 0:
        print("\nNo pivots found (zero matrix or numerical rank 0).")
    else:
        # L_block is p x p, U_block is p x n -> product is p x n
        L_block = L_logical[:p, :p]    # (p, p)
        U_block = U_logical[:p, :]     # (p, n)
        PAQ_block = (Pmat @ A0 @ Qmat)[:p, :]  # (p, n)

        LU_recon = L_block @ U_block           # (p, n)
        error = np.linalg.norm(PAQ_block - LU_recon)

        print("\nVerification (rectangular-safe):")
        print("PAQ_block shape:", PAQ_block.shape)
        print("L_block shape:", L_block.shape, "U_block shape:", U_block.shape)
        print("‖PAQ_block - L_block @ U_block‖_F =", error)
        if error < 1e-10:
            print("✅ Decomposition verified successfully.")
        else:
            print("❌ Decomposition error is large.")
