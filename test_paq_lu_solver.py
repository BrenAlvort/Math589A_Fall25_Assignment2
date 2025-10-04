# test_paq_lu_solver.py
import os
import importlib.util
import numpy as np

HERE = os.path.dirname(__file__)
MODULE_FILENAME = os.path.join(HERE, "589pa2p2_Alvarez_Ortiz.py")

def load_user_module():
    spec = importlib.util.spec_from_file_location("user_module", MODULE_FILENAME)
    user = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user)
    return user

def test_square_full_rank_unique_solution():
    np.random.seed(0)
    user = load_user_module()
    n = 5
    A = np.random.randn(n, n)
    A += np.eye(n) * 0.1
    x_true = np.random.randn(n)
    b = A @ x_true
    out = user.solve_paq_lu(A, b)
    assert out['status']['has_solution'] is True
    p = out['status']['pivot_count']
    assert p == n
    c = out['c']
    assert np.linalg.norm(A @ c - b) < 1e-9
    assert np.allclose(c, x_true, atol=1e-8)

def test_underdetermined_free_variables():
    np.random.seed(1)
    user = load_user_module()
    m, n = 3, 6
    A = np.random.randn(m, n)
    while np.linalg.matrix_rank(A) < m:
        A = np.random.randn(m, n)
    x_true = np.random.randn(n)
    b = A @ x_true
    out = user.solve_paq_lu(A, b)
    assert out['status']['has_solution'] is True
    p = out['status']['pivot_count']
    assert p == np.linalg.matrix_rank(A)
    n_free = n - p
    if n_free > 0:
        x_free = np.random.randn(n_free)
        x = out['param'](x_free)
        assert np.linalg.norm(A @ x - b) < 1e-9
        assert np.linalg.norm(A @ out['c'] - b) < 1e-9

def test_overdetermined_consistent_full_col_rank():
    np.random.seed(2)
    user = load_user_module()
    m, n = 8, 4
    A = np.random.randn(m, n)
    while np.linalg.matrix_rank(A) < n:
        A = np.random.randn(m, n)
    x_true = np.random.randn(n)
    b = A @ x_true
    out = user.solve_paq_lu(A, b)
    assert out['status']['has_solution'] is True
    p = out['status']['pivot_count']
    assert p == n
    c = out['c']
    assert np.linalg.norm(A @ c - b) < 1e-9
    assert np.allclose(c, x_true, atol=1e-8)

def test_inconsistent_system_detected():
    user = load_user_module()
    m, n = 4, 3
    A = np.zeros((m, n))
    b = np.array([1.0, 0.0, 0.0, 0.0])
    out = user.solve_paq_lu(A, b)
    assert out['status']['has_solution'] is False
    assert out['status']['inconsistent'] is True

def test_rank_deficient_consistent():
    user = load_user_module()
    A = np.array([[1., 0.],
                  [0., 0.]])
    b = np.array([1., 0.])
    out = user.solve_paq_lu(A, b)
    assert out['status']['has_solution'] is True
    p = out['status']['pivot_count']
    assert p == 1
    for _ in range(5):
        x_free = np.random.randn(1)
        x = out['param'](x_free)
        assert np.linalg.norm(A @ x - b) < 1e-9
