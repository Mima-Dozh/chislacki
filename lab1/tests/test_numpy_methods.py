import contextlib
import importlib
import io
import unittest

import numpy as np

from lab1 import problem1 as lu
from lab1 import problem2 as sweep
from lab1 import problem3 as iterations
from lab1 import problem4 as rotations
from lab1 import problem5 as qr


class LUTests(unittest.TestCase):
    def test_factorization_solve_inverse_and_input_unchanged(self):
        rng = np.random.default_rng(12)
        for size in (1, 2, 5, 9):
            for complex_input in (False, True):
                matrix = rng.normal(size=(size, size)) + 10 * np.eye(size)
                if complex_input:
                    matrix = matrix + 1j * rng.normal(size=(size, size))
                original = matrix.copy()
                factors = lu.get_LU(matrix)
                lower, upper = np.array(lu.get_L(factors)), np.array(lu.get_U(factors))
                np.testing.assert_allclose(lower @ upper, np.array(lu.get_P(factors)) @ matrix, atol=1e-12)
                for rhs in (rng.normal(size=size), rng.normal(size=(size, 3))):
                    original_rhs = rhs.copy()
                    result = lu.solve_lu(factors, rhs)
                    np.testing.assert_allclose(result, np.linalg.solve(matrix, rhs), atol=1e-12)
                    np.testing.assert_array_equal(rhs, original_rhs)
                np.testing.assert_allclose(lu.get_inverse(factors), np.linalg.inv(matrix), atol=1e-12)
                np.testing.assert_allclose(lu.get_determinant(factors), np.linalg.det(matrix), rtol=1e-12)
                np.testing.assert_array_equal(matrix, original)


def compact_rows(a, b, c):
    if len(b) == 1:
        return [[b[0]]]
    return [[b[0], c[0]]] + [[a[i], b[i], c[i]] for i in range(1, len(b) - 1)] + [[a[-1], b[-1]]]


class SweepTests(unittest.TestCase):
    def test_variant15(self):
        a = [0, -7, 4, -4, -1]
        b = [16, -16, 12, 12, 7]
        c = [-8, 5, 3, -7, 0]
        d = [0, -123, -68, 104, 20]
        p, q = sweep.get_P_Q(compact_rows(a, b, c), d)
        result = sweep.get_X(p, q)
        np.testing.assert_allclose(result, [2, 4, -9, 8, 4], atol=1e-12)
        sweep.check_answers(compact_rows(a, b, c), d, result)
        np.testing.assert_allclose(sweep.get_det_P(compact_rows(a, b, c), p), -339136)

    def test_diagonals_singleton_and_complex(self):
        rng = np.random.default_rng(15)
        for size in (1, 2, 8):
            a = np.r_[0, rng.normal(size=size - 1)]
            c = np.r_[rng.normal(size=size - 1), 0]
            for complex_input in (False, True):
                b = np.full(size, 5.0) + (1j if complex_input else 0)
                d = rng.normal(size=size)
                originals = [v.copy() for v in (a, b, c, d)]
                matrix = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
                p, q = sweep.get_P_Q(compact_rows(a, b, c), d)
                result = sweep.get_X(p, q)
                np.testing.assert_allclose(result, np.linalg.solve(matrix, d), atol=1e-12)
                sweep.check_answers(compact_rows(a, b, c), d, result)
                np.testing.assert_allclose(sweep.get_det_P(compact_rows(a, b, c), p), np.linalg.det(matrix), rtol=1e-12)
                for value, original in zip((a, b, c, d), originals):
                    np.testing.assert_array_equal(value, original)


class QRTests(unittest.TestCase):
    def test_householder_qr_zero_pivot_and_complex(self):
        for matrix in (np.zeros((3, 3)), np.array([[0, 2], [1, 3]]),
                       np.array([[1j, 2, 0], [3, 1, 4j], [0, -1j, 1]])):
            original = matrix.copy()
            q, r = map(np.array, qr.qr_decomposition(matrix))
            np.testing.assert_allclose(q @ r, matrix, atol=1e-12)
            np.testing.assert_allclose(q.conj().T @ q, np.eye(len(q)), atol=1e-12)
            np.testing.assert_allclose(np.tril(r, -1), 0, atol=1e-12)
            np.testing.assert_array_equal(matrix, original)

    def test_eigenvalues(self):
        rng = np.random.default_rng(9)
        examples = [np.zeros((3, 3)), np.eye(4), np.array([[2.]]),
                    np.array([[0, -1], [1, 0]]), np.array([[1, 2, 3], [0, 1, 2], [0, 0, 1]])]
        for size in (3, 5, 8):
            matrix = rng.normal(size=(size, size))
            examples.extend([matrix, matrix + 1j * rng.normal(size=(size, size))])
        for matrix in examples:
            original = matrix.copy()
            values, iterations = qr.qr_eigenvalues(matrix, tolerance=1e-11)
            expected = list(np.linalg.eigvals(matrix))
            for value in values:
                index = int(np.argmin(np.abs(np.asarray(expected) - value)))
                np.testing.assert_allclose(value, expected.pop(index), atol=1e-7, rtol=1e-7)
            self.assertLess(iterations, 10000)
            np.testing.assert_array_equal(matrix, original)

    def test_iteration_limit(self):
        with self.assertRaises(RuntimeError):
            qr.qr_eigenvalues([[1, 2, 3], [4, 5, 6], [7, 8, 10]], max_iterations=1)
        with self.assertRaises(ValueError):
            qr.qr_eigenvalues(np.eye(2), tolerance=0)


class ImportTests(unittest.TestCase):
    def test_no_output_on_import(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            for module in (lu, sweep, iterations, rotations, qr):
                importlib.reload(module)
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
