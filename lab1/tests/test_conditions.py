import unittest
from unittest.mock import patch

import numpy as np

from lab1 import conditions as c
from lab1 import problem1 as lu, problem2 as sweep, problem3 as it, problem4 as jacobi, problem5 as qr


class ConditionTests(unittest.TestCase):
    def test_dominance_is_not_necessary_for_lu_or_sweep(self):
        a = [[1, 2], [2, 1]]
        self.assertFalse(c.diagonal_dominance(a))
        np.testing.assert_allclose(lu.solve_lu(lu.get_LU(a), [3, 3]), [1, 1])
        p, q = sweep.get_P_Q(a, [3, 3])
        np.testing.assert_allclose(sweep.get_X(p, q), [1, 1])
        with self.assertRaises(ValueError):
            sweep.get_P_Q([[0, 1], [1, 1]], [1, 2])
        np.testing.assert_allclose(lu.solve_lu(lu.get_LU([[0, 1], [1, 1]]), [1, 2]), [1, 1])

    def test_spd_without_dominance(self):
        a = [[1, 2], [2, 5]]
        self.assertFalse(c.diagonal_dominance(a))
        self.assertTrue(c.positive_definite(a))
        for invalid in ([[1, 2], [2, 1]], [[1, 1], [0, 1]], [[0]]):
            self.assertFalse(c.positive_definite(invalid))

    def test_convergence_when_norm_exceeds_one(self):
        for a in ([[1, -2], [-.1, 1]], [[1, 2], [2, 5]]):
            b = [1, 3]
            for seidel, method in ((False, it.simple_iterations), (True, it.seidel_method)):
                report = it.check_conditions(a, b, seidel)
                self.assertGreaterEqual(report['norm'], 1)
                self.assertLess(report['rho'], 1)
                self.assertIsNotNone(report['certificate'])
                x, _, history = method(a, b, tolerance=1e-8)
                error = np.linalg.norm(np.array(x) - np.linalg.solve(a, b), np.inf)
                self.assertLessEqual(history[-1], 1e-8)
                self.assertLessEqual(error, history[-1] + 1e-12)

    def test_spectral_estimate_and_boundary(self):
        for t, expected in (([[0, .2], [.2, 0]], .2), ([[0, 2], [2, 0]], 2), ([[0, 1], [1, 0]], 1)):
            rho, status = c.spectral_diagnostic(t)
            self.assertAlmostEqual(rho, expected)
            if expected == 1:
                self.assertIn('пограничный', status)
        self.assertIsNone(c.contraction_certificate([[0, 1], [1, 0]]))
        with patch('lab1.problem5.qr_eigenvalues', side_effect=RuntimeError('limit')):
            rho, status = c.spectral_diagnostic([[1]])
            self.assertIsNone(rho)
            self.assertIn('не удалось', status)

    def test_rotation_does_not_require_positive_definiteness(self):
        a = [[0, 1], [1, 0]]
        self.assertTrue(c.real_symmetric(a))
        self.assertFalse(c.positive_definite(a))
        np.testing.assert_allclose(jacobi.jacobi_eigenvalues(a)[0], [-1, 1])

    def test_qr_small_blocks_and_singular_matrix(self):
        for a in ([[0]], [[0, -1], [1, 0]], [[1, 2, 3], [0, 0, -1], [0, 1, 0]]):
            self.assertTrue(c.block_triangular_small(a))
            self.assertEqual(qr.qr_eigenvalues(a)[1], 0)
        a = [[1, 7, -1], [-2, 2, -2], [9, -7, 3]]
        self.assertFalse(c.block_triangular_small(a))
        values, _ = qr.qr_eigenvalues(a)
        qr.check_answers(values, a)


if __name__ == '__main__':
    unittest.main()
