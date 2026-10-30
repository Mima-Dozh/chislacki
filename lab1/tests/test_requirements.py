"""Проверки требований, проблемных случаев и чтения файлов."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from lab1 import problem1 as lu
from lab1 import problem3 as iterations
from lab1 import problem4 as rotations

ROOT = Path(__file__).resolve().parents[1]


class PivotTests(unittest.TestCase):
    def test_required_swaps_and_multiple_rhs(self):
        for a in ([[0, 1], [1, 1]], [[1, 1, 0], [1, 1, 1], [0, 1, 1]],
                  [[1, 2, 3], [4, 8, 1], [7, 2, 4]]):
            a = np.array(a, dtype=float)
            factors = lu.get_LU(a)
            self.assertGreater(factors.swaps, 0)
            np.testing.assert_allclose(np.array(lu.get_P(factors)) @ a,
                                       np.array(lu.get_L(factors)) @ lu.get_U(factors), atol=1e-12)
            for rhs in (np.arange(1, len(a) + 1), np.eye(len(a))):
                np.testing.assert_allclose(lu.solve_lu(factors, rhs), np.linalg.solve(a, rhs), atol=1e-12)
            np.testing.assert_allclose(lu.get_determinant(factors), np.linalg.det(a), atol=1e-12)
            np.testing.assert_allclose(lu.get_inverse(factors), np.linalg.inv(a), atol=1e-12)

    def test_singular_and_bad_shapes(self):
        for a in ([[1, 2], [2, 4]], [], [[1, 2]], [[1, float('nan')], [0, 1]]):
            with self.assertRaises(ValueError):
                lu.get_LU(a)


class IterationTests(unittest.TestCase):
    def test_actual_error_bounded_by_reported_error(self):
        systems = [([[1, -.99], [-.99, 1]], [.01, .01]), ([[4, 1], [1, 3]], [1, 2])]
        data = json.loads((ROOT / 'data/problem3.json').read_text())
        systems.append((data['a'], data['b']))
        for a, b in systems:
            expected = np.linalg.solve(a, b)
            for method in (iterations.simple_iterations, iterations.seidel_method):
                x, count, history = method(a, b, tolerance=1e-6)
                error = np.linalg.norm(np.array(x) - expected, np.inf)
                self.assertLessEqual(error, history[-1] + 1e-12)
                self.assertLessEqual(history[-1], 1e-6)
                self.assertEqual(count, len(history))

    def test_seidel_iteration_matrix(self):
        alpha, _ = iterations.iteration_form([[4, 1, -1], [1, 5, 2], [2, 1, 6]], [1, 2, 3])
        alpha = np.array(alpha)
        expected = np.linalg.solve(np.eye(3) - np.tril(alpha, -1), np.triu(alpha))
        np.testing.assert_allclose(iterations.iteration_matrix(alpha, True), expected, atol=1e-14)

    def test_diagonal_and_limits(self):
        for method in (iterations.simple_iterations, iterations.seidel_method):
            x, count, history = method([[2, 0], [0, 4]], [6, 8])
            self.assertEqual(x, [3, 2])
            self.assertEqual(history[-1], 0)
            for a in ([[0, 1], [1, 2]], [[1, 2], [2, 1]]):
                with self.assertRaises(ValueError):
                    method(a, [1, 1])
            with self.assertRaises(RuntimeError):
                method([[1, -.99], [-.99, 1]], [.01, .01], max_iterations=1)


class RotationTests(unittest.TestCase):
    def test_eigenpairs_and_orthogonality(self):
        rng = np.random.default_rng(41)
        examples = [[[3]], np.zeros((3, 3)), np.eye(3), [[2, 1], [1, 2]]]
        for n in (3, 5, 8):
            a = rng.normal(size=(n, n))
            examples.append(a + a.T)
        for a in examples:
            original = np.array(a, copy=True)
            values, vectors, history = rotations.jacobi_eigenvalues(a, tolerance=1e-9)
            v = np.array(vectors)
            np.testing.assert_allclose(values, np.linalg.eigvalsh(a), atol=1e-9)
            self.assertLessEqual(np.linalg.norm(np.array(a) @ v - v * values), 1.01e-9)
            np.testing.assert_allclose(v.T @ v, np.eye(len(a)), atol=1e-12)
            self.assertLessEqual(history[-1][1], 1e-9)
            self.assertTrue(all(new[1] <= old[1] + 1e-14 for old, new in zip(history, history[1:])))
            np.testing.assert_array_equal(a, original)

    def test_invalid_matrix_and_limit(self):
        for a in ([[1, 2], [0, 1]], [[1j]], [], [[1, 2, 3]]):
            with self.assertRaises(ValueError):
                rotations.jacobi_eigenvalues(a)
        with self.assertRaises(RuntimeError):
            rotations.jacobi_eigenvalues([[1, 2, 3], [2, 4, 5], [3, 5, 6]], max_iterations=1)


class FileAndDependencyTests(unittest.TestCase):
    def test_default_inputs_and_required_output(self):
        expected = {1: ('P:', 'L:', 'U:', 'Обратная матрица:', 'Определитель:'),
                    2: ('Прогоночные коэффициенты:', 'Решение системы:'),
                    3: ('Заданная точность ε:', 'Эталонное решение', 'Итераций:', 'Сравнение скорости'),
                    4: ('Заданная точность ε:', 'v_1', 'Матрица собственных векторов', 'A V:', 'V Λ:', 'True'),
                    5: ('Заданная точность ε:', 'Q:', 'R:', 'Собственные значения:', 'Проверка собственных значений пройдена')}
        with tempfile.TemporaryDirectory() as tmp:
            for i, phrases in expected.items():
                p = subprocess.run([sys.executable, str(ROOT / f'problem{i}.py')], cwd=tmp,
                                   capture_output=True, text=True, check=True)
                for phrase in phrases:
                    self.assertIn(phrase, p.stdout)

    def test_custom_and_invalid_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.json'
            path.write_text(json.dumps({'a': [[0, 1], [1, 1]], 'b': [2, 3]}))
            p = subprocess.run([sys.executable, '-m', 'lab1.problem1', str(path)],
                               cwd=ROOT.parent, capture_output=True, text=True, check=True)
            self.assertIn('Решение: [1.0, 2.0]', p.stdout)
            for content in ('{', '{}'):
                path.write_text(content)
                p = subprocess.run([sys.executable, str(ROOT / 'problem1.py'), str(path)], capture_output=True)
                self.assertEqual(p.returncode, 2)

    def test_methods_work_without_site_packages(self):
        # -S отключает site-packages: все методы обязаны работать без NumPy.
        code = '''
from lab1 import problem1 as p1, problem2 as p2, problem3 as p3, problem4 as p4, problem5 as p5
assert p1.solve_lu(p1.get_LU([[0, 1], [1, 1]]), [2, 3]) == [1, 2]
p, q = p2.get_P_Q([[2]], [6]); assert p2.get_X(p, q) == [3]
for method in (p3.simple_iterations, p3.seidel_method):
    assert method([[2, 0], [0, 4]], [6, 8])[0] == [3, 2]
assert len(p4.jacobi_eigenvalues([[2, 1], [1, 2]])[0]) == 2
assert len(p5.qr_eigenvalues([[0, -1], [1, 0]])[0]) == 2
p5.qr_decomposition([[1, 2], [3, 4]])
'''
        subprocess.run([sys.executable, '-S', '-c', code], cwd=ROOT.parent, check=True, capture_output=True)


if __name__ == '__main__':
    unittest.main()
