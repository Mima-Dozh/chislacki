import unittest
import numpy as np
from lab2 import problem1, problem2


def bisect(func, left, right):
    for _ in range(60):
        middle = (left + right) / 2
        if func(left) * func(middle) <= 0:
            right = middle
        else:
            left = middle
    return (left + right) / 2


class MethodsTests(unittest.TestCase):
    def test_scalar_accuracy_and_bounds(self):
        reference = bisect(problem1.func, 1, 1.5)
        for method in (problem1.simple_iteration, problem1.newton_method):
            for eps in (1e-4, 1e-6, 1e-9):
                root, history = method(eps=eps)
                self.assertLessEqual(abs(root - reference), eps)
                for _, x, bound, _ in history:
                    self.assertLessEqual(abs(x - reference), bound + 1e-14)

    def test_negative_root_accuracy_and_bounds(self):
        reference = bisect(problem1.func, -0.8, -0.5)
        for method in (problem1.simple_iteration, problem1.newton_method):
            for eps in (1e-4, 1e-6, 1e-9, 1e-16):
                root, history = method(eps=eps, negative=True)
                self.assertLessEqual(abs(root - reference), eps + 2e-16)
                self.assertLess(abs(problem1.func(root)), 3 * eps + 2e-16)
                for _, x, bound, _ in history:
                    self.assertTrue(-0.8 <= x <= -0.5)
                    self.assertLessEqual(abs(x - reference), bound + 1e-14)

    def test_system_accuracy_and_bounds(self):
        def reduced(x):
            y = (np.exp(x) + x) / 4
            return x*x/16 + y*y/4 - 1
        x = bisect(reduced, 1.6, 1.8)
        reference = np.array([x, (np.exp(x) + x) / 4])
        for method in (problem2.simple_iteration, problem2.newton_method):
            for eps in (1e-4, 1e-6, 1e-9):
                root, history = method(eps=eps)
                self.assertLessEqual(np.linalg.norm(root - reference, ord=np.inf), eps)
                for _, point, bound, _ in history:
                    self.assertTrue(1.6 <= point[0] <= 1.8 and 1.75 <= point[1] <= 1.85)
                    self.assertLessEqual(np.linalg.norm(point - reference, ord=np.inf), bound + 1e-14)

    def test_negative_system_accuracy_and_bounds(self):
        def reduced(x):
            y = (np.exp(x) + x) / 4
            return x*x/16 + y*y/4 - 1
        x = bisect(reduced, -3.7, -3.5)
        reference = np.array([x, (np.exp(x) + x) / 4])
        for method in (problem2.simple_iteration, problem2.newton_method):
            for eps in (1e-4, 1e-6, 1e-9, 1e-12):
                root, history = method(eps=eps, region=problem2.SECOND_REGION)
                self.assertLessEqual(np.linalg.norm(root - reference, ord=np.inf), eps)
                for _, point, bound, _ in history:
                    self.assertTrue(-3.7 <= point[0] <= -3.5 and -0.95 <= point[1] <= -0.85)
                    self.assertLessEqual(np.linalg.norm(point - reference, ord=np.inf), bound + 1e-14)

    def test_jacobian(self):
        point = np.array(problem2.FIRST_REGION.x0)
        h = 1e-5
        numerical = np.column_stack([(problem2.func(point + d*h) - problem2.func(point - d*h))/(2*h)
                                     for d in np.eye(2)])
        np.testing.assert_allclose(problem2.jacobian(point), numerical, atol=1e-9)

    def test_iteration_limit(self):
        for module in (problem1, problem2):
            for method in (module.simple_iteration, module.newton_method):
                with self.assertRaises(RuntimeError):
                    method(eps=1e-12, max_iterations=1)


if __name__ == '__main__':
    unittest.main()
