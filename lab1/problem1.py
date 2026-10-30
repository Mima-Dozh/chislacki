"""LU-разложение с частичным выбором опорного элемента: P A = L U."""

from dataclasses import dataclass
import numbers

if __package__:
    from .input_data import read_input
    from .conditions import diagonal_dominance, show
    from .matrix_ops import matrix, identity, number, matmul, print_matrix
else:
    from input_data import read_input
    from conditions import diagonal_dominance, show
    from matrix_ops import matrix, identity, number, matmul, print_matrix


@dataclass
class LUFactors:
    lu: list
    permutation: list
    swaps: int


def get_LU(a):
    lu = matrix(a)
    n = len(lu)
    permutation = list(range(n))
    swaps = 0
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(lu[i][k]))
        if lu[pivot][k] == 0:
            raise ValueError('Матрица вырождена: нулевой опорный элемент')
        if pivot != k:
            # Переставляются также уже вычисленные элементы L слева от k.
            lu[k], lu[pivot] = lu[pivot], lu[k]
            permutation[k], permutation[pivot] = permutation[pivot], permutation[k]
            swaps += 1
        for i in range(k + 1, n):
            lu[i][k] /= lu[k][k]
            for j in range(k + 1, n):
                lu[i][j] -= lu[i][k] * lu[k][j]
    return LUFactors(lu, permutation, swaps)


def get_L(factors):
    lu = factors.lu
    return [[lu[i][j] if i > j else float(i == j) for j in range(len(lu))]
            for i in range(len(lu))]


def get_U(factors):
    lu = factors.lu
    return [[lu[i][j] if i <= j else 0.0 for j in range(len(lu))]
            for i in range(len(lu))]


def get_P(factors):
    n = len(factors.lu)
    return [[float(factors.permutation[i] == j) for j in range(n)] for i in range(n)]


def _substitute(a, b, forward):
    a = matrix(a)
    rhs = list(b)
    if len(rhs) != len(a):
        raise ValueError('Размер правой части не совпадает с размером матрицы')
    is_vector = isinstance(rhs[0], numbers.Number)
    rhs = [[number(value)] for value in rhs] if is_vector else matrix(rhs, square=False)
    x = [[0.0] * len(rhs[0]) for _ in a]
    order = range(len(a)) if forward else range(len(a) - 1, -1, -1)
    for i in order:
        if a[i][i] == 0:
            raise ValueError('Нулевой диагональный элемент при подстановке')
        indices = range(i) if forward else range(i + 1, len(a))
        for j in range(len(rhs[0])):
            x[i][j] = (rhs[i][j] - sum(a[i][k] * x[k][j] for k in indices)) / a[i][i]
    return [row[0] for row in x] if is_vector else x


def get_y(l, b):
    return _substitute(l, b, forward=True)


def get_x(u, y):
    return _substitute(u, y, forward=False)


def solve_lu(factors, b):
    if len(b) != len(factors.lu):
        raise ValueError('Размер правой части не совпадает с размером матрицы')
    pb = [b[i] for i in factors.permutation]
    return get_x(get_U(factors), get_y(get_L(factors), pb))


def get_determinant(factors):
    result = (-1) ** factors.swaps
    for i, row in enumerate(factors.lu):
        result *= row[i]
    return result


def get_inverse(factors):
    return solve_lu(factors, identity(len(factors.lu)))


def main():
    data = read_input(1)
    a, b = matrix(data['a']), data['b']
    print('Условия LU с выбором опорного элемента:')
    print('  Необходимо и достаточно для единственного решения: det(A) != 0.')
    try:
        factors = get_LU(a)
    except ValueError:
        print('  Проверка опорных элементов: обнаружен нулевой — решение через LU невозможно.')
        raise
    print('  Все опорные элементы ненулевые: условие выполнено в машинной арифметике.')
    l, u, p = get_L(factors), get_U(factors), get_P(factors)
    x = solve_lu(factors, b)
    inverse = get_inverse(factors)
    print_matrix('A:', a)
    print('b:', b)
    print_matrix('P:', p)
    print_matrix('L:', l)
    print_matrix('U:', u)
    print_matrix('P A:', matmul(p, a))
    print_matrix('L U:', matmul(l, u))
    print('Решение:', x)
    print('Определитель:', get_determinant(factors))
    print_matrix('Обратная матрица:', inverse)

    import numpy as np
    np.testing.assert_allclose(np.array(p) @ a, np.array(l) @ u, atol=1e-10)
    np.testing.assert_allclose(np.array(a) @ x, b, atol=1e-10)
    np.testing.assert_allclose(np.array(a) @ inverse, np.eye(len(a)), atol=1e-10)
    print('Эталон NumPy:', np.linalg.solve(a, b))
    print('Проверки P A = L U, A x = b и A A^-1 = I пройдены')


if __name__ == '__main__':
    main()
