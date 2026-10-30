"""Отражения Хаусхолдера и QR-алгоритм"""

import cmath

if __package__:
    from .input_data import read_input
    from .conditions import block_triangular_small, show
    from .matrix_ops import matrix, number, identity, matmul, norm_vector, norm_frobenius, validate_iteration, print_matrix
else:
    from input_data import read_input
    from conditions import block_triangular_small, show
    from matrix_ops import matrix, number, identity, matmul, norm_vector, norm_frobenius, validate_iteration, print_matrix

epsilon = 1e-6
MAX_ITERATIONS = 10000


def get_v(a, k=0):
    a = matrix(a)
    if not 0 <= k < len(a):
        raise ValueError('Неверный номер столбца')
    v = [0.0] * len(a)
    column = [a[i][k] for i in range(k, len(a))]
    length = norm_vector(column)
    if length:
        phase = column[0] / abs(column[0]) if column[0] != 0 else 1
        v[k:] = column
        v[k] += phase * length
    return [[value] for value in v]


def get_H(v):
    values = list(v)
    if not values:
        raise ValueError('Вектор отражения должен быть непустым')
    # Поддерживаем вектор и столбец, возвращаемый get_v.
    if hasattr(values[0], '__iter__'):
        values = [row[0] for row in values]
    values = [number(value) for value in values]
    scale = max(abs(value) for value in values)
    h = identity(len(values))
    if not scale:
        return h
    values = [value / scale for value in values]
    denominator = sum(abs(value) ** 2 for value in values)
    for i in range(len(values)):
        for j in range(len(values)):
            h[i][j] -= 2 * values[i] * values[j].conjugate() / denominator
    return h


def qr_decomposition(a):
    r = matrix(a)
    q = identity(len(r))
    for k in range(len(r) - 1):
        h = get_H(get_v(r, k))
        r = matmul(h, r)
        q = matmul(q, h)
    return q, r


def _pair_eigenvalues(block):
    half_trace = (block[0][0] + block[1][1]) / 2
    root = cmath.sqrt(((block[0][0] - block[1][1]) / 2) ** 2 + block[0][1] * block[1][0])
    return [half_trace + root, half_trace - root]


def qr_eigenvalues(a, tolerance=epsilon, max_iterations=MAX_ITERATIONS):
    """Возвращает собственные значения и число QR-шагов.

    tolerance — относительный порог отделения нижних блоков.
    Он не гарантирует такую же ошибку корней плохо обусловленного спектра.
    """
    validate_iteration(tolerance, max_iterations)
    current = matrix(a)
    n = len(current)
    scale = max(abs(value) for row in current for value in row)
    if scale == 0:
        return [0.0] * n, 0
    current = [[complex(value / scale) for value in row] for row in current]
    values = [0j] * n
    active, iterations = n, 0
    while active:
        block = [row[:active] for row in current[:active]]
        threshold = tolerance * norm_frobenius(block)
        if active == 1 or norm_vector(block[-1][:-1]) <= threshold:
            values[active - 1] = block[-1][-1]
            active -= 1
            continue
        if active == 2 or norm_frobenius([row[:-2] for row in block[-2:]]) <= threshold:
            values[active - 2:active] = _pair_eigenvalues([row[-2:] for row in block[-2:]])
            active -= 2
            continue
        if iterations >= max_iterations:
            raise RuntimeError(f'QR-алгоритм не сошёлся за {max_iterations} итераций')
        q, r = qr_decomposition(block)
        print_matrix(f'q {iterations + 1}:', q)
        print_matrix(f'r {iterations + 1}:', r)
        next_block = matmul(r, q)
        print_matrix(f'next_block {iterations + 1}:', next_block)
        print_matrix(f'current {iterations + 1}:', block)
        print_matrix(f'matmul(r, q):', matmul(q, r))
        for i in range(active):
            for j in range(active):
                current[i][j] = next_block[i][j]
        iterations += 1
    return [number(value * scale) for value in values], iterations


def check_answers(values, a, tolerance=epsilon):
    """Только проверка готового результата через независимый эталон NumPy."""
    import numpy as np
    expected = list(np.linalg.eigvals(a))
    if len(values) != len(expected):
        raise RuntimeError('Неверное количество собственных значений')
    scale = max(np.linalg.norm(a), np.finfo(float).tiny)
    for value in values:
        value = number(value)
        index = min(range(len(expected)), key=lambda i: abs(expected[i] - value))
        if abs(expected.pop(index) - value) > 10 * tolerance * scale:
            raise RuntimeError('Собственные значения не совпадают с проверкой NumPy')


def main():
    data = read_input(5)
    a = matrix(data['a'])
    tolerance, limit = data['epsilon'], data['max_iterations']
    print('Условия QR-разложения и QR-итераций:')
    print('  Применимость: квадратная матрица с конечными элементами — выполнено')
    print('  QR-разложение существует и для вырожденной матрицы; det(A) != 0 не требуется.')
    show('Достаточный частный случай завершения без QR-шагов: отделённые блоки 1x1/2x2', block_triangular_small(a))
    print('  Используется QR-алгоритм без сдвигов: A = QR, затем A = RQ.')
    print('  Общий необходимый и достаточный критерий сходимости здесь не проверяется.')
    print('  Норма A < 1 и различные модули собственных значений не используются как запрет запуска.')
    print('  Фактическое завершение проверяется отделением блоков с допуском ε и лимитом шагов.')
    q, r = qr_decomposition(a)
    values, iterations = qr_eigenvalues(a, tolerance, limit)
    print_matrix('Исходная матрица A:', a)
    print('Заданная точность ε:', tolerance)
    print_matrix('Q:', q)
    print_matrix('R:', r)
    print('Число QR-итераций:', iterations)
    print('Собственные значения:', values)
    import numpy as np
    print('Норма невязки QR - A:', np.linalg.norm(np.array(q) @ r - a))
    print('Норма невязки Q* Q - I:', np.linalg.norm(np.array(q).conj().T @ q - np.eye(len(a))))
    print('Эталонные значения NumPy:', np.linalg.eigvals(a))
    check_answers(values, a, tolerance)
    print('Проверка собственных значений пройдена')


if __name__ == '__main__':
    main()
