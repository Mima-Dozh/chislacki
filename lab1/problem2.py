"""Метод прогонки: собственные вычисления на списках Python."""

if __package__:
    from .input_data import read_input
    from .conditions import show
    from .matrix_ops import number, vector
else:
    from input_data import read_input
    from conditions import show
    from matrix_ops import number, vector


def _rows(x):
    rows = [[number(v) for v in row] for row in x]
    n = len(rows)
    if not n:
        raise ValueError('Система должна быть непустой')
    expected = [1] if n == 1 else [2] + [3] * (n - 2) + [2]
    if [len(row) for row in rows] != expected:
        raise ValueError('Нужны строки [b,c], [a,b,c], ..., [a,b]; для n=1 — [b]')
    return rows


def get_P_Q(x, y):
    rows = _rows(x)
    n = len(rows)
    rhs = vector(y, n)
    p, q = [0.0] * n, [0.0] * n
    for i, row in enumerate(rows):
        a = row[0] if i else 0
        b = row[1] if i else row[0]
        c = row[-1] if i + 1 < n else 0
        denominator = b + (a * p[i - 1] if i else 0)
        if denominator == 0:
            raise ValueError(f'Нулевой знаменатель прогонки в строке {i + 1}')
        p[i] = -c / denominator
        q[i] = (rhs[i] - (a * q[i - 1] if i else 0)) / denominator
    return p, q


def get_det_P(x, p):
    rows = _rows(x)
    p = vector(p, len(rows))
    determinant = rows[0][0]
    for i in range(1, len(rows)):
        determinant *= rows[i][1] + rows[i][0] * p[i - 1]
    return determinant


def get_X(p, q):
    p = vector(p, len(q))
    x = vector(q, len(p))
    for i in range(len(x) - 2, -1, -1):
        x[i] += p[i] * x[i + 1]
    return x


def check_answers(x, y, answers):
    import numpy as np
    rows = _rows(x)
    result = []
    for i, row in enumerate(rows):
        value = (row[1] if i else row[0]) * answers[i]
        if i:
            value += row[0] * answers[i - 1]
        if i + 1 < len(rows):
            value += row[-1] * answers[i + 1]
        result.append(value)
    np.testing.assert_allclose(result, y, atol=1e-6, rtol=1e-6)


def main():
    data = read_input(2)
    x, y = data['x'], data['y']
    rows = _rows(x)
    print('Условия прогонки:')
    dominant = all(abs(row[1] if i else row[0]) >
                   (abs(row[0]) if i else 0) + (abs(row[-1]) if i + 1 < len(rows) else 0)
                   for i, row in enumerate(rows))
    show('Достаточное условие: строгое диагональное преобладание', dominant)
    print('  Необходимо и достаточно для этой прогонки: все знаменатели ненулевые.')
    print('  Это эквивалентно ненулевым ведущим главным минорам в точной арифметике.')
    p, q = get_P_Q(x, y)
    print('  Проверка знаменателей: выполнено; отсутствие преобладания не запрещает прогонку.')
    answer = get_X(p, q)
    print('Прогоночные коэффициенты:')
    print(f"{'i':>3} {'P':>16} {'Q':>16}")
    for i, (pi, qi) in enumerate(zip(p, q), start=1):
        print(f'{i:3d} {pi:16.10f} {qi:16.10f}')
    print('Решение системы:', answer)
    print('Определитель:', get_det_P(x, p))
    check_answers(x, y, answer)
    print('Проверка A x = b пройдена')


if __name__ == '__main__':
    main()
