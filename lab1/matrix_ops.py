"""Элементарные операции над списками; готовые численные библиотеки не используются."""

import math
import numbers


def number(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Number):
        raise ValueError('Элементы должны быть числами')
    result = complex(value)
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise ValueError('Элементы должны быть конечными')
    return result.real if result.imag == 0 else result


def matrix(data, square=True, real=False):
    rows = [list(row) for row in data]
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError('Матрица должна быть непустой и прямоугольной')
    if square and len(rows) != len(rows[0]):
        raise ValueError('Матрица должна быть квадратной')
    result = [[number(value) for value in row] for row in rows]
    if real and any(isinstance(value, complex) for row in result for value in row):
        raise ValueError('Матрица должна быть вещественной')
    return result


def vector(data, size):
    values = list(data)
    if len(values) != size:
        raise ValueError('Размер правой части не совпадает с размером матрицы')
    return [number(value) for value in values]


def identity(n):
    return [[float(i == j) for j in range(n)] for i in range(n)]


def transpose(a):
    return [list(column) for column in zip(*a)]


def matmul(a, b):
    if len(a[0]) != len(b):
        raise ValueError('Несогласованные размеры матриц')
    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]


def matvec(a, x):
    return [sum(value * component for value, component in zip(row, x)) for row in a]


def subtract(a, b):
    return [[x - y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]


def norm_frobenius(a):
    return math.hypot(*(abs(value) for row in a for value in row))


def norm_inf(a):
    return max(sum(abs(value) for value in row) for row in a)


def norm_vector(x):
    return math.hypot(*(abs(value) for value in x))


def validate_iteration(tolerance, max_iterations):
    if isinstance(tolerance, bool) or not isinstance(tolerance, numbers.Real) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Точность должна быть положительной и конечной')
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, numbers.Integral) or max_iterations < 1:
        raise ValueError('Предел итераций должен быть положительным целым числом')


def print_matrix(name, a, precision=6):
    """Вывод с выравниванием столбцов; форматирование не меняет сами числа."""
    def format_real(value):
        if value == 0:
            value = 0.0  # Не показываем отрицательный ноль.
        if value != 0 and (abs(value) < 10 ** -precision or abs(value) >= 1e9):
            return f'{value:.{precision}e}'
        return f'{value:.{precision}f}'

    def format_value(value):
        if isinstance(value, complex):
            if value.imag == 0:
                return format_real(value.real)
            sign = '+' if value.imag >= 0 else '-'
            return f'{format_real(value.real)} {sign} {format_real(abs(value.imag))}j'
        return format_real(value)

    rows = [[format_value(value) for value in row] for row in a]
    print(name)
    if rows:
        widths = [max(len(value) for value in column) for column in zip(*rows)]
        for row in rows:
            print('  [ ' + '  '.join(value.rjust(width) for value, width in zip(row, widths)) + ' ]')
    print()
