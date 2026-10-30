"""Вращения Якоби для вещественной симметрической матрицы."""

import math

if __package__:
    from .input_data import read_input
    from .conditions import real_symmetric, show
    from .matrix_ops import matrix, identity, transpose, matmul, subtract, norm_frobenius, validate_iteration, print_matrix
else:
    from input_data import read_input
    from conditions import real_symmetric, show
    from matrix_ops import matrix, identity, transpose, matmul, subtract, norm_frobenius, validate_iteration, print_matrix

epsilon = 1e-6
MAX_ITERATIONS = 10000


def jacobi_eigenvalues(a, tolerance=epsilon, max_iterations=MAX_ITERATIONS):
    """Возвращает значения, матрицу векторов-столбцов и историю (шаг, оценка)."""
    validate_iteration(tolerance, max_iterations)
    current = matrix(a, real=True)
    n = len(current)
    if any(current[i][j] != current[j][i] for i in range(n) for j in range(i)):
        raise ValueError('Для метода вращений матрица должна быть симметрической')
    vectors = identity(n)
    history = []
    for iteration in range(max_iterations):
        error = math.hypot(*(current[i][j] for i in range(n) for j in range(n) if i != j))
        history.append((iteration, error))
        if error <= tolerance:
            order = sorted(range(n), key=lambda i: current[i][i])
            return [current[i][i] for i in order], [[row[i] for i in order] for row in vectors], history
        p, q = max(((i, j) for i in range(n) for j in range(i + 1, n)),
                   key=lambda ij: abs(current[ij[0]][ij[1]]))
        angle = 0.5 * math.atan2(2 * current[p][q], current[q][q] - current[p][p])
        c, s = math.cos(angle), math.sin(angle)
        rotation = identity(n)
        rotation[p][p] = rotation[q][q] = c
        rotation[p][q], rotation[q][p] = s, -s
        current = matmul(matmul(transpose(rotation), current), rotation)

        vectors = matmul(vectors, rotation)

    raise RuntimeError(f'Метод вращений не сошёлся за {max_iterations} итераций')

def main():
    data = read_input(4)
    a = matrix(data['a'], real=True)
    tolerance, limit = data['epsilon'], data['max_iterations']
    print('Условия метода вращений Якоби:')
    show('Вещественная симметрическая матрица', real_symmetric(a))
    print('  Симметричность необходима и достаточна для вещественной ортогональной диагонализации.')
    print('  Она гарантирует сходимость вращений с выбором максимального элемента в точной арифметике.')
    print('  Диагональное преобладание, невырожденность и положительная определённость не требуются.')
    values, vectors, history = jacobi_eigenvalues(a, tolerance, limit)
    print_matrix('Исходная матрица A:', a)
    print('Заданная точность ε:', tolerance)
    print('Число вращений:', len(history) - 1)
    print('Собственные значения:', values)
    for i, value in enumerate(values):
        print(f'λ_{i + 1} = {value:.10g}; v_{i + 1} = {[row[i] for row in vectors]}')
    print_matrix('Матрица собственных векторов V (столбцы):', vectors)
    diagonal = [[values[i] if i == j else 0.0 for j in range(len(a))] for i in range(len(a))]
    av, vl = matmul(a, vectors), matmul(vectors, diagonal)
    print_matrix('Λ:', diagonal)
    print_matrix('A V:', av)
    print_matrix('V Λ:', vl)
    residual = norm_frobenius(subtract(av, vl))
    print('Норма невязки ||A V - V Λ||_F:', residual)
    print('Проверка A V = V Λ с допуском ε:', residual <= tolerance)
    print('||V^T V - I||_F:', norm_frobenius(subtract(matmul(transpose(vectors), vectors), identity(len(a)))))
    print('Итерация   Норма внедиагональной части')
    for iteration, error in history:
        print(f'{iteration:8d}   {error:.10e}')

    import numpy as np
    reference = np.linalg.eigvalsh(a)
    print('Эталонные собственные значения NumPy:', reference)
    print('Максимальная ошибка относительно NumPy:', np.max(np.abs(np.array(values) - reference)))


if __name__ == '__main__':
    main()
