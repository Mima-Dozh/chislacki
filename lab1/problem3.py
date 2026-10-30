"""Методы Якоби и Зейделя с апостериорной оценкой ошибки."""

if __package__:
    from .input_data import read_input
    from .conditions import diagonal_dominance, positive_definite, contraction_certificate, spectral_diagnostic, show
    from .matrix_ops import matrix, vector, matvec, norm_inf, number, validate_iteration, print_matrix
else:
    from input_data import read_input
    from conditions import diagonal_dominance, positive_definite, contraction_certificate, spectral_diagnostic, show
    from matrix_ops import matrix, vector, matvec, norm_inf, number, validate_iteration, print_matrix

epsilon = 1e-6
MAX_ITERATIONS = 10000


def iteration_form(a, b):
    a = matrix(a)
    rhs = vector(b, len(a))
    if any(a[i][i] == 0 for i in range(len(a))):
        raise ValueError('Для итерационной формы нужны ненулевые диагональные элементы')
    alpha = [[-a[i][j] / a[i][i] if i != j else 0.0 for j in range(len(a))]
             for i in range(len(a))]
    return alpha, [rhs[i] / a[i][i] for i in range(len(a))]


def iteration_matrix(alpha, seidel=False):
    if not seidel:
        return [row[:] for row in alpha]
    # T = (I-B)^-1 C: прямой ход по строго нижней части B.
    n = len(alpha)
    t = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            t[i][j] = (alpha[i][j] if j > i else 0.0) + sum(
                alpha[i][k] * t[k][j] for k in range(i))
    return t


def check_conditions(a, b, seidel=False):
    a = matrix(a)
    alpha, _ = iteration_form(a, b)
    t = iteration_matrix(alpha, seidel)
    rho, status = spectral_diagnostic(t)
    return {'norm': norm_inf(t), 'rho': rho, 'spectral_status': status,
            'dominant': diagonal_dominance(a),
            'spd': positive_definite(a), 'certificate': contraction_certificate(t)}


def _iterate(a, b, tolerance, max_iterations, seidel):
    """Возвращает решение, число шагов и историю верхних оценок ошибки.

    При ||T||<1 используется q/(1-q)*||dx||.
    Иначе ищется ||T^m||<1 и оценивается ошибка через невязку итерации.
    Неудача поиска сертификата не является доказательством расходимости.
    """
    validate_iteration(tolerance, max_iterations)
    alpha, beta = iteration_form(a, b)
    t = iteration_matrix(alpha, seidel)
    q = norm_inf(t)
    certificate = contraction_certificate(t)
    if certificate is None:
        raise ValueError('Не удалось подтвердить ||T^m|| < 1 для m <= 64; '
                         'надёжный критерий ошибки недоступен. Это не доказательство расходимости.')
    _, _, inverse_bound = certificate
    x = beta[:]
    history = []
    for iteration in range(1, max_iterations + 1):
        previous = x[:]
        if seidel:
            for i in range(len(x)):
                x[i] = beta[i] + sum(alpha[i][j] * x[j] for j in range(i)) + sum(
                    alpha[i][j] * previous[j] for j in range(i + 1, len(x)))
        else:
            x = [beta[i] + value for i, value in enumerate(matvec(alpha, previous))]
        x = [number(value) for value in x]
        difference = [new - old for new, old in zip(x, previous)]
        if q < 1:
            error = q / (1 - q) * max(abs(value) for value in difference)
        else:
            # Невязка неподвижной точки в x_new равна T*(x_new-x_old).
            residual = matvec(t, difference)
            error = inverse_bound * max(abs(value) for value in residual)
        history.append(error)
        if error <= tolerance:
            return x, iteration, history
    raise RuntimeError(f'Точность не достигнута за {max_iterations} итераций')


def simple_iterations(a, b, tolerance=epsilon, max_iterations=MAX_ITERATIONS):
    return _iterate(a, b, tolerance, max_iterations, seidel=False)


def seidel_method(a, b, tolerance=epsilon, max_iterations=MAX_ITERATIONS):
    return _iterate(a, b, tolerance, max_iterations, seidel=True)


def main():
    data = read_input(3)
    a = matrix(data['a'])
    b = vector(data['b'], len(a))
    tolerance, limit = data['epsilon'], data['max_iterations']
    validate_iteration(tolerance, limit)
    print_matrix('A:', a)
    print('b:', b)
    print('Заданная точность ε:', tolerance)
    # Только эталон для сравнения, не часть итерационных методов.
    import numpy as np
    for name, seidel in [('Якоби', False), ('Зейдель', True)]:
        print(f'Условия сходимости: {name}')
        check = check_conditions(a, b, seidel)
        print('  Необходимо и достаточно для сходимости при любом x0: rho(T) < 1.')
        print(f"  Численная оценка rho(T): {check['rho']}; {check['spectral_status']}.")
        print('  Оценка спектра собственным QR приблизительная, не строгий сертификат.')
        show(f"Достаточное условие ||T||_inf < 1, норма = {check['norm']:.8g}", check['norm'] < 1)
        show('Достаточное условие: строгое диагональное преобладание A', check['dominant'])
        if seidel:
            show('Достаточное условие: A вещественная симметрическая положительно определённая', check['spd'])
        certificate = check['certificate']
        if certificate:
            print(f'  Найдено ||T^{certificate[0]}||_inf = {certificate[1]:.8g} < 1.')
        else:
            print('  Достаточный сертификат степеней не найден; расходимость этим не доказана.')
    reference = np.linalg.solve(a, b)
    print('Эталонное решение NumPy:', reference)
    for name, method in [('Простые итерации', simple_iterations), ('Зейдель', seidel_method)]:
        x, count, history = method(a, b, tolerance, limit)
        print(f'\n{name}:\nx = {x}\nИтераций: {count}')
        print('Апостериорная оценка абсолютной ошибки:', history[-1])
        print('Ошибка относительно эталона:', np.linalg.norm(np.array(x) - reference, ord=np.inf))
        print('Норма невязки ||Ax-b||_inf:', np.linalg.norm(np.array(a) @ x - b, ord=np.inf))
    print('\nСравнение скорости сходимости (начальное приближение beta):')
    print('Точность     Якоби   Зейдель')
    tolerances = data.get('comparison_epsilons', [tolerance])
    for value in tolerances:
        validate_iteration(value, limit)
    for value in sorted(set(tolerances + [tolerance]), reverse=True):
        counts = []
        for method in (simple_iterations, seidel_method):
            try:
                counts.append(str(method(a, b, value, limit)[1]))
            except RuntimeError:
                counts.append('предел')
        print(f'{value:9.1e} {counts[0]:>8} {counts[1]:>9}')
    print('Скорость сравнивается по числу итераций при одинаковой требуемой точности.')


if __name__ == '__main__':
    main()
