"""Проверки условий: теоретический критерий отдельно от численной диагностики."""

if __package__:
    from .matrix_ops import matrix, identity, matmul, norm_inf
else:
    from matrix_ops import matrix, identity, matmul, norm_inf


def diagonal_dominance(a):
    a = matrix(a)
    return all(abs(row[i]) > sum(abs(v) for j, v in enumerate(row) if j != i)
               for i, row in enumerate(a))


def real_symmetric(a):
    a = matrix(a)
    return all(not isinstance(v, complex) for row in a for v in row) and all(
        a[i][j] == a[j][i] for i in range(len(a)) for j in range(i))


def positive_definite(a):
    """Достаточный признак для Зейделя: вещественная SPD; проверка LDL^T."""
    a = matrix(a)
    if not real_symmetric(a):
        return False
    n = len(a)
    l, d = identity(n), [0.0] * n
    for i in range(n):
        d[i] = a[i][i] - sum(l[i][k] ** 2 * d[k] for k in range(i))
        if d[i] <= 0:
            return False
        for j in range(i + 1, n):
            l[j][i] = (a[j][i] - sum(l[j][k] * l[i][k] * d[k] for k in range(i))) / d[i]
    return True


def contraction_certificate(t, max_power=64):
    """Если ||T^m||<1, то ||(I-T)^-1|| <= ||I+...+T^(m-1)||/(1-||T^m||).

    Отсутствие сертификата за max_power шагов не доказывает расходимость.
    При q=||T||<1 сохраняем обычную оценку q/(1-q)*||dx||.
    """
    t = matrix(t)
    power, series = identity(len(t)), identity(len(t))
    for m in range(1, max_power + 1):
        power = matmul(power, t)
        q = norm_inf(power)
        if q < 1:
            return m, q, norm_inf(series) / (1 - q)
        if m != max_power:
            series = [[x + y for x, y in zip(row, add)] for row, add in zip(series, power)]
        if q == float('inf') or q != q:
            break
    return None


def spectral_diagnostic(t):
    """Приближённая проверка rho(T)<1 собственным QR, не доказательство у границы."""
    if __package__:
        from .problem5 import qr_eigenvalues
    else:
        from problem5 import qr_eigenvalues
    try:
        values, _ = qr_eigenvalues(t, tolerance=1e-12, max_iterations=10000)
        rho = max(abs(value) for value in values)
        # Зазор — диагностический, а не строгая граница ошибки QR.
        status = 'пограничный результат' if abs(rho - 1) <= 1e-7 else ('выполнено по оценке' if rho < 1 else 'не выполнено по оценке')
        return rho, status
    except (ValueError, RuntimeError, OverflowError, ZeroDivisionError):
        return None, 'не удалось оценить собственным QR'


def block_triangular_small(a):
    """Частный достаточный случай QR: уже отделённые блоки размера 1 или 2."""
    a = matrix(a)
    active = len(a)
    while active:
        if active == 1 or all(a[active - 1][j] == 0 for j in range(active - 1)):
            active -= 1
        elif active == 2 or all(a[i][j] == 0 for i in range(active - 2, active) for j in range(active - 2)):
            active -= 2
        else:
            return False
    return True


def show(label, passed):
    print(f"  {label}: {'выполнено' if passed else 'не выполнено'}")
