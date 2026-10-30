"""Задание 2.1, вариант 15: sin(x) - x² + 1 = 0."""

from pathlib import Path
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

epsilon = 1e-16
MAX_ITERATIONS = 10000
INTERVAL = (1.0, 1.5)
X0 = 1.5
NEGATIVE_INTERVAL = (-0.8, -0.5)
NEGATIVE_X0 = -0.8
NEGATIVE_Q = 0.3
NEGATIVE_M = np.cos(0.8) + 1


def func(x):
    return np.sin(x) - x ** 2 + 1


def derivative(x):
    return np.cos(x) - 2 * x


def phi(x, negative=False):
    return x - 0.5 * func(x) if negative else np.sqrt(1 + np.sin(x))


# На [1, 1.5] phi переводит отрезок в себя, |phi'| <= q < 1.
q = np.cos(INTERVAL[0]) / (2 * np.sqrt(1 + np.sin(INTERVAL[0])))
# |f'| >= m; по теореме о среднем |x - x*| <= |f(x)| / m.
m = 2 * INTERVAL[0] - np.cos(INTERVAL[0])


def simple_iteration(x0=None, eps=epsilon, max_iterations=MAX_ITERATIONS, *, negative=False):
    x = float((NEGATIVE_X0 if negative else X0) if x0 is None else x0)
    contraction = NEGATIVE_Q if negative else q
    history = []
    for iteration in range(1, max_iterations + 1):
        previous = x
        x = phi(previous, negative=negative)
        error = contraction / (1 - contraction) * abs(x - previous)
        residual = abs(func(x))
        error = min(error, residual / (NEGATIVE_M if negative else m))
        history.append((iteration, x, error, residual))
        if error <= eps:
            return x, history
    raise RuntimeError('Метод простой итерации не сошёлся')


def newton_method(x0=None, eps=epsilon, max_iterations=MAX_ITERATIONS, *, negative=False):
    x = float((NEGATIVE_X0 if negative else X0) if x0 is None else x0)
    derivative_bound = NEGATIVE_M if negative else m
    history = []
    for iteration in range(1, max_iterations + 1):
        x -= func(x) / derivative(x)
        residual = abs(func(x))
        error = residual / derivative_bound
        history.append((iteration, x, error, residual))
        if error <= eps:
            return x, history
    raise RuntimeError('Метод Ньютона не сошёлся')


def main():
    print('sin(x) - x² + 1 = 0; вариант 15')
    print('Отрезок:', INTERVAL, 'Начальное приближение:', X0, 'epsilon:', epsilon)
    print('Коэффициент сжатия q:', q)
    fig = plt.figure(figsize=(14, 13), layout="constrained")
    layout = fig.add_gridspec(3, 2)
    axes = [fig.add_subplot(layout[0, :]),
            fig.add_subplot(layout[1, 0]), fig.add_subplot(layout[1, 1]),
            fig.add_subplot(layout[2, 0]), fig.add_subplot(layout[2, 1])]
    grid = np.linspace(-3, 3, 1200)
    axes[0].plot(grid, func(grid), label='sin(x) - x² + 1')
    axes[0].axhline(0, color='black', linewidth=0.7)
    axes[0].axvline(0, color='black', linewidth=0.7)
    axes[0].axvspan(*INTERVAL, alpha=0.15, label='Отрезок [1, 1.5]')
    axes[0].axvspan(*NEGATIVE_INTERVAL, color='tab:orange', alpha=0.15,
                   label='Отрезок [-0.8, -0.5]')
    axes[0].set(xlabel='x', ylabel='f(x)', title='Общий вид: оба пересечения с осью x',
                xlim=(-3, 3))
    for negative, description in [(False, 'Положительный'), (True, 'Отрицательный')]:
        interval = NEGATIVE_INTERVAL if negative else INTERVAL
        initial = NEGATIVE_X0 if negative else X0
        local_ax = axes[3 if negative else 1]
        convergence_ax = axes[4 if negative else 2]
        local_grid = np.linspace(interval[0] - 0.1, interval[1] + 0.1, 400)
        local_ax.plot(local_grid, func(local_grid), label='sin(x) - x² + 1')
        local_ax.axhline(0, color='black', linewidth=0.7)
        local_ax.axvspan(*interval, alpha=0.15, label=f'Отрезок {interval}')
        local_ax.scatter([initial], [func(initial)], label='Начальное приближение', zorder=3)
        local_ax.set(xlabel='x', ylabel='f(x)', title=f'{description} корень: увеличение')
        convergence_ax.axhline(epsilon, linestyle='--', color='gray', label='epsilon')
        convergence_ax.set(xlabel='Итерация', ylabel='Верхняя оценка ошибки',
                           title=f'{description} корень: сходимость методов')
        print(f'\n{description} корень: отрезок {interval}')
        print('Коэффициент сжатия q:', NEGATIVE_Q if negative else q)
        for name, method in [('Простая итерация', simple_iteration), ('Ньютон', newton_method)]:
            root, history = method(negative=negative)
            print('\n' + name)
            print('Шаг          x          Оценка ошибки       |f(x)|')
            for i, x, error, residual in history:
                print(f'{i:3d} {x:14.10f} {error:16.6e} {residual:16.6e}')
            print(f'{description} корень:', root, 'Число итераций:', len(history))
            convergence_ax.semilogy([h[0] for h in history], [max(h[2], 1e-16) for h in history],
                             'o-', label=name)
        axes[0].scatter([root], [func(root)], zorder=3, label=f'x ≈ {root:.6f}')
        local_ax.scatter([root], [func(root)], zorder=4, label=f'Корень ≈ {root:.6f}')
    for ax in axes:
        ax.grid(True)
        ax.legend()
    output = Path(__file__).with_name('problem1.png')
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print('\nПростая итерация сходится линейно; Ньютон вблизи корня — квадратично.')
    print('Графики:', output)


if __name__ == '__main__':
    main()
