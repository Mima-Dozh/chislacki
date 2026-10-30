"""Задание 2.2, вариант 15: x²/16 + y²/4 = 1 и 4*y = exp(x) + x."""

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

if __package__ and '.' in __package__:
    from ..lab1.problem1 import get_LU, solve_lu
else:
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lab1.problem1 import get_LU, solve_lu

epsilon = 1e-12
MAX_ITERATIONS = 10000000
CONTRACTION_FACTOR = 0.3


@dataclass(frozen=True)
class SolutionRegion:
    name: str
    x_interval: tuple[float, float]
    y_interval: tuple[float, float]
    x0: tuple[float, float]
    negative: bool = False


FIRST_REGION = SolutionRegion(
    name='Первое', x_interval=(1.6, 1.8), y_interval=(1.75, 1.85), x0=(1.7, 1.8),
)
SECOND_REGION = SolutionRegion(
    name='Второе', x_interval=(-3.7, -3.5), y_interval=(-0.95, -0.85),
    x0=(-3.6, -0.9), negative=True,
)


def norm_inf(vector):
    return max(abs(value) for value in vector)


def func(point):
    x, y = point
    return np.array([x ** 2 / 16 + y ** 2 / 4 - 1,
                     4 * y - np.exp(x) - x])


def jacobian(point):
    x, y = point
    return np.array([[x / 8, y / 2],
                     [-np.exp(x) - 1, 4]])


def phi(point, negative=False):
    x, y = point
    if negative:
        y = (np.exp(x) + x) / 4
        x = -4 * np.sqrt(1 - y ** 2 / 4)
        return np.array([x, (np.exp(x) + x) / 4])
    return np.array([np.log(4 * y - x), 2 * np.sqrt(1 - x ** 2 / 16)])


def simple_iteration(x0=None, eps=epsilon, max_iterations=MAX_ITERATIONS, region=FIRST_REGION):
    x = np.array(region.x0 if x0 is None else x0, dtype=float)
    contraction = CONTRACTION_FACTOR
    history = []
    for iteration in range(1, max_iterations + 1):
        previous = x
        x = phi(previous, negative=region.negative)
        error = contraction / (1 - contraction) * norm_inf(x - previous)
        history.append((iteration, x.copy(), error, norm_inf(func(x))))
        if error <= eps:
            return x, history
    raise RuntimeError('Метод простой итерации не сошёлся')


def newton_method(x0=None, eps=epsilon, max_iterations=MAX_ITERATIONS, region=FIRST_REGION):
    x = np.array(region.x0 if x0 is None else x0, dtype=float)
    contraction = CONTRACTION_FACTOR
    history = []
    for iteration in range(1, max_iterations + 1):
        x += solve_lu(get_LU(jacobian(x)), -func(x))
        error = norm_inf(phi(x, negative=region.negative) - x) / (1 - contraction)
        history.append((iteration, x.copy(), error, norm_inf(func(x))))
        if error <= eps:
            return x, history
    raise RuntimeError('Метод Ньютона не сошёлся')


def main():
    fig = plt.figure(figsize=(14, 14), layout="constrained")
    layout = fig.add_gridspec(3, 2)
    axes = [fig.add_subplot(layout[0, :]), fig.add_subplot(layout[1, 0]),
            fig.add_subplot(layout[1, 1]), fig.add_subplot(layout[2, 0]),
            fig.add_subplot(layout[2, 1])]
    angle = np.linspace(0, 2 * np.pi, 1000)
    grid = np.linspace(-4.5, 4.5, 1200)
    axes[0].plot(4 * np.cos(angle), 2 * np.sin(angle), label='x²/16 + y²/4 = 1')
    axes[0].plot(grid, (np.exp(grid) + grid) / 4, label='y = (exp(x) + x)/4')
    axes[0].axhline(0, color='black', linewidth=0.7)
    axes[0].axvline(0, color='black', linewidth=0.7)
    axes[0].set(xlim=(-4.5, 4.5), ylim=(-2.5, 3), xlabel='x', ylabel='y',
                title='Общий вид: оба пересечения')
    axes[0].set_aspect('equal', adjustable='box')
    for index, region in enumerate((FIRST_REGION, SECOND_REGION)):
        description = region.name
        xi, yi = region.x_interval, region.y_interval
        initial = region.x0
        local_ax, convergence_ax = axes[2 * index + 1:2 * index + 3]
        grid = np.linspace(xi[0] - 0.1, xi[1] + 0.1, 400)
        local_ax.plot(grid, (-2 if region.negative else 2) * np.sqrt(1 - grid ** 2 / 16),
                      label='Эллипс')
        local_ax.plot(grid, (np.exp(grid) + grid) / 4, label='y = (exp(x) + x)/4')
        local_ax.set(xlabel='x', ylabel='y', title=f'{description} решение: увеличение')
        color = 'tab:orange' if region.negative else 'tab:green'
        for ax in (axes[0], local_ax):
            ax.add_patch(Rectangle((xi[0], yi[0]), xi[1] - xi[0], yi[1] - yi[0],
                                   color=color, alpha=0.2, label=f'Область {xi} × {yi}'))
        local_ax.scatter(*initial, label='Начальное приближение', zorder=3)
        print(f'\n{description} решение: область {xi} × {yi}, x0 = {initial}')
        print('Коэффициент сжатия q:', CONTRACTION_FACTOR)
        for name, method in [('Простая итерация', simple_iteration), ('Ньютон', newton_method)]:
            root, history = method(region=region)
            print('\n' + name)
            print('Шаг          x              y          Оценка ошибки      ||F||∞')
            for i, point, error, residual in history:
                print(f'{i:3d} {point[0]:14.10f} {point[1]:14.10f} {error:16.6e} {residual:16.6e}')
            print(f'{description} решение:', root, 'Число итераций:', len(history))
            convergence_ax.semilogy([h[0] for h in history], [max(h[2], 1e-16) for h in history],
                                    'o-', label=name)
        for ax in (axes[0], local_ax):
            ax.scatter(*root, zorder=4, label=f'({root[0]:.6f}, {root[1]:.6f})')
        convergence_ax.axhline(epsilon, linestyle='--', color='gray', label='epsilon')
        convergence_ax.set(xlabel='Итерация', ylabel='Верхняя оценка ошибки (норма ∞)',
                           title=f'{description} решение: сходимость')
    for ax in axes:
        ax.grid(True)
        ax.legend()
    output = Path(__file__).with_name('problem2.png')
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print('\nПростая итерация сходится линейно; Ньютон вблизи решения — квадратично.')
    print('Графики:', output)


if __name__ == '__main__':
    main()
