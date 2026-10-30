import numpy as np
from scipy.optimize import brentq

def y1_plus(x):
    rad = 5*np.sin(x) + 9 - x**2
    return np.sqrt(rad) if rad >= 0 else np.nan

def y1_minus(x):
    rad = 5*np.sin(x) + 9 - x**2
    return -np.sqrt(rad) if rad >= 0 else np.nan

def y2_A(x):
    a = -3.0
    b = 2*x - np.cos(x)
    c = 2*x**2 - 4*x + 3
    D = b*b - 4*a*c
    if D < 0: return np.nan
    return (-b + np.sqrt(D)) / (2*a)

def y2_B(x):
    a = -3.0
    b = 2*x - np.cos(x)
    c = 2*x**2 - 4*x + 3
    D = b*b - 4*a*c
    if D < 0: return np.nan
    return (-b - np.sqrt(D)) / (2*a)

def find_intersections(yf, yg, x_min=-6, x_max=6, n=4000):
    xs = np.linspace(x_min, x_max, n)
    vals = []
    for t in xs:
        a = yf(t); b = yg(t)
        vals.append(np.nan if (not np.isfinite(a) or not np.isfinite(b)) else (a-b))
    vals = np.array(vals)

    roots = []
    for i in range(n-1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i+1]) and vals[i]*vals[i+1] < 0:
            xL, xR = xs[i], xs[i+1]
            h = lambda z: yf(z) - yg(z)
            x0 = brentq(h, xL, xR)
            y0 = yf(x0)  # = yg(x0)
            roots.append((x0, y0))
    return roots

pairs = [
    (y1_plus,  y2_A),
    (y1_plus,  y2_B),
    (y1_minus, y2_A),
    (y1_minus, y2_B),
]

points = []
for f, g in pairs:
    points += find_intersections(f, g, -6, 6)

# уберём почти одинаковые точки
uniq = []
for p in points:
    if all((abs(p[0]-q[0]) > 1e-5 or abs(p[1]-q[1]) > 1e-5) for q in uniq):
        uniq.append(p)

print("Начальные приближения (x0, y0):")
for p in uniq:
    print(p)
