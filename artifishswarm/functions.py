import numpy as np


def rastrigin(x):
    n = len(x)
    A = 10
    sum = 0
    for xi in x:
        sum += xi * xi - A * np.cos(2 * np.pi * xi)
    return A * n + sum


def rosenbrock(x):
    n = len(x)
    val = 0
    for i in range(0, n - 1):
        val += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
    return val


def beale(x):
    return (1.5 - x[0] + x[0] * x[1]) ** 2 + \
        (2.25 - x[0] + x[0] * x[1] ** 2) ** 2 + \
        (2.625 - x[0] + x[0] * x[1] ** 3) ** 2


def bukin6(x):
    return 100 * np.sqrt(np.fabs(x[1] - 0.01 * x[0] ** 2)) + \
        0.01 * np.fabs(x[0] + 10)


def levi13(x):
    return np.sin(3 * np.pi * x[0]) ** 2 + \
        (x[0] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[1]) ** 2) + \
        (x[1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[1]) ** 2)


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def eggholder(x):
    def sinsqrtabs(x):
        return np.sin(np.sqrt(np.fabs(x)))

    y47 = x[1] + 47
    A = -y47 * sinsqrtabs(x[0] / 2 + y47)
    B = -x[0] * sinsqrtabs(x[0] - y47)
    return A + B
