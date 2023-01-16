import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from artifishswarm.functions import beale, ackley, rastrigin, rosenbrock, bukin6, levi13, himmelblau, eggholder


def visualize(
        func, *,
        y_min, x_min, y_max, x_max,
        step=0.01, log=False):
    for i in range(10):
        fish = pd.read_csv(f"data/{func.__name__}_{i}.csv", usecols=['x', 'y'])
        plt.scatter(fish['x'], fish['y'], c='red', s=5)

    @np.vectorize
    def npvec_wrapper(func, a, b):
        return func(np.array([a, b]))

    x, y = np.mgrid[x_min:x_max:step, y_min:y_max:step]
    z = npvec_wrapper(func, x, y)

    plt.imshow(
        np.flip(z.transpose(), axis=0),
        norm=(LogNorm() if log else None),
        extent=[x_min, x_max, y_min, y_max])
    plt.colorbar()
    plt.title(func.__name__)
    plt.savefig(f"plots/{func.__name__}.png")
    plt.show()


def main():
    visualize(
        ackley,
        x_min=-5, y_min=-5, x_max=5, y_max=5,
        log=False)
    visualize(
        rastrigin,
        x_min=-5, y_min=-5, x_max=5, y_max=5,
        log=False)
    visualize(
        rosenbrock,
        x_min=-5, y_min=-5, x_max=5, y_max=5,
        log=True)
    visualize(
        beale,
        x_min=-5, y_min=-5, x_max=5, y_max=5,
        log=True)
    visualize(
        bukin6,
        x_min=-15, y_min=-3, x_max=-5, y_max=5,
        log=False)
    visualize(
        levi13,
        x_min=-10, y_min=-10, x_max=10, y_max=10,
        log=False)
    visualize(
        himmelblau,
        x_min=-5, y_min=-5, x_max=5, y_max=5,
        log=True)
    visualize(
        eggholder,
        x_min=-1000, y_min=-1000, x_max=1000, y_max=1000,
        step=10,
        log=False)


if __name__ == '__main__':
    main()
