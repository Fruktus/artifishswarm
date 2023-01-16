from copy import copy
import numpy as np
from multiprocessing import Pool

import pandas as pd

from artifishswarm import AFSA
from artifishswarm.functions import ackley, beale, bukin6, eggholder, himmelblau, levi13, rastrigin, rosenbrock


afsa_params = {
    'func': None,
    'dimensions': 2,  # all of our functions can work with 2 dimensions
    'population': 100,  # other parameters chosen empirically
    'max_iterations': 500,
    'vision': 0.02,
    'crowding_factor': 0.5,
    'step': 0.01,
    'search_retries': 10,
    'low': -5,
    'high': 5,
    'save_history': False,
}

functions = {
    'ackley': (ackley, [0.0, 0.0], 0.0),
    'beale': (beale, [3.0, 0.5], 0.0),
    'bukin6': (bukin6, [-10.0, 1.0], 0.0),
    'eggholder': (eggholder, [512.0, 404.2319], -959.6407),
    'himmelblau': (himmelblau, [-2.805118, 3.131312], 0.0),
    'levi13': (levi13, [1.0, 1.0], 0.0),
    'rastrigin': (rastrigin, [0.0, 0.0], 0.0),
    'rosenbrock': (rosenbrock, [1.0, 1.0], 0.0)
}

def run_experiment(func_name: str, afsa_params: dict, num_tries=10):
    optimum_x = []
    optimum_y = []

    for i in range(num_tries):
        afsa = AFSA(**afsa_params)
        afsa.run()
        df = pd.DataFrame(data=afsa.fish, columns=['x', 'y'])
        df.to_csv(f"data/{func_name}_{i}.csv", sep=',')
        optimum_x.append(afsa.result[0])
        optimum_y.append(afsa.result[1])

    return optimum_x, optimum_y


def worker(job_params):
    func_name = job_params.pop('func_name')
    func_params = job_params['func']
    job_params['func'] = func_params[0]
    func_bext_x = func_params[1]
    func_best_y = func_params[2]

    res_x, res_y = run_experiment(func_name, job_params)
    print(f'{func_name} best_x: {np.average(res_x, axis=0)} std: {np.std(res_x, axis=0)} (actual: {func_bext_x}), best_y: {np.average(res_y)} std: {np.std(res_y)} (actual: {func_best_y})')


def main():
    jobs = []
    for f_name, f in functions.items():
        job = copy(afsa_params)
        job['func_name'] = f_name
        job['func'] = f
        job['optimize_towards'] = 'min'
        jobs.append(job)

    with Pool(processes=len(functions)) as pool:  # The experiments are independent so we can parallelize them
        pool.map(worker, jobs)

if __name__ == '__main__':
    main()
