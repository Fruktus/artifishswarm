from copy import copy
import numpy as np
from multiprocessing import Pool

from artifishswarm import AFSA
from artifishswarm.functions import ackley, beale, bukin6, eggholder, himmelblau, levi13, rastrigin, rosenbrock


afsa_params = {
    'func': None,
    'dimensions': 2,  # our function takes one value and returns one value
    'population': 100,  # other parameters chosen empirically
    'max_iterations': 500,
    'vision': 0.5,
    'crowding_factor': 0.5,
    'step': 0.01,
    'search_retries': 10,
    'low': -2,
    'high': 2,
    'save_history': False,
}

functions = {
    'ackley': ackley,
    'beale': beale,
    'bukin6': bukin6,
    'eggholder': eggholder,
    'himmelblau': himmelblau,
    'levi13': levi13,
    'rastrigin': rastrigin,
    'rosenbrock': rosenbrock
}

def run_experiment(afsa_params: dict, num_tries=10):
    optimum_x = []
    optimum_y = []

    for _ in range(num_tries):
        afsa = AFSA(**afsa_params)
        afsa.run()
        optimum_x.append(afsa.result[0])
        optimum_y.append(afsa.result[1])

    return optimum_x, optimum_y


def worker(job_params):
    func_name = job_params.pop('func_name')

    res_x, res_y = run_experiment(job_params)
    print(f'{func_name} best_x: {np.average(res_x, axis=0)} std: {np.std(res_x, axis=0)}, best_y: {np.average(res_y)} std: {np.std(res_y)} actual_y: 0.0')


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
