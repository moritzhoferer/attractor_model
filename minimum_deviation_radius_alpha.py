#! /usr/bin/python3

from socket import gethostname

import numpy as np
import pandas as pd

import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
num_cores = multiprocessing.cpu_count()

import config
from minimum_deviation import get_radius, get_size
from model import StatModel

def main():
    # Just on version 3
    version: int = 3

    filename = 'deviation_radius_alpha-parallel.csv'
    size_params = np.arange(2, 7, dtype=int)
    filepath = config.data_dir + f'v{version}/' + filename
        
    system_sizes = get_size(size_params)
    radius_array = get_radius(size_params)

    rho_array = [2**-8] 
    eps_min_inverse = 8
    eps_array = np.arange(1/eps_min_inverse, 1., 1/eps_min_inverse) 

    alpha_array = np.sort(np.append(
        np.array([.25, .5, .99]),
        np.arange(.75, 1., .05)
    ))
       
    r = []
    
    for rho in rho_array:
        # 1e-14 is subtracted to ensure that now off-diagonal value equals zero.
        eps_max = np.sqrt((1/rho)-1) - 1 - 1e-14
        
        for alpha in alpha_array:

            for eps in eps_array:
                eps_a = min(eps, eps_max)

                for size, radius in zip(system_sizes, radius_array):
                    print('Eps: ', eps_a, '\tSize: ', size, '\tAlpha: ', alpha)
                    middle = (size - 1) // 2
                    reduction = 0

                    def solve_problem(strategies: dict) -> dict:
                        system = StatModel(size=size, rho=rho, strategies=strategies, radius=None, version=version)
                        if system.stationary_solution_exists:
                            system.state = system.analytic_stationary_solution
                            result = system.election_result
                            
                            out = {}
                            out['Size'] = system.size
                            out['Rho'] = system.rho
                            out['Alpha'] = alpha
                            for k in strategies.keys():
                                out['Str {}'.format(k)] = system.strategies[k]['pos']
                                out['Eps {}'.format(k)] = system.strategies[k]['eps']
                                out['Rad {}'.format(k)] = system.strategies[k]['rad']
                                out[k] = result[k]
                        else:
                            out = None
                        return out
                    
                    params = []
                    while radius - reduction > 0:
                        eps_b = min(eps * (radius / (radius - reduction))**((1-alpha)/alpha), eps_max)

                        strategies = {
                            'A':{'pos': middle, 'eps': eps_a, 'rad': radius},
                            'B':{'pos': middle+1, 'eps': eps_b, 'rad': radius - reduction}
                            }
                        params.append(strategies)
                        reduction += 1

                    r += Parallel(n_jobs=num_cores)(delayed(solve_problem)(i) for i in tqdm(params))

                pd.DataFrame(r).to_csv(
                    filepath,
                    index=False,
                    )

if __name__ == "__main__":
    main()
