#! ./venv/bin/python

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
    for version in range(1,4):

        print("Choosen device: ", gethostname())
        
        filename = 'minimum_deviation_plus_radius.csv'
        if gethostname() == 'mtec-mip-507':
            size_params = np.arange(2, 7, dtype=int)
            filepath = config.data_dir + f'v{version}/' + filename
        else:
            size_params = np.arange(2, 7, dtype=int)
            filepath = config.data_test_dir + filename

        system_sizes = get_size(size_params)
        radius_array = get_radius(size_params)

        rho_array = [2**-8] 

        if version == 3:
            eps_min_inverse = 16
            eps_array = np.arange(1/eps_min_inverse, 1.2, 1/eps_min_inverse) 
        else:
            eps_array = rho_array[0] * np.arange( .125, .75, .125)
        
        r = []

        # for eps in eps_array:
        for rho in rho_array:
            # 1e-14 is subtracted to ensure that now off-diagonal value equals zero. 
            if version == 1:
                eps_max = .25 - .5 * rho - 1e-14
            elif version == 2:
                eps_max = .5 - rho- 1e-14
            elif version == 3:
                eps_max = np.sqrt((1/rho)-1) - 1 - 1e-14
            
            for eps in eps_array:
                eps_a = min(eps, eps_max)

                for size, radius in zip(system_sizes, radius_array):
                    print('Eps: ', eps_a, '\tSize: ', size)
                    middle = (size - 1) // 2
                    reduction = 0

                    def solve_problem(strategies: dict) -> dict:
                        system = StatModel(size=size, rho=rho, strategies=strategies, radius=None, version=version)
                        if system.stationary_solution_exists:
                            system.state = system.analytic_stationary_solution
                            # Alternatively calculate the stationary solution numerically 
                            # system.state = system.stationary_solution
                            result = system.election_result
                            
                            out = {}
                            out['Size'] = system.size
                            out['Rho'] = system.rho
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
                        eps_b = min(eps * radius / (radius - reduction), eps_max)

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
