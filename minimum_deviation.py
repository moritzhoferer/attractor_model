#! ./venv/bin/python

from socket import gethostname

import numpy as np
import pandas as pd
import config

from model import StatModel


def get_size(x):
    return 3 ** x


def get_radius(x):
    return (3 ** (x - 2) + 1) // 2


def main():
    for version in range(1,4):
        filename = 'minimum_deviation.csv'
        size_params = np.arange(2, 6, dtype=int)
        filepath = config.data_dir + f'v{version}/' +filename

        system_sizes = get_size(size_params)
        radius_array = get_radius(size_params)

        rho_array =  [2**(-8)] 

        if version == 3:
            eps_array = np.arange( .125, 1.1, .125)
        else:
            eps_array = rho_array[0] * np.arange( .125, .75, .125)

        out = {'Size': [], 'Rho': [], 'Eps': [], 'A': [], 'B': []}

        for rho in rho_array:
            for eps in eps_array:
                for size, radius in zip(system_sizes, radius_array):
                    middle = (size - 1) // 2
                    
                    strategies = {
                        'A':{'pos': middle, 'eps': eps, 'rad': radius},
                        'B':{'pos': middle+1, 'eps': eps, 'rad': radius}
                        }

                    system = StatModel(size=size, rho=rho, strategies=strategies, radius=None, version=version)
                    if not system.stationary_solution_exists:
                        print("---- WARNING -----")
                        break
                    system.state = system.stationary_solution

                    result = system.election_result
                    out['Size'].append(size)
                    out['Rho'].append(rho)
                    out['Eps'].append(eps)
                    for k in strategies.keys():
                        out[k].append(result[k])

        pd.DataFrame(out).to_csv(
            filepath,
            index=False,
            )


if __name__ == "__main__":
    main()
