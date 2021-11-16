#! ./venv/bin/python

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd 

import matplotlib.pyplot as plt 


def func(x, a, b, c):
    return a * x**b + c


def func_v2(x, a, b, c):
    return 1 / (x + ((x)/9) * a + c)


# TODO def func_v3():
def func_v3(x, a, b, c):
    """
    Function that returns the margin of victory.

    Args:
        x: Opinion spectrum resolution
        a: Strength of influence for both candidates
        b: Factor for correction
        c: Offset for correction

    Returns:
        Margin of victory
    """
    _rE = (x/9 + 1)/2
    _denominator = x - 2 * _rE
    _denominator += 2 * (1 + a)
    _denominator += 2 * (_rE - 1) * (1 + a)**2
    return 1 / _denominator 


def get_vote_share_different(x, eps,):
    _rE = (x/9 + 1)/2
    _denominator = x - 2 * _rE
    _denominator += 2 * (1 + eps)
    _denominator += 2 * (_rE - 1) * (1 + eps)**2
    return 1 / _denominator


for version in range(1,4):
# version = 3
    data_filename = f'./data/v{version}/' + 'minimum_deviation.csv'
    plot_filename = f'./graphics/v{version}/' + 'minimum_deviation_{}.pdf'
    color_str = 'C{:d}'
    marker_list = ['o', '^', 's', 'h', 'v', 'D',]

    df = pd.read_csv(data_filename, ) # index_col='Size')

    for j, rho in enumerate(df['Rho'].unique()):
        fig, ax = plt.subplots()
        for i, eps in enumerate(df['Eps'].unique()):
            sub_df = df[(df['Eps'] == eps) & (df['Rho'] == rho)]
            sub_df = sub_df.set_index('Size')
            data = sub_df['A'] - sub_df['B']
            ax.plot(
                data,
                marker_list[i%6],
                markerfacecolor='None',
                markeredgecolor=color_str.format(i),
                label= eps if version ==3 else eps/rho,
            )



        ax.set_xlabel(r'Size $N$')
        ax.set_ylabel(r'Vote-share diff. $v_{A} - v_{B}$')
        ax.set_xscale('log')

        ax.set_ylim(0, .12)

        legend_title = r'$\tilde{\epsilon}$' if version==3 else r'$\tilde{\epsilon}/\rho$'
        ax.legend(title=legend_title, ncol=2)

        fig.tight_layout(pad=1.)
        fig.savefig(plot_filename.format(str(np.log2(rho)) + '_linlog'))

        for j, eps in enumerate(df['Eps'].unique()):
            sub_df = df[(df['Eps'] == eps) & (df['Rho'] == rho)]
            sub_df = sub_df.set_index('Size')
            data = sub_df['A'] - sub_df['B']

            if version == 3:
                ax.plot(
                    data.index.values,
                    get_vote_share_different(data.index.values, eps),
                    '-',
                    color=color_str.format(j),
                    alpha=.7,
                )

        ax.set_xlabel(r'Size $N$')
        ax.set_ylabel(r'Vote-share diff. $v_{A} - v_{B}$')
        ax.set_yscale('log')
        ax.set_ylim(1e-4, 1e0)
        fig.tight_layout(pad=1.5)
        fig.savefig(plot_filename.format(str(np.log2(rho)) + '_loglog'))
        plt.close(fig=fig)
