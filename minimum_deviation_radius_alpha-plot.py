#! /usr/bin/python3

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd 

import matplotlib.pyplot as plt 

import config

if 'momo' in plt.style.available:
    plt.style.use('momo')

version = 3
sub_dir = f"v{version}/"
data_path = config.data_dir + sub_dir + 'deviation_radius_alpha-parallel.csv'
plot_path = config.graphics_dir + sub_dir +'deviation_radius_alpha_size_{}_eps_{}.png'

df = pd.read_csv(data_path, )

rho = df['Rho'].unique()[0]
for size in df['Size'].unique():
    for eps_A in df['Eps A'].unique():
        marker_index = 0
        fig, ax = plt.subplots()
        ax.plot([-.1,1.1],[0.,0.],'k-',)
        for alpha in df['Alpha'].unique():

            sub_df = df[(df['Size']==size) & (df['Eps A']==eps_A) & (df['Alpha']==alpha)]

            x_data =  sub_df['Rad B'] / sub_df['Rad A']
            y_data = sub_df['A'] - sub_df['B']

            ax.plot(
                x_data,
                y_data,
                # config.marker_list[marker_index%len(config.marker_list)]+'--',
                '-',
                # markerfacecolor='None',
                label='{:.2f}'.format(alpha),
                )

            marker_index += 1

        ax.set_ylabel(r'Vote-share diff. $v_{A} - v_{B}$')
        ax.set_xlabel(r'Relative rad. $\frac{r_{B}}{\tilde{r}\left(N\right)}$')
        legend_title = r'$\alpha$'
        ax.legend(title=legend_title, ncol=1, fontsize=6)
        
        # ax.set_xscale('log')

        ax.set_xlim(-.05, 1.05)
        ax.set_ylim(-.004, .0005)

        fig.tight_layout(pad=.25)
        fig.savefig(plot_path.format(int(size), eps_A))
        plt.close(fig=fig)
