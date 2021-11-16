#! ./venv/bin/python

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd 

import matplotlib.pyplot as plt 

import config

for version in range(1, 4):
    sub_dir = f"v{version}/"
    data_path = config.data_dir + sub_dir + 'minimum_deviation_plus_radius.csv'
    plot_diff_path = config.graphics_dir + sub_dir +'minimum_deviation_plus_radius_diff_{}.pdf'
    plot_conclusion_path = config.graphics_dir + sub_dir + 'scaling_deviation_radius.pdf'

    df = pd.read_csv(data_path, )

    rho = df['Rho'].unique()[0]
    for size in df['Size'].unique():
        fig, ax = plt.subplots()
        ax.plot([-.1,1.1],[0.,0.],'k-',)
        marker_index = 0
        for _, eps_A in enumerate(df['Eps A'].unique()):
            if version == 3:
                tester = (eps_A%.125==0.) and (eps_A<=1.)
            else:
                tester = (eps_A/rho%.125==0.)

            if tester:
                sub_df = df[(df['Size']==size) & (df['Eps A']==eps_A)]

                # x_data =  1 - sub_df['Rad B'] / sub_df['Rad A']
                x_data =  sub_df['Rad B'] / sub_df['Rad A']
                y_data = sub_df['A'] - sub_df['B']

                ax.plot(
                    x_data,
                    y_data,
                    config.marker_list[marker_index%len(config.marker_list)]+'--',
                    markerfacecolor='None',
                    label=eps_A if version==3 else eps_A/rho, #int(-np.log2(eps_A)),
                    )

                marker_index += 1
        
        ax.set_ylabel(r'Vote-share diff. $v_{A} - v_{B}$')
        ax.set_xlabel(r'Relative rad. $\frac{r_{B}}{\tilde{r}\left(N\right)}$')
        legend_title = r'$\tilde{\epsilon}$' if version==3 else r'$\tilde{\epsilon}/\rho$'
        ax.legend(title=legend_title, ncol=2, fontsize=6)
        
        # ax.set_xscale('log')

        ax.set_xlim(-.05, 1.05)
        ax.set_ylim(-.08, .02)

        fig.tight_layout(pad=.25)
        fig.savefig(plot_diff_path.format(int(size)))#np.log2(rho/eps_A))))
        plt.close(fig=fig)

    fig, ax = plt.subplots()
    # filter for size to give colors
    df = df[df['B']>df['A']]
    for maker_index, size in enumerate(sorted(df['Size'].unique())):
        sub_df = df[(df['Size'] == size)]

        x_data = []
        y_data = []
        
        for eps in sub_df['Eps A'].unique():
            idx = sub_df[sub_df['Eps A']==eps]['B'].idxmin()
            s = sub_df.loc[idx]
            x_data.append(eps if version==3 else eps/rho)
            y_data.append(s['Rad B']/s['Rad A'])

        ax.plot(
            x_data,
            y_data,
            config.marker_list[maker_index%len(config.marker_list)],
            markerfacecolor='None',
            label=size,
        )

    xaxis_label = r'General influence strength ' if version==3 else r'$\tilde{\epsilon}/\rho$'
    ax.set_xlabel(xaxis_label)
    ax.set_ylabel(r'Max. rel. radius $\frac{\max\{r_{B}\}}{\tilde{r}\left(N\right)}$')

    ax.legend(title=r'$N=$', ncol=2, fontsize=6,)

    ax.set_xlim(0, 1.2)

    fig.tight_layout(pad=.25)
    fig.savefig(plot_conclusion_path)
    plt.close(fig=fig)
