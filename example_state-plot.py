#! ./venv/bin/python

from model import StatModel
import matplotlib.pyplot as plt


# My personal style sheet
if 'momo' in plt.style.available:
    plt.style.use('momo')


# def main() -> None:


if __name__ == '__main__':
    # main()
    for version in range(1,4):
        size = 9
        rho_0 = 2**(-8)  # float
        immediate_influence = .5 * 1
        str_a_0 = 2  # int
        str_b_0 = 7  # int
        r_a = 2
        r_b = 1

        strategies_0 = {
            'A':{'pos': str_a_0, 'eps': immediate_influence/r_a, 'rad': r_a},
            'B':{'pos': str_b_0, 'eps': immediate_influence/r_b, 'rad': r_b}
            }

        p = StatModel(size, rho_0, strategies_0, version=version)
        p.state = p.stationary_solution
        
        fig, ax = p.plot(stat_sol=True)
        fig.tight_layout(pad=.25)
        fig.savefig(f'./graphics/v{version}/example_state.pdf')
