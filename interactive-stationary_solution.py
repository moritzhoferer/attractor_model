#! ./venv/bin/python

from model import AnimateStationarySolution


def main() -> None:
    """
    This function shows the stationary solution for the setup of Fig. 1
    """
    size = 9
    rho_0 = .05  # float
    eps_0 = 1.  # float
    str_a_0 = 2  # int
    str_b_0 = 7  # int
    strategies_0 = {
        'A':{'pos': str_a_0, 'eps': eps_0, 'rad': 2},
        'B':{'pos': str_b_0, 'eps': eps_0, 'rad': 1}
        }

    p = AnimateStationarySolution(size, rho_0, strategies_0, title='Scaling and stationary solution')
    p.show()


if __name__ == '__main__':
    main()
