#!/usr/bin/python3.6

# TODO Format code as class voterAttractorModel

import warnings

import pandas as pd
import numpy as np
from scipy.linalg import eig

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Slider, Button, RadioButtons

# My personal style sheet
if 'momo' in plt.style.available:
    plt.style.use('momo')


class StatModel(object):
    def __init__(self, size: int = 9, rho: float = .01, strategies: dict = None, 
        radius: int = 1, version: int = 3):
        self._state = None
        self.state = np.ones(size) / size 
        self.rho = rho
        self.strategies = strategies
        if self.strategies:
            for k in self.strategies.keys():
                if ('rad' not in self.strategies[k].keys()) and (radius is not None):
                    self.strategies[k]['rad'] = radius
                self.strategies[k]['init eps'] = self.strategies[k]['eps']
                self.strategies[k]['init rad'] = self.strategies[k]['rad']
        self.version = version
        self.counter = 0

        self.party_colors = {'A': 'blue', 'B': 'red'}
        self.background_alpha = .2

    @property
    def state(self) -> np.array:
        """
        Returns:
            State of the system.
        """
        return self._state

    @state.setter
    def state(self, array: np.array):
        if sum(array < 0) > 0:
            raise ValueError('All entries have to be larger or equal 0')
        try:
            if self._state is not None:
                if len(self.state) != len(array):
                    warnings.warn(f'You are changing the size of the system from {self.size} to {len(array)}')
        except AttributeError:
            pass
        self._state = array

    
    @property
    def rho(self) -> float:
        """
        Return:
            Diffusion probability to neighboring opinion state
        """
        return self._rho

    @rho.setter
    def rho(self, value: float):
        if value <= 0.:
            raise ValueError('The diffusion probability has to be positive.')
        elif value > .5:
            raise ValueError('To high value, as remain probability becomes negative.')
        self._rho = value


    @property
    def size(self):
        return len(self.state)

    @size.setter
    def size(self, value: int):
        if value < 1:
            raise ValueError('The size has to be at least one!')
        _out = np.ones(value)
        self.state =  _out / len(_out)

    @property
    def version(self):
        return self._version

    @version.setter
    def version(self, value: int):
        if value in [1, 2, 3]:
            self._version = int(value)
        else:
            raise ValueError(f'Requested version {value} is not defined.')

    @property
    def trans_matrix(self) -> np.ndarray:
        """
        Get right-stochastic transition matrix for given size, diffusion
        probability, and strategies.

        The function returns a right-stochastic transition matrix T for given size,
        diffusion probability, and strategies. Entry i,j gives the probability from
        state i to j.

        Returns:
            Transition matrix which implements all given parameters.
        """
        # Off-diagonals carry transition to the next opinion to the left and to the right
        _tm = np.diag([self.rho] * (self.size-1), k=1)
        _tm += np.diag([self.rho] * (self.size-1), k=-1)

        if self.strategies:
            for strategy in self.strategies.values():
                if self.version==1:
                    # OLD [BEGIN]
                    for r in range(-strategy['rad']+1, strategy['rad']):
                        op_i = strategy['pos'] + r

                        # Attraction of voters in lower states than option i (from the left)
                        if op_i > 0 and op_i < self.size:
                            _tm[op_i-1, op_i] += strategy['eps']
                            # _tm[strategy['pos']+r-1, strategy['pos']+r-1] -= strategy['eps']
                        
                        # Attraction of voters in larger states than option i (from the right)
                        if op_i + 1 < self.size and op_i > -1:
                            _tm[op_i+1, op_i] += strategy['eps']
                            # _tm[strategy['pos']+r+1, strategy['pos']+r+1] -= strategy['eps']
                    # OLD [END]

                elif self.version==2:
                    # NEW [BEGIN] Additive approach
                    # Attraction from the left (le = left edge)
                    _le = strategy['pos'] - strategy['rad']
                    if _le > -1:
                        _tm[_le, _le+1] += strategy['eps']
                    # Attraction from the right (re = right edge)
                    _re = strategy['pos'] + strategy['rad']
                    if _re < self.size:
                        _tm[_re, _re-1] += strategy['eps']
                    # NEW [END]

                elif self.version==3:
                    # NEW [BEGIN] Amplifying approach
                    # Attraction from the left (le = left edge)
                    _le = strategy['pos'] - strategy['rad']
                    if _le > -1:
                        _tm[_le, _le+1] *= 1 + strategy['eps']
                    # Attraction from the right (re = right edge)
                    _re = strategy['pos'] + strategy['rad']
                    if _re < self.size:
                        _tm[_re, _re-1] *= 1 + strategy['eps']
                    # NEW [END]
                else:
                    NotImplementedError("Version not implemented!")

        # The diagonal carries probability to stay with current opinion
        _tm += np.diag(1 - np.sum(_tm, axis=1))

        if np.any(_tm < 0):
            print('Warning: At least one negative probability')

        return _tm


    @property
    def stationary_solution_exists(self) -> bool:
        """
        Function returns if a unique stationary solution exits

        Returns:
            Solution exists or nto (Boolean)

        TODO Add reference to literature paper/book that defines criteria
        """
        _tm = self.trans_matrix
        _output = not np.any(_tm < 0) or np.all(_tm < 1)
        return _output


    @property
    def stationary_solution(self) -> np.array:
        """
        Get the stationary solution of a transition matrix.

        Calculate eigenvalues and right eigenvectors for the transition matrix.
        Look up for the index of the eigenvalue closest to one.
        Get the respective normalized eigenvector which represents the stationary solution.
        Source of inspiration:
        http://people.duke.edu/~ccc14/sta-663-2016/homework/Homework02_Solutions.html

        Returns:
            Stationary solution (1d array)

        TODO:
            Check that the transition matrix does not have cyclic solutions.
        """
        _evals, _evecs = eig(self.trans_matrix, left=True, right=False)
        _id = np.argmin(np.abs(_evals - 1))
        return normalize_vec(np.real(_evecs[:, _id]))


    @property
    def analytic_stationary_solution(self) -> np.array:
        """
        Get the analytically solved stationary solution of the transition matrix

        TODO Implement solution from Latouche, G. and Ramaswami, V., 1999. 

        Source: Latouche, G. and Ramaswami, V., 1999. Introduction to matrix
        analytic methods in stochastic modeling. Society for Industrial and 
        Applied Mathematics.


        Return:
            Stationary solution (1d array) 
        """
        _tm = self.trans_matrix
        _result = np.ones(self.size)
        for i in range(1, self.size):
            _result[i] = _result[i-1] * _tm[i-1,i] / _tm[i, i-1]
        return normalize_vec(_result)

    
    @property
    def election_result(self) -> dict:
        """
        Get the share for each party given the current state and strategies of parties.

        Each voter votes for the party which is closest to his ideology.
        If the distance is equal to two or more she vote uniformly random form the closest ones.

        Return:
            Vote-share for each party.
        """
        _pos = pd.DataFrame(self.strategies).T.pos
        _result = {i: 0 for i in _pos.index}
        for n, share in enumerate(self.state):
            dist = abs(_pos - n)
            winners = dist[dist == dist.min()].index
            for i in winners:
                _result[i] += share / len(winners)
        return _result


    @property
    def max_eps(self) -> float:
        if self.strategies:
            return (1-2*self.rho)/len(self.strategies)
        else:
            warnings.warn("No strategies defined.")
            return None

    def evolution(self, iterations: int = 1):
        """
        Time evolution form a given state with a given transition matrix for n
        interations.

        Args:
            init_state: Initial state to start from (1d array)
            
        Returns:
            State of the system after all iterations.
        """
        self.state.dot(np.linalg.matrix_power(self.trans_matrix, iterations))
        self.counter += iterations

    def plot(self, stat_sol=False):
        self.fig, self.ax = plt.subplots()
        self.xdata = np.arange(self.size)
        self.ydata = self.state
        
        self.background = []
        for state in range(self.size):
            dist = abs(pd.DataFrame(self.strategies).T.pos - state)
            winners = dist[dist == dist.min()].index
            if len(winners) > 1:
                self.background.append(
                    self.ax.axvspan(
                        state - .5, state + .5,
                        facecolor='grey',
                        alpha=self.background_alpha
                    )
                )
            else:
                self.background.append(
                    self.ax.axvspan(
                        state - .5, state + .5,
                        facecolor=self.party_colors[winners[0]],
                        alpha=self.background_alpha
                    )
                )

        self.line, = self.ax.plot(
            self.xdata,
            self.ydata,
            'ko-',
            linewidth=9./self.size,
            markerfacecolor='None',
            markersize=45./self.size,
        )
        if stat_sol:
            self.analytic_data = self.analytic_stationary_solution
            self.analytic_sol, = self.ax.plot(
                self.xdata,
                self.analytic_data,
                'r--',
                linewidth=9./self.size,
            )
        
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        first_tick = int((self.size//9 - 1)/2)
        self.ax.set_xticks(np.arange(first_tick, int(self.size), int(self.size//9)))
        self.ax.set_xlim(-.5, self.size - .5)
        self.ax.set_ylim(.0, 2.5/self.size)
        self.ax.set_xlabel(r'Opinion state $i$')
        self.ax.set_ylabel(r'Opinion-share $s_i$')

        self.result = self.election_result
        self.result_text = self.ax.text(
            self.size/9 - 1, 2 / self.size,
            'A: {a:.2f}\nB: {b:.2f}'.format(a=self.result['A'], b=self.result['B'])
        )

        return self.fig, self.ax


class AnimateStationarySolution(StatModel):

    def __init__(self, size: int, rho: float, strategies: dict,
                 title: str = None, const_infl: bool = False):
        super().__init__(size=size, rho=rho, strategies=strategies)
        self.const_infl = const_infl
        if self.const_infl:
            for k in self.strategies.keys():
                strategies[k]['rad'] = self.equivalence_radius
        
        self.fig, self.ax = plt.subplots(figsize=[4.,2.5],)
        self.fig.subplots_adjust(bottom=0.42, top=.9)
        if title:
            self.fig.canvas.set_window_title(title) 
        # Define axes to place sliders in
        self.slider_width, self.slider_height = .25, .035
        # From documentation: 4-tuple of floats rect = [left, bottom, width, height]
        self.ax_size = plt.axes([.175, .18, self.slider_width, self.slider_height])
        self.ax_rho = plt.axes([0.625, 0.18, self.slider_width, self.slider_height])
        self.ax_eps_a = plt.axes([0.175, 0.13, self.slider_width, self.slider_height])
        self.ax_eps_b = plt.axes([0.625, 0.13, self.slider_width, self.slider_height])
        self.ax_r_a = plt.axes([0.175, 0.08, self.slider_width, self.slider_height])
        self.ax_r_b = plt.axes([0.625, 0.08, self.slider_width, self.slider_height])
        self.ax_str_a = plt.axes([0.175, 0.03, self.slider_width, self.slider_height])
        self.ax_str_b = plt.axes([0.625, 0.03, self.slider_width, self.slider_height])

        # Define sliders
        self.slider_size = Slider(self.ax_size, r'$x$', 2, 5, valinit=log3(self.size), valfmt='%d', valstep=1,)
        self.slider_rho = Slider(self.ax_rho, r'$\rho$', .01, .25, valinit=self.rho, valstep=.01)
        self.slider_eps_a = Slider(self.ax_eps_a, r'$\epsilon_A$', .0, 2., valinit=self.strategies['A']['eps'], valstep=.25)
        self.slider_eps_b = Slider(self.ax_eps_b, r'$\epsilon_B$', .0, 2., valinit=self.strategies['B']['eps'], valstep=.25)
        self.slider_r_a = Slider(self.ax_r_a, r'$r_{A}$', 1, self.size-1, valinit=self.strategies['A']['rad'], valfmt='%d', valstep=1,)
        self.slider_r_b = Slider(self.ax_r_b, r'$r_{B}$', 1, self.size-1, valinit=self.strategies['B']['rad'], valfmt='%d', valstep=1,)
        self.slider_str_a = Slider(self.ax_str_a, r'$S_A$', 0, self.size-1, valinit=self.strategies['A']['pos'], valfmt='%d', valstep=1)
        self.slider_str_b = Slider(self.ax_str_b, r'$S_B$', 0, self.size-1, valinit=self.strategies['B']['pos'], valfmt='%d', valstep=1)
        
        self.plot()


    @property
    def equivalence_radius(self) -> int:
        if self.size < 9:
            raise ValueError("No equivalence scale-invariant radius defined for resolutions < 9. Returning 0")
        else:
            return int((self.size/9 + 1)//2)



    def plot(self):
        if self.size < 82:
            self.state = self.stationary_solution
        else:
            self.state = self.analytic_stationary_solution
        self.xdata = np.arange(self.size)
        self.ydata = self.state
        
        self.background = []
        for state in range(self.size):
            dist = abs(pd.DataFrame(self.strategies).T.pos - state)
            winners = dist[dist == dist.min()].index
            if len(winners) > 1:
                self.background.append(
                    self.ax.axvspan(
                        state - .5, state + .5,
                        facecolor='grey',
                        alpha=self.background_alpha
                    )
                )
            else:
                self.background.append(
                    self.ax.axvspan(
                        state - .5, state + .5,
                        facecolor=self.party_colors[winners[0]],
                        alpha=self.background_alpha
                    )
                )

        self.line, = self.ax.plot(
            self.xdata,
            self.ydata,
            'ko-',
            linewidth=9./self.size,
            markerfacecolor='None',
            markersize=45./self.size,
        )
        if self.size < 82:
            self.analytic_data = self.analytic_stationary_solution
            self.analytic_sol, = self.ax.plot(
                self.xdata,
                self.analytic_data,
                'r--',
                linewidth=9./self.size,
            )

        self.ax.set_xlim(-.5, self.size - .5)
        self.ax.set_ylim(.0, 4.5/self.size)
        self.ax.set_xlabel(r'Opinion state $i$')
        self.ax.set_ylabel(r'Opinion-share $s_i$')
        self.ax.set_xticks(np.arange(int((self.size/9-1)/2), self.size, self.size/9))

        self.result = self.election_result
        self.result_text = self.ax.text(
            self.size/18 - 1/2, 3.15 / self.size,
            'Election result\nA: {a:.2f}\nB: {b:.2f}'.format(a=self.result['A'], b=self.result['B']),
            fontsize='smaller',
        )

        # Functions to call, when sliders are updated
        self.slider_size.on_changed(self.update_size)
        self.slider_rho.on_changed(self.update)
        self.slider_eps_a.on_changed(self.update)
        self.slider_eps_b.on_changed(self.update)
        self.slider_r_a.on_changed(self.update_radius)
        self.slider_r_b.on_changed(self.update_radius)
        self.slider_str_a.on_changed(self.update)
        self.slider_str_b.on_changed(self.update)


    def update(self, val):
        self.rho = self.slider_rho.val
        self.strategies['A']['pos'] = int(self.slider_str_a.val)
        self.strategies['B']['pos'] = int(self.slider_str_b.val)
        self.strategies['A']['eps'] = self.slider_eps_a.val
        self.strategies['B']['eps'] = self.slider_eps_b.val
    
        self.update_plot()


    def update_radius(self, val):
        self.strategies['A']['rad'] = int(self.slider_r_a.val)
        self.strategies['B']['rad'] = int(self.slider_r_b.val)
        if self.const_infl:
            self.strategies['A']['eps'] = min(
                    self.strategies['A']['init eps']*self.equivalence_radius/self.strategies['A']['rad'],
                    self.max_eps
                )
            self.slider_eps_a.set_val(self.strategies['A']['eps'])
            self.strategies['B']['eps'] = min(
                    self.strategies['B']['init eps']*self.equivalence_radius/self.strategies['B']['rad'],
                    self.max_eps
                )
            self.slider_eps_b.set_val(self.strategies['B']['eps'])
        else:
            self.update_plot()



    def update_plot(self):
        self.state = self.stationary_solution
        self.line.set_ydata(self.state)
        self.analytic_data = self.analytic_stationary_solution
        self.analytic_sol.set_ydata(self.analytic_data)
        
        for b in self.background:
            b.remove()
        self.background = []
        for state in range(self.size):
            dist = abs(pd.DataFrame(self.strategies).T.pos - state)
            winners = dist[dist == dist.min()].index
            if len(winners) > 1:
                self.background.append(
                    self.ax.axvspan(state - .5, state + .5, facecolor='grey', alpha=self.background_alpha)
                )
            else:
                self.background.append(
                    self.ax.axvspan(state - .5, state + .5, facecolor=self.party_colors[winners[0]], alpha=self.background_alpha)
                )
        
        self.result = self.election_result #(self.ydata, self.strategies)
        self.result_text.set_text(
            'A: {a:.2f}\nB: {b:.2f}'.format(a=self.result['A'], b=self.result['B'])
            )
        self.fig.canvas.draw_idle()


    def update_size(self, val) -> None:
        _old_scale_param = int(log3(self.size))
        _new_scale_param = self.slider_size.val
        self.size = int(3**self.slider_size.val)

        self.slider_str_a.valmax = self.size - 1
        self.slider_str_a.ax.set_xlim(self.slider_str_a.valmin, self.slider_str_a.valmax)

        self.slider_str_b.valmax = self.size - 1
        self.slider_str_b.ax.set_xlim(self.slider_str_b.valmin, self.slider_str_b.valmax)
        
        self.slider_r_a.valmax = self.size - 1
        self.slider_r_a.ax.set_xlim(self.slider_r_a.valmin, self.slider_r_a.valmax)
        
        self.slider_r_b.valmax = self.size - 1
        self.slider_r_b.ax.set_xlim(self.slider_r_b.valmin, self.slider_r_b.valmax)

        self.slider_r_a.set_val(int((3**(val-2) + 1)/2))
        self.slider_r_b.set_val(int((3**(val-2) + 1)/2))
        
        # Down-scaling
        if _old_scale_param > _new_scale_param:
            self.slider_str_a.set_val(
                self.slider_str_a.val // (3**(_old_scale_param - _new_scale_param))
                )
            self.slider_str_b.set_val(
                self.slider_str_b.val // (3**(_old_scale_param - _new_scale_param))
            )
        # Up-scaling
        elif _old_scale_param < _new_scale_param:
            while _old_scale_param < _new_scale_param:
                _old_scale_param += 1
                self.slider_str_a.set_val(
                    self.slider_str_a.val * 3 + 1
                )
                self.slider_str_b.set_val(
                    self.slider_str_b.val * 3 + 1
                )
                        
        self.ax.clear()
        self.plot()


    def show(self) -> None:
        self.fig.show()


def log3(value: int) -> int:
    return np.log(value)/np.log(3)


def normalize_vec(vec: np.ndarray) -> np.ndarray:
    """ Normalize vector that all entries sum up to 1."""
    return vec / np.sum(vec)


# TODO def plot_stationary_sol(sol):


def plot_phasespace(df: pd.DataFrame, path: str = None):
    from matplotlib.cm import get_cmap
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator

    if 'momo' in plt.style.available:
        plt.style.use('momo')

    cm = get_cmap('viridis')
    levels = np.linspace(.0, 1., 11)
    norm = BoundaryNorm(levels, ncolors=cm.N, clip=True)

    fig, ax = plt.subplots()
    im = ax.imshow(df, cmap=cm, norm=norm,)

    _ = fig.colorbar(
        im,
        label=r'Share party A',
        orientation='vertical',
        pad=.05,
        aspect=20,
        shrink=.75,
        ticks=levels,
    )

    ax.set_xlabel(r'Strategy party A')
    ax.set_ylabel(r'Strategy party B')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    if path:
        fig.tight_layout(pad=.2)
        fig.savefig(path)
        plt.close(fig=fig)
    else:
        fig.show()
