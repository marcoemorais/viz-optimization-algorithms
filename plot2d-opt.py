#!/usr/bin/env python3
"""
Create visualizations of optimization algorithms solutions in 2d.
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


#
# Test Functions
#

def rosenbrock(x):
    """
    rosenbrock evaluates Rosenbrock function at vector x

    Parameters
    ----------
    x : array
        x is a D-dimensional vector, [x1, x2, ..., xD]

    Returns
    -------
    float
        scalar result
    """
    D = len(x)
    i, iplus1 = np.arange(0,D-1), np.arange(1,D)
    return np.sum(100*(x[iplus1] - x[i]**2)**2 + (1-x[i])**2)


def goldstein_price(x):
    """
    goldstein_price evaluates Goldstein-Price function at vector x

    Parameters
    ----------
    x : array
        x is a 2-dimensional vector, [x1, x2]

    Returns
    -------
    float
        scalar result
    """
    a = (x[0] + x[1] + 1)**2
    b = 19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2
    c = (2*x[0] - 3*x[1])**2
    d = 18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2
    return (1. + a*b) * (30. + c*d)


def bartels_conn(x):
    """
    bartels_conn evaluates Bartels-Conn function at vector x

    Parameters
    ----------
    x : array
        x is a 2-dimensional vector, [x1, x2]

    Returns
    -------
    float
        scalar result
    """
    a = np.abs(x[0]**2 + x[1]**2 + x[0]*x[1])
    b = np.abs(np.sin(x[0]))
    c = np.abs(np.cos(x[1]))
    return a + b +c


def egg_crate(x):
    """
    egg_crate evaluates Egg Crate function at vector x

    Parameters
    ----------
    x : array
        x is a 2-dimensional vector, [x1, x2]

    Returns
    -------
    float
        scalar result
    """
    return x[0]**2 + x[1]**2 + 25.*(np.sin(x[0])**2 + np.sin(x[1])**2)


#
# Surface Generation
#

def surface(fx, start=-30, stop=30, num=60):
    """
    surface evaluates fx at regularly spaced grid of points

    Parameters
    ----------
    fx : func
        fx is a vector valued function that returns a scalar result
    start : float
        lower bound of the coordinate grid
    stop : float
        upper bound of the coordinate grid
    num : int
        number of points along one dimension of the grid

    Returns
    -------
    array
        2D array formed by evaluating fx at each grid point
    """
    x = np.linspace(start=start, stop=stop, num=num)
    x1, x2 = np.meshgrid(x, x, indexing='ij')
    X = np.vstack((x1.ravel(), x2.ravel()))
    z = np.apply_along_axis(fx, 0, X).reshape(num,num)
    return x1, x2, z


#
# Solution Results
#

def load_steps(**params):
    """Return solution steps based on simulation properties."""
    savefn = os.path.join(params['base_dirn'],
                          params['savefn_fmt'].format(**params))
    return np.load(savefn)


def load_meta(**params):
    """Return metafile based on simulation properties."""
    metafn = os.path.join(params['base_dirn'],
                          params['metafn_fmt'].format(**params))
    return json.load(open(metafn, 'r'))


def plot2d_solutions(**params):
    """
    plot2d_solutions creates 2d solution plot from simulation results
    """
    algstr = params['alg'].replace('_',' ').title()
    funcstr = params['func'].replace('_',' ').title()
    ngridpts = params.get('ngridpts', 500)
    bounds = params['bounds']
    trial = params['trial']  # Single trial only.
    xkmind = params.get('xkmind', slice(2))
    color = params.get('color', 'darkorange')
    ticker_locator = params.get('ticker_locator', 'LinearLocator')
    colorbar_label = params.get('colorbar_label', 'z')
    show_legend = params.get('show_legend', True)

    # Imbue title with simulation meta information.
    meta = load_meta(**params)
    expmin, expxkmin = meta['exp_fxkmin'], meta['exp_xkmin']
    expminstr = 'abs $\\min(f)$={0:.0f}'.format(expmin)
    algstr = algstr if len(algstr) > 4 else algstr.upper()
    algmeta = [('nx0','n'),('T0','$T_0$'),
               ('alpha','$\\alpha$'),('tol','tol')]
    algmetastr = ' '.join(['{0}={1}'.format(n2, meta[n1])
                           for n1, n2 in algmeta if n1 in meta])
    nitstr = 'nit={0:d}'.format(meta['nsteps'][trial-1])
    minfx = meta['f(xk)'][trial-1]
    minfmt = '.2e' if minfx < 1e-1 else '.1f'
    minstr = '$\\min(f)$={0:{1}}'.format(minfx, minfmt)
    metastrs = [algstr, minstr, nitstr, algmetastr]
    titlestr = ' '.join([s for s in metastrs if len(s) > 0])
    suptitlestr = 'Solution Trajectories: {0} Function'.format(funcstr)

    # Generate surface for filled contour plot.
    fx = globals()[params['func']]
    start, stop = np.min(bounds[::2]), np.max(bounds[1::2])
    x1, x2, z = surface(fx, start, stop, ngridpts)

    fig = plt.figure(figsize=(8,6))

    # Plot 2d filled contour.
    locator = getattr(ticker, ticker_locator)
    cs = plt.contourf(x1, x2, z, locator=locator(), cmap='viridis_r',
                      alpha=0.7)

    # Plot expected minimum.
    plt.scatter(expxkmin[0], expxkmin[1], marker='D', c='red', s=30,
                label=expminstr)

    # Plot initial point.
    x0 = np.array(meta['x0'][trial-1]).reshape(-1,2)
    plt.scatter(x0[:,0], x0[:,1], marker='X', c='dodgerblue', s=30,
                label='$x_0$')

    # Plot solution trajectory.
    steps = load_steps(**params)
    nx0 = meta.get('nx0', 1)  # Multiple particles?
    xks = steps[:,xkmind]
    xks = np.clip(xks, a_min=bounds[::2], a_max=bounds[1::2])
    nxks = 0 if np.isnan(xks).any() else len(xks)//nx0
    for p in range(nx0):
        p0, pN, pstep = p, nxks, nx0
        plt.plot(xks[p0:pN:pstep,0], xks[p0:pN:pstep,1],
                 marker='.', ms=5, markevery=0.25,
                 ls='-', lw=1, c=color,
                 label='$x_k$, trial={:d}'.format(trial))

    plt.suptitle(suptitlestr)
    plt.title(titlestr)
    plt.xlabel('x1')
    plt.xlim(bounds[:2])
    plt.ylabel('x2')
    plt.ylim(bounds[2:])
    plt.colorbar(cs, label=colorbar_label)
    if show_legend:
        plt.legend()
    if params.get('plot2dfn_fmt') is not None:
        imgn = params['plot2dfn_fmt'].format(**params)
        plotfn = os.path.join(params['base_dirn'], imgn)
        plt.savefig(plotfn)
    else:
        plt.show()
    plt.close(fig)


#
# Solution Plots
#

def plot2d(**kwargs):
    """Plot simulation results in 2d."""
    params = [
        {
            'alg': 'gradient_descent',
            'func': 'rosenbrock',
            'bounds': [-2.,2.,-2.,2.],
            'ticker_locator': 'LogLocator',
            'colorbar_label': 'log(z)',
        },
        {
            'alg': 'bfgs',
            'func': 'rosenbrock',
            'bounds': [-2.,2.,-2.,2.],
            'ticker_locator': 'LogLocator',
            'colorbar_label': 'log(z)',
        },
        {
            'alg': 'simulated_annealing',
            'func': 'rosenbrock',
            'bounds': [-2.,2.,-2.,2.],
            'xkmind': slice(3,5),
            'ticker_locator': 'LogLocator',
            'colorbar_label': 'log(z)',
        },
        {
            'alg': 'particle_swarm',
            'func': 'rosenbrock',
            'bounds': [-2.,2.,-2.,2.],
            'xkmind': slice(4,6),
            'ticker_locator': 'LogLocator',
            'colorbar_label': 'log(z)',
            'color': None,
        },
        {
            'alg': 'gradient_descent',
            'func': 'goldstein_price',
            'bounds': [-2.,2.,-2.,2.],
            'ticker_locator': 'LogLocator',
            'colorbar_label': 'log(z)',
        },
        {
            'alg': 'bfgs',
            'func': 'goldstein_price',
            'bounds': [-2.,2.,-2.,2.],
            'ticker_locator': 'LogLocator',
            'colorbar_label': 'log(z)',
        },
        {
            'alg': 'simulated_annealing',
            'func': 'goldstein_price',
            'bounds': [-2.,2.,-2.,2.],
            'xkmind': slice(3,5),
            'ticker_locator': 'LogLocator',
            'colorbar_label': 'log(z)',
        },
        {
            'alg': 'particle_swarm',
            'func': 'goldstein_price',
            'bounds': [-2.,2.,-2.,2.],
            'xkmind': slice(4,6),
            'ticker_locator': 'LogLocator',
            'colorbar_label': 'log(z)',
            'color': None,
        },
        {
            'alg': 'simulated_annealing',
            'func': 'bartels_conn',
            'bounds': [-5.,5.,-5.,5.],
            'xkmind': slice(3,5),
        },
        {
            'alg': 'particle_swarm',
            'func': 'bartels_conn',
            'bounds': [-5.,5.,-5.,5.],
            'xkmind': slice(4,6),
            'color': None,
        },
        {
            'alg': 'simulated_annealing',
            'func': 'egg_crate',
            'bounds': [-5.,5.,-5.,5.],
            'xkmind': slice(3,5),
        },
        {
            'alg': 'particle_swarm',
            'func': 'egg_crate',
            'bounds': [-5.,5.,-5.,5.],
            'xkmind': slice(4,6),
            'color': None,
        }
    ]
    # One set of parameters for each algo-func combination.
    for param in params:
        param.update(kwargs)
        # Create one-plot-per-trial.
        for trial in range(1,param['ntrials']+1):
            param.update(trial=trial)
            plot2d_solutions(**param)


if __name__ == '__main__':
    opts = {
        'ntrials': 12,
        'base_dirn': './sims/',
        'savefn_fmt': '{alg}-{func}-steps-{trial:02d}.npy',
        'metafn_fmt': '{alg}-{func}-meta.json',
        'plot2dfn_fmt': '{alg}-{func}-plot2d-{trial:02d}.png',
    }
    plot2d(**opts)
