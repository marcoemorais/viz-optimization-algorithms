#!/usr/bin/env python3
"""
Perform simulations of the Simulated Annealing optimization algorithm.

Simulation output is written to files prefixed by {algorithm}-{test-function}.
The *-meta.json file holds input parameters and summary results.
The *-steps-{dd}.npy holds a numpy array with iteration history for nth trial.

The directory produced by this command is shown below:

sims
├── simulated_annealing-bartels_conn-meta.json
├── simulated_annealing-bartels_conn-steps-01.npy
├── simulated_annealing-bartels_conn-steps-02.npy
├── ...
├── simulated_annealing-egg_crate-meta.json
├── simulated_annealing-egg_crate-steps-01.npy
├── simulated_annealing-egg_crate-steps-02.npy
├── ...
├── simulated_annealing-goldstein_price-meta.json
├── simulated_annealing-goldstein_price-steps-01.npy
├── simulated_annealing-goldstein_price-steps-02.npy
├── ...
├── simulated_annealing-rosenbrock-meta.json
├── simulated_annealing-rosenbrock-steps-01.npy
├── simulated_annealing-rosenbrock-steps-02.npy
├── ...

"""

import json
import os
import time

from functools import partial

import numpy as np
from numpy import save
from numpy.random import seed


#
# Simulated Annealing Method
#

def simulated_annealing(fx, x0, mean, cov, tk, niter):
    """
    simulated_annealing returns the point xk where fx is minimum

    Parameters
    ----------
    fx : function
        function to minimize
    x0 : numpy.ndarray
        initial guess for xk
    mean : numpy.ndarray
        means of multivariate normal transition distribution
    cov : numpy.ndarray
        covariance of multivariate normal transition distribution
    tk : function
        annealing schedule as a function of iteration number
    niter : int
        number of iterations

    Returns
    -------
    numpy.ndarray
        point xk where fx is minimum
    numpy.ndarray
        current and minimum position and value history
        [[x0, fx(x0), xk_min, fx(xk_min)],
         [x1, fx(x1), xk_min, fx(xk_min)],...]
    """

    # Initialize solution at x0.
    xk, fxk = x0, fx(x0)
    xk_min, fxk_min = xk, fxk

    # Setup random transition distribution.
    mvnorm = partial(np.random.multivariate_normal, mean, cov)

    # Save current and minimum position and value to history.
    steps = np.zeros((niter, (x0.size+1)*2))
    steps[0,:] = np.hstack((x0, fxk, xk_min, fxk_min))

    # Perform fixed number of iterations.
    for k in range(1,niter):

        # Generate a new random point.
        xk1 = xk + mvnorm()

        # Evaluate the function at the new point.
        fxk1 = fx(xk1)

        # Compute the change in the objective function.
        delta_fxk = fxk1 - fxk

        # If objective function is improved or escape current position,
        # then update xk, fxk with the new position.
        if delta_fxk < 0. or np.random.random() < np.exp(-fxk1/tk(k)):
            xk, fxk = xk1, fxk1
            if fxk1 < fxk_min:
                xk_min, fxk_min = xk1, fxk1

        # Save iteration history.
        steps[k,:] = np.hstack((xk1, fxk1, xk_min, fxk_min))

    return xk_min, steps


#
# Test Function: Rosenbrock Function
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


#
# Test Function: Goldstein-Price Function
#

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


#
# Test Function: Bartels-Conn Function
#

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


#
# Test Function: Egg Crate Function
#

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
# Simulation functions.
#

def init_meta(**params):
    """Initialize simulation metadata with common properties."""
    meta = {
        'alg': params['alg'],
        'func': params['func'],
        'seed': params['seed'],
        'ntrials': params['ntrials'],
        'elapsed_sec': [None]*params['ntrials'],
        'nsteps': [None]*params['ntrials'],
        'x0': [None]*params['ntrials'],
        'f(x0)': [None]*params['ntrials'],
        'xk': [None]*params['ntrials'],
        'f(xk)': [None]*params['ntrials'],
    }
    return meta


def randx0(bounds):
    """Return random initial position x0 based on domain boundaries."""
    x0 = np.zeros(len(bounds)//2)
    for ind, (xmin,xmax) in enumerate(zip(bounds[0::2],bounds[1::2])):
        x0[ind] = xmin + 0.8*(xmax-xmin)*np.random.random()
    return x0


def write_savefn(steps, **params):
    """Write the simulation save file."""
    savefn = os.path.join(params['base_dirn'],
                          params['savefn_fmt'].format(**params))
    save(savefn, steps)
    os.chmod(savefn, 0o444)


def write_metafn(meta, **params):
    """Write the simulation metadata file."""
    metafn = os.path.join(params['base_dirn'],
                          params['metafn_fmt'].format(**params))
    json.dump(meta, open(metafn, 'w'))


def sim_simulated_annealing_rosenbrock(**kwargs):
    """Simulate Gradient Descent on the Rosenbrock function."""
    params = dict(kwargs)
    params.update(func='rosenbrock')
    meta = init_meta(**params)
    meta.update(bounds=[-2.,2.,-2.,2.])
    meta.update(mean=[1.,1.])
    meta.update(cov=[[1.,0.],[0.,1.]])
    meta.update(T0=1.)
    meta.update(niter=20000)

    seed(params['seed'])
    fx = rosenbrock
    mean, cov = np.array(meta['mean']), np.array(meta['cov'])
    tk = lambda k: meta['T0']/k
    niter = meta['niter']

    for ind, trial in enumerate(range(1,params['ntrials']+1)):
        params.update(trial=trial)
        x0 = randx0(meta['bounds'])
        t0 = time.perf_counter()
        xk, steps = simulated_annealing(fx, x0, mean, cov, tk, niter)
        t1 = time.perf_counter()
        meta['elapsed_sec'][ind] = t1-t0
        meta['nsteps'][ind] = len(steps)
        meta['x0'][ind] = x0.tolist()
        meta['f(x0)'][ind] = fx(x0)
        meta['xk'][ind] = xk.tolist()
        meta['f(xk)'][ind] = fx(xk)
        write_savefn(steps, **params)

    write_metafn(meta, **params)


def sim_simulated_annealing_goldstein_price(**kwargs):
    """Simulate Simulated Annealing on the Goldstein-Price function."""
    params = dict(kwargs)
    params.update(func='goldstein_price')
    meta = init_meta(**params)
    meta.update(bounds=[-2.,2.,-2.,2.])
    meta.update(mean=[1.,1.])
    meta.update(cov=[[1.,0.],[0.,1.]])
    meta.update(T0=1.)
    meta.update(niter=20000)

    seed(params['seed'])
    fx = goldstein_price
    mean, cov = np.array(meta['mean']), np.array(meta['cov'])
    tk = lambda k: meta['T0']/k
    niter = meta['niter']

    for ind, trial in enumerate(range(1,params['ntrials']+1)):
        params.update(trial=trial)
        x0 = randx0(meta['bounds'])
        t0 = time.perf_counter()
        xk, steps = simulated_annealing(fx, x0, mean, cov, tk, niter)
        t1 = time.perf_counter()
        meta['elapsed_sec'][ind] = t1-t0
        meta['nsteps'][ind] = len(steps)
        meta['x0'][ind] = x0.tolist()
        meta['f(x0)'][ind] = fx(x0)
        meta['xk'][ind] = xk.tolist()
        meta['f(xk)'][ind] = fx(xk)
        write_savefn(steps, **params)

    write_metafn(meta, **params)


def sim_simulated_annealing_bartels_conn(**kwargs):
    """Simulate Simulated Annealing on the Bartels-Conn function."""
    params = dict(kwargs)
    params.update(func='bartels_conn')
    meta = init_meta(**params)
    meta.update(bounds=[-5.,5.,-5.,5.])
    meta.update(mean=[1.,1.])
    meta.update(cov=[[1.,0.],[0.,1.]])
    meta.update(T0=1.)
    meta.update(niter=20000)

    seed(params['seed'])
    fx = bartels_conn
    mean, cov = np.array(meta['mean']), np.array(meta['cov'])
    tk = lambda k: meta['T0']/k
    niter = meta['niter']

    for ind, trial in enumerate(range(1,params['ntrials']+1)):
        params.update(trial=trial)
        x0 = randx0(meta['bounds'])
        t0 = time.perf_counter()
        xk, steps = simulated_annealing(fx, x0, mean, cov, tk, niter)
        t1 = time.perf_counter()
        meta['elapsed_sec'][ind] = t1-t0
        meta['nsteps'][ind] = len(steps)
        meta['x0'][ind] = x0.tolist()
        meta['f(x0)'][ind] = fx(x0)
        meta['xk'][ind] = xk.tolist()
        meta['f(xk)'][ind] = fx(xk)
        write_savefn(steps, **params)

    write_metafn(meta, **params)


def sim_simulated_annealing_egg_crate(**kwargs):
    """Simulate Simulated Annealing on the Egg Crate function."""
    params = dict(kwargs)
    params.update(func='egg_crate')
    meta = init_meta(**params)
    meta.update(bounds=[-5.,5.,-5.,5.])
    meta.update(mean=[1.,1.])
    meta.update(cov=[[1.,0.],[0.,1.]])
    meta.update(T0=1.)
    meta.update(niter=30000)

    seed(params['seed'])
    fx = egg_crate
    mean, cov = np.array(meta['mean']), np.array(meta['cov'])
    tk = lambda k: meta['T0']/k
    niter = meta['niter']

    for ind, trial in enumerate(range(1,params['ntrials']+1)):
        params.update(trial=trial)
        x0 = randx0(meta['bounds'])
        t0 = time.perf_counter()
        xk, steps = simulated_annealing(fx, x0, mean, cov, tk, niter)
        t1 = time.perf_counter()
        meta['elapsed_sec'][ind] = t1-t0
        meta['nsteps'][ind] = len(steps)
        meta['x0'][ind] = x0.tolist()
        meta['f(x0)'][ind] = fx(x0)
        meta['xk'][ind] = xk.tolist()
        meta['f(xk)'][ind] = fx(xk)
        write_savefn(steps, **params)

    write_metafn(meta, **params)


def sim_simulated_annealing(**kwargs):
    """Run simulations using Simulated Annealing on each test function."""
    os.makedirs(kwargs['base_dirn'], exist_ok=True)
    os.chmod(kwargs['base_dirn'], 0o755)
    sim_simulated_annealing_rosenbrock(**kwargs)
    sim_simulated_annealing_goldstein_price(**kwargs)
    sim_simulated_annealing_bartels_conn(**kwargs)
    sim_simulated_annealing_egg_crate(**kwargs)


if __name__ == '__main__':
    opts = {
        'alg': 'simulated_annealing',
        'ntrials': 10,
        'seed': 8517,
        'base_dirn': './sims/',
        'savefn_fmt': '{alg}-{func}-steps-{trial:02d}.npy',
        'metafn_fmt': '{alg}-{func}-meta.json',
    }
    sim_simulated_annealing(**opts)
