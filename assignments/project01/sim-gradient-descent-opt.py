#!/usr/bin/env python3
"""
Perform simulations of the Gradient Descent optimization algorithm.

Simulation output is written to files prefixed by {algorithm}-{test-function}.
The *-meta.json file holds input parameters and summary results.
The *-steps-{dd}.npy holds a numpy array with iteration history for nth trial.

The directory produced by this command is shown below:

sims
├── gradient_descent-goldstein_price-meta.json
├── gradient_descent-goldstein_price-steps-01.npy
├── gradient_descent-goldstein_price-steps-02.npy
├── ...
├── gradient_descent-rosenbrock-meta.json
├── gradient_descent-rosenbrock-steps-01.npy
├── gradient_descent-rosenbrock-steps-02.npy
├── ...

"""

import json
import os
import time

# The numpy interface of autograd wraps all numpy ops with autodiff.
import autograd.numpy as np

from autograd import grad
from numpy import save
from numpy.random import seed


#
# Gradient Descent Method
#

def gradient_descent(fx, gradfx, x0, alpha, tol, maxiter):
    """
    gradient_descent returns the point xk where fx is minimum

    Parameters
    ----------
    fx : function
        function to minimize
    gradfx : function
        gradient of function to minimize
    x0 : numpy.ndarray
        initial guess for xk
    alpha : float
        learning rate
    tol : float
        convergence threshold
    maxiter : int
        maximum number of iterations

    Returns
    -------
    numpy.ndarray
        point xk where fx is minimum
    numpy.ndarray
        position and value history
        [[x0, fx(x0), gradfx(x0)],
         [x1, fx(x1), gradfx(x1)],...]
    """

    xk, fxk, gradfxk = x0, fx(x0), gradfx(x0)

    # Save current and minimum position and value to history.
    steps = np.zeros((maxiter, (x0.size*2)+1))
    steps[0,:] = np.hstack((x0, fxk, gradfxk))

    # Repeat up to maximum number of iterations.
    for k in range(1,maxiter):

        # Stop iteration when gradient is near zero.
        if np.linalg.norm(gradfxk) < tol:
            steps = steps[:-(maxiter-k),:]
            break

        # Update xk based on product of learning rate and gradient.
        xk = xk - alpha * gradfxk

        # Evaluate gradient at new value of xk.
        gradfxk = gradfx(xk)

        # Evaluate the function at new value of xk.
        fxk = fx(xk)

        # Save iteration history.
        steps[k,:] = np.hstack((xk, fxk, gradfxk))

    return xk, steps


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


def sim_gradient_descent_rosenbrock(**kwargs):
    """Simulate Gradient Descent on the Rosenbrock function."""
    params = dict(kwargs)
    params.update(func='rosenbrock')
    meta = init_meta(**params)
    meta.update(bounds=[-2.,2.,-2.,2.])
    meta.update(alpha=1e-3)
    meta.update(tol=1e-2)
    meta.update(maxiter=20000)

    seed(params['seed'])
    fx, gradfx = rosenbrock, grad(rosenbrock)
    alpha, tol, maxiter = meta['alpha'], meta['tol'], meta['maxiter']

    for ind, trial in enumerate(range(1,params['ntrials']+1)):
        params.update(trial=trial)
        x0 = randx0(meta['bounds'])
        t0 = time.perf_counter()
        xk, steps = gradient_descent(fx, gradfx, x0, alpha, tol, maxiter)
        t1 = time.perf_counter()
        meta['elapsed_sec'][ind] = t1-t0
        meta['nsteps'][ind] = len(steps)
        meta['x0'][ind] = x0.tolist()
        meta['f(x0)'][ind] = fx(x0)
        meta['xk'][ind] = xk.tolist()
        meta['f(xk)'][ind] = fx(xk)
        write_savefn(steps, **params)

    write_metafn(meta, **params)


def sim_gradient_descent_goldstein_price(**kwargs):
    """Simulate Gradient Descent on the Goldstein-Price function."""
    params = dict(kwargs)
    params.update(func='goldstein_price')
    meta = init_meta(**params)
    meta.update(bounds=[-2.,2.,-2.,2.])
    meta.update(alpha=1e-5)
    meta.update(tol=1e-2)
    meta.update(maxiter=20000)

    seed(params['seed'])
    fx, gradfx = goldstein_price, grad(goldstein_price)
    alpha, tol, maxiter = meta['alpha'], meta['tol'], meta['maxiter']

    for ind, trial in enumerate(range(1,params['ntrials']+1)):
        params.update(trial=trial)
        x0 = randx0(meta['bounds'])
        t0 = time.perf_counter()
        xk, steps = gradient_descent(fx, gradfx, x0, alpha, tol, maxiter)
        t1 = time.perf_counter()
        meta['elapsed_sec'][ind] = t1-t0
        meta['nsteps'][ind] = len(steps)
        meta['x0'][ind] = x0.tolist()
        meta['f(x0)'][ind] = fx(x0)
        meta['xk'][ind] = xk.tolist()
        meta['f(xk)'][ind] = fx(xk)
        write_savefn(steps, **params)

    write_metafn(meta, **params)


def sim_gradient_descent(**kwargs):
    """Run simulations using Gradient Descent on each test function."""
    os.makedirs(kwargs['base_dirn'], exist_ok=True)
    os.chmod(kwargs['base_dirn'], 0o755)
    sim_gradient_descent_rosenbrock(**kwargs)
    sim_gradient_descent_goldstein_price(**kwargs)


if __name__ == '__main__':
    opts = {
        'alg': 'gradient_descent',
        'ntrials': 10,
        'seed': 8517,
        'base_dirn': './sims/',
        'savefn_fmt': '{alg}-{func}-steps-{trial:02d}.npy',
        'metafn_fmt': '{alg}-{func}-meta.json',
    }
    sim_gradient_descent(**opts)
