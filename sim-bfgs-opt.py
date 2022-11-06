#!/usr/bin/env python3
"""
Perform simulations of the BFGS optimization algorithm.

Simulation output is written to files prefixed by {algorithm}-{test-function}.
The *-meta.json file holds input parameters and summary results.
The *-steps-{dd}.npy holds a numpy array with iteration history for nth trial.

The directory produced by this command is shown below:

sims
├── bfgs-goldstein_price-meta.json
├── bfgs-goldstein_price-steps-01.npy
├── bfgs-goldstein_price-steps-02.npy
├── ...
├── bfgs-rosenbrock-meta.json
├── bfgs-rosenbrock-steps-01.npy
├── bfgs-rosenbrock-steps-02.npy
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

import scipy.optimize as opt


#
# BFGS Method
#

def bfgs(fx, gradfx, x0, tol, maxiter):
    """
    bfgs returns the point xk where fx is minimum

    Parameters
    ----------
    fx : function
        function to minimize
    gradfx : function
        gradient of function to minimize
    x0 : numpy.ndarray
        initial guess for xk
    tol : float
        convergence threshold
    maxiter : int
        maximum number of iterations

    Returns
    -------
    numpy.ndarray
        vector xk where fx is minimum
    numpy.ndarray
        position and value history
        [[x0, fx(x0), gradfx(x0)],
         [x1, fx(x1), gradfx(x1)],...]
    """

    xk, gradfxk, Bk = x0, gradfx(x0), np.eye(x0.size)

    # Save current and minimum position and value to history.
    steps = np.zeros((maxiter, (x0.size*2)+1))
    steps[0,:] = np.hstack((x0, fx(x0), gradfxk))

    # Repeat up to maximum number of iterations.
    for k in range(1,maxiter):

        # Stop iteration when gradient is near zero.
        if np.linalg.norm(gradfxk) < tol:
            steps = steps[:-(maxiter-k),:]
            break

        # Solve Bk*sk = -grad(xk) for sk.
        sk = np.linalg.solve(Bk, -1. * gradfxk)

        # Update xk and evaluate gradient at new value of xk.
        xk = xk + sk
        gradfxk1 = gradfx(xk)

        # Compute difference in gradients.
        yk = gradfxk1 - gradfxk

        # Update approximate Hessian.
        term1 = np.outer(yk, yk.T) / np.dot(yk.T, sk)
        term2a = np.dot(np.dot(Bk, np.outer(sk, sk.T)), Bk)
        term2b = np.dot(np.dot(sk.T, Bk), sk)
        Bk = Bk + term1 - (term2a / term2b)

        # Update the gradient at xk.
        gradfxk = gradfxk1

        # Save iteration history.
        steps[k,:] = np.hstack((xk, fx(xk), gradfxk))

    return xk, steps


# NOTE(mmorais): bfgs is failing on line search, use scipy instead.
def scipy_bfgs(fx, gradfx, x0, tol, maxiter):
    """
    scipy_bfgs wraps scipy implementation of bfgs
    """
    # Save current and minimum position and value to history.
    steps = np.zeros((maxiter+1, (x0.size*2)+1))
    steps[0,:] = np.hstack((x0, fx(x0), gradfx(x0)))
    def make_callback():
        k = 1
        def callback(xk):
            nonlocal k
            # Save iteration history.
            steps[k,:] = np.hstack((xk, fx(xk), gradfx(xk)))
            k = k + 1
        return callback

    # Invoke scipy minimize with BFGS option.
    res = opt.minimize(fx, x0, method='BFGS', jac=gradfx, tol=tol,
                       options={'maxiter': maxiter}, callback=make_callback())

    # Copy OptimizeResult to equivalent returned from bfgs.
    xk = res.x
    if res.nit < maxiter:
        steps = steps[:-(maxiter-res.nit),:]
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
        'x0func': params['x0func'].__name__,
        'elapsed_sec': [None]*params['ntrials'],
        'nsteps': [None]*params['ntrials'],
        'x0': [None]*params['ntrials'],
        'f(x0)': [None]*params['ntrials'],
        'xk': [None]*params['ntrials'],
        'f(xk)': [None]*params['ntrials'],
    }
    return meta


def randx0(**params):
    """Return random initial position x0 based on domain boundaries."""
    ntrials, bounds = params['ntrials'], params['bounds']
    x0s = np.zeros((ntrials, len(bounds)//2))
    for i in range(ntrials):
        for j, (xmin,xmax) in enumerate(zip(bounds[0::2],bounds[1::2])):
            x0s[i,j] = xmin + 0.8*(xmax-xmin)*np.random.random()
    return x0s


def tilex0(**params):
    """Return tiled initial position x0 based on test function."""
    func = params['func']
    if func in set(('rosenbrock','goldstein_price')):
        x1 = np.array([-1.5,0.0,1.5])
        x2 = np.array([1.8,0.8,-0.8,-1.8])
        x0s = np.transpose([np.tile(x1, len(x2)), np.repeat(x2, len(x1))])
        return x0s
    raise ValueError('no tiling for function named: {0}', func)


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


def sim_bfgs_rosenbrock(**kwargs):
    """Simulate BFGS on the Rosenbrock function."""
    params = dict(kwargs)
    params.update(func='rosenbrock')
    meta = init_meta(**params)
    meta.update(bounds=[-2.,2.,-2.,2.])
    meta.update(tol=1e-2)
    meta.update(maxiter=1000)
    meta.update(exp_xkmin=[1.,1.])
    meta.update(exp_fxkmin=0.)

    seed(params['seed'])
    fx, gradfx = rosenbrock, grad(rosenbrock)
    tol, maxiter = meta['tol'], meta['maxiter']

    trials = range(1,params['ntrials']+1)
    x0s = params['x0func'](**params)

    for ind, (trial,x0) in enumerate(zip(trials,x0s)):
        params.update(trial=trial)
        t0 = time.perf_counter()
        xk, steps = scipy_bfgs(fx, gradfx, x0, tol, maxiter)
        t1 = time.perf_counter()
        meta['elapsed_sec'][ind] = t1-t0
        meta['nsteps'][ind] = len(steps)
        meta['x0'][ind] = x0.tolist()
        meta['f(x0)'][ind] = fx(x0)
        meta['xk'][ind] = xk.tolist()
        meta['f(xk)'][ind] = fx(xk)
        write_savefn(steps, **params)

    write_metafn(meta, **params)


def sim_bfgs_goldstein_price(**kwargs):
    """Simulate BFGS on the Goldstein-Price function."""
    params = dict(kwargs)
    params.update(func='goldstein_price')
    meta = init_meta(**params)
    meta.update(bounds=[-2.,2.,-2.,2.])
    meta.update(tol=1e-2)
    meta.update(maxiter=1000)
    meta.update(exp_xkmin=[0.,-1.])
    meta.update(exp_fxkmin=3.)

    seed(params['seed'])
    fx, gradfx = goldstein_price, grad(goldstein_price)
    tol, maxiter = meta['tol'], meta['maxiter']

    trials = range(1,params['ntrials']+1)
    x0s = params['x0func'](**params)

    for ind, (trial,x0) in enumerate(zip(trials,x0s)):
        params.update(trial=trial)
        t0 = time.perf_counter()
        xk, steps = scipy_bfgs(fx, gradfx, x0, tol, maxiter)
        t1 = time.perf_counter()
        meta['elapsed_sec'][ind] = t1-t0
        meta['nsteps'][ind] = len(steps)
        meta['x0'][ind] = x0.tolist()
        meta['f(x0)'][ind] = fx(x0)
        meta['xk'][ind] = xk.tolist()
        meta['f(xk)'][ind] = fx(xk)
        write_savefn(steps, **params)

    write_metafn(meta, **params)


def sim_bfgs(**kwargs):
    """Run simulations using BFGS on each test function."""
    os.makedirs(kwargs['base_dirn'], exist_ok=True)
    os.chmod(kwargs['base_dirn'], 0o755)
    sim_bfgs_rosenbrock(**kwargs)
    sim_bfgs_goldstein_price(**kwargs)


if __name__ == '__main__':
    opts = {
        'alg': 'bfgs',
        'ntrials': 12,
        'x0func': tilex0,
        'seed': 8517,
        'base_dirn': './sims/',
        'savefn_fmt': '{alg}-{func}-steps-{trial:02d}.npy',
        'metafn_fmt': '{alg}-{func}-meta.json',
    }
    sim_bfgs(**opts)
