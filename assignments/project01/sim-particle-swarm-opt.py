#!/usr/bin/env python3
"""
Perform simulations of the Particle Swarm optimization algorithm.

Simulation output is written to files prefixed by {algorithm}-{test-function}.
The *-meta.json file holds input parameters and summary results.
The *-steps-{dd}.npy holds a numpy array with iteration history for nth trial.

The directory produced by this command is shown below:

sims
├── particle_swarm-bartels_conn-meta.json
├── particle_swarm-bartels_conn-steps-01.npy
├── particle_swarm-bartels_conn-steps-02.npy
├── ...
├── particle_swarm-egg_crate-meta.json
├── particle_swarm-egg_crate-steps-01.npy
├── particle_swarm-egg_crate-steps-02.npy
├── ...
├── particle_swarm-goldstein_price-meta.json
├── particle_swarm-goldstein_price-steps-01.npy
├── particle_swarm-goldstein_price-steps-02.npy
├── ...
├── particle_swarm-rosenbrock-meta.json
├── particle_swarm-rosenbrock-steps-01.npy
├── particle_swarm-rosenbrock-steps-02.npy
├── ...

"""

import json
import os
import time

import numpy as np
from numpy import save
from numpy.random import seed


#
# Particle Swarm Method
#

def particle_swarm(fx, x0s, omega, p1, p2, bounds, niter):
    """
    particle_swarm returns the point xk where fx is minimum

    Parameters
    ----------
    fx : function
        function to minimize
    x0s : numpy.ndarray
        initial positions of particles in swarm
    omega : float
        inertia coefficient
    p1 : float
        momentum coefficient towards min position of current particle
    p2 : float
        momentum coefficient towards min position among all particles
    bounds : numpy.ndarray
        domain boundaries [x1_min, x1_max, ..., xn_min, xn_max]
    niter : int
        number of iterations

    Returns
    -------
    numpy.ndarray
        point xk where fx is minimum
    numpy.ndarray
        current and minimum position and value history for each particle
        [[x_1,0, fx(x_1,0), xk_1,min, fx(xk_1,min),...,
            x_n,0, fx(x_n,0), xk_n,min, fx(xk_n,min)],
         [x_1,1, fx(x_1,1), xk_1,min, fx(xk_1,min),...,
            x_n,1, fx(x_n,1), xk_n,min, fx(xk_n,min)],
    """

    # Initialize swarm with position, velocity, and min position.
    pos = np.copy(x0s)
    x0sdelta = np.max(x0s, axis=0) - np.min(x0s, axis=0)
    vel = (np.random.random(x0s.shape)-0.5)*x0sdelta
    posmin, fxmin = np.copy(x0s), np.apply_along_axis(fx, 1, x0s)

    # Global minimum position.
    xk_min, fxk_min = posmin[np.argmin(fxmin),:], np.min(fxmin)

    # Save position, velocity, and min position by particle to history.
    npart, ndim = x0s.shape[0], x0s.shape[1]
    steps = np.zeros((npart*niter, ndim*3+1))
    steps[:npart,:] = np.hstack((pos, vel, posmin, fxmin[:,np.newaxis]))

    # Perform fixed number of iterations.
    for k in range(1,niter):

        # Compute new velocity of each particle.
        rs = np.random.random((npart,2))
        vel = omega*vel + p1*rs[0]*(posmin-pos) + p2*rs[1]*(xk_min-pos)

        # Update the position of each particle based on velocity.
        pos = pos + vel
        pos = np.clip(pos, a_min=bounds[::2], a_max=bounds[1::2])

        # Evaluate the objective function at each new position.
        fxpart = np.apply_along_axis(fx, 1, pos)

        # If objective function is improved,
        # then replace particle minimum position and value.
        inds = fxpart < fxmin
        posmin[inds,:], fxmin[inds] = pos[inds,:], fxpart[inds]

        # If global objective function is improved,
        # then replace global minimum position and value.
        ind = np.argmin(fxmin)
        if fxmin[ind] < fxk_min:
            xk_min, fxk_min = posmin[ind,:], fxmin[ind]

        # Save particle history.
        ind0 = k*npart
        steps[ind0:ind0+npart,:] = np.hstack((pos, vel, posmin,
                                              fxmin[:,np.newaxis]))

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
        'x0func': params['x0func'].__name__,
        'elapsed_sec': [None]*params['ntrials'],
        'nsteps': [None]*params['ntrials'],
        'x0': [None]*params['ntrials'],
        'f(x0)': [None]*params['ntrials'],
        'xk': [None]*params['ntrials'],
        'f(xk)': [None]*params['ntrials'],
    }
    return meta


def randx0(nx0, **params):
    """Return random initial position x0 based on domain boundaries."""
    ntrials, bounds = params['ntrials'], params['bounds']
    x0s = []  # Each trial holds nx0 particles.
    for _ in range(ntrials):
        particles = np.zeros((nx0, len(bounds)//2))
        for i in range(nx0):
            for j, (xmin,xmax) in enumerate(zip(bounds[0::2],bounds[1::2])):
                particles[i,j] = xmin + 0.8*(xmax-xmin)*np.random.random()
        x0s.append(particles)
    return x0s


def tilex0_particles(func):
    """Returns tiled particles for a single trial."""
    if func in set(('rosenbrock','goldstein_price')):
        x1 = np.array([-1.5,0.0,1.5])
        x2 = np.array([1.8,0.8,-0.8,-1.8])
        x0s = np.transpose([np.tile(x1, len(x2)), np.repeat(x2, len(x1))])
        # Remove points close to the center.
        return x0s[np.invert((x0s[:,0]==0) & (np.abs(x0s[:,1])==0.8))]
    elif func in set(('bartels_conn','egg_crate')):
        x1 = np.array([-3.5,0.0,3.5])
        x2 = np.array([4.,1.5,-1.5,-4.])
        x0s = np.transpose([np.tile(x1, len(x2)), np.repeat(x2, len(x1))])
        # Remove points close to the center.
        return x0s[np.invert((x0s[:,0]==0) & (np.abs(x0s[:,1])==1.5))]
    else:
        raise ValueError('no tiling for function named: {0}', func)


def tilex0(nx0, **params):
    """Return tiled initial position x0 based on test function."""
    ntrials, func = params['ntrials'], params['func']
    x0s = []  # Each trial holds nx0 particles.
    particles = tilex0_particles(func)
    for _ in range(ntrials):
        inds = np.random.choice(range(particles.shape[0]), nx0, replace=False)
        x0s.append(particles[inds])
    return x0s


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


def sim_particle_swarm_rosenbrock(**kwargs):
    """Simulate Gradient Descent on the Rosenbrock function."""
    params = dict(kwargs)
    params.update(func='rosenbrock')
    meta = init_meta(**params)
    meta.update(bounds=[-2.,2.,-2.,2.])
    meta.update(nx0=4)
    meta.update(omega=1.)
    meta.update(p1=1.)
    meta.update(p2=1.)
    meta.update(niter=50)
    meta.update(exp_xkmin=[1.,1.])
    meta.update(exp_fxkmin=0.)

    seed(params['seed'])
    fx = rosenbrock
    omega, p1, p2 = meta['omega'], meta['p1'], meta['p2']
    bounds, niter = meta['bounds'], meta['niter']

    trials = range(1,params['ntrials']+1)
    x0s = params['x0func'](meta['nx0'], **params)

    for ind, (trial,x0) in enumerate(zip(trials,x0s)):
        params.update(trial=trial)
        t0 = time.perf_counter()
        xk, steps = particle_swarm(fx, x0, omega, p1, p2, bounds, niter)
        t1 = time.perf_counter()
        meta['elapsed_sec'][ind] = t1-t0
        meta['nsteps'][ind] = len(steps)
        meta['x0'][ind] = x0.tolist()
        meta['f(x0)'][ind] = np.apply_along_axis(fx, 1, x0).tolist()
        meta['xk'][ind] = xk.tolist()
        meta['f(xk)'][ind] = fx(xk)
        write_savefn(steps, **params)

    write_metafn(meta, **params)


def sim_particle_swarm_goldstein_price(**kwargs):
    """Simulate Particle Swarm on the Goldstein-Price function."""
    params = dict(kwargs)
    params.update(func='goldstein_price')
    meta = init_meta(**params)
    meta.update(bounds=[-2.,2.,-2.,2.])
    meta.update(nx0=4)
    meta.update(omega=1.)
    meta.update(p1=1.)
    meta.update(p2=1.)
    meta.update(niter=500)
    meta.update(exp_xkmin=[0.,-1.])
    meta.update(exp_fxkmin=3.)

    seed(params['seed'])
    fx = goldstein_price
    omega, p1, p2 = meta['omega'], meta['p1'], meta['p2']
    bounds, niter = meta['bounds'], meta['niter']

    trials = range(1,params['ntrials']+1)
    x0s = params['x0func'](meta['nx0'], **params)

    for ind, (trial,x0) in enumerate(zip(trials,x0s)):
        params.update(trial=trial)
        t0 = time.perf_counter()
        xk, steps = particle_swarm(fx, x0, omega, p1, p2, bounds, niter)
        t1 = time.perf_counter()
        meta['elapsed_sec'][ind] = t1-t0
        meta['nsteps'][ind] = len(steps)
        meta['x0'][ind] = x0.tolist()
        meta['f(x0)'][ind] = np.apply_along_axis(fx, 1, x0).tolist()
        meta['xk'][ind] = xk.tolist()
        meta['f(xk)'][ind] = fx(xk)
        write_savefn(steps, **params)

    write_metafn(meta, **params)


def sim_particle_swarm_bartels_conn(**kwargs):
    """Simulate Particle Swarm on the Bartels-Conn function."""
    params = dict(kwargs)
    params.update(func='bartels_conn')
    meta = init_meta(**params)
    meta.update(bounds=[-5.,5.,-5.,5.])
    meta.update(nx0=3)
    meta.update(omega=1.)
    meta.update(p1=1.)
    meta.update(p2=1.)
    meta.update(niter=100)
    meta.update(exp_xkmin=[0.,0.])
    meta.update(exp_fxkmin=1.)

    seed(params['seed'])
    fx = bartels_conn
    omega, p1, p2 = meta['omega'], meta['p1'], meta['p2']
    bounds, niter = meta['bounds'], meta['niter']

    trials = range(1,params['ntrials']+1)
    x0s = params['x0func'](meta['nx0'], **params)

    for ind, (trial,x0) in enumerate(zip(trials,x0s)):
        params.update(trial=trial)
        t0 = time.perf_counter()
        xk, steps = particle_swarm(fx, x0, omega, p1, p2, bounds, niter)
        t1 = time.perf_counter()
        meta['elapsed_sec'][ind] = t1-t0
        meta['nsteps'][ind] = len(steps)
        meta['x0'][ind] = x0.tolist()
        meta['f(x0)'][ind] = np.apply_along_axis(fx, 1, x0).tolist()
        meta['xk'][ind] = xk.tolist()
        meta['f(xk)'][ind] = fx(xk)
        write_savefn(steps, **params)

    write_metafn(meta, **params)


def sim_particle_swarm_egg_crate(**kwargs):
    """Simulate Particle Swarm on the Egg Crate function."""
    params = dict(kwargs)
    params.update(func='egg_crate')
    meta = init_meta(**params)
    meta.update(bounds=[-5.,5.,-5.,5.])
    meta.update(nx0=4)
    meta.update(omega=1.)
    meta.update(p1=1.)
    meta.update(p2=1.)
    meta.update(niter=500)
    meta.update(exp_xkmin=[0.,0.])
    meta.update(exp_fxkmin=0.)

    seed(params['seed'])
    fx = egg_crate
    omega, p1, p2 = meta['omega'], meta['p1'], meta['p2']
    bounds, niter = meta['bounds'], meta['niter']

    trials = range(1,params['ntrials']+1)
    x0s = params['x0func'](meta['nx0'], **params)

    for ind, (trial,x0) in enumerate(zip(trials,x0s)):
        params.update(trial=trial)
        t0 = time.perf_counter()
        xk, steps = particle_swarm(fx, x0, omega, p1, p2, bounds, niter)
        t1 = time.perf_counter()
        meta['elapsed_sec'][ind] = t1-t0
        meta['nsteps'][ind] = len(steps)
        meta['x0'][ind] = x0.tolist()
        meta['f(x0)'][ind] = np.apply_along_axis(fx, 1, x0).tolist()
        meta['xk'][ind] = xk.tolist()
        meta['f(xk)'][ind] = fx(xk)
        write_savefn(steps, **params)

    write_metafn(meta, **params)


def sim_particle_swarm(**kwargs):
    """Run simulations using Particle Swarm on each test function."""
    os.makedirs(kwargs['base_dirn'], exist_ok=True)
    os.chmod(kwargs['base_dirn'], 0o755)
    sim_particle_swarm_rosenbrock(**kwargs)
    sim_particle_swarm_goldstein_price(**kwargs)
    sim_particle_swarm_bartels_conn(**kwargs)
    sim_particle_swarm_egg_crate(**kwargs)


if __name__ == '__main__':
    opts = {
        'alg': 'particle_swarm',
        'ntrials': 12,
        'x0func': tilex0,
        'seed': 8517,
        'base_dirn': './sims/',
        'savefn_fmt': '{alg}-{func}-steps-{trial:02d}.npy',
        'metafn_fmt': '{alg}-{func}-meta.json',
    }
    sim_particle_swarm(**opts)
