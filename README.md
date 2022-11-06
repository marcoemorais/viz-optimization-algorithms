# viz-optimization-algorithms

Jupyter notebooks, scripts, and results associated with the paper _Visualization of Optimization Algorithms_ by Marco Morais (Morais, 2020).

Paper is available as [tex](Optimization-Visualization-mmorais2.tex) and [pdf](Optimization-Visualization-mmorais2.pdf).

I completed this project as the end of semester assignment for [CS 519 Scientific Visualization](https://uiucmcs.org/courses/CS-519-Scientific-Visualization) at UIUC.

If you find this repo helpful, please cite this paper and star this repository. Thank you!

```tex
@article{morais2020,
  title={Visualization of Optimization Algorithms},
  author={Morais, Marco},
  year={2020}
}
```

## Abstract
```
Optimization algorithms seek to find the best solution $x^{\*}$ from a set S such that $f(x^{\*}) \\leq f(x)$ for all $x$ in S. For this project we describe and implement a handful of optimization algorithms, evaluate their performance on some well known test functions, and create visualizations to build some intuition and verify their function.  Finally, we perform a comparative analysis between algorithms using repeated trials on these test functions in order to draw broader conclusions.
```

## Notebooks
Self contained notebook that contains all of the code and results that appear in the paper.

[Optimization-Visualization-mmorais2](Optimization-Visualization-mmorais2.ipynb)

### Test Functions

#### Rosenbrock
[Test-Function-Rosenbrock](Test-Function-Rosenbrock.ipynb)

#### Goldstein-Price
[Test-Function-Goldstein-Price](Test-Function-Goldstein-Price.ipynb)

#### Bartels-Conn
[Test-Function-Bartels-Conn](Test-Function-Bartels-Conn.ipynb)

#### Egg Crate
[Test-Function-Egg-Crate](Test-Function-Egg-Crate.ipynb)

### Optimization Algorithms

#### Gradient Descent
[Gradient-Descent](Gradient-Descent.ipynb)

#### BFGS
[BFGS](BFGS.ipynb)

#### Simulated Annealing
[Simulated-Annealing](Simulated-Annealing.ipynb)

#### Particle Swarm
[Particle-Swarm](Particle-Swarm.ipynb)

#### Misc

[Line-Search](Line-Search.ipynb)

### Visualizations

#### Test Functions
[Test-Function-Visualization-2D](Test-Function-Visualization-2D.ipynb)

#### Solutions, 2D
[Solution-Visualization-2D](Solution-Visualization-2D.ipynb)

#### Solutions, 3D
[Solution-Visualization-3D](Solution-Visualization-3D.ipynb)

#### Animations, 2D
[Solution-Animation-2D](Solution-Animation-2D.ipynb)

### Results

[Simulation-Results](Simulation-Results.ipynb)
