# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

## Solutions

| Problem | File | Function |
| --- | --- | --- |
| Sigmoid (NumPy) | [sigmoid-numpy.py](sigmoid-numpy/sigmoid-numpy.py) | `sigmoid(x)` |
| Logistic regression training | [logistic-regression-training.py](logistic-regression-training/logistic-regression-training.py) | `train_logistic_regression(X, y, lr=0.1, steps=1000)` |

## Running examples

From the repository root, install NumPy:

```sh
python -m pip install numpy
```

The solution files define functions; running them directly does not print results.
Use `runpy` to load the files with their existing hyphenated names and call the functions:

```sh
python -c 'import runpy; solution = runpy.run_path("sigmoid-numpy/sigmoid-numpy.py"); print(solution["sigmoid"]([-1, 0, 1]))'
python -c 'import runpy; solution = runpy.run_path("logistic-regression-training/logistic-regression-training.py"); print(solution["train_logistic_regression"]([[0], [1], [2], [3]], [0, 0, 1, 1]))'
```

The first example prints sigmoid probabilities. The second prints the learned
weight vector and bias.
