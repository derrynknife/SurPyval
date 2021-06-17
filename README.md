# SurPyval - Survival Analysis in Python

[![PyPI - Version](https://img.shields.io/badge/pypi-v0.4.0-success)](https://pypi.org/project/surpyval/)
![PyPI - Python Version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)
[![Documentation Status](https://readthedocs.org/projects/surpyval/badge/?version=latest)](https://surpyval.readthedocs.io/en/latest/?badge=latest)


Yet another Python survival analysis tool. 

This is another pure python survival analysis tool so why was it needed? The intent of this package was to closely mimic the scipy API as close as possible with a simple `.fit()` method for any type of distribution (parametric or non-parametric); other survival analysis packages don't completely mimic that API. Further, there is currently (at the time of writing) no pacakage that can take an arbitrary comination of observed, censored, and truncated data. Finally, surpyval is unique in that it can be used with multiple parametric estimation methods. This allows for an analyst to determine a distribution for the parameters if another method fails. The parametric methods available are Maximum Likelihood Estimation (MLE), Probability Plotting (MPP), Mean Square Error (MSE), Method of Moments (MOM), and Maximum Product of Spacing (MPS). Surpyval can, for each type of estimator, take the following types of input data:

| Method | Para/Non-Para | Observed | Censored | Truncated |
| ------ | ---- |-----|------|------|
| **MLE** | Parametric | Yes | Yes | Yes |
| **MPP** | Parametric | Yes | Yes | Limited |
| **MSE** | Parametric | Yes | Yes | Limited |
| **MOM** | Parametric | Yes | No | No |
| **MPS** | Parametric | Yes | Yes | No |
| **Kaplan-Meier** | Non-Parametric | Yes | Right only | Left only |
| **Nelson-Aalen** | Non-Parametric | Yes | Right only | Left only |
| **Fleming-Harrington** | Non-Parametric | Yes | Right only | Left only |
| **Turnbull** | Non-Parametric | Yes | Yes | Yes |

SurPyval also offers many different distributions for users, and because of the flexible implementation adding new distributions is easy. The currently available distributions are:

| Distribution |
| ---- |
| **Weibull** |
| **Normal** |
| **LogNormal** |
| **Gamma** |
| **Beta** |
| **Uniform** |
| **Exponential** |
| **Exponentiated Weibull** |
| **Gumbel** |
| **Logistic** |
| **LogLogistic** |

This project spawned from a Reliaility Engineering project; due to the history of reliability engineers estimating parameters from a probability plot. SurPyval has continued this tradition to ensure that any parametric distribution can have the estimate plotted on a probability plot. These visualisations enable an analyst to get a sense of the goodness of fit of the parametric distribution with the non-parametric distribution.

# Install and Quick Intro

SurPyval can be installed via pip using the PyPI [repository](https://pypi.org/project/surpyval/)

```bash
pip install surpyval
```

If you're familiar with survival analysis, and Weibull plotting, the following is a quick start.

```python
from surpyval import Weibull
from surpyval.datasets import BoforsSteel

# Fetch some data that comes with SurPyval
data = BoforsSteel.df

x = data['x']
n = data['n']

model = Weibull.fit(x=x, n=n, offset=True)
model.plot();
```

![Weibull Data and Distribution](docs/images/weibull_plot.png)

# Documentation

SurPyval is well documented, and improving, at the main [documentation](https://surpyval.readthedocs.io/en/latest/).

# Contact

Email [derryn](mailto:derryn.knife@gmail.com) if you want any features or to see how SurPyval can be used for you.

