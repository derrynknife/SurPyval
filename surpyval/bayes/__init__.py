"""
Bayesian fitting of parametric distributions.

Say what is already believed (a prior on a parameter, on a B-life, or an
earlier fit), and ``Weibull.fit_bayes(x, c, priors=...)`` combines it with
the data -- any censoring, truncation or intervals, exactly as ``fit``
takes them -- into a model used like any other: its ``sf``, ``qf``,
``mean`` and ``random`` are those of a new unit's life with the
uncertainty in the parameters averaged in, and ``cri``,
``quantile_cri``, ``mean_cri`` and ``param_cri`` give credible intervals.

The posterior of one to three free parameters is computed on a fine,
self-checking grid (deterministic, no seed needed), or in closed form
where one exists (an Exponential with a Gamma prior, a Bernoulli with a
Beta prior, a Normal mean with the spread known).

>>> import surpyval as sp
>>> model = sp.Weibull.fit_bayes(
...     [2100, 3400, 4050, 5000, 5000], c=[0, 0, 0, 1, 1],
...     priors={"beta": sp.prior.around(2, between=(1.5, 3))})
>>> type(model).__name__
'BayesianParametric'

The priors are in ``surpyval.prior``; the likelihood every fit uses is
``log_likelihood_fn``, for anyone who wants it in another sampler; and
:meth:`BayesianParametric.from_draws` brings another sampler's draws
back as a model.
"""

from surpyval.bayes import prior
from surpyval.bayes.likelihood import LogLikelihood, log_likelihood_fn
from surpyval.bayes.model import BayesianParametric
from surpyval.bayes.posterior import Posterior

__all__ = [
    "BayesianParametric",
    "LogLikelihood",
    "Posterior",
    "log_likelihood_fn",
    "prior",
]
