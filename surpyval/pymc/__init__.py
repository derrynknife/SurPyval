"""Helpers for survival analysis in PyMC, for people who already use it.

SurPyval does not depend on PyMC, not even as an optional extra:
``import surpyval`` and ``import surpyval.pymc`` never import it. Each
helper imports PyMC (and pytensor) when it is called, and refuses with
what to install when it is absent.

- :func:`censored_likelihood` adds SurPyval's exact likelihood of any
  censored and truncated data to the open model, for the families in
  :data:`SUPPORTED` and for the regression fitters on them
  (:data:`REGRESSION`: proportional hazards, accelerated failure time
  and accelerated life, with covariates ``Z``);
- :func:`log_likelihood` is the same term as a pytensor expression;
- :func:`prior_from_model` turns a fitted SurPyval model into a joint
  prior, and :func:`prior_from_spec` turns a plain description of each
  parameter's prior into PyMC variables;
- :func:`to_surpyval` turns the posterior draws back into a SurPyval
  model (:class:`~surpyval.bayes.BayesianParametric`), with its
  functions, credible intervals and plots;
- :func:`pooled_likelihood` builds groups (sites, fleets) sharing one
  population distribution (hierarchical, partial pooling), and
  :func:`group_models` turns its draws into a SurPyval model per group
  and one for a new group.

See :doc:`/PyMC with SurPyval` for a worked example.
"""

from surpyval.pymc.likelihood import (
    REGRESSION,
    SUPPORTED,
    censored_likelihood,
    log_likelihood,
)
from surpyval.pymc.pooling import group_models, pooled_likelihood
from surpyval.pymc.posterior import to_surpyval
from surpyval.pymc.priors import (
    PRIOR_FAMILIES,
    prior_from_model,
    prior_from_spec,
)

__all__ = [
    "REGRESSION",
    "SUPPORTED",
    "PRIOR_FAMILIES",
    "censored_likelihood",
    "group_models",
    "log_likelihood",
    "pooled_likelihood",
    "prior_from_model",
    "prior_from_spec",
    "to_surpyval",
]
