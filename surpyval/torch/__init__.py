"""Helpers for survival analysis in PyTorch, for people who train their
own networks.

SurPyval does not depend on PyTorch, not even as an optional extra:
``import surpyval`` and ``import surpyval.torch`` never import it. Each
helper imports torch when it is called, and refuses with what to install
when it is absent. There are no training loops, architectures or tuning
here: only SurPyval's likelihood, for your own network and loop.

- :func:`targets` turns lifetime data (failures, right, left and interval
  censoring, left and right truncation, counts) into the tensors of a
  :class:`Targets`;
- :func:`constrain` maps a network's raw outputs into the parameters of
  a distribution, each inside its range;
- :func:`nll` is SurPyval's exact censored and truncated negative
  log-likelihood as a loss, for the families in :data:`SUPPORTED`, of
  one distribution or a mixture of ``K`` (``mixture=``, the model of Deep
  Survival Machines), and :class:`NLLLoss` the same as a
  ``torch.nn.Module``;
- :func:`cox_nll` is the Cox partial likelihood (Efron or Breslow ties) for
  a network's log relative hazards, and :func:`discrete_time_nll` the
  discrete-time (logistic) hazard likelihood for a network's interval
  logits: losses only, equal to ``CoxPH``'s and a discrete lifetime
  model's negative log-likelihood;
- :func:`sf` gives each row's survival curve as an ``(N, T)`` tensor, the
  predictions ``surpyval.metrics.score_predictions`` scores, and
  :func:`to_surpyval` any row as an ordinary SurPyval model.

>>> import pytest; torch = pytest.importorskip("torch")
>>> import surpyval as sp
>>> import surpyval.torch as spt
>>> X = torch.tensor([[0.0], [1.0], [0.0], [1.0]], dtype=torch.float64)
>>> y = spt.targets(x=[12, 30, 18, 41], c=[0, 0, 1, 1])
>>> net = torch.nn.Linear(1, 2).double()
>>> loss = spt.nll(sp.Weibull, y, spt.constrain(sp.Weibull, net(X)))
>>> loss.backward()
"""

from typing import Any

from surpyval.torch.data import Targets, targets
from surpyval.torch.loss import SUPPORTED, constrain, nll
from surpyval.torch.predict import sf, to_surpyval
from surpyval.torch.survival_losses import cox_nll, discrete_time_nll

__all__ = [
    "SUPPORTED",
    "NLLLoss",
    "Targets",
    "constrain",
    "cox_nll",
    "discrete_time_nll",
    "nll",
    "sf",
    "targets",
    "to_surpyval",
]


def __getattr__(name: str) -> Any:
    # NLLLoss subclasses torch.nn.Module, so it is built (and torch
    # imported) when first asked for.
    if name == "NLLLoss":
        from surpyval.torch.loss import nll_loss_class

        return nll_loss_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(set(globals()) | set(__all__))
