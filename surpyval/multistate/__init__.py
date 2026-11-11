"""
Multi-state models: units that move between several states.

A multi-state model follows each unit through a set of states -- healthy,
ill and dead; working, degraded and failed -- and estimates how likely it
is to be in each state over time. Competing risks (one starting state and
an absorbing state per cause) and ordinary survival (two states) are its
simplest cases.

- :class:`TransitionMatrix`: the states and the allowed transitions.
- :class:`MultiState`: the non-parametric (Aalen-Johansen) transition and
  state occupation probabilities, with their variances and bounds.
- :class:`MultiStateProportionalHazards`: a Cox model for each transition
  (with effects shared across transitions where asked), and the
  probabilities predicted from it for given covariates.
- :func:`to_stacked` and :func:`from_stacked`: the long form the models
  take (one row per spell in a state) and the stacked form (one row per
  possible transition, as R's ``mstate`` uses).
"""

from surpyval.multistate._data import from_stacked, to_stacked
from surpyval.multistate.nonparametric import MultiState
from surpyval.multistate.regression import MultiStateProportionalHazards
from surpyval.multistate.transitions import TransitionMatrix

__all__ = [
    "MultiState",
    "MultiStateProportionalHazards",
    "TransitionMatrix",
    "from_stacked",
    "to_stacked",
]
