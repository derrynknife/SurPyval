"""Removed in v0.23: ``surpyval.utils.score.score`` is
``surpyval.metrics.concordance_index`` (#653).

Importing this module raises an ``ImportError`` that says so, rather than
Python's bare "No module named 'surpyval.utils.score'".
"""

raise ImportError(
    "surpyval.utils.score was removed in v0.23; its score(x, c, risk) is "
    "surpyval.metrics.concordance_index(x, c, risk, ties='harrell') "
    "(from surpyval.metrics import concordance_index).",
    name=__name__,
)
