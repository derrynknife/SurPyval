"""Random cut points: the extremely randomised survival tree.

In words: a searched tree tries every value of a feature as a cut and
keeps the best. An extremely randomised tree (Geurts, Ernst and Wehenkel,
2006: "extra trees") draws one cut a feature instead, uniformly between
the smallest and largest value the feature takes at the node, and keeps
the best of these few by the same split statistic. The cut is chosen
without looking at the outcome, so every tree is more different from the
others and a node costs one score a feature, not one a distinct value.

The statistic, the leaf models, the minimum leaf sizes and the choice of
features are the tree's own: this module only reduces each candidate
feature to its one random cut. It does so by replacing the feature with
the indicator "above the cut", whose only possible cut is between its two
values, and giving the node's split rule those columns; the rule's answer
is then the feature and the random cut that scored best (or no split, if
no candidate's cut leaves both children large enough).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from surpyval.utils.validation import check_option

SPLITRULES = ("searched", "random")


def parse_splitrule(splitrule: str) -> str:
    """Validate ``splitrule``; the name."""
    check_option(
        "splitrule",
        splitrule,
        SPLITRULES,
        "'searched' (the default) tries every value of a feature as a "
        "cut; 'random' draws one cut a feature (an extremely randomised "
        "tree).",
    )
    return splitrule


def random_cuts(
    Z: NDArray, candidates: Iterable[int], rng: Any
) -> tuple[NDArray, dict[int, float]]:
    """``Z`` with each candidate feature replaced by the indicator of
    being above its random cut, and the cut each feature's indicator
    stands for.

    The cut is drawn uniformly on (min, max) of the feature at the node
    and then read as the largest value of the feature at or below it, so
    that "at most the cut goes left" is the split the indicator makes
    (a tree splits at observed values). A feature that is constant at the
    node has no cut: its indicator is constant, which no split rule
    accepts.
    """
    Zr = np.array(Z, dtype=float, copy=True)
    cuts: dict[int, float] = {}
    for u in candidates:
        column = np.asarray(Z[:, u], dtype=float)
        lo, hi = float(column.min()), float(column.max())
        if not lo < hi:
            Zr[:, u] = 0.0
            continue
        drawn = float(rng.uniform(lo, hi))
        cut = float(column[column <= drawn].max())
        cuts[int(u)] = cut
        Zr[:, u] = (column > cut).astype(float)
    return Zr, cuts
