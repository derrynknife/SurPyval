"""The case bootstrap of a fitted copula model (``method="bootstrap"`` on
``covariance``, ``standard_errors``, ``param_cb``, ``cb`` and the
dependence measures' bounds).

The units are resampled, not the model: each resample draws the fit's
units with replacement (a row with a count ``n`` is that many units),
each with its own pair of values, censoring codes, intervals and
truncation window, and the whole fit is done again on it, the margins
this fit estimated and then the copula (or all at once for
``how="MLE"``). A margin passed to the fit already fitted is known, and
is used as it is in every refit. Nothing is assumed of the margins, so it
is the bound of a fit with a non-parametric margin on data the rank-based
variance does not cover (censored, truncated or tied data, or one
parametric and one non-parametric margin), and the only bound of a joint
probability with a non-parametric margin; a fit with parametric margins
has it as well as its Wald bound.

The bounds are the percentile intervals of the refits
(:func:`surpyval.utils.linalg.percentile_bounds`), and the covariance is
the refits' sample covariance. A refit that fails is left out; one with
no finite maximum (a parameter running off to its limit) is kept in the
percentiles, at the end it ran to, and left out of the covariance, which
it would swamp. Both are counted in one warning.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as onp
import numpy.typing as npt

from surpyval.utils.no_maximum import warn_bootstrap_refits
from surpyval.utils.rng import as_generator

#: What a refit that raises is counted as failing with.
REFIT_ERRORS = (
    ValueError,
    RuntimeError,
    ArithmeticError,
    onp.linalg.LinAlgError,
)

_ADVICE = (
    "treat the bounds as rough, or fit a family the resamples have a "
    "maximum for"
)


@dataclass
class Refits:
    """The refits of one bootstrap of a copula model: the fitted models,
    whether each reached a finite maximum, and how many of the ``n_boot``
    did not, stopped at an unverified one, or failed outright."""

    models: list
    finite: npt.NDArray
    n_boot: int
    no_maximum: int
    unverified: int
    failed: int

    def warn(self) -> None:
        with warnings.catch_warnings():
            # (A warning is given once per place; this one's counts
            # differ by bootstrap.)
            warnings.simplefilter("always")
            warn_bootstrap_refits(
                self.n_boot,
                no_maximum=self.no_maximum,
                unverified=self.unverified,
                failed=self.failed,
                advice=_ADVICE,
            )

    def vectors(self) -> npt.NDArray:
        """Every parameter of each refit (the copula's, then each
        parametric margin's, as ``covariance(margins=True)`` lays them
        out): one row per refit."""
        return onp.array(
            [m._parameter_layout().values for m in self.models], dtype=float
        ).reshape(len(self.models), -1)

    def covariance(self) -> npt.NDArray:
        """The sample covariance of the refits that reached a finite
        maximum (``nan`` with fewer than two)."""
        rows = self.vectors()[self.finite]
        size = rows.shape[1]
        if rows.shape[0] < 2:
            return onp.full((size, size), onp.nan)
        return onp.atleast_2d(onp.cov(rows, rowvar=False, ddof=1))


def check_resampled(model: Any) -> None:
    """Refuse the bootstrap of a model that cannot be refitted: one with
    no data. (The data's counts ``n`` are whole numbers of units, which
    the data's own checks hold them to.)"""
    if model.k is None:
        raise ValueError(
            "A bootstrap refits the model to resamples of its data, and "
            "this model was built from its parameters with `from_params`: "
            "fit it to data for bounds."
        )
    if model.data is None:
        raise ValueError(
            "A bootstrap refits the model to resamples of its data, and a "
            "model restored from a dictionary keeps none: refit the model "
            "to the data for bootstrap bounds."
        )


def refits(model: Any, n_boot: int, random_state: Any) -> Refits:
    """Refit ``model`` to ``n_boot`` resamples of its units (see the
    module docstring)."""
    from surpyval.multivariate.parametric.data import MultivariateSurpyvalData

    check_resampled(model)
    data = model.data
    rng = as_generator(random_state)
    counts = onp.asarray(data.n, dtype=float)
    total = int(round(counts.sum()))
    share = counts / counts.sum()
    specs = list(model._margin_specs or model.margins)
    # The search starts from the estimate, where it is inside the space.
    inside = (
        not model._parameter_layout().on_bound()[: model.params.size].any()
    )
    init = model.params if inside and model.params.size else None
    models = []
    finite = []
    no_maximum = unverified = failed = 0
    for _ in range(n_boot):
        drawn = rng.multinomial(total, share)
        keep = drawn > 0
        resample = MultivariateSurpyvalData(
            data.x[keep],
            c=data.c[keep],
            n=drawn[keep],
            t=data.t[keep],
            xl=data.xl[keep],
            xr=data.xr[keep],
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = model.copula.fit(
                    resample, margins=specs, how=model.method, init=init
                )
        except REFIT_ERRORS:
            failed += 1
            continue
        if not onp.all(onp.isfinite(fit.params)):
            failed += 1
            continue
        no_maximum += fit.maximum == "no finite maximum"
        unverified += fit.maximum == "unverified"
        finite.append(fit.maximum != "no finite maximum")
        models.append(fit)
    return Refits(
        models,
        onp.asarray(finite, dtype=bool),
        n_boot,
        int(no_maximum),
        int(unverified),
        failed,
    )
