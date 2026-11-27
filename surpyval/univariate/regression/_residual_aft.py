"""
The predictions of a semi-parametric accelerated-failure-time model whose
error distribution is the Kaplan-Meier estimate of its residuals.

Buckley-James and the rank-based (Gehan and log-rank) estimators fit the
same model,

.. math::
    \\log T = -\\beta' Z + \\varepsilon ,

with the distribution of :math:`\\varepsilon` left unspecified, and differ
only in how they estimate :math:`\\beta`. Once it is estimated, each
predicts from the (Efron tail-corrected) Kaplan-Meier estimate of the
residuals :math:`e_i = \\log x_i + \\beta' Z_i`:
``sf(t | Z) = S_eps(log t + beta'Z)``. This base class holds those
predictions, so the two models give them in one way.

A host class sets ``beta`` (in the accelerated-failure sign: a positive
coefficient shortens life), ``params``, ``_resid`` and ``_resid_surv``
(the sorted residuals and the survival at each) and ``_data``, the fitted
``(Y, delta, Z, w)`` or ``None``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.diagnostics.mixin import DiagnoseMixin
from surpyval.serialisation import SerialisableMixin
from surpyval.utils.shapes import covariate_rows, keeps_query_shape

from ._concordance import ConcordanceMixin
from ._prediction import (
    ConditionalSurvivalMixin,
    paired_probabilities,
    step_quantiles,
)
from ._summary import coefficient_names
from .regression_data import LinearPredictorMixin, NoLikelihoodMixin

# How far below a residual Kaplan-Meier step a query's residual may round
# and still be at the step, relative to max(|r|, 1) (see ``_resid_sf``).
RESID_ROUNDING = 1e-12


def residual_km(
    e: npt.NDArray, delta: npt.NDArray, w: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Weighted Kaplan-Meier of the residuals with an Efron tail-correction.

    Returns the sorted unique residual values, the (right-continuous) survival
    at each, and the probability mass (jump) the estimator places at each.
    Forcing the largest residual to be an event makes the estimator proper, so
    the jumps sum to one and conditional expectations above any point are
    finite.
    """
    order = np.argsort(e, kind="mergesort")
    e = e[order]
    delta = delta[order].astype(float).copy()
    w = w[order].astype(float)

    # Efron tail-correction: the largest residual(s) act as an event.
    delta[e == e[-1]] = 1.0

    uniq, first_idx = np.unique(e, return_index=True)
    w_cum = np.concatenate([[0.0], np.cumsum(w)])
    total = w_cum[-1]
    # At-risk weight for each unique time (residuals >= that time).
    r = total - w_cum[first_idx]

    d = np.zeros(uniq.shape[0])
    grp = np.searchsorted(uniq, e)
    np.add.at(d, grp, np.where(delta == 1.0, w, 0.0))

    with np.errstate(divide="ignore", invalid="ignore"):
        surv = np.cumprod(1.0 - d / r)
    surv = np.clip(surv, 0.0, 1.0)
    s_prev = np.concatenate([[1.0], surv[:-1]])
    jumps = s_prev - surv
    return uniq, surv, jumps


class ResidualAFTModel(
    ConditionalSurvivalMixin,
    NoLikelihoodMixin,
    LinearPredictorMixin,
    ConcordanceMixin,
    DiagnoseMixin,
    SerialisableMixin,
):
    """The predictions shared by the semi-parametric AFT models that read
    a residual Kaplan-Meier (see the module docstring)."""

    beta: npt.NDArray
    params: npt.NDArray
    _resid: npt.NDArray
    _resid_surv: npt.NDArray
    _data: tuple | None

    @property
    def parameter_names(self) -> list[str]:
        """The names of ``params``, entry by entry: each covariate's
        column (a formula, ``fit_from_df`` or a DataFrame ``Z``), else
        ``coef_0``, ``coef_1``, ... (#614), as in the parametric
        regression models."""
        return coefficient_names(self, len(self.params))

    def _concordance_risk(self, x: npt.NDArray, Z: Any) -> npt.NDArray:
        Z_arr = np.asarray(self._prepare_Z(Z), dtype=float)
        return -(Z_arr.reshape(x.size, -1) @ self._coef())

    def _concordance_data(self) -> tuple | None:
        if self._data is None:
            return None
        Y, delta, Z, w = self._data
        return np.exp(Y), (delta == 0).astype(int), w, Z

    def _resid_sf(self, r: npt.NDArray) -> npt.NDArray:
        # Right-continuous residual survival at query points ``r``. A
        # residual within RESID_ROUNDING of a step is at the step: a time
        # is mapped to its residual by ``log t + beta'Z``, and ``qf``'s time
        # ``exp(r_k - beta'Z)`` mapped back lands an ulp either side of
        # ``r_k``, depending on how numpy's exp and log round (its AVX2 and
        # AVX-512 kernels differ). Landing below it, ``ff(qf(p))`` was the
        # step before p for one random query in ten, and on CI's runners
        # in #662's test. 1e-12 on the log scale is 1e-12 relative in time.
        tol = RESID_ROUNDING * np.maximum(np.abs(r), 1.0)
        idx = np.searchsorted(self._resid, r + tol, side="right") - 1
        out = np.where(
            idx < 0,
            1.0,
            self._resid_surv[np.clip(idx, 0, len(self._resid_surv) - 1)],
        )
        return out

    def _linear_predictor(self, x: npt.NDArray, Z: Any) -> npt.NDArray:
        """``beta'Z`` for each time in ``x``: ``Z`` is one covariate vector
        (used at every time) or one row per time, paired in the order
        given, as for the other regression models (#426)."""
        Z_arr = np.asarray(self._prepare_Z(Z), dtype=float)
        p = self.beta.size
        if Z_arr.ndim == 0:
            Z_arr = Z_arr.reshape(1)
        if Z_arr.ndim == 1:
            Z_arr = Z_arr.reshape(1, -1)
        if Z_arr.ndim != 2 or Z_arr.shape[1] != p:
            raise ValueError(
                "Z must be one covariate vector of length {} or one such "
                "row per time; got an array of shape {}.".format(
                    p, np.shape(Z)
                )
            )
        if Z_arr.shape[0] not in (1, x.size):
            raise ValueError(
                "Z has {} covariate rows but there are {} times; give one "
                "covariate vector, or one row per time.".format(
                    Z_arr.shape[0], x.size
                )
            )
        # An aliased coefficient (nan, #476) is predicted with as 0.
        return Z_arr @ np.where(np.isnan(self.beta), 0.0, self.beta)

    @keeps_query_shape
    def sf(self, x: npt.ArrayLike, Z: npt.ArrayLike) -> npt.NDArray:
        """Survival ``P(T > x | Z) = S_eps(log x + beta'Z)``; ``Z`` is one
        covariate vector (used at every time) or one row per time in
        ``x``, paired in the order given."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        # beta is the accelerated-failure (negated) slope, so the residual
        # r = log t - gamma'Z = log t + beta'Z. At and below time 0 nothing
        # has failed: survival 1 (log(0) = -inf gives that already, but
        # warned, and a negative time gave nan).
        positive = x > 0
        with np.errstate(divide="ignore"):
            r = np.log(np.where(positive, x, 1.0)) + self._linear_predictor(
                x, Z
            )
        # A missing covariate (a DataFrame row with a nan) gives nan, as in
        # the other families; the residual lookup read it as the last step
        # (survival 0). So does a missing time, which ``positive`` read as
        # "not after time 0" (survival 1).
        out = np.where(positive, self._resid_sf(r), 1.0)
        return np.where(np.isnan(r) | np.isnan(x), np.nan, out)

    @keeps_query_shape
    def ff(self, x: npt.ArrayLike, Z: npt.ArrayLike) -> npt.NDArray:
        """Failure probability ``1 - sf(x, Z)``; ``Z`` as for
        :meth:`sf`."""
        return 1.0 - self.sf(x, Z)

    @keeps_query_shape
    def Hf(self, x: npt.ArrayLike, Z: npt.ArrayLike) -> npt.NDArray:
        """Cumulative hazard ``-log sf(x, Z)``; ``Z`` as for
        :meth:`sf`."""
        with np.errstate(divide="ignore"):
            # 0.0 - log: where sf is 1, -log(1) is -0.0 (#746).
            return 0.0 - np.log(self.sf(x, Z))

    @keeps_query_shape
    def qf(self, p: npt.ArrayLike, Z: npt.ArrayLike) -> npt.NDArray:
        """
        The quantile function: the first time at which the predicted
        failure probability ``ff(x, Z)`` reaches ``p`` (#662), ``nan``
        where it never does -- the residual Kaplan-Meier stops at the
        last residual, above ``1 - p`` where the data end censored. It is
        :math:`e^{q_\\epsilon(p) - \\beta' Z}`, :math:`q_\\epsilon` the
        residual Kaplan-Meier's quantile, taken as the non-parametric
        ``qf`` takes it (a curve within ``1e-9`` of ``p`` reaches it).
        ``Z`` is paired with ``p`` as :meth:`sf` pairs it with ``x``. A
        probability outside [0, 1] gives ``nan``, with a warning.

        Examples
        --------
        >>> from surpyval import BuckleyJames
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> x, c = df["week"].values, 1 - df["arrest"].values
        >>> model = BuckleyJames.fit(x, df[["fin", "age", "prio"]].values, c=c)
        >>> rows = [[0, 25, 3], [1, 25, 3], [0, 25, 10]]
        >>> b10 = model.qf(0.1, rows)
        >>> b10.round(2)
        array([21.71, 28.33, 14.38])
        >>> model.ff(b10, rows).round(3)
        array([0.102, 0.102, 0.102])
        """
        rows = covariate_rows(
            np.asarray(self._prepare_Z(Z), dtype=float), self.beta.size
        )
        u, rows, _ = paired_probabilities(p, rows)
        lp = self._linear_predictor(np.empty(u.size), rows)
        resid = step_quantiles(
            (1.0 - np.asarray(self._resid_surv, dtype=float))[None, :],
            self._resid,
            u,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            return np.exp(resid - lp)
