"""``concordance`` on the regression models (#512).

Harrell's C of a fitted model: how well its risk scores rank the subjects'
times (:func:`surpyval.metrics.concordance_index`). Each family says what
its risk score is -- a higher score predicting an earlier event -- through
``_concordance_risk``, and which training data it kept through
``_concordance_data``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.metrics.concordance import (
    ConcordanceResult,
    concordance_estimate,
    concordance_index,
)


class ConcordanceMixin:
    """Adds :meth:`concordance` to a regression model."""

    def _concordance_risk(self, x: npt.NDArray, Z: Any) -> npt.NDArray:
        """The risk score of each row of ``Z`` (higher = earlier event);
        ``x`` are the times being scored, one per row."""
        raise NotImplementedError  # pragma: no cover - every host has it

    def _concordance_data(self) -> tuple | None:
        """The training ``(x, c, n, Z)``, or ``None`` if not kept."""
        return None

    def concordance(
        self,
        x: npt.ArrayLike | None = None,
        c: npt.ArrayLike | None = None,
        Z: Any = None,
        tie_tol: float = 1e-8,
        ties: str = "therneau",
        *,
        weighting: str = "harrell",
        tau: float | None = None,
    ) -> float:
        """Harrell's (or Uno's) concordance index (C) of the model's risk
        scores.

        The proportion of usable pairs of subjects -- the one with the
        earlier time had the event -- in which the model gives that
        subject the higher risk: 1 ranks every pair correctly, 0.5 is
        chance. With no arguments it scores the data the model was fitted
        to (each row counted ``n`` times); pass ``x``, ``c`` and ``Z``
        together to score other data, such as a test set. Two events
        at the same time are not a pair (``ties="therneau"``, as R's
        ``concordance`` and lifelines) unless ``ties="harrell"``; see
        :func:`surpyval.metrics.concordance_index` for the treatment of
        tied times and scores.

        The risk score, higher meaning an earlier event, is for each
        family:

        - ``CoxPH``, the frailty models and ``AdditiveHazards`` (Lin-Ying):
          the linear predictor :math:`\\beta'Z` (the log hazard ratio, or
          the covariates' additive hazard), as R's ``concordance`` and
          lifelines use;
        - the parametric families (PH, AFT, PO, AH and the accelerated
          life models): the cumulative hazard :math:`H(t^* \\mid Z)` at
          :math:`t^*`, the median of the times scored. For every family
          whose covariates act through a linear predictor this ranks the
          rows exactly as the linear predictor does, with the sign that
          means a higher risk (:math:`\\beta'Z` for PH, AFT and AH,
          :math:`-\\beta'Z` for PO), and :math:`t^*` does not matter; it
          can only matter for an accelerated life model whose life model
          sets a shape parameter;
        - ``BuckleyJames``: :math:`-\\beta'Z`, since the model is linear in
          the log time;
        - ``ProportionalOdds``: :math:`-\\beta'Z`, the log of the odds of
          failure relative to the baseline's.

        Parameters
        ----------
        x : array_like, optional
            Observed times.
        c : array_like, optional
            Censoring flags, 0 an event and 1 right censored.
        Z : array_like or DataFrame, optional
            Covariates, one row per time (a DataFrame for a model fitted
            with ``fit_from_df``).
        tie_tol : float, optional
            Scores within this of each other are tied. Default ``1e-8``.
        ties : {"therneau", "harrell"}, optional
            ``"therneau"`` (the default, as R's ``survival::concordance``
            and lifelines): two events at the same time are not a usable
            pair. ``"harrell"`` (Harrell's original definition): they are,
            counting 1 if their scores are tied, else 0.5.
        weighting : {"harrell", "uno"}, optional
            ``"harrell"`` (default): every usable pair weighs 1. ``"uno"``:
            Uno's inverse-probability-of-censoring weighted C, each pair
            weighted by ``1 / G(t-)^2`` (see
            :func:`surpyval.metrics.concordance_estimate`).
        tau : float, optional
            Count only the pairs whose earlier time is at most ``tau``.
            Default ``None``, every pair.

        Returns
        -------
        float
            The concordance index. For its standard error and confidence
            interval use :meth:`concordance_estimate`.

        Raises
        ------
        ValueError
            If only some of ``x``, ``c`` and ``Z`` are given, or none are
            and the model does not keep its training data (a model
            restored with ``from_dict``, or one fitted with time-varying
            covariates, whose rows are intervals of a subject's history).

        Examples
        --------
        >>> import surpyval as sp
        >>> from surpyval.datasets import load_lung
        >>> lung = load_lung().dropna(subset=["ph.ecog"])
        >>> lung["censored"] = 1 - lung["status"]  # status 1 is a death
        >>> model = sp.CoxPH.fit_from_df(
        ...     lung, x_col="time", c_col="censored",
        ...     Z_cols=["age", "sex", "ph.ecog"],
        ... )
        >>> round(model.concordance(), 4)
        0.6371
        >>> round(model.concordance(ties="harrell"), 4)
        0.6369
        """
        x_arr, c_arr, risk = self._concordance_inputs(x, c, Z)
        if np.isnan(risk).any():
            # A missing covariate leaves a subject's score, and so the
            # index, undefined (as the forests' concordance).
            return float("nan")
        if weighting == "harrell" and tau is None:
            return concordance_index(x_arr, c_arr, risk, tie_tol, ties)
        return concordance_estimate(
            x_arr, c_arr, risk, weighting, tau, tie_tol=tie_tol, ties=ties
        ).estimate

    def concordance_estimate(
        self,
        x: npt.ArrayLike | None = None,
        c: npt.ArrayLike | None = None,
        Z: Any = None,
        *,
        weighting: str = "harrell",
        tau: float | None = None,
        alpha: float = 0.05,
        tie_tol: float = 1e-8,
        ties: str = "therneau",
        ci_method: str = "normal",
    ) -> ConcordanceResult:
        """The model's concordance index with its standard error and
        confidence interval (R's ``concordance(fit)``).

        The index and the scores are :meth:`concordance`'s; the standard
        error is the infinitesimal jackknife of
        :func:`surpyval.metrics.concordance_estimate`. With no data it
        scores the data the model was fitted to, each row repeated ``n``
        times as its own subject, and so as R does with the rows
        repeated, not with ``n`` as case weights. The standard error
        treats the scores as fixed: it does not include the uncertainty of
        the fitted coefficients, as R's does not.

        Parameters
        ----------
        x : array_like, optional
            Observed times.
        c : array_like, optional
            Censoring flags, 0 an event and 1 right censored.
        Z : array_like or DataFrame, optional
            Covariates, one row per time.
        weighting : {"harrell", "uno"}, optional
            Harrell's C (default) or Uno's IPCW C.
        tau : float, optional
            Count only the pairs whose earlier time is at most ``tau``.
        alpha : float, optional
            Significance level of the interval. Default ``0.05``.
        tie_tol : float, optional
            Scores within this of each other are tied. Default ``1e-8``.
        ties : {"therneau", "harrell"}, optional
            How two events at the same time count. Default
            ``"therneau"``.
        ci_method : {"normal", "logit"}, optional
            The scale of the interval. Default ``"normal"``.

        Returns
        -------
        ConcordanceResult
            The estimate, ``se``, ``ci`` and the weighted pair counts.

        Raises
        ------
        ValueError
            As :meth:`concordance`; or if a covariate is missing, which
            leaves a subject's score undefined.

        Examples
        --------
        >>> import surpyval as sp
        >>> from surpyval.datasets import load_lung
        >>> lung = load_lung().dropna(subset=["ph.ecog"])
        >>> lung["censored"] = 1 - lung["status"]  # status 1 is a death
        >>> model = sp.CoxPH.fit_from_df(
        ...     lung, x_col="time", c_col="censored",
        ...     Z_cols=["age", "sex", "ph.ecog"],
        ... )
        >>> res = model.concordance_estimate()
        >>> round(res.estimate, 4), round(res.se, 4)
        (0.6371, 0.0251)
        """
        x_arr, c_arr, risk = self._concordance_inputs(x, c, Z)
        if np.isnan(risk).any():
            raise ValueError(
                "A covariate is missing for {} of {} subjects, so their "
                "risk scores are undefined: drop those rows from x, c and "
                "Z".format(int(np.isnan(risk).sum()), risk.size)
            )
        return concordance_estimate(
            x_arr, c_arr, risk, weighting, tau, alpha, tie_tol, ties, ci_method
        )

    def _concordance_inputs(
        self,
        x: npt.ArrayLike | None,
        c: npt.ArrayLike | None,
        Z: Any,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """The times, flags and risk scores to score: those given, or the
        training data's with each row repeated ``n`` times."""
        given = [v is not None for v in (x, c, Z)]
        if any(given) and not all(given):
            raise ValueError(
                "Pass x, c and Z together to score new data, or none of "
                "them to score the data the model was fitted to."
            )
        if all(given):
            x_arr = np.asarray(x, dtype=float).ravel()
            c_arr = np.asarray(c).ravel()
            return x_arr, c_arr, self._concordance_risk(x_arr, Z)
        data = self._concordance_data()
        if data is None:
            raise ValueError(
                f"This {type(self).__name__} does not keep the data it was "
                "fitted to (it was restored with from_dict, or fitted with "
                "time-varying covariates); pass x, c and Z to score data."
            )
        x_arr, c_arr, n_arr, Z_arr = data
        rows = np.repeat(np.arange(x_arr.size), np.asarray(n_arr, int))
        x_arr = np.asarray(x_arr, dtype=float)[rows]
        risk = self._concordance_risk(x_arr, np.asarray(Z_arr)[rows])
        return x_arr, np.asarray(c_arr)[rows], risk
