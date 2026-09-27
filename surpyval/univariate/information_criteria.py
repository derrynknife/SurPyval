"""Shared information-criterion methods (#298).

``Parametric`` and ``ParametricRegressionModel`` carried near-identical
``neg_ll``/``aic``/``bic``/``aic_c`` implementations. They differed only
in how the fitted data is stored (an xcnt dict versus a ``SurpyvalData``
object), which the ``_ic_sample_size_from_data`` hook absorbs. ``_ic_k``
is the parameter count of the ``aic``/``bic`` penalty and (through
``_ic_k_aic_c``) of the small-sample correction: ``self.k`` by default,
and the number of *estimated* parameters for univariate parametric models,
which exclude the ones fixed at fit time. Regression models set ``self.k``
to their number of estimated parameters directly.

:func:`ic_sample_size` is the one definition of the sample size ``n`` of
BIC's :math:`k \\ln n` penalty and of AIC_c's correction, shared by every
model in the library that reports either -- univariate parametric,
Royston-Parmar, regression (including frailty and time-varying-covariate
fits), recurrent-event and copula models -- so their criteria count the
same data the same way.
"""

from typing import Any

import numpy as np
import numpy.typing as npt


def ic_sample_size(
    c: npt.ArrayLike, n: npt.ArrayLike, n_rows: "float | None" = None
) -> float:
    r"""The sample size ``n`` of BIC and of AIC_c's correction term.

    It is the number of *observed failures*: every observation that is not
    right-censored -- exact, left-censored (failed before ``x``) and
    interval-censored (failed within ``[xl, xr]``) -- weighted by its count
    ``n``. A right-censored unit has not failed and adds nothing
    (Volinsky and Raftery, 2000). For recurrent-event data the same codes
    count the
    observed events, with a left- or interval-censored row adding the
    number of events it holds. For joint (copula) data, where ``c`` has one
    column per series, a row counts when at least one of its series is not
    right-censored.

    When there is no observed failure the number of rows is used instead
    (weighted by ``n``, or ``n_rows`` when given, for data whose rows are
    not the units observed, such as time-varying-covariate episodes), so
    the criteria stay finite: :math:`\ln 0` made BIC :math:`-\infty`, and
    so the "best" model of any comparison.

    Parameters
    ----------
    c : array_like
        Censoring codes (``0`` exact, ``1`` right, ``-1`` left, ``2``
        interval), one per row, or one column per series for joint data.
    n : array_like
        The count (weight) of each row.
    n_rows : float, optional
        The fallback when there is no observed failure. Defaults to the
        weighted number of rows, ``sum(n)``.

    Returns
    -------
    float
        The sample size.

    Examples
    --------
    >>> from surpyval.univariate.information_criteria import ic_sample_size
    >>> ic_sample_size([0, 1, 2, -1], [3, 5, 2, 1])
    6.0
    >>> ic_sample_size([1, 1], [3, 5])
    8.0
    """
    codes = np.asarray(c)
    weights = np.asarray(n, dtype=float)
    failed = codes != 1
    if failed.ndim > 1:
        # A joint row is a failure observation if any series failed in it.
        failed = failed.reshape(failed.shape[0], -1).any(axis=1)
    n_failures = float(weights[failed].sum())
    if n_failures > 0:
        return n_failures
    return float(weights.sum()) if n_rows is None else float(n_rows)


class InformationCriteriaMixin:
    """Log-likelihood based model-selection criteria.

    Requires the host class to set ``self.k`` (penalised parameter
    count) and ``self._neg_ll`` when fitted, and to implement
    ``_ic_sample_size_from_data`` returning :func:`ic_sample_size` of the
    data it was fitted to. A host whose sample size is not a function of
    the data it carries (a model restored without its data, or fitted to
    rows that are not the units observed) sets ``_ic_n`` instead.
    """

    # Set by the host class when fitted; annotated (not assigned) so the
    # hasattr-based caching below still works.
    k: int
    _neg_ll: float
    _bic: float
    _aic: float
    _aic_c: float
    # The stored sample size of the criteria, which takes precedence over
    # the data when set: restored by ``from_dict`` so a model saved without
    # its data reports the same ``bic``/``aic_c`` as the fitted one.
    _ic_n: "float | None" = None

    def _ic_sample_size_from_data(self) -> float:
        raise NotImplementedError

    def _ic_sample_size(self) -> float:
        """The sample size of ``bic`` and ``aic_c``; see
        :func:`ic_sample_size`."""
        if self._ic_n is not None:
            return float(self._ic_n)
        return self._ic_sample_size_from_data()

    def _ic_sample_size_or_none(self) -> "float | None":
        """The sample size for ``to_dict``, or ``None`` when the model has
        neither data nor a stored value (built from parameters)."""
        try:
            return self._ic_sample_size()
        except ValueError:
            return None

    @staticmethod
    def _restored_ic_n(model_dict: dict) -> "float | None":
        """The ``ic_n`` a ``to_dict`` dictionary stored, if any."""
        value: Any = model_dict.get("ic_n")
        return None if value is None else float(value)

    def _ic_k(self) -> int:
        # The parameter count used in the aic and bic penalties.
        return self.k

    def _ic_k_aic_c(self) -> int:
        # The parameter count used in the aic_c correction term; the same
        # as the aic penalty it corrects unless a host overrides it.
        return self._ic_k()

    def neg_ll(self) -> float:
        r"""

        The negative log-likelihood for the model, if it was fit with the
        ``fit()`` method. Not available if fit with the ``from_params()``
        method.

        Returns
        -------

        neg_ll : float
            The negative log-likelihood of the model

        Examples
        --------

        >>> from surpyval import Weibull
        >>> import numpy as np
        >>> np.random.seed(1)
        >>> x = Weibull.random(100, 10, 3)
        >>> model = Weibull.fit(x)
        >>> model.neg_ll()
        262.52685642390634
        """
        # A model restored from a dict keeps its fitted negative
        # log-likelihood even when the data were not saved with it.
        if getattr(self, "_neg_ll", None) is not None:
            return self._neg_ll
        if getattr(self, "data", None) is None:
            raise ValueError("Must have been fit with data")

        return self._neg_ll

    def bic(self) -> float:
        r"""

        The Bayesian Information Criterion (BIC) for the model, if it
        was fit with the ``fit()`` method. Not available if fit with the
        ``from_params()`` method.

        Returns
        -------

        bic : float
            The BIC of the model

        Examples
        --------

        >>> from surpyval import Weibull
        >>> import numpy as np
        >>> np.random.seed(1)
        >>> x = Weibull.random(100, 10, 3)
        >>> model = Weibull.fit(x)
        >>> model.bic()
        np.float64(534.2640532197888)

        References
        ----------

        `Bayesian Information Criterion for Censored Survival Models
        <https://www.jstor.org/stable/2677130>`_.

        Notes
        -----
        The sample size :math:`d` in the penalty :math:`k \ln d` is the
        number of observed failures: every unit whose failure was observed,
        exactly or within a left- or interval-censored window, weighted by
        its count; right-censored units add nothing. Should that be zero,
        the number of units is used instead, so the criterion is always
        finite. Every model in SurPyval that reports a BIC counts
        :math:`d` this way (see "Information criteria" in the Parametric
        Estimation guide).
        """
        if hasattr(self, "_bic"):
            return self._bic
        n_penalty = self._ic_sample_size()
        self._bic = self._ic_k() * np.log(n_penalty) + 2 * self.neg_ll()
        return self._bic

    def aic(self) -> float:
        r"""
        The Aikake Information Criterion (AIC) for the model, if it was
        fit with the ``fit()`` method. Not available if fit with the
        ``from_params()`` method.

        Returns
        -------

        aic : float
            The AIC of the model

        Examples
        --------

        >>> from surpyval import Weibull
        >>> import numpy as np
        >>> np.random.seed(1)
        >>> x = Weibull.random(100, 10, 3)
        >>> model = Weibull.fit(x)
        >>> model.aic()
        529.0537128478127
        """
        if hasattr(self, "_aic"):
            return self._aic
        self._aic = 2 * self._ic_k() + 2 * self.neg_ll()
        return self._aic

    def aic_c(self) -> float:
        r"""
        The Corrected Aikake Information Criterion (AIC) for the model,
        if it was fit with the ``fit()`` method. Not available if fit with
        the ``from_params()`` method.

        Returns
        -------

        aic_c : float
            The Corrected AIC of the model

        Examples
        --------

        >>> from surpyval import Weibull
        >>> import numpy as np
        >>> np.random.seed(1)
        >>> x = Weibull.random(100, 10, 3)
        >>> model = Weibull.fit(x)
        >>> model.aic_c()
        529.1774241880189

        Notes
        -----
        The correction :math:`(2k^2 + 2k)/(N - k - 1)` uses the same
        sample size :math:`N` as :meth:`bic`: the number of observed
        failures (exact, left- or interval-censored, weighted by their
        counts), or the number of units when there is none. Right-censored
        units do not add to it. The correction only exists for
        :math:`N > k + 1`; otherwise the corrected criterion is undefined
        and ``nan`` is returned (the formula gave ``inf`` at
        :math:`N = k + 1`, and a value *below* ``aic()`` for smaller
        :math:`N`, which would have won a comparison).
        """
        if hasattr(self, "_aic_c"):
            return self._aic_c
        k = self._ic_k_aic_c()
        n = self._ic_sample_size()
        if n - k - 1 <= 0:
            self._aic_c = float("nan")
        else:
            self._aic_c = self.aic() + (2 * k**2 + 2 * k) / (n - k - 1)
        return self._aic_c
