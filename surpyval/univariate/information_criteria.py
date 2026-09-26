"""Shared information-criterion methods (#298).

``Parametric`` and ``ParametricRegressionModel`` carried near-identical
``neg_ll``/``aic``/``bic``/``aic_c`` implementations. They differed only
in how the fitted data is stored (an xcnt dict versus a ``SurpyvalData``
object), which the ``_ic_counts`` hook absorbs. ``_ic_k`` is the parameter
count of the ``aic``/``bic`` penalty and (through ``_ic_k_aic_c``) of the
small-sample correction: ``self.k`` by default, and the number of
*estimated* parameters for univariate parametric models, which exclude the
ones fixed at fit time. Regression models set ``self.k`` to their number
of estimated parameters directly.
"""

import numpy as np


class InformationCriteriaMixin:
    """Log-likelihood based model-selection criteria.

    Requires the host class to set ``self.k`` (penalised parameter
    count), ``self._neg_ll`` and ``self.data`` when fitted, and to
    implement ``_ic_counts`` returning ``(n_observed, n_total)`` — the
    weighted count of exactly-observed events and of all observations.
    """

    # Set by the host class when fitted; annotated (not assigned) so the
    # hasattr-based caching below still works.
    k: int
    _neg_ll: float
    _bic: float
    _aic: float
    _aic_c: float

    def _ic_counts(self) -> tuple[int, int]:
        raise NotImplementedError

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
        The sample size in the penalty :math:`k \ln d` is the number of
        observed events :math:`d` that each model's ``_ic_counts`` reports
        (for a univariate parametric model, every unit whose failure was
        observed, exactly or within a left- or interval-censored window).
        Should that be zero, the number of units is used instead, so the
        criterion is always finite: :math:`\ln 0` made it :math:`-\infty`,
        and so the "best" model of any comparison.
        """
        if hasattr(self, "_bic"):
            return self._bic
        n_observed, n_total = self._ic_counts()
        n_penalty = n_observed if n_observed > 0 else n_total
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
        np.float64(529.1774241880189)

        Notes
        -----
        The correction :math:`(2k^2 + 2k)/(N - k - 1)`, with :math:`N` the
        number of observations, only exists for :math:`N > k + 1`. With
        fewer observations the corrected criterion is undefined and
        ``nan`` is returned (the formula gave ``inf`` at
        :math:`N = k + 1`, and a value *below* ``aic()`` for smaller
        :math:`N`, which would have won a comparison).
        """
        if hasattr(self, "_aic_c"):
            return self._aic_c
        k = self._ic_k_aic_c()
        _, n = self._ic_counts()
        if n - k - 1 <= 0:
            self._aic_c = float("nan")
        else:
            self._aic_c = self.aic() + (2 * k**2 + 2 * k) / (n - k - 1)
        return self._aic_c
