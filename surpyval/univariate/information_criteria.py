"""Shared information-criterion methods (#298).

``Parametric`` and ``ParametricRegressionModel`` carried near-identical
``neg_ll``/``aic``/``bic``/``aic_c`` implementations. They differed only
in how the fitted data is stored (an xcnt dict versus a ``SurpyvalData``
object) and in one deliberate convention difference: the small-sample
AIC correction term uses ``self.k`` for univariate models but
``len(self.params)`` for regression models. Both are preserved exactly
through the ``_ic_counts``/``_ic_k_aic_c`` hooks so fitted values do not
change.
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

    def _ic_counts(self):
        raise NotImplementedError

    def _ic_k_aic_c(self):
        # The parameter count used in the aic_c correction term.
        return self.k

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
        262.52685642385734
        """
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
        534.2640532196908

        References
        ----------

        `Bayesian Information Criterion for Censored Survival Models
        <https://www.jstor.org/stable/2677130>`_.

        """
        if hasattr(self, "_bic"):
            return self._bic
        n_observed, _ = self._ic_counts()
        self._bic = self.k * np.log(n_observed) + 2 * self.neg_ll()
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
        529.0537128477147
        """
        if hasattr(self, "_aic"):
            return self._aic
        self._aic = 2 * self.k + 2 * self.neg_ll()
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
        529.1774241879209
        """
        if hasattr(self, "_aic_c"):
            return self._aic_c
        k = self._ic_k_aic_c()
        _, n = self._ic_counts()
        self._aic_c = self.aic() + (2 * k**2 + 2 * k) / (n - k - 1)
        return self._aic_c
