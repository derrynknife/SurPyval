"""
The fitted shared-frailty model returned by :class:`FrailtyFitter`.

A shared-frailty model is a proportional-hazards model with an extra random
multiplier ``u`` on the hazard that is *shared* by every observation in a
group (a lot, a site, a repairable unit):

.. math::
    h(t \\mid Z, u) = u \\, h_0(t) \\, e^{\\beta' Z},

with the frailties drawn once per group from a Gamma distribution of mean 1 and
variance :math:`\\theta` (``theta``), or, with ``family="lognormal"``, as
``u = exp(w)`` with ``w`` normal of mean 0 and variance :math:`\\theta`.
``theta`` measures the unexplained between-group variability; ``theta = 0``
recovers an ordinary parametric PH model.

Because the frailty enters multiplicatively on the *cumulative* hazard, a Gamma
frailty integrates out of a group's likelihood in closed form, and the same
conjugacy gives each group's posterior frailty in closed form (a log-normal
frailty's are computed by quadrature) -- so prediction comes in two flavours:

* **marginal** (population-averaged), integrating the frailty out --
  ``S(t \\mid Z) = (1 + \\theta\\, e^{\\beta'Z} H_0(t))^{-1/\\theta}`` for the
  gamma frailty -- the right curve for a new unit from an unknown group; and
* **conditional**, on a supplied frailty value or on an *observed* group's
  posterior frailty ``S(t \\mid Z, u) = e^{-u e^{\\beta'Z} H_0(t)}``.
"""

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import ndtri as _z

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
    to_native,
)
from surpyval.univariate.information_criteria import (
    InformationCriteriaMixin,
    ic_sample_size,
)
from surpyval.utils import is_missing_event
from surpyval.utils.covariates import loaded_coefficient_names
from surpyval.utils.linalg import standard_errors_of
from surpyval.utils.no_maximum import maximum_entry, restored_maximum
from surpyval.utils.validation import (
    BOUNDS,
    check_option,
    no_covariance_error,
)

from .._concordance import ConcordanceMixin
from .._prediction import ConditionalSurvivalMixin
from ..regression_data import (
    prepare_Z,
    restore_covariate_meta,
    serialise_covariate_meta,
)
from .families import frailty_cv2, kendall_tau, lognormal_log_integral

if TYPE_CHECKING:
    import pandas as pd


def _standard_error(variance: Any) -> np.ndarray:
    """``sqrt`` of a variance, ``nan`` where it is negative.

    With ``theta`` on its boundary at zero the numerical information matrix
    is barely invertible and a variance can come out negative: the standard
    error is then unavailable, which ``nan`` says without the
    invalid-value warning a bare ``sqrt`` would emit.
    """
    variance = np.asarray(variance, dtype=float)
    return np.sqrt(np.where(variance >= 0, variance, np.nan))


class _SharedFrailty(
    ConditionalSurvivalMixin,
    InformationCriteriaMixin,
    ConcordanceMixin,
    SerialisableMixin,
):
    """What the fitted shared-frailty models have in common: the frailty
    (its family, variance ``theta`` and each group's posterior), the
    coefficients, the marginal and conditional predictions, and the
    parameter table. A subclass gives the baseline (:meth:`_H0`,
    :meth:`_h0`, :meth:`_baseline_names`)."""

    def __init__(self) -> None:
        self.kind = "Frailty"
        self.family = "gamma"
        self.dist: Any = None
        self.dist_params: np.ndarray = np.array([])
        self.beta: np.ndarray = np.array([])
        self.theta: float = 0.0
        self.k_dist: int = 0
        self.feature_names: "list[str] | None" = None
        self.formula: "str | None" = None
        self._model_spec: Any = None
        self.group_labels: list = []
        self.frailties: dict = {}
        self._covariance: "np.ndarray | None" = None
        self.parameter_names: "list[str]" = []
        self.n_obs: int = 0
        self.n_events: int = 0
        self.n_groups: int = 0
        # Count-weighted numbers of events and observations, from which
        # the sample size of the information criteria follows
        # (``n_events``/``n_obs`` count rows).
        self.n_events_weighted: float = 0.0
        self.n_obs_weighted: float = 0.0
        self._neg_ll: float = 0.0
        # The number of estimated parameters -- the baseline, the
        # coefficients and theta -- the ``k`` of the information criteria.
        self.k: int = 0
        # What the fit reached, one of ``MAXIMUM_STATES``
        # (``surpyval.utils.no_maximum``), as its warnings say; "unknown"
        # for a model restored from a dict saved without it.
        self.maximum: str = "unknown"

    def covariance(self) -> np.ndarray:
        """The parameters' covariance, in the order of
        ``parameter_names`` (#605). It was an attribute before v0.23.

        Raises a ``ValueError`` where the model has none (its information
        was singular)."""
        if self._covariance is None:
            raise no_covariance_error()
        return np.asarray(self._covariance)

    # -- information criteria (InformationCriteriaMixin) -------------------

    def _ic_sample_size_from_data(self) -> float:
        # The shared rule (ic_sample_size) on the fitted data, which the
        # stored weighted counts summarise: a frailty fit takes only
        # events (c=0) and right-censored rows (c=1).
        n_censored = self.n_obs_weighted - self.n_events_weighted
        return ic_sample_size([0, 1], [self.n_events_weighted, n_censored])

    # -- covariate / frailty resolution ------------------------------------

    def _concordance_risk(self, x: np.ndarray, Z: Any) -> np.ndarray:
        # The conditional log hazard ratio beta'Z (a unit of mean frailty).
        if self.beta.size == 0:
            return np.zeros(x.size)
        Zp = prepare_Z(Z, self.feature_names, self._model_spec)
        Zp = np.asarray(Zp, dtype=float).reshape(x.size, -1)
        return Zp @ np.where(np.isnan(self.beta), 0.0, self.beta)

    def _eta(self, Z: Any) -> np.ndarray:
        """The hazard multiplier ``exp(beta'Z)`` for a covariate setting."""
        if self.beta.size == 0:
            return np.array(1.0)
        if Z is None:
            raise ValueError(
                "This model was fit with covariates; 'Z' is required."
            )
        Zp = prepare_Z(Z, self.feature_names, self._model_spec)
        Zp = np.atleast_2d(np.asarray(Zp, dtype=float))
        # An aliased coefficient (nan, #476) is predicted with as 0.
        eta = np.exp(Zp @ np.where(np.isnan(self.beta), 0.0, self.beta))
        return eta[0] if eta.shape[0] == 1 else eta

    def _resolve_frailty(self, group: Any, frailty: Any) -> "float | None":
        """Return the frailty value to condition on, or ``None`` (marginal)."""
        if group is not None and frailty is not None:
            raise ValueError("Pass at most one of 'group' or 'frailty'.")
        if frailty is not None:
            return float(frailty)
        if group is not None and is_missing_event(group):
            # A missing group label (NaN, pandas NA) predicts nan, as a
            # missing stratum does in a stratified Cox model.
            return float("nan")
        if group is not None:
            key = group
            if key not in self.frailties:
                key = str(group)
            if key not in self.frailties:
                raise KeyError(
                    f"Unknown group {group!r}; known groups are "
                    f"{list(self.frailties)[:8]}..."
                )
            return float(self.frailties[key])
        return None

    # -- prediction --------------------------------------------------------

    def _cumulative(self, s: Any, u: "float | None") -> np.ndarray:
        """The cumulative hazard of a unit whose ``eta H0`` is ``s``:
        ``u s`` given its frailty ``u``, or, for ``u`` ``None``, the
        marginal ``-log E[exp(-u s)]`` (the frailty's Laplace transform)."""
        if u is not None:
            return u * s
        if self.family == "lognormal":
            return -lognormal_log_integral(0.0, s, self.theta)
        if self.theta < 1e-12:
            # theta -> 0 is the no-frailty PH limit
            # log(1 + theta s)/theta -> s; dividing by a zero theta
            # (e.g. frailty-free data, or a restored model) gave NaN
            # (#262).
            return s
        return np.log1p(self.theta * s) / self.theta

    def Hf(
        self, x: Any, Z: Any = None, group: Any = None, frailty: Any = None
    ) -> np.ndarray:
        """Cumulative hazard (marginal, or conditional on a frailty)."""
        x = np.asarray(x, dtype=float)
        s = self._eta(Z) * self._H0(x)
        return self._cumulative(s, self._resolve_frailty(group, frailty))

    def sf(
        self, x: Any, Z: Any = None, group: Any = None, frailty: Any = None
    ) -> np.ndarray:
        """Survival function (marginal by default)."""
        return np.exp(-self.Hf(x, Z, group=group, frailty=frailty))

    def ff(
        self, x: Any, Z: Any = None, group: Any = None, frailty: Any = None
    ) -> np.ndarray:
        """CDF / failure function."""
        # 1 - exp(-H) without the cancellation of 1 - sf for a small H
        return -np.expm1(-self.Hf(x, Z, group=group, frailty=frailty))

    def hf(
        self, x: Any, Z: Any = None, group: Any = None, frailty: Any = None
    ) -> np.ndarray:
        """Hazard function (marginal by default)."""
        x = np.asarray(x, dtype=float)
        eta = self._eta(Z)
        H0 = self._H0(x)
        h0 = self._h0(x)
        u = self._resolve_frailty(group, frailty)
        if u is None and self.family == "lognormal":
            # eta h0 times the mean frailty of the survivors to t
            s = eta * H0
            mean_u = np.exp(
                lognormal_log_integral(1.0, s, self.theta)
                - lognormal_log_integral(0.0, s, self.theta)
            )
            return eta * h0 * mean_u
        if u is None:
            return eta * h0 / (1.0 + self.theta * eta * H0)
        return u * eta * h0

    def df(
        self, x: Any, Z: Any = None, group: Any = None, frailty: Any = None
    ) -> np.ndarray:
        """Density function."""
        return self.hf(x, Z, group=group, frailty=frailty) * self.sf(
            x, Z, group=group, frailty=frailty
        )

    # -- inference ---------------------------------------------------------

    @property
    def frailty_variance(self) -> float:
        """
        The variance of the frailty scaled to mean 1, :math:`\\mathrm{Var}(u)
        / E(u)^2`, which compares the families: :math:`\\hat\\theta` for the
        gamma frailty (mean 1, variance ``theta``) and
        :math:`e^{\\hat\\theta} - 1` for the log-normal (``theta`` the
        variance of :math:`\\log u`). A baseline whose scale multiplies the
        hazard (Weibull, Exponential) absorbs the frailty's mean, so this
        is the spread the data identify.

        Examples
        --------
        >>> from surpyval import FrailtyModel
        >>> model = FrailtyModel()
        >>> model.family, model.theta = "lognormal", 0.5
        >>> round(model.frailty_variance, 4)
        0.6487
        """
        return frailty_cv2(self.family, float(self.theta))

    @property
    def kendall_tau(self) -> float:
        """
        Kendall's tau between the event times of two units of one group
        (no covariates, no censoring): the within-group dependence the
        frailty induces, on the same scale for every family (Hougaard 2000,
        section 4.2). ``theta / (theta + 2)`` for the gamma frailty; by
        quadrature for the log-normal.

        Examples
        --------
        >>> from surpyval import FrailtyModel
        >>> model = FrailtyModel()
        >>> model.theta = 0.5
        >>> model.kendall_tau
        0.2
        """
        return kendall_tau(self.family, float(self.theta))

    @property
    def aliased(self) -> np.ndarray:
        """The columns of ``Z`` whose coefficients the data cannot
        determine (#476): a constant column where the baseline's scale is
        the intercept, or a linear combination of the others. Their
        ``beta`` is ``nan`` (R's ``NA``), and predictions take it as 0."""
        return np.flatnonzero(np.isnan(np.asarray(self.beta, dtype=float)))

    def standard_errors(self) -> np.ndarray:
        """Wald standard errors of the parameters, an array in the order of
        ``parameter_names`` and of ``covariance()`` (``nan`` where a
        variance is not positive), as on every model.

        .. versionchanged:: 0.23
           An array (#613); it was a dict keyed by parameter name. The
           standard error of ``theta``, the last parameter, is
           ``standard_errors()[-1]``.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullFrailty
        >>> rng = np.random.default_rng(0)
        >>> group = np.repeat(np.arange(30), 4)
        >>> Z = rng.normal(size=(120, 1))
        >>> u = rng.gamma(2.0, 0.5, 30)[group]
        >>> x = 10 * (rng.exponential(size=120) / (u * np.exp(0.5 * Z[:, 0])))
        >>> model = WeibullFrailty.fit(x, Z, groups=group)
        >>> se = model.standard_errors()
        >>> se.shape == (len(model.parameter_names),)
        True
        """
        return standard_errors_of(self.covariance())

    def param_cb(
        self,
        name: str,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> np.ndarray:
        """Wald confidence bound on a named parameter.

        The bound is formed on a scale chosen from the parameter's support (log
        for the positive baseline parameters and ``theta``, natural for the
        unbounded coefficients) so the interval stays valid.
        """
        cov = self.covariance()
        if name not in self.parameter_names:
            raise ValueError(
                "Unknown parameter {!r}; expected one of {}".format(
                    name, self.parameter_names
                )
            )
        idx = self.parameter_names.index(name)
        est = self._param_vector()[idx]
        se = float(_standard_error(cov[idx, idx]))
        positive = name == "theta" or (
            idx < self.k_dist and self.dist.bounds[idx][0] == 0
        )
        check_option("bound", bound, BOUNDS)
        if bound == "two-sided":
            q = _z(1 - alpha_ci / 2)
            signs = np.array([-1.0, 1.0])
        elif bound == "lower":
            q = _z(1 - alpha_ci)
            signs = np.array([-1.0])
        else:
            q = _z(1 - alpha_ci)
            signs = np.array([1.0])
        if positive:
            if est <= 0:
                # A boundary estimate (theta -> 0: no detectable frailty)
                # has no log-scale Wald interval; dividing by zero gave
                # NaN/ZeroDivision (#262).
                return np.zeros_like(signs, dtype=float)
            # Near the boundary ``se / est`` is huge (theta ~ 1e-14 on
            # frailty-free data), and the upper bound really is beyond any
            # float: ``exp`` of the log-scale half-width is inf, the lower
            # bound underflows to 0. That is the answer, not an accident,
            # so the overflow is not reported as one.
            with np.errstate(over="ignore"):
                return est * np.exp(signs * q * se / est)
        return est + signs * q * se

    @property
    def params(self) -> np.ndarray:
        """Every estimated parameter, in the order of ``parameter_names``: the
        baseline's parameters, the coefficients, then ``theta``."""
        return self._param_vector()

    def _param_vector(self) -> np.ndarray:
        return np.concatenate(
            [self.dist_params, self.beta, [self.theta]]
        ).astype(float)

    def summary(self, alpha_ci: float = 0.05) -> "pd.DataFrame":
        """
        The parameter table, in the layout of the parametric regression
        models' :meth:`summary` (#484): the baseline distribution's
        parameters, the regression coefficients and the frailty variance
        ``theta``, each with its standard error and a two-sided
        ``1 - alpha_ci`` Wald interval; for the coefficients also the
        hazard ratio ``exp(coef)`` (conditional on the frailty), the Wald
        statistic ``z`` and its two-sided p-value. The coefficients are
        named by ``feature_names`` for a model fitted with ``fit_from_df``.

        The baseline parameters' and ``theta``'s intervals are those of
        :meth:`param_cb`, which stay in the parameter's support. Without a
        stored covariance there are no standard errors or intervals
        (``nan``), nor for an aliased coefficient (#476), whose value is
        ``nan`` too.

        .. versionchanged:: 0.22
           Returns a ``DataFrame``; it returned the text ``repr`` prints.

        Parameters
        ----------
        alpha_ci : float, optional
            The intervals' total tail probability. Default 0.05.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``(part, name)``, ``part`` one of ``"baseline"``,
            ``"coefficients"`` or ``"frailty"``, with the columns of
            ``CoxPH``'s :meth:`summary`.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullFrailty
        >>> rng = np.random.default_rng(0)
        >>> g = np.repeat(np.arange(30), 5)
        >>> z = rng.normal(size=g.size)
        >>> u = rng.gamma(2.0, 0.5, 30)[g]
        >>> x = 10 * rng.exponential(size=g.size) / (u * np.exp(0.5 * z))
        >>> model = WeibullFrailty.fit(x=x, Z=z[:, None], groups=g)
        >>> list(model.summary().index)  # doctest: +NORMALIZE_WHITESPACE
        [('baseline', 'alpha'), ('baseline', 'beta'),
         ('coefficients', 'coef_0'), ('frailty', 'theta')]
        """
        import pandas as pd

        from .._summary import coefficient_table

        params = self._param_vector()
        k = self.k_dist
        n_beta = self.beta.size
        se = np.full(params.shape, np.nan)
        if self._covariance is not None:
            with np.errstate(all="ignore"):
                se = np.asarray(
                    _standard_error(np.diag(self._covariance)), dtype=float
                )
        level = "{:g}%".format(100 * (1 - alpha_ci))
        columns = list(coefficient_table([], [], [], alpha_ci).columns)

        def others(indices: "list[int]") -> pd.DataFrame:
            # A parameter with a support: its estimate, standard error and
            # the support-respecting interval of ``param_cb``.
            rows = []
            for i in indices:
                bounds = np.full(2, np.nan)
                if np.isfinite(se[i]):
                    try:
                        with (
                            warnings.catch_warnings(),
                            np.errstate(all="ignore"),
                        ):
                            warnings.simplefilter("ignore")
                            bounds = np.asarray(
                                self.param_cb(
                                    self.parameter_names[i], alpha_ci
                                ),
                                dtype=float,
                            ).ravel()
                    except (ValueError, ArithmeticError):
                        pass
                rows.append(
                    {
                        "coef": params[i],
                        "se(coef)": se[i],
                        "coef lower " + level: bounds[0],
                        "coef upper " + level: bounds[-1],
                    }
                )
            return pd.DataFrame(rows, columns=columns)

        names = list(self.parameter_names[k : k + n_beta])
        coef = slice(k, k + n_beta)
        coefs = coefficient_table(names, params[coef], se[coef], alpha_ci)
        parts = [coefs.reset_index(drop=True), others([k + n_beta])]
        if k:
            parts.insert(0, others(list(range(k))))
        table = pd.concat(parts)
        index = (
            [("baseline", name) for name in self._baseline_names()]
            + [("coefficients", name) for name in names]
            + [("frailty", "theta")]
        )
        table.index = pd.MultiIndex.from_tuples(index, names=["part", "name"])
        return table

    def _family_line(self) -> str:
        if self.family == "lognormal":
            return (
                "lognormal (log u ~ N(0, theta)); Var(u)/E(u)^2 = "
                "{:.4g}, Kendall's tau = {:.4g}".format(
                    self.frailty_variance, self.kendall_tau
                )
            )
        return "gamma (mean 1, variance theta); Kendall's tau = {:.4g}".format(
            self.kendall_tau
        )

    # -- the baseline, given by a subclass --------------------------------

    def _H0(self, x: np.ndarray) -> np.ndarray:
        """The baseline cumulative hazard at ``x``."""
        raise NotImplementedError

    def _h0(self, x: np.ndarray) -> np.ndarray:
        """The baseline hazard at ``x``."""
        raise NotImplementedError

    def _baseline_names(self) -> "list[str]":
        """The names of the baseline's parameters, in ``params``."""
        raise NotImplementedError


class FrailtyModel(_SharedFrailty):
    """A fitted shared-frailty proportional-hazards model.

    See :class:`FrailtyFitter` for how one is produced. Prediction methods
    (:meth:`sf`, :meth:`ff`, :meth:`hf`, :meth:`Hf`, :meth:`df`) return the
    *marginal* (population) curve by default; pass ``group=`` to condition on
    an observed group's posterior frailty, or ``frailty=`` to condition on a
    supplied frailty value.

    :meth:`neg_ll`, :meth:`aic`, :meth:`bic` and :meth:`aic_c` use the
    marginal likelihood and count every estimated parameter (baseline,
    coefficients and ``theta``), on the same data conventions as the
    parametric regression models, so a frailty fit can be compared directly
    with the proportional-hazards fit (``WeibullPH`` for ``WeibullFrailty``)
    of the same data -- the model it reduces to at ``theta = 0``.

    ``params`` is every estimated parameter in one vector, in the order of
    ``parameter_names``: the baseline distribution's parameters, then the
    covariate coefficients (each named by its covariate's column, else
    ``coef_0``, ``coef_1``, ...; #614), then the frailty
    variance ``theta`` -- the order of :meth:`standard_errors` and of the
    stored ``covariance``. ``dist_params``, ``beta`` and ``theta`` hold the
    same values by part.

    Examples
    --------
    Thirty groups of six units, each group sharing a gamma frailty:

    >>> import numpy as np
    >>> from surpyval import WeibullFrailty
    >>> rng = np.random.default_rng(4)
    >>> groups = np.repeat(np.arange(30), 6)
    >>> u = rng.gamma(2.0, 0.5, 30)[groups]
    >>> Z = rng.binomial(1, 0.5, (180, 1))
    >>> H = rng.exponential(1, 180) / (u * np.exp(0.5 * Z[:, 0]))
    >>> x = 10 * H**0.5  # Weibull baseline, alpha 10 and beta 2
    >>> model = WeibullFrailty.fit(x, Z=Z, groups=groups)
    >>> round(model.theta, 3)
    0.432
    >>> model.parameter_names
    ['alpha', 'beta', 'coef_0', 'theta']
    >>> model.params.round(3)
    array([10.442,  1.962,  0.399,  0.432])

    The population curve, and the curve for group 0 given its posterior
    frailty:

    >>> model.sf([5, 10], [1]).round(4)
    array([0.721 , 0.3411])
    >>> model.sf([5, 10], [1], group=0).round(4)
    array([0.7226, 0.2821])
    """

    #: The rows fitted, ``{"x", "c", "w", "Z", "inv"}`` (``Z`` the columns
    #: whose coefficients were estimated, ``inv`` each row's group), for
    #: ``param_cb(method="lr")``; not saved by :meth:`to_dict`.
    _fit_data: "dict | None" = None
    #: The likelihood-ratio searches of ``param_cb(method="lr")``, with
    #: what they have found, while the parameters stay as they are; not
    #: pickled (``__getstate__``).
    _lr_search: Any = None

    def __getstate__(self) -> dict:
        # The searches' caches are rebuilt where they are needed (#617).
        state = dict(self.__dict__)
        state.pop("_lr_search", None)
        return state

    # -- inference ---------------------------------------------------------

    def param_cb(
        self,
        name: str,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "wald",
    ) -> np.ndarray:
        """Confidence bound(s) on a named parameter.

        Two methods, as for the parametric regression models; ``"wald"``
        is the default:

        - ``"wald"`` -- from the stored covariance, on a scale chosen from
          the parameter's support (log for the positive baseline
          parameters and ``theta``, natural for the unbounded
          coefficients) so the interval stays valid.
        - ``"lr"`` -- the profile-likelihood (likelihood-ratio) interval
          (#617): the values whose profile deviance of the marginal
          likelihood, every other parameter re-fitted, stays below the
          :math:`\\chi^2_1` critical value (aliases ``"likelihood"``,
          ``"likelihood-ratio"``, ``"profile"``). It need not be symmetric
          about the estimate; where the deviance stays below the critical
          value to the edge of the space (``theta`` down to 0, no
          detectable frailty), the bound is that edge. A side that cannot
          be found is ``nan``, with a warning. It needs the data the model
          was fitted to, which a model restored from a dict does not keep.

        ``theta = 0`` is the edge of its space: where the true ``theta`` is
        0 the deviance of ``theta`` is half a point mass at 0 and half a
        :math:`\\chi^2_1` (Self and Liang 1987), so the likelihood-ratio
        interval, which takes the :math:`\\chi^2_1` critical value, is
        conservative there.

        Parameters
        ----------
        name : str
            One of :attr:`parameter_names`.
        alpha_ci : float, optional
            Total tail probability of the bound(s). Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds are returned as ``[lower, upper]``.
        method : {'wald', 'lr'}, optional
            As above. Default ``'wald'``.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullFrailty
        >>> rng = np.random.default_rng(4)
        >>> groups = np.repeat(np.arange(30), 6)
        >>> u = rng.gamma(2.0, 0.5, 30)[groups]
        >>> Z = rng.binomial(1, 0.5, (180, 1))
        >>> H = rng.exponential(1, 180) / (u * np.exp(0.5 * Z[:, 0]))
        >>> model = WeibullFrailty.fit(10 * H**0.5, Z=Z, groups=groups)
        >>> model.param_cb("theta").round(3)
        array([0.226, 0.825])
        >>> model.param_cb("theta", method="lr").round(3)
        array([0.22 , 0.819])
        """
        from .._likelihood_ratio import is_lr

        if not is_lr(method):
            return super().param_cb(name, alpha_ci, bound)
        from .._likelihood_ratio import profile_interval

        check_option("bound", bound, BOUNDS)
        if name not in self.parameter_names:
            raise ValueError(
                "Unknown parameter {!r}; expected one of {}".format(
                    name, self.parameter_names
                )
            )
        return profile_interval(self._lr_region(), name, alpha_ci, bound)

    def _lr_region(self) -> Any:
        """The likelihood-ratio searches over the marginal likelihood of
        every parameter (``LikelihoodRegion``), kept while the parameters
        are as they are; an aliased coefficient is held (its bound is
        ``nan``)."""
        from .._likelihood_ratio import LikelihoodRegion
        from .frailty_fitter import FrailtyFitter

        data = self._fit_data
        if data is None:
            raise ValueError(
                "Likelihood-ratio bounds need the data the model was fitted "
                "to, which a model restored from a dict does not keep; use "
                "method='wald'."
            )
        # (an aliased coefficient, nan, is held at 0)
        params = np.nan_to_num(self._param_vector(), nan=0.0)
        point = params.tobytes()
        search = self._lr_search
        if search is not None and search.point == point:
            return search
        k = self.k_dist
        aliased = k + self.aliased
        n_beta = self.beta.size - aliased.size
        kept = np.setdiff1d(np.arange(params.size), aliased)
        fitter = FrailtyFitter.create(self.dist, self.family)

        def neg_ll(theta: np.ndarray) -> float:
            return fitter._neg_ll_natural(
                theta[kept],
                data["x"],
                data["c"],
                data["w"],
                data["Z"],
                data["inv"],
                n_beta,
            )

        bounds = [
            *self.dist.bounds,
            *((None, None),) * self.beta.size,
            (0, None),
        ]
        search = LikelihoodRegion(
            neg_ll,
            params,
            list(self.parameter_names),
            bounds,
            set(aliased.tolist()),
            self._covariance,
            point,
        )
        self._lr_search = search
        return search

    # -- the parametric baseline -----------------------------------------

    def _H0(self, x: np.ndarray) -> np.ndarray:
        return self.dist.Hf(x, *self.dist_params)

    def _h0(self, x: np.ndarray) -> np.ndarray:
        return self.dist.hf(x, *self.dist_params)

    def _baseline_names(self) -> "list[str]":
        return list(self.dist.parameter_names)

    def __repr__(self) -> str:
        from .._summary import coefficient_repr, format_table

        out = (
            "Shared-Frailty Regression SurPyval Model"
            "\n========================================"
            f"\nDistribution        : {self.dist.name}"
            f"\nFrailty             : {self._family_line()}"
            f"\nGroups              : {self.n_groups}"
            f"  (observations {self.n_obs}, events {self.n_events})"
        )
        if self.dist is None or not self.parameter_names:
            return out
        table = self.summary()
        estimates = {
            "coef": "estimate",
            "se(coef)": "se",
            "coef lower 95%": "lower 95%",
            "coef upper 95%": "upper 95%",
        }

        def block(part: str) -> str:
            rows = table.loc[part].rename(columns=estimates)
            rows.index.name = None
            return format_table(rows, list(estimates.values()))

        out += (
            "\nBaseline            : {} parameters; Wald 95% "
            "intervals\n".format(self.dist.name)
        ) + block("baseline")
        if self.beta.size:
            out += (
                "\nCoefficients        : exp(coef) is the hazard ratio "
                "given the frailty; Wald 95% intervals\n"
            ) + coefficient_repr(table.loc["coefficients"])
        out += "\nFrailty variance    : Wald 95% interval\n" + block("frailty")
        return out

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a plain, JSON-serialisable ``dict``."""
        out: dict[str, Any] = {
            "model": "FrailtyModel",
            "kind": self.kind,
            "family": self.family,
            "distribution": self.dist.name,
            "dist_params": np.asarray(self.dist_params, float).tolist(),
            "beta": np.asarray(self.beta, float).tolist(),
            "theta": float(self.theta),
            "k_dist": int(self.k_dist),
            "param_names": list(self.parameter_names),
            "group_labels": [str(g) for g in self.group_labels],
            "frailties": {str(k): float(v) for k, v in self.frailties.items()},
            "n_obs": int(self.n_obs),
            "n_events": int(self.n_events),
            "n_groups": int(self.n_groups),
            "n_events_weighted": float(self.n_events_weighted),
            "n_obs_weighted": float(self.n_obs_weighted),
            "_neg_ll": to_native(self._neg_ll),
            **maximum_entry(self.maximum),
        }
        if self._covariance is not None:
            out["covariance"] = np.asarray(self._covariance, float).tolist()
        serialise_covariate_meta(self, out)
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "FrailtyModel":
        """Rebuild a model from a :meth:`to_dict` dictionary."""
        import surpyval as surv
        from surpyval.univariate.parametric.parametric_fitter import (
            ParametricFitter,
        )

        require_model_tag(model_dict, "FrailtyModel", "a frailty model")

        dist = getattr(surv, model_dict["distribution"], None)
        if not isinstance(dist, ParametricFitter):
            raise ValueError(
                f"Unknown distribution '{model_dict['distribution']}'"
            )

        out = cls()
        out.family = model_dict.get("family", "gamma")
        out.dist = dist
        out.dist_params = np.array(model_dict["dist_params"], dtype=float)
        out.beta = np.array(model_dict["beta"], dtype=float)
        out.theta = float(model_dict["theta"])
        out.k_dist = int(model_dict["k_dist"])
        # A dict saved before v0.23 named the coefficients beta_j: they
        # load with the names the model has now (#614).
        out.parameter_names = loaded_coefficient_names(
            model_dict["param_names"],
            out.k_dist,
            out.beta.size,
            model_dict.get("feature_names"),
        )
        out.k = len(out.parameter_names)
        out.group_labels = list(model_dict.get("group_labels", []))
        out.frailties = {
            k: float(v) for k, v in model_dict.get("frailties", {}).items()
        }
        out.n_obs = int(model_dict.get("n_obs", 0))
        out.n_events = int(model_dict.get("n_events", 0))
        out.n_groups = int(model_dict.get("n_groups", 0))
        # Dicts written before these were stored have unit weights.
        out.n_events_weighted = float(
            model_dict.get("n_events_weighted", out.n_events)
        )
        out.n_obs_weighted = float(model_dict.get("n_obs_weighted", out.n_obs))
        out._neg_ll = float(model_dict.get("_neg_ll", 0.0))
        out.maximum = restored_maximum(model_dict)
        if "covariance" in model_dict:
            out._covariance = np.array(model_dict["covariance"], dtype=float)
        restore_covariate_meta(out, model_dict)
        return out
