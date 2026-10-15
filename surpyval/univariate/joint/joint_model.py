"""The fitted joint model of a marker and a lifetime, :class:`JointModel`
(#891)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
from autograd import make_jvp
from autograd.scipy.special import logsumexp
from scipy.stats import norm

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.univariate.information_criteria import (
    InformationCriteriaMixin,
    ic_sample_size,
)
from surpyval.utils.linalg import bound_signs, standard_errors_of
from surpyval.utils.no_maximum import maximum_entry, restored_maximum
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.validation import (
    BOUNDS,
    check_alpha_ci,
    check_option,
    no_covariance_error,
)

from ._likelihood import (
    GH_PREDICT,
    JointArrays,
    adaptive_nodes,
    censored_modes,
    log_cumhaz,
    log_prior,
    log_readings,
)

if TYPE_CHECKING:
    import pandas as pd

#: The functions ``cb`` bounds ('R' and 'F' are 'sf' and 'ff').
_CB_ON = ("sf", "R", "ff", "F")


def parameter_names(names: "list[str]") -> "list[str]":
    """The names of a joint model's parameters, for covariates ``names``:
    the marker's line (``marker_intercept``, ``marker_slope`` and a
    ``marker_<name>`` per covariate), its random effects
    (``var_intercept``, ``var_slope``, ``corr_intercept_slope``), the
    noise of a reading (``sd_noise``), the Weibull baseline (``alpha``,
    ``beta``), the hazard's covariate coefficients (``<name>``) and the
    marker's effect on the log hazard (``marker_effect``)."""
    return (
        ["marker_intercept", "marker_slope"]
        + [f"marker_{name}" for name in names]
        + ["var_intercept", "var_slope", "corr_intercept_slope", "sd_noise"]
        + ["alpha", "beta"]
        + list(names)
        + ["marker_effect"]
    )


def internal_theta(params: Any, p: int, scales: tuple) -> dict:
    """The internal parameters (``_likelihood``'s scale) of the reported
    ``params``, given the ``scales`` ``(time, marker centre, marker
    spread, covariate means)``; differentiable by autograd."""
    s, c, k, zbar = scales
    zbar = np.asarray(zbar, dtype=float)
    intercept, slope = params[0], params[1]
    delta_rep = params[2 : 2 + p]
    v0, v1, corr, sd = params[2 + p : 6 + p]
    alpha, beta = params[6 + p], params[7 + p]
    gamma = params[8 + p : 8 + 2 * p]
    effect = params[8 + 2 * p]
    delta = delta_rep / k
    d00 = v0 / k**2
    d11 = v1 * s**2 / k**2
    L00 = anp.sqrt(d00)
    L10 = corr * anp.sqrt(d11)
    L11 = anp.sqrt(d11 * (1.0 - corr**2))
    return {
        "b0": (intercept - c) / k + anp.dot(delta, zbar),
        "b1": slope * s / k,
        "delta": delta,
        "L00": L00,
        "L10": L10,
        "L11": L11,
        "sigma": sd / k,
        "scale": alpha * anp.exp(-anp.dot(gamma, zbar) / beta) / s,
        "shape": beta,
        "gamma": gamma,
        "assoc": effect * k,
    }


def reported_params(theta: dict, scales: tuple) -> Any:
    """The reported parameters of the internal ``theta`` (the inverse of
    :func:`internal_theta`); differentiable by autograd."""
    s, c, k, zbar = scales
    zbar = np.asarray(zbar, dtype=float)
    L00, L10, L11 = theta["L00"], theta["L10"], theta["L11"]
    d00, d01, d11 = L00**2, L00 * L10, L10**2 + L11**2
    gamma = theta["gamma"]
    return anp.concatenate(
        [
            anp.stack(
                [
                    c + k * (theta["b0"] - anp.dot(theta["delta"], zbar)),
                    k * theta["b1"] / s,
                ]
            ),
            k * theta["delta"],
            anp.stack(
                [
                    k**2 * d00,
                    (k / s) ** 2 * d11,
                    d01 / anp.sqrt(d00 * d11),
                    k * theta["sigma"],
                    s
                    * theta["scale"]
                    * anp.exp(anp.dot(gamma, zbar) / theta["shape"]),
                    theta["shape"],
                ]
            ),
            gamma,
            anp.reshape(theta["assoc"] / k, (1,)),
        ]
    )


class JointModel(InformationCriteriaMixin, SerialisableMixin):
    """
    A fitted joint model of a marker and a lifetime, returned by
    :func:`~surpyval.univariate.joint.joint_ph.JointPH` fitters such as
    ``WeibullJointPH``.

    Each item's marker is a straight line in age with its own intercept
    and slope (bivariate normal across items) plus normal noise on each
    reading; the hazard is Weibull proportional hazards in the line's
    current, noise-free value and the item's covariates:

    .. math::
        m_i(t) = (\\beta_0 + b_{i0}) + (\\beta_1 + b_{i1}) t
        + \\delta' z_i, \\qquad y_{ij} = m_i(t_{ij}) + \\varepsilon_{ij},

    .. math::
        h_i(t) = \\frac{\\beta}{\\alpha}\\left(\\frac{t}{\\alpha}
        \\right)^{\\beta - 1} \\exp\\left(\\gamma' z_i + \\zeta
        \\left(m_i(t) - c\\right)\\right),

    with :math:`b_i \\sim N(0, D)` and
    :math:`\\varepsilon_{ij} \\sim N(0, \\sigma^2)`. The marker enters the
    hazard centred at ``marker_center`` (:math:`c`, the mean reading of
    the data), so ``alpha`` and ``beta`` are the baseline of an item with
    its marker at :math:`c` and its covariates at 0.

    ``params`` holds, in the order of ``parameter_names``: the marker's
    ``marker_intercept``, ``marker_slope`` and ``marker_<covariate>``
    (:math:`\\beta_0, \\beta_1, \\delta`); the random effects'
    ``var_intercept``, ``var_slope`` and ``corr_intercept_slope``; the
    readings' ``sd_noise`` (:math:`\\sigma`); the Weibull ``alpha`` and
    ``beta``; the hazard's covariate coefficients (:math:`\\gamma`, named
    by their covariates); and ``marker_effect`` (:math:`\\zeta`, the
    change in the log hazard per unit of the marker). ``neg_ll()`` is the
    negative log-likelihood of the readings and the lifetimes together
    (so ``aic()`` and ``bic()`` compare joint models of the same data,
    not a joint model with a survival-only one).

    ``sf(x, item)`` predicts for one item still running: its survival to
    each age ``x`` given its readings so far and that it was running at
    its right censored age, over the distribution of its line given
    those; ``cb`` bounds it.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import ItemLifetime, SurpyvalData, WeibullJointPH
    >>> rng = np.random.default_rng(3)
    >>> items = []
    >>> for k in range(60):
    ...     start, rate = rng.normal(2.0, 0.4), rng.normal(0.02, 0.005)
    ...     life = 100 * rng.weibull(2.0) * np.exp(-0.9 * (start - 2))
    ...     end = min(life, 120.0)
    ...     ages = np.arange(0.0, end, 10.0)
    ...     readings = start + rate * ages + rng.normal(0, 0.2, ages.size)
    ...     outcome = "observed" if life < 120 else "right_censored"
    ...     items.append(
    ...         ItemLifetime(k, **{outcome: end},
    ...                      markers=dict(zip(ages, readings)))
    ...     )
    >>> model = WeibullJointPH.fit(SurpyvalData.from_items(items))
    >>> model.parameter_names[-1]
    'marker_effect'
    >>> new = ItemLifetime("new", right_censored=40, markers={0: 2.5, 30: 3.4})
    >>> model.sf([30, 40, 80], new).shape
    (3,)
    """

    #: The baseline distribution (``Weibull``).
    dist: Any
    #: The fitted parameters, in the order of ``parameter_names``.
    params: npt.NDArray
    #: The name of each entry of ``params``.
    parameter_names: "list[str]"
    #: The covariates' names, which name their coefficients (``coef_0``,
    #: ... where the data did not name them).
    feature_names: "list[str]"
    #: The marker value at which the hazard's baseline is: the mean
    #: reading of the data the model was fitted to.
    marker_center: float
    #: The number of estimated parameters, the ``k`` of the criteria.
    k: int
    #: What the fit reached (``surpyval.utils.no_maximum``).
    maximum: str = "unknown"
    #: The numbers of items, failures and readings fitted (count
    #: weighted).
    n_items: float = 0.0
    n_events: float = 0.0
    n_readings: float = 0.0
    #: The scales of the internal fit: ``(time, marker centre, marker
    #: spread, covariate means)``. Predictions are computed there.
    _scales: tuple
    _covariance: "npt.NDArray | None" = None
    _neg_ll: float

    kind = "Joint (Weibull proportional hazards, current value)"

    # -- the parts -------------------------------------------------------

    @property
    def _p(self) -> int:
        return len(self.feature_names)

    def covariance(self) -> npt.NDArray:
        """The parameters' covariance, in the order of
        ``parameter_names``: the inverse of the observed information (the
        Hessian of the negative log-likelihood, by autograd) at the
        maximum. Raises a ``ValueError`` where the model has none."""
        if self._covariance is None:
            raise no_covariance_error()
        return np.array(self._covariance)

    def standard_errors(self) -> npt.NDArray:
        """The parameters' standard errors, the square roots of the
        diagonal of :meth:`covariance`, in the order of
        ``parameter_names``."""
        return standard_errors_of(self.covariance())

    def _ic_sample_size_from_data(self) -> float:
        return ic_sample_size(
            [0, 1], [self.n_events, self.n_items - self.n_events]
        )

    # -- the table -------------------------------------------------------

    def summary(self, alpha_ci: float = 0.05) -> "pd.DataFrame":
        """
        The parameter table: each parameter with its standard error, a
        two-sided ``1 - alpha_ci`` Wald interval and, for the
        coefficients, the Wald statistic ``z`` and its p-value. Indexed by
        ``(part, name)``: the ``"marker"`` part (its line's
        ``intercept``, ``slope`` and covariate coefficients, the random
        effects' ``var_intercept``, ``var_slope`` and
        ``corr_intercept_slope``, and ``sd_noise``) and the
        ``"survival"`` part (the Weibull ``alpha`` and ``beta``, the
        covariate coefficients and ``marker_effect``; ``exp(coef)`` is
        the hazard ratio of one unit of a covariate or of the marker).
        The intervals of the positive parameters (variances, ``sd_noise``,
        ``alpha``, ``beta``) are formed on the log scale and that of the
        correlation on Fisher's ``atanh`` scale, so each stays in its
        parameter's range.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ItemLifetime, SurpyvalData, WeibullJointPH
        >>> rng = np.random.default_rng(3)
        >>> items = []
        >>> for k in range(60):
        ...     start, rate = rng.normal(2.0, 0.4), rng.normal(0.02, 0.005)
        ...     life = 100 * rng.weibull(2.0) * np.exp(-0.9 * (start - 2))
        ...     end = min(life, 120.0)
        ...     ages = np.arange(0.0, end, 10.0)
        ...     y = start + rate * ages + rng.normal(0, 0.2, ages.size)
        ...     outcome = "observed" if life < 120 else "right_censored"
        ...     items.append(
        ...         ItemLifetime(k, **{outcome: end},
        ...                      markers=dict(zip(ages, y)))
        ...     )
        >>> model = WeibullJointPH.fit(SurpyvalData.from_items(items))
        >>> model.summary().index[-1]
        ('survival', 'marker_effect')
        """
        import pandas as pd

        from surpyval.univariate.regression._summary import (
            coefficient_table,
        )

        check_alpha_ci(alpha_ci)
        params = np.asarray(self.params, dtype=float)
        try:
            se = self.standard_errors()
        except ValueError:
            se = np.full(params.shape, np.nan)
        p = self._p
        names = self.parameter_names
        q = norm.ppf(1 - alpha_ci / 2)
        positive = {2 + p, 3 + p, 5 + p, 6 + p, 7 + p}
        hazard = set(range(8 + p, 9 + 2 * p))
        rows = []
        for j, name in enumerate(names):
            row = coefficient_table(
                [name],
                [params[j]],
                [se[j]],
                alpha_ci,
                exp=j in hazard,
            )
            level = "{:g}%".format(100 * (1 - alpha_ci))
            with np.errstate(all="ignore"):
                if j in positive:
                    half = q * se[j] / params[j]
                    row["coef lower " + level] = params[j] * np.exp(-half)
                    row["coef upper " + level] = params[j] * np.exp(half)
                    row["z"] = row["p"] = np.nan
                elif j == 4 + p:
                    half = q * se[j] / (1 - params[j] ** 2)
                    centre = np.arctanh(params[j])
                    row["coef lower " + level] = np.tanh(centre - half)
                    row["coef upper " + level] = np.tanh(centre + half)
                    row["z"] = row["p"] = np.nan
            rows.append(row)
        table = pd.concat(rows)
        index = [
            ("marker", name[len("marker_") :] if j < 2 + p else name)
            for j, name in enumerate(names[: 6 + p])
        ] + [("survival", name) for name in names[6 + p :]]
        table.index = pd.MultiIndex.from_tuples(index, names=["part", "name"])
        return table

    def __repr__(self) -> str:
        from surpyval.univariate.regression._summary import format_table

        out = (
            "Joint SurPyval Model"
            "\n===================="
            "\nMarker              : a line per item, random intercept "
            "and slope, normal noise"
            f"\nSurvival            : {self.dist.name} proportional hazards "
            "in the marker's current value"
            f"\nMarker centre       : {self.marker_center:.6g}"
            f"\nData                : {self.n_items:g} items, "
            f"{self.n_events:g} failures, {self.n_readings:g} readings"
            f"\nFit                 : maximum {self.maximum}"
        )
        table = self.summary()
        columns = {
            "coef": "estimate",
            "se(coef)": "se",
            "coef lower 95%": "lower 95%",
            "coef upper 95%": "upper 95%",
        }

        def block(part: str) -> str:
            rows = table.loc[part].rename(columns=columns)
            rows.index.name = None
            return format_table(rows, list(columns.values()))

        out += "\nMarker part         : Wald 95% intervals\n" + block("marker")
        out += "\nSurvival part       : Wald 95% intervals\n" + block(
            "survival"
        )
        return out

    # -- prediction ------------------------------------------------------

    def _item_arrays(self, item: Any) -> tuple[JointArrays, float]:
        """``item``'s data on the internal scale, and the age it was last
        seen running there."""
        from surpyval.utils.item_history import covariate_row_in
        from surpyval.utils.item_lifetime import ItemLifetime

        if not isinstance(item, ItemLifetime):
            raise ValueError(
                "The item is an ItemLifetime record, the question the "
                "prediction answers: ItemLifetime(id, right_censored=age, "
                "markers={age: value, ...}, covariates=...); got "
                f"{type(item).__name__}."
            )
        if item.outcome != "right_censored":
            raise ValueError(
                f"Item {item.id!r} is {item.outcome}, not still running: "
                "a prediction is for an item still running at its "
                "right_censored age, given its readings so far."
            )
        p = self._p
        if p == 0 and item.covariates is not None:
            raise ValueError(
                f"Item {item.id!r} has covariates, and the model was "
                "fitted without any; leave them out of the item."
            )
        if p:
            if item.covariates is None:
                raise ValueError(
                    f"Item {item.id!r} has no covariates; the model was "
                    f"fitted on {self.feature_names!r}."
                )
            have = item.covariate_names
            if have is not None and set(have) != set(self.feature_names):
                raise ValueError(
                    f"Item {item.id!r} has covariates {list(have)!r}; the "
                    f"model was fitted on {self.feature_names!r}."
                )
            z = np.asarray(
                covariate_row_in(
                    item, self.feature_names if have is not None else None
                ),
                dtype=float,
            )
            if z.size != p:
                raise ValueError(
                    f"Item {item.id!r} has {z.size} covariate(s); the "
                    f"model was fitted on {p}."
                )
        else:
            z = np.zeros(0)
        s, c, k, zbar = self._scales
        readings = item.markers or {}
        ages = np.array(list(readings), dtype=float) / s
        values = (np.array(list(readings.values()), dtype=float) - c) / k
        end = float(item.value) / s
        arrays = JointArrays.build(
            [1.0],
            [end],
            [0.0],
            [0.0],
            (z - np.asarray(zbar, dtype=float)).reshape(1, -1),
            [ages],
            [values],
        )
        return arrays, end

    def _log_sf_function(self, x: npt.NDArray, item: Any) -> Any:
        """``(f, after)``: ``f(params)``, the log of the item's survival
        to each age of ``x`` after its right censored age ``u``, given
        that age and its readings, as a function of the parameters, and
        the mask of those ages.

        The survival is ``I(x) / I(u)``, ``I(t) = int p(y | b) S(t | b)
        phi(b) db``: each integral is that of an item censored at ``t``,
        by adaptive Gauss-Hermite quadrature on nodes centred at the mode
        of its own integrand (a grid centred for ``u`` alone misses where
        ``S(x | b)`` puts its weight far beyond ``u``). The nodes are
        placed at the fitted parameters and held while differentiating.
        """
        arrays, end = self._item_arrays(item)
        s = self._scales[0]
        p = self._p
        after = np.isfinite(x) & (x / s > end)
        times = np.concatenate([[end], x[after] / s])
        many = JointArrays(
            **{
                name: np.repeat(getattr(arrays, name), times.size, axis=0)
                for name in ("w T d L Z n_read Sy St Syy Sty Stt".split())
            }
        )
        theta = internal_theta(np.asarray(self.params), p, self._scales)
        mode, root = censored_modes(theta, many, times)
        b, logw = adaptive_nodes(mode, root, GH_PREDICT)

        def f(params: Any) -> Any:
            th = internal_theta(params, p, self._scales)
            H = anp.exp(log_cumhaz(th, many, b, times))
            log_I = logsumexp(
                logw + log_readings(th, many, b) + log_prior(th, b) - H,
                axis=1,
            )
            return log_I[1:] - log_I[0]

        return f, after

    @keeps_query_shape
    def sf(self, x: npt.ArrayLike, item: Any) -> npt.NDArray:
        """
        The survival of ``item`` to each age ``x``, given its readings
        and that it was still running at its right censored age:

        .. math::
            P(T > x \\mid T > u, y) =
            \\frac{\\int S(x \\mid b) p(y \\mid b) \\phi(b) db}
            {\\int S(u \\mid b) p(y \\mid b) \\phi(b) db},

        over its line's random effects ``b``, each integral by adaptive
        Gauss-Hermite quadrature (15 points a dimension, centred at the
        mode of its integrand). An age at or before ``u`` gives
        1; ``nan`` gives ``nan``. The result has the shape of ``x``.

        Parameters
        ----------
        x : array_like
            The ages.
        item : ItemLifetime
            The item: still running (``right_censored`` at its age now),
            with its readings so far (``markers``, if any) and its
            covariates (as the model was fitted with).

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ItemLifetime, SurpyvalData, WeibullJointPH
        >>> rng = np.random.default_rng(3)
        >>> items = []
        >>> for k in range(60):
        ...     start, rate = rng.normal(2.0, 0.4), rng.normal(0.02, 0.005)
        ...     life = 100 * rng.weibull(2.0) * np.exp(-0.9 * (start - 2))
        ...     end = min(life, 120.0)
        ...     ages = np.arange(0.0, end, 10.0)
        ...     y = start + rate * ages + rng.normal(0, 0.2, ages.size)
        ...     outcome = "observed" if life < 120 else "right_censored"
        ...     items.append(
        ...         ItemLifetime(k, **{outcome: end},
        ...                      markers=dict(zip(ages, y)))
        ...     )
        >>> model = WeibullJointPH.fit(SurpyvalData.from_items(items))
        >>> item = ItemLifetime("P", right_censored=40, markers={0: 2.0})
        >>> float(model.sf(40, item))
        1.0
        """
        x = np.asarray(x, dtype=float)
        f, after = self._log_sf_function(x, item)
        out = np.where(np.isnan(x), np.nan, 1.0)
        if after.any():
            # far beyond the data the survival underflows to 0
            with np.errstate(all="ignore"):
                out[after] = np.exp(f(np.asarray(self.params, dtype=float)))
        return out

    @keeps_query_shape
    def ff(self, x: npt.ArrayLike, item: Any) -> npt.NDArray:
        """
        The probability that ``item`` fails by each age ``x``, given its
        readings and that it was still running at its right censored age:
        ``1 - sf(x, item)``.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ItemLifetime, SurpyvalData, WeibullJointPH
        >>> rng = np.random.default_rng(3)
        >>> items = []
        >>> for k in range(60):
        ...     start, rate = rng.normal(2.0, 0.4), rng.normal(0.02, 0.005)
        ...     life = 100 * rng.weibull(2.0) * np.exp(-0.9 * (start - 2))
        ...     end = min(life, 120.0)
        ...     ages = np.arange(0.0, end, 10.0)
        ...     y = start + rate * ages + rng.normal(0, 0.2, ages.size)
        ...     outcome = "observed" if life < 120 else "right_censored"
        ...     items.append(
        ...         ItemLifetime(k, **{outcome: end},
        ...                      markers=dict(zip(ages, y)))
        ...     )
        >>> model = WeibullJointPH.fit(SurpyvalData.from_items(items))
        >>> item = ItemLifetime("P", right_censored=40, markers={0: 2.0})
        >>> float(model.ff(20, item))
        0.0
        """
        return 1.0 - self.sf(x, item)

    @keeps_query_shape
    def cb(
        self,
        x: npt.ArrayLike,
        item: Any,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """
        Confidence bounds on the item's survival (``on="sf"``, the
        default) or failure probability (``on="ff"``) to each age ``x``,
        from the uncertainty of the parameters: the delta method, with
        the parameters' covariance, on ``log(-log sf)`` (so the bounds
        stay in (0, 1)). The adaptive nodes are held at the fitted
        parameters while differentiating. An age at or before the item's
        right censored age gives ``[1, 1]`` (``[0, 0]`` on ``ff``).

        Parameters
        ----------
        x : array_like
            The ages.
        item : ItemLifetime
            The item, as for :meth:`sf`.
        on : str, optional
            ``"sf"`` (or ``"R"``) or ``"ff"`` (or ``"F"``).
        alpha_ci : float, optional
            The total tail probability: 0.05 (the default) gives 95%
            bounds.
        bound : str, optional
            ``"two-sided"`` (``[lower, upper]`` on a last axis),
            ``"lower"`` or ``"upper"``.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ItemLifetime, SurpyvalData, WeibullJointPH
        >>> rng = np.random.default_rng(3)
        >>> items = []
        >>> for k in range(60):
        ...     start, rate = rng.normal(2.0, 0.4), rng.normal(0.02, 0.005)
        ...     life = 100 * rng.weibull(2.0) * np.exp(-0.9 * (start - 2))
        ...     end = min(life, 120.0)
        ...     ages = np.arange(0.0, end, 10.0)
        ...     y = start + rate * ages + rng.normal(0, 0.2, ages.size)
        ...     outcome = "observed" if life < 120 else "right_censored"
        ...     items.append(
        ...         ItemLifetime(k, **{outcome: end},
        ...                      markers=dict(zip(ages, y)))
        ...     )
        >>> model = WeibullJointPH.fit(SurpyvalData.from_items(items))
        >>> item = ItemLifetime("P", right_censored=40, markers={0: 2.0})
        >>> model.cb([20, 60], item).shape
        (2, 2)
        """
        check_alpha_ci(alpha_ci)
        check_option("on", on, _CB_ON)
        check_option("bound", bound, BOUNDS)
        cov = self.covariance()
        x = np.asarray(x, dtype=float)
        f, after = self._log_sf_function(x, item)
        tail, signs = bound_signs(alpha_ci, bound)
        q = norm.ppf(1 - tail)
        out = np.ones(x.shape + (signs.size,))
        out[np.isnan(x)] = np.nan
        if after.any():
            params = np.asarray(self.params, dtype=float)

            def log_H(v: Any) -> Any:
                return anp.log(-f(v))

            # where the survival underflows to 0, the bound is nan
            with np.errstate(all="ignore"):
                jvp = make_jvp(log_H)(params)
                columns = [jvp(e)[1] for e in np.eye(params.size)]
                centre = log_H(params)
                J = np.stack(columns, axis=-1)
                se = np.sqrt(np.einsum("mi,ij,mj->m", J, cov, J))
                # an upper bound on log H is a lower bound on sf
                out[after] = np.exp(
                    -np.exp(centre[:, None] - signs[None, :] * q * se[:, None])
                )
        if on in ("ff", "F"):
            out = 1.0 - out[..., ::-1] if signs.size == 2 else 1.0 - out
        return out if bound == "two-sided" else out[..., 0]

    # -- saving and loading ----------------------------------------------

    def to_dict(self) -> dict:
        """
        The model as a plain, JSON-serialisable ``dict``: its parameters
        and their names, the covariates' names, the marker centre and the
        scales its predictions are computed on, the covariance, the
        fitted negative log-likelihood and the counts of the data (not
        the data). :meth:`from_dict` rebuilds it, and it predicts and
        bounds as the original.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval
        >>> from surpyval import ItemLifetime, SurpyvalData, WeibullJointPH
        >>> rng = np.random.default_rng(3)
        >>> items = []
        >>> for k in range(60):
        ...     start, rate = rng.normal(2.0, 0.4), rng.normal(0.02, 0.005)
        ...     life = 100 * rng.weibull(2.0) * np.exp(-0.9 * (start - 2))
        ...     end = min(life, 120.0)
        ...     ages = np.arange(0.0, end, 10.0)
        ...     y = start + rate * ages + rng.normal(0, 0.2, ages.size)
        ...     outcome = "observed" if life < 120 else "right_censored"
        ...     items.append(
        ...         ItemLifetime(k, **{outcome: end},
        ...                      markers=dict(zip(ages, y)))
        ...     )
        >>> model = WeibullJointPH.fit(SurpyvalData.from_items(items))
        >>> restored = surpyval.from_dict(model.to_dict())
        >>> bool(np.allclose(restored.params, model.params))
        True
        """
        s, c, k, zbar = self._scales
        out: dict[str, Any] = {
            "model": "JointModel",
            "distribution": self.dist.name,
            "params": np.asarray(self.params, dtype=float).tolist(),
            "parameter_names": list(self.parameter_names),
            "feature_names": list(self.feature_names),
            "marker_center": float(self.marker_center),
            "scales": {
                "time": float(s),
                "marker_center": float(c),
                "marker_spread": float(k),
                "covariate_means": np.asarray(zbar, dtype=float).tolist(),
            },
            "n_items": float(self.n_items),
            "n_events": float(self.n_events),
            "n_readings": float(self.n_readings),
            "_neg_ll": float(self._neg_ll),
            **maximum_entry(self.maximum),
        }
        if self._covariance is not None:
            out["covariance"] = np.asarray(self._covariance).tolist()
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "JointModel":
        """
        Rebuild a model from a :meth:`to_dict` dictionary.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ItemLifetime, SurpyvalData, WeibullJointPH
        >>> from surpyval.univariate.joint import JointModel
        >>> rng = np.random.default_rng(3)
        >>> items = []
        >>> for k in range(60):
        ...     start, rate = rng.normal(2.0, 0.4), rng.normal(0.02, 0.005)
        ...     life = 100 * rng.weibull(2.0) * np.exp(-0.9 * (start - 2))
        ...     end = min(life, 120.0)
        ...     ages = np.arange(0.0, end, 10.0)
        ...     y = start + rate * ages + rng.normal(0, 0.2, ages.size)
        ...     outcome = "observed" if life < 120 else "right_censored"
        ...     items.append(
        ...         ItemLifetime(k, **{outcome: end},
        ...                      markers=dict(zip(ages, y)))
        ...     )
        >>> model = WeibullJointPH.fit(SurpyvalData.from_items(items))
        >>> restored = JointModel.from_dict(model.to_dict())
        >>> restored.parameter_names == model.parameter_names
        True
        """
        import surpyval

        require_model_tag(model_dict, "JointModel", "a joint model")
        if model_dict["distribution"] != "Weibull":
            raise ValueError(
                "A joint model has a Weibull baseline; the dict names "
                f"{model_dict['distribution']!r}."
            )
        out = cls()
        out.dist = surpyval.Weibull
        out.params = np.array(model_dict["params"], dtype=float)
        out.parameter_names = list(model_dict["parameter_names"])
        out.feature_names = list(model_dict["feature_names"])
        out.marker_center = float(model_dict["marker_center"])
        scales = model_dict["scales"]
        out._scales = (
            float(scales["time"]),
            float(scales["marker_center"]),
            float(scales["marker_spread"]),
            np.array(scales["covariate_means"], dtype=float),
        )
        out.n_items = float(model_dict.get("n_items", 0.0))
        out.n_events = float(model_dict.get("n_events", 0.0))
        out.n_readings = float(model_dict.get("n_readings", 0.0))
        out._neg_ll = float(model_dict["_neg_ll"])
        out.k = len(out.params)
        out.maximum = restored_maximum(model_dict)
        if "covariance" in model_dict:
            out._covariance = np.array(model_dict["covariance"], dtype=float)
        if len(out.parameter_names) != len(out.params):
            raise ValueError(
                "The dict's 'parameter_names' and 'params' differ in length."
            )
        return out
