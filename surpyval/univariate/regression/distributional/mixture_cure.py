"""Mixture-cure regression: covariates on the fraction that ever fails and
on the lifetime of those that do.

Some units never have the event: parts without the defect, patients who
are cured. A unit is *susceptible* with probability :math:`\\pi(z)`, and a
susceptible unit fails at a time with survival function
:math:`S_u(t \\mid z)`, so the population's survival levels off above zero,

.. math::
    S(t \\mid z) = 1 - \\pi(z) + \\pi(z)\\, S_u(t \\mid z).

``MixtureCure(Weibull)`` fits both parts at once by maximum likelihood.
It is the limited-failure-population model (``Weibull.fit(..., lfp=True)``)
with covariates: the *incidence* :math:`\\pi(z)` (the univariate fit's
``lfp_p``) has its own linear predictor through a logit (or probit, or
complementary log-log) link, and the *latency* :math:`S_u` is the
distribution with covariates on its parameters, as in a distributional
regression (covariates on the scale alone are the AFT model).

The model is a distributional regression of the *defective* distribution,
the lifetime distribution with one more parameter, ``lfp_p``. So the
likelihood is the regressions' own (every censoring, truncation, count
and weight of SurPyval's data model), a failure contributing
:math:`\\pi f_u`, a unit still running :math:`1 - \\pi + \\pi S_u`, and
late entry dividing by the same at the entry age; and the fit, its
checks, the bounds and the saved model are that family's.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import autograd.numpy as np
import numpy as onp
import numpy.typing as npt

from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
)
from surpyval.utils.shapes import check_paired_rows, covariate_rows

from . import _links
from .fitter import DistributionalFitter, uniform_draws
from .model import DistributionalRegressionModel

#: The latency distributions a mixture-cure regression takes (lifetimes
#: on the positive half-line), each with the parameter that sets how long
#: a susceptible unit lasts: the one the covariates are on by default.
LATENCIES = {
    "Exponential": "failure_rate",
    "Weibull": "alpha",
    "Gamma": "beta",
    "LogNormal": "mu",
    "LogLogistic": "alpha",
}

#: The name of the susceptible probability, the univariate fits' name
#: for the proportion that ever fails.
INCIDENCE = "lfp_p"

#: The largest and smallest starting value of ``lfp_p``.
_START = (0.02, 0.98)


class CuredDistribution:
    """A lifetime distribution with a fraction that never fails: the
    latency distribution's parameters, then ``lfp_p``, the probability of
    ever failing. Its functions are the population's (``sf`` levels off
    at ``1 - lfp_p``)."""

    def __init__(self, latency: Any) -> None:
        self.latency = latency
        self.name = latency.name + "MixtureCure"
        self.parameter_names = [*latency.parameter_names, INCIDENCE]
        self.bounds = (*latency.bounds, (0, 1))
        self.support = latency.support
        self.k = len(self.parameter_names)

    def __reduce__(self) -> Any:
        return (_cured, (self.latency.name,))

    def __repr__(self) -> str:
        return f"{self.name} (a {self.latency.name} latency)"

    def sf(self, x: Numeric, *params: Boxable) -> Boxable:
        *lat, p = params
        return 1.0 - p + p * self.latency.sf(x, *lat)

    def ff(self, x: Numeric, *params: Boxable) -> Boxable:
        *lat, p = params
        return p * self.latency.ff(x, *lat)

    def df(self, x: Numeric, *params: Boxable) -> Boxable:
        *lat, p = params
        return p * self.latency.df(x, *lat)

    def hf(self, x: Numeric, *params: Boxable) -> Boxable:
        return self.df(x, *params) / self.sf(x, *params)

    def Hf(self, x: Numeric, *params: Boxable) -> Boxable:
        return -self.log_sf(x, *params)

    def log_sf(self, x: Numeric, *params: Boxable) -> Boxable:
        *lat, p = params
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.logaddexp(
                np.log1p(-p), np.log(p) + self.latency.log_sf(x, *lat)
            )

    def log_ff(self, x: Numeric, *params: Boxable) -> Boxable:
        *lat, p = params
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(p) + self.latency.log_ff(x, *lat)

    def log_df(self, x: Numeric, *params: Boxable) -> Boxable:
        *lat, p = params
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(p) + self.latency.log_df(x, *lat)

    def qf(self, u: Numeric, *params: Boxable) -> Boxable:
        """The population's quantile: infinite at and above ``lfp_p``,
        the units that never fail."""
        *lat, p = params
        u = onp.asarray(u, dtype=float)
        with onp.errstate(all="ignore"):
            ratio = u / p
            inside = ratio < 1
            value = self.latency.qf(onp.where(inside, ratio, 0.5), *lat)
            out = onp.where(inside, value, onp.inf)
        return onp.where(onp.isnan(ratio), onp.nan, out)

    def mean(self, *params: float) -> float:
        *lat, p = params
        return float(self.latency.mean(*lat)) if p >= 1 else float("inf")

    def fit_from_surpyval_data(self, data: Any) -> Any:
        """The start of the search: the limited-failure fit without
        covariates, its proportion kept off the ends."""
        from surpyval.utils.warnings import ignoring_warnings

        with ignoring_warnings():
            try:
                fitted = self.latency.fit_from_surpyval_data(data, lfp=True)
                params = [float(v) for v in fitted.params]
                p = float(fitted.lfp_p)
            except Exception:
                fitted = self.latency.fit_from_surpyval_data(data)
                params = [float(v) for v in fitted.params]
                p = 0.5
        if not onp.all(onp.isfinite(params)) or not onp.isfinite(p):
            fitted = self.latency.fit_from_surpyval_data(data)
            params, p = [float(v) for v in fitted.params], 0.5

        class _Start:
            pass

        start = _Start()
        start.params = onp.array(  # type: ignore[attr-defined]
            [*params, min(max(p, _START[0]), _START[1])]
        )
        return start

    def fit(self, **kwargs: Any) -> Any:
        """The limited-failure fit without covariates (the null model of
        the global likelihood-ratio test)."""
        return self.latency.fit(lfp=True, **kwargs)

    def from_params(self, params: Any) -> Any:
        """The univariate limited-failure model with these parameters."""
        *lat, p = (float(v) for v in params)
        return self.latency.from_params(lat, lfp_p=p)


_CURED: dict[str, CuredDistribution] = {}


def _cured(name: str) -> CuredDistribution:
    """The cured form of the latency distribution called ``name`` (one
    object per distribution, so fitters and models share it)."""
    import surpyval

    if name not in _CURED:
        _CURED[name] = CuredDistribution(getattr(surpyval, name))
    return _CURED[name]


def _check_latency(latency: Any) -> None:
    from surpyval.univariate.parametric.parametric_fitter import (
        ParametricFitter,
    )

    name = getattr(latency, "name", None)
    if not isinstance(latency, ParametricFitter):
        raise ValueError(
            "MixtureCure takes the latency's distribution (Weibull, "
            "LogNormal, ...); got {!r}.".format(latency)
        )
    if name not in LATENCIES:
        raise ValueError(
            "Mixture-cure regression is not available with a {} latency: "
            "the time to failure of the susceptible units is one of {} "
            "(lifetimes on the positive half-line).".format(
                name, ", ".join(LATENCIES)
            )
        )


class MixtureCureFitter(DistributionalFitter):
    """A mixture-cure regression fitter; build one with
    :func:`MixtureCure`.

    The parameters are the latency distribution's at the reference row
    (``Z = 0``, or the covariate means with ``center=True``), then
    ``lfp_p``, the probability there of ever failing, then each
    parameter's coefficients on its link scale, named
    ``<parameter>_<column>``. Every coefficient has SurPyval's sign: a
    positive one means a shorter life (on ``lfp_p``, a higher chance of
    ever failing).
    """

    fitter_kind = "mixture-cure regression fitter"
    name_suffix = ""

    @staticmethod
    def _check_distribution(distribution: Any) -> None:
        if not isinstance(distribution, CuredDistribution):
            raise ValueError(
                "MixtureCureFitter is built by MixtureCure(latency), e.g. "
                "MixtureCure(Weibull)."
            )

    def _model_class(self) -> Any:
        return MixtureCureModel

    def _say_verdict(self, verdict: Any, values: dict) -> None:
        """As the family's, except where the only thing running off is
        the susceptible probability towards 1: the data then show no
        cured fraction, and the warning says so."""
        from surpyval.utils.no_maximum import warn_no_maximum

        from .._fit_skeleton import NO_MAXIMUM_CONSEQUENCE

        p = float(values.get(INCIDENCE, onp.nan))
        only_incidence = (
            verdict.no_maximum
            and not verdict.runaway
            and list(verdict.baseline or ()) == [INCIDENCE]
        )
        if only_incidence and p > 0.5:
            warn_no_maximum(
                "the likelihood keeps increasing as lfp_p ({:.4g}), the "
                "probability of ever failing at the reference row, runs "
                "on towards 1: the data show no sign of units that never "
                "fail there (no plateau at the end of follow-up)".format(p),
                NO_MAXIMUM_CONSEQUENCE,
                "fit the latency alone, without a cure fraction (an AFT "
                "model or Distributional({})), or follow the units for "
                "longer".format(self.dist.latency.name),
            )
            return
        super()._say_verdict(verdict, values)

    def _repr_details(self) -> list[str]:
        return [f"{self.dist.latency.name} latency"]

    @property
    def latency(self) -> Any:
        """The distribution of the time to failure of a susceptible
        unit."""
        return self.dist.latency

    @property
    def signs(self) -> tuple[float, ...]:
        latency = self.dist.latency.name
        return tuple(
            1.0 if name == INCIDENCE else _links.sign(latency, name)
            for name in self.parameter_names
        )

    def _resolve_covariates(
        self,
        covariates: Mapping[str, Any] | None,
        p: int,
        names: list[str] | None,
    ) -> tuple[npt.NDArray, ...]:
        """As the distributional regression's, with this default: every
        column on the incidence (``lfp_p``) and on the latency's life
        parameter (its scale, location or rate), none on its shape."""
        if covariates is None:
            life = LATENCIES[self.dist.latency.name]
            covariates = {INCIDENCE: list(range(p)), life: list(range(p))}
        return super()._resolve_covariates(covariates, p, names)

    def random(
        self,
        size: int,
        Z: npt.ArrayLike,
        *params: float,
        random_state: Any = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """``size`` draws at each covariate row of ``Z``, row by row, and
        the row each was drawn at. A unit that never fails is drawn as
        ``inf``, as the univariate limited-failure model draws it."""
        from surpyval.utils.rng import as_generator

        Z_arr = onp.atleast_2d(onp.asarray(Z, dtype=float))
        rng = None if random_state is None else as_generator(random_state)
        x, Z_out = [], []
        for row in Z_arr:
            dist_params = [
                float(onp.ravel(v)[0])
                for v in self._dist_params_at(row, params)
            ]
            U = uniform_draws(size, rng)
            x.append(onp.asarray(self.dist.qf(U, *dist_params), dtype=float))
            Z_out.append(onp.tile(row, (size, 1)))
        return onp.concatenate(x), onp.vstack(Z_out)


class MixtureCureModel(DistributionalRegressionModel):
    r"""
    A fitted mixture-cure regression (:func:`MixtureCure`).

    A unit with covariates :math:`z` is susceptible (ever fails) with
    probability :math:`\pi(z)`, and a susceptible unit's time to failure
    has the latency distribution at that row's parameters:

    .. math::
        S(t \mid z) = 1 - \pi(z) + \pi(z)\, S_u(t \mid z), \qquad
        g\bigl(\pi(z)\bigr) = g(\pi) + x_\pi(z)' b_\pi .

    ``sf``, ``ff``, ``df``, ``hf``, ``Hf``, ``qf`` and the bounds are the
    population's: the survival levels off at the cure probability, the
    quantile function is infinite beyond the susceptible probability,
    and :meth:`mean` is infinite. Each takes ``susceptible=True`` for
    the latency alone, the life of a unit that does fail.

    ``params`` are the latency's parameters at ``Z = 0`` (at ``center``
    for a fit with ``center=True``), then ``lfp_p``, the susceptible
    probability there, then each parameter's coefficients, named
    ``<parameter>_<column>``. A positive coefficient means a shorter
    life: on ``lfp_p``, a higher chance of ever failing (``exp`` of a
    logit coefficient is the odds ratio of failing); on a scale or
    location, an earlier failure, as in the AFT models.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import WeibullMixtureCure
    >>> rng = np.random.default_rng(5)
    >>> lot = rng.binomial(1, 0.5, 600)
    >>> fails = rng.random(600) < np.where(lot == 1, 0.5, 0.2)
    >>> life = 100 * rng.weibull(2.0, 600)
    >>> x = np.where(fails & (life < 400), life, 400.0)
    >>> c = np.where(fails & (life < 400), 0, 1)
    >>> model = WeibullMixtureCure.fit(x, lot, c=c)
    >>> model.parameter_names
    ['alpha', 'beta', 'lfp_p', 'alpha_coef_0', 'lfp_p_coef_0']
    >>> model.susceptible_probability([[0], [1]]).round(2)
    array([0.22, 0.53])
    """

    _dict_model = "MixtureCureModel"
    _repr_title = "Mixture-Cure Regression SurPyval Model"

    @classmethod
    def _fitter_from_dict(cls, model_dict: dict, links: tuple) -> Any:
        latency = str(model_dict.get("latency"))
        if latency not in LATENCIES:
            raise ValueError(f"Unknown latency distribution {latency!r}")
        dist = _cured(latency)
        return MixtureCureFitter(
            dist, links=dict(zip(dist.parameter_names, links))
        )

    def to_dict(self) -> dict:
        """
        Serialise the fitted model to a plain, JSON-ready ``dict``: the
        latency distribution, each parameter's link and columns, the
        parameters and, where it can be computed, their covariance.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval
        >>> rng = np.random.default_rng(5)
        >>> lot = rng.binomial(1, 0.5, 600)
        >>> fails = rng.random(600) < np.where(lot == 1, 0.5, 0.2)
        >>> life = 100 * rng.weibull(2.0, 600)
        >>> x = np.where(fails & (life < 400), life, 400.0)
        >>> c = np.where(fails & (life < 400), 0, 1)
        >>> model = surpyval.WeibullMixtureCure.fit(x, lot, c=c)
        >>> restored = surpyval.from_dict(model.to_dict())
        >>> bool(np.allclose(restored.sf(50, [1]), model.sf(50, [1])))
        True
        """
        return super().to_dict()

    def _subclass_dict_entries(self) -> dict:
        return {"latency": self.latency.name}

    @property
    def latency(self) -> Any:
        """The distribution of the time to failure of a susceptible
        unit."""
        dist: Any = self.distribution
        return dist.latency

    @property
    def incidence_link(self) -> str:
        """The link of the susceptible probability: ``"logit"``,
        ``"probit"`` or ``"cloglog"``."""
        return str(self.links[-1])

    # -- the two parts ------------------------------------------------------

    def _parts(self, Z: Any) -> tuple[list[npt.NDArray], npt.NDArray]:
        """The latency's parameters and the susceptible probability at
        each row of ``Z``."""
        rows = self._rows(Z)
        with onp.errstate(all="ignore"):
            values = self.model._dist_params_at(rows, self._eval_params())
        values = [
            onp.asarray(v, dtype=float) * onp.ones(rows.shape[0])
            for v in values
        ]
        missing = ~onp.all(onp.isfinite(rows), axis=1)
        values = [onp.where(missing, onp.nan, v) for v in values]
        return values[:-1], values[-1]

    def susceptible_probability(self, Z: Any) -> Any:
        """
        The probability that a unit with covariates ``Z`` ever fails,
        :math:`\\pi(z)`: one value per row of ``Z`` (a number for one
        row).

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullMixtureCure
        >>> rng = np.random.default_rng(5)
        >>> lot = rng.binomial(1, 0.5, 600)
        >>> fails = rng.random(600) < np.where(lot == 1, 0.5, 0.2)
        >>> life = 100 * rng.weibull(2.0, 600)
        >>> x = np.where(fails & (life < 400), life, 400.0)
        >>> c = np.where(fails & (life < 400), 0, 1)
        >>> model = WeibullMixtureCure.fit(x, lot, c=c)
        >>> round(float(model.susceptible_probability([1])), 2)
        0.53
        """
        _, pi = self._parts(Z)
        return float(pi[0]) if pi.size == 1 else pi

    def cure_probability(self, Z: Any) -> Any:
        """
        The probability that a unit with covariates ``Z`` never fails,
        :math:`1 - \\pi(z)`: the level its survival curve settles at.
        """
        _, pi = self._parts(Z)
        out = 1.0 - pi
        return float(out[0]) if out.size == 1 else out

    def cure_given_survival(self, x: npt.ArrayLike, Z: Any) -> npt.NDArray:
        r"""
        The probability that a unit still running at age ``x`` is one
        that never fails:

        .. math::
            P(\text{cured} \mid T > x, z)
            = \frac{1 - \pi(z)}{1 - \pi(z) + \pi(z)\, S_u(x \mid z)} .

        It starts at the cure probability and rises towards 1 as the
        unit outlives the susceptible ones: the chance that a part still
        in service is free of the defect, or that a patient still
        event-free is cured. ``x`` and ``Z`` pair as for :meth:`sf`.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullMixtureCure
        >>> rng = np.random.default_rng(5)
        >>> lot = rng.binomial(1, 0.5, 600)
        >>> fails = rng.random(600) < np.where(lot == 1, 0.5, 0.2)
        >>> life = 100 * rng.weibull(2.0, 600)
        >>> x = np.where(fails & (life < 400), life, 400.0)
        >>> c = np.where(fails & (life < 400), 0, 1)
        >>> model = WeibullMixtureCure.fit(x, lot, c=c)
        >>> model.cure_given_survival([0, 100, 300], [1]).round(2)
        array([0.47, 0.73, 1.  ])
        """
        t = onp.asarray(x, dtype=float)
        S = onp.asarray(self.sf(t, Z), dtype=float)
        cure = onp.asarray(self.cure_probability(Z), dtype=float)
        with onp.errstate(all="ignore"):
            out = cure / S
        return onp.where(onp.isnan(S), onp.nan, onp.minimum(out, 1.0))

    def _latency_call(self, fn: str, x: Any, Z: Any) -> npt.NDArray:
        """The latency distribution's ``fn`` at ``x`` for each row of
        ``Z``, paired as the model's own functions pair them."""
        t = onp.asarray(x, dtype=float)
        lat, _ = self._parts(Z)
        check_paired_rows(t.size, lat[0].size)
        size = max(t.size, lat[0].size)
        flat = onp.broadcast_to(t.reshape(-1), (size,))
        lat = [onp.broadcast_to(v, (size,)) for v in lat]
        lower = float(self.latency.support[0])
        at_start = {"sf": 1.0, "ff": 0.0, "Hf": 0.0, "df": 0.0, "hf": 0.0}
        below = flat <= lower if onp.isfinite(lower) else flat < -onp.inf
        with onp.errstate(all="ignore"):
            inside = onp.where(below, lower + 1.0, flat)
            out = onp.asarray(
                getattr(self.latency, fn)(inside, *lat), dtype=float
            )
            if fn != "qf":
                out = onp.where(below & (flat < lower), at_start[fn], out)
                out = onp.where(
                    below & (flat == lower) & (fn in ("sf", "ff", "Hf")),
                    at_start[fn],
                    out,
                )
        out = onp.where(onp.isnan(flat + lat[0]), onp.nan, out)
        if t.ndim == 0 and size == 1:
            return out[0]
        return out.reshape(t.shape) if t.size == size else out

    def sf(
        self, x: npt.ArrayLike, Z: Any, *, susceptible: bool = False, **kw: Any
    ) -> npt.NDArray:
        """
        Survival at ``x`` for covariates ``Z``: the population's, which
        levels off at :meth:`cure_probability`. ``susceptible=True``
        gives the latency's, the survival of a unit that does fail.
        """
        if susceptible:
            return self._latency_call("sf", x, Z)
        return super().sf(x, Z, **kw)

    def ff(
        self, x: npt.ArrayLike, Z: Any, *, susceptible: bool = False, **kw: Any
    ) -> npt.NDArray:
        """Failure probability by ``x``; as :meth:`sf`. The population's
        rises to :meth:`susceptible_probability`, not to 1."""
        if susceptible:
            return self._latency_call("ff", x, Z)
        return super().ff(x, Z, **kw)

    def df(
        self, x: npt.ArrayLike, Z: Any, *, susceptible: bool = False, **kw: Any
    ) -> npt.NDArray:
        """Density at ``x``; as :meth:`sf`."""
        if susceptible:
            return self._latency_call("df", x, Z)
        return super().df(x, Z, **kw)

    def hf(
        self, x: npt.ArrayLike, Z: Any, *, susceptible: bool = False, **kw: Any
    ) -> npt.NDArray:
        """Hazard rate at ``x``; as :meth:`sf`. The population's falls
        towards 0 as the survivors become mostly the cured."""
        if susceptible:
            return self._latency_call("hf", x, Z)
        return super().hf(x, Z, **kw)

    def Hf(
        self, x: npt.ArrayLike, Z: Any, *, susceptible: bool = False, **kw: Any
    ) -> npt.NDArray:
        """Cumulative hazard at ``x``; as :meth:`sf`."""
        if susceptible:
            return self._latency_call("Hf", x, Z)
        return super().Hf(x, Z, **kw)

    def qf(
        self,
        p: npt.ArrayLike,
        Z: Any,
        *,
        grid: bool = False,
        susceptible: bool = False,
    ) -> npt.NDArray:
        """
        The quantile at probability ``p`` for covariates ``Z``. The
        population's is infinite at and beyond the susceptible
        probability (that fraction never fails); ``susceptible=True``
        gives the latency's, the age by which a fraction ``p`` of the
        units that do fail have failed.
        """
        if not susceptible:
            return super().qf(p, Z, grid=grid)
        from surpyval.utils.validation import warn_outside_unit_interval

        u = onp.asarray(p, dtype=float)
        u = onp.where(warn_outside_unit_interval(u), onp.nan, u)
        if grid:
            rows = covariate_rows(self._prepare_Z(Z), self._n_covariates())
            return onp.stack(
                [self._latency_call("qf", u.reshape(-1), row) for row in rows]
            )
        return self._latency_call("qf", u, Z)

    def _quantile_cb_rows(
        self,
        probs: npt.NDArray,
        rows: npt.NDArray,
        alpha_ci: float,
        bound: str,
        method: str,
        *args: Any,
        **kwargs: Any,
    ) -> npt.NDArray:
        """``quantile_cb``'s bounds (see the regressions' own). With
        ``method="lr"`` the upper bound is ``inf`` wherever the
        likelihood region holds parameters whose susceptible probability
        at the row is at most ``p``: a fraction ``p`` of such units may
        never fail, so the quantile is infinite there. Whether it does is
        the profile lower bound on ``lfp_p`` at the row, at the same
        level; that side is not searched, and the other is the one-sided
        bound at the same critical value."""
        sup = super()._quantile_cb_rows
        if method != "lr" or bound == "lower":
            return sup(probs, rows, alpha_ci, bound, method, *args, **kwargs)
        side = "two-sided" if bound == "two-sided" else "lower"
        lfp = onp.asarray(
            self.parameter_cb(
                "lfp_p", rows, alpha_ci=alpha_ci, bound=side, method="lr"
            ),
            dtype=float,
        )
        lowest = lfp[:, 0] if side == "two-sided" else lfp.reshape(-1)
        endless = lowest <= probs
        two = bound == "two-sided"
        out = onp.full((probs.size, 2) if two else probs.size, onp.nan)
        if (~endless).any():
            out[~endless] = sup(
                probs[~endless],
                rows[~endless],
                alpha_ci,
                bound,
                method,
                *args,
                **kwargs,
            )
        if endless.any():
            if two:
                # the lower end of a two-sided interval at alpha is the
                # one-sided bound at alpha / 2
                out[endless, 0] = sup(
                    probs[endless],
                    rows[endless],
                    alpha_ci / 2,
                    "lower",
                    method,
                    *args,
                    **kwargs,
                )
                out[endless, 1] = onp.inf
            else:
                out[endless] = onp.inf
        return out

    def mean(self, Z: Any, *, susceptible: bool = False) -> Any:
        """
        The mean life of a unit with covariates ``Z``. The population's
        is infinite wherever some units never fail (returned as ``inf``,
        without a warning, as the univariate limited-failure model's);
        ``susceptible=True`` gives the latency's mean, the mean life of
        the units that do fail.
        """
        lat, pi = self._parts(Z)
        out = onp.full(pi.shape, onp.nan)
        with onp.errstate(all="ignore"):
            for j in onp.flatnonzero(~onp.isnan(pi)):
                out[j] = float(self.latency.mean(*(float(v[j]) for v in lat)))
        if not susceptible:
            out = onp.where((pi < 1) & ~onp.isnan(out), onp.inf, out)
        elif onp.any(onp.isposinf(out)):
            warnings.warn(
                "The latency's mean is infinite at {} of the {} covariate "
                "rows (for example a log-logistic shape at or below 1). "
                "Use a quantile, qf(p, Z, susceptible=True), "
                "instead.".format(int(onp.sum(onp.isposinf(out))), out.size),
                UserWarning,
                stacklevel=2,
            )
        return float(out[0]) if out.size == 1 else out

    def distribution_at(self, Z: Any) -> Any:
        """
        The distribution of a unit with the covariate row ``Z``, as a
        univariate limited-failure model (``Weibull.from_params(...,
        lfp_p=...)``), with every univariate function.
        """
        return super().distribution_at(Z)

    def _times_at_hazard(
        self, params: Any, center: Any, H_target: Any, Z: Any
    ) -> npt.NDArray:
        """The times at which the population's cumulative hazard reaches
        ``H_target``, row by row of ``Z`` (the parametric bootstrap's
        draws): the population's quantile function, in closed form,
        infinite where the target is past the plateau."""
        rows = onp.asarray(self._centred(Z, center), dtype=float)
        with onp.errstate(all="ignore"):
            u = -onp.expm1(-onp.asarray(H_target, dtype=float))
            values = self.model._dist_params_at(rows, params)
            values = [
                onp.asarray(v, dtype=float) * onp.ones(rows.shape[0])
                for v in values
            ]
            return onp.asarray(self.distribution.qf(u, *values), dtype=float)

    # -- follow-up ------------------------------------------------------------

    def follow_up(self) -> dict:
        r"""
        Whether the units were followed for long enough to tell a cured
        fraction from failures still to come (Maller and Zhou's check).

        A cure fraction is only identified by a *plateau*: a stretch at
        the end of follow-up in which units are still under observation
        and nothing fails. With :math:`t_f` the largest failure time and
        :math:`t_{max}` the largest time of all, the check counts the
        failures :math:`N` in the last stretch of the failure times as
        long as the plateau, :math:`(2 t_f - t_{max},\ t_f]`. Few
        failures there says the failures had thinned out before
        follow-up ended; many says they were still arriving at their
        usual rate. The p-value is Maller and Zhou's approximation,
        :math:`(1 - N/n)^n`: small values are evidence that follow-up
        was sufficient.

        Returns
        -------
        dict
            ``{"last_failure", "last_time", "plateau", "n_plateau",
            "n_failures_in_window", "n", "p_value", "sufficient"}``:
            the largest failure time and the largest time, their
            difference, the number of units censored after the last
            failure, :math:`N`, the number of units, the p-value, and
            whether it is below 0.05.

        Raises
        ------
        ValueError
            For a model without its data, or data with left or interval
            censoring or truncation (the check is for exact and
            right-censored times).

        References
        ----------
        Maller, R. A. and Zhou, X. (1996), "Survival Analysis with
        Long-Term Survivors", Wiley, section 4.3.
        """
        data = getattr(self, "data", None)
        if data is None:
            raise ValueError(
                "follow_up needs the data the model was fitted to, which "
                "a restored model does not keep."
            )
        return _follow_up(data)


def _follow_up(data: Any) -> dict:
    c = onp.asarray(data.c)
    if onp.any((c != 0) & (c != 1)) or bool(onp.any(data.truncated_mask)):
        raise ValueError(
            "The follow-up check is for exact and right-censored times "
            "without truncation; these data have left or interval "
            "censoring or truncation. Look at the non-parametric estimate "
            "(Turnbull.fit) for a plateau instead."
        )
    x = onp.asarray(data.x, dtype=float)
    n = onp.asarray(data.n, dtype=float)
    failed = c == 0
    if not failed.any():
        raise ValueError(
            "There are no failures, so there is nothing to check the "
            "follow-up against."
        )
    t_f = float(x[failed].max())
    t_max = float(x.max())
    window = failed & (x > 2 * t_f - t_max) & (x <= t_f)
    N = float(n[window].sum())
    total = float(n.sum())
    p_value = float((1.0 - N / total) ** total)
    return {
        "last_failure": t_f,
        "last_time": t_max,
        "plateau": t_max - t_f,
        "n_plateau": float(n[(~failed) & (x > t_f)].sum()),
        "n_failures_in_window": N,
        "n": total,
        "p_value": p_value,
        "sufficient": bool(p_value < 0.05),
    }


def MixtureCure(
    latency: Any,
    incidence: str = "logit",
    covariates: Mapping[str, Any] | None = None,
    links: Mapping[str, str] | None = None,
) -> MixtureCureFitter:
    r"""
    A mixture-cure regression fitter: a fraction of the units never
    fails, and covariates act both on that fraction and on the lifetime
    of the rest.

    .. math::
        S(t \mid z) = 1 - \pi(z) + \pi(z)\, S_u(t \mid z)

    :math:`\pi(z)` is the probability of ever failing (the *incidence*;
    ``lfp_p`` in the model), with ``g(\pi(z))`` linear in the covariates;
    :math:`S_u` is the ``latency`` distribution with covariates on its
    parameters. Both parts are fitted together by maximum likelihood,
    with every censoring, truncation, count and weight of SurPyval's
    data model. It is ``latency.fit(..., lfp=True)`` with covariates.

    Parameters
    ----------
    latency : ParametricFitter
        The distribution of the time to failure of a susceptible unit:
        Weibull, LogNormal, LogLogistic, Gamma or Exponential.
    incidence : {"logit", "probit", "cloglog"}, optional
        The link of the susceptible probability. Default ``"logit"``
        (logistic regression: ``exp`` of a coefficient is an odds
        ratio).
    covariates : dict, optional
        The default of ``fit``'s ``covariates``: each parameter's
        columns of ``Z``, e.g. ``{"lfp_p": [0, 1], "alpha": [0]}``. By
        default every column acts on ``lfp_p`` and on the latency's life
        parameter (the scale ``alpha`` of a Weibull or LogLogistic,
        ``mu`` of a LogNormal, the rate of a Gamma or Exponential), and
        none on its shape.
    links : dict, optional
        A link per latency parameter (the default is log for a positive
        parameter and identity for one on the whole line).

    Returns
    -------
    MixtureCureFitter
        With ``fit``, ``fit_from_df`` (``formulas={"lfp_p": ...,
        "alpha": ...}`` gives each part its own formula) and
        ``fit_penalised``, each returning a :class:`MixtureCureModel`.

    Examples
    --------
    >>> from surpyval import MixtureCure, Weibull
    >>> MixtureCure(Weibull)
    WeibullMixtureCure: mixture-cure regression fitter (Weibull latency)
    >>> MixtureCure(Weibull, incidence="probit").links
    ('log', 'log', 'probit')
    """
    _check_latency(latency)
    if incidence not in _links.PROBABILITY_LINKS:
        raise ValueError(
            "incidence = {!r} is not a link of a probability; use one of "
            "{}.".format(incidence, list(_links.PROBABILITY_LINKS))
        )
    given = dict(links or {})
    if INCIDENCE in given and given[INCIDENCE] != incidence:
        raise ValueError(
            "links['lfp_p'] and incidence both name the incidence's link; "
            "give incidence only."
        )
    given[INCIDENCE] = incidence
    return MixtureCureFitter(_cured(latency.name), covariates, given)
