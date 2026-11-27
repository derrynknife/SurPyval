"""Promotion-time (bounded cumulative hazard) cure regression.

Some units never have the event. The mixture-cure model
(:mod:`.mixture_cure`) splits the population into the cured and the
susceptible. The promotion-time model (Yakovlev and Tsodikov 1996; Chen,
Ibrahim and Sinha 1999) gets a cured fraction another way: a unit's
cumulative hazard rises to a ceiling :math:`\\theta` instead of without
end,

.. math::
    H(t \\mid z) = \\theta(z)\\, F_0(t), \\qquad
    S(t \\mid z) = \\exp\\{-\\theta(z)\\, F_0(t)\\},

with :math:`F_0` a lifetime distribution function. The survival levels
off at the cure fraction :math:`\\exp\\{-\\theta(z)\\}`. Its biological
reading: a unit carries a Poisson number, mean :math:`\\theta`, of latent
causes (say, cancer cells left after treatment), each of which would
cause the event after a time with distribution :math:`F_0`; the unit is
cured if it carries none. With the covariates on :math:`\\theta` alone
(through a log link, the default) it is a proportional hazards model,
:math:`H(t \\mid z) = e^{b'z} H_0(t)`, whose baseline cumulative hazard
:math:`H_0 = \\theta F_0` is bounded.

As the mixture cure model is, it is a distributional regression of a
*defective* distribution: the latency's parameters and one more,
``theta``. The likelihood is the regressions' own (every censoring,
truncation, count and weight of SurPyval's data model), a failure
contributing :math:`\\theta f_0 e^{-\\theta F_0}`, a unit still running
:math:`e^{-\\theta F_0}`.
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
from surpyval.utils.aliases import canonical_name
from surpyval.utils.shapes import check_paired_rows, covariate_rows

from . import _links
from .fitter import DistributionalFitter, uniform_draws
from .mixture_cure import _follow_up
from .model import DistributionalRegressionModel

#: The distributions :math:`F_0` a promotion-time model takes: the
#: continuous lifetimes on the positive half-line whose distribution
#: function rises to 1. (A Gompertz with a negative shape does not, so
#: it is not offered.)
LATENCIES = (
    "Weibull",
    "Exponential",
    "Gamma",
    "LogNormal",
    "LogLogistic",
    "Rayleigh",
    "InverseGaussian",
    "BirnbaumSaunders",
    "GeneralizedGamma",
    "ExpoWeibull",
)

#: The ceiling of the cumulative hazard; the cure fraction is exp(-theta).
THETA = "theta"

#: The smallest and largest starting value of theta (cure fractions of
#: 0.98 and 0.02).
_START = (-onp.log(0.98), -onp.log(0.02))


class PromotionTimeDistribution:
    """A lifetime distribution whose cumulative hazard is bounded: the
    latency's parameters, then ``theta``, with :math:`S(t) =
    \\exp\\{-\\theta F_0(t)\\}`."""

    def __init__(self, latency: Any) -> None:
        self.latency = latency
        self.name = latency.name + "PromotionTimeCure"
        self.parameter_names = [*latency.parameter_names, THETA]
        self.bounds = (*latency.bounds, (0, None))
        self.support = latency.support
        self.k = len(self.parameter_names)

    def __reduce__(self) -> Any:
        return (_promotion_time, (self.latency.name,))

    def __repr__(self) -> str:
        return f"{self.name} (a {self.latency.name} latency)"

    def _F0(self, x: Numeric, lat: Any) -> Boxable:
        return self.latency.ff(x, *lat)

    def Hf(self, x: Numeric, *params: Boxable) -> Boxable:
        *lat, theta = params
        return theta * self._F0(x, lat)

    def log_sf(self, x: Numeric, *params: Boxable) -> Boxable:
        return -self.Hf(x, *params)

    def sf(self, x: Numeric, *params: Boxable) -> Boxable:
        return np.exp(-self.Hf(x, *params))

    def ff(self, x: Numeric, *params: Boxable) -> Boxable:
        return -np.expm1(-self.Hf(x, *params))

    def log_ff(self, x: Numeric, *params: Boxable) -> Boxable:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(-np.expm1(-self.Hf(x, *params)))

    def hf(self, x: Numeric, *params: Boxable) -> Boxable:
        *lat, theta = params
        return theta * self.latency.df(x, *lat)

    def df(self, x: Numeric, *params: Boxable) -> Boxable:
        return self.hf(x, *params) * self.sf(x, *params)

    def log_df(self, x: Numeric, *params: Boxable) -> Boxable:
        *lat, theta = params
        with np.errstate(divide="ignore", invalid="ignore"):
            return (
                np.log(theta)
                + self.latency.log_df(x, *lat)
                - theta * self._F0(x, lat)
            )

    def qf(self, u: Numeric, *params: Boxable) -> Boxable:
        """The population's quantile: infinite at and above the
        probability of ever failing, ``1 - exp(-theta)``."""
        *lat, theta = params
        u = onp.asarray(u, dtype=float)
        with onp.errstate(all="ignore"):
            # F0(t) = -log(1 - u) / theta
            target = -onp.log1p(-u) / theta
            inside = target < 1
            value = self.latency.qf(onp.where(inside, target, 0.5), *lat)
            out = onp.where(inside, value, onp.inf)
        return onp.where(onp.isnan(target), onp.nan, out)

    def mean(self, *params: float) -> float:
        # Some units never fail, whatever theta (exp(-theta) > 0)
        return float("inf")

    def _fit_from_surpyval_data(self, data: Any) -> Any:
        """The start of the search: the latency and cure fraction of the
        mixture-cure (limited-failure) fit without covariates, the cure
        fraction ``1 - lfp_p`` read as ``exp(-theta)``."""
        from surpyval.utils.warnings import ignoring_warnings

        with ignoring_warnings():
            try:
                fitted = self.latency._fit_from_surpyval_data(data, lfp=True)
                params = [float(v) for v in fitted.params]
                p = float(fitted.lfp_p)
            except Exception:
                fitted = self.latency._fit_from_surpyval_data(data)
                params = [float(v) for v in fitted.params]
                p = 0.5
        if not onp.all(onp.isfinite(params)) or not onp.isfinite(p):
            fitted = self.latency._fit_from_surpyval_data(data)
            params, p = [float(v) for v in fitted.params], 0.5
        theta = -onp.log1p(-min(max(p, 0.02), 0.98))

        class _Start:
            pass

        start = _Start()
        start.params = onp.array(  # type: ignore[attr-defined]
            [*params, min(max(theta, _START[0]), _START[1])]
        )
        return start

    def fit(self, **kwargs: Any) -> Any:
        """The promotion-time fit without covariates (the null model of
        the global likelihood-ratio test)."""
        x = kwargs.pop("x")
        n = len(onp.atleast_1d(onp.asarray(x, dtype=float)))
        if onp.ndim(x) == 2:
            n = onp.asarray(x).shape[0]
        fitter = PromotionTimeCureFitter(self)
        return fitter.fit(x, onp.zeros((n, 0)), covariates={}, **kwargs)

    def from_params(self, params: Any) -> Any:
        """Refused: SurPyval has no univariate promotion-time model."""
        raise ValueError(
            "A promotion-time cure model has no univariate distribution "
            "object in SurPyval, so distribution_at is not available. Use "
            "the model's own functions at the row: sf(x, Z), ff, hf, Hf, "
            "df and qf, and cure_probability(Z)."
        )


_PROMOTION: dict[str, PromotionTimeDistribution] = {}


def _promotion_time(name: str) -> PromotionTimeDistribution:
    """The promotion-time form of the latency distribution called
    ``name`` (one object per distribution, so fitters and models share
    it)."""
    import surpyval

    if name not in _PROMOTION:
        _PROMOTION[name] = PromotionTimeDistribution(getattr(surpyval, name))
    return _PROMOTION[name]


def _check_latency(latency: Any) -> None:
    from surpyval.univariate.parametric.parametric_fitter import (
        ParametricFitter,
    )

    name = getattr(latency, "name", None)
    if not isinstance(latency, ParametricFitter):
        raise ValueError(
            "PromotionTimeCure takes the distribution F0 (Weibull, "
            "LogNormal, ...); got {!r}.".format(latency)
        )
    if canonical_name(latency) == "Gompertz":
        raise ValueError(
            "A promotion-time cure model is not available with a Gompertz "
            "F0: with a negative shape the Gompertz's own distribution "
            "function stops short of 1, so its cure fraction could not be "
            "told apart from theta's. A Gompertz with a negative shape is "
            "itself a cure model: fit Gompertz (or Distributional(Gompertz)) "
            "instead."
        )
    if canonical_name(latency) not in LATENCIES:
        raise ValueError(
            "A promotion-time cure model is not available with a {} F0: "
            "F0 is one of {} (continuous lifetimes on the positive "
            "half-line).".format(name, ", ".join(LATENCIES))
        )


class PromotionTimeCureFitter(DistributionalFitter):
    """A promotion-time cure regression fitter; build one with
    :func:`PromotionTimeCure`.

    The parameters are the latency distribution's at the reference row
    (``Z = 0``, or the covariate means with ``center=True``), then
    ``theta``, the ceiling of the cumulative hazard there (the cure
    fraction is ``exp(-theta)``), then each parameter's coefficients on
    its link scale, named ``<parameter>_<column>``. Every coefficient has
    SurPyval's sign: a positive one means a shorter life (on ``theta``,
    a larger cumulative hazard and a smaller cure fraction; with the log
    link ``exp`` of it is a hazard ratio).
    """

    fitter_kind = "promotion-time cure fitter"
    name_suffix = ""

    @staticmethod
    def _check_distribution(distribution: Any) -> None:
        if not isinstance(distribution, PromotionTimeDistribution):
            raise ValueError(
                "PromotionTimeCureFitter is built by "
                "PromotionTimeCure(latency), e.g. "
                "PromotionTimeCure(Weibull)."
            )

    def _model_class(self) -> Any:
        return PromotionTimeCureModel

    def _say_verdict(self, verdict: Any, values: dict) -> None:
        """As the family's, except where theta runs off towards infinity
        (the cure fraction towards 0), alone or with the latency's
        parameters carrying it there: the data then show no cured
        fraction, and the warning says so."""
        from surpyval.utils.no_maximum import warn_no_maximum

        from .._fit_skeleton import NO_MAXIMUM_CONSEQUENCE

        theta = float(values.get(THETA, onp.nan))
        names = list(verdict.baseline or ())
        theta_runs = (
            verdict.no_maximum and not verdict.runaway and THETA in names
        )
        if theta_runs and theta > 1.0:
            warn_no_maximum(
                "the likelihood keeps increasing as theta ({:.4g}), the "
                "ceiling of the cumulative hazard at the reference row, "
                "runs on towards infinity, and the cure fraction "
                "exp(-theta) towards 0: the data show no sign of units "
                "that never fail there (no plateau at the end of "
                "follow-up), so the cure fraction is not "
                "identified".format(theta),
                NO_MAXIMUM_CONSEQUENCE,
                "fit F0 alone, without a cure fraction (a PH or AFT model, "
                "or Distributional({})), or follow the units for "
                "longer".format(self.dist.latency.name),
            )
            return
        super()._say_verdict(verdict, values)

    def _repr_details(self) -> list[str]:
        return [f"{self.dist.latency.name} F0"]

    @property
    def latency(self) -> Any:
        """The distribution :math:`F_0` of the promotion times."""
        return self.dist.latency

    @property
    def signs(self) -> tuple[float, ...]:
        latency = self.dist.latency.name
        return tuple(
            1.0 if name == THETA else _links.sign(latency, name)
            for name in self.parameter_names
        )

    def _resolve_covariates(
        self,
        covariates: Mapping[str, Any] | None,
        p: int,
        names: list[str] | None,
    ) -> tuple[npt.NDArray, ...]:
        """As the distributional regression's, with this default: every
        column on ``theta`` and none on F0, the proportional hazards
        form."""
        if covariates is None:
            covariates = {THETA: list(range(p))}
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
        ``inf``."""
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


class PromotionTimeCureModel(DistributionalRegressionModel):
    r"""
    A fitted promotion-time cure regression (:func:`PromotionTimeCure`).

    A unit with covariates :math:`z` has the bounded cumulative hazard
    :math:`H(t \mid z) = \theta(z) F_0(t \mid z)`, so

    .. math::
        S(t \mid z) = \exp\{-\theta(z) F_0(t \mid z)\}, \qquad
        P(\text{cured} \mid z) = e^{-\theta(z)}, \qquad
        \log \theta(z) = \log \theta + x_\theta(z)' b_\theta .

    ``sf``, ``ff``, ``df``, ``hf``, ``Hf``, ``qf`` and the bounds are the
    population's: the survival levels off at the cure probability, the
    quantile function is infinite beyond the probability of ever
    failing, and :meth:`mean` is infinite. Each takes
    ``susceptible=True`` for the life of a unit that does fail, whose
    survival is :math:`(S(t) - e^{-\theta}) / (1 - e^{-\theta})`.

    ``params`` are F0's parameters at ``Z = 0`` (at ``center`` for a fit
    with ``center=True``), then ``theta`` there, then each parameter's
    coefficients, named ``<parameter>_<column>``. A positive coefficient
    means a shorter life; on ``theta`` (log link) ``exp`` of it is the
    hazard ratio, the same at every age, as in a Cox model.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import WeibullPromotionTimeCure
    >>> rng = np.random.default_rng(5)
    >>> z = rng.binomial(1, 0.5, 600)
    >>> theta = 0.5 * np.exp(0.8 * z)
    >>> causes = rng.poisson(theta)
    >>> life = np.array([
    ...     (100 * rng.weibull(2.0, k)).min() if k else np.inf
    ...     for k in causes
    ... ])
    >>> x, c = np.minimum(life, 400.0), (life > 400).astype(int)
    >>> model = WeibullPromotionTimeCure.fit(x, z, c=c)
    >>> model.parameter_names
    ['alpha', 'beta', 'theta', 'theta_coef_0']
    >>> model.cure_probability([[0], [1]]).round(2)
    array([0.65, 0.29])
    """

    _dict_model = "PromotionTimeCureModel"
    _repr_title = "Promotion-Time Cure Regression SurPyval Model"

    @classmethod
    def _fitter_from_dict(cls, model_dict: dict, links: tuple) -> Any:
        latency = str(model_dict.get("latency"))
        if canonical_name(latency) not in LATENCIES:
            raise ValueError(f"Unknown latency distribution {latency!r}")
        dist = _promotion_time(latency)
        return PromotionTimeCureFitter(
            dist, links=dict(zip(dist.parameter_names, links))
        )

    def _subclass_dict_entries(self) -> dict:
        return {"latency": self.latency.name}

    @property
    def latency(self) -> Any:
        """The distribution :math:`F_0` of the promotion times."""
        dist: Any = self.distribution
        return dist.latency

    # -- the two parts ------------------------------------------------------

    def _parts(self, Z: Any) -> tuple[list[npt.NDArray], npt.NDArray]:
        """F0's parameters and theta at each row of ``Z``."""
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

    def theta_at(self, Z: Any) -> Any:
        """The ceiling of the cumulative hazard, :math:`\\theta(z)`, at
        each row of ``Z`` (a number for one row)."""
        _, theta = self._parts(Z)
        return float(theta[0]) if theta.size == 1 else theta

    def cure_probability(self, Z: Any) -> Any:
        """
        The probability that a unit with covariates ``Z`` never fails,
        :math:`e^{-\\theta(z)}`: the level its survival curve settles at.
        """
        _, theta = self._parts(Z)
        out = onp.exp(-theta)
        return float(out[0]) if out.size == 1 else out

    def susceptible_probability(self, Z: Any) -> Any:
        """The probability that a unit with covariates ``Z`` ever fails,
        :math:`1 - e^{-\\theta(z)}`."""
        _, theta = self._parts(Z)
        out = -onp.expm1(-theta)
        return float(out[0]) if out.size == 1 else out

    def cure_given_survival(self, x: npt.ArrayLike, Z: Any) -> npt.NDArray:
        r"""
        The probability that a unit still running at age ``x`` is one
        that never fails, :math:`e^{-\theta(z)} / S(x \mid z)
        = \exp\{-\theta(z) (1 - F_0(x))\}`: the cure probability at
        ``x = 0``, rising towards 1. ``x`` and ``Z`` pair as for
        :meth:`sf`.
        """
        t = onp.asarray(x, dtype=float)
        S = onp.asarray(self.sf(t, Z), dtype=float)
        cure = onp.asarray(self.cure_probability(Z), dtype=float)
        with onp.errstate(all="ignore"):
            out = cure / S
        return onp.where(onp.isnan(S), onp.nan, onp.minimum(out, 1.0))

    def _susceptible_call(self, fn: str, x: Any, Z: Any) -> npt.NDArray:
        """The distribution of a unit that does fail, its ``fn`` at
        ``x`` for each row of ``Z``: with :math:`F_0` and
        :math:`q = 1 - e^{-\theta}`, :math:`F_u = (1 - e^{-\theta
        F_0}) / q` and :math:`f_u = \theta f_0 e^{-\theta F_0} / q`."""
        t = onp.asarray(x, dtype=float)
        lat, theta = self._parts(Z)
        check_paired_rows(t.size, theta.size)
        size = max(t.size, theta.size)
        flat = onp.broadcast_to(t.reshape(-1), (size,))
        lat = [onp.broadcast_to(v, (size,)) for v in lat]
        theta = onp.broadcast_to(theta, (size,))
        lower = float(self.latency.support[0])
        with onp.errstate(all="ignore"):
            q = -onp.expm1(-theta)
            if fn == "qf":
                u = flat
                target = -onp.log1p(-u * q) / theta
                out = onp.asarray(self.latency.qf(target, *lat), dtype=float)
            else:
                inside = onp.where(flat <= lower, lower + 1.0, flat)
                F0 = onp.asarray(self.latency.ff(inside, *lat), dtype=float)
                f0 = onp.asarray(self.latency.df(inside, *lat), dtype=float)
                F0 = onp.where(flat <= lower, 0.0, F0)
                f0 = onp.where(flat < lower, 0.0, f0)
                Fu = onp.expm1(-theta * F0) / onp.expm1(-theta)
                # S_u = e^{-theta F0} (1 - e^{-theta (1 - F0)}) / q
                Su = onp.exp(-theta * F0) * -onp.expm1(-theta * (1.0 - F0)) / q
                fu = theta * f0 * onp.exp(-theta * F0) / q
                out = {
                    "sf": Su,
                    "ff": Fu,
                    "df": fu,
                    "hf": fu / Su,
                    "Hf": -onp.log(Su),
                }[fn]
        out = onp.where(onp.isnan(flat + theta), onp.nan, out)
        if t.ndim == 0 and size == 1:
            return out[0]
        return out.reshape(t.shape) if t.size == size else out

    def sf(
        self, x: npt.ArrayLike, Z: Any, *, susceptible: bool = False, **kw: Any
    ) -> npt.NDArray:
        """
        Survival at ``x`` for covariates ``Z``: the population's, which
        levels off at :meth:`cure_probability`. ``susceptible=True``
        gives the survival of a unit that does fail.
        """
        if susceptible:
            return self._susceptible_call("sf", x, Z)
        return super().sf(x, Z, **kw)

    def ff(
        self, x: npt.ArrayLike, Z: Any, *, susceptible: bool = False, **kw: Any
    ) -> npt.NDArray:
        """Failure probability by ``x``; as :meth:`sf`. The population's
        rises to :meth:`susceptible_probability`, not to 1."""
        if susceptible:
            return self._susceptible_call("ff", x, Z)
        return super().ff(x, Z, **kw)

    def df(
        self, x: npt.ArrayLike, Z: Any, *, susceptible: bool = False, **kw: Any
    ) -> npt.NDArray:
        """Density at ``x``; as :meth:`sf`."""
        if susceptible:
            return self._susceptible_call("df", x, Z)
        return super().df(x, Z, **kw)

    def hf(
        self, x: npt.ArrayLike, Z: Any, *, susceptible: bool = False, **kw: Any
    ) -> npt.NDArray:
        """Hazard rate at ``x``; as :meth:`sf`. The population's is
        :math:`\\theta f_0`, which falls towards 0 with :math:`f_0`."""
        if susceptible:
            return self._susceptible_call("hf", x, Z)
        return super().hf(x, Z, **kw)

    def Hf(
        self, x: npt.ArrayLike, Z: Any, *, susceptible: bool = False, **kw: Any
    ) -> npt.NDArray:
        """Cumulative hazard at ``x``; as :meth:`sf`. The population's
        rises to ``theta``."""
        if susceptible:
            return self._susceptible_call("Hf", x, Z)
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
        population's is infinite at and beyond the probability of ever
        failing; ``susceptible=True`` gives the age by which a fraction
        ``p`` of the units that do fail have failed.
        """
        if not susceptible:
            return super().qf(p, Z, grid=grid)
        from surpyval.utils.validation import warn_outside_unit_interval

        u = onp.asarray(p, dtype=float)
        u = onp.where(warn_outside_unit_interval(u), onp.nan, u)
        if grid:
            rows = covariate_rows(self._prepare_Z(Z), self._n_covariates())
            return onp.stack(
                [
                    self._susceptible_call("qf", u.reshape(-1), row)
                    for row in rows
                ]
            )
        return self._susceptible_call("qf", u, Z)

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
        """``quantile_cb``'s bounds, as the mixture cure model's: with
        ``method="lr"`` the upper bound is ``inf`` wherever the
        likelihood region holds parameters whose probability of ever
        failing at the row, ``1 - exp(-theta)``, is at most ``p``: read
        from the profile lower bound on ``theta`` at the row."""
        sup = super()._quantile_cb_rows
        if method != "lr" or bound == "lower":
            return sup(probs, rows, alpha_ci, bound, method, *args, **kwargs)
        side = "two-sided" if bound == "two-sided" else "lower"
        theta = onp.asarray(
            self._parameter_at_rows_cb(
                THETA, rows, alpha_ci, side, "lr", 200, None
            ),
            dtype=float,
        )
        lowest = theta[:, 0] if side == "two-sided" else theta.reshape(-1)
        endless = -onp.expm1(-lowest) <= probs
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
        is infinite (some units never fail; returned as ``inf`` without a
        warning); ``susceptible=True`` gives the mean life of a unit that
        does fail, by numerical integration of its survival.
        """
        from scipy.integrate import quad

        _, theta = self._parts(Z)
        out = onp.full(theta.shape, onp.nan)
        finite = ~onp.isnan(theta)
        if not susceptible:
            out[finite] = onp.inf
        else:
            rows = covariate_rows(self._prepare_Z(Z), self._n_covariates())
            rows = onp.broadcast_to(rows, (theta.size, rows.shape[1]))
            for j in onp.flatnonzero(finite):
                row = rows[j : j + 1]

                def surv(t: float, row: Any = row) -> float:
                    return float(self._susceptible_call("sf", t, row))

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out[j] = quad(surv, 0.0, onp.inf, limit=200)[0]
        return float(out[0]) if out.size == 1 else out

    def distribution_at(self, Z: Any) -> Any:
        """Refused: SurPyval has no univariate promotion-time model; use
        the model's functions at the row (``sf(x, Z)``, ...)."""
        return self.distribution.from_params([])  # raises, saying so

    def _times_at_hazard(
        self, params: Any, center: Any, H_target: Any, Z: Any
    ) -> npt.NDArray:
        """The times at which the population's cumulative hazard reaches
        ``H_target``, row by row of ``Z`` (the parametric bootstrap's
        draws), infinite where the target is past ``theta``."""
        rows = onp.asarray(self._centred(Z, center), dtype=float)
        with onp.errstate(all="ignore"):
            u = -onp.expm1(-onp.asarray(H_target, dtype=float))
            values = self.model._dist_params_at(rows, params)
            values = [
                onp.asarray(v, dtype=float) * onp.ones(rows.shape[0])
                for v in values
            ]
            return onp.asarray(self.distribution.qf(u, *values), dtype=float)

    def follow_up(self) -> dict:
        """
        Whether the units were followed for long enough to tell a cured
        fraction from failures still to come: Maller and Zhou's check,
        as :meth:`MixtureCureModel.follow_up
        <surpyval.univariate.regression.distributional.mixture_cure.MixtureCureModel.follow_up>`.
        """
        data = getattr(self, "data", None)
        if data is None:
            raise ValueError(
                "follow_up needs the data the model was fitted to, which "
                "a restored model does not keep."
            )
        return _follow_up(data)


def PromotionTimeCure(
    latency: Any,
    covariates: Mapping[str, Any] | None = None,
    links: Mapping[str, str] | None = None,
) -> PromotionTimeCureFitter:
    r"""
    A promotion-time (bounded cumulative hazard) cure regression fitter:
    the cumulative hazard rises to a ceiling :math:`\theta(z)` instead
    of without end, so a fraction :math:`e^{-\theta(z)}` of the units
    never fails.

    .. math::
        S(t \mid z) = \exp\{-\theta(z)\, F_0(t)\}

    In words: think of each unit as carrying a random (Poisson) number of
    hidden causes of failure, on average :math:`\theta`, each of which
    would cause the event after a time drawn from :math:`F_0`; the unit
    fails at the first, and never fails if it carries none. The
    covariates act on :math:`\theta` through a log link by default, so
    ``exp`` of a coefficient is a hazard ratio, constant in time: the
    model is a proportional hazards model with the bounded baseline
    cumulative hazard :math:`\theta F_0(t)`. That is the difference from
    :func:`MixtureCure`, where the covariates move the cured fraction
    and the timing of the failures separately.

    Parameters
    ----------
    latency : ParametricFitter
        The distribution :math:`F_0`: Weibull, Exponential, Gamma,
        LogNormal, LogLogistic, Rayleigh, InverseGaussian,
        BirnbaumSaunders, GeneralizedGamma or ExpoWeibull.
    covariates : dict, optional
        The default of ``fit``'s ``covariates``: each parameter's
        columns of ``Z``, e.g. ``{"theta": [0, 1], "alpha": [0]}``. By
        default every column acts on ``theta`` and none on F0.
    links : dict, optional
        A link per parameter (the default is log for a positive
        parameter, ``theta`` included, and identity for one on the whole
        line).

    Returns
    -------
    PromotionTimeCureFitter
        With ``fit``, ``fit_from_df`` and ``fit_penalised``, each
        returning a :class:`PromotionTimeCureModel`.

    Examples
    --------
    >>> from surpyval import PromotionTimeCure, Weibull
    >>> PromotionTimeCure(Weibull)
    WeibullPromotionTimeCure: promotion-time cure fitter (Weibull F0)
    >>> PromotionTimeCure(Weibull).links
    ('log', 'log', 'log')
    """
    _check_latency(latency)
    return PromotionTimeCureFitter(
        _promotion_time(latency.name), covariates, dict(links or {})
    )
