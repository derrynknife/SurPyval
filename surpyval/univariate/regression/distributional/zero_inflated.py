"""Zero-inflated regression: covariates on the share of units that fail at
time zero and on the lifetime of the rest.

Some units are dead on arrival: they fail the moment they are switched
on, or are found failed before use. A unit is dead on arrival with
probability :math:`f_0(z)`; a unit that works fails later, at a time with
distribution function :math:`F_u(t \\mid z)`, so the population's
distribution function starts with a jump at 0,

.. math::
    F(t \\mid z) = f_0(z) + \\bigl(1 - f_0(z)\\bigr)\\, F_u(t \\mid z).

``ZeroInflated(Weibull)`` is ``Weibull.fit(..., zi=True)`` with
covariates: the share :math:`f_0(z)` (the univariate fit's ``f0``) has
its own linear predictor through a logit (or probit, or complementary
log-log) link, and the *latency* :math:`F_u` is the distribution with
covariates on its parameters, as in a distributional regression. It is
the mirror image of the mixture-cure regression (:mod:`.mixture_cure`),
whose extra mass sits at infinity rather than at 0.

The likelihood is the regressions' own, of the zero-inflated
distribution: a failure at 0 contributes :math:`f_0`, a failure at
:math:`t > 0` contributes :math:`(1 - f_0) f_u(t)`, a unit still working
at :math:`t` contributes :math:`(1 - f_0) S_u(t)`, and a unit known only
to have failed by :math:`t` (left censored, or in an interval that starts
at 0) contributes :math:`f_0 + (1 - f_0) F_u(t)`.

Two routes reach its maximum:

* When every row says whether its unit was dead on arrival (no left
  censoring, no interval starting at 0, no right truncation without late
  entry), the likelihood factorises into a binary regression of "dead on
  arrival, yes or no" and the latency's regression on the units that
  worked. The two share no parameter, so each is fitted on its own and
  together they are the joint maximum, with a block-diagonal covariance
  ("two-stage").
* Otherwise a row that does not say couples the parts, and both are
  fitted together ("joint").
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, NamedTuple

import autograd.numpy as np
import numpy as onp
import numpy.typing as npt
from autograd import grad, hessian

from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
)
from surpyval.utils.aliases import canonical_name
from surpyval.utils.shapes import check_paired_rows, covariate_rows
from surpyval.utils.surpyval_data import SurpyvalData, accepts_surpyval_data
from surpyval.utils.units import takes_unit
from surpyval.utils.warnings import ignoring_warnings

from .._aliasing import (
    aliased_columns,
    constant_columns,
    dataframe_covariates,
    fit_columns,
)
from .._fit_skeleton import covariate_center
from ..regression_data import truncation_window
from . import _links
from .fitter import DistributionalFitter, uniform_draws
from .model import DistributionalRegressionModel

#: The latency distributions a zero-inflated regression takes (lifetimes
#: on the positive half-line), each with the parameter that sets how long
#: a working unit lasts: the one the covariates are on by default.
LATENCIES = {
    "Exponential": "failure_rate",
    "Weibull": "alpha",
    "Gamma": "beta",
    "LogNormal": "mu",
    "LogLogistic": "alpha",
}

#: The name of the dead-on-arrival probability, the univariate fits'
#: name for the proportion failing at time 0.
INCIDENCE = "f0"

#: The smallest and largest starting value of ``f0``.
_START = (0.02, 0.98)

#: The routes of a fit.
METHODS = ("auto", "two-stage", "joint")


class ZeroInflatedDistribution:
    """A lifetime distribution with a share of units failing at time 0:
    the latency distribution's parameters, then ``f0``, the probability
    of failing at 0. Its functions are the population's for ``x > 0``
    (``ff`` starts at ``f0``); the fitter supplies their values at 0.
    ``log_df`` at 0 is the log of the jump, ``log f0``, the likelihood of
    a failure at 0."""

    def __init__(self, latency: Any) -> None:
        self.latency = latency
        self.name = latency.name + "ZeroInflated"
        self.parameter_names = [*latency.parameter_names, INCIDENCE]
        self.bounds = (*latency.bounds, (0, 1))
        self.support = latency.support
        self.k = len(self.parameter_names)

    def __reduce__(self) -> Any:
        return (_zero_inflated, (self.latency.name,))

    def __repr__(self) -> str:
        return f"{self.name} (a {self.latency.name} latency)"

    def sf(self, x: Numeric, *params: Boxable) -> Boxable:
        *lat, f0 = params
        return (1.0 - f0) * self.latency.sf(x, *lat)

    def ff(self, x: Numeric, *params: Boxable) -> Boxable:
        *lat, f0 = params
        return f0 + (1.0 - f0) * self.latency.ff(x, *lat)

    def df(self, x: Numeric, *params: Boxable) -> Boxable:
        """The density of the failures after 0 (the jump at 0, ``f0``,
        is not a density)."""
        *lat, f0 = params
        return (1.0 - f0) * self.latency.df(x, *lat)

    def hf(self, x: Numeric, *params: Boxable) -> Boxable:
        """The hazard after 0: the latency's (a unit still working at
        ``x > 0`` worked at 0)."""
        *lat, _ = params
        return self.latency.hf(x, *lat)

    def Hf(self, x: Numeric, *params: Boxable) -> Boxable:
        return -self.log_sf(x, *params)

    def log_sf(self, x: Numeric, *params: Boxable) -> Boxable:
        *lat, f0 = params
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log1p(-f0) + self.latency.log_sf(x, *lat)

    def log_ff(self, x: Numeric, *params: Boxable) -> Boxable:
        *lat, f0 = params
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.logaddexp(
                np.log(f0), np.log1p(-f0) + self.latency.log_ff(x, *lat)
            )

    def log_df(self, x: Numeric, *params: Boxable) -> Boxable:
        """The log-likelihood of a failure at ``x``: ``log f0`` at 0 (the
        jump), ``log(1 - f0) + log f_u(x)`` after."""
        *lat, f0 = params
        x = onp.asarray(x, dtype=float)
        zero = x == 0
        # The latency's density at 0 can be infinite (a Weibull shape
        # below 1), so it is evaluated at a stand-in there: np.where
        # evaluates both branches, and an infinity would put a nan in the
        # gradient of the one selected.
        safe = onp.where(zero, 1.0, x)
        with np.errstate(divide="ignore", invalid="ignore"):
            after = np.log1p(-f0) + self.latency.log_df(safe, *lat)
            return np.where(zero, np.log(f0) * np.ones(after.shape), after)

    def qf(self, u: Numeric, *params: Boxable) -> Boxable:
        """The population's quantile: 0 up to ``f0``, the units dead on
        arrival, then the latency's."""
        *lat, f0 = params
        u = onp.asarray(u, dtype=float)
        with onp.errstate(all="ignore"):
            ratio = (u - f0) / (1.0 - f0)
            after = ratio > 0
            value = self.latency.qf(onp.where(after, ratio, 0.5), *lat)
            out = onp.where(after, value, 0.0)
        return onp.where(onp.isnan(ratio), onp.nan, out)

    def mean(self, *params: float) -> float:
        *lat, f0 = params
        return float((1.0 - f0) * self.latency.mean(*lat))

    def fit_from_surpyval_data(self, data: Any) -> Any:
        """The start of the search: the zero-inflated fit without
        covariates, its share kept off the ends."""
        with ignoring_warnings():
            try:
                fitted = self.latency.fit_from_surpyval_data(data, zi=True)
                params = [float(v) for v in fitted.params]
                f0 = float(fitted.f0)
            except Exception:
                worked = data[~_doa_rows(data)]
                fitted = self.latency.fit_from_surpyval_data(worked)
                params = [float(v) for v in fitted.params]
                f0 = 0.1
        if not onp.all(onp.isfinite(params)) or not onp.isfinite(f0):
            fitted = self.latency.fit_from_surpyval_data(
                data[~_doa_rows(data)]
            )
            params, f0 = [float(v) for v in fitted.params], 0.1

        class _Start:
            pass

        start = _Start()
        start.params = onp.array(  # type: ignore[attr-defined]
            [*params, min(max(f0, _START[0]), _START[1])]
        )
        return start

    def fit(self, **kwargs: Any) -> Any:
        """The zero-inflated fit without covariates (the null model of
        the global likelihood-ratio test), on the rows as the regression
        reads them (an interval starting at 0 holds the jump at 0)."""
        data = SurpyvalData(
            kwargs.pop("x"),
            kwargs.pop("c", None),
            kwargs.pop("n", None),
            kwargs.pop("t", None),
        )
        x, c, n, t, _ = _zero_inflated_rows(data)
        return self.latency.fit(x=x, c=c, n=n, t=t, zi=True, **kwargs)

    def from_params(self, params: Any) -> Any:
        """The univariate zero-inflated model with these parameters."""
        *lat, f0 = (float(v) for v in params)
        return self.latency.from_params(lat, f0=f0)


_ZERO_INFLATED: dict[str, ZeroInflatedDistribution] = {}


def _zero_inflated(name: str) -> ZeroInflatedDistribution:
    """The zero-inflated form of the latency distribution called ``name``
    (one object per distribution, so fitters and models share it)."""
    import surpyval

    if name not in _ZERO_INFLATED:
        _ZERO_INFLATED[name] = ZeroInflatedDistribution(
            getattr(surpyval, name)
        )
    return _ZERO_INFLATED[name]


def _check_latency(latency: Any) -> None:
    from surpyval.univariate.parametric.parametric_fitter import (
        ParametricFitter,
    )

    name = getattr(latency, "name", None)
    if not isinstance(latency, ParametricFitter):
        raise ValueError(
            "ZeroInflated takes the latency's distribution (Weibull, "
            "LogNormal, ...); got {!r}.".format(latency)
        )
    if canonical_name(latency) not in LATENCIES:
        raise ValueError(
            "Zero-inflated regression is not available with a {} latency: "
            "the time to failure of the units that work is one of {} "
            "(lifetimes on the positive half-line).".format(
                name, ", ".join(LATENCIES)
            )
        )


# -- the rows -----------------------------------------------------------------


def _bounds_of(data: SurpyvalData) -> tuple[npt.NDArray, npt.NDArray]:
    x = onp.asarray(data.x, dtype=float)
    if x.ndim == 2:
        return x[:, 0], x[:, 1]
    return x, x


def _doa_rows(data: SurpyvalData) -> npt.NDArray:
    """The rows of units that failed at time 0."""
    lo, _ = _bounds_of(data)
    return (onp.asarray(data.c) == 0) & (lo == 0)


def _zero_inflated_rows(
    data: SurpyvalData,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, Any]:
    """The rows as a zero-inflated fit reads them, ``(x, c, n, t, w)``.

    An interval that starts at (or before) 0 holds the jump at 0, so it
    is the same as left censoring at its upper end ("failed by
    ``xr``"), and becomes that; one that ends at 0 is a failure at 0, as
    is left censoring at 0. A left truncation at or before 0 is no
    truncation (a unit cannot enter before it exists), and becomes
    ``-inf``. Times below 0 are refused."""
    lo, hi = _bounds_of(data)
    c = onp.asarray(data.c).copy()
    n = onp.asarray(data.n).copy()
    t = onp.asarray(data.t, dtype=float).copy()
    negative = ((lo < 0) & (c != -1)) | (hi < 0)
    if onp.any(negative):
        raise ValueError(
            "A zero-inflated regression is of lifetimes, which start at 0; "
            "{} row(s) have a time below 0. Measure the times from when "
            "each unit was put in service.".format(int(onp.sum(negative)))
        )
    lo, hi = lo.copy(), hi.copy()
    from_zero = (c == 2) & (lo <= 0)
    c[from_zero] = -1
    lo[from_zero] = hi[from_zero]
    at_zero = (c == -1) & (hi == 0)
    c[at_zero] = 0
    lo[at_zero] = 0.0
    hi[at_zero] = 0.0
    t[t[:, 0] <= 0, 0] = -onp.inf
    if onp.any(c == 2):
        x: npt.NDArray = onp.column_stack([lo, hi])
        x[c != 2, 0] = hi[c != 2]
    else:
        x = hi
    w = None if data.w is None else onp.asarray(data.w, dtype=float)
    return x, c, n, t, w


def _untruncated_at_zero(t: Any) -> Any:
    """``t`` with a left truncation at or before 0 taken off (``-inf``):
    a unit cannot enter observation before it exists, and a unit dead on
    arrival with ``tl = 0`` is then not refused as entering at its own
    failure time."""
    if t is None:
        return None
    t = onp.array(t, dtype=float)
    if t.ndim == 2 and t.shape[1] == 2:
        t[t[:, 0] <= 0, 0] = -onp.inf
    return t


class RowKinds(NamedTuple):
    """What each row says about whether its unit was dead on arrival."""

    #: Failed at 0.
    doa: npt.NDArray
    #: Known to have worked: a failure or a censoring after 0 (or still
    #: working at 0).
    worked: npt.NDArray
    #: Entered after 0, so could not have been dead on arrival: it says
    #: nothing about ``f0``.
    late: npt.NDArray
    #: Does not say: left censored, or right truncated without late
    #: entry (its window holds the jump at 0).
    coupled: npt.NDArray


def row_kinds(x: npt.NDArray, c: npt.NDArray, t: npt.NDArray) -> RowKinds:
    """Each row's kind (of the rows from :func:`_zero_inflated_rows`)."""
    x = onp.asarray(x, dtype=float)
    lo = x[:, 0] if x.ndim == 2 else x
    c = onp.asarray(c)
    late = t[:, 0] > 0
    coupled = ~late & ((c == -1) | onp.isfinite(t[:, 1]))
    doa = (c == 0) & (lo == 0) & ~coupled
    worked = ~doa & ~coupled & ~late
    return RowKinds(doa, worked, late, coupled)


def _incidence_fit(
    y: npt.NDArray, Z: npt.NDArray, weight: npt.NDArray, link: str
) -> npt.NDArray:
    """The binary regression of the first stage: the maximum-likelihood
    ``f0`` at ``Z = 0`` and its coefficients, ``f0(z) = g^-1(g(f0) +
    z'b)``, from the outcomes ``y`` (1 dead on arrival) with weights."""
    from scipy.optimize import minimize

    tiny = float(onp.finfo(float).tiny)
    share = float(onp.sum(weight * y) / onp.sum(weight))
    share = min(max(share, 1e-6), 1 - 1e-6)
    m = Z.shape[1]

    def neg_ll(v: Any) -> Any:
        eta = _links.link(link, share) + v[0]
        if m:
            eta = eta + np.dot(Z, v[1:])
        if link == "logit":
            log_p = -np.logaddexp(0.0, -eta)
            log_q = -np.logaddexp(0.0, eta)
        else:
            p = _links.inverse(link, eta)
            log_p = np.log(np.maximum(p, tiny))
            log_q = np.log(np.maximum(1.0 - p, tiny))
        return -np.sum(weight * (y * log_p + (1.0 - y) * log_q))

    with onp.errstate(all="ignore"):
        res = minimize(
            neg_ll,
            onp.zeros(m + 1),
            jac=grad(neg_ll),
            hess=hessian(neg_ll),
            method="trust-exact",
            options={"gtol": 1e-10},
        )
    v = onp.asarray(res.x, dtype=float)
    f0 = float(_links.inverse(link, _links.link(link, share) + v[0]))
    return onp.r_[f0, v[1:]]


# -- the fitter ---------------------------------------------------------------


class ZeroInflatedFitter(DistributionalFitter):
    """A zero-inflated regression fitter; build one with
    :func:`ZeroInflated`.

    The parameters are the latency distribution's at the reference row
    (``Z = 0``, or the covariate means with ``center=True``), then
    ``f0``, the probability there of failing at time 0, then each
    parameter's coefficients on its link scale, named
    ``<parameter>_<column>``. Every coefficient has SurPyval's sign: a
    positive one means a shorter life (on ``f0``, a higher chance of
    being dead on arrival).
    """

    fitter_kind = "zero-inflated regression fitter"
    name_suffix = ""

    @staticmethod
    def _check_distribution(distribution: Any) -> None:
        if not isinstance(distribution, ZeroInflatedDistribution):
            raise ValueError(
                "ZeroInflatedFitter is built by ZeroInflated(latency), e.g. "
                "ZeroInflated(Weibull)."
            )

    def _model_class(self) -> Any:
        return ZeroInflatedModel

    def _say_verdict(self, verdict: Any, values: dict) -> None:
        """As the family's, except where the only thing running off is
        ``f0`` towards 0: the data then show no units dead on arrival,
        and the warning says so."""
        from surpyval.utils.no_maximum import warn_no_maximum

        from .._fit_skeleton import NO_MAXIMUM_CONSEQUENCE

        f0 = float(values.get(INCIDENCE, onp.nan))
        only_incidence = (
            verdict.no_maximum
            and not verdict.runaway
            and list(verdict.baseline or ()) == [INCIDENCE]
        )
        if only_incidence and f0 < 0.5:
            warn_no_maximum(
                "the likelihood keeps increasing as f0 ({:.4g}), the "
                "probability of failing at time 0 at the reference row, "
                "runs on towards 0: the data show no sign of units dead "
                "on arrival there".format(f0),
                NO_MAXIMUM_CONSEQUENCE,
                "fit the latency alone, without a share failing at 0 (an "
                "AFT model or Distributional({}))".format(
                    self.dist.latency.name
                ),
            )
            return
        super()._say_verdict(verdict, values)

    def _repr_details(self) -> list[str]:
        return [f"{self.dist.latency.name} latency"]

    @property
    def latency(self) -> Any:
        """The distribution of the time to failure of a unit that
        works."""
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
        column on ``f0`` and on the latency's life parameter (its scale,
        location or rate), none on its shape."""
        if covariates is None:
            life = LATENCIES[canonical_name(self.dist.latency)]
            covariates = {INCIDENCE: list(range(p)), life: list(range(p))}
        return super()._resolve_covariates(covariates, p, names)

    def _call(self, fn: str, x: Numeric, Z: Numeric, params: tuple) -> Any:
        """As the family's, with the jump at 0: at ``x = 0`` the
        survival is ``1 - f0`` and the distribution function ``f0`` (the
        family's gives the values at the start of the support, 1 and 0,
        which hold below 0)."""
        out = super()._call(fn, x, Z, params)
        if fn not in _AT_ZERO:
            return out
        xs = np.atleast_1d(np.asarray(x, dtype=float))
        zero = xs == 0
        if not onp.any(zero):
            return out
        f0 = self._dist_params_at(Z, params)[-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            value = _AT_ZERO[fn](f0)
        return np.where(zero, value, out)

    def random(
        self,
        size: int,
        Z: npt.ArrayLike,
        *params: float,
        random_state: Any = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """``size`` draws at each covariate row of ``Z``, row by row, and
        the row each was drawn at. A unit dead on arrival is drawn as
        ``0``, as the univariate zero-inflated model draws it."""
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

    # -- fitting -------------------------------------------------------------

    @dataframe_covariates
    @takes_unit
    @accepts_surpyval_data
    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        t: npt.ArrayLike | None = None,
        init: npt.ArrayLike | None = None,
        fixed: dict[str, float] | None = None,
        center: bool = False,
        tl: npt.ArrayLike | None = None,
        tr: npt.ArrayLike | None = None,
        covariates: Mapping[str, Any] | None = None,
        links: Mapping[str, str] | None = None,
        w: npt.ArrayLike | float | None = None,
        method: str = "auto",
    ) -> Any:
        """
        Fit by maximum likelihood, with every censoring, truncation,
        count and weight of SurPyval's data model.

        A failure at time 0 (``x = 0``, ``c = 0``) is a unit dead on
        arrival. An interval that starts at 0, and left censoring, say
        only that the unit failed by its time, perhaps at 0. A left
        truncation at or before 0 is no truncation.

        Parameters
        ----------
        x, Z, c, n, t, tl, tr, init, fixed, center, covariates, links, w
            As for :meth:`DistributionalFitter.fit`. By default every
            column of ``Z`` acts on ``f0`` and on the latency's life
            parameter.
        method : {"auto", "two-stage", "joint"}, optional
            How the maximum is found. ``"two-stage"``: when every row
            says whether its unit was dead on arrival, the likelihood is
            a binary regression of that (on every row but the late
            entries, which could not have been) times the latency's
            regression on the units that worked; each is fitted on its
            own, and together they are the joint maximum (which the joint
            search then confirms, and whose information it gives).
            ``"joint"``: both parts searched together from the
            zero-inflated fit without covariates. ``"auto"`` (the
            default) is two-stage where the rows allow it, else joint
            (left censoring, an interval starting at 0, or right
            truncation without late entry), and joint for a fit with
            ``fixed`` parameters or an ``init``. ``model.fit_route`` says
            which was used.

        Returns
        -------
        ZeroInflatedModel

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullZeroInflated
        >>> rng = np.random.default_rng(4)
        >>> lot = rng.binomial(1, 0.5, 500)
        >>> doa = rng.random(500) < np.where(lot == 1, 0.15, 0.05)
        >>> x = np.where(doa, 0.0, 100 * rng.weibull(2.0, 500))
        >>> model = WeibullZeroInflated.fit(x, lot)
        >>> model.parameter_names
        ['alpha', 'beta', 'f0', 'alpha_coef_0', 'f0_coef_0']
        >>> model.fit_route
        'two-stage'
        >>> model.doa_probability([[0], [1]]).round(2)
        array([0.06, 0.17])
        """
        if method not in METHODS:
            raise ValueError(
                f"method = {method!r}: use one of {list(METHODS)}."
            )
        t = _untruncated_at_zero(truncation_window(x, t, tl, tr))
        data = SurpyvalData(x, c, n, t, group_and_sort=False, w=w)
        xz, cz, nz, tz, wz = _zero_inflated_rows(data)
        kinds = row_kinds(xz, cz, tz)
        separable = not onp.any(kinds.coupled)
        if method == "two-stage" and not separable:
            raise ValueError(
                "method='two-stage' needs every row to say whether its unit "
                "was dead on arrival; {} row(s) do not (left censored, an "
                "interval starting at 0, or right truncated without late "
                "entry), and couple the two parts. Use method='auto' or "
                "'joint'.".format(int(onp.sum(nz[kinds.coupled])))
            )
        fixed_f0 = fixed is not None and INCIDENCE in fixed
        if separable and not fixed_f0:
            _check_estimable(kinds, nz)
        route = "joint"
        start = init
        if separable and method != "joint" and init is None and not fixed:
            start = self._two_stage_start(
                xz, Z, cz, nz, tz, wz, kinds, covariates, links, center, fixed
            )
            if start is not None:
                route = "two-stage"
        model = DistributionalFitter.fit(
            self,
            xz,
            Z,
            c=cz,
            n=nz,
            t=tz,
            init=start,
            fixed=fixed,
            center=center,
            covariates=covariates,
            links=links,
            w=wz,
        )
        model.fit_route = route
        return model

    def _two_stage_start(
        self,
        x: npt.NDArray,
        Z: Any,
        c: npt.NDArray,
        n: npt.NDArray,
        t: npt.NDArray,
        w: npt.NDArray | None,
        kinds: RowKinds,
        covariates: Mapping[str, Any] | None,
        links: Mapping[str, str] | None,
        center: bool,
        fixed: dict[str, float] | None,
    ) -> npt.NDArray | None:
        """The two stages' estimates, in the order of the model's
        ``params`` (at ``Z = 0``, or at the covariate means for
        ``center=True``), or ``None`` where a stage cannot be fitted (the
        joint search then starts from the fit without covariates)."""
        from .fitter import Distributional

        Z_arr = onp.asarray(Z, dtype=float)
        if Z_arr.ndim == 1:
            Z_arr = Z_arr.reshape(-1, 1)
        finite = onp.all(onp.isfinite(Z_arr), axis=1)
        p = Z_arr.shape[1]
        names = fit_columns()
        if covariates is None:
            covariates = self._covariates
        positions = self._resolve_covariates(covariates, p, names)
        link_names = self._resolve_links(
            self._links_given if links is None else links
        )
        weight = n * (1.0 if w is None else w)
        K = self.k_dist
        lat_names = self.parameter_names[: K - 1]
        try:
            with ignoring_warnings():
                # Stage 1: dead on arrival, yes or no
                rows = (kinds.doa | kinds.worked) & finite
                cols = positions[-1]
                kept = ~_aliased(Z_arr[finite][:, cols], n[finite])
                fitted = _incidence_fit(
                    kinds.doa[rows].astype(float),
                    Z_arr[rows][:, cols[kept]],
                    weight[rows].astype(float),
                    link_names[-1],
                )
                first = onp.zeros(cols.size + 1)
                first[0] = fitted[0]
                first[1:][kept] = fitted[1:]
                # Stage 2: the latency on the units that worked after 0
                lo = x[:, 0] if x.ndim == 2 else x
                rows = ~kinds.doa & ~((c == 1) & (lo == 0)) & finite
                second = Distributional(
                    self.dist.latency,
                    {k: v.tolist() for k, v in zip(lat_names, positions)},
                    dict(zip(lat_names, link_names[:-1])),
                ).fit(
                    x[rows],
                    Z_arr[rows],
                    c=c[rows],
                    n=n[rows],
                    t=t[rows],
                    w=None if w is None else w[rows],
                )
        except Exception:
            return None
        if second.maximum != "verified":
            return None
        lat = onp.asarray(second.params, dtype=float)
        base = onp.r_[lat[: K - 1], first[0]]
        coefs = onp.r_[lat[K - 1 :], first[1:]]
        if center and p:
            fixed_intercept = any(
                name in (fixed or {}) and positions[k].size
                for k, name in enumerate(self.parameter_names)
            )
            at = covariate_center(Z_arr[finite], n[finite])
            if not fixed_intercept and onp.all(onp.isfinite(at)):
                start = 0
                signs = self.signs
                for k, cols in enumerate(positions):
                    m = cols.size
                    if m:
                        shift = signs[k] * float(
                            at[cols] @ coefs[start : start + m]
                        )
                        base[k] = float(
                            _links.move(link_names[k], base[k], shift)
                        )
                    start += m
        return onp.r_[base, coefs]

    @dataframe_covariates
    @takes_unit
    @accepts_surpyval_data
    def fit_penalised(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        t: npt.ArrayLike | None = None,
        penalizer: Any = 0.1,
        l1_ratio: float = 0.0,
        standardise: bool = True,
        tl: npt.ArrayLike | None = None,
        tr: npt.ArrayLike | None = None,
        tol: float = 1e-10,
        w: npt.ArrayLike | float | None = None,
        covariates: Mapping[str, Any] | None = None,
        links: Mapping[str, str] | None = None,
    ) -> Any:
        """
        The family's penalised fit (:meth:`DistributionalFitter.\
fit_penalised`), on the rows as :meth:`fit` reads them: both parts are
        searched together, the penalty on every coefficient.
        """
        t = _untruncated_at_zero(truncation_window(x, t, tl, tr))
        data = SurpyvalData(x, c, n, t, group_and_sort=False, w=w)
        xz, cz, nz, tz, wz = _zero_inflated_rows(data)
        model = DistributionalFitter.fit_penalised(
            self,
            xz,
            Z,
            c=cz,
            n=nz,
            t=tz,
            penalizer=penalizer,
            l1_ratio=l1_ratio,
            standardise=standardise,
            tol=tol,
            w=wz,
            covariates=covariates,
            links=links,
        )
        model.fit_route = "joint"
        return model


#: Each function's value at ``x = 0``, from ``f0``.
_AT_ZERO = {
    "sf": lambda f0: 1.0 - f0,
    "ff": lambda f0: f0,
    "Hf": lambda f0: -np.log1p(-f0),
    "log_sf": lambda f0: np.log1p(-f0),
    "log_ff": lambda f0: np.log(f0),
}


def _aliased(Z: npt.NDArray, n: npt.NDArray) -> npt.NDArray:
    """The columns of ``Z`` the data cannot determine beside an
    intercept, as the family's fit finds them (its coefficients are held
    at 0 there), without its warning."""
    if Z.shape[1] == 0 or Z.shape[0] == 0:
        return onp.zeros(Z.shape[1], dtype=bool)
    constant = constant_columns(Z)
    Zc = Z - covariate_center(Z, n)
    gram = Zc.T @ (onp.asarray(n, dtype=float)[:, None] * Zc)
    out = onp.zeros(Z.shape[1], dtype=bool)
    out[aliased_columns(gram, Z.shape[0], constant)] = True
    return out


def _check_estimable(kinds: RowKinds, n: npt.NDArray) -> None:
    """Refuse data that cannot estimate ``f0``: every row says whether
    its unit was dead on arrival, and none was, or all were."""
    doa = float(onp.sum(n[kinds.doa]))
    worked = float(onp.sum(n[kinds.worked]))
    if doa == 0:
        raise ValueError(
            "No unit failed at time 0 and every row says its unit worked "
            "(or entered late), so f0, the share dead on arrival, is "
            "estimated as 0 and the model is the latency's regression "
            "alone. Fit that instead (Distributional(latency) or an AFT "
            "model); a failure at 0 is x = 0 with c = 0."
        )
    if worked == 0:
        raise ValueError(
            "Every unit that could have been dead on arrival was ({:g} "
            "failed at time 0), so f0 is 1 and there is no life to fit "
            "among the rest. Check that x = 0 marks only the units that "
            "failed on arrival.".format(doa)
        )


# -- the model ----------------------------------------------------------------


class ZeroInflatedModel(DistributionalRegressionModel):
    r"""
    A fitted zero-inflated regression (:func:`ZeroInflated`).

    A unit with covariates :math:`z` is dead on arrival (fails at time 0)
    with probability :math:`f_0(z)`, and a unit that works has the
    latency distribution at that row's parameters:

    .. math::
        F(t \mid z) = f_0(z) + \bigl(1 - f_0(z)\bigr) F_u(t \mid z), \qquad
        g\bigl(f_0(z)\bigr) = g(f_0) + x_{f_0}(z)' b_{f_0} .

    ``sf``, ``ff``, ``Hf``, ``qf``, :meth:`mean` and the bounds are the
    population's: the distribution function jumps to :math:`f_0(z)` at
    0, the quantile function is 0 up to :math:`f_0(z)`, and the mean
    counts the units dead on arrival as lasting 0. ``df`` and ``hf`` are
    those of the failures after 0 (the jump has no density). Each takes
    ``susceptible=True`` for the latency alone: the life of a unit that
    worked.

    ``params`` are the latency's parameters at ``Z = 0`` (at ``center``
    for a fit with ``center=True``), then ``f0``, the probability there
    of failing at time 0, then each parameter's coefficients, named
    ``<parameter>_<column>``. A positive coefficient means a worse
    outcome: on ``f0``, a higher chance of being dead on arrival (``exp``
    of a logit coefficient is the odds ratio); on a scale or location, an
    earlier failure, as in the AFT models.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import WeibullZeroInflated
    >>> rng = np.random.default_rng(4)
    >>> lot = rng.binomial(1, 0.5, 500)
    >>> doa = rng.random(500) < np.where(lot == 1, 0.15, 0.05)
    >>> x = np.where(doa, 0.0, 100 * rng.weibull(2.0, 500))
    >>> model = WeibullZeroInflated.fit(x, lot)
    >>> model.parameter_names
    ['alpha', 'beta', 'f0', 'alpha_coef_0', 'f0_coef_0']
    >>> round(float(model.ff(0, [1])), 2)
    0.17
    """

    _dict_model = "ZeroInflatedModel"
    _repr_title = "Zero-Inflated Regression SurPyval Model"

    #: How the fit found its maximum: ``"two-stage"`` or ``"joint"``
    #: (``None`` for a restored model).
    fit_route: str | None = None

    @classmethod
    def _fitter_from_dict(cls, model_dict: dict, links: tuple) -> Any:
        latency = str(model_dict.get("latency"))
        if canonical_name(latency) not in LATENCIES:
            raise ValueError(f"Unknown latency distribution {latency!r}")
        dist = _zero_inflated(latency)
        return ZeroInflatedFitter(
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
        >>> rng = np.random.default_rng(4)
        >>> lot = rng.binomial(1, 0.5, 500)
        >>> doa = rng.random(500) < np.where(lot == 1, 0.15, 0.05)
        >>> x = np.where(doa, 0.0, 100 * rng.weibull(2.0, 500))
        >>> model = surpyval.WeibullZeroInflated.fit(x, lot)
        >>> restored = surpyval.from_dict(model.to_dict())
        >>> bool(np.allclose(restored.sf(50, [1]), model.sf(50, [1])))
        True
        """
        return super().to_dict()

    def _subclass_dict_entries(self) -> dict:
        return {"latency": self.latency.name}

    @property
    def latency(self) -> Any:
        """The distribution of the time to failure of a unit that
        works."""
        dist: Any = self.distribution
        return dist.latency

    @property
    def incidence_link(self) -> str:
        """The link of ``f0``: ``"logit"``, ``"probit"`` or
        ``"cloglog"``."""
        return str(self.links[-1])

    # -- the two parts ------------------------------------------------------

    def _parts(self, Z: Any) -> tuple[list[npt.NDArray], npt.NDArray]:
        """The latency's parameters and ``f0`` at each row of ``Z``."""
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

    def doa_probability(self, Z: Any) -> Any:
        """
        The probability that a unit with covariates ``Z`` is dead on
        arrival (fails at time 0), :math:`f_0(z)`: one value per row of
        ``Z`` (a number for one row).

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullZeroInflated
        >>> rng = np.random.default_rng(4)
        >>> lot = rng.binomial(1, 0.5, 500)
        >>> doa = rng.random(500) < np.where(lot == 1, 0.15, 0.05)
        >>> x = np.where(doa, 0.0, 100 * rng.weibull(2.0, 500))
        >>> model = WeibullZeroInflated.fit(x, lot)
        >>> round(float(model.doa_probability([1])), 2)
        0.17
        """
        _, f0 = self._parts(Z)
        return float(f0[0]) if f0.size == 1 else f0

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
        is ``1 - f0(z)`` from time 0 on (the units dead on arrival have
        failed). ``susceptible=True`` gives the latency's, the survival
        of a unit that worked.
        """
        if susceptible:
            return self._latency_call("sf", x, Z)
        return super().sf(x, Z, **kw)

    def ff(
        self, x: npt.ArrayLike, Z: Any, *, susceptible: bool = False, **kw: Any
    ) -> npt.NDArray:
        """Failure probability by ``x``; as :meth:`sf`. The population's
        starts at :meth:`doa_probability` at time 0."""
        if susceptible:
            return self._latency_call("ff", x, Z)
        return super().ff(x, Z, **kw)

    def df(
        self, x: npt.ArrayLike, Z: Any, *, susceptible: bool = False, **kw: Any
    ) -> npt.NDArray:
        """Density at ``x``: the population's of the failures after 0,
        ``(1 - f0) f_u``; ``susceptible=True`` gives ``f_u``."""
        if susceptible:
            return self._latency_call("df", x, Z)
        return super().df(x, Z, **kw)

    def hf(
        self, x: npt.ArrayLike, Z: Any, *, susceptible: bool = False, **kw: Any
    ) -> npt.NDArray:
        """Hazard rate at ``x`` after 0: a unit still working there
        worked at 0, so it is the latency's either way."""
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
        population's is 0 up to the probability of being dead on arrival;
        ``susceptible=True`` gives the latency's, the age by which a
        fraction ``p`` of the units that worked have failed.
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

    def mean(self, Z: Any, *, susceptible: bool = False) -> Any:
        """
        The mean life of a unit with covariates ``Z``: the population's
        counts a unit dead on arrival as lasting 0, ``(1 - f0) E[T_u]``;
        ``susceptible=True`` gives the latency's, the mean life of the
        units that worked.
        """
        lat, f0 = self._parts(Z)
        out = onp.full(f0.shape, onp.nan)
        with onp.errstate(all="ignore"):
            for j in onp.flatnonzero(~onp.isnan(f0)):
                out[j] = float(self.latency.mean(*(float(v[j]) for v in lat)))
        if onp.any(onp.isposinf(out)):
            warnings.warn(
                "The latency's mean is infinite at {} of the {} covariate "
                "rows (for example a log-logistic shape at or below 1). "
                "Use a quantile, qf(p, Z), instead.".format(
                    int(onp.sum(onp.isposinf(out))), out.size
                ),
                UserWarning,
                stacklevel=2,
            )
        if not susceptible:
            out = (1.0 - f0) * out
        return float(out[0]) if out.size == 1 else out

    def distribution_at(self, Z: Any) -> Any:
        """
        The distribution of a unit with the covariate row ``Z``, as a
        univariate zero-inflated model (``Weibull.from_params(...,
        f0=...)``), with every univariate function.
        """
        return super().distribution_at(Z)

    def _times_at_hazard(
        self, params: Any, center: Any, H_target: Any, Z: Any
    ) -> npt.NDArray:
        """The times at which the population's cumulative hazard reaches
        ``H_target``, row by row of ``Z`` (the parametric bootstrap's
        draws): the population's quantile function, in closed form, 0
        where the target is within the jump at 0."""
        rows = onp.asarray(self._centred(Z, center), dtype=float)
        with onp.errstate(all="ignore"):
            u = -onp.expm1(-onp.asarray(H_target, dtype=float))
            values = self.model._dist_params_at(rows, params)
            values = [
                onp.asarray(v, dtype=float) * onp.ones(rows.shape[0])
                for v in values
            ]
            return onp.asarray(self.distribution.qf(u, *values), dtype=float)


def ZeroInflated(
    latency: Any,
    incidence: str = "logit",
    covariates: Mapping[str, Any] | None = None,
    links: Mapping[str, str] | None = None,
) -> ZeroInflatedFitter:
    r"""
    A zero-inflated regression fitter: a share of the units fails at time
    0 (dead on arrival), and covariates act both on that share and on the
    lifetime of the rest.

    .. math::
        F(t \mid z) = f_0(z) + \bigl(1 - f_0(z)\bigr) F_u(t \mid z)

    :math:`f_0(z)` is the probability of failing at time 0 (``f0`` in the
    model), with ``g(f_0(z))`` linear in the covariates; :math:`F_u` is
    the ``latency`` distribution with covariates on its parameters. It is
    ``latency.fit(..., zi=True)`` with covariates, and the mirror image
    of :func:`MixtureCure`. Where every row says whether its unit was
    dead on arrival the two parts are fitted in two stages (a binary
    regression, then the latency's regression on the units that worked),
    which is the joint maximum; otherwise together (see
    :meth:`ZeroInflatedFitter.fit`).

    Parameters
    ----------
    latency : ParametricFitter
        The distribution of the time to failure of a unit that works:
        Weibull, LogNormal, LogLogistic, Gamma or Exponential.
    incidence : {"logit", "probit", "cloglog"}, optional
        The link of ``f0``. Default ``"logit"`` (logistic regression:
        ``exp`` of a coefficient is an odds ratio).
    covariates : dict, optional
        The default of ``fit``'s ``covariates``: each parameter's
        columns of ``Z``, e.g. ``{"f0": [0, 1], "alpha": [0]}``. By
        default every column acts on ``f0`` and on the latency's life
        parameter (the scale ``alpha`` of a Weibull or LogLogistic,
        ``mu`` of a LogNormal, the rate of a Gamma or Exponential), and
        none on its shape.
    links : dict, optional
        A link per latency parameter (the default is log for a positive
        parameter and identity for one on the whole line).

    Returns
    -------
    ZeroInflatedFitter
        With ``fit``, ``fit_from_df`` (``formulas={"f0": ..., "alpha":
        ...}`` gives each part its own formula) and ``fit_penalised``,
        each returning a :class:`ZeroInflatedModel`.

    Examples
    --------
    >>> from surpyval import ZeroInflated, Weibull
    >>> ZeroInflated(Weibull)
    WeibullZeroInflated: zero-inflated regression fitter (Weibull latency)
    >>> ZeroInflated(Weibull, incidence="probit").links
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
            "links['f0'] and incidence both name the link of f0; give "
            "incidence only."
        )
    given[INCIDENCE] = incidence
    return ZeroInflatedFitter(_zero_inflated(latency.name), covariates, given)
