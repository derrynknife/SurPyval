r"""Equivalent time scales: one life measured on two clocks.

Many units age on more than one clock at once. A gas turbine runs for
hours and is started and stopped; a pump runs for hours and is cycled; a
vehicle is driven for kilometres and for engine hours. Each start (cycle,
landing, ...) wears the unit as some number of hours of running would, so
the age that predicts failure is a combination of the two,

.. math::
    T = h + k\,s ,

``h`` the first clock (hours, say), ``s`` the second (starts) and
:math:`k \ge 0` what one unit of the second clock is worth in units of
the first: "one start is worth :math:`k` hours". This is the
*equivalent operating hours* of gas-turbine practice, and the *ideal
time scale* of the statistics literature (Kordonsky and Gertsbakh, 1993;
Duchesne and Lawless, 2000). ``Equivalence(Weibull)`` estimates
:math:`k` from the data, together with the lifetime distribution's
parameters on :math:`T`, by maximum likelihood.

The likelihood
--------------
Each unit is observed at one point :math:`(h_i, s_i)`: at its failure, or
still running. Between installation and that point it is assumed to have
used its two clocks at a constant ratio, :math:`\rho_i = s_i / h_i` starts
per hour, so its combined clock runs at :math:`1 + k \rho_i` per hour.
That makes the model an accelerated-failure-time model on the hours
scale, with acceleration factor :math:`1 + k\rho_i`: the unit's life in
hours has survival :math:`S(h(1 + k\rho_i); \theta)`, and so

- a failure contributes the density *in hours*,
  :math:`f(T_i; \theta)\,(1 + k\rho_i)`, :math:`T_i = h_i + k s_i`;
- a unit still running contributes :math:`S(T_i; \theta)`; one that had
  failed before it was inspected (left-censored), :math:`F(T_i; \theta)`;
- a unit that entered observation late, at :math:`(h_{l,i}, s_{l,i})`, is
  divided by :math:`S(h_{l,i} + k s_{l,i}; \theta)`.

The factor :math:`1 + k\rho_i` is the Jacobian of the change of clock.
Leaving it out -- fitting the distribution to the :math:`T_i` as if they
were ordinary failure times -- gives a likelihood whose maximum in
:math:`k` is badly biased. A larger :math:`k` stretches every unit's
:math:`T`, and a density on a stretched clock is lower by the stretch,
so without the factor the likelihood falls with :math:`k` for a reason
that has nothing to do with the data, and the estimate is pulled
towards 0 however large the sample. The factor is exactly that stretch
for each failure, so it cannot be dropped.

A unit that failed with no hours on its first clock (:math:`h_i = 0`,
:math:`s_i > 0`) used only the second: its combined clock is
:math:`T = k s`, and its density is taken on the starts scale,
:math:`f(k s_i; \theta)\,k`. Such a failure is impossible at
:math:`k = 0`, so it bounds :math:`k` away from 0.

:math:`k` is identifiable only when units differ in how they use the two
clocks: if every unit has the same ratio :math:`\rho`, then
:math:`T = h(1 + k\rho)` for every unit, and :math:`1 + k\rho` is
absorbed by the distribution's scale. The fit refuses such data.

The minimum-CV shortcut of Kordonsky and Gertsbakh (choose :math:`k` to
make the coefficient of variation of the failures' :math:`T` smallest)
is not used: it ignores censoring and truncation and has no likelihood,
so it gives no intervals or tests.

References
----------
Duchesne, T. and Lawless, J. (2000). Alternative time scales and
failure time models. *Lifetime Data Analysis* 6, 157-179.

Kordonsky, K. B. and Gertsbakh, I. B. (1993). Choice of the best time
scale for system reliability analysis. *European Journal of Operational
Research* 65, 235-246.

Self, S. G. and Liang, K.-Y. (1987). Asymptotic properties of maximum
likelihood estimators and likelihood ratio tests under nonstandard
conditions. *Journal of the American Statistical Association* 82,
605-610.
"""

from __future__ import annotations

from typing import Any

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
from autograd import grad, hessian, jacobian
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import chi2, norm

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.univariate.information_criteria import (
    InformationCriteriaMixin,
    ic_sample_size,
)
from surpyval.utils.aliases import canonical_name
from surpyval.utils.linalg import (
    cb_link,
    sf_link_bound,
    standard_errors_of,
    wald_bound_on_support,
)
from surpyval.utils.no_maximum import (
    maximum_entry,
    restored_maximum,
    warn_no_maximum,
    warn_unverified,
)
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.units import takes_unit
from surpyval.utils.validation import (
    BOUNDS,
    check_alpha_ci,
    check_option,
    option_error,
)
from surpyval.utils.warnings import ignoring_warnings

#: The lifetime distributions an equivalent-scale model takes: those on
#: the positive half-line that are closed under a change of scale, so that
#: swapping the clocks' roles is a reparameterisation.
DISTRIBUTIONS = ("Weibull", "LogNormal", "LogLogistic", "Gamma", "Exponential")

#: The functions on the combined scale whose Wald bounds ``cb`` gives.
_CB_ON = ("sf", "R", "ff", "F", "Hf")

#: How far ``log(k / k_ref)`` may run before the fit says ``k`` has no
#: finite maximum (e^25: the first clock no longer matters at all).
_RUNAWAY = 25.0

#: Below this multiple of the reference ``k`` a fitted ``k`` is the
#: boundary value 0.
_ZERO = 1e-9


def _check_distribution(dist: Any) -> None:
    from surpyval.univariate.parametric.parametric_fitter import (
        ParametricFitter,
    )

    if not isinstance(dist, ParametricFitter):
        raise ValueError(
            "Equivalence takes a lifetime distribution (Weibull, "
            "LogNormal, LogLogistic, Gamma or Exponential); got "
            f"{dist!r}."
        )
    if canonical_name(dist) not in DISTRIBUTIONS:
        raise ValueError(
            f"An equivalent time scale is not available with the "
            f"{dist.name} distribution: it takes one of "
            f"{', '.join(DISTRIBUTIONS)} (lifetimes on the positive "
            "half-line, whose scale the combined clock can stretch)."
        )


def _as_1d(values: Any, name: str) -> np.ndarray:
    try:
        out = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be numbers, one per unit; got {values!r}."
        ) from None
    if out.ndim == 0:
        out = out[None]
    if out.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional, one value per unit; got an "
            f"array of shape {out.shape}."
        )
    return out


class _Data:
    """The validated data of a fit: each unit's two clocks ``h`` and
    ``s``, censoring ``c``, counts ``n`` and entry point ``(hl, sl)``."""

    def __init__(
        self,
        x1: Any,
        x2: Any,
        c: Any = None,
        n: Any = None,
        tl: Any = None,
        clocks: tuple[str, str] = ("x1", "x2"),
    ) -> None:
        a, b = clocks
        h = _as_1d(x1, "x1")
        s = _as_1d(x2, "x2")
        if h.size == 0:
            raise ValueError("x1 and x2 are empty: there is nothing to fit.")
        if h.size != s.size:
            raise ValueError(
                f"x1 ({a}) and x2 ({b}) must have one value per unit each; "
                f"got {h.size} and {s.size}."
            )
        for name, clock, v in (("x1", a, h), ("x2", b, s)):
            bad = ~np.isfinite(v)
            if bad.any():
                rows = np.flatnonzero(bad)[:5].tolist()
                label = name if clock == name else f"{name} ({clock})"
                raise ValueError(
                    f"{label} has {int(bad.sum())} missing or "
                    f"infinite value(s), at rows {rows}: a unit's age on "
                    "each clock is needed; drop those rows or fill them."
                )
            if (v < 0).any():
                rows = np.flatnonzero(v < 0)[:5].tolist()
                label = name if clock == name else f"{name} ({clock})"
                raise ValueError(
                    f"{label} has negative values, at rows "
                    f"{rows}: an age on a clock is 0 or more."
                )
        c = np.zeros(h.size) if c is None else _as_1d(c, "c")
        n = np.ones(h.size) if n is None else _as_1d(n, "n")
        if c.size == 1 and h.size > 1:
            c = np.full(h.size, c[0])
        if n.size == 1 and h.size > 1:
            n = np.full(h.size, n[0])
        if c.size != h.size or n.size != h.size:
            raise ValueError(
                "c and n must have one value per unit, as x1 does; got "
                f"{c.size} and {n.size} for {h.size} units."
            )
        if not np.isin(c, (0, 1, -1, 2)).all():
            raise ValueError(
                "c must be 0 (failed), 1 (still running) or -1 (failed "
                f"before it was seen); got {sorted(set(c.tolist()))}."
            )
        if (c == 2).any():
            raise ValueError(
                "Interval-censored units (c=2) are not supported by an "
                "equivalent-scale fit: an interval on two clocks needs "
                "both ends on each, so give each such unit as failed "
                "before its later inspection (c=-1) or still running at "
                "its earlier one (c=1)."
            )
        if not np.isfinite(n).all() or (n <= 0).any():
            raise ValueError(
                "n, the count of each row, must be positive; got "
                f"{n[~np.isfinite(n) | (n <= 0)][:5].tolist()}."
            )
        zero = (h == 0) & (s == 0)
        if (zero & (c != 1)).any():
            rows = np.flatnonzero(zero & (c != 1))[:5].tolist()
            raise ValueError(
                f"Rows {rows} failed at 0 on both clocks ({a} = {b} = 0): "
                "a failure needs some age on one of them. Check those "
                "rows, or drop them."
            )
        if tl is None:
            hl = np.zeros(h.size)
            sl = np.zeros(h.size)
        else:
            tl_arr = np.asarray(tl, dtype=float)
            if tl_arr.ndim == 1 and tl_arr.size == 2:
                tl_arr = np.tile(tl_arr, (h.size, 1))
            if tl_arr.shape != (h.size, 2):
                raise ValueError(
                    "tl, each unit's age at entry, is a pair "
                    f"({a}, {b}) per unit: an array of shape "
                    f"({h.size}, 2); got shape {tl_arr.shape}."
                )
            if not np.isfinite(tl_arr).all() or (tl_arr < 0).any():
                raise ValueError(
                    "tl must hold ages of 0 or more on both clocks, with "
                    "no missing values."
                )
            hl, sl = tl_arr[:, 0], tl_arr[:, 1]
            if (hl > h).any() or (sl > s).any():
                rows = np.flatnonzero((hl > h) | (sl > s))[:5].tolist()
                raise ValueError(
                    f"Rows {rows} entered observation (tl) later than they "
                    "were last seen on one of the clocks: each entry age "
                    "must be at or before the unit's age in x1 and x2."
                )
        # A unit seen still running at 0 on both clocks says nothing.
        keep = ~(zero & (c == 1))
        self.h, self.s, self.c, self.n = h[keep], s[keep], c[keep], n[keep]
        self.hl, self.sl = hl[keep], sl[keep]
        self.clocks = (a, b)
        if not (self.c != 1).any():
            raise ValueError(
                "There are no failures (every c is 1): with no failure "
                "there is nothing to fit a lifetime to."
            )
        self._check_identifiable()
        # Index sets of the likelihood's terms
        obs = self.c == 0
        self.i_obs_h = np.flatnonzero(obs & (self.h > 0))
        self.i_obs_0 = np.flatnonzero(obs & (self.h == 0))
        self.i_obs = np.flatnonzero(obs)
        self.i_right = np.flatnonzero(self.c == 1)
        self.i_left = np.flatnonzero(self.c == -1)
        self.i_tl = np.flatnonzero((self.hl > 0) | (self.sl > 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            self.rho = np.where(self.h > 0, self.s / self.h, np.inf)

    def _check_identifiable(self) -> None:
        a, b = self.clocks
        h, s = self.h, self.s
        if (s == 0).all():
            raise ValueError(
                f"Every unit has {b} = 0, so the data say nothing about "
                f"what one unit of {b} is worth: k cannot be estimated. "
                f"Fit the distribution to {a} alone (e.g. Weibull.fit)."
            )
        if (h == 0).all():
            raise ValueError(
                f"Every unit has {a} = 0, so the data say nothing about "
                f"what {a} is worth beside {b}: k cannot be estimated. "
                f"Fit the distribution to {b} alone."
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = np.where(h > 0, s / h, np.inf)
        finite = rho[np.isfinite(rho)]
        same = (
            np.isfinite(rho).all()
            and finite.size > 0
            and np.allclose(finite, finite[0], rtol=1e-9, atol=0)
        )
        if same:
            raise ValueError(
                f"Every unit used the clocks at the same rate ({b} per "
                f"{a} = {finite[0]:.6g}), so k cannot be estimated: "
                f"then {a} + k {b} = {a} (1 + k * {finite[0]:.6g}) for "
                "every unit, and the factor (1 + k * rate) is absorbed by "
                "the distribution's scale, whatever k is. k is learned "
                "from units that use the clocks at different rates; with "
                f"one rate, fit the distribution to {a} alone."
            )

    @property
    def size(self) -> float:
        return float(self.n.sum())

    def to_arrays(self) -> dict:
        return dict(
            h=self.h, s=self.s, c=self.c, n=self.n, hl=self.hl, sl=self.sl
        )

    def k_ref(self) -> float:
        """A natural unit for ``k``: the typical hours per start, so the
        search is the same whatever units the clocks are in."""
        both = (self.h > 0) & (self.s > 0)
        if both.any():
            return float(np.median(self.h[both]) / np.median(self.s[both]))
        return float(
            np.median(self.h[self.h > 0]) / np.median(self.s[self.s > 0])
        )


def _log_lik(dist: Any, d: _Data, theta: Any, k: Any) -> Any:
    """The log-likelihood in the first clock's scale (see the module
    docstring), as an autograd-differentiable function of the
    distribution's parameters ``theta`` and ``k``."""
    T = d.h + k * d.s
    ll = 0.0
    if d.i_obs.size:
        ll = ll + anp.sum(d.n[d.i_obs] * dist.log_df(T[d.i_obs], *theta))
    if d.i_obs_h.size:
        ll = ll + anp.sum(d.n[d.i_obs_h] * anp.log1p(k * d.rho[d.i_obs_h]))
    if d.i_obs_0.size:
        ll = ll + anp.sum(d.n[d.i_obs_0]) * anp.log(k)
    if d.i_right.size:
        ll = ll + anp.sum(d.n[d.i_right] * dist.log_sf(T[d.i_right], *theta))
    if d.i_left.size:
        ll = ll + anp.sum(d.n[d.i_left] * dist.log_ff(T[d.i_left], *theta))
    if d.i_tl.size:
        Tl = d.hl[d.i_tl] + k * d.sl[d.i_tl]
        pos = Tl > 0
        safe = anp.where(pos, Tl, 1.0)
        ll = ll - anp.sum(
            d.n[d.i_tl] * anp.where(pos, dist.log_sf(safe, *theta), 0.0)
        )
    return ll


def _profile(dist: Any, d: _Data, k: float) -> tuple[float, np.ndarray]:
    """The log-likelihood maximised over the distribution's parameters at
    a fixed ``k``, and where: at fixed ``k`` the fit is the
    distribution's own fit to the combined ages (the Jacobian does not
    involve the distribution's parameters)."""
    T = d.h + k * d.s
    tl = None
    if d.i_tl.size:
        tl = d.hl + k * d.sl
    with ignoring_warnings():
        fitted = dist.fit(T, d.c, d.n, tl=tl)
    theta = np.asarray(fitted.params, dtype=float)
    return float(_log_lik(dist, d, list(theta), k)), theta


class _Transform:
    """The fit's search vector: ``log(k / k_ref)``, then each parameter
    relative to its start (as a log for a positive parameter), so that
    it starts at 0 whatever units the clocks are in."""

    def __init__(self, dist: Any, theta0: Any, k_ref: float) -> None:
        self.positive = [b[0] == 0 for b in dist.bounds]
        self.theta0 = np.asarray(theta0, dtype=float)
        self.k_ref = float(k_ref)

    def to_natural(self, v: Any) -> tuple[list, Any]:
        theta = []
        for j, pos in enumerate(self.positive):
            if pos:
                theta.append(self.theta0[j] * anp.exp(v[j + 1]))
            else:
                theta.append(self.theta0[j] + v[j + 1])
        return theta, self.k_ref * anp.exp(v[0])

    def to_search(self, theta: Any, k: float) -> np.ndarray:
        v = [np.log(k / self.k_ref)]
        for j, pos in enumerate(self.positive):
            if pos:
                v.append(np.log(theta[j] / self.theta0[j]))
            else:
                v.append(theta[j] - self.theta0[j])
        return np.array(v, dtype=float)


def _search_objective(dist: Any, d: _Data, tr: _Transform) -> Any:
    def fun(v: Any) -> Any:
        theta, k = tr.to_natural(v)
        return -_log_lik(dist, d, theta, k)

    return fun


class EquivalenceFitter:
    """A fitter of an equivalent time scale; build one with
    :func:`Equivalence`, or use a prebuilt one
    (``WeibullEquivalence``, ...).

    Examples
    --------
    >>> from surpyval import Equivalence, Weibull
    >>> Equivalence(Weibull)
    WeibullEquivalence: an equivalent time scale x1 + k x2, Weibull on it
    """

    def __init__(self, dist: Any) -> None:
        _check_distribution(dist)
        self.dist = dist
        self.name = canonical_name(dist) + "Equivalence"

    def __repr__(self) -> str:
        return (
            f"{self.name}: an equivalent time scale x1 + k x2, "
            f"{self.dist.name} on it"
        )

    def __reduce__(self) -> Any:
        return (Equivalence, (self.dist,))

    @property
    def parameter_names(self) -> list[str]:
        """The names of a fitted model's ``params``: the distribution's,
        then ``k``."""
        return [*self.dist.parameter_names, "k"]

    @takes_unit
    def fit(
        self,
        x1: npt.ArrayLike,
        x2: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        tl: npt.ArrayLike | None = None,
        clocks: tuple[str, str] | None = None,
        init: npt.ArrayLike | None = None,
    ) -> EquivalenceModel:
        r"""
        Fit the combined clock ``x1 + k x2`` and the distribution on it by
        maximum likelihood.

        Parameters
        ----------
        x1 : array_like
            Each unit's age on the first clock (hours, say): at its
            failure, or when last seen running.
        x2 : array_like
            The same units' ages on the second clock (starts, cycles,
            ...). ``k`` is what one unit of this clock is worth on the
            first.
        c : array_like, optional
            Censoring: 0 failed at that age (the default), 1 still
            running, -1 failed before it was seen.
        n : array_like, optional
            The count of units in each row.
        tl : array_like, optional
            Late entry: each unit's age on both clocks when observation
            began, an array of shape ``(len(x1), 2)`` (or one pair for
            every unit).
        clocks : tuple of two str, optional
            Names of the two clocks, used when the model describes ``k``
            (default ``("x1", "x2")``; ``fit_from_df`` uses the columns'
            names).
        init : array_like, optional
            A start for the search: the distribution's parameters, then
            ``k``. The default start is searched as well, and the better
            answer kept.

        Returns
        -------
        EquivalenceModel

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullEquivalence
        >>> rng = np.random.default_rng(1)
        >>> rate = rng.choice([0.02, 0.1, 0.5], 300)   # starts per hour
        >>> T = 2000 * rng.weibull(2.0, 300)          # equivalent hours
        >>> hours = T / (1 + 10 * rate)               # one start = 10 h
        >>> starts = rate * hours
        >>> model = WeibullEquivalence.fit(hours, starts)
        >>> round(model.k, 1)
        9.8
        """
        names = ("x1", "x2") if clocks is None else tuple(clocks)
        if len(names) != 2:
            raise ValueError(f"clocks names the two clocks; got {names!r}.")
        d = _Data(x1, x2, c, n, tl, clocks=(str(names[0]), str(names[1])))
        return _fit(self, d, init)

    @takes_unit
    def fit_from_df(
        self,
        df: Any,
        x_cols: list[str],
        c_col: str | None = None,
        n_col: str | None = None,
        tl_cols: list[str] | None = None,
        init: npt.ArrayLike | None = None,
    ) -> EquivalenceModel:
        """
        Fit from a DataFrame: ``x_cols`` names the two clocks' columns,
        first the one ``k`` is measured in (``["hours", "starts"]``
        reports what one start is worth in hours); ``tl_cols`` the two
        columns of the ages at entry, in the same order. The columns'
        names name the clocks in the model's description.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from surpyval import WeibullEquivalence
        >>> rng = np.random.default_rng(1)
        >>> rate = rng.choice([0.02, 0.1, 0.5], 300)
        >>> T = 2000 * rng.weibull(2.0, 300)
        >>> hours = T / (1 + 10 * rate)
        >>> df = pd.DataFrame({"hours": hours, "starts": rate * hours})
        >>> model = WeibullEquivalence.fit_from_df(
        ...     df, x_cols=["hours", "starts"])
        >>> model.clocks
        ('hours', 'starts')
        """
        x_cols = list(x_cols)
        if len(x_cols) != 2:
            raise ValueError(
                "x_cols names the two clocks' columns, e.g. "
                f"['hours', 'starts']; got {x_cols!r}."
            )
        for col in [*x_cols, c_col, n_col, *(tl_cols or [])]:
            if col is not None and col not in df.columns:
                raise ValueError(
                    f"Column {col!r} is not in the DataFrame, whose "
                    f"columns are {list(df.columns)}."
                )
        tl = None
        if tl_cols is not None:
            tl_cols = list(tl_cols)
            if len(tl_cols) != 2:
                raise ValueError(
                    "tl_cols names the two columns of the ages at entry, "
                    f"in the order of x_cols; got {tl_cols!r}."
                )
            tl = df[tl_cols].to_numpy(dtype=float)
        return self.fit(
            df[x_cols[0]].to_numpy(dtype=float),
            df[x_cols[1]].to_numpy(dtype=float),
            c=None if c_col is None else df[c_col].to_numpy(),
            n=None if n_col is None else df[n_col].to_numpy(),
            tl=tl,
            clocks=(str(x_cols[0]), str(x_cols[1])),
            init=init,
        )

    @takes_unit
    def from_params(
        self,
        params: npt.ArrayLike,
        clocks: tuple[str, str] | None = None,
    ) -> EquivalenceModel:
        """
        A model with given parameters: the distribution's, then ``k``.
        It predicts but has no bounds (no covariance).

        Examples
        --------
        >>> from surpyval import WeibullEquivalence
        >>> model = WeibullEquivalence.from_params([2000, 2, 10])
        >>> float(model.equivalent(100, 5))
        150.0
        """
        params = np.asarray(params, dtype=float).ravel()
        if params.size != len(self.parameter_names):
            raise ValueError(
                f"params holds {len(self.parameter_names)} values, "
                f"{self.parameter_names}; got {params.size}."
            )
        if params[-1] < 0 or not np.isfinite(params[-1]):
            raise ValueError(f"k must be 0 or more; got {params[-1]}.")
        model = EquivalenceModel()
        model.dist = self.dist
        model.params = params
        model.clocks = ("x1", "x2") if clocks is None else tuple(clocks)
        model.maximum = "not applicable"
        model._covariance = None
        model._neg_ll = None
        model._null_neg_ll = None
        model._ic_n = None
        model.data = None
        return model


def _fit(
    fitter: EquivalenceFitter, d: _Data, init: Any = None
) -> EquivalenceModel:
    from surpyval.univariate.parametric.fitters import (
        is_local_minimum,
        preconditioned_bfgs,
    )

    dist = fitter.dist
    k_ref = d.k_ref()
    can_be_zero = d.i_obs_0.size == 0 and not (
        d.i_left.size and (d.h[d.i_left] == 0).any()
    )

    # The start: the best k of a coarse profile.
    candidates: list[tuple[float, float, np.ndarray]] = []
    for mult in (1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0):
        k = mult * k_ref
        try:
            ll, theta = _profile(dist, d, k)
        except Exception:
            continue
        if np.isfinite(ll):
            candidates.append((ll, k, theta))
    null = None
    if can_be_zero:
        try:
            ll0, theta0 = _profile(dist, d, 0.0)
            if np.isfinite(ll0):
                null = (ll0, theta0)
        except Exception:
            null = None
    if not candidates:
        raise ValueError(
            f"The {dist.name} distribution could not be fitted to the "
            "combined ages at any k: check the data (are there enough "
            "distinct failure ages?)."
        )
    starts = [max(candidates, key=lambda t: t[0])]
    if init is not None:
        init = np.asarray(init, dtype=float).ravel()
        if init.size != len(fitter.parameter_names):
            raise ValueError(
                f"init holds {len(fitter.parameter_names)} values, "
                f"{fitter.parameter_names}; got {init.size}."
            )
        if init[-1] > 0:
            starts.insert(0, (np.nan, float(init[-1]), init[:-1]))

    best = None
    for _, k_start, theta_start in starts:
        tr = _Transform(dist, theta_start, k_start)
        fun = _search_objective(dist, d, tr)
        jac = grad(fun)
        v0 = np.zeros(len(theta_start) + 1)
        with np.errstate(all="ignore"), ignoring_warnings():
            res = preconditioned_bfgs(
                fun, v0, jac=jac, floor=1.0, obj_scale=d.size
            )
        value = float(fun(res.x))
        if not np.isfinite(value):
            continue
        if best is None or value < best[0]:
            best = (value, tr, np.asarray(res.x, dtype=float))
    if best is None:
        raise ValueError(
            "The equivalent-scale search failed from every start: check "
            "the data, or give a start with init."
        )
    value, tr, v = best
    theta_list, k_hat = tr.to_natural(v)
    theta = np.array([float(t) for t in theta_list])
    k_hat = float(k_hat)

    maximum = "verified"
    # On the boundary: k = 0 is at least as good as the search's answer.
    boundary = False
    if null is not None and (
        k_hat < _ZERO * k_ref or null[0] >= -value - 1e-10 * d.size
    ):
        boundary = True
        theta, k_hat = np.asarray(null[1], dtype=float), 0.0
        value = -null[0]

    with np.errstate(all="ignore"):
        if boundary:
            # The distribution's parameters at a verified maximum, and the
            # likelihood falling as k moves off 0.
            tr0 = _Transform(dist, theta, 1.0)

            def fun_theta(u: Any) -> Any:
                th, _ = tr0.to_natural(anp.concatenate([anp.zeros(1), u]))
                return -_log_lik(dist, d, th, 0.0)

            ok = is_local_minimum(
                fun_theta,
                grad(fun_theta),
                hessian(fun_theta),
                np.zeros(len(theta)),
                floor=1.0,
                obj_scale=d.size,
            )
            slope = grad(lambda kk: _log_lik(dist, d, list(theta), kk))(0.0)
            ok = ok and float(slope) * k_ref / d.size < 1e-4
            if not ok:
                maximum = "unverified"
        elif v[0] - np.log(k_ref / tr.k_ref) > _RUNAWAY:
            maximum = "no finite maximum"
        else:
            fun = _search_objective(dist, d, tr)
            if not is_local_minimum(
                fun, grad(fun), hessian(fun), v, floor=1.0, obj_scale=d.size
            ):
                maximum = "unverified"

    a, b = d.clocks
    if maximum == "no finite maximum":
        warn_no_maximum(
            f"k, what one unit of {b} is worth in {a}, runs off towards "
            f"infinity (k = {k_hat:.4g}): the failures are explained by "
            f"{b} alone, and {a} adds nothing",
            "The reported k, its standard error and its bounds are "
            "meaningless",
            f"fit the distribution to {b} alone, or swap the clocks "
            f"(fit({b}, {a})), where the same finding is k = 0",
        )
    elif maximum == "unverified":
        warn_unverified("The equivalent-scale fit")

    model = EquivalenceModel()
    model.dist = dist
    model.params = np.append(theta, k_hat)
    model.clocks = d.clocks
    model.data = d.to_arrays()
    model._data = d
    model.maximum = maximum
    model._neg_ll = float(-_log_lik(dist, d, list(theta), k_hat))
    model._null_neg_ll = None if null is None else float(-null[0])
    model._ic_n = None
    model._covariance = _covariance(dist, d, model.params)
    return model


def _covariance(dist: Any, d: _Data, params: np.ndarray) -> Any:
    """The inverse observed information over (the distribution's
    parameters, ``k``), or ``None`` where it is not positive definite."""

    def nll(phi: Any) -> Any:
        return -_log_lik(
            dist, d, [phi[j] for j in range(len(phi) - 1)], phi[-1]
        )

    with np.errstate(all="ignore"):
        try:
            H = np.asarray(hessian(nll)(np.asarray(params, dtype=float)))
            if not np.isfinite(H).all():
                return None
            np.linalg.cholesky(0.5 * (H + H.T))
            cov = np.linalg.inv(H)
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
            return None
    return 0.5 * (cov + cov.T)


def _broadcast(t: Any, rate: Any) -> tuple[np.ndarray, np.ndarray, tuple]:
    t_arr = np.asarray(t, dtype=float)
    r_arr = np.asarray(rate, dtype=float)
    if (r_arr[np.isfinite(r_arr)] < 0).any():
        raise ValueError(
            "rate, the second clock's units per unit of the first "
            "(starts per hour), must be 0 or more."
        )
    tb, rb = np.broadcast_arrays(t_arr, r_arr)
    return tb.ravel(), rb.ravel(), tb.shape


def _reshape(values: np.ndarray, shape: tuple, two_sided: bool) -> Any:
    out = np.asarray(values, dtype=float)
    out = out.reshape(shape + ((2,) if two_sided else ()))
    if out.ndim == 0:
        return out[()]
    return out


def _one_rate(rate: Any) -> float:
    """``rate`` as the one usage rate ``at_usage`` binds: a finite number,
    0 or more, refused (saying what to do) otherwise."""
    arr = np.asarray(rate, dtype=float)
    if arr.ndim != 0:
        if arr.size != 1:
            raise ValueError(
                "at_usage takes one rate and gives the life of a unit used "
                f"at it; got {arr.size} rates. Call it once per rate, "
                "[model.at_usage(r) for r in rates], or pass every rate to "
                "the *_at_usage methods, model.sf_at_usage(x, rates)."
            )
        arr = arr.reshape(())
    r = float(arr)
    if not np.isfinite(r) or r < 0:
        raise ValueError(
            "rate, the second clock's units per unit of the first (starts "
            f"per hour), must be a finite number, 0 or more; got {rate!r}."
        )
    return r


class LifeAtUsage:
    r"""
    The lifetime distribution of a unit used at one constant rate: a
    fitted equivalent-scale model (:func:`Equivalence`) with the rate
    bound, as a plain life model on the first clock.

    ``model.at_usage(rate)`` returns one. ``rate`` is the second clock's
    units per unit of the first, in the model's own units: starts per
    hour for a model fitted to ``(hours, starts)``. A unit used at that
    rate ages :math:`a = 1 + k \cdot rate` units of the combined clock per
    unit of the first, so its life on the first clock is the combined life
    divided by :math:`a`. Every function takes ages (or probabilities) on
    the first clock only, as a univariate model's do, and answers on the
    first clock: ``sf(x)`` is ``model.sf_at_usage(x, rate)``, and so for
    ``ff``, ``hf``, ``qf``, ``mean``, ``cb`` (``cb_at_usage``) and
    ``quantile_cb`` (``quantile_cb_at_usage``); ``df(x)`` is the density
    per unit of the first clock, :math:`a f(a x)`, and ``Hf(x)`` the
    cumulative hazard :math:`H(a x)`. The second clock's age at a first
    clock age ``x`` is ``rate * x``.

    ``random(size)`` is ``qf(U)`` for uniform ``U``, as
    ``Parametric.random`` is, so the same ``random_state`` gives common
    random numbers across models (and the first clock's ages that
    ``model.random(size, rate)`` draws). The bounds (``cb``,
    ``quantile_cb``, ``rmst``) carry the uncertainty of :math:`k` and of
    the distribution together; a model built from given parameters has
    none, and they refuse, saying so.

    It is a view, not a copy: it holds the model and the rate, and every
    call goes to the model, so it has no ``to_dict`` of its own; save the
    model (``model.to_dict()``) and the rate.

    Attributes
    ----------
    model : EquivalenceModel
        The fitted equivalent-scale model.
    rate : float
        The usage rate, the second clock's units per unit of the first.

    Examples
    --------
    >>> from surpyval import WeibullEquivalence
    >>> model = WeibullEquivalence.from_params(
    ...     [2000, 2, 10], clocks=("hours", "starts"), unit="hours")
    >>> life = model.at_usage(0.1)  # 0.1 starts per hour
    >>> life
    LifeAtUsage(WeibullEquivalence, clocks=('hours', 'starts'), rate=0.1)
    >>> life.acceleration  # 1 + k rate: two equivalent hours per hour
    2.0
    >>> life.sf([500, 1000]).round(4)
    array([0.7788, 0.3679])
    >>> float(life.qf(0.1).round(1))  # its B10 life, in hours
    324.6
    >>> bool(life.sf(1000) == model.sf_at_usage(1000, 0.1))
    True
    """

    def __init__(self, model: Any, rate: Any) -> None:
        self.model = model
        self.rate = _one_rate(rate)

    @property
    def clocks(self) -> tuple:
        """The model's names of its two clocks, first then second."""
        return tuple(self.model.clocks)

    @property
    def unit(self) -> str | None:
        """The model's unit label, as typed (the unit of this life's
        ages), or ``None``."""
        return getattr(self.model, "unit", None)

    @property
    def acceleration(self) -> float:
        """``1 + k rate``: the combined clock's units per unit of the
        first, at this rate."""
        return 1.0 + self.model.k * self.rate

    def sf(self, x: npt.ArrayLike) -> Any:
        """The survival at ages ``x`` on the first clock:
        ``model.sf_at_usage(x, rate)``."""
        return self.model.sf_at_usage(x, self.rate)

    def ff(self, x: npt.ArrayLike) -> Any:
        """The failure probability at ages ``x`` on the first clock:
        ``model.ff_at_usage(x, rate)``."""
        return self.model.ff_at_usage(x, self.rate)

    def df(self, x: npt.ArrayLike) -> Any:
        """The density per unit of the first clock at ages ``x``:
        ``a model.df(a x)``, ``a`` the :attr:`acceleration`."""
        tb, rb, shape = _broadcast(x, self.rate)
        a = self.acceleration
        out = np.asarray(self.model._base.df(tb * a), dtype=float) * a
        return _reshape(out, shape, False)

    def hf(self, x: npt.ArrayLike) -> Any:
        """The hazard per unit of the first clock at ages ``x``:
        ``model.hf_at_usage(x, rate)``."""
        return self.model.hf_at_usage(x, self.rate)

    def Hf(self, x: npt.ArrayLike) -> Any:
        """The cumulative hazard at ages ``x`` on the first clock:
        ``model.Hf(a x)``, ``a`` the :attr:`acceleration`."""
        tb, rb, shape = _broadcast(x, self.rate)
        out = np.asarray(
            self.model._base.Hf(tb * self.acceleration), dtype=float
        )
        return _reshape(out, shape, False)

    def qf(self, p: npt.ArrayLike) -> Any:
        """The age on the first clock by which a fraction ``p`` have
        failed: ``model.qf_at_usage(p, rate)``."""
        return self.model.qf_at_usage(p, self.rate)

    def random(self, size: int, random_state: Any = None) -> np.ndarray:
        """
        Draw ``size`` lives on the first clock of units used at this
        rate.

        Each draw is ``qf(u)`` for one uniform ``u``, drawn as
        ``Parametric.random`` draws them, so the same ``random_state``
        gives common random numbers across models, and the same draws as
        the first clock of ``model.random(size, rate)``.

        Parameters
        ----------
        size : int
            The number of draws.
        random_state : None, int or numpy.random.Generator, optional
            ``None`` (the default) draws from numpy's global generator, so
            ``np.random.seed`` reproduces the draw; an int or a
            ``Generator`` gives a stream of its own.

        Returns
        -------
        numpy array
            ``size`` lives on the first clock.

        Examples
        --------
        >>> from surpyval import WeibullEquivalence
        >>> model = WeibullEquivalence.from_params([2000, 2, 10])
        >>> life = model.at_usage(0.1)
        >>> hours, _ = model.random(3, rate=0.1, random_state=7)
        >>> bool((life.random(3, random_state=7) == hours).all())
        True
        """
        from surpyval.univariate.parametric.parametric import uniform_draws

        u = np.asarray(uniform_draws(int(size), random_state), dtype=float)
        T = np.asarray(self.model.dist.qf(u, *self.model.dist_params))
        return np.asarray(T, dtype=float) / self.acceleration

    def mean(self, tau: float | None = None) -> float:
        """
        The mean life on the first clock, ``model.mean_at_usage(rate)``;
        with ``tau``, the restricted mean up to ``tau`` on the first clock
        (the area under :meth:`sf` from 0 to ``tau``), the point value of
        :meth:`rmst`.

        Examples
        --------
        >>> from surpyval import WeibullEquivalence
        >>> life = WeibullEquivalence.from_params([2000, 2, 10]).at_usage(0.1)
        >>> round(life.mean(), 1)
        886.2
        >>> round(life.mean(tau=500), 1)
        461.3
        """
        if tau is None:
            return float(self.model.mean_at_usage(self.rate))
        tau = _check_tau(tau)
        a = self.acceleration
        return float(self.model._base.mean(tau=tau * a)) / a

    def rmst(self, tau: float, alpha_ci: float = 0.05) -> dict:
        """
        The restricted mean life up to ``tau`` on the first clock, with
        its Wald interval: the delta method over every parameter (``k``
        included), on the logit of its fraction of ``tau`` so that it
        stays in ``(0, tau)``, as ``Parametric.rmst`` forms it.

        Returns
        -------
        dict
            ``{"rmst", "se", "lower", "upper", "tau"}``, as
            ``Parametric.rmst``.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullEquivalence
        >>> rng = np.random.default_rng(1)
        >>> rate = rng.choice([0.02, 0.1, 0.5], 300)
        >>> hours = 2000 * rng.weibull(2.0, 300) / (1 + 10 * rate)
        >>> model = WeibullEquivalence.fit(hours, rate * hours)
        >>> res = model.at_usage(0.1).rmst(500)
        >>> round(res["lower"], 1) < round(res["rmst"], 1) < res["upper"]
        True
        """
        from surpyval.univariate.parametric._likelihood_ratio import (
            central_jacobian,
        )

        check_alpha_ci(alpha_ci)
        tau = _check_tau(tau)
        try:
            cov = self.model.covariance()
        except ValueError as error:
            raise ValueError(
                "{} The point estimate alone is life.mean(tau={:g}).".format(
                    str(error), tau
                )
            ) from None
        dist, r = self.model.dist, self.rate

        def rmst_of(phi: npt.NDArray) -> Any:
            a = 1.0 + phi[-1] * r
            base = dist.from_params(list(phi[:-1]))
            return np.array([float(base.mean(tau=tau * a)) / a])

        phi = np.asarray(self.model.params, dtype=float)
        value = float(rmst_of(phi)[0])
        with np.errstate(all="ignore"):
            g = np.ravel(central_jacobian(rmst_of, phi))
            se = float(np.sqrt(max(float(g @ cov @ g), 0.0)))
            z = norm.ppf(1.0 - alpha_ci / 2.0)
            if tau == 0 or se == 0 or not 0 < value < tau:
                lower = upper = value
            else:
                psi = np.log(value) - np.log(tau - value)
                se_psi = se * (1.0 / value + 1.0 / (tau - value))
                lower = float(tau * expit(psi - z * se_psi))
                upper = float(tau * expit(psi + z * se_psi))
        return {
            "rmst": value,
            "se": se,
            "lower": lower,
            "upper": upper,
            "tau": tau,
        }

    def cb(self, x: npt.ArrayLike, **kwargs: Any) -> Any:
        """
        Wald confidence bounds on ``sf``, ``ff`` or ``Hf`` at ages ``x``
        on the first clock: ``model.cb_at_usage(x, rate, **kwargs)``
        (``on="sf"`` by default, and its ``alpha_ci`` and ``bound``).

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullEquivalence
        >>> rng = np.random.default_rng(1)
        >>> rate = rng.choice([0.02, 0.1, 0.5], 300)
        >>> hours = 2000 * rng.weibull(2.0, 300) / (1 + 10 * rate)
        >>> model = WeibullEquivalence.fit(hours, rate * hours)
        >>> model.at_usage(0.5).cb(500).round(3)
        array([0.061, 0.14 ])
        """
        return self.model.cb_at_usage(x, self.rate, **kwargs)

    def quantile_cb(self, p: npt.ArrayLike, **kwargs: Any) -> Any:
        """Wald confidence bounds on the life ``qf(p)``:
        ``model.quantile_cb_at_usage(p, rate, **kwargs)``."""
        return self.model.quantile_cb_at_usage(p, self.rate, **kwargs)

    def __repr__(self) -> str:
        return "LifeAtUsage({}Equivalence, clocks={!r}, rate={:g})".format(
            self.model.dist.name, self.clocks, self.rate
        )


def _check_tau(tau: Any) -> float:
    """A restricted mean's horizon: a finite number, 0 or more."""
    try:
        t = float(tau)
    except (TypeError, ValueError):
        t = np.nan
    if np.ndim(tau) != 0 or not np.isfinite(t) or t < 0:
        raise ValueError(
            "tau, the horizon of the restricted mean on the first clock, "
            f"must be one finite number, 0 or more; got {tau!r}."
        )
    return t


class EquivalenceModel(InformationCriteriaMixin, SerialisableMixin):
    r"""
    A fitted equivalent time scale (:func:`Equivalence`): the combined
    clock :math:`T = x_1 + k\,x_2` and a lifetime distribution on it.

    ``params`` are the distribution's parameters on :math:`T`, then
    :math:`k`, what one unit of the second clock is worth on the first
    (``parameter_names`` names them). ``sf``, ``ff``, ``df``, ``hf``,
    ``Hf``, ``qf``, ``mean`` and ``cb`` are the distribution's on the
    combined scale; :meth:`equivalent` gives a unit's :math:`T`; the
    ``*_at_usage`` methods give the life of a unit that uses the second
    clock at a constant ``rate`` per unit of the first (starts per hour),
    in the first clock's units, with bounds that include :math:`k`'s
    uncertainty.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import WeibullEquivalence
    >>> rng = np.random.default_rng(1)
    >>> rate = rng.choice([0.02, 0.1, 0.5], 300)
    >>> T = 2000 * rng.weibull(2.0, 300)
    >>> hours = T / (1 + 10 * rate)
    >>> model = WeibullEquivalence.fit(hours, rate * hours,
    ...                                    clocks=("hours", "starts"))
    >>> model.parameter_names
    ['alpha', 'beta', 'k']
    >>> model.param_cb("k").round(1)
    array([ 8. , 12.1])
    >>> float(model.qf_at_usage(0.1, rate=0.1).round(0))  # B10 in hours
    312.0
    """

    _repr_title = "Equivalence SurPyval Model"

    dist: Any
    params: np.ndarray
    clocks: tuple
    data: Any
    maximum: str
    _covariance: Any
    _null_neg_ll: Any
    _neg_ll: Any
    _data: Any

    # -- what it is ----------
    @property
    def parameter_names(self) -> list[str]:
        """The distribution's parameter names, then ``"k"``."""
        return [*self.dist.parameter_names, "k"]

    @property
    def covariance_names(self) -> list[str]:
        """The names of :meth:`covariance`'s rows: ``parameter_names``."""
        return self.parameter_names

    @property
    def k(self) -> float:  # type: ignore[override]
        """What one unit of the second clock is worth in units of the
        first (hours per start)."""
        return float(self.params[-1])

    @property
    def dist_params(self) -> np.ndarray:
        """The distribution's parameters on the combined scale."""
        return np.asarray(self.params[:-1], dtype=float)

    @property
    def _base(self) -> Any:
        """The distribution on the combined scale, as a univariate model
        (for its functions)."""
        cached = self.__dict__.get("_base_model")
        if cached is None or not np.array_equal(
            cached.params, self.dist_params
        ):
            cached = self.dist.from_params(list(self.dist_params))
            self.__dict__["_base_model"] = cached
        return cached

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state.pop("_base_model", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def _parameter_bounds(self) -> list[tuple]:
        """The support of each of ``params``: the distribution's, then
        ``k >= 0``."""
        return [*self.dist.bounds, (0, None)]

    def _ic_k(self) -> float:
        # The number of fitted parameters (``k``, the attribute, is the
        # clocks' exchange rate).
        return float(len(self.params))

    # -- the combined clock ----------
    def equivalent(self, x1: npt.ArrayLike, x2: npt.ArrayLike) -> Any:
        """
        The combined age ``x1 + k x2`` (equivalent hours) of units with
        those ages on the two clocks.

        Examples
        --------
        >>> from surpyval import WeibullEquivalence
        >>> model = WeibullEquivalence.from_params([2000, 2, 10])
        >>> model.equivalent([100, 200], [5, 0])
        array([150., 200.])
        """
        out = np.asarray(x1, dtype=float) + self.k * np.asarray(
            x2, dtype=float
        )
        return out[()] if out.ndim == 0 else out

    # -- on the combined scale ----------
    @keeps_query_shape
    def sf(self, x: npt.ArrayLike) -> Any:
        """
        The survival function at combined ages ``x``.

        Examples
        --------
        >>> from surpyval import WeibullEquivalence
        >>> model = WeibullEquivalence.from_params([2000, 2, 10])
        >>> float(model.sf(2000).round(4))
        0.3679
        """
        return self._base.sf(x)

    @keeps_query_shape
    def ff(self, x: npt.ArrayLike) -> Any:
        """
        The failure probability at combined ages ``x``.

        Examples
        --------
        >>> from surpyval import WeibullEquivalence
        >>> model = WeibullEquivalence.from_params([2000, 2, 10])
        >>> float(model.ff(2000).round(4))
        0.6321
        """
        return self._base.ff(x)

    @keeps_query_shape
    def df(self, x: npt.ArrayLike) -> Any:
        """
        The density at combined ages ``x`` (per unit of the combined
        clock).

        Examples
        --------
        >>> from surpyval import WeibullEquivalence
        >>> model = WeibullEquivalence.from_params([2000, 2, 10])
        >>> float(model.df(1000).round(7))
        0.0003894
        """
        return self._base.df(x)

    @keeps_query_shape
    def hf(self, x: npt.ArrayLike) -> Any:
        """
        The hazard rate at combined ages ``x`` (per unit of the combined
        clock).

        Examples
        --------
        >>> from surpyval import WeibullEquivalence
        >>> model = WeibullEquivalence.from_params([2000, 2, 10])
        >>> float(model.hf(1000))
        0.0005
        """
        return self._base.hf(x)

    @keeps_query_shape
    def Hf(self, x: npt.ArrayLike) -> Any:
        """
        The cumulative hazard at combined ages ``x``.

        Examples
        --------
        >>> from surpyval import WeibullEquivalence
        >>> model = WeibullEquivalence.from_params([2000, 2, 10])
        >>> float(model.Hf(2000))
        1.0
        """
        return self._base.Hf(x)

    @keeps_query_shape
    def qf(self, p: npt.ArrayLike) -> Any:
        """
        The combined age by which a fraction ``p`` has failed.

        Examples
        --------
        >>> from surpyval import WeibullEquivalence
        >>> model = WeibullEquivalence.from_params([2000, 2, 10])
        >>> float(model.qf(0.1).round(1))
        649.2
        """
        return self._base.qf(p)

    def mean(self) -> float:
        """
        The mean life on the combined clock.

        Examples
        --------
        >>> from surpyval import WeibullEquivalence
        >>> model = WeibullEquivalence.from_params([2000, 2, 10])
        >>> round(model.mean(), 1)
        1772.5
        """
        return float(self._base.mean())

    # -- along a usage path ----------
    def _T(self, t: Any, rate: Any, k: Any = None) -> Any:
        k = self.k if k is None else k
        return t * (1.0 + k * rate)

    def sf_at_usage(self, x: npt.ArrayLike, rate: npt.ArrayLike) -> Any:
        """
        The survival at age ``x`` on the first clock (hours) of a unit
        that uses the second clock at ``rate`` per unit of the first
        (starts per hour): ``sf(x (1 + k rate))``. ``x`` and ``rate``
        broadcast together.

        Examples
        --------
        >>> from surpyval import WeibullEquivalence
        >>> model = WeibullEquivalence.from_params([2000, 2, 10])
        >>> model.sf_at_usage(1000, rate=[0.0, 0.1]).round(4)
        array([0.7788, 0.3679])
        """
        tb, rb, shape = _broadcast(x, rate)
        out = np.asarray(self._base.sf(self._T(tb, rb)), dtype=float)
        return _reshape(out, shape, False)

    def ff_at_usage(self, x: npt.ArrayLike, rate: npt.ArrayLike) -> Any:
        """
        The failure probability at age ``x`` on the first clock of a unit
        used at ``rate`` (see :meth:`sf_at_usage`).

        Examples
        --------
        >>> from surpyval import WeibullEquivalence
        >>> model = WeibullEquivalence.from_params([2000, 2, 10])
        >>> float(model.ff_at_usage(1000, rate=0.1).round(4))
        0.6321
        """
        tb, rb, shape = _broadcast(x, rate)
        out = np.asarray(self._base.ff(self._T(tb, rb)), dtype=float)
        return _reshape(out, shape, False)

    def hf_at_usage(self, x: npt.ArrayLike, rate: npt.ArrayLike) -> Any:
        """
        The hazard per unit of the first clock (per hour) at age ``x`` of
        a unit used at ``rate``: ``hf(T) (1 + k rate)``.

        Examples
        --------
        >>> from surpyval import WeibullEquivalence
        >>> model = WeibullEquivalence.from_params([2000, 2, 10])
        >>> float(model.hf_at_usage(500, rate=0.1))
        0.001
        """
        tb, rb, shape = _broadcast(x, rate)
        T = self._T(tb, rb)
        out = np.asarray(self._base.hf(T), dtype=float) * (1.0 + self.k * rb)
        return _reshape(out, shape, False)

    def qf_at_usage(self, p: npt.ArrayLike, rate: npt.ArrayLike) -> Any:
        """
        The age on the first clock (hours) by which a fraction ``p`` of
        units used at ``rate`` have failed: the B-life along that usage,
        ``qf(p) / (1 + k rate)``. The second clock's age then is ``rate``
        times it.

        Examples
        --------
        >>> from surpyval import WeibullEquivalence
        >>> model = WeibullEquivalence.from_params([2000, 2, 10])
        >>> model.qf_at_usage(0.1, rate=[0.0, 0.1]).round(1)
        array([649.2, 324.6])
        """
        pb, rb, shape = _broadcast(p, rate)
        q = np.asarray(self._base.qf(pb), dtype=float)
        return _reshape(q / (1.0 + self.k * rb), shape, False)

    def mean_at_usage(self, rate: npt.ArrayLike) -> Any:
        """
        The mean life on the first clock of units used at ``rate``.

        Examples
        --------
        >>> from surpyval import WeibullEquivalence
        >>> model = WeibullEquivalence.from_params([2000, 2, 10])
        >>> float(model.mean_at_usage(0.1).round(1))
        886.2
        """
        r = np.asarray(rate, dtype=float)
        out = self.mean() / (1.0 + self.k * r)
        return out[()] if np.ndim(out) == 0 else out

    def at_usage(self, rate: float) -> LifeAtUsage:
        """
        The life of a unit used at one constant ``rate``, as a plain life
        model on the first clock: an object whose ``sf``, ``ff``, ``df``,
        ``hf``, ``Hf``, ``qf``, ``random``, ``mean``, ``rmst``, ``cb`` and
        ``quantile_cb`` take ages (or probabilities) on the first clock
        only, each this model's ``*_at_usage`` at ``rate``. Hand it to
        code that expects a univariate model ("the life of a pump making
        0.1 starts per hour").

        Parameters
        ----------
        rate : float
            The second clock's units per unit of the first, in the
            model's units (starts per hour for a model fitted to hours
            and starts): one finite number, 0 or more. Several rates are
            refused: call it once per rate.

        Returns
        -------
        LifeAtUsage
            A view on this model at ``rate``.

        Examples
        --------
        >>> from surpyval import WeibullEquivalence
        >>> model = WeibullEquivalence.from_params([2000, 2, 10])
        >>> round(model.at_usage(0.1).mean(), 1)
        886.2
        """
        return LifeAtUsage(self, rate)

    # -- uncertainty ----------
    def covariance(self) -> np.ndarray:
        """
        The covariance of ``params`` (the distribution's parameters, then
        ``k``): the inverse observed information at the maximum. Raises
        a ``ValueError`` where there is none (a model from
        ``from_params``, or an information matrix that is not positive
        definite).

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullEquivalence
        >>> rng = np.random.default_rng(1)
        >>> rate = rng.choice([0.02, 0.1, 0.5], 300)
        >>> hours = 2000 * rng.weibull(2.0, 300) / (1 + 10 * rate)
        >>> model = WeibullEquivalence.fit(hours, rate * hours)
        >>> model.covariance().shape
        (3, 3)
        """
        if self._covariance is None:
            raise ValueError(
                "This model has no covariance: it was built from given "
                "parameters (from_params), or the information matrix at "
                "the fit was not positive definite. Fit it to data for "
                "bounds."
            )
        return np.asarray(self._covariance)

    def standard_errors(self) -> np.ndarray:
        """
        The standard errors of ``params``, the square roots of the
        diagonal of :meth:`covariance`.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullEquivalence
        >>> rng = np.random.default_rng(1)
        >>> rate = rng.choice([0.02, 0.1, 0.5], 300)
        >>> hours = 2000 * rng.weibull(2.0, 300) / (1 + 10 * rate)
        >>> model = WeibullEquivalence.fit(hours, rate * hours)
        >>> model.standard_errors().shape
        (3,)
        """
        return standard_errors_of(self.covariance())

    def _index(self, name: str) -> int:
        names = self.parameter_names
        if name not in names:
            raise ValueError(
                f"Unknown parameter {name!r}; the model's parameters are "
                f"{names}."
            )
        return names.index(name)

    def param_cb(
        self,
        name: str,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str | None = None,
    ) -> Any:
        """
        The confidence bound on a parameter, ``k`` or one of the
        distribution's.

        ``method="wald"`` (the default) is a symmetric bound from the
        standard error on the log scale of a positive parameter (so it
        stays positive). Where ``k`` is fitted at 0 (the second clock
        does not matter) it has no Wald bound: ``method="lr"`` gives the
        profile-likelihood bound on ``k``, which starts at 0 there and
        holds its level near the boundary.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullEquivalence
        >>> rng = np.random.default_rng(1)
        >>> rate = rng.choice([0.02, 0.1, 0.5], 300)
        >>> hours = 2000 * rng.weibull(2.0, 300) / (1 + 10 * rate)
        >>> model = WeibullEquivalence.fit(hours, rate * hours)
        >>> model.param_cb("k").round(2)
        array([ 8. , 12.1])
        >>> model.param_cb("k", method="lr").round(2)
        array([ 7.99, 12.12])
        """
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        method = "wald" if method is None else str(method).lower()
        i = self._index(name)
        if method == "lr":
            if name != "k":
                raise ValueError(
                    "method='lr' is available for k only: use the "
                    f"default Wald bound for {name!r}."
                )
            return self._k_lr_bound(alpha_ci, bound)
        if method != "wald":
            raise option_error("method", method, ("wald", "lr"))
        p_hat = float(self.params[i])
        var = float(self.covariance()[i, i])
        lower: float | None
        upper: float | None
        advice: str | None
        if name == "k":
            lower, upper = 0.0, None
            advice = "use method='lr', the profile-likelihood bound"
        else:
            lo, hi = self.dist.bounds[i]
            lower = None if lo is None else float(lo)
            upper = None if hi is None else float(hi)
            advice = None
        out = wald_bound_on_support(
            p_hat, var, lower, upper, alpha_ci, bound, name, advice
        )
        out = np.asarray(out, dtype=float)
        return out if bound == "two-sided" else out[0]

    def _profile_ll(self, k: float) -> float:
        d = self.__dict__.get("_data")
        if d is None:
            raise ValueError(
                "Likelihood-ratio bounds on k need the data the model was "
                "fitted to, which a model restored from a dictionary or "
                "built from parameters does not have: refit it, or use the "
                "Wald bound (method='wald')."
            )
        return _profile(self.dist, d, float(k))[0]

    def _k_lr_bound(self, alpha_ci: float, bound: str) -> Any:
        side = alpha_ci / 2.0 if bound == "two-sided" else alpha_ci
        crit = norm.ppf(1.0 - side) ** 2
        ll_hat = -self.neg_ll()
        k_hat = self.k
        d = self.__dict__.get("_data")
        k_ref = d.k_ref() if d is not None else max(k_hat, 1.0)
        target = ll_hat - crit / 2.0

        def gap(k: float) -> float:
            return self._profile_ll(k) - target

        with np.errstate(all="ignore"):
            # lower end
            if k_hat <= 0:
                lower = 0.0
            else:
                at_zero = (
                    self._profile_ll(0.0)
                    if self._null_neg_ll is not None
                    else -np.inf
                )
                if at_zero >= target:
                    lower = 0.0
                else:
                    lo = 0.0 if np.isfinite(at_zero) else k_hat * 1e-12
                    if not np.isfinite(at_zero):
                        while gap(lo) > 0 and lo > 0:
                            lo *= 1e-3
                    lower = brentq(gap, lo, k_hat, xtol=1e-10 * k_ref)
            # upper end
            hi = max(k_hat, 1e-3 * k_ref) * 2.0
            upper = np.inf
            for _ in range(60):
                if gap(hi) < 0:
                    upper = brentq(
                        gap,
                        hi / 2.0 if hi > 2 * k_hat else k_hat,
                        hi,
                        xtol=1e-10 * k_ref,
                    )
                    break
                hi *= 2.0
                if hi > 1e12 * k_ref:
                    break
        if bound == "two-sided":
            return np.array([lower, upper])
        return lower if bound == "lower" else upper

    def k_lr_test(self) -> dict:
        """
        The likelihood-ratio test that the second clock does not matter
        (``k = 0``): the statistic, twice the gain in log-likelihood from
        fitting ``k``, and its p-value. ``k = 0`` is on the edge of the
        parameter space, so the statistic is referred to an even mixture
        of 0 and a chi-square with 1 degree of freedom (Self and Liang,
        1987): the p-value is half the chi-square's. Where a unit failed
        with no age on the first clock, ``k = 0`` is impossible and the
        p-value is 0.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullEquivalence
        >>> rng = np.random.default_rng(1)
        >>> rate = rng.choice([0.02, 0.1, 0.5], 300)
        >>> hours = 2000 * rng.weibull(2.0, 300) / (1 + 10 * rate)
        >>> test = WeibullEquivalence.fit(hours, rate * hours).k_lr_test()
        >>> test["p_value"] < 1e-10
        True
        """
        if self._neg_ll is None:
            raise ValueError(
                "The test needs a fitted model; this one was built from "
                "given parameters."
            )
        if self._null_neg_ll is None:
            stat = np.inf
        else:
            stat = max(2.0 * (self._null_neg_ll - self._neg_ll), 0.0)
        p = 0.5 * float(chi2.sf(stat, 1)) if stat > 0 else 1.0
        return {
            "statistic": float(stat),
            "p_value": p,
            "null": "k = 0",
            "reference": "0.5 chi2(0) + 0.5 chi2(1)",
        }

    def _check_cb(self, method: Any, on: str) -> None:
        if method not in (None, "wald", "Wald"):
            raise ValueError(
                f"method={method!r}: the bounds of an equivalent-scale "
                "model are Wald (delta-method) bounds, which carry the "
                "uncertainty of k and the distribution together. A "
                "likelihood-ratio or bootstrap bound of the distribution "
                "alone would hold k fixed and be too narrow."
            )
        if on not in _CB_ON:
            raise option_error("on", on, _CB_ON)

    def _sf_band(
        self,
        T_of: Any,
        n_points: int,
        alpha_ci: float,
        bound: str,
        on: str,
    ) -> np.ndarray:
        """The Wald band on ``on`` at the combined ages ``T_of(phi)``."""
        cov = self.covariance()
        phi = np.asarray(self.params, dtype=float)
        dist = self.dist

        def S(p: Any) -> Any:
            return dist.sf(T_of(p), *[p[j] for j in range(len(p) - 1)])

        with np.errstate(all="ignore"):
            J = np.asarray(jacobian(S)(phi), dtype=float).reshape(
                n_points, phi.size
            )
            var = np.einsum("ij,jk,ik->i", J, cov, J)
            se = np.sqrt(np.maximum(var, 0.0))
            T = np.asarray(T_of(phi), dtype=float)
            sf_hat = np.asarray(dist.sf(T, *phi[:-1]), dtype=float)
            ff_hat = np.asarray(dist.ff(T, *phi[:-1]), dtype=float)
            name = {"R": "sf", "F": "ff"}.get(on, on)
            out = sf_link_bound(
                sf_hat, se, alpha_ci, bound, cb_link(dist), ff_hat, name
            )
        nan = ~np.isfinite(T)
        if nan.any():
            out = np.array(out, dtype=float)
            out[nan] = np.nan
        return out

    def cb(
        self,
        x: npt.ArrayLike,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str | None = None,
    ) -> Any:
        """
        Wald confidence bounds on ``sf``, ``ff`` or ``Hf`` at combined
        ages ``x``. A two-sided bound puts ``[lower, upper]`` on the last
        axis.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullEquivalence
        >>> rng = np.random.default_rng(1)
        >>> rate = rng.choice([0.02, 0.1, 0.5], 300)
        >>> hours = 2000 * rng.weibull(2.0, 300) / (1 + 10 * rate)
        >>> model = WeibullEquivalence.fit(hours, rate * hours)
        >>> model.cb([1000, 2000]).round(3)
        array([[0.696, 0.814],
               [0.254, 0.428]])
        """
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        self._check_cb(method, on)
        xb = np.asarray(x, dtype=float)
        flat = xb.ravel()
        if flat.size == 0:
            return np.empty(xb.shape + ((2,) if bound == "two-sided" else ()))
        out = self._sf_band(lambda p: flat, flat.size, alpha_ci, bound, on)
        return _reshape(out, xb.shape, bound == "two-sided")

    def cb_at_usage(
        self,
        x: npt.ArrayLike,
        rate: npt.ArrayLike,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str | None = None,
    ) -> Any:
        """
        Wald bounds on ``sf``, ``ff`` or ``Hf`` at age ``x`` on the first
        clock of a unit used at ``rate`` (see :meth:`sf_at_usage`). They
        include the uncertainty in ``k`` (the delta method over every
        parameter).

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullEquivalence
        >>> rng = np.random.default_rng(1)
        >>> rate = rng.choice([0.02, 0.1, 0.5], 300)
        >>> hours = 2000 * rng.weibull(2.0, 300) / (1 + 10 * rate)
        >>> model = WeibullEquivalence.fit(hours, rate * hours)
        >>> model.cb_at_usage(500, rate=0.5).round(3)
        array([0.061, 0.14 ])
        """
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        self._check_cb(method, on)
        tb, rb, shape = _broadcast(x, rate)
        if tb.size == 0:
            return np.empty(shape + ((2,) if bound == "two-sided" else ()))
        out = self._sf_band(
            lambda p: tb * (1.0 + p[-1] * rb), tb.size, alpha_ci, bound, on
        )
        return _reshape(out, shape, bound == "two-sided")

    def quantile_cb_at_usage(
        self,
        p: npt.ArrayLike,
        rate: npt.ArrayLike,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> Any:
        """
        Wald bounds on the B-life along a usage path,
        :meth:`qf_at_usage`: the age on the first clock by which a
        fraction ``p`` of units used at ``rate`` have failed. The bound
        is formed on the log of the life, by the delta method over every
        parameter (``k`` included), so it stays positive.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullEquivalence
        >>> rng = np.random.default_rng(1)
        >>> rate = rng.choice([0.02, 0.1, 0.5], 300)
        >>> hours = 2000 * rng.weibull(2.0, 300) / (1 + 10 * rate)
        >>> model = WeibullEquivalence.fit(hours, rate * hours)
        >>> model.quantile_cb_at_usage(0.1, rate=0.1).round(0)
        array([273., 356.])
        """
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        cov = self.covariance()
        pb, rb, shape = _broadcast(p, rate)
        two = bound == "two-sided"
        if pb.size == 0:
            return np.empty(shape + ((2,) if two else ()))
        phi = np.asarray(self.params, dtype=float)
        theta, k = phi[:-1], phi[-1]
        dist = self.dist
        with np.errstate(all="ignore"):
            q = np.asarray(self._base.qf(pb), dtype=float)
            life = q / (1.0 + k * rb)
            # d log q / d theta = -(dF/dtheta) / (f(q) q), at q fixed
            dF = np.asarray(
                jacobian(
                    lambda th: dist.ff(q, *[th[j] for j in range(len(th))])
                )(theta),
                dtype=float,
            ).reshape(pb.size, theta.size)
            f = np.asarray(dist.df(q, *theta), dtype=float)
            g = np.empty((pb.size, phi.size))
            g[:, :-1] = -dF / (f * q)[:, None]
            g[:, -1] = -rb / (1.0 + k * rb)
            var = np.einsum("ij,jk,ik->i", g, cov, g)
            se = np.sqrt(np.maximum(var, 0.0))
            side = alpha_ci / 2.0 if two else alpha_ci
            z = norm.ppf(1.0 - side)
            lo = life * np.exp(-z * se)
            hi = life * np.exp(z * se)
        if two:
            out = np.stack([lo, hi], axis=-1)
        else:
            out = lo if bound == "lower" else hi
        return _reshape(out, shape, two)

    # -- drawing ----------
    def random(
        self,
        size: int,
        rate: npt.ArrayLike,
        random_state: Any = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Draw ``size`` lives of units used at ``rate`` (one rate for every
        unit, or one per unit), as their ages at failure on the two
        clocks: ``(x1, x2)``, with ``x2 = rate * x1``. ``random_state``
        follows the package's seed rule: ``None`` draws from numpy's
        global stream, anything else from its own.

        Examples
        --------
        >>> from surpyval import WeibullEquivalence
        >>> model = WeibullEquivalence.from_params([2000, 2, 10])
        >>> hours, starts = model.random(3, rate=0.1, random_state=1)
        >>> bool(((hours + 10 * starts) > 0).all())
        True
        """
        from surpyval.univariate.parametric.parametric import uniform_draws

        r = np.broadcast_to(np.asarray(rate, dtype=float), (int(size),))
        if (r < 0).any() or not np.isfinite(r).all():
            raise ValueError(
                "rate must be finite and 0 or more (second clock per unit "
                "of the first)."
            )
        U = uniform_draws(int(size), random_state)
        T = np.asarray(self.dist.qf(U, *self.dist_params), dtype=float)
        x1 = T / (1.0 + self.k * r)
        return x1, r * x1

    # -- information criteria ----------
    def _ic_sample_size_from_data(self) -> float:
        if self.data is None:
            raise ValueError("Must have been fit with data")
        return ic_sample_size(self.data["c"], self.data["n"])

    # -- saving ----------
    def to_dict(self) -> dict:
        """
        The model as a plain, JSON-ready ``dict``: the distribution, the
        clocks' names, the parameters, their covariance and the fit's
        log-likelihoods (not the data). ``surpyval.from_dict`` rebuilds
        it.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval
        >>> rng = np.random.default_rng(1)
        >>> rate = rng.choice([0.02, 0.1, 0.5], 300)
        >>> hours = 2000 * rng.weibull(2.0, 300) / (1 + 10 * rate)
        >>> model = surpyval.WeibullEquivalence.fit(hours, rate * hours)
        >>> restored = surpyval.from_dict(model.to_dict())
        >>> bool(np.allclose(restored.cb([1000]), model.cb([1000])))
        True
        """
        out: dict[str, Any] = {
            "model": "EquivalenceModel",
            "distribution": self.dist.name,
            "clocks": list(self.clocks),
            "params": np.asarray(self.params, dtype=float).tolist(),
            "parameter_names": self.parameter_names,
            "_neg_ll": None if self._neg_ll is None else float(self._neg_ll),
            "_null_neg_ll": (
                None if self._null_neg_ll is None else float(self._null_neg_ll)
            ),
            "ic_n": self._ic_sample_size_or_none(),
            **maximum_entry(self.maximum),
        }
        if self._covariance is not None:
            out["covariance"] = np.asarray(self._covariance).tolist()
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> EquivalenceModel:
        """
        Rebuild a model from :meth:`to_dict`'s dictionary.

        Examples
        --------
        >>> from surpyval import WeibullEquivalence
        >>> from surpyval.univariate.equivalence import (
        ...     EquivalenceModel)
        >>> model = WeibullEquivalence.from_params([2000, 2, 10])
        >>> EquivalenceModel.from_dict(model.to_dict()).k
        10.0
        """
        import surpyval

        require_model_tag(
            model_dict, "EquivalenceModel", "an equivalent-scale model"
        )
        name = str(model_dict["distribution"])
        if canonical_name(name) not in DISTRIBUTIONS:
            raise ValueError(
                f"Unknown distribution {name!r}; an equivalent-scale model "
                f"takes one of {', '.join(DISTRIBUTIONS)}."
            )
        out = cls()
        out.dist = getattr(surpyval, canonical_name(name))
        out.params = np.asarray(model_dict["params"], dtype=float)
        out.clocks = tuple(model_dict.get("clocks", ("x1", "x2")))
        cov = model_dict.get("covariance")
        out._covariance = None if cov is None else np.asarray(cov, float)
        neg = model_dict.get("_neg_ll")
        out._neg_ll = None if neg is None else float(neg)
        null = model_dict.get("_null_neg_ll")
        out._null_neg_ll = None if null is None else float(null)
        out._ic_n = cls._restored_ic_n(model_dict)
        out.maximum = restored_maximum(model_dict)
        out.data = None
        return out

    # -- reading it ----------
    def reading(self, alpha_ci: float = 0.05) -> str:
        """
        ``k`` in plain words, with its confidence interval at
        ``alpha_ci``.

        Examples
        --------
        >>> from surpyval import WeibullEquivalence
        >>> model = WeibullEquivalence.from_params(
        ...     [2000, 2, 14.2], clocks=("hours", "starts"))
        >>> model.reading()
        "one unit of 'starts' is worth 14.2 units of 'hours'"
        """
        a, b = self.clocks
        text = f"one unit of {b!r} is worth {self.k:.4g} units " f"of {a!r}"
        if self._covariance is None and self.__dict__.get("_data") is None:
            return text
        level = f"{100 * (1 - alpha_ci):.4g}%"
        with ignoring_warnings():
            try:
                method = "wald" if self.k > 0 else "lr"
                if method == "lr" and self.__dict__.get("_data") is None:
                    return text
                lo, hi = self.param_cb("k", alpha_ci, method=method)
            except ValueError:
                return text
        if not (np.isfinite(lo) and np.isfinite(hi) or hi == np.inf):
            return text
        return f"{text} ({level} CI {lo:.4g} to {hi:.4g})"

    def __repr__(self) -> str:
        lines = [
            self._repr_title,
            "=" * len(self._repr_title),
            f"Distribution        : {self.dist.name} on {self.clocks[0]} "
            f"+ k {self.clocks[1]}",
            f"Fitted by           : "
            f"{'MLE' if self._neg_ll is not None else 'given parameters'}",
        ]
        if self.maximum in ("unverified", "no finite maximum"):
            lines.append(f"Maximum             : {self.maximum}")
        lines.append(f"k                   : {self.reading()}")
        if self._neg_ll is not None:
            test = self.k_lr_test()
            lines.append(
                f"Test of k = 0       : LR {test['statistic']:.4g}, "
                f"p = {test['p_value']:.3g}"
            )
        lines.append("Parameters          :")
        for name, value in zip(self.parameter_names, self.params):
            lines.append(f"{name:>10}: {value:.6g}")
        return "\n".join(lines)


def Equivalence(dist: Any) -> EquivalenceFitter:
    r"""
    An equivalent-time-scale fitter for a lifetime distribution: the
    combined clock :math:`T = x_1 + k\,x_2` (equivalent hours: hours plus
    :math:`k` hours per start), :math:`k \ge 0` estimated by maximum
    likelihood together with the distribution on :math:`T`.

    ``dist`` is one of ``Weibull``, ``LogNormal``, ``LogLogistic``,
    ``Gamma`` and ``Exponential``; ``WeibullEquivalence`` and the
    others are prebuilt.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import Equivalence, LogNormal
    >>> rng = np.random.default_rng(2)
    >>> rate = rng.uniform(0.0, 0.4, 400)
    >>> T = np.exp(rng.normal(7, 0.5, 400))
    >>> hours = T / (1 + 5 * rate)
    >>> model = Equivalence(LogNormal).fit(hours, rate * hours)
    >>> model.parameter_names
    ['mu', 'sigma', 'k']
    """
    return EquivalenceFitter(dist)


def _prebuilt(name: str) -> EquivalenceFitter:
    import surpyval

    return EquivalenceFitter(getattr(surpyval, name))


WeibullEquivalence = _prebuilt("Weibull")
LogNormalEquivalence = _prebuilt("LogNormal")
LogLogisticEquivalence = _prebuilt("LogLogistic")
GammaEquivalence = _prebuilt("Gamma")
ExponentialEquivalence = _prebuilt("Exponential")
