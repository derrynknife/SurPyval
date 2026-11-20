r"""Excess hazard regression: covariate effects on the hazard beyond the
population's own.

The hazard of unit :math:`i` is the population hazard of its age,
calendar year and stratum, read from a rate table, plus an excess hazard
that the covariates act on proportionally (Estève et al., 1990):

.. math::
    \lambda_i(t) = \lambda_{P,i}(t) + \lambda_0(t)\, e^{\beta' Z_i} .

The model is additive on the hazard scale -- the background and the
excess add -- and multiplicative within the excess: :math:`e^{\beta_j}`
is the *excess hazard ratio* of a unit of covariate :math:`j`. The
baseline excess hazard :math:`\lambda_0` is constant within intervals of
follow-up time (one interval by default, as R's relsurv; ``breaks`` gives
more). Everything is estimated by maximum likelihood,

.. math::
    \ell = \sum_i n_i \Big[ \delta_i \log\big(\lambda_{P,i}(x_i)
    + \lambda_0(x_i) e^{\beta' Z_i}\big) - e^{\beta' Z_i} \Lambda_0(x_i)
    - \Lambda_{P,i}(x_i) \Big],

which needs no cause of death: an event of any cause contributes the
total hazard at its time. The last term does not depend on the
parameters; it is kept, so that the log-likelihood is that of the data.

The model's ``sf``, ``ff``, ``hf``, ``Hf``, ``df`` and ``qf`` are those
of the excess hazard alone, the *net* survival :math:`S_E(t \mid Z) =
\exp(-\Lambda_0(t) e^{\beta' Z})` of a unit with covariates :math:`Z`;
:meth:`ExcessHazard.observed_sf` multiplies in a unit's expected
survival from the rate table for its all-cause survival.

References
----------
Estève, J., Benhamou, E., Croasdale, M. and Raymond, L. (1990).
Relative survival and the estimation of net survival: elements for
further discussion. *Statistics in Medicine* 9, 529-538.

Pohar, M. and Stare, J. (2006). Relative survival analysis in R.
*Computer Methods and Programs in Biomedicine* 81, 272-278.
"""

from __future__ import annotations

from typing import Any

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
from autograd import grad, hessian
from scipy.optimize import minimize
from scipy.stats import norm

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
from surpyval.univariate.parametric.fitters import is_local_minimum
from surpyval.univariate.parametric.fitters.runaway import (
    runaway_coefficients,
)
from surpyval.univariate.regression._aliasing import (
    aliased_columns,
    constant_columns,
    covariate_columns,
    dataframe_covariates,
    fit_columns,
    warn_aliased,
)
from surpyval.utils.covariates import coefficient_floor, coefficient_names
from surpyval.utils.no_maximum import (
    maximum_entry,
    restored_maximum,
    warn_no_maximum,
    warn_unverified,
)
from surpyval.utils.rng import as_generator
from surpyval.utils.shapes import (
    check_paired_rows,
    covariate_rows,
    keeps_query_shape,
)
from surpyval.utils.surpyval_data import refuses_surpyval_data_with
from surpyval.utils.units import takes_unit
from surpyval.utils.validation import (
    BOUNDS,
    check_alpha_ci,
    check_option,
    warn_outside_unit_interval,
)
from surpyval.utils.warnings import ignoring_warnings, warn_no_covariance

from ._data import REFUSAL, RelativeData, check_data
from .rate_table import RateTable

_refuses_data = refuses_surpyval_data_with(REFUSAL)

#: The functions cb bounds.
_CB_ON = ("sf", "R", "ff", "F", "Hf")

#: Below this many expected excess events an interval's excess hazard is
#: taken to have run to 0 (the likelihood rises as it falls further).
_NO_EXCESS = 1e-6

#: A coefficient beyond this many standard deviations of its covariate
#: per unit of log excess hazard has run off (an excess hazard ratio of
#: e^30 between units a standard deviation apart).
_RUNAWAY = 30.0


# ---------------------------------------------------------------------------
# The likelihood
# ---------------------------------------------------------------------------
def _cuts(breaks: Any, data: RelativeData) -> npt.NDArray:
    """The interior cut points of the baseline's intervals."""
    events = np.repeat(data.x[data.c == 0], data.n[data.c == 0].astype(int))
    if breaks is None:
        return np.zeros(0)
    if isinstance(breaks, (int, np.integer)) and not isinstance(
        breaks, (bool, np.bool_)
    ):
        if breaks < 1:
            raise ValueError(
                f"'breaks' as a number is how many intervals the baseline "
                f"excess hazard is constant on, at least 1; got {breaks}."
            )
        q = np.arange(1, int(breaks)) / int(breaks)
        cuts = np.unique(np.quantile(events, q)) if q.size else np.zeros(0)
        return cuts[(cuts > 0) & (cuts < events.max())]
    try:
        cuts = np.asarray(breaks, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(
            "'breaks' must be the number of intervals, or the times at "
            f"which the baseline excess hazard may change; got {breaks!r}."
        ) from None
    cuts = cuts.reshape(-1)
    if (
        not np.all(np.isfinite(cuts))
        or np.any(cuts <= 0)
        or np.any(np.diff(cuts) <= 0)
    ):
        raise ValueError(
            "'breaks' as times must be finite, above 0 and strictly "
            f"increasing (the cut points between intervals); got {cuts}."
        )
    return cuts


def _exposure(x: npt.NDArray, cuts: npt.NDArray) -> npt.NDArray:
    """The time each unit spends in each interval up to its time."""
    edges = np.concatenate(([0.0], cuts, [np.inf]))
    t = np.clip(x[:, None], edges[None, :-1], edges[None, 1:])
    return np.maximum(t - edges[None, :-1], 0.0)


def _interval(x: npt.NDArray, cuts: npt.NDArray) -> npt.NDArray:
    """The interval each time is in, each closed on the right."""
    return np.searchsorted(cuts, x, side="left")


def _check_method(method: Any) -> None:
    """Refuse a bound the model does not have, saying what to use."""
    if method in ("lr", "bootstrap"):
        raise ValueError(
            f"method={method!r} is not built for the excess hazard model "
            "yet: its bounds are Wald bounds, method='wald'. For bootstrap "
            "bounds, refit ExcessHazard to the rows resampled with "
            "replacement and take the percentiles of the refits' values."
        )
    check_option("method", method, ("wald",))


def _describe(cuts: npt.NDArray, k: int) -> str:
    edges = np.concatenate(([0.0], cuts, [np.inf]))
    return f"({edges[k]:g}, {edges[k + 1]:g}]"


class _Problem:
    """The data of the likelihood, with the covariates of the columns
    kept centred and scaled (the search's own units)."""

    def __init__(
        self,
        table: RateTable,
        data: RelativeData,
        cuts: npt.NDArray,
        kept: npt.NDArray,
    ) -> None:
        self.n = data.n
        self.event = (data.c == 0).astype(float)
        self.D = _exposure(data.x, cuts)
        self.k = _interval(data.x, cuts)
        self.K = cuts.size + 1
        Z = np.zeros((data.size, 0)) if data.Z is None else data.Z
        Zk = Z[:, kept]
        w = self.n / self.n.sum()
        self.mean = w @ Zk
        self.sd = np.sqrt(w @ (Zk - self.mean) ** 2)
        self.Zs = (Zk - self.mean) / np.where(self.sd > 0, self.sd, 1.0)
        self.lam_P = np.where(
            self.event > 0,
            table._rate(data.s, data.age + data.x, data.year + data.x),
            0.0,
        )
        self.H_P = table._H(data.x, data.age, data.year, data.s)
        self.n_obs = float(ic_sample_size(data.c, data.n))
        self.events = np.bincount(
            self.k, weights=self.n * self.event, minlength=self.K
        )
        edges = np.concatenate(([0.0], cuts, [np.inf]))
        upper = np.minimum(data.x[:, None], edges[None, 1:])
        lower = np.minimum(data.x[:, None], edges[None, :-1])
        rows = np.repeat(np.arange(data.size), self.K)
        H_up = table._H(
            upper.ravel(), data.age[rows], data.year[rows], data.s[rows]
        ).reshape(upper.shape)
        H_low = table._H(
            lower.ravel(), data.age[rows], data.year[rows], data.s[rows]
        ).reshape(lower.shape)
        self.expected = self.n @ (H_up - H_low)
        self.at_risk = self.n @ self.D

    def nll(self, v: Any) -> Any:
        theta = v[: self.K]
        eta = anp.dot(self.Zs, v[self.K :])
        excess = anp.exp(theta[self.k] + eta)
        log_h = anp.log(self.lam_P + excess)
        logl = (
            anp.sum(self.n * self.event * log_h)
            - anp.sum(self.n * anp.exp(eta) * anp.dot(self.D, anp.exp(theta)))
            - anp.sum(self.n * self.H_P)
        )
        return -logl

    def start(self) -> npt.NDArray:
        excess = np.maximum(self.events - self.expected, 0.1 * self.events)
        theta = np.log(np.maximum(excess, 0.5) / self.at_risk)
        return np.concatenate((theta, np.zeros(self.Zs.shape[1])))

    def to_search(self, log_rate: Any, beta: Any) -> npt.NDArray:
        """Natural log-rates and coefficients (of the kept columns) in the
        search's units."""
        return np.linalg.solve(
            self.jacobian(), np.concatenate((log_rate, beta))
        )

    def jacobian(self) -> npt.NDArray:
        """d(natural log-rates, coefficients) / d(search vector): a
        linear map."""
        p = self.Zs.shape[1]
        sd = np.where(self.sd > 0, self.sd, 1.0)
        J = np.eye(self.K + p)
        J[: self.K, self.K :] = -(self.mean / sd)[None, :]
        J[self.K :, self.K :] = np.diag(1.0 / sd)
        return J

    def to_natural(self, v: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        u = self.jacobian() @ v
        return u[: self.K], u[self.K :]


def _minimise(problem: _Problem, starts: list[npt.NDArray]) -> Any:
    fun = problem.nll
    jac = grad(fun)
    hess = hessian(fun)
    best = None
    for v0 in starts:
        with np.errstate(all="ignore"), ignoring_warnings():
            try:
                res = minimize(
                    fun,
                    v0,
                    jac=jac,
                    hess=hess,
                    method="trust-exact",
                    options={"gtol": 1e-9, "maxiter": 500},
                )
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                continue
        if not np.isfinite(res.fun):
            continue
        if best is None or res.fun < best.fun:
            best = res
    if best is None:
        raise ValueError(
            "The excess hazard fit found no point with a finite "
            "likelihood: check that the rate table's rates are those of "
            "the units' population, and give 'init' if you have a start."
        )
    return best, jac, hess


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class ExcessHazard(InformationCriteriaMixin, SerialisableMixin):
    r"""
    Excess hazard regression (Estève et al., 1990): the hazard of each
    unit is its population's hazard, from a :class:`surpyval.RateTable`,
    plus an excess hazard :math:`\lambda_0(t) e^{\beta' Z}` that the
    covariates act on proportionally, :math:`\lambda_0` constant within
    intervals of follow-up time. See the module
    :mod:`surpyval.univariate.relative.excess_hazard`.

    ``params`` are the baseline excess hazard rates, one per interval
    (``rate_0``, ...; the intervals are ``intervals``), then the
    coefficients, named by their columns (``parameter_names``). ``sf``,
    ``ff``, ``hf``, ``Hf``, ``df`` and ``qf`` are the net (excess-only)
    functions of a unit with covariates ``Z``: ``sf(x, Z)`` is its net
    survival. :meth:`observed_sf` is a unit's all-cause survival.

    Examples
    --------
    Simulated patients aged 50 to 80 whose treatment halves the excess
    hazard, with population rates rising 9% a year of age:

    >>> import numpy as np
    >>> import surpyval as sp
    >>> rng = np.random.default_rng(1)
    >>> ages = np.arange(0, 121)
    >>> table = sp.RateTable(5e-5 * np.exp(0.09 * ages), ages=ages)
    >>> age = rng.uniform(50, 80, 3000)
    >>> treated = rng.binomial(1, 0.5, 3000)
    >>> excess = rng.exponential(1 / (0.15 * np.exp(-0.7 * treated)))
    >>> grid = np.linspace(0, 60, 601)
    >>> u = rng.uniform(size=3000)
    >>> background = np.array([np.interp(-np.log(v), table.Hf(grid, a), grid)
    ...                        for v, a in zip(u, age)])
    >>> time = np.minimum(np.minimum(excess, background), 8.0)
    >>> model = sp.ExcessHazard.fit(time, treated, c=(time == 8.0),
    ...                             rate_table=table, age=age)
    >>> model.parameter_names
    ['rate_0', 'coef_0']
    >>> model.params.round(3)  # the truth is 0.15 and -0.7
    array([ 0.146, -0.716])
    """

    params: npt.NDArray
    breaks: npt.NDArray
    aliased: npt.NDArray
    feature_names: list | None
    rate_table: RateTable | None
    maximum: str
    _cov_log: npt.NDArray | None
    _names: list

    # -- fitting ----------
    @classmethod
    @takes_unit
    @_refuses_data
    @dataframe_covariates
    def fit(
        cls,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        rate_table: RateTable | None = None,
        age: npt.ArrayLike | None = None,
        year: npt.ArrayLike | None = None,
        stratum: Any = None,
        breaks: Any = None,
        init: npt.ArrayLike | None = None,
    ) -> ExcessHazard:
        r"""
        Fit the excess hazard model by maximum likelihood.

        Parameters
        ----------
        x : array_like
            Each unit's follow-up time from time 0, in the rate table's
            unit.
        Z : array_like
            The covariates, one row per row of ``x`` (a DataFrame's
            columns name the coefficients).
        c : array_like, optional
            0 where the follow-up ended in the event (of any cause), 1
            where it was right censored. Defaults to all 0.
        n : array_like, optional
            Counts of identical units. Defaults to 1 per row.
        rate_table : RateTable
            The population's hazards (required).
        age : array_like
            Each unit's age at time 0, in the table's unit (required).
        year : array_like, optional
            Each unit's calendar time at time 0, where the table is by
            year.
        stratum : array_like, optional
            Each unit's stratum of the table, where it has strata.
        breaks : int or array_like, optional
            The intervals of follow-up time the baseline excess hazard is
            constant on. ``None`` (the default, as R's relsurv): one
            interval, a constant excess hazard. A number ``k``: ``k``
            intervals with equal numbers of events. An array: the times
            at which the hazard may change. Every interval needs events.
        init : array_like, optional
            A start: the baseline rates, then the coefficients. The fit is
            also started from its own default start, and the better kept.

        Returns
        -------
        ExcessHazard

        A row with a missing age, year, stratum or covariate is dropped,
        with one warning; a missing time is refused. A covariate column
        that is constant, or a combination of the others, is aliased: its
        coefficient is ``nan``, with one warning.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.01, 0.03], ages=[0, 70])
        >>> x = [1.2, 2.5, 3.1, 4.0, 4.4, 5.0, 6.3, 7.5, 8.0, 9.1]
        >>> c = [0, 0, 0, 1, 0, 0, 1, 0, 1, 1]
        >>> Z = [1, 1, 1, 0, 1, 0, 0, 1, 0, 0]
        >>> age = [62, 70, 58, 66, 75, 61, 69, 72, 55, 64]
        >>> model = sp.ExcessHazard.fit(x, Z, c=c, rate_table=table, age=age)
        >>> model.params.round(3)
        array([0.021, 2.466])
        """
        if rate_table is None:
            raise ValueError(
                "'rate_table' is required: the population's hazards, a "
                "surpyval.RateTable."
            )
        if Z is None:
            raise ValueError(
                "'Z' is required: the covariates, one row per row of x. "
                "For net survival without covariates use "
                "surpyval.RelativeSurvival."
            )
        data = check_data(x, c, n, rate_table, age, year, stratum, Z=Z)
        assert data.Z is not None
        p = data.Z.shape[1]
        cuts = _cuts(breaks, data)
        events = np.bincount(
            _interval(data.x[data.c == 0], cuts),
            weights=data.n[data.c == 0],
            minlength=cuts.size + 1,
        )
        if np.any(events == 0):
            k = int(np.flatnonzero(events == 0)[0])
            raise ValueError(
                f"'breaks': the interval {_describe(cuts, k)} has no events, "
                "so its excess hazard cannot be estimated. Merge it with a "
                "neighbour (fewer or other breaks)."
            )
        kept = cls._kept_columns(data)
        problem = _Problem(rate_table, data, cuts, kept)
        starts = [problem.start()]
        if init is not None:
            starts.append(cls._init_start(problem, init, kept, p))
        res, jac, hess = _minimise(problem, starts)
        res.start = starts[0]
        out = cls.__new__(cls)
        out.breaks = cuts
        out.rate_table = rate_table
        out.aliased = np.setdiff1d(np.arange(p), kept)
        names = fit_columns()
        out.feature_names = (
            names if names is not None and len(names) == p else None
        )
        rate_names = [f"rate_{k}" for k in range(cuts.size + 1)]
        out._names = rate_names + coefficient_names(
            p, out.feature_names, taken=rate_names
        )
        log_rate, beta_kept = problem.to_natural(res.x)
        beta = np.full(p, np.nan)
        beta[kept] = beta_kept
        out.params = np.concatenate((np.exp(log_rate), beta))
        out.k = int(problem.K + kept.size)
        out._neg_ll = float(res.fun)
        out._ic_n = problem.n_obs
        out.maximum = cls._judge(problem, res, jac, hess, cuts)
        out._cov_log = cls._covariance_of(problem, res, hess, out.maximum)
        return out

    @staticmethod
    def _kept_columns(data: RelativeData) -> npt.NDArray:
        """The columns of Z the fit can estimate; the others are aliased,
        with one warning."""
        assert data.Z is not None
        Z = data.Z
        p = Z.shape[1]
        w = data.n
        centred = Z - (w @ Z) / w.sum()
        gram = centred.T @ (w[:, None] * centred)
        constant = constant_columns(Z)
        aliased = aliased_columns(gram, int(w.sum()), constant)
        if aliased.size:
            warn_aliased(
                aliased,
                "each is constant (the baseline excess hazard absorbs it) "
                "or a combination of the other columns",
            )
        return np.setdiff1d(np.arange(p), aliased)

    @staticmethod
    def _init_start(
        problem: _Problem, init: Any, kept: npt.NDArray, p: int
    ) -> npt.NDArray:
        try:
            v = np.asarray(init, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            raise ValueError(
                "'init' must be numbers: the baseline rates, then the "
                "coefficients."
            ) from None
        if v.size != problem.K + p:
            raise ValueError(
                f"'init' has {v.size} values; give the {problem.K} baseline "
                f"rate(s), then the {p} coefficient(s)."
            )
        if np.any(v[: problem.K] <= 0) or not np.all(np.isfinite(v)):
            raise ValueError(
                "'init' must be finite, with the baseline rates above 0."
            )
        return problem.to_search(np.log(v[: problem.K]), v[problem.K :][kept])

    @staticmethod
    def _judge(
        problem: _Problem, res: Any, jac: Any, hess: Any, cuts: npt.NDArray
    ) -> str:
        """What the fit reached, warning where it is not a verified
        maximum."""
        theta, gamma = res.x[: problem.K], res.x[problem.K :]
        expected = np.exp(theta) * (
            (problem.n * np.exp(problem.Zs @ gamma)) @ problem.D
        )
        with ignoring_warnings():
            runaway = runaway_coefficients(
                problem.nll, res.x, list(range(res.x.size)), res.start
            )
        gone = np.union1d(
            np.flatnonzero(expected < _NO_EXCESS),
            [k for k in runaway if k < problem.K],
        ).astype(int)
        if gone.size:
            which = ", ".join(_describe(cuts, int(k)) for k in gone)
            warn_no_maximum(
                f"the baseline excess hazard of the interval(s) {which} "
                "runs to 0: there are no more events there than the "
                "population rates expect",
                "Those rates, their standard errors and bounds are "
                "meaningless",
                "merge the interval(s) with a neighbour (fewer or other "
                "breaks), or check that the rate table is that of the "
                "units' own population",
            )
            return "no finite maximum"
        if np.any(np.abs(gamma) > _RUNAWAY) or any(
            k >= problem.K for k in runaway
        ):
            warn_no_maximum(
                "a coefficient runs off to infinity: a covariate separates "
                "the units with excess events from those without",
                "Its coefficient, standard error and bounds are "
                "meaningless",
                "remove or regroup that covariate",
            )
            return "no finite maximum"
        if is_local_minimum(
            problem.nll, jac, hess, res.x, obj_scale=problem.n_obs
        ):
            return "verified"
        warn_unverified("The excess hazard fit", advice="try another 'init'")
        return "unverified"

    @staticmethod
    def _covariance_of(
        problem: _Problem, res: Any, hess: Any, maximum: str
    ) -> npt.NDArray | None:
        """The covariance of the natural log-rates and the coefficients
        of the kept columns, or None."""
        with np.errstate(all="ignore"):
            try:
                H = np.asarray(hess(res.x), dtype=float)
                np.linalg.cholesky(0.5 * (H + H.T))
                cov = np.linalg.inv(H)
            except (np.linalg.LinAlgError, ValueError):
                if maximum != "no finite maximum":
                    warn_no_covariance()
                return None
        J = problem.jacobian()
        return J @ cov @ J.T

    @classmethod
    @takes_unit
    def fit_from_df(
        cls,
        df: Any,
        x_col: str,
        Z_cols: Any,
        age_col: str,
        rate_table: RateTable,
        c_col: str | None = None,
        n_col: str | None = None,
        year_col: str | None = None,
        stratum_col: str | None = None,
        breaks: Any = None,
        init: npt.ArrayLike | None = None,
    ) -> ExcessHazard:
        """
        Fit to the columns of a DataFrame, one row per unit (or group of
        identical units, with ``n_col``): the columns are the arguments of
        :meth:`fit`, and ``Z_cols`` the covariates, which name the
        coefficients.

        Examples
        --------
        >>> import pandas as pd
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.01, 0.03], ages=[0, 70])
        >>> df = pd.DataFrame({
        ...     "years": [1.2, 2.5, 3.1, 4.0, 4.4, 5.0, 6.3, 7.5, 8.0, 9.1],
        ...     "censored": [0, 0, 0, 1, 0, 0, 1, 0, 1, 1],
        ...     "treated": [1, 1, 1, 0, 1, 0, 0, 1, 0, 0],
        ...     "age": [62, 70, 58, 66, 75, 61, 69, 72, 55, 64]})
        >>> model = sp.ExcessHazard.fit_from_df(
        ...     df, "years", ["treated"], "age", table, c_col="censored")
        >>> model.parameter_names
        ['rate_0', 'treated']
        """
        from surpyval.utils.dataframe import (
            check_columns,
            frame_column,
            frame_columns,
        )

        names = [Z_cols] if isinstance(Z_cols, str) else list(Z_cols)
        check_columns(
            df,
            x_col=x_col,
            Z_cols=names,
            age_col=age_col,
            c_col=c_col,
            n_col=n_col,
            year_col=year_col,
            stratum_col=stratum_col,
        )

        def column(name: str | None, arg: str) -> Any:
            return None if name is None else frame_column(df, name, arg)

        Z = frame_columns(df, names, "Z_cols").astype(float)
        with covariate_columns(names, Z):
            return cls.fit(
                frame_column(df, x_col, "x_col", time=True),
                Z,
                c=column(c_col, "c_col"),
                n=column(n_col, "n_col"),
                rate_table=rate_table,
                age=column(age_col, "age_col"),
                year=column(year_col, "year_col"),
                stratum=column(stratum_col, "stratum_col"),
                breaks=breaks,
                init=init,
            )

    # -- what it is ----------
    @property
    def parameter_names(self) -> list[str]:
        """The baseline rates' names (``rate_0``, ...), then the
        coefficients' (their columns' names, else ``coef_0``, ...)."""
        return list(self._names)

    @property
    def n_intervals(self) -> int:
        """How many intervals the baseline excess hazard is constant on."""
        return int(self.breaks.size + 1)

    @property
    def intervals(self) -> list[tuple[float, float]]:
        """The intervals of follow-up time the baseline excess hazard is
        constant on, as ``(start, end)`` pairs (the last ends at
        ``inf``)."""
        edges = np.concatenate(([0.0], self.breaks, [np.inf]))
        return [(float(a), float(b)) for a, b in zip(edges[:-1], edges[1:])]

    @property
    def rates(self) -> npt.NDArray:
        """The baseline excess hazard of each interval (at ``Z = 0``)."""
        return np.asarray(self.params[: self.n_intervals], dtype=float)

    @property
    def coef(self) -> npt.NDArray:
        """The coefficients: the logs of the excess hazard ratios (``nan``
        for an aliased column)."""
        return np.asarray(self.params[self.n_intervals :], dtype=float)

    def _parameter_bounds(self) -> list[tuple]:
        """The support of each of ``params``."""
        return [(0, None)] * self.n_intervals + [(None, None)] * (
            self.coef.size
        )

    # -- the functions ----------
    def _rows(self, Z: Any) -> npt.NDArray:
        if hasattr(Z, "columns") and self.feature_names is not None:
            missing = [f for f in self.feature_names if f not in Z.columns]
            if missing:
                raise ValueError(
                    f"Z is missing the column(s) {missing} the model was "
                    f"fitted with: {self.feature_names}."
                )
            Z = Z[self.feature_names].to_numpy(dtype=float)
        try:
            rows = covariate_rows(Z, self.coef.size)
        except (TypeError, ValueError):
            raise ValueError(
                "Z must be numbers: one row of the model's "
                f"{self.coef.size} covariate(s)."
            ) from None
        if rows.ndim != 2 or rows.shape[1] != self.coef.size:
            raise ValueError(
                f"Z has {rows.shape[-1]} column(s); the model has "
                f"{self.coef.size} covariate(s)."
            )
        return rows

    def _risk(self, x: npt.NDArray, Z: Any) -> tuple[npt.NDArray, npt.NDArray]:
        """The times and each one's relative excess hazard, paired."""
        rows = self._rows(Z)
        check_paired_rows(x.size, rows.shape[0], grid=False)
        kept = np.isfinite(self.coef)
        eta = rows[:, kept] @ self.coef[kept]
        n = max(x.size, eta.size)
        return np.broadcast_to(x, (n,)), np.broadcast_to(np.exp(eta), (n,))

    def _H0(self, x: npt.NDArray) -> npt.NDArray:
        t = np.where(np.isnan(x), 0.0, np.maximum(x, 0.0))
        out = _exposure(t, self.breaks) @ self.rates
        return np.where(np.isnan(x), np.nan, out)

    def _h0(self, x: npt.NDArray) -> npt.NDArray:
        t = np.where(np.isnan(x), 0.0, x)
        out = self.rates[_interval(t, self.breaks)]
        out = np.where(t < 0, 0.0, out)
        return np.where(np.isnan(x), np.nan, out)

    @keeps_query_shape
    def Hf(self, x: npt.ArrayLike, Z: Any) -> npt.NDArray:
        r"""
        The cumulative excess hazard :math:`\Lambda_0(x) e^{\beta' Z}` at
        ``x`` of a unit with covariates ``Z`` (one row for every time, or
        one row per time).

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.01, 0.03], ages=[0, 70])
        >>> x = [1.2, 2.5, 3.1, 4.0, 4.4, 5.0, 6.3, 7.5, 8.0, 9.1]
        >>> c = [0, 0, 0, 1, 0, 0, 1, 0, 1, 1]
        >>> Z = [1, 1, 1, 0, 1, 0, 0, 1, 0, 0]
        >>> age = [62, 70, 58, 66, 75, 61, 69, 72, 55, 64]
        >>> model = sp.ExcessHazard.fit(x, Z, c=c, rate_table=table, age=age)
        >>> model.Hf([5, 5], [[0], [1]]).round(3)
        array([0.104, 1.229])
        """
        t, risk = self._risk(np.asarray(x, dtype=float), Z)
        return self._H0(t) * risk

    @keeps_query_shape
    def hf(self, x: npt.ArrayLike, Z: Any) -> npt.NDArray:
        r"""
        The excess hazard :math:`\lambda_0(x) e^{\beta' Z}` at ``x`` of a
        unit with covariates ``Z`` (the rate of the interval ``x`` is in,
        each interval closed on the right).

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.01, 0.03], ages=[0, 70])
        >>> x = [1.2, 2.5, 3.1, 4.0, 4.4, 5.0, 6.3, 7.5, 8.0, 9.1]
        >>> c = [0, 0, 0, 1, 0, 0, 1, 0, 1, 1]
        >>> Z = [1, 1, 1, 0, 1, 0, 0, 1, 0, 0]
        >>> age = [62, 70, 58, 66, 75, 61, 69, 72, 55, 64]
        >>> model = sp.ExcessHazard.fit(x, Z, c=c, rate_table=table, age=age)
        >>> model.hf(5, [1]).round(3)
        np.float64(0.246)
        """
        t, risk = self._risk(np.asarray(x, dtype=float), Z)
        return self._h0(t) * risk

    @keeps_query_shape
    def sf(self, x: npt.ArrayLike, Z: Any) -> npt.NDArray:
        """
        Net survival at ``x`` of a unit with covariates ``Z``: its survival
        were the excess hazard the only one.

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.01, 0.03], ages=[0, 70])
        >>> x = [1.2, 2.5, 3.1, 4.0, 4.4, 5.0, 6.3, 7.5, 8.0, 9.1]
        >>> c = [0, 0, 0, 1, 0, 0, 1, 0, 1, 1]
        >>> Z = [1, 1, 1, 0, 1, 0, 0, 1, 0, 0]
        >>> age = [62, 70, 58, 66, 75, 61, 69, 72, 55, 64]
        >>> model = sp.ExcessHazard.fit(x, Z, c=c, rate_table=table, age=age)
        >>> model.sf(5, [[0], [1]]).round(3)
        array([0.901, 0.293])
        """
        return np.exp(-self.Hf(np.asarray(x, dtype=float).ravel(), Z))

    @keeps_query_shape
    def ff(self, x: npt.ArrayLike, Z: Any) -> npt.NDArray:
        """
        The net probability of the event by ``x``, ``1 - sf(x, Z)``.

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.01, 0.03], ages=[0, 70])
        >>> x = [1.2, 2.5, 3.1, 4.0, 4.4, 5.0, 6.3, 7.5, 8.0, 9.1]
        >>> c = [0, 0, 0, 1, 0, 0, 1, 0, 1, 1]
        >>> Z = [1, 1, 1, 0, 1, 0, 0, 1, 0, 0]
        >>> age = [62, 70, 58, 66, 75, 61, 69, 72, 55, 64]
        >>> model = sp.ExcessHazard.fit(x, Z, c=c, rate_table=table, age=age)
        >>> model.ff(5, [1]).round(3)
        np.float64(0.707)
        """
        return -np.expm1(-self.Hf(np.asarray(x, dtype=float).ravel(), Z))

    @keeps_query_shape
    def df(self, x: npt.ArrayLike, Z: Any) -> npt.NDArray:
        """
        The net density, ``hf(x, Z) * sf(x, Z)``.

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.01, 0.03], ages=[0, 70])
        >>> x = [1.2, 2.5, 3.1, 4.0, 4.4, 5.0, 6.3, 7.5, 8.0, 9.1]
        >>> c = [0, 0, 0, 1, 0, 0, 1, 0, 1, 1]
        >>> Z = [1, 1, 1, 0, 1, 0, 0, 1, 0, 0]
        >>> age = [62, 70, 58, 66, 75, 61, 69, 72, 55, 64]
        >>> model = sp.ExcessHazard.fit(x, Z, c=c, rate_table=table, age=age)
        >>> model.df(5, [1]).round(3)
        np.float64(0.072)
        """
        t = np.asarray(x, dtype=float).ravel()
        return self.hf(t, Z) * self.sf(t, Z)

    @keeps_query_shape
    def qf(self, p: npt.ArrayLike, Z: Any) -> npt.NDArray:
        """
        The net quantile: the time by which a unit with covariates ``Z``
        has the event with net probability ``p`` (``inf`` where the
        excess hazard of the last interval is 0 and it never does). A
        probability outside [0, 1] gives ``nan``, with a warning.

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.01, 0.03], ages=[0, 70])
        >>> x = [1.2, 2.5, 3.1, 4.0, 4.4, 5.0, 6.3, 7.5, 8.0, 9.1]
        >>> c = [0, 0, 0, 1, 0, 0, 1, 0, 1, 1]
        >>> Z = [1, 1, 1, 0, 1, 0, 0, 1, 0, 0]
        >>> age = [62, 70, 58, 66, 75, 61, 69, 72, 55, 64]
        >>> model = sp.ExcessHazard.fit(x, Z, c=c, rate_table=table, age=age)
        >>> model.qf(0.5, [1]).round(3)
        np.float64(2.821)
        """
        u = np.asarray(p, dtype=float).ravel()
        outside = warn_outside_unit_interval(u)
        u = np.where(outside, np.nan, u)
        u, risk = self._risk(u, Z)
        with np.errstate(divide="ignore", invalid="ignore"):
            target = -np.log1p(-u) / risk
        edges = np.concatenate(([0.0], self.breaks))
        widths = np.diff(edges)
        cum = np.concatenate(([0.0], np.cumsum(self.rates[:-1] * widths)))
        safe = np.where(np.isnan(target), 0.0, target)
        k = np.searchsorted(cum, safe, side="right") - 1
        k = np.clip(k, 0, self.rates.size - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = edges[k] + (safe - cum[k]) / self.rates[k]
        t = np.where(safe == 0, 0.0, t)
        t = np.where(np.isinf(safe), np.inf, t)
        return np.where(np.isnan(target), np.nan, t)

    def random(
        self, size: int, Z: Any, random_state: Any = None
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Draw net lifetimes, ``size`` for each covariate row of ``Z``:
        lifetimes under the excess hazard alone.

        Returns
        -------
        x : numpy.ndarray
            The draws, row by row.
        Z : numpy.ndarray
            The covariate row of each draw.

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.01, 0.03], ages=[0, 70])
        >>> x = [1.2, 2.5, 3.1, 4.0, 4.4, 5.0, 6.3, 7.5, 8.0, 9.1]
        >>> c = [0, 0, 0, 1, 0, 0, 1, 0, 1, 1]
        >>> Z = [1, 1, 1, 0, 1, 0, 0, 1, 0, 0]
        >>> age = [62, 70, 58, 66, 75, 61, 69, 72, 55, 64]
        >>> model = sp.ExcessHazard.fit(x, Z, c=c, rate_table=table, age=age)
        >>> draws, rows = model.random(3, [[0], [1]], random_state=1)
        >>> draws.shape, rows.shape
        ((6,), (6, 1))
        """
        rows = self._rows(Z)
        rows = np.repeat(rows, int(size), axis=0)
        if random_state is None:
            u = np.random.uniform(0, 1, rows.shape[0])
        else:
            u = as_generator(random_state).uniform(0, 1, rows.shape[0])
        return np.asarray(self.qf(u, rows), dtype=float), rows

    def observed_sf(
        self,
        x: npt.ArrayLike,
        Z: Any,
        age: npt.ArrayLike,
        year: npt.ArrayLike | None = None,
        stratum: Any = None,
    ) -> Any:
        """
        A unit's all-cause survival at ``x``: its expected survival from
        the rate table (for its ``age``, ``year`` and ``stratum`` at time
        0) times its net survival ``sf(x, Z)``.

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.01, 0.03], ages=[0, 70])
        >>> x = [1.2, 2.5, 3.1, 4.0, 4.4, 5.0, 6.3, 7.5, 8.0, 9.1]
        >>> c = [0, 0, 0, 1, 0, 0, 1, 0, 1, 1]
        >>> Z = [1, 1, 1, 0, 1, 0, 0, 1, 0, 0]
        >>> age = [62, 70, 58, 66, 75, 61, 69, 72, 55, 64]
        >>> model = sp.ExcessHazard.fit(x, Z, c=c, rate_table=table, age=age)
        >>> float(model.observed_sf(5, [1], age=68).round(3))
        0.262
        """
        if self.rate_table is None:
            raise ValueError(
                "This model has no rate table (it was restored without "
                "one), so it has no expected survival to multiply in."
            )
        expected = np.asarray(self.rate_table.sf(x, age, year, stratum))
        net = np.asarray(self.sf(x, Z))
        out = expected * net
        return out[()] if out.ndim == 0 else out

    # -- uncertainty ----------
    def _require_covariance(self) -> npt.NDArray:
        if self._cov_log is None:
            raise ValueError(
                "This model has no covariance: its information matrix at "
                "the fit was not positive definite (see the fit's "
                "warning). Use fewer breaks or covariates for bounds."
            )
        return self._cov_log

    def covariance(self) -> npt.NDArray:
        """
        The covariance of ``params`` (the baseline rates, then the
        coefficients), from the inverse observed information at the
        maximum; ``nan`` for an aliased coefficient.

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.01, 0.03], ages=[0, 70])
        >>> x = [1.2, 2.5, 3.1, 4.0, 4.4, 5.0, 6.3, 7.5, 8.0, 9.1]
        >>> c = [0, 0, 0, 1, 0, 0, 1, 0, 1, 1]
        >>> Z = [1, 1, 1, 0, 1, 0, 0, 1, 0, 0]
        >>> age = [62, 70, 58, 66, 75, 61, 69, 72, 55, 64]
        >>> model = sp.ExcessHazard.fit(x, Z, c=c, rate_table=table, age=age)
        >>> model.covariance().shape
        (2, 2)
        """
        cov = self._require_covariance()
        K = self.n_intervals
        kept = np.flatnonzero(np.isfinite(self.coef))
        idx = np.concatenate((np.arange(K), K + kept))
        J = np.concatenate((self.rates, np.ones(kept.size)))
        out = np.full((self.params.size,) * 2, np.nan)
        out[np.ix_(idx, idx)] = cov * np.outer(J, J)
        return out

    def standard_errors(self) -> npt.NDArray:
        """
        The standard errors of ``params``, the square roots of the
        diagonal of :meth:`covariance`.

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.01, 0.03], ages=[0, 70])
        >>> x = [1.2, 2.5, 3.1, 4.0, 4.4, 5.0, 6.3, 7.5, 8.0, 9.1]
        >>> c = [0, 0, 0, 1, 0, 0, 1, 0, 1, 1]
        >>> Z = [1, 1, 1, 0, 1, 0, 0, 1, 0, 0]
        >>> age = [62, 70, 58, 66, 75, 61, 69, 72, 55, 64]
        >>> model = sp.ExcessHazard.fit(x, Z, c=c, rate_table=table, age=age)
        >>> model.standard_errors().round(3)
        array([0.031, 1.557])
        """
        with np.errstate(invalid="ignore"):
            return np.sqrt(np.diag(self.covariance()))

    def _log_space(self) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """The estimates (log-rates, kept coefficients), their covariance,
        and the kept columns."""
        cov = self._require_covariance()
        kept = np.flatnonzero(np.isfinite(self.coef))
        est = np.concatenate((np.log(self.rates), self.coef[kept]))
        return est, cov, kept

    def param_cb(
        self,
        name: str,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "wald",
    ) -> Any:
        """
        The Wald confidence bounds of one parameter, named as in
        ``parameter_names``: a baseline rate's on the log scale (so they
        stay above 0), a coefficient's as it is (``nan`` for an aliased
        one). ``method`` is ``"wald"``, the one built: ``"lr"`` and
        ``"bootstrap"`` are refused, saying what to do instead.

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.01, 0.03], ages=[0, 70])
        >>> x = [1.2, 2.5, 3.1, 4.0, 4.4, 5.0, 6.3, 7.5, 8.0, 9.1]
        >>> c = [0, 0, 0, 1, 0, 0, 1, 0, 1, 1]
        >>> Z = [1, 1, 1, 0, 1, 0, 0, 1, 0, 0]
        >>> age = [62, 70, 58, 66, 75, 61, 69, 72, 55, 64]
        >>> model = sp.ExcessHazard.fit(x, Z, c=c, rate_table=table, age=age)
        >>> model.param_cb("coef_0").round(3)
        array([-0.586,  5.518])
        """
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        _check_method(method)
        if name not in self._names:
            raise ValueError(
                f"Unknown parameter {name!r}; the model's parameters are "
                f"{self._names}."
            )
        j = self._names.index(name)
        alpha = alpha_ci / 2 if bound == "two-sided" else alpha_ci
        z = norm.ppf(1 - alpha)
        K = self.n_intervals
        if j >= K and not np.isfinite(self.params[j]):
            lo = hi = np.nan
        else:
            est, cov, kept = self._log_space()
            pos = j if j < K else K + int(np.flatnonzero(kept == j - K)[0])
            se = float(np.sqrt(cov[pos, pos]))
            lo, hi = est[pos] - z * se, est[pos] + z * se
            if j < K:
                lo, hi = np.exp(lo), np.exp(hi)
        if bound == "lower":
            return np.float64(lo)
        if bound == "upper":
            return np.float64(hi)
        return np.array([lo, hi], dtype=float)

    @keeps_query_shape
    def cb(
        self,
        x: npt.ArrayLike,
        Z: Any,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "wald",
    ) -> npt.NDArray:
        r"""
        Wald confidence bounds on net survival (or ``ff`` or ``Hf``) at
        ``x`` for covariates ``Z``, by the delta method on the log of the
        cumulative excess hazard (so the bounds on ``sf`` stay inside
        [0, 1]). ``method`` is ``"wald"``, the one built: ``"lr"`` and
        ``"bootstrap"`` are refused, saying what to do instead.

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.01, 0.03], ages=[0, 70])
        >>> x = [1.2, 2.5, 3.1, 4.0, 4.4, 5.0, 6.3, 7.5, 8.0, 9.1]
        >>> c = [0, 0, 0, 1, 0, 0, 1, 0, 1, 1]
        >>> Z = [1, 1, 1, 0, 1, 0, 0, 1, 0, 0]
        >>> age = [62, 70, 58, 66, 75, 61, 69, 72, 55, 64]
        >>> model = sp.ExcessHazard.fit(x, Z, c=c, rate_table=table, age=age)
        >>> model.cb(5, [1]).round(3)
        array([0.041, 0.623])
        """
        check_alpha_ci(alpha_ci)
        check_option("on", on, _CB_ON)
        check_option("bound", bound, BOUNDS)
        _check_method(method)
        on = {"R": "sf", "F": "ff"}.get(on, on)
        est, cov, kept = self._log_space()
        t = np.asarray(x, dtype=float)
        rows = self._rows(Z)
        check_paired_rows(t.size, rows.shape[0], grid=False)
        n = max(t.size, rows.shape[0])
        t = np.broadcast_to(t, (n,))
        rows = np.broadcast_to(rows, (n, rows.shape[1]))
        H = np.asarray(self.Hf(t, rows), dtype=float).reshape(n)
        safe = np.where(np.isnan(t), 0.0, np.maximum(t, 0.0))
        part = _exposure(safe, self.breaks) * self.rates[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            g_rate = part / part.sum(axis=1, keepdims=True)
        g = np.concatenate((g_rate, rows[:, kept]), axis=1)
        se = np.sqrt(np.einsum("ij,jk,ik->i", g, cov, g))
        alpha = alpha_ci / 2 if bound == "two-sided" else alpha_ci
        z = norm.ppf(1 - alpha)
        with np.errstate(over="ignore", invalid="ignore"):
            H_low = np.where(H > 0, H * np.exp(-z * se), H)
            H_high = np.where(H > 0, H * np.exp(z * se), H)
        if on == "Hf":
            low, high = H_low, H_high
        elif on == "sf":
            low, high = np.exp(-H_high), np.exp(-H_low)
        else:
            low, high = -np.expm1(-H_low), -np.expm1(-H_high)
        if bound == "lower":
            return low
        if bound == "upper":
            return high
        return np.stack([low, high], axis=-1)

    def summary(self, alpha_ci: float = 0.05) -> Any:
        """
        The coefficients as a table: each with its excess hazard ratio
        ``exp(coef)``, standard error, Wald interval at ``alpha_ci`` and
        test of 0.

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.01, 0.03], ages=[0, 70])
        >>> x = [1.2, 2.5, 3.1, 4.0, 4.4, 5.0, 6.3, 7.5, 8.0, 9.1]
        >>> c = [0, 0, 0, 1, 0, 0, 1, 0, 1, 1]
        >>> Z = [1, 1, 1, 0, 1, 0, 0, 1, 0, 0]
        >>> age = [62, 70, 58, 66, 75, 61, 69, 72, 55, 64]
        >>> model = sp.ExcessHazard.fit(x, Z, c=c, rate_table=table, age=age)
        >>> model.summary()[["coef", "exp(coef)", "p"]].round(3)
                    coef  exp(coef)      p
        covariate
        coef_0     2.466     11.778  0.113
        """
        from surpyval.univariate.regression._summary import coefficient_table

        check_alpha_ci(alpha_ci)
        K = self.n_intervals
        se = self.standard_errors()[K:]
        return coefficient_table(self._names[K:], self.coef, se, alpha_ci)

    # -- information criteria ----------
    def _ic_sample_size_from_data(self) -> float:
        raise ValueError("The model keeps no data to count")

    # -- reading it ----------
    def __repr__(self) -> str:
        title = "Excess Hazard SurPyval Model"
        lines = [
            title,
            "=" * len(title),
            "Model        : population hazard + baseline excess hazard "
            "x exp(beta'Z)",
            f"Baseline     : constant on {self.n_intervals} interval(s)",
        ]
        if self.maximum in ("unverified", "no finite maximum"):
            lines.append(f"Maximum      : {self.maximum}")
        if self._neg_ll is not None:
            lines.append(f"Log-like     : {-self._neg_ll:.6g}")
        lines.append("Parameters   :")
        for name, value in zip(self._names, self.params):
            lines.append(f"{name:>12}: {value:.6g}")
        return "\n".join(lines)

    # -- saving ----------
    def to_dict(self) -> dict:
        """
        The model as a plain, JSON-ready ``dict`` (its rate table
        included); ``surpyval.from_dict`` rebuilds it.

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.01, 0.03], ages=[0, 70])
        >>> x = [1.2, 2.5, 3.1, 4.0, 4.4, 5.0, 6.3, 7.5, 8.0, 9.1]
        >>> c = [0, 0, 0, 1, 0, 0, 1, 0, 1, 1]
        >>> Z = [1, 1, 1, 0, 1, 0, 0, 1, 0, 0]
        >>> age = [62, 70, 58, 66, 75, 61, 69, 72, 55, 64]
        >>> model = sp.ExcessHazard.fit(x, Z, c=c, rate_table=table, age=age)
        >>> restored = sp.from_dict(model.to_dict())
        >>> bool(restored.sf(5, [1]) == model.sf(5, [1]))
        True
        """
        out: dict[str, Any] = {
            "model": "ExcessHazard",
            "params": self.params.tolist(),
            "parameter_names": list(self._names),
            "breaks": self.breaks.tolist(),
            "aliased": self.aliased.tolist(),
            "k": int(self.k),
            "_neg_ll": None if self._neg_ll is None else float(self._neg_ll),
            "ic_n": self._ic_sample_size_or_none(),
            "covariance": (
                None if self._cov_log is None else self._cov_log.tolist()
            ),
            "rate_table": (
                None if self.rate_table is None else self.rate_table.to_dict()
            ),
            **maximum_entry(self.maximum),
        }
        if self.feature_names is not None:
            out["feature_names"] = to_native(self.feature_names)
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> ExcessHazard:
        """
        Rebuild a model from :meth:`to_dict`'s dictionary.

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.01, 0.03], ages=[0, 70])
        >>> x = [1.2, 2.5, 3.1, 4.0, 4.4, 5.0, 6.3, 7.5, 8.0, 9.1]
        >>> c = [0, 0, 0, 1, 0, 0, 1, 0, 1, 1]
        >>> Z = [1, 1, 1, 0, 1, 0, 0, 1, 0, 0]
        >>> age = [62, 70, 58, 66, 75, 61, 69, 72, 55, 64]
        >>> model = sp.ExcessHazard.fit(x, Z, c=c, rate_table=table, age=age)
        >>> sp.ExcessHazard.from_dict(model.to_dict()).parameter_names
        ['rate_0', 'coef_0']
        """
        require_model_tag(model_dict, "ExcessHazard", "an excess hazard model")
        out = cls.__new__(cls)
        out.params = np.asarray(model_dict["params"], dtype=float)
        out._names = [str(n) for n in model_dict["parameter_names"]]
        out.breaks = np.asarray(model_dict["breaks"], dtype=float).reshape(-1)
        out.aliased = np.asarray(
            model_dict.get("aliased", []), dtype=int
        ).reshape(-1)
        names = model_dict.get("feature_names")
        out.feature_names = None if names is None else list(names)
        out.k = int(model_dict.get("k", out.params.size - out.aliased.size))
        neg = model_dict.get("_neg_ll")
        if neg is not None:
            out._neg_ll = float(neg)
        out._ic_n = cls._restored_ic_n(model_dict)
        cov = model_dict.get("covariance")
        out._cov_log = None if cov is None else np.asarray(cov, dtype=float)
        table = model_dict.get("rate_table")
        out.rate_table = None if table is None else RateTable.from_dict(table)
        out.maximum = restored_maximum(model_dict)
        return out


def search_space(
    model: ExcessHazard, data: dict
) -> tuple[Any, npt.NDArray, float, npt.NDArray]:
    """The negative log-likelihood of an excess hazard model's data in
    its natural log-rates and coefficients (of the columns kept), the
    model's point there, the number of observations and the per-component
    floors of a check of the maximum (each coefficient's covariate's
    unit). For an independent check of a fit's maximum."""
    d = check_data(
        data["x"],
        data.get("c"),
        data.get("n"),
        data["rate_table"],
        data["age"],
        data.get("year"),
        data.get("stratum"),
        Z=data["Z"],
    )
    assert d.Z is not None
    assert model.rate_table is not None
    kept = np.flatnonzero(np.isfinite(model.coef))
    problem = _Problem(model.rate_table, d, model.breaks, kept)
    K = model.n_intervals
    inverse = np.linalg.inv(problem.jacobian())

    def fun(u: Any) -> Any:
        return problem.nll(anp.dot(inverse, u))

    u = np.concatenate((np.log(model.rates), model.coef[kept]))
    floor = coefficient_floor(
        u.size, [(K + j, int(col)) for j, col in enumerate(kept)], d.Z
    )
    return fun, u, problem.n_obs, floor
