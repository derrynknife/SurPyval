r"""Net survival: survival from the event of interest alone, beyond what
the population's own hazard explains.

When the cause of each event is not recorded (deaths in a cancer
registry, where the death certificate is unreliable), the hazard of the
units studied is split into two parts: the hazard the population they
come from would have anyway, :math:`\lambda_{P,i}(t)`, read from a rate
table for the unit's age, calendar year and stratum, and an *excess*
hazard :math:`\lambda_E(t)` due to the condition studied:

.. math::
    \lambda_i(t) = \lambda_{P,i}(t) + \lambda_E(t).

*Net survival* :math:`S_E(t) = \exp(-\Lambda_E(t))` is the survival the
units would have if the excess hazard were the only one: what is left of
survival once the background mortality is taken away. Both estimators
here estimate the cumulative excess hazard :math:`\Lambda_E` as an
observed cumulative hazard less an expected one, over the distinct
follow-up times :math:`u_k`, with :math:`Y_i(t)` unit :math:`i`'s at-risk
indicator and :math:`\Lambda_{P,i}` its expected cumulative hazard
(:meth:`surpyval.RateTable.Hf`).

**Ederer II** (the default of relative survival life tables; Ederer and
Heise, 1959, in its continuous form): the Nelson-Aalen estimate less the
average expected hazard of the units still at risk,

.. math::
    \hat\Lambda_E(t) = \sum_{u_k \le t} \frac{d_k}{Y(u_k)}
    - \int_0^t \frac{\sum_i Y_i(s) \lambda_{P,i}(s)}{Y(s)}\, ds .

It is the *relative survival ratio*, observed over expected survival in
the units still followed. When the excess hazard depends on age (or on
anything the population hazard depends on), the units that survive drift
towards those with lower population hazards, and Ederer II no longer
estimates net survival.

**Pohar Perme** (Pohar Perme, Stare and Estève, 2012), the default:
each unit weighted by the inverse of its expected survival,
:math:`w_i(t) = 1 / S_{P,i}(t)`, which puts back, in expectation, the
units the population hazard removed, so that the risk set keeps the
make-up of the units at time 0:

.. math::
    \hat\Lambda_E(t) = \sum_{u_k \le t}
    \frac{\sum_{i: \text{event at } u_k} w_i(u_k)}{Y^w(u_k)}
    - \int_0^t \frac{\sum_i Y_i(s) w_i(s) \lambda_{P,i}(s)}{Y^w(s)}\,ds,
    \qquad Y^w(t) = \sum_i Y_i(t) w_i(t) .

It is a consistent estimator of net survival whatever the excess hazard
depends on, provided the population hazard is that of the units' own
population. Its weights grow with follow-up, so it is noisier late in
follow-up, among old units most.

The expected parts are integrated exactly over each stretch between the
distinct times (the Pohar Perme one is :math:`\log Y^w`'s increase there,
since :math:`dw_i/ds = w_i \lambda_{P,i}`). The variances are those of
the observed parts, :math:`\sum d_k / Y(u_k)^2` for Ederer II and
:math:`\sum_{\text{events}} w_i^2 / Y^w(u_k)^2` for Pohar Perme, the
expected parts being taken as known (Pohar Perme et al., 2012).

Net survival is a difference of hazards, not a probability of a set of
units: it can rise (where fewer units fail than the population rates
expect) and exceed 1. It is reported as it is, not cut.

References
----------
Ederer, F. and Heise, H. (1959). Instructions to IBM 650 programmers in
processing survival computations. Methodological note 10, End Results
Evaluation Section, National Cancer Institute.

Pohar Perme, M., Stare, J. and Estève, J. (2012). On estimation in
relative survival. *Biometrics* 68, 113-120.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.surpyval_data import refuses_surpyval_data_with
from surpyval.utils.units import takes_unit
from surpyval.utils.validation import BOUNDS, check_alpha_ci, check_option

from ._data import REFUSAL, check_data, expected_on_grid
from .rate_table import RateTable

#: The estimators of net survival.
METHODS = ("pohar-perme", "ederer2")
_NAMES = {"pohar-perme": "Pohar Perme", "ederer2": "Ederer II"}

#: The scales a pointwise bound is formed on.
BOUND_TYPES = ("log", "normal")

#: The functions a bound is on.
_CB_ON = ("sf", "R", "ff", "F", "Hf")

_refuses_data = refuses_surpyval_data_with(REFUSAL)


def _estimate(
    table: RateTable,
    x: Any,
    c: Any,
    n: Any,
    age: Any,
    year: Any,
    stratum: Any,
    method: str,
) -> dict[str, npt.NDArray]:
    """The increments of the observed and expected cumulative hazards and
    of the variance at each distinct time."""
    data = check_data(x, c, n, table, age, year, stratum)
    grid = np.unique(data.x[data.x > 0])
    K = grid.size
    pos = np.searchsorted(grid, data.x)
    followed = data.x > 0
    event = data.c == 0
    leaving = np.bincount(pos[followed], weights=data.n[followed], minlength=K)
    at_risk = np.cumsum(leaving[::-1])[::-1]
    events = np.bincount(pos[event], weights=data.n[event], minlength=K)
    if method == "ederer2":
        spent = np.zeros(K)
        for i, m, H in expected_on_grid(table, data, grid):
            spent[:m] += data.n[i] * np.diff(H, prepend=0.0)
        observed = events / at_risk
        expected = spent / at_risk
        variance = events / at_risk**2
    else:
        end = np.zeros(K)
        start = np.zeros(K)
        weighted = np.zeros(K)
        squared = np.zeros(K)
        for i, m, H in expected_on_grid(table, data, grid):
            if m == 0:
                continue
            w = np.exp(H)
            end[:m] += data.n[i] * w
            start[:m] += data.n[i] * np.exp(np.concatenate(([0.0], H[:-1])))
            if event[i]:
                weighted[m - 1] += data.n[i] * w[-1]
                squared[m - 1] += data.n[i] * w[-1] ** 2
        observed = weighted / end
        expected = np.log(end) - np.log(start)
        variance = squared / end**2
    return {
        "x": grid,
        "at_risk": at_risk,
        "events": events,
        "observed": observed,
        "expected": expected,
        "variance": variance,
        "n_units": np.array(float(data.n.sum())),
    }


class RelativeSurvival(SerialisableMixin):
    r"""
    Net survival from relative survival data: the survival of the units
    studied once the hazard of their background population is taken away,
    estimated by Pohar Perme's estimator (the default) or Ederer II
    (see the module :mod:`surpyval.univariate.relative.net_survival` for
    both).

    The expected hazards come from a :class:`surpyval.RateTable` of the
    population's rates by attained age (and calendar year and stratum):
    each unit's age, year and stratum at time 0 place it in the table, and
    its age and the calendar advance with its follow-up.

    ``sf`` is the net survival :math:`S_E(t) = \exp(-\Lambda_E(t))`, ``ff``
    its complement and ``Hf`` the cumulative excess hazard
    :math:`\Lambda_E`. The observed part steps at the event times and the
    expected part changes continuously; between two follow-up times the
    expected part is taken as linear in time (exact where the rates of the
    units at risk do not change between them). Before time 0 net survival
    is 1, and after the last follow-up time it holds its last value.

    Net survival is not a probability of a set of units: where fewer
    units fail than the population rates expect it rises, and it can
    exceed 1. It is reported as it is.

    Examples
    --------
    Simulated patients aged 50 to 80, with population rates rising 9% a
    year of age and an excess hazard of 0.1 a year:

    >>> import numpy as np
    >>> import surpyval as sp
    >>> rng = np.random.default_rng(3)
    >>> ages = np.arange(0, 121)
    >>> table = sp.RateTable(5e-5 * np.exp(0.09 * ages), ages=ages)
    >>> age = rng.uniform(50, 80, 2000)
    >>> excess = rng.exponential(10, 2000)
    >>> u = rng.uniform(size=2000)
    >>> # the population time to death: invert the expected survival
    >>> grid = np.linspace(0, 60, 601)
    >>> background = np.array([np.interp(-np.log(v), table.Hf(grid, a), grid)
    ...                        for v, a in zip(u, age)])
    >>> time = np.minimum(np.minimum(excess, background), 10.0)
    >>> c = (time == 10.0).astype(int)
    >>> model = sp.RelativeSurvival.fit(time, c, rate_table=table, age=age)
    >>> model.sf([1, 5]).round(3)  # the truth is exp(-0.1 t): 0.905, 0.607
    array([0.896, 0.591])
    """

    method: str
    x: npt.NDArray
    at_risk: npt.NDArray
    events: npt.NDArray
    n_units: float
    _observed: npt.NDArray
    _expected: npt.NDArray
    _variance: npt.NDArray

    # -- fitting ----------
    @classmethod
    @takes_unit
    @_refuses_data
    def fit(
        cls,
        x: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        rate_table: RateTable | None = None,
        age: npt.ArrayLike | None = None,
        year: npt.ArrayLike | None = None,
        stratum: Any = None,
        method: str = "pohar-perme",
    ) -> RelativeSurvival:
        r"""
        Estimate net survival.

        Parameters
        ----------
        x : array_like
            Each unit's follow-up time from time 0 (diagnosis,
            installation), in the rate table's unit.
        c : array_like, optional
            0 where the unit's follow-up ended in the event (of any cause:
            the cause need not be known), 1 where it was right censored.
            Defaults to all 0.
        n : array_like, optional
            Counts of identical units. Defaults to 1 per row.
        rate_table : RateTable
            The population's hazards (required).
        age : array_like
            Each unit's age at time 0, in the table's unit (required).
        year : array_like, optional
            Each unit's calendar time at time 0 (e.g. 2011.5), where the
            table is by year.
        stratum : array_like, optional
            Each unit's stratum of the table (e.g. its sex), where the
            table has strata: one label, or one per row.
        method : {"pohar-perme", "ederer2"}, optional
            The estimator. Defaults to ``"pohar-perme"``, which estimates
            net survival whatever the excess hazard depends on; Ederer II
            is the relative survival ratio of the units still followed.

        Returns
        -------
        RelativeSurvival

        A row with a missing age, year or stratum is dropped, with one
        warning; a missing time is refused.

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.01, 0.03], ages=[0, 70])
        >>> model = sp.RelativeSurvival.fit(
        ...     [2, 3, 5, 6, 8], c=[0, 0, 1, 0, 1], rate_table=table,
        ...     age=[60, 64, 66, 70, 75], method="ederer2")
        >>> model.Hf([2, 3]).round(4)
        array([0.164, 0.394])
        """
        check_option("method", method, METHODS)
        if rate_table is None:
            raise ValueError(
                "'rate_table' is required: the population's hazards, a "
                "surpyval.RateTable."
            )
        parts = _estimate(rate_table, x, c, n, age, year, stratum, method)
        return cls._build(method, parts)

    @classmethod
    def _build(cls, method: str, parts: dict) -> RelativeSurvival:
        out = cls.__new__(cls)
        out.method = method
        out.x = np.asarray(parts["x"], dtype=float)
        out.at_risk = np.asarray(parts["at_risk"], dtype=float)
        out.events = np.asarray(parts["events"], dtype=float)
        out.n_units = float(parts["n_units"])
        out._observed = np.cumsum(np.asarray(parts["observed"], float))
        out._expected = np.cumsum(np.asarray(parts["expected"], float))
        out._variance = np.cumsum(np.asarray(parts["variance"], float))
        return out

    @classmethod
    @takes_unit
    def fit_from_df(
        cls,
        df: Any,
        x_col: str,
        age_col: str,
        rate_table: RateTable,
        c_col: str | None = None,
        n_col: str | None = None,
        year_col: str | None = None,
        stratum_col: str | None = None,
        method: str = "pohar-perme",
    ) -> RelativeSurvival:
        """
        Estimate net survival from the columns of a DataFrame, one row per
        unit (or per group of identical units, with ``n_col``): the
        columns are the arguments of :meth:`fit`.

        Examples
        --------
        >>> import pandas as pd
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.01, 0.03], ages=[0, 70])
        >>> df = pd.DataFrame({"years": [2, 3, 5, 6, 8],
        ...                    "censored": [0, 0, 1, 0, 1],
        ...                    "age": [60, 64, 66, 70, 75]})
        >>> model = sp.RelativeSurvival.fit_from_df(
        ...     df, "years", "age", table, c_col="censored")
        >>> model.sf(3).round(4)
        np.float64(0.6819)
        """
        from surpyval.utils.dataframe import check_columns, frame_column

        check_columns(
            df,
            x_col=x_col,
            age_col=age_col,
            c_col=c_col,
            n_col=n_col,
            year_col=year_col,
            stratum_col=stratum_col,
        )

        def column(name: str | None, arg: str) -> Any:
            return None if name is None else frame_column(df, name, arg)

        return cls.fit(
            frame_column(df, x_col, "x_col", time=True),
            c=column(c_col, "c_col"),
            n=column(n_col, "n_col"),
            rate_table=rate_table,
            age=column(age_col, "age_col"),
            year=column(year_col, "year_col"),
            stratum=column(stratum_col, "stratum_col"),
            method=method,
        )

    # -- the estimate ----------
    def _parts(
        self, t: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """The cumulative excess hazard and its variance at the times
        ``t`` (flat), and where ``t`` is missing."""
        t = np.asarray(t, dtype=float)
        missing = np.isnan(t)
        safe = np.where(missing, 0.0, t)
        idx = np.searchsorted(self.x, safe, side="right") - 1
        held = np.clip(idx, 0, None)
        observed = np.where(idx < 0, 0.0, self._observed[held])
        variance = np.where(idx < 0, 0.0, self._variance[held])
        expected = np.interp(
            safe,
            np.concatenate(([0.0], self.x)),
            np.concatenate(([0.0], self._expected)),
            left=0.0,
        )
        H = observed - expected
        H = np.where(missing, np.nan, H)
        variance = np.where(missing, np.nan, variance)
        return H, variance, missing

    @keeps_query_shape
    def Hf(self, x: npt.ArrayLike) -> npt.NDArray:
        r"""
        The cumulative excess hazard :math:`\hat\Lambda_E` at ``x``: the
        observed cumulative hazard less the expected one (see the class).
        It falls where fewer units fail than the population rates expect.

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.0], ages=[0])  # no background hazard
        >>> model = sp.RelativeSurvival.fit([1, 2, 3, 4], rate_table=table,
        ...                                 age=50)
        >>> model.Hf([1, 2]).round(4)  # the Nelson-Aalen estimate
        array([0.25  , 0.5833])
        """
        return self._parts(np.asarray(x, dtype=float))[0]

    @keeps_query_shape
    def sf(self, x: npt.ArrayLike) -> npt.NDArray:
        r"""
        Net survival :math:`\exp(-\hat\Lambda_E(x))` at ``x``. It can rise,
        and exceed 1, where fewer units fail than the population rates
        expect.

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.01, 0.03], ages=[0, 70])
        >>> model = sp.RelativeSurvival.fit(
        ...     [2, 3, 5, 6, 8], c=[0, 0, 1, 0, 1], rate_table=table,
        ...     age=[60, 64, 66, 70, 75])
        >>> model.sf([0, 2, 3]).round(4)
        array([1.    , 0.8516, 0.6819])
        """
        return np.exp(-self._parts(np.asarray(x, dtype=float))[0])

    @keeps_query_shape
    def ff(self, x: npt.ArrayLike) -> npt.NDArray:
        """
        The net probability of the event by ``x``, ``1 - sf(x)``.

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.01, 0.03], ages=[0, 70])
        >>> model = sp.RelativeSurvival.fit(
        ...     [2, 3, 5, 6, 8], c=[0, 0, 1, 0, 1], rate_table=table,
        ...     age=[60, 64, 66, 70, 75])
        >>> model.ff(3).round(4)
        np.float64(0.3181)
        """
        return -np.expm1(-self._parts(np.asarray(x, dtype=float))[0])

    # -- uncertainty ----------
    @keeps_query_shape
    def cb(
        self,
        x: npt.ArrayLike,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        bound_type: str = "log",
    ) -> npt.NDArray:
        r"""
        Pointwise confidence bounds on net survival (or on ``ff`` or
        ``Hf``) at ``x``.

        Parameters
        ----------
        x : array_like
            The times.
        on : {"sf", "ff", "Hf"}, optional
            The function bounded (``"R"`` and ``"F"`` are ``"sf"`` and
            ``"ff"``). Defaults to ``"sf"``.
        alpha_ci : float, optional
            The significance level: 0.05 (the default) gives a 95%
            two-sided interval.
        bound : {"two-sided", "lower", "upper"}, optional
            Defaults to ``"two-sided"`` (``[lower, upper]`` on the last
            axis). A one-sided bound puts all of ``alpha_ci`` on one side.
        bound_type : {"log", "normal"}, optional
            ``"log"`` (the default, as R's relsurv): a normal interval for
            the cumulative excess hazard, :math:`\hat\Lambda_E \pm z
            \hat\sigma`, so :math:`\hat S_E e^{\pm z \hat\sigma}` on net
            survival. ``"normal"``: :math:`\hat S_E \pm z \hat S_E
            \hat\sigma` on net survival itself. Neither is cut at 1: net
            survival can exceed it. (The log(-log) scale of the
            Kaplan-Meier bounds is not offered: the cumulative excess
            hazard can be 0 or below.) The bounds on ``ff`` are one
            minus those on ``sf``, and those on ``Hf`` minus their
            logarithm, so a ``"normal"`` lower bound on survival below 0
            gives an upper bound on ``Hf`` of NaN.

        Returns
        -------
        numpy.ndarray
            The bounds: the query's shape, with a last axis of
            ``[lower, upper]`` when two-sided.

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.01, 0.03], ages=[0, 70])
        >>> model = sp.RelativeSurvival.fit(
        ...     [2, 3, 5, 6, 8], c=[0, 0, 1, 0, 1], rate_table=table,
        ...     age=[60, 64, 66, 70, 75])
        >>> model.cb(3).round(3)
        array([0.37 , 1.258])
        """
        check_alpha_ci(alpha_ci)
        check_option("on", on, _CB_ON)
        check_option("bound", bound, BOUNDS)
        check_option("bound_type", bound_type, BOUND_TYPES)
        on = {"R": "sf", "F": "ff"}.get(on, on)
        H, variance, _ = self._parts(np.asarray(x, dtype=float))
        se = np.sqrt(variance)
        alpha = alpha_ci / 2 if bound == "two-sided" else alpha_ci
        z = norm.ppf(1 - alpha)
        S = np.exp(-H)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if bound_type == "log":
                S_low, S_high = S * np.exp(-z * se), S * np.exp(z * se)
            else:
                S_low, S_high = S - z * S * se, S + z * S * se
            if on == "sf":
                low, high = S_low, S_high
            elif on == "ff":
                low, high = 1.0 - S_high, 1.0 - S_low
            else:
                low, high = -np.log(S_high), -np.log(S_low)
        if bound == "lower":
            return low
        if bound == "upper":
            return high
        return np.stack([low, high], axis=-1)

    def standard_errors(self, x: npt.ArrayLike) -> Any:
        """
        The standard error of the cumulative excess hazard
        :meth:`Hf` at ``x`` (that of its observed part).

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.0], ages=[0])
        >>> model = sp.RelativeSurvival.fit([1, 2, 3, 4], rate_table=table,
        ...                                 age=50)
        >>> model.standard_errors([1, 2]).round(4)
        array([0.25  , 0.4167])
        """
        t = np.asarray(x, dtype=float)
        out = np.sqrt(self._parts(t.ravel())[1]).reshape(t.shape)
        return out[()] if out.ndim == 0 else out

    # -- reading it ----------
    @property
    def table(self) -> Any:
        """
        The estimate at each distinct follow-up time, as a DataFrame: the
        number at risk and of events there, the increments of the
        observed and expected cumulative hazards, the cumulative excess
        hazard ``Hf``, its standard error and net survival ``sf``.

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.0], ages=[0])
        >>> model = sp.RelativeSurvival.fit([1, 2, 2, 4], rate_table=table,
        ...                                 age=50)
        >>> model.table[["at_risk", "events", "Hf"]].round(4)
           at_risk  events      Hf
        x
        1.0    4.0     1.0  0.2500
        2.0    3.0     2.0  0.9167
        4.0    1.0     1.0  1.9167
        """
        import pandas as pd

        H = self._observed - self._expected
        return pd.DataFrame(
            {
                "at_risk": self.at_risk,
                "events": self.events,
                "observed": np.diff(self._observed, prepend=0.0),
                "expected": np.diff(self._expected, prepend=0.0),
                "Hf": H,
                "se(Hf)": np.sqrt(self._variance),
                "sf": np.exp(-H),
            },
            index=pd.Index(self.x, name="x"),
        )

    def __repr__(self) -> str:
        title = "Relative Survival SurPyval Model"
        last = float(self.x[-1])
        lines = [
            title,
            "=" * len(title),
            f"Estimator    : {_NAMES[self.method]} (net survival)",
            f"Data         : {self.n_units:g} units, "
            f"{self.events.sum():g} events at "
            f"{int(np.count_nonzero(self.events))} times",
            f"Follow-up    : to {last:g}",
            f"Net survival : {float(self.sf(last)):.4g} at {last:g}",
        ]
        return "\n".join(lines)

    # -- saving ----------
    def to_dict(self) -> dict:
        """
        The estimate as a plain, JSON-ready ``dict``;
        ``surpyval.from_dict`` rebuilds it.

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.01, 0.03], ages=[0, 70])
        >>> model = sp.RelativeSurvival.fit(
        ...     [2, 3, 5, 6, 8], c=[0, 0, 1, 0, 1], rate_table=table,
        ...     age=[60, 64, 66, 70, 75])
        >>> restored = sp.from_dict(model.to_dict())
        >>> bool(restored.sf(3) == model.sf(3))
        True
        """
        out = {
            "model": "RelativeSurvival",
            "method": self.method,
            "x": self.x.tolist(),
            "at_risk": self.at_risk.tolist(),
            "events": self.events.tolist(),
            "n_units": float(self.n_units),
            "observed": np.diff(self._observed, prepend=0.0).tolist(),
            "expected": np.diff(self._expected, prepend=0.0).tolist(),
            "variance": np.diff(self._variance, prepend=0.0).tolist(),
        }
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> RelativeSurvival:
        """
        Rebuild an estimate from :meth:`to_dict`'s dictionary.

        Examples
        --------
        >>> import surpyval as sp
        >>> table = sp.RateTable([0.01, 0.03], ages=[0, 70])
        >>> model = sp.RelativeSurvival.fit(
        ...     [2, 3, 5, 6, 8], c=[0, 0, 1, 0, 1], rate_table=table,
        ...     age=[60, 64, 66, 70, 75])
        >>> sp.RelativeSurvival.from_dict(model.to_dict()).method
        'pohar-perme'
        """
        require_model_tag(
            model_dict, "RelativeSurvival", "a relative survival estimate"
        )
        method = str(model_dict["method"])
        check_option("method", method, METHODS)
        parts = {
            key: np.asarray(model_dict[key], dtype=float)
            for key in (
                "x",
                "at_risk",
                "events",
                "observed",
                "expected",
                "variance",
            )
        }
        parts["n_units"] = np.asarray(model_dict["n_units"], dtype=float)
        return cls._build(method, parts)
