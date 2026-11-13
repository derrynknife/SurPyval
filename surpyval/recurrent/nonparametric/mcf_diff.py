"""
The comparison of two groups' mean cumulative functions (MCFs).

:func:`mcf_diff` takes two fitted
:class:`~surpyval.recurrent.NonParametricCounting` estimates, one per
group of items (two fleets, two designs, treated and untreated patients),
and answers two questions:

- how far apart are the groups' MCFs at each time, with a confidence
  interval (the difference of the MCFs and its Lawless-Nadeau robust
  variance);
- is there evidence, over the whole follow-up, that one group has events
  at a higher rate than the other (the pseudo-score tests of Cook, Lawless
  and Nadeau 1996)?

References
----------
Lawless, J. F. and Nadeau, C. (1995), "Some simple robust methods for the
analysis of recurrent events", Technometrics 37, 158-168.
Cook, R. J., Lawless, J. F. and Nadeau, C. (1996), "Robust tests for
treatment comparisons based on recurrent event responses", Biometrics 52,
557-571.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2, norm

from surpyval import plotting
from surpyval.recurrent.inference import check_alpha_ci
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.units import label_axes, same_unit
from surpyval.utils.validation import BOUNDS, check_option

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from surpyval.utils.event_history_data import EventHistoryData

    from .mcf import NonParametricCounting

#: The ``test_variance`` values of :func:`mcf_diff`.
TEST_VARIANCES = ("robust", "poisson")

#: The weights of the pseudo-score tests, in the order they are reported.
TEST_WEIGHTS = ("constant", "linear")


@dataclass
class MCFDiffResult:
    """
    The result of :func:`mcf_diff`: the difference between two groups'
    mean cumulative functions (group a minus group b) with its pointwise
    confidence interval, and the tests of whether the groups' event rates
    differ over the follow-up.

    Attributes
    ----------
    x : numpy.ndarray
        The times at which the difference changes: every observed time of
        either group, up to the end of the shorter follow-up.
    difference : numpy.ndarray
        The MCF of group a minus the MCF of group b at each ``x``: how
        many more events per item group a has had by then.
    var : numpy.ndarray
        The variance of ``difference``: the sum of the groups'
        Lawless-Nadeau robust variances (the groups are independent).
    origin : float
        Where both MCFs are defined from (the later of the two origins);
        the difference is 0 from there to the first ``x``.
    alpha_ci : float
        The significance level of the tests and one minus the confidence
        level of :meth:`diff_cb` (unless it is given another).
    test_variance : str
        The variance of the test statistics: ``"robust"`` (no assumption
        that the items share one Poisson process) or ``"poisson"``.
    tests : dict
        One entry per weight, ``"constant"`` and ``"linear"``, each a dict
        of ``score`` (the weighted sum of the differences of the groups'
        MCF increments), ``score_var`` (its variance), ``statistic`` (the
        chi-squared statistic ``score**2 / score_var``, one degree of
        freedom) and ``p_value``.
    statistic : float
        The constant-weight test's chi-squared statistic.
    p_value : float
        The constant-weight test's p-value.
    tau : float
        The last time both groups have items under observation; the tests
        use the times up to it.
    n_items : list of int
        The items in group a and in group b.
    n_events : list of int
        The events in group a and in group b.
    unit : str or None
        The time unit of the models, when they have one.

    Examples
    --------
    >>> from surpyval.recurrent import NonParametricCounting, mcf_diff
    >>> a = NonParametricCounting.fit(
    ...     [2, 5, 9, 10, 4, 10], i=[1, 1, 1, 1, 2, 2], c=[0, 0, 0, 1, 0, 1]
    ... )
    >>> b = NonParametricCounting.fit(
    ...     [6, 10, 3, 8, 10], i=[1, 1, 2, 2, 2], c=[0, 1, 0, 0, 1]
    ... )
    >>> result = mcf_diff(a, b)
    >>> result.diff([5, 10])
    array([1. , 0.5])
    >>> round(result.statistic, 4), round(result.p_value, 4)
    (0.4, 0.5271)
    """

    x: npt.NDArray
    difference: npt.NDArray
    var: npt.NDArray
    origin: float
    alpha_ci: float
    test_variance: str
    tests: dict
    statistic: float
    p_value: float
    tau: float
    n_items: list = field(default_factory=list)
    n_events: list = field(default_factory=list)
    unit: str | None = None

    @property
    def se(self) -> npt.NDArray:
        """The standard error of ``difference``, ``sqrt(var)``."""
        return np.sqrt(self.var)

    def _lookup(self, x: npt.ArrayLike, values: npt.NDArray) -> npt.NDArray:
        """``values`` as a right-continuous step function of time, 0 from
        the origin to the first time and NaN outside ``[origin, last]``
        (and at a NaN time)."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        idx = np.searchsorted(self.x, x, side="right") - 1
        out = np.asarray(values, dtype=float)[np.clip(idx, 0, None)]
        out = out.astype(float)
        out[idx < 0] = 0.0
        last = float(self.x.max()) if self.x.size else self.origin
        out[(x > last) | (x < self.origin) | np.isnan(x)] = np.nan
        return out

    @keeps_query_shape
    def diff(self, x: npt.ArrayLike) -> npt.NDArray:
        """
        The difference of the MCFs, group a minus group b, at ``x``.

        Parameters
        ----------
        x : array like
            The times at which to evaluate the difference.

        Returns
        -------
        numpy array
            The difference at each ``x``: a step function, 0 before the
            first observed time, and NaN before the origin and past the
            end of the shorter follow-up, where one of the MCFs is not
            estimated.
        """
        return self._lookup(x, self.difference)

    @keeps_query_shape
    def diff_cb(
        self,
        x: npt.ArrayLike,
        bound: str = "two-sided",
        *,
        alpha_ci: float | None = None,
    ) -> npt.NDArray:
        """
        Pointwise confidence bounds on the difference of the MCFs at ``x``:
        the difference plus or minus ``z`` standard errors (Wald bounds;
        a difference can be negative, so there is no log scale).

        Parameters
        ----------
        x : array like
            The times at which to compute the bounds.
        bound : str, optional
            ``"two-sided"`` (the default), ``"upper"`` or ``"lower"``.
        alpha_ci : float, optional
            The total tail probability of the bound(s): a two-sided
            ``1 - alpha_ci`` interval, or a one-sided bound exceeded with
            probability ``alpha_ci``. Defaults to the result's
            ``alpha_ci``. Keyword only.

        Returns
        -------
        numpy array
            Two-sided bounds as one row per query, ``[lower, upper]``;
            one-sided bounds as a 1-D array. 0 before the first observed
            time and NaN where :meth:`diff` is.

        Examples
        --------
        >>> from surpyval.recurrent import NonParametricCounting, mcf_diff
        >>> a = NonParametricCounting.fit(
        ...     [2, 5, 9, 10, 4, 10], i=[1, 1, 1, 1, 2, 2],
        ...     c=[0, 0, 0, 1, 0, 1],
        ... )
        >>> b = NonParametricCounting.fit(
        ...     [6, 10, 3, 8, 10], i=[1, 1, 2, 2, 2], c=[0, 1, 0, 0, 1]
        ... )
        >>> mcf_diff(a, b).diff_cb([5, 10]).round(4)
        array([[ 0.02  ,  1.98  ],
               [-1.0495,  2.0495]])
        """
        check_option("bound", bound, BOUNDS)
        if alpha_ci is None:
            alpha_ci = self.alpha_ci
        check_alpha_ci(alpha_ci)
        centre = self._lookup(x, self.difference)
        se = self._lookup(x, self.se)
        if bound == "two-sided":
            z = norm.ppf(1 - alpha_ci / 2)
            return np.column_stack([centre - z * se, centre + z * se])
        z = norm.ppf(1 - alpha_ci)
        return centre + z * se if bound == "upper" else centre - z * se

    def plot(
        self,
        *,
        alpha_ci: float | None = None,
        plot_bounds: bool = True,
        ax: Axes | None = None,
    ) -> Axes:
        """
        Plot the difference of the MCFs as a step function, with its
        pointwise confidence bounds and a line at 0 (no difference). The
        arguments are keyword only.

        Parameters
        ----------
        alpha_ci : float, optional
            The total tail probability of the two-sided bounds. Defaults
            to the result's ``alpha_ci``.
        plot_bounds : bool, optional
            Whether to draw the bounds. Defaults to :code:`True`.
        ax : matplotlib Axes, optional
            The axes to draw on. Defaults to the current axes.

        Returns
        -------
        matplotlib Axes
        """
        if alpha_ci is None:
            alpha_ci = self.alpha_ci
        check_alpha_ci(alpha_ci)
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gcf().gca()
        x = np.hstack([[self.origin], self.x])
        y = np.hstack([[0.0], self.difference])
        themed = plotting.theme_enabled()
        (line,) = ax.step(
            x,
            y,
            where="post",
            label="Difference in MCF",
            color=plotting.next_color(ax),
            linewidth=plotting.LINE_WIDTH if themed else None,
        )
        if plot_bounds:
            cb = self.diff_cb(x, alpha_ci=alpha_ci)
            label = f"{(1 - alpha_ci) * 100:g}% confidence bounds"
            if themed:
                ax.fill_between(
                    x,
                    cb[:, 0],
                    cb[:, 1],
                    step="post",
                    color=line.get_color(),
                    alpha=plotting.BAND_ALPHA,
                    linewidth=0,
                    label=label,
                )
            else:
                ax.step(x, cb, where="post", label=label, color="red")
        ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
        label_axes(ax, self.unit, "Difference in MCF (a - b)")
        plotting.style_axes(ax)
        return ax

    def to_dict(self) -> dict:
        """
        The result as a dict of native Python values, strict JSON: a
        missing or infinite number is written as ``None`` and recorded
        under ``"non_finite"`` (as every SurPyval ``to_dict`` does), so
        that :meth:`from_dict` restores it.

        Examples
        --------
        >>> import json
        >>> from surpyval.recurrent import (
        ...     MCFDiffResult,
        ...     NonParametricCounting,
        ...     mcf_diff,
        ... )
        >>> a = NonParametricCounting.fit([2, 5, 8], i=[1, 1, 1], c=[0, 0, 1])
        >>> b = NonParametricCounting.fit([4, 8], i=[1, 1], c=[0, 1])
        >>> d = mcf_diff(a, b).to_dict()
        >>> isinstance(json.dumps(d, allow_nan=False), str)
        True
        >>> MCFDiffResult.from_dict(d).diff([3, 6]).tolist()
        [1.0, 1.0]
        """
        from surpyval.serialisation import encode_non_finite

        out: dict = {"result": type(self).__name__}
        for f in fields(self):
            out[f.name] = _native(getattr(self, f.name))
        return encode_non_finite(out)

    @classmethod
    def from_dict(cls, result_dict: dict) -> MCFDiffResult:
        """
        The result :meth:`to_dict` wrote.

        See Also
        --------
        to_dict
        """
        from surpyval.serialisation import decode_non_finite

        if result_dict.get("result") != cls.__name__:
            raise ValueError(
                "Must create an MCFDiffResult from a dict its to_dict "
                "wrote (with 'result': 'MCFDiffResult')"
            )
        values = decode_non_finite(dict(result_dict))
        names = {f.name for f in fields(cls)}
        missing = sorted(names - set(values))
        if missing:
            raise ValueError(
                "the dict is missing {}".format(", ".join(missing))
            )
        kwargs = {name: values[name] for name in names}
        for name in ("x", "difference", "var"):
            kwargs[name] = np.asarray(kwargs[name], dtype=float)
        return cls(**kwargs)

    def __repr__(self) -> str:
        title = "Two-sample MCF comparison (a - b)"
        lines = [
            title,
            "=" * len(title),
            "Items            : {} in a, {} in b".format(*self.n_items),
            "Events           : {} in a, {} in b".format(*self.n_events),
        ]
        if self.x.size:
            last = float(self.x[-1])
            lines.append(
                "Difference       : {:.4g} at t = {:.6g} "
                "(SE {:.4g})".format(
                    float(self.difference[-1]),
                    last,
                    float(np.sqrt(self.var[-1])),
                )
            )
        lines.append(
            "Test variance    : {}".format(
                "robust (Lawless-Nadeau)"
                if self.test_variance == "robust"
                else "Poisson"
            )
        )
        for weight in TEST_WEIGHTS:
            test = self.tests[weight]
            lines.append(
                "{:<17s}: chi-squared {:.4g} (1 DoF), p-value {:.4g}".format(
                    weight.capitalize() + " weight",
                    test["statistic"],
                    test["p_value"],
                )
            )
        return "\n".join(lines)


def _native(value: Any) -> Any:
    """``value`` with numpy arrays and scalars made native, through dicts
    and lists."""
    if isinstance(value, np.ndarray):
        return [float(v) for v in value.ravel()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(v) for v in value]
    return value


def _step(
    times: npt.NDArray, values: npt.NDArray, x: npt.NDArray
) -> npt.NDArray:
    """A right-continuous step function through ``(times, values)`` at
    ``x``, 0 before the first time."""
    idx = np.searchsorted(times, x, side="right") - 1
    out = np.asarray(values, dtype=float)[np.clip(idx, 0, None)]
    return np.where(idx < 0, 0.0, out)


def _group_data(model: Any, name: str) -> EventHistoryData:
    """The data ``model`` was fitted to, or a plain refusal."""
    from .mcf import NonParametricCounting

    # NonParametricCounting is the fitter instance; its class is the type
    # of every fitted estimate.
    if not isinstance(model, type(NonParametricCounting)) or not hasattr(
        model, "mcf_hat"
    ):
        raise TypeError(
            "'{}' must be a fitted NonParametricCounting estimate (the "
            "result of NonParametricCounting.fit on one group's data); "
            "got {!r}.".format(name, type(model).__name__)
        )
    data = getattr(model, "data", None)
    if data is None or model.var is None:
        raise ValueError(
            "'{}' carries no per-item data (it was built by from_xrd, "
            "from_dict or a simulation), so neither the robust variance "
            "of the difference nor the tests can be computed. Fit it with "
            "NonParametricCounting.fit(x, i=i, c=c) on that group's "
            "data.".format(name)
        )
    return data


class _Group:
    """One group's at-risk counts, events and per-item residuals on the
    test grid."""

    def __init__(self, data: EventHistoryData, grid: npt.NDArray) -> None:
        self.data = data
        self.grid = grid
        self.entry, self.exit = data.item_observation_windows()
        # Items split into observation windows are regrouped under their
        # original item, one cluster each, as the MCF's own variance is.
        window_map = getattr(data, "window_map", None) or {}
        keys: dict = {}
        self.cluster = np.array(
            [
                keys.setdefault(
                    window_map[item][0] if item in window_map else item,
                    len(keys),
                )
                for item in data.ids
            ],
            dtype=np.intp,
        )
        self.n_clusters = len(keys)
        # At risk at t: entry < t <= exit.
        entries = np.sort(self.entry)
        exits = np.sort(self.exit)
        self.r = np.searchsorted(entries, grid, side="left") - np.searchsorted(
            exits, grid, side="left"
        )
        is_event = np.asarray(data.c) == 0
        x = np.asarray(data.x, dtype=float)
        col = np.searchsorted(grid, x)
        on_grid = (col < grid.size) & (
            grid[np.minimum(col, max(grid.size - 1, 0))] == x
        )
        self.ev = is_event & on_grid
        self.ev_col = col[self.ev]
        self.ev_n = np.asarray(data.n, dtype=float)[self.ev]
        self.ev_item = data.item_rows()[1][self.ev]
        self.d = np.bincount(self.ev_col, self.ev_n, minlength=grid.size)
        self.n_events = int(np.sum(np.asarray(data.n)[is_event]))
        self.dL = np.where(
            self.r > 0, self.d / np.where(self.r > 0, self.r, 1), 0
        )

    def robust_var(self, w: npt.NDArray) -> float:
        """The robust variance of this group's part of the score
        ``sum_j w_j dL_j``: the sum over items of the square of
        ``sum_j w_j delta_i(t_j) / r(t_j) (n_i(t_j) - dL(t_j))``."""
        inv_r = 1.0 / self.r
        cum = np.r_[0.0, np.cumsum(w * self.dL * inv_r)]
        # The grid times in each item's window (entry, exit].
        lo = np.searchsorted(self.grid, self.entry, side="right")
        hi = np.searchsorted(self.grid, self.exit, side="right")
        expected = cum[hi] - cum[np.minimum(lo, hi)]
        observed = np.bincount(
            self.ev_item,
            self.ev_n * w[self.ev_col] * inv_r[self.ev_col],
            minlength=self.entry.size,
        )
        per_cluster = np.bincount(
            self.cluster, observed - expected, minlength=self.n_clusters
        )
        return float(np.sum(per_cluster**2))


def mcf_diff(
    model_a: NonParametricCounting,
    model_b: NonParametricCounting,
    alpha_ci: float = 0.05,
    test_variance: str = "robust",
) -> MCFDiffResult:
    r"""
    Compare the mean cumulative functions (MCFs) of two groups of items:
    the difference between them at each time, with a confidence interval,
    and a test of whether one group has events at a higher rate than the
    other over the follow-up.

    The MCF of a group is the average number of events per item by time
    :math:`t`. Fit one :class:`NonParametricCounting` per group (two fleets
    of pumps, two maintenance policies, treated and untreated patients)
    and pass both. The difference :math:`\hat M_a(t) - \hat M_b(t)` says
    how many more events per item group a has had by :math:`t`; its
    variance is the sum of the groups' Lawless-Nadeau robust variances,
    which do not assume the items share one Poisson process, and its
    pointwise interval is the difference plus or minus :math:`z` standard
    errors. ``intervention_test`` compares the same systems before and
    after a change; this compares two separate groups.

    The tests (Cook, Lawless and Nadeau 1996, the pseudo-score tests of
    R's ``reda::mcfDiff.test``) add up, over the times where both groups
    have items under observation, the differences between the groups' MCF
    increments, each weighted by how much information that time carries:

    .. math::
        U = \sum_j w(t_j) \big(d\hat M_a(t_j) - d\hat M_b(t_j)\big),
        \qquad
        w(t) = \frac{r_a(t)\, r_b(t)}{r_a(t) + r_b(t)},

    with :math:`r_a, r_b` the items under observation. The constant
    weight treats every time alike; the linear weight,
    :math:`w(t) (\tau - t)/\tau` with :math:`\tau` the last time both
    groups are observed, gives early times more say, which compares the
    average gap between the curves over the follow-up rather than the gap
    at its end. :math:`U^2 / \widehat{Var}(U)` is chi-squared with one
    degree of freedom when the groups' rates are the same. The robust
    variance of :math:`U` sums, over the items of both groups, the square
    of each item's weighted deviations from its group's increments; the
    Poisson variance, :math:`\sum_j w_j^2 (d\hat M_a/r_a + d\hat M_b/r_b)`,
    is right only when every item follows the same Poisson process.

    Parameters
    ----------
    model_a : NonParametricCounting
        Group a's MCF, fitted with ``NonParametricCounting.fit`` (it keeps
        the per-item data the robust variance needs).
    model_b : NonParametricCounting
        Group b's MCF, fitted the same way.
    alpha_ci : float, optional
        The significance level of the tests and one minus the confidence
        level of the pointwise interval. Defaults to 0.05.
    test_variance : str, optional
        The variance of the test statistics: ``"robust"`` (the default)
        or ``"poisson"``.

    Returns
    -------
    MCFDiffResult
        The difference (``diff``, ``diff_cb``, ``plot``) and the tests
        (``tests``, ``statistic``, ``p_value``).

    Raises
    ------
    TypeError
        If a model is not a fitted ``NonParametricCounting``.
    ValueError
        If a model has no per-item data (built by ``from_xrd``,
        ``from_dict`` or a simulation), ``test_variance`` is not one of its
        values, or the groups are never under observation at the same
        time.

    Examples
    --------
    Two groups of two systems each, observed to t = 10:

    >>> from surpyval.recurrent import NonParametricCounting, mcf_diff
    >>> a = NonParametricCounting.fit(
    ...     [2, 5, 9, 10, 4, 10], i=[1, 1, 1, 1, 2, 2], c=[0, 0, 0, 1, 0, 1]
    ... )
    >>> b = NonParametricCounting.fit(
    ...     [6, 10, 3, 8, 10], i=[1, 1, 2, 2, 2], c=[0, 1, 0, 0, 1]
    ... )
    >>> result = mcf_diff(a, b)
    >>> result.diff([3, 5, 10])
    array([0. , 1. , 0.5])
    >>> round(result.tests["linear"]["p_value"], 4)
    0.294

    Notes
    -----
    The difference is estimated from the later of the two origins to the
    end of the shorter follow-up; beyond it one MCF is not estimated, and
    ``diff`` is NaN there (R's ``reda::mcfDiff`` carries the shorter
    group's MCF at its last value instead). It agrees with
    ``reda::mcfDiff`` and the tests with ``reda::mcfDiff.test`` (both
    variances) to rounding on the times both estimate.

    The robust tests rest on large-sample theory. With few items per group
    they reject a true null somewhat more often than ``alpha_ci``: in a
    simulation with strongly varying rates (an exponential frailty), 20
    items per group rejected 7-8% of the time at a 5% level, and 80 items
    per group 5%.

    References
    ----------
    Lawless, J. F. and Nadeau, C. (1995), "Some simple robust methods for
    the analysis of recurrent events", Technometrics 37, 158-168.

    Cook, R. J., Lawless, J. F. and Nadeau, C. (1996), "Robust tests for
    treatment comparisons based on recurrent event responses", Biometrics
    52, 557-571.
    """
    check_alpha_ci(alpha_ci)
    check_option("test_variance", test_variance, TEST_VARIANCES)
    data_a = _group_data(model_a, "model_a")
    data_b = _group_data(model_b, "model_b")

    # The difference: on every time of either group, from the later origin
    # to the end of the shorter follow-up.
    origin = max(model_a._origin(), model_b._origin())
    last = min(float(model_a.x.max()), float(model_b.x.max()))
    x = np.union1d(model_a.x, model_b.x)
    x = x[(x >= origin) & (x <= last)]
    difference = _step(model_a.x, model_a.mcf_hat, x) - _step(
        model_b.x, model_b.mcf_hat, x
    )
    # Both variances are there (_group_data refuses a model without).
    var = _step(model_a.x, np.asarray(model_a.var), x) + _step(
        model_b.x, np.asarray(model_b.var), x
    )

    # The tests: on the times both groups have items at risk.
    grid = np.union1d(model_a.x, model_b.x)
    a = _Group(data_a, grid)
    b = _Group(data_b, grid)
    both = (a.r > 0) & (b.r > 0)
    if not both.any():
        raise ValueError(
            "The two groups never have items under observation at the "
            "same time, so their MCFs cannot be compared. Check that both "
            "models use the same time scale (and origin)."
        )
    grid = grid[both]
    a, b = _Group(data_a, grid), _Group(data_b, grid)
    tau = float(grid.max())
    w_constant = a.r * b.r / (a.r + b.r)
    weights = {
        "constant": w_constant,
        # tau = 0 (every common time at 0) leaves nothing to weight.
        "linear": w_constant * ((tau - grid) / tau if tau != 0 else 0.0),
    }
    tests = {}
    for name in TEST_WEIGHTS:
        w = weights[name]
        score = float(np.sum(w * (a.dL - b.dL)))
        if test_variance == "robust":
            score_var = a.robust_var(w) + b.robust_var(w)
        else:
            score_var = float(np.sum(w**2 * (a.dL / a.r + b.dL / b.r)))
        if score_var > 0:
            statistic = score**2 / score_var
            p_value = float(chi2.sf(statistic, 1))
        else:
            # No events at a common time: nothing to compare.
            statistic, p_value = 0.0, 1.0
        tests[name] = {
            "score": score,
            "score_var": score_var,
            "statistic": float(statistic),
            "p_value": p_value,
        }
    unit_a = getattr(model_a, "unit", None)
    unit_b = getattr(model_b, "unit", None)
    return MCFDiffResult(
        x=x,
        difference=difference,
        var=var,
        origin=float(origin),
        alpha_ci=float(alpha_ci),
        test_variance=test_variance,
        tests=tests,
        statistic=tests["constant"]["statistic"],
        p_value=tests["constant"]["p_value"],
        tau=tau,
        n_items=[len(set(a.cluster.tolist())), len(set(b.cluster.tolist()))],
        n_events=[a.n_events, b.n_events],
        unit=unit_a if same_unit(unit_a, unit_b) else None,
    )
