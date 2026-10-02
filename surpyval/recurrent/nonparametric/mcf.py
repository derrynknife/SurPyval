from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.univariate.nonparametric.nonparametric import (
    _BOUNDS,
    _check_option,
    _check_support,
    _on_support,
    _support_from_dict,
)
from surpyval.utils.dataframe import RecurrentDataFrameMixin
from surpyval.utils.fitter import singleton_fitter
from surpyval.utils.recurrent_event_data import RecurrentEventData
from surpyval.utils.recurrent_utils import (
    handle_xicn,
    reject_unsupported_nonparametric,
)
from surpyval.utils.shapes import keeps_query_shape

if TYPE_CHECKING:
    from matplotlib.axes import Axes


# The ``interp`` values of the MCF and its bounds.
_MCF_INTERP = ("step", "linear")

# How ``set_support`` names the range an MCF's bounds must contain.
_MCF_RANGE = (
    "the origin of the MCF, where observation begins",
    "the last observed time",
)


@singleton_fitter
class NonParametricCounting(RecurrentDataFrameMixin, SerialisableMixin):
    """
    The non-parametric (Nelson-Aalen) estimate of the mean cumulative
    function (MCF), the expected number of events per item by time
    :math:`t`:

    .. math::
        \\hat{M}(t) = \\sum_{t_j \\le t} \\frac{d_j}{r_j},

    with :math:`d_j` the events at :math:`t_j` and :math:`r_j` the number of
    items under observation then. Its variance is the Lawless-Nadeau robust
    estimate, which does not assume the items share one Poisson process.

    ``NonParametricCounting`` is an instance of this class; its ``fit``
    returns a new, fitted instance, which carries ``mcf``, ``mcf_cb`` and
    ``plot``.
    """

    # Set on the instance the fit returns, not in __init__ -- the
    # singleton fitter is called on a bare class and hands back a
    # populated one. Annotated (not assigned) so the attributes have
    # declared types without becoming class-level defaults shared by
    # every instance.
    x: npt.NDArray
    r: npt.NDArray
    d: npt.NDArray
    mcf_hat: npt.NDArray
    #: ``None`` on simulated models, which carry no variance.
    var: "npt.NDArray | None"
    data: RecurrentEventData
    #: Where observation begins: the MCF is 0 from here to the first event
    #: and undefined (NaN) before it. That is time 0 unless an item enters
    #: earlier (a negative ``tl``), which makes negative times observed.
    origin: float = 0.0
    #: The ``(lower, upper)`` interval the MCF is defined on, set by
    #: :meth:`set_support`; ``None`` (the default) when it has not been set.
    support: "tuple[float, float] | None" = None

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise this fitted MCF (mean cumulative function) estimate to a
        plain, JSON-serialisable dict.

        Stores the step arrays that ``mcf``/``mcf_cb`` read: the event times
        ``x``, the estimate ``mcf_hat`` and its variance ``var`` (the
        Lawless-Nadeau robust variance for a fitted MCF).
        The raw ``data`` is not stored (it is only needed to re-fit or to plot
        raw counts).

        See Also
        --------
        from_dict, to_json, from_json
        """
        out = {
            "model": "NonParametricCounting",
            "x": np.asarray(self.x, dtype=float).tolist(),
            "mcf_hat": np.asarray(self.mcf_hat, dtype=float).tolist(),
            # A simulated MCF has no variance; ``None`` (JSON null) keeps
            # it that way on reload. ``asarray(None)`` stored a NaN, and
            # the restored ``mcf_cb`` returned NaN bounds instead of saying
            # there are none.
            "var": (
                None
                if self.var is None
                else np.asarray(self.var, dtype=float).tolist()
            ),
            "origin": float(self.origin),
        }
        # Only when set: without it the dictionary is readable by v0.20.
        if self.support is not None:
            out["support"] = [float(v) for v in self.support]
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "NonParametricCounting":
        """
        Rebuild an MCF estimate from a :meth:`to_dict` dictionary.

        See Also
        --------
        to_dict, to_json, from_json
        """
        require_model_tag(
            model_dict, "NonParametricCounting", "an MCF estimate"
        )
        out = cls()
        out.x = np.array(model_dict["x"], dtype=float)
        out.mcf_hat = np.array(model_dict["mcf_hat"], dtype=float)
        var = model_dict["var"]
        # Older files stored a missing variance as a single NaN; read that
        # (and null) as no variance.
        var_arr = None if var is None else np.array(var, dtype=float)
        if var_arr is not None and var_arr.ndim == 0 and np.isnan(var_arr):
            var_arr = None
        out.var = var_arr
        out.origin = float(model_dict.get("origin", 0.0))
        support = _support_from_dict(model_dict)
        if support is not None:
            out.set_support(*support)
        return out

    def set_support(
        self, lower: float, upper: float
    ) -> "NonParametricCounting":
        """
        Give the MCF an explicit support, ``[lower, upper]``.

        Without one, the MCF is 0 from the origin (where observation
        begins: time 0, or the earliest entry) to the first event, and NaN
        before the origin and after the last observed time, since nothing
        is known there. With a support set, :meth:`mcf` and :meth:`mcf_cb`
        (for every ``interp``) are

        - 0 in ``[lower, origin)``;
        - the value at the last observed time, carried, in
          ``(last, upper]``;
        - NaN outside ``[lower, upper]``.

        Carrying the last value says that no events happen after the last
        observed time, which is the analyst's claim to make, not the
        data's. The variable need not be time: ``lower`` may be negative,
        and either bound infinite. The bounds are kept by ``to_dict``.

        Parameters
        ----------
        lower : float
            The lower end of the support; at most the origin.
        upper : float
            The upper end; at least the last observed time, and above
            ``lower``.

        Returns
        -------
        NonParametricCounting
            The model itself, so the call can be chained.

        Raises
        ------
        ValueError
            If a bound is NaN or not a number, ``lower`` is not below
            ``upper``, or the bounds do not contain ``[origin, last]``.

        Examples
        --------
        >>> from surpyval.recurrent import NonParametricCounting
        >>> x = [3, 9, 20, 35, 56, 60, 11, 44, 60]
        >>> i = [1, 1, 1, 1, 1, 1, 2, 2, 2]
        >>> c = [0, 0, 0, 0, 0, 1, 0, 0, 1]
        >>> model = NonParametricCounting.fit(x, i=i, c=c)
        >>> model.mcf([-5, 30, 80])
        array([nan, 2. , nan])
        >>> model.set_support(-10, 100).mcf([-20, -5, 30, 80, 120])
        array([nan, 0. , 2. , 3.5, nan])
        """
        self.support = _check_support(
            lower,
            upper,
            self._origin(),
            float(self.x.max()),
            _MCF_RANGE,
        )
        return self

    def _within_support(
        self,
        x: npt.ArrayLike,
        f: Callable[[npt.ArrayLike], npt.NDArray],
    ) -> npt.NDArray:
        """``f(x)``, restricted to the support when one is set (see
        :meth:`set_support`); the MCF and its bounds start at 0."""
        if self.support is None:
            return f(x)
        return _on_support(
            self.support, self._origin(), float(self.x.max()), x, f, 0.0
        )

    @keeps_query_shape
    def mcf(self, x: npt.ArrayLike, interp: str = "step") -> npt.NDArray:
        """
        The estimated mean cumulative function at ``x``.

        Parameters
        ----------
        x : array like
            The times at which to evaluate the MCF.
        interp : str, optional
            ``"step"`` (the default) for the right-continuous step estimate,
            or ``"linear"`` to interpolate linearly between event times
            (from 0 at time 0).

        Returns
        -------
        numpy array
            The MCF at each ``x``: 0 before the first event time (for
            ``"linear"``, rising from 0 at the origin to the first event),
            and NaN beyond the last observed time and before the origin.
            The origin is time 0, or the earliest entry when an item enters
            before it (a negative ``tl``). With a support set (see
            :meth:`set_support`) it is 0 from ``lower`` to the origin, the
            value at the last observed time from there to ``upper``, and
            NaN outside them.
        """
        _check_option("interp", interp, _MCF_INTERP)
        return self._within_support(x, lambda q: self._mcf(q, interp))

    def _mcf(self, x: npt.ArrayLike, interp: str) -> npt.NDArray:
        # ``mcf`` without the bounds (see ``set_support``).
        x = np.atleast_1d(np.asarray(x, dtype=float))
        grid, values = self._curve_from_origin(self.mcf_hat)
        # Let's not assume we can predict above the highest measurement
        if interp == "step":
            idx = np.searchsorted(self.x, x, side="right") - 1
            mcf = self.mcf_hat[np.clip(idx, 0, None)].astype(float)
            mcf[idx < 0] = 0
        elif interp == "linear":
            mcf = np.interp(x, grid, values)
        else:
            _check_option("interp", interp, _MCF_INTERP)
        # ... nor at a missing time: NaN in, NaN out (the step lookup put
        # a NaN past every time, at the last value, #382).
        mcf[(x > self.x.max()) | (x < self._origin()) | np.isnan(x)] = np.nan
        return mcf

    def _origin(self) -> float:
        """Where observation begins; never after the first time on the
        grid (a from_xrd triple may start below 0)."""
        return float(min(getattr(self, "origin", 0.0), self.x.min()))

    def _curve_from_origin(
        self, values: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """The grid and ``values`` with the MCF's starting point, 0 at the
        origin, prepended (unless the first time is the origin itself), for
        linear interpolation. ``values`` may be ``(k, len(x))``."""
        origin = self._origin()
        values = np.asarray(values, dtype=float)
        if self.x.min() > origin:
            zero = np.zeros(values.shape[:-1] + (1,))
            return (
                np.hstack([[origin], self.x]),
                np.concatenate([zero, values], axis=-1),
            )
        return np.asarray(self.x, dtype=float), values

    @keeps_query_shape
    def mcf_cb(
        self,
        x: npt.ArrayLike,
        bound: str = "two-sided",
        interp: str = "step",
        *,
        alpha_ci: float = 0.05,
        bound_type: str = "exp",
        dist: str = "z",
    ) -> npt.NDArray:
        """
        Confidence bounds for the MCF at the query times ``x``.

        Two-sided bounds return one row per query with columns ordered
        ``[lower, upper]`` (matching the parametric ``cif_cb``); one-sided
        bounds return a 1-D array. Queries before the first event return 0
        (for ``interp="linear"``, bounds rising from 0 at the origin, as
        the MCF does); queries after the last observed time or before the
        origin return NaN, mirroring :meth:`mcf`. With a support set (see
        :meth:`set_support`) they are 0 from ``lower`` to the origin, the
        bounds at the last observed time from there to ``upper``, and NaN
        outside them.

        Parameters
        ----------
        x : array like
            The times at which to compute the bounds.
        bound : str, optional
            ``"two-sided"`` (the default), ``"upper"`` or ``"lower"``.
        interp : str, optional
            ``"step"`` (the default) or ``"linear"``, as for :meth:`mcf`.
        alpha_ci : float, optional
            The total tail probability of the bound(s): a two-sided
            ``1 - alpha_ci`` interval, or a one-sided bound exceeded with
            probability ``alpha_ci``. Defaults to 0.05. Keyword only, as
            are the arguments after it.
        bound_type : str, optional
            ``"exp"`` (the default) for bounds on the log scale,
            :math:`\\hat{M} e^{\\pm z \\sqrt{V} / \\hat{M}}`, which stay
            positive; or ``"normal"`` for Wald bounds
            :math:`\\hat{M} \\pm z \\sqrt{V}`.
        dist : str, optional
            Only ``"z"``, the normal critical value.

        Returns
        -------
        numpy array
            The bound(s) at each ``x``.

        Raises
        ------
        ValueError
            If the model carries no variance (an MCF built from simulated
            data), or ``bound``, ``interp``, ``bound_type`` or ``dist`` is
            not one of its values.
        """
        # Up front, as unknown values: 'both' used to fail as an
        # UnboundLocalError ('stat'), and an unknown interp returned the
        # unselected bounds (#416).
        _check_option("bound", bound, _BOUNDS)
        _check_option("interp", interp, _MCF_INTERP)
        return self._within_support(
            x,
            lambda q: self._mcf_cb(
                q, bound, interp, alpha_ci, bound_type, dist
            ),
        )

    def _mcf_cb(
        self,
        x: npt.ArrayLike,
        bound: str,
        interp: str,
        alpha_ci: float,
        bound_type: str,
        dist: str,
    ) -> npt.NDArray:
        # ``mcf_cb`` without the bounds (see ``set_support``).
        # The stored variance (Lawless-Nadeau robust for a fitted MCF, the
        # per-step one for ``from_xrd``) with a normal (z) critical value.
        _check_option("bound_type", bound_type, ("exp", "normal"))
        if dist != "z":
            raise ValueError(
                "'dist' must be 'z'. The 't' option (Student-t with the "
                "at-risk count as degrees of freedom) has been removed: it "
                "had no asymptotic justification, was undefined once the "
                "risk set fell to one item, and widened the bounds "
                "arbitrarily as the risk set shrank. The normal ('z') "
                "critical value is what the asymptotic theory of the MCF "
                "estimator justifies."
            )
        x = np.atleast_1d(np.asarray(x, dtype=float))
        if bound in ["upper", "lower"]:
            stat = norm.ppf(alpha_ci, 0, 1)
            if bound == "upper":
                stat = -stat
        elif bound == "two-sided":
            stat = norm.ppf(alpha_ci / 2, 0, 1)
            # Row 0 carries the negative multiplier (lower bound), row 1
            # the positive one, so two-sided output is [lower, upper] —
            # it used to be [upper, lower], inconsistent with the
            # parametric cif_cb (#285).
            stat = np.array([1, -1]).reshape(2, 1) * stat

        if self.var is None:
            raise ValueError(
                "This model carries no variance (a simulated MCF), so "
                "confidence bounds are unavailable."
            )
        if bound_type == "exp":
            # Log-scale (exponential) bounds
            # Before the first event (possible for one cause of a
            # cause-specific MCF) the estimate and its variance are both
            # 0, and so are the bounds -- not 0/0.
            positive = self.mcf_hat > 0
            ratio = np.divide(
                np.sqrt(self.var),
                self.mcf_hat,
                out=np.zeros_like(self.mcf_hat, dtype=float),
                where=positive,
            )
            mcf_cb = self.mcf_hat * np.exp(stat * ratio)
        else:
            # Normal (Wald) bounds: estimate +- z * standard error. This
            # used to scale the standard error by the estimate again
            # (sqrt(var * mcf**2)), giving far too wide, negative bounds.
            mcf_cb = self.mcf_hat + np.sqrt(self.var) * stat
        # Let's not assume we can predict above the highest measurement
        # ... nor at a missing time (see ``_mcf``).
        invalid = (x > self.x.max()) | (x < self._origin()) | np.isnan(x)
        if interp == "step":
            # Select by query position FIRST, then mask the query-length
            # result: the masks used to be applied to the grid-length
            # array, which zeroed whole bound rows, wrapped out-of-range
            # queries to the last grid value, and crashed with an
            # IndexError for more queries than bounds (#285).
            idx = np.searchsorted(self.x, x, side="right") - 1
            safe_idx = np.clip(idx, 0, None)
            below = idx < 0
            if bound == "two-sided":
                mcf_cb = mcf_cb[:, safe_idx].T
                mcf_cb[below, :] = 0
                mcf_cb[invalid, :] = np.nan
            else:
                mcf_cb = mcf_cb[safe_idx]
                mcf_cb[below] = 0
                mcf_cb[invalid] = np.nan
        elif interp == "linear":
            # From 0 at the origin, as the linear MCF itself is: before the
            # first event the bounds used to be held at the first event's,
            # so they did not contain the interpolated MCF.
            grid, bounds = self._curve_from_origin(mcf_cb)
            if bound == "two-sided":
                R1 = np.interp(x, grid, bounds[0, :])
                R2 = np.interp(x, grid, bounds[1, :])
                mcf_cb = np.vstack([R1, R2]).T
            else:
                mcf_cb = np.interp(x, grid, bounds)
            mcf_cb[invalid] = np.nan
        return mcf_cb

    def plot(
        self,
        *,
        alpha_ci: float = 0.05,
        plot_bounds: bool = True,
        ax: "Axes | None" = None,
        start: float = 0.0,
    ) -> "Axes":
        """
        Plot the MCF as a step function, with its confidence bounds.
        The arguments are keyword only.

        Parameters
        ----------
        alpha_ci : float, optional
            The total tail probability of the two-sided bounds: a
            ``1 - alpha_ci`` interval. Defaults to 0.05.
        plot_bounds : bool, optional
            Whether to draw the bounds (skipped if the model has no
            variance). Defaults to :code:`True`.
        ax : matplotlib Axes, optional
            The axes to draw on. Defaults to the current axes.
        start : float, optional
            The time the step plot starts from, at an MCF of 0. Defaults
            to 0.

        Returns
        -------
        matplotlib Axes
        """
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gcf().gca()

        # Prepend the start point so the step plot always begins from it
        # (the MCF is 0 before the first observed event).
        if start is not None and start < self.x.min():
            x = np.hstack([[start], self.x])
            mcf_hat = np.hstack([[0.0], self.mcf_hat])
        else:
            x = self.x
            mcf_hat = self.mcf_hat

        ax.step(x, mcf_hat, where="post", label="MCF")
        if plot_bounds:
            if self.var is not None:
                cb = self.mcf_cb(self.x, bound="two-sided", alpha_ci=alpha_ci)
                if start is not None and start < self.x.min():
                    cb = np.vstack([[0.0, 0.0], cb])
                ax.step(
                    x,
                    cb,
                    where="post",
                    label=f"{(1 - alpha_ci) * 100:g}% Confidence Bounds",
                    color="red",
                )
        return ax

    @classmethod
    def from_xrd(
        cls, x: npt.ArrayLike, r: npt.ArrayLike, d: npt.ArrayLike
    ) -> "NonParametricCounting":
        """Build the Nelson-Aalen MCF from an ``(x, r, d)`` triple; the
        single home of the estimator.

        An ``(x, r, d)`` triple does not say which item each event came
        from, so the variance here is the per-step (naive) one, which
        assumes that the ``d`` events at a time all happened to different
        items (so ``d <= r``; a step with more events than items at risk
        makes the variance NaN from there on). Each of the ``r`` items at
        risk then contributes 1 or 0 events, and the step's increment
        ``d / r`` has estimated variance

        .. math::

            \\frac{1}{r^2} \\sum_k \\Big(n_k - \\frac{d}{r}\\Big)^2
            = \\frac{d (r - d)}{r^3},

        the squared deviations of the items' counts from the step's mean
        :math:`d/r`. Steps are treated as independent of one another. That
        is right for a Poisson process but understates the variance when
        items differ in their rates, because it has no within-item
        covariance. :meth:`fit` (and ``CauseSpecificMCF``) have the per-item
        data and replace it with the Lawless-Nadeau robust variance.

        Examples
        --------
        Two events at a time when three items are at risk:

        >>> from surpyval.recurrent import NonParametricCounting
        >>> model = NonParametricCounting.from_xrd([1.0], [3], [2])
        >>> model.mcf_hat, model.var
        (array([0.66666667]), array([0.07407407]))
        """
        out = cls()
        x, r, d = np.asarray(x), np.asarray(r), np.asarray(d)
        out.x, out.r, out.d = x, r, d
        out.mcf_hat = np.cumsum(d / r)
        # Centred on the step's mean d / r. It used to be centred on 1 / r
        # (the mean only when d == 1), which overstated the variance of
        # tied steps; for d == 1 the two agree. d (r - d) / r^3 is 0 when
        # there are no events, so no masking is needed. More events than
        # items at risk breaks the one-event-per-item assumption, and the
        # triple cannot say how the events were shared out, so the
        # variance from that step on is unknown (NaN), not negative.
        step_var = np.where(d <= r, d * (r - d) / r**3, np.nan)
        out.var = np.cumsum(step_var)
        return out

    def fit_from_recurrent_data(
        self, data: RecurrentEventData
    ) -> "NonParametricCounting":
        """
        Fit the MCF from a prepared
        :class:`~surpyval.utils.recurrent_event_data.RecurrentEventData`,
        as built by ``surpyval.handle_xicn``. :meth:`fit` builds one from
        its arrays and calls this; the same restrictions on censoring and
        truncation apply.

        Returns
        -------
        NonParametricCounting
            The fitted estimate.
        """
        out = self._point_estimate(data)
        out.var = _lawless_nadeau_var(data, out.x, out.r, out.d)
        return out

    def _point_estimate(
        self, data: RecurrentEventData
    ) -> "NonParametricCounting":
        """:meth:`fit_from_recurrent_data` without the Lawless-Nadeau
        variance, which costs about (items x distinct times) and which the
        simulations, returning only the MCF, discard."""
        reject_unsupported_nonparametric(data, "NonParametricCounting")
        out = type(self).from_xrd(*data.to_xrd())
        out.data = data
        out.origin = _observation_origin(data)
        return out

    def fit(
        self,
        x: npt.ArrayLike,
        i: npt.ArrayLike | None = None,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        tl: npt.ArrayLike | float | None = None,
        tr: npt.ArrayLike | float | None = None,
        windows: dict | None = None,
    ) -> "NonParametricCounting":
        """
        Fit a nonparametric (Nelson-Aalen) MCF.

        Parameters
        ----------
        x : array like
            Event (and censoring) times.
        i : array like, optional
            Item / subject id for each row. Defaults to a single item.
        c : array like, optional
            Censoring flag for each row: 0 an observed event, 1 the
            right-censored end of an item's observation. Left- (-1) and
            interval- (2) censored rows are not supported and raise a
            ``ValueError``.
        n : array like, optional
            Count of events at each row. Defaults to 1.
        tl : array like or scalar, optional
            Left-truncation (delayed-entry) time of each item: a scalar for
            every item, or one value per row (the same on every row of an
            item). An item only
            enters the at-risk set once observation begins at ``tl``, so
            earlier event times are estimated over a smaller risk set.
        tr : array like or scalar, optional
            Right-truncation time of each item, given like ``tl``: the end
            of its observation
            window. The item stays in the at-risk set up to ``tr`` and
            leaves it after, exactly as if it had an end-of-observation
            (``c=1``) row at ``tr`` -- the same window-close the parametric
            NHPP fits integrate to.
        windows : dict, optional
            Gapped (multi-window) observation: a mapping ``{item: [(start,
            end), ...]}`` giving each item's disjoint observation windows.
            When given, every row in ``x`` must be an observed event (``c=0``).
            Each window becomes its own at-risk period, so an item is correctly
            absent from the risk set during a gap. Mutually exclusive with
            ``tl``/``tr``.

        Returns
        -------
        NonParametricCounting
            The fitted estimate.

        Examples
        --------
        Two systems observed to t = 60 (the ``c=1`` rows):

        >>> from surpyval.recurrent import NonParametricCounting
        >>> x = [3, 9, 20, 35, 56, 60, 11, 44, 60]
        >>> i = [1, 1, 1, 1, 1, 1, 2, 2, 2]
        >>> c = [0, 0, 0, 0, 0, 1, 0, 0, 1]
        >>> model = NonParametricCounting.fit(x, i=i, c=c)
        >>> model.mcf([10, 30, 60])
        array([1. , 2. , 3.5])
        >>> model.mcf_cb([10, 30, 60])
        array([[0.25009765, 3.99843816],
               [1.00019529, 3.999219  ],
               [1.93248007, 6.33900458]])
        """
        data = handle_xicn(x, i, c, n, tl=tl, tr=tr, windows=windows)
        return self.fit_from_recurrent_data(data)


def _observation_origin(data: RecurrentEventData) -> float:
    """Where the MCF starts: time 0, or the earliest entry ``tl`` when an
    item enters before 0 (its negative times are then observed; the MCF
    used to be NaN there)."""
    tl = np.asarray(data.tl, dtype=float)
    finite = tl[np.isfinite(tl)]
    return float(min(0.0, finite.min())) if finite.size else 0.0


# The blocks the Lawless-Nadeau sum is centred in (see there).
_LN_BLOCKS = 64


def _lawless_nadeau_var(
    data: RecurrentEventData,
    x: npt.NDArray,
    r: npt.NDArray,
    d: npt.NDArray,
    counted: "npt.NDArray | None" = None,
) -> npt.NDArray:
    """The Lawless-Nadeau robust variance of the Nelson-Aalen MCF.

    With :math:`\\delta_k(t)` item ``k``'s at-risk indicator,
    :math:`n_k(t)` its events at ``t`` and :math:`\\hat{m}(t) = d(t)/r(t)`
    the MCF increment,

    .. math::

        \\widehat{Var}\\,\\hat{M}(t) = \\sum_k \\Big[ \\sum_{t_j \\le t}
        \\frac{\\delta_k(t_j)}{r(t_j)} \\big(n_k(t_j) -
        \\hat{m}(t_j)\\big) \\Big]^2 .

    Each item's deviations are summed over time *before* squaring, so
    an item with a high rate throughout adds its covariance across
    steps; that is what makes the variance robust to items differing in
    their rates (it does not assume a Poisson process). Items split
    into observation windows are regrouped under their original item.
    With a single item there is nothing to compare it with, and the
    variance is zero.

    ``counted`` is an optional row mask restricting which events count
    (``n_k``); ``d`` must count the same events. The cause-specific MCF
    passes the rows of one cause, so the other causes' events count as
    non-events while the risk set stays shared.

    The sum is formed without a pass over the time grid per item, which
    was O(items x times) (#521). With :math:`W(t) = \\sum_{t_j \\le t}
    \\hat{m}(t_j) / r(t_j)`, a cluster's (an item's, or a windowed item's
    windows') running sum is :math:`C(t) = D(t) - \\alpha(t) W(t)`: its
    events' :math:`n / r` and its windows' :math:`W` at entry and exit
    make up :math:`D`, and :math:`\\alpha` counts its windows open at
    :math:`t`. Both change only at its own rows and window ends, so
    :math:`\\sum C^2 = \\sum D^2 - 2 W \\sum \\alpha D + W^2 \\sum
    \\alpha^2` is accumulated from the changes, in O(rows + times), with
    :math:`D` and :math:`W` measured from the start of each of a few
    blocks of the time grid so that the three sums do not cancel (the
    result agrees with the sum over items to about 1e-13).
    """
    x_out = data.midpoints if data.x.ndim == 2 else data.x
    is_event = (data.c == 0) | (data.c == 2) | (data.c == -1)
    if counted is not None:
        is_event = is_event & counted
    m = len(x)
    if m == 0:
        return np.zeros(0)
    col = np.searchsorted(x, x_out)
    dm = np.where(r > 0, d / np.where(r > 0, r, 1), 0.0)
    inv_r = np.where(r > 0, 1.0 / np.where(r > 0, r, 1), 0.0)
    W = np.cumsum(inv_r * dm)
    window_map = getattr(data, "window_map", None) or {}
    # The same windows the risk set ``r`` was built from, so each item's
    # at-risk indicator agrees with its share of ``r`` (including a
    # right-truncation close past its last row).
    entry, exit_ = data.item_observation_windows()
    # Each item's at-risk run of the grid, lo..hi (empty when lo > hi).
    lo = np.searchsorted(x, entry, side="left")
    hi = np.searchsorted(x, exit_, side="right") - 1
    # Items split into observation windows are regrouped under their
    # original item, as one cluster.
    keys: dict = {}
    cluster = np.array(
        [
            keys.setdefault(
                window_map[item][0] if item in window_map else item, len(keys)
            )
            for item in data.items
        ],
        dtype=np.intp,
    )
    # Each row's item, as its position in ``data.items``.
    row_item = data.item_rows()[1]
    # An event counts where its item is at risk, as the at-risk indicator
    # multiplied it before.
    ev = is_event & (col >= lo[row_item]) & (col <= hi[row_item])
    open_ = lo <= hi
    closes = open_ & (hi + 1 < m)
    W_before = np.where(lo > 0, W[np.maximum(lo - 1, 0)], 0.0)
    # The changes to (D, alpha) of each cluster, and the grid time from
    # which each holds: its events, its windows opening and closing.
    where = np.concatenate([col[ev], lo[open_], hi[closes] + 1])
    owner = np.concatenate(
        [
            cluster[row_item[ev]],
            cluster[open_],
            cluster[closes],
        ]
    )
    d_D = np.concatenate(
        [
            np.asarray(data.n, dtype=float)[ev] * inv_r[col[ev]],
            W_before[open_],
            -W[hi[closes]],
        ]
    )
    d_alpha = np.concatenate(
        [
            np.zeros(int(ev.sum())),
            np.ones(int(open_.sum())),
            -np.ones(int(closes.sum())),
        ]
    )
    if where.size == 0:
        return np.zeros(m)
    # Each cluster's state after each of its changes, in time order: a
    # running sum within each cluster, taken a step at a time across all
    # clusters at once (a cumsum over every cluster, less its value at
    # the cluster's start, loses the digits the clusters before it carry).
    order = np.lexsort((where, owner))
    where, owner = where[order], owner[order]
    D, alpha = d_D[order], d_alpha[order]
    first = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
    length = np.diff(np.r_[first, owner.size])
    D_prev, alpha_prev = np.zeros_like(D), np.zeros_like(alpha)
    for step in range(1, int(length.max())):
        at = first[length > step] + step
        D_prev[at], alpha_prev[at] = D[at - 1], alpha[at - 1]
        D[at] += D[at - 1]
        alpha[at] += alpha[at - 1]
    # Summed over the clusters, C^2 = D^2 - 2 W alpha D + W^2 alpha^2
    # loses to cancellation what D^2 and W^2 are larger than C^2: much,
    # where the items' counts are alike. So the time grid is cut into
    # blocks of equal growth in W, and in block b each cluster is centred
    # on the block's start, g = D - alpha W(s_b), with u = W(t) - W(s_b):
    # C = g - alpha u, with g and u no larger than C and W's growth over
    # the block.
    starts = np.unique(
        np.searchsorted(W, W[-1] * np.arange(_LN_BLOCKS) / _LN_BLOCKS)
    )
    starts = np.unique(np.r_[0, starts[starts < m]])
    block = np.searchsorted(starts, np.arange(m), side="right") - 1
    W_start = W[starts]
    # Each change, centred on its own block: what it adds to the sums.
    W_b = W_start[block[where]]
    g_new = D - alpha * W_b
    g_old = D_prev - alpha_prev * W_b
    diff = [
        np.bincount(where, g_new * g_new - g_old * g_old, minlength=m),
        np.bincount(where, alpha * g_new - alpha_prev * g_old, minlength=m),
        np.bincount(
            where, alpha * alpha - alpha_prev * alpha_prev, minlength=m
        ),
    ]
    if starts.size > 1:
        # Each cluster's state carried into a block (after its last
        # change before the block starts) moves to the new centre there.
        n_clusters = len(keys)
        key = owner.astype(np.int64) * (m + 1) + where
        b = np.arange(1, starts.size)
        query = (
            np.arange(n_clusters, dtype=np.int64)[:, None] * (m + 1)
            + starts[b][None, :]
        )
        last = np.searchsorted(key, query, side="left") - 1
        carried = (last >= 0) & (
            owner[np.maximum(last, 0)] == np.arange(n_clusters)[:, None]
        )
        last = last[carried]
        b = np.broadcast_to(b, carried.shape)[carried]
        D_c, alpha_c = D[last], alpha[last]
        g_to = D_c - alpha_c * W_start[b]
        g_from = D_c - alpha_c * W_start[b - 1]
        at = starts[b]
        diff[0] += np.bincount(at, g_to * g_to - g_from * g_from, minlength=m)
        diff[1] += np.bincount(at, alpha_c * (g_to - g_from), minlength=m)
    s_gg, s_ag, s_aa = (np.cumsum(v) for v in diff)
    u = W - W_start[block]
    return s_gg - 2.0 * u * s_ag + u * u * s_aa
