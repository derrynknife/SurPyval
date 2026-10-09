"""
This code was created for and sponsored by Cartiga (www.cartiga.com).
Cartiga makes no representations or warranties in connection with the code
and waives any and all liability in connection therewith. Your use of the
code constitutes acceptance of these terms.

Copyright 2022 Cartiga LLC
"""

from __future__ import annotations

import textwrap
from typing import Any

import numpy as np
import numpy.typing as npt

import surpyval as surv
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
    to_native,
)
from surpyval.univariate.competing_risks.aalen_johansen import (
    aalen_johansen_iif,
    aalen_johansen_variance,
)
from surpyval.univariate.competing_risks.labels import (
    label_from_native,
    ordered_labels,
)
from surpyval.univariate.nonparametric._support import (
    check_support,
    on_support,
    support_from_dict,
)
from surpyval.univariate.nonparametric.kaplan_meier import kaplan_meier as km
from surpyval.univariate.nonparametric.nelson_aalen import nelson_aalen as na
from surpyval.univariate.nonparametric.nonparametric import (
    warn_bounds_past_data,
)
from surpyval.univariate.regression.regression_data import (
    check_finite_event_times,
)
from surpyval.utils import (
    validate_cif_event,
    validate_cr_df_inputs,
    validate_cr_inputs,
    validate_event,
)
from surpyval.utils.data_formats import _get_idx
from surpyval.utils.removed_names import column_arguments
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.validation import BOUNDS, check_alpha_ci, check_option

# The functions ``cb`` can bound ('R' and 'F' are aliases of 'sf' and 'ff',
# as for the single-event estimates).
_CB_ON = ("cif", "sf", "ff", "Hf", "R", "F")


class CompetingRisks(SerialisableMixin):
    """
    Non-parametric competing-risks estimate: the Aalen-Johansen cumulative
    incidence function (CIF) of each cause, and the all-cause and
    cause-specific (net) hazards and survival.

    Each unit fails from one of several causes ``e`` (or is right-censored,
    with no cause). The cumulative incidence of cause :math:`j` is the
    probability of failing *from that cause* by time :math:`t` while the
    others still act,

    .. math::
        F_j(t) = \\sum_{t_i \\le t} S(t_{i-1}) \\frac{d_{ij}}{r_i},

    with :math:`S` the all-cause Kaplan-Meier survival, :math:`d_{ij}` the
    failures from cause :math:`j` at :math:`t_i` and :math:`r_i` the number
    at risk. The CIFs of all causes add up to the all-cause failure
    probability.

    Call the class method ``CompetingRisks.fit`` (or ``fit_from_df``); it
    returns a fitted instance.
    """

    #: The source DataFrame when fitted via ``fit_from_df`` (named so it
    #: does not shadow the density method ``df``, #253).
    source_df: Any
    # Attributes populated by ``fit`` / ``from_dict``; declared for the type
    # checker.
    event_idx_map: dict
    n_event_types: int
    x: np.ndarray
    d: np.ndarray
    r: np.ndarray
    h0: np.ndarray
    H0: np.ndarray
    S: np.ndarray
    d_e: np.ndarray
    h0_e: np.ndarray
    H0_e: np.ndarray
    IIF: np.ndarray
    CIF: np.ndarray
    #: The survival estimator ``sf``/``ff``/``Hf`` report:
    #: ``"Nelson-Aalen"`` (``exp(-H)``) or ``"Kaplan-Meier"`` (product limit).
    how: str = "Nelson-Aalen"
    #: The ``(lower, upper)`` interval the estimate is defined on, set by
    #: :meth:`set_support`; ``None`` (the default) when it has not been set.
    support: "tuple[float, float] | None" = None

    # -- serialisation -----------------------------------------------------

    _SERIALISED_ARRAYS = (
        "x",
        "d",
        "r",
        "h0",
        "H0",
        "S",
        "d_e",
        "h0_e",
        "H0_e",
        "IIF",
        "CIF",
    )

    def to_dict(self) -> dict:
        """
        Serialise this fitted nonparametric competing-risks model to a plain,
        JSON-serialisable dict: the event index map and the fitted step arrays
        (the shared baseline plus the per-event incidence and cumulative-
        incidence functions). The reloaded model reproduces every prediction
        exactly.
        """
        out: dict = {
            "model": "CompetingRisks",
            # list of [event, index] pairs to preserve the event key types
            "event_idx_map": [
                [to_native(k), int(v)] for k, v in self.event_idx_map.items()
            ],
            "n_event_types": int(self.n_event_types),
            # Stored under "method", the argument's old name, so files
            # written before the rename still load.
            "method": self.how,
        }
        for name in self._SERIALISED_ARRAYS:
            out[name] = np.asarray(getattr(self, name), dtype=float).tolist()
        # Only when set: without it the dictionary is readable by v0.20.
        if self.support is not None:
            out["support"] = [float(v) for v in self.support]
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "CompetingRisks":
        """Rebuild a nonparametric competing-risks model from a dict."""
        require_model_tag(
            model_dict, "CompetingRisks", "a competing-risks model"
        )
        out = cls()
        out.event_idx_map = {
            label_from_native(k): int(v)
            for k, v in model_dict["event_idx_map"]
        }
        out.n_event_types = int(model_dict["n_event_types"])
        # dicts written before the method was stored reported exp(-H)
        out.how = model_dict.get("method", "Nelson-Aalen")
        for name in cls._SERIALISED_ARRAYS:
            setattr(out, name, np.array(model_dict[name], dtype=float))
        support = support_from_dict(model_dict)
        if support is not None:
            out.set_support(*support)
        return out

    def __repr__(self) -> str:
        out = """\
        Competing Risk model with events:
        {events}
        """.format(events=list(self.event_idx_map.keys()))
        return textwrap.dedent(out)

    def set_support(self, lower: float, upper: float) -> "CompetingRisks":
        """
        Give the estimate an explicit support, ``[lower, upper]``.

        Without one, every function starts at its initial value before the
        first time and holds its last value after the last, however far
        from the data. With a support set, they do so only within them:

        - in ``[lower, x[0])``: ``sf`` 1, and ``ff``, ``Hf``, ``hf``,
          ``df``, ``iif`` and ``cif`` 0;
        - in ``(x[-1], upper]``: the value at the last time, carried
          (for ``hf``, ``df`` and ``iif``, as without bounds, that of the
          step containing the last time);
        - outside ``[lower, upper]``: NaN.

        ``x[0]`` and ``x[-1]`` are the first and last observed times,
        failures or censorings. The variable need not be time: ``lower``
        may be negative, and either bound infinite. The bounds are kept
        by ``to_dict``.

        Parameters
        ----------
        lower : float
            The lower end of the support; at most the first time, ``x[0]``.
        upper : float
            The upper end; at least the last time, ``x[-1]``, and above
            ``lower``.

        Returns
        -------
        CompetingRisks
            The model itself, so the call can be chained.

        Raises
        ------
        ValueError
            If a bound is NaN or not a number, ``lower`` is not below
            ``upper``, or the bounds do not contain the observed times.

        Examples
        --------
        >>> from surpyval.univariate.competing_risks import CompetingRisks
        >>> x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> e = ['a', 'b', 'a', None, 'a', 'b', 'a', None, 'b', 'a']
        >>> model = CompetingRisks.fit(x, e).set_support(0, 20)
        >>> model.cif([-1, 0.5, 5, 15, 25], 'a').round(4)
        array([   nan, 0.    , 0.3167, 0.6083,    nan])
        """
        self.support = check_support(
            lower,
            upper,
            float(self.x[0]),
            float(self.x[-1]),
            ("the first time", "the last time"),
        )
        return self

    def _within_support(
        self, x: npt.ArrayLike, f: Any, start: float
    ) -> npt.NDArray:
        """``f(x)``, restricted to the support when one is set (see
        :meth:`set_support`)."""
        if self.support is None:
            return f(x)
        return on_support(
            self.support, float(self.x[0]), float(self.x[-1]), x, f, start
        )

    def _f(self, f: str, x: npt.ArrayLike, event: Any) -> npt.NDArray:
        validate_event(self.event_idx_map, event)
        # Look up a flat copy of the query and give the result its shape
        # back: the sort-based index of a 2-D query spread it over 4-D.
        shape = np.shape(np.atleast_1d(x))
        idx, rev = _get_idx(self.x, np.ravel(x))

        if f == "h":
            arr = self.h0_e
        elif f == "H":
            arr = self.H0_e
        elif f == "IIF":
            arr = self.IIF
        elif f == "CIF":
            arr = self.CIF

        if event is None:
            out = arr.sum(axis=0)[idx][rev]
        else:
            e = self.event_idx_map[event]
            out = arr[e, idx][rev]
        # Query times before the first observed time have index -1, which
        # would otherwise wrap to the *last* step value; every step function
        # here (hazard, cumulative hazard, IIF, CIF) is zero before the
        # first event time.
        out = np.where(idx[rev] < 0, 0.0, out)
        # A missing time sorts past the last step; it has no value.
        out = np.where(np.isnan(np.ravel(x).astype(float)), np.nan, out)
        return out.reshape(shape)

    @keeps_query_shape
    def hf(self, x: npt.ArrayLike, event: Any = None) -> npt.NDArray:
        """
        Hazard (the Nelson-Aalen increment ``d / r`` at each event time, 0
        between them), all causes (``event=None``) or one cause.
        """
        return self._within_support(x, lambda q: self._f("h", q, event), 0.0)

    @keeps_query_shape
    def Hf(self, x: npt.ArrayLike, event: Any = None) -> npt.NDArray:
        """
        Cumulative hazard, all causes (``event=None``) or one cause. With the
        Nelson-Aalen method it is the sum of the hazard increments
        ``d / r``; with Kaplan-Meier it is ``-log`` of the product-limit
        survival, so that ``sf == exp(-Hf)`` either way.
        """

        def H(q: npt.ArrayLike) -> npt.NDArray:
            if self.how == "Kaplan-Meier":
                # 0.0 - log, not -log: where the survival is 1 (before
                # the first time, or the cause's) -log(1) is -0.0 (#728).
                with np.errstate(divide="ignore"):
                    return 0.0 - np.log(self._product_limit(q, event))
            return self._f("H", q, event)

        return self._within_support(x, H, 0.0)

    def _product_limit(self, x: npt.ArrayLike, event: Any) -> npt.NDArray:
        """The product-limit survival, all causes or one cause's (net)."""
        validate_event(self.event_idx_map, event)
        if event is None:
            increments = self.h0
        else:
            increments = self.h0_e[self.event_idx_map[event]]
        S = np.cumprod(1.0 - increments)
        x = np.atleast_1d(np.asarray(x, dtype=float))
        idx = np.searchsorted(self.x, x, side="right") - 1
        out = np.where(idx >= 0, S[np.maximum(idx, 0)], 1.0)
        return np.where(np.isnan(x), np.nan, out)

    @keeps_query_shape
    def sf(self, x: npt.ArrayLike, event: Any = None) -> npt.NDArray:
        """
        Survival, all causes (``event=None``) or the net survival from one
        cause, by the estimator the model was fitted with: ``exp(-H)``
        (Nelson-Aalen, the default) or the product limit (Kaplan-Meier).
        The one-cause survival treats the other causes as censoring, so it
        is *not* the probability of escaping that cause in the presence of
        the others -- use :meth:`cif` for that.
        """
        if self.how == "Kaplan-Meier":
            return self._within_support(
                x, lambda q: self._product_limit(q, event), 1.0
            )
        return np.exp(-self.Hf(x, event=event))

    @keeps_query_shape
    def ff(self, x: npt.ArrayLike, event: Any = None) -> npt.NDArray:
        """
        ``1 - sf``: all causes, or the net failure probability from one
        cause with the others removed (treated as censoring). For the
        probability of failing *from* a cause while the others still act,
        use :meth:`cif`.
        """
        return 1 - self.sf(x, event=event)

    @keeps_query_shape
    def df(self, x: npt.ArrayLike, event: Any = None) -> npt.NDArray:
        """
        ``hf * sf``: the probability mass at each event time, all causes
        (``event=None``) or from one cause's net survival.
        """
        return self.hf(x, event=event) * self.sf(x, event=event)

    @keeps_query_shape
    def iif(self, x: npt.ArrayLike, event: Any) -> npt.NDArray:
        """
        Instantaneous incidence of cause ``event``: the step of the
        cumulative incidence at each event time,
        :math:`S(t_{i-1}) d_{ij} / r_i` (0 between event times).
        """
        validate_cif_event(event)
        return self._within_support(x, lambda q: self._f("IIF", q, event), 0.0)

    @keeps_query_shape
    def cif(self, x: npt.ArrayLike, event: Any) -> npt.NDArray:
        """
        Cumulative incidence of cause ``event`` at ``x``: the probability of
        having failed from that cause by ``x``, with the other causes still
        acting. ``event`` is required.
        """
        validate_cif_event(event)
        return self._within_support(x, lambda q: self._f("CIF", q, event), 0.0)

    @keeps_query_shape
    def cb(
        self,
        x: npt.ArrayLike,
        event: Any = None,
        on: str = "cif",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        bound_type: str = "exp",
    ) -> npt.NDArray:
        r"""
        Pointwise confidence bounds on the cumulative incidence of cause
        ``event`` (``on="cif"``, the default), or on the survival, failure
        probability or cumulative hazard (``on="sf"``, ``"ff"``, ``"Hf"``)
        of all causes (``event=None``) or one cause's net ones.

        The cumulative incidence's variance is Aalen's (1978) estimate,
        the one R's ``cmprsk::cuminc`` reports as ``var`` (see
        :func:`~surpyval.univariate.competing_risks.aalen_johansen.aalen_johansen_variance`).
        It is the same whichever ``how`` the model was fitted with, as the
        cumulative incidence is. The bounds on ``sf``, ``ff`` and ``Hf``
        are those of the single-event estimate of the same data (the
        all-cause failures, or the cause's with the other causes
        censored), by the fitted ``how``: ``KaplanMeier``'s ``cb``
        (Greenwood's variance) or ``NelsonAalen``'s (Aalen's).

        The bounds hold at each ``x`` separately.

        Parameters
        ----------
        x : array_like
            The times at which to bound the function.
        event : optional
            The cause; required for ``on="cif"``. For ``sf``, ``ff`` and
            ``Hf``, ``None`` (the default) is all causes and a cause its
            net function (the other causes treated as censoring).
        on : ('cif', 'sf', 'ff', 'Hf'), str, optional
            The function to bound ('R' and 'F' are accepted for 'sf' and
            'ff'). Defaults to 'cif'.
        alpha_ci : float, optional
            The significance level: 0.05 (the default) gives a 95%
            interval.
        bound : ('two-sided', 'lower', 'upper'), str, optional
            The two-sided interval (the default), with one row per ``x``
            and the columns ``[lower, upper]``, or one side. A one-sided
            bound puts all of ``alpha_ci`` on one side, so it equals that
            end of the two-sided interval at ``2 * alpha_ci``.
        bound_type : ('exp', 'normal'), str, optional
            'exp' (the default) forms the interval on the log(-log) scale,
            :math:`\hat{F}^{\exp(\pm z\,\hat{\sigma} / (\hat{F}\log
            \hat{F}))}` for the cumulative incidence (Choudhury, 2002),
            which keeps it within [0, 1]; 'normal' is
            :math:`\hat{F} \pm z\,\hat{\sigma}`, which can leave it. For
            ``sf``, ``ff`` and ``Hf`` it is the single-event ``cb``'s
            ``bound_type``.

        Returns
        -------
        numpy array
            The bounds at each ``x``: ``[lower, upper]`` rows for a
            two-sided bound, one value per ``x`` for one side.

        Raises
        ------
        ValueError
            If ``on``, ``bound`` or ``bound_type`` is not one of the values
            above, ``alpha_ci`` is not between 0 and 1, ``event`` is not a
            cause of the model, or ``on="cif"`` without an ``event``.

        Notes
        -----
        Where the variance is zero the bounds are the estimate: before the
        cause's first event the cumulative incidence is exactly 0, and so
        are both bounds; before the first observed time they are the
        estimate's initial value (0, or 1 on ``sf``). Above the last
        observed time they are NaN, with a warning naming the times (the
        estimate only holds its last value there), unless the model has a
        support (see :meth:`set_support`): then the bounds at the last time
        are carried to ``upper``, and they are NaN outside ``[lower,
        upper]``. If a cause's cumulative incidence reaches 1 (every
        failure from that cause and no one left at risk) the 'exp' upper
        bound there is 1 and the lower bound the largest one before it.

        References
        ----------
        Aalen, O. (1978), "Nonparametric estimation of partial transition
        probabilities in multiple decrement models", *The Annals of
        Statistics*, 6(3), 534-545.

        Choudhury, J. B. (2002), "Non-parametric confidence interval
        estimation for competing risks analysis: application to
        contraceptive data", *Statistics in Medicine*, 21(8), 1129-1144.

        Examples
        --------
        >>> from surpyval.univariate.competing_risks import CompetingRisks
        >>> x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> e = ['a', 'b', 'a', None, 'a', 'b', 'a', None, 'b', 'a']
        >>> model = CompetingRisks.fit(x, e)
        >>> model.cif([0.5, 5, 9], 'a').round(4)
        array([0.    , 0.3167, 0.4333])
        >>> model.cb([0.5, 5, 9], 'a').round(4)
        array([[0.    , 0.    ],
               [0.0616, 0.6222],
               [0.106 , 0.7323]])
        >>> model.cb(5, 'a', bound='upper').round(4)
        np.float64(0.5787)

        The all-cause survival, by the fitted ``how`` (here Nelson-Aalen):

        >>> model.cb(5, on='sf').round(4)
        array([0.2551, 0.8311])
        """
        check_alpha_ci(alpha_ci)
        check_option(
            "on",
            on,
            _CB_ON,
            "Bounds on the hazard, the density or the incidence increments "
            "('hf', 'df', 'iif') are not available.",
        )
        check_option("bound", bound, BOUNDS)
        check_option("bound_type", bound_type, ("exp", "normal"))
        validate_event(self.event_idx_map, event)
        if on != "cif":
            return self._single_event(event).cb(
                x,
                on=on,
                alpha_ci=alpha_ci,
                bound=bound,
                bound_type=bound_type,
            )
        validate_cif_event(event)
        ends = self._cif_bounds(
            self.event_idx_map[event], alpha_ci, bound, bound_type
        )

        def step(q: npt.ArrayLike) -> npt.NDArray:
            idx = np.searchsorted(self.x, q, side="right") - 1
            out = np.column_stack([v[np.maximum(idx, 0)] for v in ends])
            # Exactly 0 before the first time, as the estimate.
            out = np.where((idx < 0)[:, None], 0.0, out)
            return out if bound == "two-sided" else out[:, 0]

        first, last = float(self.x[0]), float(self.x[-1])
        if self.support is not None:
            support = self.support
        else:
            # As the single-event step estimates' bounds (#665): NaN, with
            # a warning, past the last time.
            support = (-np.inf, last)
            xf = np.atleast_1d(np.asarray(x, dtype=float))
            past = np.unique(xf[xf > last])
            if past.size:
                warn_bounds_past_data("cb", last, past, "cif")
        return on_support(support, first, last, x, step, 0.0)

    def _cif_bounds(
        self, k: int, alpha_ci: float, bound: str, bound_type: str
    ) -> list[npt.NDArray]:
        """The bounds of cause ``k``'s cumulative incidence at each
        distinct time: ``[lower, upper]``, or the one side asked for."""
        from scipy.stats import norm

        F = self.CIF[k]
        var = aalen_johansen_variance(self.r, self.d, self.d_e[k])
        z = norm.ppf(1.0 - alpha_ci / (2.0 if bound == "two-sided" else 1.0))
        se = np.sqrt(var)
        if bound_type == "normal":
            lower, upper = F - z * se, F + z * se
        else:
            # The incidence is 1 only where one cause had every failure
            # and no one is left (up to the round-off of its running sum).
            one = F >= 1.0 - 1e-12
            inside = (var > 0) & ~one
            with np.errstate(divide="ignore", invalid="ignore"):
                log_F = np.log(np.where(inside, F, 0.5))
                s = z * se / (F * np.abs(log_F))
                # On the log(-log F) scale, where -log F falls as F rises.
                lower = np.where(inside, np.exp(log_F * np.exp(s)), F)
                upper = np.where(inside, np.exp(log_F * np.exp(-s)), F)
            if np.any(one & (var > 0)):
                below = lower[~one]
                top = below.max() if below.size else 0.0
                lower = np.where(one, top, lower)
                upper = np.where(one, 1.0, upper)
        if bound == "lower":
            return [lower]
        if bound == "upper":
            return [upper]
        return [lower, upper]

    def _single_event(self, event: Any) -> Any:
        """The single-event estimate of the all-cause failures
        (``event=None``) or of one cause's (the other causes censored), by
        the fitted ``how``, with the model's support."""
        from surpyval import KaplanMeier, NelsonAalen

        fitter = KaplanMeier if self.how == "Kaplan-Meier" else NelsonAalen
        d = self.d if event is None else self.d_e[self.event_idx_map[event]]
        model = fitter.from_xrd(self.x, self.r, d)
        if self.support is not None:
            model.set_support(*self.support)
        return model

    def plot(
        self,
        stacked: bool = True,
        ax: Any = None,
        plot_bounds: bool = True,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        bound_type: str = "exp",
    ) -> Any:
        """
        Plot the cumulative incidence of every cause (#485), with its
        confidence bounds.

        As the single-event estimates' ``plot``, two-sided bounds are drawn
        as a shaded band in the colour of their curve (a one-sided bound
        as a dashed step line), unless ``plot_bounds=False``. Unstacked,
        each cause's band is its cumulative incidence's :meth:`cb`
        (``on="cif"``). Stacked, a cause's bounds do not bound its layer
        (which sits on the causes below it), so the band is drawn on the
        top of the stack, the all-cause failure probability
        :math:`1 - S` (the sum of the cumulative incidences): the
        Kaplan-Meier (Greenwood) bounds of the all-cause failures, as
        ``KaplanMeier``'s plot of the same data draws them, which is
        ``cb(x, on="ff")`` of a model fitted with ``how="Kaplan-Meier"``.

        Parameters
        ----------
        stacked : bool, optional
            Stack the causes' cumulative incidences (the default), so the
            top of the stack is the all-cause failure probability
            :math:`1 - S`; ``False`` draws each as its own step curve.
        ax : matplotlib.axes.Axes, optional
            The axes to draw on; the current axes by default.
        plot_bounds : bool, optional
            Whether to draw the confidence bounds. Defaults to True.
        alpha_ci, bound, bound_type : optional
            Passed to the confidence bound calculation; see :meth:`cb`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn on.

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> import matplotlib.pyplot as plt
        >>> from surpyval.univariate.competing_risks import CompetingRisks
        >>> x = [1, 2, 3, 4, 5, 6, 7, 8]
        >>> e = ["a", "b", "a", "b", "a", None, "a", "b"]
        >>> c = [0, 0, 0, 0, 0, 1, 0, 0]
        >>> model = CompetingRisks.fit(x, e, c=c)
        >>> fig, ax = plt.subplots()
        >>> model.plot(ax=ax).get_ylabel()
        'Cumulative incidence'
        >>> plt.close(fig)
        """
        from surpyval import KaplanMeier

        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        check_option("bound_type", bound_type, ("exp", "normal"))
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gcf().gca()
        causes = sorted(
            self.event_idx_map, key=lambda e: self.event_idx_map[e]
        )
        # Each CIF is a right-continuous step function from 0 at time 0.
        x = np.concatenate([[min(0.0, float(self.x[0]))], self.x])
        cifs = [
            np.concatenate([[0.0], self.CIF[self.event_idx_map[e]]])
            for e in causes
        ]
        labels = [str(e) for e in causes]

        def draw_bounds(cb: npt.NDArray, color: Any) -> None:
            # At 0 (before the first time) the bounds are the estimate, 0.
            if bound == "two-sided":
                cb = np.vstack([[0.0, 0.0], cb])
                ax.fill_between(
                    x,
                    cb[:, 0],
                    cb[:, 1],
                    step="post",
                    alpha=0.3,
                    color=color,
                    linewidth=0,
                )
            else:
                cb = np.concatenate([[0.0], cb])
                ax.step(x, cb, where="post", color=color, linestyle="--")

        if stacked:
            ax.stackplot(x, *cifs, labels=labels, step="post", alpha=0.7)
            if plot_bounds:
                all_cause = KaplanMeier.from_xrd(self.x, self.r, self.d)
                cb = all_cause.cb(
                    self.x,
                    on="ff",
                    alpha_ci=alpha_ci,
                    bound=bound,
                    bound_type=bound_type,
                )
                draw_bounds(cb, "k")
        else:
            for cause, cif, label in zip(causes, cifs, labels):
                (line,) = ax.step(x, cif, where="post", label=label)
                if plot_bounds:
                    cb = self.cb(
                        self.x,
                        cause,
                        alpha_ci=alpha_ci,
                        bound=bound,
                        bound_type=bound_type,
                    )
                    draw_bounds(cb, line.get_color())
        ax.set_ylim(0, 1)
        if not ax.get_xlabel():
            # "Time", as the other estimates' plots (#514)
            ax.set_xlabel("Time")
        ax.set_ylabel("Cumulative incidence")
        ax.set_title(
            "Cumulative incidence by cause" + (" (stacked)" if stacked else "")
        )
        ax.legend(title="Cause")
        return ax

    @classmethod
    @column_arguments("x", "c", "n")
    def fit_from_df(
        cls,
        df: Any,
        x_col: str,
        e_col: str,
        c_col: "str | None" = None,
        n_col: "str | None" = None,
        how: str = "Nelson-Aalen",
    ) -> "CompetingRisks":
        """
        Fit from the columns of a :class:`pandas.DataFrame`.

        Parameters
        ----------
        df : DataFrame
            The data.
        x_col : str
            The column of failure / censoring times.
        e_col : str
            The column of causes (missing for a censored row).
        c_col : str, optional
            The column of censoring flags; derived from ``e_col`` if not
            given.
        n_col : str, optional
            The column of counts.
        how : str, optional
            As for :meth:`fit`.

        Returns
        -------
        CompetingRisks
            The fitted model; the frame is kept as ``source_df``.
        """
        x, c, n, e = validate_cr_df_inputs(df, x_col, e_col, c_col, n_col)
        model = cls.fit(x, e, c, n, how)
        # Keep the source frame without shadowing the ``df`` (density)
        # method (#253).
        model.source_df = df
        return model

    @classmethod
    def fit(
        cls,
        x: npt.ArrayLike,
        e: npt.ArrayLike,
        c: "npt.ArrayLike | None" = None,
        n: "npt.ArrayLike | None" = None,
        how: str = "Nelson-Aalen",
    ) -> "CompetingRisks":
        """
        Fit the non-parametric competing-risks model.

        Parameters
        ----------
        x : array_like
            Failure or censoring times.
        e : array_like
            The cause of each failure: any hashable labels (integers,
            strings, tuples, or a mix); they are sorted to fix their order
            in ``event_idx_map`` (labels of different types by type name,
            then text). A missing value (``None``, ``NaN``) marks a
            right-censored row.
        c : array_like, optional
            Censoring flags: 0 a failure (with a cause in ``e``), 1
            right-censored (with ``e`` missing). Derived from ``e`` if not
            given. Left and interval censoring are not supported.
        n : array_like, optional
            Counts. Defaults to 1.
        how : str, optional
            The all-cause survival estimator that ``sf``, ``ff`` and ``Hf``
            report: ``"Nelson-Aalen"`` (the default, ``exp(-H)``) or
            ``"Kaplan-Meier"``. The cumulative incidence always uses the
            Kaplan-Meier survival, so the CIFs add up to the Kaplan-Meier
            failure probability either way.

        Returns
        -------
        CompetingRisks
            The fitted model.

        Examples
        --------
        >>> from surpyval.univariate.competing_risks import CompetingRisks
        >>> x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> e = ['a', 'b', 'a', None, 'a', 'b', 'a', None, 'b', 'a']
        >>> model = CompetingRisks.fit(x, e)
        >>> model.cif([5, 10], 'a').round(4)
        array([0.3167, 0.6083])
        >>> model.cif([5, 10], 'b').round(4)
        array([0.1   , 0.3917])
        """
        x, c, n, e = validate_cr_inputs(x, c, n, e, how)
        check_finite_event_times(x, c)

        # The causes in a fixed order (censored rows have no cause), the
        # same for every competing-risks class; labels of different types
        # (1 and "b") are ordered too.
        causes = ordered_labels(e)
        n_event_types = len(causes)
        event_idx_map = {state: i for i, state in enumerate(causes)}

        # Get the x, r, d format agnostic of event.
        unique_x, r, d = surv.xcnt_to_xrd(x, c, n)

        # The count of events (d) of each cause (e) at each time (x). Every
        # x is one of unique_x, so its column is found with searchsorted;
        # np.add.at sums the counts in row order, as the per-row loop with
        # ``np.where(unique_x == x_i)`` did (O(n * m): 3.6 s at 1e5, #515).
        d_e = np.zeros((n_event_types, len(unique_x)))
        events = c != 1
        cause = np.fromiter(
            (event_idx_map[label] for label in e[events]),
            dtype=np.intp,
            count=int(events.sum()),
        )
        np.add.at(
            d_e, (cause, np.searchsorted(unique_x, x[events])), n[events]
        )

        if how == "Nelson-Aalen":
            S = na(r, d)
        elif how == "Kaplan-Meier":
            S = km(r, d)

        # Useful object to return to user
        model = cls()
        model.how = how
        model.n_event_types = n_event_types
        model.event_idx_map = event_idx_map

        # Store relevant data to object
        model.x = unique_x
        model.d = d
        model.r = r
        model.h0 = d / r
        model.H0 = model.h0.cumsum()
        model.S = S
        model.d_e = d_e
        model.h0_e = d_e / r
        model.H0_e = model.h0_e.cumsum(axis=1)
        # The incidence weight must be the *product-limit* (KM) survival
        # regardless of the estimator reported as sf: only KM satisfies
        # the telescoping identity sum_j S(t-)·d_j/r = 1 - S(t), so
        # pairing the discrete hazard increment with exp(-H) inflates
        # the CIF and can push the total incidence past 1 (#278).
        S_km = S if how == "Kaplan-Meier" else km(r, d)
        model.IIF = aalen_johansen_iif(S_km, model.h0_e)
        model.CIF = model.IIF.cumsum(axis=1)
        return model
