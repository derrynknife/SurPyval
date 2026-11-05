"""The joint likelihood-ratio confidence region of a two-parameter fit:
:meth:`Parametric.likelihood_contour` finds it, and
:func:`plot_likelihood_contour` draws one or more of them.

The region at significance level ``alpha_ci`` is every pair of parameter
values whose log-likelihood is within ``chi2_{2, 1 - alpha_ci} / 2`` of
the maximum: the pairs a likelihood-ratio test at that level does not
reject. Its boundary is the contour

    log L(theta) = log L(theta_hat) - chi2_{2, 1 - alpha_ci} / 2,

and ``chi2_{2, 1 - alpha_ci} = -2 log(alpha_ci)`` (the chi-squared
distribution with two degrees of freedom is an exponential). The contour
is found as the bounds of ``param_cb(method="lr")`` are, by bracketing
and root finding along rays from the estimate in the parameters' search
coordinates (their logs for a positive parameter), at angles evenly
spaced in the Wald metric, where the region is close to a circle. The
extremes of each parameter over the region (the points where the contour
is tangent to the axes) are found exactly, as the profile-likelihood
bounds at the same critical value, and are among the points.

matplotlib is imported only when a plot is drawn.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from surpyval import plotting
from surpyval.utils.validation import check_alpha_ci
from surpyval.utils.warnings import caller_stacklevel

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure

__all__ = [
    "LikelihoodContour",
    "LikelihoodContourPlot",
    "likelihood_contour",
    "plot_likelihood_contour",
]


@dataclass
class LikelihoodContour:
    """The boundary of the joint likelihood-ratio confidence region of the
    two parameters of a fit (:meth:`Parametric.likelihood_contour`).

    Attributes
    ----------
    distribution : str
        The distribution fitted.
    params : list of str
        The two parameters, in the order of the columns of ``points``.
    estimate : numpy.ndarray
        The maximum-likelihood estimate of the two parameters.
    alpha_ci : float
        The significance level: the region covers the true pair of values
        with probability ``1 - alpha_ci`` (in large samples).
    critical_value : float
        ``chi2_{2, 1 - alpha_ci} = -2 log(alpha_ci)``, the deviance at the
        contour.
    log_likelihood : float
        The maximum log-likelihood, at ``estimate``.
    log_likelihood_cut : float
        The log-likelihood on the contour,
        ``log_likelihood - critical_value / 2``.
    points : numpy.ndarray
        The contour, an ``(n, 2)`` array going once round the estimate;
        a row of ``nan`` where a ray from the estimate does not meet the
        boundary (the region is open on that side).
    bounds : numpy.ndarray
        ``[[lower, upper], [lower, upper]]``: the smallest and largest
        value of each parameter over the region (the edge of the
        parameter's space, such as 0 or ``inf``, where the region reaches
        it).
    closed : bool
        Whether every ray met the boundary, so ``points`` is a closed
        curve.

    Examples
    --------
    >>> import surpyval as sp
    >>> x = [61.9, 74.2, 82.2, 106.3, 76.2, 40.7, 83.2, 42.3, 50.4, 82.4]
    >>> c = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    >>> model = sp.Weibull.fit(x, c=c)
    >>> contour = model.likelihood_contour(alpha_ci=0.1, n_points=40)
    >>> contour.params
    ['alpha', 'beta']
    >>> contour.points.shape
    (44, 2)
    >>> contour.bounds.round(2)
    array([[ 68.54, 100.69],
           [  2.43,   7.87]])
    """

    distribution: str
    params: list
    estimate: npt.NDArray
    alpha_ci: float
    critical_value: float
    log_likelihood: float
    log_likelihood_cut: float
    points: npt.NDArray
    bounds: npt.NDArray
    closed: bool

    @property
    def x(self) -> npt.NDArray:
        """The first parameter's values along the contour."""
        return self.points[:, 0]

    @property
    def y(self) -> npt.NDArray:
        """The second parameter's values along the contour."""
        return self.points[:, 1]

    def __repr__(self) -> str:
        title = "Likelihood contour ({}, {:g}% joint region)".format(
            self.distribution, 100 * (1 - self.alpha_ci)
        )
        lines = [title, "=" * len(title)]
        for k, name in enumerate(self.params):
            lines.append(
                "{:<16} : {:.6g} ({:.6g}, {:.6g})".format(
                    name,
                    self.estimate[k],
                    self.bounds[k, 0],
                    self.bounds[k, 1],
                )
            )
        lines.append(f"Critical value   : {self.critical_value:.6g}")
        lines.append(f"Log-likelihood   : {self.log_likelihood:.6g}")
        lines.append(f"Contour at       : {self.log_likelihood_cut:.6g}")
        lines.append(
            "Points           : {}{}".format(
                len(self.points), "" if self.closed else " (open region)"
            )
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """The contour as a dict of native Python values, strict JSON: an
        infinite or missing number is written as ``None`` and recorded
        under ``"non_finite"``, so that :meth:`from_dict` restores it.

        Examples
        --------
        >>> import json
        >>> import surpyval as sp
        >>> model = sp.Weibull.fit([12, 25, 31, 40, 48, 60], c=[0] * 5 + [1])
        >>> d = model.likelihood_contour(n_points=12).to_dict()
        >>> d["result"], d["params"]
        ('LikelihoodContour', ['alpha', 'beta'])
        >>> isinstance(json.dumps(d, allow_nan=False), str)
        True
        """
        from surpyval.serialisation import encode_non_finite, to_native

        out: dict = {"result": type(self).__name__}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (np.ndarray, np.generic)):
                value = to_native(value)
            elif isinstance(value, list):
                value = list(value)
            out[f.name] = value
        return encode_non_finite(out)

    @classmethod
    def from_dict(cls, result_dict: dict) -> LikelihoodContour:
        """The contour :meth:`to_dict` wrote.

        Examples
        --------
        >>> import surpyval as sp
        >>> model = sp.Weibull.fit([12, 25, 31, 40, 48, 60], c=[0] * 5 + [1])
        >>> contour = model.likelihood_contour(n_points=12)
        >>> again = sp.LikelihoodContour.from_dict(contour.to_dict())
        >>> bool((again.points == contour.points).all())
        True
        """
        from surpyval.serialisation import decode_non_finite

        if result_dict.get("result") != cls.__name__:
            raise ValueError(
                "Must create a LikelihoodContour from a dict its to_dict "
                "wrote (with 'result': 'LikelihoodContour')"
            )
        values = decode_non_finite(dict(result_dict))
        names = [f.name for f in fields(cls)]
        missing = sorted(set(names) - set(values))
        if missing:
            raise ValueError(
                "the dict is missing {}".format(", ".join(missing))
            )
        kwargs = {name: values[name] for name in names}
        for name in ("estimate", "points", "bounds"):
            kwargs[name] = np.asarray(kwargs[name], dtype=float)
        kwargs["points"] = kwargs["points"].reshape(-1, 2)
        kwargs["params"] = list(kwargs["params"])
        return cls(**kwargs)


def _free_params(model: Any) -> list[int]:
    """The indices of the two parameters of ``model`` the contour is of,
    or a refusal that says why there are not two."""
    if model.method != "MLE":
        raise model._no_bounds_error()
    if getattr(model, "weighted", False):
        from surpyval.utils.weights import refuse_lr

        raise refuse_lr("A likelihood contour")
    if model.offset or model.lfp or model.zi:
        raise ValueError(
            "A likelihood contour is of two parameters, and this model "
            "fits more (an offset, a limited-failure or a zero-inflated "
            "part). Refit without it for the contour of the "
            "distribution's two parameters, or bound each parameter "
            "with param_cb(name, method='lr')."
        )
    names = list(model.dist.parameter_names)
    fixed = model._user_fixed_idx()
    free = [j for j in range(len(names)) if j not in fixed]
    if len(free) != 2:
        held = [names[j] for j in sorted(fixed)]
        why = f" ({', '.join(held)} fixed at fit time)" if held else ""
        raise ValueError(
            f"A likelihood contour needs exactly two fitted parameters; "
            f"this {model.dist.name} fit has {len(free)}{why}. With one, "
            "the region is an interval: param_cb(name, method='lr'). "
            "With three or more, refit with all but two fixed "
            "(fixed={...}) for the contour of those two."
        )
    return free


def likelihood_contour(
    model: Any, alpha_ci: float = 0.05, n_points: int = 100
) -> LikelihoodContour:
    """The contour of :meth:`Parametric.likelihood_contour`, found."""
    from ._likelihood_ratio import _LRPath

    check_alpha_ci(alpha_ci)
    if (
        isinstance(n_points, bool)
        or not isinstance(n_points, (int, np.integer))
        or n_points < 4
    ):
        raise ValueError(
            f"'n_points' must be a whole number of at least 4, not "
            f"{n_points!r}: the number of rays from the estimate along "
            "which the contour is found."
        )
    free = _free_params(model)
    model._ensure_surv_data()
    crit = float(-2.0 * np.log(alpha_ci))
    coords, _ = model._lr_coords()
    excess, u_hat, nll_hat = model._lr_ray_excess(free, crit)
    if not np.isfinite(nll_hat):
        raise ValueError(
            "The log-likelihood at the fitted parameters is not finite, "
            "so there is no likelihood contour about them."
        )
    L = model._lr_wald_factor(free)
    if L is None:
        # No positive definite covariance: rays evenly spaced in the
        # search coordinates themselves.
        L = np.eye(2) * 0.1

    def to_theta(u: npt.NDArray) -> npt.NDArray:
        return np.array([coords[j].from_u(v) for j, v in zip(free, u)])

    rows: list[tuple[float, npt.NDArray]] = []
    missed = 0
    with np.errstate(all="ignore"):
        for k in range(int(n_points)):
            angle = 2.0 * np.pi * k / n_points
            d = L @ np.array([np.cos(angle), np.sin(angle)])
            r = model._lr_crossing(excess, u_hat, d, crit)
            if r is None:
                missed += 1
                rows.append((angle, np.full(2, np.nan)))
            else:
                rows.append((angle, to_theta(u_hat + r * d)))

    # The extremes of each parameter over the region: its
    # profile-likelihood bounds at the same critical value, and the
    # tangent points there (the other parameter at its profile optimum).
    bounds = np.empty((2, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, j in enumerate(free):
            o = free[1 - k]
            for s, direction in enumerate((-1.0, 1.0)):
                v = float(model._lr_param_side(j, crit, direction))
                bounds[k, s] = v
                if not np.isfinite(v) or v in (coords[j].lo, coords[j].hi):
                    continue
                path = _LRPath()
                model._profile_neg_ll(j, v, path=path)
                if not path.u:
                    continue
                theta = np.empty(2)
                theta[k] = v
                theta[1 - k] = coords[o].from_u(float(path.u[-1][0]))
                u = np.array([coords[i].to_u(t) for i, t in zip(free, theta)])
                w = np.linalg.solve(L, u - u_hat)
                rows.append(
                    (float(np.arctan2(w[1], w[0]) % (2 * np.pi)), theta)
                )
    rows.sort(key=lambda row: row[0])
    points = np.array([row[1] for row in rows], dtype=float)
    if missed:
        warnings.warn(
            f"{missed} of the {n_points} rays from the estimate did not "
            "meet the likelihood contour (the region runs on to the edge "
            "of the parameters' space, or the likelihood could not be "
            "computed on the way); their points are nan. The region is "
            "open on that side: report the parameters' bounds "
            "(param_cb(name, method='lr')) rather than the contour.",
            RuntimeWarning,
            stacklevel=caller_stacklevel(),
        )
    names = list(model.dist.parameter_names)
    estimate = np.asarray(model.params, dtype=float)[free]
    return LikelihoodContour(
        distribution=str(model.dist.name),
        params=[names[j] for j in free],
        estimate=estimate,
        alpha_ci=float(alpha_ci),
        critical_value=crit,
        log_likelihood=float(-nll_hat),
        log_likelihood_cut=float(-nll_hat - crit / 2.0),
        points=points,
        bounds=bounds,
        closed=missed == 0,
    )


@dataclass
class LikelihoodContourPlot:
    """What :func:`plot_likelihood_contour` drew.

    Attributes
    ----------
    fig : matplotlib Figure
        The figure the axes are on.
    ax : matplotlib Axes
        The axes the contours are drawn on.
    params : list of str
        The two parameters: the x and the y axis.
    contours : dict
        ``{label: [LikelihoodContour, ...]}``: each model's contours, one
        per level of ``alpha_ci``.
    lines : dict
        ``{label: [Line2D, ...]}``: the lines drawn for them.
    estimates : dict
        ``{label: Line2D}``: each model's estimate, marked.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> import surpyval as sp
    >>> model = sp.Weibull.fit([52, 66, 81, 95, 104, 118, 131, 150])
    >>> out = sp.plot_likelihood_contour(model, alpha_ci=[0.1, 0.05])
    >>> out.params, list(out.contours)
    (['alpha', 'beta'], ['Weibull'])
    >>> out.ax.get_xlabel(), out.ax.get_ylabel()
    ('alpha', 'beta')
    >>> plt.close(out.fig)
    """

    fig: Figure | SubFigure
    ax: Axes
    params: list = field(default_factory=list)
    contours: dict = field(default_factory=dict)
    lines: dict = field(default_factory=dict)
    estimates: dict = field(default_factory=dict)


def _as_models(models: Any, labels: Any) -> list[tuple[str, Any]]:
    if isinstance(models, Mapping):
        pairs = [(str(k), v) for k, v in models.items()]
    elif isinstance(models, (list, tuple)):
        pairs = [(getattr(m.dist, "name", "Model"), m) for m in models]
    else:
        pairs = [(getattr(models.dist, "name", "Model"), models)]
    if not pairs:
        raise ValueError(
            "'models' is empty; pass a fitted two-parameter model (e.g. "
            "surpyval.Weibull.fit(x, c=c)), a list of them or a dict "
            "{label: model}."
        )
    if labels is not None:
        labels = [str(v) for v in labels]
        if len(labels) != len(pairs):
            raise ValueError(
                f"'labels' has {len(labels)} labels for {len(pairs)} "
                "models; give one label per model."
            )
        pairs = [(lab, m) for lab, (_, m) in zip(labels, pairs)]
    if len({lab for lab, _ in pairs}) < len(pairs):
        pairs = [(f"Model {k + 1}", m) for k, (_, m) in enumerate(pairs)]
    for lab, m in pairs:
        if not callable(getattr(m, "likelihood_contour", None)):
            raise TypeError(
                f"'models': {lab!r} ({type(m).__name__}) has no likelihood "
                "contour; pass fitted parametric distributions, e.g. "
                "surpyval.Weibull.fit(x, c=c)."
            )
    return pairs


# The line styles of successive levels of one model.
_STYLES = ("-", "--", ":", "-.")


def plot_likelihood_contour(
    models: Any,
    alpha_ci: float | Sequence[float] = 0.05,
    ax: Axes | None = None,
    *,
    n_points: int = 100,
    labels: Sequence[str] | None = None,
    colors: Any = None,
    title: str | None = None,
    legend: bool | None = None,
) -> LikelihoodContourPlot:
    """
    Draw the joint likelihood-ratio confidence region of the two
    parameters of one or more fits, with each estimate marked.

    Every pair of parameter values inside a contour is one that a
    likelihood-ratio test at that level does not reject: the region
    covers the true pair with probability ``1 - alpha_ci``. Several
    levels can be drawn for one fit (nested contours), and several fits
    on the same axes: two regions that do not overlap are evidence that
    the two samples (two designs, suppliers or test conditions) come from
    different distributions. See :meth:`Parametric.likelihood_contour`.

    Parameters
    ----------
    models : model, list or dict
        One fitted two-parameter model (e.g. a ``Weibull.fit``), a list of
        them, or ``{label: model}``. They must have the same two
        parameters.
    alpha_ci : float or list of float, optional
        The significance level of each contour drawn. Defaults to 0.05,
        the 95% joint region.
    ax : matplotlib Axes, optional
        The axes to draw on. Without one a new figure is made.
    n_points : int, optional
        The number of rays from the estimate along which each contour is
        found. Defaults to 100.
    labels : list of str, optional
        One label per model, in order; the keys of a dict otherwise, and
        the distribution's name (or "Model 1", "Model 2", ... when names
        repeat) for a list.
    colors : list or dict, optional
        One matplotlib colour per model, or ``{label: colour}``. Without
        them the models take SurPyval's palette in order.
    title : str, optional
        The title.
    legend : bool, optional
        Whether to draw a legend. Defaults to one when there are two or
        more models or levels.

    Returns
    -------
    LikelihoodContourPlot
        ``fig`` and ``ax`` to customise further, the ``contours`` drawn
        and the artists.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> import surpyval as sp
    >>> old = sp.Weibull.fit([52, 66, 81, 95, 104, 118, 131, 150])
    >>> new = sp.Weibull.fit([88, 102, 115, 127, 139, 156, 170, 191])
    >>> out = sp.plot_likelihood_contour(
    ...     {"Old design": old, "New design": new}, alpha_ci=[0.1, 0.05]
    ... )
    >>> out.params
    ['alpha', 'beta']
    >>> [len(v) for v in out.contours.values()]
    [2, 2]
    >>> plt.close(out.fig)
    """
    levels = [float(a) for a in np.atleast_1d(np.asarray(alpha_ci))]
    if not levels:
        raise ValueError(
            "'alpha_ci' is empty; give a significance level, e.g. 0.05, "
            "or a list of them."
        )
    for level in levels:
        check_alpha_ci(level)
    pairs = _as_models(models, labels)
    found = {
        label: [
            model.likelihood_contour(alpha_ci=a, n_points=n_points)
            for a in levels
        ]
        for label, model in pairs
    }
    params = next(iter(found.values()))[0].params
    for label, contours in found.items():
        if contours[0].params != params:
            raise ValueError(
                f"'models': {label!r} has the parameters "
                f"{contours[0].params}, the first model {params}; the "
                "contours of fits drawn together must be of the same two "
                "parameters (fit the same distribution)."
            )
    return _draw(found, params, levels, ax, colors, title, legend)


def _color(colors: Any, k: int, label: str) -> Any:
    if colors is None:
        return None
    if isinstance(colors, Mapping):
        return colors.get(label)
    return list(colors)[k]


def _draw(
    found: dict[str, list[LikelihoodContour]],
    params: list,
    levels: list[float],
    ax: Axes | None,
    colors: Any,
    title: str | None,
    legend: bool | None,
) -> LikelihoodContourPlot:
    import matplotlib.pyplot as plt

    fig: Figure | SubFigure
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    else:
        fig = ax.figure
    plotting.style_axes(ax)
    out = LikelihoodContourPlot(fig=fig, ax=ax, params=list(params))
    many = len(found) > 1
    for k, (label, contours) in enumerate(found.items()):
        color = plotting.next_color(ax, _color(colors, k, label))
        if color is None:  # the theme is off: matplotlib's colours
            color = f"C{k % 10}"
        out.contours[label] = contours
        out.lines[label] = []
        for s, contour in enumerate(contours):
            pts = contour.points
            if contour.closed:
                pts = np.vstack([pts, pts[:1]])
            level = f"{100 * (1 - contour.alpha_ci):g}%"
            (line,) = ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=color,
                linestyle=_STYLES[s % len(_STYLES)],
                linewidth=plotting.LINE_WIDTH,
                label=f"{label}, {level}" if many else level,
            )
            out.lines[label].append(line)
        est = contours[0].estimate
        (out.estimates[label],) = ax.plot(
            [est[0]],
            [est[1]],
            "o",
            color=color,
            markersize=plotting.MARKER_SIZE,
            markeredgecolor=plotting.INK["surface"],
            linestyle="none",
            label=f"{label}, estimate" if many else "Estimate",
        )
    ax.set_xlabel(params[0])
    ax.set_ylabel(params[1])
    if title is not None:
        ax.set_title(title)
    if legend if legend is not None else (many or len(levels) > 1):
        plotting.style_legend(ax, fontsize="small")
    return out
