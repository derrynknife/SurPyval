"""Plots of a Bayesian fit: the probability plot with its credible band,
the posterior against the prior, and curves drawn from a prior."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from surpyval.bayes.prior import DistributionPrior, NormalOnScale


def probability_plot(
    model: Any,
    heuristic: str,
    plot_bounds: bool,
    alpha_ci: float,
    ax: Any,
    plot_mle: bool,
    show_censored: bool,
    color: Any,
    label: str | None,
    **kwargs: Any,
) -> Any:
    from surpyval.univariate.parametric.probability_plotting import (
        adjust_heuristic,
        curve_plot_data,
        draw_probability_plot,
        probability_plot_data,
    )
    from surpyval.utils.units import axis_label
    from surpyval.utils.validation import check_alpha_ci

    check_alpha_ci(alpha_ci)
    dist = model.dist
    if dist.discrete or not hasattr(dist, "mpp_y_transform"):
        raise ValueError(
            f"{dist.name} has no probability plot; plot model.sf over a "
            "range of times instead."
        )
    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.gcf().gca()
    params = model.params

    def band(x: Any) -> Any:
        return model.cri(x, on="ff", alpha_ci=alpha_ci)

    if model.data is None:
        d = curve_plot_data(dist, model.ff, model.qf, model.gamma, params)
    else:
        data = model.data
        heuristic = adjust_heuristic(data.c, data.t, heuristic)
        d = probability_plot_data(
            dist=dist,
            ff=model.ff,
            x=data.x,
            c=data.c,
            n=data.n,
            t=data.t,
            heuristic=heuristic,
            gamma=model.gamma,
            params=params,
            cb_func=band if plot_bounds else None,
        )
    ax = draw_probability_plot(
        ax,
        d,
        lambda x: dist.mpp_y_transform(x, *params),
        lambda x: dist.mpp_inv_y_transform(x, *params),
        title=f"{dist.name} Probability Plot (Bayesian)",
        plot_bounds=plot_bounds and model.data is not None,
        show_censored=show_censored,
        color=color,
        label=label or "posterior predictive",
        xlabel=axis_label(model.unit),
        **kwargs,
    )
    mle = model.mle if plot_mle else None
    if mle is not None and "x_model" in d:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            y = dist.mpp_y_transform(mle.ff(d["x_model"]), *params)
        ax.plot(
            d["x_model"],
            y,
            linestyle="--",
            color="0.4",
            linewidth=1,
            label="maximum likelihood",
        )
    return ax


def _prior_density(model: Any, name: str, values: Any) -> Any:
    """The prior density of one parameter where it has one of its own."""
    bound = model._bound()
    p = model.priors.single.get(name)
    t = bound.transforms[bound.names.index(name)]
    with np.errstate(all="ignore"):
        if isinstance(p, DistributionPrior):
            return np.asarray(p.model.df(values), dtype=float)
        if isinstance(p, NormalOnScale):
            phi = t.to_phi(values)
            return np.exp(p.log_density_phi(phi, t) - t.log_jacobian(phi))
    return None


def posterior_plot(model: Any, fig: Any) -> Any:
    import matplotlib.pyplot as plt

    names = model.posterior.free_names
    k = len(names)
    extra = 1 if k == 2 else 0
    if fig is None:
        fig = plt.figure(figsize=(4 * (k + extra), 3.4))
    axes = fig.subplots(1, k + extra, squeeze=False)[0]
    for ax, name in zip(axes, names):
        values, dens = model.posterior.marginal(name)
        ax.fill_between(values, dens, alpha=0.4, label="posterior")
        ax.plot(values, dens, linewidth=1)
        prior = _prior_density(model, name, values)
        if prior is not None:
            ax.plot(values, prior, "k--", linewidth=1, label="prior")
        mle = model.mle
        if mle is not None and name in model.dist.parameter_names:
            v = mle.params[model.dist.parameter_names.index(name)]
            ax.axvline(v, color="0.4", linestyle=":", label="max. lik.")
        ax.set_xlabel(name)
        ax.set_yticks([])
        ax.legend(fontsize="small")
    if extra:
        ax = axes[-1]
        a, b = names
        lo_a, hi_a = model.posterior.quantile(a, [0.0005, 0.9995])
        lo_b, hi_b = model.posterior.quantile(b, [0.0005, 0.9995])
        ga = np.linspace(lo_a, hi_a, 80)
        gb = np.linspace(lo_b, hi_b, 80)
        A, B = np.meshgrid(ga, gb)
        try:
            dens = model.posterior.density(np.c_[A.ravel(), B.ravel()])
            ax.contour(A, B, dens.reshape(A.shape), levels=6)
        except ValueError:
            pts = model.posterior.points
            ia = model.posterior.parameter_names.index(a)
            ib = model.posterior.parameter_names.index(b)
            ax.scatter(pts[:, ia], pts[:, ib], s=1, alpha=0.2)
        mle = model.mle
        if mle is not None and set(names) <= set(model.dist.parameter_names):
            ia = model.dist.parameter_names.index(a)
            ib = model.dist.parameter_names.index(b)
            ax.plot(mle.params[ia], mle.params[ib], "kx", label="max. lik.")
            ax.legend(fontsize="small")
        ax.set_xlabel(a)
        ax.set_ylabel(b)
    fig.tight_layout()
    return fig


def prior_curves(model: Any, ax: Any, k: int, random_state: Any) -> Any:
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gcf().gca()
    curves = model.draw_models(k, random_state)
    lo = float(np.nanmin([m.qf(0.01) for m in curves]))
    hi = float(np.nanmax([m.qf(0.99) for m in curves]))
    if model.dist.support[0] == 0 and lo > 0:
        x = np.geomspace(lo, hi, 200)
        ax.set_xscale("log")
    else:
        x = np.linspace(lo, hi, 200)
    for m in curves:
        ax.plot(x, m.ff(x), color="C0", alpha=0.2, linewidth=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Fraction failed")
    ax.set_title(f"{k} curves drawn from the prior")
    return ax
