"""
Shared frailty for repairable systems: a Poisson process whose intensity
each cluster multiplies by its own unobserved factor.

Repairable units of one kind usually differ by more than their covariates
explain: some simply fail more often. A plain non-homogeneous Poisson
process (``CrowAMSAA``, ``Duane``, ``CoxLewis``) or a
proportional-intensity model (``ProportionalIntensityNHPP``) fitted to
such a fleet gets the average intensity about right, but its standard
errors and its forecast intervals are too narrow, because the counts vary
more from unit to unit than Poisson counts do. The shared-frailty model
gives each cluster ``g`` (a unit, by default, or a group of units: a
fleet, a site) one frailty ``u_g`` that multiplies the intensity of every
unit ``j`` in it,

.. math::
    \\lambda_j(t \\mid u_g) = u_g \\, \\lambda_0(t) \\, e^{\\beta' Z_j},

with ``u`` drawn from a gamma distribution of mean 1 and variance
``theta``, a log-normal one (``log u ~ N(0, theta)``), an inverse Gaussian
one of mean 1 and variance ``theta`` (``"inverse_gaussian"``), or a
positive stable one of index ``1 - theta`` (``"positive_stable"``: Laplace
transform ``exp(-s^(1 - theta))``, no finite mean, ``theta`` Kendall's
tau). ``theta = 0`` is the model without frailty.

A cluster with ``n`` events at times ``t_1 .. t_n`` and cumulative
intensity ``L`` over its observation (at ``u = 1``; delayed entry,
observation windows and covariates all enter through ``L``) has the
marginal likelihood :math:`\\prod_k \\lambda(t_k) \\, E[u^n e^{-uL}]`. For
the gamma frailty the expectation has a closed form,

.. math::
    \\log E[u^n e^{-uL}] = \\sum_{k=0}^{n-1} \\log(1 + k\\theta)
    - (n + 1/\\theta) \\log(1 + \\theta L),

the negative binomial likelihood of the count, written so that it keeps
its digits as ``theta -> 0``. For the log-normal frailty it is the
integral the joint frailty model takes
(:func:`~surpyval.recurrent.joint_frailty.log_frailty_integral`), and for
the inverse Gaussian and positive stable frailties the closed forms of
:mod:`surpyval.univariate.regression.frailty.families`.

Under the positive stable frailty the clusters' mean count is infinite
(the frailty has no mean), so the population's ``cif``, ``iif`` and
``mcf`` are refused; a cluster's given its frailty (``frailty=``), and
the fitted clusters' forecasts given their histories, are finite.

The fits are reached through ``frailty=`` on the fitters:
``CrowAMSAA.fit(x, i, c, frailty="gamma")``; :class:`SharedFrailtyModel`
is what they return.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
from autograd import hessian as _ag_hessian
from autograd.extend import defvjp, primitive
from autograd.tracer import getval
from numpy.typing import ArrayLike

from surpyval.recurrent._bounded import unconstraining_maps
from surpyval.recurrent.inference import (
    LikelihoodInferenceMixin,
    check_alpha_ci,
)
from surpyval.recurrent.joint_frailty import (
    _is_maximum,
    _minimise,
    log_frailty_integral,
)
from surpyval.recurrent.regression.proportional_intensity import (
    alias_covariates,
)
from surpyval.recurrent.serialisation import intensity_dist_by_name
from surpyval.serialisation import (
    SerialisableMixin,
    dict_numbers,
    require_model_tag,
    stamp_schema,
)
from surpyval.univariate.regression._aliasing import fit_columns
from surpyval.univariate.regression.frailty.families import (
    LAPLACE_FAMILIES,
    laplace_log_integral,
    whole_counts,
)
from surpyval.utils.covariates import coefficient_names
from surpyval.utils.no_maximum import (
    maximum_entry,
    restored_maximum,
    warn_unverified,
)
from surpyval.utils.recurrent_data import notes_text
from surpyval.utils.recurrent_utils import item_label, validate_nhpp_data
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.validation import check_option
from surpyval.utils.warnings import caller_stacklevel, ignoring_warnings

__all__ = [
    "SharedFrailtyModel",
    "frailty_moments",
    "mixed_poisson_log_moment",
]

FAMILIES = ("gamma", "lognormal", "inverse_gaussian", "positive_stable")
BOUNDS = ("two-sided", "lower", "upper")
#: Below this the frailty variance is taken as 0 (the model without
#: frailty): the two differ by less than ``theta`` times the score there.
_THETA_ZERO = 1e-12


# ---------------------------------------------------------------------------
# The frailty expectation
# ---------------------------------------------------------------------------
def mixed_poisson_log_moment(
    D: ArrayLike, A: ArrayLike, theta: float, family: str = "gamma"
) -> npt.NDArray:
    """
    ``log E[u^D e^{-A u}]`` for a frailty ``u`` of variance ``theta``,
    elementwise: what a cluster with ``D`` events and cumulative intensity
    ``A`` (at ``u = 1``) contributes to the marginal likelihood beside its
    events' intensities.

    Gamma (mean 1, variance ``theta``): the closed form
    ``sum_{k<D} log(1 + k theta) - (D + 1/theta) log(1 + theta A)``.
    Log-normal (``log u ~ N(0, theta)``): the quadrature of
    :func:`~surpyval.recurrent.joint_frailty.log_frailty_integral`.
    Inverse Gaussian and positive stable: the closed forms of
    :func:`~surpyval.univariate.regression.frailty.families.inverse_gaussian_log_integral`
    and
    :func:`~surpyval.univariate.regression.frailty.families.positive_stable_log_integral`.
    ``theta = 0`` is no frailty: ``-A``.

    Parameters
    ----------
    D : array_like of int
        Counts of events, whole numbers of at least 0.
    A : array_like
        Cumulative intensities, at least 0; broadcast with ``D``.
    theta : float
        The frailty variance (of ``log u`` for the log-normal).
    family : {'gamma', 'lognormal', 'inverse_gaussian', 'positive_stable'}
        The frailty distribution. Default ``'gamma'``.

    Returns
    -------
    numpy.ndarray
        The log-expectation, of the broadcast shape.

    Examples
    --------
    With no events it is the log of the gamma frailty's Laplace
    transform, ``-log(1 + theta A) / theta``:

    >>> import numpy as np
    >>> from surpyval.recurrent.shared_frailty import (
    ...     mixed_poisson_log_moment,
    ... )
    >>> value = mixed_poisson_log_moment(0, 3.0, 0.5)
    >>> bool(np.isclose(value, -np.log(1 + 0.5 * 3.0) / 0.5))
    True
    >>> # no frailty: a Poisson count's exp(-A)
    >>> float(mixed_poisson_log_moment(2, 3.0, 0.0))
    -3.0
    """
    check_option("family", family, FAMILIES)
    Dv, Av = np.broadcast_arrays(
        np.asarray(D, dtype=float), np.asarray(A, dtype=float)
    )
    theta = float(theta)
    if not theta >= _THETA_ZERO:
        # (a nan theta gives nan)
        return -1.0 * Av + 0.0 * Dv + 0.0 * theta
    if family in LAPLACE_FAMILIES:
        with np.errstate(divide="ignore", over="ignore"):
            return np.asarray(
                laplace_log_integral(family, Dv, Av, theta), dtype=float
            )
    if family == "lognormal":
        # In blocks: the rule holds every entry's nodes at once.
        flat_d, flat_a = Dv.reshape(-1), Av.reshape(-1)
        out = np.empty(flat_d.size)
        for at in range(0, flat_d.size, 2048):
            block = slice(at, at + 2048)
            out[block] = log_frailty_integral(
                flat_d[block],
                flat_a[block],
                0.0 * flat_a[block],
                1.0,
                theta,
                "lognormal",
            )
        return out.reshape(Dv.shape)
    top = int(np.nanmax(Dv, initial=0.0))
    table = np.concatenate(
        [[0.0], np.cumsum(np.log1p(theta * np.arange(top)))]
    )
    whole = np.where(np.isfinite(Dv), Dv, 0.0).astype(int)
    with np.errstate(invalid="ignore"):
        tail = np.log1p(theta * Av)
        return table[whole] - Dv * tail - tail / theta + 0.0 * Dv


def frailty_moments(
    n: npt.NDArray, A: npt.NDArray, theta: float, family: str
) -> tuple[npt.NDArray, npt.NDArray]:
    """The mean and variance of the frailty of clusters with ``n`` events
    and cumulative intensity ``A``, given those (``n = A = 0``: the
    frailty distribution itself).

    Examples
    --------
    >>> from surpyval.recurrent.shared_frailty import frailty_moments
    >>> mean, variance = frailty_moments([0.0], [0.0], 0.5, "gamma")
    >>> float(mean[0]), float(variance[0])
    (1.0, 0.5)
    """
    n = np.asarray(n, dtype=float)
    A = np.asarray(A, dtype=float)
    if theta < _THETA_ZERO:
        return np.ones(n.shape), np.zeros(n.shape)
    if family == "gamma":
        shape, rate = n + 1.0 / theta, A + 1.0 / theta
        return shape / rate, shape / rate**2
    base = mixed_poisson_log_moment(n, A, theta, family)
    m1 = np.exp(mixed_poisson_log_moment(n + 1, A, theta, family) - base)
    m2 = np.exp(mixed_poisson_log_moment(n + 2, A, theta, family) - base)
    return m1, np.maximum(m2 - m1**2, 0.0)


def _frailty_quantiles(
    n: npt.NDArray,
    A: npt.NDArray,
    theta: float,
    family: str,
    probs: tuple[float, ...],
) -> npt.NDArray:
    """Quantiles (one column per entry of ``probs``) of each cluster's
    frailty given its ``n`` events and cumulative intensity ``A``."""
    n = np.asarray(n, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=float).reshape(-1)
    out = np.ones((n.size, len(probs)))
    if theta < _THETA_ZERO:
        return out
    if family == "gamma":
        from scipy.stats import gamma

        shape, rate = n + 1.0 / theta, A + 1.0 / theta
        for k, q in enumerate(probs):
            out[:, k] = gamma.ppf(q, shape, scale=1.0 / rate)
        return out
    if family == "inverse_gaussian":
        # The posterior is generalised inverse Gaussian: density
        # proportional to u^(n - 3/2) exp(-(a u + b / u) / 2), a = 1/theta
        # + 2A, b = 1/theta; scipy's geninvgauss(p, sqrt(ab)) scaled by
        # sqrt(b / a).
        from scipy.stats import geninvgauss

        a, b = 1.0 / theta + 2.0 * A, 1.0 / theta
        for k, q in enumerate(probs):
            out[:, k] = geninvgauss.ppf(
                q, n - 0.5, np.sqrt(a * b), scale=np.sqrt(b / a)
            )
        return out
    if family == "positive_stable":
        # The positive stable density has no closed form: no interval.
        return np.full((n.size, len(probs)), np.nan)
    # Log-normal: the density of w = log u given the history is
    # proportional to exp(n w - A e^w - w^2 / (2 theta)), log-concave;
    # its distribution function on a grid about its mode.
    w = np.zeros(n.size)
    for _ in range(200):
        g1 = n - A * np.exp(w) - w / theta
        g2 = -A * np.exp(w) - 1.0 / theta
        step = np.clip(-g1 / g2, -2.0, 2.0)
        w = w + step
        if np.all(np.abs(step) < 1e-12 * (1.0 + np.abs(w))):
            break
    s = 1.0 / np.sqrt(A * np.exp(w) + 1.0 / theta)
    grid = np.linspace(-12.0, 12.0, 4801)
    nodes = w[:, None] + s[:, None] * grid
    with np.errstate(over="ignore", under="ignore"):
        h = (
            n[:, None] * nodes
            - A[:, None] * np.exp(nodes)
            - nodes**2 / (2.0 * theta)
        )
        dens = np.exp(h - h.max(axis=1, keepdims=True))
    cdf = np.cumsum(0.5 * (dens[:, 1:] + dens[:, :-1]), axis=1)
    cdf = np.column_stack([np.zeros(n.size), cdf]) / cdf[:, -1:]
    for row in range(n.size):
        out[row] = np.exp(np.interp(probs, cdf[row], nodes[row]))
    return out


# ---------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------
@primitive
def _segment_sum(values: Any, index: npt.NDArray, size: int) -> Any:
    """The sum of ``values`` within each segment ``index`` names."""
    return np.bincount(index, weights=values, minlength=size)


defvjp(
    _segment_sum,
    lambda ans, values, index, size: lambda g: g[index],
    argnums=(0,),
)


def cluster_of_items(
    cluster: Any, items: Any, n_rows: int, how: str
) -> dict | None:
    """
    ``cluster`` as a mapping from each item to its cluster: ``None`` (each
    item its own cluster), a mapping ``{item: cluster}``, or one label
    per row of the data (the same on every row of an item), with the
    rows' ``items``.

    Parameters
    ----------
    cluster : None, mapping or array_like
        As ``fit`` takes it.
    items : array_like or None
        The item of each row, where ``cluster`` may be one label per row.
    n_rows : int
        The number of rows.
    how : str
        How to give the clusters where an array cannot be read, for the
        refusal.

    Returns
    -------
    dict or None
        ``{item: cluster}``.

    Examples
    --------
    >>> from surpyval.recurrent.shared_frailty import cluster_of_items
    >>> cluster_of_items(["a", "a", "b"], [1, 1, 2], 3, "")
    {1: 'a', 2: 'b'}
    """
    if cluster is None:
        return None
    if isinstance(cluster, Mapping):
        return dict(cluster)
    labels = np.asarray(cluster, dtype=object).reshape(-1)
    if items is None or labels.size != n_rows:
        raise ValueError(
            "'cluster' must be a mapping {item: cluster}, or one label per "
            f"row of x ({n_rows} rows) given with the rows' items i; got "
            f"{labels.size} label(s)"
            + ("." if items is not None else " and no i.")
            + how
        )
    out: dict = {}
    for item, label in zip(np.asarray(items, dtype=object).tolist(), labels):
        if item in out and out[item] != label:
            raise ValueError(
                f"'cluster' changes within item {item_label(item)} "
                f"({out[item]!r} and {label!r}): an item belongs to one "
                "cluster, so give the same label on each of its rows."
            )
        out[item] = label
    return out


class _Layout:
    """The rows of an event history as the frailty likelihood reads them:
    each row's time and the time before it, whether it is an event, its
    cluster, and each item's cluster, age and covariates."""

    def __init__(self, data: Any, cluster: dict | None, family: str) -> None:
        x = np.asarray(data.x, dtype=float)
        c = np.asarray(data.c)
        if x.ndim != 1 or np.any((c == -1) | (c == 2)):
            raise ValueError(
                f"frailty={family!r} needs the time of every event: counts "
                "of events in intervals (c=-1 or c=2 rows) are not taken "
                "with a frailty. PanelCountRegression fits counts between "
                "inspections (without a frailty; its robust errors allow "
                "for units that differ), or leave out 'frailty'."
            )
        if np.any(np.isfinite(np.asarray(data.tr, dtype=float))):
            raise ValueError(
                f"frailty={family!r} does not take right truncation "
                "('tr', or the right column of 't'): the items would be "
                "those selected for having failed, which the frailty fit "
                "does not model. End each item's observation with a c=1 "
                "row instead, or leave out 'frailty'."
            )
        rows = np.asarray(data.i, dtype=object)
        window_map = getattr(data, "window_map", None)
        if window_map is not None:
            # Each observation window is a row group of its own: the
            # frailty is its item's.
            rows = np.array(
                [window_map[r][0] for r in rows.tolist()], dtype=object
            )
        items = list(dict.fromkeys(rows.tolist()))
        if cluster is None:
            of_item = {item: item for item in items}
        else:
            missing = [it for it in items if it not in cluster]
            if missing:
                raise ValueError(
                    "'cluster' names no cluster for {} of the {} items "
                    "({}{}): give every item's cluster.".format(
                        len(missing),
                        len(items),
                        ", ".join(item_label(m) for m in missing[:5]),
                        ", ..." if len(missing) > 5 else "",
                    )
                )
            of_item = {item: cluster[item] for item in items}
        labels = list(dict.fromkeys(of_item.values()))
        try:
            labels = sorted(labels)
        except TypeError:
            pass
        if len(labels) < 2:
            raise ValueError(
                f"frailty={family!r} needs at least 2 clusters to estimate "
                f"how much they differ; the data has {len(labels)} "
                "(by default each item is a cluster). Give several items "
                "(i), name their clusters with 'cluster', or leave out "
                "'frailty'."
            )
        position = {label: k for k, label in enumerate(labels)}
        self.labels = labels
        self.item_cluster = of_item
        self.group = np.array(
            [position[of_item[item]] for item in rows.tolist()], dtype=int
        )
        self.x = x
        self.prev = np.asarray(data.get_previous_x(), dtype=float)
        self.event = c == 0
        self.n_clusters = len(labels)
        self.n_events = np.bincount(
            self.group[self.event], minlength=self.n_clusters
        ).astype(float)
        #: Every k of sum_{k < n} log(1 + k theta), over the clusters.
        self.steps = np.concatenate(
            [np.arange(n) for n in self.n_events.astype(int)] + [np.zeros(0)]
        ).astype(float)
        Z = getattr(data, "Z", None)
        self.Z = (
            np.zeros((x.size, 0))
            if Z is None
            else np.asarray(Z, dtype=float).reshape(x.size, -1)
        )


class _Likelihood:
    """The negative log marginal likelihood of the natural parameters
    ``[*baseline, theta, *coefficients]``, differentiable by autograd. A
    class rather than a closure, so that a fitted model pickles."""

    def __init__(
        self, dist: Any, layout: _Layout, Z: npt.NDArray, family: str
    ) -> None:
        self.dist = dist
        self.layout = layout
        self.Z = np.asarray(Z, dtype=float)
        self.family = family
        self.k = len(dist.parameter_names)
        lower = getattr(dist, "support", (-np.inf, np.inf))[0]
        # At the edge of its support (age 0 of a power law) the
        # cumulative intensity is 0, and its derivatives are not taken.
        self.end_in = (
            layout.x != lower if np.isfinite(lower) else np.ones_like(layout.x)
        ).astype(float)
        self.start_in = (
            layout.prev != lower
            if np.isfinite(lower)
            else np.ones_like(layout.prev)
        ).astype(float)
        self.end_at = np.where(self.end_in > 0, layout.x, 1.0)
        self.start_at = np.where(self.start_in > 0, layout.prev, 1.0)

    def parts(self, params: Any) -> tuple[Any, Any]:
        """The sum of the events' log-intensities, and each cluster's
        cumulative intensity at ``u = 1``."""
        d = self.layout
        base = [params[j] for j in range(self.k)]
        p = self.Z.shape[1]
        if p:
            eta = anp.dot(self.Z, params[self.k + 1 :])
            weight = anp.exp(eta)
        else:
            eta = 0.0 * d.x
            weight = 1.0
        delta = self.end_in * self.dist.cif(
            self.end_at, *base
        ) - self.start_in * self.dist.cif(self.start_at, *base)
        A = _segment_sum(weight * delta, d.group, d.n_clusters)
        events = d.event
        const = anp.sum(self.dist.log_iif(d.x[events], *base)) + anp.sum(
            eta[events]
        )
        return const, A

    def __call__(self, params: Any) -> Any:
        theta = params[self.k]
        const, A = self.parts(params)
        d = self.layout
        tv = float(getval(theta))
        if not tv >= _THETA_ZERO:
            ll = const - anp.sum(A) + 0.0 * theta
        elif self.family in LAPLACE_FAMILIES:
            ll = const + anp.sum(
                laplace_log_integral(self.family, d.n_events, A, theta)
            )
        elif self.family == "gamma":
            tail = anp.log1p(theta * A)
            ll = (
                const
                + anp.sum(anp.log1p(theta * d.steps))
                - anp.sum(d.n_events * tail)
                - anp.sum(tail) / theta
            )
        else:
            ll = const + anp.sum(
                log_frailty_integral(
                    d.n_events, A, 0.0 * A, 1.0, theta, "lognormal"
                )
            )
        return -ll

    def exposure(self, params: npt.NDArray) -> npt.NDArray:
        """Each cluster's cumulative intensity at ``u = 1``."""
        with np.errstate(all="ignore"):
            return np.asarray(getval(self.parts(params)[1]), dtype=float)


class _FloatLikelihood:
    """The negative log-likelihood as a float of a plain array: what the
    likelihood-ratio searches call."""

    def __init__(self, lik: _Likelihood) -> None:
        self.lik = lik

    def __call__(self, params: Any) -> float:
        with np.errstate(all="ignore"):
            return float(self.lik(np.asarray(params, dtype=float)))


class _Search:
    """The negative log-likelihood on the search scale: the baseline's
    parameters unconstrained and in their own units, the log of
    ``theta`` (left out where it is held at 0), then the coefficients."""

    def __init__(
        self,
        lik: _Likelihood,
        bounds: list,
        units: npt.NDArray,
        no_frailty: bool = False,
    ) -> None:
        self.lik = lik
        self.k = lik.k
        self.units = np.asarray(units, dtype=float)
        self.no_frailty = no_frailty
        self.to_natural, self.to_search = unconstraining_maps(list(bounds))

    def natural(self, u: Any) -> Any:
        k = self.k
        base = self.to_natural(u[:k] * self.units)
        out = [base[j] for j in range(k)]
        if self.no_frailty:
            out.append(0.0 * u[0])
            rest = u[k:]
        else:
            if self.lik.family == "positive_stable":
                # theta = 1 - alpha in (0, 1): logit
                out.append(1.0 / (1.0 + anp.exp(-u[k])))
            else:
                out.append(anp.exp(u[k]))
            rest = u[k + 1 :]
        out += [rest[j] for j in range(self.lik.Z.shape[1])]
        return anp.array(out)

    def search(self, natural: npt.NDArray) -> npt.NDArray:
        nat = np.asarray(natural, dtype=float)
        k = self.k
        base = self.to_search(nat[:k]) / self.units
        if self.no_frailty:
            theta = []
        elif self.lik.family == "positive_stable":
            theta = [np.log(nat[k] / (1.0 - nat[k]))]
        else:
            theta = [np.log(nat[k])]
        return np.r_[base, theta, nat[k + 1 :]]

    def __call__(self, u: Any) -> Any:
        with np.errstate(all="ignore"):
            return self.lik(self.natural(u))


def _baseline_start(data: Any, dist: Any) -> npt.NDArray:
    """The baseline of the fit without covariates or frailty."""
    from surpyval.recurrent.regression.nhpp_proportional_intensity import (
        ProportionalIntensityNHPP,
    )

    with ignoring_warnings():
        return np.asarray(
            ProportionalIntensityNHPP._baseline_start(data, dist), dtype=float
        )


def fit_shared_frailty(
    dist: Any,
    data: Any,
    family: str,
    cluster: Any = None,
    init: ArrayLike | None = None,
    regression: bool = False,
) -> SharedFrailtyModel:
    """
    Fit a shared-frailty Poisson process with the baseline intensity
    ``dist`` to an event history: what ``fit(..., frailty=family)`` of
    ``CrowAMSAA``, ``Duane``, ``CoxLewis`` and
    ``ProportionalIntensityNHPP`` calls.

    Parameters
    ----------
    dist : CountingProcess
        The baseline intensity (``CrowAMSAA``, ``Duane``, ``CoxLewis``).
    data : EventHistoryData
        The event history, with ``Z`` for a regression.
    family : {'gamma', 'lognormal'}
        The frailty distribution.
    cluster : mapping or array_like, optional
        Each item's cluster, ``{item: cluster}`` or one label per row of
        the data; by default each item is its own cluster.
    init : array_like, optional
        Starting values, in the order of the model's
        ``parameter_names``: the baseline's parameters, ``theta``, the
        coefficients. The default start is tried as well.
    regression : bool, optional
        Whether the fit is a proportional-intensity one (its predictions
        then need ``Z``).

    Returns
    -------
    SharedFrailtyModel
        The fitted model.

    Examples
    --------
    >>> import surpyval
    >>> from surpyval.recurrent import CrowAMSAA
    >>> from surpyval.recurrent.shared_frailty import fit_shared_frailty
    >>> import numpy as np
    >>> counts = np.array([1, 1, 2, 3, 8, 11])
    >>> x = np.concatenate(
    ...     [np.r_[60 * (np.arange(n) + 0.5) / n, 60] for n in counts]
    ... )
    >>> i = np.repeat(np.arange(1, 7), counts + 1)
    >>> c = (x == 60).astype(int)
    >>> data = surpyval.handle_xicn(x, i, c)
    >>> model = fit_shared_frailty(CrowAMSAA, data, "gamma")
    >>> model.parameter_names
    ['alpha', 'beta', 'theta']
    """
    check_option("frailty", family, FAMILIES)
    validate_nhpp_data(data, dist)
    of_item = cluster_of_items(
        cluster,
        getattr(data, "i", None),
        len(np.asarray(data.c)),
        " With a SurpyvalData or an EventHistoryData as x, give the "
        "mapping.",
    )
    layout = _Layout(data, of_item, family)
    if family in LAPLACE_FAMILIES:
        whole_counts(layout.n_events, family)
    k = len(dist.parameter_names)
    p = layout.Z.shape[1]
    names = fit_columns() if regression else None
    aliased = np.asarray(
        (
            alias_covariates(
                layout.Z, intercept=getattr(dist, "has_scale", False)
            )
            if p
            else []
        ),
        dtype=int,
    )
    kept = np.setdiff1d(np.arange(p), aliased)
    q = kept.size
    Zk = layout.Z[:, kept]
    # Each coefficient searched in its covariate's own scale
    sd = Zk.std(axis=0) if q else np.ones(0)
    sd = np.where(sd > 0, sd, 1.0)
    lik_s = _Likelihood(dist, layout, Zk / sd, family)
    units = (
        np.asarray(dist._search_units(data), dtype=float)
        if hasattr(dist, "_search_units")
        else np.ones(k)
    )
    bounds = list(getattr(dist, "bounds", [(None, None)] * k))
    n_obs = float(layout.n_events.sum())

    # The model without frailty, from the baseline's own fit
    null = _Search(lik_s, bounds, units, no_frailty=True)
    start = np.r_[_baseline_start(data, dist), 0.0, np.zeros(q)]
    res0 = _minimise(null, null.search(start))
    null_nat = np.asarray(getval(null.natural(res0.x)), dtype=float)
    ll_null = -float(res0.fun)
    if not np.isfinite(ll_null) or res0.fun >= 1e300:
        raise ValueError(
            "The likelihood of the model without frailty could not be "
            "evaluated on these data, so the frailty fit has no start; "
            "check the data (the fit without 'frailty' says what is wrong "
            "with them)."
        )

    # A start for theta from the clusters' counts about their means
    A0 = lik_s.exposure(null_nat)
    spread = float(np.sum((layout.n_events - A0) ** 2 - layout.n_events))
    moment = float(np.clip(spread / max(np.sum(A0**2), 1e-300), 0.02, 5.0))
    if family == "positive_stable":
        # (1 - alpha, below 1: a start of Kendall's tau)
        moment = min(moment, 0.5)
    starts = []
    for theta0 in (moment, 0.5):
        s = null_nat.copy()
        s[k] = theta0
        starts.append(s)
    if abs(np.log(moment / 0.5)) < np.log(2.0):
        starts.pop()
    if init is not None:
        given = np.asarray(init, dtype=float).reshape(-1)
        if given.size != k + 1 + p:
            raise ValueError(
                f"'init' must have {k + 1 + p} values, in the order of "
                f"parameter_names ({k} baseline parameters, theta, then "
                f"{p} coefficients); got {given.size}."
            )
        if not np.all(np.isfinite(given)) or given[k] <= 0:
            raise ValueError(
                "'init' must be finite, with theta (after the baseline's "
                "parameters) positive."
            )
        if family == "positive_stable" and not given[k] < 1:
            raise ValueError(
                "'init' gives theta = {:g}; the positive stable frailty's "
                "theta is 1 - alpha, below 1.".format(given[k])
            )
        s = np.r_[given[: k + 1], given[k + 1 :][kept] * sd]
        starts.insert(0, s)
    search = _Search(lik_s, bounds, units)
    best = None
    for s in starts:
        with np.errstate(all="ignore"):
            u0 = search.search(s)
        if not np.all(np.isfinite(u0)):
            continue
        res = _minimise(search, u0)
        if best is None or (np.isfinite(res.fun) and res.fun < best.fun):
            best = res
    std = null_nat.copy()
    ll_full = -np.inf
    verified = False
    if best is not None and np.isfinite(best.fun) and best.fun < 1e300:
        verified = _is_maximum(search, best.x, n_obs)
        std = np.asarray(getval(search.natural(best.x)), dtype=float)
        ll_full = -float(best.fun)
    boundary = False
    if not (ll_full > ll_null + 1e-8) or std[k] < 1e-6:
        # theta at its boundary 0: the model without frailty.
        boundary = True
        std = null_nat.copy()
        std[k] = 0.0
        verified = _is_maximum(null, res0.x, n_obs)
        warnings.warn(
            "The frailty variance theta is estimated at its boundary 0: "
            f"the {layout.n_clusters} clusters' counts differ no more "
            "than those of a Poisson process without frailty would, so "
            "the fit is that of the model without frailty (theta = 0, "
            "its standard error and bounds nan, every frailty 1). "
            "Nothing needs to be done; the fit without 'frailty' gives "
            "the same intensity.",
            UserWarning,
            stacklevel=caller_stacklevel(),
        )
    params = np.full(k + 1 + p, np.nan)
    params[: k + 1] = std[: k + 1]
    params[k + 1 + kept] = std[k + 1 :] / sd

    out = SharedFrailtyModel()
    out.dist = dist
    out.family = family
    out.params = params
    out.n_covariates = p
    out.regression = bool(regression)
    out.feature_names = names
    out.how = "MLE"
    out.maximum = "verified" if verified else "unverified"
    if not verified:
        warn_unverified("The shared-frailty fit")
    lik = _Likelihood(dist, layout, layout.Z, family)
    out.data = data
    out._lik = lik
    out._neg_ll = _FloatLikelihood(lik)
    out._mle = params.copy()
    out._held = np.r_[
        [False] * k, [boundary], np.isnan(params[k + 1 :])
    ].astype(bool)
    out._n_obs = n_obs
    out._ll_null = ll_null
    out.cluster_ids = list(layout.labels)
    out.cluster_events = layout.n_events.copy()
    out.cluster_exposure = lik.exposure(np.nan_to_num(params, nan=0.0))
    out._item_cluster = dict(layout.item_cluster)
    return out


# ---------------------------------------------------------------------------
# The fitted model
# ---------------------------------------------------------------------------
class SharedFrailtyModel(SerialisableMixin, LikelihoodInferenceMixin):
    """
    A fitted shared-frailty Poisson process: the result of
    ``fit(..., frailty="gamma")`` (or ``"lognormal"``) of ``CrowAMSAA``,
    ``Duane``, ``CoxLewis`` and ``ProportionalIntensityNHPP``.

    Each cluster (a unit by default, or the units ``cluster=`` groups)
    has an unobserved frailty ``u`` that multiplies its intensity,
    ``u lambda_0(t) exp(beta' Z)``; ``theta`` is the variance of ``u``
    (of ``log u`` for the log-normal frailty), 0 where the clusters
    differ no more than Poisson counts do.

    Predictions are either for the population of clusters
    (``frailty=None``: the frailty averaged over its distribution, whose
    mean is 1 for the gamma and ``exp(theta / 2)`` for the log-normal) or
    for a cluster of known frailty (``frailty=u``; :meth:`frailties`
    gives each fitted cluster's estimate). :meth:`cif` and :meth:`mcf`
    are the expected number of events by an age (the same function here,
    where nothing stops the events), :meth:`iif` its rate.

    Uncertainty: :meth:`cif_cb`, :meth:`iif_cb` and :meth:`mcf_cb` are
    delta-method bounds; :meth:`param_cb` gives Wald bounds (``theta`` on
    the log scale) and profile-likelihood ones (``method="lr"``);
    :meth:`frailty_test` is the likelihood-ratio test of ``theta = 0``;
    :meth:`forecast` gives the count of failures ahead with intervals
    that carry the frailty. Not offered: bootstrap bounds and
    ``diagnose()``; the plain fits' residual diagnostics assume Poisson
    counts, which is what this model drops (compare the fits with
    :meth:`frailty_test`, ``aic()`` and :meth:`frailties`).

    Attributes
    ----------
    params : numpy.ndarray
        The parameters, named by ``parameter_names``: the baseline's,
        ``theta``, then the coefficients (``nan`` where aliased).
    family : str
        ``"gamma"`` or ``"lognormal"``.
    dist : CountingProcess
        The baseline intensity.
    cluster_ids : list
        The clusters, in the order of :meth:`frailties`.
    maximum : str
        ``"verified"`` where the fit reached a verified maximum.

    Examples
    --------
    Six systems watched to 60 hours, with 1 to 11 failures each (far
    more spread than Poisson counts of one mean would have):

    >>> from surpyval.recurrent import CrowAMSAA
    >>> import numpy as np
    >>> counts = np.array([1, 1, 2, 3, 8, 11])
    >>> x = np.concatenate(
    ...     [np.r_[60 * (np.arange(n) + 0.5) / n, 60] for n in counts]
    ... )
    >>> i = np.repeat(np.arange(1, 7), counts + 1)
    >>> c = (x == 60).astype(int)
    >>> model = CrowAMSAA.fit(x, i, c, frailty="gamma")
    >>> model.parameter_names
    ['alpha', 'beta', 'theta']
    >>> [float(v) for v in model.frailties()["frailty"].round(2)]
    [0.46, 0.46, 0.62, 0.78, 1.59, 2.08]
    """

    params: npt.NDArray
    family: str
    dist: Any
    n_covariates: int
    regression: bool
    feature_names: list | None
    maximum: str
    how: str
    data: Any
    cluster_ids: list | None
    cluster_events: npt.NDArray | None
    cluster_exposure: npt.NDArray | None
    _held: npt.NDArray
    _lik: _Likelihood
    _ll_null: float
    _item_cluster: dict
    #: What ``fit(data)`` made of the data's terminal flags ``a`` (empty
    #: from arrays); not serialised.
    notes: tuple = ()

    def __getstate__(self) -> dict:
        """The state to pickle, without the caches of the covariance and
        of the likelihood-ratio searches, which are rebuilt when needed."""
        state = dict(vars(self))
        state.pop("_lr_region", None)
        state.pop("_cov_cache", None)
        return state

    # -- names and parts ----------------------------------------------------
    @property
    def _k(self) -> int:
        return len(self.dist.parameter_names)

    def _coefficient_names(self) -> list[str]:
        return coefficient_names(
            self.n_covariates,
            self.feature_names,
            [*self.dist.parameter_names, "theta"],
        )

    def _parameter_names(self) -> list[str]:
        return [
            *self.dist.parameter_names,
            "theta",
            *self._coefficient_names(),
        ]

    def _parameter_bounds(self) -> list:
        return [
            *self.dist.bounds,
            (0, 1) if self.family == "positive_stable" else (0, None),
            *[(None, None)] * self.n_covariates,
        ]

    def _estimated(self) -> npt.NDArray:
        held = getattr(self, "_held", None)
        if held is None:
            return ~np.isnan(np.asarray(self._mle, dtype=float))
        return ~np.asarray(held, dtype=bool)

    @property
    def theta(self) -> float:
        """The frailty variance (of ``log u`` for the log-normal); 0 for
        the model without frailty."""
        return float(self.params[self._k])

    @property
    def baseline_params(self) -> npt.NDArray:
        """The baseline intensity's parameters."""
        return self.params[: self._k].copy()

    @property
    def coeffs(self) -> npt.NDArray:
        """The covariates' coefficients (``nan`` where aliased); empty
        without covariates."""
        return self.params[self._k + 1 :].copy()

    @property
    def aliased(self) -> npt.NDArray:
        """The columns of ``Z`` whose coefficients the data cannot
        determine: ``nan`` in ``coeffs``, taken as 0 in predictions."""
        return np.flatnonzero(np.isnan(self.coeffs))

    @property
    def mean_frailty(self) -> float:
        """The mean of the frailty distribution: 1 for the gamma and the
        inverse Gaussian, ``exp(theta / 2)`` for the log-normal, ``inf``
        for the positive stable (``theta > 0``)."""
        return self._mean_frailty(self._p())

    def _p(self) -> npt.NDArray:
        """The parameters as the predictions take them: an aliased
        coefficient as 0."""
        return np.nan_to_num(np.asarray(self.params, dtype=float), nan=0.0)

    def _mean_frailty(self, params: npt.NDArray) -> float:
        theta = float(params[self._k])
        if self.family == "positive_stable" and theta > 0:
            return float(np.inf)
        return float(np.exp(theta / 2)) if self.family == "lognormal" else 1.0

    # -- covariates and frailty --------------------------------------------
    def _rows(self, Z: Any, n: int) -> npt.NDArray:
        p = self.n_covariates
        if not self.regression:
            if Z is not None:
                raise ValueError(
                    "'Z' is not taken: the model was fitted without "
                    "covariates."
                )
            return np.zeros((n, 0))
        if Z is None:
            raise ValueError(
                f"'Z' is needed: the model has {p} covariate(s); give one "
                "row, or one row per time."
            )
        if type(Z).__name__ == "DataFrame":
            names = self.feature_names
            if names is not None:
                Z = Z[names]
            Z = Z.to_numpy(dtype=float)
        Zr = np.asarray(Z, dtype=float)
        if Zr.ndim <= 1:
            Zr = Zr.reshape(1, -1)
        if Zr.shape[1] != p:
            raise ValueError(
                f"'Z' must have {p} column(s), one per covariate; got "
                f"{Zr.shape[1]}."
            )
        if Zr.shape[0] == 1:
            return np.repeat(Zr, n, axis=0)
        if Zr.shape[0] != n and n != 1:
            raise ValueError(
                f"'Z' has {Zr.shape[0]} rows for {n} times: give one row, "
                "or one row per time."
            )
        return Zr

    def _paired(
        self, x: npt.NDArray, Z: Any
    ) -> tuple[npt.NDArray, npt.NDArray]:
        x = np.asarray(x, dtype=float).reshape(-1)
        Zr = self._rows(Z, x.size)
        if Zr.shape[0] != x.size and x.size == 1:
            x = np.repeat(x, Zr.shape[0])
        return x, Zr

    def _frailty(self, frailty: Any, n: int) -> npt.NDArray | None:
        if frailty is None:
            return None
        u = np.asarray(frailty, dtype=float).reshape(-1)
        if u.size == 1:
            u = np.repeat(u, n)
        if u.size != n:
            raise ValueError("'frailty' must be one value, or one per time.")
        if np.any(u < 0):
            raise ValueError("'frailty' must be at least 0.")
        return u

    def _f(
        self,
        name: str,
        params: npt.NDArray,
        x: npt.NDArray,
        Zr: npt.NDArray,
        u: npt.NDArray | None,
    ) -> npt.NDArray:
        """``cif`` or ``iif`` at the parameters ``params``."""
        k = self._k
        base = params[:k]
        lower, upper = getattr(self.dist, "support", (-np.inf, np.inf))
        with np.errstate(all="ignore"):
            inside = (x >= lower) & (x <= upper)
            at = np.where(inside, x, np.nan)
            value = np.asarray(
                getattr(self.dist, name)(at, *base), dtype=float
            )
            phi = np.exp(Zr @ params[k + 1 :]) if Zr.shape[1] else 1.0
            if (
                u is None
                and self.family == "positive_stable"
                and float(params[k]) > 0
            ):
                raise ValueError(
                    "The positive stable frailty has no finite mean, so "
                    "the population's expected count (cif, iif or mcf "
                    "with frailty=None) is infinite. Give a cluster's "
                    "frailty with frailty= (frailties() has each fitted "
                    "cluster's), forecast the fitted clusters with "
                    "forecast(), or fit family='inverse_gaussian' or "
                    "'gamma' for a population mean."
                )
            scale = self._mean_frailty(params) if u is None else u
            return value * phi * scale

    def _evaluate(
        self, name: str, x: Any, Z: Any, frailty: Any
    ) -> npt.NDArray:
        xs, Zr = self._paired(x, Z)
        u = self._frailty(frailty, xs.size)
        return self._f(name, self._p(), xs, Zr, u)

    # -- the public functions ---------------------------------------------
    @keeps_query_shape
    def cif(self, x: ArrayLike, Z: Any = None, frailty: Any = None) -> Any:
        """
        The cumulative intensity at age ``x``: the expected number of
        events by ``x``. ``frailty=None`` averages over the frailty
        (``E[u] Lambda_0(x) exp(beta' Z)``, ``E[u]`` 1 for the gamma
        frailty and ``exp(theta / 2)`` for the log-normal); ``frailty=u``
        is that of a cluster of frailty ``u``.

        Parameters
        ----------
        x : array_like
            Ages.
        Z : array_like, optional
            Covariates, for a model fitted with them: one row for every
            age, or one per age.
        frailty : float or array_like, optional
            A cluster's frailty; default the population average.

        Returns
        -------
        numpy.ndarray
            The cumulative intensity, of the shape of ``x``; ``nan``
            outside the baseline's support.

        Examples
        --------
        >>> from surpyval.recurrent import CrowAMSAA
        >>> import numpy as np
        >>> counts = np.array([1, 1, 2, 3, 8, 11])
        >>> x = np.concatenate(
        ...     [np.r_[60 * (np.arange(n) + 0.5) / n, 60] for n in counts]
        ... )
        >>> i = np.repeat(np.arange(1, 7), counts + 1)
        >>> c = (x == 60).astype(int)
        >>> model = CrowAMSAA.fit(x, i, c, frailty="gamma")
        >>> # 26 failures over the six systems' 60 hours
        >>> round(float(model.cif(60.0)), 2)
        4.33
        >>> round(float(model.cif(60.0, frailty=2.0)), 2)
        8.67
        """
        return self._evaluate("cif", x, Z, frailty)

    @keeps_query_shape
    def iif(self, x: ArrayLike, Z: Any = None, frailty: Any = None) -> Any:
        """
        The intensity (rate of events) at age ``x``, averaged over the
        frailty (``frailty=None``) or for a cluster of frailty ``u``: the
        derivative of :meth:`cif`.

        Parameters
        ----------
        x : array_like
            Ages.
        Z : array_like, optional
            Covariates, as for :meth:`cif`.
        frailty : float or array_like, optional
            A cluster's frailty; default the population average.

        Returns
        -------
        numpy.ndarray
            The intensity, of the shape of ``x``.

        Examples
        --------
        >>> from surpyval.recurrent import CoxLewis
        >>> import numpy as np
        >>> counts = np.array([1, 1, 2, 3, 8, 11])
        >>> x = np.concatenate(
        ...     [np.r_[60 * (np.arange(n) + 0.5) / n, 60] for n in counts]
        ... )
        >>> i = np.repeat(np.arange(1, 7), counts + 1)
        >>> c = (x == 60).astype(int)
        >>> model = CoxLewis.fit(x, i, c, frailty="gamma")
        >>> bool(model.iif(30.0, frailty=2.0) == 2 * model.iif(30.0))
        True
        """
        return self._evaluate("iif", x, Z, frailty)

    @keeps_query_shape
    def mcf(self, x: ArrayLike, Z: Any = None, frailty: Any = None) -> Any:
        """
        The mean cumulative function: the expected number of events by
        age ``x``, for the population (``frailty=None``, the marginal
        mean) or for a cluster of frailty ``u`` (the conditional mean).
        Nothing stops the events in this model, so it is :meth:`cif`.

        Parameters
        ----------
        x : array_like
            Ages.
        Z : array_like, optional
            Covariates, as for :meth:`cif`.
        frailty : float or array_like, optional
            A cluster's frailty; default the population average.

        Returns
        -------
        numpy.ndarray
            The mean number of events, of the shape of ``x``.

        Examples
        --------
        >>> from surpyval.recurrent import CrowAMSAA
        >>> import numpy as np
        >>> counts = np.array([1, 1, 2, 3, 8, 11])
        >>> x = np.concatenate(
        ...     [np.r_[60 * (np.arange(n) + 0.5) / n, 60] for n in counts]
        ... )
        >>> i = np.repeat(np.arange(1, 7), counts + 1)
        >>> c = (x == 60).astype(int)
        >>> model = CrowAMSAA.fit(x, i, c, frailty="gamma")
        >>> # the sixth system, at its own predicted frailty
        >>> u = model.frailties()["frailty"][6]
        >>> bool(model.mcf(60.0, frailty=u) > model.mcf(60.0))
        True
        """
        return self._evaluate("cif", x, Z, frailty)

    # -- bounds -------------------------------------------------------------
    def _delta_cb(
        self,
        name: str,
        x: Any,
        Z: Any,
        alpha_ci: float,
        bound: str,
        frailty: Any,
    ) -> npt.NDArray:
        """Delta-method bounds on the log of the function, the gradient
        by central differences over the estimated parameters."""
        from scipy.stats import norm

        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        self._check_inference()
        xs, Zr = self._paired(x, Z)
        u = self._frailty(frailty, xs.size)
        base = self._p()
        value = self._f(name, base, xs, Zr, u)
        cov, _ = self._covariance_matrix()
        free = np.flatnonzero(self._estimated())
        grad = np.zeros((xs.size, free.size))
        with np.errstate(all="ignore"):
            log_value = np.log(value)
            for col, j in enumerate(free):
                h = 1e-5 * max(abs(base[j]), 1e-3)
                up, down = base.copy(), base.copy()
                up[j] += h
                down[j] -= h
                grad[:, col] = (
                    np.log(self._f(name, up, xs, Zr, u))
                    - np.log(self._f(name, down, xs, Zr, u))
                ) / (2 * h)
            var = np.einsum("ij,jk,ik->i", grad, cov[np.ix_(free, free)], grad)
            se = np.sqrt(np.where(var >= 0, var, np.nan))
            if bound == "two-sided":
                q = norm.ppf(1 - alpha_ci / 2)
                out = np.stack(
                    [np.exp(log_value - q * se), np.exp(log_value + q * se)],
                    axis=-1,
                )
                out[value == 0] = 0.0
            else:
                q = norm.ppf(1 - alpha_ci)
                sign = -1.0 if bound == "lower" else 1.0
                out = np.where(
                    value == 0, 0.0, np.exp(log_value + sign * q * se)
                )
        return out

    @keeps_query_shape
    def cif_cb(
        self,
        x: ArrayLike,
        Z: Any = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        frailty: Any = None,
    ) -> Any:
        """
        Confidence bounds on :meth:`cif` by the delta method on its log,
        from the parameters' covariance (which carries the frailty: the
        bounds are wider than the plain fit's where ``theta > 0``).

        Parameters
        ----------
        x : array_like
            Ages.
        Z : array_like, optional
            Covariates, as for :meth:`cif`.
        alpha_ci : float, optional
            The total tail probability. Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds add a last ``[lower, upper]`` axis.
        frailty : float, optional
            As for :meth:`cif`.

        Returns
        -------
        numpy.ndarray
            The bound(s).

        Examples
        --------
        >>> from surpyval.recurrent import CrowAMSAA
        >>> import numpy as np
        >>> counts = np.array([1, 1, 2, 3, 8, 11])
        >>> x = np.concatenate(
        ...     [np.r_[60 * (np.arange(n) + 0.5) / n, 60] for n in counts]
        ... )
        >>> i = np.repeat(np.arange(1, 7), counts + 1)
        >>> c = (x == 60).astype(int)
        >>> model = CrowAMSAA.fit(x, i, c, frailty="gamma")
        >>> lo, hi = model.cif_cb(60.0)
        >>> bool(lo < model.cif(60.0) < hi)
        True
        """
        return self._delta_cb("cif", x, Z, alpha_ci, bound, frailty)

    @keeps_query_shape
    def iif_cb(
        self,
        x: ArrayLike,
        Z: Any = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        frailty: Any = None,
    ) -> Any:
        """
        Confidence bounds on :meth:`iif`, as :meth:`cif_cb`.

        Parameters
        ----------
        x : array_like
            Ages.
        Z : array_like, optional
            Covariates, as for :meth:`cif`.
        alpha_ci : float, optional
            The total tail probability. Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds add a last ``[lower, upper]`` axis.
        frailty : float, optional
            As for :meth:`cif`.

        Returns
        -------
        numpy.ndarray
            The bound(s).

        Examples
        --------
        >>> from surpyval.recurrent import CrowAMSAA
        >>> import numpy as np
        >>> counts = np.array([1, 1, 2, 3, 8, 11])
        >>> x = np.concatenate(
        ...     [np.r_[60 * (np.arange(n) + 0.5) / n, 60] for n in counts]
        ... )
        >>> i = np.repeat(np.arange(1, 7), counts + 1)
        >>> c = (x == 60).astype(int)
        >>> model = CrowAMSAA.fit(x, i, c, frailty="gamma")
        >>> model.iif_cb([20.0, 40.0]).shape
        (2, 2)
        """
        return self._delta_cb("iif", x, Z, alpha_ci, bound, frailty)

    @keeps_query_shape
    def mcf_cb(
        self,
        x: ArrayLike,
        Z: Any = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        frailty: Any = None,
    ) -> Any:
        """
        Confidence bounds on :meth:`mcf`, as :meth:`cif_cb`.

        Parameters
        ----------
        x : array_like
            Ages.
        Z : array_like, optional
            Covariates, as for :meth:`cif`.
        alpha_ci : float, optional
            The total tail probability. Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds add a last ``[lower, upper]`` axis.
        frailty : float, optional
            As for :meth:`cif`.

        Returns
        -------
        numpy.ndarray
            The bound(s).

        Examples
        --------
        >>> from surpyval.recurrent import CrowAMSAA
        >>> import numpy as np
        >>> counts = np.array([1, 1, 2, 3, 8, 11])
        >>> x = np.concatenate(
        ...     [np.r_[60 * (np.arange(n) + 0.5) / n, 60] for n in counts]
        ... )
        >>> i = np.repeat(np.arange(1, 7), counts + 1)
        >>> c = (x == 60).astype(int)
        >>> model = CrowAMSAA.fit(x, i, c, frailty="gamma")
        >>> bool(model.mcf_cb(60.0, bound="upper") > model.mcf(60.0))
        True
        """
        return self._delta_cb("cif", x, Z, alpha_ci, bound, frailty)

    # -- parameters ----------------------------------------------------------
    def _covariance_matrix(self) -> tuple[npt.NDArray, bool]:
        """The inverse of autograd's Hessian of the negative marginal
        log-likelihood at the estimate, over the estimated parameters."""
        self._check_inference()
        lik = getattr(self, "_lik", None)
        if lik is None:
            return super()._covariance_matrix()
        cached = getattr(self, "_cov_cache", None)
        if cached is not None:
            return cached
        full = self._p()
        free = self._estimated()
        n = full.size

        def neg_ll_free(values: Any) -> Any:
            parts = []
            at = 0
            for j in range(n):
                if free[j]:
                    parts.append(values[at])
                    at += 1
                else:
                    parts.append(full[j] + 0.0 * values[0])
            return lik(anp.array(parts))

        out = np.full((n, n), np.nan)
        with np.errstate(all="ignore"):
            H = np.asarray(_ag_hessian(neg_ll_free)(full[free]), dtype=float)
        found = bool(np.all(np.isfinite(H)))
        if found:
            try:
                out[np.ix_(free, free)] = np.linalg.inv(0.5 * (H + H.T))
            except np.linalg.LinAlgError:
                found = False
        self._cov_cache = (out, found)
        return out, found

    def param_cb(
        self,
        name: str,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str | None = "wald",
    ) -> npt.NDArray:
        """
        Confidence bound(s) on a named parameter.

        - ``"wald"`` (the default) -- from the covariance, on the log
          scale for the positive parameters (``theta`` among them) and
          the natural scale for the others. Where ``theta`` is estimated
          at 0 its bounds are ``nan``.
        - ``"lr"`` -- the profile-likelihood interval: the values whose
          deviance, every other parameter re-fitted, stays below the
          :math:`\\chi^2_1` critical value. Where it stays below it down
          to ``theta = 0`` the lower bound on ``theta`` is 0. It needs
          the data, which a model restored from a dict does not keep.

        Parameters
        ----------
        name : str
            One of :attr:`parameter_names`.
        alpha_ci : float, optional
            Total tail probability. Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds are ``[lower, upper]``.
        method : {'wald', 'lr'}, optional
            As above.

        Returns
        -------
        numpy.ndarray
            The bound(s).

        Examples
        --------
        >>> from surpyval.recurrent import CrowAMSAA
        >>> import numpy as np
        >>> counts = np.array([1, 1, 2, 3, 8, 11])
        >>> x = np.concatenate(
        ...     [np.r_[60 * (np.arange(n) + 0.5) / n, 60] for n in counts]
        ... )
        >>> i = np.repeat(np.arange(1, 7), counts + 1)
        >>> c = (x == 60).astype(int)
        >>> model = CrowAMSAA.fit(x, i, c, frailty="gamma")
        >>> lo, hi = model.param_cb("beta")
        >>> bool(lo < model.params[1] < hi)
        True
        """
        from surpyval.univariate.regression._likelihood_ratio import (
            LikelihoodRegion,
            is_lr,
            profile_interval,
        )

        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        names = self.parameter_names
        if not is_lr(method):
            check_option("method", method, ("wald", "lr"))
        if name not in names:
            raise ValueError(
                "Unknown parameter {!r}; expected one of {}".format(
                    name, names
                )
            )
        idx = names.index(name)
        if not is_lr(method):
            self._check_inference()
            if not self._estimated()[idx]:
                return np.array(
                    [np.nan, np.nan] if bound == "two-sided" else [np.nan]
                )
            return super().param_cb(name, alpha_ci, bound)
        lik = getattr(self, "_lik", None)
        if lik is None:
            raise ValueError(
                "Likelihood-ratio bounds need the data the model was fitted "
                "to, which a model restored from a dict does not keep; use "
                "method='wald'."
            )
        region = getattr(self, "_lr_region", None)
        if region is None:
            held = set(np.flatnonzero(~self._estimated()).tolist())
            cov, _ = self._covariance_matrix()
            region = LikelihoodRegion(
                _FloatLikelihood(lik),
                self._p(),
                list(names),
                self._parameter_bounds(),
                held,
                cov,
            )
            self._lr_region = region
        return profile_interval(region, name, alpha_ci, bound)

    def frailty_test(self) -> Any:
        """
        The likelihood-ratio test of no frailty, ``theta = 0``: do the
        clusters differ by more than Poisson counts would?

        The statistic is twice the gain in log-likelihood over the model
        without frailty. ``theta = 0`` is on the edge of its space, so
        under it the statistic is 0 half the time and
        :math:`\\chi^2_1` the other half (Self and Liang 1987): the
        p-value is half the :math:`\\chi^2_1` tail, and 1 where the
        estimate is 0.

        Returns
        -------
        pandas.DataFrame
            One row, ``theta = 0``: ``statistic``, ``p`` and
            ``reference`` (the null distribution).

        Examples
        --------
        >>> from surpyval.recurrent import CrowAMSAA
        >>> import numpy as np
        >>> counts = np.array([1, 1, 2, 3, 8, 11])
        >>> x = np.concatenate(
        ...     [np.r_[60 * (np.arange(n) + 0.5) / n, 60] for n in counts]
        ... )
        >>> i = np.repeat(np.arange(1, 7), counts + 1)
        >>> c = (x == 60).astype(int)
        >>> model = CrowAMSAA.fit(x, i, c, frailty="gamma")
        >>> test = model.frailty_test()
        >>> round(float(test["p"].iloc[0]), 4)
        0.0049
        """
        import pandas as pd
        from scipy.stats import chi2

        self._check_fitted()
        statistic = max(0.0, 2.0 * (self.log_likelihood - self._ll_null))
        if statistic <= 1e-9:
            statistic, p = 0.0, 1.0
        else:
            p = float(0.5 * chi2.sf(statistic, 1))
        return pd.DataFrame(
            {
                "statistic": [statistic],
                "p": [p],
                "reference": ["50:50 mixture of 0 and chi2(1)"],
            },
            index=["theta = 0"],
        )

    def frailties(self, alpha_ci: float = 0.05) -> Any:
        """
        Each fitted cluster's predicted frailty: the mean of its frailty
        given its history (empirical Bayes), with an equal-tailed
        interval of that distribution.

        For the gamma frailty a cluster with ``n`` events and cumulative
        intensity ``L`` (at frailty 1) has a gamma frailty given its
        history, of mean ``(n + 1/theta) / (L + 1/theta)``: its observed
        over its expected count, shrunk towards 1, and less so the more
        there is of its history. The interval is that of this
        distribution at the fitted parameters; it does not carry their
        uncertainty.

        Parameters
        ----------
        alpha_ci : float, optional
            The interval's total tail probability. Default 0.05.

        Returns
        -------
        pandas.DataFrame
            One row per cluster: ``events``, ``expected`` (its cumulative
            intensity at frailty 1), ``frailty``, and ``lower <level>``
            and ``upper <level>``.

        Examples
        --------
        >>> from surpyval.recurrent import CrowAMSAA
        >>> import numpy as np
        >>> counts = np.array([1, 1, 2, 3, 8, 11])
        >>> x = np.concatenate(
        ...     [np.r_[60 * (np.arange(n) + 0.5) / n, 60] for n in counts]
        ... )
        >>> i = np.repeat(np.arange(1, 7), counts + 1)
        >>> c = (x == 60).astype(int)
        >>> model = CrowAMSAA.fit(x, i, c, frailty="gamma")
        >>> model.frailties()["events"].tolist()
        [1.0, 1.0, 2.0, 3.0, 8.0, 11.0]
        """
        import pandas as pd

        check_alpha_ci(alpha_ci)
        if self.cluster_ids is None:
            raise ValueError(
                "This model holds no clusters (it was restored from a "
                "dictionary saved without them); refit it to the data for "
                "their frailties."
            )
        n = np.asarray(self.cluster_events, dtype=float)
        A = np.asarray(self.cluster_exposure, dtype=float)
        mean, _ = frailty_moments(n, A, self.theta, self.family)
        ends = _frailty_quantiles(
            n, A, self.theta, self.family, (alpha_ci / 2, 1 - alpha_ci / 2)
        )
        level = f"{100 * (1 - alpha_ci):g}%"
        return pd.DataFrame(
            {
                "events": n,
                "expected": A,
                "frailty": mean,
                f"lower {level}": ends[:, 0],
                f"upper {level}": ends[:, 1],
            },
            index=pd.Index(self.cluster_ids, name="cluster"),
        )

    def forecast(
        self,
        horizon: ArrayLike,
        age: ArrayLike | None = None,
        Z: Any = None,
        cluster: ArrayLike | None = None,
        limit: ArrayLike | None = None,
        alpha_ci: float = 0.05,
    ) -> Any:
        """
        Forecast the number of failures over the time ahead, with
        prediction intervals that carry the frailty:
        :func:`surpyval.forecast` for this model.

        With ``age`` left out the units are those of the fitted data,
        each at the end of its observation, and each cluster's frailty is
        that given its history (for the gamma frailty its count ahead is
        negative binomial). With ``age`` the units are new ones of the
        population, each its own cluster unless ``cluster`` groups them;
        a label of a fitted cluster stands for that cluster, with its
        history. The interval is that of the count at the fitted
        parameters, wider than a Poisson one by the frailty's spread. It
        does not carry the parameters' uncertainty: for one new unit
        that hardly matters, but a 95% interval for the fitted fleet's
        total held 90% in the calibration study (the Poisson interval
        of the fit without frailty held 82%).

        Parameters
        ----------
        horizon : scalar or array_like
            The time ahead, or several increasing times ahead.
        age : array_like, optional
            The current age of each unit; by default the fitted data's
            units at the ends of their observation.
        Z : array_like, optional
            Each unit's covariates, with ``age``, for a model that has
            them.
        cluster : array_like, optional
            Each unit's cluster, with ``age``.
        limit : scalar or array_like, optional
            An age past which a unit's failures are not counted.
        alpha_ci : float, optional
            The prediction intervals' significance level. Default 0.05.

        Returns
        -------
        Forecast
            As :func:`surpyval.forecast` returns.

        Examples
        --------
        >>> from surpyval.recurrent import CrowAMSAA
        >>> import numpy as np
        >>> counts = np.array([1, 1, 2, 3, 8, 11])
        >>> x = np.concatenate(
        ...     [np.r_[60 * (np.arange(n) + 0.5) / n, 60] for n in counts]
        ... )
        >>> i = np.repeat(np.arange(1, 7), counts + 1)
        >>> c = (x == 60).astype(int)
        >>> model = CrowAMSAA.fit(x, i, c, frailty="gamma")
        >>> ahead = model.forecast(30.0)
        >>> plain = CrowAMSAA.fit(x, i, c)
        >>> import surpyval
        >>> poisson = surpyval.forecast(plain, horizon=30.0)
        >>> bool(ahead.variance[0] > poisson.variance[0])
        True
        """
        from surpyval.forecasting import forecast

        return forecast(
            self,
            age=age,
            horizon=horizon,
            Z=Z,
            limit=limit,
            alpha_ci=alpha_ci,
            cluster=cluster,
        )

    # -- saving and showing --------------------------------------------------
    def to_dict(self) -> dict:
        """
        A plain, JSON-safe dict of the model: its baseline, family,
        parameters, the covariance (so that the Wald bounds survive) and
        each cluster's events and cumulative intensity (so that
        :meth:`frailties` does). The data are not stored: the likelihood,
        the likelihood-ratio bounds, the test and a forecast of the
        fitted units need a refit.

        Returns
        -------
        dict
            The model, for :meth:`from_dict`.

        Examples
        --------
        >>> from surpyval.recurrent import CrowAMSAA, SharedFrailtyModel
        >>> import numpy as np
        >>> counts = np.array([1, 1, 2, 3, 8, 11])
        >>> x = np.concatenate(
        ...     [np.r_[60 * (np.arange(n) + 0.5) / n, 60] for n in counts]
        ... )
        >>> i = np.repeat(np.arange(1, 7), counts + 1)
        >>> c = (x == 60).astype(int)
        >>> model = CrowAMSAA.fit(x, i, c, frailty="gamma")
        >>> again = SharedFrailtyModel.from_dict(model.to_dict())
        >>> float(again.cif(30.0)) == float(model.cif(30.0))
        True
        """
        from surpyval.serialisation import to_native

        ids = self.cluster_ids
        out = {
            "model": "SharedFrailtyModel",
            "baseline": self.dist.name,
            "family": self.family,
            "how": getattr(self, "how", "MLE"),
            "params": np.asarray(self.params, dtype=float).tolist(),
            "n_covariates": int(self.n_covariates),
            "regression": bool(self.regression),
            "feature_names": (
                None
                if self.feature_names is None
                else [str(n) for n in self.feature_names]
            ),
            "held": (
                None
                if getattr(self, "_held", None) is None
                else [bool(v) for v in self._held]
            ),
            "covariance": self._covariance_entry(),
            "cluster_ids": None if ids is None else to_native(list(ids)),
            "cluster_events": (
                None
                if ids is None
                else np.asarray(self.cluster_events, dtype=float).tolist()
            ),
            "cluster_exposure": (
                None
                if ids is None
                else np.asarray(self.cluster_exposure, dtype=float).tolist()
            ),
            **maximum_entry(self.maximum),
        }
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> SharedFrailtyModel:
        """
        Rebuild a model from :meth:`to_dict`'s dict.

        Parameters
        ----------
        model_dict : dict
            As :meth:`to_dict` returns.

        Returns
        -------
        SharedFrailtyModel
            The model, without its data.

        Examples
        --------
        >>> from surpyval.recurrent import CrowAMSAA, SharedFrailtyModel
        >>> import numpy as np
        >>> counts = np.array([1, 1, 2, 3, 8, 11])
        >>> x = np.concatenate(
        ...     [np.r_[60 * (np.arange(n) + 0.5) / n, 60] for n in counts]
        ... )
        >>> i = np.repeat(np.arange(1, 7), counts + 1)
        >>> c = (x == 60).astype(int)
        >>> model = CrowAMSAA.fit(x, i, c, frailty="gamma")
        >>> again = SharedFrailtyModel.from_dict(model.to_dict())
        >>> again.theta == model.theta
        True
        """
        require_model_tag(
            model_dict, "SharedFrailtyModel", "a shared-frailty model"
        )
        out = cls()
        check_option("family", model_dict["family"], FAMILIES)
        out.dist = intensity_dist_by_name(model_dict["baseline"])
        out.family = str(model_dict["family"])
        out.how = str(model_dict.get("how", "MLE"))
        out.params = np.asarray(dict_numbers(model_dict, "params"), float)
        out.n_covariates = int(model_dict["n_covariates"])
        out.regression = bool(model_dict.get("regression", False))
        expected = len(out.dist.parameter_names) + 1 + out.n_covariates
        if out.params.size != expected:
            raise ValueError(
                "The serialised 'params' must have one value per baseline "
                f"parameter, theta and one per covariate ({expected}); got "
                f"{out.params.size}."
            )
        names = model_dict.get("feature_names")
        out.feature_names = None if names is None else list(names)
        held = model_dict.get("held")
        if held is not None:
            out._held = np.asarray(held, dtype=bool)
        out.maximum = restored_maximum(model_dict)
        ids = model_dict.get("cluster_ids")
        if ids is None:
            out.cluster_ids = None
            out.cluster_events = None
            out.cluster_exposure = None
        else:
            out.cluster_ids = [
                tuple(v) if isinstance(v, list) else v for v in ids
            ]
            out.cluster_events = np.asarray(
                dict_numbers(model_dict, "cluster_events"), float
            )
            out.cluster_exposure = np.asarray(
                dict_numbers(model_dict, "cluster_exposure"), float
            )
        out.data = None
        out._restore_covariance(model_dict, np.nan_to_num(out.params, nan=0.0))
        if out._saved_covariance is not None:
            out._mle = out.params.copy()
        return out

    def __repr__(self) -> str:
        title = "Shared Frailty Recurrence Model"
        lines = [
            title,
            "=" * len(title),
            f"Process             : {self.dist.name}",
            f"Frailty             : {self.family}",
            "Fitted by           : MLE",
            "Parameters          :",
        ]
        for name, value in zip(self.parameter_names, self.params):
            lines.append(f"    {name:>10}: {value:.6g}")
        return "\n".join(lines) + notes_text(self) + "\n"
