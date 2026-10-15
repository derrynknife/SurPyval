"""The joint model fitter: :func:`JointPH` and ``WeibullJointPH`` (#891)."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
from autograd import hessian, jacobian, value_and_grad
from scipy.optimize import minimize

from surpyval.univariate.regression._aliasing import (
    dataframe_covariates,
    fit_columns,
)
from surpyval.utils.fitter_repr import FitterRepr
from surpyval.utils.no_maximum import warn_unverified
from surpyval.utils.surpyval_data import accepts_surpyval_data
from surpyval.utils.units import takes_unit
from surpyval.utils.warnings import caller_stacklevel, warn_no_covariance

from ._likelihood import JointArrays, Layout, neg_ll, place_nodes
from ._markers import marker_start, survival_start
from .joint_model import JointModel, parameter_names, reported_params

#: The most rounds of placing the nodes and searching from them.
_ROUNDS = 8
#: Rounds stop once the log-likelihood moves less than this between them.
_ROUND_TOL = 1e-6


class JointPHFitter(FitterRepr):
    """
    The joint model of a marker -- a measurement read repeatedly on each
    item, such as a vibration level, a wear or oil-debris reading, or a
    biomarker -- and the item's lifetime, with a proportional hazards
    survival part: built by :func:`JointPH`, ready-made as
    ``WeibullJointPH``. :meth:`fit` returns a
    :class:`~surpyval.univariate.joint.joint_model.JointModel`.
    """

    #: The ``repr`` (#614)
    fitter_kind = "joint model fitter"

    def __init__(self, distribution: Any) -> None:
        from surpyval import Weibull

        if distribution is not Weibull:
            name = getattr(distribution, "name", type(distribution).__name__)
            raise ValueError(
                "The joint models have a Weibull baseline for now (#891): "
                f"JointPH(Weibull), or WeibullJointPH; got {name}."
            )
        self.dist = distribution
        self.name = distribution.name + "JointPH"

    def _repr_details(self) -> "list[str]":
        return [f"{self.dist.name} baseline"]

    @dataframe_covariates
    @takes_unit
    @accepts_surpyval_data
    def fit(
        self,
        x: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        t: npt.ArrayLike | None = None,
        Z: npt.ArrayLike | None = None,
        markers: Any = None,
        tl: npt.ArrayLike | None = None,
        tr: npt.ArrayLike | None = None,
    ) -> JointModel:
        """
        Fit the joint model of the marker and the lifetime by maximum
        likelihood.

        Each item's marker is a straight line in age with its own
        intercept and slope (bivariate normal across items) plus normal
        noise on each reading; its hazard is Weibull proportional hazards
        in the line's current, noise-free value (centred at the mean
        reading) and its covariates, which enter the marker's line too.
        The random effects are integrated out of each item's likelihood
        by adaptive Gauss-Hermite quadrature (7 points a dimension, the
        nodes placed again at the new parameters until the
        log-likelihood stops moving), and the likelihood is maximised
        with autograd's gradient; the covariance is the inverse of its
        Hessian. The fit starts from the marker's mixed model alone
        (REML) and a survival fit with each item's predicted line.

        The data are usually a :class:`~surpyval.SurpyvalData` of
        :class:`~surpyval.ItemLifetime` records with ``markers``:
        ``WeibullJointPH.fit(SurpyvalData.from_items(items))``. Items may
        be failed (``observed``) or still running (``right_censored``),
        and may have come under observation late (``left_truncation``:
        the likelihood is then that of an item that reached that age,
        over the population's lines). Left and interval censoring and
        right truncation are refused.

        Parameters
        ----------
        x : SurpyvalData or array_like
            The data, or each item's age at failure or censoring.
        c, n, t, tl, tr : array_like, optional
            As for every fit, with arrays: the censoring flag (0 failed, 1
            still running), the counts, and the truncation.
        Z : array_like, optional
            Each item's covariates, a row each (a DataFrame's columns name
            them). They enter both the marker's line and the hazard.
        markers : list of dict
            With arrays: each item's readings, ``{age: value}`` (``{}``
            for none).

        Returns
        -------
        JointModel
            The fitted model.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ItemLifetime, SurpyvalData, WeibullJointPH
        >>> rng = np.random.default_rng(3)
        >>> items = []
        >>> for k in range(60):
        ...     start, rate = rng.normal(2.0, 0.4), rng.normal(0.02, 0.005)
        ...     life = 100 * rng.weibull(2.0) * np.exp(-0.9 * (start - 2))
        ...     end = min(life, 120.0)
        ...     ages = np.arange(0.0, end, 10.0)
        ...     y = start + rate * ages + rng.normal(0, 0.2, ages.size)
        ...     outcome = "observed" if life < 120 else "right_censored"
        ...     items.append(
        ...         ItemLifetime(k, **{outcome: end},
        ...                      markers=dict(zip(ages, y)))
        ...     )
        >>> model = WeibullJointPH.fit(SurpyvalData.from_items(items))
        >>> model.maximum
        'verified'
        """
        records, counts, names = _records(x, c, n, t, tl, tr, Z, markers)
        return _fit(self.dist, records, counts, names)


def JointPH(distribution: Any) -> JointPHFitter:
    """
    A joint model fitter of a marker and a lifetime with a proportional
    hazards survival part on the baseline ``distribution``: the Weibull
    for now (#891), which ``WeibullJointPH`` is.

    Parameters
    ----------
    distribution : ParametricFitter
        ``surpyval.Weibull``.

    Returns
    -------
    JointPHFitter
        A fitter with a ``fit(data)`` method.

    Examples
    --------
    >>> from surpyval import JointPH, Weibull
    >>> JointPH(Weibull)
    WeibullJointPH: joint model fitter (Weibull baseline)
    """
    return JointPHFitter(distribution)


# -- the data ----------------------------------------------------------------


def _records(
    x: Any,
    c: Any,
    n: Any,
    t: Any,
    tl: Any,
    tr: Any,
    Z: Any,
    markers: Any,
) -> tuple[list, npt.NDArray, "list[str]"]:
    """The items as ItemLifetime records (which check each item's
    readings against its observation), their counts, and the covariates'
    names; what the fit cannot take is refused."""
    from surpyval.utils.item_lifetime import ItemLifetime

    x = np.asarray(x, dtype=float)
    rows = len(x)
    c = np.zeros(rows, dtype=int) if c is None else np.asarray(c)
    if c.shape != (rows,):
        raise ValueError(f"'c' must hold one flag per item ({rows}).")
    unknown = set(np.unique(c).tolist()) - {0, 1, -1, 2}
    if unknown:
        raise ValueError(
            "'c' must be 0 (failed) or 1 (still running); got "
            f"{sorted(unknown)}."
        )
    if np.isin(c, (-1, 2)).any() or x.ndim != 1:
        raise ValueError(
            "The joint model fits items that failed at a known age "
            "(observed) or are still running (right censored); this data "
            "has left or interval censored items, which it does not fit "
            "yet (#891)."
        )
    if markers is None:
        raise ValueError(
            "The joint model needs the items' marker readings: fit a "
            "SurpyvalData of ItemLifetime records with markers "
            "(SurpyvalData.from_items([...])), or give markers=[{age: "
            "value, ...}, ...], one per item."
        )
    markers = list(markers)
    if len(markers) != rows:
        raise ValueError(
            f"'markers' must hold one {{age: value}} per item ({rows}); "
            f"got {len(markers)}."
        )
    n = np.ones(rows) if n is None else np.asarray(n, dtype=float)
    if n.shape != (rows,) or not (np.isfinite(n).all() and (n > 0).all()):
        raise ValueError(f"'n' must hold one count above 0 per item ({rows}).")
    if t is not None:
        t = np.asarray(t, dtype=float)
        if t.shape != (rows, 2):
            raise ValueError(
                f"'t' must be the [left, right] truncation of each item, "
                f"shape ({rows}, 2)."
            )
        tl, tr = t[:, 0], t[:, 1]
    tl = (
        np.full(rows, -np.inf)
        if tl is None
        else np.broadcast_to(np.asarray(tl, dtype=float), (rows,))
    )
    tr = (
        np.full(rows, np.inf)
        if tr is None
        else np.broadcast_to(np.asarray(tr, dtype=float), (rows,))
    )
    if np.isfinite(tr).any():
        raise ValueError(
            "The joint model does not fit right truncated items yet "
            "(#891); this data has some."
        )
    names = fit_columns()
    if Z is not None:
        Z = np.asarray(Z, dtype=float)
        Z = Z[:, None] if Z.ndim == 1 else Z
        if Z.ndim != 2 or len(Z) != rows:
            raise ValueError(
                f"'Z' must hold one row of covariates per item ({rows})."
            )
        missing = np.isnan(Z).any(axis=1)
        if missing.any():
            warnings.warn(
                f"{int(missing.sum())} item(s) with a missing covariate "
                "were left out of the joint fit.",
                UserWarning,
                stacklevel=caller_stacklevel(),
            )
        if names is None:
            names = [f"coef_{j}" for j in range(Z.shape[1])]
    else:
        missing = np.zeros(rows, dtype=bool)
        names = []
    names = [str(name) for name in names]
    every = parameter_names(names)
    clash = sorted(
        {
            a
            for a in names
            if every.count(a) > 1 or every.count(f"marker_{a}") > 1
        }
    )
    if clash:
        raise ValueError(
            f"Covariates named {clash!r} would share a parameter's name in "
            "the joint model; rename them."
        )
    outcome = {0: "observed", 1: "right_censored"}
    records, counts = [], []
    for k in range(rows):
        if missing[k]:
            continue
        records.append(
            ItemLifetime(
                k,
                **{outcome[int(c[k])]: float(x[k])},
                left_truncation=(float(tl[k]) if np.isfinite(tl[k]) else None),
                covariates=None if Z is None else Z[k],
                markers=markers[k] or None,
            )
        )
        counts.append(n[k])
    if not records:
        raise ValueError("No items are left to fit.")
    return records, np.array(counts, dtype=float), names


def _fit(
    dist: Any, records: list, counts: npt.NDArray, names: list
) -> JointModel:
    """The joint fit of ``records`` (``counts`` of each)."""
    p = len(names)
    T = np.array([r.value for r in records], dtype=float)
    d = np.array([r.outcome == "observed" for r in records], dtype=float)
    L = np.array(
        [
            0.0 if r.left_truncation is None else r.left_truncation
            for r in records
        ]
    )
    if np.any(L < 0) or np.any(T <= 0):
        raise ValueError(
            "The joint model's ages start at 0 (a Weibull baseline): every "
            "item fails or is censored after 0 and comes under observation "
            "at 0 or later (left_truncation)."
        )
    Z = (
        np.array([r.covariates for r in records], dtype=float).reshape(-1, p)
        if p
        else np.zeros((len(records), 0))
    )
    ages = [
        np.array(list((r.markers or {}).keys()), dtype=float) for r in records
    ]
    values = [
        np.array(list((r.markers or {}).values()), dtype=float)
        for r in records
    ]
    if not d.any():
        raise ValueError(
            "The joint model needs at least one failure (observed): every "
            "item is still running."
        )
    lines = sum(1 for a in ages if len(np.unique(a)) >= 2)
    if lines < 2:
        raise ValueError(
            "The joint model needs marker readings at two or more ages on "
            f"at least two items, to see how lines differ; {lines} item(s) "
            "have them."
        )
    # The internal scale: ages over their mean, the marker centred at its
    # mean reading and over its spread, the covariates centred.
    read_w = np.concatenate(
        [np.full(len(v), w) for v, w in zip(values, counts)]
    )
    all_values = np.concatenate(values)
    s = float(np.average(T, weights=counts))
    centre = float(np.average(all_values, weights=read_w))
    spread = float(
        np.sqrt(np.average((all_values - centre) ** 2, weights=read_w))
    )
    spread = spread if spread > 0 else 1.0
    zbar = np.average(Z, axis=0, weights=counts) if p else np.zeros(0)
    scales = (s, centre, spread, zbar)
    in_ages = [a / s for a in ages]
    in_values = [(v - centre) / spread for v in values]
    Zc = Z - zbar
    arrays = JointArrays.build(counts, T / s, d, L / s, Zc, in_ages, in_values)
    layout = Layout(p)
    # Starting values: the mixed model, then the survival part on its lines
    fixed, D, sigma2, blups = marker_start(in_ages, in_values, Zc)
    root = np.linalg.cholesky(D)
    theta = {
        "b0": fixed[0],
        "b1": fixed[1],
        "delta": fixed[2:],
        "L00": root[0, 0],
        "L10": root[1, 0],
        "L11": root[1, 1],
        "sigma": np.sqrt(sigma2),
    }
    theta = survival_start(arrays, theta, blups)
    u = layout.pack(theta)
    total = float(np.sum(counts))

    def objective(v: Any, nodes: tuple) -> Any:
        return neg_ll(v, layout, arrays, nodes) / total

    search = value_and_grad(objective)

    def safe(v: npt.NDArray, nodes: tuple) -> tuple:
        with np.errstate(all="ignore"):
            value, grad = search(v, nodes)
        if not np.isfinite(value) or not np.all(np.isfinite(grad)):
            return np.inf, np.zeros_like(v)
        return value, grad

    modes = blups
    previous = np.inf
    converged = False
    for _ in range(_ROUNDS):
        nodes, modes = place_nodes(u, layout, arrays, start=modes)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                safe,
                u,
                args=(nodes,),
                jac=True,
                method="BFGS",
                options={"gtol": 1e-8, "maxiter": 2000},
            )
        if np.isfinite(result.fun) and result.fun <= safe(u, nodes)[0]:
            u = result.x
        value = float(objective(u, nodes)) * total
        if abs(previous - value) < _ROUND_TOL * max(1.0, abs(value)):
            converged = True
            break
        previous = value
    # The maximum, verified at the last nodes: a zero gradient and a
    # positive-definite Hessian
    with np.errstate(all="ignore"):
        grad = value_and_grad(lambda v: neg_ll(v, layout, arrays, nodes))(u)[1]
        info = hessian(lambda v: neg_ll(v, layout, arrays, nodes))(u)
    info = 0.5 * (info + info.T)
    try:
        cov_u = np.linalg.inv(info)
        positive = bool(np.all(np.linalg.eigvalsh(info) > 0))
    except np.linalg.LinAlgError:
        cov_u, positive = None, False
    flat = bool(np.max(np.abs(grad)) < 1e-3 * max(1.0, total) ** 0.5)
    verified = converged and positive and flat and np.all(np.isfinite(u))
    if not verified:
        reason = None
        if not converged:
            reason = "the nodes' rounds did not settle"
        elif not positive:
            reason = "the Hessian is not positive definite there"
        elif not flat:
            reason = "the gradient is not zero there"
        warn_unverified("The joint fit", reason)
    params = np.asarray(reported_params(layout.unpack(u), scales), dtype=float)
    covariance = None
    if cov_u is not None and positive:
        J = jacobian(lambda v: reported_params(layout.unpack(v), scales))(u)
        covariance = J @ cov_u @ J.T
        covariance = 0.5 * (covariance + covariance.T)
    else:
        warn_no_covariance()
    n_read = float(np.sum([len(a) * w for a, w in zip(ages, counts)]))
    n_events = float(np.sum(d * counts))
    model = JointModel()
    model.dist = dist
    model.params = params
    model.feature_names = list(names)
    model.parameter_names = parameter_names(names)
    model.marker_center = centre
    model._scales = scales
    model._covariance = covariance
    # Back to the data's units: each reading's density over the spread,
    # each failure's over the time scale
    model._neg_ll = float(
        neg_ll(u, layout, arrays, nodes)
        + n_read * np.log(spread)
        + n_events * np.log(s)
    )
    model.k = len(params)
    model.maximum = "verified" if verified else "unverified"
    model.n_items = total
    model.n_events = n_events
    model.n_readings = n_read
    return model
