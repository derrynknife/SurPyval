"""The joint model fitter: :func:`JointPH` and ``WeibullJointPH`` (#891)."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
from autograd import hessian, jacobian, value_and_grad
from scipy.optimize import minimize

from surpyval.univariate.regression._aliasing import (
    aliased_columns,
    constant_columns,
    dataframe_covariates,
    fit_columns,
    warn_aliased,
)
from surpyval.utils.covariates import finite_covariate_mask
from surpyval.utils.fitter_repr import FitterRepr
from surpyval.utils.no_maximum import warn_unverified
from surpyval.utils.surpyval_data import accepts_surpyval_data
from surpyval.utils.units import takes_unit
from surpyval.utils.warnings import warn_no_covariance

from ._likelihood import (
    MAX_VARYING,
    JointArrays,
    Layout,
    Model,
    PathFunction,
    neg_ll,
    place,
)
from ._markers import marker_start, survival_start
from .joint_model import (
    JointModel,
    covariate_parts,
    parameter_names,
    path_formula,
    reported_params,
)

#: The most rounds of placing the nodes and searching from them.
_ROUNDS = 10
#: Rounds stop once the log-likelihood moves less than this between them
#: (relative, with an absolute floor of the same in log-likelihood units).
_ROUND_TOL = 1e-10
#: A verified maximum's gradient, in standard errors of each parameter.
_GTOL = 1e-4


class JointPHFitter(FitterRepr):
    """
    The joint model of a marker -- a measurement read repeatedly on each
    item, such as a vibration level, a wear or oil-debris reading, or a
    biomarker -- and the item's lifetime, the marker following a
    degradation path model and the survival part proportional hazards:
    built by :func:`JointPH`, ready-made as ``WeibullJointPH`` (a
    straight line). :meth:`fit` returns a
    :class:`~surpyval.univariate.joint.joint_model.JointModel`.
    """

    #: The ``repr`` (#614)
    fitter_kind = "joint model fitter"

    def __init__(
        self, distribution: Any, path_model: Any, varying: Any = None
    ) -> None:
        from surpyval import Weibull
        from surpyval.degradation.path_models import LinearPath, PathModel

        if distribution is not Weibull:
            name = getattr(distribution, "name", type(distribution).__name__)
            raise ValueError(
                "The joint models have a Weibull baseline for now (#891): "
                f"JointPH(Weibull, path_model), or WeibullJointPH; got {name}."
            )
        if not isinstance(path_model, PathModel):
            raise ValueError(
                "path_model is a degradation path model, such as "
                "surpyval.degradation.LinearPath or QuadraticPath, or a "
                f"PathModel of your own; got {path_model!r}."
            )
        names = list(path_model.parameter_names)
        if varying is None:
            if len(names) > MAX_VARYING:
                raise ValueError(
                    f"The {path_model.name} path has {len(names)} "
                    f"parameters; at most {MAX_VARYING} can vary between "
                    "items (the quadrature is a grid of 7 points in each): "
                    "choose them with varying=(...)."
                )
            varying = tuple(names)
        if isinstance(varying, str):
            varying = (varying,)
        varying = tuple(varying)
        unknown = [v for v in varying if v not in names]
        if unknown or not varying or len(set(varying)) < len(varying):
            raise ValueError(
                "varying names the path parameters that vary between "
                f"items, at least one, each once, of {names}; got "
                f"{list(varying)}."
            )
        if len(varying) > MAX_VARYING:
            raise ValueError(
                f"At most {MAX_VARYING} path parameters can vary between "
                "items (the quadrature is a grid of 7 points in each); "
                f"got {len(varying)}."
            )
        every = parameter_names(names, [n for n in names if n in varying], [])
        twice = sorted({n for n in every if every.count(n) > 1})
        if twice:
            raise ValueError(
                f"The {path_model.name} path's parameter names {twice} are "
                "also names of the joint model's other parameters; rename "
                "them in the path model."
            )
        self.dist = distribution
        self.path_model = path_model
        # in the path's order
        self.varying = tuple(n for n in names if n in varying)
        plain = "".join(ch for ch in path_model.name if ch.isalnum())
        self.name = (
            f"{distribution.name}JointPH"
            if path_model is LinearPath
            else f"{distribution.name}{plain}JointPH"
        )

    def _repr_details(self) -> list[str]:
        names = list(self.path_model.parameter_names)
        out = [
            f"{self.dist.name} baseline",
            "{} path, {}".format(
                self.path_model.name, path_formula(self.path_model)
            ),
        ]
        if len(self.varying) < len(names):
            out.append("varying: " + ", ".join(self.varying))
        return out

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
        marker_reference: float | None = None,
        path_covariates: Any = None,
        hazard_covariates: Any = None,
    ) -> JointModel:
        """
        Fit the joint model of the marker and the lifetime by maximum
        likelihood.

        Each item's marker follows the fitter's path model with
        parameters of its own -- those that vary between items
        multivariate normal about means that its covariates shift, the
        rest common -- read with normal noise; its hazard is Weibull
        proportional hazards in the path's current, noise-free value and
        its covariates. The varying parameters are integrated out of each
        item's likelihood by adaptive Gauss-Hermite quadrature (7 points
        a dimension, the nodes placed again at the new parameters until
        the log-likelihood stops moving), and the likelihood is maximised
        with autograd's gradient; the covariance is the inverse of its
        Hessian. The fit starts from the path fitted to each item alone,
        the spread of those fits, and a survival fit with each item's
        predicted path.

        The data are usually a :class:`~surpyval.SurpyvalData` of
        :class:`~surpyval.ItemLifetime` records with ``markers``:
        ``WeibullJointPH.fit(SurpyvalData.from_items(items))``. Items may
        be failed (``observed``), still running (``right_censored``),
        found failed at an inspection (``left_censored``: failed by that
        age) or between two (``interval_censored``), and may have come
        under observation late (``left_truncation``) or be in the data
        only because they failed by an age (``right_truncation``): the
        likelihood is then that of an item that reached its entry age and
        failed by its right truncation, over the population's paths. A
        covariate the data cannot determine (a constant, or a combination
        of others) is aliased: its coefficients are ``nan``, with one
        warning.

        Every covariate shifts every path parameter's mean and enters the
        hazard, unless ``path_covariates`` or ``hazard_covariates`` say
        otherwise: a condition may change how fast the marker rises
        without changing the risk at a given reading (path only), or the
        risk at a given reading without changing the marker (hazard
        only).

        Parameters
        ----------
        x : SurpyvalData or array_like
            The data, or each item's age at failure or censoring.
        c, n, t, tl, tr : array_like, optional
            As for every fit, with arrays: the censoring flag (0 failed, 1
            still running, -1 failed by ``x``, 2 failed in ``x``'s
            ``[lower, upper]`` row), the counts, and the truncation.
        Z : array_like, optional
            Each item's covariates, a row each (a DataFrame's columns name
            them). By default they shift every path parameter's mean and
            enter the hazard.
        markers : list of dict
            With arrays: each item's readings, ``{age: value}`` (``{}``
            for none).
        marker_reference : float, optional
            The marker value at which ``alpha`` is reported (the
            baseline's item has its marker there). Default the mean
            reading. It changes ``alpha`` only.
        path_covariates : list of str or dict, optional
            The covariates that shift the path parameters' means
            (``a_<cov>``, ``b_<cov>``, ...): a list (each path parameter),
            or ``{path parameter: [covariate, ...]}`` (one left out is not
            shifted). Default every covariate; ``[]`` for none.
        hazard_covariates : list of str, optional
            The covariates in the hazard (``<cov>``). Default every
            covariate; ``[]`` for none. Each covariate must be in one part
            or both. Unnamed covariates are ``coef_0``, ``coef_1``, ...

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
        reference = None
        if marker_reference is not None:
            try:
                reference = float(marker_reference)
            except (TypeError, ValueError):
                reference = np.nan
            if np.ndim(marker_reference) or not np.isfinite(reference):
                raise ValueError(
                    "'marker_reference' is the marker value at which alpha "
                    "is reported, one finite number; got "
                    f"{marker_reference!r}."
                )
        records, counts, names = _records(x, c, n, t, tl, tr, Z, markers)
        parts = covariate_parts(
            list(self.path_model.parameter_names),
            names,
            path_covariates,
            hazard_covariates,
        )
        return _fit(self, records, counts, names, reference, parts)

    @takes_unit
    def fit_from_df(
        self,
        df: Any,
        x_col: str | None = None,
        markers_col: str | None = None,
        c_col: str | None = None,
        n_col: str | None = None,
        tl_col: str | None = None,
        tr_col: str | None = None,
        xl_col: str | None = None,
        xr_col: str | None = None,
        Z_cols: str | list[str] | None = None,
        marker_reference: float | None = None,
        path_covariates: Any = None,
        hazard_covariates: Any = None,
    ) -> JointModel:
        """
        Fit the joint model to a DataFrame of items, one row each: the
        age it failed or was censored (``x_col``), its readings
        (``markers_col``, a column holding each item's ``{age: value}``),
        and optionally its censoring flag, count, truncation and
        covariates (``c_col``, ``n_col``, ``tl_col``, ``tr_col``,
        ``Z_cols``, which name the model's covariates). An item found
        failed between two ages has them in ``xl_col`` and ``xr_col`` (in
        place of ``x_col``; ``c_col`` 2 for those rows). As :meth:`fit`
        otherwise, ``path_covariates`` and ``hazard_covariates`` naming
        which of ``Z_cols`` enter which part.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from surpyval import WeibullJointPH
        >>> rng = np.random.default_rng(3)
        >>> rows = []
        >>> for k in range(60):
        ...     start, rate = rng.normal(2.0, 0.4), rng.normal(0.02, 0.005)
        ...     life = 100 * rng.weibull(2.0) * np.exp(-0.9 * (start - 2))
        ...     end = min(life, 120.0)
        ...     ages = np.arange(0.0, end, 10.0)
        ...     y = start + rate * ages + rng.normal(0, 0.2, ages.size)
        ...     rows.append({"age": end, "running": int(life >= 120),
        ...                  "site": k % 2, "readings": dict(zip(ages, y))})
        >>> df = pd.DataFrame(rows)
        >>> model = WeibullJointPH.fit_from_df(
        ...     df, x_col="age", markers_col="readings", c_col="running",
        ...     Z_cols=["site"],
        ... )
        >>> model.parameter_names[:4]
        ['a', 'a_site', 'b', 'b_site']
        """
        from surpyval.univariate.regression._aliasing import (
            covariate_columns,
        )
        from surpyval.utils.dataframe import check_columns

        check_columns(
            df,
            x_col=x_col,
            markers_col=markers_col,
            c_col=c_col,
            n_col=n_col,
            tl_col=tl_col,
            tr_col=tr_col,
            xl_col=xl_col,
            xr_col=xr_col,
            Z_cols=Z_cols,
        )
        if markers_col is None:
            raise ValueError(
                "markers_col names the column of each item's readings, "
                "{age: value}: the joint model needs them."
            )
        if x_col is not None and (xl_col is not None or xr_col is not None):
            raise ValueError(
                "Give x_col, or xl_col and xr_col (an interval each), not "
                "both."
            )
        if x_col is not None:
            x = df[x_col].to_numpy(dtype=float)
        elif xl_col is not None and xr_col is not None:
            x = df[[xl_col, xr_col]].to_numpy(dtype=float)
        else:
            raise ValueError(
                "x_col names the column of each item's age at failure or "
                "censoring (or xl_col and xr_col, an interval's ends)."
            )
        names = None
        Z = None
        if Z_cols is not None:
            names = [Z_cols] if isinstance(Z_cols, str) else list(Z_cols)
            Z = df[names].to_numpy(dtype=float)
        with covariate_columns(names, Z):
            return self.fit(
                x,
                c=None if c_col is None else df[c_col].to_numpy(),
                n=None if n_col is None else df[n_col].to_numpy(),
                tl=None if tl_col is None else df[tl_col].to_numpy(float),
                tr=None if tr_col is None else df[tr_col].to_numpy(float),
                Z=Z,
                markers=list(df[markers_col]),
                marker_reference=marker_reference,
                path_covariates=path_covariates,
                hazard_covariates=hazard_covariates,
            )


def JointPH(
    distribution: Any, path_model: Any = None, varying: Any = None
) -> JointPHFitter:
    """
    A joint model fitter of a marker and a lifetime: the marker follows
    the degradation ``path_model`` (``surpyval.degradation.LinearPath``
    by default; ``QuadraticPath``, ``ExponentialPath``, ... or a
    ``PathModel`` of one's own), and the hazard is proportional hazards
    on the baseline ``distribution`` (the Weibull for now, #891) in the
    path's current value. ``WeibullJointPH`` is ``JointPH(Weibull,
    LinearPath)``.

    Parameters
    ----------
    distribution : ParametricFitter
        ``surpyval.Weibull``.
    path_model : PathModel, optional
        The marker's path. Default ``LinearPath``.
    varying : sequence of str, optional
        The path parameters that vary between items (their random
        effects, multivariate normal); the others are common to all
        items. Default all of them (at most three).

    Returns
    -------
    JointPHFitter
        A fitter with a ``fit(data)`` method.

    Examples
    --------
    >>> from surpyval import JointPH, Weibull
    >>> from surpyval.degradation import QuadraticPath
    >>> JointPH(Weibull)  # doctest: +NORMALIZE_WHITESPACE
    WeibullJointPH: joint model fitter (Weibull baseline, Linear path,
    a + b t)
    >>> JointPH(Weibull, QuadraticPath, varying=("a", "b"))
    ... # doctest: +ELLIPSIS
    WeibullQuadraticJointPH: joint model fitter (..., varying: a, b)
    """
    if path_model is None:
        from surpyval.degradation.path_models import LinearPath

        path_model = LinearPath
    return JointPHFitter(distribution, path_model, varying)


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
) -> tuple[list, npt.NDArray, list[str]]:
    """The items as ItemLifetime records (which check each item's
    readings against its observation), their counts, and the covariates'
    names; what the fit cannot take is refused."""
    from surpyval.utils.item_lifetime import ItemLifetime

    x = np.asarray(x, dtype=float)
    if x.ndim not in (1, 2) or (x.ndim == 2 and x.shape[1] != 2):
        raise ValueError(
            "'x' is each item's age at failure or censoring, or a [lower, "
            "upper] row each (interval censored items); got shape "
            f"{x.shape}."
        )
    rows = len(x)
    if c is None:
        # [lower, upper] rows: an interval where they differ
        c = (
            np.where(x[:, 0] == x[:, 1], 0, 2)
            if x.ndim == 2
            else np.zeros(rows, dtype=int)
        )
    c = np.asarray(c)
    if c.shape != (rows,):
        raise ValueError(f"'c' must hold one flag per item ({rows}).")
    unknown = set(np.unique(c).tolist()) - {0, 1, -1, 2}
    if unknown:
        raise ValueError(
            "'c' must be 0 (failed), 1 (still running), -1 (failed by "
            f"x) or 2 (failed in x's interval); got {sorted(unknown)}."
        )
    if x.ndim == 1 and np.any(c == 2):
        raise ValueError(
            "An interval censored item (c = 2) needs its [lower, upper]: "
            "give x as one [lower, upper] row per item (the same age twice "
            "for the others)."
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
    names = fit_columns()
    if Z is not None:
        Z = np.asarray(Z, dtype=float)
        Z = Z[:, None] if Z.ndim == 1 else Z
        if Z.ndim != 2 or len(Z) != rows:
            raise ValueError(
                f"'Z' must hold one row of covariates per item ({rows})."
            )
        # an item with a missing covariate is left out, with one warning
        missing = ~finite_covariate_mask(Z)
        if names is None:
            names = [f"coef_{j}" for j in range(Z.shape[1])]
    else:
        missing = np.zeros(rows, dtype=bool)
        names = []
    names = [str(name) for name in names]
    outcome = {
        0: "observed",
        1: "right_censored",
        -1: "left_censored",
        2: "interval_censored",
    }
    records, counts = [], []
    for k in range(rows):
        if missing[k]:
            continue
        flag = int(c[k])
        if flag == 2:
            value: Any = [float(x[k, 0]), float(x[k, 1])]
        elif x.ndim == 1:
            value = float(x[k])
        else:
            value = float(x[k, 1] if flag == -1 else x[k, 0])
        records.append(
            ItemLifetime(
                k,
                **{outcome[flag]: value},
                left_truncation=(float(tl[k]) if np.isfinite(tl[k]) else None),
                right_truncation=(
                    float(tr[k]) if np.isfinite(tr[k]) else None
                ),
                covariates=None if Z is None else Z[k],
                markers=markers[k] or None,
            )
        )
        counts.append(n[k])
    if not records:
        raise ValueError("No items are left to fit.")
    return records, np.array(counts, dtype=float), names


def _lifetimes(records: list) -> tuple:
    """``(T, d, U, L, Rt)`` of the records: the age each left observation
    (the lower end of an interval: a left censored item's entry age, else
    0), the event flag, the upper end of an interval (``inf`` for
    none), the entry age (0 for none) and the right truncation (``inf``
    for none)."""
    T, d, U, L, Rt = [], [], [], [], []
    for r in records:
        entry = 0.0 if r.left_truncation is None else r.left_truncation
        if r.outcome == "interval_censored":
            T.append(float(r.value[0]))
            U.append(float(r.value[1]))
        elif r.outcome == "left_censored":
            # failed by its age, having reached its entry
            T.append(entry)
            U.append(float(r.value))
        else:
            T.append(float(r.value))
            U.append(np.inf)
        d.append(r.outcome == "observed")
        L.append(entry)
        Rt.append(np.inf if r.right_truncation is None else r.right_truncation)
    return (
        np.array(T, dtype=float),
        np.array(d, dtype=float),
        np.array(U, dtype=float),
        np.array(L, dtype=float),
        np.array(Rt, dtype=float),
    )


def _arrays(
    fitter: JointPHFitter, records: list, counts: Any, p: int
) -> tuple:
    """The items' arrays (covariates as given), their readings, and the
    checks of what the model needs."""
    T, d, U, L, Rt = _lifetimes(records)
    point = ~np.isfinite(U)
    if np.any(L < 0) or np.any(T[point] <= 0) or np.any(U[~point] <= 0):
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
    for r, a, v in zip(records, ages, values):
        if len(a):
            fitter.path_model.check_data(a, v)
    if not (d.any() or np.isfinite(U).any()):
        raise ValueError(
            "The joint model needs at least one failure (observed, or "
            "found failed: left or interval censored): every item is still "
            "running."
        )
    P = len(fitter.path_model.parameter_names)
    q = len(fitter.varying)
    fitted = sum(1 for a in ages if len(np.unique(a)) >= P)
    if fitted < q + 2:
        raise ValueError(
            f"The joint model needs marker readings at {P} or more ages "
            f"(the {fitter.path_model.name} path's parameters) on at least "
            f"{q + 2} items, to see how their paths differ; {fitted} "
            "item(s) have them."
        )
    return T, d, L, Z, ages, values, U, Rt


def _aliased(Z: npt.NDArray, w: npt.NDArray) -> npt.NDArray:
    """The covariate columns the data cannot determine: constant (the
    path means and the Weibull scale are intercepts), or combinations of
    others."""
    if Z.shape[1] == 0:
        return np.zeros(0, dtype=int)
    constant = constant_columns(Z)
    zc = Z - np.average(Z, axis=0, weights=w)
    gram = (zc * w[:, None]).T @ zc
    return aliased_columns(gram, len(Z), constant)


def _fit(
    fitter: JointPHFitter,
    records: list,
    counts: npt.NDArray,
    names: list,
    reference: float | None,
    parts: tuple,
) -> JointModel:
    """The joint fit of ``records`` (``counts`` of each); ``parts`` is
    ``(shifts, hazard)``, which covariates enter which part
    (:func:`~surpyval.univariate.joint.joint_model.covariate_parts`)."""
    path_model = fitter.path_model
    path_names = list(path_model.parameter_names)
    shifts, hazard = parts
    p_all = len(names)
    every = parameter_names(
        path_names, list(fitter.varying), names, shifts, hazard
    )
    clash = sorted({a for a in names if every.count(a) > 1})
    clash += sorted(
        {a for a in names for b in path_names if every.count(f"{b}_{a}") > 1}
        - set(clash)
    )
    if clash:
        raise ValueError(
            f"Covariates named {clash!r} would share a parameter's name in "
            "the joint model; rename them."
        )
    T, d, L, Z_all, ages, values, U, Rt = _arrays(
        fitter, records, counts, p_all
    )
    aliased = _aliased(Z_all, counts)
    if aliased.size:
        warn_aliased(
            aliased,
            "constant, or a combination of the others: the path means and "
            "the Weibull scale already hold a constant",
        )
    kept = np.setdiff1d(np.arange(p_all), aliased)
    Z = Z_all[:, kept]
    p = Z.shape[1]
    read_w = np.concatenate(
        [np.full(len(v), w) for v, w in zip(values, counts)]
    )
    centre = float(np.average(np.concatenate(values), weights=read_w))
    zbar = np.average(Z, axis=0, weights=counts) if p else np.zeros(0)
    reference = centre if reference is None else reference
    refs = (centre, zbar, reference)
    arrays = JointArrays.build(
        counts, T, d, L, Z - zbar, ages, values, U=U, Rt=Rt
    )
    kept_names = [names[j] for j in kept]
    layout = Layout(
        len(path_names),
        p,
        tuple(path_names.index(v) for v in fitter.varying),
        np.array(
            [[cov in shifts[n] for cov in kept_names] for n in path_names],
            dtype=bool,
        ).reshape(len(path_names), p),
        np.array([cov in hazard for cov in kept_names], dtype=bool),
    )
    model = Model(layout, PathFunction(path_model), centre)
    # Starting values: each item's path alone, their spread, then the
    # survival part on each item's predicted path
    theta = marker_start(model, ages, values, Z - zbar, counts)
    theta = survival_start(model, arrays, theta)
    u = layout.pack(theta)
    total = float(np.sum(counts))

    def objective(v: Any, placed: tuple) -> Any:
        return neg_ll(v, model, arrays, placed)

    search = value_and_grad(objective)
    curvature = hessian(objective)

    def evaluated(v: npt.NDArray, starts: tuple) -> tuple:
        """The likelihood at ``v`` by the nodes placed there (``inf``
        where they cannot be placed), with the nodes and the modes."""
        try:
            with np.errstate(all="ignore"):
                placed, starts = place(v, model, arrays, starts)
                value = float(objective(v, placed))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return np.inf, None, starts
        return (value if np.isfinite(value) else np.inf), placed, starts

    value, placed, starts = evaluated(u, (None, None))
    if placed is None:
        raise ValueError(
            "The joint fit could not start: the likelihood is not finite at "
            "its starting values (check the readings and the path model)."
        )
    converged = False
    for _ in range(_ROUNDS):
        # A search with the nodes held where they are, then the step it
        # proposes taken as far as the likelihood by nodes placed at the
        # new point improves on this one's (backtracking where not):
        # each round is a descent of the adaptive quadrature's likelihood.
        S = _whitening(curvature, u, placed)

        def whitened(v: npt.NDArray, u0: npt.NDArray = u) -> tuple:
            with np.errstate(all="ignore"):
                f, grad = search(u0 + S @ v, placed)
            if not np.isfinite(f) or not np.all(np.isfinite(grad)):
                return np.inf, np.zeros_like(v)
            return f, S.T @ grad

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                whitened,
                np.zeros(u.size),
                jac=True,
                method="BFGS",
                options={"gtol": 1e-7, "maxiter": 5000},
            )
        step = result.x if np.all(np.isfinite(result.x)) else 0 * result.x
        size = 1.0
        while True:
            trial = u + S @ (size * step)
            new_value, new_placed, new_starts = evaluated(trial, starts)
            if new_value <= value + 1e-12 * max(1.0, abs(value)):
                break
            size /= 4.0
            if size < 1e-6:
                trial, new_value, new_placed = u, value, placed
                new_starts = starts
                break
        moved = float(np.max(np.abs(size * step), initial=0.0))
        settled = abs(value - new_value) <= _ROUND_TOL * max(1.0, abs(value))
        u, value, placed, starts = trial, new_value, new_placed, new_starts
        if settled and moved < 1e-3:
            converged = True
            break
    # The maximum, verified at the last nodes: a zero gradient (in
    # standard errors) and a positive-definite Hessian
    with np.errstate(all="ignore"):
        value, grad = search(u, placed)
        info = curvature(u, placed)
    info = 0.5 * (info + info.T)
    cov_u, positive = None, False
    if np.all(np.isfinite(info)):
        try:
            positive = bool(np.all(np.linalg.eigvalsh(info) > 0))
            cov_u = np.linalg.inv(info) if positive else None
        except np.linalg.LinAlgError:
            positive = False
    flat = False
    if cov_u is not None and np.all(np.isfinite(grad)):
        sd = np.sqrt(np.abs(np.diag(cov_u)))
        flat = bool(np.max(np.abs(grad) * sd) < _GTOL)
    verified = converged and positive and flat and np.all(np.isfinite(u))
    if not verified:
        reason = None
        if not converged:
            reason = "the rounds of placing the nodes did not settle"
        elif not positive:
            reason = "the Hessian is not positive definite there"
        elif not flat:
            reason = "the gradient is not zero there"
        warn_unverified("The joint fit", reason)
    estimated = np.asarray(
        reported_params(layout.unpack(u), layout, refs), dtype=float
    )
    covariance = None
    if cov_u is not None:
        J = jacobian(
            lambda v: reported_params(layout.unpack(v), layout, refs)
        )(u)
        covariance = J @ cov_u @ J.T
        covariance = 0.5 * (covariance + covariance.T)
    else:
        warn_no_covariance()
    # every parameter, nan for an aliased covariate's
    full_names = parameter_names(
        path_names, list(fitter.varying), names, shifts, hazard
    )
    kept_names = parameter_names(
        path_names,
        list(fitter.varying),
        kept_names,
        {n: [c for c in shifts[n] if c in kept_names] for n in path_names},
        [c for c in hazard if c in kept_names],
    )
    where = [full_names.index(name) for name in kept_names]
    params = np.full(len(full_names), np.nan)
    params[where] = estimated
    out = JointModel()
    out.dist = fitter.dist
    out.path_model = path_model
    out.varying = list(fitter.varying)
    out.params = params
    out.feature_names = list(names)
    out.aliased = aliased
    every_one = all(shifts[n] == list(names) for n in path_names) and list(
        hazard
    ) == list(names)
    # saved as None where every covariate is in both parts (as before)
    out.path_covariates = (
        None if every_one else {n: list(shifts[n]) for n in path_names}
    )
    out.hazard_covariates = None if every_one else list(hazard)
    out.parameter_names = full_names
    out.marker_reference = float(reference)
    out._centre = centre
    out._zbar = zbar
    if covariance is not None:
        full_cov = np.full((len(full_names),) * 2, np.nan)
        full_cov[np.ix_(where, where)] = covariance
        out._covariance = full_cov
    out._neg_ll = float(value)
    out.k = len(estimated)
    out.maximum = "verified" if verified else "unverified"
    out.n_items = total
    out.n_events = float(np.sum(np.maximum(d, np.isfinite(U)) * counts))
    out.n_readings = float(np.sum([len(a) * w for a, w in zip(ages, counts)]))
    return out


def _whitening(curvature: Any, u: npt.NDArray, placed: tuple) -> npt.NDArray:
    """``S`` with ``S' H S = I`` for the Hessian ``H`` at ``u`` (its
    inverse Cholesky factor), so the search sees each parameter in its
    standard errors; the diagonal where ``H`` is not positive definite."""
    with np.errstate(all="ignore"):
        H = np.asarray(curvature(u, placed), dtype=float)
    H = 0.5 * (H + H.T)
    if np.all(np.isfinite(H)):
        try:
            root = np.linalg.cholesky(H)
            return np.linalg.inv(root).T
        except np.linalg.LinAlgError:
            pass
    diag = np.diag(H) if np.all(np.isfinite(np.diag(H))) else np.ones(u.size)
    return np.diag(1.0 / np.sqrt(np.where(diag > 0, diag, 1.0)))


def search_space(model: JointModel, **data: Any) -> tuple:
    """``(f, u, n)``: the negative log-likelihood the fit of ``data`` (the
    fit's arguments) searches, ``f(u)``, by the adaptive nodes placed at
    the model's parameters; the model's point ``u`` in that space; and
    the number of items. For the checks that a fit's answer is a
    maximum (``tests/conformance``)."""
    fitter = JointPH(model.dist, model.path_model, model.varying)
    records, counts, names = _records(
        data["x"],
        data.get("c"),
        data.get("n"),
        data.get("t"),
        data.get("tl"),
        data.get("tr"),
        data.get("Z"),
        data["markers"],
    )
    names = list(model.feature_names)
    T, d, L, Z_all, ages, values, U, Rt = _arrays(
        fitter, records, counts, len(names)
    )
    Z = Z_all[:, model._kept] - model._zbar
    arrays = JointArrays.build(counts, T, d, L, Z, ages, values, U=U, Rt=Rt)
    layout = model._layout
    lik = Model(layout, PathFunction(model.path_model), model._centre)
    theta = model._theta()
    u = layout.pack({k: np.asarray(v, dtype=float) for k, v in theta.items()})
    placed, _ = place(u, lik, arrays, (None, None))

    def f(v: Any) -> Any:
        return neg_ll(v, lik, arrays, placed)

    return f, u, float(np.sum(counts))
