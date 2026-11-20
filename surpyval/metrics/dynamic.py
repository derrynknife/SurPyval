"""How good are dynamic predictions: accuracy at an age, over a window.

A dynamic prediction is the chance that a unit still running at an age
``s`` fails within the next ``w``, given what was known about it at ``s``
(a joint model's readings so far, a landmark model's covariates then).
These functions measure how good those chances were, on units whose
outcome is known:

* :func:`dynamic_auc` -- discrimination: of a unit that failed within the
  window and one that did not, how often the one that failed was given
  the higher chance (0.5 is a coin toss, 1 is perfect);
* :func:`dynamic_brier` -- error: the mean squared difference between
  the predicted chance and what happened (0 is perfect; lower is better);
* :func:`dynamic_calibration` -- calibration: of the units given about
  30% risk, did about 30% fail?

Each takes the model (or several, to compare them on the same units, such
as a joint model against a Weibull fitted to the ages alone), the data,
one or more ages ``at`` and the window ``horizon``.

At each age ``s`` only the units still at risk at ``s`` (under observation
and running) are scored, and each is given its prediction from what was
known at ``s``: for a joint model, its readings up to and including age
``s``; readings after ``s`` are never used. A unit is a *case* if it
failed in ``(s, s + w]`` and a *control* if it was still running at
``s + w``; one censored inside the window is not known to be either, and
the cases and controls are weighted by the inverse probability of being
seen to the end of the window (the reverse Kaplan-Meier of the units at
risk at ``s``), as in Rizopoulos (2011) and Blanche, Proust-Lima, Loubere,
Berr, Dartigues and Jacqmin-Gadda (2015). With predictions
:math:`\\pi_i = 1 - S_i(s + w \\mid \\text{what was known at } s)`,

.. math::
    AUC(s, w) = P(\\pi_i > \\pi_j \\mid s < T_i \\le s + w,\\ T_j > s + w),
    \\qquad BS(s, w) = E[(I(T_i \\le s + w) - \\pi_i)^2 \\mid T_i > s].

The estimates are those of :func:`~surpyval.metrics.validation.auc_td`
and :func:`~surpyval.metrics.validation.brier_score` on the units at risk
at ``s``, at the horizon ``s + w``, with their predictions (the reverse
Kaplan-Meier of the units at risk at ``s`` is the censoring survival given
being under observation at ``s``). With no censoring inside the window
they are the plain AUC and mean squared error of the window's outcomes.
The standard errors, intervals and paired comparisons between models are
those of :func:`~surpyval.metrics.validation.score_predictions`.

A model scored on the units it was fitted to looks better than it is.
``cross_validate=k`` gives honest numbers: the units are split at random
into ``k`` groups, the model is refitted ``k`` times without one group,
and each unit is predicted by the fit that did not see it.

References
----------
Rizopoulos, D. (2011), "Dynamic predictions and prospective accuracy in
joint models for longitudinal and time-to-event data", Biometrics 67,
819-829.

Blanche, P., Proust-Lima, C., Loubere, L., Berr, C., Dartigues, J.-F. and
Jacqmin-Gadda, H. (2015), "Quantifying and comparing dynamic predictive
accuracy of joint models for longitudinal marker and time-to-event in
presence of censoring and competing risks", Biometrics 71, 102-113.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
from pandas import DataFrame
from scipy.stats import norm as _norm

from surpyval.utils.validation import check_alpha_ci

__all__ = [
    "DynamicAccuracy",
    "DynamicCalibration",
    "dynamic_auc",
    "dynamic_brier",
    "dynamic_calibration",
]

_SCORE_COLUMNS = [
    "model",
    "at",
    "n_at_risk",
    "n_events",
    "n_censored",
    "estimate",
    "se",
    "lower",
    "upper",
]
_COMPARISON_COLUMNS = [
    "model",
    "reference",
    "at",
    "difference",
    "se",
    "lower",
    "upper",
    "p_value",
]
_CALIBRATION_COLUMNS = [
    "model",
    "at",
    "bin",
    "risk_from",
    "risk_to",
    "n",
    "n_events",
    "n_censored",
    "predicted",
    "observed",
    "lower",
    "upper",
]


# -- the data -----------------------------------------------------------------


def _lives(data: Any) -> Any:
    """The units of ``data`` (each one's entry, end, event and covariates),
    checked for scoring."""
    from surpyval.univariate.landmark._data import Lives
    from surpyval.utils.surpyval_data import SurpyvalData

    if not isinstance(data, SurpyvalData):
        raise ValueError(
            "'data' must be a SurpyvalData of the units' lifetimes and "
            "their readings (SurpyvalData.from_items([...]), or "
            "SurpyvalData.from_dataframe(lifetimes, x_col=..., c_col=..., "
            "i_col=..., m=Measurements.from_dataframe(readings, ...))); got "
            f"{type(data).__name__}."
        )
    if np.asarray(data.x).ndim != 1 or np.isin(data.c, (-1, 2)).any():
        raise ValueError(
            "'data' has left or interval censored units, whose failure "
            "within a window is not known: give each unit's age at failure "
            "(c = 0) or at censoring while still running (c = 1)."
        )
    if getattr(data, "e", None) is not None:
        raise ValueError(
            "'data' has event types: the accuracy is of one kind of "
            "failure; give the data of that one, the others censored."
        )
    if getattr(data, "w", None) is not None:
        raise ValueError(
            "'data' has weights (w), which the accuracy does not take: "
            "build the data without them."
        )
    n = np.asarray(data.n, dtype=float)
    if np.any(n != np.round(n)):
        raise ValueError(
            "'data' has counts (n) that are not whole numbers: each row "
            "counts identical units, so give whole counts (weights are not "
            "taken)."
        )
    if data.m is not None and len(data.m.names) > 1:
        raise ValueError(
            f"'data' holds {len(data.m.names)} measurements ("
            + ", ".join(map(str, data.m.names))
            + "); the joint models read one: keep it with "
            "data.with_measurement(name)."
        )
    return Lives(data)


def _horizon(at: Any, horizon: Any) -> tuple[npt.NDArray, float]:
    from surpyval.univariate.landmark._data import check_landmarks

    return check_landmarks(at, horizon)


def _at_risk(lives: Any, s: float) -> npt.NDArray:
    """The units under observation and running at ``s``."""
    return np.flatnonzero((lives.entry <= s) & (lives.end > s))


# -- the predictions ----------------------------------------------------------


def _is_fitter(obj: Any) -> bool:
    """Whether ``obj`` fits (a fitter, ``LandmarkCox`` itself, or a
    function of the data) rather than predicts."""
    from surpyval.univariate.joint.joint_model import JointModel
    from surpyval.univariate.landmark.landmark_cox import LandmarkCox

    if isinstance(obj, (JointModel, LandmarkCox)):
        return False
    if obj is LandmarkCox or callable(getattr(obj, "fit", None)):
        return True
    return callable(obj) and not hasattr(obj, "sf")


def _named(model: Any) -> dict:
    if isinstance(model, dict):
        if not model:
            raise ValueError(
                "'model' is an empty dict; give one model, or a dict of "
                "them keyed by name."
            )
        named = {str(k): v for k, v in model.items()}
        if len(named) != len(model):
            raise ValueError("'model' has two models with one name.")
        return named
    return {"model": model}


def _joint_covariates(
    model: Any, data: Any, row: int
) -> dict | npt.NDArray | None:
    """The covariates of the unit on ``row`` as a joint model reads them:
    by name where the data names them."""
    features = list(model.feature_names)
    if not features:
        return None
    if data.Z is None:
        raise ValueError(
            f"The model was fitted on the covariates {features!r}, and "
            "'data' has none: give the units' covariates in the data."
        )
    Z = np.asarray(data.Z, dtype=float).reshape(len(data.x), -1)
    names = data.covariate_names
    if names is None:
        if Z.shape[1] != len(features):
            raise ValueError(
                f"The model was fitted on {len(features)} covariate(s) and "
                f"'data' has {Z.shape[1]}: give the data the model's "
                "covariates, named."
            )
        return Z[row]
    names = [str(n) for n in names]
    missing = [f for f in features if f not in names]
    if missing:
        raise ValueError(
            f"The model was fitted on the covariates {features!r}; 'data' "
            f"has no {missing!r}."
        )
    return {f: float(Z[row, names.index(f)]) for f in features}


def _joint_risk(
    model: Any, data: Any, lives: Any, units: npt.NDArray, s: float, w: float
) -> npt.NDArray:
    several = [lives.ids[j] for j in units if lives.rows[j].size > 1]
    if several:
        from surpyval.utils.warnings import named

        raise ValueError(
            "A joint model takes one row per unit, with its readings; "
            f"unit(s) {named(several)} have several rows in 'data'."
        )
    markers = data._markers_by_row() if data.m is not None else None
    out = np.full(units.size, np.nan)
    for k, j in enumerate(units):
        row = int(lives.rows[j][0])
        known = (
            {}
            if markers is None
            else {a: v for a, v in markers[row].items() if a <= s}
        )
        z = _joint_covariates(model, data, row)
        out[k] = float(model.ff(s + w, z, markers=known or None, given=s))
    return out


def _landmark_risk(
    model: Any, lives: Any, units: npt.NDArray, s: float, w: float
) -> npt.NDArray:
    if w > model.window * (1 + 1e-12):
        raise ValueError(
            f"'horizon' is {w:g}, and the LandmarkCox model predicts at "
            f"most its window, {model.window:g}, ahead: score a horizon of "
            f"{model.window:g} or less, or refit with horizon={w:g}."
        )
    rows, Z, _ = lives.at(s)
    names = list(lives.names)
    missing = [f for f in model.feature_names if f not in names]
    if missing:
        raise ValueError(
            f"The LandmarkCox model was fitted on {model.feature_names!r}; "
            f"'data' has no {missing!r} (its covariates and measurements "
            f"are {names!r})."
        )
    cols = [names.index(f) for f in model.feature_names]
    out = np.full(units.size, np.nan)
    known = np.isin(units, rows)
    if known.any():
        position = {int(r): k for k, r in enumerate(rows)}
        Zs = Z[[position[int(j)] for j in units[known]]][:, cols]
        out[known] = np.asarray(
            model.ff(np.full(Zs.shape[0], s + w), Zs, s), dtype=float
        ).ravel()
    return out


def _plain_risk(
    model: Any, data: Any, lives: Any, units: npt.NDArray, s: float, w: float
) -> npt.NDArray:
    sf = getattr(model, "sf", None)
    if not callable(sf):
        raise ValueError(
            "'model' must be a fitted model with sf (a JointModel, a "
            "LandmarkCox, a distribution or a regression fit), or, with "
            "cross_validate, a fitter; got "
            f"{type(model).__name__}."
        )
    n = units.size
    try:
        takes_Z = "Z" in inspect.signature(sf).parameters
    except (TypeError, ValueError):
        takes_Z = False
    with np.errstate(all="ignore"):
        if takes_Z:
            if lives.fixed is None:
                raise ValueError(
                    "The model's sf takes covariates (Z), and 'data' has "
                    "none: give the units' covariates in the data."
                )
            held = [lives._row_at(int(j), s) for j in units]
            Zq: Any = lives.fixed[held]
            if data.covariate_names is not None:
                Zq = DataFrame(
                    Zq, columns=[str(c) for c in data.covariate_names]
                )
            at_s = np.asarray(sf(np.full(n, s), Zq), dtype=float).ravel()
            at_end = np.asarray(sf(np.full(n, s + w), Zq), dtype=float).ravel()
        else:
            both = np.asarray(sf(np.array([s, s + w])), dtype=float).ravel()
            at_s, at_end = np.full(n, both[0]), np.full(n, both[1])
        risk = np.where(at_s > 0, 1.0 - at_end / at_s, np.nan)
    if risk.size != n:
        raise ValueError(
            f"model.sf returned {risk.size} values for {n} units; its "
            "sf(x, Z) convention is not supported."
        )
    return risk


def _risk(
    model: Any, data: Any, lives: Any, units: npt.NDArray, s: float, w: float
) -> npt.NDArray:
    """Each unit's predicted chance of failing in ``(s, s + w]``, given
    that it is running at ``s`` and what was known about it then; ``nan``
    where the model gives none."""
    from surpyval.univariate.joint.joint_model import JointModel
    from surpyval.univariate.landmark.landmark_cox import LandmarkCox

    if units.size == 0:
        return np.zeros(0)
    if isinstance(model, JointModel):
        return _joint_risk(model, data, lives, units, s, w)
    if isinstance(model, LandmarkCox):
        return _landmark_risk(model, lives, units, s, w)
    return _plain_risk(model, data, lives, units, s, w)


def _refit(fitter: Any, train: Any, at: npt.NDArray, w: float) -> Any:
    """The model ``fitter`` fits to ``train``: a fitter's ``fit`` (the
    readings left out for a fit that takes none), ``LandmarkCox`` fitted
    at the ages scored with the window, or a function of the data."""
    from surpyval.univariate.landmark.landmark_cox import LandmarkCox

    if fitter is LandmarkCox:
        return LandmarkCox.fit(train, at=at, horizon=w)
    fit = getattr(fitter, "fit", None)
    if callable(fit):
        try:
            takes_markers = "markers" in inspect.signature(fit).parameters
        except (TypeError, ValueError):
            takes_markers = True
        if train.m is not None and not takes_markers:
            train = train.without("m")
        return fit(train)
    return fitter(train)


def _folds(cross_validate: Any, n_units: int) -> int | None:
    if cross_validate is None:
        return None
    if (
        isinstance(cross_validate, bool)
        or not isinstance(cross_validate, (int, np.integer))
        or not 2 <= cross_validate <= n_units
    ):
        raise ValueError(
            "'cross_validate' is the number of groups the units are split "
            f"into, a whole number from 2 to the number of units ({n_units}"
            f"); got {cross_validate!r}. Leave it out to score the model as "
            "given."
        )
    return int(cross_validate)


def _window_predictions(
    model: Any,
    data: Any,
    at: Any,
    horizon: Any,
    cross_validate: Any,
    random_state: Any,
) -> tuple[list[str], npt.NDArray, float, list[dict]]:
    """The models' names, the ages, the window, and at each age the units
    scored: their ages at failure or censoring, censoring flags and each
    model's predicted chance of failing in the window."""
    models = _named(model)
    lives = _lives(data)
    s_all, w = _horizon(at, horizon)
    k = _folds(cross_validate, len(lives.ids))
    for name, m in models.items():
        label = "'model'" if len(models) == 1 else f"'model'[{name!r}]"
        if k is None and _is_fitter(m):
            raise ValueError(
                f"{label} is a fitter, which fits rather than predicts: fit "
                "it first, or give cross_validate=k to refit it on k groups "
                "of the units."
            )
        if k is not None and not _is_fitter(m):
            raise ValueError(
                f"{label} is a fitted model, and cross_validate refits: "
                "give the fitter (such as WeibullJointPH, Weibull or "
                "LandmarkCox) or a function of the data returning a fitted "
                "model."
            )
    n_units = len(lives.ids)
    risk = {name: np.full((s_all.size, n_units), np.nan) for name in models}
    if k is None:
        for i, s in enumerate(s_all):
            units = _at_risk(lives, float(s))
            for name, m in models.items():
                risk[name][i, units] = _risk(m, data, lives, units, s, w)
    else:
        from surpyval.utils.rng import as_generator

        fold = np.empty(n_units, dtype=int)
        fold[as_generator(random_state).permutation(n_units)] = (
            np.arange(n_units) % k
        )
        for f in range(k):
            rows = np.sort(
                np.concatenate(
                    [lives.rows[j] for j in np.flatnonzero(fold != f)]
                )
            )
            train = data[rows]
            fitted = {
                name: _refit(m, train, s_all, w) for name, m in models.items()
            }
            for i, s in enumerate(s_all):
                units = _at_risk(lives, float(s))
                units = units[fold[units] == f]
                for name, m in fitted.items():
                    risk[name][i, units] = _risk(m, data, lives, units, s, w)
    out = []
    dropped = []
    for i, s in enumerate(s_all):
        units = _at_risk(lives, float(s))
        ok = np.ones(units.size, dtype=bool)
        for name in models:
            ok &= np.isfinite(risk[name][i, units])
        # a row counted n is n identical units, with one prediction
        count = lives.n[units].astype(int)
        if not ok.all():
            dropped.append((float(s), int(count[~ok].sum()), count.sum()))
        units, count = units[ok], count[ok]
        out.append(
            {
                "at": float(s),
                "x": np.repeat(lives.end[units].astype(float), count),
                "c": np.repeat(np.where(lives.event[units], 0, 1), count),
                "risk": {
                    name: np.repeat(risk[name][i, units], count)
                    for name in models
                },
            }
        )
    if dropped:
        from surpyval.utils.warnings import caller_stacklevel

        warnings.warn(
            "Some units at risk had no prediction (a missing covariate, or "
            "no reading by the age for a landmark model) and were left out "
            "of the scores: "
            + ", ".join(f"{d} of {n} at {s:g}" for s, d, n in dropped)
            + ". Give every unit its covariates (and a reading by the first "
            "age scored), or score later ages.",
            UserWarning,
            stacklevel=caller_stacklevel(),
        )
    return list(models), s_all, w, out


def _window_counts(x: npt.NDArray, c: npt.NDArray, end: float) -> tuple:
    inside = x <= end
    return int(np.sum(inside & (c == 0))), int(np.sum(inside & (c == 1)))


# -- AUC and Brier score ------------------------------------------------------


class _ResultIO:
    """``to_dict`` / ``from_dict`` of a result whose numbers are tables."""

    _TABLES: tuple = ()

    def _settings(self) -> dict:
        raise NotImplementedError

    def to_dict(self) -> dict:
        """The result as a dict of native Python values, strict JSON: a
        missing number is written as ``None`` and recorded under
        ``"non_finite"``.

        Returns
        -------
        result : dict
            The settings and each table as a list of records.
        """
        from surpyval.serialisation import encode_non_finite, to_native

        out = {"result": type(self).__name__, **self._settings()}
        for table in self._TABLES:
            out[table] = [
                {k: to_native(v) for k, v in row.items()}
                for row in getattr(self, table).to_dict("records")
            ]
        return encode_non_finite(out)

    @classmethod
    def _decoded(cls, result_dict: dict) -> dict:
        from surpyval.serialisation import decode_non_finite

        if result_dict.get("result") != cls.__name__:
            raise ValueError(
                f"Must create a {cls.__name__} from a dict its to_dict wrote "
                f"(with 'result': '{cls.__name__}')."
            )
        return decode_non_finite(dict(result_dict))


def _cv_words(cross_validate: int | None) -> str:
    if cross_validate is None:
        return (
            "Scored on the units given, as the model was fitted (if it was "
            "fitted to them, the numbers are optimistic: cross_validate=k "
            "refits it without each unit's group)."
        )
    return (
        f"Cross-validated: the units split into {cross_validate} groups, "
        "each predicted by the model fitted without its group."
    )


class DynamicAccuracy(_ResultIO):
    """
    How good a model's dynamic predictions are at each age: the AUC
    (:func:`dynamic_auc`) or the Brier score (:func:`dynamic_brier`) of
    its chance of failing within the window, with standard errors,
    intervals and, for several models scored on the same units, paired
    comparisons.

    Attributes
    ----------
    metric : str
        ``"auc"`` or ``"brier"``.
    at : ndarray
        The ages scored, sorted.
    horizon : float
        The window: the chance scored is that of failing in ``(s, s +
        horizon]``.
    models : list of str
        The names of the models, in the order given.
    alpha_ci : float
        The intervals are two-sided ``1 - alpha_ci`` intervals.
    cross_validate : int or None
        The number of groups the units were split into for
        cross-validation, or ``None`` (the models scored as given).
    scores : pandas.DataFrame
        One row per model and age: ``model``, ``at``, ``n_at_risk`` (the
        units scored: at risk at the age, with a prediction from every
        model), ``n_events`` (of those, the failures in the window),
        ``n_censored`` (those censored in the window, by the end of it),
        ``estimate``, ``se``, ``lower`` and ``upper``. ``nan`` where the
        estimate is not defined: no unit at risk, or (for the AUC) no
        failure or no survivor in the window.
    comparisons : pandas.DataFrame
        One row per pair of models and age: ``model`` and ``reference``
        (each model against each listed before it), ``difference``
        (``model`` minus ``reference``), ``se``, ``lower``, ``upper`` and
        ``p_value`` (the two-sided Wald test of no difference). Empty
        with one model.

    Examples
    --------
    :func:`dynamic_brier` and :func:`dynamic_auc` return one; here a
    Weibull fitted to the ages alone, scored at two ages over a window of
    40, and written as strict JSON:

    >>> from surpyval import SurpyvalData, Weibull
    >>> from surpyval.metrics import dynamic_brier
    >>> data = SurpyvalData(x=[30, 45, 60, 80, 90, 120, 150, 150],
    ...                     c=[0, 0, 1, 0, 0, 0, 1, 1])
    >>> model = Weibull.fit(data)
    >>> result = dynamic_brier(model, data, at=[20, 50], horizon=40)
    >>> result.scores["n_at_risk"].tolist()
    [8, 6]
    >>> sorted(result.to_dict())[:3]
    ['alpha_ci', 'at', 'comparisons']
    """

    _TABLES = ("scores", "comparisons")

    def __init__(
        self,
        metric: str,
        at: npt.NDArray,
        horizon: float,
        models: list[str],
        alpha_ci: float,
        cross_validate: int | None,
        scores: DataFrame,
        comparisons: DataFrame,
    ) -> None:
        self.metric = metric
        self.at = np.asarray(at, dtype=float)
        self.horizon = float(horizon)
        self.models = list(models)
        self.alpha_ci = float(alpha_ci)
        self.cross_validate = cross_validate
        self.scores = scores
        self.comparisons = comparisons

    def _column(self, column: str, model: str | None) -> npt.NDArray:
        if model is None:
            model = self.models[0]
        if model not in self.models:
            raise ValueError(
                f"'model' must be one of {self.models}; got {model!r}."
            )
        table = self.scores
        return table.loc[table["model"] == model, column].to_numpy(float)

    def estimate(self, model: str | None = None) -> npt.NDArray:
        """The estimates at each age of ``at``, for ``model`` (the first
        by default).

        Parameters
        ----------
        model : str, optional
            The model's name.

        Returns
        -------
        estimate : ndarray
        """
        return self._column("estimate", model)

    def se(self, model: str | None = None) -> npt.NDArray:
        """The standard errors at each age of ``at``, for ``model`` (the
        first by default).

        Parameters
        ----------
        model : str, optional
            The model's name.

        Returns
        -------
        se : ndarray
        """
        return self._column("se", model)

    def _settings(self) -> dict:
        return {
            "metric": self.metric,
            "at": [float(s) for s in self.at],
            "horizon": self.horizon,
            "models": list(self.models),
            "alpha_ci": self.alpha_ci,
            "cross_validate": self.cross_validate,
        }

    @classmethod
    def from_dict(cls, result_dict: dict) -> DynamicAccuracy:
        """The result :meth:`to_dict` wrote.

        Parameters
        ----------
        result_dict : dict
            A dict written by :meth:`to_dict`.

        Returns
        -------
        result : DynamicAccuracy
        """
        d = cls._decoded(result_dict)
        return cls(
            metric=d["metric"],
            at=np.asarray(d["at"], dtype=float),
            horizon=float(d["horizon"]),
            models=list(d["models"]),
            alpha_ci=float(d["alpha_ci"]),
            cross_validate=d["cross_validate"],
            scores=DataFrame(d["scores"], columns=_SCORE_COLUMNS),
            comparisons=DataFrame(
                d["comparisons"], columns=_COMPARISON_COLUMNS
            ),
        )

    def __repr__(self) -> str:
        w = f"{self.horizon:g}"
        if self.metric == "auc":
            out = (
                "Dynamic AUC: among the units still running at each age, "
                f"how often one that failed within {w} after it had been "
                "given a higher chance of failing than one that did not "
                "(0.5 is a coin toss, 1 is perfect).\n"
            )
        else:
            out = (
                "Dynamic Brier score: the mean squared error of the "
                f"predicted chance of failing within {w} of each age, among "
                "the units still running then (0 is perfect; lower is "
                "better).\n"
            )
        level = f"{100 * (1 - self.alpha_ci):g}%"
        out += _cv_words(self.cross_validate) + "\n"
        out += f"Intervals: {level} two-sided.\n"
        out += self.scores.to_string(index=False)
        if len(self.comparisons):
            out += "\n\nComparisons (model minus reference)\n"
            out += self.comparisons.to_string(index=False)
        return out


def _accuracy(
    metric: str,
    model: Any,
    data: Any,
    at: Any,
    horizon: Any,
    cross_validate: Any,
    random_state: Any,
    alpha_ci: float,
) -> DynamicAccuracy:
    from surpyval.metrics.validation import score_predictions

    check_alpha_ci(alpha_ci)
    names, s_all, w, landmarks = _window_predictions(
        model, data, at, horizon, cross_validate, random_state
    )
    by_model: dict = {name: [] for name in names}
    comparison_rows: list = []
    for lm in landmarks:
        s, x, c = lm["at"], lm["x"], lm["c"]
        n_events, n_censored = _window_counts(x, c, s + w)
        counts = [s, x.size, n_events, n_censored]
        if x.size == 0:
            for name in names:
                by_model[name].append([name] + counts + [np.nan] * 4)
            for i, ref in enumerate(names):
                for name in names[i + 1 :]:
                    comparison_rows.append([name, ref, s] + [np.nan] * 5)
            continue
        result = score_predictions(
            x,
            c,
            {name: 1.0 - lm["risk"][name][:, None] for name in names},
            [s + w],
            metrics=(metric,),
            alpha_ci=alpha_ci,
        )
        for row in result.scores.itertuples(index=False):
            by_model[row.model].append(
                [row.model]
                + counts
                + [row.estimate, row.se, row.lower, row.upper]
            )
        for row in result.comparisons.itertuples(index=False):
            comparison_rows.append(
                [
                    row.model,
                    row.reference,
                    s,
                    row.difference,
                    row.se,
                    row.lower,
                    row.upper,
                    row.p_value,
                ]
            )
    scores = DataFrame(
        [row for name in names for row in by_model[name]],
        columns=_SCORE_COLUMNS,
    )
    return DynamicAccuracy(
        metric=metric,
        at=s_all,
        horizon=w,
        models=names,
        alpha_ci=alpha_ci,
        cross_validate=None if cross_validate is None else int(cross_validate),
        scores=scores,
        comparisons=DataFrame(comparison_rows, columns=_COMPARISON_COLUMNS),
    )


def dynamic_auc(
    model: Any,
    data: Any,
    at: npt.ArrayLike | float,
    horizon: float,
    cross_validate: int | None = None,
    random_state: Any = None,
    alpha_ci: float = 0.05,
) -> DynamicAccuracy:
    """
    How well a model's dynamic predictions rank the units: the dynamic
    AUC at each age ``s`` of ``at`` over the window ``horizon``.

    Among the units still running at ``s``, each is given its chance of
    failing within the window, from what was known about it at ``s``.
    Take one unit that failed in the window and one that did not: the AUC
    is the probability that the model gave the one that failed the higher
    chance. 0.5 is a coin toss; 1 is perfect. Units censored inside the
    window are accounted for by weighting the others (see the module
    notes); with no censoring in the window it is the plain AUC of the
    window's outcomes (ties count a half).

    Parameters
    ----------
    model : fitted model, fitter, or dict of them
        Anything that predicts a unit's chance of failing in the window
        from what was known at the age: a fitted joint model
        (``JointModel``, from ``WeibullJointPH`` or ``JointPH``), given
        each unit's covariates and its readings up to the age; a
        ``LandmarkCox``, given the covariates and the last reading of
        each measurement at or before the age; a regression model with
        ``sf(x, Z)``, given the covariates; or a distribution fitted to
        the ages alone (``sf(x)``), the same chance for every unit at
        that age. A dict of them, keyed by name, scores each on the same
        units and compares each with every one before it. With
        ``cross_validate``, the fitters instead: ``WeibullJointPH`` (or
        any ``JointPH`` fitter), ``Weibull`` or another fitter (given the
        data without the readings when its ``fit`` takes none),
        ``LandmarkCox`` (fitted at the ages ``at`` with the window
        ``horizon``), or a function of the data returning a fitted model.
    data : SurpyvalData
        The units: one row per unit with its age at failure or censoring,
        ``c`` (0 failed, 1 still running) and, for a joint model, its
        readings (``SurpyvalData.from_items`` of records with
        ``markers``, or ``SurpyvalData.from_dataframe(..., c_col=...,
        i_col=..., m=Measurements(...))``), with any late entry (``tl``)
        and covariates. Left- and interval-censored units, counts other
        than 1, weights and event types are refused.
    at : float or array_like
        The age(s) ``s`` at which the predictions are made.
    horizon : float
        The window ``w``: the chance scored is that of failing in ``(s, s
        + w]``.
    cross_validate : int, optional
        Split the units at random into this many groups and predict each
        group from the model refitted without it (``model`` then holds
        fitters). Default: score the models as given.
    random_state : optional
        The seed of the split into groups (see
        :func:`surpyval.utils.rng.as_generator`).
    alpha_ci : float, optional
        The intervals are two-sided ``1 - alpha_ci`` intervals; default
        0.05 (``alpha_ci=0.1`` for 90%).

    Returns
    -------
    DynamicAccuracy
        The AUC at each age with its standard error and interval, the
        counts behind it, and the paired comparisons between models.

    Examples
    --------
    A fleet of pumps whose vibration is read every 10 days; a joint model
    of the readings and the lifetimes, against a Weibull fitted to the
    ages alone, standing at age 40 and looking 30 ahead. The age alone
    gives every pump at 40 the same chance, so it ranks no better than a
    coin toss:

    >>> import numpy as np
    >>> from surpyval import ItemLifetime, SurpyvalData, Weibull
    >>> from surpyval import WeibullJointPH
    >>> from surpyval.metrics import dynamic_auc
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
    >>> data = SurpyvalData.from_items(items)
    >>> models = {
    ...     "age only": Weibull.fit(data.without("m")),
    ...     "joint": WeibullJointPH.fit(data),
    ... }
    >>> result = dynamic_auc(models, data, at=40, horizon=30)
    >>> result.estimate("age only")
    array([0.5])
    >>> bool(result.estimate("joint")[0] > 0.6)
    True
    """
    return _accuracy(
        "auc",
        model,
        data,
        at,
        horizon,
        cross_validate,
        random_state,
        alpha_ci,
    )


def dynamic_brier(
    model: Any,
    data: Any,
    at: npt.ArrayLike | float,
    horizon: float,
    cross_validate: int | None = None,
    random_state: Any = None,
    alpha_ci: float = 0.05,
) -> DynamicAccuracy:
    """
    How far a model's dynamic predictions are from what happened: the
    dynamic Brier score at each age ``s`` of ``at`` over the window
    ``horizon``.

    Among the units still running at ``s``, each is given its chance of
    failing within the window, from what was known about it at ``s``; the
    Brier score is the mean of the squared difference between that chance
    and 1 for a unit that failed in the window, 0 for one that did not.
    It rewards both ranking the units well and getting the chances right:
    0 is perfect, and a model that gives every unit the share that fails
    scores that share times one minus it. Units censored inside the window
    are accounted for by weighting the others (see the module notes);
    with no censoring in the window it is the plain mean squared error.
    Compare a model with a simpler one on the same units (a dict of
    models): the difference, its interval and p-value say whether, say,
    the readings cut the error.

    Parameters
    ----------
    model : fitted model, fitter, or dict of them
        Anything that predicts a unit's chance of failing in the window
        from what was known at the age: a fitted joint model
        (``JointModel``, from ``WeibullJointPH`` or ``JointPH``), given
        each unit's covariates and its readings up to the age; a
        ``LandmarkCox``, given the covariates and the last reading of
        each measurement at or before the age; a regression model with
        ``sf(x, Z)``, given the covariates; or a distribution fitted to
        the ages alone (``sf(x)``), the same chance for every unit at
        that age. A dict of them, keyed by name, scores each on the same
        units and compares each with every one before it. With
        ``cross_validate``, the fitters instead: ``WeibullJointPH`` (or
        any ``JointPH`` fitter), ``Weibull`` or another fitter (given the
        data without the readings when its ``fit`` takes none),
        ``LandmarkCox`` (fitted at the ages ``at`` with the window
        ``horizon``), or a function of the data returning a fitted model.
    data : SurpyvalData
        The units: one row per unit with its age at failure or censoring,
        ``c`` (0 failed, 1 still running) and, for a joint model, its
        readings (``SurpyvalData.from_items`` of records with
        ``markers``, or ``SurpyvalData.from_dataframe(..., c_col=...,
        i_col=..., m=Measurements(...))``), with any late entry (``tl``)
        and covariates. Left- and interval-censored units, counts other
        than 1, weights and event types are refused.
    at : float or array_like
        The age(s) ``s`` at which the predictions are made.
    horizon : float
        The window ``w``: the chance scored is that of failing in ``(s, s
        + w]``.
    cross_validate : int, optional
        Split the units at random into this many groups and predict each
        group from the model refitted without it (``model`` then holds
        fitters). Default: score the models as given.
    random_state : optional
        The seed of the split into groups (see
        :func:`surpyval.utils.rng.as_generator`).
    alpha_ci : float, optional
        The intervals are two-sided ``1 - alpha_ci`` intervals; default
        0.05 (``alpha_ci=0.1`` for 90%).

    Returns
    -------
    DynamicAccuracy
        The Brier score at each age with its standard error and interval,
        the counts behind it, and the paired comparisons between models.

    Examples
    --------
    Without censoring in the window it is the mean squared error. Eight
    units, a Weibull fitted to their ages, standing at 50 and looking 40
    ahead: six are running at 50, of which three fail by 90:

    >>> import numpy as np
    >>> from surpyval import SurpyvalData, Weibull
    >>> from surpyval.metrics import dynamic_brier
    >>> data = SurpyvalData(x=[30, 45, 60, 80, 90, 120, 150, 150],
    ...                     c=[0, 0, 0, 0, 0, 0, 1, 1])
    >>> model = Weibull.fit(data)
    >>> result = dynamic_brier(model, data, at=50, horizon=40)
    >>> p = 1 - model.sf(90) / model.sf(50)
    >>> failed = np.array([1, 1, 1, 0, 0, 0])
    >>> bool(np.isclose(result.estimate()[0], np.mean((failed - p) ** 2)))
    True

    A 90% interval:

    >>> dynamic_brier(model, data, at=50, horizon=40, alpha_ci=0.1)
    ... # doctest: +ELLIPSIS
    Dynamic Brier score: ...
    Intervals: 90% two-sided.
    ...
    """
    return _accuracy(
        "brier",
        model,
        data,
        at,
        horizon,
        cross_validate,
        random_state,
        alpha_ci,
    )


# -- calibration --------------------------------------------------------------


def _bin_edges(risk: npt.NDArray, bins: Any) -> npt.NDArray:
    """The edges of the bins of ``risk``: quantiles for a number of bins,
    else the edges given."""
    if isinstance(bins, (int, np.integer)) and not isinstance(bins, bool):
        q = np.quantile(risk, np.linspace(0.0, 1.0, int(bins) + 1))
        q = np.unique(q)
        return q if q.size > 1 else np.array([q[0], q[0]])
    return bins


def _check_bins(bins: Any) -> Any:
    if isinstance(bins, (int, np.integer)) and not isinstance(bins, bool):
        if bins < 1:
            raise ValueError(
                f"'bins' is a number of groups, 1 or more; got {bins!r}."
            )
        return int(bins)
    try:
        edges = np.asarray(bins, dtype=float)
    except (TypeError, ValueError):
        edges = np.zeros(0)
    if (
        edges.ndim != 1
        or edges.size < 2
        or not np.all(np.diff(edges) > 0)
        or edges[0] != 0
        or edges[-1] != 1
    ):
        raise ValueError(
            "'bins' is a number of groups of equal size, or the edges of "
            "the groups of predicted risk, increasing from 0 to 1 (such as "
            f"[0, 0.1, 0.2, 0.5, 1]); got {bins!r}."
        )
    return edges


def _km_failed(
    x: npt.NDArray, c: npt.NDArray, end: float, z: float
) -> tuple[float, float, float]:
    """The share failed by ``end`` (one minus the Kaplan-Meier survival of
    the units, all running at the age) and its two-sided interval
    (Greenwood's variance on the log(-log) scale; ``nan`` where nothing
    failed or nothing survived)."""
    if x.size == 0:
        return np.nan, np.nan, np.nan
    S, gw = 1.0, 0.0
    for u in np.unique(x[(c == 0) & (x <= end)]):
        r = float(np.sum(x >= u))
        d = float(np.sum((x == u) & (c == 0)))
        S *= 1.0 - d / r
        if r > d:
            gw += d / (r * (r - d))
    if not 0.0 < S < 1.0:
        return 1.0 - S, np.nan, np.nan
    half = z * np.sqrt(gw) / abs(np.log(S))
    s_low, s_high = S ** np.exp(half), S ** np.exp(-half)
    return 1.0 - S, 1.0 - s_high, 1.0 - s_low


class DynamicCalibration(_ResultIO):
    """
    Whether a model's dynamic predictions mean what they say: the units at
    each age grouped by their predicted chance of failing within the
    window, and, in each group, the mean predicted chance beside the
    share that failed. From :func:`dynamic_calibration`.

    Attributes
    ----------
    at : ndarray
        The ages scored, sorted.
    horizon : float
        The window.
    models : list of str
        The names of the models, in the order given.
    bins : int or list of float
        The number of groups (of about equal size), or their edges.
    alpha_ci : float
        The intervals on the share failed are two-sided ``1 - alpha_ci``
        intervals.
    cross_validate : int or None
        The number of groups of units for cross-validation, or ``None``.
    table : pandas.DataFrame
        One row per model, age and group: ``model``, ``at``, ``bin``
        (numbered from 1, lowest risk first), ``risk_from`` and
        ``risk_to`` (the group's edges of predicted risk; the last group
        includes its upper edge), ``n`` (its units), ``n_events`` and
        ``n_censored`` (its failures and censorings in the window),
        ``predicted`` (its mean predicted chance), ``observed`` (the
        share that failed in the window: one minus the Kaplan-Meier
        survival of the group from the age, so units censored inside the
        window count for the time they were seen), and ``lower`` and
        ``upper`` (the interval on ``observed``, Greenwood's on the
        log(-log) scale; ``nan`` where no unit, or every unit, failed).

    Examples
    --------
    >>> from surpyval import SurpyvalData, Weibull
    >>> from surpyval.metrics import dynamic_calibration
    >>> data = SurpyvalData(x=[30, 45, 60, 80, 90, 120, 150, 150],
    ...                     c=[0, 0, 1, 0, 0, 0, 1, 1])
    >>> result = dynamic_calibration(Weibull.fit(data), data, at=20,
    ...                              horizon=60, bins=1)
    >>> result.table[["n", "n_events", "n_censored"]].values.tolist()
    [[8, 3, 1]]
    """

    _TABLES = ("table",)

    def __init__(
        self,
        at: npt.NDArray,
        horizon: float,
        models: list[str],
        bins: Any,
        alpha_ci: float,
        cross_validate: int | None,
        table: DataFrame,
    ) -> None:
        self.at = np.asarray(at, dtype=float)
        self.horizon = float(horizon)
        self.models = list(models)
        self.bins = bins
        self.alpha_ci = float(alpha_ci)
        self.cross_validate = cross_validate
        self.table = table

    def _settings(self) -> dict:
        return {
            "at": [float(s) for s in self.at],
            "horizon": self.horizon,
            "models": list(self.models),
            "bins": (
                self.bins
                if isinstance(self.bins, int)
                else [float(b) for b in self.bins]
            ),
            "alpha_ci": self.alpha_ci,
            "cross_validate": self.cross_validate,
        }

    @classmethod
    def from_dict(cls, result_dict: dict) -> DynamicCalibration:
        """The result :meth:`to_dict` wrote.

        Parameters
        ----------
        result_dict : dict
            A dict written by :meth:`to_dict`.

        Returns
        -------
        result : DynamicCalibration
        """
        d = cls._decoded(result_dict)
        bins = d["bins"]
        return cls(
            at=np.asarray(d["at"], dtype=float),
            horizon=float(d["horizon"]),
            models=list(d["models"]),
            bins=bins if isinstance(bins, int) else list(bins),
            alpha_ci=float(d["alpha_ci"]),
            cross_validate=d["cross_validate"],
            table=DataFrame(d["table"], columns=_CALIBRATION_COLUMNS),
        )

    def __repr__(self) -> str:
        level = f"{100 * (1 - self.alpha_ci):g}%"
        out = (
            f"Dynamic calibration of the chance of failing within "
            f"{self.horizon:g} of each age: the units still running at the "
            "age grouped by that chance; in each group 'predicted' is the "
            "mean chance and 'observed' the share that failed (Kaplan-Meier, "
            "so units censored in the window count for the time they were "
            "seen).\n"
        )
        out += _cv_words(self.cross_validate) + "\n"
        out += f"Intervals on 'observed': {level} two-sided.\n"
        return out + self.table.to_string(index=False)


def dynamic_calibration(
    model: Any,
    data: Any,
    at: npt.ArrayLike | float,
    horizon: float,
    bins: Any = 10,
    cross_validate: int | None = None,
    random_state: Any = None,
    alpha_ci: float = 0.05,
) -> DynamicCalibration:
    """
    Whether a model's dynamic predictions mean what they say: of the units
    given about a 30% chance of failing within the window, did about 30%
    fail?

    At each age ``s`` of ``at``, the units still running are given their
    chance of failing within ``horizon``, from what was known about them
    at ``s``, and grouped by it (``bins``: a number of groups of about
    equal size, or the groups' edges). In each group the mean predicted
    chance is set beside the share that failed in the window: one minus
    the group's Kaplan-Meier survival from ``s`` to ``s + horizon``, so a
    unit censored inside the window counts for the time it was seen. A
    well-calibrated model has the two close in every group.

    Parameters
    ----------
    model : fitted model, fitter, or dict of them
        Anything that predicts a unit's chance of failing in the window
        from what was known at the age: a fitted joint model
        (``JointModel``, from ``WeibullJointPH`` or ``JointPH``), given
        each unit's covariates and its readings up to the age; a
        ``LandmarkCox``, given the covariates and the last reading of
        each measurement at or before the age; a regression model with
        ``sf(x, Z)``, given the covariates; or a distribution fitted to
        the ages alone (``sf(x)``), the same chance for every unit at
        that age. A dict of them, keyed by name, scores each on the same
        units and compares each with every one before it. With
        ``cross_validate``, the fitters instead: ``WeibullJointPH`` (or
        any ``JointPH`` fitter), ``Weibull`` or another fitter (given the
        data without the readings when its ``fit`` takes none),
        ``LandmarkCox`` (fitted at the ages ``at`` with the window
        ``horizon``), or a function of the data returning a fitted model.
    data : SurpyvalData
        The units: one row per unit with its age at failure or censoring,
        ``c`` (0 failed, 1 still running) and, for a joint model, its
        readings (``SurpyvalData.from_items`` of records with
        ``markers``, or ``SurpyvalData.from_dataframe(..., c_col=...,
        i_col=..., m=Measurements(...))``), with any late entry (``tl``)
        and covariates. Left- and interval-censored units, counts other
        than 1, weights and event types are refused.
    at : float or array_like
        The age(s) ``s`` at which the predictions are made.
    horizon : float
        The window ``w``: the chance scored is that of failing in ``(s, s
        + w]``.
    cross_validate : int, optional
        Split the units at random into this many groups and predict each
        group from the model refitted without it (``model`` then holds
        fitters). Default: score the models as given.
    random_state : optional
        The seed of the split into groups (see
        :func:`surpyval.utils.rng.as_generator`).
    bins : int or array_like, optional
        The number of groups of about equal size (default 10; fewer where
        many units share a prediction), or the edges of the groups of
        predicted chance, increasing from 0 to 1 (such as ``[0, 0.1, 0.2,
        0.3, 0.4, 1]``).
    alpha_ci : float, optional
        The intervals on the share failed are two-sided ``1 - alpha_ci``
        intervals; default 0.05 (``alpha_ci=0.1`` for 90%).

    Returns
    -------
    DynamicCalibration
        The table of the groups at each age, for each model.

    Examples
    --------
    A Weibull fitted to the ages alone gives every unit at an age the same
    chance, so all fall in one group, whose predicted chance is close to
    the share that failed:

    >>> import numpy as np
    >>> from surpyval import SurpyvalData, Weibull
    >>> from surpyval.metrics import dynamic_calibration
    >>> rng = np.random.default_rng(1)
    >>> t = 100 * rng.weibull(2.0, 300)
    >>> data = SurpyvalData(x=np.minimum(t, 150), c=(t > 150).astype(int))
    >>> result = dynamic_calibration(Weibull.fit(data), data, at=50,
    ...                              horizon=40, bins=[0, 0.5, 1])
    >>> row = result.table.iloc[0]
    >>> bool(abs(row["predicted"] - row["observed"]) < 0.05)
    True
    """
    check_alpha_ci(alpha_ci)
    spec = _check_bins(bins)
    names, s_all, w, landmarks = _window_predictions(
        model, data, at, horizon, cross_validate, random_state
    )
    z = float(_norm.ppf(1 - alpha_ci / 2))
    rows = []
    for name in names:
        for lm in landmarks:
            s, x, c = lm["at"], lm["x"], lm["c"]
            risk = lm["risk"][name]
            if risk.size == 0:
                continue
            edges = _bin_edges(risk, spec)
            inner = edges[1:-1]
            group = np.searchsorted(inner, risk, side="right")
            for b in range(edges.size - 1):
                inside = group == b
                xb, cb = x[inside], c[inside]
                n_events, n_censored = _window_counts(xb, cb, s + w)
                observed, lower, upper = _km_failed(xb, cb, s + w, z)
                rows.append(
                    [
                        name,
                        s,
                        b + 1,
                        float(edges[b]),
                        float(edges[b + 1]),
                        int(inside.sum()),
                        n_events,
                        n_censored,
                        float(risk[inside].mean()) if inside.any() else np.nan,
                        observed,
                        lower,
                        upper,
                    ]
                )
    return DynamicCalibration(
        at=s_all,
        horizon=w,
        models=names,
        bins=spec if isinstance(spec, int) else [float(e) for e in spec],
        alpha_ci=alpha_ci,
        cross_validate=None if cross_validate is None else int(cross_validate),
        table=DataFrame(rows, columns=_CALIBRATION_COLUMNS),
    )
