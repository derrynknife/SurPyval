"""The integrated discrimination improvement (IDI) and the net
reclassification improvement (NRI) of a new risk model over an old one,
at a time horizon, on right-censored data: :func:`idi_nri`.

Both ask how the predicted risks of the event by the horizon move when
the old model is replaced by the new one. The IDI is how much the
predicted risk rises, on average, among those who have the event by the
horizon, plus how much it falls among those who do not. The NRI counts
moves instead of their size: the share of those with the event whose
risk goes up, less the share whose risk goes down, plus the same for
those without the event the other way round. With cut-offs the moves are
between risk categories (the categorical NRI); without them, any rise or
fall counts (the continuous, category-free NRI).

Who has the event by the horizon is not known for a unit censored before
it, so each estimate is weighted for the censoring: by the inverse
probability of being still uncensored (IPCW, as Uno, Tian, Cai, Kohane
and Wei 2013 and the other metrics in :mod:`surpyval.metrics`), or, for
the NRI, by Kaplan-Meier estimates within the units moving up and down
(Pencina, D'Agostino and Steyerberg 2011).

The NRI is not a proper scoring rule: it can reward a model that is
worse calibrated, or a marker that carries no information, and for
nested models its test adds nothing to the likelihood-ratio test of the
added covariate. The Brier score and its difference
(:func:`~surpyval.metrics.validation.score_predictions`) and the AUC
difference are the primary comparisons; see the notes of
:func:`idi_nri`.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
from pandas import DataFrame

from surpyval.utils import validate_1d as _as_1d
from surpyval.utils.ipcw import censoring_survival, step_at, step_left_limit
from surpyval.utils.linalg import bootstrap_quantile
from surpyval.utils.rng import as_generator
from surpyval.utils.validation import alpha_ci_error, check_option

__all__ = ["idi_nri", "RiskReclassification"]

#: The rows of :attr:`RiskReclassification.estimates`, in order (the
#: categorical NRI's only with cut-offs).
_IDI_ROWS = ("idi", "idi_events", "idi_nonevents")
_NRI_PARTS = (
    "",
    "_events",
    "_nonevents",
    "_up_events",
    "_down_events",
    "_up_nonevents",
    "_down_nonevents",
)
_TESTED = {
    "idi",
    "idi_events",
    "idi_nonevents",
    "nri",
    "nri_events",
    "nri_nonevents",
    "nri_categorical",
    "nri_categorical_events",
    "nri_categorical_nonevents",
}
_COLUMNS = ("measure", "estimate", "lower", "upper", "p_value")
_CENSORING = ("ipcw", "km")
_BOUNDS = ("perturbation", "bootstrap", "none")


def _nri_rows(prefix: str) -> list[str]:
    rows = [prefix + part for part in _NRI_PARTS[:3]]
    rows += [
        "p" + part + ("" if prefix == "nri" else "_categorical")
        for part in _NRI_PARTS[3:]
    ]
    return rows


def _risk(value: Any, n: int, name: str) -> npt.NDArray:
    """A predicted risk per unit, in [0, 1], refused otherwise."""
    r = np.asarray(value, dtype=float)
    if r.ndim == 2 and r.shape[1] == 1:
        r = r[:, 0]
    if r.ndim != 1 or r.size != n:
        raise ValueError(
            "'{}' must hold one predicted risk per unit ({} values); got "
            "shape {}".format(name, n, r.shape)
        )
    bad = ~np.isfinite(r)
    if bad.any():
        raise ValueError(
            "'{}' has {} missing or infinite values: drop those units "
            "(from 'x', 'c' and both risks) before comparing the "
            "models".format(name, int(bad.sum()))
        )
    if np.any((r < 0) | (r > 1)):
        raise ValueError(
            "'{}' must hold predicted probabilities of the event by "
            "'time', in [0, 1] (from {:.4g} to {:.4g} given). Pass 1 - "
            "S(time | Z), e.g. 1 - survival_probability(model, Z, "
            "[time])[:, 0], not a linear predictor or a hazard".format(
                name, float(r.min()), float(r.max())
            )
        )
    return r


def _categories(r: npt.NDArray, cutoffs: npt.NDArray) -> npt.NDArray:
    """The risk category of each unit: 0 below the first cut-off, k at or
    above the k-th (as ``nricens``'s ``categorize``)."""
    return np.searchsorted(cutoffs, r, side="right")


def _km_at(
    x: npt.NDArray, event: npt.NDArray, v: npt.NDArray, t: float
) -> float:
    """The weighted Kaplan-Meier survival at ``t`` (risk set ``x >= s``,
    as R's ``survfit``)."""
    times, s = censoring_survival(x, event, n=v, ties="censoring_first")
    return float(step_at(times, s, np.array([t]), before=1.0)[0])


class _Data:
    """What every resample needs: the times, who is a case or a control
    at the horizon, the risk differences and the moves."""

    def __init__(
        self,
        x: npt.NDArray,
        c: npt.NDArray,
        old: npt.NDArray,
        new: npt.NDArray,
        time: float,
        cutoffs: npt.NDArray | None,
    ) -> None:
        self.x = x
        self.event = c == 0
        self.time = time
        self.case = (x <= time) & self.event
        self.control = x > time
        self.diff = new - old
        self.moves = {"nri": (new > old, new < old)}
        if cutoffs is not None:
            k_old = _categories(old, cutoffs)
            k_new = _categories(new, cutoffs)
            self.moves["nri_categorical"] = (k_new > k_old, k_new < k_old)


def _weighted_mean(w: npt.NDArray, values: npt.NDArray) -> float:
    total = w.sum()
    return float(np.dot(w, values) / total) if total > 0 else np.nan


def _statistics(d: _Data, v: npt.NDArray, censoring: str) -> dict:
    """Every estimate, with each unit weighted by ``v`` (1 for the point
    estimate, Exp(1) draws for a perturbation, counts for a bootstrap)."""
    times, g = censoring_survival(d.x, ~d.event, n=v, ties="events_first")
    g_x = step_left_limit(times, g, d.x, before=1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        w_case = np.where(d.case & (g_x > 0), v / g_x, 0.0)
    # A control's weight 1 / G(time) is the same for all: it cancels.
    w_ctrl = np.where(d.control, v, 0.0)
    out = {
        "idi_events": _weighted_mean(w_case, d.diff),
        "idi_nonevents": 0.0 - _weighted_mean(w_ctrl, d.diff),
    }
    out["idi"] = out["idi_events"] + out["idi_nonevents"]
    if censoring == "km":
        surv = np.float64(_km_at(d.x, d.event, v, d.time))
        total = v.sum()
    for prefix, (up, down) in d.moves.items():
        suffix = "" if prefix == "nri" else "_categorical"
        if censoring == "ipcw":
            p = {
                "up_events": _weighted_mean(w_case, up),
                "down_events": _weighted_mean(w_case, down),
                "up_nonevents": _weighted_mean(w_ctrl, up),
                "down_nonevents": _weighted_mean(w_ctrl, down),
            }
        else:
            p = {}
            for name, moved in (("up", up), ("down", down)):
                share = float(v[moved].sum() / total)
                if share > 0:
                    s_moved = _km_at(
                        d.x[moved], d.event[moved], v[moved], d.time
                    )
                else:
                    s_moved = 0.0
                # nan where every unit, or none, has the event by the
                # horizon (a resample can draw that)
                with np.errstate(divide="ignore", invalid="ignore"):
                    p[name + "_events"] = (1 - s_moved) * share / (1 - surv)
                    p[name + "_nonevents"] = s_moved * share / surv
        for key, value in p.items():
            out["p_" + key + suffix] = float(value)
        events = p["up_events"] - p["down_events"]
        nonevents = p["down_nonevents"] - p["up_nonevents"]
        out[prefix + "_events"] = float(events)
        out[prefix + "_nonevents"] = float(nonevents)
        out[prefix] = float(events + nonevents)
    return out


def _order(categorical: bool) -> list[str]:
    rows = list(_IDI_ROWS) + _nri_rows("nri")
    if categorical:
        rows += _nri_rows("nri_categorical")
    return rows


class RiskReclassification:
    r"""The IDI and NRI of a new risk model over an old one at a horizon,
    with their bounds, from :func:`idi_nri`.

    Attributes
    ----------
    idi : float
        The integrated discrimination improvement: ``idi_events +
        idi_nonevents``.
    nri : float
        The continuous (category-free) NRI: ``nri_events +
        nri_nonevents``, from -2 to 2.
    nri_categorical : float or None
        The categorical NRI over ``cutoffs``; ``None`` without them.
    estimates : pandas.DataFrame
        One row per measure (``measure``), with ``estimate``, ``lower``,
        ``upper`` and ``p_value``:

        - ``idi_events``: the mean rise in predicted risk among those with
          the event by the horizon; ``idi_nonevents``: the mean fall among
          those without it; ``idi`` their sum.
        - ``nri_events``: among those with the event, the share whose
          risk goes up less the share whose risk goes down;
          ``nri_nonevents``: among those without it, the share going down
          less the share going up; ``nri`` their sum. The four shares are
          ``p_up_events``, ``p_down_events``, ``p_up_nonevents`` and
          ``p_down_nonevents``.
        - With ``cutoffs``, the same for moves between categories:
          ``nri_categorical``, ``nri_categorical_events``, ...,
          ``p_up_events_categorical``, ....

        ``lower`` and ``upper`` are the percentile bounds of the
        resamples, ``p_value`` the two-sided resampling p-value of the
        measure being 0 (twice the share of resamples on the other side
        of 0, as ``survIDINRI``), ``nan`` for the shares.
    tables : dict of pandas.DataFrame
        With ``cutoffs``: the reclassification tables, counts of units by
        old category (rows) and new category (columns), for ``"all"``
        units, the ``"events"`` (event by the horizon) and the
        ``"nonevents"`` (still event-free after it). Empty without
        cut-offs. The counts are not weighted for the censoring.
    time : float
        The horizon.
    cutoffs : ndarray or None
        The category boundaries of the categorical NRI.
    censoring : str
        ``"ipcw"`` or ``"km"`` (the NRI's handling of censoring; the IDI
        is always IPCW).
    bounds : str
        ``"perturbation"``, ``"bootstrap"`` or ``"none"``.
    n_resamples : int
        The number of resamples behind the bounds (0 for none).
    alpha_ci : float
        The bounds are ``1 - alpha_ci`` two-sided.
    n, n_events, n_nonevents : int
        The number of units, those with the event by the horizon and
        those still event-free after it; the rest were censored before
        the horizon.

    Examples
    --------
    >>> from surpyval.metrics import idi_nri
    >>> x = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> c = [0, 0, 1, 0, 0, 1, 0, 1]
    >>> old = [0.5, 0.4, 0.3, 0.5, 0.2, 0.3, 0.2, 0.1]
    >>> new = [0.6, 0.5, 0.2, 0.4, 0.2, 0.2, 0.1, 0.1]
    >>> res = idi_nri(x, c, old, new, time=4.5, bounds="none")
    >>> round(res.idi, 4), round(res.nri, 4)
    (0.075, 0.75)
    """

    def __init__(
        self,
        estimates: DataFrame,
        time: float,
        cutoffs: npt.NDArray | None,
        censoring: str,
        bounds: str,
        n_resamples: int,
        alpha_ci: float,
        n: int,
        n_events: int,
        n_nonevents: int,
        tables: dict | None = None,
    ) -> None:
        self.estimates = estimates
        self.time = float(time)
        self.cutoffs = None if cutoffs is None else np.asarray(cutoffs)
        self.censoring = censoring
        self.bounds = bounds
        self.n_resamples = int(n_resamples)
        self.alpha_ci = float(alpha_ci)
        self.n = int(n)
        self.n_events = int(n_events)
        self.n_nonevents = int(n_nonevents)
        self.tables = {} if tables is None else tables

    def _value(self, measure: str) -> float:
        rows = self.estimates[self.estimates["measure"] == measure]
        return float(rows["estimate"].iloc[0])

    @property
    def idi(self) -> float:
        return self._value("idi")

    @property
    def nri(self) -> float:
        return self._value("nri")

    @property
    def nri_categorical(self) -> float | None:
        if self.cutoffs is None:
            return None
        return self._value("nri_categorical")

    def __repr__(self) -> str:
        header = "IDI and NRI of the new model over the old"
        level = round(100 * (1 - self.alpha_ci), 6)
        lines = [
            header,
            "=" * len(header),
            f"Horizon          : {self.time:g}",
            f"Units            : {self.n} ({self.n_events} with the event "
            f"by the horizon, {self.n_nonevents} without)",
            f"Censoring        : {self.censoring}",
            "Bounds           : "
            + (
                "none"
                if self.bounds == "none"
                else f"{level:g}% {self.bounds}, {self.n_resamples} "
                "resamples"
            ),
            "",
        ]
        shown = self.estimates[self.estimates["measure"].isin(_TESTED)]
        lines.append(shown.to_string(index=False, float_format="%.4g"))
        lines.append(
            "\nThe NRI is not a proper scoring rule; compare the models by "
            "the Brier score difference first (score_predictions)."
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """The result as strict JSON: a missing number is written as
        ``None`` and recorded under ``"non_finite"``."""
        from surpyval.serialisation import encode_non_finite, to_native

        out = {
            "result": type(self).__name__,
            "time": self.time,
            "cutoffs": (
                None
                if self.cutoffs is None
                else to_native(np.asarray(self.cutoffs, dtype=float))
            ),
            "censoring": self.censoring,
            "bounds": self.bounds,
            "n_resamples": self.n_resamples,
            "alpha_ci": self.alpha_ci,
            "n": self.n,
            "n_events": self.n_events,
            "n_nonevents": self.n_nonevents,
            "estimates": [
                {k: to_native(v) for k, v in row.items()}
                for row in self.estimates.to_dict("records")
            ],
            "tables": {
                name: to_native(table.to_numpy())
                for name, table in self.tables.items()
            },
        }
        return encode_non_finite(out)

    @classmethod
    def from_dict(cls, result_dict: dict) -> RiskReclassification:
        """The result :meth:`to_dict` wrote.

        Parameters
        ----------
        result_dict : dict
            A dict written by :meth:`to_dict`.

        Returns
        -------
        RiskReclassification
        """
        from surpyval.serialisation import decode_non_finite

        if result_dict.get("result") != cls.__name__:
            raise ValueError(
                "Must create a RiskReclassification from a dict its to_dict "
                "wrote (with 'result': 'RiskReclassification')"
            )
        d = decode_non_finite(dict(result_dict))
        cutoffs = (
            None
            if d["cutoffs"] is None
            else np.asarray(d["cutoffs"], dtype=float)
        )
        estimates = DataFrame(d["estimates"], columns=list(_COLUMNS))
        for col in _COLUMNS[1:]:
            estimates[col] = estimates[col].astype(float)
        tables = {}
        if cutoffs is not None:
            labels = _category_labels(cutoffs)
            for name, counts in d.get("tables", {}).items():
                tables[name] = _table(np.asarray(counts, dtype=int), labels)
        return cls(
            estimates=estimates,
            time=float(d["time"]),
            cutoffs=cutoffs,
            censoring=d["censoring"],
            bounds=d["bounds"],
            n_resamples=int(d["n_resamples"]),
            alpha_ci=float(d["alpha_ci"]),
            n=int(d["n"]),
            n_events=int(d["n_events"]),
            n_nonevents=int(d["n_nonevents"]),
            tables=tables,
        )


def _category_labels(cutoffs: npt.NDArray) -> list[str]:
    edges = [f"{v:g}" for v in cutoffs]
    labels = ["< " + edges[0]]
    labels += [f"[{a}, {b})" for a, b in zip(edges[:-1], edges[1:])]
    labels.append(">= " + edges[-1])
    return labels


def _table(counts: npt.NDArray, labels: list[str]) -> DataFrame:
    table = DataFrame(counts, index=list(labels), columns=list(labels))
    table.index.name = "old"
    table.columns.name = "new"
    return table


def _tables(
    old: npt.NDArray, new: npt.NDArray, d: _Data, cutoffs: npt.NDArray
) -> dict:
    labels = _category_labels(cutoffs)
    k = len(labels)
    k_old = _categories(old, cutoffs)
    k_new = _categories(new, cutoffs)
    out = {}
    for name, keep in (
        ("all", np.ones(old.size, dtype=bool)),
        ("events", d.case),
        ("nonevents", d.control),
    ):
        counts = np.zeros((k, k), dtype=int)
        np.add.at(counts, (k_old[keep], k_new[keep]), 1)
        out[name] = _table(counts, labels)
    return out


def _check_cutoffs(cutoffs: Any) -> npt.NDArray | None:
    if cutoffs is None:
        return None
    cuts = np.atleast_1d(np.asarray(cutoffs, dtype=float)).ravel()
    if (
        cuts.size == 0
        or not np.isfinite(cuts).all()
        or np.any((cuts <= 0) | (cuts >= 1))
        or np.any(np.diff(cuts) <= 0)
    ):
        raise ValueError(
            "'cutoffs' must be increasing risks strictly between 0 and 1, "
            "the boundaries of the risk categories (e.g. [0.1, 0.2] for "
            "low, medium and high); got {}".format(cuts.tolist())
        )
    return cuts


def idi_nri(
    x: npt.ArrayLike,
    c: npt.ArrayLike,
    risk_old: npt.ArrayLike,
    risk_new: npt.ArrayLike,
    time: float,
    cutoffs: npt.ArrayLike | None = None,
    censoring: str = "ipcw",
    bounds: str = "perturbation",
    n_resamples: int = 1000,
    alpha_ci: float = 0.05,
    random_state: Any = None,
) -> RiskReclassification:
    r"""The integrated discrimination improvement (IDI) and the net
    reclassification improvement (NRI) of a new risk model over an old
    one at the horizon ``time``, on right-censored data.

    Both models give each unit a predicted risk of the event by
    ``time``, ``risk_old`` and ``risk_new``. The units with the event by
    ``time`` are the *events*; those still event-free after it the
    *non-events*. A better model should raise the risk of the events and
    lower the risk of the non-events.

    - The **IDI** is the mean rise in predicted risk among the events
      plus the mean fall among the non-events (Pencina, D'Agostino,
      D'Agostino and Vasan 2008): the gain in the gap between the two
      groups' mean risks, the "discrimination slope".
    - The **continuous NRI** counts moves rather than their size: the
      share of events whose risk goes up less the share going down, plus
      the share of non-events going down less the share going up
      (Pencina, D'Agostino and Steyerberg 2011). It runs from -2 to 2.
    - The **categorical NRI** (with ``cutoffs``) counts only the moves
      between risk categories, such as "below 10%", "10% to 20%" and
      "20% or more".

    The event and non-event parts are reported separately, as they
    should be read: a gain among the events can be paid for by a loss
    among the non-events.

    Parameters
    ----------
    x : array_like
        Observed times.
    c : array_like
        Censoring flags: ``0`` event, ``1`` right censored. Left and
        interval censoring are refused.
    risk_old, risk_new : array_like
        The old and the new model's predicted probabilities of the event
        by ``time``, one per unit, in [0, 1]: ``1 - S(time | Z)``, e.g.
        ``1 - survival_probability(model, Z, [time])[:, 0]``.
    time : float
        The horizon ``t``: events are the units with an event at or
        before it, non-events those whose time is after it.
    cutoffs : array_like, optional
        Increasing risks in (0, 1), the boundaries of the risk categories
        of the categorical NRI; a risk equal to a cut-off is in the
        higher category. Without them only the continuous NRI is given.
    censoring : {"ipcw", "km"}, optional
        How the NRI handles units censored before ``time``. ``"ipcw"``
        (default): the events are weighted by ``1 / G(x_i-)``, the
        inverse of the Kaplan-Meier estimate of still being uncensored
        (as the IDI, the Brier score and the AUC here). ``"km"``: the
        share of events among those moving up (or down) is one minus
        their own Kaplan-Meier survival at ``time`` (Pencina et al.
        2011; ``nricens``'s ``point.method = "km"``). The IDI is always
        IPCW (Uno et al. 2013).
    bounds : {"perturbation", "bootstrap", "none"}, optional
        ``"perturbation"`` (default): Uno et al.'s (2013) perturbation
        resampling, each unit weighted by an independent standard
        exponential draw and every estimate (the censoring weights
        included) recomputed. ``"bootstrap"``: units resampled with
        replacement (as ``nricens``). ``"none"``: estimates only.
    n_resamples : int, optional
        The number of resamples. Default 1000.
    alpha_ci : float, optional
        The bounds are ``1 - alpha_ci`` two-sided percentile bounds.
        Default 0.05.
    random_state : optional
        The seed of the resampling (see :doc:`/Conventions`).

    Returns
    -------
    RiskReclassification
        ``idi``, ``nri`` and ``nri_categorical``, the table of every
        measure with its bounds (``estimates``) and the reclassification
        tables (``tables``).

    Raises
    ------
    ValueError
        If the arrays differ in length, a flag is not 0 or 1, a risk is
        missing or outside [0, 1], ``time`` has no event at or before it
        or no unit after it, or an option is not one of its choices.

    Notes
    -----
    **What the NRI is and is not.** The NRI is not a proper scoring
    rule: a model can improve it by being worse. Adding a marker that
    carries no information, or making a model's risks less well
    calibrated, can make the NRI positive and even "significant" (Hilden
    and Gerds 2014; Pepe, Fan, Feng, Gerds and Hilden 2015). For two
    nested models fitted to the same data, the test of an IDI or an NRI
    of zero asks the same question as the likelihood-ratio test of the
    added covariate (the new marker adds nothing), and is a worse test
    of it. Compare models first by the Brier score and the integrated
    Brier score difference and the AUC difference,
    :func:`~surpyval.metrics.validation.score_predictions`, which are
    proper or well behaved and come with paired standard errors; read the
    IDI and NRI as descriptions of how the predictions moved, part by
    part.

    **The estimators.** With :math:`D_i` = ``risk_new - risk_old``,
    :math:`\hat G` the Kaplan-Meier estimate of the censoring survival
    (an event taken to come before a censoring at its time), the event
    weights :math:`w_i = \delta_i I(x_i \le t) / \hat G(x_i-)` and the
    non-events :math:`I(x_i > t)`:

    .. math::
        \mathrm{IDI} = \frac{\sum_i w_i D_i}{\sum_i w_i}
        - \frac{\sum_i I(x_i > t) D_i}{\sum_i I(x_i > t)},

    and the IPCW NRI is the same with :math:`D_i` replaced by
    :math:`I(D_i > 0) - I(D_i < 0)` (the non-event part with its sign
    reversed). The Kaplan-Meier NRI uses
    :math:`P(\text{up} \mid \text{event}) = (1 - \hat S_{up}(t))
    P(\text{up}) / (1 - \hat S(t))` and
    :math:`P(\text{up} \mid \text{non-event}) = \hat S_{up}(t)
    P(\text{up}) / \hat S(t)`, and the same for down.

    The bounds treat the predicted risks as fixed: right for models
    fitted on other data and scored here, as in a validation set, and as
    ``nricens`` with ``p.std`` and ``p.new``. For models fitted to the
    same data they leave out the variation of the fit, which
    ``survIDINRI::IDI.INF`` adds by perturbing the Cox fits too; there
    they are too narrow.

    The IDI is ``survIDINRI``'s M1 (whose non-event part is the mean
    rise, so its sign is the reverse of ``idi_nonevents`` here); its M2
    is ``p_up_events - p_up_nonevents``, half the continuous NRI when no
    risk is unchanged. ``survIDINRI`` counts an event exactly at the
    horizon as a non-event; here, as in ``nricens``, it is an event.

    References
    ----------
    Pencina, M. J., D'Agostino, R. B. Sr., D'Agostino, R. B. Jr. and
    Vasan, R. S. (2008). Evaluating the added predictive ability of a new
    marker. *Statistics in Medicine* 27, 157-172.

    Pencina, M. J., D'Agostino, R. B. Sr. and Steyerberg, E. W. (2011).
    Extensions of net reclassification improvement calculations to
    measure usefulness of new biomarkers. *Statistics in Medicine* 30,
    11-21.

    Uno, H., Tian, L., Cai, T., Kohane, I. S. and Wei, L. J. (2013). A
    unified inference procedure for a class of measures to assess
    improvement in risk prediction systems with survival data.
    *Statistics in Medicine* 32, 2430-2442.

    Hilden, J. and Gerds, T. A. (2014). A note on the evaluation of novel
    biomarkers: do not rely on integrated discrimination improvement and
    net reclassification index. *Statistics in Medicine* 33, 3405-3414.

    Pepe, M. S., Fan, J., Feng, Z., Gerds, T. and Hilden, J. (2015). The
    net reclassification index (NRI): a misleading measure of prediction
    improvement even with independent test data sets. *Statistics in
    Biosciences* 7, 282-295.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.metrics import idi_nri
    >>> x = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> c = [0, 0, 1, 0, 0, 1, 0, 1]
    >>> old = [0.5, 0.4, 0.3, 0.5, 0.2, 0.3, 0.2, 0.1]
    >>> new = [0.6, 0.5, 0.2, 0.4, 0.2, 0.2, 0.1, 0.1]
    >>> res = idi_nri(x, c, old, new, time=4.5, cutoffs=[0.25, 0.45],
    ...               n_resamples=200, random_state=1)
    >>> res.estimates.set_index("measure").loc[
    ...     ["idi", "nri", "nri_categorical"], "estimate"].round(4)
    measure
    idi                0.0750
    nri                0.7500
    nri_categorical    0.1875
    Name: estimate, dtype: float64
    """
    check_option("censoring", censoring, _CENSORING)
    check_option("bounds", bounds, _BOUNDS)
    if not (
        isinstance(alpha_ci, (int, float, np.integer, np.floating))
        and 0 < alpha_ci < 1
    ):
        raise alpha_ci_error(alpha_ci)
    x_arr = _as_1d(x, "x")
    c_arr = _as_1d(c, "c")
    if c_arr.shape != x_arr.shape:
        raise ValueError("'x' and 'c' must have the same length")
    if not np.isin(c_arr, (0, 1)).all():
        raise ValueError(
            "'c' must be 0 (event) or 1 (right censored); left and "
            "interval censored units have no known status at the horizon. "
            "Drop them, or record each as an event at the end of its "
            "interval if that is how the event was found"
        )
    if not np.isfinite(x_arr).all():
        raise ValueError(
            "'x' has missing or infinite times: drop those units (from "
            "'c' and both risks too)"
        )
    n = x_arr.size
    old = _risk(risk_old, n, "risk_old")
    new = _risk(risk_new, n, "risk_new")
    try:
        t = float(time)
    except (TypeError, ValueError):
        raise ValueError(
            f"'time' must be one horizon, a number; got {time!r}"
        ) from None
    if not np.isfinite(t):
        raise ValueError(f"'time' must be a finite horizon; got {time!r}")
    cuts = _check_cutoffs(cutoffs)
    if bounds != "none":
        if (
            isinstance(n_resamples, bool)
            or not float(n_resamples).is_integer()
            or int(n_resamples) < 2
        ):
            raise ValueError(
                "'n_resamples' must be a whole number of at least 2 (1000 "
                "is usual), or pass bounds='none'; got {!r}".format(
                    n_resamples
                )
            )
    d = _Data(x_arr, c_arr, old, new, t, cuts)
    n_events = int(d.case.sum())
    n_nonevents = int(d.control.sum())
    if n_events == 0 or n_nonevents == 0:
        raise ValueError(
            "At time={:g} there are {} units with the event at or before "
            "it and {} still event-free after it; both are needed. Choose "
            "a horizon between the first event ({:g}) and the last time "
            "({:g})".format(
                t,
                n_events,
                n_nonevents,
                (
                    float(x_arr[c_arr == 0].min())
                    if (c_arr == 0).any()
                    else np.nan
                ),
                float(x_arr.max()),
            )
        )
    point = _statistics(d, np.ones(n), censoring)
    rows = _order(cuts is not None)
    lower = dict.fromkeys(rows, np.nan)
    upper = dict.fromkeys(rows, np.nan)
    p_value = dict.fromkeys(rows, np.nan)
    resamples = 0 if bounds == "none" else int(n_resamples)
    if resamples:
        rng = as_generator(random_state)
        draws = np.empty((resamples, len(rows)))
        for b in range(resamples):
            if bounds == "perturbation":
                v = rng.standard_exponential(n)
            else:
                v = np.bincount(rng.integers(0, n, n), minlength=n).astype(
                    float
                )
            s = _statistics(d, v, censoring)
            draws[b] = [s[r] for r in rows]
        failed = ~np.isfinite(draws).all(axis=1)
        if failed.any():
            warnings.warn(
                "{} of the {} resamples had no event by time={:g} or no "
                "unit after it and were left out of the bounds; with so "
                "few events the bounds are unreliable".format(
                    int(failed.sum()), resamples, t
                ),
                stacklevel=2,
            )
        ok = draws[~failed]
        lo = bootstrap_quantile(ok, alpha_ci / 2)
        hi = bootstrap_quantile(ok, 1 - alpha_ci / 2)
        for k, r in enumerate(rows):
            lower[r] = float(lo[k])
            upper[r] = float(hi[k])
            if r in _TESTED and ok.shape[0]:
                other = (ok[:, k] < 0 if point[r] > 0 else ok[:, k] > 0).mean()
                p_value[r] = float(min(1.0, 2 * other))
    estimates = DataFrame(
        {
            "measure": rows,
            "estimate": [point[r] for r in rows],
            "lower": [lower[r] for r in rows],
            "upper": [upper[r] for r in rows],
            "p_value": [p_value[r] for r in rows],
        }
    )
    tables = {} if cuts is None else _tables(old, new, d, cuts)
    return RiskReclassification(
        estimates,
        t,
        cuts,
        censoring,
        bounds,
        resamples,
        alpha_ci,
        n,
        n_events,
        n_nonevents,
        tables,
    )
