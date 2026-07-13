"""
Model-validation diagnostics for fitted parametric recurrent models.

Everything here rests on the time-rescaling theorem: if events follow an
NHPP with cumulative intensity ``L(t)``, then the transformed event times
``L(t_1) < L(t_2) < ...`` follow a unit-rate homogeneous Poisson process.
Three consequences are used:

- the rescaled interarrival times ``L(t_k) - L(t_{k-1})`` are iid Exp(1)
  (cumulative-hazard residuals), so ``1 - exp(-e)`` are iid U(0, 1)
  (probability-integral-transform residuals);
- the observed count minus the expected count over each item's window is a
  martingale evaluated at the window close (martingale residuals);
- conditional on the number of events an item has in its observation
  window, the normalised transforms ``[L(t) - L(entry)] / [L(close) -
  L(entry)]`` are iid U(0, 1), which a Cramer-von Mises statistic tests
  (the construction behind Crow's power-law goodness-of-fit test,
  generalised to any fitted intensity).

The functions take the fitted model's ``RecurrentEventData`` plus its CIF,
and are exposed as methods on ``ParametricRecurrenceModel``.
"""

import warnings

import numpy as np


def _validate_diagnostic_data(data, what):
    """
    The diagnostics need exact event times: interval-censored (2D ``x`` or
    ``c == 2``) and left-censored (``c == -1``) observations cannot be
    placed in time, so they are rejected with an explanation.
    """
    if data.x.ndim != 1:
        raise ValueError(
            "{} requires exact event times; interval-censored data is not "
            "supported.".format(what)
        )
    if np.any((data.c != 0) & (data.c != 1)):
        raise ValueError(
            "{} requires exact event times; left- or interval-censored "
            "rows (c = -1 or c = 2) are not supported.".format(what)
        )


def _per_item_windows(data):
    """
    Group the data by item and resolve each item's observation window.

    Returns a list with one entry per (sorted-unique) item:
    ``(events, entry, close, explicit_close)`` where ``events`` are the
    item's observed event times (sorted, repeated per their counts ``n``),
    ``entry`` is its observation start (its finite left-truncation bound,
    else 0), ``close`` is the time its window closes (its finite
    right-truncation bound, else its right-censoring row, else its last
    event) and ``explicit_close`` says whether that close was given by the
    data (``tr`` or a ``c = 1`` row) or inferred from the last event
    (failure-truncated).
    """
    items = []
    for item in np.unique(data.i):
        mask = data.i == item
        x, c, n = data.x[mask], data.c[mask], data.n[mask]
        events = np.sort(np.repeat(x[c == 0], n[c == 0].astype(int)))
        entry = float(data.tl[mask][0])
        entry = entry if np.isfinite(entry) else 0.0

        tr = float(data.tr[mask][0])
        if np.isfinite(tr):
            close, explicit_close = tr, True
        elif np.any(c == 1):
            close, explicit_close = float(x[c == 1].max()), True
        elif events.size:
            close, explicit_close = float(events[-1]), False
        else:
            continue
        items.append((events, entry, close, explicit_close))
    return items


def cumulative_hazard_residuals(data, cif):
    """
    Rescaled interarrival times ``cif(t_k) - cif(t_{k-1})`` for every
    observed event (with ``t_0`` each item's entry time), pooled across
    items in sorted-item then time order. Under the fitted model these are
    iid Exp(1).
    """
    _validate_diagnostic_data(data, "Residuals")
    residuals = []
    for events, entry, _, _ in _per_item_windows(data):
        if events.size == 0:
            continue
        transformed = cif(np.concatenate([[entry], events]))
        residuals.append(np.diff(transformed))
    if not residuals:
        raise ValueError("No observed events; no residuals to compute.")
    return np.concatenate(residuals)


def martingale_residuals(data, cif):
    """
    Per-item martingale residuals: the observed event count minus the
    expected count ``cif(close) - cif(entry)`` over the item's observation
    window, one per sorted-unique item. Positive values mean the item had
    more events than the model expects.
    """
    _validate_diagnostic_data(data, "Residuals")
    residuals = []
    for events, entry, close, _ in _per_item_windows(data):
        expected = float(cif(close) - cif(entry))
        residuals.append(events.size - expected)
    return np.array(residuals)


class GoodnessOfFitResult:
    """
    Result of the Cramer-von Mises goodness-of-fit test
    (:meth:`ParametricRecurrenceModel.cramer_von_mises`).

    Attributes
    ----------
    statistic : float
        The Cramer-von Mises statistic ``C^2_M`` of the pooled
        conditionally-uniform transforms of the event times.
    p_value : float
        Parametric-bootstrap p-value: the proportion of statistics from
        data simulated (and refitted) under the fitted model that are at
        least as large as the observed one.
    n_boot : int
        The number of successful bootstrap replicates behind ``p_value``.
    n_events : int
        The number of (transformed) event times in the statistic.
    n_systems : int
        The number of systems (items) in the data.
    """

    def __init__(self, statistic, p_value, n_boot, n_events, n_systems):
        self.statistic = statistic
        self.p_value = p_value
        self.n_boot = n_boot
        self.n_events = n_events
        self.n_systems = n_systems

    def __repr__(self):
        title = "Cramer-von Mises Goodness-of-Fit Test"
        return "\n".join(
            [
                title,
                "=" * len(title),
                "Null hypothesis  : events follow the fitted intensity",
                "Statistic        : {s:.6g}".format(s=self.statistic),
                "p-value          : {p:.6g} ({n} bootstrap samples)".format(
                    p=self.p_value, n=self.n_boot
                ),
            ]
        )


def _conditional_uniforms(data, cif):
    """
    The conditionally-uniform transforms behind the Cramer-von Mises
    statistic, pooled across items. For a time-truncated item every event
    contributes ``[cif(t) - cif(entry)] / [cif(close) - cif(entry)]``; for
    a failure-truncated item the last event is the (random) window close,
    so the remaining events are normalised by the transform at that last
    event and it is itself excluded (Crow's ``M = N - 1`` convention).
    """
    u = []
    n_systems = 0
    for events, entry, close, explicit_close in _per_item_windows(data):
        n_systems += 1
        used = events if explicit_close else events[:-1]
        if used.size == 0:
            continue
        span = float(cif(close) - cif(entry))
        if span <= 0:
            continue
        u.append((cif(used) - cif(entry)) / span)
    if not u:
        raise ValueError(
            "No usable event times for the Cramer-von Mises statistic."
        )
    return np.concatenate(u), n_systems


def cvm_statistic(u):
    """
    The Cramer-von Mises statistic ``1/(12M) + sum_j (u_(j) -
    (2j - 1)/(2M))^2`` of a sample ``u`` against the standard uniform.
    """
    u = np.sort(np.asarray(u, dtype=float))
    m = u.size
    j = np.arange(1, m + 1)
    return float(1.0 / (12.0 * m) + np.sum((u - (2 * j - 1) / (2 * m)) ** 2))


def cramer_von_mises(model, n_boot=200, seed=None):
    """
    Cramer-von Mises goodness-of-fit test of a fitted parametric recurrent
    model; see :meth:`ParametricRecurrenceModel.cramer_von_mises` for the
    user-facing documentation.
    """
    from surpyval.utils.recurrent_event_data import RecurrentEventData

    data = model.data
    _validate_diagnostic_data(data, "The Cramer-von Mises test")
    if not hasattr(model.dist, "inv_cif"):
        raise ValueError(
            "The Cramer-von Mises test requires an invertible CIF; {} does "
            "not define inv_cif.".format(model.dist.name)
        )

    u, n_systems = _conditional_uniforms(data, model.cif)
    observed = cvm_statistic(u)

    # Parametric bootstrap: simulate each item's window from the fitted
    # model (conditional on the fitted intensity the event count in a
    # window is Poisson and the times are iid with density proportional to
    # the intensity), refit, and recompute the statistic -- so the p-value
    # accounts for the parameters having been estimated. Failure-truncated
    # items are approximated by a window fixed at their last observed
    # event.
    rng = np.random.default_rng(seed)
    windows = _per_item_windows(data)
    statistics = []
    failures = 0
    while len(statistics) < n_boot and failures < 2 * n_boot:
        x_b, i_b, c_b, tl_b = [], [], [], []
        for item_id, (_, entry, close, _) in enumerate(windows):
            span = float(model.cif(close) - model.cif(entry))
            count = rng.poisson(span) if span > 0 else 0
            times = np.sort(
                model.inv_cif(
                    model.cif(entry) + span * rng.uniform(size=count)
                )
            )
            x_b.extend([*times, close])
            i_b.extend([item_id] * (count + 1))
            c_b.extend([0] * count + [1])
            tl_b.extend([entry] * (count + 1))
        if (np.asarray(c_b) == 0).sum() < 2:
            failures += 1
            continue
        sim_data = RecurrentEventData(
            np.asarray(x_b, dtype=float),
            np.asarray(i_b),
            np.asarray(c_b),
            np.ones(len(x_b), dtype=int),
            tl=np.asarray(tl_b, dtype=float),
        )
        try:
            # The optimiser explores extreme parameter values on its way to
            # the optimum; the resulting numerical warnings are routine, and
            # over hundreds of refits they would swamp the console.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                refit = model.dist.fit_from_recurrent_data(sim_data)
                u_b, _ = _conditional_uniforms(sim_data, refit.cif)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            failures += 1
            continue
        statistics.append(cvm_statistic(u_b))

    if not statistics:
        raise ValueError(
            "All bootstrap refits failed; cannot compute a p-value."
        )
    if failures:
        warnings.warn(
            "{} bootstrap replicate(s) failed and were resampled or "
            "dropped.".format(failures)
        )

    statistics = np.asarray(statistics)
    p_value = float(
        (1.0 + np.sum(statistics >= observed)) / (statistics.size + 1.0)
    )
    return GoodnessOfFitResult(
        statistic=observed,
        p_value=p_value,
        n_boot=int(statistics.size),
        n_events=int(u.size),
        n_systems=n_systems,
    )
