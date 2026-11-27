"""Bayesian fits of the recurrent-event processes: priors on an
intensity's parameters times the process's own likelihood.

``CrowAMSAA.fit_bayes(x, i, c, priors=...)`` and ``HPP.fit_bayes`` come
here. The posterior is computed by the rule every Bayesian fit shares
(:mod:`surpyval.bayes.result`): in closed form for an HPP with a Gamma
prior on its rate (the gamma-Poisson pair), on the grid for one to three
free parameters (so a Crow-AMSAA posterior, of two, is exact), and by the
checked Laplace approximation beyond. There is no sampler.

In words, for a reliability growth test: the test's failures say how the
failure intensity is changing (``beta`` below 1 is growth); a prior says
what was believed before, about ``beta`` itself, about the MTBF the
system should have reached by the end of the test, or from the previous
test phase; and the posterior combines them. From it come the credible
interval of the MTBF now and at a later time, the predictive
distribution of the number of failures in the next period, and the
probability that a required MTBF is met.

A unit, for WAIC and leave-one-out cross-validation, is one system: its
whole history is one observation of the process.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.special import gammaln

from surpyval.bayes.engines.conjugate import ConjugateResult, prior_family
from surpyval.bayes.result import (
    BayesianResult,
    fit_target,
    interval_levels,
)
from surpyval.bayes.target import (
    Reparameterisation,
    Target,
    register_adapter,
    register_target,
)
from surpyval.serialisation import SerialisableMixin
from surpyval.utils.event_history_data import EventHistoryData
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.warnings import ignoring_warnings

_ON = ("cif", "iif", "mtbf")
#: The largest count a predictive distribution is tabulated to, past its
#: mean (in standard deviations of the widest posterior point).
_COUNT_SPREAD = 12.0


def _natural_neg_ll(fitter: Any, data: EventHistoryData) -> Any:
    """The fitter's negative log-likelihood of ``data`` as a function of
    its parameters on their own scale (an HPP's is written on the log of
    its rate)."""
    f = fitter.create_negll_func(data)
    if type(fitter).__name__ == "HPP":
        return lambda theta: f(np.log(np.asarray(theta, dtype=float)))
    return lambda theta: f(np.asarray(theta, dtype=float))


def _fitter_spec(fitter: Any) -> Any:
    """What names an intensity model in a saved fit: its name, or (a
    model on another clock) its own specification."""
    if hasattr(fitter, "to_spec"):
        return fitter.to_spec()
    return str(fitter.name)


def _fitter_from_spec(spec: Any) -> Any:
    if isinstance(spec, dict):
        from surpyval.recurrent.parametric.clocks import clock_from_spec

        return clock_from_spec(spec)
    from surpyval.recurrent.serialisation import intensity_dist_by_name

    return intensity_dist_by_name(spec)


def _columns(theta: npt.NDArray) -> list:
    return [theta[:, j][:, None] for j in range(theta.shape[1])]


def _pieces_log_lik(
    fitter: Any, s: dict, theta: npt.NDArray, Z: dict | None = None
) -> npt.NDArray:
    """The NHPP log-likelihood of the pieces ``s`` (one item's, or every
    item's; ``EventHistoryData.split_for_nhpp_likelihood``) at rows of
    parameters, ``(N, k) -> (N,)``: the terms of
    ``nhpp_log_likelihood``, with the intensity evaluated at every row
    at once. With ``Z`` (each piece's covariate rows) the parameters
    after the baseline's are the coefficients of a proportional
    intensity, ``exp(Z beta)`` times the baseline's. It is used only
    where it agrees with the fitter's own likelihood (checked at each
    fit)."""
    k_base = len(fitter.parameter_names)
    coef = theta[:, k_base:]
    theta = theta[:, :k_base]
    cols = _columns(theta)
    rows = len(theta)

    def eta(key: str) -> Any:
        if Z is None:
            return 0.0
        return coef @ Z[key].T

    def cif(x: npt.NDArray) -> npt.NDArray:
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return np.zeros((rows, 0))
        return np.asarray(fitter.cif(x[None, :], *cols), dtype=float)

    def counted(n: npt.NDArray, delta: npt.NDArray, key: str) -> npt.NDArray:
        if delta.shape[1] == 0:
            return np.zeros(rows)
        e = eta(key)
        out: npt.NDArray = (
            n * e + n * np.log(delta) - np.exp(e) * delta - gammaln(n + 1)
        ).sum(axis=1)
        return out

    def at_least_one(delta: npt.NDArray, key: str) -> npt.NDArray:
        if delta.shape[1] == 0:
            return np.zeros(rows)
        out: npt.NDArray = np.log(-np.expm1(-np.exp(eta(key)) * delta)).sum(
            axis=1
        )
        return out

    x_o = np.asarray(s["x_o"], dtype=float)
    ll = np.zeros(rows)
    if x_o.size:
        log_iif = np.asarray(fitter.log_iif(x_o[None, :], *cols), dtype=float)
        e = eta("o")
        ll = ll + (
            log_iif + e + np.exp(e) * (cif(s["x_o_prev"]) - cif(x_o))
        ).sum(axis=1)
    if np.size(s["x_right"]):
        ll = ll + (
            np.exp(eta("right")) * (cif(s["x_right_prev"]) - cif(s["x_right"]))
        ).sum(axis=1)
    ll = ll + counted(
        s["n_left"], cif(s["x_left"]) - cif(s["x_left_prev"]), "left"
    )
    ll = ll + counted(s["n_i"], cif(s["x_i_r"]) - cif(s["x_i_l"]), "i")
    ll = ll - at_least_one(cif(s["sel_to"]) - cif(s["sel_from"]), "sel")
    ll = ll + at_least_one(cif(s["gap_to"]) - cif(s["gap_from"]), "gap")
    return ll


def _covariate_pieces(s: dict, Z: npt.NDArray) -> dict:
    """Each likelihood piece's covariate rows, gathered with the split's
    own masks (as the proportional intensity fitters do)."""
    Z = np.asarray(Z, dtype=float)
    out = {
        key: Z[s[mask]]
        for key, mask in (
            ("o", "mask_o"),
            ("right", "mask_right"),
            ("left", "mask_left"),
            ("i", "mask_i"),
        )
    }
    out["sel"] = Z[s["sel_idx"]]
    out["gap"] = Z[s["gap_idx"]]
    return out


@register_target
class RecurrenceTarget(Target):
    """A recurrent-event process's side of a Bayesian fit
    (:mod:`surpyval.bayes.target`): the fitter's own likelihood, one
    system as the unit, and replicates simulated over each system's own
    observation window."""

    tag = "recurrence"
    unit_name = "system"

    def __init__(
        self, fitter: Any, data: EventHistoryData, mle: Any = None
    ) -> None:
        self.fitter = fitter
        #: The intensity whose functions the results evaluate (the
        #: baseline, for a proportional intensity).
        self.baseline = fitter
        self.k_base = len(fitter.parameter_names)
        self.dist = fitter
        self.data = data
        self.mle = mle
        self.names = list(fitter.parameter_names)
        self.bounds = list(fitter.bounds)
        self.fixed = {}
        name = str(fitter.name)
        self.family = name if name.endswith("Process") else f"{name} process"
        self._finish()

    def _make_neg_ll(self, data: EventHistoryData) -> Any:
        """The fitter's own negative log-likelihood of ``data``, in the
        parameters' own scales."""
        return _natural_neg_ll(self.fitter, data)

    def _split(self, data: EventHistoryData) -> tuple:
        """What the all-rows likelihood needs of ``data``: its pieces
        and (a proportional intensity) their covariate rows."""
        return data.split_for_nhpp_likelihood(), None

    def _finish(self) -> None:
        data = self.data
        self._neg_ll = self._make_neg_ll(data)
        self._pieces, self._Z = self._split(data)
        entry, exit_ = data.item_observation_windows()
        lower = float(getattr(self.baseline, "support", (0.0, np.inf))[0])
        self.entry = np.where(
            np.isfinite(entry), entry, max(lower, 0.0) if lower > -1 else 0.0
        )
        self.exit = np.asarray(exit_, dtype=float)
        #: The latest time any system was observed to: "now".
        self.end = float(np.max(self.exit))
        self._unit_pieces: list | None = None
        self._unit_neg_ll: list | None = None
        self._vector = self._vector_agrees()

    def __reduce__(self) -> tuple:
        # (the likelihood's closures are rebuilt, not pickled)
        return (type(self), (self.fitter, self.data, self.mle))

    # -- the likelihood --------------------
    def neg_ll(self, theta: npt.NDArray) -> float:
        return float(self._neg_ll(np.asarray(theta, dtype=float)))

    def _trial_points(self) -> npt.NDArray:
        start = self.start()
        if start is None:
            # no fit to start from: a rate, or a scale, in the data's
            # unit of time, and every other parameter at 1
            start = np.ones(len(self.bounds))
            span = max(abs(self.end), 1e-300)
            start[0] = 1.0 / span if len(start) == 1 else span
        return np.stack([start, start * 1.07, start * 0.9])

    def _vector_agrees(self) -> bool:
        """Whether the all-rows-at-once likelihood matches the fitter's
        own at trial points; where it does not (an intensity whose
        functions do not take arrays of parameters), the fitter's own is
        called a row at a time."""
        try:
            pts = self._trial_points()
            with ignoring_warnings(), np.errstate(all="ignore"):
                fast = _pieces_log_lik(
                    self.baseline, self._pieces, pts, self._Z
                )
                slow = np.array([-self.neg_ll(p) for p in pts])
        except Exception:  # noqa: BLE001 - any failure means "do not use"
            return False
        ok = np.isfinite(slow)
        return bool(
            ok.any()
            and np.all(np.isfinite(fast[ok]))
            and np.allclose(fast[ok], slow[ok], rtol=1e-9, atol=1e-9)
        )

    def log_lik(self, theta: npt.NDArray) -> npt.NDArray:
        theta = np.atleast_2d(theta)
        if not self._vector:
            return super().log_lik(theta)
        with ignoring_warnings(), np.errstate(all="ignore"):
            out = _pieces_log_lik(self.baseline, self._pieces, theta, self._Z)
        return np.where(np.isnan(out), -np.inf, out)

    def estimates(self) -> npt.NDArray:
        return np.asarray(self.mle.params, dtype=float)

    # -- units: one system --------------------
    def _units(self) -> tuple[list, list]:
        if self._unit_pieces is None:
            first, inverse = self.data.item_rows()
            pieces, negs = [], []
            for k in range(len(first)):
                sub = self.data[np.flatnonzero(inverse == k)]
                pieces.append(self._split(sub))
                negs.append(self._make_neg_ll(sub))
            self._unit_pieces, self._unit_neg_ll = pieces, negs
        assert self._unit_neg_ll is not None
        return self._unit_pieces, self._unit_neg_ll

    def unit_log_lik(self, theta: npt.NDArray) -> npt.NDArray:
        theta = np.atleast_2d(theta)
        pieces, negs = self._units()
        out = np.empty((len(theta), len(pieces)))
        with ignoring_warnings(), np.errstate(all="ignore"):
            for u, (s, f) in enumerate(zip(pieces, negs)):
                if self._vector:
                    out[:, u] = _pieces_log_lik(
                        self.baseline, s[0], theta, s[1]
                    )
                else:
                    out[:, u] = [-float(f(row)) for row in theta]
        return out

    def unit_counts(self) -> npt.NDArray:
        return np.ones(len(self.entry))

    # -- models and replicates --------------------
    def with_params(self, theta: npt.NDArray) -> Any:
        model = self.fitter.from_params(np.asarray(theta, dtype=float))
        unit = getattr(self.mle, "unit", None)
        if unit is not None:
            model.unit = unit
        return model

    def _has_counts(self) -> bool:
        return bool(np.any(np.isin(np.asarray(self.data.c), (-1, 2))))

    def _refuse_truncated(self) -> None:
        if np.isfinite(np.asarray(self.data.tr, dtype=float)).any():
            raise ValueError(
                "A posterior predictive check simulates each system over "
                "its observation window, which right-truncated data (tr) "
                "do not have: a system is in them because it failed. "
                "Check the model with the maximum-likelihood fit's "
                "diagnostics instead."
            )

    def simulate(self, theta: npt.NDArray, rng: Any) -> list:
        """Each system's event times over its own observation window,
        from the process at ``theta``."""
        self._refuse_truncated()
        theta = np.asarray(theta, dtype=float)
        lo = np.asarray(self.fitter.cif(self.entry, *theta), dtype=float)
        hi = np.asarray(self.fitter.cif(self.exit, *theta), dtype=float)
        out = []
        for a, b in zip(np.atleast_1d(lo), np.atleast_1d(hi)):
            m = int(rng.poisson(max(b - a, 0.0)))
            u = np.sort(a + (b - a) * rng.uniform(size=m))
            out.append(np.asarray(self.fitter.inv_cif(u, *theta), dtype=float))
        return out

    def observed(self) -> list:
        self._refuse_truncated()
        _, inverse = self.data.item_rows()
        x = np.asarray(self.data.x, dtype=float)
        x = x if x.ndim == 1 else x[:, -1]
        c = np.asarray(self.data.c)
        n = np.asarray(self.data.n, dtype=float)
        out = []
        for k in range(len(self.entry)):
            rows = inverse == k
            times = x[rows & (c == 0)]
            counted = int(n[rows & np.isin(c, (-1, 2))].sum())
            # a counted row's events have no times: they are held as
            # that many missing times, so that the count is right
            out.append(np.r_[np.sort(times), np.full(counted, np.nan)])
        return out

    def statistics(self) -> dict:
        mid = 0.5 * (self.entry + self.exit)

        def events(sets: list) -> float:
            return float(sum(len(s) for s in sets))

        def later_half(sets: list) -> float:
            total = sum(len(s) for s in sets)
            late = sum(int(np.sum(s > m)) for s, m in zip(sets, mid))
            return late / total if total else float("nan")

        def most(sets: list) -> float:
            return float(max(len(s) for s in sets))

        out = {"number of events": events}
        if not self._has_counts():
            out["share of events in the later half of observation"] = (
                later_half
            )
        if len(self.entry) > 1:
            out["most events on one system"] = most
        return out

    # -- priors --------------------
    def check(self, bound: Any) -> None:
        from surpyval.bayes.prior import PowerPrior

        for j in bound.spec.joint:
            if isinstance(j, PowerPrior):
                raise ValueError(
                    "prior.from_data holds the lives of a life "
                    "distribution. For an earlier test phase use "
                    "prior.from_fit(earlier_model, weight=...), the earlier "
                    "phase's data as a power prior, or "
                    "prior.from_model(earlier_model)."
                )

    def n_events(self) -> float:
        c = np.asarray(self.data.c)
        n = np.asarray(self.data.n, dtype=float)
        return float(np.sum(c == 0) + n[np.isin(c, (-1, 2))].sum())

    def check_flat(self, bound: Any) -> None:
        improper = bound.improper_names
        if not improper:
            return
        if any(j.is_likelihood for j in bound.spec.joint):
            # an earlier phase's data may carry them
            return
        events = self.n_events()
        if events < len(improper) or self.mle is None:
            raise ValueError(
                f"{', '.join(improper)} "
                f"{'has' if len(improper) == 1 else 'have'} no prior "
                f"(flat), and {events:g} "
                f"event{'' if events == 1 else 's'} do not determine "
                f"{'it' if len(improper) == 1 else 'them'}. Give "
                f"{'it a prior' if len(improper) == 1 else 'each a prior'} "
                "(for example prior.around(value, between=(low, high)))."
            )

    def conjugate(self, bound: Any) -> Any | None:
        """An HPP with a Gamma prior on its rate: the posterior is
        ``Gamma(a + events, b + time watched)``."""
        if type(self.fitter).__name__ != "HPP" or bound.spec.joint:
            return None
        if bound.free_names != ["lambda"]:
            return None
        prior = bound.spec.single.get("lambda")
        fam = None if prior is None else prior_family(prior)
        if fam is None or fam[0] != "gamma":
            return None
        s = self._pieces
        if len(s["sel_to"]) or len(s["gap_to"]):
            # right truncation: its terms are not a Poisson count's
            return None
        a, b = fam[1]
        d_left = s["x_left"] - s["x_left_prev"]
        d_int = s["x_i_r"] - s["x_i_l"]
        events = float(len(s["x_o"]) + np.sum(s["n_left"]) + np.sum(s["n_i"]))
        watched = float(
            np.sum(s["x_o"] - s["x_o_prev"])
            + np.sum(s["x_right"] - s["x_right_prev"])
            + np.sum(d_left)
            + np.sum(d_int)
        )
        constant = float(
            np.sum(s["n_left"] * np.log(d_left))
            - np.sum(gammaln(s["n_left"] + 1))
            + np.sum(s["n_i"] * np.log(d_int))
            - np.sum(gammaln(s["n_i"] + 1))
        )
        log_z = (
            a * np.log(b)
            - gammaln(a)
            + gammaln(a + events)
            - (a + events) * np.log(b + watched)
            + constant
        )
        return ConjugateResult(
            "lambda",
            "gamma",
            (a + events, b + watched),
            float(log_z),
            f"Gamma prior on lambda: {events:g} events in {watched:g} of "
            "time watched",
        )

    def reparam(self, bound: Any) -> Any | None:
        """A power-law process's posterior in ``(log alpha, log beta)``
        is a curved ridge (``alpha`` is ``T`` over the expected events
        to the power ``1 / beta``, so with few events it ranges over
        many orders of magnitude, tied to ``beta``). The engines work
        with the log of the expected number of events by the end of
        observation, ``beta (log T - log alpha)``, in its place: the
        count the data say most about, nearly independent of the
        shape."""
        if getattr(self.baseline, "name", "") != "Crow-AMSAA":
            return None
        free = list(bound.free_names)
        if free[:2] != ["alpha", "beta"]:
            return None
        log_end = float(np.log(self.end))

        # (any further free parameters, a proportional intensity's
        # coefficients, are left as they are)
        def forward(phi: npt.NDArray) -> npt.NDArray:
            return np.column_stack(
                [
                    np.exp(phi[:, 1]) * (log_end - phi[:, 0]),
                    phi[:, 1],
                    phi[:, 2:],
                ]
            )

        def inverse(u: npt.NDArray) -> npt.NDArray:
            return np.column_stack(
                [log_end - u[:, 0] * np.exp(-u[:, 1]), u[:, 1], u[:, 2:]]
            )

        def log_det(u: npt.NDArray) -> npt.NDArray:
            out: npt.NDArray = -u[:, 1]
            return out

        return Reparameterisation(
            ["the expected events by the end of observation", *free[1:]],
            forward,
            inverse,
            log_det,
        )

    def _at(self, fname: str, t: float) -> Any:
        fitter, names = self.baseline, self.names[: self.k_base]

        def value(theta: dict) -> npt.NDArray:
            p = [np.asarray(theta[n], dtype=float) for n in names]
            with np.errstate(all="ignore"):
                if fname == "mtbf":
                    return 1.0 / np.asarray(fitter.iif(t, *p), dtype=float)
                return np.asarray(getattr(fitter, fname)(t, *p), dtype=float)

        return value

    def quantities(self) -> dict:
        """``mtbf_at``, ``intensity_at`` and ``cif_at``: the
        instantaneous MTBF, the intensity and the expected number of
        events by a time."""
        from surpyval.bayes.prior import format_number

        return {
            "mtbf_at": lambda t: (
                f"MTBF at {format_number(t)}",
                self._at("mtbf", t),
                "log",
            ),
            "intensity_at": lambda t: (
                f"intensity at {format_number(t)}",
                self._at("iif", t),
                "log",
            ),
            "cif_at": lambda t: (
                f"expected events by {format_number(t)}",
                self._at("cif", t),
                "log",
            ),
        }

    def grid_checks(self, theta: dict) -> list:
        return [self._at("iif", self.end)(theta)]

    # -- words and saving --------------------
    def describe_data(self) -> str:
        k = len(self.entry)
        events = self.n_events()
        return (
            f"{k} system{'s' if k != 1 else ''}, {events:g} "
            f"event{'s' if events != 1 else ''}, observed to {self.end:g}"
        )

    def same_data(self, other: Any) -> bool:
        return bool(self.data.equals(other.data))

    def to_dict(self) -> dict:
        return {
            "target": self.tag,
            "fitter": _fitter_spec(self.fitter),
            "mle": None if self.mle is None else self.mle.to_dict(),
            "data": self.data.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> RecurrenceTarget:
        data = EventHistoryData.from_dict(d["data"])
        mle = None
        if d.get("mle") is not None:
            import surpyval

            with ignoring_warnings():
                mle = surpyval.from_dict(d["mle"])
        return cls(_fitter_from_spec(d["fitter"]), data, mle)


def _missing_query(x: npt.NDArray, values: Any) -> npt.NDArray:
    """``values`` with NaN where the query time ``x`` is missing (a
    quantile or a constant of the posterior would otherwise give a
    number there)."""
    out = np.array(values, dtype=float)
    out[np.isnan(np.broadcast_to(x, out.shape))] = np.nan
    return out


def _is_recurrence_fit(model: Any) -> bool:
    from surpyval.recurrent.parametric.parametric_recurrence import (
        ParametricRecurrenceModel,
    )

    return (
        type(model) is ParametricRecurrenceModel
        and getattr(model, "data", None) is not None
        and getattr(model, "how", None) == "MLE"
        and hasattr(model.dist, "create_negll_func")
    )


register_adapter(
    _is_recurrence_fit,
    lambda model: RecurrenceTarget(model.dist, model.data, model),
)


class CountPrediction:
    """
    The posterior predictive distribution of the number of events in a
    period: a Poisson count whose mean is uncertain, averaged over the
    posterior of the process's parameters.

    Attributes
    ----------
    start, end : float
        The period, ``(start, end]``.
    systems : float
        How many systems the count is over.
    mean : float
        The expected number of events.
    """

    def __init__(
        self,
        means: npt.NDArray,
        weights: npt.NDArray,
        start: float,
        end: float,
        systems: float,
        gamma: tuple[float, float] | None = None,
    ) -> None:
        from scipy import stats

        self.start, self.end, self.systems = start, end, systems
        keep = weights > 0
        self._means = np.asarray(means, dtype=float)[keep]
        self._w = weights[keep] / weights[keep].sum()
        self.mean = float(self._w @ self._means)
        top = float(np.max(self._means))
        upper = int(np.ceil(top + _COUNT_SPREAD * np.sqrt(top + 1.0) + 10))
        k = np.arange(upper + 1)
        if gamma is not None:
            # the mean is Gamma(shape, rate): the count is negative
            # binomial, exactly
            shape, rate = gamma
            self.mean = shape / rate
            pmf = stats.nbinom.pmf(k, shape, rate / (rate + 1.0))
        else:
            pmf = np.zeros(len(k))
            step = max(1, 2_000_000 // len(k))
            for a in range(0, len(self._means), step):
                m = self._means[a : a + step, None]
                pmf += self._w[a : a + step] @ stats.poisson.pmf(k[None, :], m)
        self._pmf = pmf
        self._cdf = np.minimum(np.cumsum(pmf), 1.0)

    def pmf(self, k: npt.ArrayLike) -> npt.NDArray:
        """The probability of exactly ``k`` events."""
        k = np.asarray(k)
        inside = (k >= 0) & (k < len(self._pmf)) & (k == np.floor(k))
        out = np.where(
            inside, self._pmf[np.clip(k, 0, len(self._pmf) - 1).astype(int)], 0
        )
        return np.asarray(out, dtype=float)

    def cdf(self, k: npt.ArrayLike) -> npt.NDArray:
        """The probability of ``k`` events or fewer."""
        k = np.floor(np.asarray(k, dtype=float))
        idx = np.clip(k, 0, len(self._cdf) - 1).astype(int)
        return np.where(
            k < 0, 0.0, np.where(k >= len(self._cdf), 1.0, self._cdf[idx])
        )

    def quantile(self, p: npt.ArrayLike) -> npt.NDArray:
        """The smallest count with at least probability ``p`` at or
        below it."""
        p = np.asarray(p, dtype=float)
        return np.searchsorted(self._cdf, p - 1e-12, side="left").astype(float)

    def interval(self, alpha_ci: float = 0.05) -> tuple[float, float]:
        """A predictive interval for the count: its ``alpha_ci / 2`` and
        ``1 - alpha_ci / 2`` points (it holds the count with at least
        ``1 - alpha_ci`` probability)."""
        lo, hi = self.quantile(interval_levels(alpha_ci, "two-sided"))
        return float(lo), float(hi)

    def __repr__(self) -> str:
        lo, hi = self.interval(0.1)
        over = f" over {self.systems:g} systems" if self.systems != 1 else ""
        return (
            f"Predicted events in ({self.start:g}, {self.end:g}]{over}: "
            f"mean {self.mean:.4g}, 90% predictive interval "
            f"{lo:g} to {hi:g}, P(none) {float(self.pmf(0)):.3g}"
        )


class BayesianRecurrence(BayesianResult):
    """
    A Bayesian fit of a recurrent-event process (``CrowAMSAA.fit_bayes``,
    ``HPP.fit_bayes``): the posterior of the intensity's parameters, and
    what follows from it for a repairable system or a reliability growth
    test.

    ``cif`` and ``iif`` are posterior means (the expected number of
    events by a time and the expected intensity, the uncertainty in the
    parameters averaged in); ``mtbf`` is the posterior median of the
    instantaneous MTBF, ``1 / iif``. ``cri`` gives the credible interval
    of each, ``count`` the predictive distribution of the number of
    events in a period, and ``mtbf_probability`` the probability that a
    required MTBF is met.

    Attributes
    ----------
    params : numpy.ndarray
        The posterior medians of the parameters.
    posterior : Posterior
        The posterior's points and weights.
    end : float
        The latest time any system was observed to: where "the current
        MTBF" is taken.
    method : str
        ``"conjugate"`` (closed form), ``"grid"`` (exact) or
        ``"laplace"`` (approximate, with its checks in ``diagnostics``).
    mle : ParametricRecurrenceModel or None
        The maximum-likelihood fit to the same data (``None`` where the
        data have no events, and the answer rests on the prior).

    Examples
    --------
    A growth test of one prototype to 400 hours, with a belief that the
    programme is achieving growth (``beta`` about 0.7):

    >>> from surpyval.recurrent import CrowAMSAA
    >>> import surpyval as sp
    >>> x = [15, 42, 60, 98, 130, 171, 205, 260, 310, 345, 390, 400]
    >>> c = [0] * 11 + [1]
    >>> model = CrowAMSAA.fit_bayes(
    ...     x, c=c, priors={"beta": sp.prior.around(0.7, between=(0.4, 1.2))})
    >>> model.method
    'grid'
    >>> lo, hi = model.cri(400.0, on="mtbf")
    >>> bool(lo < model.mtbf(400.0) < hi)
    True
    >>> bool(0 < model.mtbf_probability(30.0) < 1)
    True
    """

    model_tag = "BayesianRecurrence"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.end = self.target.end
        self.unit = getattr(self.mle, "unit", None)

    # -- the process at every posterior point --------------------
    def _values(
        self, on: str, x: npt.NDArray, scale: npt.NDArray | None = None
    ) -> npt.NDArray:
        """``on`` at times ``x`` for every posterior point, ``(P, m)``;
        ``scale`` multiplies each point's intensity (a proportional
        intensity's ``exp(Z beta)``)."""
        fitter = self.target.baseline
        cols = _columns(self.posterior.points[:, : self.target.k_base])
        x = np.asarray(x, dtype=float).reshape(-1)
        with ignoring_warnings(), np.errstate(all="ignore"):
            if on == "cif":
                out = fitter.cif(x[None, :], *cols)
            else:
                out = fitter.iif(x[None, :], *cols)
            out = np.broadcast_to(
                np.asarray(out, dtype=float),
                (len(self.posterior.points), x.size),
            )
            if scale is not None:
                out = out * scale[:, None]
            if on == "mtbf":
                out = 1.0 / out
        return np.asarray(out, dtype=float)

    @keeps_query_shape
    def cif(self, x: npt.ArrayLike) -> npt.NDArray:
        """
        The expected number of events by ``x`` for one system: the
        posterior mean of the cumulative intensity.

        Examples
        --------
        >>> from surpyval.recurrent import HPP
        >>> import surpyval as sp
        >>> model = HPP.fit_bayes(
        ...     [100.0], c=[1],
        ...     priors={"lambda": sp.Gamma.from_params([2.0, 100.0])})
        >>> float(model.cif(50.0))  # no failures in 100: 2 / 200 a unit time
        0.5
        """
        x = np.asarray(x, dtype=float)
        exact = self._exact_rate()
        if exact is not None:
            return np.asarray(exact.mean() * x)
        return _missing_query(
            x, self.posterior.expectation(self._values("cif", x))
        )

    mcf = cif

    @keeps_query_shape
    def iif(self, x: npt.ArrayLike) -> npt.NDArray:
        """The expected intensity (events a unit time) at ``x``: the
        posterior mean."""
        x = np.asarray(x, dtype=float)
        exact = self._exact_rate()
        if exact is not None:
            return _missing_query(x, exact.mean() * np.ones_like(x))
        return _missing_query(
            x, self.posterior.expectation(self._values("iif", x))
        )

    @keeps_query_shape
    def mtbf(self, x: npt.ArrayLike) -> npt.NDArray:
        """The instantaneous MTBF at ``x``, ``1 / iif``: its posterior
        median (its posterior mean can be very large, or infinite, where
        low intensities are still credible)."""
        x = np.asarray(x, dtype=float)
        return self._quantile("mtbf", x, 0.5)

    def _exact_rate(self) -> Any | None:
        """A closed-form posterior of an HPP's rate, where there is
        one."""
        if self.posterior.exact is None:
            return None
        return self.posterior._exact(self.posterior.exact["name"])

    def _quantile(
        self,
        on: str,
        x: npt.NDArray,
        level: float,
        scale: npt.NDArray | None = None,
    ) -> npt.NDArray:
        x = np.asarray(x, dtype=float).reshape(-1)
        exact = self._exact_rate()
        if exact is not None and scale is None:
            # every function is monotone in the rate
            if on == "mtbf":
                value = np.full(x.shape, 1.0 / float(exact.ppf(1.0 - level)))
            else:
                rate = float(exact.ppf(level))
                value = rate * x if on == "cif" else np.full(x.shape, rate)
            return _missing_query(x, value)
        out = np.asarray(
            self.posterior.wquantile(self._values(on, x, scale), level),
            dtype=float,
        ).reshape(x.shape)
        if scale is not None and np.isnan(scale).any():
            # a missing covariate: no answer at any time
            return np.full(x.shape, np.nan)
        return _missing_query(x, out)

    def cri(
        self,
        x: npt.ArrayLike,
        on: str = "cif",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """
        A credible interval for the process at times ``x``.

        Parameters
        ----------
        x : array like or scalar
            The times: at or before :attr:`end` for what the data show,
            after it for a projection (the process carried on with the
            same parameters).
        on : {'cif', 'iif', 'mtbf'}, optional
            The cumulative intensity (expected events by ``x``), the
            intensity, or the instantaneous MTBF. Default ``'cif'``.
        alpha_ci : float, optional
            1 minus the credible level. Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided adds a last ``[lower, upper]`` axis.

        Examples
        --------
        >>> from surpyval.recurrent import CrowAMSAA
        >>> x = [15, 42, 60, 98, 130, 171, 205, 260, 310, 345, 390, 400]
        >>> model = CrowAMSAA.fit_bayes(x, c=[0] * 11 + [1])
        >>> model.cri([400.0, 800.0], on="mtbf").shape
        (2, 2)
        """
        on = {"mcf": "cif", "intensity": "iif"}.get(on, on)
        if on not in _ON:
            raise ValueError(
                f"on must be one of {', '.join(_ON)}; got {on!r}. "
                "(param_cri gives a parameter's interval, function_cri "
                "any function's.)"
            )
        levels = interval_levels(alpha_ci, bound)
        shape = np.shape(x)
        out = [
            np.reshape(self._quantile(on, np.asarray(x, float), level), shape)
            for level in levels
        ]
        return np.stack(out, axis=-1) if len(out) == 2 else out[0]

    def mtbf_cri(
        self,
        x: npt.ArrayLike | None = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """
        A credible interval for the instantaneous MTBF: the current one
        (``x`` left out: at :attr:`end`, the latest time observed) or
        the projected one at a later ``x``.

        Examples
        --------
        >>> from surpyval.recurrent import CrowAMSAA
        >>> x = [15, 42, 60, 98, 130, 171, 205, 260, 310, 345, 390, 400]
        >>> model = CrowAMSAA.fit_bayes(x, c=[0] * 11 + [1])
        >>> now = model.mtbf_cri()
        >>> later = model.mtbf_cri(800.0)
        >>> now.shape, later.shape
        ((2,), (2,))
        """
        at = self.end if x is None else x
        return self.cri(at, on="mtbf", alpha_ci=alpha_ci, bound=bound)

    def mtbf_probability(
        self, requirement: float, at: float | None = None
    ) -> float:
        """
        The posterior probability that the instantaneous MTBF at time
        ``at`` (by default :attr:`end`, now) is at least
        ``requirement``: the chance, given the data and the priors, that
        the requirement is met.

        Examples
        --------
        >>> from surpyval.recurrent import HPP
        >>> import surpyval as sp
        >>> model = HPP.fit_bayes(
        ...     [100.0], c=[1],
        ...     priors={"lambda": sp.Gamma.from_params([2.0, 100.0])})
        >>> round(model.mtbf_probability(50.0), 4)  # P(rate <= 1 / 50)
        0.9084
        """
        requirement = float(requirement)
        if not requirement > 0:
            raise ValueError(
                f"requirement must be a positive MTBF; got {requirement!r}."
            )
        at = self.end if at is None else float(at)
        exact = self._exact_rate()
        if exact is not None:
            return float(exact.cdf(1.0 / requirement))
        values = self._values("mtbf", np.array([at]))[:, 0]
        return float(self.posterior.weights @ (values >= requirement))

    def count(
        self,
        start: float | None = None,
        end: float | None = None,
        *,
        duration: float | None = None,
        systems: float = 1,
    ) -> CountPrediction:
        """
        The predictive distribution of the number of events in a period:
        how many failures to expect next, with the uncertainty in the
        parameters included (so it is wider than a Poisson count at the
        fitted values).

        Parameters
        ----------
        start : float, optional
            The start of the period. By default :attr:`end`, now.
        end : float, optional
            Its end; or give ``duration``.
        duration : float, optional
            The period's length from ``start``.
        systems : float, optional
            The number of systems the count is over. Default 1.

        Returns
        -------
        CountPrediction
            ``mean``, ``pmf(k)``, ``cdf(k)``, ``quantile(p)`` and
            ``interval(alpha_ci)``.

        Examples
        --------
        >>> from surpyval.recurrent import CrowAMSAA
        >>> x = [15, 42, 60, 98, 130, 171, 205, 260, 310, 345, 390, 400]
        >>> model = CrowAMSAA.fit_bayes(x, c=[0] * 11 + [1])
        >>> nxt = model.count(duration=100.0)
        >>> bool(0 < nxt.mean < 10), nxt.interval(0.1)[0] >= 0
        (True, True)
        """
        start = self.end if start is None else float(start)
        if (end is None) == (duration is None):
            raise ValueError(
                "Give the period's end or its duration (one of them)."
            )
        if end is None:
            assert duration is not None
            end = start + float(duration)
        end = float(end)
        if not end > start:
            raise ValueError(
                f"The period must end after it starts; got ({start:g}, "
                f"{end:g}]."
            )
        if not float(systems) > 0:
            raise ValueError(f"systems must be positive; got {systems!r}.")
        values = self._values("cif", np.array([start, end]))
        # (inf - inf at a grid's edge, a point of no weight, which
        # CountPrediction leaves out)
        with np.errstate(invalid="ignore"):
            means = float(systems) * (values[:, 1] - values[:, 0])
        gamma = None
        if self.posterior.exact is not None:
            # an HPP's rate is Gamma(a, b): the period's mean is
            # Gamma(a, b / exposure)
            a, b = self.posterior.exact["params"]
            gamma = (float(a), float(b) / (float(systems) * (end - start)))
        return CountPrediction(
            means, self.posterior.weights, start, end, float(systems), gamma
        )

    def _refuse_cb(self, what: str, instead: str) -> Any:
        raise ValueError(
            f"A Bayesian fit has credible intervals, not confidence "
            f"bounds: use {instead} in place of {what}."
        )

    def cif_cb(self, *args: Any, **kwargs: Any) -> Any:
        """Refuses: a Bayesian fit's interval is :meth:`cri`."""
        return self._refuse_cb("cif_cb", "cri(x, on='cif')")

    def iif_cb(self, *args: Any, **kwargs: Any) -> Any:
        """Refuses: use :meth:`cri`."""
        return self._refuse_cb("iif_cb", "cri(x, on='iif')")

    def mtbf_cb(self, *args: Any, **kwargs: Any) -> Any:
        """Refuses: use :meth:`mtbf_cri`."""
        return self._refuse_cb("mtbf_cb", "mtbf_cri(x)")

    def param_cb(self, *args: Any, **kwargs: Any) -> Any:
        """Refuses: use :meth:`param_cri`."""
        return self._refuse_cb("param_cb", "param_cri(name)")


def zero_event_data(
    x: Any, i: Any, c: Any, n: Any, t: Any, tl: Any, tr: Any, windows: Any
) -> EventHistoryData | None:
    """The data as the fit reads them, where they have no events (the
    maximum-likelihood fit refuses those, and a Bayesian fit with proper
    priors does not need it); ``None`` for anything else."""
    from surpyval.utils.recurrent_utils import handle_xicn

    if isinstance(x, EventHistoryData):
        data = x
    else:
        try:
            data = handle_xicn(
                x,
                i,
                c,
                n,
                t=t,
                tl=tl,
                tr=tr,
                as_recurrent_data=True,
                windows=windows,
            )
        except (ValueError, TypeError):
            return None
    cc = np.asarray(data.c)
    if np.any((cc != 1) & (np.asarray(data.n) > 0)):
        return None
    return data


def fit(
    fitter: Any,
    x: Any,
    i: Any = None,
    c: Any = None,
    n: Any = None,
    t: Any = None,
    tl: Any = None,
    tr: Any = None,
    *,
    priors: Any = None,
    method: str = "auto",
    importance: bool = False,
    points: int | None = None,
    windows: Any = None,
) -> BayesianRecurrence:
    """The Bayesian fit of a recurrent-event process; see the fitters'
    ``fit_bayes``."""
    from surpyval.utils.no_maximum import quiet_maximum_warnings
    from surpyval.utils.warnings import caller_stacklevel, collecting_warnings

    mle = None
    try:
        with collecting_warnings() as caught:
            with quiet_maximum_warnings(), np.errstate(all="ignore"):
                mle = fitter.fit(
                    x, i=i, c=c, n=n, t=t, tl=tl, tr=tr, windows=windows
                )
        data = mle.data
    except ValueError as e:
        data = zero_event_data(x, i, c, n, t, tl, tr, windows)
        if data is None:
            raise ValueError(
                "fit_bayes starts from the maximum-likelihood fit, which "
                f"refuses these data: {e}"
            ) from None
        caught = []
    for w in caught:
        warnings.warn(
            str(w.message), w.category, stacklevel=caller_stacklevel()
        )
    model: BayesianRecurrence = fit_target(
        RecurrenceTarget(fitter, data, mle),
        priors,
        method=method,
        importance=importance,
        points=points,
        result=BayesianRecurrence,
    )
    return model


# ---------------------------------------------------------------------------
# Proportional intensity: covariates on a process
# ---------------------------------------------------------------------------
#: The default prior on a proportional intensity's coefficient is
#: ``Normal(0, COEFFICIENT_SCALE / sd)``, ``sd`` its covariate's standard
#: deviation over the systems: the regressions' default.
COEFFICIENT_SCALE = 2.5


def _pi_fitters(mle: Any) -> tuple[Any, Any]:
    """The proportional intensity fitter of a fitted model, and the
    intensity its baseline's functions come from."""
    import surpyval.recurrent as recurrent

    if getattr(mle, "kind", "") == "HPP":
        return recurrent.ProportionalIntensityHPP, recurrent.HPP
    return recurrent.ProportionalIntensityNHPP, mle.dist


@register_target
class ProportionalIntensityTarget(RecurrenceTarget):
    """A proportional intensity model's side of a Bayesian fit: the
    baseline's parameters and one coefficient a covariate, the fitter's
    own likelihood, one system as the unit."""

    tag = "proportional-intensity"

    def __init__(self, mle: Any, data: EventHistoryData | None = None):
        self.mle = mle
        self.data = mle.data if data is None else data
        self.fitter, self.baseline = _pi_fitters(mle)
        self.k_base = len(self.baseline.parameter_names)
        self.dist = self.fitter
        self.names = list(mle._parameter_names())
        self.bounds = list(mle._parameter_bounds())
        self.fixed = {}
        base = str(self.baseline.name)
        base = base if base.endswith("Process") else f"{base} process"
        self.family = f"Proportional intensity ({base} baseline)"
        first, _ = self.data.item_rows()
        #: Each system's covariate row.
        self.Z_items = np.atleast_2d(np.asarray(self.data.Z, dtype=float))[
            first
        ]
        self._finish()

    def __reduce__(self) -> tuple:
        return (type(self), (self.mle, self.data))

    def _make_neg_ll(self, data: EventHistoryData) -> Any:
        if getattr(self.mle, "kind", "") == "HPP":
            f = self.fitter.create_negll_func(data)

            def natural(theta: npt.NDArray) -> Any:
                theta = np.asarray(theta, dtype=float)
                return f(np.concatenate([[np.log(theta[0])], theta[1:]]))

            return natural
        f = self.fitter.create_negll_func(data, self.baseline)
        return lambda theta: f(np.asarray(theta, dtype=float))

    def _split(self, data: EventHistoryData) -> tuple:
        s = data.split_for_nhpp_likelihood()
        return s, _covariate_pieces(s, np.atleast_2d(data.Z))

    def estimates(self) -> npt.NDArray:
        return np.asarray(self.mle._mle, dtype=float)

    def with_params(self, theta: npt.NDArray) -> Any:
        import copy

        theta = np.asarray(theta, dtype=float)
        model = copy.copy(self.mle)
        model.params = theta[: self.k_base]
        model.coeffs = theta[self.k_base :]
        model._mle = theta
        return model

    def simulate(self, theta: npt.NDArray, rng: Any) -> list:
        self._refuse_truncated()
        theta = np.asarray(theta, dtype=float)
        base, coef = theta[: self.k_base], theta[self.k_base :]
        lo = np.atleast_1d(self.baseline.cif(self.entry, *base))
        hi = np.atleast_1d(self.baseline.cif(self.exit, *base))
        rate = np.exp(self.Z_items @ coef)
        out = []
        for a, b, r in zip(lo, hi, rate):
            m = int(rng.poisson(max(r * (b - a), 0.0)))
            u = np.sort(a + (b - a) * rng.uniform(size=m))
            out.append(
                np.asarray(self.baseline.inv_cif(u, *base), dtype=float)
            )
        return out

    def default_priors(self, spec: Any, fixed: dict) -> dict:
        """``Normal(0, 2.5 / sd)`` for each coefficient with no prior,
        ``sd`` its covariate's standard deviation over the systems."""
        from surpyval.bayes.prior import NormalOnScale

        covered = set(spec.single) | set(fixed)
        for j in spec.joint:
            if j.is_likelihood:
                continue
            if j.names is None:
                return {}
            covered |= set(j.names)
        out = {}
        for k, name in enumerate(self.names[self.k_base :]):
            if name in covered:
                continue
            sd = float(np.std(self.Z_items[:, k]))
            if not sd > 0:
                raise ValueError(
                    f"The covariate of {name} is the same for every "
                    "system, so the data say nothing about its coefficient "
                    "and it has no scale for a default prior. Leave the "
                    f"column out, or give the coefficient a prior "
                    f"(priors={{{name!r}: ...}})."
                )
            scale = COEFFICIENT_SCALE / sd
            out[name] = NormalOnScale(
                loc=0.0,
                scale=scale,
                kind="identity",
                said=f"default: Normal(0, {scale:.4g}), which is "
                f"{COEFFICIENT_SCALE:g} over the covariate's standard "
                f"deviation ({sd:.4g})",
            )
        return out

    def conjugate(self, bound: Any) -> Any | None:
        return None

    def quantities(self) -> dict:
        return {}

    def describe_data(self) -> str:
        k = self.Z_items.shape[1]
        return (
            super().describe_data() + f", {k} covariate{'s' if k != 1 else ''}"
        )

    def to_dict(self) -> dict:
        return {
            "target": self.tag,
            "mle": self.mle.to_dict(),
            "data": self.data.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ProportionalIntensityTarget:
        import surpyval

        with ignoring_warnings():
            mle = surpyval.from_dict(d["mle"])
        return cls(mle, EventHistoryData.from_dict(d["data"]))


def _is_pi_fit(model: Any) -> bool:
    return (
        type(model).__name__ == "ProportionalIntensityModel"
        and getattr(model, "data", None) is not None
        and getattr(model, "_mle", None) is not None
        and not hasattr(model, "frailty")
    )


register_adapter(_is_pi_fit, lambda model: ProportionalIntensityTarget(model))


class BayesianProportionalIntensity(BayesianRecurrence):
    """
    A Bayesian fit of a proportional intensity model
    (``ProportionalIntensityHPP.fit_bayes``,
    ``ProportionalIntensityNHPP.fit_bayes``): the posterior of the
    baseline's parameters and the covariates' coefficients, with every
    result given for a covariate set ``Z`` (one row: one kind of
    system).

    Attributes
    ----------
    params : numpy.ndarray
        The posterior medians: the baseline's parameters, then the
        coefficients (``parameter_names``).
    default_priors : dict
        The coefficients given ``Normal(0, 2.5 / sd)`` because none was
        stated, in words.
    method : str
        ``"grid"`` (exact; up to three free parameters) or
        ``"laplace"`` (approximate, with its checks in ``diagnostics``).

    Examples
    --------
    Eight systems, half of them in a harsher duty (``Z = 1``):

    >>> import numpy as np
    >>> from surpyval.recurrent import ProportionalIntensityHPP
    >>> rng = np.random.default_rng(1)
    >>> x, Z, i, c = [], [], [], []
    >>> for k in range(8):
    ...     z = float(k % 2)
    ...     m = rng.poisson(0.02 * np.exp(0.7 * z) * 300.0)
    ...     x += [*np.sort(300.0 * rng.uniform(size=m)), 300.0]
    ...     Z += [[z]] * (m + 1)
    ...     i += [k] * (m + 1)
    ...     c += [0] * m + [1]
    >>> model = ProportionalIntensityHPP.fit_bayes(x, Z, i, c)
    >>> model.method, model.parameter_names
    ('grid', ['lambda', 'coef_0'])
    >>> lo, hi = model.cri(300.0, [1.0], on="mtbf")
    >>> bool(lo < model.mtbf(300.0, [1.0]) < hi)
    True
    >>> bool(model.mtbf(300.0, [0.0]) > model.mtbf(300.0, [1.0]))
    True
    """

    model_tag = "BayesianProportionalIntensity"

    def _scale(self, Z: Any) -> npt.NDArray:
        """``exp(Z beta)`` at every posterior point, for one covariate
        set."""
        k = self.target.k_base
        p = len(self.parameter_names) - k
        z = np.asarray(Z, dtype=float).reshape(-1)
        if z.size != p:
            raise ValueError(
                f"Z is one covariate set: {p} value"
                f"{'s' if p != 1 else ''}, one for each of "
                f"{', '.join(self.parameter_names[k:])}; got "
                f"{np.shape(Z)}. Call once for each covariate set."
            )
        with np.errstate(over="ignore"):
            out: npt.NDArray = np.exp(self.posterior.points[:, k:] @ z)
        return out

    def cif(self, x: npt.ArrayLike, Z: Any) -> npt.NDArray:
        """The expected number of events by ``x`` for a system with
        covariates ``Z``: the posterior mean."""
        x = np.asarray(x, dtype=float)
        v = self._values("cif", x, self._scale(Z))
        return np.reshape(self.posterior.expectation(v), x.shape)

    mcf = cif

    def iif(self, x: npt.ArrayLike, Z: Any) -> npt.NDArray:
        """The expected intensity at ``x`` for a system with covariates
        ``Z``: the posterior mean."""
        x = np.asarray(x, dtype=float)
        v = self._values("iif", x, self._scale(Z))
        return np.reshape(self.posterior.expectation(v), x.shape)

    def mtbf(self, x: npt.ArrayLike, Z: Any) -> npt.NDArray:
        """The instantaneous MTBF at ``x`` for a system with covariates
        ``Z``: its posterior median."""
        x = np.asarray(x, dtype=float)
        return np.reshape(
            self._quantile("mtbf", x, 0.5, self._scale(Z)), x.shape
        )

    def cri(  # type: ignore[override]
        self,
        x: npt.ArrayLike,
        Z: Any,
        on: str = "cif",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """
        A credible interval for the process of a system with covariates
        ``Z`` at times ``x``.

        Parameters
        ----------
        x : array like or scalar
            The times.
        Z : array like
            One covariate set (one value a covariate).
        on : {'cif', 'iif', 'mtbf'}, optional
            Default ``'cif'``.
        alpha_ci : float, optional
            1 minus the credible level. Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided adds a last ``[lower, upper]`` axis.
        """
        on = {"mcf": "cif", "intensity": "iif"}.get(on, on)
        if on not in _ON:
            raise ValueError(
                f"on must be one of {', '.join(_ON)}; got {on!r}. "
                "(param_cri gives a parameter's interval, function_cri "
                "any function's.)"
            )
        levels = interval_levels(alpha_ci, bound)
        shape = np.shape(x)
        scale = self._scale(Z)
        out = [
            np.reshape(
                self._quantile(on, np.asarray(x, float), level, scale), shape
            )
            for level in levels
        ]
        return np.stack(out, axis=-1) if len(out) == 2 else out[0]

    def mtbf_cri(  # type: ignore[override]
        self,
        Z: Any,
        x: npt.ArrayLike | None = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """A credible interval for the instantaneous MTBF of a system
        with covariates ``Z``: now (``x`` left out: at :attr:`end`) or
        at a later ``x``."""
        at = self.end if x is None else x
        return self.cri(at, Z, on="mtbf", alpha_ci=alpha_ci, bound=bound)

    def mtbf_probability(  # type: ignore[override]
        self, requirement: float, Z: Any, at: float | None = None
    ) -> float:
        """The posterior probability that the instantaneous MTBF of a
        system with covariates ``Z``, at time ``at`` (by default
        :attr:`end`), is at least ``requirement``."""
        requirement = float(requirement)
        if not requirement > 0:
            raise ValueError(
                f"requirement must be a positive MTBF; got {requirement!r}."
            )
        at = self.end if at is None else float(at)
        values = self._values("mtbf", np.array([at]), self._scale(Z))[:, 0]
        return float(self.posterior.weights @ (values >= requirement))

    def count(  # type: ignore[override]
        self,
        Z: Any,
        start: float | None = None,
        end: float | None = None,
        *,
        duration: float | None = None,
        systems: float = 1,
    ) -> CountPrediction:
        """The predictive distribution of the number of events in a
        period for ``systems`` systems with covariates ``Z`` (see
        :meth:`BayesianRecurrence.count`)."""
        start = self.end if start is None else float(start)
        if (end is None) == (duration is None):
            raise ValueError(
                "Give the period's end or its duration (one of them)."
            )
        if end is None:
            assert duration is not None
            end = start + float(duration)
        end = float(end)
        if not end > start:
            raise ValueError(
                f"The period must end after it starts; got ({start:g}, "
                f"{end:g}]."
            )
        if not float(systems) > 0:
            raise ValueError(f"systems must be positive; got {systems!r}.")
        values = self._values("cif", np.array([start, end]), self._scale(Z))
        # (inf - inf at a grid's edge, a point of no weight, which
        # CountPrediction leaves out)
        with np.errstate(invalid="ignore"):
            means = float(systems) * (values[:, 1] - values[:, 0])
        return CountPrediction(
            means, self.posterior.weights, start, end, float(systems)
        )


def fit_proportional_intensity(
    fitter: Any,
    x: Any,
    Z: Any,
    i: Any = None,
    c: Any = None,
    n: Any = None,
    t: Any = None,
    tl: Any = None,
    tr: Any = None,
    *,
    priors: Any = None,
    method: str = "auto",
    importance: bool = False,
    points: int | None = None,
    **fit_options: Any,
) -> BayesianProportionalIntensity:
    """The Bayesian fit of a proportional intensity model; see the
    fitters' ``fit_bayes``."""
    from surpyval.utils.no_maximum import quiet_maximum_warnings

    try:
        with quiet_maximum_warnings(), np.errstate(all="ignore"):
            mle = fitter.fit(
                x, Z, i=i, c=c, n=n, t=t, tl=tl, tr=tr, **fit_options
            )
    except ValueError as e:
        raise ValueError(
            "fit_bayes starts from the maximum-likelihood fit, which "
            f"refuses these data: {e}"
        ) from None
    aliased = np.atleast_1d(getattr(mle, "aliased", ())).astype(int)
    if aliased.size:
        names = mle._coefficient_names()
        raise ValueError(
            "The data do not determine "
            + ", ".join(names[j] for j in aliased.tolist())
            + " (the coefficient of a column that is constant or a "
            "combination of the others). Leave the column out."
        )
    model: BayesianProportionalIntensity = fit_target(
        ProportionalIntensityTarget(mle),
        priors,
        method=method,
        importance=importance,
        points=points,
        result=BayesianProportionalIntensity,
    )
    return model


# ---------------------------------------------------------------------------
# Cause-specific processes: one posterior a cause
# ---------------------------------------------------------------------------
#: The draws, and their seed, that combine the causes' posteriors into
#: the total's.
TOTAL_DRAWS = 4000
TOTAL_SEED = 20261125


class BayesianCauseSpecific(SerialisableMixin):
    """
    A Bayesian fit of cause-specific intensities
    (``CauseSpecificNHPP.fit_bayes``): one process a failure cause, each
    with its own posterior.

    The causes' likelihoods are separate (each cause's events are an
    independent thinning of the whole process), so with priors stated a
    cause at a time their posteriors are independent, and each is the
    :class:`BayesianRecurrence` of that cause's events, in ``models``.
    The total over causes (``event=None``) combines them by
    ``TOTAL_DRAWS`` joint draws with a fixed seed.

    Attributes
    ----------
    models : dict
        Each cause's Bayesian fit.
    event_types : list
        The causes.
    mle : CauseSpecificNHPP
        The maximum-likelihood fit to the same data.
    end : float
        The latest time observed.

    Examples
    --------
    >>> import surpyval as sp
    >>> from surpyval.recurrent import CauseSpecificNHPP
    >>> x = [15, 42, 60, 98, 130, 171, 205, 260, 310, 345, 390, 400]
    >>> e = ["seal", "pump", "seal", "pump", "seal", "seal", "pump",
    ...      "seal", "pump", "seal", "seal", None]
    >>> model = CauseSpecificNHPP.fit_bayes(x, c=[0] * 11 + [1], e=e)
    >>> sorted(model.models)
    ['pump', 'seal']
    >>> model.cri(400.0, "seal", on="mtbf").shape
    (2,)
    >>> lo, hi = model.cri(400.0, on="mtbf")       # all causes together
    >>> bool(lo < hi < model.cri(400.0, "seal", on="mtbf")[1])
    True
    """

    model_tag = "BayesianCauseSpecific"

    def __init__(self, models: dict, mle: Any = None) -> None:
        # The causes as plain Python values, so a fit and its saved copy
        # name them alike ('A', not np.str_('A')) in summaries and errors.
        self.models = {
            (k.item() if isinstance(k, np.generic) else k): m
            for k, m in models.items()
        }
        self.event_types = list(self.models)
        self.mle = mle
        self.end = float(max(m.end for m in self.models.values()))
        self.method = {k: m.method for k, m in self.models.items()}
        self.diagnostics = {k: m.diagnostics for k, m in self.models.items()}

    def _cause(self, event: Any) -> BayesianRecurrence:
        if event not in self.models:
            raise ValueError(
                f"No cause {event!r}; the causes are "
                f"{', '.join(repr(k) for k in self.event_types)} "
                "(event=None is all of them together)."
            )
        fit: BayesianRecurrence = self.models[event]
        return fit

    def _total(self, on: str, x: npt.NDArray) -> npt.NDArray:
        """``on`` summed over the causes at joint draws, ``(draws, m)``:
        the causes' posteriors are independent, so a joint draw is one
        from each."""
        x = np.asarray(x, dtype=float).reshape(-1)
        total = np.zeros((TOTAL_DRAWS, x.size))
        base = "cif" if on == "cif" else "iif"
        for j, fit in enumerate(self.models.values()):
            draws = fit.posterior.draws(
                TOTAL_DRAWS, random_state=TOTAL_SEED + j
            )
            cols = _columns(draws)
            fitter = fit.target.baseline
            with ignoring_warnings(), np.errstate(all="ignore"):
                total = total + np.asarray(
                    getattr(fitter, base)(x[None, :], *cols), dtype=float
                )
        return 1.0 / total if on == "mtbf" else total

    def cif(self, x: npt.ArrayLike, event: Any = None) -> npt.NDArray:
        """The expected number of events of cause ``event`` by ``x``
        (all causes for ``event=None``): the posterior mean."""
        if event is not None:
            return self._cause(event).cif(x)
        x = np.asarray(x, dtype=float)
        return sum(  # type: ignore[return-value]
            np.asarray(m.cif(x)) for m in self.models.values()
        )

    def iif(self, x: npt.ArrayLike, event: Any = None) -> npt.NDArray:
        """The expected intensity of cause ``event`` at ``x`` (all
        causes for ``event=None``): the posterior mean."""
        if event is not None:
            return self._cause(event).iif(x)
        x = np.asarray(x, dtype=float)
        return sum(  # type: ignore[return-value]
            np.asarray(m.iif(x)) for m in self.models.values()
        )

    def mtbf(self, x: npt.ArrayLike, event: Any = None) -> npt.NDArray:
        """The instantaneous MTBF at ``x``, counting failures of cause
        ``event`` only, or of any cause (``event=None``): its posterior
        median."""
        if event is not None:
            return self._cause(event).mtbf(x)
        x = np.asarray(x, dtype=float)
        return np.reshape(np.median(self._total("mtbf", x), axis=0), x.shape)

    def cri(
        self,
        x: npt.ArrayLike,
        event: Any = None,
        on: str = "cif",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """
        A credible interval for one cause's process at times ``x``, or
        for all causes together (``event=None``).

        Parameters
        ----------
        x : array like or scalar
            The times.
        event : optional
            The cause; ``None`` (the default) is the total.
        on : {'cif', 'iif', 'mtbf'}, optional
            Default ``'cif'``.
        alpha_ci : float, optional
            1 minus the credible level. Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided adds a last ``[lower, upper]`` axis.
        """
        if event is not None:
            return self._cause(event).cri(x, on, alpha_ci, bound)
        on = {"mcf": "cif", "intensity": "iif"}.get(on, on)
        if on not in _ON:
            raise ValueError(
                f"on must be one of {', '.join(_ON)}; got {on!r}."
            )
        levels = interval_levels(alpha_ci, bound)
        shape = np.shape(x)
        values = self._total(on, np.asarray(x, dtype=float))
        out = [
            np.reshape(np.quantile(values, level, axis=0), shape)
            for level in levels
        ]
        return np.stack(out, axis=-1) if len(out) == 2 else out[0]

    def param_cri(
        self,
        event: Any,
        name: str,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """A credible interval for a parameter of one cause's process."""
        return self._cause(event).param_cri(name, alpha_ci, bound)

    def mtbf_probability(
        self, requirement: float, event: Any = None, at: float | None = None
    ) -> float:
        """The posterior probability that the instantaneous MTBF at
        ``at`` (by default :attr:`end`) is at least ``requirement``,
        counting failures of cause ``event`` only or of any cause
        (``event=None``)."""
        if event is not None:
            return self._cause(event).mtbf_probability(requirement, at)
        requirement = float(requirement)
        if not requirement > 0:
            raise ValueError(
                f"requirement must be a positive MTBF; got {requirement!r}."
            )
        at = self.end if at is None else float(at)
        values = self._total("mtbf", np.array([at]))[:, 0]
        return float(np.mean(values >= requirement))

    def count(
        self,
        event: Any = None,
        start: float | None = None,
        end: float | None = None,
        *,
        duration: float | None = None,
        systems: float = 1,
    ) -> CountPrediction:
        """The predictive distribution of the number of events of cause
        ``event`` in a period, or of any cause (``event=None``); see
        :meth:`BayesianRecurrence.count`."""
        if event is not None:
            return self._cause(event).count(
                start, end, duration=duration, systems=systems
            )
        start = self.end if start is None else float(start)
        if (end is None) == (duration is None):
            raise ValueError(
                "Give the period's end or its duration (one of them)."
            )
        if end is None:
            assert duration is not None
            end = start + float(duration)
        end = float(end)
        if not end > start:
            raise ValueError(
                f"The period must end after it starts; got ({start:g}, "
                f"{end:g}]."
            )
        values = self._total("cif", np.array([start, end]))
        means = float(systems) * (values[:, 1] - values[:, 0])
        weights = np.full(len(means), 1.0 / len(means))
        return CountPrediction(means, weights, start, end, float(systems))

    def summary(self, alpha_ci: float = 0.05) -> str:
        """Each cause's fit in words, one after another."""
        parts = []
        for cause, fit in self.models.items():
            parts.append(f"Cause {cause!r}\n" + fit.summary(alpha_ci))
        return "\n\n".join(parts)

    def __repr__(self) -> str:
        return self.summary()

    def diagnose(self, replicates: int = 200, random_state: Any = 0) -> dict:
        """Each cause's checks (``BayesianRecurrence.diagnose``), by
        cause."""
        return {
            cause: fit.diagnose(replicates, random_state)
            for cause, fit in self.models.items()
        }

    def to_dict(self) -> dict:
        """A strict-JSON dictionary of the fit: each cause's, by the
        cause's position in ``event_types`` (which is saved with
        them)."""
        import surpyval
        from surpyval.serialisation import stamp_schema, to_native

        return stamp_schema(
            {
                "model": self.model_tag,
                # (a tuple label is saved as a list, and read back)
                "event_types": to_native(list(self.event_types)),
                "models": [m.to_dict() for m in self.models.values()],
                "surpyval_version": surpyval.__version__,
            }
        )

    @classmethod
    def from_dict(cls, model_dict: dict) -> BayesianCauseSpecific:
        """Rebuild a fit from :meth:`to_dict`."""
        from surpyval.serialisation import require_model_tag
        from surpyval.univariate.competing_risks.labels import (
            label_from_native,
        )

        require_model_tag(
            model_dict, cls.model_tag, "a Bayesian cause-specific fit"
        )
        models = {
            label_from_native(cause): BayesianRecurrence.from_dict(d)
            for cause, d in zip(
                model_dict["event_types"], model_dict["models"]
            )
        }
        return cls(models)


def fit_cause_specific(
    cls: Any,
    x: Any,
    i: Any = None,
    c: Any = None,
    n: Any = None,
    e: Any = None,
    tl: Any = None,
    tr: Any = None,
    *,
    dist: Any = None,
    priors: Any = None,
    method: str = "auto",
    importance: bool = False,
    points: int | None = None,
) -> BayesianCauseSpecific:
    """The Bayesian fit of cause-specific intensities; see
    ``CauseSpecificNHPP.fit_bayes``."""
    from collections.abc import Mapping

    from surpyval.utils.no_maximum import quiet_maximum_warnings

    kwargs = {} if dist is None else {"dist": dist}
    try:
        with quiet_maximum_warnings(), np.errstate(all="ignore"):
            mle = cls.fit(x, i=i, c=c, n=n, e=e, tl=tl, tr=tr, **kwargs)
    except ValueError as err:
        raise ValueError(
            "fit_bayes starts from the maximum-likelihood fit, which "
            f"refuses these data: {err}"
        ) from None
    causes = list(mle.event_types)
    by_cause = (
        isinstance(priors, Mapping)
        and len(priors) > 0
        and all(k in causes for k in priors)
    )
    if isinstance(priors, Mapping) and not by_cause:
        if any(k in causes for k in priors):
            raise ValueError(
                "priors mixes causes and parameter names. Give either one "
                "set for every cause (priors={'beta': ...}) or a set a "
                "cause (priors={cause: {'beta': ...}}); the causes are "
                f"{', '.join(repr(k) for k in causes)}."
            )
    models = {}
    for cause in causes:
        own = priors.get(cause) if by_cause else priors
        fitted = mle.models[cause]
        models[cause] = fit_target(
            RecurrenceTarget(fitted.dist, fitted.data, fitted),
            own,
            method=method,
            importance=importance,
            points=points,
            result=BayesianRecurrence,
        )
    return BayesianCauseSpecific(models, mle)


__all__ = [
    "BayesianCauseSpecific",
    "BayesianProportionalIntensity",
    "BayesianRecurrence",
    "CountPrediction",
    "ProportionalIntensityTarget",
    "RecurrenceTarget",
    "fit",
    "fit_cause_specific",
    "fit_proportional_intensity",
]
