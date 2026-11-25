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

from surpyval.bayes.engines.conjugate import ConjugateResult, _prior_family
from surpyval.bayes.result import BayesianResult, _levels, fit_target
from surpyval.bayes.target import (
    Reparameterisation,
    Target,
    register_adapter,
    register_target,
)
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


def _pieces_log_lik(fitter: Any, s: dict, theta: npt.NDArray) -> npt.NDArray:
    """The NHPP log-likelihood of the pieces ``s`` (one item's, or every
    item's; ``EventHistoryData.split_for_nhpp_likelihood``) at rows of
    parameters, ``(N, k) -> (N,)``: the terms of
    ``nhpp_log_likelihood``, with the intensity evaluated at every row
    at once. It is used only where it agrees with the fitter's own
    likelihood (checked at each fit)."""
    cols = _columns(theta)

    def cif(x: npt.NDArray) -> npt.NDArray:
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return np.zeros((len(theta), 0))
        return np.asarray(fitter.cif(x[None, :], *cols), dtype=float)

    def counted(n: npt.NDArray, delta: npt.NDArray) -> npt.NDArray:
        if delta.shape[1] == 0:
            return np.zeros(len(theta))
        out: npt.NDArray = (n * np.log(delta) - delta - gammaln(n + 1)).sum(
            axis=1
        )
        return out

    def at_least_one(delta: npt.NDArray) -> npt.NDArray:
        if delta.shape[1] == 0:
            return np.zeros(len(theta))
        out: npt.NDArray = np.log(-np.expm1(-delta)).sum(axis=1)
        return out

    x_o = np.asarray(s["x_o"], dtype=float)
    ll = np.zeros(len(theta))
    if x_o.size:
        log_iif = np.asarray(fitter.log_iif(x_o[None, :], *cols), dtype=float)
        ll = ll + (log_iif + cif(s["x_o_prev"]) - cif(x_o)).sum(axis=1)
    ll = ll + (cif(s["x_right_prev"]) - cif(s["x_right"])).sum(axis=1)
    ll = ll + counted(s["n_left"], cif(s["x_left"]) - cif(s["x_left_prev"]))
    ll = ll + counted(s["n_i"], cif(s["x_i_r"]) - cif(s["x_i_l"]))
    ll = ll - at_least_one(cif(s["sel_to"]) - cif(s["sel_from"]))
    ll = ll + at_least_one(cif(s["gap_to"]) - cif(s["gap_from"]))
    return ll


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
        self.dist = fitter
        self.data = data
        self.mle = mle
        self.names = list(fitter.parameter_names)
        self.bounds = list(fitter.bounds)
        self.fixed = {}
        name = str(fitter.name)
        self.family = name if name.endswith("Process") else f"{name} process"
        self._neg_ll = _natural_neg_ll(fitter, data)
        self._pieces = data.split_for_nhpp_likelihood()
        entry, exit_ = data.item_observation_windows()
        lower = float(getattr(fitter, "support", (0.0, np.inf))[0])
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
                fast = _pieces_log_lik(self.fitter, self._pieces, pts)
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
            out = _pieces_log_lik(self.fitter, self._pieces, theta)
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
                pieces.append(sub.split_for_nhpp_likelihood())
                negs.append(_natural_neg_ll(self.fitter, sub))
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
                    out[:, u] = _pieces_log_lik(self.fitter, s, theta)
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
        fam = None if prior is None else _prior_family(prior)
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
        if getattr(self.fitter, "name", "") != "Crow-AMSAA":
            return None
        if list(bound.free_names) != ["alpha", "beta"]:
            return None
        log_end = float(np.log(self.end))

        def forward(phi: npt.NDArray) -> npt.NDArray:
            return np.column_stack(
                [np.exp(phi[:, 1]) * (log_end - phi[:, 0]), phi[:, 1]]
            )

        def inverse(u: npt.NDArray) -> npt.NDArray:
            return np.column_stack(
                [log_end - u[:, 0] * np.exp(-u[:, 1]), u[:, 1]]
            )

        def log_det(u: npt.NDArray) -> npt.NDArray:
            out: npt.NDArray = -u[:, 1]
            return out

        return Reparameterisation(
            ["the expected events by the end of observation", "beta"],
            forward,
            inverse,
            log_det,
        )

    def _at(self, fname: str, t: float) -> Any:
        fitter, names = self.fitter, self.names

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
        from surpyval.bayes.prior import _fmt

        return {
            "mtbf_at": lambda t: (
                f"MTBF at {_fmt(t)}",
                self._at("mtbf", t),
                "log",
            ),
            "intensity_at": lambda t: (
                f"intensity at {_fmt(t)}",
                self._at("iif", t),
                "log",
            ),
            "cif_at": lambda t: (
                f"expected events by {_fmt(t)}",
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
        lo, hi = self.quantile(_levels(alpha_ci, "two-sided"))
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
    def _values(self, on: str, x: npt.NDArray) -> npt.NDArray:
        """``on`` at times ``x`` for every posterior point, ``(P, m)``."""
        fitter = self.target.fitter
        cols = _columns(self.posterior.points)
        x = np.asarray(x, dtype=float).reshape(-1)
        with ignoring_warnings(), np.errstate(all="ignore"):
            if on == "cif":
                out = fitter.cif(x[None, :], *cols)
            else:
                out = fitter.iif(x[None, :], *cols)
                if on == "mtbf":
                    out = 1.0 / np.asarray(out, dtype=float)
        return np.broadcast_to(
            np.asarray(out, dtype=float), (len(self.posterior.points), x.size)
        )

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
        return self.posterior.expectation(self._values("cif", x))

    mcf = cif

    @keeps_query_shape
    def iif(self, x: npt.ArrayLike) -> npt.NDArray:
        """The expected intensity (events a unit time) at ``x``: the
        posterior mean."""
        x = np.asarray(x, dtype=float)
        exact = self._exact_rate()
        if exact is not None:
            return np.asarray(exact.mean() * np.ones_like(x))
        return self.posterior.expectation(self._values("iif", x))

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

    def _quantile(self, on: str, x: npt.NDArray, level: float) -> npt.NDArray:
        x = np.asarray(x, dtype=float).reshape(-1)
        exact = self._exact_rate()
        if exact is not None:
            # every function is monotone in the rate
            if on == "mtbf":
                return np.full(x.shape, 1.0 / float(exact.ppf(1.0 - level)))
            rate = float(exact.ppf(level))
            return rate * x if on == "cif" else np.full(x.shape, rate)
        return np.asarray(
            self.posterior.wquantile(self._values(on, x), level), dtype=float
        ).reshape(x.shape)

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
        levels = _levels(alpha_ci, bound)
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


def _zero_event_data(
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
        data = _zero_event_data(x, i, c, n, t, tl, tr, windows)
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


__all__ = [
    "BayesianRecurrence",
    "CountPrediction",
    "RecurrenceTarget",
    "fit",
]
