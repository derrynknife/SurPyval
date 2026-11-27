"""Bayesian fits of the renewal and imperfect-repair processes
(``GeneralizedRenewal``, ``ARA`` and ``ARI``): priors on the repair
parameter and the life (or baseline intensity) times the fitter's own
likelihood.

In words: each repair leaves a system somewhere between as good as new
and as bad as old, and the repair parameter (Kijima's ``q``, ARA's and
ARI's ``rho``) says where. The order and spacing of each system's
failures carry the information about it, and a few failures a system
leave it poorly determined; often the data are consistent with one end
of its range (perfect or minimal repair), and the maximum-likelihood
estimate sits on that end, with no standard error. A posterior says how
much of the belief lies near each end, which is the question a
maintainer asks ("is a repair as good as a replacement?").

The posterior of the repair parameter and a two-parameter life (three
free parameters) is computed on the grid, not by the Laplace
approximation: a posterior piled against an end of a range is far from
a normal. The grid is laid on the parameters' unconstrained scales (the
log of a Kijima ``q``, the logit of a ``rho``), where an end of the
range is at infinity: mass near the end is a long tail, which the grid's
stretched lattice reaches (its cells grow exponentially away from the
peak, and it is widened until the density on every face is below
``1e-8`` of the peak). The mass within a distance of an end is then the
posterior probability of a tail the grid holds.

A unit, for WAIC and leave-one-out cross-validation, is one system.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.bayes.result import BayesianResult, fit_target, interval_levels
from surpyval.bayes.target import (
    Reparameterisation,
    Target,
    register_adapter,
    register_target,
)
from surpyval.utils.event_history_data import EventHistoryData
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.warnings import ignoring_warnings

#: The families with ``fit_bayes``, by ``RenewalModel._family()``.
FAMILIES = ("GeneralizedRenewal", "ARA", "ARI")
#: What the ends (and, for a Kijima ``q``, the point 1) of each family's
#: repair parameter mean: ``(name, value, side)``, ``side`` -1 for a lower
#: end (mass below ``value + within``), +1 for an upper end, 0 for an
#: inside point (mass within ``within`` either side).
_LANDMARKS = {
    "GeneralizedRenewal": (
        ("perfect repair (as good as new)", 0.0, -1),
        ("minimal repair (as bad as old)", 1.0, 0),
    ),
    "ARA": (
        ("minimal repair (as bad as old)", 0.0, -1),
        ("perfect repair (as good as new)", 1.0, 1),
    ),
    "ARI": (
        ("minimal repair (as bad as old)", 0.0, -1),
        ("the largest reduction of intensity", 1.0, 1),
    ),
}
#: The prior SurPyval gives a repair parameter given none: uniform on its
#: range, and for a Kijima ``q``, whose range has no upper end, from as
#: good as new (0) to twice as bad as old (2).
GRP_Q_UPPER = 2.0
#: The trial points of the all-rows likelihood's check against the
#: fitter's own: the restoration moved by these factors.
_TRIALS = (1.0, 0.8, 1.1)


def _columns(theta: npt.NDArray) -> list:
    return [theta[:, j][:, None] for j in range(theta.shape[1])]


# ---------------------------------------------------------------------------
# The likelihood at many parameter rows at once
# ---------------------------------------------------------------------------
class RowsLogLik:
    """
    A renewal family's log-likelihood at rows of parameters, ``(N, k) ->
    (N,)``: the terms of the fitter's own ``create_negll_func``, with the
    virtual ages (or intensity reductions) and the life's functions
    evaluated for every row at once. It is used only where it agrees
    with the fitter's own likelihood at trial points
    (:meth:`RenewalTarget._vector_agrees`).
    """

    def __init__(
        self, family: str, data: EventHistoryData, life: Any, option: Any
    ) -> None:
        from surpyval.recurrent.renewal.generalized_renewal import (
            _AGED,
            _previous_in_item,
        )
        from surpyval.recurrent.renewal.renewal_model import (
            event_positions,
            rows_by_position,
        )

        self.family, self.life, self.option = family, life, option
        x = np.asarray(data.x, dtype=float)
        item = np.asarray(data.i)
        self.c = np.asarray(data.c)
        if family == "ARI":
            from surpyval.recurrent.renewal.ari import _event_layout

            prev, observed, failure_pos, in_force = _event_layout(data)
            self.x, self.prev, self.observed = x, prev, observed
            self.failure_pos, self.in_force = failure_pos, in_force
            longest = int(failure_pos.max()) + 1 if failure_pos.size else 0
            m = option
            self.span = longest if np.isinf(m) else min(int(m), longest)
            return
        gap = np.asarray(data.get_interarrival_times(), dtype=float)
        self.gap = gap
        self.fresh = event_positions(item) == 0
        # ARA's likelihood never integrates an aged gap (its ratio is 0)
        self.aged = _AGED if family == "GeneralizedRenewal" else 0.0
        if family == "GeneralizedRenewal" and option == "i":
            self.cumulative = _previous_in_item(x, item)
        elif family == "GeneralizedRenewal":
            self.previous_gap = _previous_in_item(gap, item)
            self.by_position = rows_by_position(event_positions(item))
        else:
            from surpyval.recurrent.renewal.ara import ARAVirtualAges

            ages = ARAVirtualAges(x, item, option)
            self.T, self.groups = ages.T, ages.groups
            self.max_terms = ages.max_terms

    def ages(self, r: npt.NDArray) -> npt.NDArray:
        """The virtual age at the start of every gap, ``(N, rows)``."""
        r = np.asarray(r, dtype=float)[:, None]
        if self.family == "GeneralizedRenewal" and self.option == "i":
            out: npt.NDArray = r * self.cumulative[None, :]
            return out
        if self.family == "GeneralizedRenewal":
            v = np.zeros((len(r), self.previous_gap.size))
            for k, rows in enumerate(self.by_position):
                before = 0.0 if k == 0 else v[:, rows - 1]
                v[:, rows] = r * (before + self.previous_gap[rows][None, :])
            return v
        T = self.T
        v = np.zeros((len(r), T.size))
        weights = (1.0 - r) ** np.arange(self.max_terms)[None, :]
        for n_terms, rows in self.groups:
            lagged = T[rows[:, None] - 1 - np.arange(n_terms)]
            discounted = weights[:, :n_terms] @ lagged.T
            v[:, rows] = T[rows - 1][None, :] - r * discounted
        return v

    def __call__(self, theta: npt.NDArray) -> npt.NDArray:
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        with np.errstate(all="ignore"):
            if self.family == "ARI":
                out = self._ari(theta)
            else:
                out = self._virtual_age(theta)
        return np.where(np.isnan(out), -np.inf, out)

    def _virtual_age(self, theta: npt.NDArray) -> npt.NDArray:
        life, gap, c = self.life, self.gap, self.c
        age = self.ages(theta[:, 0])
        cols = _columns(theta[:, 1:])
        x_new = gap[None, :] + age
        log_sf_v = np.asarray(life.log_sf(age, *cols), dtype=float)
        ll_o = np.asarray(life.log_df(x_new, *cols), dtype=float) - log_sf_v
        ll_r = np.asarray(life.log_sf(x_new, *cols), dtype=float) - log_sf_v
        aged = ~self.fresh[None, :] & (gap[None, :] < self.aged * age)
        if np.any(aged):
            # Simpson's rule over a gap far shorter than its start's age,
            # as the fitter's likelihood takes it
            h0 = np.asarray(life.hf(age, *cols), dtype=float)
            hm = np.asarray(life.hf(age + 0.5 * gap, *cols), dtype=float)
            h1 = np.asarray(life.hf(x_new, *cols), dtype=float)
            drop = -gap / 6.0 * (h0 + 4.0 * hm + h1)
            ll_o = np.where(aged, np.log(h1) + drop, ll_o)
            ll_r = np.where(aged, drop, ll_r)
        ll = np.where(c == 0, ll_o, 0.0)
        ll = np.where(c == 1, ll_r, ll)
        out: npt.NDArray = ll.sum(axis=1)
        return out

    def unit_compensator(
        self, r: npt.NDArray, beta: npt.NDArray
    ) -> npt.NDArray:
        """The expected number of failures over every gap observed (the
        compensator's total), ``(N,)``, at a scale (``alpha``) of 1: for
        a Weibull life or a power-law baseline, the total at a scale
        ``alpha`` is this times ``alpha ** -beta``."""
        r = np.asarray(r, dtype=float)
        beta = np.asarray(beta, dtype=float)[:, None]
        one = np.ones_like(beta)
        with np.errstate(all="ignore"):
            if self.family == "ARI":
                base = self.life
                x, prev, observed = self.x, self.prev, self.observed
                lam = np.broadcast_to(
                    np.asarray(base.iif(x[observed][None, :], one, beta)),
                    (len(r), int(observed.sum())),
                )
                reductions = self.reductions(lam, r)
                idx = np.where(self.in_force >= 0, self.in_force, 0)
                active = np.where(
                    self.in_force[None, :] >= 0,
                    reductions[:, idx] if reductions.shape[1] else 0.0,
                    0.0,
                )
                cif_x = np.asarray(base.cif(x[None, :], one, beta))
                cif_prev = np.asarray(base.cif(prev[None, :], one, beta))
                cif_prev = np.where(prev[None, :] == 0, 0.0, cif_prev)
                total = np.sum(cif_x - cif_prev - active * (x - prev), axis=1)
            else:
                age = self.ages(r)
                end = age + self.gap[None, :]
                total = np.sum(end**beta - age**beta, axis=1)
        out: npt.NDArray = np.asarray(total, dtype=float)
        return out

    def reductions(self, lam: npt.NDArray, rho: npt.NDArray) -> npt.NDArray:
        """ARI's reduction after every failure, ``(N, failures)``, for
        the baseline intensities ``lam`` at the failures."""
        rho = np.asarray(rho, dtype=float)[:, None]
        total = np.array(lam, copy=True)
        for i in range(1, self.span):
            shifted = np.concatenate(
                [np.zeros((len(lam), i)), lam[:, :-i]], axis=1
            )
            total += ((1.0 - rho) ** i) * np.where(
                self.failure_pos >= i, shifted, 0.0
            )
        out: npt.NDArray = rho * total
        return out

    def _ari(self, theta: npt.NDArray) -> npt.NDArray:
        base = self.life
        cols = _columns(theta[:, 1:])
        x, prev, observed = self.x, self.prev, self.observed
        n = len(theta)
        lam = np.broadcast_to(
            np.asarray(base.iif(x[observed][None, :], *cols), dtype=float),
            (n, int(observed.sum())),
        )
        reductions = self.reductions(lam, theta[:, 0])
        idx = np.where(self.in_force >= 0, self.in_force, 0)
        active = np.where(
            self.in_force[None, :] >= 0,
            reductions[:, idx] if reductions.shape[1] else 0.0,
            0.0,
        )
        intensity = lam - active[:, observed]
        cif_x = np.asarray(base.cif(x[None, :], *cols), dtype=float)
        cif_prev = np.asarray(base.cif(prev[None, :], *cols), dtype=float)
        cif_prev = np.where(prev[None, :] == 0, 0.0, cif_prev)
        delta = cif_x - cif_prev
        ll = -np.sum(delta - active * (x - prev)[None, :], axis=1) + np.sum(
            np.log(intensity), axis=1
        )
        out: npt.NDArray = np.where(np.all(intensity > 0, axis=1), ll, -np.inf)
        return out


def _option(mle: Any) -> Any:
    """The family's option: a Kijima type or a memory."""
    if mle._family() == "GeneralizedRenewal":
        return mle.kijima_type
    return mle.m


def _create_negll(
    fitter: Any, family: str, data: Any, life: Any, opt: Any
) -> Any:
    if family == "GeneralizedRenewal":
        return fitter.create_negll_func(data, life, kijima=opt)
    return fitter.create_negll_func(data, life, opt)


# ---------------------------------------------------------------------------
# The target
# ---------------------------------------------------------------------------
@register_target
class RenewalTarget(Target):
    """A renewal or imperfect-repair process's side of a Bayesian fit
    (:mod:`surpyval.bayes.target`): the fitter's own likelihood, one
    system as the unit, and replicates simulated over each system's
    observation window."""

    tag = "renewal"
    unit_name = "system"

    def __init__(self, mle: Any) -> None:
        import surpyval.recurrent as recurrent

        family = mle._family()
        if family not in FAMILIES:
            raise ValueError(
                f"fit_bayes covers {', '.join(FAMILIES)}; got {family}."
            )
        self.mle = mle
        self.data = mle.data
        self.family_name = family
        self.fitter = getattr(recurrent, family)
        self.life = mle.model.dist
        self.option = _option(mle)
        self.names = list(mle.parameter_names)
        self.bounds = [tuple(b) for b in mle._parameter_bounds()]
        self.restoration_name = self.names[0]
        self.fixed = {}
        # (what a prior of a life distribution sees; a prior of old lives,
        # prior.from_data, is refused in check)
        self.dist = self.life
        self.k_life = len(self.names) - 1
        if family == "GeneralizedRenewal":
            what = f"Kijima {str(self.option).upper()}"
        else:
            what = f"memory {float(self.option):g}"
        self.family = f"{mle.kind} process ({self.life.name}, {what})"
        self.finish_result()

    def finish_result(self) -> None:
        data = self.data
        self._neg_ll = _create_negll(
            self.fitter, self.family_name, data, self.life, self.option
        )
        self._rows = RowsLogLik(self.family_name, data, self.life, self.option)
        _, exit_ = data.item_observation_windows()
        self.exit = np.asarray(exit_, dtype=float)
        #: The latest time any system was observed to.
        self.end = float(np.max(self.exit))
        self._unit_rows: list | None = None
        self._unit_neg_ll: list | None = None
        self._vector = self._vector_agrees()

    def __reduce__(self) -> tuple:
        # (the likelihood's closures are rebuilt, not pickled)
        return (type(self), (self.mle,))

    # -- the likelihood --------------------
    def neg_ll(self, theta: npt.NDArray) -> float:
        return float(self._neg_ll(np.asarray(theta, dtype=float)))

    def start(self) -> npt.NDArray | None:
        """The maximum-likelihood estimates, with a repair parameter on
        an end of its range moved just inside it (the end is at infinity
        on the parameter's unconstrained scale)."""
        theta = np.asarray(self.estimates(), dtype=float).copy()
        low, high = self.bounds[0]
        if low is not None:
            theta[0] = max(theta[0], low + 1e-4)
        if high is not None:
            theta[0] = min(theta[0], high - 1e-4)
        return theta

    def reparam(self, bound: Any) -> Any | None:
        """For a Weibull life (or a power-law baseline, ARI), the engines
        work with the log of the expected number of failures over the
        gaps observed in place of ``log alpha``: given the repair
        parameter and the shape, the scale is pinned by that count, so
        the posterior in ``(repair, log alpha, log beta)`` is a thin,
        curved ridge (the scale rises with the virtual ages a repair
        leaves), and in ``(repair, log expected failures, log beta)``
        it is nearly a product of its margins. The expected count at
        scale ``alpha`` is ``alpha ** -beta`` times its value at a scale
        of 1 (:meth:`RowsLogLik.unit_compensator`)."""
        life = getattr(self.life, "name", "")
        wanted = "Crow-AMSAA" if self.family_name == "ARI" else "Weibull"
        if life != wanted or not self._vector:
            return None
        free = list(bound.free_names)
        if free != [self.restoration_name, "alpha", "beta"]:
            return None
        t0 = bound.transforms[0]
        rows = self._rows

        def log_total(phi0: npt.NDArray, log_beta: npt.NDArray) -> Any:
            r = np.asarray(t0.to_theta(phi0), dtype=float)
            with np.errstate(all="ignore"):
                return np.log(rows.unit_compensator(r, np.exp(log_beta)))

        def forward(phi: npt.NDArray) -> npt.NDArray:
            phi = np.atleast_2d(phi)
            u1 = (
                log_total(phi[:, 0], phi[:, 2]) - np.exp(phi[:, 2]) * phi[:, 1]
            )
            return np.column_stack([phi[:, 0], u1, phi[:, 2]])

        def inverse(u: npt.NDArray) -> npt.NDArray:
            u = np.atleast_2d(u)
            beta = np.exp(u[:, 2])
            phi1 = (log_total(u[:, 0], u[:, 2]) - u[:, 1]) / beta
            return np.column_stack([u[:, 0], phi1, u[:, 2]])

        def log_det(u: npt.NDArray) -> npt.NDArray:
            out: npt.NDArray = -np.atleast_2d(u)[:, 2]
            return out

        return Reparameterisation(
            [self.restoration_name, "the expected failures observed", "beta"],
            forward,
            inverse,
            log_det,
        )

    def _vector_agrees(self) -> bool:
        """Whether the all-rows likelihood matches the fitter's own at
        trial points; where it does not, the fitter's own is called a row
        at a time."""
        try:
            start = self.start()
            assert start is not None
            pts = np.repeat(start[None, :], len(_TRIALS), axis=0)
            pts[:, 0] = np.clip(
                pts[:, 0] * np.asarray(_TRIALS),
                1e-3,
                None if self.bounds[0][1] is None else 0.999,
            )
            pts[1:, 1:] = pts[1:, 1:] * 1.05
            with ignoring_warnings(), np.errstate(all="ignore"):
                fast = self._rows(pts)
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
        out = np.empty(len(theta))
        # (in blocks: a grid of 10^5 points times the rows of the data)
        step = max(1, 2_000_000 // max(self._rows_size(), 1))
        for a in range(0, len(theta), step):
            out[a : a + step] = self._rows(theta[a : a + step])
        return out

    def _rows_size(self) -> int:
        return int(np.size(self.data.x))

    def estimates(self) -> npt.NDArray:
        return np.asarray(self.mle.params, dtype=float)

    def covariance(self) -> npt.NDArray:
        return np.asarray(self.mle.covariance(), dtype=float)

    def mle_interval(self, name: str, alpha_ci: float) -> tuple | None:
        try:
            with ignoring_warnings(), np.errstate(all="ignore"):
                ci = np.asarray(
                    self.mle.param_cb(name, alpha_ci=alpha_ci), dtype=float
                ).reshape(-1)
        except (ValueError, TypeError, KeyError, AttributeError):
            return None
        if ci.size == 2 and np.all(np.isfinite(ci)):
            return float(ci[0]), float(ci[1])
        return None

    # -- units: one system --------------------
    def _units(self) -> tuple[list, list]:
        if self._unit_rows is None:
            first, inverse = self.data.item_rows()
            rows, negs = [], []
            for k in range(len(first)):
                sub = self.data[np.flatnonzero(inverse == k)]
                rows.append(
                    RowsLogLik(self.family_name, sub, self.life, self.option)
                )
                negs.append(
                    _create_negll(
                        self.fitter,
                        self.family_name,
                        sub,
                        self.life,
                        self.option,
                    )
                )
            self._unit_rows, self._unit_neg_ll = rows, negs
        assert self._unit_neg_ll is not None
        return self._unit_rows, self._unit_neg_ll

    def unit_log_lik(self, theta: npt.NDArray) -> npt.NDArray:
        theta = np.atleast_2d(theta)
        rows, negs = self._units()
        out = np.empty((len(theta), len(rows)))
        with ignoring_warnings(), np.errstate(all="ignore"):
            for u, (r, f) in enumerate(zip(rows, negs)):
                if self._vector:
                    out[:, u] = r(theta)
                else:
                    out[:, u] = [-float(f(row)) for row in theta]
        return np.where(np.isnan(out), -np.inf, out)

    def unit_counts(self) -> npt.NDArray:
        return np.ones(len(self.exit))

    # -- models and replicates --------------------
    def with_params(self, theta: npt.NDArray) -> Any:
        return self.mle.with_params(np.asarray(theta, dtype=float))

    def simulate(self, theta: npt.NDArray, rng: Any) -> list:
        """Each system's failure times over its own observation window,
        from new, from the process at ``theta``."""
        model = self.with_params(theta)
        out = []
        for end in self.exit:
            _, _, ahead = model._simulate_from_states(
                model.states(age=[0.0]), float(end), 1, rng
            )
            out.append(np.sort(np.asarray(ahead, dtype=float)))
        return out

    def observed(self) -> list:
        failures = self.mle._states().failures
        return [np.asarray(f, dtype=float) for f in failures]

    def statistics(self) -> dict:
        def events(sets: list) -> float:
            return float(sum(len(s) for s in sets))

        def most(sets: list) -> float:
            return float(max(len(s) for s in sets))

        def shortest_gap(sets: list) -> float:
            gaps = [np.diff(np.r_[0.0, s]) for s in sets if len(s)]
            return float(np.min(np.concatenate(gaps))) if gaps else np.nan

        out = {"number of events": events, "shortest gap": shortest_gap}
        if len(self.exit) > 1:
            out["most events on one system"] = most
        return out

    # -- priors --------------------
    def default_priors(self, spec: Any, fixed: dict) -> dict:
        """The repair parameter, given no prior, is uniform over its
        range (a Kijima ``q`` from 0 to :data:`GRP_Q_UPPER`): a flat
        prior on its unconstrained scale would have no finite mass where
        the data are consistent with an end of the range."""
        from surpyval.bayes.prior import uniform

        name = self.restoration_name
        covered = set(spec.single) | set(fixed)
        for j in spec.joint:
            if j.is_likelihood:
                continue
            covered |= set(self.names if j.names is None else j.names)
        if name in covered:
            return {}
        low, high = self.bounds[0]
        upper = GRP_Q_UPPER if high is None else float(high)
        return {name: uniform(float(low), upper)}

    def check(self, bound: Any) -> None:
        from surpyval.bayes.prior import PowerPrior

        for j in bound.spec.joint:
            if isinstance(j, PowerPrior):
                raise ValueError(
                    "prior.from_data holds the lives of a life "
                    "distribution, not a repairable system's history. For "
                    "the life, use prior.from_model(life_fit, param=[...]) "
                    "of a Weibull (or other) fit to comparable lives; for "
                    "an earlier fit of the same process, "
                    "prior.from_fit(earlier_model) or "
                    "prior.from_model(earlier_model)."
                )

    def n_events(self) -> float:
        return float(np.sum(np.asarray(self.data.c) == 0))

    def check_flat(self, bound: Any) -> None:
        improper = bound.improper_names
        if self.restoration_name in improper and not any(
            j.is_likelihood for j in bound.spec.joint
        ):
            raise ValueError(
                f"{self.restoration_name} has a flat prior on its "
                "unconstrained scale, which has no finite mass where the "
                "data are consistent with an end of its range (perfect or "
                "minimal repair). Give it a proper prior: leave it out for "
                "the default (uniform over its range), or "
                f"prior.uniform(low, high), or a Beta for a judgement."
            )

    def grid_checks(self, theta: dict) -> list:
        return []

    # -- words and saving --------------------
    def describe_data(self) -> str:
        k = len(self.exit)
        events = self.n_events()
        return (
            f"{k} system{'s' if k != 1 else ''}, {events:g} "
            f"failure{'s' if events != 1 else ''}, observed to {self.end:g}"
        )

    def same_data(self, other: Any) -> bool:
        return bool(self.data.equals(other.data))

    def to_dict(self) -> dict:
        return {
            "target": self.tag,
            "mle": self.mle.to_dict(),
            "data": self.data.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> RenewalTarget:
        from surpyval.recurrent.renewal.renewal_model import RenewalModel
        from surpyval.utils.no_maximum import quiet_maximum_warnings

        data = EventHistoryData.from_dict(d["data"])
        with ignoring_warnings(), quiet_maximum_warnings():
            given = RenewalModel.from_dict(d["mle"])
            import surpyval.recurrent as recurrent

            fitter = getattr(recurrent, given._family())
            # the maximum-likelihood fit to the data, again (a saved
            # model holds its estimates, not its likelihood)
            mle = fitter._refit(given, data)
        unit = getattr(given, "unit", None)
        if unit is not None:
            mle.unit = unit
        return cls(mle)


def _is_renewal_fit(model: Any) -> bool:
    from surpyval.recurrent.renewal.renewal_model import RenewalModel

    return (
        isinstance(model, RenewalModel)
        and getattr(model, "how", None) == "MLE"
        and getattr(model, "data", None) is not None
        and model._family() in FAMILIES
    )


register_adapter(_is_renewal_fit, RenewalTarget)


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
class SimulatedCount:
    """
    The posterior predictive distribution of the number of failures in a
    period, by simulation: parameters drawn from the posterior, and each
    unit's future simulated from its state now under each draw, so the
    uncertainty in the parameters is included.

    Attributes
    ----------
    duration : float
        The period's length, from now.
    units : int
        How many units the count is over.
    mean : float
        The expected number of failures.
    unit_means : numpy.ndarray
        The expected number for each unit.
    replicates : int
        The simulated futures the distribution is made of.
    draws : int
        The posterior draws they were simulated under.
    mean_se : float
        The Monte Carlo standard error of :attr:`mean`: the standard
        deviation of the draws' own mean counts over the square root of
        the number of draws. The futures simulated under one draw share
        its parameters, so they are not independent of each other, and
        the error is larger than the count's standard deviation over the
        square root of :attr:`replicates`.
    """

    def __init__(
        self, totals: npt.NDArray, unit_means: npt.NDArray, duration: float
    ) -> None:
        # (one row per posterior draw, one column per future under it)
        by_draw = np.atleast_2d(np.asarray(totals, dtype=int))
        self.draws = int(by_draw.shape[0])
        draw_means = by_draw.mean(axis=1)
        self.mean_se = (
            float(np.std(draw_means, ddof=1) / np.sqrt(self.draws))
            if self.draws > 1
            else float("nan")
        )
        totals = by_draw.reshape(-1)
        self.duration = float(duration)
        self.unit_means = np.asarray(unit_means, dtype=float)
        self.units = int(self.unit_means.size)
        self.replicates = int(totals.size)
        self.mean = float(np.mean(totals))
        self.sd = float(np.std(totals, ddof=1)) if totals.size > 1 else 0.0
        pmf = np.bincount(totals) / totals.size
        self._pmf = pmf
        self._cdf = np.minimum(np.cumsum(pmf), 1.0)

    def pmf(self, k: npt.ArrayLike) -> npt.NDArray:
        """The probability of exactly ``k`` failures."""
        k = np.asarray(k)
        inside = (k >= 0) & (k < len(self._pmf)) & (k == np.floor(k))
        idx = np.clip(k, 0, len(self._pmf) - 1).astype(int)
        return np.asarray(np.where(inside, self._pmf[idx], 0.0), dtype=float)

    def cdf(self, k: npt.ArrayLike) -> npt.NDArray:
        """The probability of ``k`` failures or fewer."""
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
        ``1 - alpha_ci / 2`` points."""
        lo, hi = self.quantile(interval_levels(alpha_ci, "two-sided"))
        return float(lo), float(hi)

    def __repr__(self) -> str:
        lo, hi = self.interval(0.1)
        over = f" over {self.units} units" if self.units != 1 else ""
        return (
            f"Predicted failures in the next {self.duration:g}{over}: mean "
            f"{self.mean:.4g} (Monte Carlo error {self.mean_se:.2g}), 90% "
            f"predictive interval {lo:g} to {hi:g}, P(none) "
            f"{float(self.pmf(0)):.3g} ({self.replicates} simulated "
            f"futures under {self.draws} posterior draws)"
        )


class BayesianRenewal(BayesianResult):
    """
    A Bayesian fit of a renewal or imperfect-repair process
    (``GeneralizedRenewal.fit_bayes``, ``ARA.fit_bayes``,
    ``ARI.fit_bayes``): the posterior of the repair parameter and the
    life's (or baseline intensity's) parameters, and what follows from it
    for a repairable system.

    :meth:`repair_mass` gives the posterior probability that the repair
    parameter lies near each end of its range (perfect and minimal
    repair), and the summary states it. :meth:`next_failure_sf` is the
    predictive chance that each unit runs a further time without a
    failure, given its history, the uncertainty in the parameters
    averaged in; :meth:`next_failure_cri` its credible interval and
    :meth:`next_failure_qf` its quantiles. :meth:`count` is the
    predictive distribution of the number of failures in the next period.

    Attributes
    ----------
    params : numpy.ndarray
        The posterior medians, the repair parameter first.
    posterior : Posterior
        The posterior's points and weights.
    restoration_name : str
        The repair parameter's name, ``"q"`` or ``"rho"``.
    end : float
        The latest time any system was observed to.
    method : str
        ``"grid"`` (exact, to the stated integration error) or
        ``"laplace"`` (beyond three free parameters, approximate).
    mle : RenewalModel
        The maximum-likelihood fit to the same data.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.recurrent import ARA
    >>> x = np.array([3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60])
    >>> i = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    >>> c = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    >>> model = ARA.fit_bayes(x, i, c=c, m=2)
    >>> model.method
    'grid'
    >>> lo, hi = model.param_cri("rho")
    >>> bool(0 < lo < hi < 1)
    True
    """

    model_tag = "BayesianRenewal"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.end = self.target.end
        self.restoration_name = self.target.restoration_name
        self.unit = getattr(self.mle, "unit", None)

    @property
    def restoration(self) -> float:
        """The repair parameter's posterior median."""
        return float(self.params[0])

    # -- the repair parameter --------------------
    def _cdf(self, value: float) -> float:
        """The posterior probability that the repair parameter is at or
        below ``value``: the inverse of its quantile function, on a fine
        set of levels."""
        levels = np.linspace(0.0, 1.0, 4001)
        quantiles = np.asarray(
            self.posterior.quantile(self.restoration_name, levels),
            dtype=float,
        )
        quantiles = np.maximum.accumulate(
            np.where(np.isfinite(quantiles), quantiles, np.inf)
        )
        if value < quantiles[0]:
            return 0.0
        if value >= quantiles[-1]:
            return 1.0
        # (the last level whose quantile is at or below the value, then
        # linear between it and the next)
        k = int(np.searchsorted(quantiles, value, side="right")) - 1
        a, b = quantiles[k], quantiles[k + 1]
        share = 0.0 if b <= a else (value - a) / (b - a)
        return float(levels[k] + share * (levels[k + 1] - levels[k]))

    def repair_mass(self, within: float = 0.05) -> dict:
        """
        The posterior probability that the repair parameter lies within
        ``within`` of each of its landmarks: the ends of its range (for
        ARA, ``rho = 1`` is perfect repair and ``rho = 0`` minimal; for
        ARI, ``rho = 0`` is minimal repair and ``rho = 1`` the largest
        reduction), and for a Kijima ``q`` the end ``q = 0`` (perfect
        repair) and the point ``q = 1`` (minimal repair).

        Parameters
        ----------
        within : float, optional
            The distance, on the repair parameter's own scale. Default
            0.05.

        Returns
        -------
        dict
            ``{landmark: probability}``, in words ("perfect repair (as
            good as new)": 0.62, ...).

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import ARA
        >>> x = np.array([3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60])
        >>> i = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        >>> c = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        >>> mass = ARA.fit_bayes(x, i, c=c, m=2).repair_mass(0.1)
        >>> sorted(mass)
        ['minimal repair (as bad as old)', 'perfect repair (as good as new)']
        """
        within = float(within)
        if not within > 0:
            raise ValueError(
                f"within must be a positive distance; got {within!r}."
            )
        out = {}
        for name, value, side in _LANDMARKS[self.target.family_name]:
            if side < 0:
                p = self._cdf(value + within)
            elif side > 0:
                p = 1.0 - self._cdf(value - within)
            else:
                p = self._cdf(value + within) - self._cdf(value - within)
            out[name] = float(min(max(p, 0.0), 1.0))
        return out

    def _repair_lines(self, within: float = 0.05) -> list[str]:
        name = self.restoration_name
        lines = [f"Repair ({name}), posterior probability within {within:g}:"]
        for what, p in self.repair_mass(within).items():
            lines.append(f"  of {what}: {p:.3g}")
        if self.target.family_name == "GeneralizedRenewal":
            below = self._cdf(1.0)
            lines.append(
                f"  that q < 1 (better than as bad as old): {below:.3g}"
            )
        return lines

    def summary(self, alpha_ci: float = 0.05) -> str:
        """The fit in words (see :meth:`BayesianResult.summary`), with
        the posterior probability of the repair parameter's landmarks."""
        return "\n".join([super().summary(alpha_ci), *self._repair_lines()])

    # -- each unit's next failure --------------------
    def _units(self, age: Any) -> tuple[list, npt.NDArray]:
        """The units' failure histories and their times now: the fitted
        items at the end of their histories, or new units that have run
        ``age`` without a failure."""
        if age is None:
            states = self.mle._states()
            return list(states.failures), np.asarray(states.now, dtype=float)
        now = np.atleast_1d(np.asarray(age, dtype=float)).reshape(-1)
        if not np.all(np.isfinite(now)) or np.any(now < 0):
            raise ValueError(
                "age must be finite times of at least 0 (each unit's time "
                "since new, with no failure yet); got "
                f"{now[~(np.isfinite(now) & (now >= 0))][:5]}"
            )
        return [np.zeros(0) for _ in now], now

    def _state_rows(
        self, points: npt.NDArray, failures: list, now: npt.NDArray
    ) -> npt.NDArray:
        """Each unit's state now at every posterior point, ``(P, U)``:
        its virtual age (Kijima, ARA) or the intensity reduction in
        force (ARI), as ``RenewalModel.states`` gives it at one."""
        family, option = self.target.family_name, self.target.option
        r = points[:, :1]
        cols = _columns(points[:, 1:])
        out = np.zeros((len(points), len(failures)))
        with np.errstate(all="ignore"):
            for u, (f, t) in enumerate(zip(failures, now)):
                f = np.asarray(f, dtype=float)
                if f.size == 0:
                    out[:, u] = 0.0 if family == "ARI" else t
                    continue
                since = t - f[-1]
                if family == "GeneralizedRenewal" and option == "i":
                    after = r[:, 0] * f[-1]
                elif family == "GeneralizedRenewal":
                    v = np.zeros(len(points))
                    for g in np.diff(f, prepend=0.0):
                        v = r[:, 0] * (v + g)
                    after = v
                else:
                    terms = f.size if np.isinf(option) else min(option, f.size)
                    j = np.arange(int(terms))
                    weights = (1.0 - r) ** j[None, :]
                    recent = f[::-1][: int(terms)]
                    if family == "ARA":
                        after = f[-1] - r[:, 0] * (weights @ recent)
                    else:
                        lam = np.asarray(
                            self.target.life.iif(recent[None, :], *cols),
                            dtype=float,
                        )
                        out[:, u] = r[:, 0] * np.sum(weights * lam, axis=1)
                        continue
                out[:, u] = after + since
        return out

    def _sf_rows(self, x: npt.NDArray, age: Any) -> npt.NDArray:
        """The next failure's survival function at every posterior point,
        ``(P, U, len(x))``."""
        failures, now = self._units(age)
        state = self._state_rows(self.posterior.points, failures, now)
        x = np.asarray(x, dtype=float).reshape(-1)
        return self._sf_given(state, now, x[None, None, :])

    def _sf_given(
        self, state: npt.NDArray, now: npt.NDArray, x: npt.NDArray
    ) -> npt.NDArray:
        """The survival function of each unit's next failure at ``x``
        ahead (broadcast against ``(P, U, m)``), from its state at every
        posterior point."""
        life = self.target.life
        points = self.posterior.points
        cols = [c[:, :, None] for c in _columns(points[:, 1:])]
        with np.errstate(all="ignore"):
            if self.target.family_name == "ARI":
                t = now[None, :, None]
                drop = (
                    np.asarray(life.cif(t + x, *cols), dtype=float)
                    - np.asarray(life.cif(t + 0.0 * x, *cols), dtype=float)
                    - state[:, :, None] * x
                )
                out = np.exp(-drop)
            else:
                v = state[:, :, None]
                out = np.exp(
                    np.asarray(life.log_sf(v + x, *cols), dtype=float)
                    - np.asarray(life.log_sf(v + 0.0 * x, *cols), dtype=float)
                )
        out = np.where(x == 0, 1.0, out)
        return np.asarray(np.where(np.isnan(out), 0.0, out), dtype=float)

    @staticmethod
    def _ahead(x: npt.ArrayLike) -> npt.NDArray:
        values = np.asarray(x, dtype=float)
        if np.any(values[~np.isnan(values)] < 0):
            raise ValueError(
                "x is time ahead of now, so it cannot be negative; got "
                f"{values[values < 0][:5]}"
            )
        return values

    def next_failure_sf(
        self, x: npt.ArrayLike, age: npt.ArrayLike | None = None
    ) -> npt.NDArray:
        """
        The predictive chance that each unit runs a further ``x`` from
        now without a failure, given its history: the posterior mean of
        ``RenewalModel.next_failure_sf``, so the uncertainty in the
        parameters is averaged in.

        Parameters
        ----------
        x : array like
            Times ahead of now (at least 0).
        age : array like, optional
            Units that have run this long since new without a failure,
            in place of the fitted data's items (each at the end of its
            history).

        Returns
        -------
        numpy.ndarray
            ``(units, len(x))``; ``(units,)`` for a scalar ``x``.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import ARA
        >>> x = np.array([3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60])
        >>> i = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        >>> c = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        >>> model = ARA.fit_bayes(x, i, c=c, m=2)
        >>> model.next_failure_sf([0.0, 5.0]).shape   # two units
        (2, 2)
        """
        values = self._ahead(x)
        flat = np.atleast_1d(values).reshape(-1)
        rows = self._sf_rows(flat, age)
        P, U, X = rows.shape
        out = self.posterior.expectation(rows.reshape(P, U * X)).reshape(U, X)
        out = np.where(np.isnan(flat)[None, :], np.nan, out)
        return out[:, 0] if values.ndim == 0 else out

    def next_failure_cri(
        self,
        x: npt.ArrayLike,
        age: npt.ArrayLike | None = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """
        A credible interval for each unit's next-failure survival
        function at ``x`` ahead of now: the posterior's quantiles of
        ``RenewalModel.next_failure_sf``.

        Returns ``(units, len(x), 2)`` for two-sided (``(units, 2)`` for
        a scalar ``x``), or without the last axis for one side.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import ARA
        >>> x = np.array([3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60])
        >>> i = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        >>> c = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        >>> model = ARA.fit_bayes(x, i, c=c, m=2)
        >>> lo, hi = model.next_failure_cri(5.0)[0]
        >>> bool(0 < lo < float(model.next_failure_sf(5.0)[0]) < hi < 1)
        True
        """
        levels = interval_levels(alpha_ci, bound)
        values = self._ahead(x)
        flat = np.atleast_1d(values).reshape(-1)
        rows = self._sf_rows(flat, age)
        P, U, X = rows.shape
        out = []
        for level in levels:
            q = np.asarray(
                self.posterior.wquantile(rows.reshape(P, U * X), level),
                dtype=float,
            ).reshape(U, X)
            q = np.where(np.isnan(flat)[None, :], np.nan, q)
            out.append(q[:, 0] if values.ndim == 0 else q)
        return np.stack(out, axis=-1) if len(out) == 2 else out[0]

    def next_failure_qf(
        self, p: npt.ArrayLike, age: npt.ArrayLike | None = None
    ) -> npt.NDArray:
        """
        The time ahead of now by which each unit fails with predictive
        probability ``p``: the inverse of :meth:`next_failure_sf`
        (``next_failure_sf(next_failure_qf(p)) == 1 - p``), found by
        bisection to a relative 1e-10.

        Returns ``(units, len(p))``; ``(units,)`` for a scalar ``p``.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import ARA
        >>> x = np.array([3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60])
        >>> i = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        >>> c = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        >>> model = ARA.fit_bayes(x, i, c=c, m=2)
        >>> t = model.next_failure_qf(0.5)
        >>> round(float(model.next_failure_sf(t[0])[0]), 6)
        0.5
        """
        p_arr = np.asarray(p, dtype=float)
        flat = np.atleast_1d(p_arr).reshape(-1)
        if np.any((flat < 0) | (flat > 1)):
            raise ValueError(f"p must be probabilities in [0, 1]; got {p!r}.")
        failures, now = self._units(age)
        U = len(failures)
        state = self._state_rows(self.posterior.points, failures, now)
        weights = self.posterior.weights

        def sf(t: npt.NDArray) -> npt.NDArray:
            # each unit's predictive survival at its own time t[u]
            rows = self._sf_given(state, now, t[None, :, None])[:, :, 0]
            out: npt.NDArray = weights @ rows
            return out

        out = np.zeros((U, flat.size))
        scale = max(self.end, 1e-300)
        for k, level in enumerate(flat):
            if level == 0:
                continue
            if level == 1:
                out[:, k] = np.inf
                continue
            target = 1.0 - level
            lo = np.zeros(U)
            hi = np.full(U, scale)
            for _ in range(200):
                short = sf(hi) > target
                if not short.any():
                    break
                hi = np.where(short, 2.0 * hi, hi)
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                above = sf(mid) > target
                lo = np.where(above, mid, lo)
                hi = np.where(above, hi, mid)
                if np.all(hi - lo <= 1e-10 * hi):
                    break
            out[:, k] = 0.5 * (lo + hi)
        return out[:, 0] if p_arr.ndim == 0 else out

    # -- a new system --------------------
    @keeps_query_shape
    def mcf(
        self,
        x: npt.ArrayLike,
        *,
        draws: int = 200,
        items: int = 20,
        random_state: Any = 0,
    ) -> npt.NDArray:
        """
        The predictive mean cumulative function of a new system: the
        expected number of failures by ``x`` from new, the uncertainty in
        the parameters included, by simulation (``draws`` parameter sets
        from the posterior, ``items`` systems simulated under each, every
        failure repaired by the process's own rule), as
        ``RenewalModel.mcf`` is at one set.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import ARA
        >>> x = np.array([3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60])
        >>> i = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        >>> c = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        >>> model = ARA.fit_bayes(x, i, c=c, m=2)
        >>> curve = model.mcf([20.0, 60.0], draws=50, items=10)
        >>> bool(0 < curve[0] < curve[1])
        True
        """
        from surpyval.utils.rng import as_generator

        x = np.asarray(x, dtype=float)
        for label, v in (("draws", draws), ("items", items)):
            if int(v) != v or int(v) < 1:
                raise ValueError(
                    f"{label} must be a whole number of at least 1; got {v!r}."
                )
        finite = x[np.isfinite(x)]
        out = np.full(x.shape, np.nan)
        out[np.isposinf(x)] = np.inf
        if finite.size == 0:
            return out
        until = float(max(finite.max(), 0.0))
        rng = as_generator(random_state)
        times = []
        if until > 0:
            thetas = self.posterior.draws(int(draws), random_state=rng)
            with ignoring_warnings(), np.errstate(all="ignore"):
                for theta in thetas:
                    model = self.target.with_params(theta)
                    _, _, ahead = model._simulate_from_states(
                        model.states(age=[0.0]), until, int(items), rng
                    )
                    times.append(np.asarray(ahead, dtype=float))
        events = np.sort(np.concatenate(times)) if times else np.zeros(0)
        n = int(draws) * int(items)
        at = np.isfinite(x)
        out[at] = np.searchsorted(events, x[at], side="right") / n
        out[at & (x < 0)] = 0.0
        return out

    # -- failures in the next period --------------------
    def count(
        self,
        duration: float,
        age: npt.ArrayLike | None = None,
        *,
        draws: int = 400,
        items: int = 25,
        random_state: Any = 0,
    ) -> SimulatedCount:
        """
        The predictive distribution of the number of failures in the
        next ``duration``, over every unit (the fitted items from the end
        of their histories, or new units of ``age``): ``draws``
        parameter sets from the posterior, and ``items`` futures of each
        unit simulated under each, each failure repaired at once by the
        process's own rule.

        Parameters
        ----------
        duration : float
            The period's length, from each unit's time now.
        age : array like, optional
            New units that have run this long without a failure, in
            place of the fitted items.
        draws, items : int, optional
            The posterior draws and the futures simulated under each
            (``draws * items`` replicates). Defaults 400 and 25.
        random_state : int or numpy.random.Generator, optional
            The seed (0 by default, so the answer is the same on every
            run).

        Returns
        -------
        SimulatedCount
            ``mean``, ``unit_means``, ``pmf(k)``, ``cdf(k)``,
            ``quantile(p)`` and ``interval(alpha_ci)``.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import ARA
        >>> x = np.array([3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60])
        >>> i = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        >>> c = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        >>> model = ARA.fit_bayes(x, i, c=c, m=2)
        >>> nxt = model.count(20.0, draws=50, items=10)
        >>> nxt.units, bool(nxt.mean > 0)
        (2, True)
        """
        from surpyval.utils.rng import as_generator

        duration = float(duration)
        if not (np.isfinite(duration) and duration > 0):
            raise ValueError(
                f"duration must be a positive, finite time; got {duration!r}."
            )
        for label, v in (("draws", draws), ("items", items)):
            if int(v) != v or int(v) < 1:
                raise ValueError(
                    f"{label} must be a whole number of at least 1; got {v!r}."
                )
        rng = as_generator(random_state)
        failures, now = self._units(age)
        U = len(failures)
        thetas = self.posterior.draws(int(draws), random_state=rng)
        totals = np.zeros((int(draws), int(items)), dtype=int)
        per_unit = np.zeros(U)
        with ignoring_warnings(), np.errstate(all="ignore"):
            for d, theta in enumerate(thetas):
                model = self.target.with_params(theta)
                states = model._state_of(None, failures, now)
                unit, rep, _ = model._simulate_from_states(
                    states, duration, int(items), rng
                )
                totals[d] = np.bincount(rep, minlength=int(items))
                per_unit += np.bincount(unit, minlength=U)
        n = int(draws) * int(items)
        return SimulatedCount(totals, per_unit / n, duration)

    # -- what it does not have --------------------
    def param_cb(self, *args: Any, **kwargs: Any) -> Any:
        """Refuses: a Bayesian fit's interval is :meth:`param_cri`."""
        raise ValueError(
            "A Bayesian fit has credible intervals, not confidence bounds: "
            "use param_cri(name) in place of param_cb (and "
            "next_failure_cri for the next failure)."
        )


# ---------------------------------------------------------------------------
# The engine: a grid sliced along the repair parameter
# ---------------------------------------------------------------------------
#: Starting points along the repair parameter and along each of the
#: life's axes, the most each may grow to, and the most points a lattice
#: may hold.
SLICES, SLICES_MAX = 61, 481
#: The fewest slices, and points across a slice, a lattice may start
#: with (``points=`` smaller is raised to them).
SLICES_MIN, ACROSS_MIN = 21, 7
ACROSS = {1: 41, 2: 15}
ACROSS_MAX = {1: 241, 2: 61}
POINTS_MAX = 300_000
#: The step (in standard deviations of the repair parameter at the peak)
#: of the walk that finds how far the posterior reaches along it.
WALK_STEP = 0.5
#: How far (standard deviations of the repair parameter at the peak) the
#: walk may go: further than the shared grid's 60, as a tail against an
#: end of the range is as long as the prior makes it, not the curvature
#: at the peak.
REPAIR_REACH = 1e4


@dataclass
class SlicedGrid:
    """The result of :func:`sliced_grid`: the points (working
    coordinates), their normalised log weights, the evidence and the
    lattices' layout."""

    u: npt.NDArray
    log_w: npt.NDArray
    log_z: float
    error: float
    shape: tuple
    points_per_dim: list
    refinements: int
    widened: int
    mode: npt.NDArray
    reach_sd: tuple
    across_sd: float


def _wall(
    prior_ok: Any, inside: float, outside: float, y: npt.NDArray
) -> float:
    """Where along the repair parameter the prior ends, between
    ``inside`` (where it has density) and ``outside`` (where it has
    none), by bisection to the double's precision; just inside it. Only
    the prior's support is a wall: where the likelihood is zero depends
    on the life's parameters (an ARI intensity that turns negative for
    some shapes and not others), so it is never taken as one."""
    y = np.asarray(y, dtype=float)

    def finite(s: float) -> bool:
        return bool(prior_ok(s, y))

    a, b = float(inside), float(outside)
    for _ in range(200):
        mid = 0.5 * (a + b)
        if mid in (a, b):
            break
        if finite(mid):
            a = mid
        else:
            b = mid
    return a


def _pchip(x: npt.NDArray, y: npt.NDArray) -> Any:
    from scipy.interpolate import PchipInterpolator

    if len(x) == 1:
        return lambda s: np.repeat(y[:1], np.size(s), axis=0)
    return PchipInterpolator(x, y, axis=0, extrapolate=True)


def sliced_grid(
    work: Any,
    starts: list,
    summary: Any,
    *,
    points: int | None = None,
    tol: float | None = None,
    flat_names: list[str] | None = None,
    prior_ok: Any = None,
    tails: Any = None,
) -> SlicedGrid:
    """
    The posterior of a repair parameter and one or two life parameters on
    a grid laid out slice by slice along the repair parameter.

    In words: along the repair parameter's own (unconstrained) axis the
    posterior can be anything (piled against an end of its range, with a
    long tail there; flat over much of it; skewed), while given the
    repair parameter the life's parameters are close to normal. So:

    1. Find the peak, then walk out from it along the repair parameter in
       steps of half its standard deviation, finding at each step the
       life's conditional peak and spread, until the posterior (its
       conditional peak with the spread's volume) is below ``1e-8`` of
       the peak's, or the prior ends (the density is 0 beyond a uniform
       prior's end), on each side.
    2. Lay a lattice: even along the repair parameter on a stretched
       scale (fine at the peak, coarse in the tails, as the grid's), and
       across each slice on the life's conditional axes, centred on the
       conditional peak (the peaks and spreads between the walk's steps
       interpolated smoothly). The map from the lattice to the
       parameters is a change of variables whose Jacobian is each
       slice's spread, so the sum is exact whatever the interpolation.
       Across a slice the lattice is widened until every face is below
       ``1e-8`` of the peak.
    3. Evaluate a second lattice shifted by half a cell on every axis,
       compare the two's summaries (each parameter's 2.5%, 50% and 97.5%
       points), and refine both until half their difference is within
       ``tol`` of the 95% interval's width, and the probabilities of the
       intervals ``tails`` (pairs of ends on the repair parameter's
       unconstrained scale: the mass near each end of its range) within
       ``tol``; past the budget, refuse with the numbers. Each slice's
       mass is spread evenly over its cell, so an interval's end inside
       a cell takes its share of it.

    ``prior_ok(s, y)`` says whether the prior has density at ``s`` along
    the repair parameter (the life at ``y``): where a slice has no
    density at all and the prior does, the walk stops there without
    taking it as the prior's end.

    An end of the repair parameter's range is at infinity on its
    unconstrained scale (``log q``, ``logit rho``): mass piled against it
    is the long tail the walk follows, to where the density is ``1e-8``
    of the peak's, so the mass within any distance of the end is held.
    """
    from scipy.special import logsumexp

    from surpyval.bayes.engines.grid import TOLERANCE, GridError, find_mode

    mode, cov, _ = find_mode(work, starts)
    d = mode.size
    k = d - 1
    tol = TOLERANCE[d] if tol is None else tol
    walk = _walk(work, mode, float(np.sqrt(cov[0, 0])), flat_names, prior_ok)
    lattice = _Lattices(work, mode, walk, points)
    widened = lattice.widen(flat_names)
    u, _, lw, log_cell = lattice.evaluate(False)

    def summarised(
        u: npt.NDArray, lw: npt.NDArray, shift: bool, across_only: Any
    ) -> tuple:
        w = np.exp(lw - logsumexp(lw))
        out = np.asarray(summary(u, w, lattice.shape(1)), dtype=float)
        mass = lattice.masses(w, tails or [], shift, across_only)
        return out, mass

    def difference(a: tuple, b: tuple) -> float:
        # the quantiles as a share of the 95% width; the probabilities as
        # they are
        found = _difference(a[0], b[0])
        if a[1].size:
            found = max(found, float(np.max(np.abs(a[1] - b[1]))))
        return found

    # 3. the self-check
    refinements = 0
    while True:
        ua, lwa = u, lw
        ub, _, lwb, _ = lattice.evaluate(True)
        sa = summarised(ua, lwa, False, None)
        err = difference(sa, summarised(ub, lwb, True, None))
        if 0.5 * err <= tol:
            break
        # which spacing is short: the lattice shifted along one axis at a
        # time (each axis is refined where its own shift moves the
        # summaries by more than half the tolerance, and the worse one
        # always)
        uc, _, lwc, _ = lattice.evaluate(True, across_only=False)
        ud, _, lwd, _ = lattice.evaluate(True, across_only=True)
        sizes = lattice.refined(
            difference(sa, summarised(uc, lwc, True, False)),
            difference(sa, summarised(ud, lwd, True, True)),
            tol,
        )
        if refinements >= 3 or sizes is None:
            raise GridError(
                "The grid did not reach its accuracy: the summaries of two "
                f"grids of {lattice.ns} slices of "
                f"{' x '.join([str(lattice.mz)] * k)} points, offset by "
                f"half a cell, differ by {err:.2%} of the 95% interval's "
                f"width, an error of about {0.5 * err:.2%} (the target is "
                f"{tol:.2%}). Give the parameters with long tails a prior, "
                "or use method='laplace' for an approximation."
            )
        refinements += 1
        lattice.ns, lattice.mz = sizes
        u, _, lw, log_cell = lattice.evaluate(False)

    l_all = np.concatenate([lwa, lwb])
    return SlicedGrid(
        u=np.vstack([ua, ub]),
        log_w=l_all - logsumexp(l_all),
        log_z=float(logsumexp(l_all) - np.log(2.0) + log_cell),
        error=0.5 * err,
        shape=lattice.shape(2),
        points_per_dim=[lattice.ns] + [lattice.mz] * k,
        refinements=refinements,
        widened=widened,
        mode=mode,
        reach_sd=lattice.reach_sd(),
        across_sd=lattice.across_sd(),
    )


def _difference(sa: npt.NDArray, sb: npt.NDArray) -> float:
    """The largest difference of two lattices' summaries, as a share of
    the 95% interval's width."""
    width = np.abs(0.5 * (sa[2] + sb[2]) - 0.5 * (sa[0] + sb[0]))
    width = np.where(width > 0, width, 1.0)
    rel = np.abs(sa - sb) / width
    rel = np.where(np.isfinite(rel), rel, 0.0)
    return float(rel.max()) if rel.size else 0.0


def _stretched(v: float) -> float:
    from surpyval.bayes.engines.grid import grid_z_of

    return float(grid_z_of(np.asarray(v, dtype=float)))


def _slice_at(work: Any, s: float, y0: npt.NDArray) -> tuple:
    """The life's conditional peak at ``s`` along the repair parameter,
    its spread (a Cholesky factor) and the log of the peak's density
    times the spread's volume."""
    from surpyval.bayes.engines.grid import find_mode

    def f(y: npt.NDArray) -> npt.NDArray:
        y = np.atleast_2d(y)
        return work(np.column_stack([np.full(len(y), s), y]))

    y, c, value = find_mode(f, [y0])
    L = np.linalg.cholesky(c)
    return y, L, value + float(np.sum(np.log(np.diag(L))))


def _slice_from(
    work: Any, s: float, starts: list, spread: npt.NDArray | None = None
) -> Any:
    """:func:`_slice_at` from the first of ``starts`` that has density
    in the slice, or ``None`` where none has. Where none of ``starts``
    has, the points ``spread`` (a Cholesky factor of the life's spread)
    away from the first along each axis, out to 16 of them, are tried:
    the life's support can depend on the repair parameter (an ARI
    reduction that a smaller shape makes larger than the intensity), so
    the last slice's peak can be outside the next slice's support."""
    from surpyval.bayes.engines.grid import GridError

    candidates = list(starts)
    if spread is not None:
        y = np.asarray(starts[0], dtype=float)
        for size in (1.0, 2.0, 4.0, 8.0, 16.0):
            for j in range(spread.shape[1]):
                for sign in (1.0, -1.0):
                    candidates.append(y + sign * size * spread[:, j])
    for y in candidates:
        try:
            return _slice_at(work, s, y)
        except GridError:
            continue
    return None


def _walk(
    work: Any,
    mode: npt.NDArray,
    sd: float,
    flat_names: Any,
    prior_ok: Any = None,
) -> dict:
    """Step 1 of :func:`sliced_grid`: the walk out from the peak along the
    repair parameter, in even steps on the stretched scale (each a few
    standard deviations far out in a tail), to where the posterior is
    ``EDGE`` of the peak's or a prior ends."""
    from surpyval.bayes.engines.grid import (
        EDGE,
        GridError,
        grid_u_of,
        improper_posterior_message,
    )

    centre = float(mode[0])
    y0, L0, psi0 = _slice_at(work, centre, mode[1:])
    knots = {0: (y0, L0)}
    reach, walls = {}, {}
    limit = int(np.ceil(grid_u_of(REPAIR_REACH) / WALK_STEP))
    for side in (-1, 1):
        y, L = y0, L0
        for j in range(1, limit + 2):
            if j > limit:
                raise GridError(improper_posterior_message(flat_names))
            s = centre + side * sd * _stretched(j * WALK_STEP)
            found = _slice_from(work, s, [y, y0], L)
            if found is None:
                if prior_ok is not None and not prior_ok(s, y):
                    # the prior ends (a uniform prior's end): found to the
                    # double's precision, and the lattice ends there
                    inner = centre + side * sd * _stretched(
                        (j - 1) * WALK_STEP
                    )
                    walls[side] = _wall(prior_ok, inner, s, y)
                # (else no life the slice can start from has any density
                # there: the lattice reaches it, and holds no mass past
                # it)
                reach[side] = j
                break
            y, L, psi = found
            knots[side * j] = (y, L)
            if psi - psi0 < np.log(EDGE):
                reach[side] = j
                break
    keys = sorted(knots)
    v_lo, v_hi = -reach[-1] * WALK_STEP, reach[1] * WALK_STEP
    for side, at in walls.items():
        v_at = grid_u_of((at - centre) / sd)
        v_lo, v_hi = (v_at, v_hi) if side < 0 else (v_lo, v_at)
    return {
        "sd": sd,
        "s": np.array(
            [
                centre + np.sign(j) * sd * _stretched(abs(j) * WALK_STEP)
                for j in keys
            ]
        ),
        "y": np.array([knots[j][0] for j in keys]),
        "L": [knots[j][1] for j in keys],
        "walls": walls,
        "v": (v_lo, v_hi),
    }


class _Lattices:
    """Steps 2 and 3 of :func:`sliced_grid`: the lattices, the change of
    variables from them to the parameters, and their sizes."""

    def __init__(
        self, work: Any, mode: npt.NDArray, walk: dict, points: Any
    ) -> None:
        from surpyval.bayes.engines.grid import START_HALF_WIDTH, grid_u_of

        self.work, self.walk = work, walk
        self.centre = float(mode[0])
        self.d = mode.size
        k = self.k = self.d - 1
        # the spread by the log of its diagonal and its lower entries,
        # which interpolate to a valid Cholesky factor
        self.tri = np.tril_indices(k)
        entries = []
        for L in walk["L"]:
            L = L.copy()
            L[np.diag_indices(k)] = np.log(np.diag(L))
            entries.append(L[self.tri])
        self.centre_of = _pchip(walk["s"], walk["y"])
        self.spread_of = _pchip(walk["s"], np.array(entries))
        # ``points``: the starting slices, and across each slice as many
        # in proportion (an odd number, so a lattice keeps its centre)
        # (at least SLICES_MIN slices and ACROSS_MIN across: on coarser
        # lattices the two offset ones can agree by chance)
        self.ns = SLICES if points is None else max(int(points), SLICES_MIN)
        self.ns_max = max(SLICES_MAX, self.ns)
        self.mz = ACROSS[k]
        if points is not None:
            self.mz = max(
                ACROSS_MIN,
                2 * int(round(0.5 * ACROSS[k] * self.ns / SLICES)) + 1,
            )
        self.mz_max = max(ACROSS_MAX[k], self.mz)
        self.across = grid_u_of(START_HALF_WIDTH)

    def shape(self, lattices: int) -> tuple:
        return (lattices, self.ns) + (self.mz,) * self.k

    def factors(self, s: npt.NDArray) -> tuple:
        k = self.k
        entries = np.atleast_2d(self.spread_of(s))
        L = np.zeros((len(s), k, k))
        L[:, self.tri[0], self.tri[1]] = entries
        diag = np.arange(k)
        log_diag = L[:, diag, diag].copy()
        L[:, diag, diag] = np.exp(log_diag)
        return np.atleast_2d(self.centre_of(s)), L, log_diag.sum(axis=1)

    def _axes(self, shift: bool, across_only: Any) -> tuple:
        v_lo, v_hi = self.walk["v"]
        vs, ds = np.linspace(v_lo, v_hi, self.ns, retstep=True)
        vz, dz = np.linspace(-self.across, self.across, self.mz, retstep=True)
        # (across_only: True shifts the lattice across the slices only,
        # False along the repair parameter only)
        if shift and across_only is not True:
            vs = vs + 0.5 * ds
        if shift and across_only is not False:
            vz = vz + 0.5 * dz
        grids = np.meshgrid(*([vz] * self.k), indexing="ij")
        v_across = np.stack([g.ravel() for g in grids], axis=1)
        return vs, ds, v_across, dz

    def evaluate(self, shift: bool, across_only: Any = None) -> tuple:
        """``(points, log density, log weight, log cell)`` of a
        lattice."""
        from surpyval.bayes.engines.grid import STRETCH, grid_z_of

        vs, ds, v_across, dz = self._axes(shift, across_only)
        sd = self.walk["sd"]
        s = self.centre + sd * grid_z_of(vs)
        centre, L, log_det = self.factors(s)
        z = grid_z_of(v_across)
        # (slice, across) -> point
        lifes = centre[:, None, :] + np.einsum("sij,aj->sai", L, z)
        u = np.concatenate(
            [np.repeat(s[:, None, None], len(z), axis=1), lifes], axis=2
        ).reshape(-1, self.d)
        with np.errstate(all="ignore"):
            lp = np.asarray(self.work(u), dtype=float)
        lp = np.where(np.isfinite(lp), lp, -np.inf)
        log_dv = (
            np.log(sd * np.cosh(vs / STRETCH))[:, None]
            + log_det[:, None]
            + np.sum(np.log(np.cosh(v_across / STRETCH)), axis=1)[None, :]
        ).reshape(-1)
        if not (shift and across_only is not True):
            # at a prior's end the density stops: the lattice's slice on
            # it is the end of a trapezoid rule (half its weight), which
            # the shifted lattice's midpoints match to second order
            ends = np.zeros(self.ns)
            ends[0] = np.log(0.5) if -1 in self.walk["walls"] else 0.0
            ends[-1] += np.log(0.5) if 1 in self.walk["walls"] else 0.0
            log_dv = log_dv + np.repeat(ends, len(z))
        log_cell = float(np.log(ds) + self.k * np.log(dz))
        return u, lp, lp + log_dv, log_cell

    def masses(
        self, w: npt.NDArray, intervals: list, shift: bool, across_only: Any
    ) -> npt.NDArray:
        """The posterior mass of each interval ``(a, b)`` of the repair
        parameter's unconstrained scale, each slice's mass spread evenly
        over its cell on the stretched scale (clipped to the lattice's
        range)."""
        from surpyval.bayes.engines.grid import grid_u_of

        if not intervals:
            return np.zeros(0)
        vs, ds, _, _ = self._axes(shift, across_only)
        per_slice = np.asarray(w).reshape(self.ns, -1).sum(axis=1)
        v_lo, v_hi = self.walk["v"]
        left = np.clip(vs - 0.5 * ds, v_lo, v_hi)
        right = np.clip(vs + 0.5 * ds, v_lo, v_hi)
        width = np.where(right > left, right - left, 1.0)
        sd = self.walk["sd"]
        out = []
        for a, b in intervals:
            va = -np.inf if a == -np.inf else grid_u_of((a - self.centre) / sd)
            vb = np.inf if b == np.inf else grid_u_of((b - self.centre) / sd)
            overlap = np.clip(
                np.minimum(right, vb) - np.maximum(left, va), 0.0, None
            )
            out.append(float(np.sum(per_slice * overlap / width)))
        return np.asarray(out)

    def widen(self, flat_names: Any) -> int:
        """Step 2: widen across the slices until every face is below
        ``EDGE`` of the peak; how many times."""
        from surpyval.bayes.engines.grid import (
            EDGE,
            MAX_HALF_WIDTH,
            GridError,
            grid_u_of,
            improper_posterior_message,
        )

        widened = 0
        while True:
            _, lp, _, _ = self.evaluate(False)
            top = lp.max()
            if not np.isfinite(top):
                raise GridError(
                    "The posterior density is zero everywhere near its peak."
                )
            v_across = self._axes(False, None)[2]
            face = np.any(np.abs(v_across) >= self.across - 1e-12, axis=1)
            if not (
                lp.reshape(self.ns, -1)[:, face].max() - top > np.log(EDGE)
            ):
                return widened
            widened += 1
            self.across *= 1.25
            if self.across > grid_u_of(MAX_HALF_WIDTH):
                raise GridError(improper_posterior_message(flat_names))
            self.mz = int(min(self.mz_max, self.mz + 4))

    def refined(self, err_s: float, err_z: float, tol: float) -> Any:
        """The next sizes ``(slices, across)``, growing the axis whose
        own shift moved the summaries most (or the other, where it can
        grow no more); ``None`` where neither can."""
        ns, mz, k = self.ns, self.mz, self.k
        grow_s = 0.5 * err_s > 0.5 * tol or err_s >= err_z
        grow_z = 0.5 * err_z > 0.5 * tol or err_z > err_s
        ns_new = min(self.ns_max, 2 * ns - 1) if grow_s else ns
        mz_new = min(self.mz_max, 2 * mz - 1) if grow_z else mz
        if ns_new * mz_new**k > POINTS_MAX:
            cap = int((POINTS_MAX / ns_new) ** (1.0 / k)) // 2 * 2 - 1
            mz_new = max(mz, cap)
        if ns_new == ns and mz_new == mz:
            ns_new = min(self.ns_max, 2 * ns - 1)
            if ns_new * mz**k > POINTS_MAX:
                ns_new = ns
        if ns_new == ns and mz_new == mz:
            return None
        return ns_new, mz_new

    def reach_sd(self) -> tuple:
        v_lo, v_hi = self.walk["v"]
        return (_stretched(-v_lo), _stretched(v_hi))

    def across_sd(self) -> float:
        return _stretched(self.across)


def _landmark_intervals(family: str, t: Any, within: float) -> list:
    """Each of the family's landmarks' neighbourhoods (``within`` of it,
    as :meth:`BayesianRenewal.repair_mass` takes them) as an interval of
    the repair parameter's unconstrained scale ``t``."""

    def phi(value: float) -> float:
        with np.errstate(all="ignore"):
            return float(t.to_phi(value))

    out = []
    for _, value, side in _LANDMARKS[family]:
        if side < 0:
            out.append((-np.inf, phi(value + within)))
        elif side > 0:
            out.append((phi(value - within), np.inf))
        else:
            out.append((phi(value - within), phi(value + within)))
    return out


def _sliced_fit(target: RenewalTarget, priors: Any, points: Any) -> Any:
    """The posterior by :func:`sliced_grid`, as a :class:`BayesianRenewal`
    (the shared layer's binding, priors, summaries and checks)."""
    import time

    from surpyval.bayes.engines.grid import GridError
    from surpyval.bayes.fitting import check_grid_points
    from surpyval.bayes.posterior import Posterior
    from surpyval.bayes.result import (
        bind,
        finish_result,
        grid_summary_table,
        log_posterior,
        phi_mode,
    )

    started = time.perf_counter()
    points = check_grid_points(points)
    bound, defaults = bind(target, priors)
    target.check(bound)
    target.check_flat(bound)
    log_post = log_posterior(target, bound)
    rp = target.reparam(bound)
    plain_summary = grid_summary_table(target, bound)

    # (the lattice's points mapped back once: the work density and the
    # summary of the same lattice share them)
    last: list = []

    def to_phi(u: npt.NDArray) -> npt.NDArray:
        u = np.atleast_2d(u)
        if rp is None:
            return u
        for held, phi in last:
            if held is u:
                return phi
        out: npt.NDArray = rp.inverse(u)
        if len(u) > 1000:
            last.append((u, out))
            del last[:-2]
        return out

    def work(u: npt.NDArray) -> npt.NDArray:
        u = np.atleast_2d(u)
        with np.errstate(all="ignore"):
            out = log_post(to_phi(u))
            if rp is not None:
                out = out + rp.log_det(u)
        return np.where(np.isnan(out), -np.inf, out)

    def summary(u: npt.NDArray, w: npt.NDArray, shape: tuple) -> Any:
        return plain_summary(to_phi(u), w, shape)

    starts = []
    theta = target.start()
    if theta is not None:
        with np.errstate(all="ignore"):
            starts.append(bound.to_phi(np.asarray(theta, float)[None, :])[0])
    centres = bound.centres()
    if theta is not None and np.isfinite(centres).any():
        starts.append(np.where(np.isnan(centres), starts[0], centres))
    # each with the repair parameter moved towards its lower end too: the
    # support of the others can depend on it (an ARI intensity that a
    # large reduction makes negative), and near 0 every family is
    # defined (minimal repair for ARA and ARI, perfect for a Kijima q)
    t0 = bound.transforms[0]
    for start in list(starts):
        r = float(t0.to_theta(start[0]))
        for value in (0.5 * r, 0.1 * r, 1e-3):
            moved = np.array(start, dtype=float)
            with np.errstate(all="ignore"):
                moved[0] = float(t0.to_phi(value))
            starts.append(moved)
    with np.errstate(all="ignore"):
        starts = [
            s if rp is None else rp.forward(np.asarray(s)[None, :])[0]
            for s in starts
        ]
    starts = [
        s for s in starts if np.all(np.isfinite(s)) and np.isfinite(work(s))[0]
    ]
    if not starts:
        raise ValueError(
            "No starting point: the priors give no density to the "
            "maximum-likelihood fit's parameters "
            f"({', '.join(bound.free_names)}). Check the priors' ranges."
        )

    def prior_ok(s: float, y: npt.NDArray) -> bool:
        with np.errstate(all="ignore"):
            phi = to_phi(np.r_[s, y][None, :])
            return bool(np.isfinite(bound.log_prior(phi))[0])

    tails = _landmark_intervals(target.family_name, bound.transforms[0], 0.05)

    try:
        grid = sliced_grid(
            work,
            starts,
            summary,
            points=points,
            flat_names=bound.improper_names,
            prior_ok=prior_ok,
            tails=tails,
        )
    except GridError as e:
        raise ValueError(str(e)) from None
    free_t = [bound.transforms[i] for i in bound.free_idx]
    post = Posterior(
        bound.names,
        bound.free_names,
        bound.full_theta(to_phi(grid.u)),
        np.exp(grid.log_w),
        "grid",
        transforms=free_t,
        shape=grid.shape,
    )
    diag = {
        "method": "grid",
        "engine": "sliced along the repair parameter",
        "free_parameters": list(bound.free_names),
        "points_per_dim": grid.points_per_dim,
        "grid_points": int(len(grid.log_w)),
        "refinements": grid.refinements,
        "integration_error": grid.error,
        "half_width_sd": float(max(*grid.reach_sd, grid.across_sd)),
        "reach_sd": list(grid.reach_sd),
        "widened": grid.widened,
        "mode": phi_mode(bound, log_post, to_phi(grid.mode), rp),
        "log_evidence": float(grid.log_z) if bound.proper else float("nan"),
        "ok": True,
        "approximate": False,
    }
    model = BayesianRenewal._from_target(
        target, post, bound.spec, defaults, diag
    )
    return finish_result(model, bound, started)


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------
def fit(
    fitter: Any,
    x: Any,
    i: Any = None,
    c: Any = None,
    n: Any = None,
    *,
    priors: Any = None,
    method: str = "auto",
    importance: bool = False,
    points: int | None = None,
    **options: Any,
) -> BayesianRenewal:
    """The Bayesian fit of a renewal process; see the fitters'
    ``fit_bayes``."""
    from surpyval.bayes.result import METHODS, bind
    from surpyval.utils.no_maximum import quiet_maximum_warnings
    from surpyval.utils.warnings import caller_stacklevel, collecting_warnings

    if method not in METHODS:
        raise ValueError(
            f"method must be one of {', '.join(METHODS)}; got {method!r}."
        )
    try:
        with collecting_warnings() as caught:
            with quiet_maximum_warnings(), np.errstate(all="ignore"):
                mle = fitter.fit(x, i, c, n, **options)
    except ValueError as e:
        raise ValueError(
            "fit_bayes starts from the maximum-likelihood fit, which "
            f"refuses these data: {e}"
        ) from None
    for w in caught:
        warnings.warn(
            str(w.message), w.category, stacklevel=caller_stacklevel()
        )
    target = RenewalTarget(mle)
    sliced = False
    if method in ("auto", "grid") and not importance:
        bound, _ = bind(target, priors)
        sliced = (
            target.restoration_name in bound.free_names and 2 <= bound.d <= 3
        )
    model: BayesianRenewal
    if sliced:
        model = _sliced_fit(target, priors, points)
    else:
        model = fit_target(
            target,
            priors,
            method=method,
            importance=importance,
            points=points,
            result=BayesianRenewal,
        )
    _warn_default_edge(model)
    _warn_prior_only(model)
    return model


#: The repair parameter is "mostly the prior's" where its posterior 95%
#: interval is at least this share of its prior's 95% interval's width
#: (both on the parameter's own scale).
PRIOR_ONLY_SHARE = 0.75


def _warn_prior_only(model: BayesianRenewal) -> None:
    """Warn where the data say little about the repair parameter: its
    posterior 95% interval is at least :data:`PRIOR_ONLY_SHARE` of its
    prior's (for a prior on it alone)."""
    from surpyval.bayes.model import fmt, prior_interval_text
    from surpyval.utils.warnings import caller_stacklevel

    name = model.restoration_name
    if name not in model.priors.single:
        return
    bound = model._bound()
    t = bound.transforms[0]
    prior = prior_interval_text(bound, name, t, 0.05)
    if prior is None or not np.all(np.isfinite(prior)):
        return
    lo, hi = (float(v) for v in model.param_cri(name))
    width = float(prior[1] - prior[0])
    if not width > 0 or (hi - lo) < PRIOR_ONLY_SHARE * width:
        return
    life_flat = [n for n in bound.improper_names if n != name]
    if life_flat:
        why = (
            " With no prior on the life ("
            + ", ".join(life_flat)
            + "), the posterior can drift to a life whose hazard hardly "
            "changes with age (a Weibull shape near 1), where the age a "
            "repair leaves makes no difference. Give the life a prior from "
            "comparable lives (priors=surpyval.prior.from_model(life_fit)), "
            f"or {name} a prior from engineering judgement."
        )
    else:
        why = (
            f" More failures a system, or a prior on {name} from "
            "engineering judgement, would say more."
        )
    warnings.warn(
        f"The data say little about the repair: {name}'s posterior 95% "
        f"interval ({fmt(lo)} to {fmt(hi)}) is "
        f"{(hi - lo) / width:.0%} as wide as its prior's ({fmt(prior[0])} "
        f"to {fmt(prior[1])}), so the answer about the repair is mostly "
        f"the prior's.{why}",
        stacklevel=caller_stacklevel(),
    )


def _warn_default_edge(model: BayesianRenewal) -> None:
    """Warn where a Kijima ``q`` given the default prior (uniform to
    :data:`GRP_Q_UPPER`) has its posterior against that prior's upper
    end, which the data did not put there."""
    from surpyval.utils.warnings import caller_stacklevel

    name = model.restoration_name
    if model.target.family_name != "GeneralizedRenewal":
        return
    if name not in model.default_priors:
        return
    above = 1.0 - model._cdf(0.95 * GRP_Q_UPPER)
    if above > 0.01:
        warnings.warn(
            f"The posterior of q has {above:.2g} of its mass above "
            f"{0.95 * GRP_Q_UPPER:g}, against the end of its default prior "
            f"(uniform from 0 to {GRP_Q_UPPER:g}): the data allow repairs "
            "worse than that. Give q a prior over the range you believe "
            "(priors={'q': surpyval.prior.uniform(0, 5)}).",
            stacklevel=caller_stacklevel(),
        )


__all__ = [
    "BayesianRenewal",
    "RenewalTarget",
    "RowsLogLik",
    "SimulatedCount",
]
