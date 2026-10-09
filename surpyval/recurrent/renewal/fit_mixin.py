import re
import warnings
from typing import Any, Callable

import numpy as np
from autograd.tracer import getval
from numpy.typing import ArrayLike
from scipy.optimize import OptimizeResult

from surpyval.recurrent._bounded import unconstraining_maps
from surpyval.recurrent.inference import bic_sample_size
from surpyval.recurrent.renewal._search import renewal_search
from surpyval.univariate.parametric.fitters import (
    bounds_convert,
    verified_maximum,
    verify_or_polish,
)
from surpyval.utils.dataframe import RecurrentDataFrameMixin
from surpyval.utils.fitter_repr import FitterRepr
from surpyval.utils.no_maximum import (
    quiet_maximum_warnings,
    warn_no_maximum,
    warn_unverified,
)


class RenewalFitMixin(FitterRepr, RecurrentDataFrameMixin):
    """
    Shared maximum-likelihood scaffolding for the imperfect-repair fitters
    (``GeneralizedRenewal``, ``GeneralizedOneRenewal``, ``ARA``, ``ARI``).

    Each of those fits a leading restoration parameter (``q``/``rho``) together
    with the parameters of an underlying lifetime or intensity model by a
    multi-start search on the negative log-likelihood (BFGS on its
    hand-written gradient where it has one, Nelder-Mead otherwise: see
    ``_search``), then attaches the attributes that
    :class:`LikelihoodInferenceMixin` reads (``_neg_ll``, ``_mle``,
    ``_n_obs``). The genuinely model-specific pieces -- how the
    negative log-likelihood is built, whether the search runs in an
    unconstrained transform space, the multi-start values to try, and what
    counts as an observation -- are supplied by the caller. The parts that
    were copy-pasted across all four fitters live here: the multi-start loop,
    the two convergence-failure errors, picking the best start, the bounded-to-
    unbounded parameter transform, and storing the inference attributes.
    """

    #: The ``repr`` (#614)
    fitter_kind = "imperfect repair fitter"

    @staticmethod
    def _warn_if_memoryless(dist: Any, restoration_name: str) -> None:
        """Warn that the restoration parameter cannot be estimated with a
        memoryless (Exponential) life (#665). A virtual-age model's repair
        acts through the age it leaves, and an Exponential's hazard does
        not depend on age, so the likelihood is the same for every value:
        that of a homogeneous Poisson process. The fit reported q = 17
        with an interval from 6e-86 to 5e87, in silence."""
        if getattr(dist, "name", None) != "Exponential":
            return
        import warnings

        from surpyval.utils.warnings import caller_stacklevel

        warnings.warn(
            "{name} cannot be estimated with an Exponential life: its "
            "hazard does not depend on age, so the age a repair leaves "
            "makes no difference, and the likelihood is the same for "
            "every {name} (that of a homogeneous Poisson process). The "
            "reported {name}, its standard error and its bounds are "
            "meaningless; fit the HPP (surpyval.recurrent.HPP) instead, or "
            "use a life whose hazard changes with age (such as "
            "Weibull).".format(name=restoration_name),
            UserWarning,
            stacklevel=caller_stacklevel(),
        )

    @staticmethod
    def _inside_bounds(init: np.ndarray, bounds: list) -> np.ndarray:
        """A user ``init`` with any value on a finite bound of its range
        moved just inside it: the searches run in a space where the bound
        is at infinity, so a start on it (a Kijima ``q`` of 0, which
        perfect repair suggests) failed as "did not converge" (#665)."""
        out = np.array(init, dtype=float)
        for k, (lo, hi) in enumerate(bounds):
            if lo is not None and hi is not None:
                margin = 1e-4 * (hi - lo)
            else:
                margin = 1e-4 * max(1.0, abs(out[k]))
            if lo is not None and out[k] <= lo:
                out[k] = lo + margin
            elif hi is not None and out[k] >= hi:
                out[k] = hi - margin
        return out

    @staticmethod
    def _in_units(neg_ll: Callable, bounds: list) -> "tuple | None":
        """``(neg_ll in units, units)``: the likelihood of the parameters
        divided by their ``units`` (``neg_ll.search_floor``'s, for the
        parameters searched as themselves), where some unit is not 1, for
        the checks and the polish, whose units are 1 (``verified_maximum``
        differences them in steps of 1e-7 of that). A Cox-Lewis ``beta``,
        a rate per unit time, is 1e-4 on data in thousands of hours,
        where such a step is 0.1% of it and its slope was read to 0.3,
        and whether a fit was a verified maximum was a matter of luck
        (#746). ``None`` where every unit is 1."""
        floor = getattr(neg_ll, "search_floor", None)
        if floor is None:
            return None
        units = np.array(
            [
                unit if low is None and high is None else 1.0
                for unit, (low, high) in zip(floor, bounds)
            ],
            dtype=float,
        )
        if np.all(units == 1.0):
            return None

        def in_units(params: Any) -> Any:
            return neg_ll(params * units)

        return in_units, units

    @staticmethod
    def _verified_maximum(
        neg_ll: Callable, params: np.ndarray, bounds: list, n_obs: float
    ) -> bool:
        """``verified_maximum``, in the parameters' units
        (``_in_units``)."""
        scaled = RenewalFitMixin._in_units(neg_ll, bounds)
        if scaled is not None:
            neg_ll, units = scaled
            params = np.asarray(params, dtype=float) / units
        return verified_maximum(neg_ll, params, bounds, n_obs)

    @staticmethod
    def _polish_unverified(
        neg_ll: Callable, params: np.ndarray, bounds: list, n_obs: float
    ) -> np.ndarray:
        """``params``, the natural parameters a search reached, polished
        where they are not a verified maximum (``verified_maximum``, which
        holds a restoration parameter on its bound out): Nelder-Mead's
        tolerances are absolute, and a G1 renewal fit on twelve failures
        stopped with a scaled gradient of 3e-4. The polish is BFGS in the
        space the searches run in (``unconstraining_maps``), kept only
        where it improves the likelihood; ``_attach_inference`` then says
        whether the answer is a verified maximum.

        The likelihoods are written for autograd (#710), so the polish
        has their exact gradient; central differences, which can stop
        short where the likelihood is flat, are the fallback for one that
        autograd cannot differentiate (a baseline intensity written in
        plain numpy)."""
        x = np.asarray(params, dtype=float)
        if not np.all(np.isfinite(x)):
            return x
        scaled = RenewalFitMixin._in_units(neg_ll, bounds)
        if scaled is not None:
            in_units, units = scaled
            return units * RenewalFitMixin._polish_unverified(
                in_units, x / units, bounds, n_obs
            )
        if verified_maximum(neg_ll, x, bounds, n_obs):
            return x
        to_natural, to_search = unconstraining_maps(list(bounds))

        def fun(u: np.ndarray) -> Any:
            with np.errstate(all="ignore"):
                value = neg_ll(to_natural(u))
            return value if np.isfinite(getval(value)) else 1e300

        def plain(u: np.ndarray) -> float:
            return float(fun(u))

        u0 = to_search(x)
        start = OptimizeResult(x=u0, fun=plain(u0))
        with np.errstate(all="ignore"):
            try:
                polished, _ = verify_or_polish(fun, start, n_obs)
            except (TypeError, ValueError, AttributeError):
                polished, _ = verify_or_polish(
                    plain, start, n_obs, numerical=True
                )
        if polished.fun < float(neg_ll(x)):
            return np.asarray(to_natural(polished.x), dtype=float)
        return x

    @staticmethod
    def _initial_dist_params(data: Any, dist: Any) -> np.ndarray:
        """
        Initial parameters for the underlying lifetime distribution, fitted to
        the times-to-first-event when there are enough of them (these are
        genuine renewal cycles) and otherwise to the raw interarrival times.

        The fit is a start, and its warnings that it is not a maximum
        are held back: the renewal fit says what its own search reached
        (#777).
        """
        first_events = data.get_times_to_first_events()
        dist_params = None
        if len(first_events.x) >= 2:
            try:
                with quiet_maximum_warnings():
                    dist_params = dist.fit(
                        first_events.x, first_events.c, first_events.n
                    ).params
                if np.isnan(dist_params).any():
                    dist_params = None
            except Exception:
                dist_params = None
        if dist_params is None:
            with quiet_maximum_warnings():
                dist_params = dist.fit(
                    data.interarrival_times, data.c, data.n
                ).params
        return dist_params

    @staticmethod
    def _default_start(
        start: Callable[[], np.ndarray], init: "ArrayLike | None"
    ) -> "np.ndarray | None":
        """``start()``, the default initial distribution parameters. With
        a user ``init`` they are only the fallback starts (see
        ``_multistart``), so a failure to find them is ``None``, not an
        error."""
        if init is None:
            return start()
        try:
            with np.errstate(all="ignore"):
                params = np.asarray(start(), dtype=float)
        except Exception:
            return None
        return params if np.all(np.isfinite(params)) else None

    @staticmethod
    def _renewal_dist_params(data: Any, dist: Any) -> "np.ndarray | None":
        """
        The distribution fitted to the interarrival times, the MLE of an
        ordinary renewal process (perfect repair), or ``None`` if that fit
        fails. A start, quietly (``_initial_dist_params``).
        """
        try:
            with quiet_maximum_warnings():
                fitted = dist.fit(data.interarrival_times, data.c, data.n)
            params = np.asarray(fitted.params, dtype=float)
        except Exception:
            return None
        return params if np.all(np.isfinite(params)) else None

    @staticmethod
    def _aged_starts(
        data: Any, dist: Any, neg_ll: Callable, restorations: tuple
    ) -> list:
        """Starts ``[r, *lifetime]``, one for each restoration ``r``, with
        the lifetime fitted to the gaps from the virtual ages that ``r``
        leaves (each gap's end age, left truncated at its start age):
        the lifetime that is best for ``r``. ``neg_ll.virtual_ages(r)``
        gives the ages; without it there are none.

        The fallback where no default start has a finite likelihood
        (#777). The default starts share one lifetime, fitted to the gaps
        as from age 0, and a lifetime whose own fit runs off can put all
        its mass below the longest gap: an ExpoWeibull fitted to one
        item's gaps ran off to a power law ending at the longest of them,
        so a gap from any later age had zero likelihood, at every ARA
        start. A start's own fit can run off too; it is a start, and the
        renewal fit says what its own search reached, so its warnings
        are held back."""
        if getattr(neg_ll, "virtual_ages", None) is None:
            return []
        starts = []
        for r in restorations:
            try:
                with quiet_maximum_warnings(), np.errstate(all="ignore"):
                    given = RenewalFitMixin._life_data(data, neg_ll, r)
                    assert given is not None
                    x, c, n, tl = given
                    params = dist.fit(x, c, n, tl=tl).params
            except Exception:
                continue
            if np.all(np.isfinite(params)):
                starts.append([r, *params])
        return starts

    @staticmethod
    def _life_data(data: Any, neg_ll: Callable, r: float) -> "tuple | None":
        """``(x, c, n, tl)``: the times whose likelihood, with the
        restoration held at ``r``, is the fit's (up to a constant) as a
        plain fit of the life to them: each gap's end age left truncated
        at the virtual age it starts from (``neg_ll.virtual_ages``, the
        ARA and Kijima models), or each gap rescaled
        (``neg_ll.scaled_times``, G1). ``None`` for a likelihood with
        neither."""
        ages_at = getattr(neg_ll, "virtual_ages", None)
        if ages_at is not None:
            ages = np.asarray(ages_at(r), dtype=float)
            gap = np.asarray(data.get_interarrival_times(), dtype=float)
            return gap + ages, data.c, data.n, ages
        scaled = getattr(neg_ll, "scaled_times", None)
        if scaled is not None:
            return np.asarray(scaled(r), dtype=float), data.c, data.n, None
        return None

    def _judge_maximum(
        self,
        data: Any,
        dist: Any,
        neg_ll: Callable,
        params: np.ndarray,
        bounds: list,
        n_obs: float,
    ) -> tuple:
        """``(params, maximum, said)``: what the search's answer
        ``params`` is, ``"verified"``, ``"unverified"`` or ``"no finite
        maximum"`` (``MAXIMUM_STATES``), for ``_attach_inference``, with
        what the life's own fit said of a run-off (``said``).

        An answer that is not a verified maximum is on a run-off of the
        life where, with the restoration held at its value, the life
        fitted to the times that leaves (``_life_data``), whose
        likelihood is then the fit's, has no finite maximum (#777): an
        ExpoWeibull life running to its power-law limit, its ``beta`` to
        1e11 and ``mu`` to 1e-11, had the search stop somewhere on the
        ridge, called unverified. The life's own fit has the family's
        no-maximum check, and the answer takes its parameters where they
        are further up the ridge (the restoration as it is)."""
        params = np.asarray(params, dtype=float)
        if self._verified_maximum(neg_ll, params, bounds, n_obs):
            return params, "verified", None
        try:
            given = self._life_data(data, neg_ll, float(params[0]))
        except Exception:
            given = None
        if given is None:
            return params, "unverified", None
        x, c, n, tl = given
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                with np.errstate(all="ignore"):
                    life = dist.fit(x, c, n, tl=tl)
            except Exception:
                return params, "unverified", None
        if getattr(life, "maximum", None) != "no finite maximum":
            return params, "unverified", None
        said = next(
            (
                str(w.message)
                for w in caught
                if str(w.message).startswith("No finite maximum: ")
            ),
            "",
        )
        further = np.array([params[0], *life.params], dtype=float)
        with np.errstate(all="ignore"):
            if np.all(np.isfinite(further)) and neg_ll(further) < neg_ll(
                params
            ):
                params = further
        return params, "no finite maximum", said

    @staticmethod
    def _warn_run_off(model: Any, said: str) -> None:
        """Say that ``model``'s fit has no finite maximum, its life
        running off (``_judge_maximum``), in the life's own words
        ``said`` where it gave them."""
        consequence = (
            "The reported parameters are where the search stopped, and "
            "their standard errors and bounds are meaningless"
        )
        life = model.model.dist.name
        detail = said[len("No finite maximum: ") :].rstrip(".")
        what, _, advice = detail.partition(". " + consequence + "; ")
        if not advice:
            what = "the {} fitted to them has no finite maximum".format(life)
            advice = "compare a fit with another life"
        # The life's values, where its own search stopped, as reported
        for p, v in zip(model.model.dist.parameter_names, model.model.params):
            what = re.sub(
                r"\b{} \([^)]*\)".format(re.escape(p)),
                "{} ({:.4g})".format(p, float(v)),
                what,
            )
        name = model._restoration_param_name
        warn_no_maximum(
            "the {} likelihood keeps increasing as its {} life runs off: "
            "with {} held at the fitted {:.4g}, it is the {} likelihood of "
            "the times that {} gives, and there {}".format(
                model.kind,
                life,
                name,
                float(model.restoration),
                life,
                name,
                what,
            ),
            consequence,
            advice,
        )

    @staticmethod
    def _bounds_transform(
        data_x: np.ndarray, bounds: list, parameter_names: list
    ) -> tuple[Callable, Callable]:
        """
        Build the (bounded -> unbounded) parameter transforms used by the
        fitters that optimise in an unconstrained space. ``bounds`` are the
        natural-space bounds ``[(restoration bounds), *dist.bounds]`` and
        ``parameter_names`` are the names of those parameters, restoration
        first.
        """
        param_map = {name: k for k, name in enumerate(parameter_names)}
        transform, inv_trans, _, _, _ = bounds_convert(
            data_x, bounds, {}, param_map
        )
        return transform, inv_trans

    @staticmethod
    def _finite_at(neg_ll: Callable, x0: Any) -> bool:
        """Whether the likelihood is finite at the start ``x0``."""
        # (A start can overflow: a Kijima-II q of 2 doubles the ages
        # at every failure, and inf - inf warned from here, #630.)
        with np.errstate(all="ignore"):
            value = neg_ll(np.asarray(x0, dtype=float))
        return bool(np.isfinite(value))

    @staticmethod
    def _multistart(
        fit_once: Callable,
        inits: "list | None",
        user_init: "ArrayLike | None",
        neg_ll: "Callable | None" = None,
        polish: "Callable | None" = None,
    ) -> Any:
        """
        Drive the multi-start fit. ``fit_once(x0) -> OptimizeResult`` runs the
        optimiser from a single natural-space start ``x0``. Every start in
        ``inits`` is tried and the result with the lowest (finite)
        objective is returned. A user ``init`` is tried first, and the
        default ``inits`` after it where given: from a start far from the
        maximum the search can stay where it began -- an ARI baseline
        scale of 4e6, 227 below the maximum -- and that used to be
        returned in silence (#429).

        A start that stops at Nelder-Mead's evaluation cap still counts.
        When the maximum is on the boundary of the parameter space (an ARA
        repair efficiency ``rho -> 1``, a Kijima ``q -> 0``) the search runs
        off towards an infinite transformed parameter and never meets the
        convergence test, so the start that found the best likelihood was
        the one discarded, and a worse local optimum was returned as the
        MLE. ``polish(res) -> OptimizeResult`` restarts the search from
        such a result, and the better of the two is kept.

        With ``neg_ll`` (the natural-space negative log-likelihood) the
        starts at which it is not finite are skipped: they lie outside the
        model's support (e.g. an ARI repair efficiency that drives the
        intensity negative), where every vertex of Nelder-Mead's initial
        simplex is typically infinite too. The simplex can then neither
        move nor converge -- it only runs out its iterations, with scipy
        warning about ``inf - inf`` in its convergence test -- and the
        start was going to be discarded as unconverged anyway.

        Raises ``ValueError`` with the shared messages when no start reaches
        a finite likelihood.
        """

        def feasible(x0: Any) -> bool:
            if neg_ll is None:
                return True
            return RenewalFitMixin._finite_at(neg_ll, x0)

        def usable(res: Any) -> bool:
            return bool(np.isfinite(res.fun))

        def polished(res: Any) -> Any:
            if res.success or polish is None:
                return res
            again = polish(res)
            if usable(again) and again.fun <= res.fun:
                return again
            return res

        def best_of(starts: list) -> "Any | None":
            # A start far from the maximum takes the search where the
            # likelihood overflows; that is the searches' business, and
            # their floating-point warnings are not the user's (#429).
            with np.errstate(all="ignore"):
                results = [r for r in map(fit_once, starts) if usable(r)]
                if not results:
                    return None
                best = results[int(np.argmin([r.fun for r in results]))]
                return polished(best)

        if user_init is None:
            assert inits is not None
            found = best_of([x0 for x0 in inits if feasible(x0)])
            if found is None:
                raise ValueError(
                    "Could not find a good solution. "
                    + "Try using `init` for better initial guess."
                )
            return found

        if not feasible(user_init):
            raise ValueError(
                "The provided `init` has zero likelihood (it is outside "
                "the model's support for this data). Try a different "
                "initial guess."
            )
        res = best_of([user_init])
        if res is None:
            raise ValueError(
                "Optimization with the provided `init` did not "
                "converge. Try a different initial guess."
            )
        default = best_of([x0 for x0 in inits or [] if feasible(x0)])
        if default is not None and default.fun < res.fun:
            return default
        return res

    def _fit_restoration_ml(
        self,
        data: Any,
        neg_ll: Callable,
        restoration_bounds: tuple,
        restoration_name: str,
        dist: Any,
        restoration_inits: tuple,
        dist_init_params: "np.ndarray | None",
        init: "ArrayLike | None",
        renewal_restoration: "float | None" = None,
    ) -> tuple[Any, np.ndarray]:
        """
        The transform-space fitting spine shared by ``ARA``, ``ARI`` and
        ``GeneralizedRenewal``: a multi-start search on the negative
        log-likelihood over ``[restoration, *dist params]``, run in the
        unconstrained (bounded-to-unbounded) transform space
        (``renewal_search``: BFGS on the likelihood's hand-written
        gradient, #728, or Nelder-Mead where it has none), with the best
        start's answer carried onto the restoration parameter's bound where
        it stopped next to it. Each family
        supplies its restoration parameter's name, bounds and start grid,
        and the initial distribution parameters (``None`` where they could
        not be found; a user ``init`` is then tried alone). Returns
        ``(res, natural_params)``.

        ``renewal_restoration`` is a restoration value next to the one at
        which the family is an ordinary renewal process (perfect repair:
        ARA ``rho = 1``, Kijima ``q = 0``). One more start is made there,
        with the distribution fitted to the interarrival times -- that
        boundary's own MLE. From the grid's starts, which share the
        first-event fit, the search can settle on a local maximum at the
        other end and miss a perfect-repair maximum altogether.

        ``GeneralizedOneRenewal`` does not use this: its ``q`` has no
        repair bound to settle on, and it runs the same search itself
        (under box bounds, where its likelihood has no hand-written
        gradient).
        """
        bounds = [restoration_bounds, *dist.bounds]
        transform, inv_trans = self._bounds_transform(
            data.x, bounds, [restoration_name, *dist.parameter_names]
        )
        n_obs = max(float(bic_sample_size(data)), 1.0)
        search = renewal_search(neg_ll, bounds, n_obs, inv_trans)

        def fit_once(x0: np.ndarray) -> Any:
            return search.minimize(transform(np.asarray(x0, dtype=float)))

        def polish(res: Any) -> Any:
            # Restart from where a capped search stopped (``res.x`` is
            # already in the transformed space).
            return search.simplex(res.x)

        inits = None
        if dist_init_params is not None:
            inits = [[r0, *dist_init_params] for r0 in restoration_inits]
            # (With fewer than two items the start's lifetime is already
            # this one, fitted to the gaps: not fitted again.)
            if (
                renewal_restoration is not None
                and len(data.get_times_to_first_events().x) >= 2
            ):
                renewal = self._renewal_dist_params(data, dist)
                if renewal is not None and not np.allclose(
                    renewal, dist_init_params
                ):
                    inits.append([renewal_restoration, *renewal])
            if not any(self._finite_at(neg_ll, x0) for x0 in inits):
                # None is feasible: each restoration start with its own
                # lifetime (#777).
                restorations = tuple(restoration_inits)
                if renewal_restoration is not None:
                    restorations += (renewal_restoration,)
                inits += self._aged_starts(data, dist, neg_ll, restorations)
        if init is not None:
            init = np.atleast_1d(np.asarray(init, dtype=float))
            expected = 1 + len(dist.parameter_names)
            if init.shape != (expected,):
                raise ValueError(
                    "init must have {} values ([{}, {}]); got {}.".format(
                        expected,
                        restoration_name,
                        ", ".join(dist.parameter_names),
                        init.size,
                    )
                )
            init = self._inside_bounds(
                init, [restoration_bounds, *dist.bounds]
            )
        res = self._multistart(fit_once, inits, init, neg_ll, polish)
        with np.errstate(all="ignore"):
            res = search.settle(res, restoration_bounds)
        params = self._polish_unverified(
            neg_ll, inv_trans(res.x), bounds, n_obs
        )
        # What the answer is, for ``_attach_inference`` (#777)
        params, res.maximum, res.run_off = self._judge_maximum(
            data, dist, neg_ll, params, bounds, n_obs
        )
        return res, params

    def _attach_inference(
        self,
        model: Any,
        neg_ll: Callable,
        mle: ArrayLike,
        res: Any,
        data: Any,
    ) -> Any:
        """
        Store the fit artefacts and the attributes
        :class:`LikelihoodInferenceMixin` needs: ``_neg_ll`` (the negative
        log-likelihood in natural parameter space), ``_mle`` (the fitted
        parameters in that space) and ``_n_obs`` (BIC's sample size, the
        observed events, counted the same way for every model).
        Also keeps a reference to
        the fitter (``_fitter``) so the fitted model can reuse its
        family-specific rescaled-increment (time-rescaling residual) logic.
        """
        model.res = res
        model.data = data
        model.how = "MLE"
        model._fitter = self
        model._neg_ll = neg_ll
        model._mle = np.asarray(mle, dtype=float)
        model._n_obs = bic_sample_size(data)
        # The multi-start search's answer is accepted only as a
        # verified maximum (principle 13): a restoration parameter on its
        # bound held out where the likelihood is highest there. The
        # fitter may have judged it already (``_judge_maximum``), and
        # found the life running off.
        maximum = res.get("maximum") if isinstance(res, dict) else None
        if maximum is None:
            verified = self._verified_maximum(
                neg_ll,
                model._mle,
                model._parameter_bounds(),
                max(float(model._n_obs), 1.0),
            )
            maximum = "verified" if verified else "unverified"
        model.maximum = maximum
        if maximum == "unverified":
            warn_unverified("The {} fit".format(model.kind))
        elif maximum == "no finite maximum":
            self._warn_run_off(model, res.get("run_off") or "")
        return model
