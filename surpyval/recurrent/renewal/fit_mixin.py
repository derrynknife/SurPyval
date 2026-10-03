from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import OptimizeResult, minimize

from surpyval.recurrent._bounded import unconstraining_maps
from surpyval.recurrent.inference import bic_sample_size
from surpyval.univariate.parametric.fitters import (
    bounds_convert,
    verified_maximum,
    verify_or_polish,
)
from surpyval.utils.dataframe import RecurrentDataFrameMixin
from surpyval.utils.fitter_repr import FitterRepr
from surpyval.utils.no_maximum import warn_unverified


class RenewalFitMixin(FitterRepr, RecurrentDataFrameMixin):
    """
    Shared maximum-likelihood scaffolding for the imperfect-repair fitters
    (``GeneralizedRenewal``, ``GeneralizedOneRenewal``, ``ARA``, ``ARI``).

    Each of those fits a leading restoration parameter (``q``/``rho``) together
    with the parameters of an underlying lifetime or intensity model by
    multi-start Nelder-Mead on the negative log-likelihood, then attaches the
    attributes that :class:`LikelihoodInferenceMixin` reads (``_neg_ll``,
    ``_mle``, ``_n_obs``). The genuinely model-specific pieces -- how the
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
    def _polish_unverified(
        neg_ll: Callable, params: np.ndarray, bounds: list, n_obs: float
    ) -> np.ndarray:
        """``params``, the natural parameters a search reached, polished
        where they are not a verified maximum (``verified_maximum``, which
        holds a restoration parameter on its bound out): Nelder-Mead's
        tolerances are absolute, and a G1 renewal fit on twelve failures
        stopped with a scaled gradient of 3e-4. The polish is BFGS on
        central differences (the likelihoods are not written for autograd)
        in the space the searches run in (``unconstraining_maps``), kept
        only where it improves the likelihood; ``_attach_inference`` then
        says whether the answer is a verified maximum."""
        x = np.asarray(params, dtype=float)
        if not np.all(np.isfinite(x)):
            return x
        if verified_maximum(neg_ll, x, bounds, n_obs):
            return x
        to_natural, to_search = unconstraining_maps(list(bounds))

        def fun(u: np.ndarray) -> float:
            with np.errstate(all="ignore"):
                value = neg_ll(to_natural(u))
            return float(value) if np.isfinite(value) else 1e300

        u0 = to_search(x)
        start = OptimizeResult(x=u0, fun=fun(u0))
        with np.errstate(all="ignore"):
            polished, _ = verify_or_polish(fun, start, n_obs, numerical=True)
        if polished.fun < float(neg_ll(x)):
            return np.asarray(to_natural(polished.x), dtype=float)
        return x

    @staticmethod
    def _initial_dist_params(data: Any, dist: Any) -> np.ndarray:
        """
        Initial parameters for the underlying lifetime distribution, fitted to
        the times-to-first-event when there are enough of them (these are
        genuine renewal cycles) and otherwise to the raw interarrival times.
        """
        first_events = data.get_times_to_first_events()
        dist_params = None
        if len(first_events.x) >= 2:
            try:
                dist_params = dist.fit(
                    first_events.x, first_events.c, first_events.n
                ).params
                if np.isnan(dist_params).any():
                    dist_params = None
            except Exception:
                dist_params = None
        if dist_params is None:
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
        fails.
        """
        try:
            params = np.asarray(
                dist.fit(data.interarrival_times, data.c, data.n).params,
                dtype=float,
            )
        except Exception:
            return None
        return params if np.all(np.isfinite(params)) else None

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
            return bool(np.isfinite(neg_ll(np.asarray(x0, dtype=float))))

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
        ``GeneralizedRenewal``: multi-start Nelder-Mead on the negative
        log-likelihood over ``[restoration, *dist params]``, run in the
        unconstrained (bounded-to-unbounded) transform space. Each family
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

        ``GeneralizedOneRenewal`` does not use this: its likelihood only
        needs ``q > -1``, so it optimises directly under box bounds
        rather than in a transform space.
        """
        transform, inv_trans = self._bounds_transform(
            data.x,
            [restoration_bounds, *dist.bounds],
            [restoration_name, *dist.parameter_names],
        )

        def objective(p: np.ndarray) -> float:
            return neg_ll(inv_trans(p))

        def fit_once(x0: np.ndarray) -> Any:
            return minimize(
                objective,
                transform(np.asarray(x0, dtype=float)),
                method="Nelder-Mead",
            )

        def polish(res: Any) -> Any:
            # Restart from where a capped search stopped (``res.x`` is
            # already in the transformed space).
            return minimize(objective, res.x, method="Nelder-Mead")

        inits = None
        if dist_init_params is not None:
            inits = [[r0, *dist_init_params] for r0 in restoration_inits]
            if renewal_restoration is not None:
                renewal = self._renewal_dist_params(data, dist)
                if renewal is not None and not np.allclose(
                    renewal, dist_init_params
                ):
                    inits.append([renewal_restoration, *renewal])
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
        res = self._multistart(fit_once, inits, init, neg_ll, polish)
        params = self._polish_unverified(
            neg_ll,
            inv_trans(res.x),
            [restoration_bounds, *dist.bounds],
            max(float(bic_sample_size(data)), 1.0),
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
        # The multi-start Nelder-Mead's answer is accepted only as a
        # verified maximum (principle 13): a restoration parameter on its
        # bound held out where the likelihood is highest there.
        if verified_maximum(
            neg_ll,
            model._mle,
            model._parameter_bounds(),
            max(float(model._n_obs), 1.0),
        ):
            model.maximum = "verified"
        else:
            model.maximum = "unverified"
            warn_unverified("The {} fit".format(model.kind))
        return model
