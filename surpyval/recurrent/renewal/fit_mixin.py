import numpy as np

from surpyval.univariate.parametric.fitters import bounds_convert


class RenewalFitMixin:
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

    @staticmethod
    def _initial_dist_params(data, dist):
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
    def _bounds_transform(data_x, bounds, param_names):
        """
        Build the (bounded -> unbounded) parameter transforms used by the
        fitters that optimise in an unconstrained space. ``bounds`` are the
        natural-space bounds ``[(restoration bounds), *dist.bounds]`` and
        ``param_names`` are the names of those parameters, restoration first.
        """
        param_map = {name: k for k, name in enumerate(param_names)}
        transform, inv_trans, _, _, _ = bounds_convert(
            data_x, bounds, {}, param_map
        )
        return transform, inv_trans

    @staticmethod
    def _multistart(fit_once, inits, user_init):
        """
        Drive the multi-start fit. ``fit_once(x0) -> OptimizeResult`` runs the
        optimiser from a single natural-space start ``x0``. With no user
        ``init`` every start in ``inits`` is tried and the converged result
        with the lowest objective is returned; a user ``init`` is run once.

        Raises ``ValueError`` with the shared messages when nothing converges.
        """
        if user_init is None:
            results = [res for res in map(fit_once, inits) if res.success]
            if not results:
                raise ValueError(
                    "Could not find a good solution. "
                    + "Try using `init` for better initial guess."
                )
            return results[int(np.argmin([res.fun for res in results]))]

        res = fit_once(user_init)
        if not res.success:
            raise ValueError(
                "Optimization with the provided `init` did not "
                "converge. Try a different initial guess."
            )
        return res

    @staticmethod
    def _attach_inference(model, neg_ll, mle, n_obs, res, data):
        """
        Store the fit artefacts and the attributes
        :class:`LikelihoodInferenceMixin` needs: ``_neg_ll`` (the negative
        log-likelihood in natural parameter space), ``_mle`` (the fitted
        parameters in that space) and ``_n_obs``.
        """
        model.res = res
        model.data = data
        model._neg_ll = neg_ll
        model._mle = np.asarray(mle, dtype=float)
        model._n_obs = int(n_obs)
        return model
