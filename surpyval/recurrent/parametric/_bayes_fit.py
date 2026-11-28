"""``fit_bayes`` for the recurrent-event processes: the method, shared
by the fitters that have it (the computation is in
:mod:`surpyval.recurrent.bayes`)."""

from __future__ import annotations

from typing import Any

from numpy.typing import ArrayLike

from surpyval.utils.dataframe import RecurrentBayesDataFrameMixin
from surpyval.utils.deprecation import ambiguous_item_ids
from surpyval.utils.units import takes_unit


class BayesFitMixin(RecurrentBayesDataFrameMixin):
    """Gives a recurrent-event fitter ``fit_bayes`` and
    ``fit_bayes_from_df``."""

    @ambiguous_item_ids("fit_bayes(x, c, n, t)")
    @takes_unit
    def fit_bayes(
        self,
        x: ArrayLike,
        i: ArrayLike | None = None,
        c: ArrayLike | None = None,
        n: ArrayLike | None = None,
        t: ArrayLike | None = None,
        tl: ArrayLike | None = None,
        tr: ArrayLike | None = None,
        *,
        priors: Any = None,
        method: str = "auto",
        importance: bool = False,
        points: int | None = None,
        windows: dict | None = None,
    ) -> Any:
        """
        Fit the process by Bayes' rule: priors on the intensity's
        parameters times the likelihood ``fit`` uses.

        In words, for a reliability growth test: the failures say how
        the intensity is changing; a prior says what was believed
        before, about the growth rate, about the MTBF reached, or from
        an earlier test phase; the posterior combines them, and gives
        the credible interval of the MTBF now and later, the predictive
        number of failures in the next period, and the probability that
        a required MTBF is met.

        The posterior is exact: in closed form for an HPP with a Gamma
        prior on its rate, and otherwise on a grid for up to three free
        parameters (the Laplace approximation, with its checks, beyond
        three). There is no sampler.

        Parameters
        ----------
        x, i, c, n, t, tl, tr, windows
            The data, as for ``fit``: event times pooled over systems
            (``i`` names each row's system), ``c=1`` rows closing each
            system's observation, counts in windows, truncation.
        priors : dict, joint prior or list, optional
            A dict gives a prior for each parameter it names (``beta``,
            ``alpha``; an HPP's ``lambda``): ``surpyval.prior.around(
            value, between=(low, high))``, ``surpyval.prior.between(low,
            high)``, a SurPyval distribution such as
            ``surpyval.Gamma.from_params([2, 100])``, or
            ``surpyval.prior.fixed(value)``. A belief about the MTBF or
            the intensity at a time is
            ``surpyval.prior.on_quantities(fitter, mtbf_at={T: ...},
            instead_of="alpha")`` (``intensity_at`` and ``cif_at``
            likewise; ``surpyval.prior.on_function`` for any other
            function of the model). An earlier test phase is
            ``surpyval.prior.from_fit(earlier, weight=...)`` (its data,
            discounted by ``weight``) or
            ``surpyval.prior.from_model(earlier)`` (its estimates and
            their covariance). A list combines them. A parameter with no
            prior is flat on its log scale, allowed where the data
            determine it.
        method : {'auto', 'conjugate', 'grid', 'laplace'}, optional
            The engine. ``'auto'`` (the default): closed form where
            there is one, the grid for one to three free parameters, the
            Laplace approximation beyond.
        importance : bool, optional
            With the Laplace engine, reweight its draws towards the
            exact posterior. False by default.
        points : int, optional
            The grid's starting points per parameter.

        Returns
        -------
        surpyval.recurrent.bayes.BayesianRecurrence
            ``param_cri(name)``; ``cif``, ``iif``, ``mtbf`` and ``cri(x,
            on=...)``; ``mtbf_cri()`` (now, or projected to a later
            time); ``count(duration=...)``, the predictive number of
            events in a period; ``mtbf_probability(requirement)``;
            ``compare()``, ``summary()``, ``waic()`` and ``loo()`` (one
            system is the unit) and ``diagnose()``.

        Examples
        --------
        One prototype tested to 400 hours, with a belief that the
        programme is growing (``beta`` about 0.7):

        >>> import surpyval as sp
        >>> from surpyval.recurrent import CrowAMSAA
        >>> x = [15, 42, 60, 98, 130, 171, 205, 260, 310, 345, 390, 400]
        >>> c = [0] * 11 + [1]
        >>> model = CrowAMSAA.fit_bayes(
        ...     x, c=c,
        ...     priors={"beta": sp.prior.around(0.7, between=(0.4, 1.2))})
        >>> model.method
        'grid'
        >>> model.param_cri("beta").round(2)
        array([0.46, 1.16])
        >>> model.mtbf_cri().round(0)        # the MTBF now, at 400 hours
        array([ 25., 113.])
        >>> round(model.mtbf_probability(30.0), 2)
        0.92
        >>> model.count(duration=100.0).interval(0.1)
        (0.0, 5.0)
        """
        from surpyval.recurrent.bayes import fit

        return fit(
            self,
            x,
            i,
            c,
            n,
            t,
            tl,
            tr,
            priors=priors,
            method=method,
            importance=importance,
            points=points,
            windows=windows,
        )
