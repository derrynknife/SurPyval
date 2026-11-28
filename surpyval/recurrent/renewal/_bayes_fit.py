"""``fit_bayes`` for the renewal and imperfect-repair fitters
(``GeneralizedRenewal``, ``ARA``, ``ARI``); the computation is in
:mod:`surpyval.recurrent.renewal.bayes`."""

from __future__ import annotations

from typing import Any

from numpy.typing import ArrayLike

from surpyval.utils.dataframe import RecurrentBayesDataFrameMixin
from surpyval.utils.units import takes_unit


class RenewalBayesMixin(RecurrentBayesDataFrameMixin):
    """Gives a renewal fitter ``fit_bayes`` and ``fit_bayes_from_df``."""

    @takes_unit
    def fit_bayes(
        self,
        x: ArrayLike,
        i: ArrayLike | None = None,
        c: ArrayLike | None = None,
        n: ArrayLike | None = None,
        *,
        priors: Any = None,
        method: str = "auto",
        importance: bool = False,
        points: int | None = None,
        **options: Any,
    ) -> Any:
        """
        Fit the process by Bayes' rule: priors on the repair parameter and
        on the life's (for ARI the baseline intensity's) parameters, times
        the likelihood ``fit`` uses.

        In words: each repair leaves a system somewhere between as good as
        new and as bad as old, and the repair parameter says where. A few
        failures a system leave it poorly determined, and often the data
        are consistent with one end of its range (the maximum-likelihood
        estimate is then on that end, with no standard error). The
        posterior says how much belief lies near each end
        (:meth:`~surpyval.recurrent.renewal.bayes.BayesianRenewal.repair_mass`),
        and gives each unit's next failure and the failures to expect in
        the next period, with the uncertainty in the parameters included.

        With a two-parameter life the posterior (three free parameters)
        is computed on the grid, exactly, to its stated integration
        error: not by the Laplace approximation, which cannot represent a
        posterior piled against an end of a range. A life with more
        parameters (four or more free) uses the Laplace approximation,
        with its checks; fix a parameter (``prior.fixed``) to keep the
        grid.

        Parameters
        ----------
        x, i, c, n
            The data, as for ``fit``: event times pooled over systems,
            ``i`` naming each row's system, ``c=1`` rows closing each
            system's observation; or an event-history container.
        priors : dict, joint prior or list, optional
            A dict gives a prior for each parameter it names: the repair
            parameter (``q`` for ``GeneralizedRenewal``, ``rho`` for ARA
            and ARI), from engineering judgement (``surpyval.prior.uniform
            (0, 1)``, a ``surpyval.Beta.from_params([a, b])``,
            ``surpyval.prior.between(low, high)``), and the life's
            parameters. The life's prior from a non-repairable fit of
            comparable lives is ``surpyval.prior.from_model(life_fit)``
            (its estimates and their covariance; ``weight=`` discounts
            it); an earlier fit of the same process is
            ``surpyval.prior.from_fit(earlier)`` or
            ``surpyval.prior.from_model(earlier)``. A list combines them.
            The repair parameter with no prior is uniform over its range
            (a Kijima ``q`` from 0 to 2, from as good as new to twice as
            bad as old), stated in ``default_priors``; the life's
            parameters with no prior are flat on their log scale.
        method : {'auto', 'grid', 'laplace'}, optional
            The engine. ``'auto'`` (the default): the grid for one to
            three free parameters, the Laplace approximation beyond.
        importance : bool, optional
            With the Laplace engine, reweight its draws towards the exact
            posterior.
        points : int, optional
            The grid's starting points per parameter.
        **options
            The model's own options, as for ``fit``: ``dist`` and
            ``kijima`` (GeneralizedRenewal), ``dist`` and ``m`` (ARA),
            ``baseline`` and ``m`` (ARI), and ``tl``.

        Returns
        -------
        surpyval.recurrent.renewal.bayes.BayesianRenewal
            ``param_cri(name)``; ``repair_mass(within)``;
            ``next_failure_sf``, ``next_failure_cri`` and
            ``next_failure_qf`` for each unit; ``count(duration)``, the
            predictive number of failures in the next period;
            ``summary()``, ``compare()``, ``waic()``, ``loo()`` (one
            system is the unit) and ``diagnose()``.

        Examples
        --------
        Two systems observed to 60 hours, with no prior on the repair
        efficiency (uniform from 0 to 1):

        >>> import numpy as np
        >>> from surpyval.recurrent import ARA
        >>> x = np.array([3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60])
        >>> i = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        >>> c = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        >>> model = ARA.fit_bayes(x, i, c=c, m=2)
        >>> model.default_priors
        {'rho': 'Uniform(0, 1)'}
        >>> mass = model.repair_mass(within=0.1)
        >>> bool(0 < mass["perfect repair (as good as new)"] < 1)
        True
        """
        from surpyval.recurrent.renewal.bayes import fit

        return fit(
            self,
            x,
            i,
            c,
            n,
            priors=priors,
            method=method,
            importance=importance,
            points=points,
            **options,
        )
