import numpy as np
from scipy.optimize import brentq, minimize

from surpyval.recurrent.parametric.crow_amsaa import CrowAMSAA
from surpyval.recurrent.renewal.fit_mixin import RenewalFitMixin
from surpyval.utils.fitter import singleton_fitter
from surpyval.utils.recurrent_utils import (
    handle_xicn,
    reject_left_truncation,
    validate_memory,
    validate_renewal_censoring,
)


def ari_reduction(failure_intensities, rho, m):
    """
    Intensity reduction ``R_n`` in force just after the most recent failure for
    the Arithmetic Reduction of Intensity model with memory ``m`` (Doyen &
    Gaudoin, 2004).

    ``failure_intensities`` are the baseline intensities ``lambda_0(T_k)``
    evaluated at the failures so far, ordered oldest to newest. The reduction
    uses the most recent ``min(m, n)`` of them::

        R_n = rho * sum_{j=0}^{min(m, n) - 1} (1 - rho)^j lambda_0(T_{n-j})

    ``m = 1`` keeps only the last failure (ARI1) and ``m = inf`` keeps the full
    history (ARI-infinity); ``rho = 0`` gives ``R_n = 0``, i.e. a plain NHPP.
    """
    n = len(failure_intensities)
    if n == 0:
        return 0.0
    upper = n if np.isinf(m) else min(int(m), n)
    recent = np.asarray(failure_intensities[-upper:])[::-1]
    weights = (1.0 - rho) ** np.arange(upper)
    return rho * np.sum(weights * recent)


@singleton_fitter
class ARI(RenewalFitMixin):
    """
    Arithmetic Reduction of Intensity (ARI) imperfect-repair model of Doyen and
    Gaudoin (2004).

    Where the ARA/Kijima models reduce the *virtual age*, ARI reduces the
    failure *intensity* directly. For a baseline (first-failure) intensity
    ``lambda_0`` the process intensity on the interval following the n-th
    failure is::

        lambda(t) = lambda_0(t) - rho * sum_{j=0}^{min(m,n)-1}
                    (1 - rho)^j lambda_0(T_{n-j})

    so each repair subtracts a fraction ``rho`` of (a memory-weighted sum of)
    the past failure intensities. ``rho = 0`` recovers the plain NHPP defined
    by the baseline intensity. The baseline is any of the recurrent intensity
    models (``CrowAMSAA``, ``Duane``, ``CoxLewis``); ``CrowAMSAA`` (power law)
    is the default.

    There is no closed-form marginal intensity, so the mean cumulative function
    is obtained by simulation (see ``mcf`` and ``plot``).

    Examples
    --------
    >>> from surpyval.recurrent import ARI, CrowAMSAA
    >>> import numpy as np
    >>>
    >>> x = np.array([3, 9, 20, 35, 56, 4, 11, 25, 44, 70])
    >>> i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    >>>
    >>> model = ARI.fit(x, i, m=1, dist=CrowAMSAA)
    """

    @staticmethod
    def _build_sampler(model):
        dist = model.model.dist
        dp = model.model.params
        rho = model.rho
        m = model.m
        history_iif = []
        running = [0.0]
        reduction = [0.0]

        def sample(ui):
            t0 = running[0]
            red = reduction[0]
            energy = -np.log(ui)

            def g(x):
                delta = dist.cif(t0 + x, *dp) - dist.cif(t0, *dp)
                return delta - red * x - energy

            hi = 1.0
            expansions = 0
            while g(hi) < 0 and expansions < 60:
                hi *= 2.0
                expansions += 1
            xi = hi if g(hi) < 0 else brentq(g, 0.0, hi)

            running[0] = t0 + xi
            history_iif.append(dist.iif(running[0], *dp))
            reduction[0] = ari_reduction(history_iif, rho, m)
            return xi

        return sample

    def _make_model(self, baseline_dist, dist_params, rho, m):
        from surpyval.recurrent.renewal.renewal_model import RenewalModel

        model = baseline_dist.from_params(list(dist_params))
        out = RenewalModel(
            model,
            rho,
            "rho",
            "Repair Efficiency",
            "ARI Recurrence",
            self._build_sampler,
            dist_label="Baseline Intensity",
            restoration_bounds=(0, 1),
        )
        out.m = m
        return out

    def create_negll_func(self, data, dist, m):
        _, idx = np.unique(data.i, return_index=True)
        x_by_item = np.split(data.x, idx)[1:]
        c_by_item = np.split(data.c, idx)[1:]

        def negll_func(params):
            rho = params[0]
            dist_params = params[1:]

            ll = 0.0
            for x_item, c_item in zip(x_by_item, c_by_item):
                prev = 0.0
                reduction = 0.0
                history_iif = []
                for t, censor in zip(x_item, c_item):
                    # Integral of the (reduced) intensity over (prev, t]; the
                    # reduction is constant across the interval.
                    delta_cif = dist.cif(t, *dist_params) - dist.cif(
                        prev, *dist_params
                    )
                    ll -= delta_cif - reduction * (t - prev)

                    if censor == 0:
                        intensity = dist.iif(t, *dist_params) - reduction
                        if intensity <= 0:
                            return np.inf
                        ll += np.log(intensity)
                        history_iif.append(dist.iif(t, *dist_params))
                        reduction = ari_reduction(history_iif, rho, m)
                    prev = t
            return -ll

        return negll_func

    def fit_from_recurrent_data(self, data, dist=CrowAMSAA, m=1, init=None):
        """
        Fit the ARI model from recurrent data.

        Parameters
        ----------

        data : RecurrentData
            Data containing the recurrence details.
        dist : object, optional
            A recurrent baseline intensity model (``CrowAMSAA``, ``Duane``,
            ``CoxLewis``). Default is ``CrowAMSAA``.
        m : int or float, optional
            Memory of the ARI model; a positive integer or ``numpy.inf``.
            Default is 1.
        init : list, optional
            Initial parameters ``[rho, *dist_params]`` for the optimizer.

        Returns
        -------

        ARI
            A fitted ARI object.
        """
        validate_memory(m)
        validate_renewal_censoring(data.c, type(self).__name__)
        reject_left_truncation(data, type(self).__name__)

        neg_ll = self.create_negll_func(data, dist, m)
        transform, inv_trans = self._bounds_transform(
            data.x, [(0, 1), *dist.bounds], ["rho", *dist.param_names]
        )

        def fit_once(x0):
            return minimize(
                lambda p: neg_ll(inv_trans(p)),
                transform(np.asarray(x0, dtype=float)),
                method="Nelder-Mead",
            )

        if init is None:
            base_params = self._initial_baseline_params(data, dist)
            inits = [[rho_init, *base_params] for rho_init in (0.1, 0.5, 0.9)]
        else:
            inits = None
        res = self._multistart(fit_once, inits, init)

        rho, *dist_params = inv_trans(res.x)
        out = self._make_model(dist, dist_params, rho, m)
        # Only the observed failures (c == 0) contribute an intensity term, so
        # they are the events that enter the BIC sample size.
        self._attach_inference(
            out,
            neg_ll,
            [rho, *dist_params],
            int((data.c == 0).sum()),
            res,
            data,
        )
        return out

    @staticmethod
    def _initial_baseline_params(data, dist):
        """
        Initial parameters for the baseline intensity model: the plain NHPP fit
        of that baseline if it succeeds, otherwise its own parameter
        initialiser. (ARI's baseline is an intensity model, not a lifetime
        distribution, so this differs from the other repair fitters.)
        """
        try:
            base_params = np.asarray(
                dist.fit_from_recurrent_data(data).params, dtype=float
            )
            if not np.all(np.isfinite(base_params)):
                raise ValueError
        except Exception:
            base_params = np.asarray(dist.parameter_initialiser(data.x))
        return base_params

    def fit(self, x, i=None, c=None, n=None, dist=CrowAMSAA, m=1, init=None):
        """
        Fit the ARI model.

        Parameters
        ----------

        x : array_like
            An array of event times.
        i : array_like, optional
            An array of item indices.
        c : array_like, optional
            An array of censoring indicators.
        n : array_like, optional
            An array of counts.
        dist : object, optional
            A recurrent baseline intensity model. Default is ``CrowAMSAA``.
        m : int or float, optional
            Memory of the ARI model; a positive integer or ``numpy.inf``.
            Default is 1.
        init : list, optional
            Initial parameters ``[rho, *dist_params]`` for the optimizer.

        Returns
        -------

        ARI
            A fitted ARI object.
        """
        data = handle_xicn(x, i, c, n, as_recurrent_data=True)
        return self.fit_from_recurrent_data(data, dist, m, init=init)

    def fit_from_parameters(self, dist_params, rho, m=1, dist=CrowAMSAA):
        """
        Build an ARI model from given parameters.

        Parameters
        ----------

        dist_params : list
            Parameters for the baseline intensity model.
        rho : float
            Repair efficiency in ``[0, 1]``.
        m : int or float, optional
            Memory of the ARI model; a positive integer or ``numpy.inf``.
            Default is 1.
        dist : object, optional
            A recurrent baseline intensity model. Default is ``CrowAMSAA``.

        Returns
        -------

        ARI
            An ARI object built from the supplied parameters.
        """
        validate_memory(m)
        return self._make_model(dist, dist_params, rho, m)
