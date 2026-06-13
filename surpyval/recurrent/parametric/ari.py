import numpy as np
from scipy.optimize import brentq, minimize

from surpyval.recurrent.inference import LikelihoodInferenceMixin
from surpyval.recurrent.parametric.crow import Crow
from surpyval.recurrent.simulation import RecurrenceSimulationMixin
from surpyval.univariate.parametric.fitters import bounds_convert
from surpyval.utils.recurrent_utils import (
    handle_xicn,
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


class ARI(RecurrenceSimulationMixin, LikelihoodInferenceMixin):
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
    models (``Crow``, ``Duane``, ``CoxLewis``); ``Crow`` (power law) is the
    default.

    There is no closed-form marginal intensity, so the mean cumulative function
    is obtained by simulation (see ``mcf`` and ``plot``).

    Examples
    --------
    >>> from surpyval.recurrent import ARI, Crow
    >>> import numpy as np
    >>>
    >>> x = np.array([3, 9, 20, 35, 56, 4, 11, 25, 44, 70])
    >>> i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    >>>
    >>> model = ARI.fit(x, i, m=1, dist=Crow)
    """

    _restoration_param_name = "rho"

    def __init__(self, dist, dist_params, rho, m=1):
        self.dist = dist
        self.dist_params = np.asarray(dist_params, dtype=float)
        self.rho = rho
        self.m = m

    def __repr__(self):
        out = (
            "ARI Recurrence SurPyval Model"
            + "\n============================="
            + f"\nBaseline Intensity  : {self.dist.name}"
            + "\nFitted by           : MLE"
            + f"\nMemory (m)          : {self.m}"
            + f"\nRepair Efficiency   : {self.rho}"
        )

        param_string = "\n".join(
            [
                "{:>10}".format(name) + ": " + str(p)
                for p, name in zip(self.dist.param_names, self.dist_params)
            ]
        )

        return (
            out
            + "\nParameters          :\n"
            + "{params}".format(params=param_string)
        )

    @property
    def parameter_names(self):
        self._check_fitted()
        return [self._restoration_param_name, *self.dist.param_names]

    @staticmethod
    def _validate_memory(m):
        if m == np.inf:
            return
        if not (isinstance(m, (int, np.integer)) and m >= 1):
            raise ValueError(
                "m must be a positive integer or numpy.inf; got {!r}".format(m)
            )

    def _new_sequence_sampler(self):
        dist = self.dist
        dp = self.dist_params
        rho = self.rho
        m = self.m
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

    @classmethod
    def create_negll_func(cls, data, dist, m):
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

    @classmethod
    def fit_from_recurrent_data(cls, data, dist=Crow, m=1, init=None):
        """
        Fit the ARI model from recurrent data.

        Parameters
        ----------

        data : RecurrentData
            Data containing the recurrence details.
        dist : object, optional
            A recurrent baseline intensity model (``Crow``, ``Duane``,
            ``CoxLewis``). Default is ``Crow``.
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
        cls._validate_memory(m)
        validate_renewal_censoring(data.c, cls.__name__)

        if init is None:
            try:
                base_params = dist.fit_from_recurrent_data(data).params
                base_params = np.asarray(base_params, dtype=float)
                if not np.all(np.isfinite(base_params)):
                    raise ValueError
            except Exception:
                base_params = np.asarray(dist.parameter_initialiser(data.x))

        param_map = {
            name: k
            for k, name in enumerate(["rho", *dist.param_names])
        }
        transform, inv_trans, _, _, _ = bounds_convert(
            data.x, [(0, 1), *dist.bounds], {}, param_map
        )

        neg_ll = cls.create_negll_func(data, dist, m)
        neg_ll_unbounded = lambda p: neg_ll(inv_trans(p))  # noqa: E731

        if init is None:
            results = []
            for rho_init in [0.1, 0.5, 0.9]:
                x0 = transform(np.array([rho_init, *base_params]))
                res = minimize(neg_ll_unbounded, x0, method="Nelder-Mead")
                if res.success:
                    results.append(res)
            if not results:
                raise ValueError(
                    "Could not find a good solution. "
                    + "Try using `init` for better initial guess."
                )
            res = results[np.argmin([r.fun for r in results])]
        else:
            x0 = transform(np.array(init))
            res = minimize(neg_ll_unbounded, x0, method="Nelder-Mead")
            if not res.success:
                raise ValueError(
                    "Optimization with the provided `init` did not "
                    "converge. Try a different initial guess."
                )

        rho, *dist_params = inv_trans(res.x)
        out = cls(dist, dist_params, rho, m)
        out.res = res
        out.data = data
        out._neg_ll = neg_ll
        out._mle = np.asarray([rho, *dist_params], dtype=float)
        out._n_obs = int((data.c == 0).sum())
        return out

    @classmethod
    def fit(cls, x, i=None, c=None, n=None, dist=Crow, m=1, init=None):
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
            A recurrent baseline intensity model. Default is ``Crow``.
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
        return cls.fit_from_recurrent_data(data, dist, m, init=init)

    @classmethod
    def fit_from_parameters(cls, dist_params, rho, m=1, dist=Crow):
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
            A recurrent baseline intensity model. Default is ``Crow``.

        Returns
        -------

        ARI
            An ARI object built from the supplied parameters.
        """
        cls._validate_memory(m)
        return cls(dist, dist_params, rho, m)
