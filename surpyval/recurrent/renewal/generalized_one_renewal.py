from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from surpyval import Weibull
from surpyval.recurrent.renewal.fit_mixin import RenewalFitMixin
from surpyval.recurrent.renewal.renewal_model import RenewalModel
from surpyval.utils.fitter import singleton_fitter
from surpyval.utils.recurrent_utils import (
    handle_xicn,
    reject_gapped_observation,
    reject_left_truncation,
    validate_renewal_censoring,
    validate_renewal_times,
    validate_restoration,
)


@singleton_fitter
class GeneralizedOneRenewal(RenewalFitMixin):
    """
    A class to handle the G1 renewal process of Kaminskiy and Krivtsov, in
    which the jth interarrival time is the underlying lifetime distribution
    scaled by ``(1 + q) ** j``.

    Because scaling a random variable by a factor ``cj`` is equivalent to
    evaluating the base distribution on a rescaled time axis
    (``S_j(x) = S0(x / cj)``), the model is well defined for any non-negative
    lifetime distribution and does not need to know which parameter is the
    scale. Distributions whose support includes negative values (e.g. Normal,
    Gumbel) are rejected, as scaled interarrival times would not be guaranteed
    positive.

    This *is* Lam's geometric process, reparameterised. Lam's geometric
    process takes the ``k``-th interarrival time (``k = 1, 2, ...``) as a base
    lifetime scaled so that ``a ** (k - 1) * X_k`` are i.i.d., with ratio
    ``a > 1`` a deteriorating (interarrivals shrinking) system, ``a < 1`` an
    improving one and ``a = 1`` an ordinary renewal process. Writing ``j = k -
    1``, the G1 scaling ``(1 + q) ** j`` matches ``a ** -j``, so the geometric-
    process ratio is ``a = 1 / (1 + q)`` (equivalently ``q = (1 - a) / a``).
    A negative ``q`` therefore corresponds to ``a > 1`` -- deterioration -- and
    a positive ``q`` to reliability growth; fit this model when you want the
    geometric process with a parametric lifetime distribution.

    Since the Generalised One Renewal Process does not have closed form
    solutions for the instantaneous intensity function and the cumulative
    intensity function these values cannot be calculated directly with this
    class. Instead, the model can be used to simulate recurrence data which is
    fitted to a ``NonParametricCounting`` model. This model can then be used
    to calculate the cumulative intensity function.

    Examples
    --------
    >>> from surpyval import Weibull
    >>> from surpyval.recurrent import GeneralizedOneRenewal
    >>> import numpy as np
    >>>
    >>> x = np.array([1, 2, 3, 4, 4.5, 5, 5.5, 5.7, 6])
    >>>
    >>> model = GeneralizedOneRenewal.fit(x, dist=Weibull)
    >>> model
    G1 Renewal SurPyval Model
    =========================
    Distribution        : Weibull
    Fitted by           : MLE
    Restoration Factor  : -0.1730184624683848
    Parameters          :
         alpha: 1.3919045968817332
          beta: 5.008861189641614
    >>>
    >>> np.random.seed(0)
    >>> np_model = model.count_terminated_simulation(len(x), 5000)
    >>> np_model.mcf(np.array([1, 2, 3, 4, 5, 6]))
    array([0.1696    , 1.181     , 2.287     , 3.6694    , 5.58237925,
           8.54474531])
    """

    @staticmethod
    def _build_sampler(model: Any) -> Callable:
        base_params = model.model.params
        q = model.q
        j = 0

        def sample(ui: float) -> float:
            nonlocal j
            # The jth interarrival is the base lifetime scaled by (1 + q) ** j,
            # so its quantiles are the base quantiles multiplied by the same
            # factor.
            cj = (1.0 + q) ** j
            j += 1
            return cj * model.model.dist.qf(ui, *base_params)

        return sample

    def _make_model(self, underlying_model: Any, q: float) -> "RenewalModel":
        return RenewalModel(
            underlying_model,
            q,
            "q",
            "Restoration Factor",
            "G1 Renewal",
            self._build_sampler,
            restoration_bounds=(-1, None),
        )

    def _rescaled_increments(self, model: Any, data: Any) -> np.ndarray:
        """
        Per-interval cumulative-hazard increments (time-rescaling residuals)
        for a fitted G1 renewal model. The ``j``-th interarrival of an item is
        the base lifetime scaled by ``(1 + q)^j``, so on the base time axis its
        residual is ``H(x_j / (1 + q)^j)``. Aligned with ``data`` rows; iid
        Exp(1) over the observed intervals under the fitted model.
        """
        q = model.q
        _, idx = np.unique(data.i, return_index=True)
        interarrival_by_item = np.split(data.get_interarrival_times(), idx)[1:]
        scaled = np.concatenate(
            [
                np.asarray(arr, dtype=float) / (1.0 + q) ** np.arange(len(arr))
                for arr in interarrival_by_item
            ]
        )
        return np.asarray(model.model.Hf(scaled), dtype=float)

    def _refit(self, model: Any, data: Any) -> Any:
        """Refit this model family on ``data`` with the same lifetime
        distribution; used by the Cramer-von Mises bootstrap."""
        return self.fit_from_recurrent_data(data, dist=model.model.dist)

    def create_negll_func(
        self,
        x: np.ndarray,
        i: np.ndarray,
        c: np.ndarray,
        n: np.ndarray,
        dist: Any,
    ) -> Callable:
        def negll_func(params: np.ndarray) -> float:
            q = params[0]
            dist_params = params[1:]
            # Nelder-Mead's box bounds clip trial points onto the closed
            # bounds, so the optimiser does evaluate q = -1 (every scale
            # (1 + q) ** j with j > 0 is zero) and distribution parameters
            # at their limits (a Weibull alpha of 0). The likelihood is zero
            # there, so say so with inf rather than dividing by zero.
            if not q > -1 or _outside_open_bounds(dist_params, dist.bounds):
                return np.inf
            # log((1 + q) ** j) in log space, so a q near -1 does not
            # underflow the scale to zero.
            log1p_q = np.log1p(q)

            ll = 0.0
            # Far from the optimum the rescaled times can still overflow
            # (x / c_j -> inf) and the densities underflow to zero. That
            # only happens where the likelihood is negligible, and a
            # non-finite total is returned as inf below, so the arithmetic
            # warnings on the way there carry no information.
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                for item in set(i):
                    mask_item = i == item
                    x_item = np.atleast_1d(x[mask_item])
                    c_item = np.atleast_1d(c[mask_item])
                    n_item = np.atleast_1d(n[mask_item])
                    for j in range(0, len(x_item)):
                        # The jth interarrival is the base lifetime scaled
                        # by cj = (1 + q) ** j. Scaling the random variable
                        # by cj is equivalent to evaluating the base
                        # distribution on a rescaled time axis:
                        # f_j(x) = f0(x / cj) / cj and S_j(x) = S0(x / cj).
                        log_cj = j * log1p_q
                        xj = x_item[j] * np.exp(-log_cj)
                        if c_item[j] == 0:
                            ll += n_item[j] * (
                                dist.log_df(xj, *dist_params) - log_cj
                            )
                        elif c_item[j] == 1:
                            ll += n_item[j] * dist.log_sf(xj, *dist_params)
            if not np.isfinite(ll):
                return np.inf
            return -ll

        return negll_func

    @staticmethod
    def _check_dist_eligible(dist: Any) -> None:
        """
        The G1 renewal process scales interarrival times by ``(1 + q) ** j``.
        For the scaled times to remain valid the base distribution must be a
        non-negative lifetime distribution; distributions with support over
        negative values (e.g. Normal, Gumbel) are not eligible.
        """
        if dist.support[0] < 0:
            raise ValueError(
                "{} has support {} which includes negative values; the G1 "
                "renewal process requires a non-negative lifetime "
                "distribution (e.g. Weibull, Exponential, Gamma, "
                "LogNormal).".format(dist.name, dist.support)
            )

    def fit_from_recurrent_data(
        self,
        data: Any,
        dist: Any = Weibull,
        init: "ArrayLike | None" = None,
    ) -> "RenewalModel":
        """
        Fit the generalized renewal model from recurrent data.

        Parameters
        ----------

        data : RecurrentEventData
            Data containing the recurrence details.
        dist : Distribution, optional
            A surpyval distribution object. Default is Weibull.
        init : list, optional
            Initial parameters for the optimization algorithm.

        Returns
        -------

        RenewalModel
            A fitted renewal model.

        Example
        -------

        >>> from surpyval import handle_xicn
        >>> from surpyval.recurrent import GeneralizedOneRenewal
        >>> import numpy as np
        >>>
        >>> x = np.array([1, 3, 6, 9, 10, 1.4, 3, 6.7, 8.9, 11, 1, 2])
        >>> c = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 , 1])
        >>> i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3])
        >>>
        >>> rec_data = handle_xicn(x, i, c)
        >>>
        >>> model = GeneralizedOneRenewal.fit_from_recurrent_data(rec_data)
        >>> model
        G1 Renewal SurPyval Model
        =========================
        Distribution        : Weibull
        Fitted by           : MLE
        Restoration Factor  : 0.3402789091696592
        Parameters          :
             alpha: 1.4115217370254167
              beta: 3.5499343659245564
        """
        self._check_dist_eligible(dist)
        validate_renewal_censoring(data.c, type(self).__name__)
        reject_left_truncation(data, type(self).__name__)
        reject_gapped_observation(data, type(self).__name__)
        validate_renewal_times(
            data, dist, type(self).__name__, every_gap_from_new=True
        )

        neg_ll = self.create_negll_func(
            data.interarrival_times, data.i, data.c, data.n, dist
        )

        # The G1 likelihood only needs ``q > -1``, so it is optimised directly
        # under simple box bounds rather than an unconstrained transform.
        # result is sensitive to the initial value of q.
        def fit_once(x0: np.ndarray) -> Any:
            return minimize(
                neg_ll,
                np.asarray(x0, dtype=float),
                bounds=[(-1, None), *dist.bounds],
                method="Nelder-Mead",
            )

        def polish(res: Any) -> Any:
            return fit_once(res.x)

        if init is None:
            dist_params = dist.fit(
                data.interarrival_times, data.c, data.n
            ).params
            inits = [[q_init, *dist_params] for q_init in (0.0001, 1.0, 2.0)]
        else:
            init = np.atleast_1d(np.asarray(init, dtype=float))
            if init.shape != (1 + len(dist.param_names),):
                raise ValueError(
                    "init must have {} values ([q, {}]); got {}.".format(
                        1 + len(dist.param_names),
                        ", ".join(dist.param_names),
                        init.size,
                    )
                )
            inits = None
        res = self._multistart(fit_once, inits, init, neg_ll, polish)

        underlying_model = dist.from_params(list(res.x[1:]))
        q = res.x[0]
        out = self._make_model(underlying_model, q)
        self._attach_inference(out, neg_ll, res.x, res, data)
        return out

    def fit(
        self,
        x: ArrayLike,
        i: "ArrayLike | None" = None,
        c: "ArrayLike | None" = None,
        n: "ArrayLike | None" = None,
        dist: Any = Weibull,
        init: "ArrayLike | None" = None,
    ) -> "RenewalModel":
        """
        Fit the generalized renewal model.

        Parameters
        ----------

        x : array_like
            The event times, pooled over items (each row belongs to the item
            named in ``i``), measured from the start of each item's life.
        i : array_like, optional
            Identity of the item each row belongs to. Defaults to all rows
            belonging to one item.
        c : array_like, optional
            Censoring indicators: 0 an observed failure, 1 the
            right-censored end of an item's observation. Other codes raise
            a ``ValueError``. Defaults to all observed.
        n : array_like, optional
            Count of events at each row. Defaults to 1.
        dist : object, optional
            A surpyval distribution object. Default is Weibull.
        init : list, optional
            Initial parameters for the optimization algorithm.

        Returns
        -------

        RenewalModel
            A fitted renewal model.

        Example
        -------

        >>> from surpyval.recurrent import GeneralizedOneRenewal
        >>> import numpy as np
        >>>
        >>> x = np.array([1, 3, 6, 9, 10, 1.4, 3, 6.7, 8.9, 11, 1, 2])
        >>> c = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 , 1])
        >>> i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3])
        >>>
        >>> model = GeneralizedOneRenewal.fit(x, i, c=c)
        >>> model
        G1 Renewal SurPyval Model
        =========================
        Distribution        : Weibull
        Fitted by           : MLE
        Restoration Factor  : 0.3402789091696592
        Parameters          :
             alpha: 1.4115217370254167
              beta: 3.5499343659245564
        """
        data = handle_xicn(x, i, c, n)
        return self.fit_from_recurrent_data(data, dist=dist, init=init)

    def fit_from_parameters(
        self, params: ArrayLike, q: float, dist: Any = Weibull
    ) -> "RenewalModel":
        """
        Fit the generalized renewal model from given parameters.

        Parameters
        ----------

        params : list
            A list of parameters for the survival analysis distribution.
        q : float
            Restoration factor used in the G1 renewal model.
        dist : object, optional
            A surpyval distribution object. Default is Weibull.

        Returns
        -------

        RenewalModel
            A fitted renewal model.

        Example
        -------

        >>> from surpyval import Weibull
        >>> from surpyval.recurrent import GeneralizedOneRenewal
        >>>
        >>> model = GeneralizedOneRenewal.fit_from_parameters(
        ...     [10, 2],
        ...     0.2,
        ...     dist=Weibull
        ... )
        >>> model
        G1 Renewal SurPyval Model
        =========================
        Distribution        : Weibull
        Fitted by           : given parameters (not fitted)
        Restoration Factor  : 0.2
        Parameters          :
             alpha: 10
              beta: 2
        """
        self._check_dist_eligible(dist)
        validate_restoration(q, "q", (-1, None), open_lower=True)
        model = dist.from_params(params)
        return self._make_model(model, q)


def _outside_open_bounds(params: np.ndarray, bounds: Any) -> bool:
    """Whether any parameter is on or beyond its (open) bound; ``None``
    marks an unbounded side."""
    for p, (lower, upper) in zip(params, bounds):
        if (lower is not None and not p > lower) or (
            upper is not None and not p < upper
        ):
            return True
    return False
