from __future__ import annotations

from typing import Callable

from autograd import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import OptimizeResult, minimize
from scipy.special import gammaln

from surpyval.recurrent._bounded import unconstraining_maps
from surpyval.recurrent._convergence import better_result
from surpyval.recurrent.inference import bic_sample_size
from surpyval.recurrent.parametric.counting_process import IntensityModel
from surpyval.recurrent.parametric.parametric_recurrence import (
    ParametricRecurrenceModel,
)
from surpyval.univariate.parametric.fitters import verify_or_polish
from surpyval.utils.no_maximum import warn_unverified
from surpyval.utils.pickling import Rebuilt
from surpyval.utils.recurrent_event_data import RecurrentEventData
from surpyval.utils.recurrent_utils import handle_xicn, validate_nhpp_data
from surpyval.utils.validation import check_option


def nhpp_log_likelihood(
    cif: Callable,
    log_iif: Callable,
    s: dict,
    eta: "dict | None" = None,
) -> float:
    """The log-likelihood of an NHPP, for the plain and the
    proportional-intensity fitters (#350).

    ``s`` is the data split by ``RecurrentEventData.
    split_for_nhpp_likelihood``, and ``cif`` and ``log_iif`` the baseline's
    cumulative and log instantaneous intensity at the current parameters,
    as functions of time. ``eta`` holds the linear predictor of each piece
    (``"o"``, ``"right"``, ``"left"``, ``"i"`` and ``"close"``) of a
    proportional-intensity model, whose intensity is the baseline's times
    ``exp(eta)``; ``None`` is the plain NHPP, whose observed term keeps its
    own order of operations, so its likelihood is unchanged to the last bit.

    The pieces of a censoring type that is absent are empty arrays, so
    their terms vanish from the sums with no branching, and no log of 0 is
    taken.
    """

    def scaled(delta: np.ndarray, key: str) -> np.ndarray:
        return delta if eta is None else np.exp(eta[key]) * delta

    def counted(n: np.ndarray, delta: np.ndarray, key: str) -> float:
        # A Poisson count n over the window, with mean delta (times
        # exp(eta)).
        if eta is None:
            terms = n * np.log(delta) - delta - gammaln(n + 1)
        else:
            terms = (
                n * eta[key]
                + n * np.log(delta)
                - np.exp(eta[key]) * delta
                - gammaln(n + 1)
            )
        return terms.sum()

    # ll of directly observed
    x_o, x_o_prev = s["x_o"], s["x_o_prev"]
    if eta is None:
        ll = np.sum(log_iif(x_o) + cif(x_o_prev) - cif(x_o))
    else:
        delta_o = cif(x_o_prev) - cif(x_o)
        ll = (log_iif(x_o) + eta["o"] + (np.exp(eta["o"]) * delta_o)).sum()

    # ll of right censored
    ll += np.sum(scaled(cif(s["x_right_prev"]) - cif(s["x_right"]), "right"))

    # ll of left censored: the count over (entry, x]
    delta_left = cif(s["x_left"]) - cif(s["x_left_prev"])
    ll += counted(s["n_left"], delta_left, "left")

    # ll of interval censored
    ll += counted(s["n_i"], cif(s["x_i_r"]) - cif(s["x_i_l"]), "i")

    # extend the integral from each item's last in-window time to its
    # right-truncation time tr (zero when tr is infinite or already
    # coincides with a right-censoring row)
    ll += np.sum(
        scaled(cif(s["x_close_last"]) - cif(s["x_close_tr"]), "close")
    )
    return ll


class NHPPFitter(IntensityModel):

    #: The ``repr`` (#614)
    fitter_kind = "non-homogeneous Poisson process fitter"
    #: Natural-space parameter bounds, set by each concrete intensity model.
    bounds: tuple

    def create_negll_func(self, data: RecurrentEventData) -> Callable:
        s = data.split_for_nhpp_likelihood()

        def negll_func(params: np.ndarray) -> float:
            return -nhpp_log_likelihood(
                lambda x: self.cif(x, *params),
                lambda x: self.log_iif(x, *params),
                s,
            )

        return negll_func

    def _default_start(
        self,
        data: RecurrentEventData,
        x_unique: np.ndarray,
        mcf_hat: np.ndarray,
    ) -> np.ndarray:
        """The start of the least-squares search when no ``init`` is
        given: ``parameter_initialiser`` unless the model has a better one
        from the non-parametric MCF (``mcf_hat`` at ``x_unique``)."""
        return self.parameter_initialiser(data.x)

    def fit_from_recurrent_data(
        self,
        data: RecurrentEventData,
        how: str = "MLE",
        init: "ArrayLike | None" = None,
    ) -> ParametricRecurrenceModel:
        """
        Fit the NHPP model from recurrent data using either Maximum Likelihood
        Estimation (MLE) or Mean Square Error (MSE) methods.

        Parameters
        ----------

        data: RecurrentEventData
            The recurrent event data, as built by ``surpyval.handle_xicn``.
            :meth:`fit` builds one from its arrays and calls this.
        how: str, optional
            Specifies the fitting method to use, either 'MLE' for Maximum
            Likelihood Estimation or 'MSE' for Mean Square Error. Default
            is 'MLE'.
        init: array_like, optional
            Initial parameters for optimization. The default start is
            tried too, and the better fit kept.

        Returns
        -------

        ParametricRecurrenceModel
            An instance of the ParametricRecurrenceModel class containing the
            fitted model, estimated parameters, and other relevant attributes.
        """
        check_option("how", how, ("MLE", "MSE"))
        validate_nhpp_data(data, self)
        x_unqiue, r, d = data.to_xrd()
        mcf_hat = np.cumsum(d / r)
        default_init = self._default_start(data, x_unqiue, mcf_hat)
        if init is None:
            param_init = default_init
        else:
            param_init = np.atleast_1d(np.asarray(init, dtype=float))
            if param_init.shape != (len(self.parameter_names),):
                raise ValueError(
                    "init must have {} values ({}); got {}.".format(
                        len(self.parameter_names),
                        ", ".join(self.parameter_names),
                        param_init.size,
                    )
                )

        # Both searches run on an unconstrained scale: with the bounds
        # given to the optimiser it clipped trial points onto them, and a
        # positive parameter of exactly 0 (Crow-AMSAA's alpha) divided by
        # zero in the intensity.
        to_natural, to_search = unconstraining_maps(list(self.bounds))

        def fun(u: np.ndarray) -> float:
            with np.errstate(all="ignore"):
                value = np.sum(
                    (self.cif(x_unqiue, *to_natural(u)) - mcf_hat) ** 2
                )
            return float(value) if np.isfinite(value) else 1e300

        ll_func = self.create_negll_func(data) if how == "MLE" else None

        def search_ll(u: np.ndarray) -> float:
            assert ll_func is not None
            with np.errstate(all="ignore"):
                value = ll_func(to_natural(u))
            return float(value) if np.isfinite(value) else 1e300

        def search(start: np.ndarray) -> OptimizeResult:
            # The least-squares fit, and for MLE the likelihood searched
            # from it
            res = minimize(fun, to_search(np.asarray(start, dtype=float)))
            if how == "MLE":
                res = minimize(search_ll, res.x, method="Nelder-Mead")
            elif not res.success:
                # BFGS's finite-difference gradient can stop it at the
                # minimum with "precision loss" (Cox-Lewis, whose squared
                # errors span many orders of magnitude around it): finish
                # without a gradient, as the likelihood search does. The
                # simplex keeps its start, so this is never worse.
                res = minimize(fun, res.x, method="Nelder-Mead")
            return res

        res = search(param_init)
        # A start the user gave is followed by the default one, and the
        # better answer kept: from a start far from the optimum the search
        # can stay where it began -- Duane from alpha = 7.8e5, where the
        # intensity overflows -- and that was returned in silence (#429).
        if init is not None:
            res = better_result(res, search(default_init))
        what = "The {} fit".format(getattr(self, "name", "NHPP"))
        maximum = "not applicable"
        if how == "MLE":
            # Nelder-Mead's tolerances are absolute, and its answer is
            # accepted only as a verified maximum: polished where it is
            # not one (the likelihood is in plain numpy, so by central
            # differences), and said otherwise (principle 13).
            verified = False
            if res.fun < 1e300:
                res, verified = verify_or_polish(
                    search_ll, res, bic_sample_size(data), numerical=True
                )
            maximum = "verified" if verified else "unverified"
            if not verified:
                warn_unverified(what)
        elif not (res.success and res.fun < 1e300):
            warn_unverified(what)
        params = to_natural(res.x)

        model = ParametricRecurrenceModel()
        model.mcf_hat = mcf_hat
        model.res = res
        model.params = params
        model.data = data
        model.dist = self
        model.how = how
        model.maximum = maximum
        # The MLE objective is already in natural parameter space, so it serves
        # directly as the likelihood used for AIC/BIC/standard errors. The MSE
        # fit has no likelihood, so leave the inference attributes unset (the
        # inference methods then raise).
        if ll_func is not None:
            # Kept as what it is built from, so the model pickles (#573)
            model._neg_ll = Rebuilt(
                self.create_negll_func, (data,), built=ll_func
            )
            model._mle = np.asarray(params, dtype=float)
            model._n_obs = bic_sample_size(data)
        return model

    def fit(
        self,
        x: ArrayLike,
        i: "ArrayLike | None" = None,
        c: "ArrayLike | None" = None,
        n: "ArrayLike | None" = None,
        t: "ArrayLike | None" = None,
        tl: "ArrayLike | None" = None,
        tr: "ArrayLike | None" = None,
        how: str = "MLE",
        init: "ArrayLike | None" = None,
        windows: "dict | None" = None,
    ) -> ParametricRecurrenceModel:
        """
        Fit the NHPP model from the provided data. This function prepares the
        data to ensure that it is in the correct format for the fitting.

        Parameters
        ----------

        x: array_like
            The event times, pooled over items (each row belongs to the item
            named in ``i``), measured from the start of each item's life.
            The power-law models (``CrowAMSAA``, ``Duane``) are defined for
            times from 0 and need events at positive times; ``CoxLewis``
            also takes negative times inside a negative ``tl`` window.
            Data with no events, or with a single event and no observation
            after it, raise a ``ValueError``.
        i: array_like, optional
            Identity of the item each row belongs to. Defaults to all rows
            belonging to one item.
        c: array_like, optional
            Censoring indicators: 0 an observed event, 1 the right-censored
            end of an item's observation (the time it was last seen), -1
            left-censored and 2 interval-censored counts (with ``n``).
            Defaults to all observed.
        n: array_like, optional
            Number of events in each row: a count on a left- or
            interval-censored row (``c=-1`` or ``c=2``). An exact event
            (``c=0``) and an end-of-observation row (``c=1``) stand for one,
            so ``n > 1`` there is refused: repeat the row for simultaneous
            events. Defaults to 1.
        t: array_like, optional
            (N, 2) array of [left, right] truncation bounds per observation.
        tl: array_like or scalar, optional
            Left truncation (delayed entry) time of each item; the
            observation of each item begins here. A scalar applies to every
            item; an array has one value per row (the same on every row of
            an item).
        tr: array_like or scalar, optional
            Right truncation time of each item, given like ``tl``; the
            observation window closes there, as a ``c=1`` row would close it.
        how: str, optional
            Specifies the fitting method to use, either 'MLE' for Maximum
            Likelihood Estimation or 'MSE' for Mean Square Error (least
            squares between the model's cumulative intensity and the
            non-parametric MCF of the data). Default is 'MLE'; the MLE
            search starts from the MSE fit.
        init: array_like, optional
            Initial parameters for optimization. The default start is
            tried too, and the better fit kept.
        windows: dict, optional
            Gapped (multi-window) observation: a mapping ``{item: [(start,
            end), ...]}`` giving each item's disjoint observation windows,
            with unobserved gaps between them. When given, every row in ``x``
            must be an observed event (``c=0``); the windows supply the
            end-of-window censoring rows. Because event counts over disjoint
            windows are independent for an NHPP, each window is fitted as its
            own observation period. Mutually exclusive with ``t``/``tl``/
            ``tr``.

        Returns
        -------

        ParametricRecurrenceModel
            The fitted model.

        Examples
        --------
        Two systems, each observed to t = 60 (the ``c=1`` rows), whose
        failures become less frequent over time (``beta < 1``):

        >>> from surpyval.recurrent import CrowAMSAA
        >>> x = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60]
        >>> i = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        >>> c = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        >>> model = CrowAMSAA.fit(x, i=i, c=c)
        >>> model.params
        array([7.82428586, 0.73833691])
        >>> model.cif(60)
        np.float64(4.49998940938965)
        """
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
        return self.fit_from_recurrent_data(data, how, init)

    def from_params(self, params: ArrayLike) -> ParametricRecurrenceModel:
        """
        Create a model instance directly from parameters without fitting.

        Parameters
        ----------

        params: array_like
            Parameters to be used directly to create the model.

        Returns
        -------

        ParametricRecurrenceModel
            An instance of the ParametricRecurrenceModel class initialized with
            the provided parameters.

        Examples
        --------
        >>> from surpyval.recurrent import CrowAMSAA
        >>> model = CrowAMSAA.from_params([10, 1.5])
        >>> model.cif([10, 20])
        array([1.        , 2.82842712])
        """
        model = ParametricRecurrenceModel()
        model.params = np.asarray(params)
        model.dist = self
        model.how = "from_params"
        return model
