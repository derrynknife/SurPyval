from __future__ import annotations

import functools
from typing import Any, Callable

import numpy as np
from autograd import hessian, jacobian
from autograd import numpy as anp
from numpy.typing import ArrayLike
from scipy.optimize import root
from scipy.special import gammaln

from surpyval.recurrent.inference import bic_sample_size
from surpyval.recurrent.parametric.counting_process import (
    Boxable,
    CountingProcess,
)
from surpyval.recurrent.parametric.parametric_recurrence import (
    ParametricRecurrenceModel,
)
from surpyval.univariate.parametric.fitters import is_local_minimum
from surpyval.utils.fitter import singleton_fitter
from surpyval.utils.no_maximum import warn_unverified
from surpyval.utils.pickling import Rebuilt
from surpyval.utils.recurrent_event_data import RecurrentEventData
from surpyval.utils.recurrent_utils import handle_xicn, validate_nhpp_data
from surpyval.utils.validation import check_option


def _in_rate_space(neg_ll: Callable, params: ArrayLike) -> Any:
    """``neg_ll``, which takes ``log(rate)``, at the rate ``params``."""
    return neg_ll(np.log(np.asarray(params)))


@singleton_fitter
class HPP(CountingProcess):
    """
    The homogeneous Poisson process: events at the constant rate
    ``lambda``, so :math:`\\Lambda(t) = \\lambda t`. Its maximum-likelihood
    rate is the number of events divided by the total time under
    observation. ``HPP`` is an instance of this class; ``fit`` and
    ``from_params`` return a ``ParametricRecurrenceModel``.

    Examples
    --------

    >>> from surpyval import Exponential
    >>> from surpyval.recurrent import HPP
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> x = Exponential.random(10, 1e-3).cumsum()
    >>> model = HPP.fit(x)
    >>> print(model)
    Parametric Recurrence SurPyval Model
    ==================================
    Process             : Homogeneous Poisson Process
    Fitted by           : MLE
    Parameters          :
        lambda: 0.0023047023327236213
    >>> model.cif([1, 2, 3, 4, 5, 6])
    array([0.0023047 , 0.0046094 , 0.00691411, 0.00921881, 0.01152351,
           0.01382821])
    >>>
    >>> model.iif([1, 2, 3, 4, 5, 6])
    array([0.0023047, 0.0023047, 0.0023047, 0.0023047, 0.0023047, 0.0023047])
    >>>
    >>> model.inv_cif([1, 2, 3, 4, 5, 6])
    array([ 433.89551258,  867.79102516, 1301.68653774, 1735.58205032,
           2169.4775629 , 2603.37307548])
    """

    #: The ``repr`` (#614)
    fitter_kind = "homogeneous Poisson process fitter"

    def __init__(self) -> None:
        self.parameter_names = ["lambda"]
        self.has_scale = True
        self.bounds = ((0, None),)
        # A constant rate is defined at any time, so an item observed from
        # a negative ``tl`` may have events at negative times (the support
        # check in the fit only restricts the power-law models).
        self.support = (-np.inf, np.inf)
        self.name = "Homogeneous Poisson Process"

    # The base contract is variadic (*params); HPP's one parameter
    # is named for clarity, which the checker flags as a narrower
    # override. The runtime call sites all pass positionally.
    def iif(  # type: ignore[override]
        self, x: Boxable, rate: Boxable
    ) -> Boxable:
        """
        Instantaneous intensity function (IIF) or the failure rate of the
        HPP model.

        Parameters
        ----------
        x : array_like
            The values at which IIF is evaluated.
        rate : float
            The rate parameter of the HPP model.

        Returns
        -------
        ndarray
            The IIF values at specified x.
        """
        # NaN at a missing time, like every other intensity (it was the
        # rate there, #382).
        return np.where(np.isnan(x), np.nan, 1.0) * rate

    # The base contract is variadic (*params); HPP's one parameter
    # is named for clarity, which the checker flags as a narrower
    # override. The runtime call sites all pass positionally.
    def log_iif(  # type: ignore[override]
        self, x: Boxable, rate: Boxable
    ) -> Boxable:
        """
        Natural logarithm of the instantaneous intensity function (IIF) of
        the HPP model.

        Parameters
        ----------
        x : array_like
            The values at which log(IIF) is evaluated.
        rate : float
            The rate parameter of the HPP model.

        Returns
        -------
        ndarray
            The log(IIF) values at specified x.
        """
        return np.log(rate) * np.where(np.isnan(x), np.nan, 1.0)

    # The base contract is variadic (*params); HPP's one parameter
    # is named for clarity, which the checker flags as a narrower
    # override. The runtime call sites all pass positionally.
    def cif(  # type: ignore[override]
        self, x: Boxable, rate: Boxable
    ) -> Boxable:
        """
        Cumulative intensity function (CIF) of the HPP model.

        Parameters
        ----------
        x : array_like
            The values at which CIF is evaluated.
        rate : float
            The rate parameter of the HPP model.

        Returns
        -------
        ndarray
            The CIF values at specified x.
        """
        return rate * np.array(x)

    def inv_cif(self, cif: Boxable, rate: Boxable) -> Boxable:
        """
        Inverse of the cumulative intensity function (CIF) of the HPP model.

        Parameters
        ----------
        cif : array_like
            The CIF values to be inverted.
        rate : float
            The rate parameter of the HPP model.

        Returns
        -------
        ndarray
            The inverted CIF values.
        """
        return np.array(cif) / rate

    def create_negll_func(self, data: RecurrentEventData) -> Callable:
        # The pieces of the NHPP likelihood (#350): per censoring type, the
        # times and previous times, and the right-truncation window close.
        s = data.split_for_nhpp_likelihood()

        # The HPP's cumulative intensity is rate * x, so every term of the
        # likelihood is the rate (or its log) times a sum the data fix: the
        # sums are taken once here, and evaluating the likelihood costs a
        # handful of scalar operations however large the data. The terms
        # of an absent censoring type are sums over empty arrays, 0.
        len_observed = len(s["x_o"])
        observed_time = (s["x_o_prev"] - s["x_o"]).sum()
        right_censored_time = (s["x_right_prev"] - s["x_right"]).sum()
        # Right window-close: extend the integral to each item's finite
        # right-truncation time tr, cif(x_last) - cif(tr) = rate * (x_last
        # - tr). Empty / zero when no item carries a finite tr.
        right_truncation_time = (s["x_close_last"] - s["x_close_tr"]).sum()

        # A left-censored count covers the item's window from its entry
        # (its first row, so the previous time is the entry: ``tl``, or
        # the origin 0), not from time 0 whatever ``tl`` says.
        x_left = s["x_left"] - s["x_left_prev"]
        n_left = s["n_left"]
        n_log_x_left_sum = (n_left * np.log(x_left)).sum()
        x_left_sum = x_left.sum()
        n_left_sum = n_left.sum()
        n_l_factorial_sum = gammaln(n_left + 1).sum()

        delta_xi = s["x_i_r"] - s["x_i_l"]
        n_interval = s["n_i"]
        x_interval_sum = delta_xi.sum()
        n_interval_sum = n_interval.sum()
        n_log_x_interval_sum = (n_interval * np.log(delta_xi)).sum()
        n_i_factorial_sum = gammaln(n_interval + 1).sum()

        def negll_func(log_rate: np.ndarray) -> float:
            rate = anp.exp(log_rate)
            ll = len_observed * log_rate + rate * observed_time
            ll += rate * right_censored_time
            ll += rate * right_truncation_time
            ll += (
                log_rate * n_left_sum
                + n_log_x_left_sum
                - rate * x_left_sum
                - n_l_factorial_sum
            )
            ll += (
                log_rate * n_interval_sum
                + n_log_x_interval_sum
                - rate * x_interval_sum
                - n_i_factorial_sum
            )

            return -ll[0]

        return negll_func

    def fit_from_recurrent_data(
        self,
        data: RecurrentEventData,
        how: str = "MLE",
        init: "ArrayLike | None" = None,
    ) -> Any:
        """
        Fits the HPP model to recurrent data and returns the fitted model.

        Parameters
        ----------
        data : RecurrentEventData
            The recurrent event data, as built by ``surpyval.handle_xicn``.
            :meth:`fit` builds one from its arrays and calls this.
        how : str, optional
            Only ``"MLE"``; accepted so the HPP can stand wherever an NHPP
            baseline is fitted (for example ``CauseSpecificNHPP``).
        init : array_like, optional
            Initial parameter values for the optimization.

        Returns
        -------
        ParametricRecurrenceModel
            An object containing the fitted model and related information.
        """
        out = ParametricRecurrenceModel()
        out.dist = self
        out.data = data

        out.bounds = ((0, None),)
        out.support = (-np.inf, np.inf)
        out.name = "Homogeneous Poisson Process"
        check_option(
            "how",
            how,
            ("MLE",),
            "The HPP is fitted by maximum likelihood only.",
        )
        out.how = "MLE"
        validate_nhpp_data(data, self)

        neg_ll = self.create_negll_func(data)
        jac = jacobian(neg_ll)
        hess = hessian(neg_ll)

        if init is None:
            init = [0.0]
        else:
            init = np.atleast_1d(np.asarray(init, dtype=float))
            # The search runs on log(rate), so a start of 0 (or below) was
            # -inf / nan and came back as a rate of 0 with warnings.
            if init.shape != (1,) or not (
                np.isfinite(init[0]) and init[0] > 0
            ):
                raise ValueError(
                    "init must be one positive, finite rate, [rate]; got "
                    "{!r}".format(init.tolist())
                )
            init = np.log(init)

        res = root(jac, init, jac=hess)
        out.res = res
        out.params = np.exp(res.x)
        # The root of the score is accepted as the maximum only where it
        # is one (a zero gradient and a positive curvature, per event).
        n_obs = bic_sample_size(data)
        verified = bool(
            np.all(np.isfinite(res.x))
            and is_local_minimum(
                neg_ll, jac, hess, res.x, obj_scale=max(float(n_obs), 1.0)
            )
        )
        out.maximum = "verified" if verified else "unverified"
        if not verified:
            warn_unverified("The HPP fit")

        # ``neg_ll`` is parameterised by ``log_rate`` for a stable optimiser;
        # expose it in natural (rate) space so the shared likelihood-inference
        # machinery sees ``_neg_ll(_mle)`` with ``_mle`` the fitted rate.
        # (Kept as what it is built from, so the model pickles, #573.)
        out._neg_ll = functools.partial(
            _in_rate_space,
            Rebuilt(self.create_negll_func, (data,), built=neg_ll),
        )
        out._mle = np.asarray(out.params, dtype=float)
        out._n_obs = n_obs

        return out

    def fit(
        self,
        x: ArrayLike,
        i: "ArrayLike | None" = None,
        c: "ArrayLike | None" = None,
        n: "ArrayLike | None" = None,
        t: "ArrayLike | None" = None,
        tl: "ArrayLike | None" = None,
        tr: "ArrayLike | None" = None,
        init: "ArrayLike | None" = None,
        windows: "dict | None" = None,
    ) -> Any:
        """
        Fits the HPP model to the provided data and returns the fitted model.

        Parameters
        ----------
        x : array_like
            The event times, pooled over items (each row belongs to the item
            named in ``i``), measured from the start of each item's life.
        i : array_like, optional
            Identity of the item each row belongs to. Defaults to all rows
            belonging to one item.
        c : array_like, optional
            Censoring indicators: 0 an observed event, 1 the right-censored
            end of an item's observation (the time it was last seen), -1
            left-censored and 2 interval-censored counts (with ``n``).
            Defaults to all observed.
        n : array_like, optional
            Number of events in each row: a count on a left- or
            interval-censored row (``c=-1`` or ``c=2``). An exact event
            (``c=0``) and an end-of-observation row (``c=1``) stand for one,
            so ``n > 1`` there is refused: repeat the row for simultaneous
            events. Defaults to 1.
        t : array_like, optional
            (N, 2) array of [left, right] truncation bounds per observation.
        tl : array_like or scalar, optional
            Left truncation (delayed entry) time of each item: a scalar for
            every item, or one value per row (the same on every row of an
            item).
        tr : array_like or scalar, optional
            Right truncation time of each item, given like ``tl``; the
            observation window closes there, as a ``c=1`` row would close it.
        init : array_like, optional
            Initial parameter estimates for the optimization.
        windows : dict, optional
            Gapped (multi-window) observation: a mapping ``{item: [(start,
            end), ...]}`` giving each item's disjoint observation windows.
            When given, every row in ``x`` must be an observed event (``c=0``);
            the windows supply the end-of-window censoring rows. Mutually
            exclusive with ``t``/``tl``/``tr``.

        Returns
        -------
        ParametricRecurrenceModel
            The fitted model.

        Examples
        --------
        Two systems, each observed to t = 60 (the ``c=1`` rows), with nine
        failures between them: the rate is 9 / 120.

        >>> from surpyval.recurrent import HPP
        >>> x = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60]
        >>> i = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        >>> c = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        >>> model = HPP.fit(x, i=i, c=c)
        >>> model.params
        array([0.075])
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
        return self.fit_from_recurrent_data(data, init=init)

    def from_params(self, params: ArrayLike) -> ParametricRecurrenceModel:
        """
        Create an HPP model from a known rate, without fitting.

        Parameters
        ----------
        params : array_like
            ``[rate]``, the constant event rate.

        Returns
        -------
        ParametricRecurrenceModel
            A model with the given rate, for prediction and simulation.

        Examples
        --------
        >>> from surpyval.recurrent import HPP
        >>> model = HPP.from_params([0.1])
        >>> model.cif([10, 20])
        array([1., 2.])
        """
        params = np.atleast_1d(np.asarray(params, dtype=float))
        if params.shape != (1,) or not params[0] > 0:
            raise ValueError("an HPP takes one positive rate, [rate]")
        model = ParametricRecurrenceModel()
        model.params = params
        model.dist = self
        model.bounds = ((0, None),)
        model.support = (-np.inf, np.inf)
        model.name = "Homogeneous Poisson Process"
        model.how = "from_params"
        return model
