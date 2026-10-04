from __future__ import annotations

import functools
from typing import Any

import autograd.numpy as np
import numpy.typing as npt

from surpyval.univariate.parametric.discrete_fitter import (
    DiscreteParametricFitter,
)
from surpyval.univariate.parametric.parametric import draw_state
from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    reject_structural_params,
)
from surpyval.utils.autograd_gamma_compat import betainccln, betaincln

from ..parametric import Parametric
from ._discrete_tails import refine_quantile
from ._single_probability import event_counts, probability_bounds

_NO_SINGLE_N = (
    "This Binomial model was fitted to rows with different numbers of "
    "trials, so it has no single n (it is nan) and no distribution of the "
    "count of events; model.with_params([n, p]) is the model of n trials "
    "of your choice. Its p, param_cb('p') and to_dict() need no n."
)


def _needs_one_n(fn: Any, position: int) -> Any:
    """Refuse, with :data:`_NO_SINGLE_N`, a call whose number of trials
    ``n`` (the ``position``-th argument after ``self``) is ``nan``: a
    model fitted to rows of different sizes (#608). The formulas gave
    ``nan``, or a wrong number (the hazard 0), in silence."""

    @functools.wraps(fn)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        n = args[position] if len(args) > position else kwargs.get("n")
        if n is not None and np.any(np.isnan(np.asarray(n, dtype=float))):
            raise ValueError(_NO_SINGLE_N)
        return fn(self, *args, **kwargs)

    return wrapped


def _trials_per_row(n_trials: Any, rows: int) -> npt.NDArray:
    """The number of trials of each of ``rows`` rows: one number for every
    row, or one per row (#608), each a positive whole number."""
    trials = np.atleast_1d(np.asarray(n_trials, dtype=float))
    if trials.ndim != 1 or trials.size not in (1, rows):
        raise ValueError(
            "'n_trials' must be one number of trials for every row, or one "
            f"per row ({rows}); got {trials.size}"
        )
    if not (
        np.all(np.isfinite(trials)) and np.all(trials == np.round(trials))
    ):
        raise ValueError("'n_trials' must be a positive integer")
    if np.any(trials < 1):
        raise ValueError("'n_trials' must be a positive integer")
    return np.broadcast_to(trials, (rows,)).astype(int)


class Binomial_(DiscreteParametricFitter):
    r"""
    The Binomial distribution: the number of events (failures) ``k`` in a
    fixed number ``n`` of independent pass/fail trials, each with event
    probability ``p``.

    It is the recurrent (repeated-trials) counterpart of the
    :class:`Bernoulli` distribution, which is the special case ``n = 1``.
    The two agree exactly there, every function included: both follow
    the package's discrete rule :math:`R(k) = P(K > k)`, so
    ``Bernoulli.sf([0, 1], 0.3)`` and ``Binomial.sf([0, 1], 1, 0.3)`` are
    both ``[0.3, 0]``, and ``Bernoulli.sf(x, p) == Binomial.sf(x, 1, p)``
    with no offset (Bernoulli's survival function was :math:`P(X \geq x)`
    before 0.22, #344).

    The distribution is parameterised by ``n`` (the number of trials, a
    positive integer) and ``p`` (the per-trial event probability). Because
    ``n`` is an integer structural parameter, the distribution does not use
    the gradient-based MLE machinery; instead ``fit`` uses the closed-form
    maximum likelihood estimate of ``p`` for a known number of trials, in the
    same spirit as :class:`Bernoulli`.

    Examples
    --------
    >>> from surpyval import Bernoulli, Binomial
    >>> Binomial.sf([0, 1], 1, 0.3)
    array([0.3, 0. ])
    >>> Bernoulli.sf([0, 1], 0.3)
    array([0.3, 0. ])
    """

    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            k=2,
            bounds=((1, None), (0, 1)),
            # ``support`` is a pair of *exclusive* bounds: the shared
            # ``_validate_fit_inputs`` rejects data with
            # ``x <= support[0]`` or ``x >= support[1]``, so a distribution
            # declares the bound one step outside its first and last mass
            # points. The first mass point here is k = 0 -- zero events in
            # n trials is an ordinary outcome, P = 0.168 at n = 5, p = 0.3
            # -- so the lower bound is -1, as for ``Poisson``. It read 0,
            # which is ``Geometric``'s value and says zero events lie
            # outside the distribution. Nothing observed it because
            # ``Binomial`` does not inherit ``OptimisedFitMixin``, where
            # that check lives, and validates its own inputs instead.
            #
            # The upper bound stays infinite here because n is not known
            # until the model is built; ``fit`` and ``from_params`` set it
            # to n + 1 for the same reason.
            support=(-1, np.inf),
            parameter_names=["n", "p"],
            param_map={"n": 0, "p": 1},
            plot_x_scale="linear",
        )

    def df(self, x: Numeric, n: Boxable, p: Boxable) -> Boxable:
        r"""

        Probability mass function for the Binomial distribution:

        .. math::
            P(X = x) = \binom{n}{x} p^{x} (1 - p)^{n - x}

        Parameters
        ----------

        x : numpy array or scalar
            The number of events at which the mass function is evaluated
        n : integer
            The number of trials
        p : float
            The per-trial probability of an event

        Returns
        -------

        df : scalar or numpy array
            The value(s) of the mass function at x

        Examples
        --------
        >>> from surpyval import Binomial
        >>> Binomial.df(2, 5, 0.3)
        np.float64(0.3086999999999998)
        """
        from scipy.stats import binom

        return binom.pmf(x, n, p)

    def ff(self, x: Numeric, n: Boxable, p: Boxable) -> Boxable:
        r"""

        Failure (CDF) function for the Binomial distribution:

        .. math::
            F(x) = P(X \leq x) = \sum_{i=0}^{\lfloor x \rfloor}
            \binom{n}{i} p^{i} (1 - p)^{n - i}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        n : integer
            The number of trials
        p : float
            The per-trial probability of an event

        Returns
        -------

        ff : scalar or numpy array
            The value(s) of the failure function at x

        Examples
        --------
        >>> from surpyval import Binomial
        >>> Binomial.ff(2, 5, 0.3)
        np.float64(0.83692)
        """
        from scipy.stats import binom

        return binom.cdf(x, n, p)

    def sf(self, x: Numeric, n: Boxable, p: Boxable) -> Boxable:
        r"""

        Survival (reliability) function for the Binomial distribution:

        .. math::
            R(x) = P(X > x) = 1 - F(x)

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        n : integer
            The number of trials
        p : float
            The per-trial probability of an event

        Returns
        -------

        sf : scalar or numpy array
            The value(s) of the survival function at x

        Examples
        --------
        >>> from surpyval import Binomial
        >>> Binomial.sf(2, 5, 0.3)
        np.float64(0.16308)
        """
        from scipy.stats import binom

        return binom.sf(x, n, p)

    def hf(self, x: Numeric, n: Boxable, p: Boxable) -> Boxable:
        r"""

        Discrete hazard rate for the Binomial distribution; the conditional
        probability of exactly ``x`` events given at least ``x``:

        .. math::
            h(x) = \frac{P(X = x)}{P(X \geq x)}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        n : integer
            The number of trials
        p : float
            The per-trial probability of an event

        Returns
        -------

        hf : scalar or numpy array
            The value(s) of the discrete hazard rate at x
        """
        # P(X = x) / P(X >= x), on the log scale: P(X >= x) = R(x - 1).
        # The sum sf + df underflowed to 0 in the far right tail, where
        # the hazard is near 1, and it read 0 there (#458). Beyond n
        # nothing is left at risk and the hazard is 0.
        x = np.asarray(x, dtype=float)
        k = np.floor(x)
        inside = (k >= 0) & (k <= n)
        safe_k = np.where(inside, k, 0.0)
        log_hf = self.log_df(safe_k, n, p) - self.log_sf(safe_k - 1.0, n, p)
        hf = np.exp(np.where(inside, log_hf, -np.inf))
        return hf[()] if hf.ndim == 0 else hf

    def Hf(self, x: Numeric, n: Boxable, p: Boxable) -> Boxable:
        r"""

        Cumulative hazard function for the Binomial distribution:

        .. math::
            H(x) = -\ln R(x)

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        n : integer
            The number of trials
        p : float
            The per-trial probability of an event

        Returns
        -------

        Hf : scalar or numpy array
            The value(s) of the cumulative hazard function at x
        """
        # From x = n on nothing survives: H = inf.
        return -self.log_sf(x, n, p)

    def qf(self, u: Numeric, n: Boxable, p: Boxable) -> Boxable:
        r"""

        Quantile (inverse CDF) function for the Binomial distribution; the
        smallest number of events ``x`` such that :math:`F(x) \geq u`.

        Parameters
        ----------

        u : numpy array or scalar
            The values, between 0 and 1, at which the quantile is evaluated
        n : integer
            The number of trials
        p : float
            The per-trial probability of an event

        Returns
        -------

        qf : scalar or numpy array
            The quantile(s) at u

        Examples
        --------
        >>> from surpyval import Binomial
        >>> Binomial.qf(0.5, 5, 0.3)
        np.float64(1.0)
        """
        from scipy.stats import binom

        u_arr = np.asarray(u, dtype=float)
        k = refine_quantile(
            binom.ppf(u_arr, n, p),
            u_arr,
            lambda k: self.log_sf(k, n, p),
            lambda k: self.log_ff(k, n, p),
            first=0.0,
        )
        return (
            k.reshape(u_arr.shape)[()]
            if u_arr.ndim == 0
            else k.reshape(u_arr.shape)
        )

    def _log_tail(
        self, x: Numeric, n: Boxable, p: Boxable, upper: bool
    ) -> npt.NDArray:
        """log R(x) (``upper``) or log F(x), from the incomplete beta
        R(k) = I_p(k + 1, n - k) and its complement on their own log scale:
        -log(sf) lost F where it is near 0 (H of 1e-34 read 0) and was
        -inf where sf underflowed (#458)."""
        k = np.floor(np.asarray(x, dtype=float))
        inside = (k >= 0) & (k < n)
        ks = np.where(inside, k, 0.0)
        fn = betaincln if upper else betainccln
        with np.errstate(invalid="ignore"):
            out = fn(ks + 1.0, n - ks, p)
        below, above = (0.0, -np.inf) if upper else (-np.inf, 0.0)
        out = np.where(k < 0, below, np.where(k >= n, above, out))
        return out[()] if out.ndim == 0 else out

    def log_sf(self, x: Numeric, n: Boxable, p: Boxable) -> Boxable:
        return self._log_tail(x, n, p, upper=True)

    def log_ff(self, x: Numeric, n: Boxable, p: Boxable) -> Boxable:
        return self._log_tail(x, n, p, upper=False)

    def log_df(self, x: Numeric, n: Boxable, p: Boxable) -> Boxable:
        # scipy's log mass is formed on the log scale, so it stays finite
        # where the mass underflows (it read -inf from 1e-400 on, #458);
        # the fallback from hf and sf here lost it the same way.
        from scipy.stats import binom

        return binom.logpmf(x, n, p)

    def mean(self, n: Boxable, p: Boxable) -> Boxable:
        r"""

        Mean of the Binomial distribution:

        .. math::
            E[X] = n p

        Examples
        --------
        >>> from surpyval import Binomial
        >>> Binomial.mean(5, 0.3)
        1.5
        """
        return n * p

    def moment(self, m: int, n: Boxable, p: Boxable) -> Boxable:
        r"""

        m-th (raw) moment of the Binomial distribution.

        Parameters
        ----------

        m : integer
            The ordinal of the moment to calculate
        n : integer
            The number of trials
        p : float
            The per-trial probability of an event

        Returns
        -------

        moment : scalar
            The m-th raw moment of the Binomial distribution

        Examples
        --------
        >>> from surpyval import Binomial
        >>> Binomial.moment(1, 5, 0.3)
        np.float64(1.5)
        """
        from scipy.stats import binom

        return binom.moment(m, n, p)

    def entropy(self, n: Boxable, p: Boxable) -> Boxable:
        r"""

        Entropy of the Binomial distribution (in nats).

        Examples
        --------
        >>> from surpyval import Binomial
        >>> Binomial.entropy(5, 0.3)
        np.float64(1.413614855283445)
        """
        from scipy.stats import binom

        return binom.entropy(n, p)

    def random(  # type: ignore[override]
        self,
        size: int | tuple[int, ...],
        n: Boxable,
        p: Boxable,
        *,
        random_state: Any = None,
    ) -> npt.NDArray:
        r"""

        Draws random samples from the distribution in shape `size`

        Parameters
        ----------

        size : integer or tuple of positive integers
            Shape or size of the random draw
        n : integer
            The number of trials
        p : float
            The per-trial probability of an event
        random_state : int or numpy.random.Generator, optional
            Seed or generator for a draw of its own; ``None`` (the
            default) draws from numpy's global stream (see
            :meth:`ParametricFitter.random`).

        Returns
        -------

        random : scalar or numpy array
            Random values drawn from the distribution in shape `size`
        """
        from scipy.stats import binom

        # A fitted model holds n as a float (5.0), which numpy's binomial
        # draw refused: "Cannot cast scalar from dtype('float64') to
        # dtype('int64')".
        trials = np.asarray(n, dtype=float)
        if np.any(trials != np.round(trials)):
            raise ValueError(f"n must be a whole number of trials; got {n}")
        return binom.rvs(
            trials.astype(int),
            p,
            size=size,
            random_state=draw_state(random_state),
        )

    def fit(
        self,
        x: npt.ArrayLike,
        n_trials: npt.ArrayLike,
        c: npt.NDArray | None = None,
        n: npt.NDArray | None = None,
    ) -> Parametric:
        r"""

        Fit the Binomial distribution for a known number of trials,
        ``n_trials``, using the closed-form maximum likelihood estimate of
        the per-trial event probability ``p``.

        Parameters
        ----------

        x : array like
            The observed number of events for each experiment. Every value
            must be an integer in ``[0, n_trials]``.
        n_trials : integer or array like
            The (known) number of trials in each experiment: one number for
            every row, or one per row (batches of different sizes, #608).
        c : array like, optional
            Censoring flags. Censoring is not supported for the Binomial
            distribution and any non-zero flag raises a ``ValueError``.
        n : array like, optional
            The count (multiplicity) of each observation in ``x``. If
            ``None`` each observation is assumed to have occurred once.

        Returns
        -------

        model : Parametric
            A parametric model with the fitted ``[n, p]`` parameters. Its
            ``param_cb("p")`` bounds ``p`` from the events in all the
            trials, ``sum(x)`` in
            ``sum(n_trials)``, exactly (Clopper-Pearson) by default, as
            ``Bernoulli`` does: the events in all the trials are binomial
            in their total whatever the rows' sizes, so the exact interval
            holds for unequal trials too. ``param_cb("n")`` is the known
            number of trials.

        Notes
        -----
        With a different number of trials in different rows the model has
        no single ``n``: ``params`` holds ``nan`` for it, and the functions
        of the count of events (``sf``, ``df``, ``mean``, ``random``, ...)
        raise a ``ValueError`` saying so; ``p``, its bounds and the
        model's dictionary are as for equal trials. The distribution of
        the events in ``m`` trials is ``model.with_params([m, p])``.

        Examples
        --------
        >>> from surpyval import Binomial
        >>> model = Binomial.fit([2, 3, 1, 4], n_trials=5)
        >>> model.params
        array([5. , 0.5])
        >>> model.param_cb("p").round(4)
        array([0.272, 0.728])

        Batches of different sizes:

        >>> lots = Binomial.fit([1, 0, 3], n_trials=[20, 50, 80])
        >>> lots.params.round(4)
        array([  nan, 0.0267])
        >>> lots.param_cb("p").round(4)
        array([0.0073, 0.0669])
        """
        x_arr = np.atleast_1d(np.asarray(x))
        if x_arr.shape[0] == 0:
            # As every fit says it (it gave p = nan with numpy's warning)
            raise ValueError(
                "'x' is empty: at least one observation is needed"
            )

        if not np.equal(np.mod(x_arr, 1), 0).all():
            raise ValueError("'x' must contain only integer counts")

        trials = _trials_per_row(n_trials, x_arr.shape[0])

        if ((x_arr < 0) | (x_arr > trials)).any():
            raise ValueError("'x' must be between 0 and 'n_trials'")

        if c is not None and (np.atleast_1d(np.asarray(c)) != 0).any():
            raise ValueError(
                "Binomial distribution does not support censored data"
            )

        if n is None:
            n = np.ones_like(x_arr)
        n = np.atleast_1d(np.asarray(n))

        model = Parametric(self, "MLE", None, False, False, False)
        # The proportion is the exact maximum
        model.maximum = "verified"
        events = float((x_arr * n).sum())
        total = float((trials * n).sum())
        common = trials.min() == trials.max()
        # One number of trials for every row is the model's n; rows of
        # different sizes leave it none (see the Notes).
        model.params = np.array(
            [float(trials[0]) if common else np.nan, events / total]
        )
        if not common:
            # Saved with the model (to_dict), for its support.
            model._n_trials = trials
        # The events in all the trials: the bounds on p come from these
        # (#580). Their total is binomial whatever the rows' sizes.
        model._event_counts = (events, total)
        self._set_support(model, False)
        return model

    def _probability_cb(
        self,
        model: Parametric,
        name: str,
        alpha_ci: float,
        bound: str,
        method: str | None,
    ) -> npt.NDArray:
        """``model.param_cb`` (#580): bounds on ``p`` from the events in
        all the trials (see ``Bernoulli.fit``); the number of trials ``n``
        is known, so its interval is the degenerate one at its value."""
        if name == "n":
            from surpyval.utils.linalg import bound_signs

            _, signs = bound_signs(alpha_ci, bound)
            return np.full(signs.shape, float(model.params[0]))
        if name != "p":
            raise ValueError(
                "Unknown parameter {!r}; expected one of ['n', 'p']".format(
                    name
                )
            )
        return probability_bounds(
            *event_counts(model), alpha_ci, bound, method, name
        )

    def _set_support(self, model: Any, offset: bool) -> None:
        """Exclusive bounds either side of the outcomes ``{0, ..., n}``
        (see the note in ``__init__``), with ``n`` read from the model.
        ``from_dict`` restores the support through this, so a restored
        model keeps ``[-1, n + 1]``; the inherited version read the
        declared ``[-1, inf]``. A model fitted to rows of different sizes
        reaches the largest."""
        n = float(model.params[0])
        if np.isnan(n):
            n = float(np.max(np.asarray(model._n_trials, dtype=float)))
        model.support = np.array([-1, n + 1])

    def from_params(
        self,
        params: npt.ArrayLike,
        gamma: Boxable | None = None,
        lfp_p: Boxable | None = None,
        f0: Boxable | None = None,
    ) -> Parametric:
        r"""

        Create a Binomial model from the parameters ``[n, p]``.

        Parameters
        ----------

        params : array like
            The two parameters ``[n, p]``; ``n`` the (integer) number of
            trials and ``p`` the per-trial event probability.
        gamma, lfp_p, f0 : None
            Accepted so the signature matches
            :meth:`ParametricFitter.from_params`, and rejected: a
            Binomial has no offset, limited failure population or zero
            inflation. The per-trial probability lives in ``params``.

        Returns
        -------

        model : Parametric
            A parametric model with the provided parameters.

        Examples
        --------
        >>> from surpyval import Binomial
        >>> model = Binomial.from_params([5, 0.3])
        >>> model.mean()
        np.float64(1.5)
        """
        reject_structural_params(self.name, gamma, lfp_p, f0)
        params_arr = np.atleast_1d(np.asarray(params, dtype=float))

        if params_arr.shape[0] != 2:
            raise ValueError("Binomial distribution requires '[n, p]' params")

        n, prob = params_arr

        if np.mod(n, 1) != 0:
            raise ValueError("'n' must be an integer number of trials")

        if n < 1:
            raise ValueError("'n' must be a positive integer")

        if not (0 <= prob <= 1):
            raise ValueError("'p' must be between 0 and 1")

        model = Parametric(self, "given parameters", None, False, False, False)
        model.params = np.array([float(n), prob])
        self._set_support(model, False)
        return model


# The functions of the count of events need one number of trials.
for _name, _position in (
    *((name, 1) for name in ("df", "ff", "sf", "hf", "Hf", "qf")),
    *((name, 1) for name in ("log_sf", "log_ff", "log_df", "moment")),
    ("mean", 0),
    ("entropy", 0),
    ("random", 1),
):
    setattr(
        Binomial_,
        _name,
        _needs_one_n(getattr(Binomial_, _name), _position),
    )

Binomial = Binomial_("Binomial")
