from __future__ import annotations

from typing import Any

import autograd.numpy as np
import numpy.typing as npt
from scipy.stats import binom

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


class Binomial_(DiscreteParametricFitter):
    r"""
    The Binomial distribution: the number of events (failures) ``k`` in a
    fixed number ``n`` of independent pass/fail trials, each with event
    probability ``p``.

    It is the recurrent (repeated-trials) counterpart of the
    :class:`Bernoulli` distribution, which is the special case ``n = 1``.
    The two agree exactly on the probability mass there. Their survival
    functions are offset by one, which is a convention rather than a
    disagreement: this class follows the package's discrete rule
    :math:`R(k) = P(K > k)`, while Bernoulli uses :math:`P(X \geq x)` so
    that ``R(0) = 1`` and ``R(1) = p``. Hence
    ``Bernoulli.sf(x, p) == Binomial.sf(x - 1, 1, p)``.

    The distribution is parameterised by ``n`` (the number of trials, a
    positive integer) and ``p`` (the per-trial event probability). Because
    ``n`` is an integer structural parameter, the distribution does not use
    the gradient-based MLE machinery; instead ``fit`` uses the closed-form
    maximum likelihood estimate of ``p`` for a known number of trials, in the
    same spirit as :class:`Bernoulli`.
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
        n_trials: int,
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
        n_trials : integer
            The (known) number of trials in each experiment.
        c : array like, optional
            Censoring flags. Censoring is not supported for the Binomial
            distribution and any non-zero flag raises a ``ValueError``.
        n : array like, optional
            The count (multiplicity) of each observation in ``x``. If
            ``None`` each observation is assumed to have occurred once.

        Returns
        -------

        model : Parametric
            A parametric model with the fitted ``[n_trials, p]`` parameters.
            Its ``param_cb("p")`` bounds ``p`` from the events in all the
            trials, ``sum(x)`` in ``n_trials * len(x)``, exactly
            (Clopper-Pearson) by default, as ``Bernoulli`` does;
            ``param_cb("n")`` is the known ``n_trials``.

        Examples
        --------
        >>> from surpyval import Binomial
        >>> model = Binomial.fit([2, 3, 1, 4], n_trials=5)
        >>> model.params
        array([5. , 0.5])
        >>> model.param_cb("p").round(4)
        array([0.272, 0.728])
        """
        x_arr = np.atleast_1d(np.asarray(x))

        if not np.equal(np.mod(x_arr, 1), 0).all():
            raise ValueError("'x' must contain only integer counts")

        n_trials = int(n_trials)
        if n_trials < 1:
            raise ValueError("'n_trials' must be a positive integer")

        if ((x_arr < 0) | (x_arr > n_trials)).any():
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
        p = (x_arr * n).sum() / (n_trials * n.sum())
        model.params = np.array([float(n_trials), p])
        # The events in all the trials: the bounds on p come from these
        # (#580).
        model._event_counts = (
            float((x_arr * n).sum()),
            float(n_trials * n.sum()),
        )
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
        declared ``[-1, inf]``."""
        model.support = np.array([-1, float(model.params[0]) + 1])

    # Narrower than ParametricFitter.from_params, which takes
    # (params, gamma, p, f0). Unlike `fit`, this one is not resolved
    # by the OptimisedFitMixin split: every distribution has a
    # from_params. It is a parameter *rename* -- the base's `params`
    # became `params` -- so positional calls work and keyword calls
    # raise. Fixing it means renaming
    # back, with a deprecation alias, and is tracked separately.
    def from_params(
        self,
        params: npt.ArrayLike,
        gamma: Boxable | None = None,
        p: Boxable | None = None,
        f0: Boxable | None = None,
    ) -> Parametric:
        r"""

        Create a Binomial model from the parameters ``[n, p]``.

        Parameters
        ----------

        params : array like
            The two parameters ``[n, p]``; ``n`` the (integer) number of
            trials and ``p`` the per-trial event probability.
        gamma, p, f0 : None
            Accepted so the signature matches
            :meth:`ParametricFitter.from_params`, and rejected: a
            Binomial has no offset, limited failure population or zero
            inflation. The base's ``p`` is the *never-fails* proportion,
            not the per-trial probability, which lives in ``params``.

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
        reject_structural_params(self.name, gamma, p, f0)
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


Binomial = Binomial_("Binomial")
