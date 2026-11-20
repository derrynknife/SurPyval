"""The Jelinski-Moranda model of software failures: a whole number of
faults, each found at the same rate.

The software starts testing with ``N`` faults. Each causes failures at
the rate ``phi`` until it is found and removed, so the time from failure
``i - 1`` to failure ``i`` is exponential with rate ``phi (N - i + 1)``.
With ``n`` failures seen in a total test time ``T`` (the last failure's
time, or the end of testing after it), the log-likelihood is

.. math::
    \\ell(N, \\phi) = n \\log \\phi + \\sum_{i=1}^{n} \\log(N - i + 1)
    - \\phi\\, (N T - K), \\qquad
    K = \\sum_{i=1}^{n} (i - 1)\\, x_i + n\\, x_c,

with ``x_i`` the times between failures and ``x_c`` the time tested
since the last one. For a given ``N`` it is greatest at ``phi = n / (N T
- K)``, which leaves the profile likelihood of ``N``, a function of whole
numbers ``N >= n``.

Expanding the profile in ``1 / N`` gives ``const + n (K / T - (n - 1) /
2) / N``: it comes down to its limit (a constant rate ``n / T``) where
``K / T > (n - 1) / 2``, and climbs to it for ever otherwise. So the
maximum-likelihood ``N`` is finite exactly when

.. math::
    \\frac{K}{T} > \\frac{n - 1}{2},

which in the failure times ``t_i`` reads ``mean(t) < (n + 1) T / (2 n)``:
the failures come early enough in the test. (Littlewood and Verrall
1981 give the condition for a test that ends at a failure.)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.stats import chi2

from surpyval.recurrent.inference import check_alpha_ci
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.utils.fitter import singleton_fitter
from surpyval.utils.fitter_repr import FitterRepr
from surpyval.utils.no_maximum import maximum_entry, restored_maximum
from surpyval.utils.surpyval_data import SurpyvalData
from surpyval.utils.units import takes_unit
from surpyval.utils.validation import BOUNDS, check_option

#: The whole numbers of faults past which a profile is read from its
#: limit: the likelihood there is within 1e-9 of the constant rate's.
_FAR = 1e15

_NO_WALD = (
    "{what} has likelihood-ratio bounds only (method='lr'): the "
    "likelihood of the Jelinski-Moranda model is far from quadratic in "
    "N, a whole number that may have no upper bound, so a Wald bound "
    "(an estimate and a standard error) misstates it. Use method='lr'."
)
_NO_BOOTSTRAP = (
    "{what} has no bootstrap bound: a refit of a simulated campaign has "
    "no finite N a good share of the time (the same condition that "
    "refuses a fit), so its quantiles are not defined. Use method='lr', "
    "whose upper bound says when N is unbounded."
)


def _log_falling(N: Any, n: int) -> Any:
    """``sum(log(N - i + 1) for i in 1..n)``, the log of ``N! / (N -
    n)!``."""
    # Summed term by term: a difference of two log-gammas loses the sum
    # for a large N (they are of order N log N, the sum of order n log N).
    N = np.asarray(N, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sum(np.log(N[..., None] - np.arange(n)), axis=-1)


class _Campaign:
    """One campaign's sufficient statistics: ``n`` failures in a total
    test time ``T``, with ``K = sum((i - 1) x_i) + n x_c``."""

    def __init__(self, n: int, T: float, K: float) -> None:
        self.n, self.T, self.K = int(n), float(T), float(K)

    def exposure(self, N: Any) -> Any:
        """``sum((N - i + 1) x_i) + (N - n) x_c``: the fault-time tested
        if there were ``N`` faults."""
        return np.asarray(N, dtype=float) * self.T - self.K

    def log_likelihood(self, N: Any, phi: Any) -> Any:
        return (
            self.n * np.log(phi)
            + _log_falling(N, self.n)
            - phi * self.exposure(N)
        )

    def phi_given(self, N: Any) -> Any:
        """The rate that maximises the likelihood for ``N`` faults."""
        return self.n / self.exposure(N)

    def profile(self, N: Any) -> Any:
        """The log-likelihood at ``N`` faults, maximised over the rate."""
        return self.log_likelihood(N, self.phi_given(N))

    @property
    def limit(self) -> float:
        """The profile as ``N`` grows without bound: the log-likelihood
        of a constant rate ``n / T``."""
        return self.n * np.log(self.n / self.T) - self.n

    @property
    def finite(self) -> bool:
        """Whether the profile has its maximum at a finite ``N``."""
        return self.K / self.T > (self.n - 1) / 2.0

    def best_N(self) -> int:
        """The whole number of faults with the greatest profile
        likelihood (the smaller of two that tie). The profile rises and
        then falls, so the first ``N`` whose successor is no better."""
        n = self.n

        def gain(N: float) -> float:
            # profile(N + 1) - profile(N), written as one difference
            return float(
                np.log((N + 1.0) / (N + 1.0 - n))
                - n * np.log1p(self.T / self.exposure(N))
            )

        if gain(float(n)) <= 0.0:
            return n
        lo, hi = float(n), float(2 * n + 1)
        while gain(hi) > 0.0:
            lo, hi = hi, 2.0 * hi
        while hi - lo > 1.0:
            mid = np.floor((lo + hi) / 2.0)
            if gain(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        return int(hi)


def _from_data(data: SurpyvalData, c: Any) -> tuple:
    """The times and censoring flags of a campaign given as a
    ``SurpyvalData``."""
    if c is not None:
        raise TypeError(
            "x is a SurpyvalData, which holds the data: leave out 'c' "
            "(or pass arrays rather than a SurpyvalData)."
        )
    x = np.asarray(data.x, dtype=float)
    if x.ndim != 1 or np.any(np.asarray(data.n) != 1):
        raise ValueError(
            "The Jelinski-Moranda model takes each failure's time: the "
            "SurpyvalData has interval rows or counts above 1. For "
            "failures counted in periods fit GoelOkumoto."
        )
    return x, np.asarray(data.c)


def _campaign(x: ArrayLike, c: ArrayLike | None, gaps: bool) -> tuple:
    """Read one campaign: ``(statistics, failure times, end of test)``."""
    values = np.asarray(x, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            "x must be one campaign's failure times (or, with gaps=True, "
            "its times between failures) as a flat array of at least one "
            "value; got shape {}".format(values.shape)
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "x has {} missing or infinite value(s); the Jelinski-Moranda "
            "model needs every failure's time, in order. Remove the rows "
            "or supply the times.".format(int(np.sum(~np.isfinite(values))))
        )
    flags = np.zeros(values.size, dtype=int)
    if c is not None:
        flags = np.asarray(c)
        if flags.shape != values.shape:
            raise ValueError(
                "c must have one value for each value of x ({}); got "
                "{}".format(values.size, flags.size)
            )
        if not np.all(np.isin(flags, (0, 1))):
            raise ValueError(
                "c must be 0 (a failure) or 1 (the end of testing, with "
                "no failure then); got {}. The model takes each failure's "
                "time: for failures counted in periods (c=2) fit "
                "GoelOkumoto.".format(np.unique(flags).tolist())
            )
    if gaps:
        if np.any(values < 0):
            raise ValueError(
                "With gaps=True, x is the times between failures, which "
                "cannot be negative; got {}".format(values[values < 0][:5])
            )
        if np.any(flags[:-1] == 1):
            raise ValueError(
                "With gaps=True only the last value of x can be the end "
                "of testing (c=1): the time tested since the last "
                "failure."
            )
        times = np.cumsum(values)
    else:
        order = np.argsort(values, kind="stable")
        times, flags = values[order], flags[order]
        if np.any(times < 0):
            raise ValueError(
                "x is the failure times from the start of testing, which "
                "cannot be negative; got {}".format(times[times < 0][:5])
            )
    ended = flags == 1
    if ended.sum() > 1:
        raise ValueError(
            "c marks {} rows as the end of testing (c=1); one campaign "
            "has one end. Fit each campaign on its own.".format(
                int(ended.sum())
            )
        )
    failures = times[~ended]
    end = float(times[ended][0]) if ended.any() else None
    if end is not None and failures.size and end < failures[-1]:
        raise ValueError(
            "The end of testing (c=1) at {:g} is before the last failure "
            "at {:g}.".format(end, failures[-1])
        )
    n = failures.size
    if n < 2:
        raise ValueError(
            "The Jelinski-Moranda model has two parameters and needs at "
            "least 2 failures; the data have {}. Test for longer, or fit "
            "HPP (a constant rate).".format(n)
        )
    T = float(failures[-1]) if end is None else end
    if not T > 0:
        raise ValueError(
            "Every failure is at time 0, so there is no test time to "
            "estimate a rate from."
        )
    gap = np.diff(failures, prepend=0.0)
    K = float(np.sum(np.arange(n) * gap) + n * (T - failures[-1]))
    return _Campaign(n, T, K), failures, T


@singleton_fitter
class JelinskiMoranda(FitterRepr):
    """
    The Jelinski-Moranda software reliability model: the software holds
    ``N`` faults, a whole number, and each is found at the same rate
    ``phi``, so the time between failure ``i - 1`` and failure ``i`` is
    exponential with rate ``phi (N - i + 1)``. Fitted to one test
    campaign's failure times it estimates ``N`` and ``phi``, and so the
    faults left, ``N - n``, and the failure rate now, ``phi (N - n)``.

    It is not a Poisson process: the rate steps down at each failure, and
    ``N`` is a whole number, estimated as one. Its likelihood is known to
    have no finite maximum for data that show no growth, and to say
    little about how large ``N`` may be even when it has one, so ``fit``
    refuses the first case and the bounds (by the likelihood ratio)
    report an unbounded ``N`` as ``inf``.

    ``JelinskiMoranda`` is an instance of this class; ``fit`` and
    ``from_params`` return a
    :class:`~surpyval.recurrent.JelinskiMorandaModel`.

    References
    ----------
    Jelinski, Z. and Moranda, P. B. (1972), "Software reliability
    research", in Statistical Computer Performance Evaluation (ed. W.
    Freiberger), Academic Press, 465-484.

    Littlewood, B. and Verrall, J. L. (1981), "Likelihood function of a
    debugging model for computer software reliability", IEEE Transactions
    on Reliability R-30(2), 145-148.

    Examples
    --------
    Twelve failures of a program under test, at these days:

    >>> from surpyval.recurrent import JelinskiMoranda
    >>> x = [3, 8, 12, 20, 25, 33, 41, 47, 58, 70, 81, 99]
    >>> model = JelinskiMoranda.fit(x)
    >>> model.N, round(model.phi, 5)
    (15, 0.01511)
    >>> model.remaining_faults()
    3
    >>> model.remaining_faults_cb(alpha_ci=0.5)
    array([1., 8.])
    >>> model.remaining_faults_cb(alpha_ci=0.05)
    array([ 0., inf])
    """

    #: The ``repr`` (#614)
    fitter_kind = "fault-count model fitter"
    name = "Jelinski-Moranda"
    parameter_names = ["N", "phi"]

    @takes_unit
    def fit(
        self,
        x: ArrayLike,
        c: ArrayLike | None = None,
        *,
        gaps: bool = False,
    ) -> "JelinskiMorandaModel":
        """
        Fit the model to one test campaign by maximum likelihood.

        Parameters
        ----------
        x : array_like
            The failure times, from the start of testing (in any order);
            or, with ``gaps=True``, the times between failures in the
            order they happened. At least two failures.
        c : array_like, optional
            0 for a failure and 1 for the end of testing: one ``c=1``
            row, at or after the last failure (with ``gaps=True`` the
            last value of ``x``, the time tested since the last
            failure). Without one the test is taken to end at the last
            failure. Default: every row a failure.
        gaps : bool, optional
            Whether ``x`` is the times between failures. Default False:
            the failure times. Keyword only.

        Returns
        -------
        JelinskiMorandaModel
            The fitted model.

        Raises
        ------
        ValueError
            Where the data show no reliability growth, so that the
            likelihood rises for ever as ``N`` grows and there is no
            estimate; the message gives the condition and says what to
            fit instead.

        Examples
        --------
        A campaign as times between failures, with 21 more days of
        testing after the twelfth failure and none found:

        >>> from surpyval.recurrent import JelinskiMoranda
        >>> gaps = [3, 5, 4, 8, 5, 8, 8, 6, 11, 12, 11, 18, 21]
        >>> model = JelinskiMoranda.fit(gaps, c=[0] * 12 + [1], gaps=True)
        >>> model.N, round(model.phi, 5)
        (12, 0.02414)
        """
        if isinstance(x, SurpyvalData):
            x, c = _from_data(x, c)
        stats, failures, T = _campaign(x, c, gaps)
        if not stats.finite:
            raise ValueError(_no_growth(stats, failures))
        N = stats.best_N()
        model = JelinskiMorandaModel()
        model.dist = self
        model.N = N
        model.phi = float(stats.phi_given(N))
        model.params = np.array([float(N), model.phi])
        model.n = stats.n
        model.T = T
        model.K = stats.K
        model.how = "MLE"
        model.maximum = "verified"
        model.x = failures
        return model

    @takes_unit
    def fit_from_df(
        self,
        df: Any,
        x_col: str = "x",
        c_col: str | None = None,
        *,
        gaps: bool = False,
    ) -> "JelinskiMorandaModel":
        """
        Fit the model to one campaign held in a ``pandas.DataFrame``.

        Parameters
        ----------
        df : pandas.DataFrame
            One row per failure (and one for the end of testing, if it
            is given).
        x_col : str, optional
            The column of failure times (or of times between failures,
            with ``gaps=True``). Default ``"x"``.
        c_col : str, optional
            The column of 0 (a failure) and 1 (the end of testing).
            Default None: every row a failure.
        gaps : bool, optional
            As for :meth:`fit`.

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval.recurrent import JelinskiMoranda
        >>> df = pd.DataFrame({"day": [3, 8, 12, 20, 25, 33, 41, 47, 58]})
        >>> JelinskiMoranda.fit_from_df(df, x_col="day").N
        14
        """
        for name, col in (("x_col", x_col), ("c_col", c_col)):
            if col is not None and col not in df.columns:
                raise ValueError(
                    "{}={!r} is not a column of the DataFrame; its "
                    "columns are {}".format(name, col, list(df.columns))
                )
        c = None if c_col is None else df[c_col].to_numpy()
        return self.fit(df[x_col].to_numpy(), c, gaps=gaps)

    @takes_unit
    def from_params(
        self, params: ArrayLike, n: int = 0
    ) -> "JelinskiMorandaModel":
        """
        The model at given parameters, without data.

        Parameters
        ----------
        params : array_like
            ``[N, phi]``: the whole number of faults and the rate of
            each.
        n : int, optional
            The failures already found (and removed): the model then
            answers for the software as it is now. Default 0.

        Examples
        --------
        >>> from surpyval.recurrent import JelinskiMoranda
        >>> model = JelinskiMoranda.from_params([20, 0.01], n=15)
        >>> model.remaining_faults(), round(model.current_rate(), 3)
        (5, 0.05)
        """
        values = np.asarray(params, dtype=float)
        if values.shape != (2,):
            raise ValueError(
                "params must be [N, phi]; got shape {}".format(values.shape)
            )
        N, phi = float(values[0]), float(values[1])
        if not (N >= 1 and N == np.floor(N) and np.isfinite(N)):
            raise ValueError(
                "N is the number of faults, a whole number of at least 1; "
                f"got {values[0]!r}"
            )
        if not (phi > 0 and np.isfinite(phi)):
            raise ValueError(
                f"phi is each fault's rate, which is positive; got {phi!r}"
            )
        if not (isinstance(n, (int, np.integer)) and 0 <= n <= N):
            raise ValueError(
                "n is the number of failures already found, a whole "
                f"number from 0 to N = {int(N)}; got {n!r}"
            )
        model = JelinskiMorandaModel()
        model.dist = self
        model.N = int(N)
        model.phi = phi
        model.params = np.array([N, phi])
        model.n = int(n)
        model.how = "from_params"
        return model


def _no_growth(stats: _Campaign, failures: np.ndarray) -> str:
    n, T = stats.n, stats.T
    return (
        "The data show no reliability growth that the Jelinski-Moranda "
        "model can fit: the {n} failures' mean time, {mean:.6g}, is not "
        "earlier than (n + 1) T / (2 n) = {edge:.6g} (T = {T:.6g}, the "
        "test's length). The likelihood then keeps rising as the number "
        "of faults N grows without bound and each one's rate phi falls "
        "to 0, so N has no estimate. Fit HPP (a constant rate) or an "
        "NHPP growth model such as GoelOkumoto or CrowAMSAA and compare "
        "them with aic(); or test for longer, until the failures "
        "slow.".format(
            n=n,
            mean=float(np.mean(failures)),
            edge=(n + 1) * T / (2.0 * n),
            T=T,
        )
    )


class JelinskiMorandaModel(SerialisableMixin):
    """
    A fitted Jelinski-Moranda model: ``N`` faults in all, each found at
    the rate ``phi``, with ``n`` of them found so far.

    Attributes
    ----------
    N : int
        The estimated number of faults at the start of testing.
    phi : float
        The rate at which each fault causes failures.
    params : numpy.ndarray
        ``[N, phi]``.
    n : int
        The failures found (the data's, or ``from_params``'s ``n``).
    T : float
        The length of the test (a fitted model only).

    Bounds are by the likelihood ratio (``method="lr"``, the default and
    the only method): the values whose profile likelihood is within the
    chi-squared quantile of the maximum. The number of faults is a whole
    number, so its bounds are whole numbers, and an upper bound of
    ``inf`` says that the data do not rule out any number of faults,
    however large (a constant rate fits them nearly as well).

    Examples
    --------
    >>> from surpyval.recurrent import JelinskiMoranda
    >>> x = [3, 8, 12, 20, 25, 33, 41, 47, 58, 70, 81, 99]
    >>> model = JelinskiMoranda.fit(x)
    >>> model
    Jelinski-Moranda SurPyval Model
    ===============================
    Fitted by           : MLE
    Failures            : 12 in a test of 99
    Parameters          :
             N: 15
           phi: 0.015113350125944584
    Faults left         : 3
    Rate now            : 0.04534005037783375
    >>> model.param_cb("N", alpha_ci=0.5)
    array([13., 20.])
    >>> model.current_rate_cb(alpha_ci=0.5).round(4)
    array([0.0175, 0.0804])
    """

    #: What the fit reached, one of ``MAXIMUM_STATES``
    #: (``surpyval.utils.no_maximum``): a fit is the exact maximum over
    #: the whole numbers, so ``"verified"``; ``"not applicable"`` for a
    #: model built from its parameters.
    maximum: str = "not applicable"

    dist: Any
    N: int
    phi: float
    params: np.ndarray
    n: int
    how: str
    T: float
    K: float
    x: np.ndarray

    @property
    def parameter_names(self) -> list[str]:
        return ["N", "phi"]

    def __repr__(self) -> str:
        lines = [
            "Jelinski-Moranda SurPyval Model",
            "===============================",
            "Fitted by           : "
            + (
                "MLE" if self.how == "MLE" else "given parameters (not fitted)"
            ),
        ]
        if hasattr(self, "T"):
            lines.append(
                f"Failures            : {self.n} in a test of {self.T:g}"
            )
        else:
            lines.append(f"Failures            : {self.n}")
        lines += [
            "Parameters          :",
            f"{'N':>10}: {self.N}",
            f"{'phi':>10}: {self.phi}",
            f"Faults left         : {self.remaining_faults()}",
            f"Rate now            : {self.current_rate()}",
        ]
        return "\n".join(lines)

    # -- serialisation -------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise the model to a plain, JSON-serialisable dict: the
        parameters, the failures found, and for a fitted model the two
        statistics of the campaign that the likelihood (and so every
        bound) is a function of. The failure times are not stored.

        See Also
        --------
        from_dict, to_json, from_json
        """
        out: dict = {
            "model": "JelinskiMorandaModel",
            "params": [float(self.N), float(self.phi)],
            "n": int(self.n),
            "how": self.how,
            **maximum_entry(self.maximum),
        }
        if hasattr(self, "T"):
            out["test_time"] = float(self.T)
            out["weighted_time"] = float(self.K)
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "JelinskiMorandaModel":
        """
        Rebuild a model from a :meth:`to_dict` dictionary.

        See Also
        --------
        to_dict, to_json, from_json
        """
        require_model_tag(
            model_dict, "JelinskiMorandaModel", "a Jelinski-Moranda model"
        )
        out = JelinskiMoranda.from_params(
            model_dict["params"], n=int(model_dict["n"])
        )
        out.how = model_dict.get("how", "from_params")
        out.maximum = restored_maximum(model_dict)
        if model_dict.get("test_time") is not None:
            out.T = float(model_dict["test_time"])
            out.K = float(model_dict["weighted_time"])
        return out

    # -- the fitted likelihood ------------------------------------------

    def _stats(self, what: str) -> _Campaign:
        if not hasattr(self, "T") or self.how != "MLE":
            raise ValueError(
                "{} needs the likelihood of a fitted model; this one was "
                "built from its parameters. Fit it with "
                "JelinskiMoranda.fit.".format(what)
            )
        return _Campaign(self.n, self.T, self.K)

    @property
    def log_likelihood(self) -> float:
        """The log-likelihood at the fitted parameters.

        Examples
        --------
        >>> from surpyval.recurrent import JelinskiMoranda
        >>> x = [3, 8, 12, 20, 25, 33, 41, 47, 58, 70, 81, 99]
        >>> round(JelinskiMoranda.fit(x).log_likelihood, 4)
        -36.1986
        """
        stats = self._stats("log_likelihood")
        return float(stats.log_likelihood(self.N, self.phi))

    def neg_ll(self) -> float:
        """The negative log-likelihood at the fitted parameters."""
        return -self.log_likelihood

    def aic(self) -> float:
        """Akaike's criterion, ``2 * 2 + 2 neg_ll()``, comparable with
        the ``aic()`` of an NHPP model fitted to the same failure times
        (the two likelihoods are densities of the same data)."""
        return 4.0 + 2.0 * self.neg_ll()

    def aic_c(self) -> float:
        """The small-sample AIC, with the failures as the sample size:
        ``aic() + 12 / (n - 3)`` (``inf`` with three failures or
        fewer)."""
        if self.n <= 3:
            return float(np.inf)
        return self.aic() + 12.0 / (self.n - 3.0)

    def bic(self) -> float:
        """The Bayesian criterion, ``2 log(n) + 2 neg_ll()``, with the
        failures as the sample size."""
        return 2.0 * float(np.log(self.n)) + 2.0 * self.neg_ll()

    def _parameter_bounds(self) -> list:
        """The range of each parameter (``N`` is also a whole number of
        at least the failures found)."""
        return [(0, None), (0, None)]

    def profile_likelihood(self, N: ArrayLike) -> np.ndarray:
        """
        The log-likelihood at each number of faults ``N``, maximised over
        ``phi``: the curve the estimate and the bounds on ``N`` are read
        from. ``inf`` gives its limit, a constant rate's log-likelihood.

        Parameters
        ----------
        N : array_like
            Whole numbers of faults, each at least the failures found
            (below that the data are impossible: ``-inf``).

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import JelinskiMoranda
        >>> x = [3, 8, 12, 20, 25, 33, 41, 47, 58, 70, 81, 99]
        >>> model = JelinskiMoranda.fit(x)
        >>> model.profile_likelihood([12, 15, 20, np.inf]).round(3)
        array([-36.697, -36.199, -36.39 , -37.323])
        """
        stats = self._stats("profile_likelihood")
        values = np.asarray(N, dtype=float)
        out = np.full(values.shape, -np.inf)
        far = values >= _FAR
        ok = (values >= self.n) & ~far
        out[far] = stats.limit
        out[ok] = stats.profile(values[ok])
        return out

    # -- the answers ------------------------------------------------------

    def remaining_faults(self) -> int:
        """The faults left in the software, ``N - n``."""
        return int(self.N - self.n)

    def current_rate(self) -> float:
        """The failure rate now, after the ``n`` failures found: ``phi
        (N - n)`` (0 when no fault is left)."""
        return float(self.phi * (self.N - self.n))

    def mtbf(self) -> float:
        """The mean time to the next failure, ``1 / current_rate()``
        (``inf`` when no fault is left)."""
        rate = self.current_rate()
        return float(np.inf) if rate == 0 else 1.0 / rate

    def next_failure_sf(self, x: ArrayLike) -> np.ndarray:
        """
        The chance that the software runs a further ``x`` without a
        failure: ``exp(-phi (N - n) x)``, the survival function of the
        time to the next failure (the name the NHPP models use).

        Examples
        --------
        >>> from surpyval.recurrent import JelinskiMoranda
        >>> model = JelinskiMoranda.from_params([20, 0.01], n=15)
        >>> model.next_failure_sf([0.0, 10.0]).round(4)
        array([1.    , 0.6065])
        """
        values = np.asarray(x, dtype=float)
        if np.any(values < 0):
            raise ValueError(
                "x is time ahead of now, so it cannot be negative; got "
                "{}".format(values[values < 0][:5])
            )
        return np.exp(-self.current_rate() * values)

    def cif(self, x: ArrayLike) -> np.ndarray:
        """
        The expected number of failures by the test time ``x`` from the
        start, ``N (1 - exp(-phi x))`` (each fault is found by ``x`` with
        that chance): the same curve as a Goel-Okumoto process with ``a =
        N`` and ``b = phi``.

        Examples
        --------
        >>> from surpyval.recurrent import JelinskiMoranda
        >>> JelinskiMoranda.from_params([20, 0.01]).cif([0, 50, 1e6])
        array([ 0.        ,  7.86938681, 20.        ])
        """
        x = np.asarray(x, dtype=float)
        return -self.N * np.expm1(-self.phi * x)

    def iif(self, x: ArrayLike) -> np.ndarray:
        """
        The expected failure rate at the test time ``x`` from the start,
        ``N phi exp(-phi x)``: the slope of :meth:`cif`. (The rate after
        the failures actually found is :meth:`current_rate`.)

        Examples
        --------
        >>> from surpyval.recurrent import JelinskiMoranda
        >>> JelinskiMoranda.from_params([20, 0.01]).iif([0, 50]).round(4)
        array([0.2   , 0.1213])
        """
        x = np.asarray(x, dtype=float)
        return self.N * self.phi * np.exp(-self.phi * x)

    # -- bounds -------------------------------------------------------------

    @staticmethod
    def _check_method(method: str, what: str) -> None:
        check_option("method", method, ("lr", "wald", "bootstrap"))
        if method == "wald":
            raise ValueError(_NO_WALD.format(what=what))
        if method == "bootstrap":
            raise ValueError(_NO_BOOTSTRAP.format(what=what))

    @staticmethod
    def _drop(alpha_ci: float, bound: str) -> float:
        """How far below its maximum the log-likelihood may be inside
        the bounds."""
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        level = 1.0 - alpha_ci if bound == "two-sided" else 1.0 - 2 * alpha_ci
        if level <= 0:
            return 0.0
        return 0.5 * float(chi2.ppf(level, 1))

    def _N_bounds(self, drop: float) -> tuple[float, float]:
        """The least and greatest whole ``N`` whose profile likelihood is
        within ``drop`` of the maximum (the greatest ``inf`` where the
        limit is)."""
        stats = self._stats("A bound")
        floor = float(stats.profile(self.N)) - drop

        def inside(N: float) -> bool:
            return float(stats.profile(N)) >= floor

        lo, hi = float(self.n), float(self.N)
        if not inside(lo):
            while hi - lo > 1.0:  # lo outside, hi inside
                mid = np.floor((lo + hi) / 2.0)
                lo, hi = (lo, mid) if inside(mid) else (mid, hi)
            lower = hi
        else:
            lower = lo
        if stats.limit >= floor:
            return lower, np.inf
        lo, hi = float(self.N), float(2 * self.N + 1)
        while inside(hi):
            lo, hi = hi, 2.0 * hi
        while hi - lo > 1.0:  # lo inside, hi outside
            mid = np.floor((lo + hi) / 2.0)
            lo, hi = (mid, hi) if inside(mid) else (lo, mid)
        return lower, lo

    def _phi_profile(self, phi: float, stats: _Campaign) -> float:
        """The log-likelihood at ``phi``, maximised over whole ``N``. In
        ``N`` it rises and then falls (its steps ``log((N + 1) / (N + 1 -
        n)) - phi T`` fall), so the best ``N`` is the first whose step is
        not positive: ``N + 1 >= n / (1 - exp(-phi T))``."""
        n = stats.n
        best = max(float(n), np.ceil(n / -np.expm1(-phi * stats.T)) - 1.0)
        return float(
            max(
                stats.log_likelihood(N, phi)
                for N in (best, max(float(n), best - 1.0), best + 1.0)
            )
        )

    def _rate_profile(self, rate: float, stats: _Campaign) -> float:
        """The log-likelihood where the rate now is ``rate``, maximised
        over whole ``N > n`` (``phi = rate / (N - n)``) and the limit of
        many faults (a constant rate)."""
        n = stats.n
        left = np.unique(
            np.concatenate(
                [
                    np.arange(1.0, 2001.0),
                    np.round(np.geomspace(2e3, 1e12, 800)),
                ]
            )
        )
        N = n + left
        values = stats.log_likelihood(N, rate / left)
        constant = n * np.log(rate) - rate * stats.T
        return float(max(np.max(values), constant))

    @staticmethod
    def _interval(profile: Any, peak: float, floor: float) -> tuple:
        """Where ``profile``, greatest at the positive value ``peak``,
        crosses ``floor`` below and above it, searched outwards by
        halving and doubling: ``(0, ...)`` where it never does below,
        and ``(0, 0)`` where even the peak is under the floor."""

        def gap(value: float) -> float:
            return profile(value) - floor

        if gap(peak) < 0:
            return 0.0, 0.0
        lo, lower = peak, 0.0
        for _ in range(200):
            lo = lo / 2.0
            if gap(lo) < 0:
                lower = brentq(gap, lo, 2.0 * lo, xtol=1e-300, rtol=1e-12)
                break
        hi = peak
        while gap(2.0 * hi) >= 0:
            hi = 2.0 * hi
        upper = brentq(gap, hi, 2.0 * hi, xtol=1e-300, rtol=1e-12)
        return lower, upper

    @staticmethod
    def _sided(pair: tuple[float, float], bound: str) -> Any:
        if bound == "two-sided":
            return np.array(pair, dtype=float)
        return float(pair[0] if bound == "lower" else pair[1])

    def param_cb(
        self,
        name: str,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "lr",
    ) -> Any:
        """
        Likelihood-ratio confidence bounds on ``N`` or ``phi``.

        Parameters
        ----------
        name : {'N', 'phi'}
            The parameter.
        alpha_ci : float, optional
            The total tail probability of the bound(s). Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds are an array ``[lower, upper]``; a one-sided
            bound a number.
        method : {'lr'}, optional
            Only the likelihood ratio: ``"wald"`` and ``"bootstrap"`` are
            refused, saying why.

        Returns
        -------
        numpy.ndarray or float
            For ``N`` whole numbers, the upper one ``inf`` where the data
            do not bound the number of faults; for ``phi`` the lower one
            0 where they do not bound it away from 0 (the same case).

        Examples
        --------
        >>> from surpyval.recurrent import JelinskiMoranda
        >>> x = [3, 8, 12, 20, 25, 33, 41, 47, 58, 70, 81, 99]
        >>> model = JelinskiMoranda.fit(x)
        >>> model.param_cb("N")
        array([12., inf])
        >>> model.param_cb("N", alpha_ci=0.25, bound="upper")
        20.0
        >>> model.param_cb("phi").round(5)
        array([0.     , 0.03788])
        """
        check_option("name", name, ("N", "phi"))
        self._check_method(method, "param_cb")
        drop = self._drop(alpha_ci, bound)
        stats = self._stats("param_cb")
        if name == "N":
            return self._sided(self._N_bounds(drop), bound)
        floor = float(stats.profile(self.N)) - drop
        pair = self._interval(
            lambda phi: self._phi_profile(phi, stats), self.phi, floor
        )
        if stats.limit >= floor:
            pair = (0.0, pair[1])
        return self._sided(pair, bound)

    def remaining_faults_cb(
        self,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "lr",
    ) -> Any:
        """
        Likelihood-ratio confidence bounds on the faults left, ``N - n``:
        the bounds on ``N`` less the failures found. The upper bound is
        ``inf`` where the data do not bound the number of faults. See
        :meth:`param_cb` for the arguments.

        Examples
        --------
        >>> from surpyval.recurrent import JelinskiMoranda
        >>> x = [3, 8, 12, 20, 25, 33, 41, 47, 58, 70, 81, 99]
        >>> JelinskiMoranda.fit(x).remaining_faults_cb()
        array([ 0., inf])
        """
        self._check_method(method, "remaining_faults_cb")
        drop = self._drop(alpha_ci, bound)
        lower, upper = self._N_bounds(drop)
        return self._sided((lower - self.n, upper - self.n), bound)

    def current_rate_cb(
        self,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "lr",
    ) -> Any:
        """
        Likelihood-ratio confidence bounds on the failure rate now,
        ``phi (N - n)``: the rates whose profile likelihood (over the
        whole numbers of faults left, and their limit, a constant rate)
        is within the chi-squared quantile of the maximum. The lower
        bound is 0 where ``N = n``, no fault left, is inside the bounds
        on ``N``. See :meth:`param_cb` for the arguments.

        Examples
        --------
        >>> from surpyval.recurrent import JelinskiMoranda
        >>> x = [3, 8, 12, 20, 25, 33, 41, 47, 58, 70, 81, 99]
        >>> JelinskiMoranda.fit(x).current_rate_cb().round(4)
        array([0.    , 0.1709])
        """
        self._check_method(method, "current_rate_cb")
        drop = self._drop(alpha_ci, bound)
        stats = self._stats("current_rate_cb")
        floor = float(stats.profile(self.N)) - drop
        # Of the positive rates, the likeliest is at the best N with a
        # fault left
        most = max(self.N, self.n + 1)
        peak = float(stats.phi_given(most)) * (most - self.n)
        pair = self._interval(
            lambda rate: self._rate_profile(rate, stats), peak, floor
        )
        # No fault left (N = n, a rate of exactly 0) is one of the values
        if float(stats.profile(self.n)) >= floor:
            pair = (0.0, pair[1])
        return self._sided(pair, bound)

    def mtbf_cb(
        self,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "lr",
    ) -> Any:
        """
        Likelihood-ratio confidence bounds on the mean time to the next
        failure: the reciprocals of :meth:`current_rate_cb`'s, so a lower
        bound here is an upper bound on the rate, and the upper bound is
        ``inf`` where no fault left is inside the bounds.

        Examples
        --------
        >>> from surpyval.recurrent import JelinskiMoranda
        >>> x = [3, 8, 12, 20, 25, 33, 41, 47, 58, 70, 81, 99]
        >>> JelinskiMoranda.fit(x).mtbf_cb(bound="lower").round(2)
        np.float64(6.78)
        """
        self._check_method(method, "mtbf_cb")
        opposite = {
            "two-sided": "two-sided",
            "lower": "upper",
            "upper": "lower",
        }
        check_option("bound", bound, BOUNDS)
        rate = np.asarray(
            self.current_rate_cb(alpha_ci, opposite[bound]), dtype=float
        )
        with np.errstate(divide="ignore"):
            out = 1.0 / rate
        return out[::-1] if bound == "two-sided" else out
