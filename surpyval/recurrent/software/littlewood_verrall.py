"""The Littlewood-Verrall model of software failures: a fix may make the
software better or worse.

The time from failure ``i - 1`` to failure ``i`` is exponential with a
rate ``lambda_i`` that is itself uncertain: gamma with shape ``alpha``
and rate ``psi(i)``, where ``psi`` grows with ``i`` (``beta0 + beta1 i``,
or with ``i**2``). So the rates fall on average, which is reliability
growth, but any one fix may leave the rate higher than before. With the
rate integrated out each gap is Pareto,

.. math::
    f(t_i) = \\frac{\\alpha\\, \\psi(i)^{\\alpha}}
    {(\\psi(i) + t_i)^{\\alpha + 1}}, \\qquad
    S(t_i) = \\left(\\frac{\\psi(i)}{\\psi(i) + t_i}\\right)^{\\alpha},

and the gaps are independent, so the log-likelihood is a sum over them
(a test that goes on after the last failure adds the log of ``S`` at the
time since it, with ``psi(n + 1)``).

As ``alpha`` grows with ``psi / alpha`` held, the gamma closes on a
point and the gaps become exponential with the mean ``psi(i) / alpha``:
a model with no scatter in the rates. Data whose gaps are no more
scattered than that are fitted best at that edge, where ``alpha`` has no
finite estimate; the fit says so, by comparing its likelihood with that
of the limit.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq, minimize
from scipy.stats import chi2
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm

from surpyval.recurrent.inference import check_alpha_ci
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.univariate.parametric.fitters import verify_or_polish
from surpyval.utils.fitter import singleton_fitter
from surpyval.utils.fitter_repr import FitterRepr
from surpyval.utils.linalg import numerical_hessian
from surpyval.utils.no_maximum import (
    maximum_entry,
    restored_maximum,
    warn_unverified,
)
from surpyval.utils.surpyval_data import SurpyvalData
from surpyval.utils.units import takes_unit
from surpyval.utils.validation import BOUNDS, check_option

from .jelinski_moranda import _campaign, _from_data

#: The forms of ``psi(i)``: the power of the failure number it is linear
#: in.
PSI = {"linear": 1, "quadratic": 2}

#: A fit must beat the model's limit (exponential gaps, no scatter in the
#: rates) by more than this much log-likelihood to have a finite alpha.
LIMIT_MARGIN = 1e-6


class _Gaps:
    """One campaign's gaps and the likelihoods of them."""

    def __init__(self, gaps: np.ndarray, since: float, power: int) -> None:
        self.t = np.asarray(gaps, dtype=float)
        self.n = self.t.size
        self.since = float(since)
        self.power = power
        #: psi is linear in g: from its value at the first gap (g = 0) to
        #: its value at the gap after the last failure (g = 1)
        index = np.arange(1.0, self.n + 2.0) ** power
        self.g = (index - 1.0) / (index[-1] - 1.0)

    def psi(self, first: float, last: float) -> np.ndarray:
        """``psi`` at gaps 1 to n + 1, from its two ends."""
        return first + (last - first) * self.g

    def betas(self, first: float, last: float) -> tuple[float, float]:
        """``(beta0, beta1)`` of ``psi(i) = beta0 + beta1 i**power``."""
        span = float((self.n + 1.0) ** self.power - 1.0)
        beta1 = (last - first) / span
        return first - beta1, beta1

    def ends(self, beta0: float, beta1: float) -> tuple[float, float]:
        return (
            beta0 + beta1,
            beta0 + beta1 * float((self.n + 1.0) ** self.power),
        )

    def log_likelihood(self, alpha: float, first: float, last: float) -> float:
        psi = self.psi(first, last)
        own, after = psi[:-1], psi[-1]
        with np.errstate(all="ignore"):
            # alpha log(psi) - (alpha + 1) log(psi + t), written so that
            # it keeps its digits as alpha grows
            value = np.sum(
                np.log(alpha)
                - alpha * np.log1p(self.t / own)
                - np.log(own + self.t)
            ) - alpha * np.log1p(self.since / after)
        return float(value) if np.isfinite(value) else -np.inf

    def limit_log_likelihood(self, first: float, last: float) -> float:
        """The log-likelihood of exponential gaps with means linear from
        ``first`` to ``last``: the model as alpha grows without bound."""
        mean = self.psi(first, last)
        with np.errstate(all="ignore"):
            value = (
                np.sum(-np.log(mean[:-1]) - self.t / mean[:-1])
                - self.since / mean[-1]
            )
        return float(value) if np.isfinite(value) else -np.inf

    def starts(self) -> tuple[float, float]:
        """Rough mean gaps at the start and at the end."""
        third = max(self.n // 3, 1)
        scale = float(np.mean(self.t)) or 1.0
        first = max(float(np.mean(self.t[:third])), 1e-3 * scale)
        last = max(float(np.mean(self.t[-third:])), 1e-3 * scale)
        return first, last

    def best_limit(self) -> float:
        """The greatest log-likelihood of the limit."""
        first, last = self.starts()

        def objective(u: np.ndarray) -> float:
            value = -self.limit_log_likelihood(*np.exp(u))
            return value if np.isfinite(value) else 1e300

        res = minimize(
            objective,
            np.log([first, last]),
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000},
        )
        return -float(res.fun)

    def search(self, alpha: float | None = None) -> tuple:
        """The maximum of the likelihood (over the two ends of psi only,
        with ``alpha`` given): ``(u, log-likelihood, objective)`` with
        ``u`` the logs of the free parameters."""
        first, last = self.starts()

        if alpha is None:

            def objective(u: np.ndarray) -> float:
                value = -self.log_likelihood(*np.exp(u))
                return value if np.isfinite(value) else 1e300

            trials = [
                np.log(
                    [a, first * max(a - 1.0, 0.5), last * max(a - 1.0, 0.5)]
                )
                for a in (1.5, 4.0, 20.0)
            ]
        else:

            def objective(u: np.ndarray) -> float:
                value = -self.log_likelihood(alpha, *np.exp(u))
                return value if np.isfinite(value) else 1e300

            scale = max(alpha - 1.0, 0.5)
            trials = [np.log([first * scale, last * scale])]
        best = None
        for start in trials:
            res = minimize(
                objective,
                start,
                method="Nelder-Mead",
                options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 6000},
            )
            if best is None or res.fun < best.fun:
                best = res
        return best, objective


def _read(x: Any, c: Any, gaps: bool) -> tuple:
    """The gaps and the time tested since the last failure."""
    if isinstance(x, SurpyvalData):
        x, c = _from_data(x, c)
    try:
        _, failures, end = _campaign(x, c, gaps)
    except ValueError as error:
        raise ValueError(
            str(error)
            .replace("Jelinski-Moranda", "Littlewood-Verrall")
            .replace("fit GoelOkumoto", "fit GoelOkumoto (an NHPP)")
        ) from None
    if failures.size < 5:
        raise ValueError(
            "The Littlewood-Verrall model has three parameters and needs "
            "at least 5 failures; the data have {}. Test for longer, or "
            "fit a model with fewer parameters (JelinskiMoranda, "
            "GoelOkumoto).".format(failures.size)
        )
    spans = np.diff(failures, prepend=0.0)
    if np.any(spans <= 0):
        raise ValueError(
            "{} of the times between failures are 0 (failures at the "
            "same time, or one at time 0), where a Pareto density has no "
            "likelihood to compare. Record each failure's own "
            "time.".format(int(np.sum(spans <= 0)))
        )
    return spans, float(end - failures[-1]), failures


@singleton_fitter
class LittlewoodVerrall(FitterRepr):
    """
    The Littlewood-Verrall software reliability model: the time to the
    ``i``-th failure is exponential with a rate that is itself random,
    gamma with shape ``alpha`` and rate ``psi(i) = beta0 + beta1 i`` (or
    ``beta0 + beta1 i**2``). The rates fall on average as failures are
    fixed, but a fix can make things worse: the model does not suppose
    that every fix works, nor that there is a fixed number of faults.

    It is fitted by maximum likelihood to one test campaign's times
    between failures, each of which is Pareto once the rate is integrated
    out. Its predictions of the next failure carry the uncertainty about
    the rate: :meth:`LittlewoodVerrallModel.next_failure_sf` is a Pareto
    survival function, with a heavier tail than an exponential's.

    ``LittlewoodVerrall`` is an instance of this class; ``fit`` and
    ``from_params`` return a
    :class:`~surpyval.recurrent.LittlewoodVerrallModel`.

    References
    ----------
    Littlewood, B. and Verrall, J. L. (1973), "A Bayesian reliability
    growth model for computer software", Journal of the Royal Statistical
    Society C 22(3), 332-346.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.recurrent import LittlewoodVerrall
    >>> truth = LittlewoodVerrall.from_params([3.0, 10.0, 2.0])
    >>> gaps = truth.random(200, random_state=1)
    >>> model = LittlewoodVerrall.fit(gaps, gaps=True)
    >>> model.params.round(2)
    array([ 4.64, 17.65,  3.19])
    >>> model.next_failure_sf([10.0, 100.0]).round(3)
    array([0.933, 0.519])
    """

    #: The ``repr`` (#614)
    fitter_kind = "rate-uncertainty model fitter"
    name = "Littlewood-Verrall"
    parameter_names = ["alpha", "beta0", "beta1"]

    @takes_unit
    def fit(
        self,
        x: ArrayLike,
        c: ArrayLike | None = None,
        *,
        gaps: bool = False,
        psi: str = "linear",
    ) -> "LittlewoodVerrallModel":
        """
        Fit the model to one test campaign by maximum likelihood.

        Parameters
        ----------
        x : array_like
            The failure times, from the start of testing; or, with
            ``gaps=True``, the times between failures in the order they
            happened. At least five failures, at different times.
        c : array_like, optional
            0 for a failure and 1 for the end of testing (one row, after
            the last failure; with ``gaps=True`` the last value of ``x``
            is then the time tested since the last failure). Default:
            every row a failure, the test ending at the last.
        gaps : bool, optional
            Whether ``x`` is the times between failures. Default False.
            Keyword only.
        psi : {'linear', 'quadratic'}, optional
            How ``psi`` grows with the failure number ``i``: ``beta0 +
            beta1 i`` (default) or ``beta0 + beta1 i**2``. Keyword only.

        Returns
        -------
        LittlewoodVerrallModel
            The fitted model.

        Raises
        ------
        ValueError
            Where the gaps are no more scattered than exponential times
            with a growing mean would be: the likelihood then rises for
            ever as ``alpha`` grows, and there is no estimate. The
            message says so and what to fit instead.
        """
        check_option("psi", psi, tuple(PSI))
        spans, since, failures = _read(x, c, gaps)
        data = _Gaps(spans, since, PSI[psi])
        best, objective = data.search()
        fitted = -float(best.fun)
        limit = data.best_limit()
        if not fitted > limit + LIMIT_MARGIN:
            raise ValueError(
                "The times between failures are no more scattered than "
                "exponential times with a steadily changing mean would "
                "be, so the Littlewood-Verrall model's alpha has no "
                "estimate: the likelihood keeps rising as alpha grows "
                "without bound (log-likelihood {:.6g} at the search's "
                "end, {:.6g} for the limit). Its uncertainty about each "
                "rate is not needed for these data. Fit a model without "
                "it and compare with aic(): MusaOkumoto or GoelOkumoto "
                "(Poisson processes) or JelinskiMoranda.".format(fitted, limit)
            )
        res, verified = verify_or_polish(
            objective, best, float(data.n), numerical=True
        )
        if not verified:
            warn_unverified("The Littlewood-Verrall fit")
        alpha, first, last = np.exp(res.x)
        beta0, beta1 = data.betas(first, last)
        model = LittlewoodVerrallModel()
        model.dist = self
        model.psi_form = psi
        model.params = np.array([alpha, beta0, beta1])
        model.n = data.n
        model.how = "MLE"
        model.maximum = "verified" if verified else "unverified"
        model._data = data
        model._u = np.asarray(res.x, dtype=float)
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
        psi: str = "linear",
    ) -> "LittlewoodVerrallModel":
        """
        Fit the model to one campaign held in a ``pandas.DataFrame``:
        ``x_col`` the column of failure times (or of times between
        failures, with ``gaps=True``), ``c_col`` that of 0 (a failure)
        and 1 (the end of testing), if there is one. See :meth:`fit`.
        """
        for name, col in (("x_col", x_col), ("c_col", c_col)):
            if col is not None and col not in df.columns:
                raise ValueError(
                    "{}={!r} is not a column of the DataFrame; its "
                    "columns are {}".format(name, col, list(df.columns))
                )
        c = None if c_col is None else df[c_col].to_numpy()
        return self.fit(df[x_col].to_numpy(), c, gaps=gaps, psi=psi)

    @takes_unit
    def from_params(
        self, params: ArrayLike, n: int = 0, psi: str = "linear"
    ) -> "LittlewoodVerrallModel":
        """
        The model at given parameters, without data.

        Parameters
        ----------
        params : array_like
            ``[alpha, beta0, beta1]``.
        n : int, optional
            The failures already seen: the model then answers for the
            next one, number ``n + 1``. Default 0.
        psi : {'linear', 'quadratic'}, optional
            The form of ``psi``.

        Examples
        --------
        >>> from surpyval.recurrent import LittlewoodVerrall
        >>> model = LittlewoodVerrall.from_params([3.0, 10.0, 2.0], n=20)
        >>> model.psi(21)
        52.0
        >>> round(model.next_failure_mean(), 1)
        26.0
        """
        check_option("psi", psi, tuple(PSI))
        values = np.asarray(params, dtype=float)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError(
                "params must be [alpha, beta0, beta1], three finite "
                "numbers; got {}".format(np.asarray(params).tolist())
            )
        if not (isinstance(n, (int, np.integer)) and n >= 0):
            raise ValueError(
                "n is the number of failures already seen, a whole "
                f"number of at least 0; got {n!r}"
            )
        if not values[0] > 0:
            raise ValueError(
                f"alpha is a gamma shape, which is positive; got {values[0]!r}"
            )
        model = LittlewoodVerrallModel()
        model.dist = self
        model.psi_form = psi
        model.params = values
        model.n = int(n)
        model.how = "from_params"
        for i in (1, model.n + 1):
            if not model.psi(i) > 0:
                raise ValueError(
                    "psi({i}) = beta0 + beta1 * {i}{power} = {value:g} is "
                    "not positive: psi is a gamma rate, which must be "
                    "above 0 at every failure from the first to the "
                    "next (number {last}). Raise beta0, or beta1.".format(
                        i=i,
                        power="" if PSI[psi] == 1 else "**2",
                        value=model.psi(i),
                        last=model.n + 1,
                    )
                )
        return model


class LittlewoodVerrallModel(SerialisableMixin):
    """
    A fitted Littlewood-Verrall model.

    Attributes
    ----------
    params : numpy.ndarray
        ``[alpha, beta0, beta1]``: the shape of the rates' gamma
        distribution, and ``psi(i) = beta0 + beta1 i`` (or ``i**2``), its
        rate at failure ``i``.
    psi_form : str
        ``"linear"`` or ``"quadratic"``.
    n : int
        The failures seen.

    The software has no "number of faults" in this model: each failure's
    rate is drawn afresh, lower on average than the one before. What it
    gives is the distribution of the time to the next failure (number
    ``n + 1``), :meth:`next_failure_sf`, :meth:`next_failure_qf` and
    :meth:`next_failure_mean`, and of the rate now,
    :meth:`current_rate` and :meth:`current_rate_qf`.

    Examples
    --------
    >>> from surpyval.recurrent import LittlewoodVerrall
    >>> truth = LittlewoodVerrall.from_params([3.0, 10.0, 2.0])
    >>> gaps = truth.random(200, random_state=1)
    >>> model = LittlewoodVerrall.fit(gaps, gaps=True)
    >>> round(model.next_failure_qf(0.5), 1)   # the median time to the next
    106.3
    >>> round(model.current_rate(), 4)
    0.007

    Bounds: ``param_cb`` by the likelihood ratio for ``alpha`` (whose
    likelihood is lopsided, and whose upper bound is ``inf`` where the
    data do not rule out rates with no scatter) and by the delta method
    (Wald) for all three. The predictions have no confidence bounds: see
    :meth:`next_failure_sf`.
    """

    maximum: str = "not applicable"
    dist: Any
    params: np.ndarray
    psi_form: str
    n: int
    how: str
    x: np.ndarray
    _data: _Gaps
    _u: np.ndarray
    _covariance: np.ndarray

    @property
    def parameter_names(self) -> list[str]:
        return ["alpha", "beta0", "beta1"]

    def _parameter_bounds(self) -> list:
        return [(0, None), (None, None), (None, None)]

    @property
    def alpha(self) -> float:
        return float(self.params[0])

    def psi(self, i: ArrayLike) -> Any:
        """``psi`` at failure number ``i``: ``beta0 + beta1 i`` (or
        ``i**2``), the rate parameter of that failure's gamma."""
        index = np.asarray(i, dtype=float) ** PSI[self.psi_form]
        out = self.params[1] + self.params[2] * index
        return float(out) if np.ndim(out) == 0 else out

    def __repr__(self) -> str:
        names = self.parameter_names
        lines = [
            "Littlewood-Verrall SurPyval Model",
            "=================================",
            "Fitted by           : "
            + (
                "MLE" if self.how == "MLE" else "given parameters (not fitted)"
            ),
            "psi(i)              : beta0 + beta1 i"
            + ("" if self.psi_form == "linear" else "**2"),
            f"Failures            : {self.n}",
            "Parameters          :",
        ]
        lines += [f"{n:>10}: {p}" for n, p in zip(names, self.params)]
        lines.append(f"Median to next      : {self.next_failure_qf(0.5)}")
        return "\n".join(lines)

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise the model to a plain, JSON-serialisable dict: the
        parameters, the form of ``psi``, the failures seen and, for a
        fitted model, its parameter covariance. The data are not stored,
        so the likelihood (``aic``, the likelihood-ratio bound) needs a
        refit.

        See Also
        --------
        from_dict, to_json, from_json
        """
        out: dict = {
            "model": "LittlewoodVerrallModel",
            "params": [float(v) for v in self.params],
            "psi": self.psi_form,
            "n": int(self.n),
            "how": self.how,
            "covariance": (
                self.covariance().tolist() if self._has_covariance() else None
            ),
            **maximum_entry(self.maximum),
        }
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "LittlewoodVerrallModel":
        """
        Rebuild a model from a :meth:`to_dict` dictionary.

        See Also
        --------
        to_dict, to_json, from_json
        """
        require_model_tag(
            model_dict, "LittlewoodVerrallModel", "a Littlewood-Verrall model"
        )
        out = LittlewoodVerrall.from_params(
            model_dict["params"],
            n=int(model_dict["n"]),
            psi=model_dict.get("psi", "linear"),
        )
        out.how = model_dict.get("how", "from_params")
        out.maximum = restored_maximum(model_dict)
        if model_dict.get("covariance") is not None:
            out._covariance = np.asarray(model_dict["covariance"], dtype=float)
        return out

    # -- the next failure -----------------------------------------------------

    def next_failure_sf(self, x: ArrayLike) -> np.ndarray:
        """
        The chance that the software runs a further ``x`` from the last
        failure without another: ``(psi / (psi + x))**alpha`` with ``psi
        = psi(n + 1)``, a Pareto survival function. (The name the other
        models of failures use.)

        It already allows for not knowing the rate: it is the
        exponential survival averaged over the gamma distribution of the
        rate, so it has a heavier tail than any one exponential. There
        are no confidence bounds on it; the uncertainty left out is that
        of ``alpha`` and ``psi`` themselves (see :meth:`param_cb`).

        Examples
        --------
        >>> from surpyval.recurrent import LittlewoodVerrall
        >>> model = LittlewoodVerrall.from_params([3.0, 10.0, 2.0], n=20)
        >>> model.next_failure_sf([0.0, 26.0, 260.0]).round(4)
        array([1.    , 0.2963, 0.0046])
        """
        values = np.asarray(x, dtype=float)
        if np.any(values < 0):
            raise ValueError(
                "x is time ahead of now, so it cannot be negative; got "
                "{}".format(values[values < 0][:5])
            )
        psi = self.psi(self.n + 1)
        return np.exp(-self.alpha * np.log1p(values / psi))

    def _next_gap(self, gap: float) -> tuple[float, float]:
        """``(sf, log density)`` of the next gap at ``gap`` (for
        ``compare_predictions``)."""
        psi = self.psi(self.n + 1)
        log_sf = -self.alpha * np.log1p(gap / psi)
        return float(np.exp(log_sf)), float(
            np.log(self.alpha) - np.log(psi + gap) + log_sf
        )

    def next_failure_qf(self, p: ArrayLike) -> Any:
        """
        The time by which the next failure comes with probability ``p``:
        ``psi ((1 - p)**(-1 / alpha) - 1)``. ``p = 0.5`` is the median
        time to the next failure, which exists whatever ``alpha`` is.

        Examples
        --------
        >>> from surpyval.recurrent import LittlewoodVerrall
        >>> model = LittlewoodVerrall.from_params([3.0, 10.0, 2.0], n=20)
        >>> round(model.next_failure_qf(0.5), 3)
        13.516
        """
        values = np.asarray(p, dtype=float)
        if np.any((values < 0) | (values > 1)):
            raise ValueError(
                "p is a probability, between 0 and 1; got {}".format(
                    values[(values < 0) | (values > 1)][:5]
                )
            )
        with np.errstate(divide="ignore"):
            out = self.psi(self.n + 1) * np.expm1(
                -np.log1p(-values) / self.alpha
            )
        return float(out) if out.ndim == 0 else out

    def next_failure_mean(self) -> float:
        """
        The mean time to the next failure, ``psi(n + 1) / (alpha - 1)``.
        It is infinite where ``alpha <= 1`` (the Pareto's tail is too
        heavy for a mean), and that is refused: use the median,
        ``next_failure_qf(0.5)``.
        """
        if not self.alpha > 1:
            raise ValueError(
                "The mean time to the next failure is infinite: alpha = "
                "{:.4g} is not above 1, and a Pareto time with such a "
                "tail has no mean. Use the median, next_failure_qf(0.5), "
                "or another quantile.".format(self.alpha)
            )
        return float(self.psi(self.n + 1) / (self.alpha - 1.0))

    def mtbf(self) -> float:
        """The mean time to the next failure: :meth:`next_failure_mean`
        (refused where it is infinite)."""
        return self.next_failure_mean()

    def current_rate(self) -> float:
        """The expected failure rate now, ``alpha / psi(n + 1)``: the
        mean of the gamma distribution of the next failure's rate."""
        return float(self.alpha / self.psi(self.n + 1))

    def current_rate_qf(self, p: ArrayLike) -> Any:
        """
        Quantiles of the failure rate now: those of its gamma
        distribution (shape ``alpha``, rate ``psi(n + 1)``). The model
        holds that the rate is not known exactly, even with its
        parameters given; ``current_rate_qf([0.05, 0.95])`` is a 90%
        interval for it.

        Examples
        --------
        >>> from surpyval.recurrent import LittlewoodVerrall
        >>> model = LittlewoodVerrall.from_params([3.0, 10.0, 2.0], n=20)
        >>> model.current_rate_qf([0.05, 0.5, 0.95]).round(4)
        array([0.0157, 0.0514, 0.1211])
        """
        out = gamma_dist.ppf(
            np.asarray(p, dtype=float),
            self.alpha,
            scale=1.0 / self.psi(self.n + 1),
        )
        return float(out) if np.ndim(out) == 0 else out

    def random(self, size: int, random_state: Any = None) -> np.ndarray:
        """
        Draw the times between the first ``size`` failures of a campaign
        from the model (from failure 1, whatever ``n`` is): each rate
        from its gamma, then the time from that rate.

        Parameters
        ----------
        size : int
            The number of failures.
        random_state : int or numpy.random.Generator, optional
            Seed or generator for a reproducible draw.
        """
        from surpyval.utils.rng import as_generator

        if not (isinstance(size, (int, np.integer)) and size >= 1):
            raise ValueError(
                f"size is the number of failures to draw, at least 1; got "
                f"{size!r}"
            )
        psi = np.asarray(self.psi(np.arange(1, size + 1)), dtype=float)
        if np.any(psi <= 0):
            raise ValueError(
                "psi(i) is not positive for every failure up to {}: the "
                "model is not defined that far.".format(size)
            )
        rng = as_generator(random_state)
        rates = rng.gamma(self.alpha, 1.0 / psi)
        return rng.exponential(1.0 / rates)

    # -- the likelihood ------------------------------------------------------

    def _fitted(self, what: str) -> _Gaps:
        if not hasattr(self, "_data"):
            raise ValueError(
                "{} needs the likelihood of a fitted model; this one was "
                "built from its parameters or loaded from a dict. Fit it "
                "with LittlewoodVerrall.fit.".format(what)
            )
        return self._data

    @property
    def log_likelihood(self) -> float:
        """The log-likelihood at the fitted parameters."""
        data = self._fitted("log_likelihood")
        return data.log_likelihood(*np.exp(self._u))

    def neg_ll(self) -> float:
        """The negative log-likelihood at the fitted parameters."""
        return -self.log_likelihood

    def aic(self) -> float:
        """Akaike's criterion, ``2 * 3 + 2 neg_ll()``; comparable with
        the ``aic()`` of the other models fitted to the same failure
        times."""
        return 6.0 + 2.0 * self.neg_ll()

    def aic_c(self) -> float:
        """The small-sample AIC, with the failures as the sample size."""
        if self.n <= 4:
            return float(np.inf)
        return self.aic() + 24.0 / (self.n - 4.0)

    def bic(self) -> float:
        """The Bayesian criterion, ``3 log(n) + 2 neg_ll()``."""
        return 3.0 * float(np.log(self.n)) + 2.0 * self.neg_ll()

    def _has_covariance(self) -> bool:
        return hasattr(self, "_covariance") or hasattr(self, "_data")

    def covariance(self) -> np.ndarray:
        """
        The covariance of ``[alpha, beta0, beta1]``: the inverse of the
        observed information (a numerical Hessian of the log-likelihood
        in the search's parameters, carried to these by the delta
        method).
        """
        if hasattr(self, "_covariance"):
            return self._covariance
        data = self._fitted("covariance")

        def objective(u: np.ndarray) -> float:
            return -data.log_likelihood(*np.exp(u))

        hessian = numerical_hessian(objective, self._u)
        try:
            cov_u = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            raise ValueError(
                "The Littlewood-Verrall fit has no covariance: the "
                "likelihood is flat in some direction at the estimate "
                "(alpha is likely unbounded: see param_cb('alpha'))."
            ) from None
        alpha, first, last = np.exp(self._u)
        span = float((data.n + 1.0) ** data.power - 1.0)
        # d(alpha, beta0, beta1) / d(log alpha, log first, log last)
        jac = np.array(
            [
                [alpha, 0.0, 0.0],
                [0.0, first * (1.0 + 1.0 / span), -last / span],
                [0.0, -first / span, last / span],
            ]
        )
        self._covariance = jac @ cov_u @ jac.T
        return self._covariance

    def standard_errors(self) -> np.ndarray:
        """The standard errors of ``[alpha, beta0, beta1]``."""
        with np.errstate(invalid="ignore"):
            return np.sqrt(np.diag(self.covariance()))

    # -- bounds --------------------------------------------------------------

    def param_cb(
        self,
        name: str,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str | None = None,
    ) -> Any:
        """
        Confidence bounds on a parameter.

        Parameters
        ----------
        name : {'alpha', 'beta0', 'beta1'}
            The parameter.
        alpha_ci : float, optional
            The total tail probability of the bound(s). Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds are an array ``[lower, upper]``.
        method : {'lr', 'wald'}, optional
            ``"lr"``: the likelihood ratio, for ``alpha`` only (its
            default): the values of ``alpha`` whose profile likelihood
            is within the chi-squared quantile of the maximum. The upper
            bound is ``inf`` where the model's limit, rates with no
            scatter, is inside. For ``beta0`` and ``beta1`` it is
            refused (not built: use ``"wald"``).

            ``"wald"``: the delta method, from :meth:`covariance`
            (the default for ``beta0`` and ``beta1``): on the log scale
            for ``alpha``, which is positive. For ``alpha`` it is a poor
            bound where the likelihood is lopsided; compare it with
            ``"lr"``.

        Examples
        --------
        >>> from surpyval.recurrent import LittlewoodVerrall
        >>> truth = LittlewoodVerrall.from_params([3.0, 10.0, 2.0])
        >>> gaps = truth.random(200, random_state=1)
        >>> model = LittlewoodVerrall.fit(gaps, gaps=True)
        >>> model.param_cb("alpha").round(2)
        array([ 2.38, 17.7 ])
        >>> model.param_cb("alpha", method="wald").round(2)
        array([ 2.01, 10.73])
        """
        names = self.parameter_names
        check_option("name", name, tuple(names))
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        chosen = method or ("lr" if name == "alpha" else "wald")
        check_option("method", chosen, ("lr", "wald"))
        if chosen == "lr":
            if name != "alpha":
                raise ValueError(
                    "param_cb has a likelihood-ratio bound for alpha "
                    "only: for {} it is not built (psi must stay positive "
                    "along the profile). Use method='wald'.".format(name)
                )
            return self._alpha_lr(alpha_ci, bound)
        k = names.index(name)
        tail = alpha_ci / 2.0 if bound == "two-sided" else alpha_ci
        z = float(norm.ppf(1.0 - tail))
        estimate = float(self.params[k])
        se = float(self.standard_errors()[k])
        if name == "alpha":
            pair = (
                estimate * np.exp(-z * se / estimate),
                estimate * np.exp(z * se / estimate),
            )
        else:
            pair = (estimate - z * se, estimate + z * se)
        if bound == "two-sided":
            return np.array(pair)
        return float(pair[0] if bound == "lower" else pair[1])

    def _alpha_lr(self, alpha_ci: float, bound: str) -> Any:
        data = self._fitted("A likelihood-ratio bound")
        level = 1.0 - alpha_ci if bound == "two-sided" else 1.0 - 2 * alpha_ci
        drop = 0.5 * float(chi2.ppf(level, 1)) if level > 0 else 0.0
        floor = self.log_likelihood - drop

        def gap(log_alpha: float) -> float:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                best, _ = data.search(alpha=float(np.exp(log_alpha)))
            return -float(best.fun) - floor

        at = float(self._u[0])
        lo = at
        lower = 0.0
        for _ in range(60):
            lo -= 1.0
            if gap(lo) < 0:
                lower = float(np.exp(brentq(gap, lo, lo + 1.0, xtol=1e-8)))
                break
        if data.best_limit() >= floor:
            upper = np.inf
        else:
            hi = at
            for _ in range(60):
                hi += 1.0
                if gap(hi) < 0:
                    break
            upper = float(np.exp(brentq(gap, hi - 1.0, hi, xtol=1e-8)))
        if bound == "two-sided":
            return np.array([lower, upper])
        return lower if bound == "lower" else upper
