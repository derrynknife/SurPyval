"""Reliability growth of one-shot items: success/fail data by stage.

A one-shot item (a missile, an airbag, an engine start, a medical device
that must work first time) either succeeds or fails on each trial. A
development programme tests it in stages (or configurations), fixing what
failed between them, and the chance of success should rise from stage to
stage. These models fit that rise to the successes and trials of each
stage, and project the next stage:

- :class:`CrowAMSAADiscrete`, the AMSAA discrete tracking model
  (MIL-HDBK-189C section 6.2.4): the expected number of failures in the
  first ``T`` trials is :math:`\\lambda T^\\beta`, Duane's learning curve
  counted in trials, so configuration ``i``, trials
  :math:`(T_{i-1}, T_i]`, fails with chance
  :math:`f_i = \\lambda (T_i^\\beta - T_{i-1}^\\beta) / N_i`;
- :class:`LloydLipow`, :math:`R_k = R_\\infty - \\alpha / k` at stage ``k``
  (Lloyd and Lipow 1962);
- :class:`GompertzGrowth`, :math:`R(T) = a b^{c^T}`, and
  :class:`ModifiedGompertzGrowth`, :math:`R(T) = d + a b^{c^T}`, S-shaped
  growth (Virene 1968).

Each is fitted by maximum likelihood, the successes of each stage
binomial with that stage's reliability (``method="mle"``), or, as is
traditional for Lloyd-Lipow and Gompertz, by least squares on the observed
reliabilities (``method="least_squares"``: the only fit for data given as
reliabilities alone). Bounds on a reliability are Wald bounds on its
logit (from the observed information, or for least squares from the
residual variance), and, for maximum likelihood fits, likelihood-ratio
bounds (the reliability profiled).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
import pandas as pd
from autograd import hessian, jacobian
from scipy import optimize, stats

from surpyval.utils.surpyval_data import refuses_surpyval_data_with
from surpyval.utils.validation import check_alpha_ci

#: What a growth model's fit says when given a SurpyvalData.
_COUNTS_NOT_ROWS = (
    "A discrete growth model takes success/fail counts by stage, which "
    "have no rows in a SurpyvalData: pass failures= and trials= (and "
    "stage=) as arrays, one value a stage, or reliability= alone."
)

__all__ = [
    "CrowAMSAADiscrete",
    "DiscreteGrowthModel",
    "GompertzGrowth",
    "LloydLipow",
    "ModifiedGompertzGrowth",
]

_BOUNDS = ("two-sided", "lower", "upper")


class _Data:
    """The stages as arrays (the fits read them many times)."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.stage = frame["stage"].to_numpy(float)
        self.trials = frame["trials"].to_numpy(float)
        self.failures = frame["failures"].to_numpy(float)
        self.succ = self.trials - self.failures
        self.reliability = frame["reliability"].to_numpy(float)
        self.counts = bool(np.all(np.isfinite(self.trials)))

    def __len__(self) -> int:
        return len(self.stage)


def _expit(u: Any) -> Any:
    return 1.0 / (1.0 + anp.exp(-u))


def _logit(p: Any) -> Any:
    return np.log(p) - np.log1p(-p)


def _binomial_nll(r: Any, successes: Any, failures: Any) -> Any:
    """Minus the binomial log-likelihood (no constant) of reliabilities
    ``r``; ``inf`` where a reliability with outcomes against it is 0 or
    1."""
    inside = anp.all(r > 0) and anp.all(r < 1)
    if not inside:
        ok_low = anp.all((r > 0) | (successes == 0))
        ok_high = anp.all((r < 1) | (failures == 0))
        if not (ok_low and ok_high):
            return anp.inf
        r = anp.clip(r, 1e-300, 1 - 1e-16)
    return -anp.sum(successes * anp.log(r) + failures * anp.log1p(-r))


class DiscreteGrowthModel:
    """A fitted growth model of success/fail data by stage: returned by
    the ``fit`` of :class:`CrowAMSAADiscrete`, :class:`LloydLipow`,
    :class:`GompertzGrowth` and :class:`ModifiedGompertzGrowth`.

    Attributes
    ----------
    name : str
        The model.
    params : numpy.ndarray
        The fitted parameters, in the order of ``param_names``.
    param_names : tuple of str
    method : str
        ``"mle"`` or ``"least_squares"``.
    data : pandas.DataFrame
        One row per stage: ``stage``, ``trials`` and ``failures`` (``nan``
        for data given as reliabilities), and the observed
        ``reliability``.
    cov : numpy.ndarray
        The parameters' covariance: the inverse observed information for
        a maximum likelihood fit; the residual variance times
        :math:`(J^T J)^{-1}` for least squares.
    log_likelihood : float
        The binomial log-likelihood at the fit (``nan`` for data given as
        reliabilities).

    Examples
    --------
    >>> from surpyval.recurrent import LloydLipow
    >>> model = LloydLipow.fit(
    ...     failures=[6, 5, 4, 3, 3, 2], trials=[10, 10, 10, 10, 10, 10]
    ... )
    >>> model.name, model.method, model.param_names
    ('Lloyd-Lipow', 'mle', ('R_inf', 'alpha'))
    >>> round(model.reliability(6), 3), round(model.log_likelihood, 3)
    (0.725, -37.988)
    >>> model.param_cb().round(3)
           value  lower  upper
    param
    R_inf  0.798  0.523  0.934
    alpha  0.437  0.169  1.130
    """

    def __init__(
        self,
        fitter: _GrowthFitter,
        params: np.ndarray,
        method: str,
        data: _Data,
        cov: np.ndarray,
    ) -> None:
        self._fitter = fitter
        self.name = fitter.name
        self.param_names = fitter.param_names
        self.params = np.asarray(params, dtype=float)
        self.method = method
        self._d = data
        self.data = data.frame
        self.cov = cov
        if data.counts:
            r = self._fitter._stage_reliability(self.params, data)
            self.log_likelihood = -float(
                _binomial_nll(
                    r,
                    data.succ,
                    data.failures,
                )
            )
        else:
            self.log_likelihood = np.nan

    # -- point values ------------------------------------------------------

    def reliability(self, stage: npt.ArrayLike) -> Any:
        """The fitted reliability (chance of success on a trial) at
        ``stage``, the same scale as the stages given to ``fit``: for
        :class:`CrowAMSAADiscrete`, the configuration number (1, 2, ...)
        of a fitted configuration."""
        s = np.asarray(stage, dtype=float)
        value = np.asarray(self._fitter._r(self.params, s, self._d))
        return value.item() if value.ndim == 0 else value

    def next_reliability(self, trials: float | None = None) -> float:
        """The projected reliability of the next stage: the stage after
        the last one fitted (the last stage plus one); for
        :class:`CrowAMSAADiscrete`, a next configuration of ``trials``
        trials (by default as many as the last), :math:`1 - \\lambda
        ((T_K + N)^\\beta - T_K^\\beta) / N`."""
        return float(self._fitter._next(self.params, self._d, trials))

    @property
    def limit(self) -> float:
        """The reliability the growth tends to with ever more stages:
        :math:`R_\\infty` (Lloyd-Lipow), ``a`` (Gompertz), ``a + d``
        (modified Gompertz); 1 for :class:`CrowAMSAADiscrete` with
        :math:`\\beta < 1`, which grows without limit below 1."""
        return float(self._fitter._limit(self.params))

    # -- bounds ------------------------------------------------------------

    def param_cb(self, alpha_ci: float = 0.05) -> pd.DataFrame:
        """Wald confidence bounds, two-sided at ``1 - alpha_ci``, on each
        parameter, from ``cov``: on the logit of a parameter in (0, 1)
        (a reliability, ``b``, ``c``) and on the log of any other
        positive parameter, so the bounds stay in range."""
        check_alpha_ci(alpha_ci)
        z = stats.norm.ppf(1 - alpha_ci / 2)
        se = np.sqrt(np.diag(self.cov))
        unit = self._fitter.unit_params
        rows = []
        for name, p, sd in zip(self.param_names, self.params, se):
            if name in unit and 0 < p < 1:
                w = z * sd / (p * (1 - p))
                lo = p / (p + (1 - p) * np.exp(w))
                hi = p / (p + (1 - p) * np.exp(-w))
            elif p > 0:
                lo, hi = p * np.exp(-z * sd / p), p * np.exp(z * sd / p)
            else:
                lo, hi = p - z * sd, p + z * sd
            rows.append({"param": name, "value": p, "lower": lo, "upper": hi})
        return pd.DataFrame(rows).set_index("param")

    def reliability_cb(
        self,
        stage: npt.ArrayLike,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "wald",
    ) -> np.ndarray:
        """Confidence bounds on the reliability at ``stage``.

        ``method="wald"``: the delta method on the logit of the
        reliability, so the bounds stay in (0, 1):
        :math:`\\hat R / [\\hat R + (1 - \\hat R) e^{\\pm z\\,
        \\mathrm{se}(\\hat R) / (\\hat R (1 - \\hat R))}]`.
        ``method="lr"``: the likelihood-ratio bounds, the reliabilities at
        which the profile log-likelihood falls
        :math:`\\chi^2_{1}(1 - \\alpha) / 2` below its maximum
        (``1 - 2 alpha_ci`` for a one-sided bound), maximum likelihood fits
        only.

        Parameters
        ----------
        stage : array_like
            As :meth:`reliability` takes it.
        alpha_ci : float, optional
            One minus the confidence level, by default 0.05.
        bound : str, optional
            ``"two-sided"`` (rows of lower and upper), ``"lower"`` or
            ``"upper"``.
        method : str, optional
            ``"wald"`` (the default) or ``"lr"``.

        Returns
        -------
        numpy.ndarray
            ``(..., 2)`` for two-sided bounds, else the shape of
            ``stage``.
        """
        check_alpha_ci(alpha_ci)
        if bound not in _BOUNDS:
            raise ValueError(f"bound must be one of {_BOUNDS}, got {bound!r}.")
        s = np.asarray(stage, dtype=float)
        flat = s.reshape(-1)
        tail = alpha_ci / 2 if bound == "two-sided" else alpha_ci
        if method == "wald":
            out = np.array([self._wald(self._at(si), tail) for si in flat])
        elif method == "lr":
            if self.method != "mle":
                raise ValueError(
                    "method='lr' needs a likelihood: this model was fitted "
                    "by least squares. Refit with method='mle' (from "
                    "failures and trials), or use method='wald'."
                )
            out = np.array([self._lr(si, tail) for si in flat])
        else:
            raise ValueError(f"method must be 'wald' or 'lr', got {method!r}.")
        return _shape(out, s.shape, bound)

    def next_reliability_cb(
        self,
        trials: float | None = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> np.ndarray:
        """Wald bounds (on the logit) on :meth:`next_reliability`."""
        check_alpha_ci(alpha_ci)
        if bound not in _BOUNDS:
            raise ValueError(f"bound must be one of {_BOUNDS}, got {bound!r}.")
        tail = alpha_ci / 2 if bound == "two-sided" else alpha_ci
        out = self._wald(
            lambda p: self._fitter._next(p, self._d, trials), tail
        )
        return _shape(np.array([out]), (), bound)

    def _at(self, stage: float) -> Callable[[Any], Any]:
        def r(params: Any) -> Any:
            return self._fitter._r(params, stage, self._d)

        return r

    def _wald(self, f: Callable[[Any], Any], tail: float) -> np.ndarray:
        r = float(f(self.params))
        grad = np.asarray(jacobian(f)(self.params), dtype=float)
        se = float(np.sqrt(max(grad @ self.cov @ grad, 0.0)))
        if not 0 < r < 1:
            return np.array([r, r])
        z = stats.norm.ppf(1 - tail)
        w = z * se / (r * (1 - r))
        return np.array(
            [r / (r + (1 - r) * np.exp(w)), r / (r + (1 - r) * np.exp(-w))]
        )

    def _lr(self, stage: float, tail: float) -> np.ndarray:
        fitter, data = self._fitter, self._d
        best = -self.log_likelihood
        r_hat = float(fitter._r(self.params, stage, data))
        drop = stats.chi2.ppf(1 - 2 * tail, 1) / 2

        def gap(r: float) -> float:
            return (
                fitter._profile_nll(r, stage, data, self.params) - best - drop
            )

        lo = hi = r_hat
        eps = 1e-10
        lower = upper = np.nan
        if gap(eps) > 0:
            lower = optimize.brentq(gap, eps, lo, xtol=1e-10)
        else:
            lower = 0.0
        if gap(1 - eps) > 0:
            upper = optimize.brentq(gap, hi, 1 - eps, xtol=1e-10)
        else:
            upper = 1.0
        return np.array([lower, upper])

    def handbook_lower_bound(self, alpha_ci: float = 0.05) -> float:
        """MIL-HDBK-189C's approximate lower ``1 - alpha_ci`` bound on the
        last configuration's reliability, for :class:`CrowAMSAADiscrete`
        (section 6.2.4.8.1):
        :math:`1 - (1 - \\hat R_K)\\, \\chi^2_{1-\\alpha}(2n + 2) / (2n)`,
        ``n`` the total failures. It treats the last configuration's chance
        of failure as a rate with ``n`` failures behind it, as the usual
        approximation to Crow's bound on an MTBF does. The other models
        refuse it: use :meth:`reliability_cb`."""
        check_alpha_ci(alpha_ci)
        if not isinstance(self._fitter, CrowAMSAADiscrete_):
            raise ValueError(
                "handbook_lower_bound is MIL-HDBK-189C's bound for "
                "CrowAMSAADiscrete; for this model use reliability_cb(..., "
                "bound='lower')."
            )
        n = float(self.data["failures"].sum())
        r = float(self.reliability(len(self.data)))
        chi = stats.chi2.ppf(1 - alpha_ci, 2 * n + 2)
        return float(1 - (1 - r) * chi / (2 * n))

    # -- display -----------------------------------------------------------

    def plot(
        self,
        ax: Any = None,
        *,
        alpha_ci: float = 0.05,
        stages: int = 1,
    ) -> Any:
        """Plot the observed reliability of each stage, the fitted curve
        with its two-sided Wald band, and the projection ``stages`` stages
        past the last. Returns the axes."""
        import matplotlib.pyplot as plt

        from surpyval import plotting

        if ax is None:
            _, ax = plt.subplots()
        plotting.style_axes(ax)
        data = self.data
        s = data.stage
        ax.scatter(
            s, data["reliability"], color="grey", zorder=3, label="Observed"
        )
        if self._fitter.continuous_stage:
            grid = np.linspace(s.min(), s.max() + stages, 200)
        else:
            grid = np.arange(1, len(s) + 1, dtype=float)
        band = self.reliability_cb(grid, alpha_ci=alpha_ci)
        color = plotting.next_color(ax)
        (line,) = ax.plot(
            grid, self.reliability(grid), color=color, label=self.name
        )
        ax.fill_between(
            grid,
            band[:, 0],
            band[:, 1],
            color=line.get_color(),
            alpha=0.2,
            linewidth=0,
            label=f"{1 - alpha_ci:.0%} bounds",
        )
        nxt = self.next_reliability()
        ax.scatter(
            [s.max() + 1],
            [nxt],
            marker="D",
            color=line.get_color(),
            label="Next stage",
        )
        ax.set_xlabel("Stage")
        ax.set_ylabel("Reliability")
        plotting.style_legend(ax)
        return ax

    def __repr__(self) -> str:
        header = f"{self.name} growth model ({self.method})"
        lines = [header, "=" * len(header)]
        for name, value in zip(self.param_names, self.params):
            lines.append(f"{name:<7}: {value:.6g}")
        lines.append(f"Stages : {len(self.data)}")
        lines.append(f"Next   : {self.next_reliability():.6g}")
        return "\n".join(lines)


def _shape(out: np.ndarray, shape: tuple, bound: str) -> np.ndarray:
    if bound == "two-sided":
        return out.reshape(shape + (2,))
    column = 0 if bound == "lower" else 1
    result = out[:, column].reshape(shape)
    return result.item() if result.ndim == 0 else result


# -- the fitters ----------------------------------------------------------


class _GrowthFitter:
    """What each growth model supplies; the fitting is shared."""

    name: str = ""
    param_names: tuple[str, ...] = ()
    continuous_stage = True
    least_squares = True
    unit_params: tuple[str, ...] = ()

    # The model: reliability at stage s, the parameters natural.
    def _curve(self, params: Any, s: Any) -> Any:
        raise NotImplementedError

    def _r(self, params: Any, s: Any, data: _Data) -> Any:
        return self._curve(params, s)

    def _stage_reliability(self, params: Any, data: _Data) -> Any:
        return self._curve(params, data.stage)

    def _next(self, params: Any, data: _Data, trials: Any) -> Any:
        if trials is not None:
            raise ValueError(
                f"{self.name}'s next stage is the last plus one: trials "
                "is for CrowAMSAADiscrete only."
            )
        return self._curve(params, float(data.stage.max()) + 1.0)

    def _limit(self, params: Any) -> float:
        raise NotImplementedError

    # Unconstrained coordinates u <-> natural parameters.
    def _to_params(self, u: Any, data: _Data) -> Any:
        raise NotImplementedError

    def _from_params(self, params: Any, data: _Data) -> np.ndarray:
        raise NotImplementedError

    def _starts(self, data: _Data) -> list[np.ndarray]:
        raise NotImplementedError

    # Profile: the parameters with R(s) = r, from the other coordinates.
    def _pin(self, v: Any, r: float, s: float, data: _Data) -> Any:
        raise NotImplementedError

    def _unpin(self, params: Any, s: float, data: _Data) -> Any:
        raise NotImplementedError

    # -- fitting -------------------------------------------------------------

    @refuses_surpyval_data_with(_COUNTS_NOT_ROWS, argument="failures")
    def fit(
        self,
        failures: npt.ArrayLike | None = None,
        trials: npt.ArrayLike | None = None,
        stage: npt.ArrayLike | None = None,
        *,
        reliability: npt.ArrayLike | None = None,
        method: str | None = None,
    ) -> DiscreteGrowthModel:
        """Fit the growth model to success/fail data by stage.

        Parameters
        ----------
        failures : array_like, optional
            The failures of each stage, whole numbers.
        trials : array_like, optional
            The trials of each stage, whole numbers, at least 1.
        stage : array_like, optional
            The stage of each row, increasing: 1, 2, ... by default (the
            stage number of Lloyd-Lipow; any time, such as months, for
            Gompertz). Not taken by :class:`CrowAMSAADiscrete`, whose time
            is the cumulative trials.
        reliability : array_like, optional
            Keyword only: observed reliabilities (fractions in [0, 1]) in
            place of failures and trials; fitted by least squares.
        method : str, optional
            Keyword only: ``"mle"`` (binomial maximum likelihood, the
            default with failures and trials) or ``"least_squares"`` (on
            the observed reliabilities, the default and only fit with
            ``reliability``).

        Returns
        -------
        DiscreteGrowthModel

        Raises
        ------
        ValueError
            For data in which every trial succeeded (or every trial
            failed), which show no growth; for a maximum likelihood fit
            without trials; and for malformed counts or stages.
        """
        data = self._data(failures, trials, stage, reliability)
        counts = data.counts
        if method is None:
            method = "mle" if counts else "least_squares"
        if method not in ("mle", "least_squares"):
            raise ValueError(
                "method must be 'mle' or 'least_squares', got " f"{method!r}."
            )
        if method == "mle" and not counts:
            raise ValueError(
                "method='mle' needs the failures and trials of each stage; "
                "with reliabilities alone use method='least_squares'."
            )
        if method == "least_squares" and not self.least_squares:
            raise ValueError(
                f"{self.name} is fitted by maximum likelihood only: use "
                "method='mle'."
            )
        k = len(self.param_names)
        if method == "least_squares" and len(data) <= k:
            raise ValueError(
                f"A least squares fit of {self.name} needs more than {k} "
                f"stages (for the residual variance); got {len(data)}."
            )
        if method == "mle":
            if np.all(data.failures == 0):
                raise ValueError(
                    "Every trial succeeded: there is no growth to fit. "
                    "The reliability these trials demonstrate is "
                    "surpyval.test_planning.demonstrated_reliability "
                    "(or surpyval.success_run)."
                )
            if np.all(data.failures == data.trials):
                raise ValueError(
                    "Every trial failed: there is no growth to fit."
                )
            params, cov = self._fit_mle(data)
        else:
            params, cov = self._fit_ls(data)
        return DiscreteGrowthModel(self, params, method, data, cov)

    def _data(
        self,
        failures: Any,
        trials: Any,
        stage: Any,
        reliability: Any,
    ) -> _Data:
        if reliability is not None:
            if failures is not None or trials is not None:
                raise ValueError(
                    "Give either failures and trials, or reliability, "
                    "not both."
                )
            r = np.asarray(reliability, dtype=float).reshape(-1)
            if np.any(~np.isfinite(r)) or np.any(r < 0) or np.any(r > 1):
                raise ValueError(
                    "reliability must be in [0, 1] (as a fraction, not a "
                    f"percentage), got {reliability!r}."
                )
            f = np.full(r.shape, np.nan)
            n = np.full(r.shape, np.nan)
        else:
            if failures is None or trials is None:
                raise ValueError(
                    "Give the failures and the trials of each stage (or "
                    "reliability=, the observed reliability of each)."
                )
            f = np.asarray(failures, dtype=float).reshape(-1)
            n = np.asarray(trials, dtype=float).reshape(-1)
            if f.shape != n.shape:
                raise ValueError(
                    "failures and trials must have one value per stage: "
                    f"got {f.size} and {n.size}."
                )
            whole = np.all(np.isfinite(f)) and np.all(f == np.round(f))
            whole = whole and np.all(np.isfinite(n))
            whole = whole and np.all(n == np.round(n))
            if not whole or np.any(n < 1) or np.any(f < 0) or np.any(f > n):
                raise ValueError(
                    "failures and trials must be whole numbers, trials at "
                    "least 1 and failures between 0 and trials."
                )
            r = 1 - f / n
        if r.size < 2:
            raise ValueError("Growth needs at least two stages.")
        if stage is None:
            s = np.arange(1, r.size + 1, dtype=float)
        else:
            s = np.asarray(stage, dtype=float).reshape(-1)
            if s.shape != r.shape or np.any(~np.isfinite(s)):
                raise ValueError(
                    "stage must give one finite number per stage, got "
                    f"{stage!r}."
                )
            if np.any(np.diff(s) <= 0):
                raise ValueError(f"stage must increase, got {stage!r}.")
        self._check_stages(s)
        return _Data(
            pd.DataFrame(
                {"stage": s, "trials": n, "failures": f, "reliability": r}
            )
        )

    def _check_stages(self, s: np.ndarray) -> None:
        pass

    def _nll_u(self, u: Any, data: _Data) -> Any:
        params = self._to_params(u, data)
        r = self._stage_reliability(params, data)
        succ = data.succ
        return _binomial_nll(r, succ, data.failures)

    def _nll_params(self, params: Any, data: _Data) -> Any:
        r = self._stage_reliability(params, data)
        succ = data.succ
        return _binomial_nll(r, succ, data.failures)

    def _minimise(
        self, f: Callable[[np.ndarray], float], starts: list[np.ndarray]
    ) -> np.ndarray:
        """Nelder-Mead from the three best of ``starts``, each restarted
        once from where it stopped."""
        with np.errstate(all="ignore"):
            values = [f(x0) for x0 in starts]
        order = [i for i in np.argsort(values) if np.isfinite(values[i])]
        if not order:
            raise ValueError(
                f"{self.name} could not be fitted to these data: no "
                "starting point gave a finite likelihood."
            )
        best = None
        for i in order[:3]:
            x = np.asarray(starts[i], dtype=float)
            for _ in range(2):
                with np.errstate(all="ignore"):
                    res = optimize.minimize(
                        f,
                        x,
                        method="Nelder-Mead",
                        options={
                            "xatol": 1e-10,
                            "fatol": 1e-13,
                            "maxfev": 4000,
                        },
                    )
                x = res.x
            if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
                best = res
        assert best is not None
        return np.asarray(best.x)

    def _fit_mle(self, data: _Data) -> tuple[np.ndarray, np.ndarray]:
        u = self._minimise(
            lambda x: float(self._nll_u(x, data)), self._starts(data)
        )
        params = np.asarray(self._to_params(u, data), dtype=float)
        info = np.asarray(
            hessian(lambda p: self._nll_params(p, data))(params), dtype=float
        )
        cov = _inverse(info, self.name)
        return params, cov

    def _fit_ls(self, data: _Data) -> tuple[np.ndarray, np.ndarray]:
        y = data.reliability

        def sse_u(u: np.ndarray) -> float:
            params = self._to_params(u, data)
            with np.errstate(all="ignore"):
                r = np.asarray(self._stage_reliability(params, data))
            value = float(np.sum((y - r) ** 2))
            return value if np.isfinite(value) else np.inf

        u = self._minimise(sse_u, self._starts(data))
        params = np.asarray(self._to_params(u, data), dtype=float)
        params = self._polish_ls(params, data)
        return params, self._ls_cov(params, data)

    def _polish_ls(self, params: np.ndarray, data: _Data) -> Any:
        y = data.reliability

        def resid(p: np.ndarray) -> np.ndarray:
            return np.asarray(self._stage_reliability(p, data)) - y

        with np.errstate(all="ignore"):
            res = optimize.least_squares(
                resid,
                params,
                method="lm",
                xtol=1e-15,
                ftol=1e-15,
                gtol=1e-15,
            )
        if res.success and np.sum(res.fun**2) <= np.sum(resid(params) ** 2):
            return res.x
        return params

    def _ls_cov(self, params: np.ndarray, data: _Data) -> np.ndarray:
        y = data.reliability
        jac = np.asarray(
            jacobian(lambda p: self._stage_reliability(p, data))(params),
            dtype=float,
        )
        resid = y - np.asarray(self._stage_reliability(params, data))
        sigma2 = float(resid @ resid) / (len(y) - len(params))
        return sigma2 * _inverse(jac.T @ jac, self.name)

    def _profile_nll(
        self, r: float, s: float, data: _Data, params: np.ndarray
    ) -> float:
        v0 = np.asarray(self._unpin(params, s, data), dtype=float)

        def f(v: np.ndarray) -> float:
            with np.errstate(all="ignore"):
                p = self._pin(v, r, s, data)
                if p is None:
                    return np.inf
                value = float(self._nll_params(p, data))
            return value if np.isfinite(value) else np.inf

        starts = [v0] + [v0 + d for d in self._profile_steps(v0)]
        with np.errstate(all="ignore"):
            values = [f(x) for x in starts]
        order = [i for i in np.argsort(values) if np.isfinite(values[i])]
        best = np.inf
        for i in order[:2]:
            x = starts[i]
            for _ in range(2):
                with np.errstate(all="ignore"):
                    res = optimize.minimize(
                        f,
                        x,
                        method="Nelder-Mead",
                        options={
                            "xatol": 1e-9,
                            "fatol": 1e-12,
                            "maxfev": 3000,
                        },
                    )
                x = res.x
            best = min(best, float(res.fun))
        return best

    def _profile_steps(self, v0: np.ndarray) -> list[np.ndarray]:
        steps = []
        for i in range(v0.size):
            for sign in (-1.0, 1.0):
                d = np.zeros_like(v0)
                d[i] = sign * 0.5
                steps.append(d)
        return steps


def _inverse(info: np.ndarray, name: str) -> np.ndarray:
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov = np.full(info.shape, np.nan)
    if not np.all(np.isfinite(cov)) or np.any(np.diag(cov) < 0):
        import warnings

        warnings.warn(
            f"The {name} fit's information matrix is singular or not "
            "positive definite (the fit is at the edge of the parameters, "
            "or the data do not identify them): its Wald bounds are nan. "
            "Use method='lr' bounds, or more stages.",
            stacklevel=4,
        )
        cov = np.full(info.shape, np.nan)
    return cov


class LloydLipow_(_GrowthFitter):
    """Lloyd and Lipow's (1962) reliability growth model for success/fail
    data in stages: :math:`R_k = R_\\infty - \\alpha / k`.

    In plain words: the reliability climbs towards a ceiling,
    :math:`R_\\infty`, the gap closing as one over the stage number, so most
    of the gain comes in the first few stages. ``alpha`` is the gap at the
    first stage.

    ``fit(failures, trials, stage=None, *, reliability=None,
    method=None)``: the failures and trials of each stage (stages
    1, 2, ... by default; ``stage`` gives others), fitted by maximum
    likelihood (the default) or, with ``method="least_squares"``, by
    least squares on the observed reliabilities, which has the closed
    form

    .. math::
        \\hat\\alpha = \\frac{\\sum\\frac1k\\sum R_k - K\\sum\\frac{R_k}{k}}
                            {K\\sum\\frac1{k^2} - (\\sum\\frac1k)^2},\\qquad
        \\hat R_\\infty = \\frac{\\sum R_k + \\hat\\alpha\\sum\\frac1k}{K}.

    ``reliability=`` takes observed reliabilities alone (least squares).

    Examples
    --------
    >>> from surpyval.recurrent import LloydLipow
    >>> model = LloydLipow.fit(
    ...     failures=[6, 5, 4, 3, 3, 2], trials=[10, 10, 10, 10, 10, 10]
    ... )
    >>> round(model.limit, 3), round(model.next_reliability(), 3)
    (0.798, 0.735)
    """

    name = "Lloyd-Lipow"
    param_names = ("R_inf", "alpha")
    unit_params = ("R_inf",)

    def _check_stages(self, s: np.ndarray) -> None:
        if np.any(s <= 0):
            raise ValueError(
                f"Lloyd-Lipow's stages are positive (1, 2, ...), got {s}."
            )

    def _curve(self, params: Any, s: Any) -> Any:
        return params[0] - params[1] / s

    def _limit(self, params: Any) -> float:
        return float(params[0])

    def _to_params(self, u: Any, data: _Data) -> Any:
        r_inf = _expit(u[0])
        s1 = float(data.stage.min())
        return anp.array([r_inf, r_inf * s1 * _expit(u[1])])

    def _starts(self, data: _Data) -> list[np.ndarray]:
        out = []
        for r_inf in (0.6, 0.8, 0.9, 0.97, 0.995):
            for frac in (0.05, 0.3, 0.6, 0.9):
                out.append(np.array([_logit(r_inf), _logit(frac)]))
        return out

    def _fit_ls(self, data: _Data) -> tuple[np.ndarray, np.ndarray]:
        x = 1.0 / data.stage
        y = data.reliability
        K = len(y)
        denom = K * np.sum(x**2) - np.sum(x) ** 2
        alpha = (np.sum(x) * np.sum(y) - K * np.sum(x * y)) / denom
        r_inf = (np.sum(y) + alpha * np.sum(x)) / K
        params = np.array([r_inf, alpha])
        return params, self._ls_cov(params, data)

    def _pin(self, v: Any, r: float, s: float, data: _Data) -> Any:
        return anp.array([r + v[0] / s, v[0]])

    def _unpin(self, params: Any, s: float, data: _Data) -> Any:
        return np.array([params[1]])


class GompertzGrowth_(_GrowthFitter):
    """The Gompertz reliability growth model: :math:`R(T) = a b^{c^T}`.

    In plain words: an S-shaped climb from :math:`a b` at stage 0 to the
    ceiling ``a``; ``c`` sets the pace (small ``c``, fast early growth).
    All three are in (0, 1).

    ``fit(failures, trials, stage=None, *, reliability=None,
    method=None)``: as :class:`LloydLipow` takes them (stages 1, 2, ... by
    default; the stage may be any increasing time, such as months), fitted
    by maximum likelihood or by least squares on the observed
    reliabilities (Levenberg-Marquardt, from a grid of starts).

    Examples
    --------
    >>> from surpyval.recurrent import GompertzGrowth
    >>> model = GompertzGrowth.fit(
    ...     reliability=[0.58, 0.66, 0.725, 0.78, 0.82, 0.85],
    ...     stage=[0, 1, 2, 3, 4, 5],
    ... )
    >>> [round(float(p), 4) for p in model.params]
    [0.9422, 0.6152, 0.7321]
    """

    name = "Gompertz"
    param_names = ("a", "b", "c")
    unit_params = ("a", "b", "c")

    def _curve(self, params: Any, s: Any) -> Any:
        return params[0] * params[1] ** (params[2] ** s)

    def _limit(self, params: Any) -> float:
        return float(params[0])

    def _to_params(self, u: Any, data: _Data) -> Any:
        return anp.array([_expit(u[0]), _expit(u[1]), _expit(u[2])])

    def _starts(self, data: _Data) -> list[np.ndarray]:
        out = []
        top = float(min(0.999, max(data.reliability.max(), 0.05)))
        for a in (top, (1 + top) / 2, 0.999):
            for b in (0.1, 0.5, 0.9):
                for c in (0.3, 0.6, 0.9):
                    out.append(_logit(np.array([a, b, c])))
        return out

    def _pin(self, v: Any, r: float, s: float, data: _Data) -> Any:
        b, c = _expit(v[0]), _expit(v[1])
        a = r / b ** (c**s)
        if not 0 < a <= 1:
            return None
        return anp.array([a, b, c])

    def _unpin(self, params: Any, s: float, data: _Data) -> Any:
        return _logit(np.asarray(params[1:], dtype=float))


class ModifiedGompertzGrowth_(_GrowthFitter):
    """The modified Gompertz reliability growth model:
    :math:`R(T) = d + a b^{c^T}`.

    In plain words: the Gompertz curve lifted by ``d``, for growth that
    starts slowly, speeds up and levels off (an S shape the Gompertz curve
    cannot bend to). It climbs from :math:`d + a b` at stage 0 to the
    ceiling :math:`a + d`; ``b`` and ``c`` are in (0, 1), and ``d`` in
    [0, 1) with :math:`a + d \\le 1`. Fitted as :class:`GompertzGrowth`.

    Examples
    --------
    Reliabilities on the curve with ``a = 0.6``, ``b = 0.2``, ``c = 0.5``
    and ``d = 0.3`` (to four decimals) give those parameters back, and the
    ceiling ``a + d``:

    >>> from surpyval.recurrent import ModifiedGompertzGrowth
    >>> model = ModifiedGompertzGrowth.fit(
    ...     reliability=[0.42, 0.5683, 0.7012, 0.7907, 0.8426, 0.8706, 0.8851],
    ...     stage=[0, 1, 2, 3, 4, 5, 6],
    ... )
    >>> [round(float(p), 3) for p in model.params]
    [0.6, 0.2, 0.5, 0.3]
    >>> round(model.limit, 3)
    0.9
    """

    name = "Modified Gompertz"
    param_names = ("a", "b", "c", "d")
    unit_params = ("a", "b", "c", "d")

    def _curve(self, params: Any, s: Any) -> Any:
        return params[3] + params[0] * params[1] ** (params[2] ** s)

    def _limit(self, params: Any) -> float:
        return float(params[0] + params[3])

    def _to_params(self, u: Any, data: _Data) -> Any:
        d = _expit(u[3])
        return anp.array(
            [(1 - d) * _expit(u[0]), _expit(u[1]), _expit(u[2]), d]
        )

    def _starts(self, data: _Data) -> list[np.ndarray]:
        out = []
        low = float(np.clip(data.reliability.min(), 0.01, 0.95))
        top = float(np.clip(data.reliability.max(), low + 0.01, 0.999))
        for d in (0.01, low * 0.5, low * 0.9):
            for b in (0.01, 0.2, 0.6):
                for c in (0.3, 0.5, 0.8):
                    a = max(top - d, 1e-3)
                    a_frac = min(a / (1 - d), 0.999)
                    out.append(
                        np.array(
                            [_logit(a_frac), _logit(b), _logit(c), _logit(d)]
                        )
                    )
        return out

    def _pin(self, v: Any, r: float, s: float, data: _Data) -> Any:
        a, b, c = anp.exp(v[0]), _expit(v[1]), _expit(v[2])
        d = r - a * b ** (c**s)
        if not (0 <= d < 1 and a + d <= 1):
            return None
        return anp.array([a, b, c, d])

    def _unpin(self, params: Any, s: float, data: _Data) -> Any:
        p = np.asarray(params, dtype=float)
        return np.array([np.log(p[0]), _logit(p[1]), _logit(p[2])])


class CrowAMSAADiscrete_(_GrowthFitter):
    """The AMSAA (Crow) discrete reliability growth tracking model for
    one-shot items (MIL-HDBK-189C, section 6.2.4).

    In plain words: Duane's learning curve counted in trials. The expected
    number of failures in the first ``T`` trials is
    :math:`\\lambda T^\\beta`; with :math:`\\beta < 1` failures thin out as
    testing goes on. The system is tested in configurations, fixes going
    in between them, and configuration ``i`` (trials
    :math:`(T_{i-1}, T_i]`, :math:`N_i` of them) fails on a trial with
    chance

    .. math::
        f_i = \\frac{\\lambda (T_i^\\beta - T_{i-1}^\\beta)}{N_i}.

    ``fit(failures, trials)``: the failures and trials of each
    configuration, in order, by maximum likelihood (the configurations are
    binomial; the handbook's likelihood). Trial-by-trial data are
    configurations of one trial each.

    The handbook's approximate lower bound on the last configuration's
    reliability is :meth:`handbook_lower_bound`.

    Examples
    --------
    MIL-HDBK-189C's example (table IX): four configurations.

    >>> from surpyval.recurrent import CrowAMSAADiscrete
    >>> model = CrowAMSAADiscrete.fit(
    ...     failures=[5, 3, 4, 4], trials=[14, 19, 15, 20]
    ... )
    >>> [round(float(p), 3) for p in model.params]
    [0.595, 0.78]
    >>> [round(float(r), 3) for r in model.reliability([1, 2, 3, 4])]
    [0.667, 0.766, 0.794, 0.81]
    """

    name = "Crow-AMSAA discrete"
    param_names = ("lambda", "beta")
    continuous_stage = False
    least_squares = False

    def _ends(self, data: _Data) -> np.ndarray:
        return np.cumsum(data.trials)

    def _config(self, params: Any, start: Any, n: Any) -> Any:
        lam, beta = params[0], params[1]
        start = np.asarray(start, dtype=float)
        # 0 ** beta is 0, but its derivative in beta would be 0 * log 0.
        positive = start > 0
        safe = np.where(positive, start, 1.0)
        before = anp.where(positive, safe**beta, 0.0)
        return 1 - lam * ((start + n) ** beta - before) / n

    def _stage_reliability(self, params: Any, data: _Data) -> Any:
        ends = self._ends(data)
        n = data.trials
        return self._config(params, ends - n, n)

    def _r(self, params: Any, s: Any, data: _Data) -> Any:
        s_ = np.asarray(s, dtype=float)
        k = len(data)
        if np.any(s_ != np.round(s_)) or np.any(s_ < 1) or np.any(s_ > k):
            raise ValueError(
                "stage is the number of a fitted configuration, 1 to "
                f"{k}, got {s!r}: for a new configuration use "
                "next_reliability(trials)."
            )
        idx = s_.astype(int) - 1
        ends = self._ends(data)
        n = data.trials
        return self._config(params, (ends - n)[idx], n[idx])

    def _next(self, params: Any, data: _Data, trials: Any) -> Any:
        n = float(data.trials[-1]) if trials is None else trials
        if not (np.isfinite(n) and n >= 1):
            raise ValueError(f"trials must be 1 or more, got {trials!r}.")
        return self._config(params, float(self._ends(data)[-1]), float(n))

    def _limit(self, params: Any) -> float:
        return 1.0 if params[1] < 1 else 0.0

    def _to_params(self, u: Any, data: _Data) -> Any:
        return anp.exp(u)

    def _starts(self, data: _Data) -> list[np.ndarray]:
        T = float(self._ends(data)[-1])
        F = float(data.failures.sum())
        out = []
        for beta in (0.3, 0.6, 0.8, 1.0, 1.2):
            out.append(np.log([F / T**beta, beta]))
        return out

    def _check_stages(self, s: np.ndarray) -> None:
        pass

    @refuses_surpyval_data_with(_COUNTS_NOT_ROWS, argument="failures")
    def fit(
        self,
        failures: npt.ArrayLike | None = None,
        trials: npt.ArrayLike | None = None,
        stage: npt.ArrayLike | None = None,
        *,
        reliability: npt.ArrayLike | None = None,
        method: str | None = None,
    ) -> DiscreteGrowthModel:
        """Fit the AMSAA discrete model to the ``failures`` and ``trials``
        of each configuration, in order, by maximum likelihood. It takes
        no ``stage`` (its time is the cumulative trials) and no
        ``reliability``; otherwise as the other growth models' ``fit``."""
        if stage is not None:
            raise ValueError(
                "CrowAMSAADiscrete's time is the cumulative number of "
                "trials, from the trials of each configuration: it takes "
                "no stage."
            )
        if reliability is not None:
            raise ValueError(
                "CrowAMSAADiscrete is fitted from the failures and trials "
                "of each configuration, not from reliabilities."
            )
        return super().fit(failures, trials, method=method)

    def _pin(self, v: Any, r: float, s: float, data: _Data) -> Any:
        beta = anp.exp(v[0])
        idx = int(s) - 1
        ends = self._ends(data)
        n = float(data.trials[idx])
        start = float(ends[idx] - n)
        before = start**beta if start > 0 else 0.0
        lam = (1 - r) * n / ((start + n) ** beta - before)
        return anp.array([lam, beta])

    def _unpin(self, params: Any, s: float, data: _Data) -> Any:
        return np.array([np.log(params[1])])


LloydLipow = LloydLipow_()
GompertzGrowth = GompertzGrowth_()
ModifiedGompertzGrowth = ModifiedGompertzGrowth_()
CrowAMSAADiscrete = CrowAMSAADiscrete_()
