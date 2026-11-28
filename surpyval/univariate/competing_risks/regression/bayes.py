"""
Bayesian parametric competing-risks regression, by composition.

A parametric competing-risks regression's likelihood factorises over the
causes, as the model without covariates does: a unit that failed from
cause ``k`` contributes cause ``k``'s density at its covariates and every
other cause's survival there, and a unit still working contributes every
cause's survival. So, with priors stated a cause at a time, the causes'
posteriors are independent, and each is the ordinary Bayesian fit of
that cause's regression (``fitter.fit_bayes``, the Laplace engine) with
the other causes' failures right censored. Nothing new is computed for
the posterior; this module combines the causes' fits.

The results are for one covariate row at a time: ``model.at(Z)`` binds a
row and gives a :class:`BayesianCompetingRisks
<surpyval.univariate.competing_risks.parametric.bayes.BayesianCompetingRisks>`
of a unit with those covariates (its cumulative incidence, all-cause
survival, B-life and their credible intervals), and the model's own
functions (``cif(x, Z, event)``, ...) are shorthands for it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.univariate.competing_risks.labels import (
    label_from_native,
    label_mask,
    ordered_labels,
)
from surpyval.univariate.competing_risks.parametric.bayes import (
    SEED,
    BayesianCompetingRisks,
    labelled,
    split_priors,
)
from surpyval.univariate.regression.life_at import life_at_one_row
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.validation import unknown_cause_error
from surpyval.utils.warnings import ignoring_warnings

#: The joint draws behind a regression's credible intervals (one from
#: each cause's posterior, cause ``j`` with seed ``SEED + j``), as many
#: as a Bayesian regression's own draws.
DRAWS = 1024


class _CauseAt:
    """A cause's Bayesian regression at one covariate row, as the
    competing-risks combination reads a cause's fit: its posterior
    predictive functions of time, its plug-in model and its intervals."""

    def __init__(self, fit: Any, Z: Any) -> None:
        self.fit = fit
        self.Z = Z
        self.method = fit.method

    def _f(self, name: str, x: Any) -> npt.NDArray:
        x = np.asarray(x, dtype=float)
        flat = np.atleast_1d(x).ravel()
        with ignoring_warnings(), np.errstate(all="ignore"):
            out = np.asarray(getattr(self.fit, name)(flat, self.Z), float)
        return out.reshape(x.shape)

    def sf(self, x: Any) -> npt.NDArray:
        return self._f("sf", x)

    def ff(self, x: Any) -> npt.NDArray:
        return self._f("ff", x)

    def df(self, x: Any) -> npt.NDArray:
        return self._f("df", x)

    def hf(self, x: Any) -> npt.NDArray:
        return self._f("hf", x)

    def Hf(self, x: Any) -> npt.NDArray:
        return self._f("Hf", x)

    def qf(self, p: Any) -> npt.NDArray:
        return self._f("qf", p)

    def at_params(self) -> Any:
        return self.fit.at_params().at(self.Z)

    def cri(
        self,
        x: Any,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        x = np.asarray(x, dtype=float)
        flat = np.atleast_1d(x).ravel()
        out = np.asarray(
            self.fit.cri(flat, self.Z, on=on, alpha_ci=alpha_ci, bound=bound),
            dtype=float,
        )
        tail = out.shape[1:] if out.ndim > 1 else ()
        return out.reshape(x.shape + tail)

    def param_cri(
        self, name: str, alpha_ci: float = 0.05, bound: str = "two-sided"
    ) -> npt.NDArray:
        return np.asarray(self.fit.param_cri(name, alpha_ci, bound), float)

    def summary(self, alpha_ci: float = 0.05) -> str:
        return str(self.fit.summary(alpha_ci))


class BayesianCompetingRisksAt(BayesianCompetingRisks):
    """
    A Bayesian competing-risks regression at one covariate row
    (``BayesianCompetingRisksRegression.at(Z)``): the life of a unit with
    those covariates, with every function and interval of
    :class:`~surpyval.univariate.competing_risks.parametric.bayes.BayesianCompetingRisks`.

    The predictions (``sf``, ``ff``, ``cif``, ``qf``, ...) are the
    posterior predictive at the row, exact from the causes' own
    predictive functions (their posteriors are independent); the credible
    intervals come from ``DRAWS`` (1,024) joint draws, one from each
    cause's posterior with a fixed seed, each draw's regression evaluated
    at the row. It is a view: save the regression, not this.
    """

    model_tag = "BayesianCompetingRisksAt"

    def __init__(self, regression: Any, Z: Any) -> None:
        self.regression = regression
        first = regression.models[regression.causes[0]]
        self.Z = life_at_one_row(first.at_params(), Z)
        super().__init__(
            {k: _CauseAt(m, self.Z) for k, m in regression.models.items()},
            regression.mle,
        )
        self._lives: dict | None = None

    def _draw_lives(self) -> dict:
        """Each cause's regression at the row, under each of its
        ``DRAWS`` draws."""
        if self._lives is None:
            out = {}
            for j, k in enumerate(self.causes):
                fit = self.regression.models[k]
                D = fit.posterior.draws(DRAWS, random_state=SEED + j)
                with ignoring_warnings():
                    out[k] = [fit.target.with_params(d) for d in D]
            self._lives = out
        return self._lives

    def _joint(self) -> dict:
        # (the draws are models, not parameters: _at evaluates them)
        return {k: None for k in self.causes}

    def _at(
        self,
        cause: Any,
        kind: str,
        value: Any,
        extra: int,
        theta: dict | None = None,
    ) -> Any:
        lives = self._draw_lives()[cause]
        v = np.asarray(value, dtype=float)
        name = {"F": "ff", "R": "sf", "q": "qf"}[kind]
        out = np.empty((DRAWS,) + v.shape[1:])
        with ignoring_warnings(), np.errstate(all="ignore"):
            for i, life in enumerate(lives):
                row = v[i] if v.shape[0] == DRAWS else v[0]
                flat = row.ravel()
                out[i] = np.asarray(
                    getattr(life, name)(flat, self.Z), dtype=float
                ).reshape(row.shape)
        return out

    def _cif_at(self, theta: dict | None, x: Any, event: Any) -> npt.NDArray:
        if theta is not None:
            raise ValueError("A regression's draws are models, not points.")
        x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
        from surpyval.univariate.competing_risks.parametric.bayes import (
            GAUSS_NODES,
            GAUSS_WEIGHTS,
        )

        out = np.full((DRAWS, x.size), np.nan)
        ok = ~np.isnan(x)
        idx = np.flatnonzero(ok)
        for s in range(0, idx.size, 16):
            cols = idx[s : s + 16]
            xs = x[cols]
            top = self._at(
                event, "F", np.where(np.isinf(xs), 0.0, xs)[None, :], 1
            )
            top = np.where(np.isposinf(xs)[None, :], 1.0, top)
            top = np.where(np.isneginf(xs)[None, :], 0.0, top)
            top = np.clip(np.where(np.isfinite(top), top, 0.0), 0.0, 1.0)
            p = top[:, :, None] * GAUSS_NODES[None, None, :]
            t = self._at(event, "q", p, 2)
            g = np.ones(p.shape)
            for k in self.causes:
                if k != event:
                    g = g * self._at(k, "R", t, 2)
            g = np.where(np.isfinite(g), g, 0.0)
            out[:, cols] = top * np.einsum("dtq,q->dt", g, GAUSS_WEIGHTS)
        return out

    def sf_draws(self, x: npt.ArrayLike) -> npt.NDArray:
        x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
        out = np.ones((DRAWS, x.size))
        for k in self.causes:
            out = out * self._at(k, "R", x[None, :], 1)
        return out

    def quantile_draws(self, p: float) -> npt.NDArray:
        """The all-cause ``p`` quantile at the row under each joint draw,
        ``(DRAWS,)``: the root of the summed cumulative hazards, found by
        three rounds of 32 points between the causes' own quantiles
        (each round narrowing the bracket 31-fold) and linear
        interpolation of the log cumulative hazard in log time, to about
        1e-9 of the time."""
        p = float(p)
        K = len(self.causes)
        lives = self._draw_lives()
        qs = np.stack(
            [self._at(k, "q", np.array([[p]]), 1)[:, 0] for k in self.causes]
        )
        if p <= 0 or p >= 1:
            return np.min(qs, axis=0)
        hi = np.min(qs, axis=0)
        lo_p = -np.expm1(np.log1p(-p) / K)
        lo = np.min(
            np.stack(
                [
                    self._at(k, "q", np.array([[lo_p]]), 1)[:, 0]
                    for k in self.causes
                ]
            ),
            axis=0,
        )
        target = np.log(-np.log1p(-p))
        out = np.full(DRAWS, np.nan)
        with ignoring_warnings(), np.errstate(all="ignore"):
            for i in range(DRAWS):
                a, b = float(lo[i]), float(hi[i])
                if not (np.isfinite(a) and np.isfinite(b)) or b <= a:
                    out[i] = b
                    continue
                for _ in range(3):
                    t = np.linspace(a, b, 32)
                    H = sum(
                        np.asarray(lives[k][i].Hf(t, self.Z), dtype=float)
                        for k in self.causes
                    )
                    logH = np.log(np.asarray(H, dtype=float))
                    j = int(np.searchsorted(logH, target))
                    j = min(max(j, 1), 31)
                    a, b = float(t[j - 1]), float(t[j])
                    la, lb = logH[j - 1], logH[j]
                lt_a, lt_b = np.log(a), np.log(b)
                w = (target - la) / (lb - la) if lb > la else 0.5
                out[i] = float(np.exp(lt_a + w * (lt_b - lt_a)))
        return out

    def to_dict(self) -> dict:
        raise ValueError(
            "A regression at one covariate row is a view: save the "
            "regression (model.to_dict()) and the row."
        )


class BayesianCompetingRisksRegression(SerialisableMixin):
    """
    A Bayesian parametric competing-risks regression
    (``ParametricCompetingRisksRegression.fit_bayes``): a regression for
    each cause's hazard, each with its own posterior.

    The likelihood factorises over the causes, so with priors stated a
    cause at a time their posteriors are independent, and each is the
    :class:`~surpyval.bayes.BayesianRegression` of its cause, fitted with
    the other causes' failures right censored, in ``models``.

    The results are for one covariate row per call: :meth:`at` gives the
    life of a unit with those covariates as a
    :class:`BayesianCompetingRisksAt`, and :meth:`cif`, :meth:`sf`,
    :meth:`ff`, :meth:`qf`, :meth:`cri`, :meth:`quantile_cri` and
    :meth:`probability_of_cause` take the row ``Z`` and call it.

    Attributes
    ----------
    models : dict
        Each cause's Bayesian regression, by cause.
    causes : list
        The causes, in order.
    mle : ParametricCompetingRisksRegression or None
        The maximum-likelihood fit to the same data.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import ParametricCompetingRisksRegression
    >>> rng = np.random.default_rng(0)
    >>> Z = rng.binomial(1, 0.5, (200, 1)).astype(float)
    >>> t_a = rng.weibull(2.0, 200) * 10 * np.exp(-0.4 * Z[:, 0])
    >>> t_b = rng.exponential(20, 200)
    >>> x = np.minimum(t_a, t_b).round(3)
    >>> e = np.where(t_a < t_b, "a", "b")
    >>> model = ParametricCompetingRisksRegression.fit_bayes(x, Z, e)
    >>> model.method
    {'a': 'laplace', 'b': 'laplace'}
    >>> lo, hi = model.cri(10.0, [1.0], "a")
    >>> bool(lo < model.cif(10.0, [1.0], "a") < hi)
    True
    """

    model_tag = "BayesianCompetingRisksRegression"

    def __init__(self, models: dict, mle: Any = None) -> None:
        if len(models) == 0:
            raise ValueError("At least one cause's fit is required.")
        self.models = {
            (k.item() if isinstance(k, np.generic) else k): m
            for k, m in models.items()
        }
        self.causes = ordered_labels(self.models)
        self.mle = mle
        self.method = {k: self.models[k].method for k in self.causes}

    def _check_event(self, event: Any) -> None:
        if event not in self.models:
            raise unknown_cause_error(event, self.causes)

    def at(self, Z: Any) -> BayesianCompetingRisksAt:
        """The life of a unit with the one covariate row ``Z`` (kept for
        the next call at the same row: its incidence tables and draws
        are computed once)."""
        try:
            key: Any = tuple(np.asarray(Z, dtype=float).ravel().tolist())
        except (TypeError, ValueError):
            key = None
        cache = self.__dict__.setdefault("_at", {})
        if key is not None and key in cache:
            return cache[key]
        out = BayesianCompetingRisksAt(self, Z)
        if key is not None:
            if len(cache) >= 64:
                cache.pop(next(iter(cache)))
            cache[key] = out
        return out

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_at", None)
        return state

    @staticmethod
    def _missing(Z: Any) -> bool:
        """Whether the covariate row has a missing value (the results
        there are then NaN, as for every regression)."""
        if type(Z).__name__ == "DataFrame":
            return bool(Z.isna().to_numpy().any())
        try:
            return bool(np.isnan(np.asarray(Z, dtype=float)).any())
        except (TypeError, ValueError):
            return False

    def _nan(self, x: Any, extra: tuple = ()) -> npt.NDArray:
        return np.full(np.shape(np.asarray(x, dtype=float)) + extra, np.nan)

    @keeps_query_shape
    def sf(self, x: npt.ArrayLike, Z: Any, event: Any = None) -> Any:
        """The predictive survival at the row ``Z`` (all causes, or one
        cause's net survival); see :meth:`at`."""
        if self._missing(Z):
            return self._nan(x)
        return self.at(Z).sf(x, event)

    @keeps_query_shape
    def ff(self, x: npt.ArrayLike, Z: Any, event: Any = None) -> Any:
        """The predictive failure probability at the row ``Z``."""
        if self._missing(Z):
            return self._nan(x)
        return self.at(Z).ff(x, event)

    @keeps_query_shape
    def cif(self, x: npt.ArrayLike, Z: Any, event: Any = None) -> Any:
        """The predictive cumulative incidence of ``event`` at the row
        ``Z``: the probability that a new unit with those covariates has
        failed from it by ``x``, the other causes acting."""
        if self._missing(Z):
            return self._nan(x)
        return self.at(Z).cif(x, event)

    def probability_of_cause(self, Z: Any, event: Any) -> float:
        """The predictive probability that a unit with the row ``Z``
        fails from ``event``."""
        if self._missing(Z):
            return float("nan")
        return self.at(Z).probability_of_cause(event)

    @keeps_query_shape
    def qf(self, p: npt.ArrayLike, Z: Any, event: Any = None) -> Any:
        """The predictive quantile at the row ``Z``."""
        if self._missing(Z):
            return self._nan(p)
        return self.at(Z).qf(p, event)

    @keeps_query_shape
    def cri(
        self,
        x: npt.ArrayLike,
        Z: Any,
        event: Any = None,
        on: str = "cif",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> Any:
        """A credible interval at times ``x`` for a unit with the row
        ``Z``, as :meth:`BayesianCompetingRisks.cri
        <surpyval.univariate.competing_risks.parametric.bayes.BayesianCompetingRisks.cri>`.
        """
        if self._missing(Z):
            return self._nan(x, (2,) if bound == "two-sided" else ())
        return self.at(Z).cri(x, event, on, alpha_ci, bound)

    @keeps_query_shape
    def quantile_cri(
        self,
        p: npt.ArrayLike,
        Z: Any,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        point: bool = False,
    ) -> Any:
        """A credible interval on the all-cause B-life of a unit with the
        row ``Z``."""
        if self._missing(Z):
            two = bound == "two-sided" and not point
            return self._nan(np.atleast_1d(p), (2,) if two else ())
        return self.at(Z).quantile_cri(p, alpha_ci, bound, point)

    def param_cri(
        self,
        event: Any,
        name: str,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """A credible interval for a parameter of one cause's
        regression."""
        self._check_event(event)
        return np.asarray(
            self.models[event].param_cri(name, alpha_ci, bound), dtype=float
        )

    def summary(self, alpha_ci: float = 0.05) -> str:
        """Each cause's fit in words, one after another."""
        return "\n\n".join(
            f"Cause {k!r}\n" + str(self.models[k].summary(alpha_ci))
            for k in self.causes
        )

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict:
        """A strict-JSON dictionary of the fit: each cause's, with the
        causes in order."""
        import surpyval
        from surpyval.serialisation import to_native

        return stamp_schema(
            {
                "model": self.model_tag,
                "causes": [to_native(k) for k in self.causes],
                "models": [self.models[k].to_dict() for k in self.causes],
                "surpyval_version": surpyval.__version__,
            }
        )

    @classmethod
    def from_dict(cls, model_dict: dict) -> BayesianCompetingRisksRegression:
        """Rebuild a fit from :meth:`to_dict`."""
        from surpyval.bayes import BayesianRegression

        require_model_tag(
            model_dict,
            cls.model_tag,
            "a Bayesian competing-risks regression",
        )
        return cls(
            {
                label_from_native(k): BayesianRegression.from_dict(d)
                for k, d in zip(model_dict["causes"], model_dict["models"])
            }
        )


def fit(
    cls: Any,
    x: Any,
    Z: Any,
    e: Any,
    c: Any = None,
    n: Any = None,
    tl: Any = None,
    model: Any = None,
    *,
    priors: Any = None,
    method: str = "auto",
    importance: bool = False,
) -> BayesianCompetingRisksRegression:
    """The Bayesian fit of a parametric competing-risks regression; see
    ``ParametricCompetingRisksRegression.fit_bayes``."""
    from surpyval import WeibullPH
    from surpyval.univariate.competing_risks.regression import (
        parametric_competing_risks_regression as pcrr,
    )
    from surpyval.utils.no_maximum import quiet_maximum_warnings

    model = WeibullPH if model is None else model
    x, Z, e, c, n, _, tl = pcrr._validated(
        x, Z, e, c, n, None, pcrr._entry_or_none(tl)
    )
    causes = ordered_labels(e)
    if not causes:
        raise ValueError("No observed events: every row is censored.")
    fitters = pcrr._cause_fitters(model, causes)
    own = split_priors(priors, causes)
    mle = None
    try:
        with quiet_maximum_warnings(), ignoring_warnings():
            mle = cls.fit(x, Z, e, c=c, n=n, tl=tl, model=model)
    except ValueError:
        mle = None
    models = {}
    for k in causes:
        c_k = np.where(label_mask(e, k) & (c == 0), 0, 1)
        extra = {} if tl is None else {"tl": tl}
        p = own[k]
        try:
            models[k] = labelled(
                f"Cause {k!r}",
                lambda k=k, c_k=c_k, p=p, extra=extra: fitters[k].fit_bayes(
                    x,
                    Z,
                    c=c_k,
                    n=n,
                    priors=dict(p) if isinstance(p, Mapping) else p,
                    method=method,
                    importance=importance,
                    **extra,
                ),
            )
        except ValueError as err:
            raise ValueError(f"Cause {k!r}: {err}") from None
    return BayesianCompetingRisksRegression(models, mle)


__all__ = [
    "BayesianCompetingRisksAt",
    "BayesianCompetingRisksRegression",
    "DRAWS",
    "fit",
]
