from typing import Any, Callable

import numpy as np
from scipy.optimize import brentq

from surpyval.recurrent.inference import LikelihoodInferenceMixin
from surpyval.recurrent.simulation import RecurrenceSimulationMixin
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)

#: Below this cumulative hazard the quantile function is accurate enough to
#: invert it: ``1 - p = exp(-H)`` then carries a relative error of about
#: ``eps * exp(H)``, which is 5e-8 at 20 (an error of 3e-9 in ``H``).
_QF_HAZARD_LIMIT = 20.0


def conditional_gap(lifetime: Any, v: float, u: float) -> float:
    """
    Draw the time to the next failure of an item whose virtual age is
    ``v``, from the uniform ``u``, for a virtual-age renewal model with the
    lifetime distribution ``lifetime``.

    The residual life ``X`` from age ``v`` has ``P(X > x) = S(v + x) /
    S(v)``, so ``H(v + X) = H(v) - log(u)``: the draw adds an Exp(1) amount
    to the cumulative hazard ``H`` and inverts it. Working with ``H``
    rather than the survival function keeps this exact at long horizons.
    The old draw, ``qf(1 - u * sf(v)) - v``, lost all precision once
    ``sf(v)`` fell below about 1e-16 (an expected count of about 37), so
    the next age came back as ``v`` itself and the simulated MCF went flat.

    ``H`` is inverted with the quantile function while that is accurate
    (see ``_QF_HAZARD_LIMIT``) and by root finding beyond it.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        h_v = float(lifetime.Hf(v))
        target = h_v - float(np.log(u))
    if np.isinf(h_v) and h_v > 0:
        # No survival past age v (the end of a bounded support): the item
        # fails at once.
        return 0.0
    if not np.isfinite(target):
        # u == 0: an infinitely late event.
        return np.inf
    if target <= _QF_HAZARD_LIMIT:
        x = float(lifetime.qf(-np.expm1(-target)))
    else:
        x = _invert_cumulative_hazard(lifetime, v, target)
    return max(x - v, 0.0)


def _cumulative_hazard(lifetime: Any) -> Callable:
    """``H`` of ``lifetime`` as a scalar function. For a plain model (no
    offset, cure fraction or zero-inflation -- what the renewal fitters
    build) the distribution's own ``Hf`` is called directly: the model
    method's argument handling costs fifty times the arithmetic, and the
    root finder calls it dozens of times per draw."""
    if (
        getattr(lifetime, "p", None) == 1
        and getattr(lifetime, "f0", None) == 0
        and not getattr(lifetime, "gamma", 0)
    ):
        dist_hf = lifetime.dist.Hf
        params = [float(p) for p in lifetime.params]
        lower = lifetime.dist.support[0]

        def hf(x: float) -> float:
            return 0.0 if x < lower else float(dist_hf(x, *params))

        return hf
    return lambda x: float(lifetime.Hf(x))


def _invert_cumulative_hazard(lifetime: Any, v: float, target: float) -> float:
    """Solve ``H(x) = target`` for ``x > v`` (where ``H(v) < target``)."""
    hazard = _cumulative_hazard(lifetime)

    def g(x: float) -> float:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return hazard(x) - target

    lo = float(v)
    step = max(abs(lo), 1.0)
    hi = lo + step
    for _ in range(2000):
        g_hi = g(hi)
        if not g_hi < 0:
            break
        lo, step = hi, 2.0 * step
        hi = lo + step
    # Past a bounded support H can be inf or nan; bisect until hi is a
    # finite point at or above the target so the root is bracketed.
    for _ in range(200):
        if np.isfinite(g(hi)):
            break
        mid = 0.5 * (lo + hi)
        if g(mid) < 0:
            lo = mid
        else:
            hi = mid
    g_hi = g(hi)
    if not np.isfinite(g_hi):
        # H jumps to infinity at the end of the support: fail there.
        return lo
    if g_hi == 0:
        return hi
    xtol = 4 * np.finfo(float).eps * max(abs(lo), abs(hi), 1e-300)
    return float(brentq(g, lo, hi, xtol=xtol))


class RenewalModel(
    SerialisableMixin, RecurrenceSimulationMixin, LikelihoodInferenceMixin
):
    """
    A fitted renewal / imperfect-repair recurrence model.

    This is the model object returned by the renewal-family fitters
    (``GeneralizedRenewal``, ``GeneralizedOneRenewal``, ``ARA``, ``ARI``), in
    the same way that the intensity fitters (``CrowAMSAA``, ``Duane``, ...)
    return a ``ParametricRecurrenceModel``. It holds the fitted underlying
    lifetime distribution and the restoration parameter, and provides the
    simulation (``mcf``, ``plot``, ``count_terminated_simulation``,
    ``time_terminated_simulation``) and likelihood-inference
    (``log_likelihood``, ``aic``, ``bic``, ``standard_errors``) behaviour via
    the shared mixins.

    These processes have no closed-form intensity, so the mean cumulative
    function is obtained by simulation.

    Parameters
    ----------
    model : Parametric
        The fitted underlying lifetime distribution.
    restoration : float
        The fitted restoration / repair parameter (``q`` for the generalized
        and G1 renewal processes, ``rho`` for ARA and ARI).
    restoration_name : str
        The attribute/label name of the restoration parameter (e.g. ``"q"`` or
        ``"rho"``); it is also exposed as an attribute of that name.
    restoration_label : str
        Human-readable label used in ``__repr__`` (e.g. ``"Restoration
        Factor"`` or ``"Repair Efficiency"``).
    kind : str
        Display name of the process (e.g. ``"Generalized Renewal"``).
    sampler_factory : callable
        ``sampler_factory(model) -> sample`` returning a fresh per-sequence
        sampler ``sample(ui) -> interarrival`` for simulation.
    restoration_bounds : tuple, optional
        Natural-space ``(lower, upper)`` bounds of the restoration parameter
        (e.g. ``(0, 1)`` for ARA/ARI's ``rho``), used by ``param_cb`` to pick
        a transform that keeps its confidence bounds inside the support.
    """

    # Set by the GeneralizedRenewal fitter for its Kijima sampler.
    _virtual_age_function: Any

    #: Set by the fitter for the generalized-renewal family (``"i"``/``"ii"``);
    #: absent otherwise.
    kijima_type: Any
    #: Set by the fitter for the ARA/ARI families (memory); absent otherwise.
    m: Any
    #: How the parameters were obtained: ``"MLE"`` for a fit, otherwise
    #: given (``fit_from_parameters``). Kept through serialisation.
    how: str = "from_params"

    def __init__(
        self,
        model: Any,
        restoration: float,
        restoration_name: str,
        restoration_label: str,
        kind: str,
        sampler_factory: Callable,
        dist_label: str = "Distribution",
        restoration_bounds: tuple = (None, None),
    ) -> None:
        self.model = model
        self.restoration = restoration
        self._restoration_param_name = restoration_name
        self._restoration_label = restoration_label
        self._restoration_bounds = restoration_bounds
        self._dist_label = dist_label
        self.kind = kind
        self._sampler_factory = sampler_factory
        # Expose the restoration parameter under its conventional name
        # (``q``/``rho``) so existing usage keeps working.
        setattr(self, restoration_name, restoration)

    # -- serialisation -----------------------------------------------------

    def _family(self) -> str:
        """Identify the renewal family from the fitted attributes."""
        if getattr(self, "kijima_type", None) is not None:
            return "GeneralizedRenewal"
        if getattr(self, "m", None) is not None:
            if self._dist_label == "Baseline Intensity":
                return "ARI"
            return "ARA"
        return "GeneralizedOneRenewal"

    def to_dict(self) -> dict:
        """
        Serialise this fitted renewal / imperfect-repair model to a plain,
        JSON-serialisable dict.

        These processes have no closed-form intensity -- their simulation is
        driven by a sampler closure built from the underlying distribution and
        the restoration parameter -- so what is stored is the family, the
        underlying distribution (by name) and its parameters, the restoration
        parameter, and the family's discrete option (``kijima_type`` for the
        generalized renewal, memory ``m`` for ARA/ARI). On load the family's
        fitter rebuilds the sampler from those, so ``mcf`` and the simulation
        methods reproduce exactly. The likelihood/data state is not stored.

        See Also
        --------
        from_dict, to_json, from_json
        """
        out: dict = {
            "model": "RenewalModel",
            "family": self._family(),
            "dist": self.model.dist.name,
            "params": np.asarray(self.model.params, dtype=float).tolist(),
            "restoration": float(self.restoration),
            "how": self.how,
        }
        if getattr(self, "kijima_type", None) is not None:
            out["kijima_type"] = self.kijima_type
        if getattr(self, "m", None) is not None:
            out["m"] = self.m
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "RenewalModel":
        """
        Rebuild a renewal model from a :meth:`to_dict` dictionary.

        The family's fitter is used to reconstruct the model from parameters
        (which regenerates the simulation sampler), so the result predicts
        identically to the original.

        See Also
        --------
        to_dict, to_json, from_json
        """
        import surpyval.recurrent as recurrent

        require_model_tag(model_dict, "RenewalModel", "a renewal model")
        family = model_dict["family"]
        fitters: "dict[str, Any]" = {
            "GeneralizedRenewal": recurrent.GeneralizedRenewal,
            "GeneralizedOneRenewal": recurrent.GeneralizedOneRenewal,
            "ARA": recurrent.ARA,
            "ARI": recurrent.ARI,
        }
        if family not in fitters:
            raise ValueError("Unknown renewal family {!r}".format(family))
        out = cls._rebuild(fitters[family], family, model_dict)
        # A reloaded fit still says it was fitted by MLE (it carries no
        # likelihood, as a reloaded parametric model does not).
        out.how = model_dict.get("how", "from_params")
        return out

    @staticmethod
    def _rebuild(fitter: Any, family: str, model_dict: dict) -> "RenewalModel":
        from surpyval.recurrent.serialisation import intensity_dist_by_name

        params = model_dict["params"]
        restoration = model_dict["restoration"]

        if family == "ARI":
            # the ARI baseline is a recurrence intensity model
            dist = intensity_dist_by_name(model_dict["dist"])
            return fitter.fit_from_parameters(
                params, restoration, m=model_dict["m"], dist=dist
            )

        import surpyval
        from surpyval.univariate.parametric.parametric_fitter import (
            ParametricFitter,
        )

        # Restrict the lookup to known distribution fitters so an
        # untrusted model dict cannot resolve arbitrary surpyval
        # attributes (matches the guard in Parametric.from_dict).
        dist = getattr(surpyval, model_dict["dist"], None)
        if not isinstance(dist, ParametricFitter):
            raise ValueError(
                "Unknown distribution {!r}".format(model_dict["dist"])
            )
        if family == "GeneralizedRenewal":
            return fitter.fit_from_parameters(
                params,
                restoration,
                kijima=model_dict["kijima_type"],
                dist=dist,
            )
        if family == "ARA":
            return fitter.fit_from_parameters(
                params, restoration, m=model_dict["m"], dist=dist
            )
        # GeneralizedOneRenewal
        return fitter.fit_from_parameters(params, restoration, dist=dist)

    def _new_sequence_sampler(self) -> Callable:
        return self._sampler_factory(self)

    def _parameter_names(self) -> list:
        # The restoration parameter (``q``/``rho``) leads ``_mle``, followed by
        # the underlying lifetime/intensity model's parameters.
        return [self._restoration_param_name, *self.model.dist.param_names]

    def _parameter_bounds(self) -> list:
        return [self._restoration_bounds, *self.model.dist.bounds]

    def residuals(self, kind: str = "cumulative_hazard") -> np.ndarray:
        """
        Residual diagnostics for the fitted imperfect-repair model, from the
        time-rescaling theorem applied to the process's *conditional*
        intensity (each interarrival is rescaled by the cumulative hazard
        accumulated over it given the model's virtual age / intensity
        reduction), so they extend the counting-process residuals to the
        renewal / virtual-age families.

        Parameters
        ----------

        kind: {'cumulative_hazard', 'pit', 'martingale'}, optional
            ``'cumulative_hazard'`` returns the rescaled interarrival
            increments of every observed event (pooled across items); see
            below for how far they are iid Exp(1). ``'pit'`` applies the
            probability integral transform ``1 - exp(-e)`` to those residuals
            (U(0, 1) under the same conditions). ``'martingale'`` returns
            one residual per item: its observed event count minus the
            compensator (the sum of the rescaled increments) accumulated
            over its observation.

            Only complete gaps (event to event) are returned. When an
            item's observation ends at a window close rather than at an
            event, its final gap is censored and left out, and that
            selection makes the returned residuals smaller than Exp(1) on
            average -- noticeably so with few events per item (a mean
            near 0.66 with about three events per item). So they are
            exactly iid Exp(1) only for failure-truncated items; otherwise
            read a Q-Q plot against Exp(1) with this downward bias in
            mind, or use ``cramer_von_mises``, which conditions on each
            item's window correctly.

        Returns
        -------

        numpy array
            The residuals.
        """
        self._check_has_data("residuals")
        from surpyval.recurrent import diagnostics

        diagnostics._validate_diagnostic_data(self.data, "Residuals")
        increments = np.asarray(
            self._fitter._rescaled_increments(self, self.data), dtype=float
        )
        c = np.asarray(self.data.c)

        if kind in ("cumulative_hazard", "pit"):
            e = increments[c == 0]
            if kind == "pit":
                return 1.0 - np.exp(-e)
            return e
        elif kind == "martingale":
            residuals = []
            for item in np.unique(self.data.i):
                mask = self.data.i == item
                observed = int((c[mask] == 0).sum())
                residuals.append(observed - float(increments[mask].sum()))
            return np.array(residuals)
        raise ValueError(
            "`kind` must be 'cumulative_hazard', 'pit' or 'martingale'; "
            "got {!r}".format(kind)
        )

    def trend_test(
        self, test: str = "laplace", alternative: str = "two-sided"
    ) -> Any:
        """
        Run a trend test on the data this model was fitted to. The null
        hypothesis is a *homogeneous* Poisson process (no trend); the statistic
        uses only the event times and windows, not the fitted model, so it
        checks whether an imperfect-repair model was warranted at all.

        Parameters
        ----------

        test: {'laplace', 'mil_hdbk_189c'}, optional
            The trend test to run. Default is 'laplace'.
        alternative: {'two-sided', 'increasing', 'decreasing'}, optional
            The alternative hypothesis. Default is 'two-sided'.

        Returns
        -------

        TrendTestResult
            The test result, carrying the statistic, p-value and suggested
            trend direction.
        """
        self._check_has_data("trend_test")
        from surpyval.recurrent import diagnostics

        return diagnostics.trend_test(
            self.data, test=test, alternative=alternative
        )

    def cramer_von_mises(
        self, n_boot: int = 200, seed: "int | None" = None
    ) -> Any:
        """
        Cramer-von Mises goodness-of-fit test of the fitted imperfect-repair
        model.

        These processes have no marginal cumulative intensity, so the
        conditionally-uniform transforms use the compensator built from each
        interval's rescaled increment (the conditional-intensity residual):
        conditional on an item's number of events, ``Lambda(t_k) /
        Lambda(close)`` are iid U(0, 1) when the fitted model is the true one,
        and the statistic measures their departure from uniformity. Because the
        restoration and lifetime / intensity parameters were estimated from the
        same data, the p-value is a parametric bootstrap: each item is
        resimulated from the fitted model the way it was observed -- an
        item whose last row is an end-of-observation (``c=1``) row over the
        same window, with however many events the model gives it there, and
        a failure-truncated item with its observed number of events -- the
        full model is refitted, and the statistic recomputed. Each replicate
        is a multi-start optimisation, so this is much slower than the
        residual diagnostics.

        Parameters
        ----------

        n_boot: int, optional
            Number of bootstrap replicates for the p-value. Default is 200.
        seed: int or numpy.random.Generator, optional
            Seed for a reproducible p-value.

        Returns
        -------

        GoodnessOfFitResult
            The observed statistic and its bootstrap p-value.
        """
        self._check_has_data("cramer_von_mises")
        from surpyval.recurrent import diagnostics

        return diagnostics.cramer_von_mises_renewal(
            self, n_boot=n_boot, seed=seed
        )

    def __repr__(self) -> str:
        title = f"{self.kind} SurPyval Model"
        lines = [
            title,
            "=" * len(title),
            "{:<20}: {}".format(self._dist_label, self.model.dist.name),
            "Fitted by           : "
            + (
                "MLE"
                if hasattr(self, "_neg_ll") or self.how == "MLE"
                else "given parameters (not fitted)"
            ),
        ]
        if getattr(self, "kijima_type", None) is not None:
            lines.append(f"Kijima Type         : {self.kijima_type}")
        if getattr(self, "m", None) is not None:
            lines.append(f"Memory (m)          : {self.m}")
        lines.append(
            "{:<20}: {}".format(self._restoration_label, self.restoration)
        )

        param_string = "\n".join(
            "{:>10}".format(name) + ": " + str(p)
            for p, name in zip(self.model.params, self.model.dist.param_names)
        )
        return "\n".join(lines) + "\nParameters          :\n" + param_string
