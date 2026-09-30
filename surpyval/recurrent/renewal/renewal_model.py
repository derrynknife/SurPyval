from typing import Any, Callable

import numpy as np

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

_EPS = float(np.finfo(float).eps)


def solve_bracketed(
    g: Callable,
    lo: np.ndarray,
    hi: np.ndarray,
    g_lo: np.ndarray,
    g_hi: np.ndarray,
    xtol: float = 0.0,
    rtol: float = 4 * _EPS,
    maxiter: int = 400,
) -> np.ndarray:
    """
    Solve many bracketed root problems ``g_k(x) = 0`` at once, each with
    ``g_k(lo_k) < 0 < g_k(hi_k)``. ``g(x, sel)`` evaluates the problems
    whose indices are ``sel`` at the points ``x``.

    Each step takes a regula falsi point with the Illinois modification
    (halving the value kept at an end that has not moved for two steps),
    and bisects instead whenever the last two steps together did not halve
    the bracket, so every problem converges at least half as fast as
    bisection, and superlinearly once regula falsi takes hold. No point is
    taken within the tolerance of an end, so the last step closes the
    bracket instead of creeping up on the root. A problem is
    done when its bracket is narrower than ``xtol + rtol * |x|``; the
    midpoint of the final bracket is returned.
    """
    lo = np.array(lo, dtype=float)
    hi = np.array(hi, dtype=float)
    g_lo = np.array(g_lo, dtype=float)
    g_hi = np.array(g_hi, dtype=float)
    root = lo + 0.5 * (hi - lo)
    moved = np.zeros(lo.size, dtype=int)  # -1 lo moved last, +1 hi did
    bisect = np.zeros(lo.size, dtype=bool)
    # The bracket's width two steps back.
    earlier = hi - lo
    active = np.flatnonzero(
        hi - lo > xtol + rtol * np.maximum(np.abs(lo), np.abs(hi))
    )
    for _ in range(maxiter):
        if not active.size:
            break
        a = active
        lo_a, hi_a, gl, gh = lo[a], hi[a], g_lo[a], g_hi[a]
        width = hi_a - lo_a
        mid = lo_a + 0.5 * width
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            falsi = hi_a - gh * width / (gh - gl)
            x = np.where(bisect[a] | ~np.isfinite(falsi), mid, falsi)
        # Never evaluate within the tolerance of an end (Brent's tolerance
        # step). Once one end has converged, regula falsi lands on it (or
        # past it, by rounding); a point just inside it lets the next sign
        # change close the bracket.
        tol = 0.5 * (xtol + rtol * np.maximum(np.abs(lo_a), np.abs(hi_a)))
        x = np.clip(x, lo_a + tol, hi_a - tol)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            gx = np.asarray(g(x, a), dtype=float)
        exact = gx == 0
        below = gx < 0
        # A NaN counts as past the root, so the bracket still shrinks.
        above = ~below & ~exact
        new_lo = np.where(below, x, lo_a)
        new_hi = np.where(above, x, hi_a)
        new_gl = np.where(below, gx, gl)
        new_gh = np.where(above, gx, gh)
        new_gh = np.where(below & (moved[a] == -1), 0.5 * new_gh, new_gh)
        new_gl = np.where(above & (moved[a] == 1), 0.5 * new_gl, new_gl)
        moved[a] = np.where(below, -1, np.where(above, 1, 0))
        lo[a], hi[a], g_lo[a], g_hi[a] = new_lo, new_hi, new_gl, new_gh
        new_width = new_hi - new_lo
        bisect[a] = new_width > 0.5 * earlier[a]
        earlier[a] = width
        root[a] = np.where(exact, x, new_lo + 0.5 * new_width)
        done = exact | (
            new_width
            <= xtol + rtol * np.maximum(np.abs(new_lo), np.abs(new_hi))
        )
        active = a[~done]
    return root


def conditional_gaps(
    lifetime: Any, v: np.ndarray, u: np.ndarray
) -> np.ndarray:
    """
    Draw the time to the next failure of items whose virtual ages are
    ``v``, from the uniforms ``u``, for a virtual-age renewal model with
    the lifetime distribution ``lifetime``.

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
    v = np.asarray(v, dtype=float)
    u = np.asarray(u, dtype=float)
    hazard, quantile = _lifetime_functions(lifetime)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        h_v = np.broadcast_to(hazard(v), v.shape)
        target = h_v - np.log(u)
    # No survival past age v (the end of a bounded support): the item
    # fails at once.
    at_once = np.isinf(h_v) & (h_v > 0)
    # u == 0 (or an undefined H): an infinitely late event.
    never = ~at_once & ~np.isfinite(target)
    easy = ~at_once & ~never & (target <= _QF_HAZARD_LIMIT)
    hard = ~at_once & ~never & ~easy
    x = np.array(v, dtype=float)
    if easy.any():
        x[easy] = quantile(-np.expm1(-target[easy]))
    if hard.any():
        x[hard] = _invert_cumulative_hazard(hazard, v[hard], target[hard])
    gap = np.maximum(x - v, 0.0)
    gap[at_once] = 0.0
    gap[never] = np.inf
    return gap


def _lifetime_functions(lifetime: Any) -> "tuple[Callable, Callable]":
    """``H`` and the quantile function of ``lifetime``, on arrays. For a
    plain model (no offset, cure fraction or zero-inflation -- what the
    renewal fitters build) the distribution's own functions are called
    directly, skipping the model methods' argument handling."""
    if (
        getattr(lifetime, "p", None) == 1
        and getattr(lifetime, "f0", None) == 0
        and not getattr(lifetime, "gamma", 0)
    ):
        dist = lifetime.dist
        params = [float(p) for p in lifetime.params]
        lower = dist.support[0]

        def hazard(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            below = x < lower
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                h = dist.Hf(np.where(below, lower, x), *params)
            return np.where(below, 0.0, np.asarray(h, dtype=float))

        def quantile(p: np.ndarray) -> np.ndarray:
            return np.asarray(dist.qf(p, *params), dtype=float)

        return hazard, quantile

    def model_hazard(x: np.ndarray) -> np.ndarray:
        return np.asarray(lifetime.Hf(x), dtype=float)

    def model_quantile(p: np.ndarray) -> np.ndarray:
        return np.asarray(lifetime.qf(p), dtype=float)

    return model_hazard, model_quantile


def _invert_cumulative_hazard(
    hazard: Callable, v: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Solve ``H(x_k) = target_k`` for ``x_k > v_k`` (where ``H(v_k) <
    target_k``), for every ``k`` at once."""

    def g(x: np.ndarray, sel: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return hazard(x) - target[sel]

    n = v.size
    everything = np.arange(n)
    lo = np.array(v, dtype=float)
    step = np.maximum(np.abs(lo), 1.0)
    hi = lo + step
    growing = everything
    for _ in range(2000):
        if not growing.size:
            break
        growing = growing[g(hi[growing], growing) < 0]
        lo[growing] = hi[growing]
        step[growing] *= 2.0
        hi[growing] = lo[growing] + step[growing]
    # Past a bounded support H can be inf or nan; bisect until hi is a
    # finite point at or above the target so the root is bracketed.
    g_hi = g(hi, everything)
    for _ in range(200):
        bad = np.flatnonzero(~np.isfinite(g_hi))
        if not bad.size:
            break
        mid = 0.5 * (lo[bad] + hi[bad])
        up = g(mid, bad) < 0
        lo[bad[up]] = mid[up]
        hi[bad[~up]] = mid[~up]
        g_hi[bad] = g(hi[bad], bad)
    # H jumps to infinity at the end of the support: fail there.
    out = np.where(np.isfinite(g_hi), hi, lo)
    rest = np.flatnonzero(np.isfinite(g_hi) & (g_hi != 0))
    if rest.size:
        out[rest] = solve_bracketed(
            lambda x, sel: g(x, rest[sel]),
            lo[rest],
            hi[rest],
            g(lo[rest], rest),
            g_hi[rest],
            xtol=4 * _EPS * 1e-300,
        )
    return out


class DiscountedMemory:
    """
    For sequences simulated together, the discounted sum
    ``sum_{j < min(m, k)} (1 - rho)^j h_{k - j}`` of the values ``h_1 ..
    h_k`` recorded so far, newest first: the memory term of the ARA (with
    the arrival times) and ARI (with the intensities at the failures)
    models.

    ``record`` is called once per simulation round with the sequences still
    running, so every sequence it covers has the same number of values. With
    infinite memory the sum is carried forward, ``S_k = h_k + (1 - rho)
    S_{k-1}``; with memory ``m`` the last ``m`` rounds are kept and summed.
    """

    def __init__(self, n: int, rho: float, m: "int | float") -> None:
        self.n = n
        self.decay = 1.0 - rho
        self.infinite = bool(np.isinf(m))
        self.m = 0 if self.infinite else int(m)
        self.rounds = 0
        self.total = np.zeros(n)
        self.recent: list = []

    def value(self, idx: np.ndarray) -> np.ndarray:
        if self.infinite:
            return self.total[idx]
        out = np.zeros(idx.size)
        for j, row in enumerate(reversed(self.recent)):
            out += self.decay**j * row[idx]
        return out

    def record(self, idx: np.ndarray, values: np.ndarray) -> None:
        self.rounds += 1
        if self.infinite:
            self.total[idx] = values + self.decay * self.total[idx]
            return
        row = np.full(self.n, np.nan)
        row[idx] = values
        self.recent.append(row)
        if len(self.recent) > self.m:
            self.recent.pop(0)


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
        ``sampler_factory(model, n) -> step`` returning the simulation
        sampler for ``n`` sequences simulated together: ``step(idx, u)``
        draws the next interarrival times of the sequences ``idx`` from the
        uniforms ``u`` (see
        :func:`surpyval.recurrent.simulation.simulate_sequences`).
    restoration_bounds : tuple, optional
        Natural-space ``(lower, upper)`` bounds of the restoration parameter
        (e.g. ``(0, 1)`` for ARA/ARI's ``rho``), used by ``param_cb`` to pick
        a transform that keeps its confidence bounds inside the support.

    ``params`` is every parameter of the model in one vector, in the order
    of ``param_names``: the restoration parameter (``q`` or ``rho``) first,
    then the lifetime distribution's parameters (for ARI, the baseline
    intensity's) -- the order of ``parameter_names``, :meth:`covariance`,
    :meth:`standard_errors` and ``param_cb``. The restoration parameter is
    also ``restoration`` (and ``q`` or ``rho``), and the distribution's
    parameters ``model.params``.

    Examples
    --------
    ``ARA.fit`` returns one. Two systems, repaired at each failure and
    observed to time 60:

    >>> import numpy as np
    >>> from surpyval.recurrent import ARA
    >>> x = np.array([3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60])
    >>> i = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    >>> c = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    >>> model = ARA.fit(x, i, c=c, m=2)

    The repair efficiency ``rho`` is 1 (as good as new), and the Weibull
    is that of the times between failures:

    >>> round(float(model.rho), 3)
    1.0
    >>> model.model.params.round(3)
    array([13.779,  1.917])
    >>> model.param_names
    ['rho', 'alpha', 'beta']
    >>> model.params.round(3)
    array([ 1.   , 13.779,  1.917])

    The expected number of failures per system by times 20 and 60, by
    simulation:

    >>> model.mcf(np.array([20.0, 60.0]), items=1000, random_state=0).round(3)
    array([1.296, 4.56 ])
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

    def _new_batch_sampler(self, n: int) -> Callable:
        return self._sampler_factory(self, n)

    @property
    def params(self) -> np.ndarray:
        """Every parameter of the model, in the order of ``param_names``:
        the restoration parameter, then the distribution's parameters."""
        return np.concatenate(
            [[self.restoration], np.asarray(self.model.params, dtype=float)]
        ).astype(float)

    @property
    def param_names(self) -> list:
        """The names of ``params``, in order: the restoration parameter
        (``q`` or ``rho``), then the distribution's parameters. The same
        as ``parameter_names``, but available on a model built from
        parameters too."""
        return list(self._parameter_names())

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
        self,
        test: str = "laplace",
        alternative: str = "two-sided",
        *,
        alpha_ci: float = 0.05,
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
        alpha_ci: float, optional
            The significance level at which the result's ``trend`` is
            judged (default 0.05, keyword only): a trend is named only when
            ``p_value < alpha_ci``.

        Returns
        -------

        TrendTestResult
            The test result, carrying the statistic, p-value, the
            direction of the statistic and the trend concluded at
            ``alpha_ci``.
        """
        self._check_has_data("trend_test")
        from surpyval.recurrent import diagnostics

        return diagnostics.trend_test(
            self.data, test=test, alternative=alternative, alpha_ci=alpha_ci
        )

    def cramer_von_mises(
        self, n_boot: int = 200, random_state: "int | None" = None
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
        random_state: int or numpy.random.Generator, optional
            Seed for a reproducible p-value.

        Returns
        -------

        GoodnessOfFitResult
            The observed statistic and its bootstrap p-value.
        """
        self._check_has_data("cramer_von_mises")
        from surpyval.recurrent import diagnostics

        return diagnostics.cramer_von_mises_renewal(
            self, n_boot=n_boot, random_state=random_state
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
