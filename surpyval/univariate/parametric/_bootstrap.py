"""The parametric bootstrap bounds of a fitted univariate model
(``method="bootstrap"`` on ``cb``, ``quantile_cb`` and ``param_cb``,
#645).

The Wald bounds of an offset (3-parameter) fit hold the offset at its
estimate: the offset is a threshold, whose likelihood is not regular, so
the fit estimates no standard error for it. Near the offset those bounds
are far too narrow (a 90% bound on a 3-parameter Weibull's B1 life from
30 failures covered 40% of the time, #645), and the likelihood-ratio
bounds are not available for offset fits. The bootstrap includes the
offset's uncertainty, as it does every other parameter's.

The model is resampled, not the data, as for the regressions
(``univariate.regression._bootstrap``, #617): each resample draws a
failure time for every unit from the fitted model (within the unit's
truncation window), censors it as the unit was censored -- Davison &
Hinkley's conditional bootstrap, a censored unit at its own censoring
time and a failed one at a time drawn from the censoring distribution
beyond its failure -- and refits the same model to the result, offset and
fixed parameters as fitted. The bounds are the bias-corrected percentile
intervals of the refits (Efron 1987): the percentiles of the refits
moved by the share of them below the estimate. The BCa acceleration the
regressions add needs the covariance of every parameter, and an offset
fit's leaves the offset out.

On #645's design (a 3-parameter Weibull of shape 1.5, 30 failures) the
90% bound on the B1 life covered 63-64% over two runs of 100 samples,
where the Wald bound covered 35%; on B10 82-89% (Wald 69%), on ``sf``
just above the offset 66-68% (Wald 53%). It is still short at the offset
itself when the shape is below 2: the offset's estimate is biased up
(it is at most the first failure), and no interval of the refits makes
up for that -- the percentile interval covered B1 45% and the basic
(reflected) interval 75% but ``sf`` above the offset 56%.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

from surpyval.univariate.regression._bootstrap import (
    _REFIT_ERRORS,
    FAILED_SHARE,
    _Design,
    bca_bounds,
    check_n_boot,
)
from surpyval.utils.no_maximum import quiet_maximum_warnings
from surpyval.utils.rng import as_generator
from surpyval.utils.warnings import caller_stacklevel


@dataclass
class Refits:
    """The refits of one bootstrap of a model: each refit's offset and
    parameters, and how many of the ``n_boot`` refits reached no
    verified maximum or failed outright."""

    #: The model's own ``(gamma, *params)`` the refits are of.
    point: tuple
    n_boot: int
    #: The kept refits, as models (``from_params`` at their estimates).
    models: list
    #: ``(n_kept, 1 + len(params))``: each kept refit's offset (0 without
    #: one) and parameters.
    values: npt.NDArray
    no_maximum: int
    unverified: int
    failed: int
    model_maximum: str = "verified"

    def warn(self) -> None:
        """One warning: that the model itself has no finite maximum, or
        that more than ``FAILED_SHARE`` of the refits did not reach a
        verified maximum or failed."""
        if self.model_maximum == "no finite maximum":
            warnings.warn(
                "This model's likelihood has no finite maximum (as its fit "
                "warned), so the data simulated from it run off the same "
                "way and the bootstrap bounds close onto its meaningless "
                "estimate. They are not a confidence bound.",
                UserWarning,
                stacklevel=caller_stacklevel(),
            )
            return
        bad = self.no_maximum + self.unverified + self.failed
        if bad <= FAILED_SHARE * self.n_boot:
            return
        parts = []
        if self.no_maximum:
            parts.append(f"{self.no_maximum} with no finite maximum")
        if self.unverified:
            parts.append(f"{self.unverified} at an unverified maximum")
        if self.failed:
            parts.append(f"{self.failed} failed and left out")
        warnings.warn(
            f"{bad} of the {self.n_boot} bootstrap refits did not reach a "
            f"verified maximum of the likelihood ({', '.join(parts)}). "
            "A refit with no finite maximum is kept at the estimate it "
            "reached, so that it counts at its end of the bounds; with "
            "this many, the data barely determine the model and the "
            "bounds are rough.",
            UserWarning,
            stacklevel=caller_stacklevel(),
        )


def check_resamplable(model: Any) -> None:
    """Refuse, with one message each, what the bootstrap cannot resample
    faithfully."""
    if getattr(model, "data", None) is None:
        raise ValueError(
            "Bootstrap bounds need the data the model was fitted to; this "
            "model was built from parameters, or restored from a dict "
            "saved without its data (save it with to_dict(with_data=True) "
            "to keep them). Use method='wald'."
        )
    if model.lfp or model.zi:
        raise ValueError(
            "Bootstrap bounds are not available for a limited-failure or "
            "zero-inflated model yet; use method='wald'."
        )
    if getattr(model.dist, "discrete", False):
        raise ValueError(
            "Bootstrap bounds are for continuous distributions; for a "
            "discrete one use method='wald' or method='lr'."
        )


class _UnivariateDesign(_Design):
    """What a resample of a univariate model's data keeps: each unit's
    truncation window and censoring (a row with ``n = k`` is ``k``
    units); the censoring distribution is the regressions' (``_Design``).
    """

    def __init__(self, model: Any) -> None:
        check_resamplable(model)
        data = model.data
        c = np.asarray(data["c"])
        if np.any((c != 0) & (c != 1)):
            raise ValueError(
                "Bootstrap bounds resample exact and right-censored data "
                "(truncated or not); this model was fitted to left- or "
                "interval-censored data, whose inspection times a resample "
                "would need and the data do not record. Use method='wald'."
            )
        x = np.asarray(data["x"], dtype=float)
        if x.ndim == 2:
            x = x[:, 0]
        units = np.repeat(np.arange(len(c)), np.asarray(data["n"], dtype=int))
        self.x = x[units]
        self.failed = c[units] == 0
        t = np.asarray(data["t"], dtype=float).reshape(-1, 2)[units]
        self.tl, self.tr = t[:, 0], t[:, 1]
        self.truncated = bool(
            np.any(np.isfinite(self.tl)) or np.any(np.isfinite(self.tr))
        )
        self._censoring_distribution()


def _point(model: Any) -> tuple:
    return (float(model.gamma), *(float(p) for p in model.params))


def refits(model: Any, n_boot: Any, random_state: Any) -> Refits:
    """The bootstrap refits of ``model``: drawn, or those kept on the
    model for this ``n_boot`` and integer ``random_state`` while its
    parameters are as they are (``random_state=None`` or a ``Generator``
    draws afresh each time, from that stream)."""
    n_boot = check_n_boot(n_boot)
    design = _UnivariateDesign(model)
    point = _point(model)
    key = None
    if isinstance(random_state, (int, np.integer)) and not isinstance(
        random_state, bool
    ):
        key = (n_boot, int(random_state))
        kept = getattr(model, "_bootstrap_refits", None)
        if kept is not None and key in kept and kept[key].point == point:
            return kept[key]
    out = _draw(model, design, point, n_boot, as_generator(random_state))
    if key is not None:
        if getattr(model, "_bootstrap_refits", None) is None:
            model._bootstrap_refits = {}
        model._bootstrap_refits[key] = out
    return out


def _fixed(model: Any) -> dict:
    """The parameters the user fixed at fit time, by name, at their
    values."""
    names = {i: name for name, i in model.param_map.items()}
    out = {}
    for i in sorted(model._user_fixed_idx()):
        name = names[i]
        out[name] = (
            float(model.gamma)
            if name == "gamma"
            else float(model.params[model.dist.param_map[name]])
        )
    return out


def _draw(
    model: Any,
    design: _UnivariateDesign,
    point: tuple,
    n_boot: int,
    rng: np.random.Generator,
) -> Refits:
    """``n_boot`` resamples simulated from ``model`` with ``design`` and
    refitted (see the module docstring)."""
    with np.errstate(all="ignore"):
        F_lo = np.where(
            np.isfinite(design.tl), model.ff(np.nan_to_num(design.tl)), 0.0
        )
        F_hi = np.where(
            np.isfinite(design.tr), model.ff(np.nan_to_num(design.tr)), 1.0
        )
    fixed = _fixed(model) or None
    init = np.array(
        ([model.gamma] if model.offset else []) + list(model.params),
        dtype=float,
    )
    if fixed:
        free = [
            i
            for name, i in sorted(
                model.param_map.items(), key=lambda kv: kv[1]
            )
            if name not in fixed
        ]
        init = init[free]
    models, values = [], []
    counts = {"no finite maximum": 0, "unverified": 0, "failed": 0}
    for _ in range(n_boot):
        u = rng.uniform(size=design.x.size)
        with np.errstate(all="ignore"):
            T = np.asarray(model.qf(F_lo + u * (F_hi - F_lo)), dtype=float)
        C = design.censoring(rng)
        censored = T > C
        x = np.where(censored, C, T)
        kwargs: dict[str, Any] = {"c": censored.astype(int)}
        if design.truncated:
            kwargs["tl"] = design.tl
            kwargs["tr"] = design.tr
        try:
            with warnings.catch_warnings(), quiet_maximum_warnings():
                warnings.simplefilter("ignore")
                refit = model.dist.fit(
                    x,
                    offset=model.offset,
                    fixed=fixed,
                    # From the estimate: each resample's maximum is within
                    # sampling error of it
                    init=init,
                    **kwargs,
                )
        except _REFIT_ERRORS:
            counts["failed"] += 1
            continue
        if refit.maximum in counts:
            counts[refit.maximum] += 1
        models.append(refit)
        values.append(_point(refit))
    return Refits(
        point=point,
        n_boot=n_boot,
        models=models,
        values=np.array(values).reshape(len(values), len(point)),
        no_maximum=counts["no finite maximum"],
        unverified=counts["unverified"],
        failed=counts["failed"],
        model_maximum=model.maximum,
    )


def _bounds(
    model: Any,
    fits: Refits,
    value: Callable[[Any], npt.NDArray],
    alpha_ci: float,
    bound: str,
) -> npt.NDArray:
    """The bias-corrected percentile bounds on ``value(m)``, a function
    of a model with one value per query point, over the refits about its
    value at ``model``."""
    est = np.asarray(value(model), dtype=float)
    shape = est.shape
    draws = np.empty((len(fits.models),) + shape)
    with np.errstate(all="ignore"):
        for i, refit in enumerate(fits.models):
            draws[i] = np.reshape(np.asarray(value(refit), dtype=float), shape)
    cb = bca_bounds(
        draws.reshape(len(draws), -1),
        est.reshape(-1),
        np.zeros(est.size),
        alpha_ci,
        bound,
    )
    return cb.reshape(shape + cb.shape[1:])


def cb_bootstrap(
    model: Any,
    x: npt.NDArray,
    on: str,
    alpha_ci: float,
    bound: str,
    n_boot: Any,
    random_state: Any,
) -> npt.NDArray:
    """``cb(method="bootstrap")``: the bounds of ``on`` at each ``x``
    over the refits. The percentile interval goes with a monotone
    transform, so those on ``sf``, ``ff`` and ``Hf`` are one interval."""
    on = {"R": "sf", "F": "ff"}.get(on, on)
    fits = refits(model, n_boot, random_state)
    fits.warn()
    t = np.asarray(x, dtype=float)
    # On the cumulative hazard, which increases with every parameter's
    # effect the same way at each x; sf and ff are carried from it.
    fn = {"sf": "Hf", "ff": "Hf"}.get(on, on)
    side = bound
    if on == "sf" and bound != "two-sided":
        side = {"lower": "upper", "upper": "lower"}[bound]
    cb = _bounds(model, fits, lambda m: getattr(m, fn)(t), alpha_ci, side)
    if on == "sf":
        if bound == "two-sided":
            cb = cb[..., ::-1]
        with np.errstate(over="ignore"):
            return np.exp(-cb)
    if on == "ff":
        with np.errstate(over="ignore"):
            return -np.expm1(-cb)
    return cb


def quantile_cb_bootstrap(
    model: Any,
    p: npt.NDArray,
    alpha_ci: float,
    bound: str,
    n_boot: Any,
    random_state: Any,
) -> npt.NDArray:
    """``quantile_cb(method="bootstrap")``: the bounds of the refits'
    quantiles ``qf(p)``."""
    fits = refits(model, n_boot, random_state)
    fits.warn()
    return _bounds(model, fits, lambda m: m.qf(p), alpha_ci, bound)


def param_cb_bootstrap(
    model: Any,
    name: str,
    alpha_ci: float,
    bound: str,
    n_boot: Any,
    random_state: Any,
) -> npt.NDArray:
    """``param_cb(method="bootstrap")``: the bounds of the refits'
    parameter ``name`` -- the offset ``gamma`` among them. A parameter
    fixed at fit time is known: its interval is its value."""
    if name == "gamma":
        if not model.offset:
            raise ValueError("'gamma' is only estimated for offset models")
        idx = 0
    elif name in model.dist.param_map:
        idx = 1 + model.dist.param_map[name]
    else:
        valid = (["gamma"] if model.offset else []) + list(
            model.dist.parameter_names
        )
        raise ValueError(
            f"Unknown parameter {name!r}; expected one of {valid}"
        )
    if model._is_fixed_param(name):
        value = _point(model)[idx]
        return np.array([value, value] if bound == "two-sided" else [value])
    fits = refits(model, n_boot, random_state)
    fits.warn()
    est = np.array([_point(model)[idx]])
    cb = bca_bounds(
        fits.values[:, idx][:, None], est, np.zeros(1), alpha_ci, bound
    )
    return np.atleast_1d(cb[0])
