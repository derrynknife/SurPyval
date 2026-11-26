"""The joint model fitter: :func:`JointPH` and ``WeibullJointPH`` (#891).

The hazard follows the marker's current value, its slope, the area under
it, or more than one (``link``)."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
from autograd import hessian, jacobian, value_and_grad
from scipy.optimize import minimize

from surpyval.univariate.regression._aliasing import (
    aliased_columns,
    constant_columns,
    dataframe_covariates,
    fit_columns,
    warn_aliased,
)
from surpyval.utils.covariates import finite_covariate_mask
from surpyval.utils.fitter_repr import FitterRepr
from surpyval.utils.no_maximum import warn_unverified
from surpyval.utils.readings_entry import markers_by_row, takes_joint_readings
from surpyval.utils.surpyval_data import accepts_surpyval_data
from surpyval.utils.units import normalise_unit, takes_unit
from surpyval.utils.validation import check_option
from surpyval.utils.warnings import (
    caller_stacklevel,
    ignoring_warnings,
    warn_no_covariance,
)

from ._likelihood import (
    LINKS,
    MAX_VARYING,
    JointArrays,
    Layout,
    MarkerPaths,
    Model,
    PathFunction,
    block_labels,
    check_integration,
    check_lag,
    check_random_effects,
    check_weight,
    neg_ll,
    path_jacobian,
    place,
    several_markers,
)
from ._markers import (
    marker_start,
    settled_marker_start,
    survival_stage,
    survival_start,
)
from .compare import LinkComparison, compare_links
from .joint_model import (
    HOWS,
    JointModel,
    covariate_parts,
    effect_name,
    parameter_names,
    path_formula,
    reported_params,
    shareable,
    structure_words,
)

#: The most rounds of placing the nodes and searching from them.
_ROUNDS = 10
#: Rounds stop once the log-likelihood moves less than this between them
#: (relative, with an absolute floor of the same in log-likelihood units).
_ROUND_TOL = 1e-10
#: A verified maximum's gradient, in standard errors of each parameter.
_GTOL = 1e-4

#: What a built-in path cannot give the hazard to follow, and why: the
#: Lloyd-Lipow path has no finite area from age 0, and its slope and the
#: logarithmic path's are unbounded at age 0 (the power path's is too for
#: a power below 1, which only the fit can tell: ``_check_power``).
_CANNOT_FOLLOW = {
    ("Lloyd-Lipow", "cumulative"): (
        "The Lloyd-Lipow path a - b / t has no finite area from age 0, so "
        "the hazard cannot follow its cumulative"
    ),
    ("Lloyd-Lipow", "slope"): (
        "The Lloyd-Lipow path a - b / t has the slope b / t^2, unbounded at "
        "age 0, so the hazard cannot follow its slope"
    ),
    ("Logarithmic", "slope"): (
        "The Logarithmic path a + b ln(t) has the slope b / t, unbounded at "
        "age 0, so the hazard cannot follow its slope"
    ),
}


def cannot_follow(path_model: Any, links: Any) -> list[str]:
    """Why the hazard cannot follow each of ``links`` of ``path_model``
    (a built-in path, or one on links), one reason a link; empty where it
    can follow them all."""
    name = str(getattr(getattr(path_model, "base", path_model), "name", ""))
    return [
        _CANNOT_FOLLOW[(name, link)]
        for link in links
        if (name, link) in _CANNOT_FOLLOW
    ]


#: The built-in paths whose value at age 0 is infinite, so the hazard
#: cannot follow their value with a lag (before the lag it follows the
#: value at age 0).
_INFINITE_AT_ZERO = {
    "Lloyd-Lipow": "The Lloyd-Lipow path a - b / t",
    "Logarithmic": "The Logarithmic path a + b ln(t)",
}


def _check_lag_at_zero(path_model: Any, links: Any, lag: float) -> None:
    """Refuse a lag on the value of a path whose value at age 0 is
    infinite: before the lag the hazard follows the value at age 0."""
    name = str(getattr(getattr(path_model, "base", path_model), "name", ""))
    if lag > 0 and "value" in links and name in _INFINITE_AT_ZERO:
        raise ValueError(
            f"{_INFINITE_AT_ZERO[name]} is infinite at age 0, and with a "
            "lag the hazard follows the value at age 0 until the lag has "
            "passed, so it cannot follow this path's value with a lag. "
            "Use lag=0, or a path that is finite at age 0, such as "
            "LinearPath, ExponentialPath or PowerPath."
        )


def _marker_lag_lines(lag: Any, weight: Any) -> list:
    """The fitter printout's lines of the lags and weights of several
    markers (one for all, or by marker)."""
    out = []
    if isinstance(lag, Mapping):
        said = [f"{m} {v:g}" for m, v in lag.items() if v > 0]
        if said:
            out.append("lag: " + ", ".join(said))
    elif lag > 0:
        out.append(f"lag: {lag:g}")
    if isinstance(weight, Mapping):
        said = [f"{m} {w[0]} {w[1]:g}" for m, w in weight.items() if w]
        if said:
            out.append("weight: " + ", ".join(said))
    elif weight is not None:
        out.append("weight: {} {:g}".format(*weight))
    return out


def _check_start_at_zero(fitter: Any, mu: npt.NDArray) -> None:
    """Refuse a lag on the value of a path (of one's own) whose value at
    age 0 is not finite at the average item's parameters ``mu``."""
    if fitter.lag <= 0 or "value" not in fitter.link:
        return
    with np.errstate(all="ignore"):
        at_zero = np.asarray(
            fitter.path_model.path(0.0, *np.asarray(mu, dtype=float)),
            dtype=float,
        )
    if not np.all(np.isfinite(at_zero)):
        raise ValueError(
            f"The {fitter.path_model.name} path is not finite at age 0 for "
            "the average item (the readings put it there), and with a lag "
            "the hazard follows the value at age 0 until the lag has "
            "passed. Use lag=0, or a path that is finite at age 0."
        )


def _instead(path_model: Any, survival: str, refused: Any) -> str:
    """What to use in place of the ``refused`` links of ``path_model``:
    the links it can take, and for the slope a path whose slope is finite
    at age 0."""
    usable = [
        link
        for link in LINKS
        if link not in refused
        and not cannot_follow(path_model, (link,))
        and not (survival == "AFT" and link == "cumulative")
    ]
    out = "Use link=" + " or ".join(repr(link) for link in usable)
    if "slope" in refused:
        return out + (
            ", or for the slope link a path whose slope is finite at age 0, "
            "such as LinearPath, QuadraticPath or ExponentialPath."
        )
    return out + ", or another path."


def _infinite_near_zero(survival: str) -> str:
    rate = "ageing rate" if survival == "AFT" else "failure rate"
    return (
        f"(near age 0 the {rate} would be infinite for every item whose "
        "slope has the sign of its effect)"
    )


def _check_power(fitter: Any, mu: npt.NDArray, found_by: str) -> None:
    """Refuse the slope link of a power path ``a t^b`` whose average
    item's power (from ``mu``, the path parameters' means on the path
    model's scale) is below 1: its slope ``a b t^(b - 1)`` is then
    unbounded at age 0, where the cumulative hazard starts."""
    path = fitter.path_model
    base = getattr(path, "base", path)
    if "slope" not in fitter.link or getattr(base, "name", None) != "Power":
        return
    mu = np.asarray(mu, dtype=float)
    natural = path.to_natural(mu) if path is not base else mu
    power = float(natural[list(base.parameter_names).index("b")])
    if power < 1.0:
        raise ValueError(
            "The power path a t^b has the slope a b t^(b - 1), unbounded at "
            f"age 0 for a power b below 1, and {found_by} the average "
            f"item's power at b = {power:.3g}, so the hazard cannot follow "
            f"its slope {_infinite_near_zero(fitter.survival)}. "
            + _instead(path, fitter.survival, ("slope",))
        )


class JointPHFitter(FitterRepr):
    """
    The joint model of a marker -- a measurement read repeatedly on each
    item, such as a vibration level, a wear or oil-debris reading, or a
    biomarker -- and the item's lifetime, the marker following a
    degradation path model and the survival part proportional hazards:
    built by :func:`JointPH`, ready-made as ``WeibullJointPH`` (a
    straight line). :meth:`fit` returns a
    :class:`~surpyval.univariate.joint.joint_model.JointModel`.
    """

    #: The ``repr`` (#614)
    fitter_kind = "joint model fitter"
    #: The survival part: the marker multiplies the hazard (``"PH"``), or
    #: speeds up the item's ageing (``"AFT"``, :class:`JointAFTFitter
    #: <surpyval.univariate.joint.joint_aft.JointAFTFitter>`).
    survival = "PH"
    #: A start for the path part (``mu``, ``Delta``, ``L``, ``sigma``)
    #: from another link's fit of the same data (``compare_links``).
    _path_start: dict | None = None
    #: The markers' names, for a model of several markers (``path_model``
    #: a dict of their path models); ``None`` for one marker.
    markers: list | None = None
    path_model: Any
    link: Any

    @property
    def takes_several_markers(self) -> bool:
        """Whether the fitter fits several markers (a SurpyvalData's
        named measurements are its markers)."""
        return self.markers is not None

    def __init__(
        self,
        distribution: Any,
        path_model: Any,
        varying: Any = None,
        link: Any = "value",
        lag: float = 0.0,
        weight: Any = None,
        random_effects: Any = None,
    ) -> None:
        from surpyval import Weibull
        from surpyval.degradation.path_models import LinearPath, PathModel

        if distribution is not Weibull:
            name = getattr(distribution, "name", type(distribution).__name__)
            raise ValueError(
                "The joint models have a Weibull baseline for now: "
                f"Joint{self.survival}(Weibull, path_model), or "
                f"WeibullJoint{self.survival}; got {name}."
            )
        if isinstance(path_model, Mapping):
            self._set_markers(
                distribution,
                path_model,
                varying,
                link,
                lag,
                weight,
                random_effects,
            )
            return
        if not isinstance(path_model, PathModel):
            raise ValueError(
                "path_model is a degradation path model, such as "
                "surpyval.degradation.LinearPath or QuadraticPath, or a "
                f"PathModel of your own; got {path_model!r}."
            )
        names = list(path_model.parameter_names)
        if varying is None and hasattr(path_model, "default_varying"):
            # a path that says which of its parameters differ between
            # items (a spline: its level and rate, the bends common)
            varying = tuple(path_model.default_varying)
        if varying is None:
            varying = tuple(names)
        if isinstance(varying, str):
            varying = (varying,)
        varying = tuple(varying)
        unknown = [v for v in varying if v not in names]
        if unknown or not varying or len(set(varying)) < len(varying):
            raise ValueError(
                "varying names the path parameters that vary between "
                f"items, at least one, each once, of {names}; got "
                f"{list(varying)}."
            )
        try:
            links = [link] if isinstance(link, str) else list(link)
        except TypeError:
            links = []
        if (
            not links
            or any(name not in LINKS for name in links)
            or len(set(links)) < len(links)
        ):
            raise ValueError(
                "link names what of the marker's path the hazard follows: "
                "'value' (its current value), 'slope' (its rate of change) "
                "or 'cumulative' (the area under it from age 0), or a "
                f"list of them, each once; got {link!r}."
            )
        self.link = tuple(name for name in LINKS if name in links)
        reasons = cannot_follow(path_model, self.link)
        if reasons:
            refused = [
                link
                for link in self.link
                if cannot_follow(path_model, (link,))
            ]
            said = "; ".join(reasons)
            if "slope" in refused:
                said += " " + _infinite_near_zero(self.survival)
            raise ValueError(
                said + ". " + _instead(path_model, self.survival, refused)
            )
        every = parameter_names(
            names, [n for n in names if n in varying], [], links=self.link
        )
        twice = sorted({n for n in every if every.count(n) > 1})
        if twice:
            raise ValueError(
                f"The {path_model.name} path's parameter names {twice} are "
                "also names of the joint model's other parameters; rename "
                "them in the path model."
            )
        self.lag = check_lag(lag)
        _check_lag_at_zero(path_model, self.link, self.lag)
        self.weight = check_weight(weight)
        if self.weight is not None and "cumulative" not in self.link:
            raise ValueError(
                "'weight' weights the accumulated exposure, the area "
                "under the marker's path that the 'cumulative' link "
                f"follows, and the link here is {list(self.link)!r}, "
                "which has no area to weight. Add the link "
                "(link='cumulative', or link=['value', 'cumulative']), or "
                "leave weight out."
            )
        self.dist = distribution
        self.path_model = path_model
        # in the path's order
        self.varying = tuple(n for n in names if n in varying)
        if random_effects is None:
            # the path's own default (a spline: its level and rate
            # correlated, each varying bend on its own), else every pair
            chooser = getattr(path_model, "default_random_effects", None)
            random_effects = (
                chooser(self.varying) if chooser is not None else "full"
            )
        self.random_effects = check_random_effects(
            random_effects, self.varying
        )
        plain = "".join(ch for ch in path_model.name if ch.isalnum())
        self.name = (
            f"{distribution.name}Joint{self.survival}"
            if path_model is LinearPath
            else f"{distribution.name}{plain}Joint{self.survival}"
        )

    def _set_markers(
        self,
        distribution: Any,
        path_models: Mapping,
        varying: Any,
        link: Any,
        lag: float,
        weight: Any,
        random_effects: Any,
    ) -> None:
        """The fitter of several markers (``path_model`` a dict
        ``{marker: path model}``)."""
        from ._multi import (
            MarkerSet,
            check_marker_dimension,
            check_markers,
            lag_of,
            marker_lags,
            marker_links,
            marker_random_effects,
            marker_varying,
            marker_weights,
        )

        names, models = check_markers(path_models)
        self.markers = names
        self.path_model = MarkerSet(names, models)
        self.varying = marker_varying(names, models, varying)
        check_marker_dimension(self.varying)
        self.link = marker_links(names, link)
        self.lag = marker_lags(names, lag)
        self.weight = marker_weights(names, weight, self.link)
        for name, model in zip(names, models):
            own = tuple(k for m, k in self.link if m == name)
            reasons = cannot_follow(model, own)
            if reasons:
                refused = [k for k in own if cannot_follow(model, (k,))]
                said = "; ".join(reasons)
                if "slope" in refused:
                    said += " " + _infinite_near_zero(self.survival)
                raise ValueError(
                    f"Marker {name!r}: "
                    + said
                    + ". "
                    + _instead(model, self.survival, refused)
                )
            _check_lag_at_zero(model, own, lag_of(self.lag, name))
        self.dist = distribution
        if random_effects is None:
            random_effects = "full"
        self.random_effects = check_random_effects(
            marker_random_effects(random_effects, self.varying), self.varying
        )
        self.name = f"{distribution.name}Joint{self.survival}"

    def _repr_details(self) -> list[str]:
        if self.markers is not None:
            out = [f"{self.dist.name} baseline"]
            for name, model in zip(self.markers, self.path_model.models):
                out.append(
                    "{}: {} path, {}".format(
                        name, model.name, path_formula(model)
                    )
                )
            out.append("varying: " + ", ".join(self.varying))
            if self.random_effects != "full":
                out.append(
                    "random effects: " + structure_words(self.random_effects)
                )
            out.append("link: " + ", ".join(f"{m} {k}" for m, k in self.link))
            out += _marker_lag_lines(self.lag, self.weight)
            return out
        names = list(self.path_model.parameter_names)
        out = [
            f"{self.dist.name} baseline",
            "{} path, {}".format(
                self.path_model.name, path_formula(self.path_model)
            ),
        ]
        if len(self.varying) < len(names):
            out.append("varying: " + ", ".join(self.varying))
        if self.random_effects != "full":
            out.append(
                "random effects: " + structure_words(self.random_effects)
            )
        if self.link != ("value",):
            out.append("link: " + ", ".join(self.link))
        if self.lag > 0:
            out.append(f"lag: {self.lag:g}")
        if self.weight is not None:
            out.append("weight: {} {:g}".format(*self.weight))
        return out

    @dataframe_covariates
    @takes_unit
    @takes_joint_readings
    @accepts_surpyval_data
    def fit(
        self,
        x: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        t: npt.ArrayLike | None = None,
        Z: npt.ArrayLike | None = None,
        markers: Any = None,
        tl: npt.ArrayLike | None = None,
        tr: npt.ArrayLike | None = None,
        marker_reference: float | None = None,
        path_covariates: Any = None,
        hazard_covariates: Any = None,
        i: npt.ArrayLike | None = None,
        m: Any = None,
        marker_unit: str | None = None,
        how: str = "MLE",
        integration: str | None = None,
        e: npt.ArrayLike | None = None,
        shared: Any = None,
    ) -> JointModel:
        """
        Fit the joint model of the marker and the lifetime by maximum
        likelihood (or, with ``how="two-stage"``, by the quick and
        approximate two-stage estimate).

        Each item's marker follows the fitter's path model with
        parameters of its own -- those that vary between items
        multivariate normal about means that its covariates shift, the
        rest common -- read with normal noise; its hazard is Weibull
        proportional hazards in the path's current, noise-free value (or
        its slope or the area under it: the fitter's ``link``) and its
        covariates. A :func:`~surpyval.univariate.joint.joint_aft.JointAFT`
        fitter's (``WeibullJointAFT``) has the accelerated survival part
        in their place: the path and the covariates speed up the item's
        ageing, and its survival is the Weibull's at the age it has used
        up. The varying parameters are integrated out of each
        item's likelihood by adaptive Gauss-Hermite quadrature (7 points
        a dimension, the nodes placed again at the new parameters until
        the log-likelihood stops moving; ``integration`` says on which
        nodes), and the likelihood is maximised
        with autograd's gradient; the covariance is the inverse of its
        Hessian. The fit starts from the path fitted to each item alone,
        the spread of those fits, and a survival fit with each item's
        predicted path.

        That start is also a fit of its own, ``how="two-stage"``: first
        each item's path from its readings alone, pulled towards the
        fleet's average where it has few readings, then a Weibull
        proportional hazards (or accelerated) fit of the lifetimes with
        each item's predicted path taken as a known covariate. It takes a
        fraction of a second, and it is biased: it forgets that each path
        is an estimate and that the readings stop when an item fails (in
        a simulation study, the marker's effect was 8% low with readings
        every six months, where the joint fit's was within 2.4%), and its
        standard errors (of the survival part only) are too small. It is
        for a first look and for comparison with the joint fit; the model
        says it is approximate (``how``), and has no likelihood to
        compare (``aic()``, ``bic()`` and ``neg_ll()`` are refused).

        The data are usually a :class:`~surpyval.SurpyvalData` of
        :class:`~surpyval.ItemLifetime` records with ``markers``:
        ``WeibullJointPH.fit(SurpyvalData.from_items(items))``. Items may
        be failed (``observed``), still running (``right_censored``),
        found failed at an inspection (``left_censored``: failed by that
        age) or between two (``interval_censored``), and may have come
        under observation late (``left_truncation``) or be in the data
        only because they failed by an age (``right_truncation``): the
        likelihood is then that of an item that reached its entry age and
        failed by its right truncation, over the population's paths. A
        covariate the data cannot determine (a constant, or a combination
        of others) is aliased: its coefficients are ``nan``, with one
        warning.

        Every covariate shifts every path parameter's mean and enters the
        hazard, unless ``path_covariates`` or ``hazard_covariates`` say
        otherwise: a condition may change how fast the marker rises
        without changing the risk at a given reading (path only), or the
        risk at a given reading without changing the marker (hazard
        only).

        Parameters
        ----------
        x : SurpyvalData or array_like
            The data, or each item's age at failure or censoring.
        c, n, t, tl, tr : array_like, optional
            As for every fit, with arrays: the censoring flag (0 failed, 1
            still running, -1 failed by ``x``, 2 failed in ``x``'s
            ``[lower, upper]`` row), the counts, and the truncation.
        Z : array_like, optional
            Each item's covariates, a row each (a DataFrame's columns name
            them). By default they shift every path parameter's mean and
            enter the hazard.
        markers : list of dict or Measurements
            With arrays: each item's readings, ``{age: value}`` (``{}``
            for none), in the rows' order; or a
            :class:`~surpyval.Measurements` of one measurement, matched
            to the rows by unit (``i``), never by position: an item with
            no readings has none, as ``{}``, and readings of a unit with
            no row are warned of and not used. Beside a SurpyvalData
            without readings, a Measurements is attached to it by its
            units (as ``data.m``).
        marker_reference : float, optional
            The marker value at which ``alpha`` is reported (the
            baseline's item has its marker there). Default the mean
            reading. It changes ``alpha`` only. Refused where the hazard
            does not follow the marker's value (the fitter's ``link``).
        path_covariates : list of str or dict, optional
            The covariates that shift the path parameters' means
            (``a_<cov>``, ``b_<cov>``, ...): a list (each path parameter),
            or ``{path parameter: [covariate, ...]}`` (one left out is not
            shifted). Default every covariate; ``[]`` for none.
        hazard_covariates : list of str, optional
            The covariates in the hazard (``<cov>``). Default every
            covariate; ``[]`` for none. Each covariate must be in one part
            or both. Unnamed covariates are ``coef_0``, ``coef_1``, ...
        i : array_like, optional
            With arrays and ``markers`` a Measurements: each row's unit,
            one row per unit.
        m : Measurements, optional
            The other name of ``markers``, as ``SurpyvalData(..., m=...)``
            calls the readings; the same in every way. Giving both is a
            TypeError.
        marker_unit : str, optional
            The unit the marker is read in (``"mm/s"``, ``"ppm"``), a
            label kept as typed, as ``unit=`` is for the ages. Readings
            given as a ``Measurements`` with a unit (``units=``) give it
            themselves; a different one here is refused. The model keeps
            it as ``marker_unit`` and, with ``unit``, names what each
            link's effect is per (``effect_units``: a
            ``cumulative_effect`` is per ``(mm/s)·years``, say). A label
            only: nothing is converted.
        how : str, optional
            ``"MLE"`` (the default), the joint likelihood's maximum; or
            ``"two-stage"``, the approximate two-stage estimate (the
            paths from the readings alone, then the survival part with
            each item's predicted path as a known covariate): fast, but
            its estimates are biased, its survival part's standard errors
            are understated and its marker part has none.
        integration : {"grid", "pruned", "laplace"}, optional
            How each item's likelihood is integrated over its varying
            path parameters: ``"grid"``, on the full grid of 7 points in
            each (49 nodes for two, 343 for three, 2,401 for four,
            16,807 for five); ``"pruned"``, on that grid without its
            lowest-weight nodes, those that hold less than 1e-9 of its
            weight between them (2,321 nodes for four, 15,415 for five;
            none are left out for up to three); or ``"laplace"``, by
            Laplace's approximation, which has no grid: the integrand's
            height at each item's most likely parameters and its
            curvature there (``q^2 + q + 1`` evaluations for ``q``
            varying parameters: 21 for four, 43 for six). Default the
            full grid, the more accurate, up to five varying parameters,
            and Laplace's approximation beyond, where a grid is too
            large (it is refused there). Laplace's approximation is
            exact where the integrand is normal and close where each
            item has many readings; the fit warns where many items have
            few, and it does not take truncated items. The model keeps
            the choice as ``integration``; its predictions are on the
            same kind of grid, a Laplace fit's on the full grid up to
            five varying parameters and by Laplace's approximation
            beyond.
        e : array_like, optional
            The failure mode (event type, or cause) of each failure, any
            labels; ``None`` for an item still running. A SurpyvalData's
            event types (``ItemLifetime(..., event_type=...)``) are
            these. With two or more modes the model is of competing
            failure modes: each has a Weibull baseline, hazard
            coefficients and marker effects of its own
            (``"<mode>: alpha"``, ``"<mode>: beta"``, ``"<mode>:
            marker_effect"``, ...), so a reading may predict one way of
            failing and not another, and an item that fails by one mode
            is no longer at risk of the others; the marker part is one
            for all. The model's ``sf`` is the survival of every mode
            together. An item found failed (left or interval censored)
            needs its mode too. With one mode, or none given, the model
            is that of one kind of failure. With several markers each
            mode has an effect of each marker's link of its own
            (``"<mode>: vibration.marker_effect"``, ...), so one marker
            may warn of a mode and another marker of another.
            Proportional hazards and ``how="MLE"`` only: a ``JointAFT``
            fitter and the two-stage estimate refuse event types.
        shared : list of str, optional
            With failure modes: the survival parameters the modes have
            in common, one parameter each under its plain name:
            ``"beta"`` (the baselines' shape), a hazard covariate, or a
            link's effect (``"marker_effect"``, ``"slope_effect"``,
            ``"cumulative_effect"``; with several markers a marker's
            link's, ``"vibration.marker_effect"``, one marker's at a
            time). Default none: every mode has its own. The fit with
            an effect shared is nested in the fit without, so the
            likelihood-ratio test of the two
            (``model.likelihood_ratio_test(shared_model)``) asks whether
            the reading speaks to the modes differently.

        Returns
        -------
        JointModel
            The fitted model.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ItemLifetime, SurpyvalData, WeibullJointPH
        >>> rng = np.random.default_rng(3)
        >>> items = []
        >>> for k in range(60):
        ...     start, rate = rng.normal(2.0, 0.4), rng.normal(0.02, 0.005)
        ...     life = 100 * rng.weibull(2.0) * np.exp(-0.9 * (start - 2))
        ...     end = min(life, 120.0)
        ...     ages = np.arange(0.0, end, 10.0)
        ...     y = start + rate * ages + rng.normal(0, 0.2, ages.size)
        ...     outcome = "observed" if life < 120 else "right_censored"
        ...     items.append(
        ...         ItemLifetime(k, **{outcome: end},
        ...                      markers=dict(zip(ages, y)))
        ...     )
        >>> model = WeibullJointPH.fit(SurpyvalData.from_items(items))
        >>> model.maximum
        'verified'
        >>> model.integration
        'grid'
        >>> quick = WeibullJointPH.fit(
        ...     SurpyvalData.from_items(items), how="two-stage"
        ... )
        >>> quick.how, quick.maximum
        ('two-stage', 'not applicable')

        Two failure modes, the marker a warning of the first only (a
        wear-out mode the reading follows, and a random one it does
        not):

        >>> rng = np.random.default_rng(11)
        >>> items = []
        >>> for k in range(120):
        ...     start, rate = rng.normal(2.0, 0.4), rng.normal(0.02, 0.005)
        ...     worn = 110 * rng.weibull(2.5) * np.exp(-0.4 * (start - 2))
        ...     other = 250 * rng.weibull(1.0)
        ...     end = min(worn, other, 120.0)
        ...     ages = np.arange(0.0, end, 10.0)
        ...     y = start + rate * ages + rng.normal(0, 0.2, ages.size)
        ...     if end < 120:
        ...         outcome = {"observed": end}
        ...         mode = "bearing" if worn < other else "seal"
        ...     else:
        ...         outcome, mode = {"right_censored": end}, None
        ...     items.append(
        ...         ItemLifetime(k, **outcome, event_type=mode,
        ...                      markers=dict(zip(ages, y)))
        ...     )
        >>> modes = WeibullJointPH.fit(SurpyvalData.from_items(items))
        >>> modes.causes
        ['bearing', 'seal']
        >>> modes.parameter_names[-3:]
        ['seal: alpha', 'seal: beta', 'seal: marker_effect']
        """
        check_option(
            "how",
            how,
            HOWS,
            "'MLE' fits the joint likelihood; 'two-stage' is the quick, "
            "approximate estimate.",
        )
        integration = check_integration(integration, len(self.varying))
        if self.markers is not None:
            return self._fit_markers(
                x,
                c,
                n,
                t,
                Z,
                markers,
                tl,
                tr,
                marker_reference,
                path_covariates,
                hazard_covariates,
                i,
                marker_unit,
                how,
                integration,
                e,
                shared,
            )
        reference = None
        if marker_reference is not None:
            try:
                reference = float(marker_reference)
            except (TypeError, ValueError):
                reference = np.nan
            if np.ndim(marker_reference) or not np.isfinite(reference):
                raise ValueError(
                    "'marker_reference' is the marker value at which alpha "
                    "is reported, one finite number; got "
                    f"{marker_reference!r}."
                )
        if reference is not None and "value" not in self.link:
            raise ValueError(
                "'marker_reference' is the marker value at which alpha is "
                "reported, for a hazard that follows the marker's value; "
                f"this one follows its {' and '.join(self.link)} (alpha "
                "is at a slope and an area of 0): leave it out."
            )
        marker_unit = normalise_unit(marker_unit)
        markers = markers_by_row(markers, i, x)
        records, counts, names = _records(x, c, n, t, tl, tr, Z, markers)
        parts = covariate_parts(
            list(self.path_model.parameter_names),
            names,
            path_covariates,
            hazard_covariates,
        )
        causes = _causes(self, e, records, counts, how)
        model = _fit(
            self,
            records,
            counts,
            names,
            reference,
            parts,
            integration,
            how,
            causes,
            _check_shared(shared, causes, parts[1], self.link),
        )
        model.marker_unit = marker_unit
        return model

    def _fit_markers(
        self,
        x: Any,
        c: Any,
        n: Any,
        t: Any,
        Z: Any,
        markers: Any,
        tl: Any,
        tr: Any,
        marker_reference: Any,
        path_covariates: Any,
        hazard_covariates: Any,
        i: Any,
        marker_unit: Any,
        how: str,
        integration: str,
        e: Any,
        shared: Any,
    ) -> JointModel:
        """:meth:`fit` of several markers."""
        from ._multi import checked_readings, rows_by_marker

        names = list(self.markers or [])
        valued = [m for m, k in self.link if k == "value"]
        references = {}
        if marker_reference is not None:
            if not isinstance(marker_reference, Mapping):
                raise ValueError(
                    "With several markers 'marker_reference' is a dict "
                    "{marker: value}, the value of each marker whose value "
                    f"the hazard follows ({valued!r}) at which alpha is "
                    f"reported; got {marker_reference!r}."
                )
            for name, value in marker_reference.items():
                if name not in valued:
                    raise ValueError(
                        f"'marker_reference' names {name!r}; it takes the "
                        "markers whose value the hazard follows, "
                        f"{valued!r}."
                    )
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    number = np.nan
                if np.ndim(value) or not np.isfinite(number):
                    raise ValueError(
                        f"'marker_reference'[{name!r}] is one finite "
                        f"number; got {value!r}."
                    )
                references[name] = number
        units = marker_units(marker_unit, names, markers, x)
        rows = rows_by_marker(markers_by_row(markers, i, x, names), names)
        records, counts, covariates = _records(
            x, c, n, t, tl, tr, Z, [{}] * len(rows)
        )
        readings = [
            checked_readings(r, rows[int(r.id)], names) for r in records
        ]
        parts = covariate_parts(
            list(self.path_model.parameter_names),
            covariates,
            path_covariates,
            hazard_covariates,
        )
        causes = _causes(self, e, records, counts, how)
        model = _fit(
            self,
            records,
            counts,
            covariates,
            references,
            parts,
            integration,
            how,
            causes,
            _check_shared(shared, causes, parts[1], self.link),
            readings=readings,
        )
        model.marker_unit = units
        return model

    def compare_links(
        self,
        *args: Any,
        links: Any = None,
        metric: str = "aic",
        marker: str | None = None,
        **kwargs: Any,
    ) -> LinkComparison:
        """
        Fit the joint model with each of several links to the same data
        and compare them: does the risk follow the marker's level, how
        fast it is changing, the exposure accumulated so far, or more
        than one?

        Each candidate has this fitter's baseline, path model and varying
        path parameters (its own ``link`` is not used) and is fitted as
        :meth:`fit` fits, to the data given here, so the log-likelihoods
        and criteria compare directly. The result ranks the links by
        ``metric`` (lower is better), with each one's difference from the
        best and its weight (``table``), and tests each nested pair, such
        as ``"value"`` against ``"value + slope"``, by the likelihood
        ratio (``likelihood_ratio_tests()``): see
        :class:`~surpyval.univariate.joint.compare.LinkComparison`. Links
        less than 2 apart are no real preference, and the simpler of them
        is the best (fewer parameters, then the level before the slope
        before the area). The table says each row in words
        (``reads_as``) and ``verdict`` the whole comparison, such as
        "Risk follows how fast the marker is changing. Adding the level
        does not improve the fit. ..."; ``to_dict()`` gives both as
        strict JSON. Which link a marker's risk follows is a question for
        each dataset, and this is the way to ask it.

        The value link is fitted first and the others start from its
        estimates of the path, which saves about a third of the time; a
        fit from that start that does not reach a verified maximum is made
        again from its own start, so each candidate is the maximum its
        own :meth:`fit` finds.

        A fit that is not a verified maximum is shown but neither ranked
        nor tested, with one warning naming it; a link that cannot be
        fitted is listed in ``failed`` with the reason, with one warning.
        A warning about the data that every fit gives is given once.

        Parameters
        ----------
        *args, **kwargs
            The data and the options, as :meth:`fit` takes them: a
            ``SurpyvalData`` of items with markers, or arrays with
            ``markers=``; ``Z``, ``path_covariates``,
            ``hazard_covariates``, ``unit``, ... A ``marker_reference`` is
            given to the links that follow the value only (it moves their
            ``alpha`` and nothing else).
        links : list, optional
            The links to compare, two or more, each a link (``"value"``,
            ``"slope"`` or ``"cumulative"``) or a list of them, each set
            once. Default the five: ``"value"``, ``"slope"``,
            ``"cumulative"``, ``["value", "slope"]`` and ``["value",
            "cumulative"]``, less those the path cannot take, with a
            warning: a logarithmic path, whose slope is unbounded at age 0,
            leaves out the two with the slope. A Lloyd-Lipow path, which
            also has no finite area from age 0, can take only the value,
            and is refused.
        metric : str, optional
            The criterion the links are ranked by: ``"aic"`` (default),
            ``"aic_c"`` or ``"bic"``.
        marker : str, optional
            With several markers, the one whose links are compared
            (needed): each candidate gives that marker the link compared
            and every other marker the links of this fitter, so the
            comparison asks what the hazard follows of that marker with
            the others' parts as they are. ``models`` and the table are
            by that marker's link, and ``verdict`` names it. A weight
            on the marker's exposure goes to the candidates that follow
            its area.

        Returns
        -------
        LinkComparison
            The fitted candidates (``models``, by the link's name, such as
            ``"value + slope"``), the ranking (``table``), the best fit
            (``best``, ``best_link``), ``likelihood_ratio_tests()``, the
            ``verdict`` in words and ``to_dict()``.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ItemLifetime, SurpyvalData, WeibullJointPH
        >>> rng = np.random.default_rng(3)
        >>> items = []
        >>> for k in range(60):
        ...     start, rate = rng.normal(2.0, 0.4), rng.normal(0.02, 0.005)
        ...     life = 100 * rng.weibull(2.0) * np.exp(-0.9 * (start - 2))
        ...     end = min(life, 120.0)
        ...     ages = np.arange(0.0, end, 10.0)
        ...     y = start + rate * ages + rng.normal(0, 0.2, ages.size)
        ...     outcome = "observed" if life < 120 else "right_censored"
        ...     items.append(
        ...         ItemLifetime(k, **{outcome: end},
        ...                      markers=dict(zip(ages, y)))
        ...     )
        >>> data = SurpyvalData.from_items(items)
        >>> comparison = WeibullJointPH.compare_links(
        ...     data, links=["value", "slope", ["value", "slope"]]
        ... )
        >>> sorted(comparison.models)
        ['slope', 'value', 'value + slope']
        >>> list(comparison.table.columns[:4])
        ['k', 'log_likelihood', 'aic', 'aic_c']
        >>> comparison.best_label
        'value'
        >>> comparison.table.loc["slope", "reads_as"]
        'clearly worse'
        >>> comparison.verdict.split(". ")[1]
        'Adding the rate of change does not clearly improve the fit'
        >>> comparison.likelihood_ratio_tests()[["simpler", "richer", "df"]]
          simpler         richer  df
        0   value  value + slope   1
        1   slope  value + slope   1
        """
        if kwargs.get("how", "MLE") != "MLE":
            raise ValueError(
                "compare_links ranks the links by an information criterion "
                "and tests them by the likelihood ratio, which a two-stage "
                "fit has not (its two stages have no likelihood of the "
                "whole model): leave out how=, or compare two-stage fits "
                "by the accuracy of their predictions "
                "(surpyval.metrics.score_predictions)."
            )
        if self.markers is not None and marker not in self.markers:
            raise ValueError(
                "compare_links compares the links of one marker; this fitter "
                f"has several, {list(self.markers)!r}: name the one with "
                f"marker= (marker={self.markers[0]!r}); every other keeps "
                f"its links. Got marker={marker!r}."
            )
        if self.markers is None and marker is not None:
            raise ValueError(
                "'marker' names which of several markers' links to compare; "
                "this fitter has one marker: leave marker out."
            )
        return compare_links(
            self, args, kwargs, links=links, metric=metric, marker=marker
        )

    @takes_unit
    def fit_from_df(
        self,
        df: Any,
        x_col: str | None = None,
        markers_col: str | None = None,
        c_col: str | None = None,
        n_col: str | None = None,
        tl_col: str | None = None,
        tr_col: str | None = None,
        xl_col: str | None = None,
        xr_col: str | None = None,
        Z_cols: str | list[str] | None = None,
        marker_reference: float | None = None,
        path_covariates: Any = None,
        hazard_covariates: Any = None,
        i_col: str | None = None,
        m: Any = None,
        marker_unit: str | None = None,
        how: str = "MLE",
        integration: str | None = None,
        e_col: str | None = None,
        shared: Any = None,
    ) -> JointModel:
        """
        Fit the joint model to a DataFrame of items, one row each: the
        age it failed or was censored (``x_col``), its readings
        (``markers_col``, a column holding each item's ``{age: value}``),
        and optionally its censoring flag, count, truncation and
        covariates (``c_col``, ``n_col``, ``tl_col``, ``tr_col``,
        ``Z_cols``, which name the model's covariates). An item found
        failed between two ages has them in ``xl_col`` and ``xr_col`` (in
        place of ``x_col``; ``c_col`` 2 for those rows). As :meth:`fit`
        otherwise, ``path_covariates`` and ``hazard_covariates`` naming
        which of ``Z_cols`` enter which part.

        The readings may instead be kept in a long frame of their own, one
        row per reading: build a :class:`~surpyval.Measurements` from it
        (``Measurements.from_dataframe(readings, i_col=..., x_col=...,
        m_col=...)``) and give it as ``m``, with ``i_col`` naming the
        column of each item's unit in ``df``; the readings are matched to
        the items by unit. ``m`` and ``markers_col`` are two ways of
        giving the readings: give one. ``marker_unit`` is the unit the
        readings are in, as for :meth:`fit` (from ``m``'s own unit where
        it has one), and ``how`` the way to fit, as for :meth:`fit`
        (``"two-stage"`` for the quick, approximate estimate).
        ``integration`` is how each item's integral is done, ``"grid"``,
        ``"pruned"`` or ``"laplace"``, as for :meth:`fit`. ``e_col`` names
        the column of each failure's mode (missing for an item still
        running), for a model of competing failure modes, and ``shared``
        the survival parameters the modes have in common, as for
        :meth:`fit`.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from surpyval import WeibullJointPH
        >>> rng = np.random.default_rng(3)
        >>> rows = []
        >>> for k in range(60):
        ...     start, rate = rng.normal(2.0, 0.4), rng.normal(0.02, 0.005)
        ...     life = 100 * rng.weibull(2.0) * np.exp(-0.9 * (start - 2))
        ...     end = min(life, 120.0)
        ...     ages = np.arange(0.0, end, 10.0)
        ...     y = start + rate * ages + rng.normal(0, 0.2, ages.size)
        ...     rows.append({"age": end, "running": int(life >= 120),
        ...                  "site": k % 2, "readings": dict(zip(ages, y))})
        >>> df = pd.DataFrame(rows)
        >>> model = WeibullJointPH.fit_from_df(
        ...     df, x_col="age", markers_col="readings", c_col="running",
        ...     Z_cols=["site"],
        ... )
        >>> model.parameter_names[:4]
        ['a', 'a_site', 'b', 'b_site']
        """
        from surpyval.utils.dataframe import as_pandas as _as_pandas

        df = _as_pandas(df)
        from surpyval.univariate.regression._aliasing import (
            covariate_columns,
        )
        from surpyval.utils.dataframe import check_columns

        check_columns(
            df,
            x_col=x_col,
            i_col=i_col,
            markers_col=markers_col,
            c_col=c_col,
            n_col=n_col,
            tl_col=tl_col,
            tr_col=tr_col,
            xl_col=xl_col,
            xr_col=xr_col,
            Z_cols=Z_cols,
            e_col=e_col,
        )
        if m is not None and markers_col is not None:
            raise TypeError(
                "m and markers_col are two ways of giving the readings: "
                "give one."
            )
        if m is None and markers_col is None:
            raise ValueError(
                "markers_col names the column of each item's readings, "
                "{age: value} (or give them as m=, a Measurements matched "
                "to the items by i_col): the joint model needs them."
            )
        if x_col is not None and (xl_col is not None or xr_col is not None):
            raise ValueError(
                "Give x_col, or xl_col and xr_col (an interval each), not "
                "both."
            )
        if x_col is not None:
            x = df[x_col].to_numpy(dtype=float)
        elif xl_col is not None and xr_col is not None:
            x = df[[xl_col, xr_col]].to_numpy(dtype=float)
        else:
            raise ValueError(
                "x_col names the column of each item's age at failure or "
                "censoring (or xl_col and xr_col, an interval's ends)."
            )
        names = None
        Z = None
        if Z_cols is not None:
            names = [Z_cols] if isinstance(Z_cols, str) else list(Z_cols)
            Z = df[names].to_numpy(dtype=float)
        with covariate_columns(names, Z):
            return self.fit(
                x,
                c=None if c_col is None else df[c_col].to_numpy(),
                n=None if n_col is None else df[n_col].to_numpy(),
                tl=None if tl_col is None else df[tl_col].to_numpy(float),
                tr=None if tr_col is None else df[tr_col].to_numpy(float),
                Z=Z,
                markers=None if m is not None else list(df[markers_col]),
                marker_reference=marker_reference,
                path_covariates=path_covariates,
                hazard_covariates=hazard_covariates,
                i=None if i_col is None else list(df[i_col]),
                m=m,
                marker_unit=marker_unit,
                how=how,
                integration=integration,
                e=None if e_col is None else _labels(df[e_col]),
                shared=shared,
            )


def JointPH(
    distribution: Any,
    path_model: Any = None,
    varying: Any = None,
    link: Any = "value",
    lag: float = 0.0,
    weight: Any = None,
    random_effects: Any = None,
) -> JointPHFitter:
    """
    A joint model fitter of a marker and a lifetime: the marker follows
    the degradation ``path_model`` (``surpyval.degradation.LinearPath``
    by default; ``QuadraticPath``, ``ExponentialPath``, ... or a
    ``PathModel`` of one's own), and the hazard is proportional hazards
    on the baseline ``distribution`` (the Weibull for now, #891) in the
    path's current value, or in what ``link`` names. ``WeibullJointPH``
    is ``JointPH(Weibull, LinearPath)``.

    Parameters
    ----------
    distribution : ParametricFitter
        ``surpyval.Weibull``.
    path_model : PathModel or dict, optional
        The marker's path. Default ``LinearPath``. A dict ``{name: path
        model}`` is a model of several markers read on each item, each
        with its own path, varying parameters, noise and effects, their
        varying parameters correlated across the markers; ``varying``,
        ``link`` and ``random_effects`` may then be given by marker (a
        dict, or names ``"<marker>.<parameter>"``), ``random_effects``
        may be ``"by_marker"`` (correlated within each marker only), and
        the fit takes each row's readings as ``{name: {age: value}}`` (a
        ``Measurements`` of named measurements). Up to eight varying
        parameters over the markers: see "Several markers on one item"
        in the joint models' guide.
    varying : sequence of str, optional
        The path parameters that vary between items (their random
        effects, multivariate normal); the others are common to all
        items. Default all of them; for a ``SplinePath``, its ``level``
        and ``rate``. Up to five, each item's integral is on a grid by
        default; beyond five it is Laplace's approximation
        (``integration`` of :meth:`JointPHFitter.fit`).
    link : str or sequence of str, optional
        What of the marker's path the hazard follows: ``"value"`` (the
        default: its current value, so risk follows the level),
        ``"slope"`` (its rate of change: risk follows how fast it is
        changing) or ``"cumulative"`` (the area under it from age 0:
        risk follows the accumulated exposure), or a list of them, each
        with an effect of its own (``marker_effect``, ``slope_effect``,
        ``cumulative_effect``).
    lag : float or dict, optional
        How long after the marker the hazard follows it, in the
        lifetimes' unit of time (default 0: at once). With ``lag=200``
        the hazard at age ``t`` follows the path at ``t - 200``, so a
        reading today says what the risk will be 200 hours on. Before
        the lag has passed the hazard follows the path held at its age-0
        value: the value at age 0, a slope of 0 and an area of 0 (every
        link at ``max(t - lag, 0)``, the slope 0 before the lag). The lag
        is fixed, not estimated: the likelihood is not smooth in it. To
        choose one, fit each lag you have in mind and compare them by
        ``aic()`` (the same data, the same number of parameters). With
        several markers one number applies to every marker, and a dict
        ``{marker: lag}`` gives each its own (a marker left out at once,
        0).
    weight : tuple or dict, optional
        How much of the accumulated exposure (the ``"cumulative"`` link,
        which it needs) still counts, by how long ago it was. The
        default, ``None``, counts all of it equally: an hour of exposure
        ten years ago like one last week. ``("exponential", scale)``
        makes exposure fade: what was accumulated ``u`` ago counts
        ``exp(-u / scale)`` of what is accumulated now (damage that
        heals or is repaired), so the hazard follows
        :math:`\\int_0^t e^{-(t - s)/\\mathrm{scale}} m(s) ds`.
        ``("window", width)`` counts only the last ``width`` of it, the
        area under the path over ``(max(t - width, 0), t)``. The scale
        or width is in the lifetimes' unit of time and, like the lag, is
        fixed, not estimated: fit each one you have in mind and compare
        them by ``aic()``. With a ``lag`` the weighted exposure acts
        ``lag`` later (the weighted area at ``max(t - lag, 0)``). With
        several markers one weight applies to every marker with the
        ``"cumulative"`` link, and a dict ``{marker: weight}`` gives
        each its own (a marker left out unweighted).
        ``cumulative_effect`` is still per unit of the marker times a
        unit of age.
    random_effects : str or sequence, optional
        Which of the varying path parameters are correlated between
        items. ``"full"`` (the default, but for a spline's bends, below):
        every pair, a correlation each
        (``corr_a_b``, ...; three for three varying parameters, ten for
        five); ``"diagonal"``: none, each varying independently of the
        others; or blocks of names, such as ``[("a", "b"), ("c",)]``:
        those in a block are correlated with each other and with none
        outside it, every varying parameter in exactly one block. A
        correlation left out is 0, not estimated, and has no entry in
        ``params``, ``summary()`` or the saved model, so ``aic()`` counts
        the parameters estimated. With many varying parameters and a
        fleet too small to tell their correlations apart, fewer can be
        what makes the fit identifiable; nested structures compare by
        ``aic()`` or the likelihood ratio of the same data. For a
        ``SplinePath`` whose bends vary (``varying=("level", "rate",
        "s1")``), the default is a block of ``level`` and ``rate`` and
        each bend on its own: a bend's spread is told from few items'
        readings past its knot, too few to estimate its correlations
        with the rest as well.

    Returns
    -------
    JointPHFitter
        A fitter with a ``fit(data)`` method.

    Examples
    --------
    >>> from surpyval import JointPH, Weibull
    >>> from surpyval.degradation import QuadraticPath
    >>> JointPH(Weibull)  # doctest: +NORMALIZE_WHITESPACE
    WeibullJointPH: joint model fitter (Weibull baseline, Linear path,
    a + b t)
    >>> JointPH(Weibull, QuadraticPath, varying=("a", "b"))
    ... # doctest: +ELLIPSIS
    WeibullQuadraticJointPH: joint model fitter (..., varying: a, b)
    >>> JointPH(Weibull, link=["value", "slope"])
    ... # doctest: +ELLIPSIS
    WeibullJointPH: joint model fitter (..., link: value, slope)
    >>> JointPH(Weibull, lag=200.0)  # doctest: +ELLIPSIS
    WeibullJointPH: joint model fitter (..., lag: 200)
    >>> JointPH(Weibull, QuadraticPath, random_effects=[("a", "b"), "c"])
    ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    WeibullQuadraticJointPH: joint model fitter (..., random effects:
    correlated in blocks {a, b}, {c})
    >>> JointPH(Weibull, link="cumulative", weight=("exponential", 0.5))
    ... # doctest: +ELLIPSIS
    WeibullJointPH: joint model fitter (..., weight: exponential 0.5)

    Two markers, the hazard following the value of each:

    >>> from surpyval.degradation import LinearPath
    >>> two = JointPH(
    ...     Weibull, {"vibration": LinearPath, "temperature": LinearPath}
    ... )
    >>> two.varying
    ('vibration.a', 'vibration.b', 'temperature.a', 'temperature.b')
    >>> two.link
    (('vibration', 'value'), ('temperature', 'value'))
    """
    if path_model is None:
        from surpyval.degradation.path_models import LinearPath

        path_model = LinearPath
    return JointPHFitter(
        distribution,
        path_model,
        varying,
        link,
        lag,
        weight,
        random_effects=random_effects,
    )


# -- the data ----------------------------------------------------------------


def _records(
    x: Any,
    c: Any,
    n: Any,
    t: Any,
    tl: Any,
    tr: Any,
    Z: Any,
    markers: Any,
) -> tuple[list, npt.NDArray, list[str]]:
    """The items as ItemLifetime records (which check each item's
    readings against its observation), their counts, and the covariates'
    names; what the fit cannot take is refused."""
    from surpyval.utils.item_lifetime import ItemLifetime

    x = np.asarray(x, dtype=float)
    if x.ndim not in (1, 2) or (x.ndim == 2 and x.shape[1] != 2):
        raise ValueError(
            "'x' is each item's age at failure or censoring, or a [lower, "
            "upper] row each (interval censored items); got shape "
            f"{x.shape}."
        )
    rows = len(x)
    if c is None:
        # [lower, upper] rows: an interval where they differ
        c = (
            np.where(x[:, 0] == x[:, 1], 0, 2)
            if x.ndim == 2
            else np.zeros(rows, dtype=int)
        )
    c = np.asarray(c)
    if c.shape != (rows,):
        raise ValueError(f"'c' must hold one flag per item ({rows}).")
    unknown = set(np.unique(c).tolist()) - {0, 1, -1, 2}
    if unknown:
        raise ValueError(
            "'c' must be 0 (failed), 1 (still running), -1 (failed by "
            f"x) or 2 (failed in x's interval); got {sorted(unknown)}."
        )
    if x.ndim == 1 and np.any(c == 2):
        raise ValueError(
            "An interval censored item (c = 2) needs its [lower, upper]: "
            "give x as one [lower, upper] row per item (the same age twice "
            "for the others)."
        )
    if markers is None:
        raise ValueError(
            "The joint model needs the items' marker readings: fit a "
            "SurpyvalData of ItemLifetime records with markers "
            "(SurpyvalData.from_items([...])), or give markers=[{age: "
            "value, ...}, ...], one per item."
        )
    markers = list(markers)
    if len(markers) != rows:
        raise ValueError(
            f"'markers' must hold one {{age: value}} per item ({rows}); "
            f"got {len(markers)}."
        )
    n = np.ones(rows) if n is None else np.asarray(n, dtype=float)
    if n.shape != (rows,) or not (np.isfinite(n).all() and (n > 0).all()):
        raise ValueError(f"'n' must hold one count above 0 per item ({rows}).")
    if t is not None:
        t = np.asarray(t, dtype=float)
        if t.shape != (rows, 2):
            raise ValueError(
                f"'t' must be the [left, right] truncation of each item, "
                f"shape ({rows}, 2)."
            )
        tl, tr = t[:, 0], t[:, 1]
    tl = (
        np.full(rows, -np.inf)
        if tl is None
        else np.broadcast_to(np.asarray(tl, dtype=float), (rows,))
    )
    tr = (
        np.full(rows, np.inf)
        if tr is None
        else np.broadcast_to(np.asarray(tr, dtype=float), (rows,))
    )
    names = fit_columns()
    if Z is not None:
        Z = np.asarray(Z, dtype=float)
        Z = Z[:, None] if Z.ndim == 1 else Z
        if Z.ndim != 2 or len(Z) != rows:
            raise ValueError(
                f"'Z' must hold one row of covariates per item ({rows})."
            )
        # an item with a missing covariate is left out, with one warning
        missing = ~finite_covariate_mask(Z)
        if names is None:
            names = [f"coef_{j}" for j in range(Z.shape[1])]
    else:
        missing = np.zeros(rows, dtype=bool)
        names = []
    names = [str(name) for name in names]
    outcome = {
        0: "observed",
        1: "right_censored",
        -1: "left_censored",
        2: "interval_censored",
    }
    records, counts = [], []
    for k in range(rows):
        if missing[k]:
            continue
        flag = int(c[k])
        if flag == 2:
            value: Any = [float(x[k, 0]), float(x[k, 1])]
        elif x.ndim == 1:
            value = float(x[k])
        else:
            value = float(x[k, 1] if flag == -1 else x[k, 0])
        records.append(
            ItemLifetime(
                k,
                **{outcome[flag]: value},
                left_truncation=(float(tl[k]) if np.isfinite(tl[k]) else None),
                right_truncation=(
                    float(tr[k]) if np.isfinite(tr[k]) else None
                ),
                covariates=None if Z is None else Z[k],
                markers=markers[k] or None,
            )
        )
        counts.append(n[k])
    if not records:
        raise ValueError("No items are left to fit.")
    return records, np.array(counts, dtype=float), names


def marker_units(given: Any, names: list, markers: Any, x: Any) -> Any:
    """The unit each of several markers is read in, ``{marker: unit}``
    (``None`` where no marker has one): ``given`` (a dict) and the
    readings' own units (a Measurements, given or held by the data),
    which must agree."""
    from surpyval.utils.measurements import Measurements
    from surpyval.utils.surpyval_data import SurpyvalData
    from surpyval.utils.units import same_unit

    if isinstance(given, str) and len(names) == 1:
        given = {names[0]: given}
    if given is not None and not isinstance(given, Mapping):
        raise ValueError(
            "With several markers 'marker_unit' is a dict {marker: unit}, "
            f"the unit each is read in; got {given!r}."
        )
    out: dict = {name: None for name in names}
    for name, unit in (given or {}).items():
        if name not in names:
            raise ValueError(
                f"'marker_unit' names {name!r}, which is not a marker; the "
                f"markers are {names!r}."
            )
        out[name] = normalise_unit(unit)
    held = None
    if isinstance(markers, Measurements):
        held = markers
    elif isinstance(x, SurpyvalData) and x.m is not None:
        held = x.m
    if held is not None:
        for name, unit in held.units.items():
            if name not in out or unit is None:
                continue
            if out[name] is None:
                out[name] = unit
            elif not same_unit(out[name], unit):
                raise ValueError(
                    f"Marker {name!r}'s readings are in {unit!r}, but "
                    f"marker_unit gives {out[name]!r}: a model is in the "
                    "unit of its readings (no conversion is made)."
                )
    return out if any(v is not None for v in out.values()) else None


def _labels(e: Any) -> npt.NDArray:
    """``e`` as an object array of labels, a missing one (``None``,
    ``nan``, pandas' ``NA``) as ``None``; a tuple label kept whole."""
    values = list(e)
    out = np.empty(len(values), dtype=object)
    for k, value in enumerate(values):
        missing = value is None
        if not missing and not isinstance(value, (str, tuple)):
            try:
                missing = bool(value != value)
            except (TypeError, ValueError):
                missing = True
        out[k] = None if missing else value
    return out


def _causes(
    fitter: Any, e: Any, records: list, counts: npt.NDArray, how: str
) -> tuple | None:
    """``(causes, index)``: the failure modes of ``e``, in order, and
    each record's (its index among them, -1 for an item still running);
    ``None`` where no event types are given, or one mode only (the model
    of one kind of failure). What the fit of several modes cannot take
    is refused: a failure with no mode, a mode on an item still running,
    the accelerated survival part and the two-stage estimate."""
    from surpyval.univariate.competing_risks.labels import ordered_labels

    if e is None:
        return None
    if np.ndim(e) != 1:
        raise ValueError(
            "'e' is the failure mode (event type) of each item's failure, "
            "one label per item (None for an item still running); got "
            f"{np.ndim(e)} dimension(s)."
        )
    labels = _labels(e)
    rows = max(int(r.id) for r in records) + 1
    if len(labels) < rows:
        raise ValueError(
            f"'e' must hold one failure mode per item; got {len(labels)}."
        )
    mine = [labels[int(r.id)] for r in records]
    causes = ordered_labels(mine)
    if len(causes) < 2:
        return None
    running = [r.outcome == "right_censored" for r in records]
    labelled = sum(
        w
        for w, label, r in zip(counts, mine, running)
        if r and label is not None
    )
    if labelled:
        raise ValueError(
            f"{labelled:g} item(s) still running (right censored) have a "
            "failure mode in 'e'. An item that has not failed has none: "
            "set their 'e' to None."
        )
    unknown = sum(
        w
        for w, label, r in zip(counts, mine, running)
        if not r and label is None
    )
    if unknown:
        raise ValueError(
            f"{unknown:g} failure(s) have no failure mode in 'e' "
            f"(the others are {causes!r}). A failure of unknown mode is "
            "not fitted yet: give each failure's mode, or leave 'e' out "
            "to fit every failure as one kind."
        )
    if fitter.survival == "AFT":
        raise ValueError(
            "An accelerated joint model (JointAFT) takes no failure modes "
            f"('e' has {causes!r}) for now: fit them with the proportional "
            "hazards one, JointPH (WeibullJointPH), or leave 'e' out to "
            "fit every failure as one kind."
        )
    if how == "two-stage":
        raise ValueError(
            "The two-stage estimate takes no failure modes ('e' has "
            f"{causes!r}) for now: fit them with how='MLE', or leave 'e' "
            "out to fit every failure as one kind."
        )
    index = np.array(
        [-1 if label is None else causes.index(label) for label in mine],
        dtype=int,
    )
    return causes, index


def _check_shared(
    shared: Any, causes: tuple | None, hazard: list, links: tuple
) -> tuple:
    """``shared`` as a tuple of names, refused unless each is a survival
    parameter the failure modes can have in common."""
    if shared is None:
        return ()
    names = [shared] if isinstance(shared, str) else list(shared)
    if not names:
        return ()
    if causes is None:
        raise ValueError(
            "'shared' names the survival parameters that failure modes "
            "have in common, and this fit has fewer than two modes (no "
            "'e', or one mode in it): leave 'shared' out."
        )
    allowed = shareable(list(hazard), links)
    if "alpha" in names:
        raise ValueError(
            "'shared' cannot hold 'alpha': each failure mode has its own "
            "scale (two modes with one alpha and one beta would be "
            f"equally likely at every age). It can hold {allowed!r}."
        )
    wrong = [name for name in names if name not in allowed]
    twice = sorted({name for name in names if names.count(name) > 1})
    if wrong or twice:
        raise ValueError(
            "'shared' names the survival parameters the failure modes "
            f"have in common, each once, from {allowed!r}; got "
            f"{wrong + twice!r}."
        )
    return tuple(name for name in allowed if name in names)


def _check_cause_counts(
    causes: list, index: npt.NDArray, counts: npt.NDArray, own: list
) -> dict:
    """The failures of each mode, ``{mode: count}``; refused where a
    mode has fewer failures than parameters of its own (``own``)."""
    found = {
        cause: float(np.sum(counts[index == k]))
        for k, cause in enumerate(causes)
    }
    short = {cause: n for cause, n in found.items() if n < len(own)}
    if short:
        listed = ", ".join(f"{c!r}: {n:g}" for c, n in short.items())
        raise ValueError(
            "Each failure mode needs at least as many failures as it has "
            f"parameters of its own ({len(own)}: {', '.join(own)}); the "
            f"data have {listed} (every mode: "
            + ", ".join(f"{c!r}: {n:g}" for c, n in found.items())
            + "). Merge a rare mode into another in 'e', share parameters "
            "between the modes (shared=), or leave 'e' out to fit every "
            "failure as one kind."
        )
    return found


def _lifetimes(records: list) -> tuple:
    """``(T, d, U, L, Rt)`` of the records: the age each left observation
    (the lower end of an interval: a left censored item's entry age, else
    0), the event flag, the upper end of an interval (``inf`` for
    none), the entry age (0 for none) and the right truncation (``inf``
    for none)."""
    T, d, U, L, Rt = [], [], [], [], []
    for r in records:
        entry = 0.0 if r.left_truncation is None else r.left_truncation
        if r.outcome == "interval_censored":
            T.append(float(r.value[0]))
            U.append(float(r.value[1]))
        elif r.outcome == "left_censored":
            # failed by its age, having reached its entry
            T.append(entry)
            U.append(float(r.value))
        else:
            T.append(float(r.value))
            U.append(np.inf)
        d.append(r.outcome == "observed")
        L.append(entry)
        Rt.append(np.inf if r.right_truncation is None else r.right_truncation)
    return (
        np.array(T, dtype=float),
        np.array(d, dtype=float),
        np.array(U, dtype=float),
        np.array(L, dtype=float),
        np.array(Rt, dtype=float),
    )


def _arrays(
    fitter: JointPHFitter, records: list, counts: Any, p: int
) -> tuple:
    """The items' arrays (covariates as given), their readings, and the
    checks of what the model needs."""
    T, d, U, L, Rt = _lifetimes(records)
    point = ~np.isfinite(U)
    if np.any(L < 0) or np.any(T[point] <= 0) or np.any(U[~point] <= 0):
        raise ValueError(
            "The joint model's ages start at 0 (a Weibull baseline): every "
            "item fails or is censored after 0 and comes under observation "
            "at 0 or later (left_truncation)."
        )
    Z = (
        np.array([r.covariates for r in records], dtype=float).reshape(-1, p)
        if p
        else np.zeros((len(records), 0))
    )
    ages = [
        np.array(list((r.markers or {}).keys()), dtype=float) for r in records
    ]
    values = [
        np.array(list((r.markers or {}).values()), dtype=float)
        for r in records
    ]
    for r, a, v in zip(records, ages, values):
        if len(a):
            fitter.path_model.check_data(a, v)
    if not (d.any() or np.isfinite(U).any()):
        raise ValueError(
            "The joint model needs at least one failure (observed, or "
            "found failed: left or interval censored): every item is still "
            "running."
        )
    P = len(fitter.path_model.parameter_names)
    q = len(fitter.varying)
    fitted = sum(1 for a in ages if len(np.unique(a)) >= P)
    if fitted < q + 2:
        raise ValueError(
            f"The joint model needs marker readings at {P} or more ages "
            f"(the {fitter.path_model.name} path's parameters) on at least "
            f"{q + 2} items, to see how their paths differ; {fitted} "
            "item(s) have them."
        )
    return T, d, L, Z, ages, values, U, Rt


def _marker_arrays(
    fitter: JointPHFitter,
    records: list,
    readings: list,
    counts: Any,
    p: int,
) -> tuple:
    """``(T, d, L, Z, U, Rt)`` of the records, and the checks of what a
    model of several markers needs: each marker's readings inside its
    path's domain, and read at as many ages as its path has parameters
    on enough items to see how they differ."""
    T, d, U, L, Rt = _lifetimes(records)
    point = ~np.isfinite(U)
    if np.any(L < 0) or np.any(T[point] <= 0) or np.any(U[~point] <= 0):
        raise ValueError(
            "The joint model's ages start at 0 (a Weibull baseline): every "
            "item fails or is censored after 0 and comes under observation "
            "at 0 or later (left_truncation)."
        )
    Z = (
        np.array([r.covariates for r in records], dtype=float).reshape(-1, p)
        if p
        else np.zeros((len(records), 0))
    )
    if not (d.any() or np.isfinite(U).any()):
        raise ValueError(
            "The joint model needs at least one failure (observed, or "
            "found failed: left or interval censored): every item is still "
            "running."
        )
    for k, (name, model) in enumerate(
        zip(fitter.markers or [], fitter.path_model.models)
    ):
        ages = [item[k][0] for item in readings]
        for item in readings:
            a, v = item[k]
            if len(a):
                try:
                    model.check_data(a, v)
                except ValueError as error:
                    raise ValueError(f"Marker {name!r}: {error}") from None
        P = len(model.parameter_names)
        q = sum(1 for v in fitter.varying if v.split(".", 1)[0] == name)
        fitted = sum(1 for a in ages if len(np.unique(a)) >= P)
        if fitted < q + 2:
            raise ValueError(
                f"The joint model needs readings of marker {name!r} at {P} "
                f"or more ages (its {model.name} path's parameters) on at "
                f"least {q + 2} items, to see how their paths differ; "
                f"{fitted} item(s) have them."
            )
    return T, d, L, Z, U, Rt


def _marker_refs(
    markers: list, links: tuple, centres: list, zbar: Any, reference: dict
) -> tuple:
    """``refs`` (:func:`~.joint_model.reported_params`) of several
    markers: the centre and the reference of each link's term (a value's
    its marker's, 0 for a slope or an area)."""
    c = np.array(
        [centres[markers.index(m)] if k == "value" else 0.0 for m, k in links]
    )
    r = np.array(
        [reference[m] if k == "value" else 0.0 for m, k in links],
        dtype=float,
    )
    return (c, zbar, r)


def _with_placed_paths(fitter: JointPHFitter, readings: list) -> Any:
    """:func:`_with_placed_path` for several markers: each marker's path
    of joined pieces placed from its own readings' ages."""
    from ._multi import MarkerSet

    models = []
    changed = False
    for k, model in enumerate(fitter.path_model.models):
        base = getattr(model, "base", model)
        if not hasattr(base, "placed"):
            models.append(model)
            continue
        ages = [list(item[k][0]) for item in readings]
        pooled = np.array([a for item in ages for a in item], dtype=float)
        placed = base.placed(pooled)
        placed.check_readings([len(item) for item in ages])
        name = fitter.path_model.names[k]
        own = [
            v.split(".", 1)[1]
            for v in fitter.varying
            if v.split(".", 1)[0] == name
        ]
        placed.check_varying(ages, tuple(own))
        if placed is not base:
            changed = True
            model = (
                placed
                if base is model
                else type(model)(placed, dict(model.links))
            )
        models.append(model)
    if not changed:
        return fitter
    out = copy.copy(fitter)
    out.path_model = MarkerSet(fitter.path_model.names, models)
    return out


def _check_markers(
    fitter: JointPHFitter, mu: npt.NDArray, found_by: str
) -> None:
    """:func:`_check_power` and :func:`_check_start_at_zero` of the path,
    or of each marker's."""
    if fitter.markers is None:
        _check_power(fitter, mu, found_by)
        _check_start_at_zero(fitter, mu)
        return
    mu = np.asarray(mu, dtype=float)
    for name, model, piece in zip(
        fitter.markers, fitter.path_model.models, fitter.path_model.slices
    ):
        one = copy.copy(fitter)
        one.markers = None
        one.path_model = model
        one.link = tuple(k for m, k in fitter.link if m == name)
        from ._multi import lag_of, weight_of

        one.lag = lag_of(fitter.lag, name)
        one.weight = weight_of(fitter.weight, name)
        _check_power(one, mu[piece], found_by)
        _check_start_at_zero(one, mu[piece])


def _aliased(Z: npt.NDArray, w: npt.NDArray) -> npt.NDArray:
    """The covariate columns the data cannot determine: constant (the
    path means and the Weibull scale are intercepts), or combinations of
    others."""
    if Z.shape[1] == 0:
        return np.zeros(0, dtype=int)
    constant = constant_columns(Z)
    zc = Z - np.average(Z, axis=0, weights=w)
    gram = (zc * w[:, None]).T @ zc
    return aliased_columns(gram, len(Z), constant)


def _with_placed_path(fitter: JointPHFitter, records: list) -> JointPHFitter:
    """``fitter``, or a copy of it whose path has its knots placed from
    the records' reading ages: for a path made of joined pieces (a
    ``SplinePath``), which also refuses more knots than the items'
    readings support, and varying bends whose spread they cannot
    tell."""
    path: Any = fitter.path_model
    base = getattr(path, "base", path)
    if not hasattr(base, "placed"):
        return fitter
    ages = [list((r.markers or {}).keys()) for r in records]
    pooled = np.array([a for item in ages for a in item], dtype=float)
    placed = base.placed(pooled)
    placed.check_readings([len(item) for item in ages])
    placed.check_varying(ages, fitter.varying)
    if placed is base:
        return fitter
    out = copy.copy(fitter)
    if base is path:
        out.path_model = placed
    else:
        out.path_model = type(path)(placed, dict(path.links))
    return out


def _fit(
    fitter: JointPHFitter,
    records: list,
    counts: npt.NDArray,
    names: list,
    reference: Any,
    parts: tuple,
    integration: str = "grid",
    how: str = "MLE",
    causes: tuple | None = None,
    shared: tuple = (),
    readings: Any = None,
) -> JointModel:
    """The joint fit of ``records`` (``counts`` of each), by ``how``
    (:data:`HOWS`); ``parts`` is ``(shifts, hazard)``, which covariates
    enter which part
    (:func:`~surpyval.univariate.joint.joint_model.covariate_parts`);
    ``causes`` is :func:`_causes`' (the failure modes and each record's)
    and ``shared`` the survival parameters they have in common. With
    several markers, ``readings`` holds each record's readings by marker
    (``_multi.checked_readings``) and ``reference`` is ``{marker:
    value}``."""
    several = fitter.markers is not None
    markers: list = list(fitter.markers or [])
    if several:
        fitter = _with_placed_paths(fitter, readings)
    else:
        fitter = _with_placed_path(fitter, records)
    path_model = fitter.path_model
    path_names = list(path_model.parameter_names)
    shifts, hazard = parts
    p_all = len(names)
    cause_names = None if causes is None else list(causes[0])
    every = parameter_names(
        path_names,
        list(fitter.varying),
        names,
        shifts,
        hazard,
        fitter.link,
        fitter.random_effects,
        cause_names,
        shared,
        fitter.markers,
    )
    clash = sorted({a for a in names if every.count(a) > 1})
    clash += sorted(
        {a for a in names for b in path_names if every.count(f"{b}_{a}") > 1}
        - set(clash)
    )
    if clash:
        raise ValueError(
            f"Covariates named {clash!r} would share a parameter's name in "
            "the joint model; rename them."
        )
    channels = None
    if several:
        from ._multi import joined, marker_centres

        T, d, L, Z_all, U, Rt = _marker_arrays(
            fitter, records, readings, counts, p_all
        )
        each = [joined(item) for item in readings]
        ages = [a for a, _, _ in each]
        values = [v for _, v, _ in each]
        channels = [ch for _, _, ch in each]
    else:
        T, d, L, Z_all, ages, values, U, Rt = _arrays(
            fitter, records, counts, p_all
        )
    aliased = _aliased(Z_all, counts)
    if aliased.size:
        warn_aliased(
            aliased,
            "constant, or a combination of the others: the path means and "
            "the Weibull scale already hold a constant",
        )
    kept = np.setdiff1d(np.arange(p_all), aliased)
    Z = Z_all[:, kept]
    p = Z.shape[1]
    read_w = np.concatenate(
        [np.full(len(v), w) for v, w in zip(values, counts)]
    )
    zbar = np.average(Z, axis=0, weights=counts) if p else np.zeros(0)
    if several:
        centres = marker_centres(readings, counts, len(markers))
        given: dict = dict(reference or {})
        reference = {
            name: given.get(name, centres[k])
            for k, name in enumerate(markers)
            if (name, "value") in fitter.link
        }
        refs = _marker_refs(markers, fitter.link, centres, zbar, reference)
        centre: Any = 0.0
    else:
        centre = float(np.average(np.concatenate(values), weights=read_w))
        reference = centre if reference is None else reference
        refs = (centre, zbar, reference)
    arrays = JointArrays.build(
        counts,
        T,
        d,
        L,
        Z - zbar,
        ages,
        values,
        U=U,
        Rt=Rt,
        e=None if causes is None else causes[1],
        channels=channels,
        n_markers=len(markers) if several else 1,
    )
    if integration == "laplace" and how == "MLE":
        _check_laplace(arrays, len(fitter.varying))
    kept_names = [names[j] for j in kept]
    by_cause = None
    cause_layout: dict = {}
    if causes is not None and cause_names is not None:
        kept_hazard = [cov for cov in kept_names if cov in hazard]
        own = [
            name
            for name in ["alpha", *shareable(kept_hazard, fitter.link)]
            if name not in shared
        ]
        by_cause = _check_cause_counts(cause_names, causes[1], counts, own)
        cause_layout = {
            "n_causes": len(cause_names),
            "shared_beta": "beta" in shared,
            "shared_hazard": [cov in shared for cov in kept_names],
            "shared_links": [effect_name(k) in shared for k in fitter.link],
        }
    layout = Layout(
        len(path_names),
        p,
        tuple(path_names.index(v) for v in fitter.varying),
        np.array(
            [[cov in shifts[n] for cov in kept_names] for n in path_names],
            dtype=bool,
        ).reshape(len(path_names), p),
        np.array([cov in hazard for cov in kept_names], dtype=bool),
        fitter.link,
        fitter.survival,
        block_labels(fitter.random_effects, fitter.varying),
        **cause_layout,
        n_markers=len(markers) if several else 1,
    )
    path: Any
    if several:
        path = MarkerPaths(
            path_model, fitter.link, fitter.lag, fitter.weight, centres
        )
    else:
        path = PathFunction(path_model, fitter.link, fitter.lag, fitter.weight)
    model = Model(layout, path, centre, integration)
    # Starting values: each item's path alone, their spread (or, where
    # given, another link's fit of the same data: compare_links), then
    # the survival part on each item's predicted path
    theta = _given_path_start(fitter._path_start, layout)
    given_start = theta is not None
    if theta is None and several:
        from ._multi import marker_part_start

        theta = marker_part_start(model, arrays, readings, Z - zbar, counts)
    if theta is None and layout.q > MAX_VARYING:
        # many parameters an item: the items' own fits are too wild for
        # their mean and covariance to start from
        theta = settled_marker_start(model, arrays, ages, values)
        given_start = True
    if theta is None:
        theta = marker_start(model, ages, values, Z - zbar, counts)
    _check_markers(fitter, theta["mu"], "the readings put")
    total = float(np.sum(counts))
    fit_model, path_u = model, None
    if how == "two-stage":
        u, value, cov_u, maximum = _two_stage(model, arrays, theta)
    else:
        with warnings.catch_warnings(record=True) as first_warnings:
            warnings.simplefilter("always")
            try:
                u, value, cov_u, maximum = _maximise(
                    model, arrays, survival_start(model, arrays, theta)
                )
            except _NoStart:
                if given_start:
                    raise
                # the likelihood is not finite at the start from the
                # items' own fits: one more, from the settled start
                if several:
                    from ._multi import marker_part_start

                    theta = marker_part_start(
                        model,
                        arrays,
                        readings,
                        Z - zbar,
                        counts,
                        settled=True,
                    )
                else:
                    theta = settled_marker_start(model, arrays, ages, values)
                u, value, cov_u, maximum = _maximise(
                    model, arrays, survival_start(model, arrays, theta)
                )
        kept_warnings = list(first_warnings)
        # A spread between items whose sd is (nearly) 0 sits where the
        # search's log-Cholesky scale puts it at minus infinity: the
        # likelihood is flat along it there, the search can stall short
        # of the fit with that parameter common to all items (#1334), and
        # the spread has no Wald standard error. Such a fit is held on
        # that boundary, or kept off it where the likelihood rises off it
        # (:func:`_on_the_boundary`). A verified fit with every spread
        # clear of 0 is left as it is.
        held_fit = _at_a_spread_of_zero(model, arrays, u, value, maximum)
        if held_fit is not None:
            fit_model, (u, value, cov_u, maximum), kept_warnings, path_u = (
                held_fit
            )
        for caught in kept_warnings:
            warnings.warn_explicit(
                caught.message,
                caught.category,
                caught.filename,
                caught.lineno,
            )
        _check_markers(
            fitter, fit_model.layout.unpack(u)["mu"], "the fit puts"
        )
    at = fit_model.layout
    estimated = np.asarray(
        reported_params(at.unpack(u), at, refs), dtype=float
    )
    covariance = None
    if cov_u is not None:
        J = jacobian(lambda v: reported_params(at.unpack(v), at, refs))(u)
        covariance = J @ cov_u @ J.T
        covariance = 0.5 * (covariance + covariance.T)
        if how == "two-stage":
            # stage one's moment estimates have no standard errors
            n_marker = len(estimated) - (2 + layout.n_hazard + layout.n_links)
            covariance[:n_marker, :] = np.nan
            covariance[:, :n_marker] = np.nan
    else:
        warn_no_covariance()
    kept_covariates = list(kept_names)
    # every parameter, nan for an aliased covariate's
    full_names = parameter_names(
        path_names,
        list(fitter.varying),
        names,
        shifts,
        hazard,
        fitter.link,
        fitter.random_effects,
        cause_names,
        shared,
        fitter.markers,
    )
    kept_names = parameter_names(
        path_names,
        list(fitter.varying),
        kept_names,
        {n: [c for c in shifts[n] if c in kept_names] for n in path_names},
        [c for c in hazard if c in kept_names],
        fitter.link,
        fitter.random_effects,
        cause_names,
        shared,
        fitter.markers,
    )
    if fit_model is not model:
        # held on the boundary: the fit with the spreads at 0 common to
        # all items, its parameters in their places and those spreads
        # (and their correlations) 0, with no standard errors
        held = [path_names[j] for j in fit_model.layout.varying]
        fit_names = parameter_names(
            path_names,
            held,
            kept_covariates,
            {
                n: [c for c in shifts[n] if c in kept_covariates]
                for n in path_names
            },
            [c for c in hazard if c in kept_covariates],
            fitter.link,
            fitter.random_effects,
            cause_names,
            shared,
            fitter.markers,
        )
        estimated, covariance = _held_on_the_boundary(
            kept_names, fit_names, estimated, covariance
        )
    where = [full_names.index(name) for name in kept_names]
    params = np.full(len(full_names), np.nan)
    params[where] = estimated
    out = JointModel()
    out.dist = fitter.dist
    out.path_model = path_model
    out.varying = list(fitter.varying)
    out.random_effects_structure = fitter.random_effects
    out.params = params
    out.feature_names = list(names)
    out.aliased = aliased
    every_one = all(shifts[n] == list(names) for n in path_names) and list(
        hazard
    ) == list(names)
    # saved as None where every covariate is in both parts (as before)
    out.path_covariates = (
        None if every_one else {n: list(shifts[n]) for n in path_names}
    )
    out.hazard_covariates = None if every_one else list(hazard)
    out.parameter_names = full_names
    out.markers = None if not several else list(markers)
    out.link = tuple(fitter.link)
    out.lag = fitter.lag
    out.weight = fitter.weight
    out.survival = fitter.survival
    out.causes = cause_names
    out.shared = tuple(shared)
    out.n_events_by_cause = by_cause
    if several:
        out.marker_reference = dict(reference) or None
        out._centre = list(centres)
    else:
        out.marker_reference = (
            float(reference) if "value" in fitter.link else None
        )
        out._centre = centre
    out._zbar = zbar
    if covariance is not None:
        full_cov = np.full((len(full_names),) * 2, np.nan)
        full_cov[np.ix_(where, where)] = covariance
        out._covariance = full_cov
    out._neg_ll = float(value)
    # the path part where it was found, a start for another link's fit
    # (on the boundary, the search off it: the spreads at 0 small)
    found = layout.unpack(u if path_u is None else path_u)
    out._path_start = {
        key: np.array(found[key], dtype=float)
        for key in ("mu", "Delta", "L", "sigma")
    }
    out.k = len(estimated)
    out.maximum = maximum
    out.how = how
    out.integration = integration
    out.n_items = total
    out.n_events = float(np.sum(np.maximum(d, np.isfinite(U)) * counts))
    out.n_readings = float(np.sum([len(a) * w for a, w in zip(ages, counts)]))
    return out


#: A varying parameter whose spread between items moves no item's marker
#: by this fraction of the readings' noise sd, at any reading age, is
#: taken as at its lower bound, 0 (:func:`_spreads_at_zero`): so small a
#: spread is invisible in the readings.
_NO_SPREAD = 0.01

#: The spread of a parameter at 0 when the search starts off the
#: boundary (:func:`_on_the_boundary`): it moves an item's marker by at
#: most this fraction of the noise sd, above the 1% at which it is taken
#: as 0.
_OFF_START = 0.05

#: A search off the boundary must beat the fit on it by this much
#: (relative, with an absolute floor of the same in log-likelihood
#: units) to be kept: less is round-off, and the fit on the boundary,
#: with every other parameter's standard error, is reported.
_RISE_TOL = 1e-6


def _spread_reach(model: Model, arrays: JointArrays, theta: dict) -> Any:
    """How far one unit of each varying parameter moves an item's
    marker, in the noise sds of its readings, at most over the readings
    ``(q,)``: the path's derivative in it at each item's mean path
    (``mu`` and ``Delta`` of ``theta``), over the noise sd. 0 for a
    parameter no reading depends on."""
    layout = model.layout
    V = list(layout.varying)
    mu = np.asarray(theta["mu"], dtype=float)
    Delta = np.asarray(theta["Delta"], dtype=float)
    params = mu[None, :] + np.asarray(arrays.Z, dtype=float) @ Delta.T
    with np.errstate(all="ignore"):
        if several_markers(model.path):
            _, J = model.path.reading_jacobians(params, arrays)
            noise = np.asarray(
                model.path.noise(theta["sigma"], arrays), dtype=float
            )
        else:
            ages = np.where(arrays.mask, arrays.ages, 0.0)
            P = np.broadcast_to(params[:, None, :], ages.shape + (layout.P,))
            J = path_jacobian(model.path.model, ages, np.array(P))
            noise = np.full(ages.shape[1], float(np.ravel(theta["sigma"])[0]))
        moved = np.abs(np.asarray(J, dtype=float)[..., V]) / np.reshape(
            noise, (1, -1, 1)
        )
    moved = np.where(arrays.mask[..., None] & np.isfinite(moved), moved, 0.0)
    if moved.size == 0:
        return np.zeros(len(V))
    return np.max(moved, axis=(0, 1))


def _spreads_at_zero(model: Model, arrays: JointArrays, u: Any) -> list:
    """The varying parameters (their places in ``model.layout.varying``)
    whose spread between items at the internal parameters ``u`` moves no
    item's marker by :data:`_NO_SPREAD` of the noise sd at any reading
    age: at their lower bound, 0. A parameter no reading depends on is
    not judged here."""
    theta = model.layout.unpack(np.asarray(u, dtype=float))
    L = np.asarray(theta["L"], dtype=float)
    sd = np.sqrt(np.sum(L**2, axis=1))
    reach = _spread_reach(model, arrays, theta)
    return [
        k
        for k in range(model.layout.q)
        if reach[k] > 0 and not sd[k] * reach[k] >= _NO_SPREAD
    ]


#: A spread whose loss (with its correlations) costs the log-likelihood
#: less than this is not told from 0 by the data at the 5% level (half
#: the 95% point of chi-squared on one degree of freedom): where a search
#: ends unverified, such spreads are tried at 0 (:func:`_on_the_boundary`).
_NOT_TOLD = 1.92


def _spreads_not_told_from_zero(
    model: Model, arrays: JointArrays, u: Any, value: float
) -> list:
    """The varying parameters (places in ``model.layout.varying``) whose
    spread, taken to (nearly) 0 with its correlations and every other
    parameter where it is at ``u`` (its negative log-likelihood
    ``value``), costs the log-likelihood less than :data:`_NOT_TOLD`:
    one likelihood a varying parameter, by the nodes placed there."""
    layout = model.layout
    theta = _theta_of(layout, u)
    L = np.asarray(theta["L"], dtype=float)
    D = L @ L.T
    out = []
    for k in range(layout.q):
        small = D.copy()
        small[k, :] = small[:, k] = 0.0
        small[k, k] = 1e-6 * D[k, k]
        try:
            trial = dict(theta, L=np.linalg.cholesky(small))
            v = layout.pack(trial)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                placed, _ = place(v, model, arrays, (None, None))
                lost = float(neg_ll(v, model, arrays, placed)) - value
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            continue
        if np.isfinite(lost) and lost < _NOT_TOLD:
            out.append(k)
    return out


def _at_a_spread_of_zero(
    model: Model, arrays: JointArrays, u: Any, value: float, maximum: str
) -> tuple | None:
    """:func:`_on_the_boundary`'s answer where a spread is at 0 at the
    search's answer ``u`` (:func:`_spreads_at_zero`), or, the search
    unverified, is not told from 0 (:func:`_spreads_not_told_from_zero`);
    ``None`` where none is, or the answer stands."""
    if not model.layout.q:
        return None
    at_zero = _spreads_at_zero(model, arrays, u)
    if not at_zero and maximum != "verified":
        # a search stalled on its way to 0 stops short of it
        at_zero = _spreads_not_told_from_zero(model, arrays, u, value)
    if not at_zero:
        return None
    return _on_the_boundary(model, arrays, (u, value), at_zero)


def _with_varying(model: Model, keep: list) -> Model:
    """``model`` with only the varying parameters at the places ``keep``
    of its layout's varying ones: the others common to all items."""
    import dataclasses

    layout = model.layout
    nested = dataclasses.replace(
        layout,
        varying=tuple(layout.varying[k] for k in keep),
        blocks=tuple(layout.blocks[k] for k in keep),
    )
    return dataclasses.replace(model, layout=nested)


def _theta_of(layout: Layout, u: Any) -> dict:
    return {
        key: np.array(value, dtype=float)
        for key, value in layout.unpack(np.asarray(u, dtype=float)).items()
    }


def _project(theta: dict, keep: list) -> dict:
    """``theta`` with only the spread of the varying parameters at the
    places ``keep``: their covariance kept, the others' dropped."""
    out = dict(theta)
    L = np.asarray(theta["L"], dtype=float)
    D = (L @ L.T)[np.ix_(keep, keep)]
    out["L"] = np.linalg.cholesky(0.5 * (D + D.T)) if keep else D
    return out


def _embed(theta: dict, keep: list, q: int, sds: Any) -> dict:
    """``theta`` (of the varying parameters at the places ``keep``) with
    all ``q``: those not kept uncorrelated with every other, their sds
    ``sds`` (by place)."""
    out = dict(theta)
    L = np.zeros((q, q))
    L[np.ix_(keep, keep)] = np.asarray(theta["L"], dtype=float)
    for k in range(q):
        if k not in keep:
            L[k, k] = sds[k]
    out["L"] = L
    return out


def _maximised(model: Model, arrays: JointArrays, theta: dict) -> Any:
    """:func:`_maximise`'s answer from ``theta`` with the warnings it
    gave, ``(answer, warnings)``; ``None`` where it cannot start."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            answer = _maximise(model, arrays, theta)
        except (_NoStart, np.linalg.LinAlgError, ValueError):
            return None
    return answer, list(caught)


def _on_the_boundary(
    model: Model, arrays: JointArrays, first: tuple, at_zero: list
) -> tuple | None:
    """The fit where the spreads of the varying parameters ``at_zero``
    (places in ``model.layout.varying``) are at their lower bound, 0, at
    the first search's answer ``first`` ``(u, value)``.

    The search's scale for a spread is its log sd, so 0 is at minus
    infinity, where the likelihood is flat along it: a search can stall
    there short of the fit with that parameter common to all items, which
    is this model with the spread 0 (#1334). So: (1) the fit with those
    parameters common to all items (the nested model), from the first
    answer's other parameters, again while it puts another spread at 0;
    (2) a search of the whole model from that fit, those spreads small
    (:data:`_OFF_START`), which finds where the likelihood rises off the
    boundary. Where (2), or the first answer, beats (1) by more than
    :data:`_RISE_TOL`, the better of them is the fit (off the boundary);
    otherwise the fit is held on it: (1).

    Returns ``(fit_model, (u, value, cov_u, maximum), warnings,
    path_u)``: the model searched (the nested one where held on the
    boundary) with its answer and warnings, and ``path_u`` the whole
    model's internal parameters off the boundary (a start for another
    link's fit), or ``None`` where the nested fit cannot be made (the
    first answer then stands)."""
    layout = model.layout
    q = layout.q
    u1, v1 = first
    dropped = sorted(set(at_zero))
    theta = _theta_of(layout, u1)
    keep = [k for k in range(q) if k not in dropped]
    start = _project(theta, keep)
    while True:
        if not keep:
            return None
        nested = _with_varying(model, keep)
        got = _maximised(nested, arrays, start)
        if got is None:
            return None
        (u_n, v_n, cov_n, max_n), warned_n = got
        more = _spreads_at_zero(nested, arrays, u_n)
        if not more:
            break
        theta_n = _theta_of(nested.layout, u_n)
        start = _project(
            theta_n, [j for j in range(len(keep)) if j not in more]
        )
        keep = [k for j, k in enumerate(keep) if j not in more]
    # (2) off the boundary: the spreads at 0 small
    theta_n = _theta_of(nested.layout, u_n)
    reach = _spread_reach(model, arrays, theta_n)
    sds = np.where(reach > 0, _OFF_START / np.where(reach > 0, reach, 1.0), 0)
    off_start = _embed(theta_n, keep, q, sds)
    path_u = layout.pack(off_start)
    off = _maximised(model, arrays, off_start)
    best = None
    if off is not None:
        (u_o, v_o, cov_o, max_o), warned_o = off
        path_u = u_o
        if np.isfinite(v_o):
            best = (v_o, (model, (u_o, v_o, cov_o, max_o), warned_o, None))
    tol = _RISE_TOL * max(1.0, abs(v_n))
    if best is not None and best[0] < v_n - tol and best[0] <= v1:
        return best[1]
    if v1 < v_n - tol:
        # the first answer is above the boundary: it stands
        return None
    return nested, (u_n, v_n, cov_n, max_n), warned_n, path_u


def _held_on_the_boundary(
    names: list,
    fit_names: list,
    estimated: npt.NDArray,
    covariance: npt.NDArray | None,
) -> tuple:
    """The reported parameters ``estimated`` (named ``fit_names``, of the
    fit with some spreads at 0) and their ``covariance`` in the places of
    the whole model's ``names``: those spreads and their correlations 0,
    with no standard errors (``nan`` in the covariance)."""
    where = [names.index(name) for name in fit_names]
    out = np.zeros(len(names))
    out[where] = estimated
    cov = None
    if covariance is not None:
        cov = np.full((len(names),) * 2, np.nan)
        cov[np.ix_(where, where)] = covariance
    return out, cov


#: An item with at most this many readings has few for Laplace's
#: approximation, and a fit warns where more than this share of the
#: items have. From ``scripts/studies/joint_recovery/
#: laplace_few_readings.py``'s confirming run, its design and rule fixed
#: before it ran: of 20 fleets of pumps with noisy readings, 9% to 36% of
#: them read three times or fewer, every fleet with up to 15% so read was
#: within 0.074 of the full grid's standard errors, and the first one over
#: 0.1 had 16% (0.101).
_FEW_READINGS = 3
_FEW_SHARE = 0.15


def _check_laplace(arrays: JointArrays, q: int) -> None:
    """Refuse what Laplace's approximation is not for (truncated items),
    and warn where it is weak: many items with few readings."""
    on_a_grid = (
        "use integration='grid' (or 'pruned')"
        if q <= MAX_VARYING
        else f"name up to {MAX_VARYING} path parameters in varying=(...) "
        "and use integration='grid' (or 'pruned')"
    )
    truncated = int(np.sum(arrays.truncated))
    if truncated:
        raise ValueError(
            "integration='laplace' does not take truncated items "
            f"({truncated} here came under observation at an age, or are "
            "in the data because they failed by one): the chance of an "
            "item being in the data is an integral with no readings in "
            "it, which Laplace's approximation is not accurate for; "
            f"{on_a_grid}."
        )
    few = float(np.sum(arrays.w[arrays.n_read <= _FEW_READINGS]))
    total = float(np.sum(arrays.w))
    if few > _FEW_SHARE * total:
        warnings.warn(
            f"integration='laplace': {few:g} of the {total:g} items "
            f"({few / total:.0%}) have at most {_FEW_READINGS} readings. "
            "Laplace's approximation of an item's likelihood is off by a "
            "term of the order of one over its number of readings, more "
            "where the readings are noisy: on simulated fleets with "
            f"more than {_FEW_SHARE:.0%} of the items read so few times "
            "its estimates were up to 0.22 standard errors from the full "
            "grid's (0.4 with every item read so few times; the marker's "
            "effect the most). To be sure of them, "
            f"{on_a_grid}.",
            stacklevel=caller_stacklevel(),
        )


class _NoStart(ValueError):
    """The likelihood is not finite at a fit's starting values."""


def _maximise(model: Model, arrays: JointArrays, theta: dict) -> tuple:
    """The joint likelihood's maximum from the start ``theta``:
    ``(u, value, cov_u, maximum)``, the internal parameters, the negative
    log-likelihood there, their covariance (``None`` where the Hessian is
    not positive definite) and what the search reached (``"verified"`` or
    ``"unverified"``, with a warning)."""
    layout = model.layout
    u = layout.pack(theta)

    def objective(v: Any, placed: tuple) -> Any:
        return neg_ll(v, model, arrays, placed)

    search = value_and_grad(objective)
    curvature = hessian(objective)

    def evaluated(v: npt.NDArray, starts: tuple) -> tuple:
        """The likelihood at ``v`` by the nodes placed there (``inf``
        where they cannot be placed), with the nodes and the modes."""
        try:
            with np.errstate(all="ignore"):
                placed, starts = place(v, model, arrays, starts)
                value = float(objective(v, placed))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return np.inf, None, starts
        return (value if np.isfinite(value) else np.inf), placed, starts

    value, placed, starts = evaluated(u, (None, None))
    if placed is None:
        raise _NoStart(
            "The joint fit could not start: the likelihood is not finite at "
            "its starting values (check the readings and the path model)."
        )
    u, value, placed, starts, converged = _descend(
        u, value, placed, starts, search, curvature, evaluated
    )

    # the Hessian a maximum is checked by, and the covariance taken from
    def laplace_curvature(v: npt.NDArray, at: tuple) -> npt.NDArray:
        return _laplace_curvature(v, at, search, curvature, evaluated)

    checked_curvature = laplace_curvature if model.laplace else curvature
    value, grad, cov_u, positive, flat = _check_maximum(
        u, placed, search, checked_curvature
    )
    # A search can stop short of the maximum on the flat shelf where the
    # items' spread nears a correlation of +-1 (the log-scale Cholesky
    # diagonal reaches it only at -inf, and the gradient vanishes well
    # before): its answer is then not a verified maximum. Such an answer
    # gets one more search, from the same sds uncorrelated. A verified
    # answer from it is kept; otherwise the lower of the two. The first
    # answer's likelihood is not compared with a verified one: it is not
    # a maximum, and near a correlation of +-1 the adaptive nodes can be
    # far off (one fleet: 5099.8 by the nodes, 5197.8 by a grid integral,
    # against the maximum inside at 5159.0). A verified answer is kept as
    # it is, with no second search.
    inside = None
    if layout.q >= 2 and not (converged and positive and flat):
        inside = _uncorrelated(u, layout)
    if inside is not None and np.all(np.isfinite(inside)):
        retry_value, retry_placed, retry_starts = evaluated(
            inside, (None, None)
        )
        if retry_placed is not None and np.isfinite(retry_value):
            retry = _descend(
                inside,
                retry_value,
                retry_placed,
                retry_starts,
                search,
                curvature,
                evaluated,
            )
            checked = _check_maximum(
                retry[0], retry[2], search, checked_curvature
            )
            sound = retry[4] and checked[3] and checked[4]
            lower = retry[1] < value - _ROUND_TOL * max(1.0, abs(value))
            if sound or lower:
                u, _, placed, starts, converged = retry
                value, grad, cov_u, positive, flat = checked
    verified = converged and positive and flat and np.all(np.isfinite(u))
    runaway = _runaway_hazard_coefficients(objective, u, placed, layout)
    if verified and runaway:
        verified = False
    if not verified:
        reason = None
        if runaway:
            reason = (
                "the likelihood still rises along the hazard coefficient(s) "
                f"{runaway}: a covariate may separate the items that failed "
                "from those that did not"
            )
        elif not converged:
            reason = "the rounds of placing the nodes did not settle"
        elif not positive:
            reason = "the Hessian is not positive definite there"
        elif not flat:
            reason = "the gradient is not zero there"
        warn_unverified("The joint fit", reason)
    return u, value, cov_u, "verified" if verified else "unverified"


def _two_stage(model: Model, arrays: JointArrays, theta: dict) -> tuple:
    """The two-stage estimate from stage one's marker part ``theta``:
    ``(u, value, cov_u, maximum)`` as :func:`_maximise` gives them.
    There is no likelihood of the whole model (``value`` is ``nan``);
    the covariance is stage two's, of the survival part alone (the
    marker part's entries 0, to be set to ``nan`` once reported), and
    ``maximum`` is ``"not applicable"``, or ``"unverified"``, with a
    warning, where stage two's search did not reach a verified maximum
    of its own likelihood."""
    theta, cov, reason = survival_stage(model, arrays, theta)
    u = model.layout.pack(theta)
    cov_u = None
    if cov is not None:
        k = cov.shape[0]
        cov_u = np.zeros((u.size, u.size))
        cov_u[-k:, -k:] = cov
    if reason is not None:
        warn_unverified(
            "Stage two of the two-stage fit (the survival part, each "
            "item's predicted path taken as known)",
            reason,
            "check the fit, or fit the joint likelihood (how='MLE')",
        )
        return u, np.nan, cov_u, "unverified"
    return u, np.nan, cov_u, "not applicable"


def _given_path_start(start: dict | None, layout: Layout) -> dict | None:
    """A copy of the path part ``start`` (``mu``, ``Delta``, ``L``,
    ``sigma``) where its shapes are this layout's; ``None`` otherwise."""
    if start is None:
        return None
    shapes = {
        "mu": (layout.P,),
        "Delta": (layout.P, layout.p),
        "L": (layout.q, layout.q),
        "sigma": (),
    }
    out = {key: np.array(start[key], dtype=float) for key in shapes}
    if any(out[key].shape != shape for key, shape in shapes.items()):
        return None
    return out


def _whitening(curvature: Any, u: npt.NDArray, placed: tuple) -> npt.NDArray:
    """``S`` with ``S' H S = I`` for the Hessian ``H`` at ``u`` (its
    inverse Cholesky factor), so the search sees each parameter in its
    standard errors; the diagonal where ``H`` is not positive definite."""
    with np.errstate(all="ignore"):
        H = np.asarray(curvature(u, placed), dtype=float)
    H = 0.5 * (H + H.T)
    if np.all(np.isfinite(H)):
        size = _equilibration(H)
        try:
            root = np.linalg.cholesky(H * np.outer(size, size))
            return size[:, None] * np.linalg.inv(root).T
        except np.linalg.LinAlgError:
            pass
    diag = np.diag(H) if np.all(np.isfinite(np.diag(H))) else np.ones(u.size)
    return np.diag(1.0 / np.sqrt(np.where(diag > 0, diag, 1.0)))


def _descend(
    u: npt.NDArray,
    value: float,
    placed: tuple,
    starts: tuple,
    search: Any,
    curvature: Any,
    evaluated: Any,
) -> tuple:
    """Rounds of search from ``u`` (its likelihood ``value`` by the
    nodes ``placed``): ``(u, value, placed, starts, converged)``.
    ``search`` and ``curvature`` are the likelihood's value and gradient
    and its Hessian by given nodes, ``evaluated`` the likelihood by nodes
    placed at a point (``_fit``'s)."""
    converged = False
    for _ in range(_ROUNDS):
        # A search with the nodes held where they are, then the step it
        # proposes taken as far as the likelihood by nodes placed at the
        # new point improves on this one's (backtracking where not):
        # each round is a descent of the adaptive quadrature's likelihood.
        S = _whitening(curvature, u, placed)

        def whitened(v: npt.NDArray, u0: npt.NDArray = u) -> tuple:
            with np.errstate(all="ignore"):
                f, grad = search(u0 + S @ v, placed)
            if not np.isfinite(f) or not np.all(np.isfinite(grad)):
                return np.inf, np.zeros_like(v)
            return f, S.T @ grad

        with ignoring_warnings():
            result = minimize(
                whitened,
                np.zeros(u.size),
                jac=True,
                method="BFGS",
                options={"gtol": 1e-7, "maxiter": 5000},
            )
        step = result.x if np.all(np.isfinite(result.x)) else 0 * result.x
        size = 1.0
        while True:
            trial = u + S @ (size * step)
            new_value, new_placed, new_starts = evaluated(trial, starts)
            if new_value <= value + 1e-12 * max(1.0, abs(value)):
                break
            size /= 4.0
            if size < 1e-6:
                trial, new_value, new_placed = u, value, placed
                new_starts = starts
                break
        moved = float(np.max(np.abs(size * step), initial=0.0))
        settled = abs(value - new_value) <= _ROUND_TOL * max(1.0, abs(value))
        u, value, placed, starts = trial, new_value, new_placed, new_starts
        if settled and moved < 1e-3:
            converged = True
            break
    return u, value, placed, starts, converged


def _runaway_hazard_coefficients(
    objective: Any, u: npt.NDArray, placed: tuple, layout: Any
) -> list:
    """The hazard covariates (by column) along whose coefficient the
    likelihood has no finite maximum near ``u`` (a level of a covariate
    with no failures runs its coefficient off to infinity): Newton's test
    along each coefficient's profile
    (:func:`~surpyval.univariate.parametric.fitters.runaway.runaway_coefficients`),
    with the nodes held where the fit placed them. Empty for a model of
    failure modes (whose coefficients are laid out per mode) or one with
    no hazard covariates."""
    if getattr(layout, "n_causes", 1) != 1 or not layout.n_hazard:
        return []
    from surpyval.univariate.parametric.fitters.runaway import (
        runaway_coefficients,
    )

    positions = layout.hazard_positions()
    with ignoring_warnings(), np.errstate(all="ignore"):
        found = runaway_coefficients(
            lambda v: objective(v, placed),
            u,
            [pos for pos, _ in positions],
        )
    return [positions[k][1] for k in found]


def _check_maximum(
    u: npt.NDArray, placed: tuple, search: Any, curvature: Any
) -> tuple:
    """``(value, grad, cov_u, positive, flat)`` at ``u`` by the nodes
    ``placed``: the likelihood and its gradient, the inverse Hessian
    (``None`` where it is not positive definite), whether it is, and
    whether the gradient is zero in standard errors (below ``_GTOL``).
    The Hessian is judged and inverted in each parameter's own scale: a
    slope a second has a curvature 1e7 squared times one an hour's."""
    with np.errstate(all="ignore"):
        value, grad = search(u, placed)
        info = curvature(u, placed)
    info = 0.5 * (info + info.T)
    cov_u, positive = None, False
    if np.all(np.isfinite(info)):
        own = _equilibration(info)
        scaled = info * np.outer(own, own)
        try:
            positive = bool(np.all(np.linalg.eigvalsh(scaled) > 0))
            cov_u = (
                np.linalg.inv(scaled) * np.outer(own, own)
                if positive
                else None
            )
        except np.linalg.LinAlgError:
            positive = False
    flat = False
    if cov_u is not None and np.all(np.isfinite(grad)):
        sd = np.sqrt(np.abs(np.diag(cov_u)))
        flat = bool(np.max(np.abs(grad) * sd) < _GTOL)
    return value, grad, cov_u, positive, flat


#: The step of the differences of a Laplace fit's gradient, in standard
#: errors of each parameter.
_LAPLACE_STEP = 5e-3


def _laplace_curvature(
    u: npt.NDArray,
    placed: tuple,
    search: Any,
    curvature: Any,
    evaluated: Any,
) -> npt.NDArray:
    """The Hessian at ``u`` of the Laplace approximation's negative
    log-likelihood: central differences of its gradient, each taken with
    the stencils placed at the point it is taken at (``evaluated``), so
    the modes' movement with the parameters is in it to every order. The
    steps are ``_LAPLACE_STEP`` standard errors by the held stencils'
    Hessian (``curvature``), which leaves the second-order terms of that
    movement out and so is not itself the answer. ``nan`` where a
    stencil cannot be placed."""
    S = _whitening(curvature, u, placed)
    modes_at = (placed[0].mode, None)
    k = u.size
    H = np.full((k, k), np.nan)
    for j in range(k):
        grads = []
        for sign in (1.0, -1.0):
            v = u + sign * _LAPLACE_STEP * S[:, j]
            _, at, _ = evaluated(v, modes_at)
            if at is None:
                return np.full((k, k), np.nan)
            with np.errstate(all="ignore"):
                grads.append(search(v, at)[1])
        H[:, j] = S.T @ (grads[0] - grads[1]) / (2.0 * _LAPLACE_STEP)
    inverse = np.linalg.inv(S)
    return inverse.T @ (0.5 * (H + H.T)) @ inverse


def _uncorrelated(u: npt.NDArray, layout: Layout) -> npt.NDArray:
    """``u`` with the items' spread made uncorrelated, each varying
    parameter's sd kept."""
    theta = layout.unpack(np.asarray(u, dtype=float))
    theta = {k: np.asarray(v, dtype=float) for k, v in theta.items()}
    L = theta["L"]
    theta["L"] = np.diag(np.sqrt(np.sum(L**2, axis=1)))
    return layout.pack(theta)


def _equilibration(H: npt.NDArray) -> npt.NDArray:
    """``1 / sqrt(H_ii)`` (1 where that is not positive and finite):
    the scaling that puts each parameter of ``H`` in its own units, so a
    factorisation or an eigenvalue test of ``H`` does not depend on how
    different their sizes are (on the unit of time)."""
    diag = np.diag(H)
    ok = np.isfinite(diag) & (diag > 0)
    return np.where(ok, 1.0 / np.sqrt(np.where(ok, diag, 1.0)), 1.0)


def search_space(model: JointModel, **data: Any) -> tuple:
    """``(f, u, n)``: the negative log-likelihood the fit of ``data`` (the
    fit's arguments) searches, ``f(u)``, by the adaptive nodes placed at
    the model's parameters; the model's point ``u`` in that space; and
    the number of items. For the checks that a fit's answer is a
    maximum (``tests/conformance``)."""
    lik, arrays, u = likelihood_parts(model, **data)
    placed, _ = place(u, lik, arrays, (None, None))

    def f(v: Any) -> Any:
        return neg_ll(v, lik, arrays, placed)

    return f, u, float(np.sum(arrays.w))


def likelihood_parts(model: JointModel, **data: Any) -> tuple:
    """``(lik, arrays, u)``: what the likelihood of ``data`` (the fit's
    arguments) is computed from (:func:`place`, :func:`neg_ll`) and the
    model's point in the search's space."""
    from .joint_aft import JointAFT

    factory = JointAFT if model.survival == "AFT" else JointPH
    # (an accelerated model has no accumulated exposure to weight)
    weighted = {} if model.weight is None else {"weight": model.weight}
    if model.markers is not None:
        return _marker_likelihood_parts(model, factory, weighted, data)
    fitter = factory(
        model.dist,
        model.path_model,
        model.varying,
        model.link,
        model.lag,
        **weighted,
        random_effects=model.random_effects_structure,
    )
    records, counts, names = _records(
        data["x"],
        data.get("c"),
        data.get("n"),
        data.get("t"),
        data.get("tl"),
        data.get("tr"),
        data.get("Z"),
        data["markers"],
    )
    names = list(model.feature_names)
    T, d, L, Z_all, ages, values, U, Rt = _arrays(
        fitter, records, counts, len(names)
    )
    Z = Z_all[:, model._kept] - model._zbar
    causes = _causes(fitter, data.get("e"), records, counts, "MLE")
    arrays = JointArrays.build(
        counts,
        T,
        d,
        L,
        Z,
        ages,
        values,
        U=U,
        Rt=Rt,
        e=None if causes is None else causes[1],
    )
    layout = model._layout
    lik = model._likelihood_model()
    theta = model._theta()
    u = layout.pack({k: np.asarray(v, dtype=float) for k, v in theta.items()})
    return lik, arrays, u


def _marker_likelihood_parts(
    model: JointModel, factory: Any, weighted: dict, data: dict
) -> tuple:
    """:func:`likelihood_parts` of a model of several markers."""
    from ._multi import checked_readings, joined, rows_by_marker

    names = list(model.markers or [])
    links: dict = {name: [] for name in names}
    for name, link in model.link:
        links[name].append(link)
    fitter = factory(
        model.dist,
        dict(zip(names, model.path_model.models)),
        list(model.varying),
        links,
        model.lag,
        **weighted,
        random_effects=model.random_effects_structure,
    )
    rows = rows_by_marker(data["markers"], names)
    records, counts, _ = _records(
        data["x"],
        data.get("c"),
        data.get("n"),
        data.get("t"),
        data.get("tl"),
        data.get("tr"),
        data.get("Z"),
        [{}] * len(rows),
    )
    readings = [checked_readings(r, rows[int(r.id)], names) for r in records]
    p = len(model.feature_names)
    T, d, L, Z_all, U, Rt = _marker_arrays(
        fitter, records, readings, counts, p
    )
    Z = Z_all[:, model._kept] - model._zbar
    each = [joined(item) for item in readings]
    causes = _causes(fitter, data.get("e"), records, counts, "MLE")
    arrays = JointArrays.build(
        counts,
        T,
        d,
        L,
        Z,
        [a for a, _, _ in each],
        [v for _, v, _ in each],
        U=U,
        Rt=Rt,
        e=None if causes is None else causes[1],
        channels=[ch for _, _, ch in each],
        n_markers=len(names),
    )
    layout = model._layout
    lik = model._likelihood_model()
    theta = model._theta()
    u = layout.pack({k: np.asarray(v, dtype=float) for k, v in theta.items()})
    return lik, arrays, u
