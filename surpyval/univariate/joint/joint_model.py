"""The fitted joint model of a marker and a lifetime, :class:`JointModel`
(#891)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
from autograd import make_jvp
from autograd.scipy.special import logsumexp
from scipy.stats import norm

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
    to_native,
)
from surpyval.univariate.information_criteria import (
    InformationCriteriaMixin,
    ic_sample_size,
)
from surpyval.utils.linalg import bound_signs, standard_errors_of
from surpyval.utils.no_maximum import maximum_entry, restored_maximum
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.validation import (
    BOUNDS,
    check_alpha_ci,
    check_option,
    no_covariance_error,
)

from ._likelihood import (
    EFFECTS,
    GH_FIT,
    GH_PREDICT,
    INTEGRATIONS,
    JointArrays,
    Layout,
    MarkerPaths,
    Model,
    PathFunction,
    _at,
    _path_params,
    block_labels,
    cause_thetas,
    check_lag,
    check_random_effects,
    check_weight,
    continued_modes,
    log1mexp,
    log_integrals,
    log_integrand,
    scatter,
)
from ._multi import (
    lag_to_dict,
    lag_words,
    marker_lags,
    marker_weights,
    weight_to_dict,
    weight_words,
)

if TYPE_CHECKING:
    import pandas as pd

    from .dynamic import DynamicPrediction

#: The functions ``cb`` bounds ('R' and 'F' are 'sf' and 'ff').
_CB_ON = ("sf", "R", "ff", "F", "cif")

#: Each link in words, for the printout.
_LINK_WORDS = {
    "value": "current value",
    "slope": "slope",
    "cumulative": "cumulative area",
}


def _weight_words(weight: tuple | None) -> str:
    """The weight on the accumulated exposure in words, to follow the
    links' (empty with none)."""
    if weight is None:
        return ""
    if weight[0] == "window":
        return f", the area over the last {weight[1]:g} only (window)"
    return f", the area fading over {weight[1]:g} (exponential weight)"


#: The ways a joint model is fitted (``how``): the joint likelihood's
#: maximum, or the approximate two-stage estimate.
HOWS = ("MLE", "two-stage")

#: What a two-stage fit's printout and summary say of each part: stage
#: one's moment estimates have no standard errors, and stage two's
#: estimates are approximate with standard errors that are too small.
_STAGE_ONE = "two-stage, stage 1: no standard errors"
_STAGE_TWO = "two-stage, stage 2: approximate, standard errors understated"

#: Each survival part in words, for the printout.
_SURVIVAL_WORDS = {
    "PH": "proportional hazards",
    "AFT": "accelerated failure time",
}

#: Each integration in words, for the printout.
_INTEGRATION_WORDS = {
    "grid": f"adaptive Gauss-Hermite, full grid ({GH_FIT} points a dimension)",
    "pruned": (
        f"adaptive Gauss-Hermite, pruned grid ({GH_FIT} points a "
        "dimension, without the nodes of negligible weight)"
    ),
    "laplace": "Laplace's approximation (no grid)",
}

#: The formula of each built-in path model, by its name, for the
#: printout; a path model of one's own may give its ``formula``.
FORMULAS = {
    "Linear": "a + b t",
    "Quadratic": "a + b t + c t^2",
    "Exponential": "a exp(b t)",
    "Power": "a t^b",
    "Logarithmic": "a + b log(t)",
    "Lloyd-Lipow": "a - b / t",
    "Gompertz": "a exp(-b exp(-c t))",
    "Offset Exponential": "a + b exp(c t)",
    "Michaelis-Menten": "a t / (b + t)",
}


def path_formula(path_model: Any) -> str:
    """The path's formula: its own ``formula`` where it has one, else the
    built-in one of its name, else its name's parameters; a path on
    links (``LinkedPathModel``) says which parameters are on a log link.
    """
    from surpyval.degradation.stress import LinkedPathModel

    if isinstance(path_model, LinkedPathModel):
        logged = [n for n, link in path_model.links.items() if link == "log"]
        formula = path_formula(path_model.base)
        return formula + "".join(f"; {n} on a log link" for n in logged)
    own = getattr(path_model, "formula", None)
    if isinstance(own, str) and own:
        return own
    return FORMULAS.get(
        path_model.name,
        "path(t; {})".format(", ".join(path_model.parameter_names)),
    )


def covariate_parts(
    path_names: list[str],
    covariates: list[str],
    path_covariates: Any = None,
    hazard_covariates: Any = None,
) -> tuple[dict[str, list[str]], list[str]]:
    """``(shifts, hazard)``: the covariates that shift each path
    parameter's mean, ``{path parameter: [covariate, ...]}``, and those
    in the hazard, each in the order of ``covariates``, from a fit's
    ``path_covariates`` and ``hazard_covariates``. ``None`` is every
    covariate; ``path_covariates`` may be a list (each path parameter) or
    a dict by path parameter (one left out has none). A name that is not
    a covariate, or a covariate in neither part, is refused."""

    def names_of(value: Any, what: str) -> list[str]:
        if isinstance(value, str):
            value = [value]
        try:
            given = [str(v) for v in value]
        except TypeError:
            raise ValueError(
                f"{what} is a list of covariate names; got {value!r}."
            ) from None
        unknown = [v for v in given if v not in covariates]
        if unknown:
            raise ValueError(
                f"{what} names {unknown!r}, which are not covariates of the "
                f"data; its covariates are {list(covariates)!r}."
            )
        return [c for c in covariates if c in given]

    if hazard_covariates is None:
        hazard = list(covariates)
    else:
        hazard = names_of(hazard_covariates, "hazard_covariates")
    if path_covariates is None:
        shifts = {name: list(covariates) for name in path_names}
    elif isinstance(path_covariates, Mapping):
        unknown = [k for k in path_covariates if k not in path_names]
        if unknown:
            raise ValueError(
                "path_covariates names the path parameters whose means the "
                f"covariates shift, of {list(path_names)!r}; got "
                f"{unknown!r}."
            )
        shifts = {
            name: names_of(
                path_covariates.get(name, []),
                f"path_covariates[{name!r}]",
            )
            for name in path_names
        }
    else:
        every = names_of(path_covariates, "path_covariates")
        shifts = {name: list(every) for name in path_names}
    used = set(hazard).union(*shifts.values())
    unused = [c for c in covariates if c not in used]
    if unused:
        raise ValueError(
            f"The covariates {unused!r} enter neither the hazard nor the "
            "path: name them in hazard_covariates or path_covariates, or "
            "leave them out of the data."
        )
    return shifts, hazard


def parameter_names(
    path_names: list[str],
    varying: list[str],
    covariates: list[str],
    shifts: dict[str, list[str]] | None = None,
    hazard: list[str] | None = None,
    links: tuple = ("value",),
    random_effects: Any = "full",
    causes: Any = None,
    shared: Any = (),
    markers: Any = None,
) -> list[str]:
    """The names of a joint model's parameters: each path parameter
    followed by its covariates' shifts of its mean (``a``, ``a_<cov>``,
    ``b``, ``b_<cov>``, ...); the spread of the varying ones (``sd_a``,
    ``sd_b``, ..., then ``corr_a_b``, ... of each pair correlated in
    ``random_effects``, checked: every pair where it is ``"full"``, the
    default); ``sd_noise``; the Weibull ``alpha`` and ``beta``; the
    hazard's covariate coefficients (``<cov>``); and the effect of each
    of ``links``, what of the path the hazard follows (``marker_effect``
    of its value, ``slope_effect``, ``cumulative_effect``). ``shifts``
    and ``hazard`` (as :func:`covariate_parts` gives them) say which
    covariates enter which part; default every one in both.

    With ``causes`` (competing failure modes) the survival part is each
    cause's own, ``"<cause>: alpha"``, ``"<cause>: beta"``, its hazard
    coefficients and effects, cause by cause (named as
    ``ParametricCompetingRisks`` names its causes' parameters), then the
    parameters of ``shared`` (``"beta"``, a hazard covariate or an
    effect), which the causes have in common, one each under its plain
    name.

    With ``markers`` (several markers' names) the path names are
    ``"<marker>.<name>"``, ``links`` are ``(marker, link)`` pairs, the
    noise is ``"<marker>.sd_noise"`` a marker and each effect
    ``"<marker>.marker_effect"`` (:func:`effect_name`)."""
    if shifts is None:
        shifts = {name: list(covariates) for name in path_names}
    if hazard is None:
        hazard = list(covariates)
    out = []
    for name in path_names:
        out += [name] + [f"{name}_{cov}" for cov in shifts.get(name, [])]
    out += [f"sd_{name}" for name in varying]
    label = block_labels(random_effects, varying)
    out += [
        f"corr_{a}_{b}"
        for i, a in enumerate(varying)
        for j, b in enumerate(varying)
        if j > i and label[i] == label[j]
    ]
    effects = [effect_name(link) for link in links]
    noise = (
        ["sd_noise"]
        if markers is None
        else [f"{marker}.sd_noise" for marker in markers]
    )
    if causes is not None:
        every = ["alpha", "beta", *hazard, *effects]
        common = [name for name in every[1:] if name in shared]
        own = [name for name in every if name not in common]
        out += noise
        out += [f"{cause}: {name}" for cause in causes for name in own]
        return out + common
    return out + [*noise, "alpha", "beta", *hazard, *effects]


def effect_name(link: Any) -> str:
    """The name of a link's effect: ``marker_effect``,
    ``slope_effect``, ``cumulative_effect``; of a marker's link (a
    ``(marker, link)`` pair, several markers) prefixed by the marker,
    ``"vibration.marker_effect"``."""
    if isinstance(link, tuple):
        return f"{link[0]}.{EFFECTS[link[1]]}"
    return EFFECTS[link]


def link_words(link: Any) -> str:
    """A link in words: ``"current value"``; of a marker's link, the
    marker's (``"vibration's current value"``)."""
    if isinstance(link, tuple):
        return f"{link[0]}'s {_LINK_WORDS[link[1]]}"
    return _LINK_WORDS[link]


def shareable(hazard: list[str], links: tuple) -> list[str]:
    """What the causes of a joint model may share (``shared=``): the
    Weibull ``beta``, a hazard covariate's coefficient, and a link's
    effect."""
    return ["beta", *hazard, *[effect_name(link) for link in links]]


def _reported_alpha(th: dict, layout: Layout, refs: tuple) -> Any:
    """``alpha`` at the covariates' 0 and the marker's reference, of the
    internal ``th`` (one cause's)."""
    c, zbar, r = refs
    zeta = anp.reshape(th["zeta"], (layout.n_links,))
    return th["alpha"] * anp.exp(
        (
            anp.dot(th["gamma"], np.asarray(zbar, dtype=float))
            - _reference_shift(zeta, layout, r, c)
        )
        / _per_scale(layout, th["beta"])
    )


def _reported_causes(theta: dict, layout: Layout, refs: tuple) -> Any:
    """The survival part of the reported parameters with several causes
    (in the order of :attr:`Layout.cause_slots`)."""
    cols = np.flatnonzero(layout.hazard)
    each = []
    for th in cause_thetas(theta, layout):
        zeta = anp.reshape(th["zeta"], (layout.n_links,))
        each.append(
            [_reported_alpha(th, layout, refs), th["beta"]]
            + [th["gamma"][k] for k in cols]
            + [zeta[j] for j in range(layout.n_links)]
        )
    slots = layout.cause_slots
    out = []
    for j in range(layout.n_survival):
        k, col = np.argwhere(slots == j)[0]
        out.append(each[k][col])
    return anp.stack(out)


def _internal_causes(block: Any, layout: Layout, refs: tuple) -> dict:
    """``alpha``, ``beta``, ``gamma`` and ``zeta`` of every cause from
    the survival part ``block`` of the reported parameters, the inverse
    of :func:`_reported_causes`."""
    c, zbar, r = refs
    zbar = np.asarray(zbar, dtype=float)
    slots = layout.cause_slots
    n_hazard = layout.n_hazard
    alphas, betas, gammas, zetas = [], [], [], []
    for k in range(layout.n_causes):
        v = block[slots[k]]
        gamma = layout._gamma(v[2 : 2 + n_hazard])
        zeta = v[2 + n_hazard :]
        alphas.append(
            v[0]
            * anp.exp(
                -(anp.dot(gamma, zbar) - _reference_shift(zeta, layout, r, c))
                / _per_scale(layout, v[1])
            )
        )
        betas.append(v[1])
        gammas.append(gamma)
        zetas.append(zeta)
    return {
        "alpha": anp.stack(alphas),
        "beta": anp.stack(betas),
        "gamma": anp.stack(gammas),
        "zeta": anp.stack(zetas),
    }


def reported_params(theta: dict, layout: Layout, refs: tuple) -> Any:
    """The reported parameters (of the estimated covariates) of the
    internal ``theta``; ``refs`` is ``(c, zbar, r)``: the marker's centre
    in the internal hazard, the covariate means, and the marker value
    at which ``alpha`` is reported. Differentiable by autograd."""
    c, zbar, r = refs
    zbar = np.asarray(zbar, dtype=float)
    P, p, q = layout.P, layout.p, layout.q
    parts = []
    for j in range(P):
        shift = theta["Delta"][j]
        parts.append(anp.reshape(theta["mu"][j] - anp.dot(shift, zbar), (1,)))
        kept = np.flatnonzero(layout.shifts[j])
        if kept.size == p:
            parts.append(anp.reshape(shift, (p,)))
        elif kept.size:
            parts.append(anp.stack([shift[k] for k in kept]))
    same = layout.correlated
    if layout.loaded is not None:
        # the spread of the model with the parameter held as a multiple
        # of another varying freely: its varying parameters' order
        D = _loaded_spread(theta, layout)
        q = D.shape[0]
        labels = np.asarray(layout.full_blocks, dtype=int)
        same = labels[:, None] == labels[None, :]
    elif q:
        L = theta["L"]
        D = anp.dot(L, anp.transpose(L))
    if q:
        sd = anp.sqrt(anp.stack([D[i, i] for i in range(q)]))
        parts.append(sd)
        corr = [
            D[i, j] / (sd[i] * sd[j])
            for i in range(q)
            for j in range(i + 1, q)
            if same[i, j]
        ]
        if corr:
            parts.append(anp.stack(corr))
    if layout.n_causes > 1:
        parts.append(anp.reshape(theta["sigma"], (layout.n_markers,)))
        parts.append(_reported_causes(theta, layout, refs))
        return anp.concatenate(parts)
    beta, gamma = theta["beta"], theta["gamma"]
    zeta = anp.reshape(theta["zeta"], (layout.n_links,))
    alpha = theta["alpha"] * anp.exp(
        (anp.dot(gamma, zbar) - _reference_shift(zeta, layout, r, c))
        / _per_scale(layout, beta)
    )
    if layout.n_markers > 1:
        parts.append(anp.reshape(theta["sigma"], (layout.n_markers,)))
        parts.append(anp.stack([alpha, beta]))
    else:
        parts.append(anp.stack([theta["sigma"], alpha, beta]))
    if layout.hazard.all():
        parts.append(anp.reshape(gamma, (p,)))
    elif layout.hazard.any():
        parts.append(
            anp.stack([gamma[k] for k in np.flatnonzero(layout.hazard)])
        )
    parts.append(zeta)
    return anp.concatenate(parts)


def _pair(name: str, varying: list) -> tuple:
    """The two varying parameters of the correlation ``name``
    (``corr_<a>_<b>``); empty where it is not one of theirs."""
    for i, a in enumerate(varying):
        for b in varying[i + 1 :]:
            if name == f"corr_{a}_{b}":
                return (a, b)
    return ()


def _loaded_spread(theta: dict, layout: Layout) -> Any:
    """The items' spread ``D`` of a layout with a parameter held as a
    multiple of a varying one (:attr:`Layout.loaded`), in the order of
    :attr:`Layout.full_varying`: the varying ones' ``L L'``, and the
    held one's row ``lam`` times that one's (its variance ``lam^2``
    times that one's). Its rank is one short; differentiable by
    autograd."""
    L = theta["L"]
    q = layout.q
    D = anp.dot(L, anp.transpose(L)) if q else anp.zeros((0, 0))
    _, place, at, _ = layout.loaded
    lam = theta["lam"]
    row = [lam * D[place, c] for c in range(q)]
    full_rows = []
    for r in range(q + 1):
        if r == at:
            full_rows.append(
                anp.stack(row[:at] + [lam * lam * D[place, place]] + row[at:])
            )
            continue
        own = r if r < at else r - 1
        entries = [D[own, c] for c in range(q)]
        full_rows.append(
            anp.stack(entries[:at] + [lam * D[own, place]] + entries[at:])
        )
    return anp.stack(full_rows)


def _per_scale(layout: Layout, beta: Any) -> Any:
    """What a shift of the log hazard (proportional hazards) or of the
    log ageing rate (accelerated) is divided by to move ``log alpha``:
    ``beta`` and 1."""
    return 1.0 if layout.survival == "AFT" else beta


def _reference_shift(zeta: Any, layout: Layout, r: float, c: float) -> Any:
    """The log hazard (the log ageing rate, accelerated) between the
    marker's reference ``r`` and the centre ``c`` of the internal hazard:
    the value's effect times ``r - c``, and 0 where the hazard does not
    follow the value. With several markers ``r`` and ``c`` are those of
    each link's term (0 but for a value's)."""
    if np.ndim(r) or np.ndim(c):
        diff = np.asarray(r, dtype=float) - np.asarray(c, dtype=float)
        return anp.sum(zeta * diff)
    if "value" not in layout.links:
        return 0.0
    return zeta[layout.links.index("value")] * (r - c)


def internal_theta(params: Any, layout: Layout, refs: tuple) -> dict:
    """The internal parameters of the reported ``params`` (of the
    estimated covariates), the inverse of :func:`reported_params`;
    differentiable by autograd."""
    c, zbar, r = refs
    zbar = np.asarray(zbar, dtype=float)
    P, p, q = layout.P, layout.p, layout.q
    i = 0
    mus, shifts = [], []
    for j in range(P):
        n = int(layout.shifts[j].sum())
        shift = scatter(params[i + 1 : i + 1 + n], layout.shifts[j])
        mus.append(params[i] + anp.dot(shift, zbar))
        shifts.append(shift)
        i += 1 + n
    sd = params[i : i + q + layout.n_loaded]
    i += q + layout.n_loaded
    lam = None
    if layout.loaded is not None:
        # every sd of the model with the parameter held as a multiple of
        # another varying freely, but no correlation of it: it is +-1
        # with that one (``sign``), and those with the others follow
        _, place, at, sign = layout.loaded
        held = sd[at]
        sd = anp.concatenate([sd[:at], sd[at + 1 :]])
        lam = sign * held / sd[place]
    same = layout.correlated
    n_corr = len(layout.lower)
    corr = params[i : i + n_corr]
    i += n_corr
    if q:
        rows = []
        k = 0
        upper = {}
        for a in range(q):
            for b in range(a + 1, q):
                if same[a, b]:
                    upper[(a, b)] = corr[k]
                    k += 1
        for a in range(q):
            row = []
            for b in range(q):
                if a == b:
                    row.append(sd[a] ** 2)
                elif same[a, b]:
                    rho = upper[(min(a, b), max(a, b))]
                    row.append(rho * sd[a] * sd[b])
                else:
                    # uncorrelated: 0 between the blocks
                    row.append(0.0 * sd[a] * sd[b])
            rows.append(anp.stack(row))
        L = _cholesky(rows, q)
    else:
        L = anp.zeros((0, 0))
    sigma = (
        params[i]
        if layout.n_markers == 1
        else params[i : i + layout.n_markers]
    )
    i += layout.n_markers - 1
    if layout.n_causes > 1:
        return {
            **({} if lam is None else {"lam": lam}),
            "mu": anp.stack(mus),
            "Delta": anp.stack(shifts) if p else anp.zeros((P, 0)),
            "L": L,
            "sigma": sigma,
            **_internal_causes(
                params[i + 1 : i + 1 + layout.n_survival], layout, refs
            ),
        }
    alpha_r, beta = params[i + 1], params[i + 2]
    n_hazard = layout.n_hazard
    gamma = layout._gamma(params[i + 3 : i + 3 + n_hazard])
    zeta = params[i + 3 + n_hazard : i + 3 + n_hazard + layout.n_links]
    return {
        **({} if lam is None else {"lam": lam}),
        "mu": anp.stack(mus),
        "Delta": anp.stack(shifts) if p else anp.zeros((P, 0)),
        "L": L,
        "sigma": sigma,
        "alpha": alpha_r
        * anp.exp(
            -(anp.dot(gamma, zbar) - _reference_shift(zeta, layout, r, c))
            / _per_scale(layout, beta)
        ),
        "beta": beta,
        "gamma": gamma,
        "zeta": zeta,
    }


def _saved_path(path: Any) -> tuple:
    """``(key, links)``: a built-in path model by its name (a spline by
    its dict), on links (``LinkedPathModel``) with them; a path of one's
    own is refused (``NotImplementedError``)."""
    from surpyval.degradation.path_models import PATH_MODELS, path_model_key
    from surpyval.degradation.stress import LinkedPathModel

    links = None
    if isinstance(path, LinkedPathModel):
        links = dict(path.links)
        path = path.base
    key = path_model_key(path)
    # by its class: an unpickled built-in path is a copy of it
    if not isinstance(key, dict) and type(PATH_MODELS.get(key)) is not type(
        path
    ):
        raise NotImplementedError(
            f"The {path.name!r} path model is not a built-in "
            "one, so it cannot be rebuilt from a name: keep the model "
            "with pickle."
        )
    return key, links


def _restored_path(key: Any, links: Any) -> Any:
    """The path model :func:`_saved_path` saved."""
    from surpyval.degradation.path_models import PATH_MODELS, get_path_model

    if not isinstance(key, dict) and key not in PATH_MODELS:
        raise ValueError(
            f"Unknown path model {key!r}; the built-in ones are "
            f"{sorted(PATH_MODELS)}, and a spline is saved as its "
            "SplinePath.to_dict()."
        )
    path = get_path_model(key)
    if links:
        from surpyval.degradation.stress import LinkedPathModel, validate_links

        path = LinkedPathModel(path, validate_links(path, dict(links)))
    return path


def _cholesky(rows: list, q: int) -> Any:
    """The lower Cholesky factor of the ``q x q`` matrix of ``rows``, by
    its recurrence (autograd's ``cholesky`` has no forward-mode
    derivative, which ``cb`` takes)."""
    L: list = [[0.0] * q for _ in range(q)]
    for i in range(q):
        for j in range(i + 1):
            acc = rows[i][j]
            for k in range(j):
                acc = acc - L[i][k] * L[j][k]
            L[i][j] = anp.sqrt(acc) if i == j else acc / L[j][j]
    zero = 0.0 * rows[0][0]
    return anp.stack(
        [
            anp.stack([L[i][j] if j <= i else zero for j in range(q)])
            for i in range(q)
        ]
    )


def structure_words(random_effects: Any) -> str:
    """The covariance structure of the varying parameters in words, for
    the printouts and the summary's heading: ``""`` for the full one,
    ``"uncorrelated"``, or ``"correlated in blocks {a, b}, {c}"``."""
    if random_effects == "full":
        return ""
    if random_effects == "diagonal":
        return "uncorrelated"
    blocks = ", ".join(
        "{" + ", ".join(block) + "}" for block in random_effects
    )
    return f"correlated in blocks {blocks}"


def _structure_words(random_effects: Any) -> str:
    """:func:`structure_words` in brackets after a space, or nothing."""
    words = structure_words(random_effects)
    return f" ({words})" if words else ""


def _saved_structure(random_effects: Any) -> Any:
    """``random_effects`` in JSON: its name, or its blocks as lists."""
    if isinstance(random_effects, str):
        return random_effects
    return [list(block) for block in random_effects]


def _in_brackets(label: str) -> str:
    """``label`` as a factor of a compound unit: in brackets where it is
    more than one word or symbol (``"(mm/s)"``, ``"(marker unit)"``)."""
    return f"({label})" if any(ch in label for ch in " /*·^") else label


def _finite_number(name: str, value: Any) -> float:
    try:
        out = float(value) if np.ndim(value) == 0 else None
    except (TypeError, ValueError):
        out = None
    if out is None or not np.isfinite(out):
        raise ValueError(f"'{name}' must be one finite number; got {value!r}.")
    return out


class JointModel(InformationCriteriaMixin, SerialisableMixin):
    """
    A fitted joint model of a marker and a lifetime, returned by the
    :func:`~surpyval.univariate.joint.joint_ph.JointPH` fitters, such as
    ``WeibullJointPH``, and the
    :func:`~surpyval.univariate.joint.joint_aft.JointAFT` fitters, such as
    ``WeibullJointAFT`` (``survival`` says which).

    Each item's marker follows the degradation path model (``LinearPath``
    for ``WeibullJointPH``; any ``PathModel``) with parameters of its
    own: those that vary between items (``varying``) are multivariate
    normal about a mean that the covariates shift, the others are common
    to all items. Each reading is the path plus normal noise; the hazard
    is Weibull proportional hazards in the path's current, noise-free
    value and the covariates:

    .. math::
        y_{ij} = m(t_{ij}; \\theta_i) + \\varepsilon_{ij}, \\qquad
        h_i(t) = \\frac{\\beta}{\\alpha}\\left(\\frac{t}{\\alpha}
        \\right)^{\\beta - 1} \\exp\\left(\\gamma' z_i + \\zeta
        \\left(m(t; \\theta_i) - r\\right)\\right),

    so ``alpha`` and ``beta`` are the baseline of an item with its
    marker at ``marker_reference`` (:math:`r`; the mean reading of the
    data unless the fit was given one) and its covariates at 0. With
    another ``link`` the hazard follows the path's slope
    (``slope_effect`` times :math:`m'(t)`) or the area under it from age
    0 (``cumulative_effect`` times :math:`\\int_0^t m(s) ds`), in place
    of the value or beside it; the baseline is then at a slope and an
    area of 0. With a ``lag`` above 0 the hazard at age :math:`t` follows
    each of these at :math:`\\max(t - \\mathrm{lag}, 0)` (the path held at
    its age-0 value until the lag has passed, so the slope is 0 there):
    a reading acts ``lag`` later. With a ``weight`` the area counts
    older exposure for less, :math:`\\int_0^t w(t - s) m(s) ds` with
    :math:`w(u) = e^{-u/\\mathrm{scale}}` (``("exponential", scale)``)
    or :math:`w(u) = 1` for :math:`u` up to the width and 0 beyond
    (``("window", width)``).

    A model fitted to failures with their modes (event types; ``causes``
    lists them) has a hazard of this form for each failure mode
    :math:`k`, with its own :math:`\\alpha_k`, :math:`\\beta_k`,
    :math:`\\gamma_k` and :math:`\\zeta_k` (named ``"<cause>: alpha"``
    and so on; those in ``shared`` are common to the modes), and one
    marker part. An item fails by the first mode to occur, so its
    survival, which ``sf``, ``cb`` and ``predict`` give, is that of every
    mode together, :math:`\\exp(-\\sum_k H_{ik}(t))`.

    In an accelerated model (``survival == "AFT"``) the marker speeds up
    the item's ageing in place of multiplying its hazard: by age
    :math:`t` the item has used up the effective age

    .. math::
        A_i(t) = \\int_0^t \\exp\\left(\\gamma' z_i + \\zeta
        \\left(m(s; \\theta_i) - r\\right)\\right) ds, \\qquad
        S_i(t) = \\exp\\left(-\\left(A_i(t) / \\alpha\\right)^\\beta
        \\right),

    so time at a high reading uses up life faster, and that life is not
    given back when the reading falls. The parameters have the same names;
    an effect is then the log of how many times faster the item ages per
    unit of the marker or the covariate (with the slope link,
    ``slope_effect`` times :math:`m'(s)` in the exponent), and with a
    marker that never changes the model is the proportional hazards one
    with every effect divided by ``beta``.

    ``params`` holds, in the order of ``parameter_names``: each path
    parameter's mean (by the path model's own names: ``a``, ``b``, ...)
    followed by its covariates' shifts (``a_<cov>``, ...); the spread of
    the varying ones between items (``sd_a``, ``sd_b``, ...,
    ``corr_a_b``, ... of each pair correlated: every pair unless the
    fitter was given ``random_effects``); ``sd_noise``; the Weibull
    ``alpha`` and ``beta``; the hazard's covariate coefficients
    (``<cov>``); and ``marker_effect`` (:math:`\\zeta`, the change in the
    log hazard per unit of the marker; ``slope_effect`` and
    ``cumulative_effect`` for those links). An aliased covariate's
    entries are ``nan`` (``aliased``). ``neg_ll()`` is the negative
    log-likelihood of the readings and the lifetimes together (so
    ``aic()`` and ``bic()`` compare joint models of the same data).

    ``sf(x, Z, markers=..., given=...)`` predicts for one item: its
    covariates ``Z``, its readings so far and the age it was last seen
    running; ``sf(x, item)`` takes them from an ``ItemLifetime``.

    A model fitted with ``how="two-stage"`` (``how``) has the same
    parameters and predictions, from the approximate two-stage estimate:
    the paths fitted to the readings alone, then the survival part with
    each item's predicted path taken as known. Its estimates are biased
    (most where the readings are few or the path curves); its survival
    part's standard errors (and so the bounds of its predictions) are too
    small, and its marker part has none (``nan``; the bounds hold it at
    its estimates). It has no likelihood of the whole model, so
    ``neg_ll()``, ``aic()``, ``aic_c()`` and ``bic()`` are refused. The
    printout and the summary say so.

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
    ...     readings = start + rate * ages + rng.normal(0, 0.2, ages.size)
    ...     outcome = "observed" if life < 120 else "right_censored"
    ...     items.append(
    ...         ItemLifetime(k, **{outcome: end},
    ...                      markers=dict(zip(ages, readings)))
    ...     )
    >>> model = WeibullJointPH.fit(SurpyvalData.from_items(items))
    >>> model.parameter_names[:4]
    ['a', 'b', 'sd_a', 'sd_b']
    >>> model.sf([30, 40, 80], markers={0: 2.5, 30: 3.4}, given=40).shape
    (3,)
    """

    #: The baseline distribution (``Weibull``) and the marker's path model.
    dist: Any
    path_model: Any
    #: The markers' names in a model of several markers (``path_model``
    #: is then their :class:`~._multi.MarkerSet`, ``link`` holds
    #: ``(marker, link)`` pairs, ``marker_reference`` and
    #: ``marker_unit`` are dicts by marker); ``None`` for one marker.
    markers: list | None = None
    #: The names of the path parameters that vary between items.
    varying: list[str]
    #: The fitted parameters, in the order of ``parameter_names``.
    params: npt.NDArray
    parameter_names: list[str]
    #: The covariates' names (``coef_0``, ... where the data did not name
    #: them), and the columns whose coefficients the data cannot
    #: determine (``nan`` in ``params``).
    feature_names: list[str]
    aliased: npt.NDArray
    #: The covariates that shift each path parameter's mean, ``{path
    #: parameter: [covariate, ...]}``, and those in the hazard (every
    #: covariate in both, unless the fit was given ``path_covariates`` or
    #: ``hazard_covariates``).
    path_covariates: dict[str, list[str]] | None = None
    hazard_covariates: list[str] | None = None
    #: What of the marker's path the hazard follows: ``"value"`` (its
    #: current value), ``"slope"`` (its rate of change), ``"cumulative"``
    #: (the area under it from age 0), one or more.
    link: tuple = ("value",)
    #: How long after the marker the hazard follows it, in the lifetimes'
    #: unit of time (0: at once): the hazard at age ``t`` follows the
    #: path at ``max(t - lag, 0)``. Fixed by the fit's fitter, not
    #: estimated.
    lag: float = 0.0
    #: How the accumulated exposure (the ``"cumulative"`` link) is
    #: weighted by how long ago it was: ``None`` (all of it equally),
    #: ``("exponential", scale)`` (it fades as ``exp(-u / scale)``, ``u``
    #: the time since) or ``("window", width)`` (only the last ``width``
    #: counts). Fixed by the fit's fitter, not estimated.
    weight: tuple | None = None
    #: The survival part: ``"PH"`` (the marker multiplies the hazard,
    #: ``JointPH``) or ``"AFT"`` (it speeds up the item's ageing,
    #: ``JointAFT``).
    survival: str = "PH"
    #: The failure modes (causes) of a model fitted to failures with
    #: event types (``e``), in the order of its parameters: each has a
    #: Weibull baseline, hazard coefficients and effects of its own,
    #: named ``"<cause>: <name>"``. ``None`` for a model of one kind of
    #: failure.
    causes: list | None = None
    #: The survival parameters the causes have in common (the fit's
    #: ``shared``): ``"beta"``, hazard covariates and effects, each one
    #: parameter under its plain name.
    shared: tuple = ()
    #: The failures of each cause, ``{cause: count}`` (count weighted);
    #: ``None`` for a model of one kind of failure.
    n_events_by_cause: dict | None = None
    #: The marker value at which ``alpha`` is reported (``None`` where
    #: the hazard does not follow the value).
    marker_reference: Any
    #: The number of estimated parameters, the ``k`` of the criteria.
    k: int
    #: What the fit reached (``surpyval.utils.no_maximum``): ``"not
    #: applicable"`` for a two-stage fit, which maximises no likelihood
    #: of the whole model.
    maximum: str = "unknown"
    #: How it was fitted: ``"MLE"`` (the joint likelihood's maximum) or
    #: ``"two-stage"`` (the approximate two-stage estimate).
    how: str = "MLE"
    #: How each item's likelihood was integrated over its varying path
    #: parameters, in the fit and in this model's predictions: ``"grid"``
    #: (adaptive Gauss-Hermite on the full grid of 7 points in each),
    #: ``"pruned"`` (the same grid without the nodes of negligible
    #: weight) or ``"laplace"`` (Laplace's approximation, with no grid:
    #: the default beyond five varying parameters; its predictions are
    #: on the grid up to five).
    integration: str = "grid"
    #: Which varying path parameters are correlated between items:
    #: ``"full"`` (every pair), ``"diagonal"`` (none) or a tuple of blocks
    #: of names, those in one block correlated with each other and with
    #: none outside it (the fitter's ``random_effects``). A correlation
    #: outside the blocks is 0, not estimated: it has no entry in
    #: ``params``.
    random_effects_structure: Any = "full"
    #: The numbers of items, failures and readings fitted (count
    #: weighted).
    n_items: float = 0.0
    n_events: float = 0.0
    n_readings: float = 0.0
    #: The unit the marker is read in (a label, as ``unit`` is the
    #: ages'): from the readings' ``Measurements`` or ``marker_unit=`` at
    #: the fit; None is unknown. With ``unit`` it names the unit of each
    #: link's effect (``effect_units``).
    marker_unit: Any = None
    #: The marker's centre in the internal hazard (the mean reading) and
    #: the covariate means: predictions are computed there.
    _centre: Any = 0.0
    _zbar: npt.NDArray
    _covariance: npt.NDArray | None = None
    #: ``(k, j)``: the varying parameter ``k`` the fit holds as an exact
    #: multiple of the varying ``j`` about their means (a correlation of
    #: +-1 between them, :attr:`perfect_correlations`); None: none.
    _proportional: tuple | None = None
    _neg_ll: float
    #: The path part of the internal estimates (``mu``, ``Delta``, ``L``,
    #: ``sigma``), a start for another link's fit of the same data
    #: (``compare_links``); not saved.
    _path_start: dict | None = None

    @property
    def kind(self) -> str:
        return "Joint (Weibull {}, {})".format(
            _SURVIVAL_WORDS[self.survival],
            ", ".join(link_words(link) for link in self.link),
        )

    @property
    def effect_units(self) -> dict[str, str]:
        """
        What each link's effect is per, by its name: ``marker_effect`` per
        unit of the marker, ``slope_effect`` per unit of the marker per
        unit of age, and ``cumulative_effect`` per unit of the marker
        times a unit of age (the area under the path). The labels are
        ``marker_unit`` and ``unit``, as given; one not given is
        ``"marker unit"`` or ``"time unit"``. A label of more than one
        word or symbol is put in brackets in a compound unit.

        Examples
        --------
        >>> from surpyval.univariate.joint import JointModel
        >>> model = JointModel()
        >>> model.link = ("value", "cumulative")
        >>> model.marker_unit, model.unit = "mm/s", "years"
        >>> model.effect_units
        {'marker_effect': 'mm/s', 'cumulative_effect': '(mm/s)·years'}
        >>> model.unit = None
        >>> model.effect_units["cumulative_effect"]
        '(mm/s)·(time unit)'
        """
        age = getattr(self, "unit", None) or "time unit"
        if self.markers is not None:
            units = dict(self.marker_unit or {})
            out = {}
            for name, link in self.link:
                marker = units.get(name) or "marker unit"
                per = {
                    "value": marker,
                    "slope": f"{_in_brackets(marker)}/{_in_brackets(age)}",
                    "cumulative": (
                        f"{_in_brackets(marker)}·{_in_brackets(age)}"
                    ),
                }
                out[effect_name((name, link))] = per[link]
            return out
        marker = self.marker_unit or "marker unit"
        per = {
            "value": marker,
            "slope": f"{_in_brackets(marker)}/{_in_brackets(age)}",
            "cumulative": f"{_in_brackets(marker)}·{_in_brackets(age)}",
        }
        return {EFFECTS[link]: per[link] for link in self.link}

    # -- the parts -------------------------------------------------------

    @property
    def _kept(self) -> npt.NDArray:
        """The estimated covariate columns."""
        p = len(self.feature_names)
        return np.setdiff1d(np.arange(p), np.asarray(self.aliased, int))

    def _covariate_parts(self) -> tuple[dict[str, list[str]], list[str]]:
        """``(shifts, hazard)`` of every covariate (aliased ones too)."""
        names = list(self.path_model.parameter_names)
        shifts = self.path_covariates
        if shifts is None:
            shifts = {name: list(self.feature_names) for name in names}
        hazard = self.hazard_covariates
        if hazard is None:
            hazard = list(self.feature_names)
        return shifts, hazard

    @property
    def spreads_at_zero(self) -> list[str]:
        """The varying parameters whose spread between items the fit
        holds at its lower bound, 0: their ``sd_<name>`` is exactly 0
        and they are, in effect, common to all items. The fit is then
        the one with them common (its likelihood, predictions and the
        other parameters' standard errors are that fit's); their sds
        and correlations (0) have no standard errors. Compare this fit
        with the one where they are common by ``aic()``, or by
        :meth:`likelihood_ratio_test`."""
        # Only the fit's holding on the boundary (joint_ph's
        # _held_on_the_boundary) writes an sd of exactly 0; a search
        # ends at an sd above 0 (exp of its log), so an exact 0 marks it,
        # in a fitted model and in one loaded from its dict.
        names = list(self.parameter_names)
        params = np.asarray(self.params, dtype=float)
        return [
            v
            for v in self.varying
            if f"sd_{v}" in names and params[names.index(f"sd_{v}")] == 0.0
        ]

    @property
    def perfect_correlations(self) -> list[str]:
        """The correlations between two varying parameters that the fit
        holds at +1 or -1, the edge of the valid ones (as
        ``corr_<a>_<b>``): one parameter's spread between items is then
        an exact multiple of the other's, each item's departure from the
        mean in it that multiple of its departure in the other. The fit
        is the one with that one integrated over and the other a fixed
        multiple of it; the correlation has no standard error, and the
        correlations of the held parameter with the others follow from
        the other's. Empty where none is held."""
        if self._proportional is None:
            return []
        k, j = self._proportional
        varying = list(self.varying)
        a, b = sorted((k, j), key=varying.index)
        return [f"corr_{a}_{b}"]

    @property
    def _varying_fitted(self) -> list[str]:
        """The varying parameters with a spread above 0, those the
        likelihood integrates over."""
        held = set(self.spreads_at_zero)
        return [v for v in self.varying if v not in held]

    @property
    def _layout(self) -> Layout:
        names = list(self.path_model.parameter_names)
        kept = [self.feature_names[k] for k in self._kept]
        shifts, hazard = self._covariate_parts()
        varying = self._varying_fitted
        loaded = None
        if self._proportional is not None:
            # one varying parameter a multiple of another
            k, j = self._proportional
            full = list(varying)
            varying = [v for v in full if v != k]
            params = np.asarray(self.params, dtype=float)
            corr = params[
                list(self.parameter_names).index(self.perfect_correlations[0])
            ]
            loaded = (
                names.index(k),
                varying.index(j),
                full.index(k),
                float(np.sign(corr)),
            )
        return Layout(
            len(names),
            len(kept),
            tuple(names.index(v) for v in varying),
            np.array(
                [[cov in shifts.get(n, []) for cov in kept] for n in names],
                dtype=bool,
            ).reshape(len(names), len(kept)),
            np.array([cov in hazard for cov in kept], dtype=bool),
            tuple(self.link),
            self.survival,
            block_labels(self.random_effects_structure, varying),
            **self._cause_layout(kept),
            n_markers=1 if self.markers is None else len(self.markers),
            loaded=loaded,
        )

    def _cause_layout(self, kept: list) -> dict:
        """The layout's entries for the causes (none for a model of one
        kind of failure)."""
        if self.causes is None:
            return {}
        return {
            "n_causes": len(self.causes),
            "shared_beta": "beta" in self.shared,
            "shared_hazard": [cov in self.shared for cov in kept],
            "shared_links": [effect_name(k) in self.shared for k in self.link],
        }

    def _names(self) -> list[str]:
        """The parameters' names as the model's own description gives
        them (every covariate, aliased ones too)."""
        shifts, hazard = self._covariate_parts()
        return parameter_names(
            list(self.path_model.parameter_names),
            list(self.varying),
            list(self.feature_names),
            shifts,
            hazard,
            tuple(self.link),
            self.random_effects_structure,
            self.causes,
            self.shared,
            self.markers,
        )

    def _scale(self) -> float:
        """A typical age: the Weibull ``alpha`` (the smallest of the
        causes')."""
        names = list(self.parameter_names)
        if self.causes is None:
            return float(self.params[names.index("alpha")])
        return float(
            min(
                self.params[names.index(f"{cause}: alpha")]
                for cause in self.causes
            )
        )

    def _likelihood_model(self) -> Model:
        if self.markers is not None:
            return Model(
                self._layout,
                MarkerPaths(
                    self.path_model,
                    tuple(self.link),
                    self.lag,
                    self.weight,
                    self._centre,
                ),
                0.0,
                self.integration,
            )
        return Model(
            self._layout,
            PathFunction(
                self.path_model, tuple(self.link), self.lag, self.weight
            ),
            self._centre,
            self.integration,
        )

    @property
    def _estimated(self) -> npt.NDArray:
        """The entries of ``params`` that are estimated (not aliased,
        and not the spread of a parameter held at 0,
        :attr:`spreads_at_zero`)."""
        out = ~np.isnan(np.asarray(self.params, dtype=float))
        held = set(self.spreads_at_zero)
        if held:
            varying = list(self.varying)
            label = block_labels(self.random_effects_structure, varying)
            dropped = {f"sd_{v}" for v in held} | {
                f"corr_{a}_{b}"
                for i, a in enumerate(varying)
                for j, b in enumerate(varying)
                if j > i and label[i] == label[j] and {a, b} & held
            }
            for k, name in enumerate(self.parameter_names):
                if name in dropped:
                    out[k] = False
        if self._proportional is not None:
            # the held parameter's correlations: +-1 with the one it is a
            # multiple of, and those with the others follow from its
            held_k = self._proportional[0]
            for k, name in enumerate(self.parameter_names):
                if name.startswith("corr_") and held_k in _pair(
                    name, list(self.varying)
                ):
                    out[k] = False
        return out

    @property
    def _refs(self) -> tuple:
        if self.markers is not None:
            reference = dict(self.marker_reference or {})
            centres = list(self._centre)
            names = list(self.markers)
            c = np.array(
                [
                    centres[names.index(m)] if k == "value" else 0.0
                    for m, k in self.link
                ]
            )
            r = np.array(
                [
                    (
                        reference.get(m, centres[names.index(m)])
                        if k == "value"
                        else 0.0
                    )
                    for m, k in self.link
                ],
                dtype=float,
            )
            return (c, self._zbar, r)
        reference = self.marker_reference
        return (
            self._centre,
            self._zbar,
            self._centre if reference is None else reference,
        )

    def _theta(self, params: Any = None) -> dict:
        own = params is None
        if own:
            self._check_edges()
            params = np.asarray(self.params, dtype=float)[self._estimated]
        with np.errstate(all="ignore"):
            theta = internal_theta(params, self._layout, self._refs)
        if own:
            self._check_resolved(theta)
        if own and not np.all(np.isfinite(np.asarray(theta["L"], float))):
            # (a fit always gives a valid spread; a dict edited by hand,
            # or written elsewhere, may not)
            names = [
                n
                for n in self.parameter_names
                if n.startswith(("sd_", "corr_"))
            ]
            raise ValueError(
                "The model's spread between items ("
                + ", ".join(names)
                + ") is not a valid covariance matrix: its correlations "
                "are not positive definite (or an sd is not above 0). Each "
                "correlation must be between -1 and 1 and together they "
                "must be consistent; check the values in 'params' (a "
                "correlation of exactly +-1 is a fit held on its edge, "
                "saved with 'proportional')."
            )
        return theta

    def _check_edges(self) -> None:
        """Refuse a correlation of exactly +-1 that the model does not
        hold (``perfect_correlations``): a fit on that edge saves which
        parameter is the multiple of which; without it the spread is
        singular and the likelihood not defined."""
        held = set(self.perfect_correlations)
        names = list(self.parameter_names)
        params = np.asarray(self.params, dtype=float)
        for name in names:
            if name.startswith("corr_") and name not in held:
                value = params[names.index(name)]
                if np.isfinite(value) and abs(value) == 1.0:
                    raise ValueError(
                        f"{name} is exactly {value:+g}, a singular spread, "
                        "and the model does not hold it there (no "
                        "'proportional' in its dict): its likelihood is not "
                        "defined. A fit on that edge is saved with "
                        "'proportional' naming the "
                        "parameter held as a multiple of the other; otherwise "
                        "give a correlation strictly between -1 and 1."
                    )

    def _check_resolved(self, theta: dict) -> None:
        """Refuse a spread above 0 but too small for the integration to
        resolve, by the same rule as a fit's search
        (``joint_ph._resolved_spreads``, its first part): a varying
        parameter's own spread (its Cholesky diagonal) below 1e-9 of its
        mean value (``mu``, at the covariates' means). There the nodes'
        offsets are lost to rounding and the likelihood is wrong; a fit
        never reports one (its search does not step there, and a spread
        at 0 is saved as exactly 0)."""
        layout = self._layout
        L = np.asarray(theta["L"], dtype=float)
        if not L.size:
            return
        mu = np.asarray(theta["mu"], dtype=float)
        names = list(self.path_model.parameter_names)
        for k, j in enumerate(layout.varying):
            own, value = L[k, k], mu[j]
            if np.isfinite(own) and 0.0 < own < 1e-9 * abs(value):
                v = names[j]
                raise ValueError(
                    f"The spread of {v} of its own is {own:.3g}, above 0 but "
                    f"below 1e-9 of {v}'s mean ({value:.6g}): too small for "
                    "the integration over the items to resolve (its nodes "
                    "would sit within rounding of each other). A spread that "
                    f"small is 0: set sd_{v} (and its correlations) to 0 for "
                    f"the model with {v} common to all items."
                )

    def covariance(self) -> npt.NDArray:
        """The parameters' covariance, in the order of
        ``parameter_names``: the inverse of the observed information (the
        Hessian of the negative log-likelihood, by autograd) at the
        maximum, ``nan`` for an aliased coefficient. Raises a
        ``ValueError`` where the model has none."""
        if self._covariance is None:
            raise no_covariance_error()
        return np.array(self._covariance)

    def standard_errors(self) -> npt.NDArray:
        """The parameters' standard errors, the square roots of the
        diagonal of :meth:`covariance`, in the order of
        ``parameter_names``."""
        return standard_errors_of(self.covariance())

    def _refuse_two_stage(self, name: str) -> None:
        if self.how == "two-stage":
            raise ValueError(
                f"A two-stage fit has no {name}: its two stages (the paths "
                "from the readings, then the lifetimes with each item's "
                "predicted path taken as known) have no likelihood of the "
                "whole model. Compare it with the joint fit by the accuracy "
                "of their predictions (surpyval.metrics.score_predictions, "
                "dynamic_auc, dynamic_brier), or refit with how='MLE' for "
                "the likelihood and the criteria."
            )

    def neg_ll(self) -> float:
        """The negative log-likelihood of the readings and the lifetimes
        together, at the fitted parameters. Refused for a two-stage fit
        (``how="two-stage"``), which has no likelihood of the whole
        model."""
        self._refuse_two_stage("neg_ll()")
        return super().neg_ll()

    def _refuse_ic(self, name: str) -> None:
        self._refuse_two_stage(name)
        super()._refuse_ic(name)

    def likelihood_ratio_test(self, nested: JointModel) -> dict:
        """
        The likelihood-ratio test of this model against one nested in
        it, fitted to the same data: twice the gain in the
        log-likelihood, chi-squared on the difference in the number of
        parameters when the nested model holds. With failure modes, the
        fit with a parameter shared between the modes (``shared=``) is
        nested in the fit without: the test asks whether the modes
        differ in it, such as whether the marker speaks to one way of
        failing more than to another.

        The fit with fewer varying parameters (``varying`` a part of
        this one's, nothing else different) is nested too: it is this
        model with their spreads 0, on the lower bound of a variance.
        Where one parameter is left out, the statistic's reference is
        then half chi-squared on the correlations left out and half on
        them and its sd (Self and Liang, 1987; Stram and Lee, 1994): the
        p-value is the mean of the two. Where several are, chi-squared
        on the difference in parameters, which is conservative (its
        p-value too large). This is the test for a spread the fit holds
        at 0 (:attr:`spreads_at_zero`), which has no standard error.

        That ``nested`` is nested in this model is the caller's claim;
        what can be checked is: both are joint fits by maximum
        likelihood of the same numbers of items, failures and readings,
        with the same path and survival part, and ``nested`` has fewer
        parameters and no higher a log-likelihood.

        Parameters
        ----------
        nested : JointModel
            The simpler fit of the same data.

        Returns
        -------
        dict
            ``{"statistic", "df", "p_value", "log_likelihood",
            "log_likelihood_null"}``, the last the nested model's.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ItemLifetime, SurpyvalData, WeibullJointPH
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
        >>> data = SurpyvalData.from_items(items)
        >>> each = WeibullJointPH.fit(data)
        >>> one = WeibullJointPH.fit(data, shared=["marker_effect"])
        >>> test = each.likelihood_ratio_test(one)
        >>> test["df"], bool(0 <= test["p_value"] <= 1)
        (1, True)
        """
        from scipy.stats import chi2

        if not isinstance(nested, JointModel):
            raise ValueError(
                "'nested' is the simpler joint model fitted to the same "
                f"data; got {type(nested).__name__}."
            )
        for model in (self, nested):
            model._refuse_two_stage("likelihood-ratio test")
        same = (
            "n_items",
            "n_events",
            "n_readings",
            "survival",
            "link",
            "lag",
            "weight",
            "markers",
        )
        dropped = [v for v in self.varying if v not in nested.varying]
        fewer = bool(dropped) and [
            v for v in self.varying if v in nested.varying
        ] == list(nested.varying)
        differ = [
            name
            for name in same
            if getattr(self, name) != getattr(nested, name)
            # several markers: the nested model may follow fewer of
            # them (a marker's effect 0)
            and not (
                name == "link"
                and self.markers is not None
                and set(nested.link) < set(self.link)
            )
        ] + (
            ["path_model"]
            if type(self.path_model) is not type(nested.path_model)
            else []
        )
        if list(self.varying) != list(nested.varying) and not fewer:
            differ.append("varying")
        others = sorted(
            set(nested.parameter_names) ^ set(self.parameter_names)
        )
        others = [
            name
            for name in others
            if name not in self.parameter_names
            or not name.startswith(("sd_", "corr_"))
        ]
        held = self.perfect_correlations or nested.perfect_correlations
        if held and list(self.varying) == list(nested.varying):
            raise ValueError(
                "A likelihood-ratio test of a correlation held at +-1 "
                f"({held[0]}) against a free one is not taken: the two fits "
                "have the same parameters, and the null is on the edge of "
                "the valid correlations, where the statistic's reference is "
                "a mixture whose weights depend on the information matrix. "
                "The held fit is the free model's maximum, so its neg_ll() "
                "is the free model's."
            )
        if fewer and others:
            raise ValueError(
                "A likelihood-ratio test of fewer varying parameters "
                "compares two fits that differ only in their spread; "
                f"these also differ in {others!r}."
            )
        if differ:
            raise ValueError(
                "A likelihood-ratio test compares two fits of the same "
                "data with the same path, links and survival part; these "
                f"differ in {differ!r}."
            )
        df = int(self.k) - int(nested.k)
        if df < 1:
            raise ValueError(
                "'nested' is the simpler model, with fewer parameters "
                f"than this one; it has {nested.k} and this one {self.k}. "
                "Call the richer model's likelihood_ratio_test with the "
                "simpler one."
            )
        ll, ll_null = -self.neg_ll(), -nested.neg_ll()
        gain = ll - ll_null
        if gain < -1e-6 * max(1.0, abs(ll)):
            raise ValueError(
                "The nested model has a higher log-likelihood "
                f"({ll_null:.6g}) than this one ({ll:.6g}), so it is not "
                "nested in it or this fit did not reach its maximum "
                "(check both models' 'maximum')."
            )
        # a gain within round-off of 0 (two searches reaching one
        # maximum by different steps) is 0: chi-squared on 0 degrees of
        # freedom, in the boundary mixture below, is a point at 0, and a
        # statistic of 1e-13 would halve the p-value
        statistic = 2.0 * gain if gain > 1e-9 * max(1.0, abs(ll)) else 0.0
        p_value = float(chi2.sf(statistic, df))
        if fewer and len(dropped) == 1:
            # one spread on its lower bound, 0, under the null: half
            # chi-squared on its correlations, half on them and its sd
            # (Self and Liang 1987; Stram and Lee 1994)
            below = (
                float(chi2.sf(statistic, df - 1))
                if df > 1
                else float(statistic <= 0.0)
            )
            p_value = 0.5 * below + 0.5 * p_value
        return {
            "statistic": float(statistic),
            "df": df,
            "p_value": p_value,
            "log_likelihood": float(ll),
            "log_likelihood_null": float(ll_null),
        }

    def _ic_sample_size_from_data(self) -> float:
        return ic_sample_size(
            [0, 1], [self.n_events, self.n_items - self.n_events]
        )

    # -- the table -------------------------------------------------------

    def _parts(self) -> list[tuple[str, list[int]]]:
        """``(heading, positions in params)`` of each part."""
        names = list(self.path_model.parameter_names)
        shifts, hazard = self._covariate_parts()
        n_path = len(names) + sum(len(shifts.get(n, [])) for n in names)
        p = len(hazard)
        # every varying parameter's spread (one held at 0 too)
        label = block_labels(self.random_effects_structure, self.varying)
        n_spread = len(label) + sum(
            label[i] == label[j]
            for i in range(len(label))
            for j in range(i + 1, len(label))
        )
        path = list(range(n_path))
        spread = list(range(n_path, n_path + n_spread))
        i = spread[-1] + 1 if spread else n_path
        if self.markers is not None:
            return self._marker_parts(path, spread, i, p, hazard)
        at = {
            "value": "marker = {:.6g}".format(
                np.nan
                if self.marker_reference is None
                else self.marker_reference
            ),
            "slope": "slope = 0",
            "cumulative": "area = 0",
        }
        one, two = "", ""
        if self.how == "two-stage":
            one, two = f" ({_STAGE_ONE})", f" ({_STAGE_TWO})"
        if self.causes is not None:
            return [
                (
                    "path: {} ({})".format(
                        self.path_model.name, path_formula(self.path_model)
                    ),
                    path,
                ),
                (
                    "spread between items"
                    + _structure_words(self.random_effects_structure),
                    spread,
                ),
                ("noise", [i]),
                *self._cause_parts(
                    hazard,
                    ", ".join(at[link] for link in self.link),
                    ", ".join(_LINK_WORDS[link] for link in self.link)
                    + (f", lag {self.lag:g}" if self.lag > 0 else "")
                    + _weight_words(self.weight),
                ),
            ]
        return [
            (
                "path: {} ({}){}".format(
                    self.path_model.name, path_formula(self.path_model), one
                ),
                path,
            ),
            (
                "spread between items"
                + _structure_words(self.random_effects_structure)
                + one,
                spread,
            ),
            ("noise" + one, [i]),
            (
                "survival: {} {}, alpha at {}{}".format(
                    self.dist.name,
                    self.survival,
                    ", ".join(at[link] for link in self.link),
                    two,
                ),
                list(range(i + 1, i + 3 + p)),
            ),
            (
                "link: "
                + ", ".join(_LINK_WORDS[link] for link in self.link)
                + (f", lag {self.lag:g}" if self.lag > 0 else "")
                + _weight_words(self.weight)
                + two,
                list(range(i + 3 + p, i + 3 + p + len(self.link))),
            ),
        ]

    def _marker_parts(
        self, path: list, spread: list, i: int, p: int, hazard: list
    ) -> list[tuple[str, list[int]]]:
        """:meth:`_parts` of several markers: the path (each marker's in
        turn), the spread, each marker's noise, the survival part (alpha
        at each followed marker's reference) and the links."""
        one, two = "", ""
        if self.how == "two-stage":
            one, two = f" ({_STAGE_ONE})", f" ({_STAGE_TWO})"
        M = len(self.markers or [])
        reference = dict(self.marker_reference or {})
        at = []
        for name, link in self.link:
            if link == "value":
                at.append(f"{name} = {reference.get(name, np.nan):.6g}")
            elif link == "slope":
                at.append(f"{name} slope = 0")
            else:
                at.append(f"{name} area = 0")
        paths = "; ".join(
            f"{name}: {model.name} ({path_formula(model)})"
            for name, model in zip(self.markers or [], self.path_model.models)
        )
        noise = list(range(i, i + M))
        j = i + M
        links = (
            ", ".join(link_words(link) for link in self.link)
            + lag_words(self.lag, heading=True)
            + weight_words(self.weight)
        )
        if self.causes is not None:
            return [
                (f"paths: {paths}", path),
                (
                    "spread between items"
                    + _structure_words(self.random_effects_structure),
                    spread,
                ),
                ("noise", noise),
                *self._cause_parts(hazard, ", ".join(at), links),
            ]
        return [
            (f"paths: {paths}{one}", path),
            (
                "spread between items"
                + _structure_words(self.random_effects_structure)
                + one,
                spread,
            ),
            ("noise" + one, noise),
            (
                "survival: {} {}, alpha at {}{}".format(
                    self.dist.name, self.survival, ", ".join(at), two
                ),
                list(range(j, j + 2 + p)),
            ),
            (
                "links: "
                + ", ".join(link_words(link) for link in self.link)
                + lag_words(self.lag, heading=True)
                + weight_words(self.weight)
                + two,
                list(range(j + 2 + p, j + 2 + p + len(self.link))),
            ),
        ]

    def _cause_parts(self, hazard: list, at: str, links: str) -> list:
        """The survival part's headings and positions with several
        causes: each cause's baseline and coefficients, then its
        effects, then what the causes share. ``at`` says where ``alpha``
        is reported and ``links`` what the hazard follows."""
        names = list(self.parameter_names)
        effects = [effect_name(link) for link in self.link]
        out = []
        for cause in self.causes or []:
            own = [f"{cause}: {name}" for name in ["alpha", "beta", *hazard]]
            out.append(
                (
                    "survival, {}: {} {}, alpha at {}".format(
                        cause, self.dist.name, self.survival, at
                    ),
                    [names.index(name) for name in own if name in names],
                )
            )
            own = [f"{cause}: {name}" for name in effects]
            out.append(
                (
                    f"link, {cause}: {links}",
                    [names.index(name) for name in own if name in names],
                )
            )
        common = [
            name for name in ["beta", *hazard, *effects] if name in names
        ]
        out.append(
            (
                "shared by the failure modes",
                [names.index(name) for name in common],
            )
        )
        return out

    def _kinds(self, parts: list) -> tuple[set, set, set]:
        """The positions of the positive parameters, of those with a
        hazard ratio (coefficients and effects) and of the
        correlations."""
        n_sd = len(self.varying)
        positive = set(parts[2][1]) | set(parts[1][1][:n_sd])
        corr = set(parts[1][1][n_sd:])
        if self.causes is None:
            survival = parts[3][1]
            hazard = set(survival[2:]) | set(parts[4][1])
            return positive | set(survival[:2]), hazard, corr
        names = list(self.parameter_names)
        baseline = {
            j
            for j, name in enumerate(names)
            if name == "beta" or str(name).endswith((": alpha", ": beta"))
        }
        hazard = {j for _, where in parts[3:] for j in where} - baseline
        return positive | baseline, hazard, corr

    def summary(self, alpha_ci: float = 0.05) -> pd.DataFrame:
        """
        The parameter table: each parameter with its standard error, a
        two-sided ``1 - alpha_ci`` Wald interval and, for the
        coefficients, the Wald statistic ``z`` and its p-value. Indexed by
        ``(part, name)``, the parts headed by what they are: the path
        (its model and formula: its parameters' means and their
        covariates' shifts), the spread between items (``sd_<name>``,
        ``corr_<name>_<name>``; the heading says which are correlated
        where the fitter's ``random_effects`` was not ``"full"``), the
        noise (``sd_noise``), the survival
        part (the Weibull ``alpha`` -- the heading says at which marker
        value -- and ``beta``, and the covariate coefficients) and the
        link (``marker_effect``). ``exp(coef)`` is the hazard ratio of
        one unit of a covariate or of the marker; in an accelerated model
        (``survival == "AFT"``) it is how many times faster the item ages
        for that unit (its life divided by that). The intervals of the
        positive parameters (the sds, ``alpha``, ``beta``) are on the log
        scale and those of the correlations on Fisher's ``atanh`` scale,
        so each stays in its parameter's range. A two-stage fit
        (``how="two-stage"``) says so in each part's heading: the marker
        parts (stage 1) have no standard errors, and the survival part's
        and the link's (stage 2) are approximate and understated.

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
        >>> model.summary().index[0]
        ('path: Linear (a + b t)', 'a')
        """
        import pandas as pd

        from surpyval.univariate.regression._summary import (
            coefficient_table,
        )

        check_alpha_ci(alpha_ci)
        params = np.asarray(self.params, dtype=float)
        try:
            se = self.standard_errors()
        except ValueError:
            se = np.full(params.shape, np.nan)
        names = self.parameter_names
        q = norm.ppf(1 - alpha_ci / 2)
        level = f"{100 * (1 - alpha_ci):g}%"
        parts = self._parts()
        positive, hazard, corr = self._kinds(parts)
        rows, index = [], []
        for heading, positions in parts:
            for j in positions:
                row = coefficient_table(
                    [names[j]], [params[j]], [se[j]], alpha_ci, exp=j in hazard
                )
                with np.errstate(all="ignore"):
                    if j in positive:
                        half = q * se[j] / params[j]
                        row["coef lower " + level] = params[j] * np.exp(-half)
                        row["coef upper " + level] = params[j] * np.exp(half)
                        row["z"] = row["p"] = np.nan
                    elif j in corr:
                        half = q * se[j] / (1 - params[j] ** 2)
                        centre = np.arctanh(params[j])
                        row["coef lower " + level] = np.tanh(centre - half)
                        row["coef upper " + level] = np.tanh(centre + half)
                        row["z"] = row["p"] = np.nan
                rows.append(row)
                index.append((heading, names[j]))
        table = pd.concat(rows)
        table.index = pd.MultiIndex.from_tuples(index, names=["part", "name"])
        return table

    def __repr__(self) -> str:
        from surpyval.univariate.regression._summary import format_table

        if self.markers is not None:
            return self._marker_repr()
        followed = " and ".join(_LINK_WORDS[link] for link in self.link)
        followed += _weight_words(self.weight)
        if self.lag > 0:
            followed += f", {self.lag:g} earlier (lag)"
        if self.survival == "AFT":
            survival = (
                f"{self.dist.name} accelerated failure time, ageing sped "
                f"up by the marker's {followed}"
            )
        else:
            survival = (
                f"{self.dist.name} proportional hazards in the marker's "
                f"{followed}"
            )
        out = (
            "Joint SurPyval Model"
            "\n===================="
            f"\nMarker path         : {self.path_model.name}, "
            f"{path_formula(self.path_model)}; varying between items: "
            f"{', '.join(self.varying)}"
            f"{_structure_words(self.random_effects_structure)}"
            f"\nSurvival            : {survival}"
            f"\nData                : {self.n_items:g} items, "
            f"{self.n_events:g} failures, {self.n_readings:g} readings"
            f"{self._cause_line()}"
            f"\nFit                 : {self._fit_line()}"
            f"\nIntegration         : {_INTEGRATION_WORDS[self.integration]}"
            "\nEffect units        : "
        )
        out += "; ".join(
            f"{name} per {per}" for name, per in self.effect_units.items()
        )
        table = self.summary()
        columns = {
            "coef": "estimate",
            "se(coef)": "se",
            "coef lower 95%": "lower 95%",
            "coef upper 95%": "upper 95%",
        }
        for heading, positions in self._parts():
            if not positions:
                continue
            rows = table.loc[heading].rename(columns=columns)
            rows.index.name = None
            if heading.startswith("survival:") and self.causes is None:
                rows, heading = self._alpha_at_means(rows, heading, positions)
            out += "\n{}\n{}".format(
                heading[0].upper() + heading[1:],
                format_table(rows, list(columns.values())),
            )
        return out

    def _marker_repr(self) -> str:
        """The printout of a model of several markers."""
        from surpyval.univariate.regression._summary import format_table

        followed = ", ".join(link_words(link) for link in self.link)
        followed += weight_words(self.weight)
        followed += lag_words(self.lag)
        if self.survival == "AFT":
            survival = (
                f"{self.dist.name} accelerated failure time, ageing sped "
                f"up by the {followed}"
            )
        else:
            survival = (
                f"{self.dist.name} proportional hazards in the {followed}"
            )
        paths = "\n".join(
            f"  {name:<18}: {model.name}, {path_formula(model)}"
            for name, model in zip(self.markers or [], self.path_model.models)
        )
        out = (
            "Joint SurPyval Model"
            "\n===================="
            f"\nMarkers             : {len(self.markers or [])}, each with "
            "its own path, noise and effect\n"
            + paths
            + "\nVarying between items: "
            + ", ".join(self.varying)
            + _structure_words(self.random_effects_structure)
            + f"\nSurvival            : {survival}"
            f"\nData                : {self.n_items:g} items, "
            f"{self.n_events:g} failures, {self.n_readings:g} readings"
            f"{self._cause_line()}"
            f"\nFit                 : {self._fit_line()}"
            f"\nIntegration         : {_INTEGRATION_WORDS[self.integration]}"
            "\nEffect units        : "
        )
        out += "; ".join(
            f"{name} per {per}" for name, per in self.effect_units.items()
        )
        table = self.summary()
        columns = {
            "coef": "estimate",
            "se(coef)": "se",
            "coef lower 95%": "lower 95%",
            "coef upper 95%": "upper 95%",
        }
        for heading, positions in self._parts():
            if not positions:
                continue
            rows = table.loc[heading].rename(columns=columns)
            rows.index.name = None
            if heading.startswith("survival:"):
                rows, heading = self._alpha_at_means(rows, heading, positions)
            out += "\n{}\n{}".format(
                heading[0].upper() + heading[1:],
                format_table(rows, list(columns.values())),
            )
        return out

    def _alpha_at_means(
        self, rows: pd.DataFrame, heading: str, positions: list[int]
    ) -> tuple[pd.DataFrame, str]:
        """The printout's survival part with ``alpha`` at the hazard
        covariates' means first (the readable value: at the covariates'
        0, which can be far from the data, ``alpha`` can be 1e11 with a
        larger standard error), and the parameter itself, at 0, after
        it as ``alpha (Z = 0)``; unchanged where the means are 0. Its
        interval is the Wald interval on the log scale, from the delta
        method on ``log alpha - (gamma . zbar) / beta`` (``/ 1`` for an
        accelerated model), as the table's other positive parameters'."""
        import pandas as pd

        names = self.parameter_names
        kept = [self.feature_names[k] for k in self._kept]
        means = dict(zip(kept, np.asarray(self._zbar, dtype=float)))
        hazard = [j for j in positions[2:] if names[j] in means]
        shift = {names[j]: means[names[j]] for j in hazard}
        # (a mean within round-off of 0, a centred covariate, is 0)
        if not hazard or all(abs(v) < 1e-12 for v in shift.values()):
            return rows, heading
        params = np.asarray(self.params, dtype=float)
        i_alpha, i_beta = positions[0], positions[1]
        alpha, beta = params[i_alpha], params[i_beta]
        per = 1.0 if self.survival == "AFT" else beta
        moved = float(sum(params[j] * shift[names[j]] for j in hazard))
        value = alpha * np.exp(-moved / per)
        grad = np.zeros(params.size)
        grad[i_alpha] = 1.0 / alpha
        if self.survival != "AFT":
            grad[i_beta] = moved / beta**2
        for j in hazard:
            grad[j] = -shift[names[j]] / per
        try:
            cov = self.covariance()
        except ValueError:
            cov = np.full((params.size, params.size), np.nan)
        with np.errstate(all="ignore"):
            se_log = float(np.sqrt(grad @ cov @ grad))
        half = norm.ppf(0.975) * se_log
        first = pd.DataFrame(
            {
                "estimate": [value],
                "se": [value * se_log],
                "lower 95%": [value * np.exp(-half)],
                "upper 95%": [value * np.exp(half)],
            },
            index=["alpha"],
        )
        rest = rows.rename(index={"alpha": "alpha (Z = 0)"})
        at = ", ".join(f"{k} = {v:.6g}" for k, v in shift.items())
        heading = heading.replace(
            "alpha at ",
            f"alpha at the covariates' means ({at}) and ",
            1,
        )
        return pd.concat([first, rest[list(first.columns)]]), heading

    def _cause_line(self) -> str:
        """The printout's line of the failure modes (none for a model
        of one kind of failure)."""
        if self.causes is None:
            return ""
        counts = self.n_events_by_cause or {}
        line = "\nFailure modes       : " + ", ".join(
            f"{cause} ({counts.get(cause, np.nan):g} failures)"
            for cause in self.causes
        )
        # the failures of unknown mode (no label in 'e'), any of them
        unknown = float(self.n_events) - float(sum(counts.values()))
        if counts and unknown > 0.5:
            line += f"; {unknown:g} of unknown mode"
        return line

    def _fit_line(self) -> str:
        """What the printout's ``Fit`` line says."""
        if self.how != "two-stage":
            held = self.spreads_at_zero
            line = f"maximum {self.maximum}"
            if held:
                sds = ", ".join(f"sd_{v}" for v in held)
                common = ", ".join(held)
                line += (
                    f", with {sds} at the lower bound, 0 (the fit with "
                    f"{common} common to all items; no standard error "
                    "there: compare by aic())"
                )
            if self._proportional is not None:
                k, j = self._proportional
                name = self.perfect_correlations[0]
                value = self.params[list(self.parameter_names).index(name)]
                line += (
                    f", with {name} at its edge, {value:+g} ({k} an exact "
                    f"multiple of {j} about their means; no standard error "
                    "there)"
                )
            return line
        line = (
            "two-stage (approximate: the paths from the readings alone, "
            "then the lifetimes with each item's predicted path taken as "
            "known; estimates biased, standard errors understated, no AIC)"
        )
        if self.maximum == "unverified":
            line += "; stage 2 not at a verified maximum"
        return line

    # -- prediction ------------------------------------------------------

    def _question(
        self, Z: Any, markers: Any, given: Any
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, float]:
        """The covariates (of every column), readings and age the item
        was last seen running, from ``Z`` (or an ``ItemLifetime`` there),
        ``markers`` and ``given``, checked."""
        from surpyval.utils.item_history import covariate_row_in
        from surpyval.utils.item_lifetime import ItemLifetime

        p = len(self.feature_names)
        if isinstance(Z, ItemLifetime) and self.markers is not None:
            raise ValueError(
                f"Item {Z.id!r} is an ItemLifetime, whose markers are one "
                "measurement, and this model has several markers "
                f"({', '.join(self.markers)}): give Z (its covariates), "
                "markers={marker: {age: value}} and given (the age it was "
                "last seen running)."
            )
        if isinstance(Z, ItemLifetime):
            item = Z
            if markers is not None or given is not None:
                raise ValueError(
                    f"Item {item.id!r} holds its readings and the age it "
                    "was last seen running; leave out markers and given "
                    "(or give Z, markers and given instead of the item)."
                )
            if item.outcome != "right_censored":
                raise ValueError(
                    f"Item {item.id!r} is {item.outcome}, not still "
                    "running: a prediction is for an item still running "
                    "at its right_censored age, given its readings so far."
                )
            if p == 0 and item.covariates is not None:
                raise ValueError(
                    f"Item {item.id!r} has covariates, and the model was "
                    "fitted without any; leave them out of the item."
                )
            if p and item.covariates is None:
                raise ValueError(
                    f"Item {item.id!r} has no covariates; the model was "
                    f"fitted on {self.feature_names!r}."
                )
            Z = None
            if p:
                have = item.covariate_names
                if have is not None:
                    if set(have) != set(self.feature_names):
                        raise ValueError(
                            f"Item {item.id!r} has covariates "
                            f"{list(have)!r}; the model was fitted on "
                            f"{self.feature_names!r}."
                        )
                    Z = dict(
                        zip(
                            self.feature_names,
                            covariate_row_in(item, self.feature_names),
                        )
                    )
                else:
                    Z = np.asarray(item.covariates, dtype=float)
            markers, given = item.markers, item.value
        z = self._covariate_row(Z)
        if self.markers is not None:
            return self._marker_question(z, markers, given)
        ages, values = self._readings(markers)
        if given is None:
            if ages.size:
                raise ValueError(
                    "Give 'given', the age the item was last seen running "
                    "(at or after its last reading, "
                    f"{ages.max():g}): the readings alone do not say it."
                )
            given_age = 0.0
        else:
            given_age = _finite_number("given", given)
            if given_age < 0:
                raise ValueError(
                    f"'given' is an age, 0 or more; got {given_age:g}."
                )
            late = ages[ages > given_age]
            if late.size:
                raise ValueError(
                    f"A reading at age {late.max():g} is after 'given', "
                    f"{given_age:g}, the age the item was last seen "
                    f"running: give given={late.max():g} or later."
                )
        if ages.size and np.all(np.isfinite(z)):
            self.path_model.check_data(ages, values)
        return z, ages, values, given_age

    def _marker_question(
        self, z: npt.NDArray, markers: Any, given: Any
    ) -> tuple:
        """:meth:`_question`'s readings and age of several markers: the
        readings ``[(ages, values), ...]``, one marker each."""
        names = list(self.markers or [])
        if markers is None:
            markers = {}
        if not isinstance(markers, Mapping) or any(
            not isinstance(v, Mapping) for v in markers.values()
        ):
            raise ValueError(
                "With several markers 'markers' are the readings so far by "
                f"marker, {{marker: {{age: value}}}}, of {names!r}; got "
                f"{markers!r}."
            )
        unknown = [k for k in markers if k not in names]
        if unknown:
            raise ValueError(
                f"'markers' names {unknown!r}, which are not markers of the "
                f"model; its markers are {names!r}."
            )
        readings = [self._readings(markers.get(name)) for name in names]
        latest = max(
            [float(a.max()) for a, _ in readings if a.size], default=None
        )
        if given is None:
            if latest is not None:
                raise ValueError(
                    "Give 'given', the age the item was last seen running "
                    f"(at or after its last reading, {latest:g}): the "
                    "readings alone do not say it."
                )
            given_age = 0.0
        else:
            given_age = _finite_number("given", given)
            if given_age < 0:
                raise ValueError(
                    f"'given' is an age, 0 or more; got {given_age:g}."
                )
            if latest is not None and latest > given_age:
                raise ValueError(
                    f"A reading at age {latest:g} is after 'given', "
                    f"{given_age:g}, the age the item was last seen "
                    f"running: give given={latest:g} or later."
                )
        if np.all(np.isfinite(z)):
            for name, model, (a, v) in zip(
                names, self.path_model.models, readings
            ):
                if a.size:
                    model.check_data(a, v)
        ages = [a for a, _ in readings]
        values = [v for _, v in readings]
        return z, ages, values, given_age

    def _item_arrays(
        self, times: npt.NDArray, zc: npt.NDArray, ages: Any, values: Any
    ) -> JointArrays:
        """One item's arrays, a row an age of ``times`` (each its own
        ``T``), its readings in every row: ``ages`` and ``values`` (by
        marker, lists, with several markers)."""
        M = times.size
        common = (
            np.ones(M),
            np.maximum(times, 1e-300),
            np.zeros(M),
            np.zeros(M),
            np.repeat(zc, M, axis=0),
        )
        if self.markers is None:
            return JointArrays.build(*common, [ages] * M, [values] * M)
        from ._multi import joined

        a, v, ch = joined(list(zip(ages, values)))
        return JointArrays.build(
            *common,
            [a] * M,
            [v] * M,
            channels=[ch] * M,
            n_markers=len(self.markers),
        )

    def _covariate_row(self, Z: Any) -> npt.NDArray:
        """``Z`` as the row of every covariate column (a named mapping is
        read by name)."""
        names = self.feature_names
        p = len(names)
        if Z is None:
            if p:
                raise ValueError(
                    f"The model was fitted on covariates {names!r}: give "
                    "the item's, Z={...} by name or a row of "
                    f"{p} value(s)."
                )
            return np.zeros(0)
        if (
            isinstance(Z, Mapping)
            or hasattr(Z, "index")
            and hasattr(Z, "to_dict")
        ):
            given = dict(Z.items())
            if set(given) != set(names):
                raise ValueError(
                    f"Z names {sorted(given)!r}; the model was fitted on "
                    f"{names!r}."
                )
            return np.array([float(given[n]) for n in names])
        row = np.asarray(Z, dtype=float)
        if row.ndim == 2 and row.shape[0] == 1:
            row = row[0]
        if p == 0 and row.size == 0:
            return np.zeros(0)
        if row.ndim != 1 or row.size != p:
            raise ValueError(
                f"Z is one item's covariates, a row of {p} value(s) "
                f"({names!r}); got shape {np.shape(Z)}."
                + (
                    " The model was fitted without covariates: leave Z out."
                    if p == 0
                    else ""
                )
            )
        return row

    def _readings(self, markers: Any) -> tuple[npt.NDArray, npt.NDArray]:
        if markers is None:
            return np.zeros(0), np.zeros(0)
        if not isinstance(markers, Mapping):
            raise ValueError(
                "'markers' are the readings so far, a dict {age: value}."
            )
        ages, values = [], []
        for age, value in markers.items():
            age = _finite_number("markers' ages", age)
            if age < 0:
                raise ValueError(
                    f"A reading's age is 0 or more; got {age:g} in markers."
                )
            ages.append(age)
            values.append(_finite_number(f"the reading at age {age:g}", value))
        order = np.argsort(ages, kind="stable")
        return np.array(ages)[order], np.array(values)[order]

    def _conditional(
        self, x: npt.NDArray, z: npt.NDArray, ages: Any, values: Any, u: float
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """``(log_sf, log_hf)`` of one item at the ages ``x`` (each after
        ``u``; ``z`` its covariates, checked, ``ages`` and ``values`` its
        readings), at the fitted parameters: the log of its survival to
        each from ``u``, and of its hazard there given its readings and
        that it reached that age,

            h(x | y) = int p(y | theta) h(x | theta) S(x | theta) phi
                       / int p(y | theta) S(x | theta) phi,

        both integrals on the nodes of the second (centred at the mode of
        its integrand), and the survival as :meth:`sf` computes it."""
        kept = self._kept
        zc = (z[kept] - self._zbar).reshape(1, -1)
        times = np.concatenate([[u], np.asarray(x, dtype=float)])
        M = times.size
        many = self._item_arrays(times, zc, ages, values)
        model = self._likelihood_model()
        theta = self._theta()
        with np.errstate(all="ignore"):
            mode, root = continued_modes(
                theta, model.layout, model.path, many, times, model.centre
            )
            nodes = model.prediction_nodes(mode, root, theta, many, times)
            log_I = log_integrals(
                theta, model, many, nodes, times, np.zeros(M), readings=True
            )
            log_J = log_integrals(
                theta, model, many, nodes, times, np.ones(M), readings=True
            )
            # (an age the item cannot survive to: log sf -inf, and its
            # hazard there not defined, nan)
            return log_I[1:] - log_I[0], (log_J - log_I)[1:]

    def _marker_moments(
        self,
        x: npt.NDArray,
        z: npt.NDArray,
        ages: Any,
        values: Any,
        u: float,
        marker: Any = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """The mean and standard deviation of one item's noise-free
        marker at the ages ``x``, over its path parameters' distribution
        given its readings and that it reached ``u`` (adaptive
        Gauss-Hermite quadrature, at the fitted parameters); with
        several markers, of ``marker``."""
        kept = self._kept
        zc = (z[kept] - self._zbar).reshape(1, -1)
        one = self._item_arrays(np.array([u]), zc, ages, values)
        model = self._likelihood_model()
        theta = self._theta()
        t = np.array([u])
        with np.errstate(all="ignore"):
            mode, root = continued_modes(
                theta, model.layout, model.path, one, t, model.centre
            )
            nodes = model.prediction_nodes(mode, root, theta, one, t)
            if isinstance(nodes, tuple):
                log_g = np.asarray(
                    log_integrand(
                        theta, model, one, nodes, t, np.zeros(1), readings=True
                    ),
                    dtype=float,
                )[0]
                weights = np.exp(log_g - np.max(log_g))
                weights = weights / weights.sum()
                points = nodes[0]
            else:
                # Laplace: the moments over the normal it stands on
                from ._laplace import sigma_points

                points, weights = sigma_points(nodes)
            params = np.asarray(
                _path_params(theta, model.layout, one.Z, points),
                dtype=float,
            )
            x = np.asarray(x, dtype=float)
            if self.markers is not None:
                k = list(self.markers).index(marker)
                piece = model.path.slices[k]
                m = _at(model.path.parts[k], params[..., piece], x[None, :])[0]
            else:
                m = _at(model.path, params, x[None, :])[0]  # (K, M)
            m = np.asarray(m, dtype=float)
        mean = weights @ m
        var = weights @ (m - mean) ** 2
        return mean, np.sqrt(np.maximum(var, 0.0))

    def predict(
        self,
        Z: Any = None,
        markers: dict | None = None,
        given: float | None = None,
    ) -> DynamicPrediction:
        """
        The dynamic prediction of one item: its failure-age distribution
        given its covariates ``Z``, its readings so far ``markers`` and
        that it was still running at age ``given`` (an ``ItemLifetime``
        still running, as ``Z``, stands for all three), as an object with
        the conditional ``sf``, ``ff``, ``hf``, ``df`` and ``Hf``, ``qf``
        and the remaining life's quantiles, a Wald band from the
        parameters' uncertainty (``cb``), the item's smoothed and
        projected marker (``marker``, ``marker_interval``) and draws
        (``random``). Each new reading, given to ``predict`` again,
        updates it. :meth:`sf`, :meth:`ff` and :meth:`cb` give the same
        numbers for an array of ages.

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
        >>> hot = model.predict(markers={0: 2.4, 20: 2.9, 40: 3.4}, given=40)
        >>> cool = model.predict(markers={0: 1.8, 20: 1.9, 40: 2.0}, given=40)
        >>> bool(hot.remaining_life(0.5) < cool.remaining_life(0.5))
        True
        >>> bool(hot.sf(60) == model.sf(60, markers=hot.markers, given=40))
        True
        """
        from .dynamic import DynamicPrediction

        return DynamicPrediction(self, Z, markers, given)

    def _log_sf_function(
        self, x: npt.NDArray, Z: Any, markers: Any, given: Any
    ) -> Any:
        """``(f, after, missing)``: ``f(params)``, the log of the item's
        survival to each age of ``x`` after ``given`` (``u``), as a
        function of the estimated parameters; the mask of those ages; and
        whether the item's covariates are missing (``nan``).

        The survival is ``I(x) / I(u)``, ``I(t) = int p(y | theta) S(t |
        theta) phi(theta_V) d theta_V``: each integral by adaptive
        Gauss-Hermite quadrature on nodes centred at the mode of its own
        integrand, placed at the fitted parameters and held while
        differentiating."""
        z, ages, values, u = self._question(Z, markers, given)
        after = np.isfinite(x) & (x > u)
        if not np.all(np.isfinite(z)):
            return None, after, True
        kept = self._kept
        zc = (z[kept] - self._zbar).reshape(1, -1)
        times = np.concatenate([[u], x[after]])
        M = times.size
        many = self._item_arrays(times, zc, ages, values)
        model = self._likelihood_model()
        theta = self._theta()
        mode, root = continued_modes(
            theta, model.layout, model.path, many, times, model.centre
        )
        nodes = model.prediction_nodes(mode, root, theta, many, times)
        zero = np.zeros(M)

        def f(params: Any) -> Any:
            th = internal_theta(params, model.layout, self._refs)
            log_I = log_integrals(
                th, model, many, nodes, times, zero, readings=True
            )
            return log_I[1:] - log_I[0]

        return f, after, False

    def _check_mode(self, mode: Any) -> int | None:
        """The place of the failure mode ``mode`` among the model's
        ``causes`` (``None`` for every mode together)."""
        if mode is None:
            return None
        if self.causes is None:
            raise ValueError(
                "This joint model has one way of failing (it was fitted "
                "without failure modes, e=): leave 'mode' out, and "
                "cif(x) is ff(x). Fit with each failure's mode as e= for "
                "the chance of each."
            )
        names = [str(c) for c in self.causes]
        if str(mode) not in names:
            raise ValueError(
                f"'mode' is one of the model's failure modes, "
                f"{list(self.causes)!r}; got {mode!r}."
            )
        return names.index(str(mode))

    def _log_cif_function(
        self, x: npt.NDArray, Z: Any, markers: Any, given: Any
    ) -> tuple:
        """``(f, after, missing)``: ``f(params)``, the log of the item's
        chance of failing by each mode by each age of ``x`` after
        ``given`` (``u``), ``(n_modes, n_after)``, as a function of the
        estimated parameters; the mask of those ages; and whether the
        item's covariates are missing.

        Each mode's chance is the chance of failing at all by ``x``,
        ``1 - sf(x)`` (:meth:`_log_sf_function`'s), times the mode's
        share of it: over the item's path parameters given its readings
        and its survival to ``u`` (``b``, on the nodes placed there),

            share_k(x) = E[F(x | b) G_k(x | b) / sum_j G_j(x | b)]
                         / E[F(x | b)],

        with ``F(x | b) = 1 - S(x | b) / S(u | b)``, exact, and ``G_k(x
        | b) = int_u^x h_k(s | b) S(s | b) / S(u | b) ds`` by the
        cumulative hazard's Gauss-Legendre rule
        (:func:`log_cause_fails_between`). The share is taken at each
        node, inside the integral over age and the average over the
        path, so the result is ``E[G_k(x | b)]``, the mode's cumulative
        incidence from ``u``. So the modes' chances add to
        ``1 - sf(x)`` whatever the rule's error, which moves only the
        split between them. The nodes are held at the fitted parameters
        while differentiating."""
        f_sf, after, missing = self._log_sf_function(x, Z, markers, given)
        if missing:
            return None, after, True
        n_modes = len(self.causes or [])
        if not after.any():
            return (lambda params: np.zeros((n_modes, 0))), after, False
        z, ages, values, u = self._question(Z, markers, given)
        kept = self._kept
        zc = (z[kept] - self._zbar).reshape(1, -1)
        xs = np.asarray(x, dtype=float)[after]
        n = xs.size
        model = self._likelihood_model()
        theta = self._theta()
        at_u = self._item_arrays(np.array([u]), zc, ages, values)
        mode, root = continued_modes(
            theta, model.layout, model.path, at_u, np.array([u]), model.centre
        )
        nodes = model.prediction_nodes(mode, root, theta, at_u, np.array([u]))
        if not isinstance(nodes, tuple):
            raise NotImplementedError(
                "The chance of each failure mode is integrated on a grid "
                f"of the varying parameters, which a model of more than "
                f"{max(GH_PREDICT)} of them has none of (its predictions "
                "use Laplace's approximation). Fit with fewer varying "
                "parameters, or use sf for the chance of failing at all."
            )
        b, logw = nodes
        rows = np.repeat(np.array([u]), n)
        same = (np.repeat(b, n, axis=0), np.repeat(logw, n, axis=0))
        base = self._item_arrays(rows, zc, ages, values)
        by_mode = []
        for k in range(n_modes):
            each = self._item_arrays(rows, zc, ages, values)
            each.e = np.full(n, k)
            by_mode.append(each)
        zero = np.zeros(n)

        def f(params: Any) -> Any:
            th = internal_theta(params, model.layout, self._refs)
            # log weight, prior, readings and S(u | b) at each node
            at = log_integrand(th, model, base, same, rows, zero, True)
            # S(x | b) on the same nodes
            to_x = log_integrand(th, model, base, same, xs, zero, True)
            log_F = log1mexp(anp.log(at - to_x))
            log_G = anp.stack(
                [
                    log_integrand(th, model, arr, same, rows, zero, True, xs)
                    - at
                    for arr in by_mode
                ]
            )
            log_split = log_G - logsumexp(log_G, axis=0)
            log_share = (
                logsumexp((at + log_F)[None] + log_split, axis=-1)
                - logsumexp(at + log_F, axis=-1)[None]
            )
            log_total = log1mexp(anp.log(-f_sf(params)))
            return log_total[None] + log_share

        return f, after, False

    @keeps_query_shape
    def sf(
        self,
        x: npt.ArrayLike,
        Z: Any = None,
        markers: dict | None = None,
        given: float | None = None,
    ) -> npt.NDArray:
        """
        The survival of one item to each age ``x``, given its covariates
        ``Z``, its readings so far ``markers`` and that it was still
        running at age ``given``:

        .. math::
            P(T > x \\mid T > u, y) =
            \\frac{\\int S(x \\mid \\theta) p(y \\mid \\theta)
            \\phi(\\theta) d\\theta}
            {\\int S(u \\mid \\theta) p(y \\mid \\theta) \\phi(\\theta)
            d\\theta},

        over its path's varying parameters, each integral by adaptive
        Gauss-Hermite quadrature (centred at the mode of its integrand).
        With no readings and no ``given`` it is a new item's survival,
        over the population's paths. An age at or before ``given`` gives
        1; ``nan`` gives ``nan``, and so does a missing covariate. The
        result has the shape of ``x``.

        Parameters
        ----------
        x : array_like
            The ages.
        Z : dict, array_like or ItemLifetime, optional
            The item's covariates: ``{name: value}`` or a row in the
            order of ``feature_names`` (needed where the model has
            covariates). An ``ItemLifetime`` (still running) stands for
            all three arguments: its covariates, its ``markers`` and its
            right censored age as ``given``.
        markers : dict, optional
            The readings so far, ``{age: value}``, at or before ``given``.
        given : float, optional
            The age the item was last seen running (needed with
            readings).

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
        >>> float(model.sf(40, markers={0: 2.0}, given=40))
        1.0
        >>> item = ItemLifetime("P", right_censored=40, markers={0: 2.0})
        >>> same = model.sf(60, markers={0: 2.0}, given=40)
        >>> bool(model.sf(60, item) == same)
        True
        """
        x = np.asarray(x, dtype=float)
        f, after, missing = self._log_sf_function(x, Z, markers, given)
        out = np.where(np.isnan(x), np.nan, 1.0)
        if missing:
            return np.full(x.shape, np.nan)
        if after.any():
            # far beyond the data the survival underflows to 0
            with np.errstate(all="ignore"):
                params = np.asarray(self.params, dtype=float)
                out[after] = np.exp(f(params[self._estimated]))
        return out

    @keeps_query_shape
    def ff(
        self,
        x: npt.ArrayLike,
        Z: Any = None,
        markers: dict | None = None,
        given: float | None = None,
    ) -> npt.NDArray:
        """
        The probability that one item fails by each age ``x``, given its
        covariates, readings and the age it was last seen running (as
        for :meth:`sf`): ``1 - sf(x, Z, markers, given)``.

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
        >>> float(model.ff(20, markers={0: 2.0}, given=40))
        0.0
        """
        return 1.0 - self.sf(x, Z, markers, given)

    @keeps_query_shape
    def qf(
        self,
        p: npt.ArrayLike,
        Z: Any = None,
        markers: dict | None = None,
        given: float | None = None,
    ) -> npt.NDArray:
        """
        The age by which one item has failed with probability ``p``, given
        its covariates ``Z``, its readings so far ``markers`` and that it
        was still running at age ``given`` (as for :meth:`sf`): the
        inverse of :meth:`ff`, ``predict(Z, markers, given).qf(p)``.

        ``given`` for ``p = 0``; ``inf`` for 1 and where the survival
        never falls to ``1 - p`` (a path that can carry the marker away
        from failure leaves a chance of never failing); ``nan`` with a
        warning outside [0, 1]. The remaining life is ``qf(p) - given``.
        The result has the shape of ``p``.

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
        >>> t = model.qf([0.1, 0.5], markers={0: 2.0, 20: 2.5}, given=40)
        >>> bool(40 < t[0] < t[1])
        True
        >>> bool(np.allclose(model.ff(t, markers={0: 2.0, 20: 2.5},
        ...                           given=40), [0.1, 0.5]))
        True
        """
        return self.predict(Z, markers, given).qf(p)

    def mean(
        self,
        Z: Any = None,
        markers: dict | None = None,
        given: float | None = None,
        tau: float | None = None,
    ) -> float:
        """
        The expected failure age of one item, given its covariates ``Z``,
        its readings so far ``markers`` and that it was still running at
        age ``given`` (as for :meth:`sf`), or with ``tau`` the restricted
        mean to that age: ``predict(Z, markers, given).mean(tau)`` (see
        :meth:`DynamicPrediction.mean
        <surpyval.univariate.joint.dynamic.DynamicPrediction.mean>`).

        It is ``inf``, with a warning giving the chance, where the item's
        survival levels off above 0: a path whose varying parameters can
        carry the marker away from failure (a straight line with a slope
        that may be negative) leaves a chance that it never fails. Give
        ``tau`` for the restricted mean then, or use :meth:`qf`.

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
        >>> m = model.mean(markers={0: 2.0, 20: 2.5}, given=40, tau=200)
        >>> bool(40 < m < 200)
        True
        """
        return self.predict(Z, markers, given).mean(tau)

    @keeps_query_shape
    def cb(
        self,
        x: npt.ArrayLike,
        Z: Any = None,
        markers: dict | None = None,
        given: float | None = None,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        mode: Any = None,
    ) -> npt.NDArray:
        """
        Confidence bounds on one item's survival (``on="sf"``, the
        default), failure probability (``on="ff"``) or, with failure
        modes, chance of failing by the mode ``mode`` (``on="cif"``; the
        delta method on ``log(-log(1 - cif))``, the same scale) to each
        age ``x``,
        given its covariates, readings and the age it was last seen
        running (as for :meth:`sf`), from the uncertainty of the
        parameters: the delta method, with the parameters' covariance, on
        ``log(-log sf)`` (so the bounds stay in (0, 1)), the adaptive
        nodes held at the fitted parameters while differentiating. An age
        at or before ``given`` gives ``[1, 1]`` (``[0, 0]`` on ``ff``).
        A two-stage fit's bounds are from its survival part's
        uncertainty alone (its marker part held at its estimates), with
        standard errors that are too small: they are too narrow.

        Parameters
        ----------
        x : array_like
            The ages.
        Z, markers, given : optional
            The item, as for :meth:`sf` (an ``ItemLifetime`` as ``Z``
            stands for all three).
        on : str, optional
            ``"sf"`` (or ``"R"``), ``"ff"`` (or ``"F"``) or ``"cif"``.
        alpha_ci : float, optional
            The total tail probability: 0.05 (the default) gives 95%
            bounds.
        bound : str, optional
            ``"two-sided"`` (``[lower, upper]`` on a last axis),
            ``"lower"`` or ``"upper"``.
        mode : optional
            With ``on="cif"``, one of the model's failure modes
            (``causes``); ``None`` (the default) for any, which is
            ``on="ff"``.

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
        >>> model.cb([20, 60], markers={0: 2.0}, given=40).shape
        (2, 2)
        """
        check_alpha_ci(alpha_ci)
        check_option("on", on, _CB_ON)
        check_option("bound", bound, BOUNDS)
        k = self._check_mode(mode)
        if on == "cif" and k is None:
            on = "ff"
        elif on != "cif" and k is not None:
            raise ValueError(
                "'mode' is for the chance of failing by a mode, on='cif'; "
                f"got on={on!r}."
            )
        cov = self.covariance()
        x = np.asarray(x, dtype=float)
        if on == "cif" and k is not None:
            return self._cif_cb(x, Z, markers, given, k, alpha_ci, bound)
        f, after, missing = self._log_sf_function(x, Z, markers, given)
        tail, signs = bound_signs(alpha_ci, bound)
        if on in ("ff", "F"):
            # a lower bound on ff is an upper one on sf
            signs = -signs
        q = norm.ppf(1 - tail)
        out = np.ones(x.shape + (signs.size,))
        out[np.isnan(x)] = np.nan
        if missing:
            out[:] = np.nan
        elif after.any():
            est = self._estimated
            params = np.asarray(self.params, dtype=float)[est]
            sub = cov[np.ix_(est, est)]
            if self.how == "two-stage":
                # stage one's estimates have no variance: held at them
                sub = np.where(np.isfinite(sub), sub, 0.0)

            def log_H(v: Any) -> Any:
                return anp.log(-f(v))

            # where the survival underflows to 0, the bound is nan
            with np.errstate(all="ignore"):
                jvp = make_jvp(log_H)(params)
                columns = [jvp(e)[1] for e in np.eye(params.size)]
                centre = log_H(params)
                J = np.stack(columns, axis=-1)
                se = np.sqrt(np.einsum("mi,ij,mj->m", J, sub, J))
                # an upper bound on log H is a lower bound on sf
                out[after] = np.exp(
                    -np.exp(centre[:, None] - signs[None, :] * q * se[:, None])
                )
        if on in ("ff", "F"):
            out = 1.0 - out
        return out if bound == "two-sided" else out[..., 0]

    def _cif_cb(
        self,
        x: npt.NDArray,
        Z: Any,
        markers: Any,
        given: Any,
        k: int,
        alpha_ci: float,
        bound: str,
    ) -> npt.NDArray:
        """:meth:`cb` on the chance of failing by the mode at place
        ``k``: the delta method on ``log(-log(1 - cif))``, the nodes held
        at the fitted parameters (``[0, 0]`` up to ``given``)."""
        cov = self.covariance()
        f, after, missing = self._log_cif_function(x, Z, markers, given)
        tail, signs = bound_signs(alpha_ci, bound)
        q = norm.ppf(1 - tail)
        out = np.zeros(x.shape + (signs.size,))
        out[np.isnan(x)] = np.nan
        if missing:
            out[:] = np.nan
        elif after.any():
            est = self._estimated
            params = np.asarray(self.params, dtype=float)[est]
            sub = cov[np.ix_(est, est)]
            if self.how == "two-stage":
                sub = np.where(np.isfinite(sub), sub, 0.0)

            def log_H(v: Any) -> Any:
                # the cif's own cumulative hazard, log(-log(1 - cif))
                return anp.log(-anp.log1p(-anp.exp(f(v)[k])))

            with np.errstate(all="ignore"):
                jvp = make_jvp(log_H)(params)
                columns = [jvp(e)[1] for e in np.eye(params.size)]
                centre = log_H(params)
                J = np.stack(columns, axis=-1)
                se = np.sqrt(np.einsum("mi,ij,mj->m", J, sub, J))
                out[after] = -np.expm1(
                    -np.exp(centre[:, None] + signs[None, :] * q * se[:, None])
                )
        return out if bound == "two-sided" else out[..., 0]

    # -- saving and loading ----------------------------------------------

    def to_dict(self) -> dict:
        """
        The model as a plain, JSON-serialisable ``dict``: its path model
        (by its built-in name) and which of its parameters vary (and
        which of those are correlated, ``"random_effects"``, where not
        every pair is), its parameters and their names, the covariates'
        names and the aliased ones, the marker reference, the centres its
        predictions are computed at, the covariance, the fitted negative
        log-likelihood and the counts of the data (not the data).
        :meth:`from_dict` rebuilds it, and it predicts and bounds as the
        original. A built-in path on links (``LinkedPathModel``) is saved
        as its name and its links. A path model of one's own cannot be
        rebuilt from a name, and is refused (``NotImplementedError``); the
        model pickles all the same.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval
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
        >>> restored = surpyval.from_dict(model.to_dict())
        >>> bool(np.allclose(restored.params, model.params))
        True
        """
        several: dict = {}
        if self.markers is not None:
            key, links = None, None
            several = {
                "markers": [
                    dict(
                        zip(
                            ("name", "path_model", "links"),
                            (name, *_saved_path(m)),
                        )
                    )
                    for name, m in zip(self.markers, self.path_model.models)
                ]
            }
        else:
            key, links = _saved_path(self.path_model)
        out: dict[str, Any] = {
            "model": "JointModel",
            "distribution": self.dist.name,
            "path_model": key,
            "links": links,
            **several,
            "varying": list(self.varying),
            "link": [
                list(link) if isinstance(link, tuple) else link
                for link in self.link
            ],
            **lag_to_dict(self.lag),
            **weight_to_dict(self.weight),
            "survival": self.survival,
            "params": np.asarray(self.params, dtype=float).tolist(),
            "parameter_names": list(self.parameter_names),
            "feature_names": list(self.feature_names),
            "aliased": np.asarray(self.aliased, dtype=int).tolist(),
            "path_covariates": (
                None
                if self.path_covariates is None
                else {k: list(v) for k, v in self.path_covariates.items()}
            ),
            "hazard_covariates": (
                None
                if self.hazard_covariates is None
                else list(self.hazard_covariates)
            ),
            "marker_reference": (
                None
                if self.marker_reference is None
                else (
                    {k: float(v) for k, v in self.marker_reference.items()}
                    if isinstance(self.marker_reference, Mapping)
                    else float(self.marker_reference)
                )
            ),
            "centre": (
                [float(v) for v in self._centre]
                if self.markers is not None
                else float(self._centre)
            ),
            "covariate_means": np.asarray(self._zbar, dtype=float).tolist(),
            "n_items": float(self.n_items),
            "n_events": float(self.n_events),
            "n_readings": float(self.n_readings),
            **(
                {}
                if self.marker_unit is None
                else {
                    "marker_unit": (
                        dict(self.marker_unit)
                        if isinstance(self.marker_unit, Mapping)
                        else self.marker_unit
                    )
                }
            ),
            "integration": self.integration,
            **(
                {}
                if self.random_effects_structure == "full"
                else {
                    "random_effects": _saved_structure(
                        self.random_effects_structure
                    )
                }
            ),
            **self._causes_entry(),
            **(
                {}
                if self._proportional is None
                else {"proportional": list(self._proportional)}
            ),
            "_neg_ll": float(self._neg_ll),
            **maximum_entry(self.maximum),
            **({} if self.how == "MLE" else {"how": self.how}),
        }
        if self._covariance is not None:
            out["covariance"] = np.asarray(self._covariance).tolist()
        return stamp_schema(out)

    def _causes_entry(self) -> dict:
        """What :meth:`to_dict` saves of the causes (nothing for a model
        of one kind of failure, whose dict is as it was)."""
        if self.causes is None:
            return {}
        counts = self.n_events_by_cause or {}
        return {
            "causes": to_native(list(self.causes)),
            "shared": list(self.shared),
            "n_events_by_cause": [
                float(counts.get(cause, np.nan)) for cause in self.causes
            ],
        }

    @classmethod
    def from_dict(cls, model_dict: dict) -> JointModel:
        """
        Rebuild a model from a :meth:`to_dict` dictionary.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ItemLifetime, SurpyvalData, WeibullJointPH
        >>> from surpyval.univariate.joint import JointModel
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
        >>> restored = JointModel.from_dict(model.to_dict())
        >>> restored.parameter_names == model.parameter_names
        True
        """
        import surpyval

        require_model_tag(model_dict, "JointModel", "a joint model")
        if model_dict["distribution"] != "Weibull":
            raise ValueError(
                "A joint model has a Weibull baseline; the dict names "
                f"{model_dict['distribution']!r}."
            )
        out = cls()
        out.dist = surpyval.Weibull
        several = model_dict.get("markers")
        if several:
            from ._multi import MarkerSet

            out.markers = [str(entry["name"]) for entry in several]
            out.path_model = MarkerSet(
                out.markers,
                [
                    _restored_path(entry["path_model"], entry.get("links"))
                    for entry in several
                ],
            )
        else:
            out.path_model = _restored_path(
                model_dict["path_model"], model_dict.get("links")
            )
        out.varying = list(model_dict["varying"])
        out.params = np.array(model_dict["params"], dtype=float)
        out.parameter_names = list(model_dict["parameter_names"])
        out.feature_names = list(model_dict["feature_names"])
        out.aliased = np.array(model_dict.get("aliased", []), dtype=int)
        # saved before the parts' covariates could differ: every one in both
        shifts = model_dict.get("path_covariates")
        out.path_covariates = (
            None if shifts is None else {k: list(v) for k, v in shifts.items()}
        )
        hazard = model_dict.get("hazard_covariates")
        out.hazard_covariates = None if hazard is None else list(hazard)
        # saved before the hazard could follow anything but the value
        out.link = tuple(
            tuple(link) if isinstance(link, list) else link
            for link in (model_dict.get("link") or ("value",))
        )
        # saved before the lag, or with none: at once
        saved_lag = model_dict.get("lag", 0.0)
        saved_weight = model_dict.get("weight")
        if out.markers is not None and isinstance(saved_lag, Mapping):
            # a lag by marker
            out.lag = marker_lags(out.markers, saved_lag)
        else:
            out.lag = check_lag(saved_lag)
        if out.markers is not None and isinstance(saved_weight, Mapping):
            out.weight = marker_weights(
                out.markers,
                {k: tuple(v) for k, v in saved_weight.items()},
                out.link,
            )
        else:
            # saved before the weights, or with none: all the exposure
            out.weight = check_weight(
                None if saved_weight is None else tuple(saved_weight)
            )
        plain = [k[1] if isinstance(k, tuple) else k for k in out.link]
        if isinstance(out.weight, tuple) and "cumulative" not in plain:
            raise ValueError(
                "A joint model's weight is on its accumulated exposure "
                "(the 'cumulative' link); the dict has the weight "
                f"{list(out.weight)!r} and the link {list(out.link)!r}."
            )
        # saved before the accelerated survival part: proportional hazards
        survival = model_dict.get("survival") or "PH"
        if survival not in _SURVIVAL_WORDS:
            raise ValueError(
                "A joint model's survival part is 'PH' (proportional "
                "hazards) or 'AFT' (accelerated); the dict names "
                f"{survival!r}."
            )
        if survival == "AFT" and "cumulative" in plain:
            raise ValueError(
                "An accelerated joint model does not follow the area under "
                "the marker (its effective age already accumulates it); the "
                "dict's link is " + repr(list(out.link)) + "."
            )
        out.survival = survival
        reference = model_dict["marker_reference"]
        if several:
            out.marker_reference = (
                None
                if reference is None
                else {k: float(v) for k, v in reference.items()}
            )
            out._centre = [float(v) for v in model_dict["centre"]]
        else:
            out.marker_reference = (
                None if reference is None else float(reference)
            )
            out._centre = float(model_dict["centre"])
        out._zbar = np.array(model_dict["covariate_means"], dtype=float)
        out.n_items = float(model_dict.get("n_items", 0.0))
        out.n_events = float(model_dict.get("n_events", 0.0))
        out.n_readings = float(model_dict.get("n_readings", 0.0))
        # saved before the marker's unit was kept: unknown
        out.marker_unit = model_dict.get("marker_unit")
        # saved before the integration had a choice: the full grid
        integration = model_dict.get("integration") or "grid"
        if integration not in INTEGRATIONS:
            raise ValueError(
                "A joint model's integration is 'grid', 'pruned' or "
                f"'laplace'; the dict names {integration!r}."
            )
        out.integration = integration
        # saved before the structure had a choice (or full): every pair
        # correlated
        out.random_effects_structure = check_random_effects(
            model_dict.get("random_effects") or "full", out.varying
        )
        # saved before the two-stage fit: the joint likelihood's maximum
        how = model_dict.get("how") or "MLE"
        if how not in HOWS:
            raise ValueError(
                "A joint model is fitted by 'MLE' or 'two-stage'; the dict "
                f"names {how!r}."
            )
        out.how = how
        # saved before a correlation could be held at +-1: none is
        proportional = model_dict.get("proportional")
        out._proportional = (
            None if proportional is None else tuple(proportional)
        )
        neg_ll = model_dict["_neg_ll"]
        out._neg_ll = np.nan if neg_ll is None else float(neg_ll)
        out.k = int(np.sum(~np.isnan(out.params)))
        out.maximum = restored_maximum(model_dict)
        if "covariance" in model_dict:
            out._covariance = np.array(model_dict["covariance"], dtype=float)
        if len(out.parameter_names) != len(out.params):
            raise ValueError(
                "The dict's 'parameter_names' and 'params' differ in length."
            )
        # saved before the failure modes, or with one kind of failure
        causes = model_dict.get("causes")
        if causes is not None:
            from surpyval.univariate.competing_risks.labels import (
                label_from_native,
            )

            out.causes = [label_from_native(k) for k in causes]
            out.shared = tuple(model_dict.get("shared") or ())
            counts = model_dict.get("n_events_by_cause")
            if counts is not None:
                out.n_events_by_cause = {
                    cause: np.nan if count is None else float(count)
                    for cause, count in zip(out.causes, counts)
                }
            if len(out.causes) < 2 or survival != "PH":
                raise ValueError(
                    "A joint model with failure modes ('causes') has two "
                    "or more and the proportional hazards survival part; "
                    f"the dict has {out.causes!r} and {survival!r}."
                )
            expected = out._names()
            if expected != out.parameter_names:
                raise ValueError(
                    "The dict's 'parameter_names' are not those of its "
                    f"'causes' ({out.causes!r}) and 'shared' "
                    f"({list(out.shared)!r}): expected {expected!r}."
                )
        if out.random_effects_structure != "full" or several:
            expected = out._names()
            if expected != out.parameter_names:
                saved = _saved_structure(out.random_effects_structure)
                raise ValueError(
                    "The dict's 'parameter_names' are not those of its "
                    f"'random_effects' ({saved!r}): expected {expected!r}."
                )
        return out
