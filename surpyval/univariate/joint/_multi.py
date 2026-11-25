"""Joint models of several markers: two or more measurements read on each
item (a vibration level and a temperature, two biomarkers), each with a
path of its own, its own noise and its own effect on the hazard, their
items' path parameters correlated across the markers.

Item ``i``'s marker ``k`` follows its own path model with parameters
``theta_ik``, read with its own noise,

    y_ijk = path_k(t_ijk; theta_ik) + e_ijk,  e_ijk ~ N(0, sigma_k^2),

at ages of its own (the markers need not be read together). The varying
parameters of every marker are one multivariate normal vector (one
covariance ``D`` across the markers, by its Cholesky factor), and the
hazard follows each marker through its own links and effects,

    h_i(t) = h0(t) exp(gamma'z_i + sum_k sum_l zeta_kl f_l(path_k)),

so the likelihood is the single marker's with the path the markers'
paths side by side: one parameter vector of the markers' parameters in
turn (:class:`MarkerSet`), each reading of its own marker's path, and a
term in the hazard for each ``(marker, link)`` pair
(:class:`~._likelihood.MarkerPaths`). The integral over the varying
parameters is the one marker's (a grid up to ``MAX_VARYING`` of them in
all, Laplace's approximation beyond, up to ``MAX_MARKER_VARYING``).

This module holds what is particular to several markers: their names and
paths, the fitter's options, the readings by marker, and the start of the
marker part (each marker's own, put side by side).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from ._likelihood import LINKS, JointArrays, Layout, Model, PathFunction

#: The most varying path parameters of all the markers together. The
#: integral over them is Laplace's approximation beyond five (``q^2 + q
#: + 1`` evaluations of each item's integrand, 73 for eight) and their
#: covariance has ``q (q + 1) / 2`` parameters (36 for eight): beyond
#: eight the fits were not tested, and a fleet large enough to estimate
#: so many correlations is rare. Fewer is the remedy: name fewer in
#: ``varying`` or leave correlations out with ``random_effects``.
MAX_MARKER_VARYING = 8

#: The covariance structure with every marker's varying parameters
#: correlated with each other and with none of another marker's.
BY_MARKER = "by_marker"


class MarkerSet:
    """The paths of several markers as one: ``names`` and their path
    ``models``; ``parameter_names`` are each marker's own prefixed by its
    name and a dot (``"vibration.a"``), one marker after another, and
    ``slices`` where each marker's are in that vector."""

    def __init__(self, names: list, models: list) -> None:
        self.names = [str(name) for name in names]
        self.models = list(models)
        sizes = [len(m.parameter_names) for m in self.models]
        starts = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
        self.slices = [
            slice(int(a), int(b)) for a, b in zip(starts[:-1], starts[1:])
        ]
        self.parameter_names = [
            f"{name}.{p}"
            for name, model in zip(self.names, self.models)
            for p in model.parameter_names
        ]
        self.name = "; ".join(
            f"{name}: {model.name}"
            for name, model in zip(self.names, self.models)
        )

    def of(self, name: str) -> Any:
        """The path model of marker ``name``."""
        return self.models[self.names.index(name)]

    def own(self, name: str) -> list[str]:
        """Marker ``name``'s parameters by their own names."""
        return list(self.of(name).parameter_names)

    def __repr__(self) -> str:
        return f"MarkerSet({self.name})"


def qualified(marker: str, name: str) -> str:
    """``name`` of marker ``marker``: ``"vibration.a"``."""
    return f"{marker}.{name}"


def check_markers(path_models: Mapping) -> tuple[list, list]:
    """``(names, models)`` of a dict of the markers' path models, each
    name text and each model a degradation path model (a ``LinkedPath``
    one too)."""
    from surpyval.degradation.path_models import PathModel

    if not path_models:
        raise ValueError(
            "path_model as a dict names each marker's path model, "
            "{name: path model}, with one marker or more; got an empty "
            "dict."
        )
    names, models = [], []
    for name, model in path_models.items():
        if not isinstance(name, str) or not name or "." in name:
            raise ValueError(
                "Each marker's name in path_model is text without a dot "
                "(the parameters are named '<marker>.<name>'); got "
                f"{name!r}."
            )
        if not isinstance(model, PathModel):
            raise ValueError(
                f"path_model[{name!r}] is a degradation path model, such "
                "as surpyval.degradation.LinearPath or QuadraticPath, or "
                f"a PathModel of your own; got {model!r}."
            )
        names.append(name)
        models.append(model)
    return names, models


def _per_marker(value: Any, names: list, what: str) -> dict:
    """``value`` by marker: a dict by marker name (each one it leaves out
    gets ``None``), else the same for every marker."""
    if isinstance(value, Mapping):
        unknown = [k for k in value if k not in names]
        if unknown:
            raise ValueError(
                f"{what} names {unknown!r}, which are not markers; the "
                f"markers are {names!r}."
            )
        return {name: value.get(name) for name in names}
    return {name: value for name in names}


def marker_varying(names: list, models: list, varying: Any) -> tuple:
    """The varying parameters by their qualified names, in the order of
    the markers and each marker's path: ``varying`` is ``None`` (each
    marker's default: every parameter, a spline's level and rate), a
    dict ``{marker: [names]}`` (a marker left out takes its default), or
    a list of qualified names (``"vibration.a"``), those of a marker not
    named common to all items."""
    defaults = {}
    for name, model in zip(names, models):
        own = list(model.parameter_names)
        default = getattr(model, "default_varying", None)
        defaults[name] = list(default) if default is not None else own
    if varying is None:
        chosen = defaults
    elif isinstance(varying, Mapping):
        given = _per_marker(varying, names, "varying")
        chosen = {}
        for name in names:
            value = given[name]
            if value is None:
                chosen[name] = defaults[name]
                continue
            value = [value] if isinstance(value, str) else list(value)
            chosen[name] = value
    else:
        listed = [varying] if isinstance(varying, str) else list(varying)
        every = [
            qualified(name, p)
            for name, model in zip(names, models)
            for p in model.parameter_names
        ]
        unknown = [v for v in listed if v not in every]
        if unknown:
            raise ValueError(
                "varying names the path parameters that vary between "
                "items, as '<marker>.<parameter>' (or a dict {marker: "
                f"[parameter, ...]}}), of {every}; got {unknown!r}."
            )
        chosen = {
            name: [
                p
                for p in model.parameter_names
                if qualified(name, p) in listed
            ]
            for name, model in zip(names, models)
        }
    out = []
    for name, model in zip(names, models):
        own = list(model.parameter_names)
        value = chosen[name]
        unknown = [v for v in value if v not in own]
        if unknown or len(set(value)) < len(value):
            raise ValueError(
                f"varying for marker {name!r} names the parameters of its "
                f"path that vary between items, each once, of {own}; got "
                f"{list(value)}."
            )
        out += [qualified(name, p) for p in own if p in value]
    if not out:
        raise ValueError(
            "varying names no path parameter that varies between items: "
            "at least one marker needs one (the items differ in it)."
        )
    return tuple(out)


def marker_links(names: list, link: Any) -> tuple:
    """The ``(marker, link)`` pairs the hazard follows, in the markers'
    order and each marker's in the order of ``LINKS``: ``link`` is one
    link or a list of them for every marker, or a dict ``{marker: link
    or [links]}`` (a marker left out, or given ``[]``, is read and not
    in the hazard). At least one pair."""
    given = _per_marker(link, names, "link")
    out = []
    for name in names:
        value = given[name]
        if value is None:
            continue
        try:
            links = [value] if isinstance(value, str) else list(value)
        except TypeError:
            links = [value]
        bad = [k for k in links if k not in LINKS]
        if bad or len(set(links)) < len(links):
            raise ValueError(
                f"link for marker {name!r} names what of its path the "
                "hazard follows: 'value', 'slope' or 'cumulative', or a "
                f"list of them, each once (or [] for none); got {value!r}."
            )
        out += [(name, k) for k in LINKS if k in links]
    if not out:
        raise ValueError(
            "link names what the hazard follows of no marker: give at "
            "least one marker a link, such as link={"
            f"{names[0]!r}: 'value'}}."
        )
    return tuple(out)


def marker_lags(names: list, lag: Any) -> Any:
    """The lag of each marker: one number for every marker (a float, as
    with one marker), or a dict ``{marker: lag}`` (a marker left out at
    once, 0), each checked by :func:`~._likelihood.check_lag`."""
    from ._likelihood import check_lag

    if not isinstance(lag, Mapping):
        return check_lag(lag)
    given = _per_marker(lag, names, "lag")
    return {
        name: 0.0 if value is None else check_lag(value)
        for name, value in given.items()
    }


def marker_weights(names: list, weight: Any, links: tuple) -> Any:
    """The weight on each marker's accumulated exposure: one weight for
    every marker that has the 'cumulative' link (a tuple or ``None``, as
    with one marker), or a dict ``{marker: weight}`` (a marker left out
    unweighted), each checked by :func:`~._likelihood.check_weight` and
    refused for a marker whose area the hazard does not follow."""
    from ._likelihood import check_weight

    if not isinstance(weight, Mapping):
        checked = check_weight(weight)
        if checked is not None and not any(
            k == "cumulative" for _, k in links
        ):
            raise ValueError(
                "'weight' weights the accumulated exposure, the area "
                "under a marker's path that the 'cumulative' link "
                "follows, and no marker here has that link. Add it "
                "(link={marker: ['value', 'cumulative']}), or leave "
                "weight out."
            )
        return checked
    given = _per_marker(weight, names, "weight")
    out: dict = {}
    for name, value in given.items():
        checked = None if value is None else check_weight(value)
        if checked is not None and (name, "cumulative") not in links:
            raise ValueError(
                f"'weight' for marker {name!r} weights the accumulated "
                "exposure, the area under its path that the 'cumulative' "
                f"link follows, and {name!r} has no such link. Add it "
                f"(link={{{name!r}: ['value', 'cumulative']}}), or leave "
                f"{name!r} out of weight."
            )
        out[name] = checked
    return out


def lag_of(lag: Any, name: str) -> float:
    """Marker ``name``'s lag (``lag`` one for all, or by marker)."""
    return float(lag.get(name, 0.0)) if isinstance(lag, Mapping) else lag


def weight_of(weight: Any, name: str) -> Any:
    """Marker ``name``'s weight (``weight`` one for all, or by marker)."""
    return weight.get(name) if isinstance(weight, Mapping) else weight


def lag_words(lag: Any, heading: bool = False) -> str:
    """The lag in words, to follow the links' (empty with none): ``",
    30 earlier (lag)"``, or with ``heading`` the summary's ``", lag
    30"``; by marker ``", lags vibration 30, temperature 7"``."""
    if not isinstance(lag, Mapping):
        if not lag > 0:
            return ""
        return f", lag {lag:g}" if heading else f", {lag:g} earlier (lag)"
    said = [f"{name} {value:g}" for name, value in lag.items() if value > 0]
    return f", lags {', '.join(said)}" if said else ""


def weight_words(weight: Any) -> str:
    """The weights by marker in words (empty with none); one weight for
    all is the one marker's words (``joint_model._weight_words``)."""
    from .joint_model import _weight_words

    if not isinstance(weight, Mapping):
        return _weight_words(weight)
    said = [
        f"{name}{_weight_words(value)[1:]}"
        for name, value in weight.items()
        if value is not None
    ]
    return f", weights: {'; '.join(said)}" if said else ""


def lag_to_dict(lag: Any) -> dict:
    """The lag as a saved model holds it (nothing for none)."""
    if isinstance(lag, Mapping):
        if not any(v > 0 for v in lag.values()):
            return {}
        return {"lag": {k: float(v) for k, v in lag.items()}}
    return {} if lag == 0 else {"lag": float(lag)}


def weight_to_dict(weight: Any) -> dict:
    """The weight as a saved model holds it (nothing for none)."""
    if weight is None:
        return {}
    if isinstance(weight, Mapping):
        kept = {k: [v[0], float(v[1])] for k, v in weight.items() if v}
        return {"weight": kept} if kept else {}
    return {"weight": [weight[0], float(weight[1])]}


def marker_random_effects(random_effects: Any, varying: tuple) -> Any:
    """``random_effects`` with ``"by_marker"`` (each marker's varying
    parameters correlated with each other, uncorrelated with another
    marker's) written out as its blocks; anything else as given."""
    if isinstance(random_effects, str) and random_effects == BY_MARKER:
        blocks: dict = {}
        for name in varying:
            blocks.setdefault(name.split(".", 1)[0], []).append(name)
        return [tuple(block) for block in blocks.values()]
    return random_effects


def check_marker_dimension(varying: tuple) -> None:
    """Refuse more varying parameters in all than
    :data:`MAX_MARKER_VARYING`."""
    q = len(varying)
    if q > MAX_MARKER_VARYING:
        raise ValueError(
            f"{q} path parameters vary between items over the markers "
            f"({', '.join(varying)}), and a joint model of several "
            f"markers takes up to {MAX_MARKER_VARYING}: each item's "
            "integral over them is Laplace's approximation beyond five, "
            "and their covariance has q (q + 1) / 2 parameters, too many "
            "to estimate from a usual fleet beyond that. Name fewer in "
            "varying= (a parameter common to all items has no spread), "
            "for example only each marker's level."
        )


# -- the readings -------------------------------------------------------------


def rows_by_marker(markers: Any, names: list) -> list:
    """Each row's readings by marker, ``[{marker: {age: value}}, ...]``,
    from the fit's ``markers``: a list, one ``{marker: {age: value}}``
    per row (a marker with no readings of a row may be left out). A
    measurement not among the model's markers is not used."""
    if markers is None:
        raise ValueError(
            "The joint model of several markers needs the items' readings "
            "of each: a SurpyvalData with named measurements (m={marker: "
            "{unit: {age: value}}}, with i=), a Measurements of named "
            "measurements as m= with each row's unit as i=, or markers=["
            "{marker: {age: value}}, ...], one per row."
        )
    out = []
    for k, row in enumerate(list(markers)):
        if row is None:
            row = {}
        if not isinstance(row, Mapping) or any(
            not isinstance(v, Mapping) for v in row.values()
        ):
            raise ValueError(
                "With several markers, each row's readings are "
                "{marker: {age: value}} (a dict of each marker's readings, "
                f"{names!r}); row {k} has {row!r}."
            )
        unknown = [key for key in row if key not in names]
        if unknown and not getattr(row, "_measurements", False):
            raise ValueError(
                f"Row {k}'s readings name {unknown!r}, which are not "
                f"markers of the model; the markers are {names!r}."
            )
        out.append({name: dict(row.get(name) or {}) for name in names})
    return out


class ByMarker(dict):
    """A row's readings of several named measurements, ``{name: {age:
    value}}``, from a SurpyvalData or a Measurements: a model of several
    markers takes those it names; any other joint fit refuses them."""

    _measurements = True


def checked_readings(record: Any, row: dict, names: list) -> list:
    """``[(ages, values), ...]``, one marker each in ``names``' order, of
    the record's row ``row``, each marker's readings checked against the
    item's observation (as an ``ItemLifetime``'s markers are)."""
    out = []
    for name in names:
        readings = row.get(name) or {}
        if readings:
            try:
                checked = record.clone(markers=readings).markers or {}
            except ValueError as error:
                raise ValueError(f"Marker {name!r}: {error}") from None
        else:
            checked = {}
        out.append(
            (
                np.array(list(checked.keys()), dtype=float),
                np.array(list(checked.values()), dtype=float),
            )
        )
    return out


def joined(readings: list) -> tuple:
    """``(ages, values, channels)`` of one item's readings by marker
    (:func:`checked_readings`), every marker's together, each reading's
    marker in ``channels``."""
    if not readings:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=int)
    ages = np.concatenate([a for a, _ in readings])
    values = np.concatenate([v for _, v in readings])
    channels = np.concatenate(
        [np.full(len(a), k, dtype=int) for k, (a, _) in enumerate(readings)]
    )
    return ages, values, channels


def marker_centres(readings: list, counts: npt.NDArray, n: int) -> list:
    """Each marker's mean reading (count weighted), the centre of its
    value in the hazard."""
    out = []
    for k in range(n):
        values = [item[k][1] for item in readings]
        weights = np.concatenate(
            [np.full(len(v), w) for v, w in zip(values, counts)]
        )
        every = np.concatenate(values)
        out.append(float(np.average(every, weights=weights)))
    return out


# -- the start ----------------------------------------------------------------


def _marker_layout(layout: Layout, piece: slice) -> Layout:
    """The layout of one marker's path part alone: its parameters and
    their shifts, its varying ones and their blocks (those of the joint
    layout), no links."""
    V = [j for j in layout.varying if piece.start <= j < piece.stop]
    blocks = [layout.blocks[layout.varying.index(j)] for j in V]
    return Layout(
        piece.stop - piece.start,
        layout.p,
        tuple(j - piece.start for j in V),
        layout.shifts[piece],
        layout.hazard,
        (),
        layout.survival,
        blocks,
    )


def marker_part_start(
    model: Model,
    arrays: JointArrays,
    readings: list,
    Z: npt.NDArray,
    w: npt.NDArray,
    settled: bool = False,
) -> dict:
    """The start of the marker part (``mu``, ``Delta``, ``L``,
    ``sigma``) of several markers: each marker's own start, from its
    readings alone (:func:`~._markers.marker_start`, or with ``settled``
    or more than ``MAX_VARYING`` varying parameters of its own
    :func:`~._markers.settled_marker_start`), side by side, uncorrelated
    across the markers. A marker with no varying parameter has its
    common ones' start."""
    from ._likelihood import MAX_VARYING
    from ._markers import settled_marker_start

    layout = model.layout
    path = model.path
    mus, deltas, sigmas = [], [], []
    roots = []
    for k, (part, piece) in enumerate(zip(path.parts, path.slices)):
        sub = _marker_layout(layout, piece)
        one = Model(sub, PathFunction(part.model), 0.0, model.integration)
        ages = [item[k][0] for item in readings]
        values = [item[k][1] for item in readings]
        if sub.q and (settled or sub.q > MAX_VARYING):
            own = JointArrays.build(
                arrays.w,
                arrays.T,
                np.zeros(len(arrays.w)),
                np.zeros(len(arrays.w)),
                arrays.Z,
                ages,
                values,
            )
            theta = settled_marker_start(one, own, ages, values)
        else:
            theta = _start_without_spread(one, ages, values, Z, w)
        mus.append(np.asarray(theta["mu"], dtype=float))
        deltas.append(np.asarray(theta["Delta"], dtype=float))
        roots.append(np.atleast_2d(np.asarray(theta["L"], dtype=float)))
        sigmas.append(float(theta["sigma"]))
    q = layout.q
    L = np.zeros((q, q))
    at = 0
    for root in roots:
        size = root.shape[0] if root.size else 0
        L[at : at + size, at : at + size] = root
        at += size
    return {
        "mu": np.concatenate(mus),
        "Delta": np.vstack(deltas).reshape(layout.P, layout.p),
        "L": L,
        "sigma": np.array(sigmas),
    }


def _start_without_spread(
    model: Model, ages: list, values: list, Z: npt.NDArray, w: npt.NDArray
) -> dict:
    """:func:`~._markers.marker_start`, for a marker with no varying
    parameter too (its spread is then empty)."""
    from ._markers import marker_start

    if model.layout.q:
        return marker_start(model, ages, values, Z, w)
    from ._markers import per_item_fits

    _, fits, _, sigma2 = per_item_fits(model.path.model, ages, values)
    return {
        "mu": np.median(fits, axis=0),
        "Delta": np.zeros((model.layout.P, model.layout.p)),
        "L": np.zeros((0, 0)),
        "sigma": float(np.sqrt(sigma2)),
    }
