"""The registry of model families and the engine behind ``diagnose()``
(#867).

A model family is a :class:`Family`: what it applies to, the checks that
test its assumptions (:class:`Check`), the rules that turn their failures
into advice (:class:`AdviceRule`) and anything it must refuse. It is
registered with :func:`register` from its own module, listed in
:data:`FAMILY_MODULES`. :func:`diagnose` finds the family of a model, runs
its checks and builds the :class:`~surpyval.diagnostics.report.Diagnosis`;
nothing here knows any one family. The guide to adding one is
:mod:`surpyval.diagnostics`'s docstring.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple, Sequence

from .report import COLUMNS, VERDICTS, Advice, Diagnosis, make_table, verdict

#: The modules that define and register a family, imported (once) the
#: first time a model is diagnosed. A new family adds its module here.
FAMILY_MODULES: tuple[str, ...] = (
    "surpyval.diagnostics.cox",
    "surpyval.diagnostics.ph_aft",
)


class PlotOptions(NamedTuple):
    """What a panel's ``draw`` receives besides its axes: the report's
    ``alpha`` (a band's level is ``1 - alpha``), the number of bootstrap
    resamples ``n_boot`` and ``rng``, the ``numpy.random.Generator`` to
    draw them from (from ``plot``'s ``random_state``)."""

    alpha: float
    n_boot: int
    rng: Any


@dataclass(frozen=True)
class Panel:
    """One panel of a report's figure: its ``title`` and ``draw(ax,
    options)``, which draws on the matplotlib axes ``ax`` (with
    :class:`PlotOptions` ``options``). The axes get SurPyval's theme and
    the title after ``draw``."""

    title: str
    draw: Callable[[Any, PlotOptions], None]


@dataclass
class Context:
    """What a family's checks, panels, rules and notes are given.

    ``model`` is the fitted model, ``alpha`` the level, ``names`` the
    names of all the model's covariates (as its ``summary()`` names its
    coefficients), ``columns`` the positions in ``names`` of the ones to
    check (all of them unless ``diagnose(columns=...)`` chose), and
    ``notes`` a list a check may append to. :meth:`cached` computes a
    quantity once per report (residuals that several checks and the plots
    share).
    """

    model: Any
    alpha: float
    names: list[str]
    columns: list[int]
    notes: list[str] = field(default_factory=list)
    _cache: dict = field(default_factory=dict, repr=False)

    def cached(self, key: str, compute: Callable[[], Any]) -> Any:
        """``compute()``, computed the first time ``key`` is asked for and
        kept for the rest of the report."""
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    @property
    def checked(self) -> list[tuple[int, str]]:
        """``(position, name)`` of each covariate to check."""
        return [(j, self.names[j]) for j in self.columns]


@dataclass(frozen=True)
class Check:
    """
    One check of a family: the assumption or property it tests and how.

    ``name`` is the row's ``check`` in the table and the name
    ``Diagnosis.plot(which=...)`` takes (snake case, e.g.
    ``"proportional_hazards"``); ``assumption`` says in a sentence what
    holds when the check passes. ``rows(ctx)`` returns the table's rows
    for it, dicts with the keys ``test``, ``covariates`` (a sequence of
    names), ``statistic``, ``df``, ``p``, ``detail`` and the family's own
    columns; a row without a ``verdict`` gets :func:`verdict` of its
    ``p``, one with a ``verdict`` keeps it (a screen that is not a test).
    ``panels(ctx)`` returns the check's :class:`Panel` list, or is
    ``None`` for a check with nothing to draw. A check with ``rows=None``
    is drawn only (it has no row in the table).
    """

    name: str
    assumption: str
    rows: "Callable[[Context], list[dict]] | None" = None
    panels: "Callable[[Context], list[Panel]] | None" = None


@dataclass(frozen=True)
class AdviceRule:
    """
    A rule that turns a check's result into advice.

    For every row of the check ``check`` whose verdict is in ``on``
    (``("fail",)`` by default) and for which ``when(row, ctx)`` is true,
    the advice is ``message(row, ctx)``. ``features`` lists the SurPyval
    features the message tells the user to use, spelt as the API spells
    them -- ``"CoxPH.fit(strata=)"`` for an argument, ``"CoxPH.fit_tvc"``
    for a method -- relative to ``surpyval`` or, for a method of the
    fitted model, to its class (``"summary(robust=)"``); the tests resolve
    each one, so a rule cannot advise a feature that does not exist. A
    rule with ``check=None`` is about the model, not a row: it is asked
    once, with ``row=None``. The same message is given once, however many
    rows call for it.
    """

    check: "str | None"
    message: Callable[[Any, Context], str]
    features: tuple[str, ...]
    when: Callable[[Any, Context], bool] = lambda row, ctx: True
    on: tuple[str, ...] = ("fail",)


@dataclass(frozen=True)
class Family:
    """
    A model family's diagnostics.

    ``name`` heads the report (``"Cox proportional hazards"``) and
    ``issue`` is its tracking issue. ``applies(model)`` says whether a
    fitted model belongs to the family (the first registered family that
    applies is used). ``validate(model)`` raises, with a message naming
    the fix, for a model of the family that the checks cannot handle (a
    model restored without its data, one with no finite maximum, an
    option the checks do not support yet); ``None`` refuses nothing.
    ``covariates(model)`` gives the covariate names (default: as the
    model's ``summary()`` names its coefficients). ``checks`` and
    ``advice`` are the :class:`Check` and :class:`AdviceRule` lists;
    ``notes(ctx)`` the notes every report of the family carries;
    ``columns`` the family's own table columns, after ``check``.
    """

    name: str
    issue: int
    applies: Callable[[Any], bool]
    checks: tuple[Check, ...]
    advice: tuple[AdviceRule, ...] = ()
    validate: "Callable[[Any], None] | None" = None
    covariates: "Callable[[Any], list[str]] | None" = None
    notes: "Callable[[Context], list[str]] | None" = None
    columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class Pending:
    """A family not built yet: ``diagnose()`` on a model it ``applies``
    to refuses, naming the family (``label``) and its tracking
    ``issue``."""

    label: str
    issue: int
    applies: Callable[[Any], bool]


_FAMILIES: list[Family] = []
_LOADED: list[bool] = [False]


def register(family: Family) -> Family:
    """Add ``family`` to the registry (from its module, at import); a
    family of the same name is replaced. Returns ``family``.

    Examples
    --------
    >>> from surpyval.diagnostics.registry import families
    >>> "Cox proportional hazards" in [f.name for f in families()]
    True
    """
    names = {c.name for c in family.checks}
    if len(names) != len(family.checks):
        raise ValueError(
            "Family {!r} has two checks of one name; give each check its "
            "own name.".format(family.name)
        )
    clash = set(family.columns) & set(COLUMNS)
    if clash:
        raise ValueError(
            "Family {!r}'s own columns {} clash with the report's; name "
            "them otherwise.".format(family.name, sorted(clash))
        )
    for rule in family.advice:
        if rule.check is not None and rule.check not in names:
            raise ValueError(
                "An advice rule of family {!r} refers to the check {!r}, "
                "which the family does not have.".format(
                    family.name, rule.check
                )
            )
        bad = set(rule.on) - set(VERDICTS)
        if bad:
            raise ValueError(
                "An advice rule's 'on' must be among {}; got {}.".format(
                    VERDICTS, sorted(bad)
                )
            )
    _FAMILIES[:] = [f for f in _FAMILIES if f.name != family.name]
    _FAMILIES.append(family)
    return family


def families() -> list[Family]:
    """The registered families, after importing :data:`FAMILY_MODULES`."""
    if not _LOADED[0]:
        for module in FAMILY_MODULES:
            importlib.import_module(module)
        _LOADED[0] = True
    return list(_FAMILIES)


def _instance_of(path: str, **attributes: Any) -> Callable[[Any], bool]:
    """A test of whether a model is an instance of the class at ``path``
    (imported only when asked) whose attributes take the given values
    (a tuple is any of its values)."""
    module, _, name = path.rpartition(".")

    def applies(model: Any) -> bool:
        cls = getattr(importlib.import_module(module), name)
        if not isinstance(model, cls):
            return False
        for key, wanted in attributes.items():
            value = getattr(model, key, None)
            if isinstance(wanted, tuple):
                if value not in wanted:
                    return False
            elif value != wanted:
                return False
        return True

    return applies


_REG = "surpyval.univariate.regression."
_CR = "surpyval.univariate.competing_risks."

#: The families not yet built, each with its tracking issue: their models
#: have ``diagnose()``, which refuses until the family is registered (a
#: registered family that applies is found first).
PENDING: tuple[Pending, ...] = (
    Pending(
        "proportional odds and additive hazards models",
        870,
        _instance_of(
            _REG + "ParametricRegressionModel",
            kind=("Proportional Odds", "Additive Hazard"),
        ),
    ),
    Pending(
        "proportional odds and additive hazards models",
        870,
        _instance_of(_REG + "ProportionalOddsModel"),
    ),
    Pending(
        "proportional odds and additive hazards models",
        870,
        _instance_of(_REG + "AdditiveHazardsModel"),
    ),
    Pending(
        "accelerated life (life-stress) models",
        871,
        _instance_of(
            _REG + "ParametricRegressionModel", kind="Accelerated Life"
        ),
    ),
    Pending("frailty models", 872, _instance_of(_REG + "FrailtyModel")),
    Pending("frailty models", 872, _instance_of(_REG + "CoxFrailtyModel")),
    Pending(
        "recurrent-event models",
        873,
        _instance_of(
            "surpyval.recurrent.parametric.parametric_recurrence."
            "ParametricRecurrenceModel"
        ),
    ),
    Pending(
        "recurrent-event models",
        873,
        _instance_of("surpyval.recurrent.RenewalModel"),
    ),
    Pending(
        "recurrent-event models",
        873,
        _instance_of(
            "surpyval.recurrent.regression.proportional_intensity."
            "ProportionalIntensityModel"
        ),
    ),
    Pending(
        "competing-risks models",
        874,
        _instance_of(_CR + "CompetingRisksProportionalHazards"),
    ),
    Pending(
        "competing-risks models",
        874,
        _instance_of(_CR + "regression.fine_gray.FineGrayModel"),
    ),
    Pending(
        "competing-risks models",
        874,
        _instance_of(_CR + "ParametricCompetingRisks"),
    ),
    Pending(
        "univariate distributions",
        875,
        _instance_of("surpyval.univariate.parametric.Parametric"),
    ),
)


def model_label(model: Any) -> str:
    """How a report names the kind of model: its class, with the family
    of a parametric regression (``ParametricRegressionModel (Accelerated
    Failure Time)``); a Cox model is ``CoxPH``, the fitter that made it."""
    label = type(model).__name__
    kind = getattr(model, "kind", None)
    if label == "ParametricRegressionModel" and isinstance(kind, str):
        return "{} ({})".format(label, kind)
    if label == "SemiParametricRegressionModel":
        return "CoxPH"
    return label


def family_for(model: Any) -> Family:
    """The registered family of ``model``; raises ``NotImplementedError``
    naming the tracking issue for a model whose family is not built yet.

    Examples
    --------
    >>> from surpyval import CoxPH
    >>> from surpyval.diagnostics.registry import family_for
    >>> model = CoxPH.fit([1, 2, 3, 4, 5], [0, 1, 0, 1, 1])
    >>> family_for(model).name
    'Cox proportional hazards'
    """
    for family in families():
        if family.applies(model):
            return family
    label = model_label(model)
    for pending in PENDING:
        if pending.applies(model):
            raise NotImplementedError(
                "diagnose() is not yet available for {}: the diagnostics "
                "of {} are tracked in #{}.".format(
                    label, pending.label, pending.issue
                )
            )
    raise NotImplementedError(
        "diagnose() is not yet available for {}; tracked in #867.".format(
            label
        )
    )


def _check_alpha(alpha: Any) -> float:
    try:
        value = float(alpha)
    except (TypeError, ValueError):
        value = float("nan")
    if not 0 < value < 1:
        raise ValueError(
            "'alpha' must be strictly between 0 and 1 (the level each "
            "check's test is judged at, e.g. 0.05); got {!r}.".format(alpha)
        )
    return value


def _resolve_columns(columns: Any, names: list[str]) -> list[int]:
    if columns is None:
        return list(range(len(names)))
    if isinstance(columns, str):
        columns = [columns]
    try:
        asked = list(columns)
    except TypeError:
        asked = [columns]
    positions = []
    for name in asked:
        if not isinstance(name, str) or name not in names:
            raise ValueError(
                "'columns' must name the model's covariates, as its "
                "summary() names them: some of {}; got {!r}.".format(
                    names, name
                )
            )
        if names.index(name) not in positions:
            positions.append(names.index(name))
    if not positions:
        raise ValueError(
            "'columns' is empty; name at least one of {}, or leave it out "
            "to check them all.".format(names)
        )
    return sorted(positions)


def _covariate_names(model: Any) -> list[str]:
    from surpyval.univariate.regression._summary import coefficient_names

    params: Any = getattr(model, "beta", None)
    if params is None:
        params = getattr(model, "params", ())
    return list(coefficient_names(model, len(params)))


def diagnose(
    model: Any, alpha: float = 0.05, columns: "Sequence[str] | None" = None
) -> Diagnosis:
    """
    Run the checks of ``model``'s family and report: the engine behind
    every model's ``diagnose()`` (see the method for the arguments).

    Examples
    --------
    >>> from surpyval import CoxPH
    >>> from surpyval.datasets import load_rossi_static
    >>> from surpyval.diagnostics import diagnose
    >>> df = load_rossi_static()
    >>> x, c = df["week"].values, 1 - df["arrest"].values
    >>> diagnose(CoxPH.fit(x, df[["fin", "prio"]].values, c=c)).ok
    True
    """
    alpha = _check_alpha(alpha)
    family = family_for(model)
    if family.validate is not None:
        family.validate(model)
    names = (family.covariates or _covariate_names)(model)
    ctx = Context(model, alpha, names, _resolve_columns(columns, names))

    rows: list[dict] = []
    for check in family.checks:
        if check.rows is None:
            continue
        for row in check.rows(ctx):
            row = dict(row, check=check.name)
            row["covariates"] = tuple(row.get("covariates") or ())
            if row.get("verdict") is None:
                row["verdict"] = verdict(row.get("p"), alpha)
            elif row["verdict"] not in VERDICTS:
                raise ValueError(
                    "Check {!r} gave the verdict {!r}; a verdict is one of "
                    "{}.".format(check.name, row["verdict"], VERDICTS)
                )
            rows.append(row)

    advice: list[Advice] = []
    seen: set[str] = set()
    for rule in family.advice:
        if rule.check is None:
            candidates: list[Any] = [None]
        else:
            candidates = [
                r
                for r in rows
                if r["check"] == rule.check and r["verdict"] in rule.on
            ]
        for row in candidates:
            if not rule.when(row, ctx):
                continue
            message = rule.message(row, ctx)
            if message in seen:
                continue
            seen.add(message)
            advice.append(
                Advice(
                    message,
                    rule.check,
                    () if row is None else row["covariates"],
                    rule.features,
                )
            )

    notes = list(family.notes(ctx)) if family.notes is not None else []
    notes.extend(n for n in ctx.notes if n not in notes)
    return Diagnosis(
        make_table(rows, family.columns),
        advice,
        notes,
        alpha,
        family.name,
        model_label(model),
        [names[j] for j in ctx.columns],
        context=ctx,
        family_spec=family,
    )


def plot_report(
    family: Family,
    ctx: Context,
    which: "str | Sequence[str] | None",
    n_boot: int = 200,
    random_state: Any = None,
) -> Any:
    """Draw the panels of ``family``'s checks (those named in ``which``,
    or all) on one figure, three to a row; the engine behind
    :meth:`Diagnosis.plot`."""
    drawable = [c.name for c in family.checks if c.panels is not None]
    if not drawable:
        raise ValueError(
            "The {} diagnostics have no plots to draw.".format(family.name)
        )
    if which is None:
        chosen = drawable
    else:
        chosen = [which] if isinstance(which, str) else list(which)
        unknown = [w for w in chosen if w not in drawable]
        if unknown or not chosen:
            raise ValueError(
                "'which' must name checks this report can draw, some of "
                "{}; got {!r}.".format(drawable, which)
            )
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 0:
        raise ValueError(
            "'n_boot' must be a whole number of resamples, 0 or more (0 "
            "draws no bootstrap band); got {!r}.".format(n_boot)
        )
    panels: list[Panel] = []
    for check in family.checks:
        if check.name in chosen and check.panels is not None:
            panels.extend(check.panels(ctx))

    import matplotlib.pyplot as plt

    from surpyval.plotting import style_axes
    from surpyval.utils.rng import as_generator

    options = PlotOptions(ctx.alpha, n_boot, as_generator(random_state))
    ncols = max(1, min(3, len(panels)))
    nrows = max(1, -(-len(panels) // ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.4 * ncols, 3.4 * nrows), squeeze=False
    )
    flat = list(axes.flat)
    for ax, panel in zip(flat, panels):
        panel.draw(ax, options)
        style_axes(ax)
        ax.set_title(panel.title)
    for ax in flat[len(panels) :]:
        fig.delaxes(ax)
    fig.tight_layout()
    return fig
