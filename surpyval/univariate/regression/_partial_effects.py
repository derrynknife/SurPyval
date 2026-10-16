"""Partial-effects plots (#876): what a fitted regression says about one
covariate.

``model.plot_partial_effects(covariate, values)`` draws the model's
survival curve (or ``ff``, ``Hf``; a competing-risks model's cumulative
incidence) once for each of ``values`` of one covariate, the other
covariates held at reference values, in the house plot theme
(:mod:`surpyval.plotting`). It shows what the model says the covariate
does; it does not check the model against the data.

The work is done here once, for every model that offers the plot: the
covariates are read as the model names them (by position for a fit on
arrays, by column for named covariates, by the formula's variables for a
formula), the rows are built, and each curve is the model's own function
at its row. A model says only how to evaluate a curve and a band, which
rows it was fitted to, and the span of its times, through the hooks of
:class:`PartialEffectsMixin`.

matplotlib is imported only when a plot is drawn (#470).
"""

from __future__ import annotations

import numbers
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from formulaic.parser.types import Factor  # type: ignore[import-untyped]

from surpyval import plotting
from surpyval.utils.units import label_axes
from surpyval.utils.validation import check_alpha_ci

if TYPE_CHECKING:
    from matplotlib.axes import Axes

#: The number of points of the default time grid of a smooth model.
GRID_POINTS = 200

#: The y-axis label of each function drawn.
_YLABELS = {
    "sf": "Survival probability",
    "ff": "Failure probability",
    "Hf": "Cumulative hazard",
    "cif": "Cumulative incidence",
}


class PartialEffectsMixin:
    """``plot_partial_effects`` for the Cox and parametric regression
    models. A host class provides the hooks below where the defaults do
    not fit it."""

    #: The functions ``on=`` can draw.
    _PARTIAL_EFFECTS_ON: tuple[str, ...] = ("sf", "ff", "Hf")
    #: Whether the model's curves are step functions (drawn as steps).
    _PARTIAL_EFFECTS_STEP = False

    def plot_partial_effects(
        self,
        covariate: Any,
        values: Any,
        *,
        reference: "Mapping | None" = None,
        x: "npt.ArrayLike | None" = None,
        on: str = "sf",
        plot_bounds: bool = False,
        alpha_ci: float = 0.05,
        ax: "Axes | None" = None,
    ) -> "Axes":
        r"""
        Draw the model's survival curve at each of ``values`` of one
        covariate, the other covariates held at reference values: a
        partial-effects plot, showing what the model says the covariate
        does.

        Parameters
        ----------
        covariate : str or int
            The covariate to vary: a column name for a model fitted with
            named covariates (``fit_from_df(Z_cols=...)``, a DataFrame
            ``Z``); for a formula, a *variable* of the formula (``"load"``
            for ``np.log(load)``, ``"site"`` for ``C(site)``), which the
            formula then transforms, expands into its levels' columns and
            multiplies into its interactions; for a fit on arrays, the
            column's position in ``Z`` (0, 1, ...).
        values : array like
            The covariate's values, one curve each, drawn in this order:
            numbers, or, for a categorical variable of a formula, its
            levels.
        reference : dict, optional
            ``{covariate: value}`` for any of the other covariates, named
            as ``covariate`` is. Those not given are held at:

            - for a categorical variable of a formula, its reference
              level: the level whose columns are all 0 (the first level,
              or the base of ``contr.treatment(...)``);
            - otherwise, the mean of the fitted rows, weighted by their
              counts ``n`` (for a fit with ``center=True``, the
              ``center``), a 0/1 indicator included.

            A mean can be read only from a column of the design matrix
            the covariate fills by itself. A variable that enters a
            formula only transformed (``np.log(load)``) or in an
            interaction (``load:temp``), a categorical variable without
            a level coded as all zeros (``contr.sum``), and every
            covariate of a model that does not keep the rows it was
            fitted to (a model restored by ``from_dict``, a
            competing-risks regression) or whose rows are not units (a
            time-varying-covariate fit) is refused unless given here. A
            value for ``covariate`` itself is ignored, so one mapping can
            serve plots of several covariates.
        x : array like, optional
            The times, a 1-D array, at which each curve is evaluated and
            drawn, in the order given. Default: from 0 (or the earliest
            time, if before 0) to the latest fitted time, at 200 evenly
            spaced points for a parametric model, and at every distinct
            fitted time for the Cox model, whose curves are drawn as
            steps.
        on : {'sf', 'ff', 'Hf'}, optional
            The function drawn: survival (the default), failure
            probability or cumulative hazard.
        plot_bounds : bool, optional
            Draw each curve's two-sided confidence band, from the model's
            ``cb`` (its default Wald method, from the fit's covariance),
            as a pale wash of the curve's colour. Default False. A model
            without ``cb`` (the Cox model) refuses it.
        alpha_ci : float, optional
            Total tail probability of the bands. Default 0.05.
        ax : matplotlib axes, optional
            The axes to draw on; default the current axes
            (``plt.gca()``).

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn on. Each curve is a line labelled
            ``"covariate = value"`` (a fit on arrays names column ``j``
            ``Z[:, j]``) whose data are ``x`` and the model's ``on`` at
            the curve's row, ``getattr(model, on)(x, row)``.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from surpyval import CoxPH
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> df["c"] = 1 - df["arrest"]
        >>> model = CoxPH.fit_from_df(
        ...     df, x_col="week", c_col="c", Z_cols=["fin", "age", "prio"]
        ... )
        >>> ax = model.plot_partial_effects("prio", [0, 3, 8])
        >>> [line.get_label() for line in ax.get_lines()]
        ['prio = 0', 'prio = 3', 'prio = 8']

        The last point of each curve is the chance of no arrest in the
        year, for ``fin`` and ``age`` at their means:

        >>> [round(float(line.get_ydata()[-1]), 4) for line in ax.get_lines()]
        [0.8121, 0.7571, 0.6365]
        >>> plt.close(ax.figure)
        """
        return draw_partial_effects(
            self,
            covariate,
            values,
            reference=reference,
            x=x,
            on=on,
            event=None,
            plot_bounds=plot_bounds,
            alpha_ci=alpha_ci,
            ax=ax,
        )

    # -- hooks -------------------------------------------------------------

    def _partial_effects_refusal(self) -> "str | None":
        """Why the model cannot draw the plot, or None."""
        return None

    def _partial_effects_width(self) -> int:
        """The number of columns of ``Z`` (the design matrix)."""
        raise NotImplementedError

    def _partial_effects_rows(
        self,
    ) -> "tuple[npt.NDArray, npt.NDArray] | str":
        """The design rows the model was fitted to and their counts, or,
        where it has none to give, why not."""
        return "it does not keep the covariates it was fitted to"

    def _partial_effects_times(self) -> "npt.NDArray | None":
        """The times the default grid spans: the fitted times of a smooth
        model, the jump times (and the fitted times) of a step one; None
        when the model has none."""
        return None

    def _partial_effects_curve(
        self, on: str, x: npt.NDArray, row: npt.NDArray, event: Any
    ) -> npt.NDArray:
        """The function ``on`` at the times ``x`` for one design row."""
        return np.asarray(getattr(self, on)(x, row), dtype=float)

    def _partial_effects_band(
        self,
        on: str,
        x: npt.NDArray,
        row: npt.NDArray,
        alpha_ci: float,
        event: Any,
    ) -> npt.NDArray:
        """The two-sided band on ``on`` at ``x`` for one design row, an
        array of ``[lower, upper]`` rows."""
        cb = getattr(self, "cb", None)
        if cb is None:
            raise ValueError(
                "plot_bounds=True needs the model's confidence bounds on "
                "its predictions (cb), which {} does not have. Plot "
                "without bounds (plot_bounds=False).".format(
                    type(self).__name__
                )
            )
        return np.asarray(cb(x, row, on=on, alpha_ci=alpha_ci), dtype=float)


class CompetingRisksPartialEffectsMixin(PartialEffectsMixin):
    """``plot_partial_effects`` for the competing-risks regressions, which
    draw a cause's cumulative incidence (and have no ``cb``)."""

    _PARTIAL_EFFECTS_ON = ("cif",)
    _PARTIAL_EFFECTS_STEP = True

    def plot_partial_effects(  # type: ignore[override]
        self,
        covariate: Any,
        values: Any,
        *,
        event: Any = None,
        reference: "Mapping | None" = None,
        x: "npt.ArrayLike | None" = None,
        ax: "Axes | None" = None,
    ) -> "Axes":
        r"""
        Draw the cumulative incidence of cause ``event`` at each of
        ``values`` of one covariate, the other covariates held at
        reference values: a partial-effects plot.

        As the regression models' ``plot_partial_effects``, with the
        cumulative incidence (``cif``) in place of the survival curve.
        The model does not keep the covariates it was fitted to, so every
        other covariate needs its value in ``reference`` (a categorical
        variable of a formula excepted: it is held at its reference
        level). There are no confidence bands: the model has no ``cb``.

        Parameters
        ----------
        covariate : str or int
            The covariate to vary, named as the model names it (a column,
            a variable of the formula, or a position for a fit on
            arrays).
        values : array like
            Its values, one curve each.
        event : cause label
            The cause whose cumulative incidence is drawn: one of the
            fitted causes. A Fine-Gray model is of one cause, which is
            the default.
        reference : dict, optional
            ``{covariate: value}`` for the other covariates.
        x : array like, optional
            The times, a 1-D array. Default: from 0 to the last event
            time, at every event time.
        ax : matplotlib axes, optional
            The axes to draw on; default the current axes.

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn on, each curve labelled ``"covariate = value"``.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from surpyval import CompetingRisksProportionalHazards as CRPH
        >>> rng = np.random.default_rng(1)
        >>> Z = rng.uniform(0, 1, (200, 1))
        >>> x = rng.exponential(10 * np.exp(-Z[:, 0]))
        >>> e = rng.choice(["wear", "shock"], 200)
        >>> model = CRPH.fit(x, Z, e)
        >>> ax = model.plot_partial_effects(0, [0.0, 1.0], event="wear")
        >>> [line.get_label() for line in ax.get_lines()]
        ['Z[:, 0] = 0', 'Z[:, 0] = 1']
        >>> plt.close(ax.figure)
        """
        return draw_partial_effects(
            self,
            covariate,
            values,
            reference=reference,
            x=x,
            on="cif",
            event=event,
            plot_bounds=False,
            alpha_ci=0.05,
            ax=ax,
        )


# ---------------------------------------------------------------------------
# The plot
# ---------------------------------------------------------------------------
def draw_partial_effects(
    model: Any,
    covariate: Any,
    values: Any,
    *,
    reference: "Mapping | None",
    x: "npt.ArrayLike | None",
    on: str,
    event: Any,
    plot_bounds: bool,
    alpha_ci: float,
    ax: "Axes | None",
) -> "Axes":
    """Draw ``model``'s partial-effects plot (see
    :meth:`PartialEffectsMixin.plot_partial_effects`). Every curve and
    band is computed before anything is drawn, so a refusal leaves the
    axes as they were."""
    refusal = model._partial_effects_refusal()
    if refusal is not None:
        raise ValueError(refusal)
    allowed = model._PARTIAL_EFFECTS_ON
    if on not in allowed:
        raise ValueError(
            "on={!r} is not a function plot_partial_effects draws; use one "
            "of {}.".format(on, list(allowed))
        )
    if plot_bounds:
        check_alpha_ci(alpha_ci)
    values = _values(values)
    covariates = _Covariates(model)
    label = covariates.label(covariates.resolve(covariate, "covariate"))
    rows = covariates.rows(covariate, values, reference)
    times = _grid(model, x)

    curves = []
    for row in rows:
        row = row[None, :]
        y = model._partial_effects_curve(on, times, row, event)
        band = None
        if plot_bounds:
            band = model._partial_effects_band(on, times, row, alpha_ci, event)
        curves.append((y, band))

    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.gca()
    themed = plotting.theme_enabled()
    step = bool(model._PARTIAL_EFFECTS_STEP)
    band_label = "{:g}% confidence bands".format((1 - alpha_ci) * 100)
    for i, (value, (y, band)) in enumerate(zip(values, curves)):
        (line,) = ax.plot(
            times,
            y,
            color=plotting.next_color(ax),
            linewidth=plotting.LINE_WIDTH if themed else None,
            drawstyle="steps-post" if step else "default",
            label="{} = {}".format(label, _format_value(value)),
        )
        if band is not None:
            ax.fill_between(
                times,
                band[:, 0],
                band[:, 1],
                color=line.get_color(),
                alpha=plotting.BAND_ALPHA if themed else 0.2,
                linewidth=0,
                step="post" if step else None,
                label=band_label if i == 0 else None,
            )
    ylabel = _YLABELS[on]
    if on == "cif" and event is not None:
        ylabel = "{} of {}".format(ylabel, event)
    label_axes(ax, getattr(model, "unit", None), ylabel)
    plotting.style_axes(ax)
    plotting.style_legend(ax)
    return ax


def _values(values: Any) -> list:
    """The covariate values as a list, one per curve."""
    arr = np.asarray(values, dtype=object)
    if arr.ndim > 1:
        raise ValueError(
            "values must be a 1-D list of the covariate's values, one "
            "curve each; got an array of shape {}.".format(arr.shape)
        )
    out = [_native(v) for v in np.atleast_1d(arr).tolist()]
    if not out:
        raise ValueError(
            "values is empty: give the covariate's values, one curve each."
        )
    return out


def _grid(model: Any, x: Any) -> npt.NDArray:
    """The times the curves are drawn at: ``x``, or the default grid."""
    if x is not None:
        times = np.asarray(x, dtype=float)
        if times.ndim == 0:
            times = times.reshape(1)
        if times.ndim != 1 or times.size == 0:
            raise ValueError(
                "x must be a 1-D array of the times to draw the curves "
                "at; got shape {}.".format(np.shape(x))
            )
        return times
    span = model._partial_effects_times()
    if span is None:
        raise ValueError(
            "This model does not keep the times it was fitted to, so "
            "there is no default time grid: pass the times to draw at, "
            "x=np.linspace(0, t_max, 200)."
        )
    span = np.asarray(span, dtype=float)
    span = span[np.isfinite(span)]
    if span.size == 0:
        raise ValueError(
            "The model has no finite times to span, so there is no default "
            "time grid: pass the times to draw at, x=..."
        )
    lo = min(0.0, float(span.min()))
    hi = float(span.max())
    if model._PARTIAL_EFFECTS_STEP:
        return np.unique(np.concatenate([[lo], span, [hi]]))
    return np.linspace(lo, hi, GRID_POINTS)


def _native(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _format_value(value: Any) -> str:
    """A covariate value as a legend shows it: a number by ``:g``."""
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return "{:g}".format(float(value))
    return str(value)


# ---------------------------------------------------------------------------
# The covariates, as the model names them
# ---------------------------------------------------------------------------
class _Covariates:
    """The covariates of ``model`` as ``plot_partial_effects`` takes
    them, and the design rows of a partial-effects plot.

    Three layouts: a formula's variables (``_model_spec``), the named
    columns of ``Z`` (``feature_names``), or the positions of the columns
    of an array ``Z``.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.spec = getattr(model, "_model_spec", None)
        names = getattr(model, "feature_names", None)
        self.columns = None if names is None else [str(n) for n in names]
        self.width = int(model._partial_effects_width())
        if self.spec is not None:
            self.layout = "formula"
            self.names: list = _formula_variables(self.spec)
        elif self.columns is not None:
            self.layout = "named"
            self.names = list(self.columns)
        else:
            self.layout = "positional"
            self.names = list(range(self.width))

    # -- names ------------------------------------------------------------

    def resolve(self, key: Any, what: str) -> Any:
        """``key`` as one of :attr:`names`, or a ValueError naming the
        covariates and how to give one."""
        if self.layout == "positional":
            if isinstance(key, (int, np.integer)) and not isinstance(
                key, (bool, np.bool_)
            ):
                if 0 <= int(key) < self.width:
                    return int(key)
                raise ValueError(
                    "{} {} is not a column of Z: this model has {} "
                    "covariate column(s), positions 0 to {}.".format(
                        what, key, self.width, self.width - 1
                    )
                )
            raise ValueError(
                "{} {!r}: this model's covariates have no names (it was "
                "fitted on arrays), so give the column's position in Z, "
                "0 to {}, or fit with named covariates (fit_from_df, a "
                "DataFrame Z).".format(what, key, self.width - 1)
            )
        if isinstance(key, str) and key in self.names:
            return key
        if self.layout == "formula" and isinstance(key, str):
            used = _variables_of(self.spec, key)
            if used:
                raise ValueError(
                    "{} {!r} is a term of the formula {!r}, not one of its "
                    "variables: give the variable{} {} and the formula "
                    "computes {!r} from {}.".format(
                        what,
                        key,
                        _formula_text(self.model),
                        "s" if len(used) > 1 else "",
                        " and ".join(repr(v) for v in used),
                        key,
                        "them" if len(used) > 1 else "it",
                    )
                )
        kind = "variables of its formula" if self.layout == "formula" else ""
        raise ValueError(
            "{} {!r} is not a covariate of this model; its covariates "
            "are {}{}.".format(
                what,
                key,
                self.names,
                " (the {})".format(kind) if kind else "",
            )
        )

    def label(self, name: Any) -> str:
        """The name of a covariate in a legend."""
        return "Z[:, {}]".format(name) if self.layout == "positional" else name

    # -- rows -------------------------------------------------------------

    def rows(
        self, covariate: Any, values: list, reference: "Mapping | None"
    ) -> npt.NDArray:
        """The design rows, one per value of ``covariate``, the others at
        their reference values."""
        name = self.resolve(covariate, "covariate")
        if reference is None:
            reference = {}
        if not isinstance(reference, Mapping):
            raise ValueError(
                "reference must be a dict {{covariate: value}}; got "
                "{}.".format(type(reference).__name__)
            )
        given = {}
        for key, value in reference.items():
            given[self.resolve(key, "reference covariate")] = value
        refs, missing = {}, []
        for other in self.names:
            if other == name:
                continue
            if other in given:
                refs[other] = given[other]
                continue
            value, why = self._default_reference(other)
            if why is not None:
                missing.append((other, why))
            else:
                refs[other] = value
        if missing:
            raise ValueError(
                "No reference value for {}: pass reference={{{}}}.".format(
                    "; ".join(
                        "{} ({})".format(self._shown(n), why)
                        for n, why in missing
                    ),
                    ", ".join(
                        "{}: ...".format(self._quoted(n)) for n, _ in missing
                    ),
                )
            )
        if self.layout == "formula":
            return self._formula_rows(name, values, refs)
        return self._column_rows(name, values, refs)

    def _quoted(self, name: Any) -> str:
        return repr(name) if isinstance(name, str) else str(name)

    def _shown(self, name: Any) -> str:
        """A covariate as a message names it: ``Z[:, j]``, or quoted."""
        return self.label(name) if self.layout == "positional" else repr(name)

    def _column_rows(self, name: Any, values: list, refs: dict) -> Any:
        index = {n: j for j, n in enumerate(self.names)}
        try:
            column = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(
                "The values of {} must be numbers; got {}.".format(
                    self._shown(name), values
                )
            ) from None
        base = np.empty(self.width)
        for other, value in refs.items():
            try:
                base[index[other]] = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    "The reference value of {} must be a number; got "
                    "{!r}.".format(self._shown(other), value)
                ) from None
        rows = np.tile(base, (len(values), 1))
        rows[:, index[name]] = column
        return rows

    def _formula_rows(self, name: str, values: list, refs: dict) -> Any:
        import pandas as pd

        from .regression_data import prepare_Z

        frame = {other: [value] * len(values) for other, value in refs.items()}
        frame[name] = list(values)
        df = pd.DataFrame({v: frame[v] for v in self.names})
        return prepare_Z(df, self.columns, self.spec)

    # -- reference values -------------------------------------------------

    def _default_reference(self, name: Any) -> "tuple[Any, str | None]":
        """A covariate's default reference value, or why there is none."""
        if self.layout == "formula":
            level = _reference_level(self.spec, name)
            if level is not None:
                return level
            column = _bare_column(self.spec, name, self.columns or [])
            if column is None:
                return None, (
                    "it enters the formula only through {}, and the model "
                    "keeps the design matrix, not the variable".format(
                        _terms_with(self.spec, name)
                    )
                )
            j = (self.columns or []).index(column)
        else:
            j = self.names.index(name)
        return self._column_mean(j)

    def _column_mean(self, j: int) -> "tuple[Any, str | None]":
        if getattr(self.model, "is_tvc", False):
            return None, (
                "the rows of a time-varying-covariate fit are intervals, "
                "not units, so their mean is not a unit's covariates"
            )
        fitted = self.model._partial_effects_rows()
        if isinstance(fitted, str):
            return None, fitted
        Z, n = fitted
        Z = np.asarray(Z, dtype=float).reshape(len(n), -1)
        return float(np.average(Z[:, j], weights=n)), None


# ---------------------------------------------------------------------------
# Reading a formula's model spec (formulaic)
# ---------------------------------------------------------------------------
def _formula_text(model: Any) -> str:
    return str(getattr(model, "formula", None) or model._model_spec.formula)


def _data_variables(spec: Any, factor: Any) -> list[str]:
    """The data variables of a formula factor (or its text)."""
    for key, variables in spec.factor_variables.items():
        if str(key) == str(factor):
            return [str(v) for v in variables if v.source == "data"]
    return []


def _formula_variables(spec: Any) -> list[str]:
    """The data variables of a formula, in their order of appearance."""
    out: list[str] = []
    for structure in spec.structure:
        for factor in structure.term.factors:
            for name in _data_variables(spec, factor):
                if name not in out:
                    out.append(name)
    return out


def _variables_of(spec: Any, text: str) -> list[str]:
    """The data variables behind a term, factor or design column named
    ``text`` (empty where it is none of them)."""
    for structure in spec.structure:
        if text == str(structure.term) or text in structure.columns:
            out: list[str] = []
            for factor in structure.term.factors:
                for name in _data_variables(spec, factor):
                    if name not in out:
                        out.append(name)
            return out
    for factor in spec.factor_variables:
        if text == str(factor):
            return _data_variables(spec, factor)
    return []


def _terms_with(spec: Any, name: str) -> str:
    terms = [
        repr(str(s.term))
        for s in spec.structure
        if any(name in _data_variables(spec, f) for f in s.term.factors)
    ]
    return ", ".join(terms) if terms else "no term"


def _main_effect(spec: Any, factor: Any) -> Any:
    """The structure of the term that is ``factor`` alone, or None."""
    for structure in spec.structure:
        factors = list(structure.term.factors)
        if len(factors) == 1 and str(factors[0]) == str(factor):
            return structure
    return None


def _as_is(spec: Any, factor: Any, name: str) -> bool:
    """Whether ``factor`` codes the variable ``name`` as it is: ``name``
    itself or a ``C(name, ...)`` wrapper."""
    expr = str(factor)
    if _data_variables(spec, factor) != [name]:
        return False
    return expr == name or bool(
        re.match(r"C\(\s*" + re.escape(name) + r"\s*[,)]", expr)
    )


def _reference_level(spec: Any, name: str) -> "tuple[Any, str | None] | None":
    """``(level, None)`` for a categorical variable with a level coded as
    all zeros, ``(None, why)`` for a categorical one without, and None
    for a variable that is not categorical."""
    for factor, (kind, state) in spec.encoder_state.items():
        if kind is not Factor.Kind.CATEGORICAL:
            continue
        if not _as_is(spec, factor, name):
            continue
        levels = list(state.get("categories", []))
        contrasts = state.get("contrasts")
        structure = _main_effect(spec, factor)
        why = (
            "the formula codes no level of it as all zeros, so it has no "
            "reference level"
        )
        if (
            contrasts is None
            or structure is None
            or len(structure.columns) != len(levels) - 1
        ):
            return None, why
        coding = contrasts.contrasts.get_coding_matrix(
            levels, reduced_rank=True
        )
        zero = [
            level
            for level, row in zip(levels, np.asarray(coding, dtype=float))
            if not np.any(row)
        ]
        if len(zero) != 1:
            return None, why
        return _native(zero[0]), None
    return None


def _bare_column(spec: Any, name: str, columns: list[str]) -> "str | None":
    """The design column that is the numerical variable ``name`` itself,
    or None."""
    for factor, (kind, _state) in spec.encoder_state.items():
        if kind is not Factor.Kind.NUMERICAL or str(factor) != name:
            continue
        structure = _main_effect(spec, factor)
        if structure is None or len(structure.columns) != 1:
            continue
        if structure.columns[0] in columns:
            return str(structure.columns[0])
    return None
