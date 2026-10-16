"""The report ``model.diagnose()`` returns: :class:`Diagnosis` (#867).

One report for every model family. It holds the table of checks, the
advice and the notes the family's checks produced, and draws the family's
plots on request; it knows nothing about any one family. See
:mod:`surpyval.diagnostics` for the verdicts and for how a family plugs
in.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Iterable, Sequence

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from .registry import Context, Family

#: The verdicts a check can give, from best to worst.
VERDICTS = ("ok", "warn", "fail")

#: The table's columns, in order; a family's own columns (``Family.columns``,
#: e.g. the transform of a proportional-hazards test, or a cause) follow
#: ``check``.
COLUMNS = (
    "check",
    "test",
    "covariates",
    "statistic",
    "df",
    "p",
    "verdict",
    "detail",
)

#: The version of the dictionary :meth:`Diagnosis.to_dict` writes.
SCHEMA = 1


def verdict(p: "float | None", alpha: float) -> str:
    """The verdict of a test with p-value ``p`` at level ``alpha``, the
    one rule every family's tests share:

    - ``"fail"``: ``p < alpha``, evidence against the assumption at the
      chosen level;
    - ``"warn"``: ``alpha <= p < 2 * alpha``, borderline (it would fail at
      twice the level), or ``p`` is ``nan`` -- the test could not be
      computed on these data (the row's ``detail`` says why);
    - ``"ok"``: ``p >= 2 * alpha``.

    A check that is not a test (an influence or outlier screen) sets its
    own verdict, ``"ok"`` or ``"warn"``, by the rule its family documents;
    such a check never fails.

    Examples
    --------
    >>> from surpyval.diagnostics.report import verdict
    >>> verdict(0.01, 0.05), verdict(0.07, 0.05), verdict(0.5, 0.05)
    ('fail', 'warn', 'ok')
    >>> verdict(float("nan"), 0.05)
    'warn'
    """
    if p is None or not math.isfinite(float(p)):
        return "warn"
    if p < alpha:
        return "fail"
    if p < 2 * alpha:
        return "warn"
    return "ok"


class Advice(str):
    """One piece of advice: the plain-English remedy (the string itself),
    with the check and covariates that prompted it and the SurPyval
    features it names (``features``, each spelt as the API spells it,
    e.g. ``"CoxPH.fit(strata=)"``).

    Examples
    --------
    >>> from surpyval.diagnostics.report import Advice
    >>> a = Advice("Stratify on sex.", "proportional_hazards", ("sex",),
    ...            ("CoxPH.fit(strata=)",))
    >>> print(a)
    Stratify on sex.
    >>> a.covariates, a.features
    (('sex',), ('CoxPH.fit(strata=)',))
    """

    check: "str | None"
    covariates: "tuple[str, ...]"
    features: "tuple[str, ...]"

    def __new__(
        cls,
        message: str,
        check: "str | None" = None,
        covariates: Iterable[str] = (),
        features: Iterable[str] = (),
    ) -> "Advice":
        out = super().__new__(cls, message)
        out.check = check
        out.covariates = tuple(covariates)
        out.features = tuple(features)
        return out

    def to_dict(self) -> dict:
        """The advice as a JSON-safe dictionary."""
        return {
            "advice": str(self),
            "check": self.check,
            "covariates": list(self.covariates),
            "features": list(self.features),
        }

    def __reduce__(self) -> tuple:
        return (
            Advice,
            (str(self), self.check, self.covariates, self.features),
        )


class DiagnosticsFailed(Exception):
    """Raised by :meth:`Diagnosis.raise_if_failed` when a check failed;
    ``diagnosis`` is the report.

    Examples
    --------
    >>> from surpyval.diagnostics.report import DiagnosticsFailed
    >>> issubclass(DiagnosticsFailed, Exception)
    True
    """

    def __init__(self, message: str, diagnosis: "Diagnosis") -> None:
        super().__init__(message)
        self.diagnosis = diagnosis


def _json_number(value: Any) -> Any:
    """A float for JSON; ``None`` for a missing (``None``, ``nan``,
    ``pandas.NA``) or non-finite one."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format_cell(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, tuple):
        return ", ".join(str(v) for v in value)
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return "{:.{}g}".format(value, digits)
    text = str(value)
    return "" if text in ("<NA>", "nan") else text


class Diagnosis:
    """
    The report of ``model.diagnose()``: one row per check, the advice its
    failures call for, and the plots.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per check: ``check`` (the assumption or property tested,
        e.g. ``"proportional_hazards"``), the family's own columns (a
        Cox model's ``transform``), ``test`` (the test or screen, by
        name), ``covariates`` (a tuple of the covariates the row is about;
        all of them for a joint test, none for a model-wide one),
        ``statistic``, ``df`` and ``p`` (the test's statistic, its
        degrees of freedom and its p-value; ``nan`` / ``<NA>`` for a
        screen that is not a test), ``verdict`` (``"ok"``, ``"warn"`` or
        ``"fail"``, see :func:`verdict`) and ``detail`` (what was found,
        in words).
    advice : list of Advice
        Plain-English remedies, each naming the SurPyval feature that
        applies it (``Advice.features``), for the rows that failed (and
        the screens that warned), deduplicated.
    notes : list of str
        What the report could not check, or that the reader should know
        to read it (for example a check skipped for lack of data).
    alpha : float
        The level the tests were judged at.
    family : str
        The model family whose checks ran.
    model : str
        The kind of model diagnosed.

    Examples
    --------
    >>> from surpyval import CoxPH
    >>> from surpyval.datasets import load_rossi_static
    >>> df = load_rossi_static()
    >>> df["censored"] = 1 - df["arrest"]
    >>> model = CoxPH.fit_from_df(
    ...     df, x_col="week", c_col="censored", Z_cols=["fin", "age", "prio"]
    ... )
    >>> d = model.diagnose()
    >>> d.ok
    False
    >>> d.table.loc[d.table["verdict"] == "fail", ["transform", "covariates"]]
      transform covariates
    1        km     (age,)
    >>> print(d.advice[0])  # doctest: +ELLIPSIS
    The effect of age changes with time ...
    """

    def __init__(
        self,
        table: "pd.DataFrame",
        advice: Sequence[Advice],
        notes: Sequence[str],
        alpha: float,
        family: str,
        model: str,
        columns: Sequence[str] = (),
        context: "Context | None" = None,
        family_spec: "Family | None" = None,
    ) -> None:
        self.table = table
        self.advice = list(advice)
        self.notes = list(notes)
        self.alpha = float(alpha)
        self.family = family
        self.model = model
        self.columns = list(columns)
        # What plot() draws from: the family and the fitted model's
        # context. A report restored with from_dict has neither.
        self._context = context
        self._family_spec = family_spec

    # -- verdicts ---------------------------------------------------------
    @property
    def ok(self) -> bool:
        """True when no check failed (a warning does not count).

        Examples
        --------
        >>> from surpyval import CoxPH
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> x, c = df["week"].values, 1 - df["arrest"].values
        >>> CoxPH.fit(x, df[["fin", "prio"]].values, c=c).diagnose().ok
        True
        """
        return not bool((self.table["verdict"] == "fail").any())

    def raise_if_failed(self) -> "Diagnosis":
        """Raise :class:`DiagnosticsFailed` if any check failed, naming
        each failed check and the advice; otherwise return the report, so
        a pipeline can write ``model.diagnose().raise_if_failed()``.

        Examples
        --------
        >>> from surpyval import CoxPH
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> x, c = df["week"].values, 1 - df["arrest"].values
        >>> d = CoxPH.fit(x, df[["fin", "prio"]].values, c=c).diagnose()
        >>> d.raise_if_failed() is d
        True
        """
        failed = self.table[self.table["verdict"] == "fail"]
        if failed.empty:
            return self
        lines = [
            "{} of {} checks failed at alpha = {:g} ({}):".format(
                len(failed), len(self.table), self.alpha, self.model
            )
        ]
        for _, row in failed.iterrows():
            lines.append("  - " + self._describe(row))
        if self.advice:
            lines.append("Advice:")
            lines.extend("  - " + str(a) for a in self.advice)
        raise DiagnosticsFailed("\n".join(lines), self)

    def _describe(self, row: Any) -> str:
        covariates = _format_cell(tuple(row["covariates"])) or "the model"
        test = row["test"]
        extra = [
            "{} {}".format(k, row[k])
            for k in self.table.columns
            if k not in COLUMNS and isinstance(row[k], str) and row[k]
        ]
        if extra:
            test = "{} ({})".format(test, ", ".join(extra))
        p = row["p"]
        if p is not None and isinstance(p, float) and math.isfinite(p):
            return "{}: {} for {}, p = {:.3g}".format(
                row["check"], test, covariates, p
            )
        return "{}: {} for {}: {}".format(
            row["check"], test, covariates, row["detail"]
        )

    # -- machine-readable -------------------------------------------------
    def to_dict(self) -> dict:
        """The report as a strict-JSON-safe dictionary (``nan`` is
        ``None``, tuples are lists), which :meth:`from_dict` reads back.

        Examples
        --------
        >>> import json
        >>> from surpyval import CoxPH
        >>> from surpyval.datasets import load_rossi_static
        >>> from surpyval.diagnostics import Diagnosis
        >>> df = load_rossi_static()
        >>> x, c = df["week"].values, 1 - df["arrest"].values
        >>> d = CoxPH.fit(x, df[["fin", "prio"]].values, c=c).diagnose()
        >>> text = json.dumps(d.to_dict(), allow_nan=False)
        >>> Diagnosis.from_dict(json.loads(text)).to_dict() == d.to_dict()
        True
        """
        rows = []
        for record in self.table.to_dict(orient="records"):
            row: dict = {}
            for key, value in record.items():
                if key == "covariates":
                    row[key] = [str(v) for v in value]
                elif key in ("statistic", "p"):
                    row[key] = _json_number(value)
                elif key == "df":
                    number = _json_number(value)
                    row[key] = None if number is None else int(number)
                elif isinstance(value, str) or value is None:
                    row[key] = value
                else:
                    row[key] = _json_number(value)
            rows.append(row)
        return {
            "schema": SCHEMA,
            "family": self.family,
            "model": self.model,
            "alpha": self.alpha,
            "columns": list(self.columns),
            "ok": self.ok,
            "table": rows,
            "advice": [a.to_dict() for a in self.advice],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Diagnosis":
        """The report :meth:`to_dict` wrote. It has the table, advice and
        notes but not the fitted model, so it cannot :meth:`plot`.

        Examples
        --------
        >>> from surpyval import CoxPH
        >>> from surpyval.datasets import load_rossi_static
        >>> from surpyval.diagnostics import Diagnosis
        >>> df = load_rossi_static()
        >>> x, c = df["week"].values, 1 - df["arrest"].values
        >>> d = CoxPH.fit(x, df[["fin", "prio"]].values, c=c).diagnose()
        >>> Diagnosis.from_dict(d.to_dict()).ok
        True
        """
        if not isinstance(data, dict) or "table" not in data:
            raise ValueError(
                "'data' must be the dictionary Diagnosis.to_dict() wrote; "
                "got {!r}.".format(type(data).__name__)
            )
        if data.get("schema", SCHEMA) > SCHEMA:
            raise ValueError(
                "This report was written by a newer SurPyval (schema {}); "
                "this one reads schema {} and older. Upgrade "
                "SurPyval.".format(data["schema"], SCHEMA)
            )
        extra: list[str] = []
        for row in data["table"]:
            extra.extend(k for k in row if k not in COLUMNS and k not in extra)
        table = make_table(data["table"], extra)
        advice = [
            Advice(
                a["advice"],
                a.get("check"),
                a.get("covariates", ()),
                a.get("features", ()),
            )
            for a in data.get("advice", [])
        ]
        return cls(
            table,
            advice,
            data.get("notes", []),
            data["alpha"],
            data.get("family", ""),
            data.get("model", ""),
            data.get("columns", ()),
        )

    # -- plots --------------------------------------------------------------
    def plot(
        self,
        which: "str | Sequence[str] | None" = None,
        *,
        n_boot: int = 200,
        random_state: Any = None,
    ) -> Any:
        """
        Draw the family's visual checks, one panel each, on one figure.

        matplotlib is imported here, not by ``diagnose()``.

        Parameters
        ----------
        which : str or list of str, optional
            The checks to draw, by their ``check`` names (for a Cox model
            ``"proportional_hazards"``, ``"functional_form"``,
            ``"influence"``, ``"outliers"`` and ``"overall_fit"``); all of
            them by default.
        n_boot : int, optional
            Resamples for the bootstrap bands the family draws (around a
            smoothed trend, for example). Default 200.
        random_state : optional
            The seed of those resamples (SurPyval's one seed rule:
            ``None`` draws from numpy's global generator).

        Returns
        -------
        matplotlib.figure.Figure

        Examples
        --------
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> from surpyval import CoxPH
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> x, c = df["week"].values, 1 - df["arrest"].values
        >>> d = CoxPH.fit(x, df[["fin", "prio"]].values, c=c).diagnose()
        >>> fig = d.plot("proportional_hazards", n_boot=20, random_state=1)
        >>> len(fig.axes)
        2
        """
        from .registry import plot_report

        if self._context is None or self._family_spec is None:
            raise ValueError(
                "plot() needs the fitted model, which a Diagnosis restored "
                "with from_dict does not carry; call model.diagnose().plot() "
                "on the model instead."
            )
        return plot_report(
            self._family_spec,
            self._context,
            which,
            n_boot=n_boot,
            random_state=random_state,
        )

    # -- printing -----------------------------------------------------------
    def summary(self) -> str:
        """The report as text: a headline, the table, the advice and the
        notes (what ``print(d)`` shows).

        Examples
        --------
        >>> from surpyval import CoxPH
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> x, c = df["week"].values, 1 - df["arrest"].values
        >>> d = CoxPH.fit(x, df[["fin", "prio"]].values, c=c).diagnose()
        >>> print(d.summary().splitlines()[0])
        Cox proportional hazards diagnostics at alpha = 0.05: 0 failed, 0 warned, 10 ok
        """  # noqa: E501
        counts = {v: int((self.table["verdict"] == v).sum()) for v in VERDICTS}
        lines = [
            "{} diagnostics at alpha = {:g}: {} failed, {} warned, "
            "{} ok".format(
                self.family,
                self.alpha,
                counts["fail"],
                counts["warn"],
                counts["ok"],
            )
        ]
        if not self.table.empty:
            shown = self.table.drop(columns=["detail"])
            text = shown.to_string(
                index=False,
                na_rep="",
                formatters={
                    c: (lambda v, c=c: _format_cell(v, 3 if c == "p" else 4))
                    for c in shown.columns
                },
            )
            lines.extend("  " + line for line in text.splitlines())
        if self.advice:
            lines.append("Advice:")
            lines.extend("  - " + str(a) for a in self.advice)
        if self.notes:
            lines.append("Notes:")
            lines.extend("  - " + n for n in self.notes)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def make_table(rows: Sequence[dict], extra: Sequence[str] = ()) -> Any:
    """The report's table from its rows (dicts with the keys of
    :data:`COLUMNS` and of ``extra``, a family's own columns): the
    columns in order, ``covariates`` as tuples, ``df`` as a nullable
    integer.

    Examples
    --------
    >>> from surpyval.diagnostics.report import make_table
    >>> make_table([{"check": "c", "test": "t", "covariates": ["a"],
    ...              "statistic": 1.0, "df": 1, "p": 0.3,
    ...              "verdict": "ok", "detail": ""}])["covariates"][0]
    ('a',)
    """
    import pandas as pd

    columns = list(COLUMNS[:1]) + list(extra) + list(COLUMNS[1:])
    data: dict = {c: [] for c in columns}
    for row in rows:
        for c in columns:
            value = row.get(c)
            if c == "covariates":
                value = tuple(value or ())
            elif c in ("statistic", "p"):
                value = float("nan") if value is None else float(value)
            elif c == "detail":
                value = "" if value is None else str(value)
            data[c].append(value)
    table = pd.DataFrame(data, columns=columns)
    table["df"] = pd.array(
        [None if v is None else int(v) for v in data["df"]], dtype="Int64"
    )
    return table
