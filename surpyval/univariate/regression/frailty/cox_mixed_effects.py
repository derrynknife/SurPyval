"""
The mixed-effects Cox model: Gaussian random effects on the log-hazard, as
R's ``coxme`` (Therneau) fits them.

.. math::
    h_i(t) = h_0(t) \\exp\\big(\\beta' Z_i + \\textstyle\\sum_t v_{ti}' b_{t,
    g_t(i)}\\big), \\qquad b_{t,g} \\sim N(0, \\Sigma_t),

with one term ``t`` per grouping factor: its effects ``v`` are an
intercept (1) and any random slopes (columns of ``Z``), and ``b_{t,g}``
are group ``g``'s random effects, independent between groups and terms.
Nested factors (wards within hospitals) are a term for the outer factor
and one for the pair; crossed factors are a term each. A random intercept
alone is a log-normal shared frailty, ``exp(b)``.

For given variances the coefficients and the random effects maximise the
penalised partial likelihood; the variances maximise its Laplace
approximation to the integrated partial likelihood (Ripatti and Palmgren
2000; Therneau, Grambsch and Pankratz 2003). See :mod:`._mixed_cox`.
"""

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtri as _z
from scipy.stats import chi2, norm

from surpyval.diagnostics.mixin import DiagnoseMixin
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
from surpyval.univariate.regression._aliasing import dataframe_covariates
from surpyval.utils import is_missing_event
from surpyval.utils.covariates import (
    check_covariate_rows,
    coefficient_names,
    finite_covariate_mask,
)
from surpyval.utils.data_formats import xcnt_handler
from surpyval.utils.data_summary import data_summary
from surpyval.utils.dataframe import check_columns
from surpyval.utils.fitter_repr import FitterRepr
from surpyval.utils.linalg import standard_errors_of
from surpyval.utils.no_maximum import (
    maximum_entry,
    restored_maximum,
    warn_no_maximum,
    warn_unverified,
)
from surpyval.utils.shapes import (
    check_paired_rows,
    covariate_rows,
    keeps_query_shape,
)
from surpyval.utils.surpyval_data import accepts_surpyval_data
from surpyval.utils.units import takes_unit
from surpyval.utils.validation import (
    BOUNDS,
    check_alpha_ci,
    check_option,
    no_covariance_error,
)
from surpyval.utils.warnings import caller_stacklevel, collecting_warnings

from .._aliasing import expand, fit_columns
from .._fit_skeleton import covariate_center
from .._prediction import (
    ConditionalSurvivalMixin,
    paired_probabilities,
    step_quantiles,
    unique_rows,
)
from ..proportional_hazards.cox_likelihood import (
    baseline_at_origin,
    strata_labels,
)
from ..proportional_hazards.cox_ph import CoxPH
from ..regression_data import (
    FrameDesign,
    check_covariate_width,
    prepare_Z,
    restore_covariate_meta,
    serialise_covariate_meta,
)
from ..semi_parametric_regression_model import SemiParametricRegressionModel
from ._mixed_cox import SD_MAX, MixedCoxEngine, Solution, TermDesign
from .families import lognormal_log_integral

_TIE_METHODS = ("efron", "breslow")
_ORIGIN_HINT = (
    "Fit with center=True to keep the baseline at the covariate means."
)
_NO_RANDOM = (
    "random() is not available for a mixed-effects Cox model: its baseline "
    "is a step function that stops at the last event time. Draw from a "
    "parametric frailty model of the same data instead, "
    "Frailty(Weibull, family='lognormal').fit(...).random(...), or from "
    "CoxPH."
)


class RandomEffect:
    """
    One random-effects term of :class:`CoxMixedEffectsFitter`: a grouping
    factor, and the effects that vary between its groups.

    Parameters
    ----------
    groups : array_like, or a tuple of array_like
        The group of each row. A tuple of label arrays is their
        combination (``(hospital, ward)``: each ward of each hospital is a
        group), which is how a nested factor is written: a term for
        ``hospital`` and one for ``(hospital, ward)``. A row with a missing
        label is dropped from the fit, with a warning.
    slopes : int, str or list, optional
        The covariates (columns of ``Z``, by position, or by name for a
        ``Z`` with named columns) whose coefficients vary between groups:
        random slopes. Each also has its fixed coefficient in ``Z``.
    intercept : bool, optional
        Whether the term has a random intercept (default ``True``).
    correlated : bool, optional
        With several effects (an intercept and slopes), whether they are
        correlated within a group (an unstructured covariance, the
        default, as ``coxme``'s ``(1 + x | g)``) or independent (a
        diagonal covariance).
    name : str, optional
        The term's name in the output; by default ``"group"`` (with a
        number for the second and later terms), or the column names in
        ``fit_from_df``.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import RandomEffect
    >>> centre = np.repeat([1, 2, 3], 4)
    >>> term = RandomEffect(centre, slopes=[0], name="centre")
    >>> term
    RandomEffect(centre: intercept + slope on Z[:, 0], correlated)
    """

    def __init__(
        self,
        groups: Any,
        slopes: Any = None,
        intercept: bool = True,
        correlated: bool = True,
        name: "str | None" = None,
    ) -> None:
        if slopes is None:
            slopes = []
        elif isinstance(slopes, (str, int, np.integer)):
            slopes = [slopes]
        self.groups = groups
        self.slopes = list(slopes)
        self.intercept = bool(intercept)
        self.correlated = bool(correlated)
        self.name = name
        if not self.intercept and not self.slopes:
            raise ValueError(
                "A RandomEffect needs an intercept or at least one slope: "
                "pass intercept=True, or slopes=[...]."
            )

    def __repr__(self) -> str:
        parts = ["intercept"] if self.intercept else []
        parts += [
            "slope on " + (repr(s) if isinstance(s, str) else f"Z[:, {s}]")
            for s in self.slopes
        ]
        extra = ""
        if len(parts) > 1:
            extra = ", correlated" if self.correlated else ", independent"
        return "RandomEffect({}: {}{})".format(
            self.name or "group", " + ".join(parts), extra
        )


# -- the bar notation of fit_from_df --------------------------------------

_TERM = re.compile(r"\(([^()|]*)(\|\|?)([^()|]*)\)")


def parse_random(text: str) -> "list[dict]":
    """
    The terms of a random-effects formula in the bar notation of ``lme4``
    and ``coxme``: ``"(1 | g)"``, ``"(1 + x | g)"``, ``"(0 + x | g)"``,
    ``"(1 | a/b)"`` (``b`` nested in ``a``: a term for ``a`` and one for
    ``a:b``), ``"(1 | a:b)"``, several joined by ``+``, and ``||`` for
    independent effects. Each term is a dict of ``groups`` (a list of
    column names), ``slopes``, ``intercept`` and ``correlated``.

    A slope without an explicit ``1`` or ``0`` is refused: ``lme4`` reads
    ``(x | g)`` as an intercept and a slope, ``coxme`` as a slope alone.

    Examples
    --------
    >>> from surpyval.univariate.regression.frailty.cox_mixed_effects import (
    ...     parse_random,
    ... )
    >>> [t["groups"] for t in parse_random("(1 | hospital/ward)")]
    [['hospital'], ['hospital', 'ward']]
    """
    if not isinstance(text, str):
        raise TypeError(
            "random_effects must be a string such as '(1 | group)'; got "
            "{!r}.".format(text)
        )
    found = list(_TERM.finditer(text))
    rest = _TERM.sub("", text).replace("+", "").strip()
    if not found or rest:
        raise ValueError(
            "random_effects={!r} is not a random-effects formula: write "
            "each term in parentheses with a bar, '(1 | group)', "
            "'(1 + x | group)' "
            "or '(0 + x | group)', joined by '+'; '(1 | a/b)' nests b in "
            "a.".format(text)
        )
    out = []
    for match in found:
        lhs = [s.strip() for s in match.group(1).split("+") if s.strip()]
        bars = match.group(2)
        rhs = match.group(3).strip()
        intercept = None
        slopes = []
        for token in lhs:
            if token == "1":
                intercept = True
            elif token in ("0", "-1"):
                intercept = False
            else:
                slopes.append(token)
        if intercept is None:
            if slopes:
                raise ValueError(
                    "In {!r} say whether the term has a random intercept: "
                    "'(1 + {} | {})' for an intercept and slope, or "
                    "'(0 + {} | {})' for the slope alone (lme4 and coxme "
                    "read '(x | g)' differently).".format(
                        match.group(0),
                        " + ".join(slopes),
                        rhs,
                        " + ".join(slopes),
                        rhs,
                    )
                )
            intercept = True
        if not rhs:
            raise ValueError(
                "{!r} names no grouping column after the bar.".format(
                    match.group(0)
                )
            )
        if "/" in rhs:
            levels = [s.strip() for s in rhs.split("/")]
            groupings = [levels[: i + 1] for i in range(len(levels))]
        else:
            groupings = [[s.strip() for s in rhs.split(":")]]
        for groups in groupings:
            out.append(
                {
                    "groups": groups,
                    "slopes": slopes,
                    "intercept": intercept,
                    "correlated": bars == "|",
                }
            )
    return out


# -- data -----------------------------------------------------------------


def _label_array(groups: Any, n_obs: int, what: str) -> tuple:
    """A term's groups: each row's position in the sorted labels (a tuple
    of label arrays is sorted part by part and its labels joined with
    ``:``), the labels as strings, and the mask of rows with a missing
    label. Labels sort as they are (numbers numerically), not as text."""
    parts = groups if isinstance(groups, tuple) else (groups,)
    codes, texts = [], []
    missing = np.zeros(n_obs, dtype=bool)
    for part in parts:
        if isinstance(part, pd.Series):
            part = part.to_numpy()
        arr, miss = strata_labels(
            np.asarray(part, dtype=object).ravel().tolist()
        )
        if arr.shape[0] != n_obs:
            raise ValueError(
                "{} has {} label(s) but there are {} observations; give one "
                "label per row.".format(what, arr.shape[0], n_obs)
            )
        try:
            uniq, inv = np.unique(arr, return_inverse=True)
        except TypeError:
            uniq, inv = np.unique(arr.astype(str), return_inverse=True)
        codes.append(np.ravel(inv))
        texts.append(np.array([str(v) for v in uniq.tolist()], dtype=object))
        missing |= miss
    return np.column_stack(codes), texts, missing


def _term_groups(
    codes: np.ndarray, texts: "list[np.ndarray]"
) -> tuple[np.ndarray, np.ndarray]:
    """The distinct groups' labels and each row's group."""
    keys, first, inv = np.unique(
        codes, axis=0, return_index=True, return_inverse=True
    )
    labels = np.array(
        [
            ":".join(str(t[k]) for t, k in zip(texts, row))
            for row in keys.tolist()
        ],
        dtype=object,
    )
    return labels, np.ravel(inv)


def _slope_index(slope: Any, names: "list[str] | None", p: int) -> int:
    if isinstance(slope, (int, np.integer)) and not isinstance(slope, bool):
        if not 0 <= int(slope) < p:
            raise ValueError(
                "slopes={!r} is not a column of Z, which has {} "
                "column(s).".format(slope, p)
            )
        return int(slope)
    if names is not None and str(slope) in names:
        return names.index(str(slope))
    raise ValueError(
        "slopes={!r} is not a column of Z{}. A random slope is a "
        "covariate of the fixed part: add it to Z (or Z_cols/formula), and "
        "name it by its position or column name.".format(
            slope,
            "" if names is None else f" (columns {names})",
        )
    )


def _as_terms(groups: Any, random_effects: Any) -> "list[RandomEffect]":
    terms: list[RandomEffect] = []
    if groups is not None:
        terms.append(RandomEffect(groups))
    if random_effects is not None:
        if isinstance(random_effects, RandomEffect):
            random_effects = [random_effects]
        for term in random_effects:
            if not isinstance(term, RandomEffect):
                raise TypeError(
                    "random_effects must be a RandomEffect or a list of "
                    "them; got {!r}. For a random intercept alone, pass "
                    "groups= the labels.".format(term)
                )
            terms.append(term)
    if not terms:
        raise ValueError(
            "A mixed-effects Cox model needs a random effect: pass "
            "groups= (a group label per row, for a random intercept) or "
            "random_effects= (RandomEffect terms). Without one, use "
            "CoxPH.fit."
        )
    return terms


# -- the fitter -----------------------------------------------------------


class CoxMixedEffectsFitter(FitterRepr):
    """
    The mixed-effects Cox model (R's ``coxme``): a Cox model whose
    log-hazard has Gaussian random effects, a random intercept per group of
    one or more grouping factors (nested or crossed) and random slopes,
    correlated within a group. ``CoxMixedEffects`` is an instance; see
    :meth:`fit`.
    """

    #: The ``repr``
    fitter_kind = "semi-parametric mixed-effects fitter"

    def _repr_name(self) -> str:
        return "CoxMixedEffects"

    @dataframe_covariates
    @takes_unit
    @accepts_surpyval_data
    def fit(
        self,
        x: Any,
        Z: Any = None,
        c: Any = None,
        n: Any = None,
        tl: Any = None,
        groups: Any = None,
        random_effects: Any = None,
        strata: Any = None,
        tie_method: str = "efron",
        variance: Any = None,
        center: bool = False,
    ) -> "CoxMixedEffectsModel":
        """
        Fit the mixed-effects Cox model by maximum likelihood, as R's
        ``coxme``.

        In words: each group (a hospital, a batch, a machine) gets its own
        multiplier on the hazard, and possibly its own effect of a
        covariate, drawn from a normal distribution on the log scale whose
        variance is estimated. Groups with few events are pulled towards
        the average (shrinkage), and the fixed coefficients are the
        effects within a group.

        For given variances the coefficients ``beta`` and the random
        effects maximise the penalised partial likelihood ``PL(beta, b) -
        b' Sigma^{-1} b / 2`` (Newton's method); the variances maximise
        the integrated partial likelihood, the random effects integrated
        out by the Laplace approximation (Ripatti and Palmgren 2000), as
        ``coxme`` does.

        Parameters
        ----------
        x : array_like
            The observed times (exits, with ``tl``).
        Z : array_like, optional
            Fixed covariates, one row per observation. Rows with a missing
            or infinite covariate are dropped, with a warning; a column the
            partial likelihood cannot determine is aliased as in ``CoxPH``.
        c : array_like, optional
            ``0`` event, ``1`` right-censored (the only two supported).
        n : array_like, optional
            Counts of each row.
        tl : array_like, optional
            Entry (left-truncation) times, as in ``CoxPH``: start-stop
            rows of a unit keep their group.
        groups : array_like, optional
            The group of each row: a random intercept per group (R's
            ``(1 | groups)``). The shorthand for
            ``random_effects=RandomEffect(groups)``.
        random_effects : RandomEffect or list of RandomEffect, optional
            The random-effects terms: several grouping factors (nested or
            crossed), and random slopes. See :class:`RandomEffect`.
        strata : array_like, optional
            Stratum labels: a baseline per stratum, as in ``CoxPH``.
            Predictions then take ``stratum=``.
        tie_method : str, optional
            ``"efron"`` (default) or ``"breslow"``, as in ``CoxPH``.
        variance : list, optional
            Fixed variances in place of their estimates (R's ``vfixed``):
            one entry per term, a variance for a one-effect term or the
            effects' covariance matrix. The model then has no variance
            bounds, and ``lr_test`` compares that fixed model.
        center : bool, optional
            As for ``CoxPH``: ``True`` keeps the baseline at the
            covariate means.

        Returns
        -------
        CoxMixedEffectsModel

        Examples
        --------
        The kidney catheter data, a random intercept per patient (R's
        ``coxme(Surv(time, status) ~ age + sex + (1 | id), kidney)``):

        >>> import numpy as np
        >>> from surpyval import CoxMixedEffects
        >>> from surpyval.datasets import load_kidney
        >>> df = load_kidney()
        >>> Z = np.column_stack([df["age"], df["sex"] == 2])
        >>> model = CoxMixedEffects.fit(
        ...     df["time"], Z=Z, c=1 - df["status"], groups=df["id"]
        ... )
        >>> model.beta.round(4)
        array([ 0.0043, -1.3556])
        >>> model.variance_components()["variance"].round(4).tolist()
        [0.4574]
        >>> round(model.log_likelihood, 4)
        -181.9045
        """
        check_option(
            "tie_method",
            tie_method,
            _TIE_METHODS,
            "The mixed-effects fit has no exact or Kalbfleisch-Prentice "
            "likelihood.",
        )
        specs = _as_terms(groups, random_effects)
        data = _prepare(x, Z, c, n, tl, strata, specs)
        return _fit(data, tie_method, variance, center)

    @takes_unit
    def fit_from_df(
        self,
        df: pd.DataFrame,
        x_col: str,
        Z_cols: "str | list[str] | None" = None,
        c_col: "str | None" = None,
        n_col: "str | None" = None,
        tl_col: "str | None" = None,
        group_col: "str | None" = None,
        random_effects: "str | None" = None,
        strata_col: "str | None" = None,
        formula: "str | None" = None,
        tie_method: str = "efron",
        variance: Any = None,
        center: bool = False,
    ) -> "CoxMixedEffectsModel":
        """Fit from a :class:`pandas.DataFrame` naming the columns.

        The fixed part is ``Z_cols`` or ``formula`` (formulaic syntax);
        the random part is ``group_col`` (a random intercept per group),
        or ``random_effects``, a string in the bar notation of ``coxme``
        and ``lme4``:
        ``"(1 | centre)"``, ``"(1 + trt | centre)"`` (a correlated random
        intercept and slope), ``"(0 + trt | centre)"`` (a slope alone),
        ``"(1 + trt || centre)"`` (independent), ``"(1 | hospital/ward)"``
        (wards nested in hospitals), ``"(1 | a) + (1 | b)"`` (crossed). A
        slope must be a column of the fixed part.

        Examples
        --------
        >>> from surpyval import CoxMixedEffects
        >>> from surpyval.datasets import load_lung
        >>> df = load_lung()
        >>> df["censored"] = 1 - df["status"]
        >>> model = CoxMixedEffects.fit_from_df(
        ...     df, x_col="time", c_col="censored", Z_cols=["age", "sex"],
        ...     random_effects="(1 | inst)",
        ... )
        >>> list(model.random_effects)
        ['inst']
        """
        check_columns(
            df,
            x_col=x_col,
            c_col=c_col,
            n_col=n_col,
            tl_col=tl_col,
            group_col=group_col,
            strata_col=strata_col,
        )
        if (group_col is None) == (random_effects is None):
            raise ValueError(
                "Give the random part as group_col= (a random intercept per "
                "group) or random_effects= (e.g. '(1 + trt | centre)'), not "
                "{}.".format("both" if group_col else "neither")
            )
        parsed: "list[dict[str, Any]]"
        if random_effects is None:
            parsed = [
                {
                    "groups": [group_col],
                    "slopes": [],
                    "intercept": True,
                    "correlated": True,
                }
            ]
        else:
            parsed = parse_random(random_effects)
        terms = []
        for spec in parsed:
            cols = spec["groups"]
            check_columns(df, **{f"group {c}": c for c in cols})
            labels = tuple(df[c].to_numpy() for c in cols)
            terms.append(
                RandomEffect(
                    labels if len(labels) > 1 else labels[0],
                    slopes=spec["slopes"],
                    intercept=spec["intercept"],
                    correlated=spec["correlated"],
                    name=":".join(cols),
                )
            )
        design = FrameDesign(df, Z_cols, formula, optional=True)
        return design.fit(
            self.fit,
            df[x_col].values,
            Z=design.Z,
            c=None if c_col is None else df[c_col].values,
            n=None if n_col is None else df[n_col].values,
            tl=None if tl_col is None else df[tl_col].values,
            random_effects=terms,
            strata=None if strata_col is None else df[strata_col].values,
            tie_method=tie_method,
            variance=variance,
            center=center,
        )


class _Data:
    """The checked rows of a fit, and its random-effects terms."""

    x: np.ndarray
    c: np.ndarray
    w: np.ndarray
    tl: np.ndarray
    Z: np.ndarray
    truncated: bool
    names: "list[str] | None"
    strata: "np.ndarray | None"
    strata_labels: "np.ndarray | None"
    terms: "list[TermDesign]"


def _prepare(
    x: Any,
    Z: Any,
    c: Any,
    n: Any,
    tl: Any,
    strata: Any,
    specs: "list[RandomEffect]",
) -> _Data:
    x_h, c_h, n_h, _ = xcnt_handler(x, c, n, group_and_sort=False)
    c = np.asarray(c_h, dtype=int).ravel()
    if not np.all(np.isin(c, (0, 1))):
        raise ValueError(
            "The mixed-effects Cox model supports only observed (c=0) and "
            "right-censored (c=1) rows; fit interval or left-censored "
            "data with a parametric frailty model, "
            "Frailty(Weibull, family='lognormal')."
        )
    x = np.asarray(x_h, dtype=float)
    if x.ndim == 2:
        x = x[:, 0]
    n_obs = x.shape[0]
    w = np.asarray(n_h, dtype=float).ravel()
    if tl is None:
        tl_arr = np.full(n_obs, -np.inf)
    else:
        tl_arr = np.asarray(tl, dtype=float).ravel()
        if tl_arr.shape[0] != n_obs:
            raise ValueError(
                "tl has {} value(s) but there are {} observations.".format(
                    tl_arr.shape[0], n_obs
                )
            )
        if np.any(~(tl_arr < x)):
            raise ValueError(
                "Each entry time tl must be below its exit time x; {} "
                "row(s) are not.".format(int(np.sum(~(tl_arr < x))))
            )
    names = fit_columns()
    if Z is not None:
        Zm = np.atleast_2d(np.asarray(Z, dtype=float))
        if Zm.shape[0] != n_obs and Zm.shape[1] == n_obs:
            Zm = Zm.T
        check_covariate_rows(Zm, n_obs)
    else:
        Zm = np.zeros((n_obs, 0))
    p = Zm.shape[1]
    drop = ~finite_covariate_mask(Zm) if p else np.zeros(n_obs, bool)
    term_labels = []
    missing = np.zeros(n_obs, dtype=bool)
    for k, spec in enumerate(specs):
        what = "groups" if spec.name is None else repr(spec.name)
        codes, texts, miss = _label_array(spec.groups, n_obs, what)
        term_labels.append((codes, texts))
        missing |= miss
    if strata is not None:
        s_arr, s_miss = strata_labels(
            np.asarray(strata, dtype=object).ravel().tolist()
        )
        if s_arr.shape[0] != n_obs:
            raise ValueError(
                "'strata' has {} label(s) but there are {} observations."
                "".format(s_arr.shape[0], n_obs)
            )
        missing |= s_miss
    if missing.any():
        if missing.all():
            raise ValueError(
                "Every row has a missing group or stratum label; there is "
                "nothing to fit."
            )
        warnings.warn(
            "Dropped {} of {} rows with a missing group or stratum "
            "label.".format(int(missing.sum()), n_obs),
            UserWarning,
            stacklevel=caller_stacklevel(),
        )
    keep = ~(missing | drop)
    x, c, w, tl_arr, Zm = x[keep], c[keep], w[keep], tl_arr[keep], Zm[keep]
    term_labels = [(codes[keep], texts) for codes, texts in term_labels]
    if int((c == 0).sum()) == 0:
        raise ValueError("At least one event (c=0) is required.")
    data = _Data()
    data.x, data.c, data.w, data.tl, data.Z = x, c, w, tl_arr, Zm
    data.truncated = tl is not None
    data.names = names
    data.strata = None
    data.strata_labels = None
    if strata is not None:
        s_text = np.array([str(v) for v in s_arr[keep].tolist()])
        data.strata_labels, data.strata = np.unique(
            s_text, return_inverse=True
        )
    terms = []
    used = set()
    for k, (spec, (codes, texts)) in enumerate(zip(specs, term_labels)):
        uniq, inv = _term_groups(codes, texts)
        if uniq.shape[0] < 2:
            raise ValueError(
                "A random-effects term needs at least two groups; term {} "
                "has {}.".format(k + 1, uniq.shape[0])
            )
        cols = [_slope_index(s, names, p) for s in spec.slopes]
        effects = (["(Intercept)"] if spec.intercept else []) + [
            names[j] if names is not None else f"Z[:, {j}]" for j in cols
        ]
        V = np.column_stack(
            ([np.ones(x.size)] if spec.intercept else [])
            + [Zm[:, j] for j in cols]
        )
        name = spec.name
        if name is None:
            name = "group" if k == 0 else f"group{k + 1}"
        if name in used:
            at = 2
            while f"{name}[{at}]" in used:
                at += 1
            name = f"{name}[{at}]"
        used.add(name)
        terms.append(
            TermDesign(
                name,
                uniq,
                inv,
                V,
                effects,
                spec.correlated,
                cols,
                spec.intercept,
            )
        )
    data.terms = terms
    return data


def _psi_of_variance(terms: "list[TermDesign]", variance: Any) -> np.ndarray:
    """The engine's parameters of fixed variances, one entry per term."""
    if not isinstance(variance, (list, tuple)) or len(variance) != len(terms):
        raise ValueError(
            "variance must be a list with one entry per random-effects "
            "term ({}): a variance for a one-effect term, or the effects' "
            "covariance matrix.".format(len(terms))
        )
    out = []
    for term, v in zip(terms, variance):
        S = np.atleast_2d(np.asarray(v, dtype=float))
        if S.shape != (term.k, term.k) or not np.all(np.isfinite(S)):
            raise ValueError(
                "The variance of term {!r} must be a finite {} x {} "
                "matrix (a number for one effect); got {!r}.".format(
                    term.name, term.k, term.k, v
                )
            )
        sd = np.sqrt(np.clip(np.diag(S), 0, None))
        if np.any(np.diag(S) < 0) or np.any(np.linalg.eigvalsh(S) < -1e-12):
            raise ValueError(
                "The variance of term {!r} must be non-negative (positive "
                "semi-definite); got {!r}.".format(term.name, v)
            )
        if np.any(sd > SD_MAX):
            raise ValueError(
                "A fixed standard deviation must be at most {}; term {!r} "
                "has {}.".format(SD_MAX, term.name, sd.max())
            )
        part = list(sd)
        if term.correlated:
            with np.errstate(all="ignore"):
                R = S / np.outer(sd, sd)
            R = np.where(np.isfinite(R), R, 0.0)
            np.fill_diagonal(R, 1.0)
            try:
                Lc = np.linalg.cholesky(R + 1e-15 * np.eye(term.k))
            except np.linalg.LinAlgError:
                Lc = np.eye(term.k)
            for i in range(1, term.k):
                remaining = 1.0
                for j in range(i):
                    z = Lc[i, j] / np.sqrt(remaining) if remaining > 0 else 0
                    part.append(float(np.clip(z, -1 + 1e-8, 1 - 1e-8)))
                    remaining -= Lc[i, j] ** 2
        out.extend(part)
    return np.asarray(out, dtype=float)


def _fit(
    data: _Data, tie_method: str, variance: Any, center: bool
) -> "CoxMixedEffectsModel":
    x, c, w, tl, Zfull = data.x, data.c, data.w, data.tl, data.Z
    p_all = Zfull.shape[1]
    tl_cox = tl if data.truncated else None
    strata_arg = None if data.strata is None else data.strata
    maximum = "verified"
    cox = None
    if p_all:
        with collecting_warnings() as caught:
            cox = CoxPH.fit(
                x,
                Zfull,
                c,
                w,
                tl=tl_cox,
                tie_method=tie_method,
                strata=strata_arg,
                center=True,
            )
        maximum = cox.maximum
        for caught_warning in caught:
            warnings.warn(
                str(caught_warning.message),
                caught_warning.category,
                stacklevel=caller_stacklevel(),
            )
    kept = np.arange(p_all)
    if cox is not None and cox.aliased.size:
        kept = np.setdiff1d(kept, cox.aliased)
    Zk = Zfull[:, kept]
    means = covariate_center(Zk, w) if kept.size else np.zeros(0)
    engine = MixedCoxEngine(
        x, Zk - means, c, w, tl, data.strata, data.terms, tie_method
    )
    beta0 = (
        np.asarray(cox.beta, dtype=float)[kept] if cox is not None else None
    )
    null_psi = np.zeros(engine.n_psi)
    no_random = engine.solve(null_psi, beta0)
    fixed = variance is not None
    if fixed:
        psi = _psi_of_variance(data.terms, variance)
    else:
        psi, _, success = engine.maximise(engine.start())
        # A start from the boundary too: the integrated likelihood can have
        # a local maximum away from it.
        if engine.n_psi:
            alt, _, _ = engine.maximise(np.where(engine.start() > 0, 0.05, 0))
            if engine.ilik(alt) > engine.ilik(psi) + 1e-9:
                psi = alt
        upper = np.array([b[1] for b in engine.bounds()])
        at_top = np.isclose(psi, upper) & (upper == SD_MAX)
        if at_top.any():
            maximum = "no finite maximum"
            warn_no_maximum(
                "a random effect's standard deviation runs to the edge of "
                "its search, {} (groups that differ more than any finite "
                "variance explains)".format(SD_MAX),
                "The reported variances and the random effects are "
                "meaningless",
                "check for groups whose events all come first, or fix the "
                "variances with variance=",
            )
    sol = engine.solve(psi, beta0)
    if (engine.not_converged and not sol.converged) and maximum == "verified":
        maximum = "unverified"
        warn_unverified(
            "The mixed-effects Cox fit",
            "Newton's method on the penalised partial likelihood did not "
            "converge at the variances reported",
        )
    p, q = engine.p, engine.q
    try:
        inverse = np.linalg.inv(sol.hessian)
    except np.linalg.LinAlgError:
        inverse = None
    cov_beta = None if inverse is None else inverse[:p, :p]
    df_pen = float("nan")
    if inverse is not None:
        df_pen = float(p + q - np.trace(inverse[p:, p:]))

    # The baseline (per stratum) with the random effects as offsets, at
    # Z = 0 and b = 0 (or at the covariate means with center=True).
    offset = _random_part(data.terms, sol.b)
    model = CoxMixedEffectsModel()
    model.tie_method = tie_method
    model.beta = expand(sol.beta, kept, p_all) if p_all else np.zeros(0)
    model.center = (
        covariate_center(Zfull, w) if center and p_all else np.zeros(p_all)
    )
    baselines = []
    codes = (
        [None]
        if data.strata_labels is None
        else list(range(len(data.strata_labels)))
    )
    for code in codes:
        rows = np.ones(x.size, bool) if code is None else data.strata == code
        times, r, d, h0 = CoxPH.baseline(
            np.r_[sol.beta, 1.0],
            x[rows],
            c[rows],
            w[rows],
            np.column_stack([Zk[rows] - means, offset[rows]]),
            tl[rows],
            tie_method,
        )
        if kept.size and not center:
            r, h0 = baseline_at_origin(
                sol.beta, means, Zk[rows], r, h0, hint=_ORIGIN_HINT
            )
        baselines.append((times, h0))
    model.baselines = [
        {"x": t, "h0": h, "H0": np.cumsum(h)} for t, h in baselines
    ]
    model.strata = (
        None
        if data.strata_labels is None
        else [str(s) for s in data.strata_labels]
    )
    model.terms = [
        _term_record(t, part, sol.b, at)
        for t, part, at in zip(
            data.terms, engine.split(psi), _offsets(data.terms)
        )
    ]
    model.parameter_names = coefficient_names(p_all, fit_columns(), [])
    covariance = np.full((p_all, p_all), np.nan)
    if cov_beta is not None and kept.size:
        covariance[np.ix_(kept, kept)] = cov_beta
    if maximum == "no finite maximum" and cox is not None:
        off = np.flatnonzero(np.isnan(cox.standard_errors()))
        covariance[off, :] = np.nan
        covariance[:, off] = np.nan
    model._covariance = covariance
    model.log_likelihood = float(sol.ilik)
    model.log_likelihood_penalised = float(sol.ppl)
    model.log_likelihood_fitted = float(sol.pl)
    with np.errstate(all="ignore"):
        model.log_likelihood_null = -float(engine.neg_pl(np.zeros(p + q)))
    model.log_likelihood_no_random = float(no_random.ilik)
    model.df_penalised = df_pen
    model.fixed_variance = fixed
    model.k = int(kept.size) + (0 if fixed else engine.n_psi)
    model.n_obs = int(x.size)
    model.n_events = int((c == 0).sum())
    model.n_events_weighted = float(w[c == 0].sum())
    model.n_obs_weighted = float(w.sum())
    model._data_summary = data_summary(c, w, x=x)
    model.maximum = maximum
    model._psi = psi
    model._engine = engine
    model._beta0 = beta0
    return model


def _offsets(terms: "list[TermDesign]") -> "list[int]":
    out, at = [], 0
    for t in terms:
        out.append(at)
        at += t.q
    return out


def _random_part(terms: "list[TermDesign]", b: np.ndarray) -> np.ndarray:
    total = np.zeros(terms[0].inv.size)
    at = 0
    for t in terms:
        B = b[at : at + t.q].reshape(t.k, t.G)
        total += np.sum(t.V * B[:, t.inv].T, axis=1)
        at += t.q
    return total


def _term_record(
    t: TermDesign, part: np.ndarray, b: np.ndarray, at: int
) -> dict:
    S = t.covariance(part)
    sd = np.sqrt(np.clip(np.diag(S), 0, None))
    with np.errstate(all="ignore"):
        R = S / np.outer(sd, sd)
    R = np.where(np.isfinite(R), R, 0.0)
    np.fill_diagonal(R, 1.0)
    return {
        "name": t.name,
        "effects": list(t.effects),
        "intercept": bool(t.intercept),
        "slope_columns": [int(j) for j in t.slope_columns],
        "correlated": bool(t.correlated),
        "labels": [str(v) for v in t.labels.tolist()],
        "covariance": S,
        "sd": sd,
        "correlation": R,
        "blups": b[at : at + t.q].reshape(t.k, t.G).T.copy(),
    }


# -- the model ------------------------------------------------------------


class CoxMixedEffectsModel(
    ConditionalSurvivalMixin,
    InformationCriteriaMixin,
    DiagnoseMixin,
    SerialisableMixin,
):
    """
    A fitted mixed-effects Cox model, returned by
    :meth:`CoxMixedEffectsFitter.fit`.

    ``beta`` are the fixed coefficients: ``exp(beta)`` is the hazard ratio
    of two units of the same group (with the same random effects). Their
    covariance is the inverse of the penalised partial likelihood's
    information, its fixed block (``coxme``'s ``vcov``), with Wald
    intervals (:meth:`param_cb`, :meth:`summary`).

    :meth:`variance_components` gives each term's standard deviations,
    variances and correlations, and :meth:`variance_cb` their
    profile-likelihood intervals; :meth:`lr_test` tests the random effects
    together against the Cox model without them. ``random_effects`` are
    the predicted random effects of each group (the posterior modes,
    ``coxme``'s ``ranef``: best linear unbiased predictions, shrunk
    towards 0).

    The log-likelihoods are ``coxme``'s: ``log_likelihood`` the
    integrated (Laplace) partial likelihood, the one maximised and the one
    :meth:`aic` and :meth:`bic` penalise (by ``k``, the fixed
    coefficients and the variance parameters, as ``coxme``'s integrated
    AIC); ``log_likelihood_penalised`` the penalised partial likelihood at
    the fit, ``log_likelihood_fitted`` the partial likelihood there
    without the penalty (``coxme`` prints it as "Fitted"),
    ``log_likelihood_null`` at ``beta = 0`` and ``b = 0``, and
    ``log_likelihood_no_random`` the Cox model's (``CoxPH``) on the same
    rows; ``df_penalised`` is the penalised fit's effective degrees of
    freedom.

    Predictions are for a unit of a given group (``group=``: a label, or a
    ``{term: label}`` dict with several terms), or, by default, of a new
    group with random effects 0 (a typical, median group); with
    ``marginal=True`` the random effects of the terms not given in
    ``group`` are integrated out, the population-averaged curve for a unit
    of a new group. The baseline is a step function, as ``CoxPH``'s:
    ``hf`` and ``df`` are its jumps.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import CoxMixedEffects
    >>> from surpyval.datasets import load_kidney
    >>> df = load_kidney()
    >>> Z = np.column_stack([df["age"], df["sex"] == 2])
    >>> model = CoxMixedEffects.fit(
    ...     df["time"], Z=Z, c=1 - df["status"], groups=df["id"]
    ... )
    >>> test = model.lr_test()
    >>> round(test["statistic"], 3), round(test["p_value"], 4)
    (4.88, 0.0136)
    >>> model.sf([30, 100], [45, 1]).round(3)
    array([0.739, 0.545])
    >>> model.sf([30, 100], [45, 1], marginal=True).round(3)
    array([0.707, 0.524])
    """

    def __init__(self) -> None:
        self.kind = "CoxMixedEffects"
        self._neg_ll: float = float("nan")
        self.tie_method = "efron"
        self.beta: np.ndarray = np.array([])
        self.center: np.ndarray = np.array([])
        self.parameter_names: "list[str]" = []
        self.feature_names: "list[str] | None" = None
        self.formula: "str | None" = None
        self._model_spec: Any = None
        self._covariance: "np.ndarray | None" = None
        self.terms: "list[dict]" = []
        self.baselines: "list[dict]" = []
        self.strata: "list[str] | None" = None
        self.log_likelihood = float("nan")
        self.log_likelihood_penalised = float("nan")
        self.log_likelihood_fitted = float("nan")
        self.log_likelihood_null = float("nan")
        self.log_likelihood_no_random = float("nan")
        self.df_penalised = float("nan")
        self.fixed_variance = False
        self.k = 0
        self.n_obs = 0
        self.n_events = 0
        self.n_events_weighted = 0.0
        self.n_obs_weighted = 0.0
        self.maximum = "unknown"
        self._data_summary: "str | None" = None
        self._psi: "np.ndarray | None" = None
        self._engine: "MixedCoxEngine | None" = None
        self._beta0: "np.ndarray | None" = None

    def __getstate__(self) -> dict:
        # The engine holds closures over the data: a pickled model keeps
        # the fit, and refits for the variance bounds.
        state = self.__dict__.copy()
        state["_engine"] = None
        return state

    # -- the likelihood ------------------------------------------------------

    def _ic_sample_size_from_data(self) -> float:
        n_censored = self.n_obs_weighted - self.n_events_weighted
        return ic_sample_size([0, 1], [self.n_events_weighted, n_censored])

    # -- the fixed effects ---------------------------------------------------

    @property
    def params(self) -> np.ndarray:
        """The fixed coefficients, ``beta``."""
        return np.asarray(self.beta, dtype=float)

    def _parameter_bounds(self) -> "list[tuple[None, None]]":
        # The support of each parameter param_cb takes: the coefficients,
        # unbounded.
        return [(None, None)] * self.beta.size

    @property
    def aliased(self) -> np.ndarray:
        """The columns of ``Z`` the data cannot determine (``nan``
        coefficients)."""
        return np.flatnonzero(np.isnan(np.asarray(self.beta, dtype=float)))

    def covariance(self) -> np.ndarray:
        """The fixed coefficients' covariance, in the order of
        ``parameter_names``: the fixed block of the inverse of the
        penalised partial likelihood's information (``coxme``'s
        ``vcov``)."""
        if self._covariance is None:
            raise no_covariance_error()
        return np.asarray(self._covariance)

    def standard_errors(self) -> np.ndarray:
        """The fixed coefficients' Wald standard errors."""
        return standard_errors_of(self.covariance())

    @property
    def p_values(self) -> np.ndarray:
        """The fixed coefficients' Wald p-values."""
        out = np.full(self.beta.shape, np.nan)
        if self._covariance is None:
            return out
        with np.errstate(all="ignore"):
            se = np.sqrt(np.diag(self._covariance))
            return 2 * norm.sf(np.abs(self.beta / se))

    def param_cb(
        self,
        name: str,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: "str | None" = None,
    ) -> np.ndarray:
        """The Wald bound on a fixed coefficient, by name. ``method`` is
        ``'wald'`` (or ``None``); likelihood-ratio and bootstrap bounds on
        the coefficients are not available (refused), and the variance
        components' bounds are :meth:`variance_cb`."""
        check_alpha_ci(alpha_ci)
        if method is not None and str(method).lower() != "wald":
            raise ValueError(
                "The mixed-effects Cox model's param_cb gives Wald bounds "
                "on the fixed coefficients only (method='wald', or None); "
                "method={!r} is not available. The variance components' "
                "profile-likelihood bounds are variance_cb().".format(method)
            )
        if name not in self.parameter_names:
            raise ValueError(
                "Unknown parameter {!r}; expected one of {}. The variance "
                "components' bounds are variance_cb().".format(
                    name, self.parameter_names
                )
            )
        check_option("bound", bound, BOUNDS)
        idx = self.parameter_names.index(name)
        est = float(self.beta[idx])
        se = float(np.sqrt(self.covariance()[idx, idx]))
        if bound == "two-sided":
            return est + np.array([-1.0, 1.0]) * _z(1 - alpha_ci / 2) * se
        q = _z(1 - alpha_ci)
        return np.array([est - q * se if bound == "lower" else est + q * se])

    def summary(self, alpha_ci: float = 0.05) -> pd.DataFrame:
        """The fixed coefficients' table, as ``CoxPH``'s :meth:`summary`:
        each coefficient, ``exp(coef)`` (the hazard ratio within a group),
        its standard error, Wald ``z``, p-value and interval."""
        check_alpha_ci(alpha_ci)
        from .._summary import coefficient_table

        se = np.full(self.beta.shape, np.nan)
        if self._covariance is not None:
            with np.errstate(all="ignore"):
                se = np.sqrt(np.diag(self._covariance))
        return coefficient_table(
            list(self.parameter_names), self.beta, se, alpha_ci
        )

    # -- the random effects --------------------------------------------------

    @property
    def random_effects(self) -> "dict[str, pd.DataFrame]":
        """Each term's predicted random effects (``coxme``'s ``ranef``):
        a DataFrame per term, a row per group (its label, as a string) and
        a column per effect, on the log-hazard scale; ``exp`` of an
        intercept is the group's frailty."""
        out = {}
        for term in self.terms:
            out[term["name"]] = pd.DataFrame(
                np.asarray(term["blups"], dtype=float),
                index=pd.Index(term["labels"], name=term["name"]),
                columns=term["effects"],
            )
        return out

    def variance_components(self) -> pd.DataFrame:
        """
        Each term's standard deviations and variances, and the
        correlations of a correlated term, as ``coxme`` prints them:
        indexed by ``(term, parameter)`` with ``parameter`` ``"var(e)"``
        for an effect ``e`` and ``"corr(e1, e2)"``; the columns
        ``variance`` (or the correlation) and ``std dev`` (``nan`` for a
        correlation).
        """
        rows, index = [], []
        for term, (kind, label, a, b) in self._components():
            S = np.asarray(term["covariance"], dtype=float)
            if kind == "var":
                rows.append({"variance": S[a, a], "std dev": np.sqrt(S[a, a])})
            else:
                rows.append(
                    {
                        "variance": np.asarray(term["correlation"])[a, b],
                        "std dev": np.nan,
                    }
                )
            index.append((term["name"], label))
        return pd.DataFrame(
            rows,
            index=pd.MultiIndex.from_tuples(
                index, names=["term", "parameter"]
            ),
            columns=["variance", "std dev"],
        )

    def _components(self) -> "list[tuple[dict, tuple]]":
        out = []
        for term in self.terms:
            effects = term["effects"]
            for a, e in enumerate(effects):
                out.append((term, ("var", f"var({e})", a, a)))
            if term["correlated"]:
                for i in range(1, len(effects)):
                    for j in range(i):
                        out.append(
                            (
                                term,
                                (
                                    "corr",
                                    f"corr({effects[j]}, {effects[i]})",
                                    i,
                                    j,
                                ),
                            )
                        )
        return out

    def lr_test(self) -> dict:
        """
        The likelihood-ratio test of the random effects together: twice
        the integrated log-likelihood less the Cox model's partial
        log-likelihood (``CoxPH`` on the same rows), and its p-value.

        Every variance is 0 under the null, on the edge of its range, so
        the statistic is not a plain chi-square. With one variance (a
        random intercept) it is the 50:50 mixture of 0 and a chi-square
        on 1 degree of freedom (Self and Liang 1987); with ``m``
        independent variances (no correlations) the binomial mixture of
        chi-squares on 0 to ``m`` degrees of freedom. With correlations
        the exact mixture depends on the information, and the plain
        chi-square on the number of variance parameters is used: its
        p-value is larger than the true one (conservative), which
        ``reference`` says. A model fitted with ``variance=`` estimated no
        variance: its statistic is reported with a ``nan`` p-value.
        """
        stat = max(
            2.0 * (self.log_likelihood - self.log_likelihood_no_random), 0.0
        )
        if self.fixed_variance:
            # Nothing was estimated: the statistic is reported, untested.
            return {
                "statistic": float(stat),
                "p_value": float("nan"),
                "df": 0,
                "null": "no random effects",
                "reference": "variances fixed (variance=): no test",
            }
        m = int(sum(len(t["effects"]) for t in self.terms))
        n_corr = int(
            sum(
                len(t["effects"]) * (len(t["effects"]) - 1) // 2
                for t in self.terms
                if t["correlated"]
            )
        )
        if n_corr:
            df = m + n_corr
            p = float(chi2.sf(stat, df)) if stat > 0 else 1.0
            reference = f"chi2({df}) (conservative)"
        else:
            from math import comb

            weights = [comb(m, j) / 2.0**m for j in range(m + 1)]
            p = sum(
                wt * (float(chi2.sf(stat, j)) if j else float(stat <= 0))
                for j, wt in enumerate(weights)
            )
            p = float(min(max(p, 0.0), 1.0)) if stat > 0 else 1.0
            reference = " + ".join(
                f"{wt:g} chi2({j})" for j, wt in enumerate(weights)
            )
            df = m
        return {
            "statistic": float(stat),
            "p_value": p,
            "df": int(df),
            "null": "no random effects (every variance 0)",
            "reference": reference,
        }

    def variance_cb(self, alpha_ci: float = 0.05) -> pd.DataFrame:
        """
        Profile-likelihood intervals of the variance components: for each
        variance (and correlation), the values at which the integrated
        log-likelihood, maximised over every other variance parameter,
        falls ``chi2_1(1 - alpha_ci) / 2`` below its maximum. A variance
        whose profile at 0 is within that distance has a lower bound of 0.
        Indexed as :meth:`variance_components`, with the columns
        ``estimate``, ``lower`` and ``upper``.

        The correlations of a term with three or more effects are
        reported without bounds (``nan``): the search runs on canonical
        partial correlations, which are the correlations only for two
        effects. Refused for a model fitted with ``variance=`` (nothing
        estimated) and for one restored with ``from_dict`` or unpickled
        (the profile needs the data: refit it).
        """
        check_alpha_ci(alpha_ci)
        if self.fixed_variance:
            raise ValueError(
                "The variances of this model were fixed (variance=): there "
                "is nothing to bound. Refit without variance= to estimate "
                "them."
            )
        engine = self._engine
        if engine is None or self._psi is None:
            raise ValueError(
                "variance_cb() needs the data the model was fitted to, "
                "which a model restored with from_dict (or unpickled) does "
                "not keep. Refit it with CoxMixedEffects.fit and bound the "
                "fit."
            )
        crit = float(chi2.ppf(1 - alpha_ci, 1)) / 2.0
        psi_hat = np.asarray(self._psi, dtype=float)
        top = engine.ilik(psi_hat)
        bounds = engine.bounds()
        rows, index = [], []
        at = 0
        for term in self.terms:
            k = len(term["effects"])
            n_par = k + (k * (k - 1) // 2 if term["correlated"] else 0)
            for local in range(n_par):
                i = at + local
                is_sd = local < k
                lo, hi = bounds[i]
                free = np.ones(psi_hat.size, dtype=bool)
                free[i] = False

                def drop(v: float, i: int = i, free: Any = free) -> float:
                    start = psi_hat.copy()
                    start[i] = v
                    _, value, _ = engine.maximise(start, free, start)
                    return top - value - crit

                if is_sd:
                    est = psi_hat[i]
                    lower = 0.0
                    if est > 0 and drop(0.0) > 0:
                        lower = brentq(drop, 0.0, est, xtol=1e-6)
                    upper = np.nan
                    probe = max(est, 0.05)
                    for _ in range(12):
                        probe = min(probe * 2, hi)
                        if drop(probe) > 0:
                            upper = brentq(drop, est, probe, xtol=1e-6)
                            break
                        if probe >= hi:
                            break
                    rows.append(
                        {
                            "estimate": est**2,
                            "lower": lower**2,
                            "upper": upper**2,
                        }
                    )
                    label = "var({})".format(term["effects"][local])
                else:
                    pair = local - k
                    i_e, j_e = _pair(pair, k)
                    label = "corr({}, {})".format(
                        term["effects"][j_e], term["effects"][i_e]
                    )
                    est = psi_hat[i]
                    if k == 2 and np.all(psi_hat[at : at + k] > 0):
                        lower = (
                            brentq(drop, lo, est, xtol=1e-7)
                            if drop(lo) > 0
                            else -1.0
                        )
                        upper = (
                            brentq(drop, est, hi, xtol=1e-7)
                            if drop(hi) > 0
                            else 1.0
                        )
                    else:
                        lower = upper = np.nan
                    rows.append(
                        {
                            "estimate": float(
                                np.asarray(term["correlation"])[i_e, j_e]
                            ),
                            "lower": lower,
                            "upper": upper,
                        }
                    )
                index.append((term["name"], label))
            at += n_par
        engine.solve(psi_hat, self._beta0)
        return pd.DataFrame(
            rows,
            index=pd.MultiIndex.from_tuples(
                index, names=["term", "parameter"]
            ),
            columns=["estimate", "lower", "upper"],
        )

    # -- prediction ----------------------------------------------------------

    def _rows(self, Z: Any) -> np.ndarray:
        p = self.beta.size
        if p == 0:
            return np.zeros((1, 0))
        if Z is None:
            raise ValueError(
                "This model was fit with covariates; 'Z' is required."
            )
        Zp = prepare_Z(Z, self.feature_names, self._model_spec)
        check_covariate_width(Zp, list(self.parameter_names))
        return covariate_rows(Zp, p)

    def _baseline(self, stratum: Any) -> dict:
        if self.strata is None:
            if stratum is not None:
                raise ValueError(
                    "This model is not stratified; drop stratum=."
                )
            return self.baselines[0]
        if stratum is None:
            raise ValueError(
                "This model is stratified: pass stratum=, one of {}.".format(
                    self.strata
                )
            )
        key = str(stratum)
        if key not in self.strata:
            raise KeyError(
                "Unknown stratum {!r}; known strata are {}.".format(
                    stratum, self.strata
                )
            )
        return self.baselines[self.strata.index(key)]

    def _groups(self, group: Any) -> "dict[str, Any]":
        if group is None:
            return {}
        if isinstance(group, dict):
            names = [t["name"] for t in self.terms]
            unknown = [k for k in group if k not in names]
            if unknown:
                raise KeyError(
                    "Unknown term(s) {}; the terms are {}.".format(
                        unknown, names
                    )
                )
            return dict(group)
        if len(self.terms) != 1:
            raise ValueError(
                "This model has {} random-effects terms: pass group= as a "
                "dict {{term: label}} (terms {}).".format(
                    len(self.terms), [t["name"] for t in self.terms]
                )
            )
        return {self.terms[0]["name"]: group}

    def _linear(
        self, rows: np.ndarray, group: Any, marginal: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Each row's log-hazard ratio (fixed part and the given groups'
        random effects) and, with ``marginal``, the variance of the
        random effects integrated out."""
        beta = np.where(np.isnan(self.beta), 0.0, self.beta)
        lp = (
            (rows - np.asarray(self.center, dtype=float)) @ beta
            if beta.size
            else np.zeros(rows.shape[0])
        )
        given = self._groups(group)
        var = np.zeros(rows.shape[0])
        for term in self.terms:
            effects = np.column_stack(
                ([np.ones(rows.shape[0])] if term["intercept"] else [])
                + [rows[:, j] for j in term["slope_columns"]]
            )
            if term["name"] in given:
                label = given[term["name"]]
                if is_missing_event(label):
                    lp = lp + np.nan
                    continue
                if str(label) not in term["labels"]:
                    raise KeyError(
                        "Unknown group {!r} of term {!r}; known groups "
                        "are {}...".format(
                            label, term["name"], term["labels"][:8]
                        )
                    )
                b = np.asarray(term["blups"])[term["labels"].index(str(label))]
                lp = lp + effects @ b
            elif marginal:
                S = np.asarray(term["covariance"], dtype=float)
                var = var + np.einsum("ij,jk,ik->i", effects, S, effects)
        return lp, var

    @staticmethod
    def _cumulative(s: np.ndarray, var: np.ndarray) -> np.ndarray:
        """``s`` (``exp(lp) H0``) or, where ``var`` > 0, its log-normal
        Laplace transform's ``-log``: the random effects of variance
        ``var`` integrated out."""
        s = np.asarray(s, dtype=float)
        var = np.broadcast_to(var, s.shape)
        out = s.copy()
        for v in np.unique(var[var > 0]):
            m = var == v
            out[m] = -lognormal_log_integral(0.0, s[m], float(v))
        return out

    def _H_parts(
        self, x: Any, Z: Any, group: Any, marginal: bool, stratum: Any
    ) -> tuple:
        x = np.asarray(x, dtype=float)
        rows = self._rows(Z)
        if rows.shape[0] > 1:
            check_paired_rows(np.size(x), rows.shape[0], grid=False)
        lp, var = self._linear(rows, group, marginal)
        base = self._baseline(stratum)
        H0 = SemiParametricRegressionModel._baseline_step(
            base["x"], base["H0"], x
        ).reshape(x.shape)
        h0 = SemiParametricRegressionModel._baseline_step(
            base["x"], base["h0"], x
        ).reshape(x.shape)
        eta = np.exp(lp)
        if eta.size == 1:
            eta, var = eta[0], var[0]
        return x, eta, var, H0, h0

    def Hf(
        self,
        x: Any,
        Z: Any = None,
        group: Any = None,
        marginal: bool = False,
        stratum: Any = None,
    ) -> np.ndarray:
        """The cumulative hazard: of a unit of the group ``group`` (by
        default a new group, random effects 0), or with ``marginal=True``
        averaged over the random effects not given."""
        x, eta, var, H0, _ = self._H_parts(x, Z, group, marginal, stratum)
        s = np.asarray(eta * H0, dtype=float)
        return self._cumulative(s, var * np.ones(s.shape))

    def sf(
        self,
        x: Any,
        Z: Any = None,
        group: Any = None,
        marginal: bool = False,
        stratum: Any = None,
    ) -> np.ndarray:
        """The survival function (see :meth:`Hf`)."""
        return np.exp(-self.Hf(x, Z, group, marginal, stratum))

    def ff(
        self,
        x: Any,
        Z: Any = None,
        group: Any = None,
        marginal: bool = False,
        stratum: Any = None,
    ) -> np.ndarray:
        """The failure probability (see :meth:`Hf`)."""
        return -np.expm1(-self.Hf(x, Z, group, marginal, stratum))

    def hf(
        self,
        x: Any,
        Z: Any = None,
        group: Any = None,
        marginal: bool = False,
        stratum: Any = None,
    ) -> np.ndarray:
        """The hazard's jump at the latest baseline time at or before
        ``x`` (a step baseline has no hazard rate, as ``CoxPH``'s)."""
        x, eta, var, H0, h0 = self._H_parts(x, Z, group, marginal, stratum)
        s1 = np.asarray(eta * H0, dtype=float)
        s0 = np.asarray(eta * (H0 - h0), dtype=float)
        v = var * np.ones(s1.shape)
        return self._cumulative(s1, v) - self._cumulative(s0, v)

    def df(
        self,
        x: Any,
        Z: Any = None,
        group: Any = None,
        marginal: bool = False,
        stratum: Any = None,
    ) -> np.ndarray:
        """The density's jump: :meth:`hf` times :meth:`sf`."""
        return self.hf(x, Z, group, marginal, stratum) * self.sf(
            x, Z, group, marginal, stratum
        )

    @keeps_query_shape
    def qf(
        self,
        p: Any,
        Z: Any = None,
        group: Any = None,
        marginal: bool = False,
        stratum: Any = None,
    ) -> np.ndarray:
        """The quantile: the first baseline time at which the failure
        probability reaches ``p`` (``nan`` where it never does), as
        ``CoxPH.qf``; ``Z`` is paired with ``p`` as :meth:`sf` pairs it
        with ``x``."""
        rows = self._rows(Z)
        self._groups(group)
        u, rows, _ = paired_probabilities(p, rows)
        times = np.asarray(self._baseline(stratum)["x"], dtype=float)
        uniq, which = unique_rows(rows)
        F = np.empty((uniq.shape[0], times.size))
        for k in range(uniq.shape[0]):
            Zk = uniq[k : k + 1] if uniq.shape[1] else None
            F[k] = self.ff(times, Zk, group, marginal, stratum)
        return step_quantiles(F[which], times, u)

    def random(self, *args: Any, **kwargs: Any) -> Any:
        """Refused: see the message."""
        raise NotImplementedError(_NO_RANDOM)

    # -- printout and serialisation ------------------------------------------

    def __repr__(self) -> str:
        from .._summary import coefficient_repr, format_table

        out = (
            "Mixed-Effects Cox Regression SurPyval Model"
            "\n==========================================="
            "\nBaseline            : unspecified (Cox); {} ties{}".format(
                self.tie_method,
                "" if self.strata is None else "; strata " + str(self.strata),
            )
            + "\nObservations        : {} (events {})".format(
                self.n_obs, self.n_events
            )
        )
        for term in self.terms:
            out += "\nGroups              : {} ({})".format(
                term["name"], len(term["labels"])
            )
        if self.beta.size:
            out += (
                "\nFixed coefficients  : exp(coef) is the hazard ratio "
                "within a group; Wald 95% intervals\n"
            ) + coefficient_repr(self.summary())
        table = self.variance_components()
        out += "\nRandom effects      : (no standard errors; variance_cb() "
        out += "gives profile-likelihood intervals)\n"
        out += format_table(table, ["variance", "std dev"])
        out += (
            "\nLog-likelihood      : integrated {:.4f}, penalised {:.4f}, "
            "fitted {:.4f}, null {:.4f}".format(
                self.log_likelihood,
                self.log_likelihood_penalised,
                self.log_likelihood_fitted,
                self.log_likelihood_null,
            )
        )
        return out

    def to_dict(self) -> dict:
        """Serialise to a plain, JSON-serialisable ``dict``."""
        out: dict[str, Any] = {
            "model": "CoxMixedEffectsModel",
            "kind": self.kind,
            "tie_method": self.tie_method,
            "beta": np.asarray(self.beta, float).tolist(),
            "center": np.asarray(self.center, float).tolist(),
            "parameter_names": list(self.parameter_names),
            "strata": self.strata,
            "baselines": [
                {k: np.asarray(v, float).tolist() for k, v in b.items()}
                for b in self.baselines
            ],
            "terms": [
                {
                    k: (
                        np.asarray(v, float).tolist()
                        if isinstance(v, np.ndarray)
                        else v
                    )
                    for k, v in t.items()
                }
                for t in self.terms
            ],
            "_neg_ll": to_native(self._neg_ll),
            "log_likelihood_penalised": to_native(
                self.log_likelihood_penalised
            ),
            "log_likelihood_fitted": to_native(self.log_likelihood_fitted),
            "log_likelihood_null": to_native(self.log_likelihood_null),
            "log_likelihood_no_random": to_native(
                self.log_likelihood_no_random
            ),
            "df_penalised": to_native(self.df_penalised),
            "fixed_variance": bool(self.fixed_variance),
            "k": int(self.k),
            "n_obs": int(self.n_obs),
            "n_events": int(self.n_events),
            "n_events_weighted": float(self.n_events_weighted),
            "n_obs_weighted": float(self.n_obs_weighted),
            "data_summary": self._data_summary,
            **maximum_entry(self.maximum),
        }
        if self._covariance is not None:
            out["covariance"] = [
                [to_native(v) for v in row]
                for row in np.asarray(self._covariance, float).tolist()
            ]
        serialise_covariate_meta(self, out)
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "CoxMixedEffectsModel":
        """Rebuild a model from a :meth:`to_dict` dictionary (without
        its data: :meth:`variance_cb` then refuses)."""
        require_model_tag(
            model_dict, "CoxMixedEffectsModel", "a mixed-effects Cox model"
        )
        out = cls()
        out.tie_method = model_dict["tie_method"]
        out.beta = np.array(model_dict["beta"], dtype=float)
        out.center = np.array(model_dict["center"], dtype=float)
        out.parameter_names = list(model_dict["parameter_names"])
        out.strata = model_dict.get("strata")
        out.baselines = [
            {k: np.array(v, dtype=float) for k, v in b.items()}
            for b in model_dict["baselines"]
        ]
        out.terms = []
        for t in model_dict["terms"]:
            term = dict(t)
            for key in ("covariance", "sd", "correlation", "blups"):
                term[key] = np.array(t[key], dtype=float)
            out.terms.append(term)
        out.log_likelihood = -float(model_dict["_neg_ll"])
        for key in (
            "log_likelihood_penalised",
            "log_likelihood_fitted",
            "log_likelihood_null",
            "log_likelihood_no_random",
            "df_penalised",
        ):
            value = model_dict.get(key)
            setattr(out, key, float("nan") if value is None else float(value))
        out.fixed_variance = bool(model_dict.get("fixed_variance", False))
        out.k = int(model_dict["k"])
        out.n_obs = int(model_dict["n_obs"])
        out.n_events = int(model_dict["n_events"])
        out.n_events_weighted = float(model_dict["n_events_weighted"])
        out.n_obs_weighted = float(model_dict["n_obs_weighted"])
        out._data_summary = model_dict.get("data_summary")
        if "covariance" in model_dict:
            out._covariance = np.array(
                [
                    [np.nan if v is None else v for v in row]
                    for row in model_dict["covariance"]
                ],
                dtype=float,
            )
        restore_covariate_meta(out, model_dict)
        out.maximum = restored_maximum(model_dict)
        return out


def _pair(index: int, k: int) -> tuple[int, int]:
    """The ``(i, j)``, ``j < i``, of the ``index``-th correlation of a
    ``k``-effect term, row by row."""
    at = 0
    for i in range(1, k):
        for j in range(i):
            if at == index:
                return i, j
            at += 1
    raise IndexError(index)


def solution_of(model: CoxMixedEffectsModel) -> "Solution | None":
    """The penalised fit at the model's variances, where it has its data."""
    if model._engine is None or model._psi is None:
        return None
    return model._engine.solve(model._psi, model._beta0)


CoxMixedEffects = CoxMixedEffectsFitter()


# -- CoxFrailty(family="lognormal") ---------------------------------------

# The step, in log theta, of the profile's second difference that gives
# theta's standard error (as the gamma Cox frailty's).
_CURVATURE_STEP = 0.02


def lognormal_cox_frailty(
    x: np.ndarray,
    Zfull: np.ndarray,
    Zk: np.ndarray,
    means: np.ndarray,
    c: np.ndarray,
    w: np.ndarray,
    labels: np.ndarray,
    inv: np.ndarray,
    kept: np.ndarray,
    cox: Any,
    maximum: str,
    tie_method: str,
    theta: "float | None",
    center: bool,
) -> Any:
    """``CoxFrailty.fit(..., family="lognormal")``: the one-term random
    intercept of the mixed-effects engine, reported as a
    :class:`~surpyval.CoxFrailtyModel`. ``theta`` is the variance of the
    log-frailty, its standard error from the curvature of the profile
    integrated likelihood in ``log theta`` (delta method)."""
    from .cox_frailty import CoxFrailtyModel

    n_groups = labels.shape[0]
    term = TermDesign(
        "group",
        labels,
        inv,
        np.ones((x.size, 1)),
        ["(Intercept)"],
        True,
        [],
        True,
    )
    tl = np.full(x.size, -np.inf)
    engine = MixedCoxEngine(x, Zk - means, c, w, tl, None, [term], tie_method)
    p_all = Zfull.shape[1]
    beta0 = (
        np.asarray(cox.beta, dtype=float)[kept] if cox is not None else None
    )
    no_frailty = engine.solve(np.zeros(1), beta0).ilik
    theta_var = float("nan")
    if theta is None:
        psi, _, _ = engine.maximise(engine.start())
        alt, _, _ = engine.maximise(np.array([0.05]))
        if engine.ilik(alt) > engine.ilik(psi) + 1e-9:
            psi = alt
        if psi[0] >= SD_MAX * (1 - 1e-9):
            maximum = "no finite maximum"
            warn_no_maximum(
                "the log-frailty's standard deviation runs to the edge of "
                "its search, {} (groups that differ more than any finite "
                "variance explains)".format(SD_MAX),
                "The reported theta, its standard error and its bounds are "
                "meaningless",
                "check for groups whose events all come first, or fix "
                "theta with theta=",
            )
    else:
        if theta > SD_MAX**2:
            raise ValueError(
                "theta must be at most {} for the log-normal frailty (a "
                "standard deviation of {} on the log-hazard); got "
                "{!r}.".format(SD_MAX**2, SD_MAX, theta)
            )
        psi = np.array([np.sqrt(theta)])
    sol = engine.solve(psi, beta0)
    theta_hat = float(psi[0] ** 2)
    if theta is None and theta_hat > 0:
        h = _CURVATURE_STEP
        values = [
            engine.ilik(np.array([np.sqrt(theta_hat * np.exp(s * h))]))
            for s in (-1, 0, 1)
        ]
        curvature = -(values[0] - 2 * values[1] + values[2]) / h**2
        if curvature > 0:
            theta_var = float(theta_hat**2 / curvature)
        engine.solve(psi, beta0)
    if not sol.converged and maximum == "verified":
        maximum = "unverified"
        warn_unverified(
            "The log-normal Cox frailty fit",
            "Newton's method on the penalised partial likelihood did not "
            "converge at the reported theta",
        )
    p = engine.p
    try:
        cov_beta = np.linalg.inv(sol.hessian)[:p, :p]
    except np.linalg.LinAlgError:
        cov_beta = None

    times, r, d, h0 = CoxPH.baseline(
        np.r_[sol.beta, 1.0],
        x,
        c,
        w,
        np.column_stack([Zk - means, sol.b[inv]]),
        None,
        tie_method,
    )
    if kept.size and not center:
        r, h0 = baseline_at_origin(
            sol.beta, means, Zk, r, h0, hint=_ORIGIN_HINT
        )
    model = CoxFrailtyModel()
    model.family = "lognormal"
    model.tie_method = tie_method
    model.beta = expand(sol.beta, kept, p_all) if p_all else np.zeros(0)
    model.center = (
        covariate_center(Zfull, w) if center and p_all else np.zeros(p_all)
    )
    model.theta = theta_hat
    model.x = times
    model.h0 = h0
    model.H0 = np.cumsum(h0)
    model.group_labels = list(labels)
    model.frailties = {
        str(lab): float(u) for lab, u in zip(labels, np.exp(sol.b))
    }
    names = coefficient_names(p_all, fit_columns(), ["theta"]) + ["theta"]
    covariance = np.full((p_all + 1, p_all + 1), np.nan)
    if cov_beta is not None and kept.size:
        covariance[np.ix_(kept, kept)] = cov_beta
        covariance[np.ix_(kept, [p_all])] = 0.0
        covariance[np.ix_([p_all], kept)] = 0.0
    covariance[p_all, p_all] = theta_var
    if maximum == "no finite maximum" and cox is not None:
        off = np.flatnonzero(np.isnan(cox.standard_errors()))
        covariance[off, :] = np.nan
        covariance[:, off] = np.nan
    model._covariance = covariance
    model.parameter_names = names
    model.log_likelihood = float(sol.ilik)
    model.log_likelihood_no_frailty = float(no_frailty)
    model.k = int(kept.size) + (1 if theta is None else 0)
    model.n_obs = int(x.size)
    model.n_events = int((c == 0).sum())
    model.n_events_weighted = float(w[c == 0].sum())
    model.n_obs_weighted = float(w.sum())
    model.n_groups = n_groups
    model._data_summary = data_summary(c, w, x=x)
    model._fit_data = {"x": x, "c": c, "n": w, "Z": Zfull}
    model.maximum = maximum
    return model
