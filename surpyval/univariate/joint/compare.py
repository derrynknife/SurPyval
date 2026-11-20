"""Comparing what the hazard follows: one joint model fitted with each
link to the same data, ranked by an information criterion, with
likelihood-ratio tests of the nested pairs and the result said in words
(:class:`LinkComparison`, made by :meth:`JointPHFitter.compare_links
<surpyval.univariate.joint.joint_ph.JointPHFitter.compare_links>`)."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from surpyval.utils.no_maximum import quiet_maximum_warnings
from surpyval.utils.validation import check_option
from surpyval.utils.warnings import collecting_warnings

if TYPE_CHECKING:
    import pandas as pd

    from .joint_model import JointModel
    from .joint_ph import JointPHFitter

#: The links compared by default: each alone, and the slope and the area
#: each beside the value.
DEFAULT_LINKS: tuple[tuple[str, ...], ...] = (
    ("value",),
    ("slope",),
    ("cumulative",),
    ("value", "slope"),
    ("value", "cumulative"),
)

#: The criteria the links can be ranked by (lower is better).
METRICS = ("aic", "aic_c", "bic")

#: How each criterion is written in words.
_METRIC_NAMES = {"aic": "AIC", "aic_c": "AICc", "bic": "BIC"}

#: Two links whose criteria differ by less than this are "no real
#: preference" (Burnham and Anderson 2002, section 2.6), and the simpler
#: is chosen: the one with fewer parameters, then the level before the
#: slope before the area. By AIC, a link one effect richer than another
#: is then chosen only where twice its gain in log-likelihood exceeds 4:
#: a likelihood-ratio test at the 4.55% level.
NO_REAL_PREFERENCE = 2.0

#: A link this far or further above the best is "clearly worse"
#: (Burnham and Anderson: essentially no support from the data); between
#: ``NO_REAL_PREFERENCE`` and this it is "worse".
CLEARLY_WORSE = 10.0

#: Below this, a likelihood-ratio test's p-value is read as "the extra
#: effect adds to the fit".
_TEST_LEVEL = 0.05

#: What the hazard follows, in words: in a sentence, and as a noun.
_FOLLOWS = {
    "value": "the marker's level",
    "slope": "how fast the marker is changing",
    "cumulative": "the marker's accumulated exposure (the area under its "
    "path)",
}
_NOUNS = {
    "value": "the level",
    "slope": "the rate of change",
    "cumulative": "the accumulated exposure",
}

#: How far below the simpler link's log-likelihood (relative, with an
#: absolute floor of the same) the richer one may be and still be taken
#: as at its maximum: the adaptive quadrature's own error.
_LL_TOL = 1e-6


def link_label(link: Any) -> str:
    """The name of a link in the comparison: ``"value"``, ``"value +
    slope"``, ... (the links in their fixed order)."""
    return " + ".join(link)


def _and(words: list[str]) -> str:
    """``"a"``, ``"a and b"``, ``"a, b and c"``."""
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + " and " + words[-1]


def _follows(link: Any) -> str:
    """What a hazard with this link follows, in words."""
    return _and([_FOLLOWS[term] for term in link])


def _nouns(link: Any) -> str:
    """The terms of a link as nouns: ``"the level and the rate of
    change"``."""
    return _and([_NOUNS[term] for term in link])


def _p(p: float) -> str:
    """A p-value as it is written in words."""
    return "< 0.0001" if p < 1e-4 else f"{p:.2g}"


def _plain(value: Any) -> Any:
    """A table entry as strict JSON takes it: ``None`` for a number
    that is not finite, Python's own int and float."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _order(link: Any) -> tuple:
    """The links' order on a tie: the level before the slope before the
    area, each alone before it is beside another."""
    from .joint_ph import LINKS

    return tuple(LINKS.index(term) for term in link)


def _has_cumulative(link: Any) -> bool:
    """Whether ``link`` (a link or a list of them) follows the
    accumulated exposure."""
    return "cumulative" in ([link] if isinstance(link, str) else list(link))


def _candidates(fitter: JointPHFitter, links: Any) -> tuple[list, list, list]:
    """The fitters to compare, one per link; the default links left out
    because the path cannot take them, and why."""
    from .joint_ph import JointPH, cannot_follow

    left_out: list[str] = []
    reasons: list[str] = []
    if links is None:
        path = fitter.path_model
        chosen: list[Any] = []
        for link in DEFAULT_LINKS:
            why = cannot_follow(path, link)
            if why:
                left_out.append(link_label(link))
                reasons += [r for r in why if r not in reasons]
            else:
                chosen.append(link)
        if len(chosen) < 2:
            raise ValueError(
                f"The {path.name} path can give the hazard only the "
                f"{' and '.join(link_label(c) for c in chosen)} link to "
                f"follow ({'; '.join(reasons)}), so there is nothing to "
                "compare: fit that link on its own, or compare the links "
                "with another path."
            )
    else:
        if isinstance(links, str):
            raise ValueError(
                "'links' is a list of the links to compare, each a link "
                "('value', 'slope' or 'cumulative') or a list of them: "
                "links=['value', 'slope'] compares two; got "
                f"{links!r}. Leave it out to compare the default five."
            )
        try:
            chosen = list(links)
        except TypeError:
            chosen = []
        if len(chosen) < 2:
            raise ValueError(
                "'links' needs two or more links to compare, each a link "
                "('value', 'slope' or 'cumulative') or a list of them, "
                "such as ['value', 'slope', ['value', 'slope']]; got "
                f"{links!r}."
            )
    # a weight is on the accumulated exposure: the links that follow it
    fitters = [
        JointPH(
            fitter.dist,
            fitter.path_model,
            fitter.varying,
            link,
            fitter.lag,
            fitter.weight if _has_cumulative(link) else None,
        )
        for link in chosen
    ]
    seen: dict[tuple, int] = {}
    for k, candidate in enumerate(fitters):
        if candidate.link in seen:
            raise ValueError(
                f"'links' names the link {link_label(candidate.link)!r} "
                f"twice (entries {seen[candidate.link]} and {k}): give "
                "each once."
            )
        seen[candidate.link] = k
    return fitters, left_out, reasons


def _reason(error: Exception) -> str:
    """The first line of an error, cut to 120 characters."""
    text = str(error).strip().splitlines()
    first = text[0] if text else type(error).__name__
    return first if len(first) <= 120 else first[:117] + "..."


def _fit_one(
    candidate: JointPHFitter, args: tuple, kwargs: dict, start: Any
) -> tuple:
    """``(model, warnings)``: the candidate's fit, its search started
    from the path part ``start`` (another link's fit of the same data)
    where given. A warm start that does not reach a verified maximum, or
    raises, is set aside and the fit made again from its own start, so a
    warm start never gives a worse answer than the fit alone. A
    ``ValueError`` of the fit from its own start is raised."""
    if start is not None:
        candidate._path_start = start
        try:
            with collecting_warnings() as caught, quiet_maximum_warnings():
                model = candidate.fit(*args, **kwargs)
        except ValueError:
            model = None
        finally:
            candidate._path_start = None
        if model is not None and model.maximum == "verified":
            return model, caught
    with collecting_warnings() as caught, quiet_maximum_warnings():
        model = candidate.fit(*args, **kwargs)
    return model, caught


def compare_links(
    fitter: JointPHFitter,
    args: tuple,
    kwargs: dict,
    links: Any = None,
    metric: str = "aic",
) -> LinkComparison:
    """The work of :meth:`JointPHFitter.compare_links`: each candidate
    fitted to the same data, then ranked."""
    check_option(
        "metric",
        metric,
        METRICS,
        "It is the criterion the links are ranked by (lower is better).",
    )
    fitters, left_out, reasons = _candidates(fitter, links)
    if left_out:
        warnings.warn(
            "; ".join(reasons)
            + f": {len(left_out)} link(s) were left out of the comparison "
            f"({', '.join(left_out)}).",
            stacklevel=3,
        )
    reference = kwargs.get("marker_reference")
    fitted: dict[str, JointModel] = {}
    failed: dict[str, str] = {}
    errors: list[Exception] = []
    said: dict[tuple, Any] = {}
    # The value link first: its path estimates start the other fits (the
    # path part changes little from link to link), or, without it, the
    # first fit's.
    queue = sorted(fitters, key=lambda f: f.link != ("value",))
    start = None
    for candidate in queue:
        label = link_label(candidate.link)
        own = dict(kwargs)
        if reference is not None and "value" not in candidate.link:
            # The value's reference moves alpha only, and only where the
            # hazard follows the value: the other links are fitted
            # without it.
            del own["marker_reference"]
        try:
            model, caught = _fit_one(candidate, args, own, start)
        except ValueError as error:
            failed[label] = _reason(error)
            errors.append(error)
            continue
        fitted[label] = model
        if start is None and model.maximum == "verified":
            start = getattr(model, "_path_start", None)
        for kept in caught:
            # A warning about the data comes from every fit: said once
            said.setdefault((str(kept.message), kept.category), kept)
    if not fitted:
        if len({str(e) for e in errors}) == 1:
            # The same for every link: about the data, as the fit says it
            raise errors[0]
        raise ValueError(
            "No link could be fitted to these data: "
            + "; ".join(f"{k} ({v})" for k, v in failed.items())
        )
    for kept in said.values():
        warnings.warn_explicit(
            kept.message, kept.category, kept.filename, kept.lineno
        )
    # in the order the links were given
    order = [link_label(f.link) for f in fitters]
    models = {label: fitted[label] for label in order if label in fitted}
    failed = {label: failed[label] for label in order if label in failed}
    out = LinkComparison(models, failed, metric)
    if failed:
        warnings.warn(
            f"compare_links could not fit {len(failed)} of "
            f"{len(fitters)} links, left out of the ranking: "
            + "; ".join(f"{k} ({v})" for k, v in failed.items())
            + ". Fit one on its own (JointPH(..., link=...).fit) for the "
            "whole message.",
            stacklevel=3,
        )
    set_aside = [
        label for label, model in models.items() if model.maximum != "verified"
    ]
    if set_aside:
        warnings.warn(
            f"compare_links set aside {len(set_aside)} fit(s) that did "
            f"not reach a verified maximum ({', '.join(set_aside)}): a "
            "criterion or a likelihood-ratio test of such a fit means "
            "nothing, so they are neither ranked nor tested. Best: "
            f"{out.best_label or 'none'}. Fit each on its own "
            "(JointPH(..., link=...).fit) to see why.",
            stacklevel=3,
        )
    return out


class LinkComparison:
    """
    One joint model fitted with each of several links -- what the hazard
    follows of the marker's path: its current value (its level), its
    slope (how fast it is changing), the area under it (the exposure it
    has accumulated), or more than one -- to the same data, compared and
    said in words. Made by :meth:`JointPHFitter.compare_links
    <surpyval.univariate.joint.joint_ph.JointPHFitter.compare_links>`,
    as in ``WeibullJointPH.compare_links(data)``.

    Every candidate has the fitter's baseline, path model and varying
    path parameters and is fitted to the same items and readings, so
    their log-likelihoods (of the readings and the lifetimes together)
    and criteria compare directly with each other -- not with a model of
    the lifetimes alone. A fit that is not a verified maximum is shown
    but neither ranked nor tested.

    Links whose criteria differ by less than 2 are "no real preference",
    and the simpler of them is the best: the one with fewer parameters,
    then the level before the slope before the area. By AIC, a link one
    effect richer than another is therefore the best only where the
    likelihood-ratio test of the extra effect rejects it at the 4.55%
    level (twice the gain in log-likelihood above 4).

    Attributes
    ----------
    models : dict of str to JointModel
        Each fitted candidate, by its link's name (``"value"``,
        ``"slope"``, ``"value + slope"``, ...).
    failed : dict of str to str
        Each link that could not be fitted, with the reason.
    metric : str
        The criterion the links are ranked by: ``"aic"``, ``"aic_c"`` or
        ``"bic"`` (lower is better).
    best : JointModel or None
        The simplest ranked fit within 2 of the lowest ``metric``
        (``None`` where no fit is a verified maximum).

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import ItemLifetime, SurpyvalData, WeibullJointPH
    >>> rng = np.random.default_rng(3)
    >>> items = []
    >>> for k in range(40):
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
    >>> comparison = WeibullJointPH.compare_links(
    ...     SurpyvalData.from_items(items), links=["value", "slope"]
    ... )
    >>> sorted(comparison.models), comparison.metric
    (['slope', 'value'], 'aic')
    >>> comparison.best.link
    ('value',)
    """

    def __init__(
        self, models: dict, failed: dict, metric: str = "aic"
    ) -> None:
        self.models: dict[str, JointModel] = dict(models)
        self.failed: dict[str, str] = dict(failed)
        self.metric = metric
        self.best: JointModel | None = self._choose()

    def _measure(self, model: JointModel) -> float:
        return float(getattr(model, self.metric)())

    def _ranked(self) -> dict[str, JointModel]:
        """The candidates that are ranked and tested: the verified fits
        with a finite criterion."""
        return {
            label: model
            for label, model in self.models.items()
            if model.maximum == "verified"
            and np.isfinite(self._measure(model))
        }

    def _choose(self) -> JointModel | None:
        """The best: of the ranked fits within ``NO_REAL_PREFERENCE`` of
        the lowest criterion, the one with the fewest parameters, then
        the level first, then the lowest criterion."""
        ranked = list(self._ranked().values())
        if not ranked:
            return None
        lowest = min(self._measure(model) for model in ranked)
        close = [
            model
            for model in ranked
            if self._measure(model) - lowest < NO_REAL_PREFERENCE
        ]
        return min(
            close,
            key=lambda m: (int(m.k), _order(m.link), self._measure(m)),
        )

    @property
    def best_link(self) -> tuple | None:
        """The link of the best fit, as its ``link`` (``("value",
        "slope")``, ...); ``None`` where no fit is ranked."""
        return None if self.best is None else tuple(self.best.link)

    @property
    def best_label(self) -> str | None:
        """The name of the best fit's link (``"value + slope"``, ...)."""
        return None if self.best is None else link_label(self.best.link)

    def _nested(self) -> dict[str, tuple[str, float]]:
        """``{richer: (simpler, p_value)}``: for each ranked link with a
        ranked link nested in it, the test of the one it is set against
        in the table -- the best, where the best is nested in it, else
        the nested link with the lowest criterion."""
        tests = self.likelihood_ratio_tests()
        tests = tests[np.isfinite(tests["p_value"].to_numpy(dtype=float))]
        out: dict[str, tuple[str, float]] = {}
        for richer, rows in tests.groupby("richer", sort=False):
            simpler = list(rows["simpler"])
            if self.best_label in simpler:
                chosen = self.best_label
            else:
                chosen = min(
                    simpler, key=lambda s: self._measure(self.models[s])
                )
            p = rows.loc[rows["simpler"] == chosen, "p_value"].iloc[0]
            out[str(richer)] = (str(chosen), float(p))
        return out

    def _test_words(self, richer: str, simpler: str, p: float) -> str:
        """What a likelihood-ratio test says, in words: ``"the level adds
        nothing (test p = 0.53)"``."""
        small = self.models[simpler].link
        extra = [
            term for term in self.models[richer].link if term not in small
        ]
        verb = "adds" if len(extra) == 1 else "add"
        said = "to the fit" if p < _TEST_LEVEL else "nothing"
        return f"{_nouns(extra)} {verb} {said} (test p = {_p(p)})"

    @property
    def table(self) -> pd.DataFrame:
        """
        One row per link, best first, then the other ranked links by
        ``metric``, indexed by the link's name: the number of parameters
        ``k``, the ``log_likelihood``, ``aic``, ``aic_c`` and ``bic``;
        ``delta``, the link's ``metric`` less the best's (below 2 either
        way is no real preference, and can be below 0 where the best was
        chosen as the simpler); ``weight``, each ranked link's share of
        ``exp(-delta / 2)`` (the Akaike weights, ranked by AIC);
        ``lr_against`` and ``lr_p_value``, where a ranked link has another
        nested in it, the likelihood-ratio test of its extra effects
        against that one (the best where the best is nested in it, else
        the nested link with the lowest criterion; every pair is in
        :meth:`likelihood_ratio_tests`); the fit's ``maximum``;
        ``status``: ``"best"``, ``"ranked"``, ``"set aside"`` (with the
        ``reason``) or ``"failed"`` (could not be fitted, with the
        ``reason``); and ``reads_as``, the row in words: ``"best"``, ``"as
        good: no real preference"`` (within 2 of the best), ``"worse"``
        (2 to 10 above), ``"clearly worse"`` (10 or more above), ``"set
        aside"`` or ``"not fitted"``, then what its test says, such as
        ``"the level adds nothing (test p = 0.53)"``.
        """
        import pandas as pd

        ranked = self._ranked()
        nested = self._nested()
        best_value = np.nan if self.best is None else self._measure(self.best)
        records: list[dict[str, Any]] = []
        for label, model in self.models.items():
            measure = self._measure(model)
            delta = measure - best_value
            if model is self.best:
                status, reason, words = "best", "", "best"
            elif label in ranked:
                status, reason = "ranked", ""
                if delta < NO_REAL_PREFERENCE:
                    words = "as good: no real preference"
                elif delta < CLEARLY_WORSE:
                    words = "worse"
                else:
                    words = "clearly worse"
            else:
                status, words = "set aside", "set aside"
                reason = (
                    "the fit is not a verified maximum"
                    if model.maximum != "verified"
                    else f"its {self.metric} is not finite"
                )
            against, p_value = nested.get(label, ("", np.nan))
            if against:
                words += "; " + self._test_words(label, against, p_value)
            records.append(
                {
                    "link": label,
                    "k": int(model.k),
                    "log_likelihood": float(model.log_likelihood),
                    "aic": float(model.aic()),
                    "aic_c": float(model.aic_c()),
                    "bic": float(model.bic()),
                    "delta": delta,
                    "lr_against": against,
                    "lr_p_value": p_value,
                    "maximum": model.maximum,
                    "status": status,
                    "reads_as": words,
                    "reason": reason,
                    "_measure": measure,
                }
            )
        for label, why in self.failed.items():
            records.append(
                {
                    "link": label,
                    "k": np.nan,
                    "log_likelihood": np.nan,
                    "aic": np.nan,
                    "aic_c": np.nan,
                    "bic": np.nan,
                    "delta": np.nan,
                    "lr_against": "",
                    "lr_p_value": np.nan,
                    "maximum": "",
                    "status": "failed",
                    "reads_as": "not fitted",
                    "reason": why,
                    "_measure": np.nan,
                }
            )
        order = {"best": 0, "ranked": 1, "set aside": 2, "failed": 3}
        records.sort(
            key=lambda r: (
                order[r["status"]],
                np.inf if np.isnan(r["_measure"]) else r["_measure"],
            )
        )
        table = (
            pd.DataFrame(records).set_index("link").drop(columns="_measure")
        )
        weight = np.full(len(table), np.nan)
        ok = table["status"].isin(["best", "ranked"]).to_numpy()
        if ok.any():
            delta = table["delta"].to_numpy()[ok]
            relative = np.exp(-0.5 * (delta - delta.min()))
            weight[ok] = relative / relative.sum()
        table.insert(table.columns.get_loc("delta") + 1, "weight", weight)
        return table

    @property
    def verdict(self) -> str:
        """
        The comparison in plain words: what the risk follows (the best
        link), whether a richer link improves on it, which links the data
        cannot tell from it, and how much worse the rest are. For
        example: ``"Risk follows how fast the marker is changing. Adding
        the level does not improve the fit. The remaining 3 links fit
        clearly worse (AIC 13.4 or more higher)."``
        """
        if self.best is None:
            return (
                "No fit reached a verified maximum, so the comparison "
                "cannot say which link the data support: fit each link on "
                "its own (JointPH(..., link=...).fit) to see why."
            )
        name = _METRIC_NAMES[self.metric]
        best = self.best
        best_value = self._measure(best)
        others = [
            model for model in self._ranked().values() if model is not best
        ]
        others.sort(key=self._measure)
        close = [
            m
            for m in others
            if self._measure(m) - best_value < NO_REAL_PREFERENCE
        ]
        worse = others[len(close) :]
        sentences = [f"Risk follows {_follows(best.link)}."]
        for model in close:
            if set(best.link) < set(model.link):
                extra = [t for t in model.link if t not in best.link]
                # a little lower, but not enough to prefer it
                how = "clearly " if self._measure(model) < best_value else ""
                sentences.append(
                    f"Adding {_nouns(extra)} does not {how}improve the fit."
                )
            else:
                gap = abs(self._measure(model) - best_value)
                sentences.append(
                    "The data cannot tell this from a risk that follows "
                    f"{_follows(model.link)} ({name} differs by "
                    f"{gap:.1f}: no real preference)."
                )
        test = self._nested().get(link_label(best.link))
        if test is not None:
            simpler, p = test
            small = self.models[simpler].link
            extra = [t for t in best.link if t not in small]
            verb = "adds" if len(extra) == 1 else "add"
            said = "to" if p < _TEST_LEVEL else "little to"
            text = (
                f"{_nouns(extra)} {verb} {said} what {_nouns(small)} alone "
                f"explains (test p = {_p(p)})."
            )
            sentences.append(text[0].upper() + text[1:])
        if worse:
            gaps = [self._measure(m) - best_value for m in worse]
            how = "clearly worse" if min(gaps) >= CLEARLY_WORSE else "worse"
            if len(worse) == 1:
                sentences.append(
                    f"A risk that follows {_follows(worse[0].link)} fits "
                    f"{how} ({name} {gaps[0]:.1f} higher)."
                )
            else:
                rest = "remaining" if close else "other"
                sentences.append(
                    f"The {rest} {len(worse)} links fit {how} ({name} "
                    f"{min(gaps):.1f} or more higher)."
                )
        left = [
            label
            for label, model in self.models.items()
            if model is not best and all(model is not m for m in others)
        ] + list(self.failed)
        if left:
            sentences.append(
                f"Not compared: {', '.join(left)} (see the table's " "reason)."
            )
        return " ".join(sentences)

    def likelihood_ratio_tests(self) -> pd.DataFrame:
        """
        The likelihood-ratio test of each nested pair of ranked links. A
        link whose terms are all in another's (``"value"`` in ``"value +
        slope"``) is that model with the other's extra effects at 0, so
        where the simpler link is right, twice the richer fit's gain in
        log-likelihood is chi-squared on as many degrees of freedom as it
        has extra effects. One row per pair: the ``simpler`` and
        ``richer`` links, the ``statistic``, ``df`` and ``p_value`` (a
        small one says the extra effects help) and a ``note``. Empty where
        no two ranked links are nested (``"value"`` and ``"slope"`` are
        not: compare those by :attr:`table`).

        Where the richer fit's log-likelihood is clearly below the
        simpler one's, it did not reach its maximum: the pair's statistic
        and p-value are ``nan``, and the ``note`` says so.
        """
        import pandas as pd
        from scipy.stats import chi2

        ranked = self._ranked()
        rows = []
        for small_label, small in ranked.items():
            for big_label, big in ranked.items():
                if not set(small.link) < set(big.link):
                    continue
                df = int(big.k - small.k)
                ll_small = float(small.log_likelihood)
                gain = float(big.log_likelihood) - ll_small
                note = ""
                if gain < -_LL_TOL * max(1.0, abs(ll_small)):
                    statistic = p_value = np.nan
                    note = (
                        "the richer fit's log-likelihood is below the "
                        "simpler one's: it did not reach its maximum"
                    )
                else:
                    # a tiny negative gain is the quadrature's round-off
                    statistic = max(2.0 * gain, 0.0)
                    p_value = float(chi2.sf(statistic, df))
                rows.append(
                    {
                        "simpler": small_label,
                        "richer": big_label,
                        "statistic": statistic,
                        "df": df,
                        "p_value": p_value,
                        "note": note,
                    }
                )
        return pd.DataFrame(
            rows,
            columns=[
                "simpler",
                "richer",
                "statistic",
                "df",
                "p_value",
                "note",
            ],
        )

    def to_dict(self) -> dict:
        """
        The comparison as a dictionary that strict JSON takes: the
        ``metric``, the ``best`` link's name and its ``best_link`` (a
        list), the ``verdict``, the ``table`` (one dictionary a row, its
        index as ``"link"``) and the ``likelihood_ratio_tests`` (one a
        pair); a number that is not finite is ``None``. The fitted models
        are not in it: save the best (or any) with its own ``to_dict``,
        as ``comparison.best.to_dict()``.
        """
        table = self.table.reset_index()
        tests = self.likelihood_ratio_tests()
        return {
            "metric": self.metric,
            "best": self.best_label,
            "best_link": None if self.best is None else list(self.best.link),
            "verdict": self.verdict,
            "table": [
                {key: _plain(value) for key, value in row.items()}
                for row in table.to_dict("records")
            ],
            "likelihood_ratio_tests": [
                {key: _plain(value) for key, value in row.items()}
                for row in tests.to_dict("records")
            ],
        }

    def __repr__(self) -> str:
        import textwrap

        shown = self.table[
            [
                "k",
                "log_likelihood",
                self.metric,
                "delta",
                "weight",
                "lr_p_value",
                "status",
            ]
        ]
        verdict = textwrap.fill(
            self.verdict,
            width=79,
            initial_indent="Verdict             : ",
            subsequent_indent=" " * 22,
        )
        out = (
            "Joint model link comparison"
            "\n==========================="
            f"\nRanked by           : {self.metric} (lower is better)"
            f"\nBest                : {self.best_label or 'none'}"
            f"\n{verdict}"
            "\n" + shown.to_string(float_format=lambda v: f"{v:.6g}")
        )
        if self.failed:
            out += "\nNot fitted          : " + "; ".join(
                f"{k} ({v})" for k, v in self.failed.items()
            )
        return out
