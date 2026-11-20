"""Comparing what the hazard follows: one joint model fitted with each
link to the same data, ranked by an information criterion, with
likelihood-ratio tests of the nested pairs (:class:`LinkComparison`, made
by :meth:`JointPHFitter.compare_links
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

#: How far below the simpler link's log-likelihood (relative, with an
#: absolute floor of the same) the richer one may be and still be taken
#: as at its maximum: the adaptive quadrature's own error.
_LL_TOL = 1e-6


def link_label(link: Any) -> str:
    """The name of a link in the comparison: ``"value"``, ``"value +
    slope"``, ... (the links in their fixed order)."""
    return " + ".join(link)


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
    fitters = [
        JointPH(
            fitter.dist, fitter.path_model, fitter.varying, link, fitter.lag
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
    models: dict[str, JointModel] = {}
    failed: dict[str, str] = {}
    errors: list[Exception] = []
    said: dict[tuple, Any] = {}
    for candidate in fitters:
        label = link_label(candidate.link)
        own = dict(kwargs)
        if reference is not None and "value" not in candidate.link:
            # The value's reference moves alpha only, and only where the
            # hazard follows the value: the other links are fitted
            # without it.
            del own["marker_reference"]
        with collecting_warnings() as caught, quiet_maximum_warnings():
            try:
                model = candidate.fit(*args, **own)
            except ValueError as error:
                failed[label] = _reason(error)
                errors.append(error)
                continue
        models[label] = model
        for kept in caught:
            # A warning about the data comes from every fit: said once
            said.setdefault((str(kept.message), kept.category), kept)
    if not models:
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
    follows of the marker's path: its current value, its slope, the area
    under it, or more than one -- to the same data, compared. Made by
    :meth:`JointPHFitter.compare_links
    <surpyval.univariate.joint.joint_ph.JointPHFitter.compare_links>`,
    as in ``WeibullJointPH.compare_links(data)``.

    Every candidate has the fitter's baseline, path model and varying
    path parameters and is fitted to the same items and readings, so
    their log-likelihoods (of the readings and the lifetimes together)
    and criteria compare directly. A fit that is not a verified maximum
    is shown but neither ranked nor tested.

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
        The ranked fit with the lowest ``metric`` (``None`` where no fit
        is a verified maximum).

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
        ranked = self._ranked()
        self.best: JointModel | None = (
            min(ranked.values(), key=self._measure) if ranked else None
        )

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

    @property
    def best_link(self) -> tuple | None:
        """The link of the best fit, as its ``link`` (``("value",
        "slope")``, ...); ``None`` where no fit is ranked."""
        return None if self.best is None else tuple(self.best.link)

    @property
    def best_label(self) -> str | None:
        """The name of the best fit's link (``"value + slope"``, ...)."""
        return None if self.best is None else link_label(self.best.link)

    @property
    def table(self) -> pd.DataFrame:
        """
        One row per link, best first, indexed by the link's name: the
        number of parameters ``k``, the ``log_likelihood``, ``aic``,
        ``aic_c`` and ``bic``; ``delta``, the link's ``metric`` less the
        best's; ``weight``, each ranked link's share of
        ``exp(-delta / 2)`` (the Akaike weights, ranked by AIC); the
        fit's ``maximum``; and ``status``: ``"best"``, ``"ranked"``,
        ``"set aside"`` (with the ``reason``) or ``"failed"`` (could not
        be fitted, with the ``reason``).
        """
        import pandas as pd

        ranked = self._ranked()
        best_value = np.nan if self.best is None else self._measure(self.best)
        records: list[dict[str, Any]] = []
        for label, model in self.models.items():
            if model is self.best:
                status, reason = "best", ""
            elif label in ranked:
                status, reason = "ranked", ""
            else:
                status = "set aside"
                reason = (
                    "the fit is not a verified maximum"
                    if model.maximum != "verified"
                    else f"its {self.metric} is not finite"
                )
            records.append(
                {
                    "link": label,
                    "k": int(model.k),
                    "log_likelihood": float(model.log_likelihood),
                    "aic": float(model.aic()),
                    "aic_c": float(model.aic_c()),
                    "bic": float(model.bic()),
                    "delta": self._measure(model) - best_value,
                    "maximum": model.maximum,
                    "status": status,
                    "reason": reason,
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
                    "maximum": "",
                    "status": "failed",
                    "reason": why,
                }
            )
        order = {"best": 0, "ranked": 0, "set aside": 1, "failed": 2}
        records.sort(
            key=lambda r: (
                order[r["status"]],
                np.inf if np.isnan(r[self.metric]) else r[self.metric],
            )
        )
        table = pd.DataFrame(records).set_index("link")
        weight = np.full(len(table), np.nan)
        ok = table["status"].isin(["best", "ranked"]).to_numpy()
        if ok.any():
            relative = np.exp(-0.5 * table["delta"].to_numpy()[ok])
            weight[ok] = relative / relative.sum()
        table.insert(table.columns.get_loc("delta") + 1, "weight", weight)
        return table

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

    def __repr__(self) -> str:
        shown = self.table[
            ["k", "log_likelihood", self.metric, "delta", "weight", "status"]
        ]
        out = (
            "Joint model link comparison"
            "\n==========================="
            f"\nRanked by           : {self.metric} (lower is better)"
            f"\nBest                : {self.best_label or 'none'}"
            "\n" + shown.to_string(float_format=lambda v: f"{v:.6g}")
        )
        if self.failed:
            out += "\nNot fitted          : " + "; ".join(
                f"{k} ({v})" for k, v in self.failed.items()
            )
        return out
