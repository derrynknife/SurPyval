"""The validation kit of the Bayesian fits: one template for
simulation-based calibration, and one for a frozen reference from
another sampler. Every family's Bayesian fit is validated with these
two, so a result's numbers mean the same thing in each.

Simulation-based calibration (SBC; Talts et al. 2018)
-----------------------------------------------------
In words: if the true parameters are drawn from the prior and the data
from them, then over many repetitions the truth falls below the
posterior's ``p`` point in a share ``p`` of them, for every ``p``. So
the posterior probability below the truth (its "PIT" value) is uniform
on (0, 1) when the posterior is computed correctly, and anything else
shows an error in the likelihood, the prior or the engine.

``sbc`` runs the repetitions; ``check_sbc`` applies the pass rules,
which are fixed here, before any family's first run:

- each parameter's central 50% and 90% credible intervals hold the
  truth in 0.50 and 0.90 of repetitions, within three binomial standard
  errors;
- each parameter's PIT values pass a Kolmogorov-Smirnov test of
  uniformity at the 0.001 level;
- no repetition's fit refuses (unless the scenario allows a share);
- repetition ``r`` uses the seed ``seed + r``, the scenario's ``seed``
  written down with it.

An approximate engine (Laplace) is measured the same way; where it is
expected to fall short the scenario states the tolerance it is held to
instead, before its first run.

A frozen reference from PyMC
----------------------------
SurPyval does not depend on PyMC. A reference posterior is computed
once, in a separate environment, by the generator script kept beside its
JSON file under ``references/``, and frozen: for each parameter its
2.5%, 50% and 97.5% points with their Monte Carlo standard errors.
``check_reference`` applies the pass rule, fixed here: each point within
four Monte Carlo standard errors of the reference's (plus the engine's
own stated integration error, as a share of the posterior's 95% width).
For the Laplace engine the rule is a stated share of the 95% interval's
width instead.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

REFERENCES = Path(__file__).parent / "references"
#: Standard errors a coverage may be from its nominal level.
SE_MULTIPLE = 3.0
#: The level of the uniformity test of the PIT values.
KS_LEVEL = 0.001
#: Monte Carlo standard errors a posterior point may be from a
#: reference's.
MCSE_MULTIPLE = 4.0


def pit(model: Any, name: str, truth: float) -> float:
    """The posterior probability that parameter ``name`` is below
    ``truth``."""
    post = model.posterior
    exact = post._exact(name)
    if exact is not None:
        return float(exact.cdf(truth))
    col = post.points[:, post.parameter_names.index(name)]
    w = post.weights
    return float(w[col < truth].sum() + 0.5 * w[col == truth].sum())


def sbc(
    draw_truth: Any,
    simulate: Any,
    fit: Any,
    names: list[str],
    reps: int,
    seed: int,
) -> dict:
    """
    Run simulation-based calibration.

    Parameters
    ----------
    draw_truth : callable
        ``draw_truth(rng) -> {name: value}``, the parameters from the
        priors.
    simulate : callable
        ``simulate(truth, rng) -> data``.
    fit : callable
        ``fit(data) -> Bayesian fit`` (with ``posterior``).
    names : list of str
        The parameters checked.
    reps, seed : int
        Repetition ``r`` uses ``numpy.random.default_rng(seed + r)``.

    Returns
    -------
    dict
        ``pit`` (``{name: array}``), ``refused`` (``[(r, message)]``),
        ``coverage`` (``{name: {0.5: share, 0.9: share}}``) and
        ``reps``.
    """
    values: dict[str, list] = {k: [] for k in names}
    refused = []
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        truth = draw_truth(rng)
        data = simulate(truth, rng)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = fit(data)
        except ValueError as e:
            refused.append((r, str(e)[:160]))
            continue
        for k in names:
            values[k].append(pit(model, k, float(truth[k])))
    out = {k: np.asarray(v) for k, v in values.items()}
    coverage = {
        k: {
            level: float(np.mean(np.abs(u - 0.5) <= level / 2))
            for level in (0.5, 0.9)
        }
        for k, u in out.items()
    }
    return {
        "pit": out,
        "refused": refused,
        "coverage": coverage,
        "reps": reps,
    }


def describe_sbc(result: dict) -> str:
    """The coverage and uniformity numbers in one line a parameter."""
    lines = []
    for k, u in result["pit"].items():
        cov = result["coverage"][k]
        ks = stats.kstest(u, "uniform").pvalue if len(u) else float("nan")
        lines.append(
            f"{k}: 50% {cov[0.5]:.3f}, 90% {cov[0.9]:.3f}, "
            f"KS p {ks:.3f} ({len(u)} fits)"
        )
    if result["refused"]:
        lines.append(f"refused: {len(result['refused'])}")
    return "; ".join(lines)


def check_sbc(
    result: dict,
    *,
    tolerance: dict | None = None,
    allow_refused: float = 0.0,
    uniform: bool = True,
) -> None:
    """Apply the pass rules to ``sbc``'s result (see the module's
    description). ``tolerance`` replaces the three-standard-error rule
    with a stated band, ``{level: half_width}``, for an approximate
    engine; ``uniform=False`` leaves out the uniformity test for one."""
    reps = result["reps"]
    refused = result["refused"]
    assert (
        len(refused) <= allow_refused * reps
    ), f"{len(refused)} of {reps} fits refused: {refused[:3]}"
    problems = []
    for k, u in result["pit"].items():
        for level in (0.5, 0.9):
            got = result["coverage"][k][level]
            if tolerance is not None:
                band = tolerance[level]
            else:
                band = SE_MULTIPLE * np.sqrt(level * (1 - level) / len(u))
            if abs(got - level) > band:
                problems.append(
                    f"{k}: {level:.0%} coverage {got:.3f} "
                    f"(allowed {level} +- {band:.3f})"
                )
        if uniform:
            p = stats.kstest(u, "uniform").pvalue
            if p < KS_LEVEL:
                problems.append(f"{k}: PIT not uniform (KS p {p:.2g})")
    assert not problems, "; ".join(problems) + " | " + describe_sbc(result)


def load_reference(name: str) -> dict:
    """A frozen reference, ``references/<name>.json``."""
    with open(REFERENCES / f"{name}.json") as f:
        out: dict = json.load(f)
    return out


def check_reference(
    model: Any,
    reference: dict,
    *,
    width_share: float | None = None,
) -> list[str]:
    """
    Compare a fit's posterior with a frozen reference's.

    ``reference`` is ``{"parameters": {name: {"q": [2.5%, 50%, 97.5%],
    "mcse": [...]}}}``. The exact engines are held to four Monte Carlo
    standard errors (plus the grid's own integration error, as a share
    of the 95% width); ``width_share`` holds an approximate engine to
    that share of the reference's 95% interval's width instead.

    Returns the differences as text, for a report; asserts the rule.
    """
    lines, problems = [], []
    err = float(model.diagnostics.get("integration_error", 0.0) or 0.0)
    for name, ref in reference["parameters"].items():
        q = np.asarray(ref["q"], dtype=float)
        mcse = np.asarray(ref["mcse"], dtype=float)
        got = np.asarray(
            model.posterior.quantile(name, [0.025, 0.5, 0.975]), dtype=float
        )
        width = q[2] - q[0]
        if width_share is None:
            allowed = MCSE_MULTIPLE * mcse + max(err, 1e-4) * width
        else:
            allowed = np.full(3, width_share * width)
        diff = np.abs(got - q)
        lines.append(
            f"{name}: differences {diff[0]:.3g}, {diff[1]:.3g}, "
            f"{diff[2]:.3g} ({np.max(diff / width):.2%} of the 95% width; "
            f"allowed {np.max(allowed / width):.2%})"
        )
        if np.any(diff > allowed):
            problems.append(lines[-1])
    assert not problems, "; ".join(problems)
    return lines
