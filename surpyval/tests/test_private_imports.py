"""No new imports of another package's private names.

A name with a leading underscore is private to its module's package: it
may change without notice, so a module elsewhere that imports it breaks
when it moves (the maintainability plan, phase 1). ``ALLOWED`` lists the
imports of that kind the package still has, as ``(importer, source,
name)``; the first test fails on any import not in it, the second on an
entry the package no longer needs, so that the list only shrinks. When
you move a helper, give it a public name in a module of the package that
shares it (as ``nonparametric._support`` does), or import it from the
module that defines it and remove its entry here.

"Package" is the directory a module is in; the tests are not checked.
"""

from __future__ import annotations

import ast
from pathlib import Path

import surpyval

PACKAGE = Path(surpyval.__file__).resolve().parent

ALLOWED: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("surpyval", "surpyval.utils.deprecation", "_message"),
        (
            "surpyval.beta.ml.forest.log_rank_split",
            "surpyval.utils.data_formats",
            "_entered_before",
        ),
        (
            "surpyval.recurrent.competing_risks.nonparametric.cause_specific_mcf",  # noqa: E501
            "surpyval.recurrent.nonparametric.mcf",
            "_MCF_RANGE",
        ),
        (
            "surpyval.recurrent.competing_risks.nonparametric.cause_specific_mcf",  # noqa: E501
            "surpyval.recurrent.nonparametric.mcf",
            "_lawless_nadeau_var",
        ),
        (
            "surpyval.recurrent.competing_risks.nonparametric.cause_specific_mcf",  # noqa: E501
            "surpyval.recurrent.nonparametric.mcf",
            "_observation_origin",
        ),
        (
            "surpyval.recurrent.nonparametric.mcf",
            "surpyval.univariate.nonparametric.nonparametric",
            "_BOUNDS",
        ),
        (
            "surpyval.recurrent.nonparametric.mcf",
            "surpyval.univariate.nonparametric.nonparametric",
            "_check_option",
        ),
        (
            "surpyval.univariate.competing_risks.nonparametric.competing_risks",  # noqa: E501
            "surpyval.utils.data_formats",
            "_get_idx",
        ),
        (
            "surpyval.univariate.competing_risks.regression.competing_risks_proportional_hazard",  # noqa: E501
            "surpyval.univariate.nonparametric.nonparametric",
            "_check_option",
        ),
        (
            "surpyval.univariate.nonparametric.nonparametric_fitter",
            "surpyval.utils.data_formats",
            "_handled_xcnt_to_xrd",
        ),
        (
            "surpyval.univariate.parametric.distributions.custom_distribution",
            "surpyval.univariate.parametric._fit_inputs",
            "_offset_start",
        ),
        (
            "surpyval.univariate.parametric.distributions.expo_weibull",
            "surpyval.univariate.parametric._fit_inputs",
            "_offset_start",
        ),
        (
            "surpyval.univariate.parametric.distributions.gamma",
            "surpyval.univariate.parametric._fit_inputs",
            "_offset_start",
        ),
        (
            "surpyval.univariate.parametric.distributions.loglogistic",
            "surpyval.univariate.parametric._fit_inputs",
            "_offset_start",
        ),
        (
            "surpyval.univariate.parametric.distributions.lognormal",
            "surpyval.univariate.parametric._fit_inputs",
            "_offset_start",
        ),
        (
            "surpyval.univariate.parametric.distributions.rayleigh",
            "surpyval.univariate.parametric._fit_inputs",
            "_offset_start",
        ),
        (
            "surpyval.univariate.parametric.parametric_fitter",
            "surpyval.utils.validation",
            "_check_x_not_empty",
        ),
        (
            "surpyval.univariate.parametric.probability_plotting",
            "surpyval.utils.numeric",
            "_round_vals",
        ),
        (
            "surpyval.univariate.regression._aliasing",
            "surpyval.utils",
            "_caller_stacklevel",
        ),
        (
            "surpyval.univariate.regression._fit_skeleton",
            "surpyval.utils",
            "_caller_stacklevel",
        ),
        (
            "surpyval.univariate.regression.accelerated_life.parameter_substitution",  # noqa: E501
            "surpyval.utils",
            "_caller_stacklevel",
        ),
        (
            "surpyval.univariate.regression.additive_hazards.additive_hazards_fitter",  # noqa: E501
            "surpyval.univariate.regression._fit_skeleton",
            "_gradient",
        ),
        (
            "surpyval.univariate.regression.frailty.cox_frailty",
            "surpyval.univariate.regression.proportional_hazards.cox_ph",
            "_baseline_at_origin",
        ),
        (
            "surpyval.univariate.regression.frailty.cox_frailty",
            "surpyval.univariate.regression.proportional_hazards.cox_ph",
            "_newton_raphson",
        ),
        (
            "surpyval.univariate.regression.frailty.cox_frailty",
            "surpyval.utils",
            "_caller_stacklevel",
        ),
        (
            "surpyval.univariate.regression.frailty.frailty_fitter",
            "surpyval.univariate.regression._fit_skeleton",
            "_gradient",
        ),
        (
            "surpyval.univariate.regression.frailty.frailty_fitter",
            "surpyval.univariate.regression.proportional_hazards.cox_ph",
            "_strata_labels",
        ),
        (
            "surpyval.univariate.regression.frailty.frailty_fitter",
            "surpyval.utils",
            "_caller_stacklevel",
        ),
        (
            "surpyval.univariate.regression.proportional_hazards.cox_ph",
            "surpyval.utils",
            "_caller_stacklevel",
        ),
        (
            "surpyval.univariate.regression.proportional_odds.proportional_odds",  # noqa: E501
            "surpyval.utils",
            "_caller_stacklevel",
        ),
        (
            "surpyval.univariate.regression.regression_data",
            "surpyval.utils",
            "_caller_stacklevel",
        ),
    }
)


def _package_of(module: str) -> str:
    """The package ``module`` is in (a package is in itself)."""
    path = PACKAGE.parent.joinpath(*module.split("."))
    return module if path.is_dir() else module.rpartition(".")[0]


def _private_imports() -> set[tuple[str, str, str]]:
    found = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE.parent).with_suffix("")
        if rel.parts[1] == "tests":
            continue
        parts = list(rel.parts)
        is_init = parts[-1] == "__init__"
        if is_init:
            parts.pop()
        module = ".".join(parts)
        package = module if is_init else module.rpartition(".")[0]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                base = package.split(".")
                base = base[: len(base) - (node.level - 1)]
                source = ".".join(
                    base + ([node.module] if node.module else [])
                )
            else:
                source = node.module or ""
            if not source.startswith("surpyval"):
                continue
            if _package_of(source) == package:
                continue
            for alias in node.names:
                name = alias.name
                if name.startswith("_") and not name.startswith("__"):
                    found.add((module, source, name))
    return found


def test_no_new_private_imports_across_packages() -> None:
    new = sorted(_private_imports() - ALLOWED)
    assert not new, (
        "These modules import another package's private names; import a "
        "public name instead (see this module's docstring): {}".format(new)
    )


def test_the_allowlist_only_shrinks() -> None:
    stale = sorted(ALLOWED - _private_imports())
    assert (
        not stale
    ), "These imports are gone; remove them from ALLOWED: {}".format(stale)
