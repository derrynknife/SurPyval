"""The conformance registry's known failures (see ``registry.py``).

Case -> property -> what goes wrong, each reason led by its issue
number; each becomes a strict xfail of that case, so the suite
stays green and fails (XPASS) the day the failure is fixed -- then
delete the entry.
"""

# ---------------------------------------------------------------------------
# Known failures: case -> property -> what goes wrong. Each becomes a
# strict xfail, so the suite stays green and fails (XPASS) the day the
# failure is fixed -- then delete the entry. Numbers are on the case's
# fixture; see the report for #379 for minimal reproductions.
# ---------------------------------------------------------------------------
KNOWN_FAILURES: dict[str, dict[str, str]] = {}


# -- option sweeps (test_options.py) ------------------------------------
# Keyed "<property>[<bound name>]" (or "interp[<value>]",
# "estimators_agree[<option>]"); grouped as in the report for #379.
def _each(props, name, reason):
    return {f"{p}[{name}]": reason for p in props}


_NEGATIVE_VARIANCE = (
    "no Wald interval exists, so param_cb is nan (with a warning saying "
    "why, #411): a parameter at or near the edge of its support has a "
    "negative variance (covariance diagonal"
)
_OPTION_FAILURES: dict[str, dict[str, str]] = {
    # H. boundary estimates with a non-positive variance. (The renewal
    # fits' restoration parameter on its edge -- GeneralizedRenewal's q at
    # 2.7e-16, ARA's rho at 1 - 3e-16, ARI's at 1 -- now has a
    # profile-likelihood interval from the edge, and the other parameters
    # the Wald intervals of the model held there, #461.)
    "Beta4": _each(
        ("cb_contains",),
        "param_cb[wald]",
        _NEGATIVE_VARIANCE + " -8.4e-5 for alpha = 1.00008, whose fit "
        "runs to the edge)",
    ),
}
# The issue tracking each case's option failures (by key where a case
# has failures of more than one kind); it leads each reason.
_OPTION_ISSUES: dict[str, str | dict[str, str]] = {
    "Beta4": "#461",
}
for _name, _failures in _OPTION_FAILURES.items():
    _issue = _OPTION_ISSUES[_name]
    _failures = {
        key: "{}: {}".format(
            _issue if isinstance(_issue, str) else _issue[key], reason
        )
        for key, reason in _failures.items()
    }
    KNOWN_FAILURES[_name] = {**KNOWN_FAILURES.get(_name, {}), **_failures}

# -- convergence (test_convergence.py) --------------------------------------
# Each starved fit returns silently -- no warning, no error -- a model that
# is not the maximum (a lower log-likelihood, "ll", than the fixture's
# fit) or, where the data have no maximum, a finite answer as if there
# were one. Grouped by the fault; the group's issue leads each reason.
_CONVERGENCE_ISSUES = {
    # regression, Fine-Gray, copula, mixture and degradation fits of data
    # whose likelihood has no maximum (the #392 class, outside univariate)
    "no maximum": "#392",
}
# (The regression, frailty and Fine-Gray fits of a covariate level with no
# events now warn that the likelihood has no finite maximum, #392.)
_CONVERGENCE_FAILURES: dict[str, tuple[str, str]] = {}
for _name, (_group, _reason) in _CONVERGENCE_FAILURES.items():
    KNOWN_FAILURES[_name] = {
        **KNOWN_FAILURES.get(_name, {}),
        "convergence": f"{_CONVERGENCE_ISSUES[_group]}: {_reason}",
    }


# -- maximum (test_maximum.py) --------------------------------------------
# Likelihood fits that do not say what they reached: no ``maximum``, and no
# check that the answer is a verified maximum. Keyed "maximum" (the
# fixture's fit) and "maximum[starved]" (the convergence property's starved
# fit); the reason leads with the issue.
_MAXIMUM_FAILURES: dict[str, str] = {
    name: (
        "#TBD-degradation-maximum: the {} fit sets no ``maximum`` and does "
        "not check that its answer is a verified maximum".format(name)
    )
    for name in ("WienerProcess", "GammaProcess", "DestructiveDegradation")
}
# (The WienerProcess's starved fit, of noise-free data, is refused.)
for _name, _reason in _MAXIMUM_FAILURES.items():
    _starved = (
        {} if _name == "WienerProcess" else {"maximum[starved]": _reason}
    )
    KNOWN_FAILURES[_name] = {
        **KNOWN_FAILURES.get(_name, {}),
        "maximum": _reason,
        **_starved,
    }


# Known failures whose outcome depends on the numpy / scipy / BLAS build,
# so they are non-strict xfails: case name -> properties. The fits started
# far from the maximum were (#427, #428, #429); they now reach it, or say
# they did not, on every build.
NON_STRICT: dict[str, frozenset[str]] = {}


# The issue that tracks each kind of known failure; its number leads the
# xfail reason, so a test report says where the fix is being worked on.
KNOWN_FAILURE_ISSUES: dict[str, str] = {
    "missing_query": "#382",
    "qf_ff": "#383",
}
# The option-sweep and outside-data failures name their issue in the
# reason itself (see _OPTION_ISSUES), as one key can fail for different
# reasons in different cases.


# Option conventions (test_options.py, CONVENTIONS) the package breaks:
# convention -> the inconsistency. Each is a strict xfail; the test's
# message lists every method concerned.
# Options named differently somewhere, keyed by the convention they break,
# each reason starting "#NNN: " (principle 21). None since #422.
KNOWN_INCONSISTENCIES: dict[str, str] = {}


def _with_issue(prop: str, reason: str) -> str:
    issue = KNOWN_FAILURE_ISSUES.get(prop)
    return "{}: {}".format(issue, reason) if issue else reason
