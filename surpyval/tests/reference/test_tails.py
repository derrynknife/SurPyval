"""Accuracy of the distribution functions in the tails and at extreme
parameters (#398), against 50-digit mpmath values.

``data/tails_mpmath.json`` (written by
``scripts/reference/tails_mpmath.py``) holds, for every closed-form
univariate distribution, the true ``sf``, ``ff``, ``df``, ``hf``, ``Hf``,
``log_sf``, ``log_ff`` and ``log_df`` on a grid of extreme parameters
(shapes 1e-3 to 1e3, scales 1e-6 to 1e6, locations +-1e6) and times (the
support edges, and the times where the CDF or the survival function is
1e-300 ... 0.5 and 1e-400), and the true ``qf`` at probabilities from
1e-300 to the double just below 1. The generator's docstring lists the
grid and the distributions it leaves out, and why.

Each value is checked against these rules:

* never NaN, never an exception, never of the wrong sign (``sf``, ``ff``,
  ``df``, ``hf`` and ``Hf`` are not negative, ``log_sf`` and ``log_ff``
  not positive);
* where the true value is infinite or exactly 0 (an edge of the support),
  or beyond the largest double, the same infinity or 0;
* where it is a normal double, a relative error of at most ``RTOL`` =
  1e-8 -- or, where the function is ill-conditioned, an absolute error of
  at most 64 ulps of its inputs' sensitivity: the generator stores
  ``s = sum_i |d f / d log(theta_i)|`` over the time and the parameters,
  and ``eps * s`` is what the rounding of the arguments alone costs any
  double-precision formula (``x / alpha`` is rounded before it is raised
  to the power ``beta``). A Normal at ``mu = 1e6``, ``sigma = 1e-6`` has
  ``s / f`` up to 1e14 in the right tail, so there only a few digits are
  meaningful, and only those are checked;
* where it is below the normal range (smaller than 2.2e-308, down to
  values that underflow to 0), 0 or a subnormal of the same sign;
* ``qf`` of a discrete distribution exactly (the generator leaves out a
  probability within 1e-9 of a step of the CDF).

The points are grouped by (distribution, function, regime), so one bug
is one test: a group fails if any of its points does, and its message
names the worst. A group known to fail is a strict xfail naming its issue
in ``KNOWN_FAILURES``.
"""

import math
import warnings
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pytest

import surpyval.univariate.parametric.distributions as distributions

from ._data import _load

SOURCE = "tails_mpmath"
RTOL = 1e-8
ULPS = 64
EPS = np.finfo(float).eps
TINY = np.finfo(float).tiny

NOT_NEGATIVE = {"sf", "ff", "df", "hf", "Hf"}
NOT_POSITIVE = {"log_sf", "log_ff"}

# (distribution, function, regime) groups known to fail, by issue.
_BY_ISSUE: dict[str, tuple[str, list[tuple[str, str, str]]]] = {}
KNOWN_FAILURES: dict[tuple[str, str, str], str] = {
    group: f"{num}: {why}"
    for num, (why, groups) in _BY_ISSUE.items()
    for group in groups
}

# Groups whose outcome depends on the numpy / scipy build.
NON_STRICT: set[tuple[str, str, str]] = set()


def _content():
    return _load(SOURCE)


def _functions():
    return _content()["settings"]["functions"]


def _groups():
    """(distribution, function, regime) -> [(case, point)], where point
    indexes ``x`` (or ``qf.u`` for the function ``qf``)."""
    groups = defaultdict(list)
    for c, case in enumerate(_content()["cases"]):
        for i, regime in enumerate(case["regime"]):
            for fn in _functions():
                groups[(case["dist"], fn, regime)].append((c, i))
        for i, regime in enumerate(case["qf"]["regime"]):
            groups[(case["dist"], "qf", regime)].append((c, i))
    return groups


GROUPS = _groups()


@lru_cache(maxsize=None)
def _package(c, fn):
    """SurPyval's ``fn`` at every point of case ``c``: floats, or the name
    of the exception a point raised."""
    case = _content()["cases"][c]
    dist = getattr(distributions, case["dist"])
    params = case["params"]
    points = case["qf"]["u"] if fn == "qf" else case["x"]
    x = np.asarray(points, dtype=float)
    method = getattr(dist, fn)

    def call(arg):
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            # Accuracy is checked here; the warnings are the business of
            # conformance/test_warnings.py (principle 22).
            warnings.simplefilter("ignore")
            out = np.asarray(method(arg, *params))
        if np.iscomplexobj(out):
            out = np.where(out.imag == 0, out.real, np.nan)
        return np.broadcast_to(out.astype(float), np.shape(arg))

    try:
        return list(call(x))
    except Exception:
        out = []
        for v in x:
            try:
                out.append(float(call(np.array([v]))[0]))
            except Exception as e:  # noqa: BLE001 - reported per point
                out.append("raised {}".format(type(e).__name__))
        return out


def _is_exact_zero(text):
    return float(text) == 0 and not any(d in text for d in "123456789")


def _problem(fn, got, true_text, sens_text, discrete_qf):
    """Why ``got`` is not an acceptable value for ``true_text``, or
    None."""
    if isinstance(got, str):
        return got
    if math.isnan(got):
        return "nan"
    if fn in NOT_NEGATIVE and got < 0:
        return "negative"
    if fn in NOT_POSITIVE and got > 0:
        return "positive"
    true = float(true_text)
    if discrete_qf or math.isinf(true) or _is_exact_zero(true_text):
        return None if got == true else "wrong"
    if abs(true) < TINY:
        same_side = got == 0 or (got > 0) == (true > 0)
        return None if abs(got) < TINY and same_side else "wrong"
    tol = max(RTOL * abs(true), ULPS * EPS * float(sens_text))
    return None if abs(got - true) <= tol else "inaccurate"


def _error(got, true):
    if isinstance(got, str) or math.isnan(got):
        return math.inf
    if true == got:
        return 0.0
    if true == 0 or math.isinf(true) or math.isinf(got):
        return math.inf
    return abs(got - true) / abs(true)


def check_group(key):
    """Every failing point of a group, worst first:
    (relative error, message)."""
    dist, fn, regime = key
    failures = []
    for c, i in GROUPS[key]:
        case = _content()["cases"][c]
        got = _package(c, fn)[i]
        if fn == "qf":
            true_text = case["qf"]["value"][i]
            sens_text = case["qf"]["sens"][i]
            where = "u={!r}".format(case["qf"]["u"][i])
        else:
            j = _functions().index(fn)
            true_text = case["values"][i][j]
            sens_text = case["sens"][i][j]
            where = "x={!r}".format(case["x"][i])
        discrete_qf = fn == "qf" and getattr(distributions, dist).discrete
        problem = _problem(fn, got, true_text, sens_text, discrete_qf)
        if problem is None:
            continue
        true = float(true_text)
        err = _error(got, true)
        failures.append(
            (
                err,
                "{} at params={} {}: got {!r}, true {} (rel err {:.2g}, "
                "sensitivity {})".format(
                    problem,
                    tuple(case["params"]),
                    where,
                    got,
                    true_text,
                    err,
                    sens_text,
                ),
            )
        )
    failures.sort(key=lambda f: -f[0])
    return failures


def _marks(key):
    if key in KNOWN_FAILURES:
        return [
            pytest.mark.xfail(
                strict=key not in NON_STRICT, reason=KNOWN_FAILURES[key]
            )
        ]
    return []


@pytest.mark.parametrize(
    "key",
    [
        pytest.param(key, id="{}.{}-{}".format(*key), marks=_marks(key))
        for key in sorted(GROUPS)
    ],
)
def test_accuracy_in_the_tails(key):
    failures = check_group(key)
    assert not failures, "{}.{} in {}: {} of {} points wrong; {}".format(
        *key, len(failures), len(GROUPS[key]), failures[0][1]
    )


def test_every_known_failure_is_a_group():
    assert set(KNOWN_FAILURES) <= set(GROUPS)
    assert NON_STRICT <= set(KNOWN_FAILURES)
    for reason in KNOWN_FAILURES.values():
        assert reason.startswith("#"), reason


def test_every_distribution_is_covered_or_skipped_with_a_reason():
    content = _content()
    covered = {case["dist"] for case in content["cases"]}
    public = {
        name
        for name in dir(distributions)
        if not name.startswith("_") and name[0].isupper()
    }
    assert public <= covered | set(content["skipped"])
    assert all(content["skipped"].values())
    # nothing is both checked and skipped
    assert not covered & set(content["skipped"])


def test_reference_file_records_its_provenance():
    content = _content()
    assert content["generator"] == "scripts/reference/tails_mpmath.py"
    assert content["software"] == "mpmath" and content["version"]
    assert content["settings"]["dps"] >= 50
    assert content["settings"]["rechecked_points"] > 0
    for case in content["cases"]:
        n = len(case["x"])
        assert len(case["regime"]) == len(case["values"]) == n
        assert len(case["sens"]) == n
        assert len(case["qf"]["u"]) == len(case["qf"]["value"])
