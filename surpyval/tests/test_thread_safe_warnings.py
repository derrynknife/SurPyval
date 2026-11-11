"""A fit in one thread never turns another thread's warnings into errors.

Python's warning filters are one list for the whole process. A fit that
ran ``warnings.filterwarnings("error")`` to notice its own warnings made
every warning raised meanwhile in any other thread an exception there: a
web server running an accelerated life fit saw unrelated requests fail.
The fits check their own results instead; these tests run a fit in one
thread while another warns, and the other thread must never raise.
"""

from __future__ import annotations

import threading
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from surpyval import AcceleratedLife, Weibull, WeibullPH
from surpyval.life_models import Power
from surpyval.utils.covariates import formula_model_matrix


def _warn_while(
    work: Callable[[], None], category: type[Warning], runs: int
) -> list[BaseException]:
    """Run ``work`` ``runs`` times in one thread while another thread
    warns (``category``) without pause; the exceptions the warning
    thread met."""
    done = threading.Event()
    errors: list[BaseException] = []
    work_errors: list[BaseException] = []

    def warner() -> None:
        while not done.is_set():
            try:
                warnings.warn("x", category)
            except BaseException as error:  # noqa: B036
                errors.append(error)
                return

    def worker() -> None:
        try:
            for _ in range(runs):
                work()
        except BaseException as error:  # noqa: B036
            work_errors.append(error)
        finally:
            done.set()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category)
        threads = [
            threading.Thread(target=warner),
            threading.Thread(target=worker),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=300)
    assert not work_errors, work_errors
    return errors


def test_accelerated_life_fit_leaves_other_threads_warnings_alone():
    rng = np.random.default_rng(20261107)
    Z = np.repeat([1.0, 2.0, 3.0, 4.0], 15)
    x = 100.0 * Z**-1.5 * rng.weibull(2.0, Z.size)

    def fit() -> None:
        model = AcceleratedLife(Weibull, Power).fit(x=x, Z=Z)
        assert np.all(np.isfinite(model.params))

    assert _warn_while(fit, DeprecationWarning, runs=5) == []


def test_formula_prediction_leaves_other_threads_warnings_alone():
    from formulaic.errors import (  # type: ignore[import-untyped]
        DataMismatchWarning,
    )

    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "t": rng.weibull(1.5, 60) * 10.0,
            "c": np.zeros(60, dtype=int),
            "z": rng.normal(size=60),
            "g": np.tile(["a", "b", "c"], 20),
        }
    )
    model = WeibullPH.fit_from_df(df, x_col="t", c_col="c", formula="z + g")
    new = pd.DataFrame({"z": [0.5, 1.0], "g": ["a", "c"]})

    def predict() -> None:
        formula_model_matrix(model._model_spec, new)

    assert _warn_while(predict, DataMismatchWarning, runs=2000) == []


# -- ignoring_warnings: an "ignore" only its own thread sees -----------------


class _Probe(UserWarning):
    """A warning only these tests raise."""


def test_ignoring_warnings_ignores_in_its_own_thread_only():
    """While one thread is inside the block, its own warnings are ignored
    and another thread's are given (here: raised, by an "error" filter)."""
    from surpyval.utils.warnings import ignoring_warnings

    inside = threading.Event()
    checked = threading.Event()
    seen: list[str] = []

    def other() -> None:
        inside.wait(timeout=60)
        try:
            warnings.warn("theirs", _Probe)
            seen.append("hidden")
        except _Probe:
            seen.append("given")
        checked.set()

    with warnings.catch_warnings():
        warnings.simplefilter("error", _Probe)
        thread = threading.Thread(target=other)
        thread.start()
        with ignoring_warnings():
            warnings.warn("mine", _Probe)  # ignored: does not raise
            inside.set()
            checked.wait(timeout=60)
            warnings.warn("mine again", _Probe)
        thread.join(timeout=60)
    assert seen == ["given"]


def test_ignoring_warnings_out_of_order_leaves_nothing_ignored():
    """Blocks left out of order by two threads leave nothing ignored.
    (``catch_warnings`` restores the list it saved: the second to leave
    put the first one's "ignore" back, for good.) Nor does the filter
    list grow with use: a block switches a filter on and off, and adds
    one only when none is at the front."""
    from surpyval.utils.warnings import ignoring_warnings

    first, second = ignoring_warnings(), ignoring_warnings()
    entered = threading.Event()
    leave = threading.Event()

    def other() -> None:
        with second:
            entered.set()
            leave.wait(timeout=60)

    first.__enter__()
    size = len(warnings.filters)
    thread = threading.Thread(target=other)
    thread.start()
    entered.wait(timeout=60)
    first.__exit__(None, None, None)  # the first in leaves first
    leave.set()
    thread.join(timeout=60)
    for _ in range(50):
        with ignoring_warnings():
            with ignoring_warnings(message="x"):
                pass
    assert len(warnings.filters) == size
    with warnings.catch_warnings():
        warnings.simplefilter("error", _Probe)
        with pytest.raises(_Probe):
            warnings.warn("after", _Probe)


def test_ignoring_warnings_takes_any_message_object():
    """A warning whose message is not a string (``warnings.warn(123)``)
    is judged as any other, in a thread that is ignoring and in one that
    is not: the filter's "never" and "always" take any object."""
    from surpyval.utils.warnings import ignoring_warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with ignoring_warnings():
            warnings.warn(123, _Probe)
        warnings.warn(456, _Probe)
    assert [w.message.args for w in caught] == [(456,)]


def test_ignoring_warnings_nests_as_the_standard_form_does():
    """The newest filter wins, as with ``catch_warnings``: a recording
    block opened inside sees the warnings, and a block inside a recording
    one hides them; a category and a message narrow what is ignored."""
    from surpyval.utils.warnings import ignoring_warnings

    with ignoring_warnings():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("recorded", _Probe)
            with ignoring_warnings():
                warnings.warn("hidden", _Probe)
            with ignoring_warnings(RuntimeWarning):
                warnings.warn("not a RuntimeWarning", _Probe)
                warnings.warn("hidden too", RuntimeWarning)
            with ignoring_warnings(message="Output seems"):
                warnings.warn("output seems independent of input", _Probe)
                warnings.warn("another message", _Probe)
            with ignoring_warnings(_Probe, "The weights"):
                warnings.warn("The weights leave little", _Probe)
                warnings.warn("The weights leave little", RuntimeWarning)
    assert [str(w.message) for w in caught] == [
        "recorded",
        "not a RuntimeWarning",
        "another message",
        "The weights leave little",
    ]
    assert caught[-1].category is RuntimeWarning


def test_ignoring_warnings_ends_with_an_exception_too():
    from surpyval.utils.warnings import ignoring_warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", _Probe)
        try:
            with ignoring_warnings():
                warnings.warn("ignored", _Probe)
                raise ValueError("stop")
        except ValueError:
            pass
        with pytest.raises(_Probe):
            warnings.warn("after", _Probe)


def test_a_quiet_fit_leaves_other_threads_warnings_alone():
    """A fit that holds its own inner warnings back (the Bayesian grid's
    maximum-likelihood start) does not hide another thread's warnings
    while it runs: every one the other thread raises is given."""
    import surpyval as sp

    x = np.array([120.0, 250.0, 310.0, 480.0, 700.0, 910.0])

    def fit() -> None:
        model = sp.Weibull.fit_bayes(
            x,
            priors={
                "alpha": sp.LogNormal.from_params([6.0, 1.0]),
                "beta": sp.LogNormal.from_params([0.5, 0.5]),
            },
        )
        assert np.all(np.isfinite(model.params))

    done = threading.Event()
    raised = 0
    hidden = 0

    def warner() -> None:
        nonlocal raised, hidden
        while not done.is_set():
            try:
                warnings.warn("theirs", _Probe)
                hidden += 1
            except _Probe:
                raised += 1

    with warnings.catch_warnings():
        warnings.simplefilter("error", _Probe)
        thread = threading.Thread(target=warner)
        thread.start()
        try:
            for _ in range(3):
                fit()
        finally:
            done.set()
            thread.join(timeout=300)
    assert raised > 0
    assert hidden == 0


#: The places that still use ``warnings.catch_warnings`` and why: each
#: collects the warnings of an inner step to read or repeat them, which
#: Python (before 3.14's context-aware warnings) can do only through the
#: shared filters.
_STILL_SHARED = {
    "surpyval/diagnostics/recurrent.py": 1,
    "surpyval/fit_best.py": 1,
    "surpyval/recurrent/renewal/fit_mixin.py": 1,
    "surpyval/univariate/parametric/_residual_life.py": 1,
    "surpyval/univariate/regression/frailty/cox_frailty.py": 1,
    # formulaic's unknown-level warning, for a formula naming its levels
    "surpyval/utils/covariates.py": 1,
    # ignoring_warnings itself, on Python 3.14 with context-aware warnings
    "surpyval/utils/warnings.py": 1,
}


def test_the_package_ignores_warnings_through_ignoring_warnings():
    """No module silences warnings through the shared filter list: a new
    ``with warnings.catch_warnings(): warnings.simplefilter("ignore")``
    is written ``with ignoring_warnings():`` (surpyval.utils.warnings)."""
    import ast
    from pathlib import Path

    import surpyval

    root = Path(surpyval.__file__).parent
    found: dict[str, int] = {}
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root.parent).as_posix()
        if relative.startswith("surpyval/tests/"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        count = sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "catch_warnings"
            for node in ast.walk(tree)
        )
        if count:
            found[relative] = count
    assert found == _STILL_SHARED
