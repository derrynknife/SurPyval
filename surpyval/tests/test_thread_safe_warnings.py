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


#: The places that use ``warnings.catch_warnings``: only the helpers
#: themselves, on Python 3.14 run with context-aware warnings, where the
#: standard form is already local to the thread.
_STILL_SHARED = {
    "surpyval/utils/warnings.py": 1,
}


def test_the_package_ignores_warnings_through_ignoring_warnings():
    """No module changes the shared filter list to ignore, collect or
    raise warnings: ``catch_warnings()`` with ``simplefilter("ignore")``
    is written ``with ignoring_warnings():``, ``catch_warnings(record=
    True)`` with ``simplefilter("always")`` is ``with
    collecting_warnings() as caught:``, and ``simplefilter("error")`` is
    ``with raising_warnings():`` (surpyval.utils.warnings)."""
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


# -- collecting_warnings and raising_warnings: one thread only ---------------


class _Other(UserWarning):
    """Another warning only these tests raise."""


def test_collecting_warnings_takes_its_own_threads_warnings_only():
    """While one thread collects, another thread's warnings are neither
    collected nor lost: they meet the filters as they are (here an
    "error" filter raises them) and the collector holds only its own."""
    from surpyval.utils.warnings import collecting_warnings

    inside = threading.Event()
    checked = threading.Event()
    seen: list[str] = []

    def other() -> None:
        inside.wait(timeout=60)
        try:
            warnings.warn("theirs", _Probe)
            seen.append("lost")
        except _Probe:
            seen.append("given")
        checked.set()

    with warnings.catch_warnings():
        warnings.simplefilter("error", _Probe)
        thread = threading.Thread(target=other)
        thread.start()
        with collecting_warnings() as caught:
            warnings.warn("mine", _Probe)  # collected: does not raise
            inside.set()
            checked.wait(timeout=60)
            warnings.warn("mine again", _Probe)
        thread.join(timeout=60)
        with pytest.raises(_Probe):
            warnings.warn("after", _Probe)
    assert seen == ["given"]
    assert [str(w.message) for w in caught] == ["mine", "mine again"]


def test_collecting_warnings_leaves_a_recording_thread_its_warnings():
    """A thread recording with the standard ``catch_warnings(record=True)``
    (pytest's way) still records its own warnings while another thread
    collects, and the collector takes none of them."""
    from surpyval.utils.warnings import collecting_warnings

    inside = threading.Event()
    done = threading.Event()
    mine: list[str] = []

    def collector() -> None:
        with collecting_warnings() as caught:
            inside.set()
            done.wait(timeout=60)
            warnings.warn("collected", _Other)
        mine.extend(str(w.message) for w in caught)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        thread = threading.Thread(target=collector)
        thread.start()
        inside.wait(timeout=60)
        for i in range(3):
            warnings.warn(f"recorded {i}", _Probe)
        done.set()
        thread.join(timeout=60)
    assert [str(w.message) for w in recorded] == [
        "recorded 0",
        "recorded 1",
        "recorded 2",
    ]
    assert mine == ["collected"]


def test_raising_warnings_raises_in_its_own_thread_only():
    """While one thread raises a category, another thread's warnings of
    it are not turned into exceptions."""
    from surpyval.utils.warnings import raising_warnings

    inside = threading.Event()
    checked = threading.Event()
    seen: list[str] = []

    def other() -> None:
        inside.wait(timeout=60)
        try:
            warnings.warn("theirs", _Probe)
            seen.append("warned")
        except _Probe:
            seen.append("raised")
        checked.set()

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        thread = threading.Thread(target=other)
        thread.start()
        with pytest.raises(_Probe):
            with raising_warnings(_Probe):
                inside.set()
                checked.wait(timeout=60)
                warnings.warn("mine", _Probe)
        thread.join(timeout=60)
    assert seen == ["warned"]
    assert [str(w.message) for w in recorded] == ["theirs"]


def test_collecting_out_of_order_leaves_nothing_collected():
    """Two threads leaving collecting blocks out of order leave no
    thread collecting and nothing shown "always", and the filter list
    does not grow with use."""
    from surpyval.utils.warnings import collecting_warnings

    first, second = collecting_warnings(), collecting_warnings()
    entered = threading.Event()
    leave = threading.Event()

    def other() -> None:
        with second:
            entered.set()
            leave.wait(timeout=60)

    def use() -> None:
        with collecting_warnings():
            with collecting_warnings(RuntimeWarning):
                pass

    use()  # (adds the filters these blocks need, once)
    size = len(warnings.filters)
    first.__enter__()
    thread = threading.Thread(target=other)
    thread.start()
    entered.wait(timeout=60)
    first.__exit__(None, None, None)  # the first in leaves first
    leave.set()
    thread.join(timeout=60)
    for _ in range(50):
        use()
    assert len(warnings.filters) == size
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("error", _Probe)
        with pytest.raises(_Probe):
            warnings.warn("after", _Probe)
        warnings.simplefilter("ignore", _Other)
        warnings.warn("ignored", _Other)
    assert recorded == []


def _as_pairs(caught: list) -> list[tuple]:
    return [(w.category, str(w.message)) for w in caught]


def _nested_body(inner_collector: Callable[[], object]) -> Callable[[], None]:
    from surpyval.utils.warnings import ignoring_warnings

    def body() -> None:
        for _ in range(3):
            warnings.warn("repeated", _Probe)
        with ignoring_warnings():
            warnings.warn("held back", _Probe)
            with inner_collector() as inner:  # type: ignore[attr-defined]
                warnings.warn("inner", _Probe)
                warnings.warn("held back too", _Other)
            assert [str(w.message) for w in inner] == ["inner"]
        with ignoring_warnings(_Other):
            warnings.warn("not an _Other", _Probe)
            warnings.warn("an _Other", _Other)
        warnings.warn("last", _Other)

    return body


class _StandardCollector:
    """``catch_warnings(record=True)`` with ``simplefilter("always",
    category)``, the form the library used before."""

    def __init__(self, category: type[Warning] = _Probe) -> None:
        self._category = category

    def __enter__(self) -> list:
        self._block = warnings.catch_warnings(record=True)
        log = self._block.__enter__()
        warnings.simplefilter("always", self._category)
        return log

    def __exit__(self, *exc: object) -> None:
        self._block.__exit__(*exc)  # type: ignore[arg-type]


def test_collecting_warnings_nests_as_the_standard_form_does():
    """In one thread the block collects what ``catch_warnings(record=
    True)`` with ``simplefilter("always")`` did: every warning each time
    it is raised, none an inner ignoring block holds back, none an inner
    collector takes, and the inner collector's category inside an
    ignoring block."""
    from surpyval.utils.warnings import collecting_warnings

    expected = [
        (_Probe, "repeated"),
        (_Probe, "repeated"),
        (_Probe, "repeated"),
        (_Probe, "not an _Other"),
        (_Other, "last"),
    ]
    with warnings.catch_warnings(record=True) as standard:
        warnings.simplefilter("always")
        _nested_body(_StandardCollector)()
    assert _as_pairs(standard) == expected
    with collecting_warnings() as caught:
        _nested_body(lambda: collecting_warnings(_Probe))()
    assert _as_pairs(caught) == expected


def test_collecting_one_category_collects_what_the_filters_let_through():
    """``collecting_warnings(category)`` is ``catch_warnings(record=True)``
    with ``simplefilter("always", category)``: that category each time,
    other warnings as the filters say, all of them collected."""
    from surpyval.utils.warnings import collecting_warnings

    def body() -> None:
        for _ in range(2):
            warnings.warn("probe", _Probe)
            warnings.warn("other", _Other)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", _Other)
        with _StandardCollector(_Probe) as standard:
            body()
        with collecting_warnings(_Probe) as mine:
            body()
        with _StandardCollector(_Other) as standard_others:
            body()
        with collecting_warnings(_Other) as others:
            body()
    # (an _Other is ignored unless collected; a _Probe not collected
    # "always" is given once, by the "default" action)
    assert _as_pairs(standard) == [(_Probe, "probe")] * 2
    assert _as_pairs(mine) == _as_pairs(standard)
    assert _as_pairs(standard_others) == [
        (_Probe, "probe"),
        (_Other, "other"),
        (_Other, "other"),
    ]
    assert _as_pairs(others) == _as_pairs(standard_others)


def test_collecting_warnings_ends_with_an_exception_too():
    from surpyval.utils.warnings import collecting_warnings

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("default")
        try:
            with collecting_warnings() as caught:
                warnings.warn("collected", _Probe)
                raise ValueError("stop")
        except ValueError:
            pass
        for _ in range(2):
            warnings.warn("given", _Probe)  # "default": once
    assert [str(w.message) for w in caught] == ["collected"]
    assert [str(w.message) for w in recorded] == ["given"]


# -- the collecting sites, with another thread warning meanwhile ------------


def _fit_best() -> None:
    import surpyval as sp

    x = np.array([3.1, 4.7, 5.2, 8.8, 10.4, 12.9, 15.3, 21.0, 26.4])
    sp.fit_best(x, metric="aic")


def _cox_frailty() -> None:
    import surpyval as sp

    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(10), 4)
    Z = rng.normal(size=(40, 1))
    frailty = rng.gamma(2.0, 0.5, 10)[groups]
    x = rng.exponential(1.0 / (frailty * np.exp(0.5 * Z[:, 0])))
    sp.CoxFrailty.fit(x, Z=Z, groups=groups)


def _residual_life() -> None:
    from surpyval import LogNormal

    LogNormal.mrl(np.array([0.5, 2.0, 30.0]), 0.0, 1.0)


def _renewal() -> None:
    # (a life on a run-off: the fit reads the life's own refit's warning)
    from surpyval import ExpoWeibull, GeneralizedRenewal

    rng = np.random.default_rng(5)
    x = np.cumsum(rng.weibull(2.0, 30) * 3)
    model = GeneralizedRenewal.fit(x, dist=ExpoWeibull, kijima="ii")
    assert model.maximum == "no finite maximum"


def _diagnose_cvm() -> None:
    from surpyval import HPP

    x = np.cumsum(np.random.default_rng(2).exponential(10.0, 15))
    HPP.fit(x).diagnose(n_boot=19, random_state=0)


def _formula() -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "t": rng.weibull(1.5, 30) * 10.0,
            "c": np.zeros(30, dtype=int),
            "g": np.tile(["a", "b", "c"], 10),
        }
    )
    model = WeibullPH.fit_from_df(
        df, x_col="t", c_col="c", formula="C(g, levels=['a', 'b', 'c'])"
    )
    with pytest.raises(ValueError):
        formula_model_matrix(model._model_spec, pd.DataFrame({"g": ["d"]}))


@pytest.mark.parametrize(
    "site",
    [
        _fit_best,
        _cox_frailty,
        _residual_life,
        _renewal,
        _diagnose_cvm,
        _formula,
    ],
    ids=[
        "fit_best",
        "cox_frailty",
        "residual_life",
        "renewal",
        "diagnose_cvm",
        "formula",
    ],
)
def test_a_collecting_site_leaves_other_threads_warnings_alone(site):
    """Each place that collects (or raises) an inner step's warnings,
    run in one thread while another warns without pause: every warning
    of the other thread is given (none collected, none hidden), and the
    site's own warnings are those it gives run alone."""
    from surpyval.utils.warnings import collecting_warnings

    with collecting_warnings() as alone:
        site()

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

    runs: list[list[tuple]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", _Probe)
        thread = threading.Thread(target=warner)
        thread.start()
        try:
            for _ in range(2):
                with collecting_warnings() as caught:
                    site()
                runs.append(_as_pairs(caught))
        finally:
            done.set()
            thread.join(timeout=300)
    assert raised > 0
    assert hidden == 0
    assert runs == [_as_pairs(alone)] * 2
