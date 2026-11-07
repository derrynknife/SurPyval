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
