"""``import surpyval`` must not import matplotlib (#363), pandas or
formulaic (#470).

A server that never plots paid for pyplot, and its backend probing, on
every cold start. matplotlib is now imported inside the plotting methods.
pandas and formulaic took half of ``import surpyval``; the models that
need them are imported on first use.
"""

import subprocess
import sys


def test_import_surpyval_does_not_load_matplotlib() -> None:
    code = (
        "import sys, surpyval, surpyval.recurrent, surpyval.degradation, "
        "surpyval.multivariate, surpyval.metrics; "
        "print(sorted(m for m in sys.modules "
        "if m == 'matplotlib' or m.startswith('matplotlib.')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "[]"


def test_plotting_still_works_after_a_lazy_import() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np

    import surpyval as surv

    x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    assert surv.KaplanMeier.fit(x).plot() is not None
    assert surv.Weibull.fit(x).plot() is not None


def test_import_surpyval_does_not_load_pandas_or_formulaic() -> None:
    # They took half of ``import surpyval`` (#470): the regression,
    # competing-risks, recurrent and degradation models and the metrics
    # load on first use, and the core imports them where they are used.
    # (scipy.stats still loads: autograd.scipy, which the distributions
    # differentiate with, imports it.)
    code = (
        "import sys, surpyval; "
        "surpyval.Weibull.fit([1.0, 2.0, 4.0, 7.0]); "
        "surpyval.KaplanMeier.fit([1.0, 2.0, 4.0, 7.0]).sf(3.0); "
        "print(sorted(m for m in ('pandas', 'formulaic', 'narwhals') "
        "if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "[]"


def test_lazy_names_are_the_packages_names() -> None:
    # Every name the subpackages export at the top level is in the lazy
    # map, and resolves to the package's own object.
    import importlib
    import types

    import surpyval
    import surpyval.univariate.regression as regression

    lazy = surpyval._LAZY  # type: ignore[attr-defined]
    exported = {
        k for k, v in lazy.items() if v == "surpyval.univariate.regression"
    }
    # The life models moved to ``surpyval.life_models``
    moved = set(surpyval._MOVED_TO_LIFE_MODELS)  # type: ignore[attr-defined]
    moved |= {"GeneralLogLinear"}
    assert exported | moved == set(regression.__all__)
    assert not exported & moved
    for name, module in lazy.items():
        assert name in dir(surpyval)
        package = importlib.import_module(module)
        assert getattr(surpyval, name) is getattr(package, name)
    for name in ("degradation", "life_models", "metrics", "recurrent"):
        assert isinstance(getattr(surpyval, name), types.ModuleType)
    assert isinstance(surpyval.univariate.regression, types.ModuleType)
    assert isinstance(surpyval.univariate.competing_risks, types.ModuleType)


def test_lazy_subpackages_in_a_fresh_interpreter() -> None:
    # ``surpyval.recurrent.X`` and ``from surpyval import CoxPH`` work with
    # nothing imported beforehand, as when ``import surpyval`` loaded them.
    code = (
        "import surpyval; from surpyval import CoxPH, FineGray; "
        "print(surpyval.recurrent.laplace.__name__, "
        "surpyval.univariate.regression.CoxPH is CoxPH, "
        "surpyval.degradation.WienerProcess is surpyval.WienerProcess)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.split() == ["laplace", "True", "True"]
