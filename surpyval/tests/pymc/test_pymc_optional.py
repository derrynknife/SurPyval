"""The PyMC helpers without PyMC: ``import surpyval`` never imports it,
and each helper refuses with what to install. These run whether or not
PyMC is installed (its absence is simulated)."""

import subprocess
import sys

import pytest

import surpyval as sp


def test_import_surpyval_does_not_import_pymc():
    code = (
        "import sys, surpyval, surpyval.pymc; "
        "print(sorted(m for m in ('pymc', 'pytensor', 'arviz') "
        "if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "[]"


@pytest.fixture
def without_pymc(monkeypatch):
    # A None entry in sys.modules makes the import raise ImportError, as
    # it does where the package is not installed.
    for module in ("pymc", "pytensor", "pytensor.tensor", "arviz"):
        monkeypatch.setitem(sys.modules, module, None)


def _calls():
    from surpyval import pymc as spm

    return [
        (
            "log_likelihood",
            lambda: spm.log_likelihood(sp.Weibull, x=[1, 2], alpha=1, beta=1),
        ),
        (
            "censored_likelihood",
            lambda: spm.censored_likelihood(
                sp.Weibull, x=[1, 2], alpha=1, beta=1
            ),
        ),
        (
            "prior_from_model",
            lambda: spm.prior_from_model(sp.Weibull.fit([1, 2, 3, 5])),
        ),
        (
            "prior_from_spec",
            lambda: spm.prior_from_spec({"beta": ("lognormal", 0, 1)}),
        ),
    ]


@pytest.mark.parametrize("index", range(4))
def test_each_helper_says_to_install_pymc(without_pymc, index):
    what, call = _calls()[index]
    with pytest.raises(ImportError, match="install pymc to use this") as err:
        call()
    assert what in str(err.value)
    assert "pip install pymc" in str(err.value)
