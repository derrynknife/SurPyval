"""The torch helpers without PyTorch: ``import surpyval`` never imports
it, and each helper refuses with what to install. These run in the main
suite, whether or not PyTorch is installed (its absence is simulated)."""

import subprocess
import sys

import pytest

import surpyval as sp


def test_import_surpyval_does_not_import_torch():
    code = (
        "import sys, surpyval, surpyval.torch; "
        "from surpyval.torch import SUPPORTED, Targets, constrain, nll, "
        "sf, targets, to_surpyval; "
        "print('torch' in sys.modules, len(SUPPORTED))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False 12"


def test_the_families_are_those_of_the_pymc_helpers():
    from surpyval import pymc as spm
    from surpyval import torch as spt

    assert spt.SUPPORTED == spm.SUPPORTED


def test_every_public_name_is_listed():
    from surpyval import torch as spt

    assert set(spt.__all__) <= set(dir(spt))
    with pytest.raises(AttributeError, match="no attribute 'NoSuchName'"):
        spt.NoSuchName


@pytest.fixture
def without_torch(monkeypatch):
    # A None entry in sys.modules makes the import raise ImportError, as
    # it does where the package is not installed.
    monkeypatch.setitem(sys.modules, "torch", None)
    from surpyval.torch.loss import nll_loss_class

    # (the class built by an earlier test, where torch is installed)
    nll_loss_class.cache_clear()
    yield
    nll_loss_class.cache_clear()


def _calls():
    from surpyval import torch as spt

    return [
        ("targets", lambda: spt.targets(x=[1, 2], c=[0, 1])),
        ("constrain", lambda: spt.constrain(sp.Weibull, [[0.0, 0.0]])),
        (
            "nll",
            lambda: spt.nll(sp.Weibull, None, {"alpha": 1.0, "beta": 1.0}),
        ),
        ("NLLLoss", lambda: spt.NLLLoss(sp.Weibull)),
        ("sf", lambda: spt.sf(sp.Weibull, [1.0], {"alpha": 1, "beta": 1})),
        (
            "to_surpyval",
            lambda: spt.to_surpyval(sp.Weibull, {"alpha": 1, "beta": 1}),
        ),
    ]


@pytest.mark.parametrize("index", range(6))
def test_each_helper_says_to_install_torch(without_torch, index):
    what, call = _calls()[index]
    with pytest.raises(ImportError, match="pip install torch") as err:
        call()
    assert str(err.value).startswith(f"{what}: the torch helpers need PyTorch")
