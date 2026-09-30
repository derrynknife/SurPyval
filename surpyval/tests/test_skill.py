"""The bundled Claude Code skill's code runs (#491).

``.claude/skills/surpyval/SKILL.md`` is what an assistant copies into a
user's code, so it drifted silently: a removed keyword (``alpha=`` for
``alpha_ci=``), removed classes, a withdrawn warning. Every ``python``
block in it is run here, in order, in one namespace, with SurPyval's
deprecation and removal errors raised, and a few of its claims are checked
against the package.
"""

import pathlib
import re
import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

SKILL = (
    pathlib.Path(__file__).resolve().parents[2]
    / ".claude"
    / "skills"
    / "surpyval"
    / "SKILL.md"
)

pytestmark = pytest.mark.skipif(
    not SKILL.exists(), reason="the skill ships with the repository only"
)


def _blocks():
    text = SKILL.read_text()
    return re.findall(r"```python\n(.*?)```", text, flags=re.S)


def test_skill_has_code():
    assert len(_blocks()) >= 4


def test_every_code_block_runs():
    namespace: dict = {}
    for k, block in enumerate(_blocks()):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            try:
                exec(compile(block, f"SKILL.md block {k}", "exec"), namespace)
            except Exception as error:  # pragma: no cover - the report
                pytest.fail(f"SKILL.md block {k} failed: {error!r}\n{block}")
    import matplotlib.pyplot as plt

    plt.close("all")


def test_regression_example_recovers_its_coefficient():
    # The old example drew Z and the noise from two default_rng(0)
    # generators, so they were correlated and Cox gave 1.04 for 0.6.
    namespace: dict = {}
    block = next(b for b in _blocks() if "WeibullPH" in b)
    exec(block, namespace)
    assert namespace["cox"].beta[0] == pytest.approx(0.6, abs=0.05)
    assert namespace["model"].params[2] == pytest.approx(0.6, abs=0.05)


def test_claims_about_the_package():
    import surpyval as sp

    text = SKILL.read_text()
    assert "0.15.2" not in text and "SeriesModel" not in text
    assert "alpha=0.05" not in text
    # logrank is at the top level, not in surpyval.utils
    assert callable(sp.logrank) and not hasattr(sp.utils, "logrank")
    # 0.22: Bernoulli's sf is P(X > x)
    np.testing.assert_allclose(
        sp.Bernoulli.from_params([0.9]).sf([0, 1]), [0.9, 0.0]
    )
