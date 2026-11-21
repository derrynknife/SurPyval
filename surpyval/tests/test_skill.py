"""The bundled Claude Code skill's code runs (#491).

``.claude/skills/surpyval/SKILL.md`` (and its ``references/``) is what an
assistant copies into a user's code, so it drifted silently: a removed
keyword (``alpha=`` for ``alpha_ci=``), removed classes, a withdrawn
warning. Every ``python`` block in it is run here, in order, in one
namespace, with SurPyval's deprecation and removal errors raised, and a
few of its claims are checked against the package. Each reference file's
blocks run the same way, in a namespace of their own.
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

REFERENCES = sorted((SKILL.parent / "references").glob("*.md"))

pytestmark = pytest.mark.skipif(
    not SKILL.exists(), reason="the skill ships with the repository only"
)


def _blocks(path=SKILL):
    text = path.read_text()
    return re.findall(r"```python\n(.*?)```", text, flags=re.S)


def test_skill_has_code():
    assert len(_blocks()) >= 4


def _run_blocks(path):
    namespace: dict = {}
    for k, block in enumerate(_blocks(path)):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            try:
                exec(
                    compile(block, f"{path.name} block {k}", "exec"), namespace
                )
            except Exception as error:  # pragma: no cover - the report
                pytest.fail(
                    f"{path.name} block {k} failed: {error!r}\n{block}"
                )
    import matplotlib.pyplot as plt

    plt.close("all")
    return namespace


def test_every_code_block_runs():
    _run_blocks(SKILL)


@pytest.mark.parametrize("path", REFERENCES, ids=lambda p: p.name)
def test_every_reference_code_block_runs(path):
    assert _blocks(path), f"{path.name} has no python block"
    _run_blocks(path)


def test_skill_links_each_reference():
    text = SKILL.read_text()
    for path in REFERENCES:
        assert f"references/{path.name}" in text, path.name


def test_choosing_a_model_flow_is_yes_or_no():
    # Every question of the flow has exactly two answers, Yes and No, and
    # the docs page asks the same numbered questions.
    path = SKILL.parent / "references" / "choosing-a-model.md"
    text = path.read_text()
    questions = re.findall(
        r"^\s*(\d+)\. \*\*(.+?)\*\*", text, flags=re.M | re.S
    )
    numbers = [int(k) for k, _ in questions]
    assert numbers == list(range(1, len(numbers) + 1)), numbers
    blocks = re.split(r"^\s*\d+\. \*\*", text, flags=re.M)[1:]
    for k, block in enumerate(blocks, start=1):
        block = block.split("\n## ")[0]
        answers = re.findall(r"^\s*- (Yes|No)\b", block, flags=re.M)
        assert answers == ["Yes", "No"], (k, answers)
    docs = SKILL.parents[3] / "docs" / "Choosing a Model.rst"
    if docs.exists():
        flow = docs.read_text().split("Stage 1: the shape of the problem")[1]
        page = re.findall(r"^\s*(\d+)\. \*\*", flow, flags=re.M)
        assert [int(k) for k in page] == numbers, page


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
