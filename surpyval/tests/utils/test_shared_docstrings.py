"""Each distribution method's parameter entries match its class's
``_param_docs`` (``surpyval.utils.docstrings``); a drifted copy fails,
naming the class, the method and the difference.
``python scripts/sync_docstrings.py --write`` rewrites the copies."""

from pathlib import Path

import pytest

import surpyval
from surpyval.utils.docstrings import copies, sync

DISTRIBUTIONS = Path(surpyval.__file__).parent / (
    "univariate/parametric/distributions"
)
FILES = sorted(
    p for p in DISTRIBUTIONS.glob("*.py") if "_param_docs =" in p.read_text()
)


def test_every_distribution_with_parameters_states_them_once():
    # 18 distributions, about nine methods each, keep a copy.
    assert len(FILES) == 18
    assert sum(len(copies(p)) for p in FILES) >= 150


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_copies_match_param_docs(path):
    drifted = [c for c in copies(path) if not c.in_step]
    assert not drifted, (
        "Parameter entries differ from _param_docs (run "
        "`python scripts/sync_docstrings.py --write`):\n"
        + "\n".join(f"{c.cls}.{c.method}\n{c.diff()}" for c in drifted)
    )


SOURCE = '''
class A:
    _param_docs = """
        a : float
            The scale.
        b : float
            The shape.
        """

    def f(self, x, a, b):
        """F.

        Parameters
        ----------
        x : float
            Where.
        a : float
            The scale (drifted).
        b : float
            The shape.

        Returns
        -------
        a : float
            Not a parameter entry.
        """

    def g(self, n):
        """G.

        Parameters
        ----------
        a : int
            Another meaning, with no b after it: not a copy.
        """
'''


def test_drift_is_named_and_written_back(tmp_path):
    path = tmp_path / "dist.py"
    path.write_text(SOURCE)
    found = copies(path)
    assert [(c.cls, c.method, c.in_step) for c in found] == [("A", "f", False)]
    assert "-            The scale.\n+            The scale (drifted)." in (
        found[0].diff()
    )
    assert [c.method for c in sync(path)] == ["f"]
    text = path.read_text()
    assert "(drifted)" not in text
    assert "Not a parameter entry." in text
    assert "Another meaning" in text
    assert all(c.in_step for c in copies(path))
