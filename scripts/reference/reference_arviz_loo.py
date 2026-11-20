"""ArviZ's PSIS-LOO and WAIC on fixed log-likelihood matrices, for
``surpyval/tests/bayes/test_criteria.py``.

Writes ``surpyval/tests/bayes/arviz_loo.json``: for each case, ArviZ's
pointwise ``elpd_loo``, Pareto ``k``, ``elpd_waic`` and ``p_waic``, with
its version and the calls. The matrices need no random numbers, so the
test rebuilds them exactly: case ``j`` is a normal model with ``sigma``
known, its posterior of ``mu`` ``N(m, s)`` taken at the ``S`` quantiles
``m + s * Phi^-1((i + 0.5) / S)``, and ``ll[i, r] = log N(y_r; mu_i, 1)``.

ArviZ is not a SurPyval dependency, and needs a matplotlib older than the
development one; run this from a separate virtual environment::

    python -m venv /tmp/arviz_venv
    /tmp/arviz_venv/bin/pip install arviz==0.20.0 matplotlib==3.9.2
    /tmp/arviz_venv/bin/python scripts/reference/reference_arviz_loo.py

from the repository root. Re-running reproduces the file.
"""

import json
import platform
from pathlib import Path

import arviz as az
import numpy as np
import scipy
from scipy import stats

OUT = (
    Path(__file__).resolve().parents[2]
    / "surpyval"
    / "tests"
    / "bayes"
    / "arviz_loo.json"
)

#: (name, posterior mean, posterior sd, draws, observations)
CASES = [
    ("well_behaved", 0.5, 0.45, 1000, [-1.0, 0.0, 0.4, 1.2, 2.0]),
    ("outlier", 1.0, 0.3, 4000, [0.2, 0.7, 1.1, 1.4, 9.0]),
    ("few_draws", 0.0, 1.0, 60, [-0.5, 0.3, 2.5]),
]


def matrix(m, s, draws, y):
    mu = m + s * stats.norm.ppf((np.arange(draws) + 0.5) / draws)
    return stats.norm.logpdf(np.asarray(y)[None, :], mu[:, None], 1.0)


def main():
    out = {
        "generator": "scripts/reference/reference_arviz_loo.py",
        "software": "arviz",
        "version": az.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
        "references": {},
    }
    for name, m, s, draws, y in CASES:
        ll = matrix(m, s, draws, y)
        idata = az.from_dict(log_likelihood={"y": ll[None, :, :]})
        loo = az.loo(idata, pointwise=True, reff=1.0)
        waic = az.waic(idata, pointwise=True)
        log_w, k = az.psislw(-ll.T, reff=1.0)
        out["references"][name] = {
            "case": {
                "posterior_mean": m,
                "posterior_sd": s,
                "draws": draws,
                "y": y,
            },
            "call": (
                "az.loo(idata, pointwise=True, reff=1.0); "
                "az.waic(idata, pointwise=True); az.psislw(-ll.T, reff=1.0)"
            ),
            "values": {
                "elpd_loo": [float(v) for v in loo.loo_i.values],
                "pareto_k": [float(v) for v in loo.pareto_k.values],
                "psislw_k": [float(v) for v in np.asarray(k)],
                "p_loo": float(loo.p_loo),
                "elpd_waic": [float(v) for v in waic.waic_i.values],
                "p_waic": float(waic.p_waic),
                "good_k": float(loo.good_k),
            },
        }
    OUT.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
