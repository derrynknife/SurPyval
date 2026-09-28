"""Reference results from Python packages for surpyval/tests/reference.

Reads the shared fixtures (``surpyval/tests/reference/data/fixtures.json``,
written by ``make_fixtures.py``) and writes ``py_lifelines.json`` and
``py_sksurv.json`` beside them, in the same layout as the R files: every
entry records the software, its version, the call, the settings and the
values.

lifelines is not a SurPyval dependency and pins an older pandas, so run
this from a separate virtual environment rather than the development one
(scikit-survival comes through ``--system-site-packages``)::

    python -m venv --system-site-packages /tmp/ref_venv
    /tmp/ref_venv/bin/pip install lifelines
    /tmp/ref_venv/bin/python scripts/reference/reference_python.py

from the repository root. Re-running reproduces the files.
"""

import json
import math
import platform
from pathlib import Path

import lifelines
import numpy as np
import pandas as pd
import sksurv
from lifelines import (
    ExponentialFitter,
    LogLogisticFitter,
    LogNormalFitter,
    WeibullAFTFitter,
    WeibullFitter,
)
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (
    brier_score,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv

DATA = (
    Path(__file__).resolve().parents[2]
    / "surpyval"
    / "tests"
    / "reference"
    / "data"
)


def _native(value):
    """Plain lists of floats with ``None`` for non-finite values."""
    if isinstance(value, dict):
        return {k: _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(v) for v in value]
    if isinstance(value, np.ndarray):
        return _native(value.tolist())
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _entry(fixture, software, version, call, settings, values, note=None):
    out = {
        "fixture": fixture,
        "software": software,
        "version": version,
        "call": call,
        "settings": settings,
        "values": _native(values),
    }
    if note is not None:
        out["note"] = note
    return out


def _write(name, references):
    out = {
        "generator": "scripts/reference/reference_python.py",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "references": references,
    }
    path = DATA / name
    path.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    print("wrote", path)


def _frame(fixtures, name):
    return pd.DataFrame(fixtures[name]["columns"]).astype(float)


# lifelines parameterisations, recorded next to each value.
UNIVARIATE = {
    "weibull": (
        WeibullFitter,
        ["lambda_", "rho_"],
        "S(t) = exp(-(t / lambda_) ** rho_)",
    ),
    "lognormal": (
        LogNormalFitter,
        ["mu_", "sigma_"],
        "S(t) = 1 - Phi((log t - mu_) / sigma_)",
    ),
    "loglogistic": (
        LogLogisticFitter,
        ["alpha_", "beta_"],
        "S(t) = 1 / (1 + (t / alpha_) ** beta_)",
    ),
    "exponential": (
        ExponentialFitter,
        ["lambda_"],
        "S(t) = exp(-t / lambda_)",
    ),
}


def lifelines_references(fixtures):
    refs = {}
    version = lifelines.__version__
    lt = _frame(fixtures, "left_truncation")
    iv = fixtures["interval"]["columns"]
    lower = np.array(iv["left"], dtype=float)
    upper = np.array(
        [np.inf if v is None else v for v in iv["right"]], dtype=float
    )
    for dist, (Fitter, names, param) in UNIVARIATE.items():
        fitter = Fitter()
        fitter.fit(lt["x"], event_observed=1 - lt["c"], entry=lt["tl"])
        refs["{}_left_truncation".format(dist)] = _entry(
            "left_truncation",
            "lifelines",
            version,
            "{}().fit(x, event_observed=1 - c, entry=tl)".format(
                Fitter.__name__
            ),
            {"parameterisation": param},
            {
                "params": [getattr(fitter, n) for n in names],
                "names": names,
                "loglik": fitter.log_likelihood_,
            },
        )
        fitter = Fitter()
        fitter.fit_interval_censoring(lower, upper)
        refs["{}_interval".format(dist)] = _entry(
            "interval",
            "lifelines",
            version,
            "{}().fit_interval_censoring(left, right)".format(Fitter.__name__),
            {
                "parameterisation": param,
                "intervals": "left == right exact, right = inf right "
                "censored, left = 0 left censored",
            },
            {
                "params": [getattr(fitter, n) for n in names],
                "names": names,
                "loglik": fitter.log_likelihood_,
            },
        )

    aft = WeibullAFTFitter()
    frame = pd.DataFrame(
        {"x": lt["x"], "event": 1 - lt["c"], "tl": lt["tl"], "z": lt["z"]}
    )
    aft.fit(frame, "x", event_col="event", entry_col="tl")
    refs["weibull_aft_left_truncation"] = _entry(
        "left_truncation",
        "lifelines",
        version,
        "WeibullAFTFitter().fit(df[x, event, tl, z], 'x', "
        "event_col='event', entry_col='tl')",
        {
            "parameterisation": "S(t | z) = exp(-(t / lambda) ** rho), "
            "log lambda = lambda_intercept + lambda_z * z, "
            "log rho = rho_intercept"
        },
        {
            "lambda_intercept": aft.params_["lambda_"]["Intercept"],
            "lambda_z": aft.params_["lambda_"]["z"],
            "rho_intercept": aft.params_["rho_"]["Intercept"],
            "loglik": aft.log_likelihood_,
        },
    )
    return refs


def sksurv_references(fixtures):
    refs = {}
    version = sksurv.__version__
    for name in ["prediction_ties", "prediction_continuous"]:
        fx = fixtures[name]
        x = np.array(fx["columns"]["x"], dtype=float)
        event = np.array(fx["columns"]["c"]) == 0
        y = Surv.from_arrays(event=event, time=x)
        S = np.array(fx["survival"], dtype=float)
        times = np.array(fx["times"], dtype=float)
        _, bs = brier_score(y, y, S, times)
        ibs = integrated_brier_score(y, y, S, times)
        auc, _ = cumulative_dynamic_auc(y, y, 1 - S, times)
        note = None
        if name == "prediction_ties":
            note = (
                "events tie with censorings here; scikit-survival weights "
                "an event by 1/G(x_i), SurPyval (like pec) by 1/G(x_i-) "
                "(#365)"
            )
        refs[name.replace("prediction", "metrics")] = _entry(
            name,
            "scikit-survival",
            version,
            "brier_score(y, y, S, times); integrated_brier_score(y, y, S, "
            "times); cumulative_dynamic_auc(y, y, 1 - S, times)",
            {"train": "the evaluation data (y) estimate G"},
            {
                "times": times,
                "brier": bs,
                "integrated_brier": ibs,
                "auc": auc,
            },
            note=note,
        )

    ties = _frame(fixtures, "ties")
    y = Surv.from_arrays(event=ties["c"] == 0, time=ties["x"])
    for method in ["breslow", "efron"]:
        model = CoxPHSurvivalAnalysis(ties=method, tol=1e-12, n_iter=200)
        model.fit(ties[["z1", "z2"]].to_numpy(), y)
        refs["cox_ties_{}".format(method)] = _entry(
            "ties",
            "scikit-survival",
            version,
            "CoxPHSurvivalAnalysis(ties='{}', tol=1e-12, n_iter=200)"
            ".fit(Z[z1, z2], y)".format(method),
            {"ties": method},
            {"coef": model.coef_},
        )
    return refs


def main():
    fixtures = json.loads((DATA / "fixtures.json").read_text())
    _write("py_lifelines.json", lifelines_references(fixtures))
    _write("py_sksurv.json", sksurv_references(fixtures))


if __name__ == "__main__":
    main()
