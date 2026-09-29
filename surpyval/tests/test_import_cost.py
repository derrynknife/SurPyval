"""``import surpyval`` must not import matplotlib (#363).

A server that never plots paid for pyplot, and its backend probing, on
every cold start. matplotlib is now imported inside the plotting methods.
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
