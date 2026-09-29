"""Known failures found by the properties, and the data they need (#379).

Each entry is a predicate on a generated data set that is true where a
known bug bites. The properties ``assume`` it away, so the rest of the
data space is still searched and the suite stays green; the bug itself
is pinned by a plain strict-xfail test with its minimal example in
``test_known_failures.py``. When that test starts to pass (XPASS, which
fails a strict xfail), delete it and the predicate here.
"""

import numpy as np


def turnbull_all_right_truncated(data):
    """Every row has a finite right truncation time.

    Turnbull's ladder is ``bounds[1:-1]``, which assumes the last bound is
    ``+inf`` (the default ``tr``). With every ``tr`` finite the last
    bound is the largest ``tr`` and the piece ending there is dropped: a
    failure at 1 observable up to 1 gives ``sf(1) == 1``, and a single
    left censored row leaves an empty ladder on which ``sf`` raises
    ``IndexError``.
    """
    tr = data.get("tr")
    return tr is not None and bool(np.all(np.isfinite(tr)))


def point_mass_supremum(data):
    """A distribution concentrated ever closer to some time ``v`` does
    not lose likelihood on any row.

    Then a family that can approach a point mass has no maximum
    likelihood fit (its likelihood grows without bound, or tends to a
    supremum it never reaches), and a local-optimum or invariance
    property does not apply. A row keeps its likelihood under a spike at
    ``v`` when:

    - it is exact at ``v``, or exact at its own right truncation time
      ``tr < v`` (its contribution ``f(x) / F(x)`` grows without bound);
    - ``v`` is in (the closure of) its set of event times: ``(x, tr]``
      right censored, ``(tl, x]`` left censored, ``(xl, xr]`` interval;
    - its window lies above ``v`` and its set starts at the window's
      start (given ``X > tl``, a spike below ``tl`` leaves its tail just
      above ``tl``), or the mirror image below ``v``.

    The fitters mean to refuse such data
    (``ParametricFitter._check_identifiable``) but catch only fewer
    distinct non-right-censored values than parameters: an exact value
    at 0.5 and a left censored one at 1 give a Weibull ``beta`` of 395.7
    with no warning, and take LogNormal about 20 s (pinned in
    ``test_known_failures.py``).
    """
    x = np.asarray(data["x"], dtype=float)
    xl = x if x.ndim == 1 else x[:, 0]
    xr = x if x.ndim == 1 else x[:, 1]
    c = np.asarray(data["c"])
    size = xl.size
    tl = np.asarray(data.get("tl", np.full(size, -np.inf)), dtype=float)
    tr = np.asarray(data.get("tr", np.full(size, np.inf)), dtype=float)
    lower = np.select([c == 1, c == -1], [xl, tl], default=xl)
    upper = np.select([c == 1, c == -1], [tr, xr], default=xr)
    exact = c == 0

    points = np.unique(np.r_[xl, xr, tl, tr])
    points = points[np.isfinite(points)]
    candidates = np.r_[
        points, (points[1:] + points[:-1]) / 2, points[0] - 1, points[-1] + 1
    ]
    for v in candidates:
        ok = np.where(
            exact,
            (xl == v) | ((xl == tr) & (xl < v)),
            ((lower <= v) & (v <= upper))
            | ((v <= tl) & (lower == tl))
            | ((v >= tr) & (upper == tr)),
        )
        if ok.all():
            return True
    return False


def truncated(data):
    """Some row is truncated (a finite ``tl`` or ``tr``).

    A parametric fit to truncated data can depend on the time unit: the
    optimiser stops at a different, worse point on the rescaled data. On
    right censored at 10.5 (observable up to 13), exact 6.5, right
    censored 2.5 and left censored 11, Normal gives mu 8.536, sigma 2.461
    (log-likelihood -4.034), but on the data times 7.3 mu 48.87, sigma
    23.04 (-4.630 in the original unit) instead of 62.31, 17.96; Gumbel's
    rescaled fit gives the data zero likelihood (pinned in
    ``test_known_failures.py``). The units property is therefore run on
    untruncated data only (generated so, rather than filtered with this).
    """
    return any(np.isfinite(data[k]).any() for k in ("tl", "tr") if k in data)
