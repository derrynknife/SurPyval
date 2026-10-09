"""CoxPH's information without per-time ``p x p`` arrays, and its
Newton-Raphson solver (#516).

The information is now ``Z' diag(q) Z`` over the rows rather than a sum of
per-event-time ``p x p`` matrices, the not-yet-entered terms are skipped
when nothing is truncated, and the coefficients come from Newton-Raphson
with step-halving instead of ``root(hybr)``. The score and information
are checked against their definition (a literal loop over the event
times), and the fits against values computed on the code before #516
(3a5ccc2).
"""

import tracemalloc
import warnings
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from surpyval import CompetingRisksProportionalHazards, CoxPH
from surpyval.univariate.regression.proportional_hazards import (
    cox_ph,
    cox_separation,
)
from surpyval.utils import validate_coxph

MONOTONE = "No finite maximum: the partial likelihood"


def _data(kind):
    """Deterministic data: ``x, Z, c, n, tl`` (``n`` and ``tl`` may be
    None) and stratum labels."""
    rng = np.random.default_rng(516)
    N, p = {"one_row": (1, 2), "small": (12, 2)}.get(kind, (240, 3))
    Z = rng.normal(size=(N, p))
    Z[:, 0] = rng.integers(0, 2, N)
    t = 10 * rng.weibull(1.5, N) * np.exp(-(Z @ np.linspace(0.5, -0.3, p)))
    c = (rng.random(N) < 0.3).astype(int)
    x = np.where(c == 1, t * rng.uniform(0.2, 1.0, N), t)
    if kind != "untied":
        # Ties: on a grid of 0.5, or of 2 for heavy ties.
        step = 2.0 if kind == "heavy_ties" else 0.5
        x = np.ceil(x / step) * step
    x = np.maximum(x, 0.05)
    n = None
    if kind in ("weighted", "heavy_ties", "truncated_weighted"):
        n = rng.integers(1, 5, N).astype(float)
    tl = None
    if kind.startswith("truncated"):
        tl = np.where(rng.random(N) < 0.4, x * rng.uniform(0, 0.9, N), 0.0)
        tl = np.floor(tl * 4) / 4
    if kind == "one_row":
        c = np.zeros(N, int)
    strata = rng.integers(0, 3, N)
    return x, Z, c, n, tl, strata


def _definition(x, Z, c, n, tl, beta, efron):
    """The score (of the negative partial log-likelihood) and the
    information, straight from their definition: a loop over the event
    times, the risk set ``tl < tau <= x`` and, for Efron, the loop over the
    ``int(d)`` tied deaths with ``c = j / d``. Also the size of the terms
    that cancel in the information, the scale of its rounding error."""
    p = Z.shape[1]
    w = n * np.exp(Z @ beta)
    score = np.zeros(p)
    info = np.zeros((p, p))
    size = 0.0
    for tau in np.unique(x[c == 0]):
        risk = (tl < tau) & (x >= tau)
        dead = (x == tau) & (c == 0)
        d = n[dead].sum()
        R, ZR = w[risk].sum(), w[risk] @ Z[risk]
        Z2R = (Z[risk].T * w[risk]) @ Z[risk]
        D, ZD = w[dead].sum(), w[dead] @ Z[dead]
        Z2D = (Z[dead].T * w[dead]) @ Z[dead]
        score -= n[dead] @ Z[dead]
        if efron:
            terms = [(j / d, 1.0) for j in range(int(d))]
        else:
            terms = [(0.0, d)]
        for k, mult in terms:
            r = R - k * D
            a = ZR - k * ZD
            score += mult * a / r
            info += mult * ((Z2R - k * Z2D) / r - np.outer(a, a) / r**2)
            size += mult * np.abs(Z2R - k * Z2D).max() / r
    return score, info, size


KINDS = [
    "untied",
    "tied",
    "heavy_ties",
    "weighted",
    "truncated",
    "truncated_weighted",
    "small",
    "one_row",
]


@pytest.mark.parametrize("kind", KINDS + ["fractional"])
@pytest.mark.parametrize("method", ["efron", "breslow"])
def test_score_and_information_match_their_definition(kind, method):
    x, Z, c, n, tl, _ = _data("weighted" if kind == "fractional" else kind)
    x, c, n, tl, Z = validate_coxph(x, c, n, Z, tl, method)
    if kind == "fractional":
        # Weights rather than counts: Efron takes int(d) terms, c = j / d.
        n = n * np.random.default_rng(1).uniform(0.4, 1.3, len(n))
    generator = {
        "efron": CoxPH.create_efron_ll_jac_hess,
        "breslow": CoxPH.create_breslow_ll_jac_hess,
    }[method]
    _, jac_hess = generator(x, Z, c, n, tl)
    rng = np.random.default_rng(3)
    for beta in [np.zeros(Z.shape[1]), rng.normal(0, 0.5, Z.shape[1])]:
        score, info = jac_hess(beta)
        want_score, want_info, size = _definition(
            x, Z, c, n, tl, beta, method == "efron"
        )
        tol = dict(rtol=1e-12, atol=1e-14 * size)
        np.testing.assert_allclose(info, want_info, **tol)
        np.testing.assert_allclose(score, want_score, **tol)
        np.testing.assert_array_equal(info, info.T)


def test_untruncated_fits_skip_the_not_yet_entered_terms():
    # They are exact zeros without truncation; the old code built them
    # (an (n, p, p) gather among them) on every evaluation.
    x, Z, c, n, tl, _ = _data("tied")
    for method in ("efron", "breslow"):
        with mock.patch.object(
            cox_ph, "not_yet_entered", side_effect=AssertionError("built")
        ):
            CoxPH.fit(x, Z, c, n=n, tie_method=method)


def test_the_information_needs_no_per_time_p_by_p_arrays():
    # The old generator kept an (n, p, p) array of z z' and built more per
    # evaluation, 29 MB each here; the new one needs O(n p), about 1 MB
    # an array.
    rng = np.random.default_rng(5)
    N, p = 4_000, 30
    Z = rng.normal(size=(N, p))
    x = rng.exponential(size=N)
    c = np.zeros(N)
    ones = np.ones(N)
    tl = np.full(N, -np.inf)
    for generator in (
        CoxPH.create_efron_ll_jac_hess,
        CoxPH.create_breslow_ll_jac_hess,
    ):
        tracemalloc.start()
        _, jac_hess = generator(x, Z, c, ones, tl)
        jac_hess(np.full(p, 0.1))
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert peak < N * p * p * 8 / 2, peak


def _counting(generator_name):
    """Patch a likelihood generator so its ``jac_hess`` counts its calls."""
    calls = []
    original = getattr(CoxPH, generator_name)

    def generator(*args):
        neg_ll, jac_hess = original(*args)

        def counted(beta):
            calls.append(1)
            return jac_hess(beta)

        return neg_ll, counted

    return mock.patch.object(CoxPH, generator_name, generator), calls


@pytest.mark.parametrize("method", ["efron", "breslow"])
def test_newton_raphson_needs_few_information_matrices(method):
    # root(hybr) evaluated the score and information about 16 times on
    # these data; Newton-Raphson needs one per step plus the start.
    x, Z, c, n, tl, _ = _data("tied")
    patch, calls = _counting(f"create_{method}_ll_jac_hess")
    with patch:
        model = CoxPH.fit(x, Z, c, tie_method=method)
    assert model.res.message == "Newton-Raphson converged"
    assert len(calls) <= 8, len(calls)


@pytest.mark.parametrize("kind", ["weighted", "truncated"])
@pytest.mark.parametrize("method", ["efron", "breslow"])
def test_newton_raphson_solves_the_score_to_rounding(kind, method):
    # Converged when a step is at most tol = 1e-10 standard errors, so the
    # score left is at the level of rounding (1e-15 standard errors here);
    # root(hybr), stopping on its relative change in beta, left 3e-13 to
    # 2.6e-12.
    x, Z, c, n, tl, _ = _data(kind)
    model = CoxPH.fit(x, Z, c, n=n, tl=tl, tie_method=method)
    score = model.jac(model.beta)[0]
    assert np.max(np.abs(score) * model.standard_errors()) < 1e-13


def test_the_root_finder_takes_over_when_newton_gives_up():
    # The fallback (the solver before #516) reaches the same maximum.
    x, Z, c, n, tl, _ = _data("truncated_weighted")
    newton = CoxPH.fit(x, Z, c, n=n, tl=tl)
    with mock.patch.object(cox_ph, "newton_raphson", return_value=None):
        fallback = CoxPH.fit(x, Z, c, n=n, tl=tl)
    assert "Newton" not in str(fallback.res.message)
    np.testing.assert_allclose(fallback.beta, newton.beta, rtol=1e-9)
    np.testing.assert_allclose(
        fallback.standard_errors(), newton.standard_errors(), rtol=1e-9
    )


def test_a_monotone_likelihood_still_warns():
    # Newton-Raphson does not converge (the step stays near 1 as the
    # coefficient runs off), so the root-finder takes over, as before,
    # and the fit warns once (#392).
    rng = np.random.default_rng(13)
    x = np.round(rng.exponential(1, 60), 1) + 0.1
    c = (rng.random(60) < 0.3).astype(int)
    Z = np.column_stack(
        [(x < np.median(x)).astype(float), rng.normal(size=60)]
    )
    with pytest.warns(UserWarning, match=MONOTONE) as w:
        model = CoxPH.fit(x, Z, c, center=True)
    assert len(w) == 1
    assert w[0].filename == __file__
    assert "Newton" not in str(model.res.message)


# Neither column runs off alone; together, in the proportion 0.25 : -1,
# they separate the events from the survivors (#728)
COMBINATION = (
    np.array([2, 1, 1.5, 0.5, 0.5, 2]),
    np.array([[0, 1.5], [0.5, 1], [-1.5, 1], [2, -1], [0, -1.5], [0, 1.5]]),
)
ORDERS = [np.arange(6), np.arange(6)[::-1], np.array([3, 0, 5, 1, 4, 2])]


@pytest.mark.parametrize("order", range(len(ORDERS)))
@pytest.mark.parametrize("method", ["efron", "breslow"])
def test_a_run_off_along_a_combination_has_no_finite_maximum(method, order):
    # It ran to (61.9, -247.6), the partial likelihood at its supremum,
    # with standard errors 0.6 and 0.5, and said "verified" (#728).
    x, Z = (a[ORDERS[order]] for a in COMBINATION)
    with pytest.warns(UserWarning, match=MONOTONE) as w:
        model = CoxPH.fit(x, Z, tie_method=method)
    assert len(w) == 1
    assert (
        "coefficients [0, 1] grow without bound together, in the "
        "proportion 0.25 : -1" in str(w[0].message)
    )
    assert model.maximum == "no finite maximum"
    assert np.isnan(model.standard_errors()).all()


def test_a_stratified_run_off_along_a_combination_is_found():
    x, Z = COMBINATION
    with pytest.warns(UserWarning, match="proportion 0.25 : -1"):
        model = CoxPH.fit(
            np.r_[x, x + 0.25], np.r_[Z, Z], strata=np.repeat([0, 1], 6)
        )
    assert model.maximum == "no finite maximum"


@pytest.mark.parametrize(
    "method, runs_off",
    [("efron", False), ("breslow", False), ("exact", True), ("kp", True)],
)
def test_the_exact_methods_ask_less_of_tied_deaths(method, runs_off):
    # Two deaths at 1 (z 1 and 2) above the survivor (z 0): Efron and
    # Breslow need the deaths level as well, the exact methods do not.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = CoxPH.fit([1, 1, 2], [[1.0], [2.0], [0.0]], tie_method=method)
    assert model.maximum == (
        "no finite maximum" if runs_off else "verified"
    ), model.beta
    assert len(w) == runs_off


@pytest.mark.parametrize("order", [[0, 1, 2, 3, 4, 5], [5, 3, 1, 0, 4, 2]])
def test_a_level_whose_one_row_is_an_event_has_no_finite_maximum(order):
    # "verified" at (-23.5, 35.9), in either order (#728, #714)
    df = pd.DataFrame(
        {
            "x": [0.5, 1, 0.5, 0.5, 0.5, 0.5],
            "c": [0, 0, 0, 0, 1, 1],
            "g": list("baaaaa"),
            "z0": [0.5, -1, -1, -1, 0.5, 0.5],
        }
    ).iloc[order]
    with pytest.warns(UserWarning, match="together, in the proportion"):
        model = CoxPH.fit_from_df(
            df, x_col="x", c_col="c", formula="z0 + C(g)"
        )
    assert model.maximum == "no finite maximum"


def test_an_ordinary_fit_does_not_look_for_a_run_off():
    # The data are asked only when the search gives cause (#728)
    x, Z, c, n, tl, _ = _data("truncated_weighted")
    with mock.patch.object(
        cox_ph, "runoff_direction", side_effect=AssertionError
    ):
        model = CoxPH.fit(x, Z, c, n=n, tl=tl)
    assert model.maximum == "verified"


def test_a_run_off_where_newton_raphson_gave_up_is_found():
    # Found by a random search (#746): Newton-Raphson gave up and BFGS
    # stopped where the score was below the verification's tolerance in
    # the units of the third covariate (a spread of 1e-7), every unit
    # within 2 of the average and the information at 0.29 of the start's;
    # none of the three triggers fired, and a likelihood with no finite
    # maximum was reported "verified". An answer Newton-Raphson did not
    # converge to now has the data asked.
    x = np.array([2.0, 1, 3, 1, 3, 4, 2])
    c = np.array([0, 0, 1, 0, 0, 0, 0])
    Z = np.array(
        [
            [-0.622, 0.08, 1.252],
            [-0.324, -1.102, -0.799],
            [1.777, -0.35, -1.184],
            [-0.302, 0.298, 0.287],
            [1.863, -0.191, -1.546],
            [1.509, 0.287, 0.3],
            [-0.679, -1.036, 1.581],
        ]
    ) * np.array([0.52, 179.0, 2.5e-8])
    with pytest.warns(UserWarning, match="No finite maximum"):
        model = CoxPH.fit(x, Z, c, tie_method="kp", center=True)
    assert model.maximum == "no finite maximum"


def test_runoff_direction_matches_every_pair_of_unit_and_time():
    # The cutting planes against the full programme, one constraint per
    # pair of a death and a unit at risk with it, on small random data
    from scipy.optimize import linprog

    rng = np.random.default_rng(728)
    for trial in range(60):
        N, p = rng.integers(4, 10), rng.integers(1, 3)
        x = rng.integers(1, 5, size=N).astype(float)
        Z = rng.integers(-2, 3, size=(N, p)).astype(float)
        c = (rng.random(N) < 0.3).astype(int)
        tl = np.where(rng.random(N) < 0.3, x - 1.5, -np.inf)
        method = ("efron", "exact")[trial % 2]
        if not np.any(c == 0):
            continue
        rows, level, gaps = [], [], np.zeros(p)
        for t in np.unique(x[c == 0]):
            D = np.flatnonzero((x == t) & (c == 0))
            S = np.flatnonzero((tl < t) & (x >= t) & ~((x == t) & (c == 0)))
            rows += [Z[j] - Z[k] for j in S for k in D]
            gaps += (Z[D].sum(0) * len(S) - Z[S].sum(0) * len(D)) / 4
            if method == "efron":
                level += [Z[k] - Z[D[0]] for k in D[1:]]
        full = linprog(
            -gaps,
            A_ub=np.array(rows) if rows else None,
            b_ub=np.zeros(len(rows)) if rows else None,
            A_eq=np.array(level) if level else None,
            b_eq=np.zeros(len(level)) if level else None,
            bounds=[(-1, 1)] * p,
        )
        found = cox_separation.runoff_direction(
            x, Z, c, np.ones(N), tl, None, method
        )
        assert (found is not None) == (-full.fun > 1e-9), (x, Z, c, tl)


# beta, se, -log L, H0 at T and sf at T for Z = 0.2, computed with the code
# before #516 (3a5ccc2): root(hybr) on the per-time p x p information.
T = np.array([0.5, 2.0, 5.0, 9.0])
FIT_CASES = {
    "tied/efron": ("tied", "efron", {}),
    "tied/breslow": ("tied", "breslow", {}),
    "untied/efron": ("untied", "efron", {}),
    "untied/breslow": ("untied", "breslow", {}),
    "heavy_ties/efron": ("heavy_ties", "efron", {}),
    "heavy_ties/breslow": ("heavy_ties", "breslow", {}),
    "truncated_weighted/efron": ("truncated_weighted", "efron", {}),
    "truncated_weighted/breslow": ("truncated_weighted", "breslow", {}),
    "tied/efron/center": ("tied", "efron", {"center": True}),
    "weighted/efron/strata": ("weighted", "efron", {"strata": True}),
    "truncated/breslow/strata": ("truncated", "breslow", {"strata": True}),
    "small/efron": ("small", "efron", {}),
}


def _fit(case):
    kind, method, kw = FIT_CASES[case]
    x, Z, c, n, tl, strata = _data(kind)
    kw = dict(kw)
    stratum = {}
    if kw.pop("strata", False):
        kw["strata"] = strata
        stratum = {"stratum": 1}
    model = CoxPH.fit(x, Z, c, n=n, tl=tl, tie_method=method, **kw)
    Zq = np.full(Z.shape[1], 0.2)
    return model, model.Hf(T, Zq, **stratum), model.sf(T, Zq, **stratum)


def _tvc_data():
    rng = np.random.default_rng(12)
    ids, xl, xr, cc, Z = [], [], [], [], []
    for s in range(150):
        end = np.round(rng.uniform(1, 10), 1)
        cuts = np.unique(np.round(rng.uniform(0.1, end - 0.1, 2), 1))
        edges = np.concatenate([[0], cuts[(cuts > 0) & (cuts < end)], [end]])
        event = rng.uniform() < 0.6
        for j in range(len(edges) - 1):
            ids.append(s)
            xl.append(edges[j])
            xr.append(edges[j + 1])
            cc.append(0 if (event and j == len(edges) - 2) else 1)
            Z.append(rng.normal(size=2))
    return ids, np.array(xl), np.array(xr), cc, np.array(Z)


def _competing_risks():
    x, Z, c, n, tl, _ = _data("tied")
    rng = np.random.default_rng(14)
    e = np.where(c == 0, rng.integers(1, 3, len(x)), np.nan)
    return CompetingRisksProportionalHazards.fit(x=x, Z=Z, c=c, e=e)


# fmt: off
OLD_FITS = {
    "tied/efron": dict(
        beta=[1.0168233666516833, 0.24658617107409916, -0.6068609530068154],
        se=[0.1733086656829556, 0.07534974947332118, 0.08727644625572786],
        neg_ll=696.9889434316875,
        H=[0.003933102644420112, 0.06171866529721044, 0.26739901631746926,
           0.5626681712449955],
        sf=[0.9960746218733602, 0.940147345662686, 0.7653676163608506,
            0.5696870117006351],
    ),
    "tied/breslow": dict(
        beta=[0.9815587920956778, 0.2332596203472142, -0.5848285554571279],
        se=[0.17290967080069733, 0.07499318138590112, 0.08689801710448683],
        neg_ll=703.2232584419438,
        H=[0.004078248328339511, 0.06232286179271347, 0.26345604806997835,
           0.5506710227079177],
        sf=[0.9959300564329123, 0.9395794834986199, 0.7683913939818383,
            0.5765627938193649],
    ),
    "untied/efron": dict(
        beta=[1.035421878334804, 0.25669222820891074, -0.614538625043452],
        se=[0.17341683813905015, 0.07559866171577093, 0.08725672610664677],
        neg_ll=694.2272015607675,
        H=[0.0038770755896804874, 0.06369154171113005, 0.2693428761538631,
           0.5672885996495851],
        sf=[0.9961304305641083, 0.9382943795763786, 0.7638812940604591,
            0.5670608852366575],
    ),
    "untied/breslow": dict(
        beta=[1.035421878334804, 0.25669222820891074, -0.614538625043452],
        se=[0.17341683813905015, 0.07559866171577093, 0.08725672610664678],
        neg_ll=694.2272015607676,
        H=[0.0038770755896804874, 0.06369154171113005, 0.2693428761538631,
           0.5672885996495851],
        sf=[0.9961304305641083, 0.9382943795763786, 0.7638812940604591,
            0.5670608852366575],
    ),
    "heavy_ties/efron": dict(
        beta=[0.9120878409001267, 0.24080073315712802, -0.5224051198275868],
        se=[0.1077074497914273, 0.04636948852328286, 0.052155599231892384],
        neg_ll=2200.3207488907397,
        H=[0.0, 0.06702945872889865, 0.2071030487469608, 0.4775998320373384],
        sf=[1.0, 0.9351676520806581, 0.8129358736479828, 0.6202703596514285],
    ),
    "heavy_ties/breslow": dict(
        beta=[0.8212453252370068, 0.20477053700252695, -0.46357711352897485],
        se=[0.10691683073769187, 0.04594673025403154, 0.05154128799935047],
        neg_ll=2256.0527866078087,
        H=[0.0, 0.06797418165749071, 0.1951864542077767, 0.4424488275334756],
        sf=[1.0, 0.9342845949454988, 0.8226812513738244, 0.642461216452044],
    ),
    "truncated_weighted/efron": dict(
        beta=[0.8255154148519428, 0.22422626798251044, -0.5022857870430274],
        se=[0.10868012082128257, 0.04606817453820939, 0.05386618906944244],
        neg_ll=2135.9459249207803,
        H=[0.005074981707684602, 0.09707868479987132, 0.3397964619103736,
           0.6769975656840892],
        sf=[0.9949378742548627, 0.9074845980799591, 0.7119152098788808,
            0.5081403623733028],
    ),
    "truncated_weighted/breslow": dict(
        beta=[0.7875026341632173, 0.2075337553256086, -0.47981348812570473],
        se=[0.10832133215563337, 0.0458880245127478, 0.05358125962131608],
        neg_ll=2157.8758987646565,
        H=[0.005259295621056187, 0.09757112491623812, 0.3332610857003709,
           0.6594435832951988],
        sf=[0.994754510260484, 0.9070378262720884, 0.7165830801350133,
            0.5171389992314207],
    ),
    "tied/efron/center": dict(
        beta=[1.0168233666516833, 0.24658617107409916, -0.6068609530068154],
        se=[0.1733086656829556, 0.07534974947332118, 0.08727644625572786],
        neg_ll=696.9889434316875,
        H=[0.003933102644420112, 0.061718665297210415, 0.2673990163174693,
           0.5626681712449955],
        sf=[0.9960746218733602, 0.940147345662686, 0.7653676163608506,
            0.5696870117006351],
    ),
    "weighted/efron/strata": dict(
        beta=[1.007559604751802, 0.23643083548053442, -0.5890768870105988],
        se=[0.11218326392999134, 0.04619859300989802, 0.05391578334572443],
        neg_ll=1713.1830631990952,
        H=[0.0, 0.0437017738542976, 0.22912197185885014, 0.45624210318206615],
        sf=[1.0, 0.9572393887228138, 0.79523153172133, 0.6336604073254175],
    ),
    "truncated/breslow/strata": dict(
        beta=[0.8957245512874277, 0.16059961190889607, -0.5589966782690301],
        se=[0.17743742432760634, 0.07574628636740607, 0.0889529115375211],
        neg_ll=502.39285110398373,
        H=[0.0, 0.06743510939472301, 0.36325283827586136, 0.8251486188150511],
        sf=[1.0, 0.9347883776316014, 0.6954105848561073, 0.43816986733918456],
    ),
    "small/efron": dict(
        beta=[0.2580213515728971, 0.982979099775542],
        se=[1.1097249628340748, 0.7517898247072821],
        neg_ll=10.423905616708964,
        H=[0.0, 0.11401904371818103, 0.3781424887272776, 2.22943611319827],
        sf=[1.0, 0.8922409641351452, 0.6851328699997384, 0.10758908109689042],
    ),
}
OLD_TVC = {
    "efron": dict(
        beta=[-0.0089484087893837, 0.12048839718896766],
        se=[0.11687328139390654, 0.09712827485140682],
        H0=[0.04761264400904183, 0.3350344498439049, 1.0933009682031525],
    ),
    "breslow": dict(
        beta=[-0.011106640430276243, 0.11682505160703706],
        se=[0.11679955182830415, 0.09716291923125654],
        H0=[0.04759625023010446, 0.3327551161511443, 1.084470712385075],
    ),
}
OLD_CRPH_BETAS = [
    [1.0787801506152894, 0.1664747498153155, -0.6621703848756292],
    [0.9110818603000075, 0.323685218291951, -0.5331761623188014],
]
OLD_CRPH_CIF = [0.0, 0.02491017642340669, 0.11237220982678978,
                0.202618857040061]
# fmt: on

# root(hybr) stopped at a relative change in beta of 1e-10, short of the
# maximum by up to 3e-12 standard errors; Newton-Raphson reaches it to
# rounding. The differences are that shortfall: at most 1.4e-13 relative
# (in beta) on these data, against the 1e-10 the old tolerance allowed.
RTOL = 1e-10


@pytest.mark.parametrize("case", sorted(FIT_CASES))
def test_fits_match_the_code_before_516(case):
    model, H, sf = _fit(case)
    old = OLD_FITS[case]
    np.testing.assert_allclose(model.beta, old["beta"], rtol=RTOL)
    np.testing.assert_allclose(model.standard_errors(), old["se"], rtol=RTOL)
    np.testing.assert_allclose(model.neg_ll(), old["neg_ll"], rtol=1e-13)
    np.testing.assert_allclose(H, old["H"], rtol=RTOL)
    np.testing.assert_allclose(sf, old["sf"], rtol=RTOL)


@pytest.mark.parametrize("method", ["efron", "breslow"])
def test_tvc_fits_match_the_code_before_516(method):
    ids, xl, xr, cc, Z = _tvc_data()
    model = CoxPH.fit_tvc(ids, xl, xr, cc, Z, tie_method=method)
    old = OLD_TVC[method]
    np.testing.assert_allclose(model.beta, old["beta"], rtol=RTOL)
    np.testing.assert_allclose(model.standard_errors(), old["se"], rtol=RTOL)
    np.testing.assert_allclose(model.H0[[20, 40, 80]], old["H0"], rtol=RTOL)


def test_competing_risks_cox_matches_the_code_before_516():
    model = _competing_risks()
    np.testing.assert_allclose(model.betas, OLD_CRPH_BETAS, rtol=RTOL)
    np.testing.assert_allclose(
        model.cif(T, np.full(3, 0.2), event=1.0), OLD_CRPH_CIF, rtol=RTOL
    )


def _neg_ll_by_definition(x, Z, c, n, tl, beta, efron):
    """The negative partial log-likelihood straight from its definition,
    each risk set's sum taken by ``logsumexp`` (so it holds at any
    ``beta``): ``R - k D`` is the survivors' weights plus ``(1 - k)``
    times the deaths'."""
    from scipy.special import logsumexp

    eta = Z @ beta
    ll = 0.0
    for tau in np.unique(x[c == 0]):
        risk = (tl < tau) & (x >= tau)
        dead = (x == tau) & (c == 0)
        d = n[dead].sum()
        ll += n[dead] @ eta[dead]
        terms = [(j / d, 1.0) for j in range(int(d))] if efron else [(0, d)]
        for k, mult in terms:
            b = np.where(dead[risk], (1 - k) * n[risk], n[risk])
            ll -= mult * logsumexp(eta[risk], b=b)
    return -ll


@pytest.mark.parametrize("kind", ["tied", "heavy_ties", "truncated_weighted"])
@pytest.mark.parametrize("method", ["efron", "breslow"])
def test_neg_ll_of_far_out_is_quiet_and_right(kind, method):
    # At a large beta it gave inf or nan, with numpy's overflow, invalid
    # and divide warnings (#728); it is the partial likelihood, in logs.
    x, Z, c, n, tl, _ = _data(kind)
    x, c, n, tl, Z = validate_coxph(x, c, n, Z, tl, method)
    generator = {
        "efron": CoxPH.create_efron_ll_jac_hess,
        "breslow": CoxPH.create_breslow_ll_jac_hess,
    }[method]
    neg_ll, _ = generator(x, Z, c, n, tl)
    rng = np.random.default_rng(728)
    for scale in [1e3, 1e5]:
        beta = scale * rng.normal(size=Z.shape[1])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = neg_ll(beta)
        want = _neg_ll_by_definition(x, Z, c, n, tl, beta, method == "efron")
        np.testing.assert_allclose(got, want, rtol=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(neg_ll(np.full(Z.shape[1], np.inf)))


def test_neg_ll_of_keeps_a_risk_set_far_below_the_entries_to_come():
    # A unit yet to enter with e^18 times the weight of the one at risk:
    # the risk set's sum, the difference of two sums of 7e7, was one
    # rounding step, and the likelihood 1.446 for 1.386 (#728)
    x = np.array([0.5, 1.0, 0.5, 2.0])
    Z = np.array([[-1.0, 0.5], [-0.5, 0.5], [1.0, -0.5], [0.5, -0.5]])
    c = np.zeros(4, int)
    n = np.ones(4)
    tl = np.array([-np.inf, 0.0, -np.inf, 1.5])
    beta = np.array([-36.16, -72.33])
    neg_ll, _ = CoxPH.create_breslow_ll_jac_hess(x, Z, c, n, tl)
    want = _neg_ll_by_definition(x, Z, c, n, tl, beta, False)
    np.testing.assert_allclose(neg_ll(beta), want, rtol=1e-12)


@pytest.mark.parametrize("method", ["efron", "breslow"])
def test_score_and_information_keep_a_risk_set_below_the_entries(method):
    # The score and information subtracted the mass yet to enter as the
    # likelihood did: in the case above the information was 2.00095 for
    # 1.99995 (and 0.50099 for 0.49999), and where the risk set's sum
    # rounded to 0 the score was -inf and the information nan. Such a risk
    # set is summed over itself (#746).
    generator = getattr(CoxPH, f"create_{method}_ll_jac_hess")
    cases = [
        (
            np.array([0.5, 1.0, 0.5, 2.0]),
            np.array([[-1.0, 0.5], [-0.5, 0.5], [1.0, -0.5], [0.5, -0.5]]),
            np.array([-np.inf, 0.0, -np.inf, 1.5]),
            np.array([-36.16, -72.33]),
        ),
        (
            np.array([1.0, 2.0, 3.0]),
            np.array([[-1.0], [0.0], [-1.0]]),
            np.array([-np.inf, 1.5, -np.inf]),
            np.array([40.0]),
        ),
    ]
    for x, Z, tl, beta in cases:
        c, n = np.zeros(x.size, int), np.ones(x.size)
        _, jac_hess = generator(x, Z, c, n, tl)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            score, info = jac_hess(beta)
        want_score, want_info = _score_information_far(
            x, Z, c, n, tl, beta, method == "efron"
        )
        np.testing.assert_allclose(score, want_score, rtol=1e-9, atol=1e-13)
        np.testing.assert_allclose(info, want_info, rtol=1e-9, atol=1e-13)


def test_a_row_sum_lost_to_the_times_before_entry_is_refused():
    # The information's row weights sum s_u over each row's times at risk
    # as a difference of cumulative sums; where the times before a row
    # entered outweigh its own by more than 1e6 that is rounding, and the
    # direct information gives way to the one in logs (#746)
    from surpyval.univariate.regression.proportional_hazards import (
        cox_likelihood as cl,
    )

    rows = cl._RiskSetRows(
        np.array([1.0, 2.0, 2.0]), np.array([1.0, 2.0]), np.array([0, 1.5, 0])
    )
    np.testing.assert_allclose(
        rows.over_risk_set_kept(np.array([2.0, 1.0])), [2.0, 1.0, 3.0]
    )
    assert rows.over_risk_set_kept(np.array([1e20, 1.0])) is None
    # A row at risk only where the values are 0 has nothing to lose
    np.testing.assert_allclose(
        rows.over_risk_set_kept(np.array([1e20, 0.0])), [1e20, 0.0, 1e20]
    )


def _tie_neg_ll_by_definition(x, Z, c, beta, method):
    """The exact (sum over the deaths' orderings) and Kalbfleisch-Prentice
    (sum over the d-subsets of the risk set) negative partial
    log-likelihoods by enumeration, in logs."""
    from itertools import combinations, permutations

    from scipy.special import logsumexp

    eta = Z @ beta
    ll = 0.0
    for tau in np.unique(x[c == 0]):
        risk = np.flatnonzero(x >= tau)
        dead = np.flatnonzero((x == tau) & (c == 0))
        ll += eta[dead].sum()
        if method == "kp":
            ll -= logsumexp(
                [eta[list(s)].sum() for s in combinations(risk, len(dead))]
            )
            continue
        survivors = np.setdiff1d(risk, dead)
        log_w = logsumexp(eta[survivors]) if survivors.size else -np.inf
        orders = [
            -sum(
                np.logaddexp(log_w, logsumexp(eta[list(o[k:])]))
                for k in range(len(o))
            )
            for o in permutations(dead)
        ]
        ll += logsumexp(orders)
    return -ll


@pytest.mark.parametrize("method", ["exact", "kp"])
def test_tie_methods_neg_ll_far_out_is_quiet_and_right(method):
    # The exact term raised IndexError at a large beta, and KP gave -inf
    # with a divide warning (#728)
    x = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0])
    Z = np.array([[0.3], [-1.2], [0.8], [1.5], [-0.4], [0.1], [2.0], [-1.0]])
    c = np.array([0, 0, 0, 0, 0, 0, 1, 0])
    generator = {
        "exact": CoxPH.create_exact_ll_jac_hess,
        "kp": CoxPH.create_kalbfleisch_prentice_ll_jac_hess,
    }[method]
    neg_ll, _ = generator(x, Z, c, np.ones(8), np.full(8, -np.inf))
    for b in [0.5, 1e3, -1e3, 1e5]:
        beta = np.array([b])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = neg_ll(beta)
        want = _tie_neg_ll_by_definition(x, Z, c, beta, method)
        np.testing.assert_allclose(got, want, rtol=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(neg_ll(np.array([1e308])))


def _score_information_far(x, Z, c, n, tl, beta, efron):
    """The score and information from their definition, each risk set's
    weights scaled by their largest, so that it holds at any ``beta``: per
    tie term the weighted mean and covariance of ``Z`` in the risk set,
    the deaths' weights times ``1 - j / d`` for Efron."""
    eta = Z @ beta
    p = Z.shape[1]
    score, info = np.zeros(p), np.zeros((p, p))
    for tau in np.unique(x[c == 0]):
        risk = (tl < tau) & (x >= tau)
        dead = (x == tau) & (c == 0)
        d = n[dead].sum()
        score -= n[dead] @ Z[dead]
        terms = [(j / d, 1.0) for j in range(int(d))] if efron else [(0, d)]
        for k, mult in terms:
            b = np.where(dead[risk], (1 - k) * n[risk], n[risk])
            w = b * np.exp(eta[risk] - eta[risk].max())
            w = w / w.sum()
            mean = w @ Z[risk]
            centred = Z[risk] - mean
            score += mult * mean
            info += mult * (centred.T * w) @ centred
    return score, info


@pytest.mark.parametrize("kind", ["tied", "heavy_ties", "truncated_weighted"])
@pytest.mark.parametrize("method", ["efron", "breslow"])
def test_score_and_information_far_out_are_quiet_and_right(kind, method):
    # At a large beta R^2 overflowed and the information's second term,
    # ZR ZR' / R^2, went to 0: the information, about 0 there, looked
    # healthy (or exp overflowed, with numpy's warnings); each risk set's
    # sums are now scaled by its own total (#746)
    x, Z, c, n, tl, _ = _data(kind)
    x, c, n, tl, Z = validate_coxph(x, c, n, Z, tl, method)
    generator = {
        "efron": CoxPH.create_efron_ll_jac_hess,
        "breslow": CoxPH.create_breslow_ll_jac_hess,
    }[method]
    _, jac_hess = generator(x, Z, c, n, tl)
    rng = np.random.default_rng(746)
    for scale in [20.0, 1e3, 1e5]:
        beta = scale * rng.normal(size=Z.shape[1])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            score, info = jac_hess(beta)
        want_score, want_info = _score_information_far(
            x, Z, c, n, tl, beta, method == "efron"
        )
        # The linear predictor itself is rounded by |eta| eps
        atol = 1e-14 * np.abs(Z @ beta).max() * n.sum()
        np.testing.assert_allclose(score, want_score, rtol=0, atol=atol)
        np.testing.assert_allclose(info, want_info, rtol=0, atol=atol)


@pytest.mark.parametrize("method", ["efron", "breslow"])
def test_information_far_out_does_not_look_healthy(method):
    # Every death but the last has the largest Z of its risk set: at beta
    # = 150 the information is e^-225 of the deaths' spread, and it was 9
    # (R^2 overflowed and ZR ZR' / R^2 was taken as 0, #746)
    x = np.arange(1.0, 7.0)
    Z = np.array([[3.0], [2.0], [1.0], [0.0], [-1.0], [0.5]])
    generator = getattr(CoxPH, f"create_{method}_ll_jac_hess")
    _, jac_hess = generator(x, Z, np.zeros(6), np.ones(6), np.full(6, -np.inf))
    score, info = jac_hess(np.array([150.0]))
    np.testing.assert_allclose(score, [2.0], rtol=1e-12)
    assert abs(info[0, 0]) < 1e-12


def _kp_score_information(x, Z, c, beta):
    """The Kalbfleisch-Prentice score and information by enumeration: the
    mean and covariance of the covariate sum of the d-subsets of each risk
    set, weighted by their scores, scaled by the largest."""
    from itertools import combinations

    eta = Z @ beta
    p = Z.shape[1]
    score, info = np.zeros(p), np.zeros((p, p))
    for tau in np.unique(x[c == 0]):
        risk = np.flatnonzero(x >= tau)
        dead = (x == tau) & (c == 0)
        score -= Z[dead].sum(axis=0)
        subsets = [list(s) for s in combinations(risk, int(dead.sum()))]
        S = np.array([Z[s].sum(axis=0) for s in subsets])
        log_w = np.array([eta[s].sum() for s in subsets])
        w = np.exp(log_w - log_w.max())
        w = w / w.sum()
        mean = w @ S
        centred = S - mean
        score += mean
        info += (centred.T * w) @ centred
    return score, info


@pytest.mark.parametrize("method", ["exact", "kp"])
def test_tie_methods_score_and_information_far_out(method):
    # KP's score and information were nan at a large beta: every product
    # of d scores underflowed on one scale; each row of the recursion now
    # has its own (#746). The exact term's held already.
    x = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0])
    z = np.array([0.3, -1.2, 0.8, 1.5, -0.4, 0.1, 2.0, -1.0])
    Z = np.column_stack([z, np.cos(np.arange(8.0))])
    c = np.array([0, 0, 0, 0, 0, 0, 1, 0])
    generator = {
        "exact": CoxPH.create_exact_ll_jac_hess,
        "kp": CoxPH.create_kalbfleisch_prentice_ll_jac_hess,
    }[method]
    neg_ll, jac_hess = generator(x, Z, c, np.ones(8), np.full(8, -np.inf))
    for b in [0.5, 1e3, -1e3, 1e5]:
        beta = np.array([b, -0.7 * b])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            score, info = jac_hess(beta)
        if method == "kp":
            want_score, want_info = _kp_score_information(x, Z, c, beta)
        else:
            # Central differences of the likelihood, and of the score
            steps = 1e-3 * max(abs(b), 1.0) * np.eye(2)
            want_score = [
                (neg_ll(beta + h) - neg_ll(beta - h)) / (2 * h.sum())
                for h in steps
            ]
            want_info = [
                (jac_hess(beta + h)[0] - jac_hess(beta - h)[0]) / (2 * h.sum())
                for h in steps
            ]
        np.testing.assert_allclose(score, want_score, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(info, want_info, rtol=1e-6, atol=1e-9)
