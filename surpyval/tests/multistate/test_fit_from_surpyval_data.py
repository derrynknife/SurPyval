"""The multi-state fits from a SurpyvalData: ``fit(data)`` is the fit of
the data's own arrays (``x``, ``e``, ``s`` as ``state``, ``tl``, ``i``,
``n``, ``Z`` named by ``covariate_names``), exactly; data without ``s``
starts every row from the one initial state (competing risks); what the
models cannot take is refused with the fix; ``s`` is the other name of
``state`` (and ``s_col`` of ``state_col``)."""

import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import surpyval as sp
from surpyval.datasets import load_mgus2
from surpyval.multistate import (
    MultiState,
    MultiStateProportionalHazards,
    ParametricMultiState,
    ParametricMultiStateRegression,
    TransitionMatrix,
)
from surpyval.univariate.competing_risks import ParametricCompetingRisks

TM = TransitionMatrix.illness_death(("mgus", "pcm", "death"))
TIMES = [12.0, 60.0, 120.0]
# The Cox fits to 250 patients warn of hazards adding up past 1 late on
# for the covariates queried; both fits alike, which is what is compared.
pytestmark = pytest.mark.filterwarnings("ignore:For the covariates")


@pytest.fixture(scope="module")
def frame():
    """The MGUS spells of the first 250 patients, with a count column
    (one patient's spells counted twice) and a second covariate."""
    df = load_mgus2(spells=True)
    df = df[df["id"] <= 250].reset_index(drop=True)
    df["n"] = np.where(df["id"] == 7, 2, 1)
    df["age10"] = df["age"] / 10.0
    return df


def _arrays(df, Z=False):
    out = dict(
        x=df["x"].to_numpy(),
        e=df["e"].to_numpy(dtype=object),
        state=df["state"].to_numpy(dtype=object),
        tl=df["tl"].to_numpy(),
        i=df["id"].tolist(),
        n=df["n"].to_numpy(),
    )
    if Z:
        out["Z"] = df[["male", "age10"]].to_numpy()
    return out


def _data(df, Z=False, **kw):
    a = _arrays(df, Z)
    return sp.SurpyvalData(
        a["x"],
        n=a["n"],
        tl=a["tl"],
        e=a["e"],
        s=a["state"],
        i=a["i"],
        Z=a.get("Z"),
        covariate_names=["male", "age10"] if Z else None,
        **kw,
    )


def _same_model(a, b, states=("mgus", "pcm", "death"), Z=None):
    """Two fits are the same: estimates, standard errors (or covariance),
    occupancy, and the saved model, exactly."""
    for state in states:
        if Z is None:
            pa, pb = a.occupancy(TIMES, state), b.occupancy(TIMES, state)
        else:
            pa = a.occupancy(TIMES, Z, state)
            pb = b.occupancy(TIMES, Z, state)
        assert_array_equal(pa, pb)
    da, db = a.to_dict(), b.to_dict()
    assert da == db


FITS = [
    (MultiState, False, {}),
    (MultiStateProportionalHazards, True, {}),
    (ParametricMultiState, False, {"dist": sp.Exponential}),
    (ParametricMultiStateRegression, True, {"model": sp.ExponentialPH}),
]
IDS = [f[0].__name__ for f in FITS]


def _array_fit(cls, df, Z, opts):
    a = _arrays(df, Z)
    if Z:
        a["Z"] = pd.DataFrame(a["Z"], columns=["male", "age10"])
    return cls.fit(**a, transitions=TM, **opts)


@pytest.mark.parametrize("cls, Z, opts", FITS, ids=IDS)
def test_fit_from_data_is_the_fit_from_arrays(frame, cls, Z, opts):
    ref = _array_fit(cls, frame, Z, opts)
    got = cls.fit(_data(frame, Z), transitions=TM, **opts)
    _same_model(ref, got, Z=[[1, 7.0]] * len(TIMES) if Z else None)
    if hasattr(ref, "params"):
        assert_array_equal(ref.params, got.params)
    if Z:
        assert got.covariate_names == ["male", "age10"]
    if hasattr(ref, "covariance") and not callable(ref.covariance):
        assert_array_equal(ref.covariance, got.covariance)
    if cls is MultiState:
        for variance in ("greenwood", "aalen", "robust"):
            assert_array_equal(
                ref.occupancy_se(TIMES, "pcm", variance=variance),
                got.occupancy_se(TIMES, "pcm", variance=variance),
            )


@pytest.mark.parametrize("cls, Z, opts", FITS, ids=IDS)
def test_unit_and_from_dataframe(frame, cls, Z, opts):
    data = sp.SurpyvalData.from_dataframe(
        frame,
        x_col="x",
        e_col="e",
        s_col="state",
        i_col="id",
        tl_col="tl",
        n_col="n",
        Z_cols=["male", "age10"] if Z else None,
        unit="months",
    )
    got = cls.fit(data, transitions=TM, **opts)
    ref = _array_fit(cls, frame, Z, opts)
    assert got.unit == "month"
    ref.unit = got.unit
    _same_model(ref, got, Z=[[0, 6.5]] * len(TIMES) if Z else None)
    # unit= that agrees is fine; one that does not is refused
    cls.fit(data, transitions=TM, unit="month", **opts)
    with pytest.raises(ValueError, match="in 'month'"):
        cls.fit(data, transitions=TM, unit="day", **opts)


def test_rows_keep_their_order(frame):
    # With s nothing is regrouped or sorted, so the rows line up with
    # the frame's.
    data = _data(frame.iloc[::-1].reset_index(drop=True), True)
    rev = frame.iloc[::-1].reset_index(drop=True)
    assert_array_equal(data.x, rev["x"].to_numpy())
    assert list(data.s) == list(rev["state"])
    assert list(data.e) == [None if v is None else v for v in rev["e"]]
    assert data.ids == rev["id"].tolist()
    assert_array_equal(data.Z, rev[["male", "age10"]].to_numpy())


# -- the other names: s for state, s_col for state_col -----------------------


@pytest.mark.parametrize("cls, Z, opts", FITS, ids=IDS)
def test_s_is_state(frame, cls, Z, opts):
    a = _arrays(frame, Z)
    state = a.pop("state")
    ref = cls.fit(**a, state=state, transitions=TM, **opts)
    got = cls.fit(**a, s=state, transitions=TM, **opts)
    z = [[1, 7.0]] * len(TIMES) if Z else None
    _same_model(ref, got, Z=z)
    # Positional calls are unchanged.
    if Z:
        pos = cls.fit(
            a["x"], a["Z"], a["e"], state, a["tl"], a["i"], a["n"], TM, **opts
        )
    else:
        pos = cls.fit(
            a["x"], a["e"], state, a["tl"], a["i"], a["n"], TM, **opts
        )
    _same_model(ref, pos, Z=z)
    with pytest.raises(TypeError, match="same argument"):
        cls.fit(**a, state=state, s=state, transitions=TM, **opts)
    args = (a["x"], a["Z"], a["e"], state) if Z else (a["x"], a["e"], state)
    with pytest.raises(TypeError, match="same argument"):
        cls.fit(*args, s=state, transitions=TM, **opts)
    # s shows in the signature, a keyword with no default but None.
    import inspect

    p = inspect.signature(cls.fit).parameters["s"]
    assert p.kind is p.KEYWORD_ONLY and p.default is None


@pytest.mark.parametrize("cls, Z, opts", FITS, ids=IDS)
def test_s_col_is_state_col(frame, cls, Z, opts):
    zc = (["male", "age10"],) if Z else ()
    ref = cls.fit_from_df(
        frame, "x", *zc, "e", "state", tl_col="tl", i_col="id",
        n_col="n", transitions=TM, **opts,
    )  # fmt: skip
    got = cls.fit_from_df(
        frame, "x", *zc, e_col="e", s_col="state", tl_col="tl",
        i_col="id", n_col="n", transitions=TM, **opts,
    )  # fmt: skip
    _same_model(ref, got, Z=[[1, 7.0]] * len(TIMES) if Z else None)
    with pytest.raises(TypeError, match="same argument"):
        cls.fit_from_df(
            frame, "x", *zc, "e", "state", s_col="state", tl_col="tl",
            transitions=TM, **opts,
        )  # fmt: skip


# -- data without s: every row starts in the one initial state ---------------


def _competing_risks_data():
    # The array identity's data in test_multistate.py
    rng = np.random.default_rng(3)
    size = 200
    x = np.round(rng.exponential(5, size), 1) + 0.1
    e = rng.choice(np.array(["a", "b", None], dtype=object), size)
    tl = np.round(x * rng.uniform(0, 0.6, size), 1)
    return x, e, tl


def _cr_data(x, e, tl):
    c = np.array([1 if v is None else 0 for v in e])
    return sp.SurpyvalData(x, c, tl=tl, e=e)


def test_without_s_is_one_starting_state_and_competing_risks():
    x, e, tl = _competing_risks_data()
    data = _cr_data(x, e, tl)
    # The rows are as given (e turns sorting off).
    assert_array_equal(data.x, x)
    # No transitions: the starting state is called "event-free", the
    # name TransitionMatrix.competing_risks gives it.
    got = MultiState.fit(data)
    ref = MultiState.fit(x, e, ["event-free"] * x.size, tl=tl)
    assert got.to_dict() == ref.to_dict()
    # Transitions with one state that nothing moves into: that state.
    tm = TransitionMatrix.competing_risks(["a", "b"], initial="alive")
    model = MultiState.fit(data, transitions=tm)
    ref = MultiState.fit(x, e, ["alive"] * x.size, tl=tl, transitions=tm)
    assert model.to_dict() == ref.to_dict()
    # ... which is CompetingRisks' cumulative incidence (the tolerance of
    # test_competing_risks_is_the_cumulative_incidence).
    cr = sp.CompetingRisks.fit(x, e, tl=tl, how="Kaplan-Meier")
    q = np.linspace(0.5, 20, 40)
    for cause in ("a", "b"):
        assert_allclose(
            model.occupancy(q, cause), cr.cif(q, cause), rtol=0, atol=1e-15
        )
    assert_allclose(model.occupancy(q, "alive"), cr.sf(q), atol=1e-15)


def test_without_s_parametric_is_parametric_competing_risks():
    # The data of test_parametric.py's special case, and its tolerances.
    rng = np.random.default_rng(7)
    ta = rng.weibull(1.8, 400) * 10
    tb = rng.weibull(0.8, 400) * 20
    c = rng.uniform(0, 25, 400)
    x = np.minimum.reduce([ta, tb, c])
    e = np.where(x == c, None, np.where(ta < tb, "a", "b"))
    cr = ParametricCompetingRisks.fit(x, e, dist=sp.Weibull)
    tm = TransitionMatrix.competing_risks(["a", "b"], initial="alive")
    ms = ParametricMultiState.fit(
        sp.SurpyvalData(x, (x == c).astype(int), e=e),
        transitions=tm,
        dist=sp.Weibull,
    )
    ref = ParametricMultiState.fit(
        x, e, ["alive"] * x.size, transitions=tm, dist=sp.Weibull
    )
    assert ms.to_dict() == ref.to_dict()
    times = np.array([0.5, 2.0, 5.0, 10.0, 20.0, 50.0])
    for k in ("a", "b"):
        assert_allclose(ms.occupancy(times, k), cr.cif(times, k), atol=1e-8)
    assert_allclose(ms.occupancy(times, "alive"), cr.sf(times), atol=1e-10)


def test_without_s_regression_with_covariates():
    x, e, tl = _competing_risks_data()
    rng = np.random.default_rng(11)
    Z = rng.uniform(size=(x.size, 1))
    c = np.array([1 if v is None else 0 for v in e])
    data = sp.SurpyvalData(x, c, tl=tl, e=e, Z=Z, covariate_names=["z"])
    got = MultiStateProportionalHazards.fit(data)
    ref = MultiStateProportionalHazards.fit(
        x, pd.DataFrame(Z, columns=["z"]), e, ["event-free"] * x.size, tl=tl
    )
    assert got.to_dict() == ref.to_dict()
    assert got.parameter_names == ["z: event-free -> a", "z: event-free -> b"]


def test_without_s_ambiguous_initial_state_is_refused():
    x, e, tl = _competing_risks_data()
    tm = TransitionMatrix.illness_death(recovery=True)
    with pytest.raises(ValueError, match=r"give each row's state.*s=s"):
        MultiState.fit(_cr_data(x, e, tl), transitions=tm)


def test_without_s_masked_causes_and_plain_survival_are_refused():
    x = np.array([1.0, 2, 3, 4])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        masked = sp.SurpyvalData(x, [0, 0, 1, 0], e=["a", None, None, "b"])
    with pytest.raises(ValueError, match=r"Rows 1 failed .*Give it in e"):
        MultiState.fit(masked)
    with pytest.raises(ValueError, match=r"has no e.*KaplanMeier"):
        MultiState.fit(sp.SurpyvalData(x, [0, 1, 0, 1]))


# -- refusals ----------------------------------------------------------------


def _small(**kw):
    base = dict(
        x=[2.0, 6, 3, 8, 4],
        e=["ill", "dead", "ill", None, "dead"],
        s=["healthy", "ill", "healthy", "ill", "healthy"],
        tl=[0.0, 2, 0, 3, 0],
        i=[1, 1, 2, 2, 3],
    )
    base.update(kw)
    return base


@pytest.mark.parametrize("cls, Z, opts", FITS, ids=IDS)
def test_data_arguments_beside_data_are_refused(cls, Z, opts):
    zkw = {"Z": [[0.0], [0], [1], [1], [0]]} if Z else {}
    data = sp.SurpyvalData(**_small(), **zkw)
    for extra in (
        {"state": ["healthy"] * 5},
        {"s": ["healthy"] * 5},
        {"tl": [0.0] * 5},
        {"n": [1] * 5},
        {"e": ["dead"] * 5},
        {"i": [1, 2, 3, 4, 5]},
    ):
        with pytest.raises(TypeError, match="SurpyvalData, which holds"):
            cls.fit(data, **extra, **opts)


@pytest.mark.parametrize("cls, Z, opts", FITS, ids=IDS)
def test_interval_and_left_censored_rows_are_refused(cls, Z, opts):
    zkw = {"Z": [[0.0], [1], [0], [1], [0], [1]]} if Z else {}
    data = sp.SurpyvalData(
        x=[[2.0, 2], [6, 6], [3, 3], [8, 8], [4, 4], [4, 6]],
        c=[0, 0, 0, 1, 0, 2],
        e=["ill", "dead", "ill", None, "dead", "dead"],
        s=["healthy", "ill", "healthy", "ill", "healthy", "healthy"],
        **zkw,
    )
    with pytest.raises(ValueError, match=r"Rows 5 are left or interval"):
        cls.fit(data, **opts)


def test_other_refusals():
    data = sp.SurpyvalData(**_small(), w=[1.0, 1, 2, 2, 1])
    with pytest.raises(ValueError, match="This fit takes no weights"):
        MultiState.fit(data)
    data = sp.SurpyvalData(**_small(), tr=10.0)
    with pytest.raises(ValueError, match="right truncation"):
        MultiState.fit(data)
    data = sp.SurpyvalData(**_small(), Z=[[0.0], [0], [1], [1], [0]])
    with pytest.raises(ValueError, match="MultiStateProportionalHazards"):
        MultiState.fit(data)
    with pytest.raises(ValueError, match="takes no covariates"):
        ParametricMultiState.fit(data)
    with pytest.raises(TypeError, match="needs covariates"):
        MultiStateProportionalHazards.fit(sp.SurpyvalData(**_small()))
    no_e = _small()
    del no_e["e"]
    with pytest.raises(ValueError, match="has no e"):
        MultiState.fit(sp.SurpyvalData(**no_e, c=[0, 0, 0, 1, 0]))
