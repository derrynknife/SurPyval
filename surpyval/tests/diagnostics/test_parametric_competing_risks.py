"""The parametric competing-risks family of ``diagnose()``: each cause's
Cox-Snell goodness of fit (the univariate family's test on the cause's
model), the alternatives, the incidences against the Aalen-Johansen
estimate, the advice, the refusals, the plots and the report's round
trip.

The Aalen-Johansen side is ``CompetingRisks.fit``, checked against R's
``cmprsk::cuminc`` in ``tests/reference/test_competing_risks.py``. The
size and power of the tests are studied in
``tests/calibration/test_size_parametric_competing_risks_diagnostics.py``.
"""

import json
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.diagnostics import Diagnosis
from surpyval.diagnostics.competing_risks import (
    _cif_steps,
    incidence_bootstrap,
    incidence_distance,
    pcr_data,
)
from surpyval.diagnostics.univariate import bootstrap
from surpyval.univariate.competing_risks import ParametricCompetingRisks

PCR = ParametricCompetingRisks


def _data(seed, n=200, wrong=False):
    """Cause ``a`` a Weibull (10, 2); ``b`` a LogNormal (2.5, 0.6), or with
    ``wrong`` 4 plus an exponential of mean 6 (no failures before 4);
    uniform censoring on (0, 25)."""
    rng = np.random.default_rng(seed)
    t_a = 10 * rng.weibull(2.0, n)
    t_b = 4 + 6 * rng.weibull(1.0, n) if wrong else rng.lognormal(2.5, 0.6, n)
    cens = rng.uniform(0, 25, n)
    first = np.minimum(t_a, t_b)
    e = np.where(cens < first, None, np.where(t_a < t_b, "a", "b"))
    return np.minimum(first, cens), e.astype(object)


def _fit(x, e, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PCR.fit(x, e, **kwargs)


@pytest.fixture(scope="module")
def right():
    x, e = _data(1)
    return _fit(x, e, dist={"a": sp.Weibull, "b": sp.LogNormal})


@pytest.fixture(scope="module")
def right_report(right):
    return right.diagnose(n_boot=49, random_state=0)


def test_a_right_model_passes(right_report):
    d = right_report
    assert isinstance(d, Diagnosis)
    assert d.family == "Competing risks (parametric)"
    assert set(d.table["check"]) == {
        "goodness_of_fit",
        "alternatives",
        "incidence",
    }
    assert set(d.table["cause"]) == {"a", "b"}
    tested = d.table[d.table["check"] != "alternatives"]
    assert (tested["verdict"] != "fail").all(), tested
    assert any("Tsiatis" in note for note in d.notes)
    assert any("Cox-Snell" in note for note in d.notes)


def test_goodness_of_fit_is_the_univariate_test_of_each_cause(right):
    # The cause's rows are the univariate bootstrap of the cause's own fit
    # (its Cox-Snell residuals through u = 1 - exp(-r)), with the cause's
    # own seed from the report's.
    d = right.diagnose(n_boot=19, random_state=5)
    seeds = np.random.default_rng(5).integers(0, 2**63 - 1, size=3)
    for cause, seed in zip(right.causes, seeds):
        boot = bootstrap(right.models[cause], 19, int(seed))
        rows = d.table[
            (d.table["check"] == "goodness_of_fit")
            & (d.table["cause"] == cause)
        ]
        assert_allclose(rows["p"], [boot["p"][k] for k in ("AD", "KS", "CvM")])
        assert_allclose(
            rows["statistic"],
            [boot["observed"][k] for k in ("AD", "KS", "CvM")],
        )


def test_the_wrong_cause_is_flagged():
    x, e = _data(2890, n=300, wrong=True)
    model = _fit(x, e, dist=sp.Weibull)
    d = model.diagnose(n_boot=49, random_state=1)
    gof = d.table[d.table["check"] == "goodness_of_fit"]
    ad = gof[gof["test"].str.startswith("Anderson")]
    assert ad.set_index("cause").loc["b", "verdict"] == "fail"
    assert ad.set_index("cause").loc["a", "verdict"] != "fail"
    incidence = d.table[d.table["check"] == "incidence"].set_index("cause")
    assert incidence.loc["b", "verdict"] == "fail"
    assert not d.ok
    text = " ".join(str(a) for a in d.advice)
    assert "cause 'b'" in text
    assert "dist={'b': surpyval." in text or "from_fitted" in text


def test_cif_steps_agree_with_cif():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lfp = sp.Weibull.fit(
            sp.Weibull.random(80, 5, 2, random_state=1),
            c=np.r_[np.zeros(40), np.ones(40)].astype(int),
            lfp=True,
        )
        offset = sp.Weibull.fit(
            3 + sp.Weibull.random(80, 5, 1.5, random_state=2), offset=True
        )
    cases = [
        {
            0: sp.Weibull.from_params([10, 0.5]),
            1: sp.Weibull.from_params([8, 3]),
        },
        {
            0: sp.LogNormal.from_params([2, 1.5]),
            1: sp.Exponential.from_params([0.05]),
            2: sp.Gamma.from_params([2, 0.3]),
        },
        {0: lfp, 1: offset},
    ]
    for models in cases:
        model = PCR.from_fitted(models)
        for size in (60, 5, 1):
            t = np.sort(np.random.default_rng(0).uniform(0.01, 40, size))
            for k in model.causes:
                assert_allclose(
                    _cif_steps(model, t, k), model.cif(t, k), atol=1e-7
                )


def test_incidence_distance_is_the_largest_gap(right):
    data = pcr_data(right)
    out = incidence_distance(right, data["x"], data["e"], data["c"], data["n"])
    aj = sp.CompetingRisks.fit(data["x"], data["e"], c=data["c"])
    t = np.linspace(0, data["x"].max(), 4001)
    for k in right.causes:
        dense = np.max(np.abs(right.cif(t, k) - aj.cif(t, k)))
        # The supremum over a dense grid can only approach it from below.
        assert dense <= out[k]["D"] + 1e-7
        assert dense >= out[k]["D"] - 0.01


def test_pcr_data_rebuilds_the_data():
    x = np.array([1.0, 2, 2, 2, 3, 4, 4, 5, 6])
    e = np.array(["a", "a", "b", None, "b", "a", "a", None, "c"], object)
    n = np.array([1, 2, 1, 1, 3, 1, 1, 2, 1])
    model = _fit(x, e, n=n, dist=sp.Exponential)
    data = pcr_data(model)

    def units(xs, es, ns):
        out = {}
        for key, count in zip(zip(xs, [str(v) for v in es]), ns):
            out[key] = out.get(key, 0.0) + float(count)
        return out

    got = units(data["x"], data["e"], data["n"])
    want = units(x, e, n)
    assert (data["c"] == (data["e"] == None)).all()  # noqa: E711
    assert got == want


def test_incidence_bootstrap_is_seeded(right):
    one = incidence_bootstrap(right, 9, random_state=3)
    two = incidence_bootstrap(right, 9, random_state=3)
    assert_allclose(one["draws"], two["draws"])
    assert one["draws"].shape == (9, 2)
    assert set(one["p"]) == {"a", "b"}


def test_delayed_entry_checks_each_cause_and_the_incidence_from_entry():
    # Null data (each cause fitted with its true distribution) with delayed
    # entry: the incidence is compared with the Aalen-Johansen estimate
    # that takes the entry times, and nothing fails.
    x, e = _data(3, n=150)
    tl = np.where(np.arange(x.size) % 3 == 0, 0.3 * x, 0.0)
    model = _fit(x, e, tl=tl, dist={"a": sp.Weibull, "b": sp.LogNormal})
    d = model.diagnose(n_boot=19, random_state=0)
    gof = d.table[d.table["check"] == "goodness_of_fit"]
    assert np.isfinite(gof["p"]).all()
    incidence = d.table[d.table["check"] == "incidence"]
    assert np.isfinite(incidence["p"]).all()
    assert (incidence["verdict"] != "fail").all(), incidence
    assert any("delayed entry" in note for note in d.notes)
    assert not any("not computed" in note for note in d.notes)
    # The observed gaps are those against the estimate with entry
    data = pcr_data(model)
    out = incidence_distance(
        model, data["x"], data["e"], data["c"], data["n"], tl=data["t"][:, 0]
    )
    rows = incidence.set_index("cause")
    for k in model.causes:
        assert rows.loc[k, "statistic"] == out[k]["D"]


def test_incidence_with_late_entry_is_conditional_on_the_first_entry():
    # Every unit enters at 2 or later, so the Aalen-Johansen estimate is
    # conditional on surviving to 2. For exponential causes (rates 0.1 and
    # 0.05) the conditional incidence is in closed form, (l_k / l) (1 -
    # exp(-l (t - 2))), and the gap is taken against it.
    model = PCR.from_fitted(
        {
            "a": sp.Exponential.from_params([0.1]),
            "b": sp.Exponential.from_params([0.05]),
        }
    )
    x = np.array([3.0, 4, 5, 6, 8, 9, 11, 12])
    e = np.array(["a", "b", "a", None, "a", "b", None, "a"], dtype=object)
    tl = np.array([2.0, 2, 2.5, 3, 2, 4, 2, 2])
    out = incidence_distance(model, x, e, tl=tl)
    aj = sp.CompetingRisks.fit(x, e, tl=tl)
    times = np.unique(np.r_[x[e != None], x.max()])  # noqa: E711
    for k, rate in (("a", 0.1), ("b", 0.05)):
        exact = rate / 0.15 * (1 - np.exp(-0.15 * (times - 2.0)))
        step = aj.cif(times, k)
        before = np.r_[0.0, step[:-1]]
        gaps = np.maximum(np.abs(exact - step), np.abs(exact - before))
        assert_allclose(out[k]["D"], gaps.max(), atol=1e-7)
        assert out[k]["time"] == times[np.argmax(gaps)]
    # Without the entry times the unconditional incidence is used
    plain = incidence_distance(model, x, e)
    assert plain["a"]["D"] != out["a"]["D"]


def test_delayed_entry_bootstrap_draws_beyond_entry():
    x, e = _data(5, n=120)
    tl = np.where(np.arange(x.size) % 2 == 0, 0.5 * x, 0.0)
    model = _fit(x, e, tl=tl, dist={"a": sp.Weibull, "b": sp.LogNormal})
    one = incidence_bootstrap(model, 9, random_state=3)
    two = incidence_bootstrap(model, 9, random_state=3)
    assert_allclose(one["draws"], two["draws"])
    assert one["failed"] == 0
    assert np.isfinite(one["draws"]).all()


def test_right_truncation_is_not_compared_with_aalen_johansen():
    x, e = _data(6, n=100)
    keep = x < 20
    x, e = x[keep], e[keep]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = {
            k: sp.Weibull.fit(x, c=np.where(e == k, 0, 1), tr=20.0)
            for k in ("a", "b")
        }
    model = PCR.from_fitted(fits)
    d = model.diagnose(n_boot=9, random_state=0)
    incidence = d.table[d.table["check"] == "incidence"]
    assert incidence["p"].isna().all()
    assert incidence["detail"].str.contains("right truncation").all()
    with pytest.raises(ValueError, match="right truncation"):
        incidence_bootstrap(model, 9)


def test_a_joint_fit_refuses_with_its_own_message():
    # Left- or interval-censored rows: the causes are fitted jointly and
    # keep no data; diagnose() says so, and what to use instead.
    x = [[2, 2], [2, 4], [4, 6], [5, 5], [6, 8], [7, 7], [8, 10], [9, 9]]
    x += [[10, 10], [12, 12]]
    e = ["a", "b", "a", "b", "a", "a", "b", None, "a", None]
    c = [-1, 2, 2, 0, 2, 0, 2, 1, 0, 1]
    joint = _fit(x, e, c=c, dist=sp.Exponential)
    with pytest.raises(NotImplementedError) as info:
        joint.diagnose()
    text = str(info.value)
    assert "left- or interval-censored rows (c = -1 or 2)" in text
    assert "does not yet check" in text
    assert "Turnbull.fit" in text and "aic()" in text
    assert "from_dict" not in text
    # Restored, the same model gets the restored model's message
    with pytest.raises(ValueError, match="from_dict") as info:
        sp.from_dict(joint.to_dict()).diagnose()
    assert "interval" not in str(info.value)


def test_restored_and_assembled_models_refuse(right):
    with pytest.raises(ValueError, match="from_dict") as info:
        sp.from_dict(right.to_dict()).diagnose()
    assert "interval" not in str(info.value)
    # Causes fitted to different data
    x, e = _data(4, n=80)
    other = _fit(x, e)
    mixed = PCR.from_fitted({"a": right.models["a"], "b": other.models["b"]})
    with pytest.raises(ValueError, match="different data") as info:
        mixed.diagnose()
    assert "models[cause].diagnose()" in str(info.value)
    # A cause's model that is not a fitted distribution
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c_a = np.where(right.models["a"].data["c"] == 0, 0, 1)
        mix = sp.MixtureModel.fit(
            right.models["a"].data["x"], c=c_a, dist=sp.Weibull, m=2
        )
    assembled = PCR.from_fitted({"a": mix, "b": right.models["b"]})
    with pytest.raises(NotImplementedError, match="Cause 'a': .*Mixture"):
        assembled.diagnose()


def test_plot_draws_by_cause(right_report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = right_report.plot(n_boot=0)
    titles = [ax.get_title() for ax in fig.axes]
    for cause in ("a", "b"):
        assert any(t.startswith(f"cause {cause}: P-P") for t in titles)
        assert any(t.startswith(f"cause {cause}: incidence") for t in titles)
        assert any("probability plot" in t and cause in t for t in titles)
    plt.close(fig)
    fig = right_report.plot("incidence")
    assert len(fig.axes) == 2
    plt.close(fig)


def test_to_dict_round_trips(right_report):
    text = json.dumps(right_report.to_dict(), allow_nan=False)
    back = Diagnosis.from_dict(json.loads(text))
    assert back.to_dict() == right_report.to_dict()
    assert all(isinstance(r["cause"], str) for r in json.loads(text)["table"])
