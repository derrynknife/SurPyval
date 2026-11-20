"""``compare_predictions``: the prequential likelihood, the u-plot and
the y-plot of one-step-ahead predictions.

The scores against sums written out here for a constant rate (whose
predictions are in closed form), the u-values' uniformity for data
simulated from the model being checked, the handling of fits refused
along the way, and the refusals. Seeds were fixed before the results
were seen. The study of how often the true model wins carries the
``calibration`` mark.
"""

import matplotlib
import numpy as np
import pytest
from scipy import stats

matplotlib.use("Agg")

from surpyval.recurrent import (  # noqa: E402
    HPP,
    CrowAMSAA,
    GoelOkumoto,
    InflectionS,
    JelinskiMoranda,
    MusaOkumoto,
    PredictionComparison,
    compare_predictions,
)

#: Failure times of one system: failures that slow down
TIMES = np.cumsum(
    [3, 5, 2, 6, 4, 9, 5, 12, 8, 7, 16, 11, 22, 9, 30, 18, 41, 25, 60, 37.0]
)


def test_constant_rate_predictions_are_the_closed_form():
    # Fitted to the first i failures (a test ended at the i-th), a
    # constant rate is i / t_i, and the next gap is exponential.
    result = compare_predictions(TIMES, HPP, start=5)
    assert isinstance(result, PredictionComparison)
    assert result.names == ["HPP"]
    np.testing.assert_array_equal(result.steps, np.arange(5, 20))
    np.testing.assert_allclose(result.gaps, np.diff(TIMES)[4:])
    rate = result.steps / TIMES[result.steps - 1]
    np.testing.assert_allclose(
        result.u["HPP"], -np.expm1(-rate * result.gaps), rtol=1e-12
    )
    log_density = np.log(rate) - rate * result.gaps
    np.testing.assert_allclose(
        result.log_density["HPP"], log_density, rtol=1e-12
    )
    assert result.prequential_log_likelihood("HPP") == pytest.approx(
        log_density.sum(), rel=1e-12
    )
    assert result.status["HPP"] == ["ok"] * 15
    assert result.predicted("HPP").all() and result.common.all()


def test_plots_in_numbers_are_the_kolmogorov_distances():
    result = compare_predictions(TIMES, HPP, start=5)
    u = np.sort(result.u["HPP"])
    m = u.size
    above = np.max(np.arange(1, m + 1) / m - u)
    below = np.max(u - np.arange(m) / m)
    distance, bias, beyond = result.u_plot("HPP")
    assert distance == pytest.approx(max(above, below), rel=1e-14)
    assert distance == pytest.approx(
        stats.kstest(result.u["HPP"], "uniform").statistic, rel=1e-12
    )
    # the gaps lengthen, so a constant rate expects failures too soon
    assert bias == "pessimistic" and below > above
    assert beyond == (distance > stats.kstwo.ppf(0.95, m))
    # y: the running share of the total of -log(1 - u), less the last
    x = -np.log1p(-result.u["HPP"])
    y = np.cumsum(x)[:-1] / x.sum()
    y_distance, drifting = result.y_plot("HPP")
    assert y_distance == pytest.approx(
        stats.kstest(y, "uniform").statistic, rel=1e-12
    )
    assert drifting == (y_distance > stats.kstwo.ppf(0.95, m - 1))


def test_poisson_process_prediction_is_the_models_own():
    result = compare_predictions(TIMES, CrowAMSAA, start=12)
    for k, i in enumerate(result.steps):
        model = CrowAMSAA.fit(TIMES[:i])
        age, gap = TIMES[i - 1], TIMES[i] - TIMES[i - 1]
        sf = np.exp(-(model.cif(age + gap) - model.cif(age)))
        assert result.u["CrowAMSAA"][k] == pytest.approx(1 - sf, rel=1e-10)
        assert result.log_density["CrowAMSAA"][k] == pytest.approx(
            np.log(model.iif(age + gap)) + np.log(sf), rel=1e-10
        )


def test_likelihood_ratio_and_best():
    result = compare_predictions(TIMES, [HPP, CrowAMSAA], start=8)
    ratio = result.likelihood_ratio("CrowAMSAA", "HPP")
    assert ratio.shape == (12,)
    assert ratio[-1] == pytest.approx(
        result.prequential_log_likelihood("CrowAMSAA")
        - result.prequential_log_likelihood("HPP"),
        rel=1e-12,
    )
    np.testing.assert_allclose(
        result.likelihood_ratio("HPP", "CrowAMSAA"), -ratio, rtol=1e-12
    )
    assert result.best == "CrowAMSAA" and ratio[-1] > 0
    table = result.summary()
    assert list(table.index) == ["HPP", "CrowAMSAA"]
    assert list(table.columns) == [
        "predicted",
        "failed",
        "log_likelihood",
        "u_distance",
        "bias",
        "biased",
        "y_distance",
        "drifting",
    ]
    assert table["predicted"].tolist() == [12, 12]
    text = repr(result)
    assert text.startswith("One-step-ahead predictions of failures 9 to 20")
    assert result.verdict in text
    assert "CrowAMSAA predicted best over the 12 predictions" in text
    with pytest.raises(ValueError, match="not one of the models compared"):
        result.u_plot("GoelOkumoto") if False else result.predicted("Duane")


def test_u_values_are_uniform_for_data_from_the_model_checked():
    # 150 or so failures of a Goel-Okumoto process, predicted by
    # Goel-Okumoto from the 40th on: the u are uniform (Kolmogorov test
    # at the 1% level, set before the run), and the model is not called
    # biased.
    truth = GoelOkumoto.from_params([200.0, 0.02])
    x = truth.time_terminated_simulation_data(80.0, random_state=20261120).x
    result = compare_predictions(x[:-1], GoelOkumoto, start=40)
    # (early on the growth does not show yet, and those fits are
    # refused: the u are of the steps predicted)
    made = result.predicted("GoelOkumoto")
    u = result.u["GoelOkumoto"][made]
    print("predicted", made.sum(), "of", made.size)
    assert made.sum() > 100
    p_value = stats.kstest(u, "uniform").pvalue
    print("failures", x.size - 1, "p-value of uniformity", p_value)
    assert p_value > 0.01
    assert result.u_plot("GoelOkumoto")[2] is False
    assert "consistently too" not in result.verdict


def test_a_refused_fit_is_a_step_with_its_reason():
    # Evenly spread failures and then a slowing: a growth model has no
    # fit until the slowing shows.
    x = np.concatenate(
        [np.arange(1.0, 13.0), 12 + np.cumsum(np.arange(2.0, 14.0))]
    )
    result = compare_predictions(x, [GoelOkumoto, HPP], start=6)
    refused = ~result.predicted("GoelOkumoto")
    assert refused[:6].all() and result.predicted("GoelOkumoto")[-6:].all()
    assert result.predicted("HPP").all()
    statuses = np.array(result.status["GoelOkumoto"])
    assert np.all(statuses[~refused] == "ok")
    assert np.all(
        statuses[refused]
        == "The data show no reliability growth that the Goel-Okumoto "
        "model can fit."
    )
    np.testing.assert_array_equal(result.common, ~refused)
    table = result.summary()
    assert table.loc["GoelOkumoto", "failed"] == refused.sum()
    assert table.loc["GoelOkumoto", "predicted"] == (~refused).sum()
    # the likelihoods are compared on the steps both predicted
    assert result.prequential_log_likelihood("HPP") == pytest.approx(
        result.log_density["HPP"][~refused].sum()
    )
    assert result.prequential_log_likelihood(
        "HPP", common=False
    ) == pytest.approx(result.log_density["HPP"].sum())
    assert (
        "GoelOkumoto gave no prediction at {} of the {} steps".format(
            refused.sum(), refused.size
        )
        in result.verdict
    )
    assert "show no reliability growth" in result.verdict


def test_jelinski_moranda_predicts_with_its_rate_now():
    result = compare_predictions(TIMES, [JelinskiMoranda, HPP], start=10)
    assert result.names == ["JelinskiMoranda", "HPP"]
    made = result.predicted("JelinskiMoranda")
    assert made.any()
    for k in np.flatnonzero(made):
        i = result.steps[k]
        rate = JelinskiMoranda.fit(TIMES[:i]).current_rate()
        gap = result.gaps[k]
        assert result.u["JelinskiMoranda"][k] == pytest.approx(
            -np.expm1(-rate * gap), rel=1e-12
        )
        if rate > 0:
            assert result.log_density["JelinskiMoranda"][k] == pytest.approx(
                np.log(rate) - rate * gap, rel=1e-12
            )
        else:
            # no fault left on that fit: the failure that came had no
            # chance, and the verdict says so
            assert result.log_density["JelinskiMoranda"][k] == -np.inf
    if np.any(np.isneginf(result.log_density["JelinskiMoranda"][made])):
        assert "no chance at all" in result.verdict
        assert result.prequential_log_likelihood("JelinskiMoranda") == -np.inf
        assert result.best == "HPP"


def test_fitted_models_stand_for_their_fitters_and_plots_draw():
    import matplotlib.pyplot as plt

    fitted = [GoelOkumoto.fit(TIMES), MusaOkumoto.fit(TIMES)]
    result = compare_predictions(TIMES, fitted, start=14)
    assert result.names == ["GoelOkumoto", "MusaOkumoto"]
    for kind in ("u", "y", "ratio"):
        fig, ax = plt.subplots()
        assert result.plot(kind, ax=ax) is ax
        assert ax.lines
        plt.close(fig)
    with pytest.raises(ValueError, match="kind"):
        result.plot("z")


def test_refusals():
    with pytest.raises(ValueError, match="in the order they happened"):
        compare_predictions(np.diff(TIMES, prepend=0.0), HPP)
    with pytest.raises(ValueError, match="one system's failure times"):
        compare_predictions(TIMES.reshape(4, 5), HPP)
    with pytest.raises(ValueError, match="no missing values"):
        compare_predictions(np.append(TIMES, np.nan), HPP)
    with pytest.raises(ValueError, match="fitter or a list of fitters"):
        compare_predictions(TIMES, [1.0])
    with pytest.raises(ValueError, match="more than once"):
        compare_predictions(TIMES, [HPP, HPP])
    # two more failures than the largest model has parameters
    with pytest.raises(ValueError, match="at least 5"):
        compare_predictions(TIMES, [HPP, InflectionS], start=4)
    with pytest.raises(ValueError, match="at least 3"):
        compare_predictions(TIMES, HPP, start=2.5)
    with pytest.raises(ValueError) as error:
        compare_predictions(TIMES, HPP, start=20)
    assert "the first prediction is of failure 21, but x has 20" in str(
        error.value
    )
    assert "Give a smaller start (at least 3)" in str(error.value)
    with pytest.raises(ValueError, match="x has 9 failures"):
        compare_predictions(TIMES[:9], HPP)  # the default start is 10
    with pytest.raises(ValueError, match="alpha"):
        compare_predictions(TIMES, HPP, alpha=5)


@pytest.mark.calibration
def test_the_true_model_wins_more_often_than_not():
    # 40 campaigns of a Goel-Okumoto process (a = 60, b = 0.03, to t =
    # 120; seeds 0 to 39, fixed before the run): Goel-Okumoto's
    # prequential likelihood beats Musa-Okumoto's in more than half.
    truth = GoelOkumoto.from_params([60.0, 0.03])
    wins, margins = 0, []
    for seed in range(40):
        x = truth.time_terminated_simulation_data(120.0, random_state=seed).x
        result = compare_predictions(
            x[:-1], [GoelOkumoto, MusaOkumoto], start=15
        )
        margin = result.prequential_log_likelihood(
            "GoelOkumoto"
        ) - result.prequential_log_likelihood("MusaOkumoto")
        margins.append(margin)
        wins += margin > 0
    print(
        "Goel-Okumoto beat Musa-Okumoto in {} of 40 campaigns; median "
        "log ratio {:.3f}".format(wins, float(np.median(margins)))
    )
    assert wins > 20
