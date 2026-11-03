"""The coefficient path of a penalised fit and the penalty chosen by AIC
or BIC: ``penalty_path``.

The references are R's glmnet 5.2 (R 4.6.2, survival 3.8.12, threshold
1e-30) on the Rossi data with Breslow ties, printed by
``scripts/reference/reference_r_penalty_path.R``: glmnet's path of 10
penalties from its ``lambda_max`` down to 0.001 times it
(``nlambda = 10, lambda.min.ratio = 1e-3``), the default path of
``penalty_path``, and its coefficients at each.
"""

import warnings

import numpy as np
import pytest

from surpyval import CoxPH, WeibullAFT, penalty_path
from surpyval.datasets import load_rossi_static
from surpyval.recurrent import ProportionalIntensityHPP
from surpyval.univariate.competing_risks import (
    CompetingRisksProportionalHazards,
)

COLS = ["fin", "age", "race", "wexp", "mar", "paro", "prio"]

# glmnet 5.2: l1_ratio -> its penalties (lambda), largest first
GLMNET_LAMBDA = {
    1.0: [
        0.08840441466,
        0.04103369439,
        0.01904615377,
        0.008840441466,
        0.004103369439,
        0.001904615377,
        0.0008840441466,
        0.0004103369439,
        0.0001904615377,
        8.840441466e-05,
    ],
    0.5: [
        0.1768088293,
        0.08206738878,
        0.03809230754,
        0.01768088293,
        0.008206738878,
        0.003809230754,
        0.001768088293,
        0.0008206738878,
        0.0003809230754,
        0.0001768088293,
    ],
    0.0: [
        88.40441466,
        41.03369439,
        19.04615377,
        8.840441466,
        4.103369439,
        1.904615377,
        0.8840441466,
        0.4103369439,
        0.1904615377,
        0.08840441466,
    ],
}
# glmnet 5.2: l1_ratio -> the coefficients of COLS at each penalty
GLMNET_PATH = {
    1.0: [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [
            -0.04434456007,
            -0.02864789315,
            0.0,
            -0.078737976,
            0.0,
            0.0,
            0.05338941275,
        ],
        [
            -0.2162734352,
            -0.04184815123,
            0.06572076693,
            -0.1262891795,
            -0.2061505428,
            0.0,
            0.07490161517,
        ],
        [
            -0.3012152503,
            -0.04933556376,
            0.1929509415,
            -0.1414366676,
            -0.3229206255,
            -0.01327199638,
            0.08456403256,
        ],
        [
            -0.3427984543,
            -0.05348777017,
            0.2568129973,
            -0.1467175382,
            -0.3796411889,
            -0.05170599758,
            0.0881217653,
        ],
        [
            -0.3621844798,
            -0.05548237092,
            0.2872833453,
            -0.149094131,
            -0.407612382,
            -0.06953863709,
            0.08973414353,
        ],
        [
            -0.3712014371,
            -0.05642316909,
            0.3016153012,
            -0.1501811921,
            -0.4209864067,
            -0.07781458353,
            0.09047440848,
        ],
        [
            -0.3753908268,
            -0.056863128,
            0.3083094188,
            -0.1506823008,
            -0.427282408,
            -0.08165569251,
            0.09081628113,
        ],
        [
            -0.3773362548,
            -0.05706805043,
            0.3114256773,
            -0.1509141491,
            -0.4302242151,
            -0.0834385264,
            0.09097459447,
        ],
        [
            -0.3782394337,
            -0.05716332082,
            0.312874094,
            -0.1510216029,
            -0.4315939183,
            -0.08426603379,
            0.09104799757,
        ],
    ],
    0.5: [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [
            -0.04018436369,
            -0.02375317269,
            0.0,
            -0.09349617068,
            0.0,
            0.0,
            0.04692468785,
        ],
        [
            -0.2026138299,
            -0.03770531448,
            0.05627246094,
            -0.1399302961,
            -0.1966254262,
            0.0,
            0.07062612991,
        ],
        [
            -0.2917187776,
            -0.04670287615,
            0.1834649599,
            -0.1508052365,
            -0.3133927847,
            -0.0120009376,
            0.08229240169,
        ],
        [
            -0.3375955645,
            -0.05204621144,
            0.2509942722,
            -0.1519519917,
            -0.3738065926,
            -0.05032679617,
            0.0870213065,
        ],
        [
            -0.3595913177,
            -0.05475911447,
            0.2842417299,
            -0.1517388245,
            -0.4045344083,
            -0.06872079188,
            0.08921307822,
        ],
        [
            -0.369958688,
            -0.05607500949,
            0.3001260094,
            -0.151457851,
            -0.4194699602,
            -0.07739587153,
            0.09023029089,
        ],
        [
            -0.3748054947,
            -0.05669875902,
            0.3076010201,
            -0.1512857468,
            -0.4265587242,
            -0.08145283413,
            0.09070247959,
        ],
        [
            -0.3770627293,
            -0.0569911521,
            0.3110931324,
            -0.151196617,
            -0.4298839478,
            -0.0833425259,
            0.09092166589,
        ],
        [
            -0.3781120776,
            -0.05712749654,
            0.312718931,
            -0.1511532272,
            -0.4314350305,
            -0.08422107659,
            0.09102340728,
        ],
    ],
    0.0: [
        [
            -9.75080175e-38,
            -1.462382377e-38,
            5.885701726e-38,
            -1.567466058e-37,
            -1.559179497e-37,
            -2.919136448e-38,
            3.03891697e-38,
        ],
        [
            -0.002338079411,
            -0.0003494210289,
            0.001412985059,
            -0.003737989936,
            -0.003718259516,
            -0.0006974941407,
            0.0007280574598,
        ],
        [
            -0.005001782902,
            -0.0007445517483,
            0.003027003472,
            -0.0079467339,
            -0.007905549085,
            -0.001486114234,
            0.001555955254,
        ],
        [
            -0.01061509994,
            -0.001567306481,
            0.006443959163,
            -0.01664367948,
            -0.01656572477,
            -0.003127011617,
            0.003294929746,
        ],
        [
            -0.02215721287,
            -0.003219705777,
            0.01354293215,
            -0.033803717,
            -0.03372193706,
            -0.00641161429,
            0.006844392493,
        ],
        [
            -0.04473747011,
            -0.006320296152,
            0.02775458284,
            -0.06464927319,
            -0.06508083196,
            -0.01249263543,
            0.01367160717,
        ],
        [
            -0.08490565857,
            -0.01151757087,
            0.05425782648,
            -0.111091764,
            -0.1152862851,
            -0.0222355674,
            0.02535687434,
        ],
        [
            -0.1455737558,
            -0.01898775453,
            0.09759157364,
            -0.1624186648,
            -0.1824294133,
            -0.03477486007,
            0.0416753612,
        ],
        [
            -0.2176883511,
            -0.02806533111,
            0.1547129889,
            -0.1962428394,
            -0.2561091765,
            -0.04777702802,
            0.05885979412,
        ],
        [
            -0.2824035744,
            -0.03736282675,
            0.2120340807,
            -0.201311475,
            -0.3224587634,
            -0.05980615814,
            0.07248878753,
        ],
    ],
}


def _rossi():
    df = load_rossi_static()
    return df["week"].to_numpy(float), df[COLS], 1 - df["arrest"].to_numpy()


@pytest.mark.parametrize("l1_ratio", sorted(GLMNET_LAMBDA))
def test_default_path_is_glmnets(l1_ratio):
    x, Z, c = _rossi()
    path = penalty_path(
        CoxPH.fit_penalised,
        x,
        Z,
        c=c,
        l1_ratio=l1_ratio,
        n_penalties=10,
        tie_method="breslow",
    )
    # The largest penalty to six digits, and so the whole grid.
    np.testing.assert_allclose(
        path.penalizers, GLMNET_LAMBDA[l1_ratio], rtol=1e-6
    )
    ref = np.asarray(GLMNET_PATH[l1_ratio])
    got = path.coefficients
    assert all(m.maximum == "verified" for m in path.models)
    if l1_ratio == 0:
        # glmnet reports its first ridge fit as 0 (1e-37): its lambda_max
        # of a ridge path is a convention, and the ridge fit there is not
        # 0, only small. Compared from the second penalty on.
        assert np.all(np.abs(ref[0]) < 1e-30)
        assert np.all(np.abs(got[0]) < 1e-2) and np.all(got[0] != 0)
        ref, got = ref[1:], got[1:]
    else:
        assert np.all(got[0] == 0)
    np.testing.assert_array_equal(got == 0, ref == 0)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-7)


def test_each_fit_is_fit_penalised_at_its_penalty():
    x, Z, c = _rossi()
    path = penalty_path(
        CoxPH.fit_penalised, x, Z, c=c, l1_ratio=0.5, penalizers=[0.01, 0.1]
    )
    np.testing.assert_array_equal(path.penalizers, [0.1, 0.01])
    for lam, coefs, model in zip(
        path.penalizers, path.coefficients, path.models
    ):
        direct = CoxPH.fit_penalised(x, Z, c=c, penalizer=lam, l1_ratio=0.5)
        np.testing.assert_array_equal(coefs, direct.params)
        assert model.aic() == direct.aic()
    assert path.coefficient_names == COLS


def test_criteria_count_the_effective_df_and_choose_the_smallest():
    x, Z, c = _rossi()
    path = penalty_path(CoxPH.fit_penalised, x, Z, c=c, l1_ratio=1)
    np.testing.assert_allclose(
        path.aic, 2 * path.effective_df + 2 * path.neg_ll
    )
    # The lasso's effective df is its number of non-zero coefficients.
    np.testing.assert_allclose(
        path.effective_df, (path.coefficients != 0).sum(axis=1)
    )
    for criterion in ("aic", "bic"):
        i = int(np.argmin(getattr(path, criterion)))
        assert path.best_penalizer(criterion) == path.penalizers[i]
        assert path.best(criterion) is path.models[i]
    table = path.table()
    assert list(table.columns) == [
        "penalizer",
        "effective_df",
        "nonzero",
        "neg_ll",
        "aic",
        "bic",
    ]
    assert "smallest BIC" in repr(path)
    # BIC's log(events) per parameter chooses a sparser fit than AIC's 2.
    assert path.best_penalizer("bic") >= path.best_penalizer("aic")
    with pytest.raises(ValueError, match="criterion"):
        path.best("cv")


def test_penalty_factor_leaves_a_column_unpenalised():
    x, Z, c = _rossi()
    factor = np.ones(len(COLS))
    factor[COLS.index("prio")] = 0
    path = penalty_path(
        CoxPH.fit_penalised,
        x,
        Z,
        c=c,
        l1_ratio=1,
        n_penalties=5,
        penalty_factor=factor,
    )
    # At the largest penalty only the unpenalised column is not 0.
    first = path.coefficients[0]
    assert first[COLS.index("prio")] != 0
    assert np.all(np.delete(first, COLS.index("prio")) == 0)


def test_other_models_take_a_path():
    x, Z, c = _rossi()
    aft = penalty_path(
        WeibullAFT.fit_penalised, x, Z, c=c, l1_ratio=1, n_penalties=4
    )
    assert aft.coefficients.shape == (4, len(COLS))
    assert np.all(aft.coefficients[0] == 0)
    hpp = penalty_path(
        ProportionalIntensityHPP.fit_penalised,
        x,
        Z.to_numpy(float),
        i=np.arange(x.size),
        c=c,
        l1_ratio=1,
        n_penalties=4,
    )
    assert np.all(hpp.coefficients[0] == 0) and hpp.coefficients.shape[1] == 7
    e = np.where(c == 0, "arrest", None).astype(object)
    e[(c == 0) & (Z["fin"].to_numpy() == 1)] = "other"
    cr = penalty_path(
        CompetingRisksProportionalHazards.fit_penalised,
        x,
        Z.to_numpy(float),
        e,
        l1_ratio=1,
        n_penalties=4,
    )
    assert cr.coefficients.shape == (4, 2 * len(COLS))
    assert np.all(cr.coefficients[0] == 0)


def test_refusals():
    x, Z, c = _rossi()
    with pytest.raises(ValueError, match="penalizers"):
        penalty_path(CoxPH.fit_penalised, x, Z, c=c, penalizer=0.1)
    with pytest.raises(ValueError, match="n_penalties"):
        penalty_path(CoxPH.fit_penalised, x, Z, c=c, n_penalties=1)
    with pytest.raises(ValueError, match="min_ratio"):
        penalty_path(CoxPH.fit_penalised, x, Z, c=c, min_ratio=2)
    with pytest.raises(ValueError, match="penalizers"):
        penalty_path(CoxPH.fit_penalised, x, Z, c=c, penalizers=[-1])
    with pytest.raises(ValueError, match="penalty_factor"):
        penalty_path(CoxPH.fit_penalised, x, Z, c=c, penalty_factor=[[1.0]])


def test_plot_draws_every_coefficient():
    import matplotlib

    matplotlib.use("Agg")
    x, Z, c = _rossi()
    path = penalty_path(CoxPH.fit_penalised, x, Z, c=c, n_penalties=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax = path.plot()
    assert len(ax.get_lines()) >= len(COLS)
    assert ax.xaxis_inverted()
