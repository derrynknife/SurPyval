"""Rotated Clayton, Gumbel and Joe copulas (#157, the ``rotation`` option).

References are pyvinecopulib 1.0.0's rotations (R VineCopula's convention:
90 is ``v - C(1 - u, v)``, 180 the survival copula, 270
``u - C(u, 1 - v)``) at ``(u, v) = (0.3, 0.7)``.
"""

import json
import warnings

import numpy as np
import pytest

import surpyval as surv
from surpyval import LogNormal, Weibull
from surpyval.multivariate import Clayton, Frank, Gaussian, Gumbel, Joe

MARGINS = [Weibull.from_params([10.0, 2.0]), LogNormal.from_params([2.5, 0.5])]

# (family, theta, rotation): (cdf, h1, h2, pdf) at (0.3, 0.7), vinecopulib
VINECOP = {
    (Clayton, 2.0, 90): (
        0.1303480788601884,
        0.5389327541530857,
        0.4610672458469143,
        1.5296104659027978,
    ),
    (Gumbel, 2.0, 180): (
        0.2848780620209499,
        0.8844021560584541,
        0.08951961352454485,
        0.6636783965240105,
    ),
    (Joe, 2.86, 270): (
        0.13208211449291057,
        0.5363700961978732,
        0.4636299038021268,
        1.5486197488093714,
    ),
}


@pytest.mark.parametrize(
    "key", list(VINECOP), ids=["Clayton90", "Gumbel180", "Joe270"]
)
def test_against_vinecopulib(key):
    family, theta, rotation = key
    rotated = family.rotated(rotation)
    got = [
        float(f(0.3, 0.7, theta))
        for f in (rotated.cdf, rotated.du, rotated.dv, rotated.pdf)
    ]
    assert got == pytest.approx(VINECOP[key], rel=1e-9)


@pytest.mark.parametrize("family", [Clayton, Gumbel, Joe])
@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_identities(family, rotation):
    theta = {"Clayton": 2.0, "Gumbel": 2.0, "Joe": 2.86}[family.name]
    rotated = family.rotated(rotation)
    g = np.r_[1e-6, np.linspace(0.02, 0.98, 49), 1 - 1e-6]
    U, V = np.meshgrid(g, g, indexing="ij")
    C = rotated.cdf(U.ravel(), V.ravel(), theta).reshape(U.shape)
    volume = C[1:, 1:] - C[:-1, 1:] - C[1:, :-1] + C[:-1, :-1]
    assert volume.min() > -1e-12
    inner = np.linspace(0.05, 0.95, 7)
    one = np.full_like(inner, 1 - 1e-12)
    np.testing.assert_allclose(
        rotated.cdf(inner, one, theta), inner, atol=1e-9
    )
    np.testing.assert_allclose(
        rotated.cdf(one, inner, theta), inner, atol=1e-9
    )
    h = 1e-6
    fd = (
        rotated.cdf(inner + h, 0.4, theta) - rotated.cdf(inner - h, 0.4, theta)
    ) / (2 * h)
    np.testing.assert_allclose(rotated.du(inner, 0.4, theta), fd, atol=1e-7)
    # dependence measures: a quarter turn negates them
    sign = -1 if rotation in (90, 270) else 1
    assert rotated.kendall_tau(theta) == sign * family.kendall_tau(theta)
    assert rotated.spearman_rho(theta) == pytest.approx(
        sign * family.spearman_rho(theta)
    )
    lower, upper = family.tail_dependence(theta)
    expected = (upper, lower) if rotation == 180 else (0.0, 0.0)
    assert rotated.tail_dependence(theta) == expected
    # conditional-inversion draws follow the rotated CDF
    u, v = rotated.sample_uv(40_000, np.array([theta]), random_state=3)
    for a in (0.2, 0.5, 0.8):
        for b in (0.2, 0.5, 0.8):
            assert np.mean((u <= a) & (v <= b)) == pytest.approx(
                float(rotated.cdf(a, b, theta)), abs=0.008
            )


def test_rotations_compose_and_are_cached():
    assert Clayton.rotated(0) is Clayton
    assert Clayton.rotated(90) is Clayton.rotated(90)
    assert Clayton.rotated(90).rotated(90) is Clayton.rotated(180)
    assert Clayton.rotated(180).rotated(180) is Clayton


@pytest.mark.parametrize("family", [Frank, Gaussian])
def test_symmetric_families_are_not_rotated(family):
    with pytest.raises(ValueError, match="not rotated"):
        family.rotated(180)
    X = family.from_params([0.5], MARGINS).random(50, random_state=0)
    with pytest.raises(ValueError, match="not rotated"):
        family.fit(X, margins=[Weibull, LogNormal], rotation=90)


@pytest.mark.parametrize("rotation", [45, -90, 360])
def test_rotation_must_be_a_quarter_turn(rotation):
    with pytest.raises(ValueError, match="0, 90, 180 or 270"):
        Clayton.rotated(rotation)


@pytest.mark.parametrize(
    "family, theta, rotation, how",
    [
        (Clayton, 2.0, 180, "IFM"),
        (Gumbel, 2.0, 90, "IFM"),
        (Joe, 2.86, 270, "MLE"),
    ],
)
def test_fit_recovers_a_rotated_copula(family, theta, rotation, how):
    truth = family.from_params([theta], MARGINS, rotation=rotation)
    X = truth.random(1000, random_state=4)
    rng = np.random.default_rng(5)
    stop = np.column_stack(
        [rng.uniform(0, 25, 1000), rng.uniform(0, 30, 1000)]
    )
    c = (X > stop).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = family.fit(
            np.minimum(X, stop),
            c=c,
            margins=[Weibull, LogNormal],
            how=how,
            rotation=rotation,
        )
    assert model.copula is family.rotated(rotation)
    assert model.params[0] == pytest.approx(theta, abs=0.35)
    # the unrotated family cannot express the data as well
    plain = family.fit(np.minimum(X, stop), c=c, margins=[Weibull, LogNormal])
    assert model.log_likelihood > plain.log_likelihood + 20


def test_rotated_model_round_trips_and_says_so():
    X = Clayton.from_params([2.0], MARGINS, rotation=180).random(
        200, random_state=1
    )
    model = Clayton.fit(X, margins=[Weibull, LogNormal], rotation=180)
    assert "Clayton (rotated 180 degrees)" in repr(model)
    d = json.loads(json.dumps(model.to_dict()))
    assert d["copula"] == "Clayton" and d["rotation"] == 180
    back = surv.from_dict(d)
    assert back.copula is Clayton.rotated(180)
    pts = [[3.0, 8.0], [12.0, 20.0]]
    np.testing.assert_array_equal(back.sf(pts), model.sf(pts))
    assert back.aic() == pytest.approx(model.aic(), rel=1e-12)
    # an unrotated model's dict is as before, with no rotation key
    assert "rotation" not in Clayton.from_params([2.0], MARGINS).to_dict()


def test_perfect_dependence_warning_follows_the_rotation():
    x1 = 10.0 * (-np.log1p(-(np.arange(1, 21) - 0.3) / 20.4)) ** 0.5
    counter = np.column_stack([x1, 100.0 - x1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Clayton.fit(counter, margins=[Weibull, Weibull], rotation=90)
    assert len(caught) == 1
    assert "perfectly discordant" in str(caught[0].message)
