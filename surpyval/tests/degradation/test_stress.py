"""Stage-2 accelerated degradation: stress-dependent path parameters."""

import warnings

import numpy as np
import pytest

from surpyval.degradation import (
    DegradationAnalysis,
    DegradationModel,
    ExponentialPath,
    LinearPath,
    LinkedPathModel,
    PathModel,
    PowerPath,
)
from surpyval.degradation.population import reml_estimate
from surpyval.degradation.stress import (
    fixed_effect_names,
    stress_design,
    validate_links,
)


def _rt(d):
    """JSON round trip, so tuples/arrays become plain lists."""
    import json

    return json.loads(json.dumps(d))


# -- LinkedPathModel -------------------------------------------------------


@pytest.mark.parametrize(
    "base,links,theta",
    [
        (LinearPath, {"b": "log"}, (2.0, 0.5)),
        (LinearPath, {"a": "log", "b": "log"}, (2.0, 0.5)),
        (LinearPath, {"b": "identity"}, (2.0, -0.5)),
        (ExponentialPath, {"b": "log"}, (2.0, 0.05)),
        (PowerPath, {"a": "log", "b": "log"}, (2.0, 0.8)),
    ],
)
def test_linked_model_matches_base(base, links, theta):
    linked = LinkedPathModel(base, links)
    x = np.linspace(1, 10, 20)
    eta = linked.to_link(theta)
    assert np.allclose(linked.to_natural(eta), theta)
    assert np.allclose(linked.path(x, *eta), base.path(x, *theta))
    level = 0.5 * (base.path(x, *theta).min() + base.path(x, *theta).max())
    assert np.allclose(
        linked.inv_path(level, *eta), base.inv_path(level, *theta)
    )
    # chain-rule Jacobian against the base class' finite differences
    analytic = linked.jacobian(x, *eta)
    numeric = PathModel.jacobian(linked, x, *eta)
    assert analytic.shape == (len(x), len(base.param_names))
    assert np.allclose(analytic, numeric, rtol=1e-4, atol=1e-6)
    # a fit on the link scale is the base fit mapped there
    y = base.path(x, *theta)
    assert np.allclose(linked.fit(x, y), linked.to_link(base.fit(x, y)))


def test_linked_model_names_and_linearity():
    linked = LinkedPathModel(LinearPath, {"b": "log"})
    assert linked.param_names == ["a", "log(b)"]
    assert linked.links == {"a": "identity", "b": "log"}
    assert not linked.linear_in_parameters
    assert LinkedPathModel(LinearPath, {"b": "identity"}).linear_in_parameters
    assert not LinkedPathModel(
        ExponentialPath, {"b": "identity"}
    ).linear_in_parameters
    assert "Linear" in repr(linked)


def test_log_link_rejects_non_positive_parameter():
    linked = LinkedPathModel(LinearPath, {"b": "log"})
    with pytest.raises(ValueError, match="not positive"):
        linked.to_link([2.0, -0.5])


def test_validate_links_errors():
    with pytest.raises(ValueError, match="non-empty dict"):
        validate_links(LinearPath, {})
    with pytest.raises(ValueError, match="non-empty dict"):
        validate_links(LinearPath, "log")
    with pytest.raises(ValueError, match="does not have"):
        validate_links(LinearPath, {"c": "log"})
    with pytest.raises(ValueError, match="Unknown link"):
        validate_links(LinearPath, {"b": "logit"})
    # returned in path-parameter order, only the named parameters
    assert list(validate_links(LinearPath, {"b": "log", "a": "identity"})) == [
        "a",
        "b",
    ]


def test_stress_design_layout():
    # linear path, b stress-dependent, two covariates
    d = stress_design([0.5, 2.0], {"b": "log"}, ["a", "b"])
    assert d.shape == (2, 4)
    assert np.allclose(d, [[1, 0, 0, 0], [0, 1, 0.5, 2.0]])
    names = fixed_effect_names(["a", "log(b)"], ["a", "b"], {"b": "log"}, 2)
    assert names == ["a", "log(b)", "log(b):Z0", "log(b):Z1"]
    # both parameters stress-dependent, one covariate
    d = stress_design(3.0, {"a": "identity", "b": "log"}, ["a", "b"])
    assert np.allclose(d, [[1, 3.0, 0, 0], [0, 0, 1, 3.0]])


# -- the widened REML ------------------------------------------------------


def test_reml_fixed_design_defaults_to_random_design():
    # with A_i = X_i the widened routine is the old one
    rng = np.random.default_rng(3)
    y_list, d_list = [], []
    for _ in range(30):
        n_k = int(rng.integers(3, 9))
        x_k = np.sort(rng.uniform(5, 60, n_k))
        y_k = rng.normal(10, 1) + rng.normal(0.4, 0.05) * x_k
        y_list.append(y_k + rng.normal(0, 1.0, n_k))
        d_list.append(np.column_stack([np.ones_like(x_k), x_k]))
    cov0 = np.diag([1.0, 0.0025])
    plain = reml_estimate(y_list, d_list, cov0, 1.0)
    explicit = reml_estimate(y_list, d_list, cov0, 1.0, a_mat_list=d_list)
    assert np.array_equal(plain[0], explicit[0])
    assert np.array_equal(plain[1], explicit[1])
    assert plain[2] == explicit[2]


def test_reml_recovers_linear_stress_effect():
    # y = a + b(z) t with b(z) = 0.2 + 0.3 z plus unit scatter: an exact
    # linear mixed model with a wider fixed-effects design
    rng = np.random.default_rng(5)
    y_list, x_list, a_list, z_rows = [], [], [], []
    t = np.arange(5.0, 45.0, 5.0)
    for z in np.repeat([0.0, 1.0, 2.0], 20):
        a = rng.normal(10.0, 1.0)
        b = 0.2 + 0.3 * z + rng.normal(0, 0.03)
        y_list.append(a + b * t + rng.normal(0, 0.5, t.size))
        design = np.column_stack([np.ones_like(t), t])
        x_list.append(design)
        d = stress_design(z, {"b": "identity"}, ["a", "b"])
        a_list.append(design @ d)
        z_rows.append(z)
    gamma, cov, sigma2, ok = reml_estimate(
        y_list, x_list, np.diag([1.0, 0.001]), 0.25, a_mat_list=a_list
    )
    assert np.allclose(gamma, [10.0, 0.2, 0.3], atol=[0.4, 0.03, 0.02])
    assert np.isclose(sigma2, 0.25, rtol=0.3)
    assert np.isclose(cov[1, 1], 0.03**2, rtol=0.6)


# -- DegradationAnalysis.fit(links=) ----------------------------------------


def adt_data(gamma=0.8, b0=0.5, intercept=10.0, seed=0, sd_log_b=0.15):
    """Linear degradation whose rate is log-linear in stress:
    ``y = a + b t``, ``log b = log b0 + gamma Z + N(0, sd_log_b^2)``,
    ``a ~ N(intercept, 1)``, measurement noise sd 0.5."""
    rng = np.random.default_rng(seed)
    times = np.arange(1, 11) * 5.0
    xs, ys, ii, ZZ = [], [], [], []
    uid = 0
    for Z in [0.0, 0.5, 1.0, 1.5]:
        for _ in range(12):
            b = b0 * np.exp(gamma * Z) * np.exp(rng.normal(0, sd_log_b))
            a = intercept + rng.normal(0, 1.0)
            y = a + b * times + rng.normal(0, 0.5, size=times.size)
            xs.append(times)
            ys.append(y)
            ii.append(np.full(times.size, uid))
            ZZ.append(np.full(times.size, Z))
            uid += 1
    return tuple(np.concatenate(v) for v in (xs, ys, ii, ZZ))


@pytest.mark.parametrize("population_method", ["moments", "reml"])
def test_log_link_recovers_arrhenius_style_rate(population_method):
    x, y, i, Z = adt_data()
    model = DegradationAnalysis.fit(
        x,
        y,
        i,
        threshold=100.0,
        Z=Z,
        links={"b": "log"},
        population_method=population_method,
    )
    assert model.links == {"b": "log"}
    assert model.path_param_fixed_names == ["a", "log(b)", "log(b):Z0"]
    fixed = dict(zip(model.path_param_fixed_names, model.path_param_fixed))
    assert fixed["a"] == pytest.approx(10.0, abs=0.5)
    assert fixed["log(b)"] == pytest.approx(np.log(0.5), abs=0.1)
    assert fixed["log(b):Z0"] == pytest.approx(0.8, abs=0.1)
    # between-unit scatter given stress: var a = 1, var log b = 0.15^2
    link_cov = model.path_param_link_cov
    assert link_cov.shape == (2, 2)
    assert np.isclose(link_cov[0, 0], 1.0, rtol=0.6)
    assert np.isclose(link_cov[1, 1], 0.15**2, rtol=0.6)
    assert model.measurement_var == pytest.approx(0.25, rel=0.3)
    # the Stage-1 life regression is still there and unchanged in kind
    assert model.is_accelerated
    assert model.life_model.params[-1] == pytest.approx(0.8, abs=0.2)
    # the pooled population summaries are still the natural-scale ones
    assert model.path_params.shape == (48, 2)
    assert (model.path_params[:, 1] > 0).all()
    assert model.path_param_mean.shape == (2,)


def test_identity_link_is_an_exact_linear_mixed_model():
    # b(z) = 0.2 + 0.3 z, additive in stress, so the identity link and
    # a linear path keep the model an exact LMM (no FOCE)
    rng = np.random.default_rng(7)
    t = np.arange(5.0, 45.0, 5.0)
    xs, ys, ii, ZZ = [], [], [], []
    for uid, z in enumerate(np.repeat([0.0, 1.0, 2.0], 16)):
        a = rng.normal(10.0, 1.0)
        b = 0.2 + 0.3 * z + rng.normal(0, 0.03)
        xs.append(t)
        ys.append(a + b * t + rng.normal(0, 0.5, t.size))
        ii.append(np.full(t.size, uid))
        ZZ.append(np.full(t.size, z))
    x, y, i, Z = (np.concatenate(v) for v in (xs, ys, ii, ZZ))
    reml = DegradationAnalysis.fit(
        x,
        y,
        i,
        threshold=60.0,
        Z=Z,
        links={"b": "identity"},
        population_method="reml",
    )
    moments = DegradationAnalysis.fit(
        x, y, i, threshold=60.0, Z=Z, links={"b": "identity"}
    )
    assert reml.path_param_fixed_names == ["a", "b", "b:Z0"]
    assert np.allclose(
        reml.path_param_fixed, [10.0, 0.2, 0.3], atol=[0.4, 0.03, 0.02]
    )
    # balanced design: REML and the two-stage moments agree closely
    assert np.allclose(
        reml.path_param_fixed, moments.path_param_fixed, rtol=1e-3, atol=1e-3
    )
    assert np.allclose(
        reml.path_param_link_cov,
        moments.path_param_link_cov,
        rtol=0.05,
        atol=1e-4,
    )


def test_links_on_a_nonlinear_path():
    # exponential path a exp(b t) with a stress-dependent rate
    rng = np.random.default_rng(9)
    t = np.arange(1.0, 13.0)
    xs, ys, ii, ZZ = [], [], [], []
    for uid, z in enumerate(np.repeat([0.0, 0.5, 1.0], 20)):
        a = rng.normal(2.0, 0.1)
        b = 0.05 * np.exp(0.6 * z) * np.exp(rng.normal(0, 0.05))
        xs.append(t)
        ys.append(a * np.exp(b * t) + rng.normal(0, 0.03, t.size))
        ii.append(np.full(t.size, uid))
        ZZ.append(np.full(t.size, z))
    x, y, i, Z = (np.concatenate(v) for v in (xs, ys, ii, ZZ))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DegradationAnalysis.fit(
            x,
            y,
            i,
            threshold=6.0,
            path="exponential",
            Z=Z,
            links={"b": "log"},
            population_method="reml",
        )
    fixed = dict(zip(model.path_param_fixed_names, model.path_param_fixed))
    assert fixed["a"] == pytest.approx(2.0, abs=0.1)
    assert fixed["log(b)"] == pytest.approx(np.log(0.05), abs=0.1)
    assert fixed["log(b):Z0"] == pytest.approx(0.6, abs=0.1)
    assert (np.linalg.eigvalsh(model.path_param_link_cov) > 0).all()


def test_links_with_two_covariates_and_two_linked_parameters():
    rng = np.random.default_rng(2)
    t = np.arange(5.0, 45.0, 5.0)
    xs, ys, ii, ZZ = [], [], [], []
    uid = 0
    for z1 in [0.0, 1.0]:
        for z2 in [0.0, 1.0, 2.0]:
            for _ in range(8):
                a = 10.0 + 2.0 * z1 + rng.normal(0, 0.5)
                b = (
                    0.5
                    * np.exp(0.4 * z1 - 0.2 * z2)
                    * np.exp(rng.normal(0, 0.05))
                )
                xs.append(t)
                ys.append(a + b * t + rng.normal(0, 0.3, t.size))
                ii.append(np.full(t.size, uid))
                ZZ.append(np.tile([z1, z2], (t.size, 1)))
                uid += 1
    x, y, i = (np.concatenate(v) for v in (xs, ys, ii))
    Z = np.vstack(ZZ)
    model = DegradationAnalysis.fit(
        x, y, i, threshold=60.0, Z=Z, links={"a": "identity", "b": "log"}
    )
    assert model.path_param_fixed_names == [
        "a",
        "a:Z0",
        "a:Z1",
        "log(b)",
        "log(b):Z0",
        "log(b):Z1",
    ]
    fixed = dict(zip(model.path_param_fixed_names, model.path_param_fixed))
    assert fixed["a:Z0"] == pytest.approx(2.0, abs=0.4)
    assert fixed["a:Z1"] == pytest.approx(0.0, abs=0.3)
    assert fixed["log(b):Z0"] == pytest.approx(0.4, abs=0.08)
    assert fixed["log(b):Z1"] == pytest.approx(-0.2, abs=0.05)


def test_links_require_Z():
    x, y, i, Z = adt_data()
    with pytest.raises(ValueError, match="Z must be given"):
        DegradationAnalysis.fit(x, y, i, threshold=100.0, links={"b": "log"})


def test_links_validation_errors_surface_from_fit():
    x, y, i, Z = adt_data()
    with pytest.raises(ValueError, match="does not have"):
        DegradationAnalysis.fit(
            x, y, i, threshold=100.0, Z=Z, links={"rate": "log"}
        )
    with pytest.raises(ValueError, match="Unknown link"):
        DegradationAnalysis.fit(
            x, y, i, threshold=100.0, Z=Z, links={"b": "sqrt"}
        )


def test_log_link_needs_positive_fitted_parameters():
    # a unit whose fitted rate is negative cannot sit on a log link
    x, y, i, Z = adt_data()
    y = y.copy()
    mask = i == 0
    y[mask] = 10.0 - 0.5 * x[mask]
    with pytest.raises(ValueError, match="not positive"):
        DegradationAnalysis.fit(
            x, y, i, threshold=100.0, Z=Z, links={"b": "log"}
        )


def test_plain_fit_has_no_stress_attributes():
    x, y, i, Z = adt_data()
    model = DegradationAnalysis.fit(x, y, i, threshold=100.0, Z=Z)
    assert model.links is None
    assert model.path_param_fixed is None
    assert model.path_param_fixed_names is None
    assert model.path_param_link_cov is None
    assert "Path Stress" not in repr(model)


def test_repr_shows_stress_population():
    x, y, i, Z = adt_data()
    model = DegradationAnalysis.fit(
        x, y, i, threshold=100.0, Z=Z, links={"b": "log"}
    )
    text = repr(model)
    assert "Path Stress Links   : b: log" in text
    assert "log(b):Z0" in text


def test_linked_model_round_trips():
    x, y, i, Z = adt_data()
    model = DegradationAnalysis.fit(
        x, y, i, threshold=100.0, Z=Z, links={"b": "log"}
    )
    restored = DegradationModel.from_dict(_rt(model.to_dict()))
    assert restored.links == {"b": "log"}
    assert restored.path_param_fixed_names == model.path_param_fixed_names
    assert np.allclose(restored.path_param_fixed, model.path_param_fixed)
    assert np.allclose(restored.path_param_link_cov, model.path_param_link_cov)
    t = np.array([100.0, 200.0])
    assert np.allclose(model.sf(t, Z=[1.0]), restored.sf(t, Z=[1.0]))
    # a plain model's dict carries the (absent) stress fields as None
    plain = DegradationAnalysis.fit(x, y, i, threshold=100.0, Z=Z)
    d = plain.to_dict()
    assert d["links"] is None and d["path_param_fixed"] is None
    assert DegradationModel.from_dict(_rt(d)).links is None


def test_fit_from_df_passes_links():
    import pandas as pd

    x, y, i, Z = adt_data()
    df = pd.DataFrame({"t": x, "deg": y, "unit": i, "stress": Z})
    model = DegradationAnalysis.fit_from_df(
        df,
        x="t",
        y="deg",
        i="unit",
        Z_cols="stress",
        threshold=100.0,
        links={"b": "log"},
    )
    assert model.path_param_fixed_names == ["a", "log(b)", "log(b):Z0"]
