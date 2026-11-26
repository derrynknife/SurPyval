"""``ParametricPanelMultiStateRegression``: a parametric regression per
transition fitted to states seen only at visits, the covariates of each
visit holding over the gap that starts at it; on the forward clock (a
non-homogeneous Markov model) and the reset clock (semi-Markov,
progressive models)."""

import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad, solve_ivp

from surpyval import ExponentialPH, Weibull, WeibullPH
from surpyval.multistate import (
    PanelMultiStateRegression,
    ParametricPanelMultiStateRegression,
    TransitionMatrix,
)
from surpyval.multistate.panel import _check_panel
from surpyval.multistate.parametric import _fitters
from surpyval.tests.multistate._inspected import (
    NAMES,
    TRANSITIONS,
    visits_covariates,
)

TM = TransitionMatrix(TRANSITIONS)
PARS = {
    ("healthy", "ill"): (6.0, 1.5),
    ("healthy", "dead"): (25.0, 1.2),
    ("ill", "dead"): (8.0, 2.0),
}
BETAS = {
    ("healthy", "ill"): (0.5, 0.6),
    ("healthy", "dead"): (0.3, 0.0),
    ("ill", "dead"): (-0.4, 0.0),
}


def _visits(n, seed, clock="forward"):
    ids, times, states, Z = visits_covariates(n, seed, PARS, BETAS, clock)
    s = np.array([NAMES[int(v)] for v in states], dtype=object)
    return dict(
        x=np.array(times),
        Z=pd.DataFrame(Z, columns=["male", "w"]),
        s=s,
        i=np.array(ids),
        obstype=np.where(s == "dead", "exact_entry", "panel"),
    )


@pytest.fixture(scope="module")
def small():
    return _visits(100, 7072)


def _ph(pars, betas, pair):
    """The WeibullPH model of a transition with the given baseline and
    coefficients."""
    model = WeibullPH.fit(
        x=[1.0, 2, 3, 4, 5, 6],
        Z=pd.DataFrame(
            {"male": [0.0, 1, 0, 1, 0, 1], "w": [0.0, 0, 1, 1, 0, 1]}
        ),
    )
    return model.with_params([*pars[pair], *betas[pair]])


def _models(pars=PARS, betas=BETAS):
    return [_ph(pars, betas, p) for p in TM.transitions]


def _h(par, t):
    a, b = par
    return b / a * (t / a) ** (b - 1)


def _H(par, t):
    a, b = par
    return (np.maximum(t, 0.0) / a) ** b


def _lik(pdata, clock):
    return ParametricPanelMultiStateRegression._likelihood(
        pdata, _fitters(WeibullPH, pdata.tm, True), clock
    )


@pytest.mark.parametrize("clock", ["forward", "reset"])
def test_exponential_ph_is_panel_multistate_regression(small, clock):
    """With constant baseline hazards both clocks are
    ``PanelMultiStateRegression``'s Markov model with covariates on the
    intensities, the covariates of a visit holding until the next (a
    progressive model: on the reset clock the time in a state does not
    matter)."""
    markov = PanelMultiStateRegression.fit(transitions=TM, **small)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = ParametricPanelMultiStateRegression.fit(
            transitions=TM, model=ExponentialPH, clock=clock, **small
        )
    assert model.maximum == "verified"
    assert model.covariate_names == ["male", "w"]
    assert_allclose(model.log_likelihood, markov.log_likelihood, rtol=1e-10)
    rates = [float(m.params[0]) for m in model._models()]
    assert_allclose(rates, markov.params[:3], rtol=1e-5)
    assert_allclose(model.betas, markov.betas, rtol=1e-4, atol=1e-6)
    # the standard errors: the rates, then the coefficients by covariate
    se = model.standard_errors().reshape(3, 3)
    se_m = markov.standard_errors()
    assert_allclose(se[:, 0], se_m[:3], rtol=1e-4)
    assert_allclose(se[:, 1:].T.ravel(), se_m[3:], rtol=1e-4)
    assert model.n_obs == markov.n_obs
    assert_allclose(model.bic(), markov.bic(), rtol=1e-9)
    z = [1.0, 0.0]
    assert_allclose(
        model.occupancy([2.0, 5.0], [z, z], "ill", from_state="healthy"),
        markov.occupancy([2.0, 5.0], [z, z], "ill", from_state="healthy"),
        rtol=1e-5,
    )


def _P_forward(pars, betas, z, t0, t1):
    """P(t0, t1 | z) of the illness-death model with Weibull
    proportional hazards on the forward clock, by an ODE solver."""
    names = ["healthy", "ill", "dead"]

    def rhs(t, p):
        P = p.reshape(3, 3)
        Q = np.zeros((3, 3))
        for (a, b), par in pars.items():
            m = np.exp(np.dot(betas[(a, b)], z))
            Q[names.index(a), names.index(b)] = m * _h(par, t)
        Q -= np.diag(Q.sum(axis=1))
        return (P @ Q).ravel()

    sol = solve_ivp(rhs, (t0, t1), np.eye(3).ravel(), rtol=1e-12, atol=1e-14)
    return sol.y[:, -1].reshape(3, 3)


def test_forward_likelihood_is_the_product_over_gaps_with_their_covariates():
    """Each gap's factor is an element of P(t_{j-1}, t_j | z_{j-1}), the
    covariates of the visit that starts it (by an ODE solver here); an
    exactly timed death adds the hazard into death for the gap's
    covariates; a censored visit sums the states it allows."""
    pars = {
        ("healthy", "ill"): (4.0, 1.5),
        ("healthy", "dead"): (12.0, 1.2),
        ("ill", "dead"): (5.0, 2.0),
    }
    betas = {
        ("healthy", "ill"): (0.7, -0.5),
        ("healthy", "dead"): (0.2, 0.4),
        ("ill", "dead"): (-0.3, 0.9),
    }
    x = [0.5, 1.5, 3.0, 0.5, 2.0, 4.5, 1.0, 2.5]
    s = ["healthy", "ill", "ill", "healthy", "healthy", "dead", "healthy"]
    s.append("alive")
    i = [1, 1, 1, 2, 2, 2, 3, 3]
    Z = np.array(
        [[1, 0], [1, 1], [1, 0], [0, 1], [0, 0], [0, 1], [1, 1], [1, 0]],
        dtype=float,
    )
    obstype = ["panel"] * 5 + ["exact_entry"] + ["panel"] * 2
    pdata = _check_panel(
        x, s, i, Z, obstype, TM, {"alive": ["healthy", "ill"]}, None, None
    )
    models = _models(pars, betas)
    lik = _lik(pdata, "forward")
    lik.plan(models)
    assert lik.n_patterns == 4

    def P(z, t0, t1):
        return _P_forward(pars, betas, np.array(z, dtype=float), t0, t1)

    def h(pair, z, t):
        return np.exp(np.dot(betas[pair], z)) * _h(pars[pair], t)

    P1 = P([1, 0], 0.5, 1.5)[0, 1] * P([1, 1], 1.5, 3.0)[1, 1]
    P24 = P([0, 0], 2.0, 4.5)
    h_hd = h(("healthy", "dead"), [0, 0], 4.5)
    h_id = h(("ill", "dead"), [0, 0], 4.5)
    P2 = P([0, 1], 0.5, 2.0)[0, 0] * (P24[0, 0] * h_hd + P24[0, 1] * h_id)
    P3 = P([1, 1], 1.0, 2.5)[0, :2].sum()
    assert_allclose(lik.neg_ll(models), -np.log(P1 * P2 * P3), rtol=1e-8)


def test_reset_likelihood_with_covariates_by_quadrature():
    """On the reset clock, the hazards of each gap those of the
    covariates of the visit that starts it, against scipy's adaptive
    quadrature: a stay in "ill" spanning three gaps with three rows of
    covariates (entered at an unknown time in the first), and a death at
    a known time from a state not seen. One shape is below 1, so that
    hazard is infinite at entry."""
    pars = {
        ("healthy", "ill"): (3.0, 0.8),
        ("healthy", "dead"): (10.0, 1.3),
        ("ill", "dead"): (4.0, 1.7),
    }
    betas = {
        ("healthy", "ill"): (0.6, -0.4),
        ("healthy", "dead"): (0.3, 0.2),
        ("ill", "dead"): (-0.5, 0.8),
    }
    hi, hd, idd = (pars[p] for p in TM.transitions)

    def m(pair, z):
        return np.exp(np.dot(betas[pair], z))

    def Sh(d0, d1, z):
        # staying healthy from duration d0 to d1 with the covariates z
        return np.exp(
            -m(TM.transitions[0], z) * (_H(hi, d1) - _H(hi, d0))
            - m(TM.transitions[1], z) * (_H(hd, d1) - _H(hd, d0))
        )

    def Si(d0, d1, z):
        return np.exp(-m(TM.transitions[2], z) * (_H(idd, d1) - _H(idd, d0)))

    x = [0, 1, 2.5, 0, 1, 2, 3]
    s = ["healthy", "healthy", "dead", "healthy", "ill", "ill", "dead"]
    i = [1, 1, 1, 2, 2, 2, 2]
    za, zb, zc = [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]
    Z = np.array([za, zb, zb, za, zb, zc, zc])
    obstype = ["panel", "panel", "exact_entry", "panel", "panel", "panel"]
    obstype.append("panel")
    pdata = _check_panel(x, s, i, Z, obstype, TM, None, None, None)
    lik = _lik(pdata, "reset")

    opts = dict(epsabs=0, epsrel=1e-12, limit=200)
    # unit 1: healthy at 0 and 1 (za then zb), dead at exactly 2.5
    stay1 = Sh(0, 1, za)
    direct = Sh(1, 2.5, zb) * m(TM.transitions[1], zb) * _h(hd, 2.5)
    via_ill = quad(
        lambda v: Sh(1, v, zb)
        * m(TM.transitions[0], zb)
        * _h(hi, v)
        * Si(0, 2.5 - v, zb)
        * m(TM.transitions[2], zb)
        * _h(idd, 2.5 - v),
        1.0,
        2.5,
        **opts,
    )[0]
    L1 = stay1 * (direct + via_ill)

    # unit 2: fell ill at u in (0, 1] (za), ill at 1 and 2 (zb over
    # (1, 2]), dead by 3 (zc over (2, 3])
    def unit2(u):
        onset = Sh(0, u, za) * m(TM.transitions[0], za) * _h(hi, u)
        to_two = Si(0, 1 - u, za) * Si(1 - u, 2 - u, zb)
        dead = 1 - Si(2 - u, 3 - u, zc)
        return onset * to_two * dead

    L2 = quad(unit2, 0, 1, **opts)[0]
    models = _models(pars, betas)
    assert_allclose(lik.neg_ll(models), -np.log(L1 * L2), rtol=1e-7)


def test_one_covariate_row_is_the_model_without_covariates():
    """Covariates constant over every gap are aliased with the baseline:
    the fit's likelihood is the plain panel model's."""
    from surpyval.multistate import ParametricPanelMultiState

    d = _visits(60, 31)
    plain = ParametricPanelMultiState.fit(
        d["x"], d["s"], d["i"], obstype=d["obstype"], transitions=TM,
        dist=Weibull,
    )  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ParametricPanelMultiStateRegression.fit(
            d["x"], np.ones(d["x"].size), d["s"], d["i"],
            obstype=d["obstype"], transitions=TM,
        )  # fmt: skip
    assert np.all(np.isnan(model.betas))
    assert_allclose(model.neg_ll(), plain.neg_ll(), rtol=1e-8)


@pytest.mark.parametrize("clock", ["forward", "reset"])
def test_recovers_the_weibull_ph_hazards(clock):
    """One seeded data set of 400 units: every parameter within three
    standard errors of the truth (the full study is the calibration
    test)."""
    d = _visits(400, 11 if clock == "forward" else 12, clock)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ParametricPanelMultiStateRegression.fit(
            transitions=TM, clock=clock, **d
        )
    assert model.maximum == "verified"
    truth = np.concatenate([[*PARS[p], *BETAS[p]] for p in TM.transitions])
    z = (model.params - truth) / model.standard_errors()
    assert np.all(np.abs(z) < 3), z


@pytest.mark.parametrize("clock", ["forward", "reset"])
def test_round_trip(small, clock):
    import surpyval as sp

    model = ParametricPanelMultiStateRegression.fit(
        transitions=TM, model=ExponentialPH, clock=clock, **small
    )
    back = sp.from_dict(model.to_dict())
    assert isinstance(back, ParametricPanelMultiStateRegression)
    assert back.covariate_names == model.covariate_names
    assert_allclose(back.params, model.params)
    assert_allclose(back.covariance(), model.covariance())
    assert back.neg_ll() == model.neg_ll()
    assert back.maximum == model.maximum
    z = [0.0, 1.0]
    assert_allclose(back.cb(4.0, z, "ill"), model.cb(4.0, z, "ill"))
    assert "Covariates: male, w" in repr(back)


def test_fit_from_df(small):
    df = pd.DataFrame(
        {
            "t": small["x"],
            "male": small["Z"]["male"],
            "w": small["Z"]["w"],
            "state": small["s"],
            "unit": small["i"],
            "how": small["obstype"],
        }
    )
    a = ParametricPanelMultiStateRegression.fit_from_df(
        df, "t", ["male", "w"], "state", "unit", obstype_col="how",
        transitions=TM, model=ExponentialPH,
    )  # fmt: skip
    b = ParametricPanelMultiStateRegression.fit(
        transitions=TM, model=ExponentialPH, **small
    )
    assert_allclose(a.params, b.params)
    assert a.covariate_names == ["male", "w"]


def test_refusals(small):
    kw = {k: v for k, v in small.items() if k != "Z"}
    with pytest.raises(ValueError, match="Z is needed"):
        ParametricPanelMultiStateRegression.fit(Z=None, transitions=TM, **kw)
    bad = small["Z"].to_numpy().copy()
    bad[3, 0] = np.nan
    with pytest.raises(ValueError, match="missing or infinite covariates"):
        ParametricPanelMultiStateRegression.fit(Z=bad, transitions=TM, **kw)
    with pytest.raises(TypeError, match="arrays of visits"):
        import surpyval as sp

        ParametricPanelMultiStateRegression.fit(
            sp.SurpyvalData(x=[1.0, 2.0]), transitions=TM
        )
