"""Fine-Gray subdistribution-hazard regression.

The model targets the cumulative incidence of a cause directly:
``F_k(t|Z) = 1 - exp(-Lambda0(t) exp(beta'Z))``. Independent right-censoring
is handled by inverse-probability-of-censoring weighting (IPCW), so these
tests exercise parameter recovery *under censoring* -- the regime where a
naive (unweighted) subdistribution risk set would be biased.
"""

import numpy as np
import pytest

from surpyval.univariate.competing_risks import (
    CompetingRisksProportionalHazards,
    FineGray,
)


def _simulate_fine_gray(N, seed, beta=(0.7, -0.4), p=0.5, cens_scale=3.0):
    """
    Simulate two-cause competing-risks data whose cause-1 cumulative incidence
    follows a Fine-Gray model with coefficients ``beta`` (Fine & Gray 1999
    construction): ``F_1(t|Z) = 1 - (1 - p(1 - e^{-t}))^{exp(beta'Z)}``.
    Cause 2 mops up the rest; exponential right-censoring is applied.
    """
    rng = np.random.default_rng(seed)
    beta = np.asarray(beta, dtype=float)
    Z = rng.uniform(-1, 1, size=(N, beta.size))
    phi = np.exp(Z @ beta)

    p1 = 1 - (1 - p) ** phi  # P(cause = 1 | Z)
    is1 = rng.uniform(size=N) < p1

    x = np.empty(N)
    e = np.empty(N, dtype=object)

    # Invert the conditional cause-1 CIF for the cause-1 event times.
    v = rng.uniform(size=N)
    w = 1 - (1 - v * p1) ** (1 / phi)  # = p (1 - e^{-t})
    t1 = -np.log(np.clip(1 - w / p, 1e-12, 1.0))
    x[is1] = t1[is1]
    e[is1] = 1

    # Cause-2 times are exponential.
    t2 = rng.exponential(1.0, size=N)
    x[~is1] = t2[~is1]
    e[~is1] = 2

    # Independent right-censoring.
    cens = rng.exponential(cens_scale, size=N)
    c = (x > cens).astype(int)
    x = np.minimum(x, cens)
    e[c == 1] = None
    return x, Z, e, c


# --- standalone FineGray --------------------------------------------------


def test_recovers_parameters_under_censoring():
    x, Z, e, c = _simulate_fine_gray(8000, 0)
    m = FineGray.fit(x, Z, e, c=c, cause=1)
    assert np.allclose(m.beta, [0.7, -0.4], atol=0.07)
    # Roughly a quarter of observations are censored in this design.
    assert 0.15 < c.mean() < 0.35


def test_baseline_cif_matches_theory_at_Z_zero():
    # At Z = 0 the model reduces to the baseline CIF p(1 - e^{-t}).
    x, Z, e, c = _simulate_fine_gray(8000, 1)
    m = FineGray.fit(x, Z, e, c=c, cause=1)
    t = np.array([0.5, 1.0, 2.0])
    expected = 0.5 * (1 - np.exp(-t))
    assert np.allclose(m.cif(t, [0.0, 0.0]), expected, atol=0.03)


def test_cif_is_monotone_and_bounded():
    x, Z, e, c = _simulate_fine_gray(4000, 2)
    m = FineGray.fit(x, Z, e, c=c, cause=1)
    t = np.linspace(0.0, 5.0, 50)
    cif = m.cif(t, [0.3, -0.2])
    assert np.all(cif >= 0) and np.all(cif <= 1)
    assert np.all(np.diff(cif) >= -1e-12)


def test_sf_and_hf_identities():
    x, Z, e, c = _simulate_fine_gray(4000, 3)
    m = FineGray.fit(x, Z, e, c=c, cause=1)
    t = np.array([0.5, 1.0, 2.0])
    Z0 = [0.1, -0.1]
    assert np.allclose(m.sf(t, Z0), 1 - m.cif(t, Z0))


def test_positive_coefficient_is_significant():
    # The strong cause-1 covariate should be clearly significant; all p-values
    # are well-defined probabilities.
    x, Z, e, c = _simulate_fine_gray(6000, 4)
    m = FineGray.fit(x, Z, e, c=c, cause=1)
    assert np.all(np.isfinite(m.p_values))
    assert np.all((m.p_values >= 0) & (m.p_values <= 1))
    assert m.p_values[0] < 0.01  # beta_0 = 0.7


def test_counts_equivalent_to_repeated_rows():
    x, Z, e, c = _simulate_fine_gray(1500, 5)
    m_rep = FineGray.fit(
        np.repeat(x, 2),
        np.repeat(Z, 2, axis=0),
        np.repeat(e, 2),
        c=np.repeat(c, 2),
        cause=1,
    )
    m_cnt = FineGray.fit(x, Z, e, c=c, n=np.full(x.size, 2), cause=1)
    assert np.allclose(m_rep.beta, m_cnt.beta, atol=1e-4)


def test_ipcw_matters_versus_naive_no_censoring():
    # With no censoring the IPCW weights are all 1, so the fit is the plain
    # subdistribution model and still recovers the truth.
    x, Z, e, c = _simulate_fine_gray(8000, 6, cens_scale=1e6)
    assert c.mean() == 0.0
    m = FineGray.fit(x, Z, e, c=c, cause=1)
    assert np.allclose(m.beta, [0.7, -0.4], atol=0.07)


# --- input handling -------------------------------------------------------


def test_cause_required_when_multiple_event_types():
    x, Z, e, c = _simulate_fine_gray(500, 7)
    with pytest.raises(ValueError, match="specify `cause`"):
        FineGray.fit(x, Z, e, c=c)


def test_unknown_cause_rejected():
    x, Z, e, c = _simulate_fine_gray(500, 8)
    with pytest.raises(ValueError, match="not observed"):
        FineGray.fit(x, Z, e, c=c, cause=99)


# --- CompetingRisksProportionalHazards integration -----------------------


def test_crph_fine_gray_matches_standalone():
    x, Z, e, c = _simulate_fine_gray(5000, 9)
    crph = CompetingRisksProportionalHazards.fit(x, Z, e, c=c, how="Fine-Gray")
    standalone = FineGray.fit(x, Z, e, c=c, cause=1)
    i1 = crph.event_idx_map[1]
    assert np.allclose(crph.betas[i1], standalone.beta, atol=1e-6)
    t = np.array([0.5, 1.0, 2.0])
    assert np.allclose(
        crph.cif(t, [0.2, -0.1], 1), standalone.cif(t, [0.2, -0.1])
    )


def test_crph_fine_gray_cif_identities():
    x, Z, e, c = _simulate_fine_gray(4000, 10)
    crph = CompetingRisksProportionalHazards.fit(x, Z, e, c=c, how="Fine-Gray")
    t = np.array([0.5, 1.0, 2.0])
    Z0 = [0.1, 0.1]
    assert np.allclose(crph.sf(t, Z0, 1) + crph.cif(t, Z0, 1), 1.0)
    assert np.allclose(crph.Hf(t, Z0, 1), -np.log(crph.sf(t, Z0, 1)))


def test_crph_fine_gray_hf_df_raise():
    x, Z, e, c = _simulate_fine_gray(2000, 11)
    crph = CompetingRisksProportionalHazards.fit(x, Z, e, c=c, how="Fine-Gray")
    with pytest.raises(ValueError, match="no pointwise"):
        crph.hf([1.0], [0.0, 0.0], 1)
    with pytest.raises(ValueError, match="no pointwise"):
        crph.df([1.0], [0.0, 0.0], 1)


def test_crph_fine_gray_requires_event():
    x, Z, e, c = _simulate_fine_gray(2000, 12)
    crph = CompetingRisksProportionalHazards.fit(x, Z, e, c=c, how="Fine-Gray")
    with pytest.raises(ValueError, match="pass `event`"):
        crph.sf([1.0], [0.0, 0.0])


def test_crph_cox_path_still_runs():
    # Regression guard: the cause-specific Cox path (a sibling of the same
    # public entry point) fits and predicts.
    x, Z, e, c = _simulate_fine_gray(3000, 13)
    crph = CompetingRisksProportionalHazards.fit(x, Z, e, c=c, how="Cox")
    cif = crph.cif(np.array([0.5, 1.0, 2.0]), [0.1, -0.1], 1)
    assert np.all(np.isfinite(cif))
