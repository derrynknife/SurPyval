"""Identities between a model's functions (#379).

For every registered model that has the functions involved:

- ``sf + ff == 1`` (for a single-cause Fine-Gray model, ``sf + cif``);
- ``Hf == -log(sf)``;
- ``df == hf * sf`` for a continuous model, and the discrete analogue
  ``df(k) == hf(k) * sf(k - 1)`` (the hazard of a discrete model is
  the probability of the event at ``k`` given survival to ``k``);
- ``qf(ff(x)) == x`` where the model's CDF is strictly increasing
  (continuous), or at the atoms (discrete);
- the causes' cumulative incidences sum to the all-cause ``ff``.
"""

import numpy as np
import pytest

from surpyval.tests.conformance.registry import call, cases_for, fitted


def _values(case, model, fname, x, Z=None):
    return np.asarray(call(case, model, fname, x, Z), dtype=float)


def _z_rows(case):
    # A model with covariates is checked with each query row as one
    # covariate vector for every time, so each curve is a whole curve.
    if case.Z is None:
        return [None]
    return [case.Z[0], case.Z[-1]]


@pytest.mark.parametrize("case", cases_for("sf_ff", needs=("sf",)))
def test_sf_plus_ff_is_one(case):
    model = fitted(case)
    ff_name = "ff" if case.has("ff") else "cif"
    for z in _z_rows(case):
        sf = _values(case, model, "sf", case.x, z)
        ff = _values(case, model, ff_name, case.x, z)
        np.testing.assert_allclose(sf + ff, 1.0, rtol=0, atol=1e-10)


@pytest.mark.parametrize("case", cases_for("Hf_sf", needs=("sf", "Hf")))
def test_cumulative_hazard_is_minus_log_sf(case):
    model = fitted(case)
    for z in _z_rows(case):
        sf = _values(case, model, "sf", case.x, z)
        Hf = _values(case, model, "Hf", case.x, z)
        # Where sf has underflowed to 0, Hf (often computed directly) can
        # still be finite; it only has to be past what sf can resolve.
        zero = sf == 0
        assert np.all(Hf[zero] > 700)
        np.testing.assert_allclose(
            Hf[~zero], -np.log(sf[~zero]), rtol=1e-8, atol=1e-10
        )


@pytest.mark.parametrize(
    "case", cases_for("df_hf_sf", needs=("sf", "hf", "df"))
)
def test_density_is_hazard_times_survival(case):
    model = fitted(case)
    x = case.x
    for z in _z_rows(case):
        df = _values(case, model, "df", x, z)
        hf = _values(case, model, "hf", x, z)
        if case.continuous:
            sf = _values(case, model, "sf", x, z)
        else:
            # P(X = k) = P(X = k | X >= k) P(X > k - 1)
            sf = _values(case, model, "sf", x - 1, z)
        keep = sf > 1e-12
        np.testing.assert_allclose(
            df[keep], (hf * sf)[keep], rtol=1e-7, atol=1e-12
        )


@pytest.mark.parametrize("case", cases_for("qf_ff", needs=("qf", "ff")))
def test_quantile_inverts_cdf(case):
    model = fitted(case)
    x = case.x
    ff = _values(case, model, "ff", x)
    if case.continuous:
        # Only where the CDF is strictly increasing is qf its inverse.
        df = _values(case, model, "df", x) if case.has("df") else 1.0
        keep = (ff > 1e-9) & (ff < 1 - 1e-9) & (np.asarray(df) > 1e-12)
        q = _values(case, model, "qf", ff[keep])
        np.testing.assert_allclose(q, x[keep], rtol=1e-6)
    else:
        # At an atom k (P(X = k) > 0) the generalised inverse gives k back.
        keep = _values(case, model, "df", x) > 1e-12
        keep &= ff < 1 - 1e-12
        q = _values(case, model, "qf", ff[keep])
        np.testing.assert_array_equal(q, x[keep])


@pytest.mark.parametrize("case", cases_for("cif_sum", needs=("cif", "ff")))
def test_cumulative_incidences_sum_to_ff(case):
    model = fitted(case)
    for z in _z_rows(case):
        total = sum(
            np.asarray(call(case, model, "cif", case.x, z, event=e), float)
            for e in case.events
        )
        ff = _values(case, model, "ff", case.x, z)
        np.testing.assert_allclose(total, ff, rtol=1e-8, atol=1e-10)
