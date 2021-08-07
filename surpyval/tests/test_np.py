import pytest
import numpy as np
import lifelines
import surpyval

def right_censor(x, tl, frac):
    c = np.random.binomial(1, frac, x.shape)
    x_out = np.copy(x)
    for i, (trunc, value) in enumerate(zip(tl, x)):
        if c[i] == 0:
            continue
        if np.isfinite(trunc):
            x_out[i] = np.random.uniform(trunc, value, 1)
        else:
            x_out[i] = value - np.abs(value * np.random.uniform(0, 1, 1))
    return x_out, c

def left_truncate(x, dist, frac, params):
    t = np.random.binomial(1, frac, x.shape)
    # Find a lower value
    tl = np.where((t == 1) & (x > 0), x * np.random.uniform(0, 1, x.size), -np.inf)
    tl = np.where((t == 1) & (x < 0), x - np.abs(x * np.random.uniform(0, 1, x.size)), tl)
    drop_due_to_truncation = dist.ff(tl, *params)
    drop_due_to_truncation[~np.isfinite(drop_due_to_truncation)] = 0
    
    keep = np.ones_like(x)
    for i, p in enumerate(drop_due_to_truncation):
        if p == 0:
            continue
        else:
            keep[i] = np.random.binomial(1, p)

    mask = keep == 1
    tl = tl[mask]
    x = x[mask]
    return x, tl

def test_kaplan_meier_against_lifelines():
    kmf = lifelines.KaplanMeierFitter()
    for i in range(100):
        test_params = []
        for b in ((1, 100), (0.5, 20)):
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = surpyval.Weibull.random(int(np.random.uniform(2, 1000, 1)), *test_params)
        n = np.ones_like(x) * int(np.random.uniform(1, 5))
        ll_est = kmf.fit(x, weights=n).predict(x).values
        surp_est = surpyval.KaplanMeier.fit(x, n=n).sf(x)
        if not np.allclose(ll_est, surp_est, 1e-15):
            raise AssertionError('Kaplan-Meier different to lifelines?!')

def test_kaplan_meier_censored_against_lifelines():
    kmf = lifelines.KaplanMeierFitter()
    for i in range(100):
        test_params = []
        for b in ((1, 100), (0.5, 20)):
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = surpyval.Weibull.random(int(np.random.uniform(2, 1000, 1)), *test_params)
        c = np.random.binomial(1, np.random.uniform(0, 1, 1), x.shape)
        x = x - np.abs(x * np.random.uniform(0, 1, x.shape))
        n = np.ones_like(x) * int(np.random.uniform(1, 5))
        ll_est = kmf.fit(x, 1-c, weights=n).predict(x).values
        surp_est = surpyval.KaplanMeier.fit(x, c=c, n=n).sf(x)
        if not np.allclose(ll_est, surp_est, 1e-15):
            raise AssertionError('Kaplan-Meier different to lifelines?!')


def test_kaplan_meier_censored_and_truncated_against_lifelines():
    kmf = lifelines.KaplanMeierFitter()
    for i in range(100):
        test_params = []
        for b in ((1, 100), (0.5, 20)):
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = surpyval.Weibull.random(int(np.random.uniform(2, 1000, 1)), *test_params)
        x, tl = left_truncate(x, surpyval.Weibull, 0.1, test_params)
        x, c = right_censor(x, tl, 0.2)
        n = np.ones_like(x) * int(np.random.uniform(1, 5))
        ll_est = kmf.fit(x, 1-c, entry=tl, weights=n).predict(x).values
        surp_est = surpyval.KaplanMeier.fit(x, c=c, n=n, tl=tl).sf(x)
        if not np.allclose(ll_est, surp_est, 1e-15):
            raise AssertionError('Kaplan-Meier different to lifelines?!')


def test_nelson_aalen_against_lifelines():
    naf = lifelines.NelsonAalenFitter(nelson_aalen_smoothing=False)
    for i in range(100):
        test_params = []
        for b in ((1, 100), (0.5, 20)):
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = surpyval.Weibull.random(int(np.random.uniform(2, 1000, 1)), *test_params)
        n = np.ones_like(x) * int(np.random.uniform(1, 5))
        ll_est = naf.fit(x, weights=n).predict(x).values
        surp_est = surpyval.NelsonAalen.fit(x, n=n).Hf(x)
        if not np.allclose(ll_est, surp_est, 1e-15):
            raise AssertionError('Kaplan-Meier different to lifelines?!')

def test_nelson_aalen_censored_against_lifelines():
    naf = lifelines.NelsonAalenFitter(nelson_aalen_smoothing=False)
    for i in range(100):
        test_params = []
        for b in ((1, 100), (0.5, 20)):
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = surpyval.Weibull.random(int(np.random.uniform(2, 1000, 1)), *test_params)
        c = np.random.binomial(1, np.random.uniform(0, 1, 1), x.shape)
        x = x - np.abs(x * np.random.uniform(0, 1, x.shape))
        n = np.ones_like(x) * int(np.random.uniform(1, 5))
        ll_est = naf.fit(x, 1-c, weights=n).predict(x).values
        surp_est = surpyval.NelsonAalen.fit(x, c=c, n=n).Hf(x)
        if not np.allclose(ll_est, surp_est, 1e-15):
            raise AssertionError('Kaplan-Meier different to lifelines?!')


def test_nelson_aalen_censored_and_truncated_against_lifelines():
    naf = lifelines.NelsonAalenFitter(nelson_aalen_smoothing=False)
    for i in range(100):
        test_params = []
        for b in ((1, 100), (0.5, 20)):
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = surpyval.Weibull.random(int(np.random.uniform(2, 1000, 1)), *test_params)
        x, tl = left_truncate(x, surpyval.Weibull, 0.1, test_params)
        x, c = right_censor(x, tl, 0.2)
        n = np.ones_like(x) * int(np.random.uniform(1, 5))
        ll_est = naf.fit(x, 1-c, entry=tl, weights=n).predict(x).values
        surp_est = surpyval.NelsonAalen.fit(x, c=c, n=n, tl=tl).Hf(x)
        if not np.allclose(ll_est, surp_est, 1e-15):
            raise AssertionError('Kaplan-Meier different to lifelines?!')


def test_fleming_harrington_same_as_nelson_aalen_with_no_counts():
    naf = lifelines.NelsonAalenFitter(nelson_aalen_smoothing=False)
    for i in range(100):
        test_params = []
        for b in ((1, 100), (0.5, 20)):
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = surpyval.Weibull.random(int(np.random.uniform(2, 1000, 1)), *test_params)
        ll_est = naf.fit(x).predict(x).values
        surp_est = surpyval.FlemingHarrington.fit(x).Hf(x)
        if not np.allclose(ll_est, surp_est, 1e-15):
            raise AssertionError('Fleming-Harrington fails different to lifelines?!')


def test_fleming_harrington_HF_less_than_or_equal_to_nelson_aalen_with_counts():
    naf = lifelines.NelsonAalenFitter(nelson_aalen_smoothing=False)
    for i in range(100):
        test_params = []
        for b in ((1, 100), (0.5, 20)):
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)

        x = surpyval.Weibull.random(int(np.random.uniform(2, 1000, 1)), *test_params)
        n = np.ones_like(x) * int(np.random.uniform(2, 5))

        ll_na_est = naf.fit(x, weights=n).predict(x).values
        surp_est = surpyval.FlemingHarrington.fit(x, n=n).Hf(x)
        # FH cumulative hazard should be less than NA Hf
        diff = surp_est - ll_na_est
        if (diff < 0).all():
            raise AssertionError('Fleming-Harrington not all below NelsonAalen')

def test_fleming_harrington_case():
    solution_h = [
        (1./10 + 1./9 + 1./8),
        (1./7 + 1./6),
        (1./5 + 1./4 + 1./3 + 1./2),
        1.
    ]
    x = [1, 2, 3, 4]
    n = [3, 2, 4, 1]
    H = np.cumsum(solution_h)
    H_est = surpyval.FlemingHarrington.fit(x=x, n=n).H
    assert np.allclose(H, H_est, 1e-15)




