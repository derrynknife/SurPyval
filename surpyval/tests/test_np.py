import lifelines
import numpy as np

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
    tl = np.where(
        (t == 1) & (x > 0), x * np.random.uniform(0, 1, x.size), -np.inf
    )
    tl = np.where(
        (t == 1) & (x < 0), x - np.abs(x * np.random.uniform(0, 1, x.size)), tl
    )
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
        x = surpyval.Weibull.random(
            int(np.random.uniform(2, 1000, 1)), *test_params
        )
        n = np.ones_like(x) * int(np.random.uniform(1, 5))
        x_test = np.random.uniform(x.min() / 2, x.max() * 2, 100)
        ll_est = kmf.fit(x, weights=n).predict(x_test).values
        surp_est = surpyval.KaplanMeier.fit(x, n=n).sf(x_test)
        assert np.allclose(ll_est, surp_est, 1e-15)


def test_kaplan_meier_censored_against_lifelines():
    kmf = lifelines.KaplanMeierFitter()
    for i in range(100):
        test_params = []
        for b in ((1, 100), (0.5, 20)):
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = surpyval.Weibull.random(
            int(np.random.uniform(2, 1000, 1)), *test_params
        )
        c = np.random.binomial(1, np.random.uniform(0, 1, 1), x.shape)
        x = x - np.abs(x * np.random.uniform(0, 1, x.shape))
        n = np.ones_like(x) * int(np.random.uniform(1, 5))
        x_test = np.random.uniform(x.min() / 2, x.max() * 2, 100)
        ll_est = kmf.fit(x, 1 - c, weights=n).predict(x_test).values
        surp_est = surpyval.KaplanMeier.fit(x, c=c, n=n).sf(x_test)
        assert np.allclose(ll_est, surp_est, 1e-15)


def test_kaplan_meier_censored_and_truncated_against_lifelines():
    kmf = lifelines.KaplanMeierFitter()
    for i in range(100):
        test_params = []
        for b in ((1, 100), (0.5, 20)):
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = surpyval.Weibull.random(
            int(np.random.uniform(2, 1000, 1)), *test_params
        )
        x, tl = left_truncate(x, surpyval.Weibull, 0.1, test_params)
        x, c = right_censor(x, tl, 0.2)
        n = np.ones_like(x) * int(np.random.uniform(1, 5))
        x_test = np.random.uniform(x.min() / 2, x.max() * 2, 100)
        ll_est = kmf.fit(x, 1 - c, entry=tl, weights=n).predict(x_test).values
        surp_est = surpyval.KaplanMeier.fit(x, c=c, n=n, tl=tl).sf(x_test)
        assert np.allclose(ll_est, surp_est, 1e-15)


def test_nelson_aalen_against_lifelines():
    naf = lifelines.NelsonAalenFitter(nelson_aalen_smoothing=False)
    for i in range(100):
        test_params = []
        for b in ((1, 100), (0.5, 20)):
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = surpyval.Weibull.random(
            int(np.random.uniform(2, 1000, 1)), *test_params
        )
        n = np.ones_like(x) * int(np.random.uniform(1, 5))
        x_test = np.random.uniform(x.min() / 2, x.max() * 2, 100)
        ll_est = naf.fit(x, weights=n).predict(x_test).values
        surp_est = surpyval.NelsonAalen.fit(x, n=n).Hf(x_test)
        assert np.allclose(ll_est, surp_est, 1e-15)


def test_nelson_aalen_censored_against_lifelines():
    naf = lifelines.NelsonAalenFitter(nelson_aalen_smoothing=False)
    for i in range(100):
        test_params = []
        for b in ((1, 100), (0.5, 20)):
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = surpyval.Weibull.random(
            int(np.random.uniform(2, 1000, 1)), *test_params
        )
        c = np.random.binomial(1, np.random.uniform(0, 1, 1), x.shape)
        x = x - np.abs(x * np.random.uniform(0, 1, x.shape))
        n = np.ones_like(x) * int(np.random.uniform(1, 5))
        x_test = np.random.uniform(x.min() / 2, x.max() * 2, 100)
        ll_est = naf.fit(x, 1 - c, weights=n).predict(x_test).values
        surp_est = surpyval.NelsonAalen.fit(x, c=c, n=n).Hf(x_test)
        assert np.allclose(ll_est, surp_est, 1e-15)


def test_nelson_aalen_censored_and_truncated_against_lifelines():
    naf = lifelines.NelsonAalenFitter(nelson_aalen_smoothing=False)
    for i in range(100):
        test_params = []
        for b in ((1, 100), (0.5, 20)):
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = surpyval.Weibull.random(
            int(np.random.uniform(2, 1000, 1)), *test_params
        )
        x, tl = left_truncate(x, surpyval.Weibull, 0.1, test_params)
        x, c = right_censor(x, tl, 0.2)
        n = np.ones_like(x) * int(np.random.uniform(1, 5))
        x_test = np.random.uniform(x.min() / 2, x.max() * 2, 100)
        ll_est = naf.fit(x, 1 - c, entry=tl, weights=n).predict(x_test).values
        surp_est = surpyval.NelsonAalen.fit(x, c=c, n=n, tl=tl).Hf(x_test)
        assert np.allclose(ll_est, surp_est, 1e-15)


def test_fleming_harrington_same_as_nelson_aalen_with_no_counts():
    naf = lifelines.NelsonAalenFitter(nelson_aalen_smoothing=False)
    for i in range(100):
        test_params = []
        for b in ((1, 100), (0.5, 20)):
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = surpyval.Weibull.random(
            int(np.random.uniform(2, 1000, 1)), *test_params
        )
        x_test = np.random.uniform(x.min() / 2, x.max() * 2, 100)
        ll_est = naf.fit(x).predict(x_test).values
        surp_est = surpyval.FlemingHarrington.fit(x).Hf(x_test)
        assert np.allclose(ll_est, surp_est, 1e-15)


def test_fleming_harrington_case():
    solution_h = [
        (1.0 / 10 + 1.0 / 9 + 1.0 / 8),
        (1.0 / 7 + 1.0 / 6),
        (1.0 / 5 + 1.0 / 4 + 1.0 / 3 + 1.0 / 2),
        1.0,
    ]
    x = [1, 2, 3, 4]
    n = [3, 2, 4, 1]
    H = np.cumsum(solution_h)
    H_est = surpyval.FlemingHarrington.fit(x=x, n=n).H
    assert np.allclose(H, H_est, 1e-15)


def test_turnbull_with_R_1():
    """
    R's icenReg solution, from code:

    ``` r
    left <-  c(1, 8, 8, 7, 7, 17, 37, 46, 46, 45)
    right <- c(7, 8, 15, 16, 14, Inf, 44, Inf, Inf, Inf)

    df <- data.frame(cbind(left, right))

    r_model <- ic_np(cbind(left, right) ~ 0, df)

    getSCurves(r_model)
    ```
    """
    x_test = [1, 7, 8, 14, 44, np.inf]
    sf_test = [1.0, 0.9, 0.7, 0.5, 0.375, 0]

    left = np.array([1, 8, 8, 7, 7, 17, 37, 46, 46, 45])
    right = np.array([7, 8, 10, 16, 14, np.inf, 44, np.inf, np.inf, np.inf])

    model = surpyval.Turnbull.fit(
        xl=left, xr=right, turnbull_estimator="Kaplan-Meier"
    )

    assert np.allclose(model.sf(x_test), sf_test, 1e-5)


def test_turnbull_with_R_2():
    """
    R's icenReg solution, from code:

    ``` r
    left <- c(1, 76, 288, 501, 579, 667, 829, 920, 1071)
    right <- c(71, 169, 344, 504, 579, 754, 829, 971, Inf)

    df <- data.frame(cbind(left, right))

    r_model <- ic_np(cbind(left, right) ~ 0, df)

    getSCurves(r_model)
    ```
    """
    x_test = [1, 71, 169, 344, 504, 579, 754, 829, 971, np.inf]
    sf_test = [
        1.0,
        0.8888889,
        0.7777778,
        0.6666667,
        0.5555556,
        0.4444444,
        0.3333333,
        0.2222222,
        0.1111111,
        0.0,
    ]

    left = [1, 76, 288, 501, 579, 667, 829, 920, 1071]
    right = [71, 169, 344, 504, 579, 754, 829, 971, np.inf]

    model = surpyval.Turnbull.fit(
        xl=left, xr=right, turnbull_estimator="Kaplan-Meier"
    )

    assert np.allclose(model.sf(x_test), sf_test, 1e-5)


[0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0]


def test_turnbull_with_R_3():
    """
    R's icenReg solution, from code:

    ``` r
    left <- seq(1, 10)
    right <- seq(2, 11)

    df <- data.frame(cbind(left, right))

    t <- ic_np(cbind(left, right) ~ 0, df)

    getSCurves(t)
    ```
    """
    x_test = np.array(range(1, 12))
    sf_test = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    left = np.array(range(1, 11))
    right = left + 1

    model = surpyval.Turnbull.fit(
        xl=left, xr=right, turnbull_estimator="Kaplan-Meier"
    )

    assert np.allclose(model.sf(x_test), sf_test, 1e-5)


def test_kaplan_meier_with_R():
    """
    R's survreg() solutions.
    """
    # Test 1
    x1 = np.array(range(1, 9))
    sf_test1 = [0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0]
    model = surpyval.KaplanMeier.fit(x1)
    assert np.allclose(model.R, sf_test1, 1e-5)

    # If repeated should be same
    n = np.ones_like(x1) * 10
    model = surpyval.KaplanMeier.fit(x1, n=n)
    assert np.allclose(model.R, sf_test1, 1e-5)

    """
    times <- c(1, 2, 3, 4, 5, 6, 7, 8)
    weights <- c(1, 1, 1, 1, 1, 1, 1, 10)
    censored <- c(1, 1, 1, 1, 1, 1, 1, 0)
    model <- survfit(Surv(times, censored) ~ 0, weights=weights)
    model$surv
    [1] 0.9411765 0.8823529 0.8235294 0.7647059 0.7058824 0.6470588 0.5882353
    [8] 0.5882353
    """
    # If repeated should be same
    n = np.ones_like(x1)
    n[-1] = 10
    c = np.zeros_like(x1)
    c[-1] = 1
    sf_test3 = [
        0.94117647,
        0.88235294,
        0.82352941,
        0.76470588,
        0.70588235,
        0.64705882,
        0.58823529,
        0.58823529,
    ]
    model = surpyval.KaplanMeier.fit(x1, c=c, n=n)
    assert np.allclose(model.R, sf_test3, 1e-5)


def test_nelson_aalen_with_R():
    """
    R's survreg() solutions.
    """
    """
    times <- c(1, 2, 3, 4, 5, 6, 7, 8)
    weights <- c(10, 10, 10, 10, 10, 10, 10, 10)
    model <- survfit(Surv(times) ~ 0, type='fh')
    model$surv
    [1] 0.88249690 0.76501706 0.64757296 0.53018790 0.41291075 0.29586348
    [7] 0.17945027 0.06601607
    """
    # Test 1
    x1 = np.array(range(1, 9))
    sf_test1 = [
        0.88249690,
        0.76501706,
        0.64757296,
        0.53018790,
        0.41291075,
        0.29586348,
        0.17945027,
        0.06601607,
    ]
    model = surpyval.NelsonAalen.fit(x1)
    assert np.allclose(model.R, sf_test1, 1e-5)

    # If repeated should be same
    n = np.ones_like(x1) * 10
    model = surpyval.NelsonAalen.fit(x1, n=n)
    assert np.allclose(model.R, sf_test1, 1e-5)

    """
    times <- c(1, 2, 3, 4, 5, 6, 7, 8)
    weights <- c(1, 1, 1, 1, 1, 1, 1, 10)
    censored <- c(1, 1, 1, 1, 1, 1, 1, 0)
    model <- survfit(Surv(times, censored) ~ 0, weights=weights, type='fh')
    model$surv
    [1] 0.9428731 0.8857473 0.8286228 0.7714999 0.7143789 0.6572603 0.6001448
    [8] 0.6001448
    """

    # If repeated should be same
    n = np.ones_like(x1)
    n[-1] = 10
    c = np.zeros_like(x1)
    c[-1] = 1
    sf_test3 = [
        0.9428731,
        0.8857473,
        0.8286228,
        0.7714999,
        0.7143789,
        0.6572603,
        0.6001448,
        0.6001448,
    ]
    model = surpyval.NelsonAalen.fit(x1, c=c, n=n)
    assert np.allclose(model.R, sf_test3, 1e-5)
