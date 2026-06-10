import numpy as np

import surpyval

# The expected survival/hazard values below were generated once with
# lifelines (KaplanMeierFitter / NelsonAalenFitter with
# nelson_aalen_smoothing=False) on the fixed datasets defined in each test,
# then hard-coded here so the suite no longer depends on lifelines at run
# time. surpyval's convention is c == 1 for right-censored observations,
# which maps to lifelines' event indicator 1 - c and entry= for the left
# truncation times.


def test_kaplan_meier_against_known_values():
    # KaplanMeier survival function with integer weights, no censoring.
    x = np.array([3.0, 5.0, 5.0, 8.0, 12.0, 12.0, 12.0, 20.0, 25.0, 30.0])
    n = np.array([2, 1, 3, 2, 1, 4, 2, 1, 5, 1])
    x_test = np.array([1.0, 4.0, 5.0, 6.0, 12.0, 19.0, 25.0, 31.0])
    expected = [
        1.0,
        0.909090909091,
        0.727272727273,
        0.727272727273,
        0.318181818182,
        0.318181818182,
        0.045454545455,
        0.0,
    ]
    model = surpyval.KaplanMeier.fit(x, n=n)
    assert np.allclose(model.sf(x_test), expected, atol=1e-9)


def test_kaplan_meier_censored_against_known_values():
    # KaplanMeier with right-censored observations (c == 1) and weights.
    x = np.array([4.0, 7.0, 9.0, 13.0, 16.0, 21.0, 28.0, 33.0, 41.0, 50.0])
    c = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 0])
    n = np.array([1, 2, 1, 3, 1, 2, 1, 1, 4, 1])
    x_test = np.array([2.0, 7.0, 10.0, 16.0, 28.0, 41.0, 55.0])
    expected = [
        1.0,
        0.823529411765,
        0.823529411765,
        0.570135746606,
        0.488687782805,
        0.097737556561,
        0.0,
    ]
    model = surpyval.KaplanMeier.fit(x, c=c, n=n)
    assert np.allclose(model.sf(x_test), expected, atol=1e-9)


def test_kaplan_meier_censored_and_truncated_against_known_values():
    # KaplanMeier with right censoring, left truncation, and weights.
    x = np.array([6.0, 9.0, 14.0, 18.0, 22.0, 27.0, 33.0, 40.0, 48.0, 55.0])
    tl = np.array([0.0, 2.0, 5.0, 5.0, 10.0, 12.0, 0.0, 20.0, 25.0, 30.0])
    c = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
    n = np.array([1, 1, 2, 1, 1, 3, 1, 1, 2, 1])
    x_test = np.array([3.0, 9.0, 18.0, 27.0, 40.0, 55.0, 60.0])
    expected = [
        1.0,
        0.833333333333,
        0.520833333333,
        0.297619047619,
        0.238095238095,
        0.0,
        0.0,
    ]
    model = surpyval.KaplanMeier.fit(x, c=c, n=n, tl=tl)
    assert np.allclose(model.sf(x_test), expected, atol=1e-9)


def test_nelson_aalen_against_known_values():
    # NelsonAalen cumulative hazard with integer weights, no censoring.
    x = np.array([3.0, 5.0, 5.0, 8.0, 12.0, 12.0, 12.0, 20.0, 25.0, 30.0])
    n = np.array([2, 1, 3, 2, 1, 4, 2, 1, 5, 1])
    x_test = np.array([1.0, 4.0, 5.0, 6.0, 12.0, 19.0, 25.0, 31.0])
    expected = [
        0.0,
        0.090909090909,
        0.290909090909,
        0.290909090909,
        0.915909090909,
        0.915909090909,
        1.892099567100,
        2.892099567100,
    ]
    model = surpyval.NelsonAalen.fit(x, n=n)
    assert np.allclose(model.Hf(x_test), expected, atol=1e-9)


def test_nelson_aalen_censored_against_known_values():
    # NelsonAalen with right-censored observations (c == 1) and weights.
    x = np.array([4.0, 7.0, 9.0, 13.0, 16.0, 21.0, 28.0, 33.0, 41.0, 50.0])
    c = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 0])
    n = np.array([1, 2, 1, 3, 1, 2, 1, 1, 4, 1])
    x_test = np.array([2.0, 7.0, 10.0, 16.0, 28.0, 41.0, 55.0])
    expected = [
        0.0,
        0.183823529412,
        0.183823529412,
        0.514592760181,
        0.657449903038,
        1.457449903038,
        2.457449903038,
    ]
    model = surpyval.NelsonAalen.fit(x, c=c, n=n)
    assert np.allclose(model.Hf(x_test), expected, atol=1e-9)


def test_nelson_aalen_censored_and_truncated_against_known_values():
    # NelsonAalen with right censoring, left truncation, and weights.
    x = np.array([6.0, 9.0, 14.0, 18.0, 22.0, 27.0, 33.0, 40.0, 48.0, 55.0])
    tl = np.array([0.0, 2.0, 5.0, 5.0, 10.0, 12.0, 0.0, 20.0, 25.0, 30.0])
    c = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
    n = np.array([1, 1, 2, 1, 1, 3, 1, 1, 2, 1])
    x_test = np.array([3.0, 9.0, 18.0, 27.0, 40.0, 55.0, 60.0])
    expected = [
        0.0,
        0.166666666667,
        0.583333333333,
        1.011904761905,
        1.211904761905,
        2.878571428571,
        2.878571428571,
    ]
    model = surpyval.NelsonAalen.fit(x, c=c, n=n, tl=tl)
    assert np.allclose(model.Hf(x_test), expected, atol=1e-9)


def test_fleming_harrington_same_as_nelson_aalen_with_no_counts():
    # With no ties and no counts the Fleming-Harrington and Nelson-Aalen
    # cumulative hazards coincide; the expected values are the lifelines
    # Nelson-Aalen estimates for this dataset.
    x = np.array([2.0, 4.0, 6.0, 8.0, 9.0, 13.0, 17.0, 22.0, 30.0, 45.0])
    x_test = np.array([1.0, 4.0, 6.0, 10.0, 22.0, 50.0])
    expected = [
        0.0,
        0.211111111111,
        0.336111111111,
        0.645634920635,
        1.428968253968,
        2.928968253968,
    ]
    fh = surpyval.FlemingHarrington.fit(x)
    na = surpyval.NelsonAalen.fit(x)
    assert np.allclose(fh.Hf(x_test), expected, atol=1e-9)
    assert np.allclose(fh.Hf(x_test), na.Hf(x_test), atol=1e-9)


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
