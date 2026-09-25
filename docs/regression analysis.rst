
Regression Analysis
===================

The time until some event happens will, almost certainly, be impacted by factors. For example, when considering how long a machine will last before failure an engineer will want to account for the operational conditions. It may operate in a humid environment, or it may operate at a higher rate. The question is then, how do we account for these variations, or 'covariates', on the time until failure?

Regression analysis is the process of capturing the effect that covariates have on the item. That is, we use data on other factors to 'regress' onto the survival distribution. The purpose of this type of regression is so that you can ask, and answer, questions like "what effect will increasing X have on the survival time?"

Survival regression differs from ordinary (least-squares) regression in two ways. First, the response is a *time* that is often not fully observed: an item still running when the test stopped is right censored, one that was only inspected occasionally is interval censored, and one that only entered the study after it had already survived a while is left truncated (see :doc:`Types of Data`). Second, the answer is a whole *distribution* of lifetimes for each covariate value, not a single predicted number. Every model on this page is therefore defined by how the covariates change a survival distribution, and every one is fitted by a likelihood that knows about censoring and truncation.

Surpyval covers five families of regression model, distinguished by *how* the covariates act on the distribution:

 - Proportional Hazards (they multiply the hazard),
 - Accelerated Failure Time (they scale the time axis),
 - Accelerated Life (they replace the life parameter with a physical stress-life relationship),
 - Proportional Odds (they multiply the odds of survival), and
 - Additive Hazards (they add to the hazard).

Around these sit a number of variations — semi-parametric versions that leave the baseline unspecified (Cox, Lin-Ying, Buckley-James), a random-effects (frailty) version for grouped data, stratified and time-varying-covariate versions, and tree-based predictors that make no structural assumption at all. There are special cases when several of these coincide, however, it is important to understand the difference between them in general. I detail the differences in the following sections; the worked, runnable versions of everything here are on the :doc:`Regression Modelling with SurPyval` page, and the complete API is under :doc:`surpyval.regression`.

**Notation.** Throughout, :math:`T` is the (random) lifetime and :math:`x` an observed time. The covariates of one unit are a row vector :math:`Z = (z_1, \dots, z_p)` and the coefficients a column vector :math:`\beta`, so :math:`\beta' Z = \beta_1 z_1 + \dots + \beta_p z_p` is a single number, the *linear predictor*. The survival function is :math:`S(x) = P(T > x)`, the CDF :math:`F = 1 - S`, the density :math:`f`, the hazard :math:`h = f / S` and the cumulative hazard :math:`H = -\log S`. A subscript :math:`0` marks the *baseline* — the distribution of a unit whose covariates are all zero. Censoring follows the surpyval convention: ``c = 0`` observed, ``c = 1`` right censored, ``c = -1`` left censored and ``c = 2`` interval censored.

The table below is the one-line summary of each family — what the covariates do, and how to read a coefficient. The sign convention matters: in the PH, AFT and additive families a *positive* coefficient means a *shorter* life, while in the proportional odds family it means a *longer* one, and in accelerated life the stress-life function is written directly in units of life.

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Family
     - Definition
     - Reading a coefficient
   * - Proportional hazards
     - :math:`h(x \mid Z) = e^{\beta' Z} h_0(x)`
     - :math:`e^{\beta_j}` is the hazard ratio per unit of :math:`z_j`; positive shortens life.
   * - Accelerated failure time
     - :math:`S(x \mid Z) = S_0(e^{\beta' Z} x)`
     - :math:`e^{\beta_j}` is the factor by which a unit of :math:`z_j` speeds up ageing; positive shortens life.
   * - Accelerated life
     - :math:`\text{life} = \phi(Z)`, a stress-life model
     - The parameters of a physical law (activation energy, power-law exponent, ...).
   * - Proportional odds
     - :math:`\frac{S(x \mid Z)}{F(x \mid Z)} = e^{\beta' Z} \frac{S_0(x)}{F_0(x)}`
     - :math:`e^{\beta_j}` is the survival-odds ratio; positive *lengthens* life.
   * - Additive hazards
     - :math:`h(x \mid Z) = h_0(x) + \beta' Z`
     - :math:`\beta_j` is an absolute change in hazard (a rate); positive shortens life.

The families really are different, and the quickest way to see it is to look at the ratio of the hazard of a "treated" unit to the baseline hazard over time. The cell below takes one baseline (a log-normal) and applies a covariate effect of the same size through each of three families. Under proportional hazards the ratio is flat by definition; under proportional odds it starts at :math:`e^{-\beta' Z}` and fades to 1; under accelerated failure time it changes shape with the baseline:

.. jupyter-execute::

    import numpy as np
    import matplotlib.pyplot as plt
    from surpyval import LogNormal

    mu, sigma = 2.0, 0.5          # baseline log-normal
    phi = 2.0                     # the covariate effect, exp(beta'Z)
    x = np.linspace(0.5, 30, 300)
    S0, F0 = LogNormal.sf(x, mu, sigma), LogNormal.ff(x, mu, sigma)
    h0 = LogNormal.hf(x, mu, sigma)

    hr_ph = np.full_like(x, phi)                            # e^{b'Z} h0 / h0
    hr_aft = phi * LogNormal.hf(phi * x, mu, sigma) / h0    # phi h0(phi x) / h0
    hr_po = 1.0 / (F0 + (1 / phi) * S0)                     # odds of survival x 1/phi

    plt.plot(x, hr_ph, label='proportional hazards')
    plt.plot(x, hr_aft, label='accelerated failure time')
    plt.plot(x, hr_po, label='proportional odds')
    plt.axhline(1.0, color='grey', lw=0.5)
    plt.xlabel('x'); plt.ylabel('h(x | Z) / h0(x)'); plt.legend()
    plt.show()

(The proportional-odds curve uses a survival-odds multiplier of :math:`1/2` so that, like the other two, the covariate is harmful.) Deciding which of these shapes matches your data is the central modelling choice; the last section, `Validating a survival predictor`_, closes with a guide to making it.

Proportional Hazards Model
--------------------------

A proportional hazards model is one in which we change the hazard rate of the distribution by some proportional amount. You may recall that every distribution can be defined by a hazard rate or a cumulative hazard rate, see the :doc:`Handy References - Aide-mémoire` page which shows that the density, CDF, and survival function can all be defined in terms of the hazard rate, h(t).

So what we can do then is assume that the covariates will affect the survival time of the thing by having some effect on the hazard rate. The general definition for a proportional hazard model is:

.. math::

	h(t|X) = \phi(X) h_{0}(t)

This is to say that the hazard rate at time t is the function (of a vector) of covariates on a 'baseline' hazard rate. Let's use a simple example, a proportional hazard model with covariates that affect a constant hazard rate. Let's say that some factory produces one widget an hour. But this is only with one machine in operation, if we add a second machine, we can produce widgets at two per hour, if we had a third, it will be three per hour. In this case the base rate is 1 and the function linking X to the base rate is to simply multiply X by the base rate.

This is to say that for this example:

.. math::

    \phi(X) = X \\
    h_{0}(t) = 1

Therefore:

.. math::

	h(t|X) = X

This is an overly simple model, but it shows how we can construct a PH model.

In this case we have a simple proportional hazard model, also, it is limited to only an increasing hazard rate, but sometimes we need to capture a negative impact. Further, we may need a way to capture more covariates. For these reasons a very common selection for the function of covariates is an exponential function.

.. math::

	\phi(X) = e^{X\cdot \beta }

Where

.. math::

	X\cdot \beta = X_{0}\beta_{0} + X_{1} \beta_{1} + ... + X_{n-1}\beta_{n-1} + X_{n}\beta_{n}

In this case the proportional term is e raised to the power of the dot product of X and beta. Using this as the covariate function is a very common choice. This is because it will not ever become negative. It can capture situations where a covariate will increase the hazard rate if its coefficient, beta, is positive, and it will decrease the hazard rate if its coefficient is negative. Also, the dot product can capture a varying number of covariates with ease. For these reasons this log-linear form is used by the Cox model and by every parametric PH model in surpyval. Although you can choose any function for your covariates there is already likely literature about your problem which might indicate which function to use.

**What a coefficient means.** Compare two units that differ by one unit in :math:`z_j` and agree on everything else. The ratio of their hazards is

.. math::

    \frac{h(t \mid z_j + 1)}{h(t \mid z_j)} = e^{\beta_j},

at *every* time :math:`t` — that is what "proportional" means. :math:`e^{\beta_j}` is the **hazard ratio**: :math:`\beta_j = 0.7` roughly doubles the instantaneous risk of failure, :math:`\beta_j = -0.7` roughly halves it. Because the hazards stay a constant multiple apart, the survival curves are powers of one another, :math:`S(t \mid Z) = S_0(t)^{e^{\beta' Z}}`, and so they never cross. The multiplier only has meaning relative to the baseline: the baseline is the unit with :math:`Z = 0`, so centring a covariate (subtracting its mean) changes the baseline but not :math:`\beta`.

For a *parametric* proportional hazards model (a known baseline such as the Weibull) surpyval uses MLE to estimate the parameters. This is a simple conversion from regular MLE since we know the relationship between a baseline distribution and the proportional hazards version. (The Cox model in the next section is *semi-parametric* — its baseline is left unspecified and its coefficients are estimated by *partial* likelihood, not full MLE.) These relationships are:

.. math::

	f(t|X) = \phi(X) h_{0}(t) e^{-\phi(X) H_{0}(t)} \\
	\\
	F(t|X) = 1 - e^{-\phi(X) H_{0}(t)} \\
	\\
	S(t|X) = e^{-\phi(X) H_{0}(t)}

It is therefore relatively simple to adjust the MLE methods to accommodate proportional hazard models. Pre-built versions exist for the Exponential, Weibull, Normal, Gumbel, Logistic, Log-Normal and Gamma baselines (``WeibullPH`` and friends), and the ``PH(distribution)`` factory builds one from other surpyval distributions; see :doc:`regression/parametric`.

The details on fitting proportional hazards model is detailed more in the :doc:`Regression Modelling with SurPyval` page.

Maximum likelihood with censoring and truncation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every *parametric* regression family in surpyval — proportional hazards, accelerated failure time, proportional odds, parametric additive hazards and accelerated life — is fitted by the same likelihood. The families differ only in the covariate-aware :math:`f(x \mid Z)`, :math:`S(x \mid Z)` and :math:`F(x \mid Z)` they plug in; the treatment of the data is shared, which is why all of them accept the same censoring and truncation.

Each row :math:`i` contributes the probability of what was actually seen, weighted by its count :math:`n_i`:

- observed at :math:`x_i` (``c = 0``): the density :math:`f(x_i \mid Z_i)`;
- right censored at :math:`x_i` (``c = 1``): :math:`S(x_i \mid Z_i)` — it was still working;
- left censored at :math:`x_i` (``c = -1``): :math:`F(x_i \mid Z_i)` — it had already failed;
- interval censored in :math:`(x_{l,i}, x_{r,i}]` (``c = 2``): :math:`F(x_{r,i} \mid Z_i) - F(x_{l,i} \mid Z_i)`.

If the row could only have been observed inside a truncation window :math:`(t_{l,i}, t_{r,i})` — for example a unit that entered the study at age :math:`t_{l,i}` having already survived that long — the contribution is divided by the probability of landing in the window, :math:`F(t_{r,i} \mid Z_i) - F(t_{l,i} \mid Z_i)`. Putting it together,

.. math::

    \log L(\theta, \beta) = \sum_i n_i \log \bigl[\text{contribution}_i\bigr]
    - \sum_{i \in \text{truncated}} n_i \log P\bigl(t_{l,i} < T \le t_{r,i} \mid Z_i\bigr).

The truncation term is where a naive analysis goes wrong: ignoring delayed entry treats every late entrant as if it had been watched from birth, so it over-represents long lives. For a window bounded on one side only, surpyval evaluates :math:`\log S(t_l \mid Z)` or :math:`\log F(t_r \mid Z)` directly rather than as a difference of CDFs, which keeps the term finite even when the probability underflows.

The estimate :math:`(\hat\theta, \hat\beta)` maximises :math:`\log L` numerically (the baseline parameters :math:`\theta` are optimised on a transformed scale that respects their support). Parameters can be held at known values with ``fixed=``, e.g. a Weibull shape known from experience.

**Uncertainty.** The covariance of the estimates is approximated by the inverse of the observed information — the numerical Hessian of :math:`-\log L` at the optimum, :math:`\widehat{\text{Cov}} = \mathcal{I}(\hat\theta, \hat\beta)^{-1}` (fixed parameters get a zero row and column). From it:

- a Wald interval on a single parameter is formed on a scale that respects its support — the natural scale for an unbounded coefficient, the log of the distance from the bound for a positive parameter such as a Weibull scale — so the interval can never leave the parameter space;
- a band on a predicted curve at covariate :math:`Z` uses the **delta method**: the gradient :math:`g` of, say, :math:`S(x \mid Z)` with respect to all parameters gives :math:`\text{se} = \sqrt{g' \,\widehat{\text{Cov}}\, g}`, and the band is built on the logit scale for :math:`S`, :math:`F` and :math:`H` (so it stays in :math:`(0, 1)`) and on the log scale for :math:`h` and :math:`f` (so it stays positive).

Both are large-sample approximations. They are least trustworthy with few failures or with a parameter near a boundary, in which case the information matrix may not be invertible and the covariance is reported as unavailable.

Semi-Parametric
^^^^^^^^^^^^^^^

Earlier pages covered 'parametric' and 'non-parametric' survival models, so what is 'semi-parametric'? A semi-parametric model is a survival model with a non-parametric baseline and parametric function that affects that baseline. Recall that a proportional hazard model can be defined as:

.. math::

	h(t|X) = \phi(X) h_{0}(t)

It is interesting to note that the phi term must be parametric, however, the baseline hazard rate need not be parametric, it can be non-parametric! Therefore, what we have is a parametric relationship of the covariates to the baseline hazard rate, but a non-parametric baseline hazard rate, therefore, a 'semi-parametric' model.

By far the most common of any regression model of any kind (parametric, non-parametric, and semi-parametric of all the accelerated life, proportional hazard, and accelerated time) is the Cox Proportional Hazard model [Cox1972reg]_, it is a semi-parametric model.

The Cox model is used in a wide variety of fields. It has been used in criminology to study the recidivism of parolees, in engineering to understand the factors affecting tire reliability, and in medical science to understand factors affecting cancer and other diseases, among many many other applications. The wide use of the model shows the utility the model has and the broad applicability to solve problems.

**The partial likelihood.** Cox's insight is that :math:`\beta` can be estimated without ever writing down :math:`h_0`. Order the distinct failure times :math:`t_1 < t_2 < \dots`, and at each one ask: *given* that exactly one unit failed at :math:`t_k`, what is the probability that it was the one that actually did? Every unit :math:`j` still at risk has hazard :math:`h_0(t_k) e^{\beta' Z_j}`, so that probability is

.. math::

    \frac{h_0(t_k)\, e^{\beta' Z_{(k)}}}{\sum_{j \in R_k} h_0(t_k)\, e^{\beta' Z_j}}
    = \frac{e^{\beta' Z_{(k)}}}{\sum_{j \in R_k} e^{\beta' Z_j}},

where :math:`Z_{(k)}` is the covariate of the unit that failed and :math:`R_k` is the **risk set** at :math:`t_k`. The unknown baseline cancels. Multiplying over failure times gives the partial likelihood,

.. math::

    \ell(\beta) = \sum_k \Bigl[\beta' Z_{(k)} - \log \sum_{j \in R_k} n_j\, e^{\beta' Z_j}\Bigr],

with counts :math:`n_j` as weights. Only the *order* of the failure times matters, which is why the Cox model cannot tell you anything about the shape of the baseline — and why it does not need to.

**Risk sets, censoring and delayed entry.** Censored units never appear in the numerator, but they do sit in the risk sets of every failure time up to their censoring time — that is how they contribute information. A unit that entered observation late (left truncation, ``tl`` in surpyval) is at risk only after it entered. surpyval uses the standard ``(entry, exit]`` convention: unit :math:`j` is in :math:`R_k` when :math:`t_{l,j} < t_k \le x_j`, so a unit entering exactly at a failure time is not at risk for it, and a unit is at risk at its own failure or censoring time. Right and interval truncation cannot be expressed in this forward-in-time comparison, so the Cox fitter accepts left truncation only.

A tiny example makes the formula concrete. Four units fail in turn at times 1, 2, 3 and 4; the first and third are "exposed" (:math:`z = 1`). The partial log-likelihood at :math:`\beta = 0.3`, computed by hand, matches the value surpyval optimises:

.. jupyter-execute::

    from surpyval import CoxPH

    x_toy = np.array([1.0, 2.0, 3.0, 4.0])
    z_toy = np.array([1.0, 0.0, 1.0, 0.0])
    b = 0.3

    by_hand = 0.0
    for k in range(4):                        # failure k: risk set is units k..3
        risk_set = z_toy[k:]
        by_hand += b * z_toy[k] - np.log(np.exp(b * risk_set).sum())

    cox_toy = CoxPH.fit(x=x_toy, Z=z_toy.reshape(-1, 1))
    print('by hand          :', round(by_hand, 6))
    print('surpyval (-neg_ll):', round(-cox_toy.neg_ll(np.array([b])), 6))

**Tied failure times.** The argument above assumes one failure at each time. With ties (times rounded to days, inspections, genuinely discrete time) there are several conventions, chosen with ``method=``:

- **Breslow** [Breslow1974reg]_ treats the :math:`d_k` tied units as if each failed against the full risk set: the log term becomes :math:`d_k \log \sum_{j \in R_k} n_j e^{\beta' Z_j}`. Simple and fast, but it biases :math:`\hat\beta` towards zero when ties are heavy.
- **Efron** [Efron1977reg]_ assumes the tied failures happened in some unknown order and removes, on average, a fraction of their weight from the risk set for each successive one. Writing :math:`\mathcal{R}_k = \sum_{j \in R_k} n_j e^{\beta' Z_j}` and :math:`\mathcal{D}_k` for the same sum over the tied failures, the term is :math:`\sum_{l=0}^{d_k - 1} \log\bigl(\mathcal{R}_k - \tfrac{l}{d_k}\mathcal{D}_k\bigr)`. It is much closer to the exact answer at almost no cost, and it is the default in R and lifelines.
- **Exact** (``'exact'``) sums the sequential contribution over every ordering of the tied failures — appropriate when the ties come from rounding a continuous time. Its cost grows as :math:`2^{d}` in the size of a tie group, so it is limited to twelve tied failures at one time. Like the next method it needs integer counts ``n``, because each row is expanded into that many tied units.
- **Kalbfleisch-Prentice** (``'kalbfleisch-prentice'`` or ``'kp'``) is the exact *discrete-time* (conditional logistic) likelihood [KalbfleischPrentice2002reg]_, for time that really is discrete: the denominator sums the product of the risk scores over every subset of the risk set of size :math:`d_k`.

With no ties all four are identical. ``CoxPH.fit`` defaults to Breslow; ``CoxPH.fit_from_df`` and the time-varying-covariate fits default to Efron.

**Estimation and uncertainty.** The score :math:`U(\beta) = \partial \ell / \partial \beta = \sum_k \bigl(Z_{(k)} - \bar Z_k\bigr)`, where :math:`\bar Z_k` is the risk-weighted mean covariate over :math:`R_k`, is solved for zero by a root finder (with a direct minimisation as fallback). The observed information :math:`\mathcal{I}(\hat\beta) = -\partial^2 \ell / \partial\beta\,\partial\beta'` is the risk-weighted covariance of :math:`Z` summed over failures; its inverse is the covariance of :math:`\hat\beta`, and the reported ``p_values`` are the Wald tests :math:`2\bigl(1 - \Phi(|\hat\beta_j| / \text{se}_j)\bigr)`. When the design is degenerate (for example a covariate that never varies inside a risk set) the information is singular and the standard error is reported as unavailable.

**The baseline and predictions.** Once :math:`\hat\beta` is known the baseline is recovered non-parametrically by the Breslow estimator — at each distinct time, the number of failures divided by the total risk score at risk,

.. math::

    \hat h_0(t_k) = \frac{d_k}{\sum_{j \in R_k} n_j e^{\hat\beta' Z_j}}, \qquad
    \hat H_0(t) = \sum_{t_k \le t} \hat h_0(t_k), \qquad
    \hat S(t \mid Z) = \exp\bigl(-e^{\hat\beta' Z}\,\hat H_0(t)\bigr).

This is a step function that jumps only at observed times, so a Cox model predicts only within the range of the data: beyond the last observed time the curve is simply held flat, and it cannot be used to extrapolate. A parametric PH model is the tool for that. Note too that the Cox model's ``hf`` returns the *jump* :math:`\hat h_0(t_k) e^{\hat\beta' Z}` of the step at the latest time :math:`t_k \le t`, not a smooth hazard rate.

Accelerated Failure Time
------------------------

An accelerated failure time (AFT) model is very similar to a proportional hazards model. The difference is where the function is applied; instead of multiplying the hazard function, an accelerated failure time model multiplies the time by the function of covariates. The general definition is:

.. math::

	f(t|X) = \phi(X)\, f_{0}(\phi(X)t)

It is called accelerated failure time since the time term is transformed by the covariates, i.e. time is 'accelerated' by the covariates.

.. math::

	t_{a} = \phi(X)t


Just like proportional hazards, there are simple transformations that apply. Note the density carries an extra :math:`\phi(X)` factor — the Jacobian of the time change of variables — while the survival and CDF do not:


.. math::

	f(t|X) = \phi(X)\, f_{0}(\phi(X)t) \\
	\\
	F(t|X) = F_{0}(\phi(X)t) \\
	\\
	S(t|X) = S_{0}(\phi(X)t) \\
	\\
	h(t|X) = \phi(X)\, h_{0}(\phi(X)t)

Given the simple transformation of the time term the MLE is feasible with an additional transformation step. This is how surpyval estimates the parameters, using the likelihood of the previous section; pre-built versions are ``WeibullAFT``, ``LogNormalAFT`` and friends, and ``AFT(distribution)`` builds one for any distribution.

**What a coefficient means.** surpyval uses :math:`\phi(Z) = e^{\beta' Z}`, so a unit with covariates :math:`Z` "ages" :math:`e^{\beta' Z}` times faster than the baseline: it reaches at time :math:`t` the state a baseline unit reaches at :math:`e^{\beta' Z} t`. Equivalently the lifetime itself is scaled, :math:`T = T_0\, e^{-\beta' Z}`, so every quantile — the median, the B10 life — is multiplied by the **time ratio** :math:`e^{-\beta' Z}`. As with PH, a positive coefficient shortens life. Taking logs,

.. math::

    \log T = \log T_0 - \beta' Z,

which is a linear regression of log-time on the covariates, with an error distribution fixed by the baseline: a log-normal baseline gives normal errors (censored linear regression on :math:`\log T`), a Weibull baseline gives extreme-value errors, a log-logistic one logistic errors. This is the most direct interpretation of any family — "this covariate costs you 20% of your life" — and the reason AFT is often preferred in engineering, where a stress speeding up a physical process is exactly the mechanism.

**Where AFT and PH meet.** For a Weibull baseline with shape :math:`k`, :math:`H_0(x) = (x / \alpha)^k`, so

.. math::

    H(x \mid Z) = H_0\bigl(e^{\beta_{AFT}' Z} x\bigr) = e^{k\,\beta_{AFT}' Z} H_0(x),

which is a proportional hazards model with :math:`\beta_{PH} = k\,\beta_{AFT}`. The Weibull (and its special case the Exponential) is the *only* distribution that is both PH and AFT [Bagdonavicius]_: a Weibull PH fit and a Weibull AFT fit to the same data have the same likelihood and differ only in how the coefficients are scaled. For every other baseline — log-normal included — the two families are genuinely different models, and the hazard-ratio plot at the top of this page shows how.

Accelerated Life
----------------

An accelerated life model is, in many cases, simply the inverse of an accelerated time model. However, there are some cases where they are different. Consider an accelerated life model with a normal distribution:

.. math::

	F(t|X) = \Phi\left(\frac{\phi(X)t - \mu}{\sigma}\right) \\

Where :math:`\Phi` is the CDF of the standard normal distribution. In this case :math:`\mu` is the expected life of the model, however, we may instead be interested in determining what effect covariates have on the expected life of an item. In this case we can simply substitute the expected life:

.. math::

	F(t|X) = \Phi\left(\frac{t - \phi(X)}{\sigma}\right) \\

An accelerated life model is, therefore, simply a model where the life parameter of a distribution is substituted with a function of the covariates, that is, it 'accelerates' the expected life, as opposed to accelerating time as per an accelerated time model. This is the standard framework of accelerated life testing (ALT) [Meeker1998]_: units are tested at elevated stress — temperature, voltage, humidity, load — so that they fail quickly, and a physical stress-life relationship carries the result back to use conditions.

For each of the distributions in Surpyval their life parameter that varies is as per the following table. The built-in stress-life functions :math:`\phi(Z)` are written as a *life* (a time). For the Exponential, Gamma and Log-Normal, whose parameter is a rate or a log-location, surpyval converts the life to that parameter:

+------------------+---------------------------------------------------------------+
| **Distribution** | **Life Param**                                                |
+------------------+---------------------------------------------------------------+
| Weibull          | alpha                                                         |
+------------------+---------------------------------------------------------------+
| Exponential      | 1./lambda (``failure_rate`` is set to :math:`1/\phi(Z)`)      |
+------------------+---------------------------------------------------------------+
| Normal           | mu                                                            |
+------------------+---------------------------------------------------------------+
| LogNormal        | mu (set to :math:`\log \phi(Z)`, so :math:`\phi` is the       |
|                  | median life)                                                  |
+------------------+---------------------------------------------------------------+
| Gamma            | 1./beta (the rate ``beta`` is set to :math:`1/\phi(Z)`, so    |
|                  | :math:`\phi` is the scale and the mean life                   |
|                  | :math:`\alpha\,\phi(Z)`)                                      |
+------------------+---------------------------------------------------------------+
| Gumbel           | mu                                                            |
+------------------+---------------------------------------------------------------+
| Logistic         | mu                                                            |
+------------------+---------------------------------------------------------------+
| LogLogistic      | Not Avail                                                     |
+------------------+---------------------------------------------------------------+
| ExpoWeibull      | Not Avail                                                     |
+------------------+---------------------------------------------------------------+
| Uniform          | Not Avail                                                     |
+------------------+---------------------------------------------------------------+
| Beta             | Not Avail                                                     |
+------------------+---------------------------------------------------------------+

Given the simple substitution into the life parameter, surpyval uses MLE to calculate the parameters: the remaining distribution parameters (for a Weibull, the shape) are shared across all stress levels — the assumption that the failure *mechanism* is the same at every stress and only its speed changes — and the life-model parameters replace the life parameter. The life parameter itself is kept in the parameter vector as a fixed placeholder (reported as ``1.0``), since its value now comes from :math:`\phi(Z)`. To start the search, surpyval fits the distribution separately at each distinct stress and regresses those lives on the stress, so the data need at least two distinct stress levels.

The built-in stress-life relationships (all are ``LifeModel`` instances, and a custom one can be written by subclassing ``LifeModel``):

.. list-table::
   :header-rows: 1
   :widths: 26 34 40

   * - Life model
     - :math:`\phi(Z)`
     - Typical use
   * - ``Power``
     - :math:`a Z^{n}`
     - Mechanical load, voltage (with :math:`n < 0`)
   * - ``InversePower``
     - :math:`1 / (a Z^{n})`
     - The inverse power law of voltage endurance and fatigue
   * - ``ExponentialLifeModel``
     - :math:`b\, e^{a / Z}`
     - Arrhenius: temperature (in kelvin) with :math:`a = E_a / k_B`
   * - ``InverseExponential``
     - :math:`1 / (b\, e^{a / Z})`
     - Reciprocal of the above
   * - ``Eyring``
     - :math:`\frac{1}{Z} e^{-(c - a/Z)}`
     - Temperature, with a :math:`1/Z` pre-factor from reaction-rate theory
   * - ``InverseEyring``
     - reciprocal of Eyring
     - Reciprocal of the above
   * - ``Linear``
     - :math:`a + b Z`
     - First-order approximation over a narrow stress range
   * - ``DualExponential``
     - :math:`c\, e^{a / Z_1} e^{b / Z_2}`
     - Two thermal-type stresses (e.g. temperature and humidity)
   * - ``DualPower``
     - :math:`c\, Z_1^{m} Z_2^{n}`
     - Two non-thermal stresses
   * - ``PowerExponential``
     - :math:`c\, e^{a / Z_1} Z_2^{n}`
     - One thermal and one non-thermal stress

**Accelerated life versus AFT.** For a Weibull, substituting a log-linear life :math:`\alpha(Z) = e^{a + b' Z}` gives :math:`S(t \mid Z) = \exp\bigl(-(t e^{-b'Z} / e^{a})^{k}\bigr)`, which is exactly an AFT model with :math:`\beta = -b`. So for scale-family distributions the two coincide under a log-linear link, with opposite signs: an accelerated life coefficient says how much *life* a unit of stress buys, an AFT coefficient how much *faster* it ages. They part company for location-family distributions (Normal, Gumbel, Logistic), where accelerated life shifts the location but AFT rescales time, and whenever the stress-life relationship is not log-linear — which is the point of having physically motivated life models. The distinction follows [Bagdonavicius]_; see also :doc:`Handy References - Aide-mémoire`.

The practical caution is extrapolation. The fitted life model is used precisely *outside* the tested stresses, so its form is an assumption about physics that the data can only weakly check: Arrhenius and a power law can fit three test temperatures equally well and still disagree substantially at use conditions, and the disagreement grows the further the use stress is from the tested range. Compare candidate life models, and prefer the one with a mechanism behind it.

Proportional Odds
-----------------

A proportional odds model acts on the *odds* rather than on the hazard [Bennett1983reg]_. In surpyval it multiplies the odds of *survival* by time :math:`x`, :math:`O(x) = S(x) / F(x)`, by :math:`e^{\beta' Z}`:

.. math::

    \frac{S(x \mid Z)}{F(x \mid Z)} = e^{\beta' Z}\, \frac{S_{0}(x)}{F_{0}(x)}.

(Writing it for the odds of *failure*, :math:`F/S`, is the same model with the sign of :math:`\beta` flipped.) Solving for the survival function and differentiating,

.. math::

    S(x \mid Z) = \frac{e^{\beta' Z} S_0(x)}{F_0(x) + e^{\beta' Z} S_0(x)}, \qquad
    h(x \mid Z) = \frac{h_0(x)}{F_0(x) + e^{\beta' Z} S_0(x)} .

**What a coefficient means.** :math:`e^{\beta_j}` is the ratio of the odds of surviving past any given time, per unit of :math:`z_j`. Because it multiplies the odds of *survival*, a **positive** coefficient **lengthens** life — the opposite sign from the PH and AFT families. Keep this in mind when comparing fits: on the same data a PO model reports coefficients of the opposite sign to a PH model.

Its defining feature is that the covariate effect *decays* over time. The hazard ratio is

.. math::

    \frac{h(x \mid Z)}{h_0(x)} = \frac{1}{F_0(x) + e^{\beta' Z} S_0(x)},

which starts at :math:`e^{-\beta' Z}` when :math:`x` is small (:math:`S_0 \approx 1`) and tends to 1 as :math:`x \to \infty` (:math:`S_0 \to 0`). Two survival curves under a proportional odds model converge rather than staying a constant multiple apart. This makes it the natural choice when a treatment or covariate matters early but its influence fades, a pattern proportional hazards cannot represent.

Two baselines make it especially natural. With a log-logistic baseline the survival odds are :math:`(x/\alpha)^{-k}`, and multiplying by :math:`e^{\beta' Z}` is the same as rescaling :math:`\alpha` — so a log-logistic PO model is also an AFT model, the log-logistic's counterpart of the Weibull's PH/AFT coincidence. With a logistic baseline the survival odds are :math:`e^{-(x - \mu)/\sigma}`, and the multiplier shifts :math:`\mu` by :math:`\sigma \beta' Z` — a location shift of the whole distribution. Pre-built versions (``LogisticPO``, ``WeibullPO``, ...) and the ``PO(distribution)`` factory are fitted by the same censored and truncated likelihood as the other parametric families. Proportional odds has no additive structure over time, so it is not available with time-varying covariates.

Additive Hazards
----------------

Where proportional hazards *multiplies* the baseline hazard, an additive hazards model *adds* to it:

.. math::

    h(t \mid X) = h_{0}(t) + \beta \cdot X.

The covariate shifts the absolute hazard by a constant amount at every time, rather than scaling it. This is often the more natural scale for risk-difference questions (excess deaths per unit time attributable to an exposure), and for reliability settings where hazards from separate mechanisms genuinely add. Integrating, the cumulative hazard is :math:`H(t \mid Z) = H_0(t) + t\, \beta' Z`, so :math:`S(t \mid Z) = S_0(t)\, e^{-t \beta' Z}`. A coefficient is a *rate*: :math:`\beta_j = 0.01` per hour means one extra failure per hundred unit-hours for each unit of :math:`z_j`, whatever the baseline is doing.

**The Lin-Ying estimator.** Like Cox, the Lin-Ying form [LinYing1994reg]_ leaves the baseline hazard unspecified, but unlike Cox it admits a *closed-form* estimator for :math:`\beta` — no iteration and nothing to converge. With :math:`Y_i(t)` the at-risk indicator, :math:`N_i(t)` the failure counting process and :math:`\bar Z(t)` the mean covariate among those at risk,

.. math::

    \hat\beta = A^{-1} b, \qquad
    A = \sum_i \int_0^{\tau} Y_i(t) \bigl(Z_i - \bar Z(t)\bigr)^{\otimes 2}\, dt, \qquad
    b = \sum_i \int_0^{\tau} \bigl(Z_i - \bar Z(t)\bigr)\, dN_i(t).

:math:`A` is the spread of the covariates in the risk set integrated over *time* — how long each covariate configuration was exposed — and :math:`b` compares the covariates of those who failed with those at risk. The variance is the Lin-Ying sandwich :math:`A^{-1} B A^{-1}`, with :math:`B = \sum_i \int (Z_i - \bar Z)^{\otimes 2} dN_i`, from which the standard errors and Wald ``p_values`` follow. The baseline cumulative hazard is a Breslow-type step sum corrected for the covariate drift, :math:`\hat H_0(t) = \sum_{t_k \le t} d_k / |R_k| - \int_0^t \bar Z(s)' \hat\beta\, ds`. Because a step function has no rate, the hazard :math:`h(t \mid Z)` is reported with a kernel-smoothed baseline (an Epanechnikov kernel over the increments of :math:`\hat H_0`, bandwidth by a normal-reference rule unless you pass ``bandwidth=``), which is least accurate near the ends of the observed time range. The semi-parametric fitter handles observed and right-censored data.

**The parametric version.** ``AH(distribution)`` and the pre-built ``WeibullAH``, ``ExponentialAH``, ... use a parametric baseline, :math:`h(x \mid Z) = h_0(x; \theta) + \beta' Z`, and are fitted by the censored and truncated likelihood of the proportional hazards section. They give a smooth, extrapolatable version of the same model.

**The positivity caveat.** Nothing constrains :math:`h_0(t) + \beta' Z` to be positive. For a strongly protective covariate the additive hazard can go negative — impossible for a real hazard. The semi-parametric estimate is returned unclamped, so a fitted survival that rises above 1 is the symptom; the parametric likelihood needs :math:`\log h` at every failure, so a fit whose optimum would need a negative hazard raises an error instead of returning an invalid model. When effects are strongly protective, the exponential link of proportional hazards, which keeps the hazard positive by construction, is the safer choice.

Semi-Parametric — Buckley-James
-------------------------------

Cox leaves the baseline *hazard* unspecified; Buckley-James [BuckleyJames1979reg]_ is the accelerated-failure-time counterpart that leaves the *error distribution* unspecified. It fits an AFT model by iterating between imputing the censored failure times from the current fit (using the Kaplan-Meier residual distribution) and re-estimating the coefficients by least squares on the completed data. The result is a semi-parametric AFT: covariate effects on the log-time scale without committing to a parametric family for the baseline.

In detail, write the model as :math:`\log T = \gamma' Z + \varepsilon` with :math:`\varepsilon` from an arbitrary distribution. If there were no censoring, least squares of :math:`y_i = \log x_i` on :math:`Z_i` would be the obvious estimator. A censored :math:`y_i` is only a lower bound, so Buckley-James replaces it by its conditional expectation given that it is at least that large:

.. math::

    \hat y_i = \delta_i\, y_i + (1 - \delta_i)\Bigl[\gamma' Z_i + \frac{\sum_{e_k > e_i} e_k\, \Delta \hat F(e_k)}{\hat S(e_i)}\Bigr],

where :math:`e_i = y_i - \gamma' Z_i` are the residuals, :math:`\hat S` is their Kaplan-Meier survival function and :math:`\Delta\hat F(e_k)` its jumps. The two steps — impute, then refit :math:`\gamma` by least squares on the :math:`\hat y_i` — alternate until :math:`\gamma` stops moving. Some details matter in practice:

- the largest residual is treated as a failure (Efron's tail correction), so the residual distribution is proper and every conditional mean is finite;
- the intercept is not identified separately from the error distribution, so only the slopes are estimated and the location lives in the residual distribution;
- the iteration can settle into a two-point cycle rather than a fixed point, a known property of the estimator; surpyval detects the cycle and averages it, and warns if the iteration has not converged;
- there is no likelihood, so there are no model-based standard errors: uncertainty comes from a percentile bootstrap.

surpyval reports the coefficients as :math:`\beta = -\gamma`, in the same sign convention as ``WeibullAFT`` (positive shortens life), and predicts with the residual distribution directly, :math:`S(t \mid Z) = \hat S_\varepsilon(\log t + \beta' Z)` — a step function, like a Kaplan-Meier. The fitter accepts observed and right-censored data with positive times.

Checking a proportional hazards fit
-----------------------------------

Every proportional hazards model rests on one assumption: that a covariate multiplies the baseline hazard by a *constant* factor for all time. If that is false — a treatment that helps early but not late, a covariate whose effect drifts — the single coefficient the model reports is a time-average that can be misleading.

The assumption is checked with the **Schoenfeld residuals**. At each event time the Schoenfeld residual for a covariate is the observed covariate value of the subject who failed minus the risk-weighted mean covariate value over everyone still at risk. If proportional hazards holds, these residuals have no trend in time; if the effect is drifting, they trend. The **Grambsch-Therneau test** [GrambschTherneau1994reg]_ formalises this by regressing the *scaled* Schoenfeld residuals on a transform of time and testing for a non-zero slope, both per covariate and jointly. A small :math:`p`-value is evidence *against* proportional hazards.

**The precise definitions.** For the unit failing at :math:`t_k` the Schoenfeld residual is

.. math::

    r_k = Z_{(k)} - \bar Z(t_k), \qquad
    \bar Z(t_k) = \frac{\sum_{j \in R_k} n_j e^{\hat\beta' Z_j} Z_j}{\sum_{j \in R_k} n_j e^{\hat\beta' Z_j}},

(for an Efron fit the mean is averaged over the Efron-reduced risk sets of the tie group, so the residuals are consistent with the likelihood that was maximised). Grambsch and Therneau showed that, if the coefficient is really a function of time :math:`\beta_j(t)`, the **scaled** residual :math:`r^*_k = \hat\beta + d\, V r_k` — with :math:`d` the number of failures and :math:`V = \mathcal{I}^{-1}` the coefficient covariance — has expectation approximately :math:`\beta(t_k)`. Plotting :math:`r^*_k` against time is therefore a picture of how the coefficient changes, centred on the fitted constant. For a time transform :math:`g` (``transform=`` in ``check_ph``: ``"km"``, the default, uses :math:`g(t) = 1 - \hat S_{KM}(t)` from a Kaplan-Meier fitted to all the data; ``"rank"`` uses average ranks; ``"identity"`` and ``"log"`` use :math:`t` and :math:`\log t`), let :math:`u = \sum_k (g_k - \bar g)\, r_k` and :math:`s = \sum_k (g_k - \bar g)^2`. Then

.. math::

    T_{\text{global}} = \frac{d}{s}\, u' V u \sim \chi^2_p, \qquad
    T_j = \frac{d\, (V u)_j^2}{s\, V_{jj}} \sim \chi^2_1,

the global and per-covariate statistics of R's ``cox.zph`` and lifelines, which surpyval reproduces. The ``"km"`` transform is the usual choice: it spreads the failures evenly and is not dominated by a few long times. The per-covariate tests are screens; with several covariates, look at the global test and at the plots.

When the test rejects, the usual remedies are to **stratify** on the offending covariate (if it is a nuisance), to let its effect change over time by including an interaction with a function of time as a time-varying covariate, or to move to a family whose effect is not constant in time (AFT, proportional odds).

SurPyval exposes several other residuals for a fitted Cox model, each answering a different question: **martingale** residuals (observed minus expected events, :math:`M_i = \delta_i - e^{\hat\beta' Z_i}\bigl(\hat H_0(x_i) - \hat H_0(t_{l,i})\bigr)`, where :math:`\delta_i = 1` for a failure; an Efron fit credits a tied failure with only its share of the baseline step at its own time) reveal non-linear covariate functional form when plotted against a covariate; **deviance** residuals, a symmetrised transform of the martingale residuals, highlight poorly-predicted individuals; **score** residuals are each observation's contribution to :math:`U(\hat\beta)`, and **dfbeta** residuals (score residuals times :math:`V`) approximate how much each observation moves :math:`\hat\beta`. The Schoenfeld, score and martingale residuals all sum to zero at the maximum of the partial likelihood. All of them use the tie method of the fit and respect delayed entry. See [TherneauGrambsch2000reg]_ for a thorough treatment.

Cluster-robust standard errors
------------------------------

The model-based standard errors assume every observation is independent. When the data are *clustered* — repeated events on the same subject, several failures from one machine, grouped sampling — that assumption is wrong and the naive errors are too small. The **Lin-Wei sandwich** [LinWei1989reg]_ (or "robust") variance corrects for it. Writing :math:`H` for the information matrix and :math:`s_c` for the sum of a cluster's score contributions, the robust covariance is

.. math::

    V_{\text{robust}} = H^{-1} \left( \sum_{c} s_c s_c^{\top} \right) H^{-1},

which reduces to the usual variance when there is one observation per cluster and there is no within-cluster correlation. Since the dfbeta residual of an observation is its score residual times :math:`H^{-1}`, this is the same as summing the dfbeta residuals within each cluster, :math:`D_c = H^{-1} s_c`, and forming :math:`\sum_c D_c D_c^{\top}` — which is how surpyval computes it.

The intuition for why clustering matters: if every observation were accidentally entered twice, a naive analysis would think it had twice the data and shrink the standard errors by :math:`\sqrt 2`; declaring each pair a cluster tells the sandwich that the copies carry no new information, and the error returns to its correct size. For a start-stop (time-varying-covariate) Cox fit the rows of one subject are correlated by construction, so the robust variance clusters by subject unless told otherwise.

Shared frailty
--------------

Cluster-robust errors *correct* for within-cluster correlation but do not *model* it. A **shared-frailty** model does the opposite: it introduces the correlation explicitly through an unobserved random effect [Hougaard2000reg]_. Each group :math:`g` (a manufacturing lot, a site, a repairable unit) is given a **frailty** :math:`u_g` — a random multiplier shared by every member of the group — acting on the hazard:

.. math::

    h\bigl(t \mid Z, u_g\bigr) = u_g \, h_0(t) \, e^{\beta' Z},

with the frailties drawn once per group from a Gamma distribution of mean 1 and variance :math:`\theta`. The frailty is the survival analogue of a random intercept: it absorbs whatever unmeasured feature makes a whole group fail faster or slower than its covariates predict, and :math:`\theta` measures that between-group variability (:math:`\theta = 0` recovers ordinary proportional hazards). This is the *conditional* / random-effects counterpart of the *marginal* cluster-robust correction above — same within-group correlation, modelled rather than merely accounted for.

Because the frailty multiplies the *cumulative* hazard, a Gamma frailty integrates out of a group's likelihood in closed form. Writing :math:`D_g` for the number of events in group :math:`g` and :math:`H_g = \sum_{j \in g} e^{\beta' z_j} H_0(t_j)` for the sum of its members' cumulative hazards, the group contributes

.. math::

    \sum_{\text{events}} \log\!\bigl(h_0 \, e^{\beta' Z}\bigr)
    - \tfrac{1}{\theta}\log\theta - \log\Gamma\!\bigl(\tfrac{1}{\theta}\bigr)
    + \log\Gamma\!\bigl(D_g + \tfrac{1}{\theta}\bigr)
    - \bigl(D_g + \tfrac{1}{\theta}\bigr)\log\!\bigl(H_g + \tfrac{1}{\theta}\bigr)

to the marginal log-likelihood, which is maximised jointly over the baseline parameters, :math:`\beta`, and :math:`\theta`. The same conjugacy makes the **posterior** frailty of an observed group a closed form, :math:`\hat u_g = (D_g + 1/\theta)/(H_g + 1/\theta)` — an empirical-Bayes estimate shrunk toward 1, larger for groups that fail early. Standard errors come from the numerical Hessian of the marginal likelihood; the Wald interval for :math:`\theta` (and for the positive baseline parameters) is formed on the log scale so it stays positive. The closed form requires observed and right-censored data only.

The distinction between the two curves the model can draw matters. Integrating the frailty out gives the **marginal** (population-averaged) survival of a unit from an *unknown* group, :math:`S(t \mid Z) = (1 + \theta \, e^{\beta' Z} H_0(t))^{-1/\theta}` — a Laplace transform of the frailty distribution, and always heavier-tailed than the baseline. Conditioning on a value :math:`u` gives :math:`S(t \mid Z, u) = e^{-u \, e^{\beta' Z} H_0(t)}`, used with :math:`\hat u_g` to predict a *new* member of an *already-observed* group. A subtle consequence is that a mixture of groups makes the **population** hazard bend down over time even when every group's hazard rises, because the frail groups fail first and leave robust survivors — so an apparent decreasing hazard can be a heterogeneity artifact rather than a real one. Identification requires within-group replication: with a single group, or one observation per group, :math:`\theta` is confounded with the baseline shape and cannot be estimated.

The same selection effect changes what the coefficients mean. The marginal hazard is

.. math::

    h(t \mid Z) = \frac{e^{\beta' Z} h_0(t)}{1 + \theta\, e^{\beta' Z} H_0(t)},

so the *population* hazard ratio between two covariate values starts at :math:`e^{\beta}` and shrinks towards 1 over time, even though *within* every group it is exactly :math:`e^{\beta}`. :math:`\beta` in a frailty model is a within-group (conditional) effect, and it is typically larger in magnitude than the coefficient an ordinary PH fit reports on the same data. The cell below shows the bending for a baseline whose hazard increases, with :math:`\theta = 1`:

.. jupyter-execute::

    from surpyval import Weibull

    t = np.linspace(0.01, 30, 300)
    h0, H0 = Weibull.hf(t, 10, 1.5), Weibull.Hf(t, 10, 1.5)
    for theta in [0.0, 0.5, 1.0]:
        plt.plot(t, h0 / (1 + theta * H0), label=f'theta = {theta:g}')
    plt.xlabel('t'); plt.ylabel('population (marginal) hazard'); plt.legend()
    plt.show()

Every group's hazard is the rising :math:`\theta = 0` curve times its own :math:`u_g`, yet the population hazard for :math:`\theta = 1` rises and then falls. As :math:`\theta \to 0` the marginal cumulative hazard :math:`\log(1 + \theta e^{\beta' Z} H_0)/\theta` tends continuously to the proportional-hazards one, :math:`e^{\beta' Z} H_0(t)`, and surpyval's marginal predictions switch to that limit when :math:`\theta` is numerically zero. Because :math:`\theta` cannot be negative, data with no between-group heterogeneity push the estimate onto that boundary, where the Wald interval is no longer meaningful. A :math:`\hat\theta` at or very near zero says the grouping explains nothing beyond the covariates: fit and report the ordinary proportional-hazards model instead, and compare its coefficients with the frailty fit's.

Stratification
--------------

When proportional hazards fails for a *nuisance* covariate — a study site, a batch, a device generation you would rather not model — the standard remedy is **stratification**: allow a separate baseline hazard :math:`h_{0,g}(t)` for each stratum :math:`g` while sharing the coefficients :math:`\beta`. Because the partial likelihood is summed *within* strata, risk sets never cross a stratum boundary and the nuisance factor is removed from the comparison without ever estimating its effect. The Cox partial likelihood factorises across strata, so this is a small change to the estimation with a large gain in robustness:

.. math::

    \ell(\beta) = \sum_{g} \ell_g(\beta),

each :math:`\ell_g` being the ordinary partial likelihood of stratum :math:`g` alone. There is one Breslow baseline per stratum, so a prediction must say which stratum it is for. The price is that the stratifying variable gets no coefficient — you learn nothing about its effect, which is the point when it is a nuisance and a loss when it is not. The residual diagnostics and robust variance above assume a single baseline, so they are not available on a stratified fit.

Time-varying covariates
-----------------------

A covariate can change *during* a subject's follow-up — a dose is raised, a treatment begins, a machine is moved to a harsher environment. Such a covariate path is represented in the **counting-process (start-stop) format**: each subject contributes one row per interval :math:`(x_l, x_r]` on which its covariate vector is constant, and only the interval ending at the subject's event carries the terminal status. surpyval writes this with its usual vocabulary — the subject id ``i``, the interval bounds ``xl`` / ``xr``, and the censoring flag ``c`` (``0`` event, ``1`` right-censored interval end). A subject's rows may not overlap, it may have at most one event, and that event must be on its last interval; gaps between intervals are allowed for the hazard-based families (the subject is simply not at risk in the gap). Equivalently the path can be given as a **timeline** — one row per covariate change, each value holding until the next row, with the first time the entry and the last row carrying the exit time and status — which surpyval expands into the same intervals.

**Fitting.** Whether a subject's episodes can be fitted as if they were independent observations depends on the family. Where the cumulative hazard is *additive over disjoint intervals* — proportional hazards, additive hazards, and (semi-parametric) Cox — a subject splits **exactly** into one left-truncated (delayed-entry) observation per constant-covariate interval:

.. math::

    H\bigl(x_r \mid Z\bigr) - H\bigl(x_l \mid Z\bigr)

is the interval's contribution and the subject's likelihood is the product over its intervals, so the ordinary maximum-likelihood fitter recovers the same estimate from the reshaped rows (the *episode-splitting identity*). Accelerated failure time is different: the covariate rescales the *time axis*, so the baseline is evaluated at the subject's accumulated **accelerated age** :math:`\psi = \sum_k e^{\beta' z_k}\,(b_k - a_k)`, and the episode entry ages in that sum depend on :math:`\beta`. The episodes therefore do not separate into independent rows, and AFT is fitted with a dedicated accumulated-age likelihood that re-accumulates :math:`\psi` per subject on each optimiser step. Proportional odds has neither structure and is not fitted from time-varying data.

For AFT the subject's contribution is

.. math::

    \bigl[e^{\beta' z_{\text{last}}}\, h_0(\psi)\bigr]^{\delta}\, e^{-H_0(\psi)},
    \qquad \psi = \int_0^{T} e^{\beta' Z(u)}\, du ,

with :math:`\delta = 1` for a subject whose last interval ends in failure. The integral runs from time zero, which has a consequence: the accelerated age accumulated *before* a subject entered observation, or during a gap in its record, depends on covariate values that were never observed. Rather than guess them, surpyval's AFT time-varying fit requires each subject's intervals to start at 0 and to be contiguous, and refuses delayed entry and gaps with an error that points to the Cox time-varying fit, whose risk-set likelihood never needs the unobserved history. Information criteria for this fit count subjects, not interval rows.

**Evaluation.** Given an already-fitted model and a covariate path :math:`Z(t)`, the survival :math:`S(t \mid Z(\cdot))` is exact when the path is **piecewise-constant** (a step function). For proportional and additive hazards (and Cox) it is the sum of the per-segment cumulative-hazard increments; for AFT it is the baseline evaluated at the accumulated accelerated age. A continuously-varying covariate would break the exactness of these segment sums, so the step-valued requirement is a correctness precondition, not a convenience. Unlike some packages, surpyval is willing to evaluate a model along a *future* covariate path — because that path is *supplied* as a plan or hypothesis (mission phases, a duty cycle, a scheduled load), making :math:`S(t \mid Z(\cdot))` a well-posed conditional question rather than a claim to know the future.

Two details of the evaluation are worth knowing. For Cox, the baseline hazard jumps at observed event times, and a jump that falls exactly on a covariate change time is weighted by the *old* covariate — the same ``(x_l, x_r]`` convention as the fit, where a unit is still at risk at the end of its interval. And conditional survival — the probability of surviving to :math:`x` for a unit already known to have survived to age :math:`g` along the same path — is

.. math::

    S(x \mid T > g, Z(\cdot)) = \exp\bigl(-[H(x \mid Z(\cdot)) - H(g \mid Z(\cdot))]\bigr), \qquad x \ge g .

**Identifiability.** A time-varying covariate's effect is estimated from contrast between units at the same time: at each failure, did the unit that failed carry a different covariate value from the others still at risk? If every unit follows the same covariate path on the same clock, the covariate is perfectly confounded with time and its coefficient cannot be separated from the baseline. Staggered starts, different schedules and idle periods are what make the effect measurable.

Validating a survival predictor
-------------------------------

Information criteria (AIC, BIC) compare how well models fit the data they were trained on. To judge how well a model *predicts*, it must be scored on held-out data, and the metrics must account for censoring. Two right-censored-standard measures are used, both handling censoring by inverse-probability-of-censoring weighting (IPCW):

- The **Brier score** :math:`BS(t)` is the weighted mean squared error between the predicted survival :math:`S(t \mid Z)` and the survival indicator :math:`\mathbb{1}(T > t)`; the **integrated Brier score** averages it over a time grid. Lower is better, and a useful model scores below the marginal Kaplan-Meier reference.
- The **time-dependent AUC** (Uno's cumulative/dynamic estimator) measures discrimination as a function of the horizon — the probability that a subject who has failed by :math:`t` was assigned a higher risk than one still event-free. 0.5 is chance, 1.0 is perfect.

IPCW works by up-weighting the subjects whose outcome is known at :math:`t` by the inverse of their probability of not having been censored yet, estimated by a Kaplan-Meier of the *censoring* times, so the scored sample stands in for the full one. The metrics are model-agnostic: they take a matrix of predicted survival probabilities, so parametric, semi-parametric and tree-based predictors can be compared on the same footing (see :doc:`comparison_and_validation`).

Concordance
^^^^^^^^^^^

The oldest discrimination measure is **Harrell's concordance index** [Harrell1982reg]_. It asks, over every *comparable* pair of subjects, whether the model ranked them correctly: the one that failed first should have the higher risk score. A pair is comparable when the earlier time is an observed failure — if the earlier subject was censored we do not know who failed first. Then

.. math::

    C = \frac{\#\{\text{concordant pairs}\} + \tfrac12\, \#\{\text{pairs with tied scores}\}}{\#\{\text{comparable pairs}\}},

with 0.5 for a random ranking and 1 for a perfect one. Ties in *time* need conventions, and surpyval's are: a failure and a censoring at the same time form a fully comparable pair, because the censored subject is known to have outlived the failure (so a lower score for the failure counts 0, not 0.5); two failures at the same time count 1 if their scores tie and 0.5 otherwise; two censorings at the same time are not comparable. For a proportional hazards model the linear predictor :math:`\beta' Z` is a natural risk score. :math:`C` measures ranking only — a model can have an excellent :math:`C` and badly miscalibrated survival probabilities, which is what the Brier score is for. A four-subject example shows the counting:

.. jupyter-execute::

    from surpyval.utils.score import score

    x_c = [1.0, 2.0, 3.0, 4.0]
    c_c = [0, 1, 0, 0]              # the second subject is censored at 2
    risk = [0.9, 0.2, 0.5, 0.1]     # higher score = expected to fail earlier
    # comparable pairs: (1,2) (1,3) (1,4) (3,4); (2,*) are not, 2 was censored
    # all four are ranked correctly
    score(x_c, c_c, risk)

Survival trees and forests
^^^^^^^^^^^^^^^^^^^^^^^^^^

All the families above impose a structure — a link function, a linear predictor. A **survival tree** imposes none: it recursively splits the data on one covariate at a time, at the threshold that best separates the survival of the two halves, and fits a survival model in each leaf [LeBlancCrowley1993reg]_. A prediction for a new unit drops it down the tree to a leaf and returns that leaf's distribution. Trees find interactions and thresholds by themselves ("high temperature matters, but only for the old design"), at the cost of being step functions of the covariates and of being unstable — a slightly different sample can grow a different tree.

surpyval's trees couple the split rule with the leaf model (``kind=``):

- ``"non-parametric"`` scores a candidate split by the standardised **log-rank** statistic between the two children,

  .. math::

      L = \frac{\sum_j \bigl(d_{j,L} - Y_{j,L}\, d_j / Y_j\bigr)}
      {\sqrt{\sum_j \frac{Y_{j,L}}{Y_j}\Bigl(1 - \frac{Y_{j,L}}{Y_j}\Bigr)\frac{Y_j - d_j}{Y_j - 1}\, d_j}},

  summed over the pooled distinct times, with :math:`d` deaths and :math:`Y` numbers at risk (left child and total). The at-risk counts use the same ``(entry, exit]`` convention as everywhere else, so left truncation is handled; the leaves are Nelson-Aalen estimates. A risk-set statistic only exists for observed and right-censored data, so this kind rejects left and interval censoring and right truncation.
- ``"weibull"`` (the default) and ``"exponential"`` score a split by the gain in the maximised full log-likelihood of a Weibull (or exponential [DavisAnderson1989reg]_) model in each child. Because it uses the full likelihood of the proportional hazards section, this works for every kind of censoring and truncation; the leaves are the fitted Weibull or exponential models.

A **random survival forest** [Ishwaran2008reg]_ averages many trees, each grown on a bootstrap resample of the data and allowed to consider only a random subset of the covariates at each split. The averaging trades the high variance of a single deep tree for a little bias, and usually predicts much better. The forest's survival curve is the average of the trees' leaf survival curves (or, optionally, the survival implied by their averaged cumulative hazards), and its risk score for concordance is the leaf cumulative hazard summed over the evaluation times. In surpyval both live in ``surpyval.beta.ml`` — tested and usable, but with an interface that may still change (see :doc:`surpyval.beta`).

Choosing a model
^^^^^^^^^^^^^^^^

There is no universally best family, but the questions below settle most cases.

- **Is the effect a constant multiple of the risk?** Fit a Cox model and run the Grambsch-Therneau test. If it passes, proportional hazards is a defensible and very interpretable default, and a Cox fit makes no assumption about the baseline.
- **Do you need to extrapolate in time, or predict the full distribution?** Cox cannot predict beyond the observed times. Use a parametric family, and choose the baseline by likelihood (AIC/BIC) and by comparing the fitted curves with non-parametric estimates.
- **Is the mechanism "the covariate speeds the clock up"?** Use AFT — a stress that accelerates a physical process — or, when the covariate is a controlled stress with a known physical law and you must extrapolate to use conditions, accelerated life. Remember that for a Weibull, AFT and PH are the same model.
- **Does the effect fade over time?** Proportional odds represents exactly that; a PH model forced onto such data reports a time-averaged hazard ratio.
- **Is the question about excess risk?** Additive hazards reports risk differences, in units of failures per unit time.
- **Are units grouped?** For honest standard errors, cluster-robust errors; to model and quantify the between-group variation (and to predict for a known group), a shared frailty; for a nuisance grouping that violates PH, stratification.
- **Do covariates change during follow-up?** The start-stop format, with Cox, PH, AH or AFT.
- **Is the covariate structure unknown or strongly non-linear?** A random survival forest, validated on held-out data, and compared against a simpler model on the same metrics.

Whatever the choice, check it: residuals and the PH test for the assumption, information criteria for the fit, and held-out Brier scores, AUC and concordance for prediction. For worked examples on how to do regression analysis — including checking the proportional hazards assumption, robust and stratified fits, and validating predictions — see the :doc:`Regression Modelling with SurPyval` page.

.. rubric:: References

.. [Cox1972reg] Cox, D.R., 1972. Regression models and life-tables. *Journal of the Royal Statistical Society: Series B*, 34(2), pp.187-220.

.. [Breslow1974reg] Breslow, N., 1974. Covariance analysis of censored survival data. *Biometrics*, 30(1), pp.89-99.

.. [Efron1977reg] Efron, B., 1977. The efficiency of Cox's likelihood function for censored data. *Journal of the American Statistical Association*, 72(359), pp.557-565.

.. [KalbfleischPrentice2002reg] Kalbfleisch, J.D. and Prentice, R.L., 2002. *The Statistical Analysis of Failure Time Data*, 2nd ed. Wiley.

.. [Bennett1983reg] Bennett, S., 1983. Analysis of survival data by the proportional odds model. *Statistics in Medicine*, 2(2), pp.273-277.

.. [LinYing1994reg] Lin, D.Y. and Ying, Z., 1994. Semiparametric analysis of the additive risk model. *Biometrika*, 81(1), pp.61-71.

.. [BuckleyJames1979reg] Buckley, J. and James, I., 1979. Linear regression with censored data. *Biometrika*, 66(3), pp.429-436.

.. [GrambschTherneau1994reg] Grambsch, P.M. and Therneau, T.M., 1994. Proportional hazards tests and diagnostics based on weighted residuals. *Biometrika*, 81(3), pp.515-526.

.. [TherneauGrambsch2000reg] Therneau, T.M. and Grambsch, P.M., 2000. *Modeling Survival Data: Extending the Cox Model*. Springer.

.. [LinWei1989reg] Lin, D.Y. and Wei, L.J., 1989. The robust inference for the Cox proportional hazards model. *Journal of the American Statistical Association*, 84(408), pp.1074-1078.

.. [Hougaard2000reg] Hougaard, P., 2000. *Analysis of Multivariate Survival Data*. Springer.

.. [Harrell1982reg] Harrell, F.E., Califf, R.M., Pryor, D.B., Lee, K.L. and Rosati, R.A., 1982. Evaluating the yield of medical tests. *JAMA*, 247(18), pp.2543-2546.

.. [LeBlancCrowley1993reg] LeBlanc, M. and Crowley, J., 1993. Survival trees by goodness of split. *Journal of the American Statistical Association*, 88(422), pp.457-467.

.. [DavisAnderson1989reg] Davis, R.B. and Anderson, J.R., 1989. Exponential survival trees. *Statistics in Medicine*, 8(8), pp.947-961.

.. [Ishwaran2008reg] Ishwaran, H., Kogalur, U.B., Blackstone, E.H. and Lauer, M.S., 2008. Random survival forests. *The Annals of Applied Statistics*, 2(3), pp.841-860.
