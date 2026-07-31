"""
Demonstration tests for the offset parameter-vs-distribution divergence.

The threshold/location parameter ``gamma`` of an offset distribution is
non-regular: it trades off against the shape and scale parameters. A fit
can therefore land on a wildly wrong ``(gamma, *params)`` tuple while the
*distribution* it implies is still close to the truth. These tests pin
that behaviour down with numbers so it does not surprise anyone, and so a
future "fix" to the offset initialisers can be judged on the divergence
that actually matters (KL / Wasserstein) rather than on the parameter
error alone. (#257 was exactly such a fix for the Gamma MPP path: its
test now asserts parameter recovery as well.)
"""

import numpy as np
from scipy.stats import wasserstein_distance

from surpyval import Gamma, Rayleigh

N_FIT = 10_000
N_EVAL = 100_000
TRUE_GAMMA = 10.0


def _kl_true_vs_fit(dist, true_params, true_gamma, fit_model, n=N_EVAL):
    """Monte-Carlo KL(true || fit) in nats, using the offset log-density."""
    x = dist.random(n, *true_params) + true_gamma
    log_true = dist.log_df(x - true_gamma, *true_params)
    log_fit = fit_model.dist.log_df(x - fit_model.gamma, *fit_model.params)
    mask = np.isfinite(log_true) & np.isfinite(log_fit)
    return float(np.mean(log_true[mask] - log_fit[mask]))


def _summary(dist, true_params, how, seed=0):
    """Fit with an offset and return parameter vs distribution metrics."""
    np.random.seed(seed)
    x = dist.random(N_FIT, *true_params) + TRUE_GAMMA
    fit = dist.fit(x, offset=True, how=how)
    true = dist.from_params(list(true_params), gamma=TRUE_GAMMA)

    # Worst relative error across gamma and the shape/scale parameters.
    param_errs = [abs(fit.gamma - TRUE_GAMMA) / abs(TRUE_GAMMA)]
    for fp, tp in zip(fit.params, true_params):
        param_errs.append(abs(float(fp) - tp) / abs(tp))
    max_param_rel_err = max(param_errs)

    # Distribution-level divergence.
    xs_true = dist.random(N_EVAL, *true_params) + TRUE_GAMMA
    xs_fit = fit.random(N_EVAL)
    w1 = wasserstein_distance(xs_true, xs_fit)
    kl = _kl_true_vs_fit(dist, true_params, TRUE_GAMMA, fit)

    return {
        "fit": fit,
        "true": true,
        "max_param_rel_err": max_param_rel_err,
        "mean_rel_err": abs(fit.mean() - true.mean()) / true.mean(),
        "median_rel_err": abs(float(fit.qf(0.5)) - float(true.qf(0.5)))
        / float(true.qf(0.5)),
        "std_rel_err": abs(xs_fit.std() - xs_true.std()) / xs_true.std(),
        "wasserstein": w1,
        "wasserstein_frac_std": w1 / xs_true.std(),
        "kl_nats": kl,
    }


def test_mpp_offset_gamma_parameters_recovered():
    """MPP offset on Gamma used to land on an absurd parameter tuple
    (huge shape, gamma far off) that merely mimicked the true
    distribution. This was the divergence this module existed to pin
    down; #257 fixed the initialiser (multi-started shape search) and
    the rr="x" inversion, so the *parameters* are now recovered too."""
    s = _summary(Gamma, (3.0, 2.0), how="MPP")

    # The parameters are now close to the truth.
    assert abs(s["fit"].gamma - TRUE_GAMMA) < 1.0, s["fit"].gamma
    assert s["max_param_rel_err"] < 0.20  # within 20%

    # ...and the distribution remains essentially exact.
    assert s["mean_rel_err"] < 0.01  # mean within 1%
    assert s["median_rel_err"] < 0.03  # median within 3%
    assert s["std_rel_err"] < 0.10  # spread within 10%
    assert s["kl_nats"] < 0.01
    assert s["wasserstein_frac_std"] < 0.05


def test_mom_offset_rayleigh_parameters_recovered():
    """MOM offset on Rayleigh used to stop far from the moment-matching
    solution (the numeric path ran with tol=1e-1 and no convergence
    check), biasing gamma low and the spread high. #275 tightened the
    optimisation, so the parameters are now recovered too."""
    s = _summary(Rayleigh, (3.0,), how="MOM")

    # The parameters are now close to the truth.
    assert abs(s["fit"].gamma - TRUE_GAMMA) < 0.5, s["fit"].gamma
    assert s["max_param_rel_err"] < 0.10  # within 10%

    # ...and the distribution matches.
    assert s["mean_rel_err"] < 0.02  # mean within 2%
    assert s["median_rel_err"] < 0.02  # median within 2%
    assert s["std_rel_err"] < 0.10  # spread within 10%
    assert s["kl_nats"] < 0.02


def test_mle_offset_is_the_accurate_baseline():
    """For contrast: MLE recovers both the parameters and the
    distribution to high accuracy, so the divergences above are a
    property of the MOM/MPP initialisers, not of offsetting itself."""
    for dist, params in [(Rayleigh, (3.0,)), (Gamma, (3.0, 2.0))]:
        s = _summary(dist, params, how="MLE")
        assert s["max_param_rel_err"] < 0.05  # parameters within 5%
        assert s["kl_nats"] < 0.01  # distribution essentially identical
        assert s["wasserstein_frac_std"] < 0.05
