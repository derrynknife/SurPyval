"""
Demonstration tests for the offset parameter-vs-distribution divergence.

The threshold/location parameter ``gamma`` of an offset distribution is
non-regular: it trades off against the shape and scale parameters. A fit
can therefore land on a wildly wrong ``(gamma, *params)`` tuple while the
*distribution* it implies is still close to the truth. These tests pin
that behaviour down with numbers so it does not surprise anyone, and so a
future "fix" to the offset initialisers can be judged on the divergence
that actually matters (KL / Wasserstein) rather than on the parameter
error alone.

See DEVELOPMENT.md section 2.
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


def test_mpp_offset_gamma_parameters_absurd_but_distribution_close():
    """MPP offset on Gamma recovers an absurd parameter tuple
    (gamma far negative, shape inflated by orders of magnitude) yet the
    implied distribution is almost indistinguishable from the truth: a
    high-shape Gamma parked near the origin mimics the offset Gamma."""
    s = _summary(Gamma, (3.0, 2.0), how="MPP")

    # The parameters are wildly wrong - this is the "failure".
    assert s["fit"].gamma < 5.0, s["fit"].gamma
    assert s["max_param_rel_err"] > 5.0  # >500% off

    # ...but the distribution barely moves.
    assert s["mean_rel_err"] < 0.01  # mean within 1%
    assert s["median_rel_err"] < 0.03  # median within 3%
    assert s["std_rel_err"] < 0.10  # spread within 10%
    assert s["kl_nats"] < 0.20  # KL well under a fifth of a nat
    assert s["wasserstein_frac_std"] < 0.25

    # The headline: distribution error is far smaller than parameter error.
    assert s["mean_rel_err"] < s["max_param_rel_err"] / 100


def test_mom_offset_rayleigh_biased_but_central_predictions_hold():
    """MOM offset on Rayleigh biases gamma low and, unlike the Gamma
    case, the divergence is *modest, not negligible*: the spread is
    visibly wrong even though the central predictions stay accurate."""
    s = _summary(Rayleigh, (3.0,), how="MOM")

    # gamma is biased away from the truth.
    assert abs(s["fit"].gamma - TRUE_GAMMA) > 0.4

    # Central predictions remain trustworthy...
    assert s["mean_rel_err"] < 0.02  # mean within 2%
    assert s["median_rel_err"] < 0.02  # median within 2%

    # ...but the distribution is genuinely off in its spread: this is the
    # "modest, not negligible" caveat made concrete.
    assert s["std_rel_err"] > 0.10  # std off by more than 10%
    assert 0.02 < s["kl_nats"] < 0.15  # small, but not zero


def test_mle_offset_is_the_accurate_baseline():
    """For contrast: MLE recovers both the parameters and the
    distribution to high accuracy, so the divergences above are a
    property of the MOM/MPP initialisers, not of offsetting itself."""
    for dist, params in [(Rayleigh, (3.0,)), (Gamma, (3.0, 2.0))]:
        s = _summary(dist, params, how="MLE")
        assert s["max_param_rel_err"] < 0.05  # parameters within 5%
        assert s["kl_nats"] < 0.01  # distribution essentially identical
        assert s["wasserstein_frac_std"] < 0.05
