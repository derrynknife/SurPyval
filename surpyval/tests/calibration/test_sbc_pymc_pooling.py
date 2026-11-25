"""Simulation-based calibration of the PyMC pooling template
(``surpyval.pymc.pooled_likelihood``, ``group_models``): the coverage of
its 90% credible intervals on the population's centre ``mu``, its spread
``tau`` and each group's ``alpha``.

In words: draw the population and each group's alpha from the priors,
simulate each group's lives from them, sample the posterior with PyMC,
and ask how often the 90% interval holds the truth. When the model and
the likelihood are right the truth's posterior probability (its "PIT"
value) is uniform, so the interval holds it in 90% of repetitions. The
rules are the validation kit's (``surpyval/tests/bayes/kit.py``):
coverage within three binomial standard errors of 90% (over the
repetitions; for the groups' alphas, the repetitions' share averaged
over the groups, held to the same band), uniformity by
Kolmogorov-Smirnov at 0.001 (``mu`` and ``tau``).

The scenario and its seeds were fixed before the first run (26 Nov
2026): five groups of 12 Weibull lives, each unit watched to 1,500;
``mu`` Normal(log 1000, 0.3), ``tau`` half-normal (scale 0.3), each
group's ``log alpha`` Normal(mu, tau); ``beta`` log-normal about 2
(sigma 0.2), shared. The non-centred form. PyMC: 2 chains of 500 draws
after 500 tuning steps, seed ``67100 + 2 r`` for the data and ``67101 +
2 r`` for the sampler of repetition ``r``. 200 repetitions. Skipped
without PyMC (not a dependency of SurPyval).

The lives are simulated from the model's definition, not with the
package's simulators.
"""

import warnings

import numpy as np
import pytest
from scipy import stats

import surpyval as sp
from surpyval.tests.bayes import kit

pytestmark = pytest.mark.calibration

REPS = 200
GROUPS, UNITS, END = 5, 12, 1500.0
MU, MU_SD, TAU_SCALE = np.log(1000.0), 0.3, 0.3
BETA, BETA_SD = 2.0, 0.2


def _one(r):
    pm = pytest.importorskip("pymc")

    from surpyval.pymc import pooled_likelihood

    rng = np.random.default_rng(67100 + 2 * r)
    mu = MU + MU_SD * rng.standard_normal()
    tau = abs(TAU_SCALE * rng.standard_normal())
    beta = BETA * np.exp(BETA_SD * rng.standard_normal())
    log_alpha = mu + tau * rng.standard_normal(GROUPS)
    life = np.exp(log_alpha)[:, None] * rng.weibull(beta, (GROUPS, UNITS))
    x = np.minimum(life, END).ravel()
    c = (life > END).astype(int).ravel()
    group = np.repeat(np.arange(GROUPS), UNITS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pm.Model():
            b = pm.LogNormal("beta", mu=np.log(BETA), sigma=BETA_SD)
            pooled_likelihood(
                sp.Weibull,
                x=x,
                c=c,
                group=group,
                pool={"alpha": (MU, MU_SD)},
                shared={"beta": b},
                tau_scale=TAU_SCALE,
            )
            trace = pm.sample(
                draws=500,
                tune=500,
                chains=2,
                random_seed=67101 + 2 * r,
                progressbar=False,
                compute_convergence_checks=False,
            )
    post = trace.posterior
    d_mu = np.asarray(post["groups_alpha_mu"]).ravel()
    d_tau = np.asarray(post["groups_alpha_tau"]).ravel()
    z = np.asarray(post["groups_alpha_z"]).reshape(-1, GROUPS)
    d_alpha = d_mu[:, None] + d_tau[:, None] * z
    return (
        float(np.mean(d_mu < mu)),
        float(np.mean(d_tau < tau)),
        np.mean(d_alpha < log_alpha[None, :], axis=0),
    )


def run(reps=REPS):
    """The PIT values of ``mu`` and ``tau``, ``(reps,)``, and of each
    group's alpha, ``(reps, GROUPS)``."""
    rows = [_one(r) for r in range(reps)]
    mu = np.array([r[0] for r in rows])
    tau = np.array([r[1] for r in rows])
    alpha = np.stack([r[2] for r in rows])
    return mu, tau, alpha


def test_the_pooling_template_holds_its_level():
    pytest.importorskip("pymc")
    mu, tau, alpha = run()
    level = 0.9
    band = kit.SE_MULTIPLE * np.sqrt(level * (1 - level) / REPS)
    problems = []
    for name, u in (("mu", mu), ("tau", tau), ("alpha", alpha)):
        got = float(np.mean(np.abs(u - 0.5) <= level / 2))
        if abs(got - level) > band:
            problems.append(f"{name}: 90% coverage {got:.3f}")
    for name, u in (("mu", mu), ("tau", tau)):
        p = stats.kstest(u, "uniform").pvalue
        if p < kit.KS_LEVEL:
            problems.append(f"{name}: PIT not uniform (KS p {p:.2g})")
    assert not problems, problems
