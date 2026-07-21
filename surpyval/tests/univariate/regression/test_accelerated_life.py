"""Accelerated Life (parameter-substitution) fitting.

Guards the life-parameter map that couples each distribution to its
stress-relationship life model, and checks that an Exponential Accelerated
Life model fits and recovers the underlying stress-life relationship (a
regression: its life parameter was mis-named ``"lambda"`` where the
distribution actually calls it ``"failure_rate"``, so the fit raised
``KeyError: 'lambda'``).
"""

import numpy as np

import surpyval
from surpyval import AcceleratedLife, Exponential, Power, Weibull
from surpyval.univariate.regression.accelerated_life.accelerated_life import (
    _LIFE_PARAM_MAP,
)


def test_life_param_map_names_a_real_parameter():
    # Every distribution's declared life parameter must be one of that
    # distribution's actual parameters, or ``fit`` fails when it looks the
    # index up in ``param_map``.
    for dist_name, (life_param, _, _) in _LIFE_PARAM_MAP.items():
        dist = getattr(surpyval, dist_name)
        assert life_param in dist.param_names, (
            f"{dist_name} life parameter {life_param!r} is not in "
            f"{dist.param_names}"
        )


def _al_stress_data(phi, stresses, per=400, seed=0):
    rng = np.random.default_rng(seed)
    xs, Zs = [], []
    for s in stresses:
        xs.append(rng.exponential(phi(s), per))
        Zs.append(np.full(per, s))
    x = np.concatenate(xs)
    Z = np.concatenate(Zs).reshape(-1, 1)
    return x, Z


def test_exponential_accelerated_life_fits_and_recovers():
    # Exponential lifetimes whose mean life follows a power law in the stress.
    stresses = [1.0, 2.0, 4.0, 8.0]

    def true_life(s):
        return 2000.0 * s**-1.5

    x, Z = _al_stress_data(true_life, stresses)

    # Must not raise (previously KeyError: 'lambda').
    model = AcceleratedLife(Exponential, Power).fit(x=x, Z=Z)

    # The fitted model's mean life at each stress (= 1 / failure_rate) should
    # recover the true power-law life within sampling error.
    for s in stresses:
        rate = np.ravel(
            model.model.param_transform(
                model.reg_model.phi(np.array([[s]]), *model.phi_params)
            )
        )[0]
        assert np.isclose(1.0 / rate, true_life(s), rtol=0.15)


def test_exponential_accelerated_life_round_trips():
    stresses = [1.0, 2.0, 4.0, 8.0]
    x, Z = _al_stress_data(lambda s: 2000.0 * s**-1.5, stresses, seed=1)
    model = AcceleratedLife(Exponential, Power).fit(x=x, Z=Z)

    restored = surpyval.from_dict(model.to_dict())
    xq = np.linspace(1.0, 500.0, 10)
    Zq = np.full((xq.size, 1), 2.0)
    assert np.allclose(model.sf(xq, Zq), restored.sf(xq, Zq), equal_nan=True)


def test_weibull_accelerated_life_still_fits():
    # A guard that the fix did not disturb the other (already-working)
    # distributions.
    stresses = [1.0, 2.0, 4.0, 8.0]
    rng = np.random.default_rng(2)
    xs, Zs = [], []
    for s in stresses:
        life = 500.0 * s**-1.0
        xs.append(life * rng.weibull(2.2, 200))
        Zs.append(np.full(200, s))
    x = np.concatenate(xs)
    Z = np.concatenate(Zs).reshape(-1, 1)

    model = AcceleratedLife(Weibull, Power).fit(x=x, Z=Z)
    assert np.all(np.isfinite(model.params))
