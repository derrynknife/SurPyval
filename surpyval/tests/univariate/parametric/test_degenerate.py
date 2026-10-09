"""The degenerate distributions: InstantlyOccurs (point mass at 0) and
NeverOccurs (point mass at +inf).

Previously these lived in ``parametric/__init__.py`` with a partial API
(no df/hf/qf/mean) and no serialisation; they are now first-class
members of the distributions package.
"""

import numpy as np
import pytest

import surpyval
import surpyval as surv
from surpyval import ExactEventTime, InstantlyOccurs, NeverOccurs

X = np.array([0.0, 1.0, 5.0, 100.0])


def test_never_occurs_api():
    assert (NeverOccurs.sf(X) == 1.0).all()
    assert (NeverOccurs.ff(X) == 0.0).all()
    assert (NeverOccurs.df(X) == 0.0).all()
    assert (NeverOccurs.hf(X) == 0.0).all()
    assert (NeverOccurs.Hf(X) == 0.0).all()
    assert (NeverOccurs.qf(np.array([0.1, 0.5, 0.99])) == np.inf).all()
    assert NeverOccurs.mean() == np.inf
    assert (NeverOccurs.random(5) == np.inf).all()


def test_instantly_occurs_api():
    assert (InstantlyOccurs.sf(X) == 0.0).all()
    assert (InstantlyOccurs.ff(X) == 1.0).all()
    assert (InstantlyOccurs.hf(X) == np.inf).all()
    assert (InstantlyOccurs.Hf(X) == np.inf).all()
    df = InstantlyOccurs.df(X)
    assert df[0] == np.inf and (df[1:] == 0.0).all()
    assert (InstantlyOccurs.qf(np.array([0.1, 0.9])) == 0.0).all()
    assert InstantlyOccurs.mean() == 0.0
    assert (InstantlyOccurs.random(5) == 0.0).all()


def test_degenerate_serialisation_round_trip():
    # Stateless: the class is the model, so from_dict returns the class
    # itself and identity is preserved (the survival-tree leaves rely on
    # ``model is NeverOccurs``).
    for cls in (NeverOccurs, InstantlyOccurs):
        d = cls.to_dict()
        assert d["model"] == cls.name
        assert d["schema"] == surpyval.serialisation.required_schema(d)
        assert surpyval.from_dict(d) is cls


def test_import_paths_preserved():
    # The historical import locations must keep working, and must resolve
    # to the same class objects.
    from surpyval.univariate.parametric import (
        InstantlyOccurs as from_parametric_i,
    )
    from surpyval.univariate.parametric import NeverOccurs as from_parametric_n
    from surpyval.univariate.parametric.distributions.degenerate import (
        NeverOccurs as from_module_n,
    )

    assert from_parametric_n is NeverOccurs is from_module_n
    assert from_parametric_i is InstantlyOccurs


def test_exact_event_time_refuses_density_and_hazard():
    # A point mass has no density: all of its probability sits at T, so
    # the density is a Dirac delta rather than a function of x. These
    # used to return inf at T (and, for hf, at every x after it), which
    # integrates to inf rather than 1 and propagates silently into
    # whatever consumes it.
    import pytest

    from surpyval import ExactEventTime

    x = np.array([4.0, 5.0, 6.0])
    with pytest.raises(NotImplementedError, match="no density"):
        ExactEventTime.df(x, 5.0)
    with pytest.raises(NotImplementedError, match="no hazard rate"):
        ExactEventTime.hf(x, 5.0)
    # log_df is inherited and reaches hf, so it refuses too rather than
    # returning the nan that log(inf) - inf used to give.
    with pytest.raises(NotImplementedError):
        ExactEventTime.log_df(x, 5.0)


def test_exact_event_time_keeps_the_functions_that_are_defined():
    # sf, ff and Hf are genuine step functions and are unaffected.
    from surpyval import ExactEventTime

    x = np.array([4.0, 4.999, 5.0, 6.0])
    np.testing.assert_array_equal(ExactEventTime.sf(x, 5.0), [1, 1, 0, 0])
    np.testing.assert_array_equal(ExactEventTime.ff(x, 5.0), [0, 0, 1, 1])
    Hf = np.asarray(ExactEventTime.Hf(x, 5.0), dtype=float)
    np.testing.assert_array_equal(Hf, [0.0, 0.0, np.inf, np.inf])
    # Hf is -log R(x), and used to be an alias for hf that happened to
    # take the same two values.
    with np.errstate(divide="ignore"):
        expected = -np.log(np.asarray(ExactEventTime.sf(x, 5.0), dtype=float))
    np.testing.assert_array_equal(Hf, expected)


def test_exact_event_time_still_fits_and_serialises():
    # The estimator brackets T between the censoring bounds and never
    # touches a density, so refusing df and hf cannot affect it.
    from surpyval import ExactEventTime

    model = ExactEventTime.fit(x=[1.0, 2.0, 8.0, 9.0], c=[1, 1, -1, -1])
    np.testing.assert_allclose(model.params, [5.0])
    restored = surpyval.from_dict(model.to_dict())
    np.testing.assert_allclose(restored.params, model.params)


# ---------------------------------------------------------------------------
# JSON round trip and the class tag check.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dist", [NeverOccurs, InstantlyOccurs])
def test_degenerate_to_json_round_trip(tmp_path, dist):
    path = tmp_path / "d.json"
    dist.to_json(path)
    assert surpyval.from_json(path) is dist
    assert dist.from_json(path) is dist
    assert surpyval.from_dict(dist.to_dict()) is dist


def test_degenerate_from_dict_checks_the_tag(tmp_path):
    with pytest.raises(ValueError, match="InstantlyOccurs"):
        InstantlyOccurs.from_dict(NeverOccurs.to_dict())
    path = tmp_path / "n.json"
    NeverOccurs.to_json(path)
    with pytest.raises(ValueError, match="InstantlyOccurs"):
        InstantlyOccurs.from_json(path)


# ---------------------------------------------------------------------------
# ``ExactEventTime``: an informative error, and its quantile,
# mean and moments (#257).
# ---------------------------------------------------------------------------


def test_exact_event_time_informative_error():
    with pytest.raises(ValueError, match="right-censored"):
        ExactEventTime.fit(np.array([1.0, 2.0, 3.0]), c=np.array([1, 1, 1]))


def test_exact_event_time_has_a_quantile_mean_and_moments():
    # A point mass has no density and no hazard rate -- df and hf raise,
    # and say why -- but its quantile, mean and moments are all exact and
    # trivial. They were simply missing, so a caller reaching for the mean
    # of a known event time got an AttributeError.
    T = 5.0
    np.testing.assert_allclose(
        np.asarray(ExactEventTime.qf([0.01, 0.5, 0.99], T), dtype=float), T
    )
    assert ExactEventTime.mean(T) == T
    assert ExactEventTime.moment(1, T) == T
    assert ExactEventTime.moment(3, T) == T**3
    # And the ones that genuinely do not exist still refuse.
    for method in ("df", "hf"):
        with pytest.raises(NotImplementedError):
            getattr(ExactEventTime, method)(np.array([1.0]), T)


# ---------------------------------------------------------------------------
# ``ExactEventTime`` checks; ``FixedEventProbability.mean``.
# ---------------------------------------------------------------------------


def test_exact_event_time_refuses_contradictory_checks():
    with pytest.raises(ValueError, match="contradict"):
        surv.ExactEventTime.fit([5, 3], [1, -1])


def test_fixed_event_probability_mean_is_its_first_moment():
    model = surv.FixedEventProbability.from_params([0.3])
    assert model.mean() == pytest.approx(0.3)
    assert model.mean() == pytest.approx(model.moment(1))


def test_746_default_Hf_is_plus_zero_where_sf_is_one():
    # The default Hf of a Distribution is -log sf; -log(1) is -0.0 (#746).
    from surpyval.distribution import Distribution

    class Never(Distribution):
        def sf(self, x):
            return np.ones_like(np.asarray(x, dtype=float))

        def ff(self, x):
            return 1.0 - self.sf(x)

    H = Never().Hf([0.0, 1.0])
    assert np.all(H == 0) and not np.any(np.signbit(H))
