"""CopulaModel to_dict/from_dict round-trip (#299).

CopulaModel.to_dict used to be the only serialiser without a schema
stamp or a reader — it wrote dictionaries nothing could load.
"""

import numpy as np
import pytest

import surpyval
from surpyval import Weibull
from surpyval.multivariate import Clayton, CopulaModel, Gaussian

PTS = np.array([[3.0, 2.0], [8.0, 4.0], [1.0, 6.0]])


def _model(copula, theta):
    return copula.from_params(
        theta,
        margins=[Weibull.from_params([10, 2]), Weibull.from_params([5, 1.5])],
    )


@pytest.mark.parametrize("copula,theta", [(Clayton, 2.0), (Gaussian, 0.6)])
def test_round_trip_preserves_predictions(copula, theta):
    m = _model(copula, theta)
    d = m.to_dict()
    assert d["schema"] == 1
    assert d["parameterization"] == "copula"
    r = surpyval.from_dict(d)
    assert isinstance(r, CopulaModel)
    np.testing.assert_allclose(r.sf(PTS), m.sf(PTS), rtol=0, atol=0)
    np.testing.assert_allclose(r.cdf(PTS), m.cdf(PTS), rtol=0, atol=0)
    np.testing.assert_allclose(r.pdf(PTS), m.pdf(PTS), rtol=0, atol=0)
    assert r.method == m.method


def test_json_file_round_trip(tmp_path):
    m = _model(Clayton, 2.0)
    fp = tmp_path / "copula.json"
    m.to_json(str(fp))
    r = CopulaModel.from_json(str(fp))
    np.testing.assert_allclose(r.sf(PTS), m.sf(PTS), rtol=0, atol=0)


def test_fitted_model_round_trips():
    m0 = _model(Clayton, 2.0)
    x = m0.random(150, random_state=3)
    m = Clayton.fit(x, margins=[Weibull, Weibull])
    r = surpyval.from_dict(m.to_dict())
    np.testing.assert_allclose(r.sf(PTS), m.sf(PTS), rtol=0, atol=0)
    np.testing.assert_allclose(
        np.atleast_1d(r.params), np.atleast_1d(m.params), rtol=0, atol=0
    )


def test_unknown_family_rejected():
    m = _model(Clayton, 2.0)
    d = m.to_dict()
    d["copula"] = "NotACopula"
    with pytest.raises(ValueError, match="Unknown copula family"):
        surpyval.from_dict(d)
