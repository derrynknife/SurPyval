import pytest
import surpyval as surv
from surpyval.univariate.parametric.parametric import Parametric

def test_from_dict_security_validation():
    # Attempt to load a dictionary with a non-distribution attribute
    payload = {
        "parameterization": "parametric",
        "distribution": "__builtins__",
        "how": "MLE",
        "offset": False,
        "lfp": False,
        "zi": False,
        "params": [1, 2]
    }

    with pytest.raises(ValueError, match="Invalid distribution: __builtins__"):
        Parametric.from_dict(payload)

def test_from_dict_valid_distribution():
    # Fit a simple model to get a valid dict
    x = [1, 2, 3, 4, 5]
    model = surv.Weibull.fit(x)
    model_dict = model.to_dict()

    # Should not raise any error
    loaded_model = Parametric.from_dict(model_dict)
    assert loaded_model.dist.name == "Weibull"
