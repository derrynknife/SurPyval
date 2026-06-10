import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from surpyval.utils.surpyval_data import SurpyvalData


@pytest.fixture
def basic_data():
    """Fixture for basic uncensored data"""
    x = np.array([1.0, 2.0, 3.0])
    return SurpyvalData(x)


@pytest.fixture
def censored_data():
    """Fixture for right censored data"""
    x = np.array([1.0, 2.0, 3.0])
    c = np.array([0, 1, 1])  # 2 and 3 are censored
    return SurpyvalData(x, c)


@pytest.fixture
def interval_data():
    """Fixture for interval censored data"""
    xl = np.array([1.0, 2.0, 3.0])
    xr = np.array([2.0, 3.0, 4.0])
    return SurpyvalData(xl=xl, xr=xr)


def test_basic_initialization(basic_data):
    """Test basic initialization with uncensored data"""
    assert len(basic_data.x) == 3
    assert basic_data.x_min == 1.0
    assert basic_data.x_max == 3.0
    assert np.all(basic_data.c == 0)  # All uncensored
    assert np.all(basic_data.n == 1)  # Default counts


def test_censored_data_initialization(censored_data):
    """Test initialization with right censored data"""
    assert len(censored_data.x) == 3
    assert len(censored_data.x_r) == 2  # Two right censored points
    assert len(censored_data.x_o) == 1  # One uncensored point
    assert censored_data.x_o[0] == 1.0  # First point is uncensored


def test_interval_data_initialization(interval_data):
    """Test initialization with interval censored data"""
    assert len(interval_data.x) == 3
    assert interval_data.x.ndim == 2  # Should be 2D for interval data
    assert interval_data.x[:, 0].tolist() == [1.0, 2.0, 3.0]
    assert interval_data.x[:, 1].tolist() == [2.0, 3.0, 4.0]


def test_iteration(basic_data):
    """Test iteration over data points"""
    points = list(basic_data)
    assert len(points) == 3
    for point in points:
        assert len(point) == 4  # x, c, n, t values


def test_indexing(basic_data):
    """Test indexing functionality"""
    subset = basic_data[0:2]
    assert isinstance(subset, SurpyvalData)
    assert len(subset.x) == 2
    assert subset.x_max == 2.0


def test_add_covariates():
    """Test adding covariates"""
    x = np.array([1.0, 2.0, 3.0])
    c = np.array([0, 1, 1])
    data = SurpyvalData(x, c)
    Z = np.array([[1, 2], [3, 4], [5, 6]])
    data.add_covariates(Z)
    assert np.array_equal(data.Z, Z)
    assert len(data.Z_o) == 1  # One uncensored point
    assert len(data.Z_r) == 2  # Two right censored points


def test_to_xrd_simple():
    """Test conversion to xrd format for simple data"""
    x = np.array([1.0, 2.0, 3.0])
    data = SurpyvalData(x)
    x_rd, r, d = data.to_xrd()
    assert len(x_rd) == len(r) == len(d)
    assert np.all(x_rd == x)


@pytest.mark.parametrize(
    "x,c",
    [
        (np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0])),  # All uncensored
        (np.array([1.0, 2.0, 3.0]), np.array([0, 1, 1])),  # Right censored
        (np.array([1.0, 2.0, 3.0]), np.array([0, -1, -1])),  # Left censored
    ],
)
def test_data_types(x, c):
    """Test different data type combinations"""
    data = SurpyvalData(x, c)
    assert data.x.shape == x.shape
    assert data.c.shape == c.shape


def test_different_length():
    """Test handling of invalid data"""
    with pytest.raises(ValueError):
        SurpyvalData(x=[1], c=[0, 1])

    with pytest.raises(ValueError):
        SurpyvalData(x=[1], n=[2, 1])


def test_interval_check_same_as_censoring():
    # x = [1, 2, [3, 5], [6, 7], 8, 9]
    # c = [0, 1, 2, 2, 1, 0]
    # SurpyvalData(x=x, c=c)

    with pytest.raises(ValueError):
        x = [1, 2, [3, 5], [6, 7], 8, 9]
        c = [0, 1, 1, 0, 1, 0]
        SurpyvalData(x=x, c=c)


def test_repr(basic_data):
    """Test string representation"""
    repr_str = repr(basic_data)
    assert "SurpyvalData" in repr_str
    assert "x=" in repr_str
    assert "c=" in repr_str
    assert "n=" in repr_str
    assert "t=" in repr_str


# --- scalar __getitem__ ---


def test_getitem_scalar_returns_surpyval_data(basic_data):
    result = basic_data[0]
    assert isinstance(result, SurpyvalData)
    assert len(result.x) == 1


def test_getitem_scalar_correct_values(basic_data):
    result = basic_data[1]
    assert result.x[0] == basic_data.x[1]
    assert result.c[0] == basic_data.c[1]
    assert result.n[0] == basic_data.n[1]


def test_getitem_negative_index(basic_data):
    result = basic_data[-1]
    assert isinstance(result, SurpyvalData)
    assert result.x[0] == basic_data.x[-1]


# --- to_json / from_json ---


def test_to_json_returns_string(basic_data):
    s = basic_data.to_json()
    assert isinstance(s, str)
    parsed = json.loads(s)
    assert set(parsed.keys()) == {"x", "c", "n", "t"}


def test_to_json_no_Z_key_when_no_covariates(basic_data):
    parsed = json.loads(basic_data.to_json())
    assert "Z" not in parsed


def test_to_json_includes_Z_when_covariates_present():
    x = np.array([1.0, 2.0, 3.0])
    c = np.array([0, 1, 1])
    data = SurpyvalData(x, c)
    Z = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    data.add_covariates(Z)
    parsed = json.loads(data.to_json())
    assert "Z" in parsed
    assert np.allclose(parsed["Z"], Z.tolist())


def test_from_json_string_roundtrip(basic_data):
    s = basic_data.to_json()
    restored = SurpyvalData.from_json(s)
    assert np.allclose(restored.x, basic_data.x)
    assert np.array_equal(restored.c, basic_data.c)
    assert np.array_equal(restored.n, basic_data.n)
    assert np.allclose(restored.t, basic_data.t)


def test_from_json_file_roundtrip(basic_data):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    try:
        basic_data.to_json(path)
        restored = SurpyvalData.from_json(path)
        assert np.allclose(restored.x, basic_data.x)
        assert np.array_equal(restored.c, basic_data.c)
    finally:
        path.unlink(missing_ok=True)


def test_from_json_preserves_covariates():
    x = np.array([1.0, 2.0, 3.0])
    c = np.array([0, 1, 0])
    data = SurpyvalData(x, c)
    Z = np.array([[0.1], [0.2], [0.3]])
    data.add_covariates(Z)
    restored = SurpyvalData.from_json(data.to_json())
    assert np.allclose(restored.Z, Z)


def test_from_json_no_covariates_Z_is_none(basic_data):
    restored = SurpyvalData.from_json(basic_data.to_json())
    assert restored.Z is None


def test_to_json_saves_to_file(basic_data):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    try:
        result = basic_data.to_json(path)
        assert result is None
        assert path.exists()
        assert json.loads(path.read_text())["x"] == basic_data.x.tolist()
    finally:
        path.unlink(missing_ok=True)
