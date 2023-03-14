import numpy as np
import pytest
from numpy.testing import assert_array_equal
from numpy.typing import NDArray

from surpyval import RandomSurvivalForest


@pytest.fixture
def get_x_Z_c_samples() -> tuple[NDArray, NDArray, NDArray]:
    # Set random seed
    rng = np.random.default_rng(seed=0)

    # Samples
    x_left = rng.uniform(5, 10, 50)
    x_right = rng.uniform(100, 105, 50)
    x = np.concatenate((x_left, x_right))

    # Covariants
    z_0_irrelevant = rng.uniform(0, 1, 100)
    z_1_relevant = np.concatenate(
        (rng.uniform(0, 1, 50), rng.uniform(100, 101, 50))
    )
    Z = np.array([z_0_irrelevant, z_1_relevant]).transpose()
    c = np.array([0] * len(x))

    return x, Z, c


def test_forest_n_trees(get_x_Z_c_samples: tuple[NDArray, NDArray, NDArray]):
    # Get samples
    x, Z, c = get_x_Z_c_samples

    # Make forest w/ 5 trees
    forest = RandomSurvivalForest(x=x, Z=Z, c=c, n_trees=5)

    # Assert there are only 5 trees
    assert forest.n_trees == 5
    assert len(forest.trees) == 5


def test_forest_bootstrap_false(
    get_x_Z_c_samples: tuple[NDArray, NDArray, NDArray]
):
    # Get samples
    x, Z, c = get_x_Z_c_samples

    # Make forest w/ 2 trees
    forest = RandomSurvivalForest(x=x, Z=Z, c=c, n_trees=2, bootstrap=False)

    assert not forest.bootstrap  # == False
    assert_array_equal(forest.trees[0].x, x)
    assert_array_equal(forest.trees[1].x, x)


def test_forest_bootstrap_true(
    get_x_Z_c_samples: tuple[NDArray, NDArray, NDArray]
):
    # Get samples
    x, Z, c = get_x_Z_c_samples

    # Make forest w/ 1 tree, bootstrapped (boostrap=True by default)
    forest = RandomSurvivalForest(x=x, Z=Z, c=c, n_trees=1)

    # Assert all samples are not as given
    assert forest.bootstrap  # == False
    with pytest.raises(AssertionError):
        assert_array_equal(forest.trees[0].x, x)


def test_forest_sf_scalar_x(
    get_x_Z_c_samples: tuple[NDArray, NDArray, NDArray]
):
    # Get samples
    x, Z, c = get_x_Z_c_samples

    # Make forest w/ 5 trees
    forest = RandomSurvivalForest(x=x, Z=Z, c=c, n_trees=5)

    # Make a sf() call for x=1 || 100, and Z = [0.5, 0.5]
    # Should be pretty     low || high
    sf_1 = forest.sf(x=1, Z=[0.5, 0.5])
    assert isinstance(sf_1, np.ndarray)
    assert pytest.approx(sf_1, abs=0.05) == np.array([0.97])
    # (Veeery approximate)     ^^^^^^^^

    sf_100 = forest.sf(x=100, Z=[0.5, 0.5])
    assert isinstance(sf_100, np.ndarray)
    assert pytest.approx(sf_100, abs=0.1) == np.array([0.1])
    # (Veeery approximate)       ^^^^^^^


def test_forest_sf_vector_x(
    get_x_Z_c_samples: tuple[NDArray, NDArray, NDArray]
):
    # Get samples
    x, Z, c = get_x_Z_c_samples

    # Make forest w/ 5 trees
    forest = RandomSurvivalForest(x=x, Z=Z, c=c, n_trees=5)

    # Make a sf() call for x=[1, 5, 50, 75, 100], and Z = [0.5, 0.5]
    sf = forest.sf(x=[1, 5, 50, 75, 100], Z=[0.5, 0.5])
    assert isinstance(sf, np.ndarray)
    assert len(sf) == 5


def test_forest_all_functions(
    get_x_Z_c_samples: tuple[NDArray, NDArray, NDArray]
):
    # Get samples
    x, Z, c = get_x_Z_c_samples

    # Make forest w/ 5 trees
    forest = RandomSurvivalForest(x=x, Z=Z, c=c, n_trees=5)

    # Make a sf(), ff(), df(), hf(), and Hf() call for x=1 and Z = [0.5, 0.5]
    sf = forest.sf(x=1, Z=[0.5, 0.5])
    assert len(sf) == 1 and isinstance(sf[0], float)

    ff = forest.ff(x=1, Z=[0.5, 0.5])
    assert len(ff) == 1 and isinstance(ff[0], float)

    df = forest.df(x=1, Z=[0.5, 0.5])
    assert len(df) == 1 and isinstance(df[0], float)

    hf = forest.hf(x=1, Z=[0.5, 0.5])
    assert len(hf) == 1 and isinstance(hf[0], float)

    Hf = forest.Hf(x=1, Z=[0.5, 0.5])
    assert len(Hf) == 1 and isinstance(Hf[0], float)
