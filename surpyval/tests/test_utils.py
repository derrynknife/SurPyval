import numpy as np
import pytest

from surpyval.utils import (
    fsli_handler,
    group_xcnt,
    xcn_to_fs,
    xcnt_handler,
    xcnt_sort,
    xrd_handler,
)


def test_group_xcnt():
    x = np.array([1, 1, 3, 3])
    c = np.array([0, 0, 0, 0])
    n = np.ones_like(x).astype(int)
    t = np.vstack([np.ones_like(x) * -np.inf, np.ones_like(x) * np.inf]).T

    x_g, c_g, n_g, _ = group_xcnt(x, c, n, t)
    assert np.all(x_g == np.array([1, 3]))
    assert np.all(c_g == np.array([0, 0]))
    assert np.all(n_g == np.array([2, 2]))


def test_xcnt_sort():
    x = np.array([1, 2, 1, 2])
    c = np.array([1, 0, 0, 1])
    n = np.ones_like(x).astype(int)
    t = np.vstack([np.ones_like(x) * -np.inf, np.ones_like(x) * np.inf]).T

    x_s, c_s, n_s, _ = xcnt_sort(x, c, n, t)
    assert np.all(x_s == np.array([1, 1, 2, 2]))
    assert np.all(c_s == np.array([0, 1, 0, 1]))
    assert np.all(n_s == np.array([1, 1, 1, 1]))


def test_fsli_handler():
    f = [2, 4, 6, 8]
    s = [1, 3, 5, 7]
    l = [10, 12]
    i = [[2, 5], [6, 9]]

    _ = fsli_handler(f, s, l, i)
    _ = fsli_handler(s=[1, 2])
    _ = fsli_handler(l=[1, 2])
    _ = fsli_handler(s=[1, 2])

    # Test that call can't be empty arrays
    with pytest.raises(Exception):
        _ = fsli_handler([], [], [], [])

    # Test that all can't be None
    with pytest.raises(Exception):
        _ = fsli_handler()

    # Test that f cannot be a jagged array
    with pytest.raises(Exception):
        _ = fsli_handler(f=[[1, 2], 2, 3])

    # Test that s cannot be a jagged array
    with pytest.raises(Exception):
        _ = fsli_handler(s=[[1, 2], 2, 3])

    # Test that l cannot be a jagged array
    with pytest.raises(Exception):
        _ = fsli_handler(l=[[1, 2], 2, 3])

    # Test that i cannot be a jagged array
    with pytest.raises(Exception):
        _ = fsli_handler(i=[[1, 2], 3])

    # Test that i cannot have a second dim other than 2
    with pytest.raises(Exception):
        _ = fsli_handler(i=[[1, 2, 3], [3, 4, 5]])

    # Test that i cannot have a second dim other than 2
    with pytest.raises(Exception):
        _ = fsli_handler(i=[1, 2, 3])

    # Test that i cannot have rows where the first element is equal
    # to the second element
    with pytest.raises(Exception):
        _ = fsli_handler(i=[[1, 1], [2, 3]])

    # Test that i cannot have rows where the first element is greater
    # than the second element
    with pytest.raises(Exception):
        _ = fsli_handler(i=[[10, 1], [2, 3]])


def test_xrd_handler():
    with pytest.raises(
        ValueError,
        match="'x' must be an array of scalar numbers with real values.",
    ):
        xrd_handler(["a", "b"], [1, 2], [1, 2])

    with pytest.raises(
        ValueError,
        match="'x' must be an array of scalar numbers with real values.",
    ):
        xrd_handler([[1, 2], 3, 4], [1, 2], [1, 2])

    with pytest.raises(ValueError, match="'r' must be an array of integers."):
        xrd_handler([1, 2], [1.2, 2.3], [1, 2])

    with pytest.raises(ValueError, match="'d' must be an array of integers."):
        xrd_handler([1, 2], [1, 2], [1.5, 2.7])

    with pytest.raises(
        ValueError, match="'x' must be a one dimensional array"
    ):
        xrd_handler([[1, 2], [3, 4]], [1, 2, 3, 4], [1, 2, 3, 4])

    with pytest.raises(
        ValueError, match="'x' array not the same length as 'r' array"
    ):
        xrd_handler([1, 2, 3], [1, 2], [1, 2, 3])

    with pytest.raises(
        ValueError, match="'x' array not the same length as 'd' array"
    ):
        xrd_handler([1, 2, 3], [1, 2, 3], [1, 2])

    with pytest.raises(
        ValueError, match="'d' array cannot have any negative values"
    ):
        xrd_handler([1, 2, 3], [1, 2, 3], [1, 2, -3])

    with pytest.raises(
        ValueError,
        match="'r' at risk item count array cannot have any negative values",
    ):
        xrd_handler([1, 2, 3], [1, -2, 3], [1, 2, 3])

    with pytest.raises(
        ValueError,
        match="cannot have more deaths/failures than there are items at risk",
    ):
        xrd_handler([1, 2, 3], [1, 2, 3], [1, 2, 4])


def test_xcn_to_fs():
    # Test with defaults
    x = [1, 2, 3]
    f, s = xcn_to_fs(x)
    assert np.array_equal(
        f, [1, 2, 3]
    )  # default c is all zeros, n is all ones
    assert len(s) == 0

    # Test repeats with default n
    x = [1, 2, 3]
    c = [0, 1, 0]
    f, s = xcn_to_fs(x, c)
    assert np.array_equal(f, [1, 3])  # both 1 and 3 have c as 0
    assert np.array_equal(s, [2])  # 2 has c as 1

    # Test repeats with given n
    x = [1, 2, 3, 4]
    c = [0, 1, 0, 1]
    n = [2, 3, 2, 1]
    f, s = xcn_to_fs(x, c, n)
    assert np.array_equal(
        f, [1, 1, 3, 3]
    )  # 1 repeated twice, 3 repeated twice
    assert np.array_equal(
        s, [2, 2, 2, 4]
    )  # 2 repeated thrice, 4 repeated once


def test_xcnt_handler():
    # Test valid x and default c, n, t
    x = np.array([1, 2, 3, 4])
    x, c, n, t = xcnt_handler(x)
    assert np.array_equal(x, [1, 2, 3, 4])
    assert np.array_equal(c, [0, 0, 0, 0])
    assert np.array_equal(n, [1, 1, 1, 1])
    assert np.array_equal(t, [[-np.inf, np.inf] for _ in range(4)])

    # Test valid x as list
    x = [1, 2, 3, 4]
    x, c, n, t = xcnt_handler(x)
    assert np.array_equal(x, [1, 2, 3, 4])

    # Test with missing input
    with pytest.raises(ValueError, match="Must enter some data!"):
        xcnt_handler()

    # Test with conflicting inputs for x and xl, xr
    with pytest.raises(
        ValueError, match="Must use either 'x' or both 'xl and 'xr'"
    ):
        xcnt_handler(x=[1, 2, 3], xl=[1, 2], xr=[3, 4])

    # Test with conflicting inputs for x and xl, xr
    with pytest.raises(
        ValueError, match="Must use either 'x' or both 'xl and 'xr'"
    ):
        xcnt_handler(x=[1, 2, 3], xl=[1, 2])

    # Test with conflicting inputs for x and xl, xr
    with pytest.raises(
        ValueError, match="Must use either 'x' or both 'xl and 'xr'"
    ):
        xcnt_handler(x=[1, 2, 3], xr=[3, 4])

    # Test with only xl input
    with pytest.raises(
        ValueError, match="Must use either 'x' or both 'xl and 'xr'"
    ):
        xcnt_handler(xl=[1, 2])

    # Test with only xr input
    with pytest.raises(
        ValueError, match="Must use either 'x' or both 'xl and 'xr'"
    ):
        xcnt_handler(xr=[3, 4])

    # Test mismatched dims for xl and xr
    with pytest.raises(
        ValueError, match="'xl' and 'xr' must be the same length"
    ):
        xcnt_handler(xl=[1, 2, 3], xr=[3, 4])

    # Test no array as sequence
    with pytest.raises(
        ValueError,
        match="'xl' and 'xr' must be an array of scalar"
        " numbers with real values.",
    ):
        xcnt_handler(xl=[1, 2], xr=[3, [3, 4]])

    # Test no array as sequence
    with pytest.raises(
        ValueError,
        match="'xl' and 'xr' must be an array of scalar numbers "
        "with real values.",
    ):
        xcnt_handler(xl=[1, [2, 3]], xr=[3, 4])

    # Test with valid xl and xr input
    x, _, _, _ = xcnt_handler(xl=[1, 2], xr=[3, 4])
    assert np.array_equal(x, [[1, 3], [2, 4]])

    # Test valid jagged array
    x, _, _, _ = xcnt_handler(x=[1, [2, 3]])
    assert np.array_equal(x, [[1, 1], [2, 3]])

    # Test invalid jagged array
    with pytest.raises(
        ValueError,
        match="Each element of 'x' must be either scalar or array-like"
        "of no more than length 2",
    ):
        xcnt_handler(x=[1, [2, 3, 4]])

    # Test with wrong dimension for x - list
    with pytest.raises(
        ValueError, match="Variable 'x' array must be one or two dimensional"
    ):
        xcnt_handler(x=np.array([[[1], [2]], [[3], [4]]]))

    # Test with wrong dimension for x - numpy array
    with pytest.raises(ValueError, match="Dimension 1 must be equal to 2"):
        xcnt_handler(x=np.array(np.array([[1, 2, 3], [3, 4, 5]])))

    # Test with valid x input
    x, c, n, t = xcnt_handler(x=[1, 2, 3, 4])
    assert np.array_equal(x, [1, 2, 3, 4])
    assert np.array_equal(c, [0, 0, 0, 0])
    assert np.array_equal(n, [1, 1, 1, 1])
    assert np.array_equal(t, [[-np.inf, np.inf] for _ in range(4)])

    # Test with x input as 2d array but with left interval
    # greater than right interval
    with pytest.raises(
        ValueError,
        match="All left intervals must be <= to right intervals",
    ):
        xcnt_handler(x=[[3, 2], [5, 4]])

    # Test with c input not matching the shape of x
    with pytest.raises(
        ValueError,
        match="censoring flag array must be same length as variable array",
    ):
        xcnt_handler(x=[1, 2, 3], c=[0, 1])

    # Test with n input not matching the shape of x
    with pytest.raises(
        ValueError, match="count array must be same length as variable array."
    ):
        xcnt_handler(x=[1, 2, 3], n=[1, 2])

    # Test with t, tl and tr
    with pytest.raises(ValueError, match="Cannot use 't' with 'tl' or 'tr'"):
        xcnt_handler(x=[1, 2], t=[[0, 3], [1, 4]], tl=[0, 1])
