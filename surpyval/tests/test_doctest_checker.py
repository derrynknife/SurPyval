"""The numeric fallback used by the ``--doctest-modules`` run.

``conftest._numerically_equal`` decides whether two blocks of doctest
output that differ as text are the same as numbers. It is the mechanism
that lets the docstring examples record real, untrimmed output while
still passing on every supported Python, so it needs its own tests: too
strict and the doctest step fails on a toolchain difference, too loose
and it stops catching the documentation drift it was added to catch.

The "tolerated" cases below are not invented. Each is a real pair of
outputs seen for the same example on two different Pythons in CI.
"""

import doctest

import pytest

from conftest import _numerically_equal

# The ``ProportionalIntensityHPP`` example, verbatim: the documented
# block on the left, the block CI produced on the right. Kept whole
# rather than reduced to one coefficient because the thing that broke
# here was not a number -- it was the ``<BLANKLINE>`` markers, which the
# text comparison strips from the expected output before matching and
# which the numeric fallback has to strip too. A test on an extracted
# line would have gone on passing.
HPP_DOCUMENTED = """\
Base Rate Parameters:
    lambda  :  0.012395105741757225
<BLANKLINE>
Covariate Coefficients:
   beta_0  :  0.06397367067847898
   beta_1  :  0.011491178797116433
   beta_2  :  -0.02147901865302258
<BLANKLINE>
"""

HPP_OBSERVED = """\
Base Rate Parameters:
    lambda  :  0.012395109943236718

Covariate Coefficients:
   beta_0  :  0.06397361219411789
   beta_1  :  0.011491197469342556
   beta_2  :  -0.021479544898597686

"""

# (documented output, output seen on another Python)
TOLERATED = [
    pytest.param(
        "     alpha: 913.84662107\n      beta: 1.4781707110680866\n",
        "     alpha: 913.8468395959314\n      beta: 1.4781709287931744\n",
        id="crow-amsaa-fit",
    ),
    pytest.param(
        # The loosest genuine disagreement found: 9e-4 relative on ``b``.
        "     alpha: 1.478202089169939\n         b: 4.199455086392048e-05\n",
        "     alpha: 1.4781024854527343\n         b: 4.203204430005199e-05\n",
        id="duane-fit",
    ),
    pytest.param(
        "   beta_2  :  -0.02147901865302258\n",
        "   beta_2  :  -0.021479544898597686\n",
        id="hpp-proportional-intensity-coefficient",
    ),
    pytest.param(
        # A restoration factor whose true value is zero; the optimiser
        # stops on whatever mantissa it stops on. Only ``abs_tol`` can
        # rescue this one.
        "Restoration Factor  : 1.3316262291443964e-16\n",
        "Restoration Factor  : 1.1551809284521243e-16\n",
        id="numerical-zero",
    ),
    pytest.param(
        "1.8227536487527594\n",
        "1.822753648752769\n",
        id="scipy-special-last-digits",
    ),
    pytest.param(
        # numpy re-pads the columns when an element gets shorter, so the
        # closing bracket moves.
        "     alpha: [ 6.32508961 17.37701969]\n",
        "     alpha: [ 6.3250866 17.377018 ]\n",
        id="array-repadded",
    ),
    pytest.param(HPP_DOCUMENTED, HPP_OBSERVED, id="repr-with-blank-lines"),
]

# Every one of these is a real defect this sweep found in the docstrings,
# or the class of defect it found. None may be forgiven.
REJECTED = [
    pytest.param(
        "3\n",
        "3.332162203618775\n",
        id="value-from-another-parameterisation",
    ),
    pytest.param(
        "11.229\n", "10.533288486847923\n", id="documented-variance-was-wrong"
    ),
    pytest.param(
        "array([1, 2, 3, 4, 5])\n",
        "array([1., 2., 3., 4., 5.])\n",
        id="dtype-changed",
    ),
    pytest.param(
        "array([0.83333333, 0.66666667])\n",
        "array([0.16666667, 0.33333333])\n",
        id="example-called-the-wrong-function",
    ),
    pytest.param(
        "array([1.0, 2.0])\n", "array([1.0, 2.0, 3.0])\n", id="shape-changed"
    ),
    pytest.param("alpha: 1.0\n", "beta: 1.0\n", id="different-label"),
    pytest.param("array([1.0])\n", "3.0\n", id="scalar-against-array"),
    pytest.param("", "1.0\n", id="expected-nothing"),
    pytest.param("1.0\n", "", id="produced-nothing"),
    pytest.param(
        # 2e-3 relative, twice the tolerance.
        "8.929795115692489\n",
        "8.947654505923874\n",
        id="drifted-past-the-tolerance",
    ),
]


@pytest.mark.parametrize("want, got", TOLERATED)
def test_the_same_number_under_a_different_toolchain_is_accepted(want, got):
    assert _numerically_equal(want, got)


@pytest.mark.parametrize("want, got", REJECTED)
def test_a_real_difference_is_still_a_failure(want, got):
    assert not _numerically_equal(want, got)


def test_output_with_no_numbers_is_not_silently_accepted():
    # Nothing to compare numerically, so the fallback must decline and
    # leave the verdict to the ordinary text comparison.
    assert not _numerically_equal("Fitted by : MLE\n", "Fitted by : MPS\n")
    assert not _numerically_equal("Fitted by : MLE\n", "Fitted by : MLE\n")


def test_the_checker_is_installed_on_the_doctest_base_class():
    # The examples' precision depends on this patch being live for the
    # whole doctest run, including under pytest's own subclass.
    checker = doctest.OutputChecker()
    assert checker.check_output("1.0000000\n", "1.0000001\n", 0)
    assert not checker.check_output("1.0\n", "2.0\n", 0)


def test_the_whole_comparison_path_handles_blank_lines():
    # Not the fallback in isolation but the method doctest actually
    # calls, with the flags the doctest step actually runs under. The
    # ``<BLANKLINE>`` handling lives in the text comparison, so only
    # this route proves the two halves agree about it.
    checker = doctest.OutputChecker()
    assert checker.check_output(
        HPP_DOCUMENTED, HPP_OBSERVED, doctest.NORMALIZE_WHITESPACE
    )
    assert not checker.check_output(
        HPP_DOCUMENTED,
        HPP_OBSERVED.replace("0.0123951", "0.0198765"),
        doctest.NORMALIZE_WHITESPACE,
    )
