"""Which of a joint model's varying path parameters are correlated between
items (``random_effects=``): every pair (``"full"``, the default), none
(``"diagonal"``) or blocks of names.

The checks:

- the option: its one form, and its refusals, each saying what to give;
- the layout: the Cholesky factor's entries between blocks are not
  parameters, so the spread is block diagonal, and the full layout is
  the one it always was;
- the likelihood: a model with some correlations left out has the
  log-likelihood, and the predictions, of the full model at the same
  parameters with those correlations 0, through the search's own
  ``search_space``;
- fits: a diagonal fit has no correlation, counts its parameters in
  ``aic()`` and is nested in the full fit; ``"full"`` given is the
  default number for number; a fleet simulated without the correlation
  is recovered (``_fleet.simulate``, seed 451, chosen before the fit was
  seen); blocks of a three-parameter path; the accelerated and the
  two-stage fits; ``compare_links``;
- saving and loading, and the schema that a SurPyval 0.28 reader
  refuses.
"""

import functools
import pickle
import warnings

import numpy as np
import pytest

import surpyval
from surpyval import JointAFT, JointModel, JointPH, Weibull, WeibullJointPH
from surpyval.degradation import QuadraticPath
from surpyval.serialisation import required_schema
from surpyval.tests.univariate.joint._fleet import CENTRE, TRUTH, fleet
from surpyval.tests.univariate.joint.test_joint_integration import (
    ALPHA,
    BETA,
    EFFECT,
    MEANS,
    NOISE,
    SDS,
    simulate,
)
from surpyval.univariate.joint._likelihood import (
    Layout,
    block_labels,
    check_random_effects,
)
from surpyval.univariate.joint.compare import _candidates
from surpyval.univariate.joint.joint_model import parameter_names
from surpyval.univariate.joint.joint_ph import search_space

# -- the option ---------------------------------------------------------------


@pytest.mark.parametrize(
    "given, expected",
    [
        ("full", "full"),
        ("diagonal", "diagonal"),
        ([("a", "b", "c")], "full"),
        (["a", "b", "c"], "diagonal"),
        ([("c",), ("a",), ("b",)], "diagonal"),
        ([("a", "b"), ("c",)], (("a", "b"), ("c",))),
        ([("a", "b"), "c"], (("a", "b"), ("c",))),
        # in the path's order, the blocks by their first
        ([["c", "a"], ["b"]], (("a", "c"), ("b",))),
        ((["b"], ("c", "a")), (("a", "c"), ("b",))),
    ],
)
def test_the_structure_has_one_form(given, expected):
    assert check_random_effects(given, ("a", "b", "c")) == expected


def test_one_varying_parameter_is_the_full_structure():
    assert check_random_effects("diagonal", ("a",)) == "full"
    fitter = JointPH(Weibull, varying="a", random_effects="diagonal")
    assert fitter.random_effects == "full"


def test_the_blocks_label_the_varying_parameters():
    varying = ("a", "b", "c")
    assert block_labels("full", varying) == (0, 0, 0)
    assert block_labels("diagonal", varying) == (0, 1, 2)
    assert block_labels((("a", "c"), ("b",)), varying) == (0, 1, 0)


@pytest.mark.parametrize(
    "given, match",
    [
        ("independent", r"'full' .*'diagonal' .*\['a', 'b', 'c'\]"),
        (3, r"'full' .*'diagonal' .*got 3"),
        ({"a": "b"}, r"'full' .*'diagonal' .*got \{'a': 'b'\}"),
        ([("a", "d"), ("b", "c")], r"\['d'\].*those are \['a', 'b', 'c'\]"),
        ([("a", "b"), ("b", "c")], r"\['b'\] more than once.*exactly one"),
        ([("a", "b")], r"leaves out \['c'\].*block of its own.*\('c',\)"),
        ([("a", "b", "c"), ()], "an empty block"),
        ([], r"leaves out \['a', 'b', 'c'\]"),
    ],
)
def test_a_structure_that_does_not_fit_the_varying_parameters_is_refused(
    given, match
):
    with pytest.raises(ValueError, match=match):
        JointPH(Weibull, QuadraticPath, random_effects=given)


def test_a_common_parameter_has_no_correlation_to_choose():
    with pytest.raises(ValueError, match="common to all items"):
        JointPH(
            Weibull,
            QuadraticPath,
            varying=("a", "b"),
            random_effects=[("a", "c"), ("b",)],
        )


def test_the_fitters_say_their_structure():
    assert "random effects" not in repr(WeibullJointPH)
    assert repr(JointPH(Weibull, random_effects="diagonal")).endswith(
        "random effects: uncorrelated)"
    )
    blocks = JointAFT(Weibull, QuadraticPath, random_effects=[("a", "c"), "b"])
    assert blocks.random_effects == (("a", "c"), ("b",))
    assert repr(blocks).endswith(
        "random effects: correlated in blocks {a, c}, {b})"
    )


# -- the layout ---------------------------------------------------------------


def test_the_full_layout_is_unchanged():
    layout = Layout(3, 1, (0, 1, 2))
    assert layout.blocks == (0, 0, 0)
    assert layout.lower == [(1, 0), (2, 0), (2, 1)]
    assert layout.n_spread == 6
    assert layout.size == 3 + 3 + 6 + 3 + 1 + 1
    assert layout.hazard_positions() == [(15, 0)]


def test_a_block_layout_has_no_factor_between_its_blocks():
    layout = Layout(3, 1, (0, 1, 2), blocks=(0, 1, 0))
    assert layout.lower == [(2, 0)]
    assert layout.n_spread == 4
    assert layout.size == 3 + 3 + 4 + 3 + 1 + 1
    assert layout.hazard_positions() == [(13, 0)]
    u = np.random.default_rng(1).normal(size=layout.size)
    theta = layout.unpack(u)
    L = np.asarray(theta["L"])
    assert L[1, 0] == 0.0 and L[2, 1] == 0.0 and L[2, 0] != 0.0
    D = L @ L.T
    assert D[0, 1] == D[1, 2] == 0.0
    # (the log of the exp of the sds: to rounding)
    np.testing.assert_allclose(layout.pack(theta), u, rtol=1e-14)


def test_the_names_follow_the_structure():
    names = ["a", "b", "c"]
    full = parameter_names(names, names, [])
    assert [n for n in full if n.startswith("corr")] == [
        "corr_a_b",
        "corr_a_c",
        "corr_b_c",
    ]
    diagonal = parameter_names(names, names, [], random_effects="diagonal")
    assert diagonal == [n for n in full if not n.startswith("corr")]
    blocks = parameter_names(
        names, names, [], random_effects=(("a", "c"), ("b",))
    )
    assert [n for n in blocks if n.startswith("corr")] == ["corr_a_c"]


# -- the likelihood -----------------------------------------------------------


def _model_at(structure, corr):
    """A quadratic joint model at the truth of the integration tests'
    fleets, every parameter varying, with the structure ``structure`` and
    the correlations ``corr`` (by name; those not given 0), without a
    fit."""
    names = ["a", "b", "c"]
    structure = check_random_effects(structure, names)
    values = dict(zip(names, MEANS[:3]))
    values.update({f"sd_{n}": s for n, s in zip(names, SDS[:3])})
    values.update(
        sd_noise=NOISE, alpha=ALPHA, beta=BETA, marker_effect=EFFECT, **corr
    )
    out = JointModel()
    out.dist = Weibull
    out.path_model = QuadraticPath
    out.varying = names
    out.random_effects_structure = structure
    out.parameter_names = parameter_names(
        names, names, [], random_effects=structure
    )
    out.params = np.array([values.get(n, 0.0) for n in out.parameter_names])
    out.feature_names = []
    out.aliased = np.array([], dtype=int)
    out.marker_reference = 3.6
    out._centre = 3.6
    out._zbar = np.zeros(0)
    out.k = len(out.params)
    return out


@functools.cache
def _quadratic_fleet():
    return simulate(40, seed=4511, q=3)


def _neg_ll(model, data):
    f, u, _ = search_space(model, **data)
    return float(f(u))


@pytest.mark.parametrize(
    "structure, corr",
    [
        ("diagonal", {}),
        ([("a", "c"), ("b",)], {"corr_a_c": 0.4}),
        ([("a",), ("b", "c")], {"corr_b_c": -0.3}),
    ],
)
def test_leaving_out_a_correlation_is_holding_it_at_0(structure, corr):
    data = _quadratic_fleet()
    structured = _model_at(structure, corr)
    full = _model_at("full", corr)
    assert structured.k < full.k
    np.testing.assert_allclose(
        _neg_ll(structured, data), _neg_ll(full, data), rtol=1e-13
    )
    question = dict(markers={0.0: 3.1, 0.5: 3.6, 1.0: 3.8}, given=1.0)
    np.testing.assert_allclose(
        structured.sf([1.5, 2.0, 2.5], **question),
        full.sf([1.5, 2.0, 2.5], **question),
        rtol=1e-12,
    )


# -- fits ---------------------------------------------------------------------


@functools.cache
def _fleet():
    """Lines (``a + b t``, ``a`` and ``b`` independent) read every 0.1
    years for up to 3."""
    return simulate(120, seed=4512, q=2)


@functools.cache
def _fits():
    data = _fleet()
    return WeibullJointPH.fit(**data), JointPH(
        Weibull, random_effects="diagonal"
    ).fit(**data)


def test_a_diagonal_fit_has_no_correlation():
    full, diagonal = _fits()
    assert diagonal.maximum == "verified"
    assert diagonal.random_effects_structure == "diagonal"
    assert diagonal.parameter_names == [
        n for n in full.parameter_names if n != "corr_a_b"
    ]
    assert diagonal.k == full.k - 1 == len(diagonal.params)
    assert diagonal.covariance().shape == (diagonal.k,) * 2
    np.testing.assert_allclose(
        diagonal.aic(), 2 * diagonal.k + 2 * diagonal.neg_ll()
    )
    # the full model nests it
    assert full.neg_ll() <= diagonal.neg_ll() + 1e-8
    parts = set(diagonal.summary().index.get_level_values("part"))
    assert "spread between items (uncorrelated)" in parts
    assert "spread between items" in set(
        full.summary().index.get_level_values("part")
    )
    assert "varying between items: a, b (uncorrelated)" in repr(diagonal)
    bounds = diagonal.cb(_X, **_QUESTION)
    assert bounds.shape == (2, 2) and np.all(np.isfinite(bounds))


_X = [1.5, 2.0]
_QUESTION = dict(markers={0.0: 3.1, 0.5: 3.6, 1.0: 3.8}, given=1.0)


def test_a_diagonal_fit_is_the_full_likelihood_with_the_correlation_at_0():
    full, diagonal = _fits()
    held = JointModel.from_dict(diagonal.to_dict())
    held.random_effects_structure = "full"
    held.parameter_names = full.parameter_names
    values = dict(zip(diagonal.parameter_names, diagonal.params))
    held.params = np.array([values.get(n, 0.0) for n in full.parameter_names])
    held._covariance = None
    data = _fleet()
    np.testing.assert_allclose(
        _neg_ll(held, data), _neg_ll(diagonal, data), rtol=1e-12
    )
    # ... which is the fit's own
    np.testing.assert_allclose(
        _neg_ll(diagonal, data), diagonal.neg_ll(), rtol=1e-9
    )
    np.testing.assert_allclose(
        held.sf(_X, **_QUESTION), diagonal.sf(_X, **_QUESTION), rtol=1e-12
    )


def test_full_given_is_the_default_number_for_number():
    full, _ = _fits()
    given = JointPH(Weibull, random_effects=[("a", "b")]).fit(**_fleet())
    assert given.random_effects_structure == "full"
    np.testing.assert_array_equal(given.params, full.params)
    np.testing.assert_array_equal(given.covariance(), full.covariance())
    assert given.neg_ll() == full.neg_ll()
    assert "random_effects" not in full.to_dict()


def test_a_fleet_without_the_correlation_is_recovered():
    truth = dict(TRUTH, corr_a_b=0.0)
    data = fleet(300, seed=451, truth=truth)
    model = JointPH(Weibull, random_effects="diagonal").fit(
        data, marker_reference=CENTRE
    )
    assert model.maximum == "verified"
    expected = np.array([truth[n] for n in model.parameter_names])
    z = (model.params - expected) / model.standard_errors()
    assert np.all(np.abs(z) < 3), dict(zip(model.parameter_names, z))


@functools.cache
def _blocks():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return JointPH(
            Weibull, QuadraticPath, random_effects=[("a", "b"), ("c",)]
        ).fit(**_quadratic_fleet())


def test_blocks_of_a_three_parameter_path_fit():
    model = _blocks()
    assert model.random_effects_structure == (("a", "b"), ("c",))
    assert model.parameter_names[3:8] == [
        "sd_a",
        "sd_b",
        "sd_c",
        "corr_a_b",
        "sd_noise",
    ]
    assert np.all(np.isfinite(model.params))
    assert "(correlated in blocks {a, b}, {c})" in repr(model)
    # the fit's likelihood is the full layout's with the others at 0
    held = _model_at("full", {})
    held.params = np.array(
        [
            dict(zip(model.parameter_names, model.params)).get(n, 0.0)
            for n in held.parameter_names
        ]
    )
    held.marker_reference = model.marker_reference
    held._centre = model._centre
    data = _quadratic_fleet()
    np.testing.assert_allclose(
        _neg_ll(held, data), _neg_ll(model, data), rtol=1e-12
    )


def test_the_accelerated_and_two_stage_fits_take_the_structure():
    data = _fleet()
    accelerated = JointAFT(Weibull, random_effects="diagonal").fit(**data)
    assert accelerated.maximum == "verified"
    assert "corr_a_b" not in accelerated.parameter_names
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quick = JointPH(Weibull, random_effects="diagonal").fit(
            **data, how="two-stage"
        )
    assert quick.parameter_names == _fits()[1].parameter_names
    assert np.all(np.isfinite(quick.params))


def test_compare_links_keeps_the_structure():
    fitter = JointPH(Weibull, random_effects="diagonal")
    fitters, _, _ = _candidates(fitter, None)
    assert [f.random_effects for f in fitters] == ["diagonal"] * len(fitters)


# -- saving -------------------------------------------------------------------


def test_the_structure_is_saved_and_restored():
    _, diagonal = _fits()
    saved = diagonal.to_dict()
    assert saved["random_effects"] == "diagonal"
    restored = surpyval.from_dict(saved)
    assert restored.random_effects_structure == "diagonal"
    assert restored.parameter_names == diagonal.parameter_names
    np.testing.assert_array_equal(
        restored.sf(_X, **_QUESTION), diagonal.sf(_X, **_QUESTION)
    )
    np.testing.assert_array_equal(
        restored.cb(_X, **_QUESTION), diagonal.cb(_X, **_QUESTION)
    )
    unpickled = pickle.loads(pickle.dumps(diagonal))
    assert unpickled.random_effects_structure == "diagonal"


def test_blocks_are_saved_as_lists():
    model = _blocks()
    saved = model.to_dict()
    assert saved["random_effects"] == [["a", "b"], ["c"]]
    restored = JointModel.from_dict(saved)
    assert restored.random_effects_structure == (("a", "b"), ("c",))
    question = dict(markers={0.0: 3.1, 0.5: 3.6, 1.0: 3.8}, given=1.0)
    np.testing.assert_array_equal(
        restored.sf([1.5, 2.0], **question), model.sf([1.5, 2.0], **question)
    )


def test_a_structured_model_saves_as_schema_5_which_0_28_refuses():
    # SurPyval 0.28 reads schema 4 and below; it would ignore
    # "random_effects" and read the parameters after the sds as every
    # pair's correlation, misplacing each one after them.
    full, diagonal = _fits()
    assert full.to_dict()["schema"] < 5
    assert diagonal.to_dict()["schema"] == 5
    saved = full.to_dict()
    assert required_schema(dict(saved, random_effects="full")) < 5
    assert required_schema(dict(saved, random_effects="diagonal")) == 5
    assert required_schema(dict(saved, random_effects=[["a"], ["b"]])) == 5


def test_a_saved_model_without_the_structure_is_full():
    full, _ = _fits()
    saved = full.to_dict()
    assert "random_effects" not in saved
    assert JointModel.from_dict(saved).random_effects_structure == "full"


def test_a_saved_structure_that_does_not_fit_is_refused():
    _, diagonal = _fits()
    saved = diagonal.to_dict()
    with pytest.raises(ValueError, match="'random_effects'"):
        JointModel.from_dict(dict(saved, random_effects="banded"))
    with pytest.raises(ValueError, match=r"leaves out \['b'\]"):
        JointModel.from_dict(dict(saved, random_effects=[["a"]]))
    # names of the full structure under a diagonal one
    full, _ = _fits()
    edited = dict(full.to_dict(), random_effects="diagonal")
    with pytest.raises(ValueError, match="'parameter_names' are not those"):
        JointModel.from_dict(edited)
