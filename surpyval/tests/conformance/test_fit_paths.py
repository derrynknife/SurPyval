"""Alternate fit paths agree (#379).

A model that can be reached several ways -- ``fit``, ``fit_from_df``
(with column names or a formula), ``from_params`` with the fitted
parameters, ``fit_from_surpyval_data`` / ``fit_from_recurrent_data``,
Cox's ``fit_tvc`` with one interval per subject -- gives the same
predictions from the same data and the default options of each.
"""

from dataclasses import replace

import numpy as np
import pytest

from surpyval.tests.conformance.registry import (
    cases_for,
    fitted,
    predictions,
    refit,
)


def _path_params():
    # A known failure of one path is listed as "fit_paths[<path>]".
    params = []
    for param in cases_for("fit_paths"):
        case = param.values[0]
        for path in case.paths:
            marks = list(param.marks)
            reason = case.xfail.get(f"fit_paths[{path}]")
            if reason:
                marks.append(pytest.mark.xfail(strict=True, reason=reason))
            params.append(
                pytest.param(case, path, id=f"{case.name}-{path}", marks=marks)
            )
    return params


@pytest.mark.parametrize("case, path", _path_params())
def test_fit_paths_agree(case, path):
    ref = predictions(case, fitted(case))
    other = refit(replace(case, fit=case.paths[path]), case.data())
    got = predictions(case, other)
    for key in ref:
        np.testing.assert_allclose(
            got[key],
            ref[key],
            rtol=case.rtol,
            atol=case.rtol * 1e-2,
            err_msg=key,
        )
