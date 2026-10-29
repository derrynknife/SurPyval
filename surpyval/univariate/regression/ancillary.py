"""Covariates on a baseline distribution's other parameters (lifelines'
``ancillary``, flexsurv's ``anc``).

In an accelerated failure time model the covariates move one thing, the
time scale. With ``Ancillary(fitter, "beta")`` they move another of the
baseline's parameters as well, row by row: a Weibull AFT whose shape
depends on the covariates,

.. math::
    H(x \\mid Z) = \\left(\\frac{e^{b' Z} x}{\\alpha}\\right)^{\\beta(Z)},
    \\qquad \\beta(Z) = \\beta\\, e^{d' Z},

so the covariates change the shape of the hazard, not only its level. A
parameter bounded below by 0 (a shape, a scale) moves on the log scale,
:math:`p\\, e^{d' Z}`; an unbounded one (a location) by :math:`p + d' Z`.
``beta`` (or ``sigma``, ...) is then the parameter at ``Z = 0``, and the
coefficients ``d`` are named ``<parameter>_<column>``.

Every column of ``Z`` enters the linear predictor and each ancillary
parameter; hold an ancillary coefficient at 0 with ``fixed`` to leave a
column out of it. A parameter the family's covariates already move (the
Weibull AFT's scale ``alpha``) would not be identified, and is refused.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import autograd.numpy as np
import numpy.typing as npt

from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
)
from surpyval.univariate.regression._aliasing import dataframe_covariates
from surpyval.utils.surpyval_data import SurpyvalData, accepts_surpyval_data
from surpyval.utils.units import takes_unit

from ._covariate_link import CovariateLink
from ._fit_skeleton import (
    ORIGIN_MAPS,
    LogLinearPhi,
    MirroredDistributionAttrs,
    coefficient_map,
    fit_log_linear,
    mirror_distribution,
    optimise_nm_tnc,
)
from ._kinds import ACCELERATED_FAILURE_TIME
from ._likelihood import regression_neg_ll
from .parametric_regression_model import ParametricRegressionModel
from .regression_data import DataFrameRegressionMixin, truncation_window

#: The display (and saved) name of the link: the linear predictor's
#: ``exp(beta'Z)`` and the ancillary parameters' own.
LINK_NAME = "Log Linear [exp(beta'Z)] with ancillary covariates"


def _family_kind(family: Any) -> str:
    from .accelerated_failure_time import AFTFitter

    if isinstance(family, AFTFitter) and type(family) is AFTFitter:
        return ACCELERATED_FAILURE_TIME
    raise ValueError(
        "Ancillary takes a parametric AFT fitter (WeibullAFT, LogNormalAFT, "
        "AFT(Gamma), ...); got {!r}. Under proportional hazards or odds "
        "the baseline's scale and its other parameters interact (a Weibull "
        "PH with covariates on its shape has H = e^(b'Z) alpha^(-beta(Z)) "
        "x^beta(Z)), which is not the model R's flexsurv or lifelines "
        "fit; for the Weibull, Ancillary(WeibullAFT, 'beta') is the "
        "covariates-on-shape model.".format(family)
    )


class AncillaryFitter(MirroredDistributionAttrs, DataFrameRegressionMixin):
    """An AFT fitter whose covariates also move some of its baseline
    distribution's parameters; build one with :func:`Ancillary`.

    The parameters are the baseline's (at ``Z = 0``), the linear
    predictor's coefficients (one per column, named by the column or
    ``coef_j``), then for each ancillary parameter one coefficient per
    column, named ``<parameter>_<coefficient name>``.
    """

    fitter_kind = "regression fitter with ancillary covariates"

    def __init__(self, family: Any, parameters: Iterable[str]) -> None:
        kind = _family_kind(family)
        parameters = (
            (parameters,) if isinstance(parameters, str) else tuple(parameters)
        )
        mirror_distribution(self, family.dist)
        names = list(family.parameter_names)
        if not parameters:
            raise ValueError(
                "Name at least one parameter for the covariates to move "
                "(one of {}).".format(names)
            )
        unknown = [p for p in parameters if p not in names]
        if unknown:
            raise ValueError(
                "Unknown parameter(s) {} for `parameters`; the {} baseline "
                "has {}.".format(unknown, family.dist.name, names)
            )
        if len(set(parameters)) != len(parameters):
            raise ValueError(
                f"`parameters` names {parameters} more than once."
            )
        moved = ORIGIN_MAPS.get((kind, family.dist.name), ((), None))[0]
        clash = [p for p in parameters if names.index(p) in moved]
        if clash:
            raise ValueError(
                "The covariates of the {} model already move {} of the {} "
                "baseline, so covariates on it as well would not be "
                "identified. Choose another parameter (one of {}).".format(
                    kind,
                    clash,
                    family.dist.name,
                    [
                        p
                        for i, p in enumerate(names)
                        if i not in moved and p not in parameters
                    ]
                    or "none: every parameter is moved already",
                )
            )
        links = []
        for p in parameters:
            lower, upper = family.bounds[names.index(p)]
            if lower == 0 and upper is None:
                links.append("log")
            elif lower is None and upper is None:
                links.append("identity")
            else:
                raise ValueError(
                    "{!r} is bounded on both sides ({}, {}), and has no "
                    "link for covariates here; choose an unbounded or a "
                    "positive parameter.".format(p, lower, upper)
                )
        self.family = family
        self.kind = kind
        self.ancillary = parameters
        self.links = tuple(links)
        self.name = "{}AFT[ancillary: {}]".format(
            family.dist.name, ", ".join(parameters)
        )

    def __reduce__(self) -> Any:
        return (AncillaryFitter, (self.family, self.ancillary))

    # -- the model ---------------------------------------------------------

    def _split(self, Z: Numeric, params: tuple) -> tuple[list, tuple]:
        """The baseline's parameters for each row of ``Z`` (the ancillary
        ones moved by their covariates) and the linear predictor's
        coefficients."""
        Z_arr: Any = np.atleast_2d(np.asarray(Z, dtype=float))
        k, p = self.k_dist, Z_arr.shape[1]
        dist = list(params[:k])
        life = tuple(params[k : k + p])
        anc = params[k + p :]
        for a, (name, link) in enumerate(zip(self.ancillary, self.links)):
            j = self.param_map[name]
            lin = np.dot(Z_arr, np.array(list(anc[a * p : (a + 1) * p])))
            if link == "log":
                with np.errstate(over="ignore"):
                    dist[j] = dist[j] * np.exp(lin)
            else:
                dist[j] = dist[j] + lin
        return dist, life

    def _call(self, fn: str, x: Numeric, Z: Numeric, params: tuple) -> Any:
        dist, life = self._split(Z, params)
        return getattr(self.family, fn)(x, Z, *dist, *life)

    def Hf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        """Cumulative hazard at ``x`` for covariates ``Z``; ``params`` the
        baseline's, the coefficients, then the ancillary coefficients."""
        return self._call("Hf", x, Z, params)

    def hf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        """Hazard rate; as :meth:`Hf`."""
        return self._call("hf", x, Z, params)

    def sf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        """Survival function; as :meth:`Hf`."""
        return self._call("sf", x, Z, params)

    def ff(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        """Failure function; as :meth:`Hf`."""
        return self._call("ff", x, Z, params)

    def df(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        """Density; as :meth:`Hf`."""
        return self._call("df", x, Z, params)

    def log_sf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        return self._call("log_sf", x, Z, params)

    def log_ff(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        return self._call("log_ff", x, Z, params)

    def log_df(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        return self._call("log_df", x, Z, params)

    def neg_ll(self, data: SurpyvalData, *params: Boxable) -> Boxable:
        return regression_neg_ll(self, data, *params)

    @staticmethod
    def _life_phi(Z: Numeric, *params: Boxable) -> Boxable:
        """``exp(b'Z)``, the linear predictor's multiplier (the first
        ``Z.shape[1]`` of the covariate parameters)."""
        Z_arr: Any = np.atleast_2d(np.asarray(Z, dtype=float))
        return LogLinearPhi.phi(Z_arr, *params[: Z_arr.shape[1]])

    def _param_map(self, Z: npt.NDArray) -> dict[str, int]:
        life = coefficient_map(Z, self.param_map)
        names = sorted(life, key=life.__getitem__)
        out = dict(life)
        for name in self.ancillary:
            for col in names:
                out[f"{name}_{col}"] = len(out)
        return out

    def link(self, pmap: dict[str, int]) -> CovariateLink:
        """The covariate link a fitted model carries."""
        return CovariateLink(LINK_NAME, pmap, self._life_phi)

    @dataframe_covariates
    @takes_unit
    @accepts_surpyval_data
    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        t: npt.ArrayLike | None = None,
        init: npt.ArrayLike | None = None,
        fixed: dict[str, float] | None = None,
        center: bool = False,
        tl: npt.ArrayLike | None = None,
        tr: npt.ArrayLike | None = None,
    ) -> ParametricRegressionModel:
        """
        Fit by maximum likelihood, with every censoring and truncation of
        the family's ``fit``.

        Parameters
        ----------
        x, Z, c, n, t, tl, tr : array_like
            As for the family's ``fit``.
        init : array_like, optional
            Initial values: the baseline's, the coefficients, then the
            ancillary coefficients. Default the univariate fit and zeros.
        fixed : dict, optional
            Parameters to hold, by name; an ancillary coefficient held at
            0 leaves its column out of that parameter.
        center : bool, optional
            ``True`` reports the baseline, the ancillary parameters
            included, at the covariate means (``model.center``) instead of
            at ``Z = 0``.

        Returns
        -------
        ParametricRegressionModel
            The fitted model.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Ancillary, WeibullAFT
        >>> rng = np.random.default_rng(3)
        >>> Z = rng.binomial(1, 0.5, (400, 1))
        >>> shape = 1.2 * np.exp(0.6 * Z[:, 0])
        >>> x = 10 * rng.weibull(shape) * np.exp(-0.3 * Z[:, 0])
        >>> model = Ancillary(WeibullAFT, "beta").fit(x, Z)
        >>> model.parameter_names
        ['alpha', 'beta', 'coef_0', 'beta_coef_0']
        """
        t = truncation_window(x, t, tl, tr)
        return fit_log_linear(
            self,
            x,
            Z,
            c,
            n,
            t,
            init,
            fixed,
            center,
            kind=self.kind,
            optimiser=optimise_nm_tnc,
            reg_model=self.link,
            phi_bounds=lambda Z_: ((None, None),)
            * (Z_.shape[1] * (1 + len(self.ancillary))),
            phi_param_map=self._param_map,
            phi_init=lambda Z_: np.zeros(
                Z_.shape[1] * (1 + len(self.ancillary))
            ),
            log_linear=False,
        )


def Ancillary(fitter: Any, parameters: str | Iterable[str]) -> AncillaryFitter:
    """
    A regression fitter whose covariates also move the baseline
    distribution's ``parameters`` (lifelines' ``ancillary``, flexsurv's
    ``anc``): for example the shape of a Weibull AFT,
    ``Ancillary(WeibullAFT, "beta")``.

    Parameters
    ----------
    fitter : AFTFitter
        The AFT family's baseline, e.g. ``WeibullAFT``, ``LogNormalAFT``,
        ``AFT(Gamma)``. Proportional hazards and odds are refused (see
        the error for why).
    parameters : str or list of str
        The baseline parameters the covariates move. A positive one moves
        on the log scale (``beta * exp(d'Z)``), an unbounded one by
        ``+ d'Z``. One the family's covariates already move (the AFT's
        scale) is refused.

    Returns
    -------
    AncillaryFitter

    Examples
    --------
    >>> from surpyval import Ancillary, LogNormalAFT, WeibullAFT
    >>> Ancillary(WeibullAFT, "beta").name
    'WeibullAFT[ancillary: beta]'
    >>> Ancillary(LogNormalAFT, ["sigma"]).links
    ('log',)
    """
    return AncillaryFitter(fitter, parameters)
