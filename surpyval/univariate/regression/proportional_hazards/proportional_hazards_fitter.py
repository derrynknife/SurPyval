import inspect
from typing import Any, Callable

import autograd.numpy as np
import numpy as onp
import numpy.typing as npt
from scipy.optimize import brentq

from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
)
from surpyval.utils.surpyval_data import SurpyvalData

from .._fit_skeleton import (
    HazardIdentitiesMixin,
    LogLinearPhi,
    MirroredDistributionAttrs,
    assemble_regression_model,
    make_objective,
    mirror_distribution,
    optimise_ph,
    prepare_regression_fit,
)
from .._likelihood import regression_neg_ll
from ..parametric_regression_model import ParametricRegressionModel
from ..regression_data import DataFrameRegressionMixin
from ..tvc_fit import TVCFitMixin


class Phi:
    # Lightweight namespace whose attributes are populated by the fitter.
    phi: Any
    phi_param_map: Any
    name: str


class ProportionalHazardsFitter(
    MirroredDistributionAttrs,
    HazardIdentitiesMixin,
    TVCFitMixin,
    DataFrameRegressionMixin,
):
    """
    Parametric proportional hazards fitter: a parametric baseline hazard
    multiplied by a covariate function,

    .. math::
        h(x \\mid Z) = \\phi(Z)\\, h_0(x), \\qquad
        H(x \\mid Z) = \\phi(Z)\\, H_0(x),

    with :math:`\\phi(Z) = e^{\\beta' Z}` for the pre-built instances
    (``WeibullPH``, ``ExponentialPH``, ...) and for ``PH(dist)``. A positive
    coefficient raises the hazard (shortens life).

    Use the pre-built instances or the ``PH`` factory rather than building
    this class directly; the constructor exists for a custom ``phi(Z,
    *params)`` with its own bounds and parameter names. ``fit`` returns a
    :class:`~surpyval.univariate.regression.parametric_regression_model.ParametricRegressionModel`.

    Parameters
    ----------
    name : str
        The fitter's name (e.g. ``"WeibullLinearRR"``).
    dist : ParametricFitter
        The baseline distribution (e.g. ``Weibull``).
    phi : callable
        The covariate function, with the signature ``phi(Z, *params)``,
        written with ``autograd.numpy`` so the likelihood can be
        differentiated; it must be positive at the fitted parameters.
    phi_name : str
        A display name for ``phi``, shown in the fitted model's ``repr``.
    phi_bounds : tuple or callable
        The ``(lower, upper)`` bounds of each ``phi`` parameter (``None``
        for unbounded), or a function of the covariate matrix returning
        them. The bounds are how ``phi`` is kept positive.
    phi_param_map : dict or callable
        ``{name: position}`` of the ``phi`` parameters, or a function of the
        covariate matrix returning it.
    phi_init : callable, optional
        A function of the covariate matrix returning starting values for
        the ``phi`` parameters. Defaults to zeros.

    A model with a custom ``phi`` predicts, and gives bounds, like the
    pre-built ones, but cannot be serialised (``phi`` cannot be rebuilt
    from a name).

    Examples
    --------
    An excess-relative-risk model, :math:`\\phi(z) = 1 + \\beta z` with
    :math:`\\beta > 0`:

    >>> import numpy as np
    >>> import autograd.numpy as anp
    >>> from surpyval import ProportionalHazardsFitter, Weibull
    >>> def linear_rr(Z, *params):
    ...     return 1.0 + anp.dot(Z, anp.array(params))
    >>> WeibullLinearRR = ProportionalHazardsFitter(
    ...     "WeibullLinearRR", Weibull, linear_rr, "Linear [1 + beta'Z]",
    ...     phi_bounds=lambda Z: ((0, None),) * Z.shape[1],
    ...     phi_param_map=lambda Z: {
    ...         f"beta_{i}": i for i in range(Z.shape[1])
    ...     },
    ...     phi_init=lambda Z: np.full(Z.shape[1], 0.5),
    ... )
    >>> rng = np.random.default_rng(0)
    >>> dose = rng.uniform(0, 4, 400)
    >>> x = 10 * (-np.log(rng.uniform(size=400)) / (1 + 0.5 * dose)) ** 0.5
    >>> WeibullLinearRR.fit(x=x, Z=dose).params.round(3)
    array([10.15 ,  2.081,  0.604])
    """

    def __init__(
        self,
        name: str,
        dist: Any,
        phi: Callable,
        phi_name: str,
        phi_bounds: "Callable[[npt.NDArray], tuple] | tuple",
        phi_param_map: "Callable[[npt.NDArray], dict] | dict",
        phi_init: "Callable[[npt.NDArray], npt.NDArray] | None" = None,
    ) -> None:
        # Compare names and kinds rather than the signature's string
        # form so an annotated phi (e.g. ``LogLinearPhi.phi``) passes.
        phi_sig = list(inspect.signature(phi).parameters.values())
        if not (
            len(phi_sig) == 2
            and phi_sig[0].name == "Z"
            and phi_sig[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            and phi_sig[1].name == "params"
            and phi_sig[1].kind is inspect.Parameter.VAR_POSITIONAL
        ):
            raise ValueError(
                "PH function must have the signature '(Z, *params)'"
            )

        self.name = name
        mirror_distribution(self, dist)
        self.phi = phi
        self.phi_name = phi_name
        self.Hf_dist = self.dist.Hf
        self.hf_dist = self.dist.hf
        self.sf_dist = self.dist.sf
        self.ff_dist = self.dist.ff
        self.df_dist = self.dist.df
        self.phi_init = phi_init
        self.phi_bounds = phi_bounds
        self.phi_param_map = phi_param_map

    def Hf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        """
        Cumulative hazard :math:`\\phi(Z) H_0(x)` at ``x`` for covariates
        ``Z``; ``params`` are the distribution parameters followed by the
        covariate coefficients.
        """
        dist_params = np.array(params[0 : self.k_dist])
        phi_params = np.array(params[self.k_dist :])
        Hf_raw = self.Hf_dist(x, *dist_params)
        return self.phi(Z, *phi_params) * Hf_raw

    def hf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        """
        Hazard rate :math:`\\phi(Z) h_0(x)` at ``x`` for covariates ``Z``;
        ``params`` as for :meth:`Hf`.
        """
        dist_params = np.array(params[0 : self.k_dist])
        phi_params = np.array(params[self.k_dist :])
        hf_raw = self.hf_dist(x, *dist_params)
        return self.phi(Z, *phi_params) * hf_raw

    def mpp_inv_y_transform(self, y: Numeric, *params: Boxable) -> Numeric:
        return y

    def mpp_y_transform(self, y: Numeric, *params: Boxable) -> Numeric:
        return y

    def random(
        self, size: int, Z: npt.ArrayLike, *params: float
    ) -> tuple[npt.NDArray, npt.NDArray]:
        dist_params = np.array(params[0 : self.k_dist])
        phi_params = np.array(params[self.k_dist :])
        Z_arr = np.atleast_2d(np.asarray(Z, dtype=float))
        x = []
        Z_out = []
        for row in Z_arr:
            phi = self.phi(row, *phi_params)
            U = np.random.uniform(0, 1, size)
            x.append(
                self._invert_cumulative_hazard(-np.log(U) / phi, dist_params)
            )
            Z_out.append(np.tile(row, (size, 1)))
        return np.concatenate(x), np.vstack(Z_out)

    def _invert_cumulative_hazard(
        self, h: npt.NDArray, dist_params: npt.NDArray
    ) -> npt.NDArray:
        """
        The times at which the baseline cumulative hazard reaches ``h``.

        ``S(x|Z) = S0(x)^phi``, so a draw ``U`` of the survival is reached
        where ``H0(x) = -log(U) / phi``. Through the quantile function that
        is ``qf(1 - exp(-h))``; for a very small hazard multiplier ``h`` is
        so large that ``1 - exp(-h)`` rounds to 1 and ``qf`` returns
        ``inf``, although the time is finite. Those draws are solved on
        ``log H0(x) = log h`` directly.
        """
        h = np.asarray(h, dtype=float)
        with onp.errstate(divide="ignore", over="ignore", invalid="ignore"):
            out = np.asarray(
                self.dist.qf(-np.expm1(-h), *dist_params), dtype=float
            )
        lost = ~np.isfinite(out) & np.isfinite(h)
        if lost.any():
            out = out.copy()
            start = float(self.dist.qf(0.5, *dist_params))
            for k in np.flatnonzero(lost):
                target = np.log(h[k])

                def gap(t: float) -> float:
                    with onp.errstate(all="ignore"):
                        return float(
                            np.log(self.dist.Hf(t, *dist_params)) - target
                        )

                upper = max(start, 1.0)
                for _ in range(2000):
                    if gap(upper) >= 0:
                        break
                    upper *= 2.0
                lower = upper / 2.0 if upper > start else 0.0
                out[k] = brentq(gap, lower, upper, xtol=1e-300, rtol=1e-12)
        return out

    def neg_ll(self, data: SurpyvalData, *params: Boxable) -> Boxable:
        return regression_neg_ll(self, data, *params)

    @staticmethod
    def create(distribution: Any) -> "ProportionalHazardsFitter":
        """
        Create a Proportional Hazards fitter for the given distribution using
        exp(beta'Z) as the hazard multiplier.

        Parameters
        ----------
        distribution : ParametricFitter
            A surpyval parametric distribution (e.g. ``Weibull``,
            ``Exponential``).

        Returns
        -------
        ProportionalHazardsFitter
            A configured fitter with a ``.fit(x, Z, ...)`` method.
        """
        return ProportionalHazardsFitter.create_general_log_linear_fitter(
            f"{distribution.name}PH", distribution
        )

    @classmethod
    def create_general_log_linear_fitter(
        cls, name: str, distribution: Any
    ) -> "ProportionalHazardsFitter":
        return cls(
            name,
            distribution,
            LogLinearPhi.phi,
            LogLinearPhi.NAME_E,
            LogLinearPhi.phi_bounds,
            phi_param_map=LogLinearPhi.make_param_map,
            phi_init=lambda Z: np.zeros(Z.shape[1]),
        )

    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        t: npt.ArrayLike | None = None,
        init: npt.ArrayLike | None = None,
        fixed: dict[str, float] | None = None,
    ) -> ParametricRegressionModel:
        """
        Fit the proportional hazards model to the data.

        Parameters
        ----------

        x : array_like
            The observed event times.
        Z : array_like
            The covariates to fit the model to.
        c : array_like, optional
            The censoring indicators.
        n : array_like, optional
            The number of observations at each time.
        t : array_like, optional
            Truncation bounds: an (N, 2) array of the left and right
            truncation times of each observation.
        init : array_like, optional
            The initial values for the parameters: the distribution
            parameters followed by the covariate coefficients.
        fixed : dict, optional
            A dictionary of parameters to fix to a specific value, by name
            (a distribution parameter such as ``"beta"``, or a coefficient
            ``"beta_0"``, ``"beta_1"``, ...).

        Returns
        -------

        ParametricRegressionModel
            The fitted model.

        Examples
        --------

        >>> from surpyval import WeibullPH
        >>> from surpyval.datasets import load_tires_data
        >>> data = load_tires_data()
        >>> x = data['Survival'].values
        >>> c = data['Censoring'].values
        >>> Z = data[[
        ...     'Wedge gauge', 'Interbelt gauge', 'Peel force',
        ...     'Wedge gauge×peel force'
        ... ]].values
        >>> model = WeibullPH.fit(x=x, Z=Z, c=c)
        >>> model
        Parametric Regression SurPyval Model
        ====================================
        Kind                : Proportional Hazard
        Distribution        : Weibull
        Regression Model    : Log Linear [e^(beta'Z)]
        Fitted by           : MLE
        Distribution        :
             alpha: 0.2425513627560218
              beta: 16.057785182711932
        Regression Model    :
            beta_0: -9.165062726518311
            beta_1: -7.998573055929788
            beta_2: -27.50318580568538
            beta_3: 18.385445332039488
        >>> model = WeibullPH.fit(x=x, Z=Z, c=c, fixed={"beta": 15})
        >>> model
        Parametric Regression SurPyval Model
        ====================================
        Kind                : Proportional Hazard
        Distribution        : Weibull
        Regression Model    : Log Linear [e^(beta'Z)]
        Fitted by           : MLE
        Distribution        :
             alpha: 0.237729668424067
              beta: 15.0
        Regression Model    :
            beta_0: -8.62832691738283
            beta_1: -7.617529362323243
            beta_2: -25.952367249502934
            beta_3: 17.270148387391387
        """
        data, prep = prepare_regression_fit(
            self,
            x,
            Z,
            c,
            n,
            t,
            init,
            fixed,
            self.phi_bounds,
            self.phi_param_map,
            self.phi_init,
        )
        init_t, bounds, pmap, transform, inv_trans, const, fixed = prep

        with np.errstate(all="ignore"):

            fun = make_objective(self, data, inv_trans, const)

            res = optimise_ph(fun, init_t)

        params = inv_trans(const(res.x))

        # Keep this fitter's possibly-custom phi (and its historical
        # serialisation name) rather than assuming log-linear.
        reg_model = Phi()
        reg_model.phi = self.phi
        reg_model.phi_param_map = pmap
        reg_model.name = self.phi_name

        return assemble_regression_model(
            self,
            "Proportional Hazard",
            reg_model,
            data,
            res,
            params,
            bounds,
            pmap,
            fixed,
        )
