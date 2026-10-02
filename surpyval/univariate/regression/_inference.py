"""The parameter covariance and Wald confidence bounds of a fitted
parametric regression model.

``InferenceMixin`` holds them for
:class:`~surpyval.univariate.regression.parametric_regression_model.ParametricRegressionModel`,
which inherits it: ``covariance`` (the inverse of the observed
information, exact where the fit kept it), ``standard_errors``,
``param_cb``, and ``cb``, the bounds on the functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import autograd.numpy as np
import numpy.typing as npt

from surpyval.utils.linalg import (
    cb_link,
    delta_method_se,
    link_band,
    log_transformed_cb,
    numerical_hessian,
    sf_link_from_H,
    wald_bound_on_support,
)
from surpyval.utils.shapes import check_paired_rows, keeps_query_shape
from surpyval.utils.validation import BOUNDS, CB_ON, check_option
from surpyval.utils.warnings import warn_no_covariance

from ._bounds import logit_sf_bound

if TYPE_CHECKING:
    import pandas as pd

    from surpyval.univariate.parametric.parametric_fitter import (
        ParametricFitter,
    )
    from surpyval.utils.deprecation import CallableList
    from surpyval.utils.surpyval_data import SurpyvalData

    from ._covariate_link import CovariateLink
    from .accelerated_life.lifemodel import LifeModel


class InferenceMixin:
    """The covariance and confidence bounds of a
    :class:`ParametricRegressionModel`, which inherits this mixin.

    Separated from ``ParametricRegressionModel`` to keep the inference in
    one place; every method here reads the fitted model through the
    attributes and helpers ``ParametricRegressionModel`` defines.
    """

    if TYPE_CHECKING:
        # Supplied by ParametricRegressionModel, the one class that
        # inherits this mixin. Declared rather than defined so the methods
        # below type check without the mixin pretending to own them.
        params: npt.NDArray
        k_dist: int
        kind: str
        fixed: dict[str, float]
        distribution: ParametricFitter
        reg_model: "CovariateLink | LifeModel"
        model: Any
        data: SurpyvalData
        res: Any
        center: "npt.NDArray | None"
        _fit_centring: "tuple | None"
        _information: "tuple | None"
        _covariance_cache: "tuple | None"
        _restored_covariance: "npt.NDArray | None"
        _restored: bool

        @property
        def parameter_names(self) -> CallableList: ...
        @property
        def aliased(self) -> npt.NDArray: ...
        @property
        def life_parameter(self) -> "str | None": ...
        def _eval_params(self) -> npt.NDArray: ...
        def _held(self) -> set: ...
        def _life_relation(self) -> str: ...

        def _prepare_Z(
            self, Z: "npt.ArrayLike | pd.DataFrame"
        ) -> npt.NDArray: ...

        def _centred(
            self, Z: npt.ArrayLike, center: "npt.NDArray | None" = None
        ) -> Any: ...

        def _warn_if_hazard_negative(
            self,
            x: npt.ArrayLike,
            Z: npt.NDArray,
            valid: Any = True,
            stacklevel: int = 4,
        ) -> None: ...

    # -- confidence bounds -------------------------------------------------

    def _check_inference(self) -> None:
        # A model deserialised with a stored covariance can produce bounds
        # without the original data.
        if self._restored_covariance is not None:
            return
        if self._restored:
            # Restored without a covariance: to_dict stores one only when
            # it was finite at fit time, and the data are not stored.
            raise ValueError(
                "Confidence bounds are unavailable: this model was restored "
                "from a dict that carries no parameter covariance (it could "
                "not be computed when the model was saved), and a restored "
                "model does not keep the data to recompute it."
            )
        if not hasattr(self, "data") or getattr(self, "res", None) is None:
            raise ValueError(
                "Confidence bounds are only available for models fit from "
                "data; from_params models carry no likelihood."
            )

    def covariance(self) -> npt.NDArray:
        """
        Approximate covariance matrix of the fitted parameters, ordered to
        match :attr:`parameter_names`. Computed as the inverse of the Hessian
        of the negative log-likelihood at the MLE (the observed
        information): the exact one the fit computed with autograd to check
        its answer. It falls back to a numerical Hessian where there is
        none at the fitted parameters: an accelerated-life fit (which makes
        no such check), an AFT time-varying fit (whose likelihood autograd
        cannot differentiate), a fit with no finite maximum, one whose
        Hessian there is not positive definite, or a model whose
        parameters or data have changed since the fit. Fixed parameters get
        a zero row/column. It is computed once, and kept while the
        parameters stay as they are.

        A parameter driven to a boundary breaks the Wald approximation; the
        covariance is then returned filled with ``nan`` (with a warning).

        For a fit on centred covariates (#463) the information is that of
        the centred fit, carried to the reported parameters (the baseline
        at ``Z = 0``) by the jacobian of the map between them.
        """
        restored = self._restored_covariance
        if restored is not None:
            return restored
        self._check_inference()
        _, _, cov = self._inference_state()
        if self._fit_centring is not None:
            J = self._fit_centring[2]
            cov = J @ cov @ J.T
        if self.aliased.size:
            # No variance for a coefficient that was not estimated (#476).
            cov = np.array(cov, dtype=float)
            cov[self.k_dist + self.aliased, :] = np.nan
            cov[:, self.k_dist + self.aliased] = np.nan
        return cov

    def _inference_state(
        self,
    ) -> "tuple[npt.NDArray, npt.NDArray | None, npt.NDArray]":
        """``(params, center, covariance)`` of the parameterisation the
        confidence bounds are computed in: that of the centred fit behind
        a model that reports its baseline at 0 (#463), where the
        coefficients and the baseline are not nearly collinear, else the
        model's own."""
        restored = self._restored_covariance
        if restored is not None:
            return (
                np.asarray(self._eval_params(), dtype=float),
                self.center,
                restored,
            )
        if self._fit_centring is not None:
            params, center = self._fit_centring[:2]
        else:
            params, center = self._eval_params(), self.center
        p_hat = np.asarray(params, dtype=float)
        return p_hat, center, self._observed_covariance(p_hat, center)

    def _observed_covariance(
        self, p_hat: npt.NDArray, center: "npt.NDArray | None"
    ) -> npt.NDArray:
        """The inverse of the Hessian of the negative log-likelihood at
        ``p_hat``, the baseline at ``center``: the exact one the fit kept
        (``_information``) if it was computed at that point, else a
        numerical one. A covariance computed is kept, and returned (as a
        copy) while the point is the same."""
        point = self._covariance_point(p_hat, center)
        cached = self._covariance_cache
        if cached is not None and _same_point(cached[0], point):
            return cached[1].copy()
        names = self.parameter_names
        held = self._held()
        free = [i for i, nm in enumerate(names) if nm not in held]
        n = len(names)
        cov = np.zeros((n, n))
        if not free:
            return cov
        step = self._hessian_step(p_hat)[free]
        info = self._information
        if info is not None and _same_point(info[0], point):
            H = info[1]
        else:
            H = self._numerical_information(p_hat, center, free, step)
        bad = not np.all(np.isfinite(H))
        if not bad:
            # Invert in step-scaled coordinates: with a parameter many
            # orders of magnitude from the others the raw information
            # matrix is too ill-conditioned to invert directly.
            try:
                cov_free = np.linalg.inv(H * np.outer(step, step)) * np.outer(
                    step, step
                )
            except np.linalg.LinAlgError:
                bad = True
        if bad:
            warn_no_covariance()
            return np.full((n, n), np.nan)
        cov[np.ix_(free, free)] = cov_free
        self._covariance_cache = (point, cov.copy())
        return cov

    def _covariance_point(
        self, p_hat: npt.ArrayLike, center: "npt.ArrayLike | None"
    ) -> tuple:
        """What the covariance at ``p_hat``, the baseline at ``center``,
        is a function of: those two (copied, and a centre of zeros read as
        none), the data and the names of the fixed parameters. Two points
        are the same by :func:`_same_point`."""
        centre = None
        if center is not None and np.any(center):
            centre = np.array(center, dtype=float)
        return (
            np.array(p_hat, dtype=float),
            centre,
            getattr(self, "data", None),
            tuple(sorted(self.fixed)),
        )

    def _numerical_information(
        self,
        p_hat: npt.NDArray,
        center: "npt.NDArray | None",
        free: list,
        step: npt.NDArray,
    ) -> npt.NDArray:
        """The numerical Hessian of the negative log-likelihood in the free
        parameters at ``p_hat``, the baseline at ``center``."""
        data = self.data
        if center is not None and np.any(center):
            from ._fit_skeleton import centred_copy

            data = centred_copy(data, center)

        def neg_ll_free(free_vals: npt.NDArray) -> float:
            full = p_hat.copy()
            full[free] = free_vals
            return self.model.neg_ll(data, *full)

        return numerical_hessian(neg_ll_free, p_hat[free], step)

    def _parameter_bounds(self) -> list:
        """``(lower, upper)`` for every entry of ``params``: the
        distribution's support bounds, then the life model's parameter
        bounds for an accelerated-life model (the other families'
        coefficients are unbounded)."""
        n_phi = len(self.params) - self.k_dist
        phi_bounds: Any = ((None, None),) * n_phi
        if self.kind == "Accelerated Life":
            declared = getattr(self.reg_model, "phi_bounds", phi_bounds)
            if callable(declared):
                declared = declared(np.asarray(self.data.Z))
            phi_bounds = declared
        return [*self.distribution.bounds, *phi_bounds]

    def _hessian_step(self, p_hat: npt.NDArray) -> npt.NDArray:
        """Finite-difference step for the covariance Hessian, and the
        scale its inversion (numerical or exact) is done in.

        The usual ``eps**(1/3) * max(|p|, 1e-2)``, except that a parameter
        closer to one of its bounds than a few steps gets a step relative
        to that distance. The absolute floor is far larger than, say, an
        accelerated-life coefficient of 5.6e-22 (``InversePower``'s ``a``
        for lives in the thousands), so the difference stepped outside the
        support and the covariance came back nan.
        """
        h = np.finfo(float).eps ** (1.0 / 3.0)
        step = h * np.maximum(np.abs(p_hat), 1e-2)
        for i, (lower, upper) in enumerate(self._parameter_bounds()):
            gaps = [
                p_hat[i] - lower if lower is not None else np.inf,
                upper - p_hat[i] if upper is not None else np.inf,
            ]
            gap = min(gaps)
            if 0 < gap < 10 * step[i]:
                step[i] = h * gap
        return step

    def standard_errors(self) -> npt.NDArray:
        """
        Standard errors of the fitted parameters (square roots of the diagonal
        of :meth:`covariance`), ordered to match :attr:`parameter_names`.
        """
        with np.errstate(invalid="ignore"):
            return np.sqrt(np.diag(self.covariance()))

    def param_cb(
        self,
        name: str,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """
        Confidence bound(s) on a single fitted parameter.

        Wald bounds from the observed information, computed on a scale chosen
        from the parameter's support so the result stays inside it: log for a
        one-sided-bounded distribution parameter (e.g. a positive scale), the
        natural scale for the unbounded covariate coefficients.

        Parameters
        ----------
        name : str
            The parameter to bound; one of :attr:`parameter_names`.
        alpha_ci : float, optional
            Total tail probability of the bound(s). Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds are returned as ``[lower, upper]``.
        """
        self._check_inference()
        names = self.parameter_names
        if name not in names:
            raise ValueError(
                "Unknown parameter {!r}; expected one of {}".format(
                    name, names
                )
            )
        if name == self.life_parameter:
            raise ValueError(
                "{!r} is not a parameter of this accelerated life model: it "
                "is {} at each stress. Bound the life-model parameters "
                "({}) instead, or the predictions with cb().".format(
                    name,
                    self._life_relation(),
                    ", ".join(names[self.k_dist :]),
                )
            )
        idx = names.index(name)
        p_hat = float(self.params[idx])
        var = float(self.covariance()[idx, idx])

        # Distribution parameters carry the distribution's support bounds; the
        # covariate coefficients are unbounded.
        dist_bounds = list(self.distribution.bounds)
        n_phi = len(names) - self.k_dist
        all_bounds = dist_bounds + [(None, None)] * n_phi
        lower, upper = all_bounds[idx]
        return wald_bound_on_support(
            p_hat, var, lower, upper, alpha_ci, bound, name=name
        )

    @keeps_query_shape
    def cb(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | pd.DataFrame",
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        r"""
        Confidence bounds on a predicted function at covariate vector ``Z``.

        The bounds propagate the fitted parameter covariance through the
        requested function by the delta method. ``sf``/``ff``/``Hf`` are
        derived from one bound on the baseline family's probability-plot
        scale, as for the univariate models: ``log H`` for a Weibull,
        Exponential, Rayleigh or Gumbel baseline, the normal quantile of
        ``F`` for a Normal or LogNormal one, the logit of ``F`` for the rest
        (#504; every band was on the logit before v0.22). Each keeps ``sf``
        in ``(0, 1)``, and is formed from the cumulative hazard so the ``Hf``
        bound has no ceiling where ``sf`` underflows. ``hf``/``df`` use a
        log-scale bound (so they stay positive).

        Parameters
        ----------
        x : array like or scalar
            Times at which to evaluate the bound(s).
        Z : array like
            A single covariate vector, used at every ``x`` (one row per
            ``x`` is paired element-wise, as for :meth:`sf`).
        on : {'sf', 'ff', 'Hf', 'hf', 'df'}, optional
            The function to bound. Default ``'sf'``.
        alpha_ci : float, optional
            Total tail probability of the bound(s). Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds put ``[lower, upper]`` on the last axis.

        Returns
        -------
        numpy array
            The confidence bound(s) on ``on`` at each ``x``.
        """
        self._check_inference()
        check_option("on", on, CB_ON)
        check_option("bound", bound, BOUNDS)
        x = np.atleast_1d(np.asarray(x, dtype=float))
        # In the parameterisation of the centred fit when there is one
        # (#463): the bounds are the same function of the data, and there
        # the coefficients are not nearly collinear with the baseline.
        if np.ndim(self._prepare_Z(Z)) == 2:
            # Rows and times paired, as for sf (#488).
            check_paired_rows(
                np.size(x), np.shape(self._prepare_Z(Z))[0], grid=False
            )
        params, center, cov = self._inference_state()
        Zp = self._centred(self._prepare_Z(Z), center)
        if self.kind == "Additive Hazard":
            # Not below the support, where nothing has happened yet.
            self._warn_if_hazard_negative(
                x,
                self._centred(self._prepare_Z(Z)),
                np.asarray(x) >= self.distribution.support[0],
                stacklevel=4,
            )

        if on in ("hf", "df"):
            fn = self.model.hf if on == "hf" else self.model.df
            est = np.asarray(fn(x, Zp, *params), dtype=float)
            se = delta_method_se(lambda p: fn(x, Zp, *p), params, cov)
            return log_transformed_cb(est, se, alpha_ci, bound)

        # Below the support nothing has happened yet: H is 0 there, as
        # for sf (the bound is then the estimate, sf = 1). An additive
        # hazard's beta'Z x is not 0 at a negative x, and gave a band
        # around a survival of about 0.9 where sf is 1.
        lower = self.distribution.support[0]
        below = np.asarray(x) < lower
        x_in = np.where(below, lower + 1.0 if np.isfinite(lower) else 0.0, x)

        def H_of(p: npt.NDArray) -> npt.NDArray:
            H = np.asarray(self.model.Hf(x_in, Zp, *p), dtype=float)
            return np.where(below, 0.0, H)

        def sf_of(p: npt.NDArray) -> npt.NDArray:
            S = np.asarray(self.model.sf(x_in, Zp, *p), dtype=float)
            return np.where(below, 1.0, S)

        return self._sf_bounds(
            H_of,
            sf_of,
            params,
            cov,
            np.shape(x),
            on,
            alpha_ci,
            bound,
        )

    @property
    def _cb_link(self) -> str:
        """The scale of the Wald bands on ``sf``/``ff``/``Hf``: the baseline
        family's probability-plot scale, as for the univariate models
        (#477, #504)."""
        return cb_link(self.distribution)

    def _sf_bounds(
        self,
        H_of: Any,
        sf_of: Any,
        params: npt.NDArray,
        cov: npt.NDArray,
        shape: tuple,
        on: str,
        alpha_ci: float,
        bound: str,
    ) -> npt.NDArray:
        """The bounds of :meth:`cb` and :meth:`cb_tvc` on ``sf``, ``ff`` or
        ``Hf``: a Wald bound on the baseline family's scale (``log H``,
        the normal quantile of ``F`` or the logit of ``F``, see
        :attr:`_cb_link`), formed from the cumulative hazard ``H_of(p)``
        and propagated by the delta method, carried to the scale of
        ``on``. ``sf_of(p)`` gives the survival where ``H`` is negative (an
        additive hazard's ``sf`` above 1)."""
        link = self._cb_link
        name = {"R": "sf", "F": "ff"}.get(on, on)

        # Formed from the cumulative hazard, so the Hf bound has no
        # ceiling where sf underflows (#418).
        def u_of(p: npt.NDArray) -> npt.NDArray:
            return sf_link_from_H(H_of(p), link)

        H_hat = np.asarray(H_of(params), dtype=float)
        H_hat = np.broadcast_to(H_hat, np.broadcast_shapes(H_hat.shape, shape))
        u_hat = np.broadcast_to(u_of(params), H_hat.shape)
        with np.errstate(invalid="ignore"):
            # inf - inf where sf is 1 or 0: the bounds are the estimate
            # there (link_band)
            se_u = delta_method_se(u_of, params, cov)
        cb = link_band(u_hat, se_u, alpha_ci, bound, link, name)
        # A negative H (an additive hazards sf above 1, documented) has no
        # point on these scales: it keeps the clipped-sf logit bound.
        negative = H_hat < 0
        if not negative.any():
            return cb
        sf_hat = np.asarray(sf_of(params), dtype=float)
        se = delta_method_se(sf_of, params, cov)

        def end(sign: float, tail: float) -> npt.NDArray:
            # One end on the scale of ``on``; sign +1 is sf's upper.
            sf_c = logit_sf_bound(sf_hat, se, sign, tail)
            if name == "sf":
                return sf_c
            with np.errstate(divide="ignore"):
                return 1.0 - sf_c if name == "ff" else -np.log(sf_c)

        # ff and Hf decrease in sf: their lower end is sf's upper.
        flip = -1.0 if name == "sf" else 1.0
        if bound == "two-sided":
            fallback = np.stack(
                [end(flip, alpha_ci / 2.0), end(-flip, alpha_ci / 2.0)],
                axis=-1,
            )
            negative = negative[..., None]
        else:
            sign = flip if bound == "lower" else -flip
            fallback = end(sign, alpha_ci)
        return np.where(negative, fallback, cb)


def _same_point(a: tuple, b: tuple) -> bool:
    """Whether ``a`` and ``b``, as ``_covariance_point`` gives them, are
    the same point: equal parameters and centres, the same data object and
    the same fixed parameters."""
    params_a, centre_a, data_a, fixed_a = a
    params_b, centre_b, data_b, fixed_b = b
    if data_a is not data_b or fixed_a != fixed_b:
        return False
    if not np.array_equal(params_a, params_b):
        return False
    if centre_a is None or centre_b is None:
        return centre_a is None and centre_b is None
    return bool(np.array_equal(centre_a, centre_b))
