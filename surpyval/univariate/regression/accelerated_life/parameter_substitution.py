from __future__ import annotations

import copy
import warnings
from typing import Any, Callable

import autograd.numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult, minimize

from surpyval.univariate.parametric.fitters import (
    EachParameter,
    bounds_convert,
    identity,
    verify_or_polish,
)
from surpyval.univariate.parametric.fitters.runaway import (
    search_derivatives,
)
from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    OptimisedFitMixin,
)
from surpyval.univariate.regression._aliasing import dataframe_covariates
from surpyval.utils import _caller_stacklevel
from surpyval.utils.covariates import (
    coefficient_floor,
    coefficient_names,
)
from surpyval.utils.rng import as_generator
from surpyval.utils.surpyval_data import SurpyvalData

from .._aliasing import (
    aliased_columns,
    constant_columns,
    fit_columns,
    warn_aliased,
)
from .._fit_skeleton import (
    FixedWithAliased,
    HazardIdentitiesMixin,
    MirroredDistributionAttrs,
    assemble_regression_model,
    check_baseline_support,
    check_fixed_and_init,
    covariate_center,
    drop_nonfinite_covariates,
    finish_search,
    finite_start,
    free_baseline,
    free_coefficients,
    keep_information,
    make_objective,
    mirror_distribution,
    one_sided_positions,
    require_finite_fit,
    uniform_draws,
)
from .._likelihood import regression_neg_ll
from ..parametric_regression_model import ParametricRegressionModel
from ..regression_data import DataFrameRegressionMixin, truncation_window
from .lifemodel import LifeModel


def _search(
    fun: Callable[[npt.NDArray], Any],
    x0: npt.NDArray,
    n_obs: float,
    floor: "float | npt.ArrayLike" = 1.0,
) -> tuple[OptimizeResult, bool]:
    """Minimise ``fun`` from ``x0`` with Nelder-Mead then TNC, as the fit
    always searched, and whether the answer is verifiably a minimum (see
    ``verify_or_polish``, which polishes one that is not, each component
    in units of at least ``floor``)."""
    res1 = minimize(fun, x0, method="Nelder-Mead", options={"maxiter": 1000})
    res2 = minimize(fun, res1.x, method="TNC")
    res, verified = verify_or_polish(
        fun, res2 if res2.success else res1, n_obs, floor=floor
    )
    if verified:
        res = _newton_finish(fun, res)
    return res, verified


def _newton_finish(fun: Callable[[npt.NDArray], Any], res: Any) -> Any:
    """``res``, a verified minimum of ``fun``, taken the rest of the way
    by Newton's method (up to three steps, each kept only where it does
    not raise ``fun``). The searches' tolerances stop them anywhere in a
    neighbourhood of the minimum that is wide along a flat direction (a
    constant factor ``c`` against the coefficients of stresses far from
    0), so the answer depended on the path: 1e-4 apart in the parameters
    for one fit with and without an aliased column, once ``c`` was
    searched on its log scale (#634). From inside the neighbourhood
    Newton's method converges to the minimum itself."""
    x = np.asarray(res.x, dtype=float)
    f = float(res.fun)
    for _ in range(3):
        try:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Output seems independent")
                H, g = search_derivatives(fun, x) or (None, None)
                if H is None or not (
                    np.all(np.isfinite(H)) and np.all(np.isfinite(g))
                ):
                    break
                step = np.linalg.solve(H, g)
                trial = x - step
                f_trial = float(fun(trial))
        except (np.linalg.LinAlgError, ValueError, ArithmeticError):
            break
        if not (np.all(np.isfinite(trial)) and f_trial <= f):
            break
        x, f = trial, f_trial
    res.x, res.fun = x, f
    return res


class _LifeOfLogScale:
    """A life model's ``phi`` for the fit's objective, taking the
    parameters it searches on the log scale (``log_scale_parameters``) as
    their logs, the search's own values: the life is then one exponent of
    them (``LifeModel.log_life``), which neither underflows where ``c``
    alone would (1e-322, where the search could go no further and the fit
    could only say "unverified") nor overflows where ``e^(a / U)`` alone
    would (#634). A class, not a closure, so the model that keeps the
    objective pickles (#573)."""

    def __init__(self, life_model: LifeModel) -> None:
        self.life_model = life_model

    def __call__(self, Z: Any, *params: Any) -> Any:
        return np.exp(self.life_model.log_life(Z, *params))


def _coefficient_units(
    fitter: Any,
    fixed: dict,
    phi_param_map: dict,
    Z: "npt.ArrayLike | None",
) -> npt.NDArray:
    """The search's ``floor``: each free life-model parameter that is a
    column's coefficient (``LifeModel.coefficient_columns``) in its
    covariate's units, as the other regressions search theirs (#577,
    #612); 1 for every other component."""
    columns = fitter.life_model.coefficient_columns()
    names = [
        *fitter.param_map,
        *sorted(phi_param_map, key=phi_param_map.__getitem__),
    ]
    free = [name for name in names if name not in fixed]
    coefs = [(k, columns[nm]) for k, nm in enumerate(free) if nm in columns]
    return coefficient_floor(len(free), coefs, Z)


class ParameterSubstitutionFitter(
    MirroredDistributionAttrs, HazardIdentitiesMixin, DataFrameRegressionMixin
):
    """
    Accelerated life fitter: the life parameter of a distribution is
    replaced by a function of the stress, :math:`L(Z)`, given by a life
    model (``Power``, ``Eyring``, ...), while the other distribution
    parameters are shared by every stress level.

    Which parameter carries the life, and how, depends on the
    distribution: :math:`\\alpha = L(Z)` for Weibull, :math:`\\mu = L(Z)`
    for Normal, Gumbel and Logistic, :math:`\\mu = \\ln L(Z)` for LogNormal,
    and the rate is :math:`1 / L(Z)` for Exponential (``failure_rate``)
    and Gamma (``beta``). Create one with
    ``AcceleratedLife(distribution, life_model)`` rather than directly.
    """

    #: The ``repr`` (#614)
    fitter_kind = "accelerated life fitter"
    name_suffix = "AL"

    def _repr_details(self) -> "list[str]":
        return [*super()._repr_details(), self.life_model.name + " life model"]

    def __init__(
        self,
        kind: str,
        name: str,
        distribution: OptimisedFitMixin,
        life_model: LifeModel,
        life_parameter: str,
        baseline: list[str] | str | None = None,
        param_transform: Callable[[Boxable], Boxable] | None = None,
        inverse_param_transform: Callable[[Boxable], Boxable] | None = None,
        life_relation: str = "L(Z)",
    ) -> None:
        if baseline is None:
            baseline = []
        elif not isinstance(baseline, list):
            # Baseline used if using a function that deviates from some number,
            # e.g. np.exp(np.dot(Z, beta))
            baseline = [baseline]

        self.name = name
        self.kind = kind
        mirror_distribution(self, distribution)
        self.life_model = life_model
        self.phi = life_model.phi
        self.Hf_dist = self.dist.Hf
        self.hf_dist = self.dist.hf
        self.sf_dist = self.dist.sf
        self.ff_dist = self.dist.ff
        self.df_dist = self.dist.df
        self.baseline = baseline
        self.life_parameter = life_parameter
        self.life_relation = life_relation
        self.fixed = {life_parameter: 1.0}

        self.param_transform: Callable[..., Any]
        self.inverse_param_transform: Callable[..., Any]
        if param_transform is None:
            # (Module-level, not lambdas, so a fitted model pickles, #573)
            self.param_transform = identity
            self.inverse_param_transform = identity
        else:
            # Supplied as a pair -- accelerated_life.py passes both or
            # neither -- so the inverse is not None here.
            assert inverse_param_transform is not None
            self.param_transform = param_transform
            self.inverse_param_transform = inverse_param_transform

    def _with_life_model(
        self, life_model: LifeModel
    ) -> "ParameterSubstitutionFitter":
        """This fitter with ``life_model`` in place of its own (the
        resolved form of a life model with a parameter per stress
        column, see ``LifeModel.resolve``)."""
        out = copy.copy(self)
        out.life_model = life_model
        out.phi = life_model.phi
        return out

    def _stress_matrix(self, Z: Numeric) -> npt.NDArray:
        """``Z`` as a 2-D array with one row per stress vector.

        A scalar or 0-d stress is one stress (``AxisError`` from
        ``np.unique`` before); a 1-D array is one stress per entry for a
        single-stress life model (#261) but, for a two-stress model, a
        single row ``[T, V]`` of the right length -- which ``phi`` already
        accepted while ``sf``/``hf``/``cb``/``random`` raised
        ``IndexError``.
        """
        Z_arr = np.asarray(Z, dtype=float)
        n_stresses = getattr(self.life_model, "n_stresses", 1)
        if Z_arr.ndim == 0:
            return Z_arr.reshape(1, 1)
        if Z_arr.ndim == 1:
            if (
                n_stresses is not None
                and n_stresses > 1
                and Z_arr.shape[0] == n_stresses
            ):
                return Z_arr.reshape(1, -1)
            return Z_arr.reshape(-1, 1)
        return Z_arr

    def Hf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        return self._at_rows(self.Hf_dist, x, Z, params)

    def _at_rows(
        self, f: Callable[..., Boxable], x: Numeric, Z: Numeric, params: tuple
    ) -> Boxable:
        """The distribution's function ``f`` at ``x``, with the life
        parameter of each row of ``Z`` (``_dist_params_by_row``).

        Evaluated once over every row. It used to be evaluated over every
        row once per distinct stress and the rows at that stress kept, so
        a fit to a continuous stress cost the square of the rows: a
        ``GeneralLogLinear`` fit to 120 rows of distinct stresses made 157
        substitutions per likelihood evaluation and took 9 s (#592)."""
        x = np.array(x)
        Z_arr = self._stress_matrix(Z)
        if Z_arr.shape[0] == 0:
            return np.zeros_like(x)
        values = f(x, *self._dist_params_by_row(Z_arr, params))
        return self._nan_at_unknown_stress(values, Z_arr)

    def _dist_params_by_row(
        self, Z_arr: npt.NDArray, params: tuple
    ) -> list[Boxable]:
        """The distribution's parameters at every row of ``Z_arr`` at
        once: those in ``params``, with the life parameter's slot holding
        the (transformed) life at each row, an array that broadcasts
        against ``x`` row for row (a scalar for a single row, as
        ``_dist_params_at`` gives it).

        A life model whose ``phi`` takes the rows of a stress matrix
        (``LifeModel.phi_takes_rows``, the built-in ones) gives every
        row's life in one call. Another one -- a custom life model written
        for a single stress row -- is called once per distinct row, as it
        always was, and the lives taken to the rows. A row with a missing
        stress is given another row's, and ``_nan_at_unknown_stress`` makes
        its value nan: a nan life would poison the gradient of every
        parameter through the sum over rows.

        A list, not ``np.where`` over the slots (see ``_dist_params_at``,
        #555)."""
        known = np.isfinite(Z_arr).all(axis=1)
        if not known.all():
            stand_in = (
                Z_arr[known][0] if known.any() else np.ones(Z_arr.shape[1])
            )
            Z_arr = np.where(known[:, None], Z_arr, stand_in)
        phi_params = params[self.k_dist :]
        if getattr(self.life_model, "phi_takes_rows", False):
            life = np.reshape(self.phi(Z_arr, *phi_params), (-1,))
        else:
            stresses, inverse = np.unique(Z_arr, axis=0, return_inverse=True)
            lives = np.array(
                [np.reshape(self.phi(s, *phi_params), ()) for s in stresses]
            )
            life = lives[np.reshape(inverse, (-1,))]
        life = self.param_transform(life)
        if Z_arr.shape[0] == 1:
            life = np.reshape(life, ())
        life_idx = self.param_map[self.life_parameter]
        return [
            life if k == life_idx else params[k] for k in range(self.k_dist)
        ]

    def _dist_params_at(
        self, stress: npt.NDArray, params: tuple
    ) -> list[Boxable]:
        """The distribution's parameters at the stress row ``stress``:
        those in ``params``, with the life parameter's slot replaced by
        the (transformed) life the life model gives there.

        Built as a list, not by ``np.where`` over the slots: autograd's
        ``where`` does not reduce its gradient to the shape of a broadcast
        argument (the life, one value against a row of them), and its
        second derivative through one is wrong -- a LogNormal ``Power``
        model's exact information was 4e-5 off in ``n``, its ``Hf``'s
        second derivative 13% off (#555)."""
        life = self.param_transform(self.phi(stress, *params[self.k_dist :]))
        life_idx = self.param_map[self.life_parameter]
        return [
            np.reshape(life, ()) if k == life_idx else params[k]
            for k in range(self.k_dist)
        ]

    @staticmethod
    def _nan_at_unknown_stress(values: Boxable, Z_arr: npt.NDArray) -> Boxable:
        # A row with a missing stress matches no stress level, so it kept
        # the initial 0 (a survival of 1); its prediction is unknown.
        known = np.isfinite(Z_arr).all(axis=1)
        if known.all():
            return values
        return np.where(known, values, np.nan)

    def hf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        return self._at_rows(self.hf_dist, x, Z, params)

    # sf/ff/df and the log identities come from HazardIdentitiesMixin;
    # Hf and hf above already do the scalar/1-D stress coercion (#261),
    # so the identities need no preamble of their own.

    def mpp_inv_y_transform(self, y: Numeric, *params: Boxable) -> Numeric:
        return y

    def mpp_y_transform(self, y: Numeric, *params: Boxable) -> Numeric:
        return y

    def random(
        self,
        size: int,
        Z: Numeric,
        *params: Boxable,
        random_state: Any = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Draw ``size`` samples at each distinct stress in ``Z``.

        ``Z`` is a scalar stress, a 1-D array of stresses (one stress
        variable), or one row per stress for a multi-stress life model.
        Returns the draws and the stress row each was drawn at.
        ``random_state`` seeds the draw: ``None`` (the default) draws from
        numpy's global generator, so ``np.random.seed`` reproduces it; an
        int or a ``numpy.random.Generator`` gives a stream of its own.
        """
        x = []
        Z_out = []
        # A scalar or 1-D stress is one stress variable: make it a column,
        # as ``Hf``/``hf`` do. (A former ``(low, high)`` tuple option that
        # drew the stresses uniformly was unreachable through the fitted
        # model, whose ``random`` converts ``Z`` to an array first, and
        # returned ``size`` draws per random stress -- ``size**2`` in all.)
        Z_arr = self._stress_matrix(Z)
        # One stream for every stress (an int seed would otherwise restart
        # and give every stress the same uniforms).
        rng = None if random_state is None else as_generator(random_state)

        for stress in np.unique(Z_arr, axis=0):
            dist_params_i = self._dist_params_at(stress, params)
            U = uniform_draws(size, rng)
            x.append(self.dist.qf(U, *dist_params_i))
            if np.isscalar(stress):
                cols = 1
            else:
                cols = len(stress)
            Z_out.append(np.ones((size, cols)) * stress)
        return np.array(x).flatten(), np.concatenate(Z_out)

    def neg_ll(self, data: SurpyvalData, *params: Boxable) -> Boxable:
        return regression_neg_ll(self, data, *params)

    def _check_stresses(self, Z_arr: npt.NDArray) -> None:
        """Refuse stresses the life model is not defined at.

        ``Power`` (``a Z**n``), ``InversePower``, ``DualPower`` and the
        power column of ``PowerExponential`` need strictly positive
        stresses. A non-positive one used to reach the log-linear starting
        fit and fail there with an SVD ``LinAlgError`` and LAPACK messages
        on stderr. The Arrhenius-type models (``Exponential``,
        ``InverseExponential``, the Eyring models, the temperature column
        of ``DualExponential`` and ``PowerExponential``) read a column as
        an absolute temperature: a value <= 0 there is refused naming
        kelvin, and a column below 200 K throughout warns, as a
        temperature typed in degrees Celsius (#654).
        """
        from .lifemodel import KELVIN_WARNING_BELOW

        life_model = self.life_model
        cols = getattr(life_model, "positive_stress_columns", ())
        kelvin = getattr(life_model, "kelvin_stress_columns", ())
        n_stresses = getattr(life_model, "n_stresses", None)
        if n_stresses is not None and Z_arr.shape[1] != n_stresses:
            raise ValueError(
                "The {} life model takes {} stress column(s); Z has "
                "{}.".format(life_model.name, n_stresses, Z_arr.shape[1])
            )
        for col in sorted({*cols, *kelvin}):
            values = np.asarray(Z_arr[:, col], dtype=float)
            if not values.size:
                continue
            lowest = float(np.min(values))
            if col in kelvin and lowest <= 0:
                raise ValueError(
                    "The {} life model reads column {} of Z as an absolute "
                    "temperature, in kelvin, which must be positive; its "
                    "lowest value is {:g}. Kelvin is degrees Celsius plus "
                    "273.15: for a Z in degrees Celsius pass Z + "
                    "273.15.".format(life_model.name, col, lowest)
                )
            if lowest <= 0:
                raise ValueError(
                    "The {} life model needs strictly positive stresses "
                    "(the lowest in column {} of Z is {:g}): it raises the "
                    "stress to a "
                    "power or takes its logarithm. Shift or rescale the "
                    "stress, or use a life model defined there (e.g. "
                    "life_models.Linear or life_models.GeneralLogLinear)."
                    "".format(life_model.name, col, lowest)
                )
            highest = float(np.max(values))
            if (
                col in kelvin
                and getattr(life_model, "warns_below_kelvin", True)
                and highest < KELVIN_WARNING_BELOW
            ):
                warnings.warn(
                    "Every stress in column {} of Z is below {:g} K (the "
                    "highest is {:g}): the {} life model reads it as an "
                    "absolute temperature, in kelvin. Did you pass degrees "
                    "Celsius? Add 273.15.".format(
                        col,
                        KELVIN_WARNING_BELOW,
                        highest,
                        life_model.name,
                    ),
                    UserWarning,
                    stacklevel=_caller_stacklevel(),
                )

    def _aliased_stresses(
        self, Z: npt.NDArray, n: npt.NDArray, fixed: dict
    ) -> tuple[str, ...]:
        """The life-model parameters the data cannot determine (#503),
        with one warning naming their stress columns.

        A life model whose log-life is linear in terms of the stresses
        (``LifeModel._stress_terms``: ``log s`` for a power term, ``1 / s``
        for an exponential one) determines a term's parameter only where
        the term is not constant (the constant factor, ``c``, absorbs a
        constant one) or a linear combination of the others: with equal
        stress columns ``DualPower``'s ``c s1^m s2^n`` is ``c s^(m + n)``,
        and only ``m + n`` is determined. The check is that of the
        regressions (:mod:`.._aliasing`) on the centred terms, the later
        of two collinear columns aliased. A parameter the caller fixed is
        an offset, left out.
        """
        found = self.life_model._stress_terms(Z)
        if found is None:
            return ()
        terms, names, intercept = found
        terms = np.asarray(terms, dtype=float)
        free = np.array([k for k, nm in enumerate(names) if nm not in fixed])
        if free.size == 0 or not np.all(np.isfinite(terms)):
            return ()
        T = terms[:, free]
        if intercept:
            T = T - covariate_center(T, n)
            constant = constant_columns(terms[:, free])
        else:
            constant = np.all(T == 0, axis=0)
        gram = T.T @ (n[:, None] * T)
        aliased = free[aliased_columns(gram, T.shape[0], constant)]
        if aliased.size == 0:
            return ()
        if intercept:
            how = (
                "the {} life model's log-life is linear in a term of each "
                "stress (log s for a power term, 1 / s for an exponential "
                "one), and their terms are constant (the constant factor "
                "absorbs them) or a linear combination of the other "
                "columns' terms"
            )
        else:
            how = (
                "the {} life model's log-life is linear in the stresses, "
                "and these are all zero or a linear combination of the "
                "other columns"
            )
        warn_aliased(aliased, how.format(self.life_model.name))
        return tuple(names[k] for k in aliased.tolist())

    def _one_level_message(self, Z: Any, free_phi: list) -> str:
        level = np.unique(Z, axis=0)[0]
        level_text = (
            str(float(level[0])) if level.size == 1 else str(level.tolist())
        )
        out = (
            "An accelerated life model needs at least two distinct stress "
            "levels to fit how life changes with stress; Z has one ({}), "
            "which fixes only the life at that stress. ".format(level_text)
        )
        if len(free_phi) > 1:
            return out + (
                "Test at more stress levels, fit the distribution alone, or "
                "fix all but one of the {} life model's parameters ({}) with "
                "`fixed` and give a start with `init`.".format(
                    self.life_model.name, ", ".join(free_phi)
                )
            )
        return out + (
            "With the other life-model parameters fixed, give a start with "
            "`init` (the distribution's parameters, then the life model's)."
        )

    @dataframe_covariates
    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        t: npt.ArrayLike | None = None,
        init: npt.ArrayLike | None = None,
        fixed: dict[str, float] | None = None,
        tl: npt.ArrayLike | None = None,
        tr: npt.ArrayLike | None = None,
    ) -> ParametricRegressionModel:
        """
        Fit the accelerated life model by maximum likelihood.

        Parameters
        ----------

        x : array_like
            The observed event times.
        Z : array_like
            The stress of each observation: a 1-D array for a one-stress
            life model, or one column per stress (two for ``DualPower``,
            ``DualExponential``, ``PowerExponential``; any number for
            ``GeneralLogLinear``, one coefficient each). Designed for a few
            controlled stress levels: without ``init`` the starting point
            comes from fitting the distribution at each distinct stress
            level, so at least two levels are needed. ``Power``,
            ``InversePower``, ``DualPower`` and the second (power) stress
            of ``PowerExponential`` need strictly positive stresses; the
            Arrhenius-type models (``Exponential``, ``InverseExponential``,
            the Eyring models, the first stress of ``DualExponential`` and
            ``PowerExponential``) an absolute temperature, in kelvin (a
            value <= 0 is refused, and a stress below 200 K throughout
            warns, as degrees Celsius would be). Rows with a missing or
            infinite stress are dropped, with a warning.
        c : array_like, optional
            The censoring indicators (0 observed, 1 right, -1 left, 2
            interval). Defaults to all observed.
        n : array_like, optional
            The count of observations at each time. Defaults to 1.
        t : array_like, optional
            Truncation bounds: an (N, 2) array of the left and right
            truncation times of each observation.
        tl, tr : array_like or float, optional
            The left / right truncation times of each observation (or one
            for every observation), the columns of ``t``, which they
            replace (#662).
        init : array_like, optional
            Initial parameter values: the distribution parameters (with any
            value in the life parameter's slot) followed by the life-model
            parameters. Where the fit from them does not reach a verified
            maximum, the default start is tried too and the better kept.
        fixed : dict, optional
            Parameters to hold fixed, by name (a distribution parameter or
            a life-model parameter such as ``"n"``).

        Returns
        -------

        ParametricRegressionModel
            The fitted model. The life parameter's slot in ``params`` and
            ``dist_params`` holds a placeholder value of 1 (it is replaced
            by the life model at each stress); the life-model parameters
            are in ``phi_params``.

        Examples
        --------

        >>> import numpy as np
        >>> from surpyval import Weibull, AcceleratedLife
        >>> from surpyval.life_models import Power
        >>> np.random.seed(1)
        >>> stress = np.repeat([20.0, 30.0, 40.0], 40)
        >>> x = Weibull.random(120, 10, 3) * (100.0 / stress)
        >>> model = AcceleratedLife(Weibull, Power).fit(x, Z=stress)
        >>> model.params.round(3)
        array([  1.   ,   2.831, 558.686,  -0.828])
        """
        t = truncation_window(x, t, tl, tr)
        # ``x`` goes through the data handler before anything reads it as
        # an array: the documented ragged form ``[10, [11, 13], ...]`` is
        # not a rectangular array, and ``np.asarray(x)`` on it raised a raw
        # numpy error.
        data = SurpyvalData(x=x, c=c, n=n, t=t, group_and_sort=False)
        # A 1-D stress vector (one stress variable) becomes a single column
        # so the per-stress masking in the initialiser works (#261). As
        # floats, so a ``None`` is a missing value, dropped with the rest.
        Z_arr = np.asarray(Z, dtype=float)
        if Z_arr.ndim == 1:
            Z_arr = Z_arr.reshape(-1, 1)
        if Z_arr.ndim == 2:
            # A life model with a parameter per stress column
            # (GeneralLogLinear) is fitted, and carried by the model, in
            # its form for Z's columns.
            life_model = self.life_model.resolve(Z_arr.shape[1])
            # Its column coefficients named by the columns, or coef_j
            # (#614), unique among the other parameters' names
            columns = life_model.coefficient_columns()
            if columns:
                others = [
                    *self.param_map,
                    *(k for k in life_model.phi_param_map if k not in columns),
                ]
                life_model = life_model.named(
                    coefficient_names(len(columns), fit_columns(), others)
                )
            if life_model is not self.life_model:
                return self._with_life_model(life_model).fit(
                    x, Z_arr, c=c, n=n, t=t, init=init, fixed=fixed
                )
        data, Z_arr = drop_nonfinite_covariates(data, Z_arr)
        self._check_stresses(Z_arr)
        data.add_covariates(Z_arr)
        check_baseline_support(self, data)
        # The per-stress fallback start uses each row's time (the midpoint
        # of an interval row).
        x_arr: npt.NDArray = (
            data.x if data.x.ndim == 1 else data.x.mean(axis=1)
        )
        life_parameter_idx = self.param_map[self.life_parameter]
        if fixed is None:
            fixed = {}

        def default_init() -> npt.NDArray:
            # The distribution fitted at each distinct stress, with the life
            # model fitted through the per-stress life parameters.
            stress_data = []
            params_at_Z = []

            # How do I make this work when there is only one failure per
            # stress?
            base_line_dist_init = self.dist.fit_from_surpyval_data(data).params

            for s in np.unique(data.Z, axis=0):
                mask = (data.Z == s).all(axis=1)
                with warnings.catch_warnings():
                    warnings.filterwarnings("error")
                    try:
                        params_at_s = self.dist.fit_from_surpyval_data(
                            data[mask]
                        ).params
                        params_at_Z.append(params_at_s)
                    except Exception:
                        # The mean time at the level is a life: its life
                        # parameter is that life on the parameter's scale
                        # (a LogNormal's mu its log, a Gamma's beta its
                        # reciprocal). Taken as the parameter itself, a
                        # LogNormal on a continuous stress, with one row a
                        # level, had lives of exp(time) and a log-likelihood
                        # not finite at the start (#621).
                        params_at_s = np.copy(base_line_dist_init)
                        params_at_s[life_parameter_idx] = self.param_transform(
                            x_arr[mask].mean()
                        )
                        params_at_Z.append(params_at_s)
                    finally:
                        stress_data.append(s)

            params_at_Z = np.array(params_at_Z)
            dist_init = params_at_Z.mean(axis=0)

            stress_data = np.array(stress_data)

            if len(params_at_Z) < 2:
                # One level identifies the life at that stress, not how it
                # changes with stress; no ``init`` changes that (#489).
                raise ValueError(self._one_level_message(data.Z, free_phi))

            parameter_data = params_at_Z[:, life_parameter_idx]

            parameter_data = self.inverse_param_transform(parameter_data)

            # Every life model's phi_init is (life, Z). There used to be
            # a branch here for a "(Z)"-only signature, chosen by
            # comparing str(inspect.signature(...)) == "(Z)", and another
            # for a non-callable phi_init. Neither could run: all ten
            # life models are callable with the two-argument signature.
            phi_init = self.life_model.phi_init(parameter_data, stress_data)
            return np.array([*dist_init, *phi_init])

        if callable(self.life_model.phi_param_map):
            phi_param_map = self.life_model.phi_param_map(data.Z)
        else:
            phi_param_map = self.life_model.phi_param_map
        # A stress effect the data cannot determine is aliased (#503): held
        # at 0 in the fit, reported as nan.
        aliased = self._aliased_stresses(
            Z_arr, np.asarray(data.n, dtype=float), fixed
        )
        fixed = {**fixed, **dict.fromkeys(aliased, 0.0)}
        # The life-model parameters the fit estimates, and the distinct
        # stress levels that identify them: each level pins down one life.
        free_phi = [k for k in phi_param_map if k not in fixed]
        n_levels = len(np.unique(data.Z, axis=0))

        # Keep the merged map local: assigning it to ``self.param_map``
        # mutated the fitter, so a second ``fit()`` re-merged on top of the
        # already-merged map and produced out-of-range indices (#261).
        param_map = {
            **self.param_map,
            **{k: v + len(self.param_map) for k, v in phi_param_map.items()},
        }
        check_fixed_and_init(fixed, init, param_map, self.fixed)

        user_init = init is not None and len(np.atleast_1d(init)) > 0
        init = np.array(init) if user_init else default_init()

        if self.baseline != []:
            baseline_model = self.dist.fit_from_surpyval_data(data)
            baseline_fixed = {
                k: baseline_model.params[baseline_model.param_map[k]]
                for k in self.baseline
            }
            fixed = {**baseline_fixed, **fixed}

        if self.fixed != {}:
            fixed = {**self.fixed, **fixed}

        # Dynamic or static bounds determination
        if callable(self.life_model.phi_bounds):
            bounds = (*self.bounds, *self.life_model.phi_bounds(data.Z))
        else:
            bounds = (*self.bounds, *self.life_model.phi_bounds)

        # A factor that multiplies the life (``c``) is searched on the log
        # scale over its whole range: linearly beyond 1, a fit stopped short
        # of its maximum at c = 1e22 (#634)
        log_scale = [
            len(self.param_map) + phi_param_map[name]
            for name in getattr(self.life_model, "log_scale_parameters", ())
            if name in phi_param_map
        ]
        units = [np.inf if i in log_scale else 1.0 for i in range(len(bounds))]
        transform, inv_trans, const, fixed_idx, not_fixed = bounds_convert(
            data.x, bounds, fixed, param_map, units
        )

        init = transform(init)[not_fixed]

        with np.errstate(all="ignore"):

            if log_scale:
                # The objective's life from the logs the search runs on
                # (``_LifeOfLogScale``); the model is built with the life
                # model's own ``phi`` and the natural parameters.
                searcher = copy.copy(self)
                searcher.phi = _LifeOfLogScale(self.life_model)
                on_log_scale = EachParameter(
                    [
                        identity if i in log_scale else f
                        for i, f in enumerate(inv_trans.funcs)
                    ]
                )
                fun = make_objective(searcher, data, on_log_scale, const)
            else:
                fun = make_objective(self, data, inv_trans, const)
            init = finite_start(
                fun,
                init,
                (
                    (lambda: transform(default_init())[not_fixed])
                    if user_init
                    else None
                ),
            )

            n_obs = float(np.sum(data.n))
            floor = _coefficient_units(self, fixed, phi_param_map, data.Z)
            res, verified = _search(fun, init, n_obs, floor)
            start = init
            # From a start far from the maximum the search can stop short
            # of it, silently: InversePower started with its first
            # parameter x1e6 ended 14.7 below the maximum (#428). The
            # default start is then tried too, and the better kept.
            if user_init and not verified:
                try:
                    default = finite_start(
                        fun, transform(default_init())[not_fixed], None
                    )
                except ValueError:
                    # No default start (a single stress level, say)
                    default = None
                if default is not None:
                    alt, alt_verified = _search(fun, default, n_obs, floor)
                    if alt.fun < res.fun or not np.isfinite(res.fun):
                        res, verified = alt, alt_verified
                        start = default

        require_finite_fit(float(res.fun))
        identifiable = n_levels >= len(free_phi)
        if not identifiable:
            # Fewer levels than free life-model parameters: the likelihood
            # is flat along a ridge of them, wherever the search stopped.
            warnings.warn(
                "The life-stress relationship is not identifiable: {} "
                "distinct stress level(s) for the {} free parameters of the "
                "{} life model ({}). The likelihood is flat along a ridge "
                "of them, so their values, standard errors and bounds are "
                "meaningless (the predictions at the observed stresses are "
                "not); test at more stress levels, or fix all but {} of "
                "them with `fixed`.".format(
                    n_levels,
                    len(free_phi),
                    self.life_model.name,
                    ", ".join(free_phi),
                    n_levels,
                ),
                stacklevel=_caller_stacklevel(),
            )
        # Store the full merged fixed dict (baseline-derived + fitter-level
        # + user-supplied), not just the fitter's own -- otherwise standard
        # errors are reported for parameters that were held fixed (#261).
        # It holds the life-parameter placeholder (its value is replaced by
        # the life model, so it is not a parameter at all), any baseline
        # parameters, the user's fixed values and the aliased ones, which
        # are reported as nan, R's NA (#503), and predict with 0.
        held = FixedWithAliased(fixed)
        held.aliased = tuple(aliased)
        model = assemble_regression_model(
            self,
            self.kind,
            self.life_model,
            data,
            res,
            inv_trans(const(res.x)),
            bounds,
            phi_param_map,
            held,
        )
        if identifiable:
            # One warning for what the search found, as for the other
            # regressions (#392, #555): a life-model parameter that runs
            # off (stress levels with no failures at one end, say), or
            # else a search that did not reach a verified maximum. (A
            # ridge of non-identifiable parameters was said above.)
            verdict = finish_search(
                fun,
                res,
                free_coefficients(self, fixed, phi_param_map),
                start,
                n_obs,
                verified=verified,
                what="The accelerated life fit",
                floor=floor,
                # (one searched on the log scale already is judged there)
                one_sided=one_sided_positions(
                    [
                        (None, None) if i in log_scale else b
                        for i, b in enumerate(bounds)
                    ],
                    not_fixed,
                ),
                baseline=free_baseline(self, fixed),
                dist=self.dist.name,
                values=dict(zip(self.param_map, model.params)),
            )
            model.maximum = verdict.maximum
            # The exact observed information for the covariance, which
            # was a numerical Hessian.
            keep_information(
                model,
                verdict.no_maximum,
                verdict.derivatives,
                inv_trans,
                const,
                res.x,
                None,
            )
        else:
            # A ridge has no single maximum (said above, in its own words)
            model.maximum = "unverified"
        model.fun = fun
        return model
