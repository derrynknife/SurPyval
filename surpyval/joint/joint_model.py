"""Joint models of a longitudinal marker and a failure time (#891).

A marker read repeatedly on each unit -- a vibration level, an oil-debris
count, a wear measurement, a biomarker -- often drives the unit's risk of
failure. It is read with noise, only now and then, and the readings stop
when the unit fails. The obvious analyses are biased: a time-varying
covariate takes the last noisy reading as the truth (and pulls its effect
towards zero), and a model of the readings alone ignores that the units
whose marker rose fastest left the data first. The joint model describes
both at once, sharing each unit's random effects between its path and its
hazard, and its dynamic prediction gives each unit's survival given its
own readings so far: its remaining useful life, updated as readings
arrive.

The model (v1):

- the **longitudinal part** is a linear mixed model: a straight path
  with a random intercept and slope per unit, ``m_i(t) = (path_intercept
  + b0_i) + (path_slope + b1_i) t`` (plus the path covariates' effects on
  both), read with normal noise ``sigma``;
- the **survival part** is a Weibull proportional hazards model, linked
  by the path's current value: ``h_i(t) = (beta / alpha) (t /
  alpha)^(beta - 1) exp(gamma' z_i + association (m_i(t) -
  marker_centre))``;
- **estimation** is joint maximum likelihood, the random effects
  integrated out by adaptive Gauss-Hermite quadrature (Rizopoulos 2012).

Not yet (the API leaves room for each, without a break): an AFT or
accelerated-life survival part through cumulative exposure (``survival=``),
links to the path's slope or its cumulative value (``link=``), a spline
path (``path=``), several markers or a latent health index, and the
two-stage estimate as an option for comparison.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.univariate.information_criteria import InformationCriteriaMixin
from surpyval.utils.linalg import (
    delta_method_se,
    sf_link_bound,
    standard_errors_of,
    wald_bound_on_support,
)
from surpyval.utils.no_maximum import (
    maximum_entry,
    restored_maximum,
    warn_no_maximum,
    warn_unverified,
)
from surpyval.utils.rng import as_generator
from surpyval.utils.shapes import check_paired_rows, keeps_query_shape
from surpyval.utils.units import takes_unit
from surpyval.utils.validation import (
    BOUNDS,
    check_alpha_ci,
    check_option,
    warn_outside_unit_interval,
)

from ._data import JointData, joint_data
from ._fit import FitResult, fit_joint, observed_failures
from ._likelihood import Q
from ._params import (
    PATH_TERMS,
    kept_positions,
    parameter_names,
    unpack_natural,
)
from ._predict import Evaluator, draw_units, marginal, marginal_random
from .dynamic import DynamicPrediction

__all__ = ["JointModel"]

#: The options of v1, each the only value it takes yet.
SURVIVAL_MODELS = ("weibull_ph",)
LINKS = ("value",)
PATHS = ("linear",)

#: Parameters that are positive, and the correlation, for their bounds.
_POSITIVE = ("alpha", "beta", "sd_intercept", "sd_slope", "sigma")


def _check_choice(name: str, value: Any, accepted: tuple, later: str) -> None:
    if value not in accepted:
        raise ValueError(
            f"{name}={value!r} is not available yet: v1 of the joint model "
            f"has {name}={accepted[0]!r} only ({later} are planned, #891)."
        )


class JointModel(InformationCriteriaMixin, SerialisableMixin):
    """
    A joint model of a longitudinal marker and a failure time: a linear
    mixed model of the marker's readings and a Weibull proportional
    hazards model of the failure, linked by the marker's current
    (noise-free) value. See the module documentation for the model.

    ``JointModel.fit`` (or ``fit_from_df``) fits one to data;
    ``JointModel.from_params`` builds one from known parameters. A fitted
    model gives

    - for a **new unit** (no readings yet): its survival ``sf(x, Z)``
      and the other functions of its failure time, averaged over the
      units' random effects, with Wald bounds ``cb``;
    - for a **unit with readings** (:meth:`predict`): its survival from
      now given its readings, the dynamic prediction, updated as readings
      arrive.

    Attributes
    ----------
    params : numpy.ndarray
        The fitted parameters, in the order of ``parameter_names``: the
        Weibull scale ``alpha`` and shape ``beta`` of the baseline (for a
        unit whose marker stays at ``marker_centre``), the hazard
        coefficients (named by their columns), the ``association`` (the
        log hazard ratio per unit of the marker's current value), the
        path's ``path_intercept`` and ``path_slope`` (and, per path
        covariate ``w``, ``path_intercept:w`` and ``path_slope:w``), the
        random effects' ``sd_intercept``, ``sd_slope`` and ``corr``, and
        the noise ``sigma``. An aliased coefficient is ``nan``.
    parameter_names : list of str
        The names of ``params``.
    marker_centre : float
        The marker value the baseline is at (the mean reading of the
        fitted data), which conditions the fit and gives ``alpha`` a
        meaning; it is not a parameter.
    maximum : str
        What the fit reached: ``"verified"``, ``"unverified"`` or ``"no
        finite maximum"`` (``"not applicable"`` built from parameters).

    Examples
    --------
    Pumps whose vibration rises at a pump-specific rate, and whose hazard
    rises with it. ``simulate`` draws a fleet from a model built from
    known parameters, with a reading every 30 days for two years; the fit
    recovers the association.

    >>> from surpyval.joint import JointModel
    >>> truth = JointModel.from_params(
    ...     {"alpha": 900.0, "beta": 1.4, "association": 0.9,
    ...      "path_intercept": 2.8, "path_slope": 0.0045,
    ...      "sd_intercept": 0.5, "sd_slope": 0.002, "corr": 0.0,
    ...      "sigma": 0.6},
    ...     marker_centre=3.5,
    ... )
    >>> fleet = truth.simulate(150, 30.0, 730.0, random_state=1)
    >>> model = JointModel.fit_from_df(
    ...     fleet, x_col="x", c_col="c", i_col="i",
    ...     marker_x_col="marker_x", marker_y_col="marker_y",
    ... )
    >>> model.maximum
    'verified'
    >>> lo, hi = model.param_cb("association")
    >>> bool(lo < 0.9 < hi)
    True
    """

    _model_tag = "JointModel"
    _human_name = "joint"

    #: The survival part, link and path (the only ones of v1).
    survival: str = "weibull_ph"
    link: str = "value"
    path: str = "linear"
    #: What the fit reached (see the class docstring).
    maximum: str = "not applicable"
    #: The covariate columns' names (from ``fit_from_df``), or ``None``.
    Z_cols: "list[str] | None" = None
    path_Z_cols: "list[str] | None" = None
    #: The fitted data (``None`` for a model built from parameters or
    #: restored from a dict).
    data: "JointData | None" = None
    #: The fit's internal state (its search point and quadrature nodes),
    #: for the checks of its derivatives and maximum; not saved.
    _fit_state: "FitResult | None" = None
    #: The covariance of the estimated parameters (``None`` where there is
    #: none).
    _covariance: "npt.NDArray | None" = None
    _neg_ll: "float | None" = None  # type: ignore[assignment]

    def __init__(
        self,
        params: npt.ArrayLike,
        parameter_names: list[str],
        n_Z: int,
        n_path_Z: int,
        marker_centre: float = 0.0,
        n_quad: int = 7,
        n_legendre: int = 20,
    ) -> None:
        self.params = np.asarray(params, dtype=float).copy()
        self.parameter_names = list(parameter_names)
        self.n_Z = int(n_Z)
        self.n_path_Z = int(n_path_Z)
        self.marker_centre = float(marker_centre)
        self.n_quad = int(n_quad)
        self.n_legendre = int(n_legendre)
        if self.params.shape != (len(self.parameter_names),):
            raise ValueError(
                "params must have one value per parameter name; got "
                f"{self.params.shape} for {len(self.parameter_names)}."
            )
        self._check_values()

    def _check_values(self) -> None:
        nat = unpack_natural(self._eval_params(), self.n_Z, self.n_path_Z)
        for name in _POSITIVE:
            value = self.params[self.parameter_names.index(name)]
            if not (np.isfinite(value) and value > 0):
                raise ValueError(
                    f"{name} must be positive and finite; got {value!r}."
                )
        if not (abs(nat.corr) < 1):
            raise ValueError(
                "corr, the correlation of the random intercept and slope, "
                f"must be strictly between -1 and 1; got {nat.corr!r}."
            )
        if not np.all(np.isfinite(self._eval_params())):
            raise ValueError("Every parameter must be finite.")

    # -- names ----------------------------------------------------------------
    @property
    def _z_names(self) -> list[str]:
        return self.parameter_names[2 : 2 + self.n_Z]

    @property
    def _w_names(self) -> list[str]:
        start = 3 + self.n_Z + Q
        names = self.parameter_names[start : start + Q * self.n_path_Z]
        return [n.split(":", 1)[1] for n in names[::Q]]

    @property
    def aliased(self) -> npt.NDArray:
        """The columns of ``Z`` whose coefficients the data cannot
        determine (constant, or a combination of the others): ``nan`` in
        ``params``, and 0 in every prediction."""
        gamma = self.params[2 : 2 + self.n_Z]
        return np.flatnonzero(np.isnan(gamma))

    @property
    def aliased_path(self) -> npt.NDArray:
        """The columns of ``path_Z`` whose effects the data cannot
        determine, as :attr:`aliased` for ``Z``."""
        start = 3 + self.n_Z + Q
        eff = self.params[start : start + Q * self.n_path_Z]
        return np.flatnonzero(np.isnan(eff[::Q]))

    @property
    def covariance_names(self) -> list[str]:
        """The names of the rows of :meth:`covariance`: the estimated
        parameters (an aliased coefficient is not one)."""
        return [
            n
            for n, v in zip(self.parameter_names, self.params)
            if np.isfinite(v)
        ]

    def _estimates(self) -> npt.NDArray:
        return self.params[np.isfinite(self.params)]

    def _eval_params(self) -> npt.NDArray:
        """``params`` with an aliased coefficient as 0."""
        return np.where(np.isnan(self.params), 0.0, self.params)

    def _values_with(self, estimates: npt.NDArray) -> npt.NDArray:
        """The parameters with the estimated ones replaced by
        ``estimates`` (aliased ones 0), for the delta method."""
        values = self._eval_params()
        values[np.isfinite(self.params)] = estimates
        return values

    def _evaluator(self, values: "npt.NDArray | None" = None) -> Evaluator:
        return Evaluator.build(
            self._eval_params() if values is None else values,
            self.n_Z,
            self.n_path_Z,
            self.marker_centre,
            self.n_legendre,
        )

    def __getattr__(self, name: str) -> Any:
        # Each parameter by name: model.association, model.sigma, ...
        names = self.__dict__.get("parameter_names")
        if names is not None and name in names:
            return float(self.__dict__["params"][names.index(name)])
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def _parameter_bounds(self) -> list:
        """Each parameter's support, ``(lower, upper)`` (``None`` for
        unbounded), in the order of ``parameter_names``."""
        out: list[tuple[float | None, float | None]] = []
        for name in self.parameter_names:
            if name in _POSITIVE:
                out.append((0, None))
            elif name == "corr":
                out.append((-1, 1))
            else:
                out.append((None, None))
        return out

    @property
    def D(self) -> npt.NDArray:
        """The covariance matrix of the random intercept and slope."""
        return unpack_natural(self._eval_params(), self.n_Z, self.n_path_Z).D

    # -- building -----------------------------------------------------------
    @classmethod
    def from_params(
        cls,
        params: dict,
        marker_centre: float = 0.0,
        n_quad: int = 7,
        n_legendre: int = 20,
        **kwargs: Any,
    ) -> "JointModel":
        """
        A joint model from known parameters.

        Parameters
        ----------
        params : dict
            Each parameter by name (see :attr:`parameter_names`): the
            Weibull ``alpha`` and ``beta``, the ``association``, the
            path's ``path_intercept`` and ``path_slope``, the random
            effects' ``sd_intercept``, ``sd_slope`` and ``corr``, and the
            noise ``sigma``. A hazard covariate's coefficient is given by
            its name, in the order of the columns of ``Z``; a path
            covariate ``w``'s effects as ``path_intercept:w`` and
            ``path_slope:w`` (both).
        marker_centre : float, optional
            The marker value at which the baseline is ``alpha``'s Weibull.
            Default 0.
        n_quad, n_legendre : int, optional
            As for :meth:`fit`.
        unit : str, optional
            The unit of the times (a label, see ``fit``).

        Examples
        --------
        >>> from surpyval.joint import JointModel
        >>> model = JointModel.from_params(
        ...     {"alpha": 1000.0, "beta": 1.5, "load": 0.3,
        ...      "association": 0.8, "path_intercept": 3.0,
        ...      "path_slope": 0.004, "sd_intercept": 0.5,
        ...      "sd_slope": 0.002, "corr": 0.0, "sigma": 0.5},
        ...     marker_centre=3.0,
        ... )
        >>> model.parameter_names[:4]
        ['alpha', 'beta', 'load', 'association']
        >>> round(float(model.sf(500, Z=[1.0])), 4)
        0.2804
        """
        return _from_params(
            cls, params, marker_centre, n_quad, n_legendre, **kwargs
        )

    @classmethod
    @takes_unit
    def fit(
        cls,
        x: Any = None,
        c: Any = None,
        Z: Any = None,
        i: Any = None,
        tl: Any = None,
        tr: Any = None,
        *,
        marker_x: Any = None,
        marker_y: Any = None,
        marker_i: Any = None,
        path_Z: Any = None,
        xl: Any = None,
        xr: Any = None,
        survival: str = "weibull_ph",
        link: str = "value",
        path: str = "linear",
        n_quad: int = 7,
        n_legendre: int = 20,
        init: "npt.ArrayLike | None" = None,
        max_iter: int = 1000,
        marker_centre: "float | None" = None,
    ) -> "JointModel":
        """
        Fit a joint model by maximum likelihood.

        The data are in two parts: one row per unit (its failure or
        censoring time, its censoring, truncation and covariates, as for
        any SurPyval regression) and one row per marker reading (its
        age, value and unit). The random effects are integrated out by
        adaptive Gauss-Hermite quadrature with ``n_quad`` points per
        dimension, the nodes held fixed while the parameters are
        searched and then moved to each unit's new posterior mode, until
        the estimates stop moving (as R's ``JM`` does). The start is the
        two-stage estimate: the readings fitted alone as a linear mixed
        model by REML, then the survival part with each unit's predicted
        path.

        Parameters
        ----------
        x : array_like
            Each unit's failure time, censoring time, or ``[lower,
            upper]`` interval (with ``c=2``). Positive.
        c : array_like, optional
            Each unit's censoring flag: 0 failed at ``x`` (the default),
            1 right censored (still running at ``x``), -1 left censored
            (failed before ``x``), 2 interval censored.
        Z : array_like, optional
            The hazard covariates, one row per unit (constant over its
            life).
        i : array_like, optional
            Each unit's id, which ``marker_i`` refers to. Default
            ``0, 1, ...``, the row numbers.
        tl, tr : scalar or array_like, optional
            Left and right truncation, per unit: a unit is in the data
            only because it failed in ``(tl, tr]`` (for left truncation,
            because it was still running at ``tl``, its entry). The
            random effects keep their distribution at age 0: the
            likelihood of a truncated unit is divided by the probability
            of its window, averaged over the random effects.
        marker_x, marker_y, marker_i : array_like
            The readings: their ages, values and units (required). A
            reading is taken at or before its unit's ``x`` (the upper end
            of an interval). A unit may have no readings.
        path_Z : array_like, optional
            Covariates of the path, one row per unit: each shifts both the
            path's intercept and its slope. Pass the same columns as
            ``Z`` to have a covariate act on both parts.
        xl, xr : array_like, optional
            The ends of each unit's interval, in place of ``x``.
        survival, link, path : str, optional
            The survival part (``"weibull_ph"``), the link (``"value"``,
            the path's current value) and the path (``"linear"``): the
            only ones of v1.
        n_quad : int, optional
            Gauss-Hermite points per random effect (``n_quad**2`` nodes per
            unit). Default 7; fewer than 5 is too coarse for units with
            few readings.
        n_legendre : int, optional
            Gauss-Legendre points for each cumulative hazard. Default 20.
        init : array_like, optional
            A starting point, in the order of ``parameter_names``.
        max_iter : int, optional
            The most iterations of each search. Default 1000.
        marker_centre : float, optional
            The marker value at which ``alpha`` is the baseline's Weibull
            scale (a "healthy" level, say). Default: the mean reading.
            It changes ``alpha`` and nothing else: every prediction is
            the same whatever it is.
        unit : str, optional
            The unit of the times, a label (``"hour"``, ``"cycle"``)
            shown on the model and kept by ``to_dict``.

        Returns
        -------
        JointModel
            The fitted model. Its ``maximum`` says what the fit reached.

        Warns
        -----
        UserWarning
            Where the fit did not reach a verified maximum, where the
            likelihood has no finite maximum (a hazard coefficient or the
            association running off to infinity, as a covariate that
            separates the failures from the survivors makes it), and where
            a covariate column is aliased.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.joint import JointModel
        >>> truth = JointModel.from_params(
        ...     {"alpha": 900.0, "beta": 1.4, "association": 0.9,
        ...      "path_intercept": 2.8, "path_slope": 0.0045,
        ...      "sd_intercept": 0.5, "sd_slope": 0.002, "corr": 0.0,
        ...      "sigma": 0.6},
        ...     marker_centre=3.5,
        ... )
        >>> df = truth.simulate(150, 30.0, 730.0, random_state=1)
        >>> units = df.groupby("i").first()
        >>> model = JointModel.fit(
        ...     units["x"], units["c"], i=units.index,
        ...     marker_x=df["marker_x"], marker_y=df["marker_y"],
        ...     marker_i=df["i"],
        ... )
        >>> model.parameter_names[:3]
        ['alpha', 'beta', 'association']
        """
        _check_choice("survival", survival, SURVIVAL_MODELS, "AFT and ALT")
        _check_choice("link", link, LINKS, "'slope' and 'cumulative'")
        _check_choice("path", path, PATHS, "spline paths")
        if int(n_quad) < 1 or int(n_legendre) < 1:
            raise ValueError("n_quad and n_legendre must be positive.")
        if marker_centre is not None and not np.isfinite(marker_centre):
            raise ValueError("marker_centre must be a finite number.")
        data = joint_data(
            x, c, Z, i, tl, tr, xl, xr, marker_x, marker_y, marker_i, path_Z
        )
        z_names = _column_names(cls._z_cols_of(Z), data.Z.shape[1], "coef")
        w_names = _column_names(cls._z_cols_of(path_Z), data.W.shape[1], "w")
        names = parameter_names(z_names, w_names)
        if init is not None:
            init = _init_values(init, names)
        if len(set(names)) != len(names):
            raise ValueError(
                "The covariate columns' names collide with the model's "
                f"parameter names ({names}): rename the columns."
            )
        result = fit_joint(
            data,
            n_quad=int(n_quad),
            n_legendre=int(n_legendre),
            init=init,
            max_iter=int(max_iter),
            centre=marker_centre,
        )
        return cls._from_result(
            result,
            data,
            names,
            n_quad,
            n_legendre,
            (cls._z_cols_of(Z), cls._z_cols_of(path_Z)),
        )

    @staticmethod
    def _z_cols_of(Z: Any) -> "list[str] | None":
        if isinstance(Z, pd.DataFrame):
            return [str(c) for c in Z.columns]
        if isinstance(Z, pd.Series) and Z.name is not None:
            return [str(Z.name)]
        return None

    @classmethod
    def _from_result(
        cls,
        result: FitResult,
        data: JointData,
        names: list[str],
        n_quad: int,
        n_legendre: int,
        cols: tuple = (None, None),
    ) -> "JointModel":
        p, pw = data.Z.shape[1], data.W.shape[1]
        prep = result.prepared
        params = np.full(len(names), np.nan)
        kept = kept_positions(p, pw, prep.z_kept, prep.w_kept)
        params[kept] = result.natural
        model = cls.__new__(cls)
        model.params = params
        model.parameter_names = names
        model.n_Z, model.n_path_Z = p, pw
        model.marker_centre = prep.scaling.y_c
        model.n_quad, model.n_legendre = int(n_quad), int(n_legendre)
        model.maximum = result.maximum
        model._covariance = result.covariance
        model.data = data
        model._fit_state = result
        model.Z_cols, model.path_Z_cols = cols
        model._set_fit_stats(result.neg_ll, observed_failures(data))
        model._say_fit(result, names)
        return model

    def _say_fit(self, result: FitResult, names: list[str]) -> None:
        """The fit's warnings: aliased columns, then what it reached."""
        _warn_aliased(self)
        if result.maximum == "no finite maximum":
            gone = [
                (self._z_names + ["association"])[j] for j in result.runaway
            ]
            warn_no_maximum(
                "the likelihood keeps increasing as {} grow{} without bound "
                "(a covariate separates the failures from the survivors, as "
                "a group with no failures does)".format(
                    ", ".join(gone), "s" if len(gone) == 1 else ""
                ),
                "The fit stopped where the increase became too small to "
                "follow, and those values, their standard errors and bounds "
                "are meaningless",
                "consider removing or coarsening the covariate",
            )
        elif result.maximum == "unverified":
            warn_unverified(
                "The joint model's fit",
                advice=(
                    "check the fit (a random-effect variance near zero, or "
                    "too few readings per unit, leaves the likelihood flat), "
                    "or try another `init` or a larger `max_iter`"
                ),
            )

    @classmethod
    @takes_unit
    def fit_from_df(
        cls,
        df: pd.DataFrame,
        x_col: str = "x",
        c_col: "str | None" = None,
        i_col: str = "i",
        marker_x_col: str = "marker_x",
        marker_y_col: str = "marker_y",
        Z_cols: "str | list[str] | None" = None,
        path_Z_cols: "str | list[str] | None" = None,
        tl_col: "str | None" = None,
        tr_col: "str | None" = None,
        xl_col: "str | None" = None,
        xr_col: "str | None" = None,
        **fit_kwargs: Any,
    ) -> "JointModel":
        """
        Fit a joint model from a DataFrame with one row per reading (the
        "long" layout of R's ``JM`` data, such as
        :func:`~surpyval.datasets.load_pbc2`): each row a reading
        (``marker_x_col``, ``marker_y_col``) of a unit (``i_col``), with
        the unit's failure or censoring time, censoring flag, truncation
        and covariates repeated on each of its rows. A unit's columns must
        be the same on all of its rows. A row whose reading age and value
        are both missing is a unit with no reading there (so a unit with
        no readings is one such row).

        Parameters
        ----------
        df : pandas.DataFrame
            The data, one row per reading.
        x_col, c_col, i_col : str, optional
            The units' times (default ``"x"``), censoring flags (default:
            none, every unit failed) and ids (default ``"i"``).
        marker_x_col, marker_y_col : str, optional
            The readings' ages and values (defaults ``"marker_x"`` and
            ``"marker_y"``).
        Z_cols, path_Z_cols : str or list of str, optional
            The hazard and path covariates (by name; their names are the
            coefficients').
        tl_col, tr_col, xl_col, xr_col : str, optional
            Truncation, and interval ends in place of ``x_col``.
        **fit_kwargs
            The other arguments of :meth:`fit`.

        Examples
        --------
        The longitudinal primary biliary cirrhosis data: serum bilirubin
        (on the log scale) read at each visit, and death (transplantation
        censored). The association is the log hazard ratio per unit of
        log bilirubin.

        >>> import numpy as np
        >>> from surpyval.datasets import load_pbc2
        >>> from surpyval.joint import JointModel
        >>> df = load_pbc2()
        >>> df["log_bilirubin"] = np.log(df["serBilir"])
        >>> df["censored"] = 1 - df["status2"]
        >>> model = JointModel.fit_from_df(
        ...     df, x_col="years", c_col="censored", i_col="id",
        ...     marker_x_col="year", marker_y_col="log_bilirubin",
        ... )
        >>> round(model.association, 2)
        1.24
        """
        frame = _long_frame(
            df,
            x_col,
            c_col,
            i_col,
            marker_x_col,
            marker_y_col,
            Z_cols,
            path_Z_cols,
            tl_col,
            tr_col,
            xl_col,
            xr_col,
        )
        return cls.fit(**frame, **fit_kwargs)

    @classmethod
    @takes_unit
    def fit_from_items(
        cls,
        items: Any,
        path_covariates: "bool | list | None" = None,
        **fit_kwargs: Any,
    ) -> "JointModel":
        """
        Fit a joint model to items' histories: a list of
        :class:`~surpyval.ItemHistory`, or an
        :class:`~surpyval.EventHistoryData` made from them. Each item's
        ``markers`` are its readings; its failure (exact, in an interval
        or before an age; at most one), end of observation and truncation
        are the unit's event; its ``covariates`` are the hazard
        covariates ``Z``.

        Parameters
        ----------
        items : list of ItemHistory, or EventHistoryData
            The units.
        path_covariates : bool or list, optional
            Which covariates also act on the path (``path_Z``): ``True``
            for all of them, or a list of their names (or column
            numbers). Default none.
        **fit_kwargs
            The other arguments of :meth:`fit`.

        Examples
        --------
        >>> from surpyval import ItemHistory
        >>> from surpyval.joint import JointModel
        >>> truth = JointModel.from_params(
        ...     {"alpha": 900.0, "beta": 1.4, "association": 0.9,
        ...      "path_intercept": 2.8, "path_slope": 0.0045,
        ...      "sd_intercept": 0.5, "sd_slope": 0.002, "corr": 0.0,
        ...      "sigma": 0.6},
        ...     marker_centre=3.5,
        ... )
        >>> df = truth.simulate(100, 30.0, 730.0, random_state=2)
        >>> items = [
        ...     ItemHistory(
        ...         i,
        ...         observed=None if g["c"].iat[0] else g["x"].iat[0],
        ...         right_censored=g["x"].iat[0],
        ...         markers=dict(zip(g["marker_x"], g["marker_y"])),
        ...     )
        ...     for i, g in df.groupby("i")
        ... ]
        >>> model = JointModel.fit_from_items(items)
        >>> model.maximum
        'verified'
        """
        return cls.fit(**_from_items(items, path_covariates), **fit_kwargs)

    # -- a new unit -----------------------------------------------------------
    def _rows(self, x: npt.NDArray, Z: Any, path_Z: Any) -> tuple:
        """The covariate rows paired with the times ``x``."""
        z = _rows_for("Z", Z, self.n_Z, x.size)
        w = _rows_for("path_Z", path_Z, self.n_path_Z, x.size)
        n = max(x.size, z.shape[0], w.shape[0])
        if x.size == 1 and n > 1:
            x = np.repeat(x, n)
        z = np.broadcast_to(z, (n, self.n_Z)) if z.shape[0] == 1 else z
        w = np.broadcast_to(w, (n, self.n_path_Z)) if w.shape[0] == 1 else w
        return x, np.ascontiguousarray(z), np.ascontiguousarray(w)

    def _marginal(
        self, x: npt.ArrayLike, Z: Any, path_Z: Any
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """``(x, log sf, log df)`` of a new unit, ``nan`` where the time
        or a covariate is missing; nothing has happened by age 0."""
        xs, z, w = self._rows(np.asarray(x, dtype=float).ravel(), Z, path_Z)
        log_sf = np.zeros(xs.size)
        log_df = np.full(xs.size, -np.inf)
        missing = np.isnan(xs) | np.isnan(z).any(1) | np.isnan(w).any(1)
        inside = (xs > 0) & np.isfinite(xs) & ~missing
        if np.any(inside):
            ls, ld = marginal(
                self._evaluator(), xs[inside], z[inside], w[inside]
            )
            log_sf[inside], log_df[inside] = ls, ld
        log_sf[np.isposinf(xs)] = -np.inf
        log_sf[missing] = np.nan
        log_df[missing] = np.nan
        return xs, log_sf, log_df

    @keeps_query_shape
    def sf(self, x: npt.ArrayLike, Z: Any = None, path_Z: Any = None) -> Any:
        """
        The survival function of a new unit (one with no readings yet):
        ``S(x | z, w)``, the conditional survival averaged over the
        units' random effects.

        Parameters
        ----------
        x : array_like
            The ages.
        Z, path_Z : array_like, optional
            The unit's hazard and path covariates (required where the
            model has them): one row per age, or one row for every age.
            A missing (``nan``) value gives ``nan`` for its row.

        Examples
        --------
        >>> from surpyval.joint import JointModel
        >>> model = JointModel.from_params(
        ...     {"alpha": 1000.0, "beta": 1.5, "association": 0.8,
        ...      "path_intercept": 3.0, "path_slope": 0.004,
        ...      "sd_intercept": 0.5, "sd_slope": 0.002, "corr": 0.0,
        ...      "sigma": 0.5},
        ...     marker_centre=3.0,
        ... )
        >>> model.sf([250, 500, 1000]).round(4)
        array([0.7977, 0.3682, 0.0429])
        """
        return np.exp(self._marginal(x, Z, path_Z)[1])

    @keeps_query_shape
    def ff(self, x: npt.ArrayLike, Z: Any = None, path_Z: Any = None) -> Any:
        """The failure function of a new unit, ``1 - sf`` (see
        :meth:`sf`)."""
        return -np.expm1(self._marginal(x, Z, path_Z)[1])

    @keeps_query_shape
    def Hf(self, x: npt.ArrayLike, Z: Any = None, path_Z: Any = None) -> Any:
        """The cumulative hazard of a new unit, ``-log sf`` (see
        :meth:`sf`)."""
        return 0.0 - self._marginal(x, Z, path_Z)[1]

    @keeps_query_shape
    def df(self, x: npt.ArrayLike, Z: Any = None, path_Z: Any = None) -> Any:
        """The density of a new unit's failure time (see :meth:`sf`)."""
        return np.exp(self._marginal(x, Z, path_Z)[2])

    @keeps_query_shape
    def hf(self, x: npt.ArrayLike, Z: Any = None, path_Z: Any = None) -> Any:
        """The hazard of a new unit, ``df / sf``: the population hazard of
        the units still running (see :meth:`sf`)."""
        x, log_sf, log_df = self._marginal(x, Z, path_Z)
        with np.errstate(invalid="ignore"):
            out = np.exp(log_df - log_sf)
        out = np.where(x < 0, 0.0, out)
        return np.where(np.isposinf(x), np.nan, out)

    @keeps_query_shape
    def qf(self, p: npt.ArrayLike, Z: Any = None, path_Z: Any = None) -> Any:
        """The quantiles of a new unit's failure time (``nan``, with a
        warning, for a probability outside [0, 1])."""
        probs = np.asarray(p, dtype=float).ravel()
        probs = np.where(warn_outside_unit_interval(probs), np.nan, probs)
        probs, z, w = self._rows(probs, Z, path_Z)
        out = np.full(probs.size, np.nan)
        ok = np.isfinite(probs) & ~np.isnan(z).any(1) & ~np.isnan(w).any(1)
        out[ok & (probs <= 0)] = 0.0
        out[ok & (probs >= 1)] = np.inf
        inner = np.flatnonzero(ok & (probs > 0) & (probs < 1))
        if inner.size:
            out[inner] = self._quantiles(probs[inner], z[inner], w[inner])
        return out

    def _quantiles(
        self, p: npt.NDArray, z: npt.NDArray, w: npt.NDArray
    ) -> npt.NDArray:
        """Bisection in log time for ``log sf = log(1 - p)``; ``inf``
        beyond the share of new units that ever fail."""
        from .dynamic import _bisect_log_time

        ev = self._evaluator()
        return _bisect_log_time(
            lambda t: marginal(ev, t, z, w)[0],
            np.log1p(-p),
            float(self.params[0]),
        )

    def random(
        self,
        size: int,
        Z: Any = None,
        path_Z: Any = None,
        random_state: "int | np.random.Generator | None" = None,
    ) -> npt.NDArray:
        """
        Failure times of ``size`` new units with covariates ``Z`` and
        ``path_Z`` (one row each, for all of them): each unit's random
        effects drawn from their distribution, then its failure time
        from its hazard.

        ``random_state`` is ``None`` (numpy's global generator, so
        ``np.random.seed`` reproduces a draw), an int, or a
        ``numpy.random.Generator``.
        """
        rng = as_generator(random_state)
        z = _rows_for("Z", Z, self.n_Z, 1)
        w = _rows_for("path_Z", path_Z, self.n_path_Z, 1)
        if z.shape[0] != 1 or w.shape[0] != 1:
            raise ValueError(
                "random draws units with one row of covariates: pass Z and "
                "path_Z as single rows (draw each group separately)."
            )
        return marginal_random(self._evaluator(), int(size), z[0], w[0], rng)

    def simulate(
        self,
        n_units: int,
        reading_interval: float,
        end: float,
        Z: Any = None,
        path_Z: Any = None,
        random_state: "int | np.random.Generator | None" = None,
    ) -> pd.DataFrame:
        """
        A fleet drawn from the model: ``n_units`` new units, each read
        every ``reading_interval`` from age 0 until it fails or the
        observation ends at ``end`` (right censored there).

        Returns a DataFrame with one row per reading, in the layout of
        :meth:`fit_from_df` (its default column names): ``i``, ``x``,
        ``c``, ``marker_x`` and ``marker_y``, with the covariates (as
        ``Z_0``, ... and ``path_Z_0``, ... or the model's column names).

        Parameters
        ----------
        n_units : int
            The number of units.
        reading_interval : float
            The time between readings.
        end : float
            The age at which observation ends.
        Z, path_Z : array_like, optional
            The covariates: one row for every unit, or one per unit.
        random_state : int or numpy.random.Generator, optional
            As for :meth:`random`.

        Examples
        --------
        >>> from surpyval.joint import JointModel
        >>> model = JointModel.from_params(
        ...     {"alpha": 1000.0, "beta": 1.5, "association": 0.8,
        ...      "path_intercept": 3.0, "path_slope": 0.004,
        ...      "sd_intercept": 0.5, "sd_slope": 0.002, "corr": 0.0,
        ...      "sigma": 0.5},
        ...     marker_centre=3.0,
        ... )
        >>> df = model.simulate(5, 100.0, 1000.0, random_state=0)
        >>> list(df.columns)
        ['i', 'x', 'c', 'marker_x', 'marker_y']
        """
        return _simulate(
            self, n_units, reading_interval, end, Z, path_Z, random_state
        )

    # -- a unit with readings -------------------------------------------------
    def predict(
        self,
        marker_x: Any,
        marker_y: Any = None,
        Z: Any = None,
        path_Z: Any = None,
        *,
        age: "float | None" = None,
    ) -> DynamicPrediction:
        """
        The dynamic prediction of one unit: its failure-age distribution
        given its readings so far and that it is still running at
        ``age``.

        Parameters
        ----------
        marker_x, marker_y : array_like
            The unit's readings: their ages and values (no readings at
            all is a new unit that has run to ``age``). ``marker_x`` may
            instead be an :class:`~surpyval.ItemHistory` with
            ``markers``, whose covariates are then ``Z``.
        Z, path_Z : array_like, optional
            The unit's hazard and path covariates (one row each).
        age : float, optional
            The unit's age now, at or after its last reading. Default:
            the age of its last reading.

        Returns
        -------
        DynamicPrediction
            Its ``sf(x)`` is ``P(T > x | T > age, readings)``, with
            ``remaining_life``, ``cb``, ``marker`` and the other
            functions of the unit's failure age.

        Examples
        --------
        >>> from surpyval.joint import JointModel
        >>> model = JointModel.from_params(
        ...     {"alpha": 1000.0, "beta": 1.5, "association": 0.8,
        ...      "path_intercept": 3.0, "path_slope": 0.004,
        ...      "sd_intercept": 0.5, "sd_slope": 0.002, "corr": 0.0,
        ...      "sigma": 0.5},
        ...     marker_centre=3.0,
        ... )
        >>> pred = model.predict([0, 100, 200], [3.1, 3.6, 4.4])
        >>> pred.age
        200.0
        >>> round(float(pred.sf(400)), 4)
        0.5331
        """
        t, y, z, w, now = _prediction_inputs(
            self, marker_x, marker_y, Z, path_Z, age
        )
        return DynamicPrediction(self, t, y, z, w, now)

    # -- inference ------------------------------------------------------------
    def covariance(self) -> npt.NDArray:
        """
        The covariance of the estimated parameters, in the order of
        :attr:`covariance_names`: the inverse of the observed information
        (autograd's Hessian of the negative log-likelihood, with the
        quadrature nodes at the fit), carried to the parameters by the
        delta method. Raises a ``ValueError`` where there is none: a model
        built from its parameters or restored without it, or a fit whose
        information could not be inverted.
        """
        if self._covariance is None:
            if self.maximum in ("verified", "unverified", "no finite maximum"):
                why = (
                    "the observed information at the fit is not positive "
                    "definite"
                )
            else:
                why = (
                    "it was built from its parameters, or restored from a "
                    "dict saved without its covariance; refit it to the data"
                )
            raise ValueError(
                f"This joint model has no parameter covariance: {why}."
            )
        return np.array(self._covariance, dtype=float)

    def standard_errors(self) -> npt.NDArray:
        """The standard errors of the estimated parameters, the square
        roots of the diagonal of :meth:`covariance`, in the order of
        :attr:`covariance_names`."""
        return standard_errors_of(self.covariance())

    def param_cb(
        self, name: str, alpha_ci: float = 0.05, bound: str = "two-sided"
    ) -> npt.NDArray:
        """
        The Wald confidence bound on one parameter, on a scale that keeps
        it in its support: the log of a positive one (``alpha``,
        ``beta``, the standard deviations and ``sigma``), the generalised
        logit of ``corr`` on (-1, 1), and the natural scale of the others.

        Parameters
        ----------
        name : str
            One of :attr:`covariance_names`.
        alpha_ci : float, optional
            The total tail probability. Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Default ``'two-sided'``, ``[lower, upper]``.
        """
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        names = self.covariance_names
        check_option("name", name, names)
        idx = names.index(name)
        var = float(self.covariance()[idx, idx])
        lower: "float | None" = None
        upper: "float | None" = None
        if name in _POSITIVE:
            lower = 0.0
        elif name == "corr":
            lower, upper = -1.0, 1.0
        return wald_bound_on_support(
            float(self._estimates()[idx]),
            var,
            lower,
            upper,
            alpha_ci,
            bound,
            name=name,
        )

    def cb(
        self,
        x: npt.ArrayLike,
        Z: Any = None,
        path_Z: Any = None,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """
        Wald confidence bounds on a new unit's ``sf``, ``ff`` or ``Hf`` at
        times ``x`` (see :meth:`sf`), by the delta method from
        :meth:`covariance`, formed on ``log(-log sf)`` so they stay in
        [0, 1].

        Parameters
        ----------
        x : array_like
            The times.
        Z, path_Z : array_like, optional
            The covariates, as for :meth:`sf`.
        on : {'sf', 'ff', 'Hf'}, optional
            The function to bound (``'R'`` and ``'F'`` are ``'sf'`` and
            ``'ff'``). Default ``'sf'``.
        alpha_ci : float, optional
            The total tail probability. Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Default ``'two-sided'``, ``[lower, upper]`` on a last axis.
        """
        check_option("on", on, ("sf", "R", "ff", "F", "Hf"))
        on = {"R": "sf", "F": "ff"}.get(on, on)
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        cov = self.covariance()
        x_arr = np.asarray(x, dtype=float)
        flat, z, w = self._rows(np.atleast_1d(x_arr).ravel(), Z, path_Z)
        _, log_sf, _ = self._marginal(flat, z, w)

        def sf_of(estimates: npt.NDArray) -> npt.NDArray:
            model = self._copy_with(self._values_with(estimates))
            return np.exp(model._marginal(flat, z, w)[1])

        with np.errstate(all="ignore"):
            se = delta_method_se(sf_of, self._estimates(), cov)
        se = np.where(flat <= 0, 0.0, se)
        out = sf_link_bound(
            np.exp(log_sf),
            se,
            alpha_ci,
            bound,
            "loglog",
            ff_hat=-np.expm1(log_sf),
            on=on,
        )
        shape = (flat.size,) if flat.size != x_arr.size else x_arr.shape
        return out.reshape(shape + ((2,) if bound == "two-sided" else ()))

    def _copy_with(self, values: npt.NDArray) -> "JointModel":
        """A copy of this model with the parameters ``values``."""
        model = type(self).__new__(type(self))
        model.__dict__.update(self.__dict__)
        params = np.array(values, dtype=float)
        params[np.isnan(self.params)] = np.nan
        model.params = params
        return model

    def neg_ll_of(self, params: npt.ArrayLike) -> float:
        """
        The negative log-likelihood of the fitted data at the parameters
        ``params`` (the estimated ones, in the order of
        :attr:`covariance_names`), by adaptive quadrature with the nodes
        placed at those parameters: the model's likelihood, whose maximum
        :meth:`neg_ll` reports.
        """
        from ._fit import natural_neg_ll, prepare
        from ._likelihood import adaptive_neg_ll
        from ._params import internal_from_natural

        if self.data is None:
            raise ValueError(
                "This joint model keeps no data (it was built from its "
                "parameters, or restored from a dict): refit it."
            )
        values = np.asarray(params, dtype=float).ravel()
        state = self._fit_state
        if state is not None:
            prepared, last = state.prepared, state.nodes
        else:
            z_kept = np.setdiff1d(np.arange(self.n_Z), self.aliased)
            w_kept = np.setdiff1d(np.arange(self.n_path_Z), self.aliased_path)
            prepared = prepare(
                self.data,
                z_kept,
                w_kept,
                self.n_quad,
                self.n_legendre,
                self.marker_centre,
            )
            last = None
        theta = internal_from_natural(
            values, prepared.prob.layout, prepared.scaling
        )
        with np.errstate(all="ignore"):
            value = adaptive_neg_ll(theta, prepared.prob, last)
        return natural_neg_ll(value, prepared)

    # -- information criteria -------------------------------------------------
    def _set_fit_stats(self, neg_ll: float, n_failures: int) -> None:
        self._neg_ll = float(neg_ll)
        self._ic_n = float(max(n_failures, 1))

    def _ic_k(self) -> int:
        return int(np.sum(np.isfinite(self.params)))

    def _ic_sample_size_from_data(self) -> float:
        raise ValueError(
            "This joint model has no likelihood: it was built from its "
            "parameters; fit it to data for neg_ll, aic and bic."
        )

    def neg_ll(self) -> float:
        """
        The negative log-likelihood of the fitted model: of the readings
        and the failure times together (the random effects integrated
        out), in the data's own units. It ranks joint models fitted to
        the same readings and failures by :meth:`aic` or :meth:`bic`; it
        is not comparable with a model of the failure times alone. Raises
        a ``ValueError`` for a model built from its parameters.
        """
        if self._neg_ll is None:
            raise ValueError(
                "This joint model has no likelihood: it was built from its "
                "parameters, or restored from a dict saved without it; fit "
                "it to data for neg_ll, aic and bic."
            )
        return float(self._neg_ll)

    def aic(self) -> float:
        """Akaike's information criterion, ``2 k + 2 neg_ll()``, ``k`` the
        number of estimated parameters."""
        return float(super().aic())

    def bic(self) -> float:
        """The Bayesian information criterion, ``k ln n + 2 neg_ll()``,
        ``n`` the number of units whose failure was observed (exactly or
        within a window), as every SurPyval model counts it."""
        return float(super().bic())

    def aic_c(self) -> float:
        """The small-sample corrected AIC (``nan`` where ``n <= k + 1``)."""
        return float(super().aic_c())

    # -- reporting -----------------------------------------------------------
    def summary(self, alpha_ci: float = 0.05) -> pd.DataFrame:
        """
        The parameter table: each parameter's estimate, standard error and
        two-sided ``1 - alpha_ci`` Wald interval (that of
        :meth:`param_cb`), with, for the hazard coefficients and the
        association, ``exp(coef)`` (the hazard ratio) and the Wald test of
        0. Indexed by ``(part, name)``, ``part`` one of ``"survival"``,
        ``"association"``, ``"path"`` and ``"random effects"``.
        """
        check_alpha_ci(alpha_ci)
        return _summary(self, alpha_ci)

    def __repr__(self) -> str:
        return _repr(self)

    # -- serialisation --------------------------------------------------------
    def to_dict(self) -> dict:
        """The model as a plain dict (strict JSON), with its covariance and
        likelihood, so a restored model keeps ``cb``, ``param_cb``,
        ``predict`` and ``aic``. The data are not kept."""
        out: dict = {
            "model": self._model_tag,
            "survival": self.survival,
            "link": self.link,
            "path": self.path,
            "parameter_names": list(self.parameter_names),
            "params": [float(v) for v in self.params],
            "n_Z": self.n_Z,
            "n_path_Z": self.n_path_Z,
            "marker_centre": self.marker_centre,
            "n_quad": self.n_quad,
            "n_legendre": self.n_legendre,
        }
        if self.Z_cols is not None:
            out["Z_cols"] = list(self.Z_cols)
        if self.path_Z_cols is not None:
            out["path_Z_cols"] = list(self.path_Z_cols)
        if self._covariance is not None:
            out["covariance"] = np.asarray(self._covariance, float).tolist()
        if self._neg_ll is not None:
            out["_neg_ll"] = float(self._neg_ll)
            out["ic_n"] = self._ic_n
        out.update(maximum_entry(self.maximum))
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "JointModel":
        """Rebuild a joint model from :meth:`to_dict`'s dict."""
        require_model_tag(model_dict, cls._model_tag, "a joint model")
        for key in ("survival", "link", "path"):
            accepted = {
                "survival": SURVIVAL_MODELS,
                "link": LINKS,
                "path": PATHS,
            }[key]
            _check_choice(key, model_dict.get(key), accepted, "others")
        params = np.array(
            [np.nan if v is None else v for v in model_dict["params"]],
            dtype=float,
        )
        model = cls.__new__(cls)
        model.params = params
        model.parameter_names = list(model_dict["parameter_names"])
        model.n_Z = int(model_dict["n_Z"])
        model.n_path_Z = int(model_dict["n_path_Z"])
        model.marker_centre = float(model_dict["marker_centre"])
        model.n_quad = int(model_dict.get("n_quad", 7))
        model.n_legendre = int(model_dict.get("n_legendre", 20))
        model.Z_cols = model_dict.get("Z_cols")
        model.path_Z_cols = model_dict.get("path_Z_cols")
        model.maximum = restored_maximum(model_dict)
        if model_dict.get("covariance") is not None:
            model._covariance = np.asarray(
                model_dict["covariance"], dtype=float
            )
        if model_dict.get("_neg_ll") is not None:
            model._set_fit_stats(model_dict["_neg_ll"], model_dict["ic_n"])
        return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _column_names(given: "list[str] | None", p: int, prefix: str) -> list[str]:
    if given is not None and len(given) == p:
        return list(given)
    return [f"{prefix}_{j}" for j in range(p)]


def _init_values(init: Any, names: list[str]) -> npt.NDArray:
    values = np.asarray(init, dtype=float).ravel()
    if values.size != len(names):
        raise ValueError(
            f"init must have one value per parameter ({len(names)}: "
            f"{names}); got {values.size}."
        )
    return values


def _warn_aliased(model: "JointModel") -> None:
    """One warning for the aliased columns of ``Z``, in the form every
    regression gives it, and one for those of ``path_Z``."""
    import warnings

    from surpyval.univariate.regression._aliasing import (
        covariate_columns,
        warn_aliased,
    )
    from surpyval.utils.warnings import caller_stacklevel

    how = (
        "it is constant over the units, or a linear combination of the "
        "other columns (the hazard has its own intercept, the Weibull "
        "scale)"
    )
    if model.aliased.size:
        with covariate_columns(model.Z_cols):
            warn_aliased(model.aliased, how)
    if model.aliased_path.size:
        names = model._w_names
        shown = ", ".join(
            "{} ({!r})".format(j, names[j]) for j in model.aliased_path
        )
        warnings.warn(
            "Covariate column(s) {} of path_Z cannot be estimated: constant "
            "over the units, or a linear combination of the other columns "
            "(the path has its own intercept and slope). Their effects are "
            "aliased -- reported as nan, with nan standard errors -- and "
            "the model is fitted on the other columns; predictions take an "
            "aliased effect as 0. Remove the column(s) from path_Z to fit "
            "without this warning.".format(shown),
            UserWarning,
            stacklevel=caller_stacklevel(),
        )


def _rows_for(name: str, value: Any, p: int, n_x: int) -> npt.NDArray:
    """Covariate rows for a prediction: ``(m, p)``."""
    if p == 0:
        if value is not None and np.size(value) > 0:
            raise ValueError(
                f"This joint model was fitted without {name}; do not pass "
                "it."
            )
        return np.zeros((1, 0))
    if value is None:
        raise ValueError(
            f"This joint model depends on {name} ({p} column(s)): pass one "
            "row for every time, or one per time."
        )
    if isinstance(value, pd.DataFrame):
        value = value.to_numpy(dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1) if p == 1 else arr.reshape(1, -1)
    if arr.shape[1] != p:
        raise ValueError(
            f"{name} must have {p} column(s); got {arr.shape[1]}."
        )
    check_paired_rows(n_x, arr.shape[0], grid=False)
    return arr


def _from_params(
    cls: type,
    params: dict,
    marker_centre: float,
    n_quad: int,
    n_legendre: int,
    **kwargs: Any,
) -> JointModel:
    unit = kwargs.pop("unit", None)
    if kwargs:
        raise TypeError(
            f"from_params() got unexpected arguments {sorted(kwargs)}"
        )
    if not isinstance(params, dict):
        raise ValueError(
            "params must be a dict of each parameter by name (see the "
            "docstring of from_params)."
        )
    fixed = {"alpha", "beta", "association", "corr", "sigma"}
    fixed |= {f"path_{t}" for t in PATH_TERMS} | {
        f"sd_{t}" for t in PATH_TERMS
    }
    missing = sorted(fixed - set(params))
    if missing:
        raise ValueError(f"params is missing {missing}.")
    w_names: list[str] = []
    for key in params:
        if key.startswith("path_intercept:"):
            w_names.append(key.split(":", 1)[1])
    for w in w_names:
        if f"path_slope:{w}" not in params:
            raise ValueError(
                f"params gives path_intercept:{w} but not path_slope:{w}; "
                "a path covariate has both."
            )
    z_names = [
        k for k in params if k not in fixed and not k.startswith("path_")
    ]
    names = parameter_names(z_names, w_names)
    values = [float(params[n]) for n in names]
    model = cls(
        values,
        names,
        len(z_names),
        len(w_names),
        marker_centre,
        n_quad,
        n_legendre,
    )
    if unit is not None:
        from surpyval.utils.units import set_unit

        set_unit(model, unit)
    return model


def _long_frame(
    df: pd.DataFrame,
    x_col: str,
    c_col: "str | None",
    i_col: str,
    marker_x_col: str,
    marker_y_col: str,
    Z_cols: "str | list[str] | None",
    path_Z_cols: "str | list[str] | None",
    tl_col: "str | None",
    tr_col: "str | None",
    xl_col: "str | None",
    xr_col: "str | None",
) -> dict:
    """The arguments of ``fit`` from a long DataFrame."""
    from surpyval.utils.dataframe import check_columns, require_frame

    df = require_frame(df)
    Z_cols = [Z_cols] if isinstance(Z_cols, str) else Z_cols
    path_Z_cols = (
        [path_Z_cols] if isinstance(path_Z_cols, str) else path_Z_cols
    )
    unit_cols = [
        col
        for col in [
            None if xl_col is not None else x_col,
            c_col,
            tl_col,
            tr_col,
            xl_col,
            xr_col,
            *(Z_cols or []),
            *(path_Z_cols or []),
        ]
        if col is not None
    ]
    check_columns(
        df,
        i_col=i_col,
        marker_x_col=marker_x_col,
        marker_y_col=marker_y_col,
        **{f"col_{k}": v for k, v in enumerate(unit_cols)},
    )
    if df[i_col].isna().any():
        raise ValueError(f"{i_col!r} (the unit ids) has missing values.")
    groups = df.groupby(i_col, sort=False)
    varying = [
        col
        for col in unit_cols
        if (groups[col].nunique(dropna=False) > 1).any()
    ]
    if varying:
        raise ValueError(
            f"Column(s) {varying} vary within a unit: a unit's time, "
            "censoring, truncation and covariates must be the same on each "
            "of its rows (one row per reading)."
        )
    units = groups.first()
    t = df[marker_x_col].to_numpy(dtype=float)
    y = df[marker_y_col].to_numpy(dtype=float)
    none = np.isnan(t) & np.isnan(y)
    out: dict = {
        "i": units.index.to_numpy(),
        "marker_x": t[~none],
        "marker_y": y[~none],
        "marker_i": df[i_col].to_numpy()[~none],
    }
    if xl_col is not None or xr_col is not None:
        out["xl"] = units[xl_col].to_numpy(dtype=float)
        out["xr"] = units[xr_col].to_numpy(dtype=float)
    else:
        out["x"] = units[x_col].to_numpy(dtype=float)
    for key, col in (("c", c_col), ("tl", tl_col), ("tr", tr_col)):
        if col is not None:
            out[key] = units[col].to_numpy()
    if Z_cols:
        out["Z"] = units[Z_cols]
    if path_Z_cols:
        out["path_Z"] = units[path_Z_cols]
    return out


def _prediction_inputs(
    model: JointModel,
    marker_x: Any,
    marker_y: Any,
    Z: Any,
    path_Z: Any,
    age: "float | None",
) -> tuple:
    from surpyval.utils.item_history import ItemHistory

    t: npt.NDArray
    y: npt.NDArray
    if isinstance(marker_x, ItemHistory):
        item = marker_x
        markers = item.markers or {}
        t = np.array([float(a) for a in markers], dtype=np.float64)
        y = np.array([float(v) for v in markers.values()], dtype=np.float64)
        if Z is None and item.covariates is not None and model.n_Z:
            Z = item.covariates
    else:
        t = np.atleast_1d(np.asarray(marker_x, dtype=float)).ravel()
        if marker_y is None:
            raise ValueError("marker_y is required with marker_x.")
        y = np.atleast_1d(np.asarray(marker_y, dtype=float)).ravel()
    if t.size != y.size:
        raise ValueError(
            "marker_x and marker_y must have the same length; got "
            f"{t.size} and {y.size}."
        )
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(y))):
        raise ValueError(
            "The readings must be finite: drop a missing reading before "
            "predicting."
        )
    if np.any(t < 0):
        raise ValueError("marker_x must not be negative: they are ages.")
    order = np.argsort(t, kind="stable")
    t, y = t[order], y[order]
    last = float(t[-1]) if t.size else 0.0
    now = last if age is None else float(age)
    if not np.isfinite(now) or now < 0:
        raise ValueError(f"age must be a finite age, at least 0; got {age!r}.")
    if now < last:
        raise ValueError(
            f"age ({now:g}) is before the last reading ({last:g}): the "
            "prediction is made at or after the readings it uses."
        )
    z = _rows_for("Z", Z, model.n_Z, 1)
    w = _rows_for("path_Z", path_Z, model.n_path_Z, 1)
    if z.shape[0] != 1 or w.shape[0] != 1:
        raise ValueError(
            "predict is for one unit: pass Z and path_Z as single rows."
        )
    if np.isnan(z).any() or np.isnan(w).any():
        raise ValueError(
            "The unit's covariates must not be missing (nan) for a "
            "prediction."
        )
    return t, y, z[0], w[0], now


def _simulate(
    model: JointModel,
    n_units: int,
    reading_interval: float,
    end: float,
    Z: Any,
    path_Z: Any,
    random_state: Any,
) -> pd.DataFrame:
    if not (reading_interval > 0 and end > 0):
        raise ValueError("reading_interval and end must be positive.")
    n = int(n_units)
    rng = as_generator(random_state)
    z = _rows_for("Z", Z, model.n_Z, n)
    w = _rows_for("path_Z", path_Z, model.n_path_Z, n)
    z = np.broadcast_to(z, (n, model.n_Z))
    w = np.broadcast_to(w, (n, model.n_path_Z))
    times, U = draw_units(
        model._evaluator(),
        np.ascontiguousarray(z),
        np.ascontiguousarray(w),
        rng,
    )
    x = np.minimum(times, end)
    c = (times > end).astype(int)
    sigma = float(model.params[model.parameter_names.index("sigma")])
    rows = []
    for u in range(n):
        ages = np.arange(0.0, x[u], reading_interval)
        if ages.size == 0:
            ages = np.array([0.0])
        m = U[u, 0] + U[u, 1] * ages + model.marker_centre
        y = m + rng.normal(0.0, sigma, ages.size)
        frame = {
            "i": np.full(ages.size, u),
            "x": np.full(ages.size, x[u]),
            "c": np.full(ages.size, c[u]),
            "marker_x": ages,
            "marker_y": y,
        }
        for j, name in enumerate(model.Z_cols or model._z_names):
            frame[name] = np.full(ages.size, z[u, j])
        for j, name in enumerate(model.path_Z_cols or model._w_names):
            frame["path_" + name] = np.full(ages.size, w[u, j])
        rows.append(pd.DataFrame(frame))
    return pd.concat(rows, ignore_index=True)


def _summary(model: JointModel, alpha_ci: float) -> pd.DataFrame:
    from scipy.stats import norm

    names = model.parameter_names
    try:
        se_kept = model.standard_errors()
    except ValueError:
        se_kept = np.full(len(model.covariance_names), np.nan)
    se = np.full(len(names), np.nan)
    se[np.isfinite(model.params)] = se_kept
    level = "{:g}%".format(100 * (1 - alpha_ci))
    rows, index = [], []
    for j, name in enumerate(names):
        value = model.params[j]
        lo = hi = np.nan
        if np.isfinite(se[j]):
            import warnings

            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                lo, hi = model.param_cb(name, alpha_ci)
        row = {
            "coef": value,
            "se(coef)": se[j],
            "coef lower " + level: lo,
            "coef upper " + level: hi,
        }
        if j == 0 or j == 1:
            part = "survival"
        elif j < 2 + model.n_Z:
            part = "survival"
        elif name == "association":
            part = "association"
        elif name.startswith("path_"):
            part = "path"
        elif name == "sigma":
            part = "path"
        else:
            part = "random effects"
        if 2 <= j <= 2 + model.n_Z:
            with np.errstate(all="ignore"):
                row["exp(coef)"] = np.exp(value)
                row["z"] = value / se[j]
                row["p"] = 2 * norm.sf(abs(value / se[j]))
        rows.append(row)
        index.append((part, name))
    columns = [
        "coef",
        "exp(coef)",
        "se(coef)",
        "coef lower " + level,
        "coef upper " + level,
        "z",
        "p",
    ]
    return pd.DataFrame(
        rows,
        columns=columns,
        index=pd.MultiIndex.from_tuples(index, names=["part", "name"]),
    )


def _repr(model: JointModel) -> str:
    lines = [
        "Joint Model (longitudinal marker and failure time)",
        "==================================================",
        "Survival part       : Weibull proportional hazards",
        "Link                : current value of the marker",
        "Path                : linear, random intercept and slope",
        "Marker centre       : {:.6g}".format(model.marker_centre),
    ]
    if model.data is not None:
        lines.append(
            "Units               : {} ({} failed), {} readings".format(
                model.data.n,
                observed_failures(model.data),
                model.data.n_readings,
            )
        )
    lines.append("Parameters          :")
    try:
        se = dict(zip(model.covariance_names, model.standard_errors()))
    except ValueError:
        se = {}
    width = max(len(n) for n in model.parameter_names)
    for name, value in zip(model.parameter_names, model.params):
        text = "    {}  {:>12.6g}".format(name.ljust(width), value)
        if name in se:
            text += "  (se {:.4g})".format(se[name])
        lines.append(text)
    if model._neg_ll is not None:
        lines.append(
            "Log-likelihood      : {:.6g} (maximum: {})".format(
                -model._neg_ll, model.maximum
            )
        )
    return "\n".join(lines)


def _from_items(items: Any, path_covariates: Any) -> dict:
    """The arguments of ``fit`` from items' histories."""
    from surpyval.utils.event_history_data import EventHistoryData
    from surpyval.utils.item_history import ItemHistory, covariate_names_of

    if isinstance(items, EventHistoryData):
        items = items.to_items()
    items = list(items)
    if not items or not all(isinstance(h, ItemHistory) for h in items):
        raise ValueError(
            "items must be a non-empty list of ItemHistory objects (or an "
            "EventHistoryData made from them)."
        )
    ids, x, c, tl, tr = [], [], [], [], []
    t_all, y_all, i_all = [], [], []
    for h in items:
        _check_joint_item(h)
        ids.append(h.id)
        lo, hi, flag = _item_event(h)
        x.append([lo, hi])
        c.append(flag)
        tl.append(0.0 if h.left_truncation is None else h.left_truncation)
        tr.append(np.inf if h.right_truncation is None else h.right_truncation)
        for age, value in (h.markers or {}).items():
            if isinstance(value, tuple):
                raise ValueError(
                    f"Item {h.id!r} has several markers per reading; the "
                    "joint model of v1 takes one marker (several markers "
                    "with a latent health index are planned, #891)."
                )
            t_all.append(age)
            y_all.append(value)
            i_all.append(h.id)
    out: dict = {
        "x": np.asarray(x, dtype=float),
        "c": np.asarray(c),
        "i": np.asarray(ids, dtype=object),
        "tl": np.asarray(tl),
        "tr": np.asarray(tr),
        "marker_x": np.asarray(t_all, dtype=float),
        "marker_y": np.asarray(y_all, dtype=float),
        "marker_i": np.asarray(i_all, dtype=object),
    }
    with_cov = [h.covariates is not None for h in items]
    if any(with_cov):
        if not all(with_cov):
            raise ValueError(
                "Covariates were given for some items but not all: every "
                "unit needs its covariates."
            )
        names = covariate_names_of(items)
        rows = np.asarray([h.covariates for h in items], dtype=float)
        cols = names or [f"coef_{j}" for j in range(rows.shape[1])]
        Z = pd.DataFrame(rows, columns=cols)
        out["Z"] = Z
        if path_covariates is True:
            out["path_Z"] = Z.copy()
        elif path_covariates:
            picked = [
                cols[k] if isinstance(k, (int, np.integer)) else k
                for k in path_covariates
            ]
            unknown = [k for k in picked if k not in cols]
            if unknown:
                raise ValueError(
                    f"path_covariates names {unknown}, which are not "
                    f"covariates of the items ({cols})."
                )
            out["path_Z"] = Z[picked]
    elif path_covariates:
        raise ValueError("path_covariates was given, but the items have none.")
    return out


def _check_joint_item(h: Any) -> None:
    if h.time_varying_covariates is not None or h.has_states:
        raise ValueError(
            f"Item {h.id!r} has time-varying covariates or states, which "
            "the joint model does not use: data.without(...) leaves them "
            "out."
        )
    if h.has_event_types:
        raise ValueError(
            f"Item {h.id!r} has event types: the joint model has one "
            "failure mode. Fit the failures of one type "
            "(data.for_event_type(t))."
        )
    failures = len(h.observed) + int(np.sum(h.interval_censored_counts))
    failures += 0 if h.left_censored is None else h.left_censored_counts
    if failures > 1:
        raise ValueError(
            f"Item {h.id!r} has {failures} failures: a joint model has one "
            "failure per unit (its first: data.to_first_events(), or each "
            "life as a unit)."
        )


def _item_event(h: Any) -> tuple[float, float, int]:
    """``(lower, upper, c)``: an item's failure, or its end."""
    if len(h.observed):
        age = float(h.observed[0])
        return age, age, 0
    if len(h.interval_censored):
        lo, hi = (float(v) for v in h.interval_censored[0])
        return lo, hi, 2
    if h.left_censored is not None:
        return float(h.left_censored), float(h.left_censored), -1
    end = h._end()
    if end is None:
        raise ValueError(
            f"Item {h.id!r} has no failure and no end of observation."
        )
    return float(end), float(end), 1
