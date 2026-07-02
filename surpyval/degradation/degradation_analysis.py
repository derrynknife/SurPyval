"""Degradation analysis.

Classic (pseudo-failure-time) degradation analysis: a degradation
measurement is tracked over time on each unit, a
:class:`~surpyval.degradation.PathModel` is fitted to each unit's
measurements, each fitted path is extrapolated to the failure threshold
to get that unit's pseudo failure time, and a lifetime distribution is
fitted to the pseudo failure times. Units whose fitted path never
reaches the threshold are treated as right censored at their last
observed time.
"""

import warnings
from numbers import Number

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from surpyval.univariate.parametric import Weibull
from surpyval.univariate.parametric.parametric import Parametric

from .path_models import PathModel, get_path_model


def _clip_psd(matrix: npt.NDArray) -> tuple[npt.NDArray, bool]:
    """
    Project a symmetric matrix onto the positive semi-definite cone
    by clipping negative eigenvalues to zero.

    Returns the projected matrix and whether any eigenvalue was
    *materially* negative (beyond floating-point noise).
    """
    matrix = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(matrix)
    tol = 1e-10 * max(np.abs(eigvals).max(), np.finfo(float).tiny)
    clipped = bool((eigvals < -tol).any())
    eigvals = np.clip(eigvals, 0.0, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T, clipped


class DegradationModel:
    """
    A fitted degradation analysis model.

    This is the model object returned by
    :meth:`DegradationAnalysis.fit`. It holds the per-unit fitted
    degradation paths, the pseudo failure times extrapolated from them,
    and the lifetime distribution fitted to those pseudo failure times.
    The usual lifetime functions (``sf``, ``ff``, ``df``, ``hf``,
    ``Hf``, ``qf``, ``mean``, ``random``) are forwarded to the fitted
    life model, and the failure time of a new, partially observed unit
    can be estimated from its trajectory with
    :meth:`predict_failure_time` / :meth:`predict_remaining_life`.

    Parameters
    ----------
    x, y, i : ndarray
        The degradation data: measurement times, measurements, and the
        unit each measurement belongs to.
    units : ndarray
        The distinct unit identifiers.
    threshold : float
        The degradation level at which a unit is considered failed.
    path_model : PathModel
        The degradation path model fitted to each unit.
    path_params : ndarray
        Per-unit fitted path parameters, one row per entry of
        ``units``.
    pseudo_failure_times : ndarray
        Per-unit pseudo failure time: the extrapolated threshold
        crossing time, or the unit's last observed time for censored
        units.
    c : ndarray
        Per-unit censor flags: 0 where the fitted path crosses the
        threshold, 1 (right censored) where it never does.
    life_model : Parametric
        The lifetime distribution fitted to the pseudo failure times.
    measurement_var : float
        Pooled estimate of the measurement-error variance around the
        per-unit paths (the per-unit residual sums of squares over the
        total residual degrees of freedom). Zero when every unit has
        exactly as many measurements as path parameters.
    path_param_mean : ndarray
        Mean of the per-unit fitted path parameters: the estimated
        population mean path.
    path_param_cov : ndarray
        Noise-corrected estimate of the *between-unit* covariance of
        the true path parameters (Lu-Meeker two-stage): the sample
        covariance of the per-unit estimates minus the average
        least-squares estimation covariance, projected onto the
        positive semi-definite cone.
    path_param_sample_cov : ndarray
        The raw (uncorrected) sample covariance of the per-unit
        fitted path parameters. This overstates the between-unit
        variability because each per-unit estimate also carries
        least-squares estimation noise.
    """

    x: npt.NDArray
    y: npt.NDArray
    i: npt.NDArray
    units: npt.NDArray
    threshold: float
    path_model: PathModel
    path_params: npt.NDArray
    pseudo_failure_times: npt.NDArray
    c: npt.NDArray
    life_model: Parametric
    measurement_var: float
    path_param_mean: npt.NDArray
    path_param_cov: npt.NDArray
    path_param_sample_cov: npt.NDArray

    def __init__(
        self,
        x,
        y,
        i,
        units,
        threshold,
        path_model,
        path_params,
        pseudo_failure_times,
        c,
        life_model,
        measurement_var,
        path_param_mean,
        path_param_cov,
        path_param_sample_cov,
    ):
        self.x = x
        self.y = y
        self.i = i
        self.units = units
        self.threshold = threshold
        self.path_model = path_model
        self.path_params = path_params
        self.pseudo_failure_times = pseudo_failure_times
        self.c = c
        self.life_model = life_model
        self.measurement_var = measurement_var
        self.path_param_mean = path_param_mean
        self.path_param_cov = path_param_cov
        self.path_param_sample_cov = path_param_sample_cov
        self._unit_index = {unit: idx for idx, unit in enumerate(units)}

    def path(self, x: npt.ArrayLike, unit) -> npt.NDArray:
        """Evaluate the fitted degradation path of ``unit`` at ``x``."""
        idx = self._unit_index[unit]
        return self.path_model.path(x, *self.path_params[idx])

    def predict_failure_time(
        self, x: npt.ArrayLike, y: npt.ArrayLike
    ) -> float:
        """
        Estimate the failure time of a new unit from its (partial)
        degradation trajectory.

        Fits this model's path model to the new unit's measurements
        and extrapolates the fitted path to this model's failure
        threshold, exactly as was done for each unit during fitting.

        Parameters
        ----------
        x : array like
            Times at which the new unit's measurements were taken.
        y : array like
            The new unit's degradation measurements.

        Returns
        -------
        float
            The time at which the new unit's fitted path reaches the
            threshold. This can be smaller than the last observed time
            if the trajectory has already crossed the threshold.
            Returns ``nan`` (with a warning) if the fitted path never
            reaches the threshold.
        """
        x_arr, y_arr = self._handle_new_trajectory(x, y)
        params = self.path_model.fit(x_arr, y_arr)
        t = float(self.path_model.inv_path(self.threshold, *params))
        if not (np.isfinite(t) and t > 0):
            warnings.warn(
                "The fitted degradation path of the new trajectory never "
                "reaches the threshold {}; returning nan".format(
                    self.threshold
                ),
                stacklevel=2,
            )
            return float("nan")
        return t

    def predict_remaining_life(
        self, x: npt.ArrayLike, y: npt.ArrayLike
    ) -> float:
        """
        Estimate the remaining life of a new unit from its (partial)
        degradation trajectory.

        This is :meth:`predict_failure_time` minus the new unit's last
        observed time. A negative value means the fitted path crossed
        the threshold before the last observation (the unit is
        predicted to have already failed); ``nan`` (with a warning)
        means the fitted path never reaches the threshold.
        """
        x_arr, y_arr = self._handle_new_trajectory(x, y)
        return self.predict_failure_time(x_arr, y_arr) - float(x_arr.max())

    def _handle_new_trajectory(
        self, x: npt.ArrayLike, y: npt.ArrayLike
    ) -> tuple[npt.NDArray, npt.NDArray]:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be one dimensional")
        if len(x) != len(y):
            raise ValueError(
                "x and y must have the same length; got {} and {}".format(
                    len(x), len(y)
                )
            )
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError("x and y must contain only finite values")
        n_params = len(self.path_model.param_names)
        if len(x) < n_params or len(np.unique(x)) < 2:
            raise ValueError(
                "The trajectory needs at least {} measurements at 2 or "
                "more distinct times to fit the {} path model".format(
                    n_params, self.path_model.name
                )
            )
        return x, y

    def sf(self, x: npt.ArrayLike) -> npt.NDArray:
        """Survival function of the fitted life model."""
        return self.life_model.sf(x)

    def ff(self, x: npt.ArrayLike) -> npt.NDArray:
        """CDF of the fitted life model."""
        return self.life_model.ff(x)

    def df(self, x: npt.ArrayLike) -> npt.NDArray:
        """Density of the fitted life model."""
        return self.life_model.df(x)

    def hf(self, x: npt.ArrayLike) -> npt.NDArray:
        """Hazard rate of the fitted life model."""
        return self.life_model.hf(x)

    def Hf(self, x: npt.ArrayLike) -> npt.NDArray:
        """Cumulative hazard of the fitted life model."""
        return self.life_model.Hf(x)

    def qf(self, p: npt.ArrayLike) -> npt.NDArray:
        """Quantile function of the fitted life model."""
        return self.life_model.qf(p)

    def mean(self) -> float:
        """Mean of the fitted life model."""
        return self.life_model.mean()

    def random(self, size: int) -> npt.NDArray:
        """Random pseudo failure times from the fitted life model."""
        return self.life_model.random(size)

    def plot(self, ax=None):
        """
        Plot the degradation data, the fitted per-unit paths (extended
        to each unit's pseudo failure time), and the failure threshold.

        Parameters
        ----------
        ax : matplotlib axes, optional
            An axes object to draw the plot on. Creates a new one if
            not provided.

        Returns
        -------
        matplotlib axes
            An axes object with the plot.
        """
        if ax is None:
            ax = plt.gcf().gca()

        for idx, unit in enumerate(self.units):
            mask = self.i == unit
            x_unit = self.x[mask]
            y_unit = self.y[mask]
            start, end = x_unit.min(), x_unit.max()
            if self.c[idx] == 0:
                pseudo = self.pseudo_failure_times[idx]
                start, end = min(start, pseudo), max(end, pseudo)
            x_plot = np.linspace(start, end, 200)
            (line,) = ax.plot(
                x_plot,
                self.path_model.path(x_plot, *self.path_params[idx]),
                linewidth=1,
                alpha=0.8,
            )
            ax.scatter(x_unit, y_unit, s=12, color=line.get_color())

        ax.axhline(
            self.threshold, color="k", linestyle="--", label="Threshold"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Degradation")
        ax.legend()
        return ax

    def __repr__(self):
        param_string = "\n".join(
            [
                f"{name:>10}: {p}"
                for p, name in zip(
                    self.life_model.params, self.life_model.dist.param_names
                )
            ]
        )
        return (
            "Degradation Analysis SurPyval Model"
            "\n==================================="
            f"\nPath Model          : {self.path_model.name}"
            f"\nThreshold           : {self.threshold}"
            f"\nNumber of Units     : {len(self.units)}"
            f"\nCensored Units      : {int((self.c == 1).sum())}"
            f"\nLife Distribution   : {self.life_model.dist.name}"
            "\nParameters          :\n" + param_string
        )


class DegradationAnalysis_:
    """
    Pseudo-failure-time degradation analysis.

    Fits a degradation path model to each unit's measurements,
    extrapolates each fitted path to the failure ``threshold`` to
    obtain per-unit pseudo failure times, and fits a lifetime
    distribution to those times. Units whose fitted path never reaches
    the threshold at a positive finite time are right censored at their
    last observed time (with a warning).

    Examples
    --------

    >>> import numpy as np
    >>> from surpyval.degradation import DegradationAnalysis
    >>> # 4 units measured every 100 hours; degradation grows linearly
    >>> # at a different rate per unit; failure is defined at level 150.
    >>> x = np.tile(np.arange(100, 1100, 100), 4)
    >>> slopes = np.repeat([0.31, 0.28, 0.44, 0.37], 10)
    >>> i = np.repeat([1, 2, 3, 4], 10)
    >>> y = 10 + slopes * x
    >>> model = DegradationAnalysis.fit(x, y, i, threshold=150)
    >>> print(model)
    Degradation Analysis SurPyval Model
    ===================================
    Path Model          : Linear
    Threshold           : 150.0
    Number of Units     : 4
    Censored Units      : 0
    Life Distribution   : Weibull
    Parameters          :
         alpha: 441.4780882117898
          beta: 6.987078993008337
    >>> model.pseudo_failure_times
    array([451.61290323, 500.        , 318.18181818, 378.37837838])
    """

    def fit(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        i: npt.ArrayLike,
        threshold: float,
        path: "str | PathModel" = "linear",
        distribution=Weibull,
        how: str = "MLE",
    ) -> DegradationModel:
        """
        Fit a degradation analysis model.

        Parameters
        ----------
        x : array like
            Times at which the degradation measurements were taken.
        y : array like
            The degradation measurements.
        i : array like
            The unit each measurement belongs to. Must have the same
            length as ``x`` and ``y``.
        threshold : float
            The degradation level at which a unit is defined to have
            failed.
        path : str or PathModel, optional
            The degradation path model fitted to each unit: one of
            ``"linear"`` (default), ``"exponential"``, ``"power"``,
            ``"logarithmic"``, ``"lloyd-lipow"``, or a
            :class:`~surpyval.degradation.PathModel` instance.
        distribution : ParametricFitter, optional
            The lifetime distribution fitted to the pseudo failure
            times. Defaults to ``Weibull``.
        how : str, optional
            The method used to fit the lifetime distribution (passed
            to ``distribution.fit``). Defaults to ``"MLE"``.

        Returns
        -------
        DegradationModel
            The fitted degradation model, with the per-unit paths,
            pseudo failure times, and the fitted life model.
        """
        x_arr, y_arr, i_arr = self._handle_xyi(x, y, i)

        if not isinstance(threshold, Number) or not np.isfinite(threshold):
            raise ValueError("threshold must be a finite number")
        threshold = float(threshold)

        path_model = get_path_model(path)

        units = np.unique(i_arr)
        if len(units) < 2:
            raise ValueError(
                "Degradation analysis requires at least 2 units; "
                "got {}".format(len(units))
            )

        n_params = len(path_model.param_names)
        path_params = np.empty((len(units), n_params))
        pseudo = np.empty(len(units))
        last_time = np.empty(len(units))
        rss_total = 0.0
        dof_total = 0
        estimation_cov_sum = np.zeros((n_params, n_params))

        for idx, unit in enumerate(units):
            mask = i_arr == unit
            x_unit, y_unit = x_arr[mask], y_arr[mask]
            if len(x_unit) < n_params or len(np.unique(x_unit)) < 2:
                raise ValueError(
                    "Unit {} needs at least {} measurements at 2 or more "
                    "distinct times to fit the {} path model".format(
                        unit, n_params, path_model.name
                    )
                )
            params = path_model.fit(x_unit, y_unit)
            path_params[idx] = params
            pseudo[idx] = path_model.inv_path(threshold, *params)
            last_time[idx] = x_unit.max()

            residuals = y_unit - path_model.path(x_unit, *params)
            rss_total += residuals @ residuals
            dof_total += len(x_unit) - n_params
            jacobian = path_model.jacobian(x_unit, *params)
            jtj = jacobian.T @ jacobian
            try:
                estimation_cov_sum += np.linalg.inv(jtj)
            except np.linalg.LinAlgError:
                estimation_cov_sum += np.linalg.pinv(jtj)

        # Two-stage (Lu-Meeker) noise correction: the scatter of the
        # per-unit estimates is Sigma + V_i, so subtracting the average
        # estimation covariance leaves the between-unit covariance.
        measurement_var = rss_total / dof_total if dof_total > 0 else 0.0
        path_param_mean = path_params.mean(axis=0)
        path_param_sample_cov = np.atleast_2d(
            np.cov(path_params, rowvar=False, ddof=1)
        )
        mean_estimation_cov = measurement_var * estimation_cov_sum / len(units)
        path_param_cov, was_clipped = _clip_psd(
            path_param_sample_cov - mean_estimation_cov
        )
        if was_clipped:
            warnings.warn(
                "The noise-corrected between-unit covariance of the path "
                "parameters was not positive semi-definite (the estimation "
                "noise is comparable to the between-unit scatter); negative "
                "eigenvalues were clipped to zero. With this few units or "
                "measurements per unit, path_param_cov is unreliable",
                stacklevel=2,
            )

        events = np.isfinite(pseudo) & (pseudo > 0)
        if not events.any():
            raise ValueError(
                "No unit's fitted degradation path reaches the threshold "
                "{}; check the threshold and the path model".format(threshold)
            )
        if not events.all():
            warnings.warn(
                "The fitted degradation path(s) of unit(s) {} never reach "
                "the threshold {}; these units are treated as right "
                "censored at their last observed time".format(
                    list(units[~events]), threshold
                ),
                stacklevel=2,
            )

        pseudo_failure_times = np.where(events, pseudo, last_time)
        c = np.where(events, 0, 1)

        life_model = distribution.fit(x=pseudo_failure_times, c=c, how=how)

        return DegradationModel(
            x=x_arr,
            y=y_arr,
            i=i_arr,
            units=units,
            threshold=threshold,
            path_model=path_model,
            path_params=path_params,
            pseudo_failure_times=pseudo_failure_times,
            c=c,
            life_model=life_model,
            measurement_var=measurement_var,
            path_param_mean=path_param_mean,
            path_param_cov=path_param_cov,
            path_param_sample_cov=path_param_sample_cov,
        )

    def fit_from_df(
        self,
        df: pd.DataFrame,
        x: str = "x",
        y: str = "y",
        i: str = "i",
        **fit_kwargs,
    ) -> DegradationModel:
        """
        Fit a degradation analysis model from a DataFrame.

        Parameters
        ----------
        df : DataFrame
            DataFrame with the degradation data.
        x : str, optional
            Column of the measurement times. Defaults to ``"x"``.
        y : str, optional
            Column of the degradation measurements. Defaults to
            ``"y"``.
        i : str, optional
            Column of the unit identifiers. Defaults to ``"i"``.
        **fit_kwargs
            Remaining arguments (``threshold``, ``path``,
            ``distribution``, ``how``) passed to :meth:`fit`.

        Returns
        -------
        DegradationModel
            The fitted degradation model.
        """
        return self.fit(
            df[x].to_numpy(), df[y].to_numpy(), df[i].to_numpy(), **fit_kwargs
        )

    @staticmethod
    def _handle_xyi(
        x: npt.ArrayLike, y: npt.ArrayLike, i: npt.ArrayLike
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        i = np.atleast_1d(np.asarray(i))
        if x.ndim != 1 or y.ndim != 1 or i.ndim != 1:
            raise ValueError("x, y, and i must be one dimensional")
        if not (len(x) == len(y) == len(i)):
            raise ValueError(
                "x, y, and i must have the same length; got {}, {}, "
                "and {}".format(len(x), len(y), len(i))
            )
        if len(x) == 0:
            raise ValueError("x, y, and i must not be empty")
        if not np.isfinite(x).all():
            raise ValueError("x must contain only finite values")
        if not np.isfinite(y).all():
            raise ValueError("y must contain only finite values")
        return x, y, i


DegradationAnalysis = DegradationAnalysis_()
