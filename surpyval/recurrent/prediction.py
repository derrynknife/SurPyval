"""Which model predicts best: one-step-ahead predictions of the next
failure, scored (Abdel-Ghaly, Chan and Littlewood 1986).

Models of a growing reliability often fit the failures seen equally well
and predict very differently, so a fit to the past does not choose
between them. What does is how each would have predicted, had it been
used all along: at each failure, fit the model to the failures before
it, take its distribution of the time to the next one, and see where
the time that came falls in it.

- The *prequential likelihood* is the product of the predictive
  densities at the times that came: the higher, the better the
  predictions. The ratio of two models' is how much more one was right
  than the other.
- ``u`` is the predictive distribution function at the time that came.
  Were the predictions right, the ``u`` would be uniform on (0, 1): the
  *u-plot* is their empirical distribution against the diagonal, and its
  Kolmogorov distance measures the bias. ``u`` mostly small says the
  times came sooner than predicted: the predictions were too optimistic.
- The *y-plot* looks for a bias that changes with time, which the u-plot
  averages out: with ``x = -log(1 - u)`` (exponential, were the
  predictions right), ``y_k`` is the share of the total of ``x`` reached
  by the ``k``-th prediction, uniform in order.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import kstwo

from surpyval import plotting
from surpyval.utils.deprecation import REMOVED_IN_NEXT, renamed_arguments
from surpyval.utils.validation import check_option
from surpyval.utils.warnings import ignoring_warnings

#: The first prediction is made from this many failures unless ``start``
#: says otherwise.
DEFAULT_START = 10


def _fitter_name(fitter: Any) -> str:
    """The public name of a fitter (``GoelOkumoto``), for the table."""
    if hasattr(fitter, "to_spec") and hasattr(fitter, "base"):
        # A model on another clock: "EffortClock(GoelOkumoto)"
        return fitter._repr_name()
    return type(fitter).__name__.rstrip("_")


def _parameter_count(fitter: Any) -> int:
    return len(getattr(fitter, "parameter_names", ())) or 1


def _next_gap(model: Any, age: float, gap: float) -> tuple[float, float]:
    """``(u, log density)`` of the gap that came, in the model's
    distribution of the time from ``age`` to the next failure."""
    with np.errstate(all="ignore"):
        if hasattr(model, "_next_gap"):
            # a model of the gaps themselves, which says both
            sf, log_density = model._next_gap(gap)
        elif hasattr(model, "current_rate"):
            # a constant rate until the next failure (Jelinski-Moranda)
            rate = float(model.current_rate())
            sf = float(np.exp(-rate * gap))
            log_density = np.log(rate) - rate * gap if rate > 0 else -np.inf
        else:
            # a Poisson process: the hazard of the gap is the intensity
            sf = float(model.next_failure_sf(gap, age=[age])[0])
            log_density = float(np.log(model.iif(age + gap))) + np.log(sf)
    return 1.0 - float(sf), float(log_density)


def _kolmogorov(values: np.ndarray) -> tuple[float, int]:
    """The Kolmogorov distance of ``values`` from the uniform on (0, 1),
    and the sign of the largest gap (+1: the empirical distribution is
    above the diagonal there, the values are too small)."""
    v = np.sort(values)
    m = v.size
    above = np.max(np.arange(1, m + 1) / m - v)
    below = np.max(v - np.arange(0, m) / m)
    return float(max(above, below)), (1 if above >= below else -1)


class PredictionComparison:
    """
    The result of :func:`compare_predictions`: each model's one-step-ahead
    predictions of the failures of one system, and their scores.

    Attributes
    ----------
    names : list of str
        The models, in the order given.
    steps : numpy.ndarray
        The number of failures each prediction was made from: the
        prediction at step ``i`` is of failure ``i + 1``.
    gaps : numpy.ndarray
        The time from failure ``i`` to failure ``i + 1``, which each
        prediction is of.
    u : dict
        ``{name: array}``: each model's predictive distribution function
        at the gap that came, ``nan`` where it made no prediction.
    log_density : dict
        ``{name: array}``: the log of its predictive density there.
    status : dict
        ``{name: list}``: ``"ok"`` for each step predicted, and
        otherwise why not (the fit's refusal, in its own words).
    alpha : float
        The level of the u-plot's and y-plot's Kolmogorov tests.

    Examples
    --------
    >>> from surpyval.recurrent import (
    ...     CrowAMSAA, GoelOkumoto, MusaOkumoto, compare_predictions)
    >>> truth = GoelOkumoto.from_params([60, 0.03])
    >>> x = truth.time_terminated_simulation_data(120.0, random_state=3).x
    >>> result = compare_predictions(
    ...     x[:-1], [GoelOkumoto, MusaOkumoto, CrowAMSAA], start=15)
    >>> result.summary()[["predicted", "failed", "log_likelihood"]].round(2)
                 predicted  failed  log_likelihood
    GoelOkumoto         39       6          -67.64
    MusaOkumoto         39       6          -69.72
    CrowAMSAA           45       0          -75.43
    >>> result.best
    'GoelOkumoto'
    """

    def __init__(
        self,
        names: list[str],
        steps: np.ndarray,
        gaps: np.ndarray,
        u: dict,
        log_density: dict,
        status: dict,
        alpha: float,
        unit: str | None = None,
    ) -> None:
        self.names = names
        self.steps = steps
        self.gaps = gaps
        self.u = u
        self.log_density = log_density
        self.status = status
        self.alpha = alpha
        self.unit = unit

    # -- which steps ------------------------------------------------------

    def predicted(self, name: str) -> np.ndarray:
        """A mask of the steps at which ``name`` made a prediction."""
        self._check_name(name)
        return ~np.isnan(self.u[name])

    @property
    def common(self) -> np.ndarray:
        """A mask of the steps at which every model made a prediction:
        the ones their prequential likelihoods are compared on."""
        out = np.ones(self.steps.size, dtype=bool)
        for name in self.names:
            out &= self.predicted(name)
        return out

    def _check_name(self, name: str) -> None:
        if name not in self.names:
            raise ValueError(
                f"{name!r} is not one of the models compared: {self.names}"
            )

    # -- the scores ---------------------------------------------------------

    def prequential_log_likelihood(
        self, name: str, common: bool = True
    ) -> float:
        """
        The log of ``name``'s prequential likelihood: the sum of the logs
        of its predictive densities at the gaps that came.

        Parameters
        ----------
        name : str
            The model.
        common : bool, optional
            Over the steps every model predicted (default), so that the
            models' values compare; ``False`` for all of its own steps.

        Returns
        -------
        float
            ``-inf`` where the model gave a gap that came no chance at
            all (a Jelinski-Moranda fit with no fault left, say).
        """
        mask = self.common if common else self.predicted(name)
        return float(np.sum(self.log_density[name][mask]))

    def likelihood_ratio(self, a: str, b: str) -> np.ndarray:
        """
        The log of the prequential likelihood ratio of model ``a`` to
        model ``b`` as the predictions accumulate: at each step both
        predicted, the running sum of ``log density_a - log density_b``.
        A curve that keeps rising says ``a`` predicts better throughout;
        its last value is the difference of the two models'
        :meth:`prequential_log_likelihood`.

        Returns
        -------
        numpy.ndarray
            One value per step both models predicted (those steps are
            ``steps[predicted(a) & predicted(b)]``).
        """
        mask = self.predicted(a) & self.predicted(b)
        with np.errstate(invalid="ignore"):
            return np.cumsum(
                self.log_density[a][mask] - self.log_density[b][mask]
            )

    def u_plot(self, name: str) -> tuple[float, str, bool]:
        """
        The u-plot of ``name`` in numbers: ``(distance, bias, beyond)``.

        ``distance`` is the Kolmogorov distance of its ``u`` from the
        uniform; ``bias`` is ``"optimistic"`` (the times came sooner
        than predicted: the largest gap has the ``u`` too small) or
        ``"pessimistic"`` (they came later); ``beyond`` says whether the
        distance is past the Kolmogorov test's critical value at
        ``alpha`` for that many predictions, that is, whether the bias
        is more than chance. (The test takes the ``u`` as independent,
        which they are only roughly: read ``beyond`` as a guide.)
        """
        values = self.u[name][self.predicted(name)]
        if values.size == 0:
            return float("nan"), "none", False
        distance, sign = _kolmogorov(values)
        critical = float(kstwo.ppf(1.0 - self.alpha, values.size))
        return (
            distance,
            "optimistic" if sign > 0 else "pessimistic",
            bool(distance > critical),
        )

    def _y(self, name: str) -> np.ndarray:
        values = self.u[name][self.predicted(name)]
        with np.errstate(divide="ignore"):
            x = -np.log1p(-values)
        if x.size < 2 or not np.all(np.isfinite(x)) or x.sum() <= 0:
            return np.array([])
        return np.cumsum(x)[:-1] / x.sum()

    def y_plot(self, name: str) -> tuple[float, bool]:
        """
        The y-plot of ``name`` in numbers: ``(distance, beyond)``, the
        Kolmogorov distance of its ``y`` from the uniform and whether it
        is past the critical value at ``alpha``: whether the
        predictions' bias changed as the failures went on (too
        optimistic early and too pessimistic late, say), which the
        u-plot cannot see. ``nan`` with fewer than three predictions,
        or where a ``u`` is exactly 1.
        """
        y = self._y(name)
        if y.size < 2:
            return float("nan"), False
        distance, _ = _kolmogorov(y)
        return distance, bool(distance > kstwo.ppf(1.0 - self.alpha, y.size))

    @property
    def best(self) -> str:
        """The model with the greatest prequential likelihood over the
        steps every model predicted (the first of any that tie)."""
        scores = [self.prequential_log_likelihood(n) for n in self.names]
        return self.names[int(np.argmax(scores))]

    def summary(self) -> Any:
        """
        One row per model, as a ``pandas.DataFrame``:

        - ``predicted`` and ``failed``: the steps it predicted, and
          those at which its fit gave no model (``status`` says why);
        - ``log_likelihood``: its prequential log-likelihood over the
          steps every model predicted;
        - ``u_distance``, ``bias``, ``biased``: its u-plot
          (:meth:`u_plot`);
        - ``y_distance``, ``drifting``: its y-plot (:meth:`y_plot`).
        """
        import pandas as pd

        rows = []
        for name in self.names:
            made = int(self.predicted(name).sum())
            u_distance, bias, biased = self.u_plot(name)
            y_distance, drifting = self.y_plot(name)
            rows.append(
                {
                    "predicted": made,
                    "failed": int(self.steps.size - made),
                    "log_likelihood": self.prequential_log_likelihood(name),
                    "u_distance": u_distance,
                    "bias": bias,
                    "biased": biased,
                    "y_distance": y_distance,
                    "drifting": drifting,
                }
            )
        return pd.DataFrame(rows, index=self.names)

    @property
    def verdict(self) -> str:
        """The comparison in words: which model predicted best and by
        how much, whose predictions were biased and which way, and which
        fits failed along the way."""
        shared = int(self.common.sum())
        lines = []
        if shared == 0:
            lines.append(
                "No step was predicted by every model, so their "
                "prequential likelihoods cannot be compared."
            )
        elif len(self.names) > 1:
            scores = {
                n: self.prequential_log_likelihood(n) for n in self.names
            }
            order = sorted(self.names, key=lambda n: -scores[n])
            best, second = order[0], order[1]
            gap = scores[best] - scores[second]
            if not np.isfinite(scores[best]):
                lines.append(
                    "Every model gave some failure that came no chance "
                    "at all, so none can be preferred."
                )
            elif gap < np.log(2.0):
                lines.append(
                    "{} and {} predicted about equally well over the {} "
                    "predictions every model made (prequential "
                    "log-likelihoods {:.2f} and {:.2f}: a ratio under 2 "
                    "to 1).".format(
                        best, second, shared, scores[best], scores[second]
                    )
                )
            else:
                ratio = (
                    "more than a million"
                    if gap > np.log(1e6)
                    else f"{np.exp(gap):.3g}"
                )
                lines.append(
                    "{} predicted best over the {} predictions every "
                    "model made: its prequential log-likelihood is "
                    "{:.2f}, against {:.2f} for {}, the next best (a "
                    "ratio of {} to 1).".format(
                        best,
                        shared,
                        scores[best],
                        scores[second],
                        second,
                        ratio,
                    )
                )
        for name in self.names:
            made = int(self.predicted(name).sum())
            failed = int(self.steps.size - made)
            if failed:
                first = next(s for s in self.status[name] if s != "ok")
                lines.append(
                    "{} gave no prediction at {} of the {} steps (its "
                    "fit was refused; the first: {})".format(
                        name, failed, self.steps.size, first
                    )
                )
            if made and not np.all(
                np.isfinite(self.log_density[name][self.predicted(name)])
            ):
                lines.append(
                    "{} gave a failure that came no chance at all (it "
                    "predicted that no more failures could "
                    "happen).".format(name)
                )
            distance, bias, biased = self.u_plot(name)
            if biased:
                came = "sooner" if bias == "optimistic" else "later"
                lines.append(
                    "{}'s predictions were consistently too {}: the "
                    "failures came {} than it predicted (u-plot distance "
                    "{:.2f}, beyond chance at the {:g} level).".format(
                        name, bias, came, distance, self.alpha
                    )
                )
            y_distance, drifting = self.y_plot(name)
            if drifting:
                lines.append(
                    "{}'s bias changed as the failures went on (y-plot "
                    "distance {:.2f}, beyond chance at the {:g} "
                    "level).".format(name, y_distance, self.alpha)
                )
        if len(lines) == (1 if len(self.names) > 1 and shared else 0):
            lines.append(
                "No model's predictions show a bias beyond chance (the "
                "u-plots and y-plots are within their {:g} "
                "bands).".format(self.alpha)
            )
        return " ".join(lines)

    def __repr__(self) -> str:
        table = self.summary()
        table = table.assign(
            log_likelihood=table["log_likelihood"].round(3),
            u_distance=table["u_distance"].round(3),
            y_distance=table["y_distance"].round(3),
        )
        return (
            "One-step-ahead predictions of failures {} to {}\n"
            "{}\n{}\n\n{}".format(
                int(self.steps[0]) + 1,
                int(self.steps[-1]) + 1,
                "=" * 48,
                table.to_string(),
                self.verdict,
            )
        )

    def plot(self, kind: str = "u", ax: Any = None) -> Any:
        """
        Draw the u-plot, the y-plot or the prequential likelihood ratios.

        Parameters
        ----------
        kind : {'u', 'y', 'ratio'}, optional
            ``"u"`` (default): each model's ``u`` as an empirical
            distribution against the diagonal, with the Kolmogorov band
            at ``alpha`` for the fewest predictions made; a curve above
            the diagonal is optimistic. ``"y"``: the same for ``y``.
            ``"ratio"``: the log prequential likelihood ratio of each
            model to the first, step by step (above 0: it predicts
            better than the first).
        ax : matplotlib axes, optional
            The axes to draw on. A new one by default.

        Returns
        -------
        matplotlib axes
        """
        check_option("kind", kind, ("u", "y", "ratio"))
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gcf().gca()
        if kind == "ratio":
            base = self.names[0]
            for name in self.names[1:]:
                mask = self.predicted(name) & self.predicted(base)
                ax.step(
                    self.steps[mask] + 1,
                    self.likelihood_ratio(name, base),
                    where="post",
                    label=f"{name} / {base}",
                )
            ax.axhline(0.0, color="grey", linewidth=0.8)
            ax.set_xlabel("Failure predicted")
            ax.set_ylabel("Log prequential likelihood ratio")
        else:
            fewest = None
            for name in self.names:
                values = (
                    self.u[name][self.predicted(name)]
                    if kind == "u"
                    else self._y(name)
                )
                if values.size == 0:
                    continue
                v = np.sort(values)
                ax.step(
                    np.concatenate([[0.0], v, [1.0]]),
                    np.concatenate([[0.0], np.arange(1, v.size + 1), [v.size]])
                    / v.size,
                    where="post",
                    label=name,
                )
                fewest = v.size if fewest is None else min(fewest, v.size)
            ax.plot([0, 1], [0, 1], color="grey", linewidth=0.8)
            if fewest:
                band = float(kstwo.ppf(1.0 - self.alpha, fewest))
                grid = np.array([0.0, 1.0])
                for sign in (-1.0, 1.0):
                    ax.plot(
                        grid,
                        grid + sign * band,
                        color="grey",
                        linestyle=":",
                        linewidth=0.8,
                    )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel(kind)
            ax.set_ylabel("Share of predictions at or below")
        ax.set_title(
            {
                "u": "u-plot",
                "y": "y-plot",
                "ratio": "Prequential likelihood ratio",
            }[kind]
        )
        plotting.style_axes(ax)
        plotting.style_legend(ax)
        return ax


@renamed_arguments(REMOVED_IN_NEXT, FutureWarning, alpha="alpha_ci")
def compare_predictions(
    x: ArrayLike,
    models: Any,
    start: int | None = None,
    *,
    alpha_ci: float = 0.05,
) -> PredictionComparison:
    """
    Compare models of one system's failures by how well each would have
    predicted the next failure, had it been used all along (Littlewood's
    prequential comparison).

    For each failure from number ``start + 1`` on, every model is fitted
    to the failures before it (as a test ended at the last of them), and
    its distribution of the time to the next failure is scored at the
    time that came. See :class:`PredictionComparison` for the scores:
    the prequential likelihood (which model's predictions were best), the
    u-plot (were they biased, and which way) and the y-plot (did the
    bias change).

    Parameters
    ----------
    x : array_like
        One system's failure times, from the start of testing: every
        failure, at increasing times.
    models : fitter or list of fitters
        The models to compare: any of the Poisson-process fitters
        (``GoelOkumoto``, ``YamadaDelayedS``, ``InflectionS``,
        ``MusaOkumoto``, ``CrowAMSAA``, ``Duane``, ``CoxLewis``, ``HPP``)
        ``JelinskiMoranda`` and ``LittlewoodVerrall``. A fitted model
        stands for its fitter.
    start : int, optional
        The number of failures the first prediction is made from.
        Default 10; at least two more than the parameters of the
        largest model, and fewer than the failures.
    alpha_ci : float, optional
        The level of the Kolmogorov tests behind ``biased`` and
        ``drifting``. Default 0.05. Keyword only.

    Returns
    -------
    PredictionComparison
        Its ``repr`` is the table and the verdict.

    Notes
    -----
    A fit that is refused at some step (a growth model on failures that
    show no growth so far, for instance) makes no prediction there. The
    step is kept, with the reason, in ``status``; the likelihoods are
    compared on the steps every model predicted, and the table counts
    the failed steps. Nothing is left out silently.

    References
    ----------
    Abdel-Ghaly, A. A., Chan, P. Y. and Littlewood, B. (1986),
    "Evaluation of competing software reliability predictions", IEEE
    Transactions on Software Engineering SE-12(9), 950-967.

    Examples
    --------
    >>> from surpyval.recurrent import (
    ...     GoelOkumoto, MusaOkumoto, compare_predictions)
    >>> truth = GoelOkumoto.from_params([60, 0.03])
    >>> x = truth.time_terminated_simulation_data(120.0, random_state=3).x
    >>> result = compare_predictions(
    ...     x[:-1], [GoelOkumoto, MusaOkumoto], start=15)
    >>> result.best
    'GoelOkumoto'
    >>> round(result.prequential_log_likelihood("GoelOkumoto"), 2)
    -67.64
    >>> result.u_plot("GoelOkumoto")[1:]
    ('pessimistic', False)
    """
    times = np.asarray(x, dtype=float)
    if times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError(
            "x must be one system's failure times as a flat array with "
            "no missing values; got shape {}. For several systems, "
            "compare the predictions of each on its own.".format(times.shape)
        )
    if np.any(np.diff(times) < 0) or np.any(times <= 0):
        raise ValueError(
            "x must be the failure times in the order they happened, "
            "each above 0 (cumulative times from the start of testing, "
            "not the times between failures: use numpy.cumsum on those)."
        )
    fitters = list(models) if isinstance(models, (list, tuple)) else [models]
    fitters = [getattr(m, "dist", m) for m in fitters]
    if not fitters or not all(hasattr(f, "fit") for f in fitters):
        raise ValueError(
            "models must be a fitter or a list of fitters (GoelOkumoto, "
            "MusaOkumoto, CrowAMSAA, JelinskiMoranda, ...), each with a "
            "fit method."
        )
    names = [_fitter_name(f) for f in fitters]
    if len(set(names)) != len(names):
        raise ValueError(
            f"models has a fitter more than once ({names}); give each once."
        )
    if not (isinstance(alpha_ci, float) and 0 < alpha_ci < 1):
        raise ValueError(
            "alpha_ci is a level between 0 and 1, such as 0.05; got "
            f"{alpha_ci!r}"
        )
    least = max(_parameter_count(f) for f in fitters) + 2
    first = DEFAULT_START if start is None else start
    if not (isinstance(first, (int, np.integer)) and first >= least):
        raise ValueError(
            "start is the number of failures the first prediction is "
            "made from: a whole number of at least {} (two more than the "
            "parameters of the largest model compared); got "
            "{!r}".format(least, start)
        )
    if first >= times.size:
        raise ValueError(
            "With start = {} the first prediction is of failure {}, but "
            "x has {} failures. Give a smaller start (at least {}), or "
            "more failures.".format(first, first + 1, times.size, least)
        )
    steps = np.arange(first, times.size)
    gaps = times[steps] - times[steps - 1]
    u = {name: np.full(steps.size, np.nan) for name in names}
    log_density = {name: np.full(steps.size, np.nan) for name in names}
    status: dict = {name: [] for name in names}
    for name, fitter in zip(names, fitters):
        for k, i in enumerate(steps):
            try:
                with ignoring_warnings():
                    model = fitter.fit(times[:i])
            except ValueError as refusal:
                status[name].append(str(refusal).split(": ", 1)[0] + ".")
                continue
            u[name][k], log_density[name][k] = _next_gap(
                model, float(times[i - 1]), float(gaps[k])
            )
            status[name].append("ok")
    return PredictionComparison(
        names, steps, gaps, u, log_density, status, alpha_ci
    )
