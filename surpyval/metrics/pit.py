"""Is a model's predicted distribution right? The PIT and D-calibration.

A model that predicts a whole distribution for each case (a regression
model, a Cox model, a forest, a neural network giving each case its own
parameters) can be checked with one simple idea. For each case, read the
case's own predicted CDF at what actually happened: ``u_i = F_i(y_i)``,
the predicted probability of an outcome at or below the one seen. If the
predictions are right, these numbers are spread evenly between 0 and 1,
whatever the cases' covariates: a tenth of them below 0.1, a tenth between
0.1 and 0.2, and so on. Too many near 0 or 1 means the predicted spread is
too narrow; a pile in the middle means it is too wide; more on one side
than the other means the predictions are shifted. This is the probability
integral transform (PIT) histogram of Dawid (1984) and Gneiting, Balabdaoui
and Raftery (2007). It is the CDF *at* the outcome, not its inverse.

A case censored at ``t`` (still running when last seen) has not had its
outcome yet, so its ``u`` is only known to be above ``F_i(t)``. If the
model is right, its ``u`` is equally likely to be anywhere between
``F_i(t)`` and 1, so its weight is spread evenly over that range: each
bin gets the share of ``(F_i(t), 1]`` it covers. This is D-calibration
(Haider, Hoehn, Davis and Greiner, 2020), written there on the survival
scale (``S_i(t) = 1 - F_i(t)``, weight spread *below* ``S_i(t)``); the two
are the same histogram read from opposite ends. With no censoring it is
the plain PIT histogram.

The test: with ``B`` equal bins and ``N`` cases, each bin should hold
``N / B``; Pearson's statistic :math:`\\sum_b (O_b - N/B)^2 / (N/B)` is
compared with a chi-square on ``B - 1`` degrees of freedom, giving one
p-value for the model. A small p-value says the predictions are not
calibrated.

**With censoring this test is conservative.** Spreading a censored case's
weight puts its expected share in each bin rather than one random count,
so the bins vary less than the chi-square assumes and the p-value is too
large: a right model is rejected less often than the level says. In a
simulation of 400 data sets of 200 cases under the true model, the 5%
test rejected 5.25% of them with no censoring, 1% with 30% censored and
none with 82% censored. A model it rejects is badly off; one it passes
may still be off. ``randomise=True`` instead draws each censored case's
``u`` at random from its range (where it lies if the model is right),
and the test then has its stated level (5.25%, 5.25%, 3.75% and 4.75% at
no, 30%, 57% and 82% censoring in the same simulation), at the cost of a
result that depends on the draw (set ``seed``).

Things to keep in mind:

* **Calibration is not sharpness.** A model that predicts the same
  population curve for everyone is calibrated and useless. Pair this with
  a score of how informative the predictions are:
  :func:`~surpyval.metrics.score_predictions` (Brier score, AUC) or the
  concordance index.
* **The cases are assumed independent.** Repeated forecasts of one series
  (the same unit predicted at several ages) are not, and the p-value is
  then too small. Give each forecast's horizon with ``by=`` to see one
  histogram per horizon, each of independent cases.
* **Discrete outcomes or atoms.** Where the predicted CDF jumps at the
  outcome (a step-function model on the data it was fitted to, a
  probability mass at zero), ``u`` is not uniform even for a right model.
  ``randomise=True`` draws ``u`` uniformly between the CDF just below the
  outcome and at it (the randomised PIT), which is uniform for a right
  model.
* **Recalibration** (mapping predictions through the empirical PIT
  distribution to correct them) is not done here.

References
----------
Dawid, A. P. (1984), "Statistical theory: the prequential approach",
JRSS A 147, 278-292.

Gneiting, T., Balabdaoui, F. and Raftery, A. E. (2007), "Probabilistic
forecasts, calibration and sharpness", JRSS B 69, 243-268.

Haider, H., Hoehn, B., Davis, S. and Greiner, R. (2020), "Effective ways to
build and evaluate individual survival distributions", JMLR 21(85), 1-63.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from pandas import DataFrame
from scipy.stats import chi2 as _chi2

from surpyval.utils.validation import validate_1d as _as_1d

__all__ = ["DCalibration", "d_calibration", "pit_values"]


def _paired(out: Any, n: int, what: str) -> npt.NDArray:
    """A model's paired output as a vector of ``n`` values."""
    arr = np.asarray(out, dtype=float)
    if arr.ndim == 2 and arr.shape == (n, n):
        # A model returning every case at every time: case i at x_i.
        arr = np.diagonal(arr)
    arr = arr.ravel()
    if arr.shape[0] != n:
        raise ValueError(
            "model.{} returned {} values for {} cases; pass the predicted "
            "CDF at each outcome as F= instead".format(what, arr.shape[0], n)
        )
    return arr


def pit_values(
    model: Any, x: npt.ArrayLike, Z: npt.ArrayLike | None = None
) -> npt.NDArray:
    """Each case's predicted CDF at its own time, ``F_i(x_i)``.

    Parameters
    ----------
    model : object
        A fitted model with a CDF. Without ``Z``, any model with ``ff(x)``
        (a fitted distribution, a non-parametric estimate). With ``Z``,
        any model whose ``sf(x, Z)`` pairs ``x`` with the rows of ``Z``
        (the regression families, ``CoxPH``, ``AdditiveHazards``, the
        ``surpyval.ml`` tree and forest, joint models).
    x : array_like
        Each case's time (its outcome, or the time it was censored).
    Z : array_like or pandas.DataFrame, optional
        Each case's covariates, one row per case.

    Returns
    -------
    ndarray
        ``F_i(x_i)``, one per case.

    Examples
    --------
    >>> from surpyval import Weibull
    >>> from surpyval.metrics import pit_values
    >>> model = Weibull.from_params([10.0, 2.0])
    >>> pit_values(model, [5.0, 10.0]).round(4)
    array([0.2212, 0.6321])
    """
    x_arr = _as_1d(x, "x")
    n = x_arr.shape[0]
    if Z is None:
        return _paired(model.ff(x_arr), n, "ff")
    if not isinstance(Z, DataFrame):
        Z_arr = np.asarray(Z, dtype=float)
        if Z_arr.ndim == 1:
            Z_arr = Z_arr.reshape(-1, 1)
        if Z_arr.shape[0] != n:
            raise ValueError(
                f"'Z' has {Z_arr.shape[0]} rows for {n} cases in 'x'"
            )
        Z = Z_arr
    elif len(Z) != n:
        raise ValueError(f"'Z' has {len(Z)} rows for {n} cases in 'x'")
    return 1.0 - _paired(model.sf(x_arr, Z), n, "sf")


def _left_pit(model: Any, x: npt.NDArray, Z: Any) -> npt.NDArray:
    """The predicted CDF just below each time, ``F_i(x_i-)``."""
    below = np.nextafter(x, -np.inf)
    out = pit_values(model, np.where(x > 0, below, x), Z)
    return np.where(x > 0, out, 0.0)


def _spread(
    u: npt.NDArray, c: npt.NDArray, n: npt.NDArray, edges: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """The weight each bin receives from the events and the censored.

    An event puts its weight ``n_i`` in the bin holding ``u_i`` (the last
    bin includes 1). A case censored at ``u_i`` spreads ``n_i`` evenly over
    ``(u_i, 1]``; one censored at ``u_i = 1`` puts it all in the last bin.
    """
    B = edges.shape[0] - 1
    ev = c == 0
    idx = np.clip(np.searchsorted(edges, u[ev], side="right") - 1, 0, B - 1)
    events = np.bincount(idx, weights=n[ev], minlength=B).astype(float)
    uc = u[~ev]
    nc = n[~ev]
    lo, hi = edges[:-1], edges[1:]
    width = 1.0 - uc
    with np.errstate(divide="ignore", invalid="ignore"):
        overlap = np.clip(
            hi[None, :] - np.maximum(lo[None, :], uc[:, None]), 0, None
        )
        share = np.where(width[:, None] > 0, overlap / width[:, None], 0.0)
    share[width <= 0, B - 1] = 1.0
    censored = (share * nc[:, None]).sum(axis=0)
    return events, censored


class DCalibration:
    """
    The PIT / D-calibration histogram of a model's predictions and its
    chi-square test of uniformity. From :func:`d_calibration`.

    Attributes
    ----------
    edges : ndarray
        The ``B + 1`` bin edges on the probability scale, from 0 to 1.
    events : ndarray
        The weight in each bin from the cases with an outcome.
    censored : ndarray
        The weight in each bin from the censored cases, each spread over
        the range its outcome can still take.
    observed : ndarray
        ``events + censored``.
    expected : float
        What each bin should hold if the predictions are right: ``N / B``.
    n : float
        ``N``, the number of cases (the sum of their counts).
    n_censored : float
        How many of them were censored.
    statistic : float
        Pearson's chi-square statistic.
    df : int
        Its degrees of freedom, ``B - 1``.
    p_value : float
        The chance of a statistic this large or larger if the predictions
        are calibrated.
    randomised : bool
        Whether each ``u`` was drawn (``randomise=True``): the censored
        cases are then counted in ``events`` at their drawn ``u`` and
        ``censored`` is all zero.
    groups : dict
        With ``by=``, the result of each group (for example each forecast
        horizon), keyed by its label; otherwise empty.

    Examples
    --------
    >>> from surpyval import Weibull
    >>> from surpyval.metrics import d_calibration
    >>> model = Weibull.from_params([10.0, 2.0])
    >>> result = d_calibration([3, 6, 8, 11, 14, 20], [0, 0, 1, 0, 0, 1],
    ...                        model=model, bins=4)
    >>> result.observed.round(3)
    array([1.   , 1.052, 1.474, 2.474])
    >>> round(result.p_value, 3)
    0.817
    """

    def __init__(
        self,
        edges: npt.NDArray,
        events: npt.NDArray,
        censored: npt.NDArray,
        n_censored: float,
        randomised: bool,
        groups: dict | None = None,
    ) -> None:
        self.edges = np.asarray(edges, dtype=float)
        self.events = np.asarray(events, dtype=float)
        self.censored = np.asarray(censored, dtype=float)
        self.observed = self.events + self.censored
        self.n = float(self.observed.sum())
        self.n_censored = float(n_censored)
        B = self.observed.shape[0]
        self.expected = self.n / B
        self.df = B - 1
        if self.expected > 0:
            self.statistic = float(
                ((self.observed - self.expected) ** 2).sum() / self.expected
            )
            self.p_value = float(_chi2.sf(self.statistic, self.df))
        else:
            self.statistic = float("nan")
            self.p_value = float("nan")
        self.randomised = bool(randomised)
        self.groups = dict(groups or {})

    @property
    def table(self) -> DataFrame:
        """One row per bin: ``bin`` (from 1), ``u_from``, ``u_to``,
        ``events``, ``censored``, ``observed`` and ``expected``."""
        B = self.observed.shape[0]
        return DataFrame(
            {
                "bin": np.arange(1, B + 1),
                "u_from": self.edges[:-1],
                "u_to": self.edges[1:],
                "events": self.events,
                "censored": self.censored,
                "observed": self.observed,
                "expected": np.full(B, self.expected),
            }
        )

    def plot(self, ax: Any = None) -> Any:
        """The histogram, as shares of the cases, with the events and the
        censored stacked and the share each bin should hold if the
        predictions are right drawn across it.

        Returns
        -------
        matplotlib.axes.Axes

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from surpyval import Weibull
        >>> from surpyval.metrics import d_calibration
        >>> result = d_calibration([3, 6, 8, 11], [0, 0, 1, 0],
        ...                        model=Weibull.from_params([10.0, 2.0]),
        ...                        bins=4)
        >>> fig, ax = plt.subplots()
        >>> ax = result.plot(ax=ax)
        >>> plt.close(fig)
        """
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gcf().gca()
        total = self.n if self.n > 0 else 1.0
        lo = self.edges[:-1]
        width = np.diff(self.edges)
        ax.bar(
            lo,
            self.events / total,
            width=width,
            align="edge",
            label="outcomes",
            edgecolor="white",
        )
        ax.bar(
            lo,
            self.censored / total,
            width=width,
            bottom=self.events / total,
            align="edge",
            label="censored (spread)",
            edgecolor="white",
            alpha=0.6,
        )
        ax.axhline(
            1.0 / self.observed.shape[0],
            color="black",
            linestyle="--",
            label="if calibrated",
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel("Predicted CDF at the outcome, u = F(t)")
        ax.set_ylabel("Share of cases")
        ax.set_title(
            "Calibration: chi-square p = {:.3g} on {} df".format(
                self.p_value, self.df
            )
        )
        ax.legend()
        return ax

    def to_dict(self) -> dict:
        """The result as a dict of native Python values (strict JSON)."""
        from surpyval.serialisation import encode_non_finite

        return encode_non_finite(
            {
                "result": "DCalibration",
                "edges": [float(v) for v in self.edges],
                "events": [float(v) for v in self.events],
                "censored": [float(v) for v in self.censored],
                "n_censored": self.n_censored,
                "randomised": self.randomised,
                "groups": {
                    str(k): v.to_dict() for k, v in self.groups.items()
                },
            }
        )

    @classmethod
    def from_dict(cls, result_dict: dict) -> DCalibration:
        """Rebuild a result from :meth:`to_dict`."""
        from surpyval.serialisation import decode_non_finite

        if result_dict.get("result") != "DCalibration":
            raise ValueError(
                "Must create a DCalibration from a dict its to_dict wrote "
                "(with 'result': 'DCalibration')."
            )
        d = decode_non_finite(dict(result_dict))
        return cls(
            np.asarray(d["edges"], dtype=float),
            np.asarray(d["events"], dtype=float),
            np.asarray(d["censored"], dtype=float),
            d["n_censored"],
            d["randomised"],
            {k: cls.from_dict(v) for k, v in d.get("groups", {}).items()},
        )

    def __repr__(self) -> str:
        lines = [
            "D-calibration of {:g} cases ({:g} censored) in {} bins{}".format(
                self.n,
                self.n_censored,
                self.observed.shape[0],
                ", randomised PIT" if self.randomised else "",
            ),
            "  chi-square = {:.4g} on {} df, p = {:.4g}".format(
                self.statistic, self.df, self.p_value
            ),
            "  observed per bin: "
            + ", ".join(f"{v:.3g}" for v in self.observed)
            + f" (expected {self.expected:.3g} each)",
        ]
        if self.groups:
            lines.append("  by group:")
            for k, v in self.groups.items():
                lines.append(f"    {k}: n = {v.n:g}, p = {v.p_value:.4g}")
        return "\n".join(lines)


def _edges(bins: Any) -> npt.NDArray:
    if isinstance(bins, (int, np.integer)) and not isinstance(bins, bool):
        if bins < 2:
            raise ValueError(
                "'bins' must be at least 2 (Haider et al. use 10); got "
                "{}".format(bins)
            )
        return np.linspace(0.0, 1.0, int(bins) + 1)
    edges = _as_1d(bins, "bins")
    if (
        edges.shape[0] < 3
        or edges[0] != 0.0
        or edges[-1] != 1.0
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError(
            "'bins' as edges must increase from 0 to 1 with at least two "
            "bins, such as [0, 0.25, 0.5, 0.75, 1]; or pass the number of "
            "equal bins"
        )
    if not np.allclose(np.diff(edges), edges[1] - edges[0]):
        # Unequal bins would need expected counts in proportion to their
        # widths; the test here assumes equal ones.
        raise ValueError(
            "'bins' must be equal in width (the chi-square test expects "
            "the same count in each); pass the number of bins"
        )
    return edges


def d_calibration(
    x: npt.ArrayLike | None = None,
    c: npt.ArrayLike | None = None,
    n: npt.ArrayLike | None = None,
    *,
    model: Any = None,
    Z: npt.ArrayLike | None = None,
    F: npt.ArrayLike | None = None,
    F_left: npt.ArrayLike | None = None,
    bins: Any = 10,
    randomise: bool = False,
    by: npt.ArrayLike | None = None,
    seed: Any = None,
) -> DCalibration:
    """
    Whether a model's predicted distributions are calibrated: the PIT
    histogram (no censoring) or D-calibration (right censoring) and its
    chi-square test of uniformity.

    Each case's ``u_i = F_i(x_i)``, its own predicted CDF at its time,
    should be uniform on (0, 1) if the predictions are right. Events
    count in the bin holding ``u_i``; a case censored at ``x_i`` is known
    only to have ``u > F_i(x_i)``, and its weight is spread evenly over
    ``(F_i(x_i), 1]``. See the module notes for what the result does and
    does not say.

    Parameters
    ----------
    x : array_like, optional
        Each case's time: its outcome, or when it was censored. Needed
        with ``model``; with ``F`` it is not used.
    c : array_like, optional
        0 for an outcome, 1 for right censored (default all 0). Left and
        interval censoring are refused.
    n : array_like, optional
        Each case's count (default 1).
    model : object, optional
        A fitted model with a CDF, as :func:`pit_values` takes it: any
        model with ``ff(x)`` without ``Z``, or ``sf(x, Z)`` paired with
        the rows of ``Z`` with it.
    Z : array_like or pandas.DataFrame, optional
        Each case's covariates, for ``model``.
    F : array_like, optional
        Instead of ``model``, each case's predicted CDF at its time,
        ``F_i(x_i)``, from any source (a network's predicted
        distributions, another package).
    F_left : array_like, optional
        With ``F`` and ``randomise=True``, the predicted CDF just below
        each time, ``F_i(x_i-)`` (default ``F``: no jump). With ``model``
        it is computed.
    bins : int or array_like, default 10
        The number of equal bins on (0, 1), or their edges.
    randomise : bool, default False
        Draw each case's ``u`` at random from the range it can take if the
        model is right: an outcome's uniformly between ``F_i(x_i-)`` and
        ``F_i(x_i)`` (the randomised PIT, for discrete outcomes or
        predicted distributions with jumps), a censored case's uniformly
        between ``F_i(x_i)`` and 1 (in place of spreading its weight).
        The test then has its stated level with censoring, where the
        default is conservative; the result depends on the draw.
    by : array_like, optional
        A label per case (such as each forecast's horizon). The result
        then holds, in ``groups``, one histogram and test per label as
        well as the overall one. Use it when cases are repeated forecasts
        of the same series: the overall p-value assumes independent cases.
    seed : int or numpy.random.Generator, optional
        For ``randomise=True``.

    Returns
    -------
    DCalibration

    Examples
    --------
    A right model passes; one with the scale wrong fails:

    >>> import numpy as np
    >>> from surpyval import Weibull
    >>> from surpyval.metrics import d_calibration
    >>> rng = np.random.default_rng(3)
    >>> t = 10 * rng.weibull(2.0, 500)
    >>> cens = 12 * rng.uniform(size=500)
    >>> x, c = np.minimum(t, cens), (cens < t).astype(int)
    >>> right = d_calibration(x, c, model=Weibull.from_params([10, 2]))
    >>> wrong = d_calibration(x, c, model=Weibull.from_params([6, 2]))
    >>> right.p_value > 0.05, wrong.p_value < 0.001
    (True, True)

    The predicted CDFs can come from anywhere:

    >>> d_calibration(F=[0.1, 0.35, 0.6, 0.9], bins=2).observed
    array([2., 2.])
    """
    edges = _edges(bins)
    if (model is None) == (F is None):
        raise ValueError(
            "pass either model= (a fitted model, with Z= for one with "
            "covariates) or F= (each case's predicted CDF at its time), "
            "not both or neither"
        )
    if model is not None:
        if x is None:
            raise ValueError("model= needs 'x', each case's time")
        x_arr = _as_1d(x, "x")
        u = pit_values(model, x_arr, Z)
        u_left = _left_pit(model, x_arr, Z) if randomise else u
    else:
        if Z is not None:
            raise ValueError("'Z' is used with model=, not with F=")
        assert F is not None
        u = _as_1d(F, "F")
        if F_left is not None:
            if not randomise:
                raise ValueError("'F_left' is used only with randomise=True")
            u_left = _as_1d(F_left, "F_left")
            if u_left.shape != u.shape:
                raise ValueError("'F' and 'F_left' must have the same length")
            if np.any(u_left > u):
                raise ValueError(
                    "'F_left' must not exceed 'F': it is the CDF just "
                    "below each time"
                )
        else:
            u_left = u
        if x is not None and _as_1d(x, "x").shape != u.shape:
            raise ValueError("'x' and 'F' must have the same length")
    N = u.shape[0]
    c_arr = np.zeros(N) if c is None else _as_1d(c, "c")
    if c_arr.shape[0] != N:
        raise ValueError("'c' must have one flag per case")
    if not np.isin(c_arr, (0, 1)).all():
        raise ValueError(
            "'c' must be 0 (outcome) or 1 (right censored); left and "
            "interval censoring are not supported by D-calibration"
        )
    n_arr = np.ones(N) if n is None else _as_1d(n, "n")
    if n_arr.shape[0] != N or np.any(n_arr < 0):
        raise ValueError("'n' must be one non-negative count per case")
    if np.any(~np.isfinite(u)) or np.any((u < 0) | (u > 1)):
        raise ValueError(
            "the predicted CDF at {} of the {} cases is missing or outside "
            "[0, 1]; check the model's predictions (a missing covariate "
            "gives nan) and drop or fix those cases".format(
                int(np.sum(~np.isfinite(u) | (u < 0) | (u > 1))), N
            )
        )
    c_draw = c_arr
    if randomise:
        # An outcome's u is drawn between the CDF just below it and at it;
        # a censored case's between the CDF at its time and 1, where it
        # lies if the model is right, so it is then counted as an outcome.
        rng = np.random.default_rng(seed)
        v = rng.uniform(size=N)
        ev = c_arr == 0
        u = np.where(ev, u_left + v * (u - u_left), u + v * (1.0 - u))
        c_arr = np.zeros(N)
    labels = None if by is None else np.asarray(by)
    if labels is not None and labels.shape[0] != N:
        raise ValueError("'by' must have one label per case")

    def one(mask: npt.NDArray, groups: dict | None = None) -> DCalibration:
        events, censored = _spread(u[mask], c_arr[mask], n_arr[mask], edges)
        n_cens = (
            float(n_arr[mask][c_draw[mask] == 1].sum())
            if randomise
            else float(n_arr[mask][c_arr[mask] == 1].sum())
        )
        return DCalibration(edges, events, censored, n_cens, randomise, groups)

    groups = None
    if labels is not None:
        groups = {}
        for lab in dict.fromkeys(labels.tolist()):
            groups[lab] = one(labels == lab)
    return one(np.ones(N, dtype=bool), groups)
