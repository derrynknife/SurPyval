"""Reliability growth projection: the MTBF once delayed fixes are in (#607).

The AMSAA-Crow projection model (ACPM; Crow 1983, MIL-HDBK-189C section
6.2) and Crow's (2004) extended model. A test of ``k`` systems, each run
from 0 to ``T``, surfaces failure modes of three kinds:

- **A modes**, which will not be fixed;
- **BC modes**, fixed during the test (their effect is already in the
  intensity the test demonstrates);
- **BD modes**, whose fixes are delayed to the end of the test, each with a
  fix-effectiveness factor (FEF) ``d_i``: the fraction of its intensity the
  fix removes.

With ``N_i`` failures of BD mode ``i`` (``K`` distinct BD modes seen), the
projected intensity of one system after the delayed fixes is

.. math::
    r_P = \\lambda_{CA} - \\frac{N_{BD}}{kT}
          + \\sum_{i=1}^{K} (1 - d_i) \\frac{N_i}{kT} + \\bar d\\, h(T),

where :math:`\\lambda_{CA}` is the intensity the test demonstrates,
:math:`\\bar d` the mean FEF of the BD modes seen, and :math:`h(T)` the
rate at which new BD modes were still being found at the end of the test:
the first occurrences of the BD modes are a power-law process, and
:math:`h(T) = K \\bar\\beta / (kT)` with the unbiased
:math:`\\bar\\beta = (K - 1) / K \\cdot \\hat\\beta`,
:math:`\\hat\\beta = K / \\sum_i \\ln(T / t_i)` (``t_i`` mode ``i``'s first
occurrence). The term corrects for the fixes being judged on the modes
seen, whose rates ``N_i / (kT)`` over-state what fixing them removes: the
modes not yet seen carry the rest of the B-mode intensity, which on average
is :math:`h(T)` (Crow 1983).

Without BC modes the system did not change during the test (test-find-
test), so :math:`\\lambda_{CA} = N / (kT)` and the projection is the ACPM:
:math:`r_P = N_A/(kT) + \\sum (1 - d_i) N_i/(kT) + \\bar d\\, h(T)`. With BC
modes the system grew during the test, and :math:`\\lambda_{CA}` is the
Crow-AMSAA intensity at ``T`` fitted to every failure (Crow 2004).

The growth potential is the intensity once every BD mode has been found and
fixed with these FEFs, :math:`r_{GP} = \\lambda_{CA} - N_{BD}/(kT) +
\\sum (1 - d_i) N_i/(kT)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass(frozen=True)
class GrowthProjection:
    """The projected reliability of a growth test once its delayed fixes
    are in: returned by :meth:`CrowAMSAA.projection
    <surpyval.recurrent.parametric.crow_amsaa.CrowAMSAA.projection>`.

    Intensities and MTBFs are those of one system.

    Attributes
    ----------
    T : float
        The end of the test (every system run from 0 to ``T``).
    systems : int
        The number of systems tested, ``k``.
    failures : dict
        The number of failures of each kind, keys ``"A"``, ``"BC"`` and
        ``"BD"``.
    modes : pandas.DataFrame
        One row per BD mode seen, indexed by its label: its number of
        failures (``failures``), its first occurrence (``first``, the
        earliest over the systems), its fix-effectiveness factor (``fef``)
        and its intensity before and after its fix (``intensity`` and
        ``projected``, ``N_i / (kT)`` and ``(1 - d_i) N_i / (kT)``).
    beta : float
        The unbiased shape of the BD modes' first occurrences,
        ``(K - 1) / K`` times its maximum-likelihood estimate (``nan``
        without a BD mode).
    mean_fef : float
        The mean fix-effectiveness factor of the BD modes seen (``nan``
        without one).
    new_mode_intensity : float
        ``h(T)``, the rate at which new BD modes were being found at the
        end of the test (0 with fewer than two BD modes).
    demonstrated_intensity : float
        The intensity the test demonstrates at ``T``, before the delayed
        fixes: ``N / (kT)`` without BC modes, the Crow-AMSAA intensity at
        ``T`` with them.
    projected_intensity : float
        The intensity once the delayed fixes are in.
    growth_potential_intensity : float
        The intensity once every BD mode has been found and fixed with
        these effectiveness factors.
    model : ParametricRecurrenceModel
        The Crow-AMSAA fit to every failure.
    """

    T: float
    systems: int
    failures: dict
    modes: pd.DataFrame
    beta: float
    mean_fef: float
    new_mode_intensity: float
    demonstrated_intensity: float
    projected_intensity: float
    growth_potential_intensity: float
    model: Any

    @property
    def demonstrated_mtbf(self) -> float:
        """The MTBF the test demonstrates, ``1 / demonstrated_intensity``."""
        return _reciprocal(self.demonstrated_intensity)

    @property
    def projected_mtbf(self) -> float:
        """The MTBF once the delayed fixes are in,
        ``1 / projected_intensity``."""
        return _reciprocal(self.projected_intensity)

    @property
    def growth_potential_mtbf(self) -> float:
        """The growth potential MTBF, ``1 / growth_potential_intensity``:
        the most these fixes can reach."""
        return _reciprocal(self.growth_potential_intensity)

    def __repr__(self) -> str:
        title = "Reliability growth projection (AMSAA-Crow)"
        n = self.failures
        lines = [
            title,
            "=" * len(title),
            "Test                : {} system{} to T = {:g}".format(
                self.systems, "" if self.systems == 1 else "s", self.T
            ),
            "Failures            : {} A, {} BC, {} BD ({} BD mode{})".format(
                n["A"],
                n["BC"],
                n["BD"],
                len(self.modes),
                "" if len(self.modes) == 1 else "s",
            ),
            "Mean FEF            : {:.4g}".format(self.mean_fef),
            "New BD modes        : beta = {:.4g}, h(T) = {:.4g}".format(
                self.beta, self.new_mode_intensity
            ),
            "                      {:>12} {:>12}".format("intensity", "MTBF"),
        ]
        for label, rate, mtbf in (
            (
                "Demonstrated",
                self.demonstrated_intensity,
                self.demonstrated_mtbf,
            ),
            ("Projected", self.projected_intensity, self.projected_mtbf),
            (
                "Growth potential",
                self.growth_potential_intensity,
                self.growth_potential_mtbf,
            ),
        ):
            lines.append(
                "{:<20}: {:>12.4g} {:>12.4g}".format(label, rate, mtbf)
            )
        return "\n".join(lines)


def _reciprocal(value: float) -> float:
    return float(np.inf) if value <= 0 else float(1.0 / value)


def _labels(values: Any, name: str) -> set:
    """The set of mode labels in ``values`` (an iterable, or ``None``)."""
    if values is None:
        return set()
    if isinstance(values, (str, bytes)):
        raise ValueError(
            "{} must be a collection of mode labels, not a single string; "
            "got {!r}".format(name, values)
        )
    return set(values)


def _missing(label: Any) -> bool:
    return label is None or (isinstance(label, float) and np.isnan(label))


def growth_projection(
    fitter: Any,
    x: npt.ArrayLike,
    modes: npt.ArrayLike,
    fef: dict,
    i: npt.ArrayLike | None = None,
    c: npt.ArrayLike | None = None,
    bc: Any = None,
) -> GrowthProjection:
    """See :meth:`CrowAMSAA.projection`."""
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    mode_arr = np.empty(x_arr.size, dtype=object)
    given = list(np.atleast_1d(np.asarray(modes, dtype=object)))
    if len(given) != x_arr.size:
        raise ValueError(
            "modes must give one label per row of x ({} rows); got "
            "{}".format(x_arr.size, len(given))
        )
    mode_arr[:] = given
    c_arr = (
        np.zeros(x_arr.size, dtype=int)
        if c is None
        else np.atleast_1d(np.asarray(c))
    )
    i_arr = (
        np.ones(x_arr.size, dtype=int)
        if i is None
        else np.atleast_1d(np.asarray(i))
    )
    if c_arr.shape != x_arr.shape or i_arr.shape != x_arr.shape:
        raise ValueError("x, i, c and modes must have the same length")
    if not np.all(np.isin(c_arr, (0, 1))):
        raise ValueError(
            "A growth projection takes exact failures (c=0) and the end of "
            "each system's test (c=1); got c values {}".format(
                sorted(set(c_arr.tolist()) - {0, 1})
            )
        )
    if not isinstance(fef, dict):
        raise ValueError(
            "fef must be a dict of BD mode label -> fix-effectiveness "
            "factor; got {}".format(type(fef).__name__)
        )
    events = c_arr == 0
    unlabelled = [k for k in np.flatnonzero(events) if _missing(mode_arr[k])]
    if unlabelled:
        raise ValueError(
            "Every failure (c=0) needs a mode label; {} have none (rows "
            "{})".format(len(unlabelled), unlabelled[:5])
        )
    seen = set(mode_arr[events].tolist())
    bd_modes = set(fef)
    bc_modes = _labels(bc, "bc")
    both = bd_modes & bc_modes
    if both:
        raise ValueError(
            "A mode is either fixed during the test (bc) or delayed (fef), "
            "not both: {}".format(sorted(map(str, both)))
        )
    unknown = (bd_modes | bc_modes) - seen
    if unknown:
        raise ValueError(
            "These modes have no failures in the data: {}. Only modes seen "
            "in the test can be classified.".format(sorted(map(str, unknown)))
        )
    factors = {}
    for label, value in fef.items():
        d = float(value)
        if not 0.0 <= d <= 1.0:
            raise ValueError(
                "A fix-effectiveness factor is a fraction in [0, 1]; mode "
                "{!r} has {}".format(label, value)
            )
        factors[label] = d

    model = fitter.fit(x_arr, i=i_arr, c=c_arr)
    T, n_events, terminated = model._crow_design()
    if terminated != "time":
        raise ValueError(
            "A growth projection needs a time-terminated test: each system "
            "run from 0 to the end of the test, T, recorded as a c=1 row at "
            "T. Add that row (the time the test stopped)."
        )
    k = len(model.data.items)
    total_time = k * T

    t_ev = x_arr[events]
    m_ev = mode_arr[events]
    kind = np.array(
        [
            "BD" if m in bd_modes else "BC" if m in bc_modes else "A"
            for m in m_ev
        ]
    )
    counts = {key: int(np.sum(kind == key)) for key in ("A", "BC", "BD")}

    labels = [m for m in fef if m in seen]
    n_i = np.array([np.sum(m_ev == m) for m in labels], dtype=float)
    first = np.array([t_ev[m_ev == m].min() for m in labels], dtype=float)
    d_i = np.array([factors[m] for m in labels], dtype=float)
    table = pd.DataFrame(
        {
            "failures": n_i.astype(int),
            "first": first,
            "fef": d_i,
            "intensity": n_i / total_time,
            "projected": (1.0 - d_i) * n_i / total_time,
        },
        index=pd.Index(labels, name="mode"),
    )

    K = len(labels)
    if K == 0:
        beta = mean_fef = float("nan")
        h = 0.0
    else:
        mean_fef = float(d_i.mean())
        log_sum = float(np.sum(np.log(T / first)))
        # (K - 1) / K * K / log_sum; 0 for K = 1, and a first occurrence
        # at T itself (log_sum 0) has no information on the shape.
        beta = (K - 1) / log_sum if log_sum > 0 else float("nan")
        h = K * beta / total_time if K > 1 and log_sum > 0 else 0.0

    if counts["BC"] == 0:
        demonstrated = n_events / total_time
    else:
        demonstrated = float(model.iif(T))
    remaining = demonstrated - n_i.sum() / total_time
    potential = remaining + float(table["projected"].sum())
    projected = potential + (mean_fef * h if K else 0.0)
    return GrowthProjection(
        T=float(T),
        systems=k,
        failures=counts,
        modes=table,
        beta=float(beta),
        mean_fef=float(mean_fef),
        new_mode_intensity=float(h),
        demonstrated_intensity=float(demonstrated),
        projected_intensity=float(projected),
        growth_potential_intensity=float(potential),
        model=model,
    )
