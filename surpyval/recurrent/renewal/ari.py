from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from numpy.typing import ArrayLike

from surpyval.recurrent.parametric.crow_amsaa import CrowAMSAA

if TYPE_CHECKING:
    from surpyval.recurrent.renewal.renewal_model import RenewalModel
from surpyval.recurrent.renewal.fit_mixin import RenewalFitMixin
from surpyval.utils.deprecation import renamed_arguments
from surpyval.utils.fitter import singleton_fitter
from surpyval.utils.pickling import Rebuilt
from surpyval.utils.recurrent_utils import (
    handle_xicn,
    measure_from_entry,
    reject_gapped_observation,
    validate_intensity_model,
    validate_memory,
    validate_nhpp_data,
    validate_renewal_censoring,
    validate_restoration,
)


def ari_reduction(
    failure_intensities: np.ndarray, rho: float, m: "int | float"
) -> "np.ndarray | float":
    """
    Intensity reduction ``R_n`` in force just after the most recent failure for
    the Arithmetic Reduction of Intensity model with memory ``m`` (Doyen &
    Gaudoin, 2004).

    ``failure_intensities`` are the baseline intensities ``lambda_0(T_k)``
    evaluated at the failures so far, ordered oldest to newest. The reduction
    uses the most recent ``min(m, n)`` of them::

        R_n = rho * sum_{j=0}^{min(m, n) - 1} (1 - rho)^j lambda_0(T_{n-j})

    ``m = 1`` keeps only the last failure (ARI1) and ``m = inf`` keeps the full
    history (ARI-infinity); ``rho = 0`` gives ``R_n = 0``, i.e. a plain NHPP.
    """
    n = len(failure_intensities)
    if n == 0:
        return 0.0
    upper = n if np.isinf(m) else min(int(m), n)
    recent = np.asarray(failure_intensities[-upper:])[::-1]
    weights = (1.0 - rho) ** np.arange(upper)
    return rho * np.sum(weights * recent)


def _reduction_sequence(
    failure_intensities: np.ndarray,
    position: int,
    rho: float,
    m: "int | float",
) -> np.ndarray:
    """``R_n`` after every failure at once, for failures from many items.

    ``failure_intensities`` holds ``lambda_0`` at each *observed* failure
    with the items laid end to end, and ``position`` gives each failure's
    index within its own item, which is what keeps one item's history
    from leaking into the next.

    This is ``ari_reduction`` evaluated at every prefix, but summed over
    the *window offset* rather than over the failures. The offset only
    ever runs to ``min(m, longest item)``, so the loop is a handful of
    whole-array passes -- for ARI1 exactly one -- in place of one Python
    step per failure. The offset ``i`` contributes only where the item
    actually has ``i`` earlier failures, which is the ``position >= i``
    mask and reproduces ``upper = min(m, n)`` above.
    """
    lam = np.asarray(failure_intensities, dtype=float)
    if lam.size == 0:
        return lam
    pos = np.asarray(position)
    longest = int(pos.max()) + 1
    span = longest if np.isinf(m) else min(int(m), longest)

    q = 1.0 - rho
    total = np.array(lam, copy=True)
    for i in range(1, span):
        shifted = np.concatenate([np.zeros(i), lam[:-i]])
        total += (q**i) * np.where(pos >= i, shifted, 0.0)
    return rho * total


def _event_layout(data: Any) -> tuple:
    """Per-row bookkeeping shared by the likelihood and the residuals.

    ``prev`` is the previous event time, restarting at 0 for each item.
    ``observed`` marks the failures. ``failure_pos`` gives each failure's
    ordinal within its own item, which bounds the reduction window.
    ``in_force`` indexes the reduction sequence at the reduction acting
    over each row's interval, or ``-1`` where the item has not failed
    yet and the intensity is still the unreduced baseline.

    Rows are assumed grouped by item, as ``handle_xicn`` leaves them.
    """
    x = np.asarray(data.x, dtype=float)
    item = np.asarray(data.i)
    observed = np.asarray(data.c) == 0

    starts = np.empty(x.size, dtype=bool)
    starts[0] = True
    starts[1:] = item[1:] != item[:-1]

    prev = np.empty_like(x)
    prev[0] = 0.0
    prev[1:] = x[:-1]
    prev[starts] = 0.0

    # Failures strictly before each row, globally then per item.
    before = np.cumsum(observed) - observed
    first_rows = np.flatnonzero(starts)
    lengths = np.diff(np.append(first_rows, x.size))
    before_item = before - np.repeat(before[first_rows], lengths)

    # Within an item the most recent earlier failure is also the most
    # recent one globally, so the global running count indexes it -- the
    # per-item count only decides whether one exists at all.
    in_force = np.where(before_item > 0, before - 1, -1)
    failure_pos = before_item[observed]
    return prev, observed, failure_pos, in_force


@singleton_fitter
class ARI(RenewalFitMixin):
    """
    Arithmetic Reduction of Intensity (ARI) imperfect-repair model of Doyen and
    Gaudoin (2004).

    Where the ARA/Kijima models reduce the *virtual age*, ARI reduces the
    failure *intensity* directly. For a baseline (first-failure) intensity
    ``lambda_0`` the process intensity on the interval following the n-th
    failure is::

        lambda(t) = lambda_0(t) - rho * sum_{j=0}^{min(m,n)-1}
                    (1 - rho)^j lambda_0(T_{n-j})

    so each repair subtracts a fraction ``rho`` of (a memory-weighted sum of)
    the past failure intensities. ``rho = 0`` recovers the plain NHPP defined
    by the baseline intensity (minimal repair) and ``rho = 1`` removes the
    most intensity a repair can; it is not a renewal process (ARI has no
    repair as good as new). The fitted model prints ``rho`` with its
    standard error and Wald interval, and the conclusion of
    ``repair_test()``, the likelihood-ratio tests of the fit against this
    maximal repair (``rho = 1``) and minimal repair (``rho = 0``). The
    baseline, ``baseline=``, is any of the recurrent intensity models
    (``CrowAMSAA``, ``Duane``, ``CoxLewis``); ``CrowAMSAA`` (power law) is
    the default. It is named ``baseline`` rather than ``dist`` because it
    is not a lifetime distribution, as ARA's and GeneralizedRenewal's
    ``dist`` is (#507); ``dist=`` still works, with a
    ``DeprecationWarning``, until v0.23.

    There is no closed-form marginal intensity, so the mean cumulative function
    is obtained by simulation (see ``mcf`` and ``plot``).

    Examples
    --------
    >>> from surpyval.recurrent import ARI, CrowAMSAA
    >>> import numpy as np
    >>>
    >>> x = np.array([3, 9, 20, 35, 56, 4, 11, 25, 44, 70])
    >>> i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    >>>
    >>> model = ARI.fit(x, i, m=1, baseline=CrowAMSAA)
    """

    @staticmethod
    def _build_sampler(model: Any, n: int) -> Callable:
        from surpyval.recurrent.renewal.renewal_model import DiscountedMemory
        from surpyval.utils.numeric import solve_bracketed

        dist = model.model.dist
        dp = model.model.params
        rho = model.rho
        # The baseline intensities at the failures so far, discounted over
        # the last m of them: the intensity reduction is rho times this.
        memory = DiscountedMemory(n, rho, model.m)
        running = np.zeros(n)

        def step(idx: np.ndarray, u: np.ndarray) -> np.ndarray:
            t0 = running[idx]
            reduction = rho * memory.value(idx)
            energy = -np.log(u)
            cif0 = np.asarray(dist.cif(t0, *dp), dtype=float)

            def g(x: np.ndarray, sel: np.ndarray) -> np.ndarray:
                cif = np.asarray(dist.cif(t0[sel] + x, *dp), dtype=float)
                return cif - cif0[sel] - reduction[sel] * x - energy[sel]

            everything = np.arange(idx.size)
            hi = np.ones(idx.size)
            growing = everything
            for _ in range(60):
                growing = growing[g(hi[growing], growing) < 0]
                if not growing.size:
                    break
                hi[growing] *= 2.0
            g_hi = g(hi, everything)
            # Still short of the energy after 60 doublings: take hi, as the
            # scalar sampler did.
            gap = hi.copy()
            rest = np.flatnonzero(g_hi > 0)
            if rest.size:
                gap[rest] = solve_bracketed(
                    lambda x, sel: g(x, rest[sel]),
                    np.zeros(rest.size),
                    hi[rest],
                    -energy[rest],
                    g_hi[rest],
                    xtol=2e-12,
                )
            arrival = t0 + gap
            running[idx] = arrival
            memory.record(idx, np.asarray(dist.iif(arrival, *dp), dtype=float))
            return gap

        return step

    def _make_model(
        self,
        baseline: Any,
        baseline_params: ArrayLike,
        rho: float,
        m: "int | float",
    ) -> "RenewalModel":
        from surpyval.recurrent.renewal.renewal_model import RenewalModel

        model = baseline.from_params(np.asarray(baseline_params).tolist())
        out = RenewalModel(
            model,
            rho,
            "rho",
            "Repair Efficiency",
            "ARI Recurrence",
            self._build_sampler,
            dist_label="Baseline Intensity",
            restoration_bounds=(0, 1),
        )
        out.m = m
        return out

    def _rescaled_increments(self, model: Any, data: Any) -> np.ndarray:
        """
        Per-interval compensator increments (time-rescaling residuals) for a
        fitted ARI model: the integral of the reduced intensity over each
        interval, ``[Lambda_0(t) - Lambda_0(prev)] - R * (t - prev)``, where
        ``R`` is the intensity reduction in force after the previous event.
        Aligned with ``data`` rows; iid Exp(1) over the observed intervals
        under the fitted model.
        """
        rho, m = model.rho, model.m
        dist = model.model
        x = np.asarray(data.x, dtype=float)
        prev, observed, failure_pos, in_force = _event_layout(data)

        lam = np.asarray(dist.iif(x[observed]), dtype=float)
        reductions = _reduction_sequence(lam, failure_pos, rho, m)
        active = np.where(in_force >= 0, reductions[in_force], 0.0)

        delta_cif = np.asarray(dist.cif(x) - dist.cif(prev), dtype=float)
        return delta_cif - active * (x - prev)

    def _refit(self, model: Any, data: Any) -> Any:
        """Refit this model family on ``data`` with the same baseline
        intensity and memory; used by the Cramer-von Mises bootstrap."""
        return self.fit_from_recurrent_data(
            data, baseline=model.model.dist, m=model.m
        )

    @renamed_arguments(dist="baseline")
    def create_negll_func(
        self, data: Any, baseline: Any, m: "int | float"
    ) -> Callable:
        """The negative log-likelihood of ``[rho, *baseline_params]`` on
        ``data``, for the baseline intensity model ``baseline`` and memory
        ``m``."""
        x = np.asarray(data.x, dtype=float)
        prev, observed, failure_pos, in_force = _event_layout(data)
        gap = x - prev
        x_failures = x[observed]

        def negll_func(params: np.ndarray) -> float:
            rho = params[0]
            baseline_params = params[1:]

            # lambda_0 at the failures drives the reductions; every row
            # then picks up whichever reduction was in force over its own
            # interval (`in_force` is -1 before the item's first failure,
            # where the baseline is unreduced).
            lam = baseline.iif(x_failures, *baseline_params)
            reductions = _reduction_sequence(lam, failure_pos, rho, m)
            active = np.where(in_force >= 0, reductions[in_force], 0.0)

            # A non-positive intensity is outside the model's support.
            # Checked before the log so it returns inf rather than
            # warning its way to a nan, as the scalar loop did by
            # returning early.
            intensity = lam - active[observed]
            if not np.all(intensity > 0):
                return np.inf

            delta_cif = baseline.cif(x, *baseline_params) - baseline.cif(
                prev, *baseline_params
            )
            ll = -np.sum(delta_cif - active * gap) + np.sum(np.log(intensity))
            return -ll

        return negll_func

    @renamed_arguments(dist="baseline")
    def fit_from_recurrent_data(
        self,
        data: Any,
        baseline: Any = CrowAMSAA,
        m: "int | float" = 1,
        init: "ArrayLike | None" = None,
    ) -> "RenewalModel":
        """
        Fit the ARI model from recurrent data.

        Parameters
        ----------

        data : RecurrentEventData
            Data containing the recurrence details.
            An item with delayed entry (a ``tl``) is taken to be as
            new at entry, with its times counted from there (see
            :meth:`fit`).
        baseline : object, optional
            A recurrent baseline intensity model (``CrowAMSAA``, ``Duane``,
            ``CoxLewis``). Default is ``CrowAMSAA``. Its old name,
            ``dist``, works until v0.23 with a ``DeprecationWarning``.
        m : int or float, optional
            Memory of the ARI model; a positive integer or ``numpy.inf``.
            Default is 1.
        init : list, optional
            Initial parameters ``[rho, *baseline_params]`` for the
            optimizer.

        Returns
        -------

        RenewalModel
            A fitted renewal model.
        """
        validate_intensity_model(baseline, type(self).__name__)
        validate_memory(m)
        validate_renewal_censoring(data.c, type(self).__name__)
        reject_gapped_observation(data, type(self).__name__)
        # Delayed entry: as new at entry (#615).
        data = measure_from_entry(data, type(self).__name__)
        # The baseline is an NHPP intensity, with the same needs: some
        # events, times inside its support (no event at t = 0 for a power
        # law) and more than one failure-truncated event.
        validate_nhpp_data(data, baseline)

        neg_ll = self.create_negll_func(data, baseline, m)
        base_params0 = self._default_start(
            lambda: self._initial_baseline_params(data, baseline), init
        )
        res, params = self._fit_restoration_ml(
            data,
            neg_ll,
            (0, 1),
            "rho",
            baseline,
            (0.1, 0.5, 0.9),
            base_params0,
            init,
        )
        rho, *baseline_params = params
        out = self._make_model(baseline, baseline_params, rho, m)
        # The likelihood kept as what it is built from, so the model
        # pickles (#573).
        neg_ll = Rebuilt(
            self.create_negll_func, (data, baseline, m), built=neg_ll
        )
        self._attach_inference(out, neg_ll, [rho, *baseline_params], res, data)
        return out

    @staticmethod
    def _initial_baseline_params(data: Any, baseline: Any) -> np.ndarray:
        """
        Initial parameters for the baseline intensity model: the plain NHPP fit
        of that baseline if it succeeds, otherwise its own parameter
        initialiser. (ARI's baseline is an intensity model, not a lifetime
        distribution, so this differs from the other repair fitters.)
        """
        try:
            base_params = np.asarray(
                baseline.fit_from_recurrent_data(data).params, dtype=float
            )
            if not np.all(np.isfinite(base_params)):
                raise ValueError
        except Exception:
            base_params = np.asarray(baseline.parameter_initialiser(data.x))
        return base_params

    @renamed_arguments(dist="baseline")
    def fit(
        self,
        x: ArrayLike,
        i: "ArrayLike | None" = None,
        c: "ArrayLike | None" = None,
        n: "ArrayLike | None" = None,
        baseline: Any = CrowAMSAA,
        m: "int | float" = 1,
        init: "ArrayLike | None" = None,
        tl: "ArrayLike | None" = None,
    ) -> "RenewalModel":
        """
        Fit the ARI model.

        Parameters
        ----------

        x : array_like
            The event times, pooled over items (each row belongs to the item
            named in ``i``), measured from the start of each item's life.
        i : array_like, optional
            Identity of the item each row belongs to. Defaults to all rows
            belonging to one item.
        c : array_like, optional
            Censoring indicators: 0 an observed failure, 1 the
            right-censored end of an item's observation. Other codes raise
            a ``ValueError``. Defaults to all observed.
        n : array_like, optional
            Count of events at each row. Defaults to 1.
        baseline : object, optional
            A recurrent baseline intensity model (``CrowAMSAA``, ``Duane``,
            ``CoxLewis``). Default is ``CrowAMSAA``. Unlike ARA's and
            GeneralizedRenewal's ``dist``, it is not a lifetime
            distribution: passing one (e.g. ``Weibull``) raises a
            ``ValueError`` that names the alternatives. Its old name,
            ``dist``, works until v0.23 with a ``DeprecationWarning``
            (#507).
        m : int or float, optional
            Memory of the ARI model; a positive integer or ``numpy.inf``.
            Default is 1.
        init : list, optional
            Initial parameters ``[rho, *baseline_params]`` for the
            optimizer.
        tl : array_like or scalar, optional
            Delayed entry: the time each item's observation began, when
            its failures before then were not recorded (a scalar for every
            item, or one value per row, the same on every row of an item).
            The item is taken to be **as new at entry**, as after an
            overhaul: the baseline intensity's clock restarts at ``tl``,
            with no reduction from earlier repairs, so its times count
            from there and its history before entry plays no part. That
            is exact for an item renewed at entry and an assumption
            otherwise; the fitted model's ``data`` hold the times from
            entry.

        Returns
        -------

        RenewalModel
            A fitted renewal model.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.recurrent import ARI, CrowAMSAA
        >>> x = np.array([3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60])
        >>> i = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        >>> c = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        >>> model = ARI.fit(x, i, c=c, m=1, baseline=CrowAMSAA)
        >>> model.model.params.round(3)
        array([3.508, 1.3  ])
        >>> round(float(model.rho), 3)
        1.0
        """
        # Before the data: a lifetime distribution here (as ARA takes)
        # failed deep inside the fit (#495).
        validate_intensity_model(baseline, type(self).__name__)
        data = handle_xicn(x, i, c, n, tl=tl)
        return self.fit_from_recurrent_data(data, baseline, m, init=init)

    @renamed_arguments(dist="baseline")
    def fit_from_df(
        self,
        df: Any,
        x_col: str,
        i_col: "str | None" = None,
        c_col: "str | None" = None,
        n_col: "str | None" = None,
        tl_col: "str | None" = None,
        tr_col: "str | None" = None,
        **fit_options: Any,
    ) -> "RenewalModel":
        """
        Fit to an event log held in the columns of a
        :class:`pandas.DataFrame`.

        As every recurrent ``fit_from_df``: the column names are passed in
        place of the arrays :meth:`fit` takes, and every other :meth:`fit`
        option (``baseline``, ``m``, ``init``) is passed to it unchanged.
        ``dist=``, the old name of ``baseline``, works until v0.23 with a
        ``DeprecationWarning`` (#507).

        Parameters
        ----------
        df : pandas.DataFrame
            The event log.
        x_col : str
            Column of event (and end-of-observation) times.
        i_col : str, optional
            Column of item / unit ids. Defaults to a single item.
        c_col : str, optional
            Column of censoring flags (0 an event, 1 the end of a unit's
            observation).
        n_col : str, optional
            Column of event counts per row.
        tl_col, tr_col : str, optional
            Refused: ARI takes no truncation.
        **fit_options
            Every other option of :meth:`fit`.

        Returns
        -------
        RenewalModel
            The model :meth:`fit` returns.

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval.recurrent import ARI, CrowAMSAA
        >>> log = pd.DataFrame({
        ...     "hours": [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60],
        ...     "unit": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        ...     "c": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        ... })
        >>> model = ARI.fit_from_df(
        ...     log, x_col="hours", i_col="unit", c_col="c",
        ...     baseline=CrowAMSAA, m=1,
        ... )
        >>> model.model.params.round(3)
        array([3.508, 1.3  ])
        """
        return super().fit_from_df(
            df,
            x_col,
            i_col=i_col,
            c_col=c_col,
            n_col=n_col,
            tl_col=tl_col,
            tr_col=tr_col,
            **fit_options,
        )

    @renamed_arguments(dist="baseline", dist_params="baseline_params")
    def fit_from_parameters(
        self,
        baseline_params: ArrayLike,
        rho: float,
        m: "int | float" = 1,
        baseline: Any = CrowAMSAA,
    ) -> "RenewalModel":
        """
        Build an ARI model from given parameters.

        Parameters
        ----------

        baseline_params : list
            Parameters for the baseline intensity model. Its old name,
            ``dist_params``, works until v0.23 with a
            ``DeprecationWarning``.
        rho : float
            Repair efficiency in ``[0, 1]``.
        m : int or float, optional
            Memory of the ARI model; a positive integer or ``numpy.inf``.
            Default is 1.
        baseline : object, optional
            A recurrent baseline intensity model. Default is ``CrowAMSAA``.
            Its old name, ``dist``, works until v0.23 with a
            ``DeprecationWarning`` (#507).

        Returns
        -------

        RenewalModel
            A model built from the supplied parameters, for simulation.

        Examples
        --------
        >>> from surpyval.recurrent import ARI, Duane
        >>> model = ARI.fit_from_parameters([1.0, 1.5], 0.6, baseline=Duane)
        >>> model.parameter_names
        ['rho', 'alpha', 'b']
        """
        validate_intensity_model(baseline, type(self).__name__)
        validate_memory(m)
        validate_restoration(rho, "rho", (0, 1))
        return self._make_model(baseline, baseline_params, rho, m)
