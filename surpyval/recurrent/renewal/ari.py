import numpy as np
from scipy.optimize import brentq, minimize

from surpyval.recurrent.parametric.crow_amsaa import CrowAMSAA
from surpyval.recurrent.renewal.fit_mixin import RenewalFitMixin
from surpyval.utils.fitter import singleton_fitter
from surpyval.utils.recurrent_utils import (
    handle_xicn,
    reject_gapped_observation,
    reject_left_truncation,
    validate_memory,
    validate_renewal_censoring,
)


def ari_reduction(failure_intensities, rho, m):
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


def _reduction_sequence(failure_intensities, position, rho, m):
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


def _event_layout(data):
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
    by the baseline intensity. The baseline is any of the recurrent intensity
    models (``CrowAMSAA``, ``Duane``, ``CoxLewis``); ``CrowAMSAA`` (power law)
    is the default.

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
    >>> model = ARI.fit(x, i, m=1, dist=CrowAMSAA)
    """

    @staticmethod
    def _build_sampler(model):
        dist = model.model.dist
        dp = model.model.params
        rho = model.rho
        m = model.m
        history_iif = []
        running = [0.0]
        reduction = [0.0]

        def sample(ui):
            t0 = running[0]
            red = reduction[0]
            energy = -np.log(ui)

            def g(x):
                delta = dist.cif(t0 + x, *dp) - dist.cif(t0, *dp)
                return delta - red * x - energy

            hi = 1.0
            expansions = 0
            while g(hi) < 0 and expansions < 60:
                hi *= 2.0
                expansions += 1
            xi = hi if g(hi) < 0 else brentq(g, 0.0, hi)

            running[0] = t0 + xi
            history_iif.append(dist.iif(running[0], *dp))
            reduction[0] = ari_reduction(history_iif, rho, m)
            return xi

        return sample

    def _make_model(self, baseline_dist, dist_params, rho, m):
        from surpyval.recurrent.renewal.renewal_model import RenewalModel

        model = baseline_dist.from_params(list(dist_params))
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

    def _rescaled_increments(self, model, data):
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

    def _refit(self, model, data):
        """Refit this model family on ``data`` with the same baseline
        intensity and memory; used by the Cramer-von Mises bootstrap."""
        return self.fit_from_recurrent_data(
            data, dist=model.model.dist, m=model.m
        )

    def create_negll_func(self, data, dist, m):
        x = np.asarray(data.x, dtype=float)
        prev, observed, failure_pos, in_force = _event_layout(data)
        gap = x - prev
        x_failures = x[observed]

        def negll_func(params):
            rho = params[0]
            dist_params = params[1:]

            # lambda_0 at the failures drives the reductions; every row
            # then picks up whichever reduction was in force over its own
            # interval (`in_force` is -1 before the item's first failure,
            # where the baseline is unreduced).
            lam = dist.iif(x_failures, *dist_params)
            reductions = _reduction_sequence(lam, failure_pos, rho, m)
            active = np.where(in_force >= 0, reductions[in_force], 0.0)

            # A non-positive intensity is outside the model's support.
            # Checked before the log so it returns inf rather than
            # warning its way to a nan, as the scalar loop did by
            # returning early.
            intensity = lam - active[observed]
            if not np.all(intensity > 0):
                return np.inf

            delta_cif = dist.cif(x, *dist_params) - dist.cif(
                prev, *dist_params
            )
            ll = -np.sum(delta_cif - active * gap) + np.sum(np.log(intensity))
            return -ll

        return negll_func

    def fit_from_recurrent_data(self, data, dist=CrowAMSAA, m=1, init=None):
        """
        Fit the ARI model from recurrent data.

        Parameters
        ----------

        data : RecurrentData
            Data containing the recurrence details.
        dist : object, optional
            A recurrent baseline intensity model (``CrowAMSAA``, ``Duane``,
            ``CoxLewis``). Default is ``CrowAMSAA``.
        m : int or float, optional
            Memory of the ARI model; a positive integer or ``numpy.inf``.
            Default is 1.
        init : list, optional
            Initial parameters ``[rho, *dist_params]`` for the optimizer.

        Returns
        -------

        ARI
            A fitted ARI object.
        """
        validate_memory(m)
        validate_renewal_censoring(data.c, type(self).__name__)
        reject_left_truncation(data, type(self).__name__)
        reject_gapped_observation(data, type(self).__name__)

        neg_ll = self.create_negll_func(data, dist, m)
        transform, inv_trans = self._bounds_transform(
            data.x, [(0, 1), *dist.bounds], ["rho", *dist.param_names]
        )

        def fit_once(x0):
            return minimize(
                lambda p: neg_ll(inv_trans(p)),
                transform(np.asarray(x0, dtype=float)),
                method="Nelder-Mead",
            )

        if init is None:
            base_params = self._initial_baseline_params(data, dist)
            inits = [[rho_init, *base_params] for rho_init in (0.1, 0.5, 0.9)]
        else:
            inits = None
        res = self._multistart(fit_once, inits, init)

        rho, *dist_params = inv_trans(res.x)
        out = self._make_model(dist, dist_params, rho, m)
        # Only the observed failures (c == 0) contribute an intensity term, so
        # they are the events that enter the BIC sample size.
        self._attach_inference(
            out,
            neg_ll,
            [rho, *dist_params],
            int((data.c == 0).sum()),
            res,
            data,
        )
        return out

    @staticmethod
    def _initial_baseline_params(data, dist):
        """
        Initial parameters for the baseline intensity model: the plain NHPP fit
        of that baseline if it succeeds, otherwise its own parameter
        initialiser. (ARI's baseline is an intensity model, not a lifetime
        distribution, so this differs from the other repair fitters.)
        """
        try:
            base_params = np.asarray(
                dist.fit_from_recurrent_data(data).params, dtype=float
            )
            if not np.all(np.isfinite(base_params)):
                raise ValueError
        except Exception:
            base_params = np.asarray(dist.parameter_initialiser(data.x))
        return base_params

    def fit(self, x, i=None, c=None, n=None, dist=CrowAMSAA, m=1, init=None):
        """
        Fit the ARI model.

        Parameters
        ----------

        x : array_like
            An array of event times.
        i : array_like, optional
            An array of item indices.
        c : array_like, optional
            An array of censoring indicators.
        n : array_like, optional
            An array of counts.
        dist : object, optional
            A recurrent baseline intensity model. Default is ``CrowAMSAA``.
        m : int or float, optional
            Memory of the ARI model; a positive integer or ``numpy.inf``.
            Default is 1.
        init : list, optional
            Initial parameters ``[rho, *dist_params]`` for the optimizer.

        Returns
        -------

        ARI
            A fitted ARI object.
        """
        data = handle_xicn(x, i, c, n)
        return self.fit_from_recurrent_data(data, dist, m, init=init)

    def fit_from_parameters(self, dist_params, rho, m=1, dist=CrowAMSAA):
        """
        Build an ARI model from given parameters.

        Parameters
        ----------

        dist_params : list
            Parameters for the baseline intensity model.
        rho : float
            Repair efficiency in ``[0, 1]``.
        m : int or float, optional
            Memory of the ARI model; a positive integer or ``numpy.inf``.
            Default is 1.
        dist : object, optional
            A recurrent baseline intensity model. Default is ``CrowAMSAA``.

        Returns
        -------

        ARI
            An ARI object built from the supplied parameters.
        """
        validate_memory(m)
        return self._make_model(dist, dist_params, rho, m)
