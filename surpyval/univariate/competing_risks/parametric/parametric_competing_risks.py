"""
Parametric competing-risks model.

Where :class:`~surpyval.univariate.competing_risks.CompetingRisks` estimates
each cause's cumulative incidence non-parametrically (a step function),
``ParametricCompetingRisks`` fits a *parametric distribution* to each cause's
cause-specific hazard and assembles smooth, extrapolatable cumulative-incidence
functions from them.

The key fact that makes this simple and exact is that the parametric
cause-specific likelihood **factorises across causes**: the contribution of an
observation is the cause-specific density if it failed from that cause and the
cause-specific survival otherwise, so maximising the joint likelihood is the
same as fitting each cause's distribution independently with the *other*
causes' events treated as right-censored. The cumulative incidence of cause
:math:`k` is then

.. math::
    \\mathrm{CIF}_k(t) = \\int_0^t h_k(u)\\,S(u)\\,du
                       = \\int_0^t f_k(u)\\!\\!\\prod_{j\\neq k} S_j(u)\\,du,

with the all-cause survival :math:`S(u) = \\prod_j S_j(u) = \\exp(-\\sum_j
H_j(u))`. The cause CIFs sum to the all-cause failure probability
:math:`1 - S(t)`.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

from surpyval.univariate.parametric import Weibull
from surpyval.utils import check_e_and_x, xcnt_handler


def _validate(x, c, n, e):
    """Wrangle ``x``/``c``/``n`` and check the event labels: ``e`` is the
    per-observation cause, and must be ``None`` exactly for censored rows."""
    x, c, n, _ = xcnt_handler(x, c, n, group_and_sort=False)
    x, c, n = (np.asarray(a, dtype=float) for a in (x, c, n))
    e = np.asarray(e, dtype=object)
    check_e_and_x(e, x)
    if (-1 in c) or (2 in c):
        raise ValueError(
            "Left or interval censoring is not supported by competing risks."
        )
    if any(ev is not None for ev in e[c == 1]) or any(
        ev is None for ev in e[c != 1]
    ):
        raise ValueError(
            "None can only be used as the event type for a censored "
            "observation (c = 1)."
        )
    return x, c, n, e


class ParametricCompetingRisks:
    """
    A fitted parametric competing-risks model: one parametric distribution per
    cause, combined into cumulative-incidence functions. Build with
    :meth:`fit` or :meth:`fit_from_df`.
    """

    causes: list
    models: dict

    def __repr__(self):
        dists = ", ".join(
            "{}: {}".format(k, self.models[k].dist.name) for k in self.causes
        )
        return (
            "Parametric Competing Risks SurPyval Model\n"
            "=========================================\n"
            "Causes              : {causes}\n"
            "Cause distributions : {dists}".format(
                causes=list(self.causes), dists=dists
            )
        )

    # -- cause-specific and all-cause functions ---------------------------

    def Hf(self, x, event=None):
        """Cumulative hazard. ``event=None`` gives the all-cause cumulative
        hazard :math:`\\sum_k H_k`; ``event=k`` gives cause ``k``'s."""
        if event is not None:
            self._check_event(event)
            return self.models[event].Hf(x)
        return sum(self.models[k].Hf(x) for k in self.causes)

    def hf(self, x, event=None):
        """Hazard rate. ``event=None`` is the all-cause hazard
        :math:`\\sum_k h_k`; ``event=k`` is the cause-specific hazard."""
        if event is not None:
            self._check_event(event)
            return self.models[event].hf(x)
        return sum(self.models[k].hf(x) for k in self.causes)

    def sf(self, x):
        """All-cause survival :math:`S(t) = \\prod_k S_k(t)`."""
        return np.exp(-self.Hf(x))

    def ff(self, x):
        """All-cause failure probability :math:`1 - S(t)` (the total
        cumulative incidence over all causes)."""
        return -np.expm1(-self.Hf(x))

    def iif(self, x, event):
        """
        Instantaneous incidence function (the subdistribution density) of a
        cause: :math:`f_k^{\\mathrm{sub}}(t) = h_k(t) S(t) = f_k(t)
        \\prod_{j\\neq k} S_j(t)`.
        """
        self._check_event(event)
        x = np.asarray(x, dtype=float)
        others = np.ones_like(x, dtype=float)
        for j in self.causes:
            if j != event:
                others = others * self.models[j].sf(x)
        return self.models[event].df(x) * others

    def cif(self, x, event=None):
        """
        Cumulative incidence function. ``event=k`` returns
        :math:`\\int_0^t f_k^{\\mathrm{sub}}(u)\\,du`, the probability of
        having failed from cause ``k`` by ``t``; ``event=None`` gives the
        all-cause incidence :math:`1 - S(t) = \\sum_k \\mathrm{CIF}_k(t)`.
        """
        if event is None:
            return self.ff(x)
        self._check_event(event)
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        upper = max(float(x_arr.max()), np.finfo(float).tiny)
        grid = np.linspace(0.0, upper, 4000)
        integrand = self.iif(grid, event)
        # A hazard may diverge at 0 (e.g. Weibull shape < 1); the mass there
        # is finite and negligible on a fine grid, so drop non-finite points.
        integrand = np.where(np.isfinite(integrand), integrand, 0.0)
        cif_grid = cumulative_trapezoid(integrand, grid, initial=0.0)
        out = np.interp(x_arr, grid, cif_grid)
        return out if np.ndim(x) else float(out[0])

    def probability_of_cause(self, event):
        """
        The eventual probability that a unit fails from ``event``,
        :math:`\\mathrm{CIF}_k(\\infty)`. These sum to one over all causes.
        """
        self._check_event(event)
        # Integrate out to where every cause's survival has essentially
        # vanished (a high quantile), rather than a crude multiple of the
        # scale, so the CIF grid stays concentrated where the mass is.
        upper = max(self._tail(k) for k in self.causes)
        return self.cif(upper, event)

    def random(self, size, random_state=None):
        """
        Draw ``size`` samples of ``(time, cause)`` from the fitted model, using
        the latent-failure-time representation: draw a latent time from each
        cause's distribution and take the earliest, whose cause is recorded.
        Returns a structured array with fields ``x`` and ``e``.
        """
        rng = np.random.default_rng(random_state)
        latent = np.column_stack(
            [self.models[k].random(size) for k in self.causes]
        )
        # random() draws are independent of rng; shuffle-free, deterministic
        # given each model's own draw. Use rng only if a model ignores it.
        del rng
        idx = np.argmin(latent, axis=1)
        x = latent[np.arange(size), idx]
        e = np.array([self.causes[i] for i in idx], dtype=object)
        out = np.empty(size, dtype=[("x", float), ("e", object)])
        out["x"] = x
        out["e"] = e
        return out

    # -- goodness of fit (the joint likelihood factorises over causes) ----

    def neg_ll(self):
        """Total negative log-likelihood: the sum over the per-cause fits."""
        return float(sum(self.models[k].neg_ll() for k in self.causes))

    def aic(self):
        """Akaike information criterion of the joint model."""
        return float(sum(self.models[k].aic() for k in self.causes))

    def bic(self):
        """Bayesian information criterion of the joint model."""
        return float(sum(self.models[k].bic() for k in self.causes))

    # -- helpers ----------------------------------------------------------

    def _check_event(self, event):
        if event not in self.models:
            raise ValueError(
                "Unknown cause {!r}; fitted causes are {}.".format(
                    event, list(self.causes)
                )
            )

    def _tail(self, k):
        # The time by which cause k has essentially certainly occurred, used
        # as the finite upper limit for CIF(inf); its very high quantile, or a
        # large multiple of the mean if the quantile is unavailable.
        model = self.models[k]
        try:
            q = float(np.ravel(model.qf(1.0 - 1e-6))[0])
            if np.isfinite(q) and q > 0:
                return q
        except Exception:
            pass
        return 50.0 * float(model.mean())

    # -- construction -----------------------------------------------------

    @classmethod
    def fit(cls, x, e, c=None, n=None, dist=Weibull, how="MLE"):
        """
        Fit a parametric distribution to each cause's cause-specific hazard.

        Parameters
        ----------
        x : array_like
            Observed times.
        e : array_like
            The cause of each observation; ``None`` for a censored row.
        c : array_like, optional
            Censoring flag (0 observed, 1 right-censored). Defaults to all
            observed. Left/interval censoring is not supported.
        n : array_like, optional
            Counts per observation.
        dist : ParametricFitter or dict, optional
            The distribution fitted to each cause (default ``Weibull``). Pass a
            ``{cause: distribution}`` mapping to use a different distribution
            per cause.
        how : str, optional
            Estimation method passed to each distribution's ``fit`` (default
            ``"MLE"``).

        Returns
        -------
        ParametricCompetingRisks
            The fitted model.
        """
        x, c, n, e = _validate(x, c, n, e)

        causes = sorted({ev for ev in e[c == 0]})
        if not causes:
            raise ValueError("No observed events to fit a cause to.")

        models = {}
        for k in causes:
            # Cause k observed where its event occurred; every other event and
            # every censored row is right-censored for cause k.
            c_k = np.where((e == k) & (c == 0), 0, 1).astype(int)
            distribution = dist[k] if isinstance(dist, dict) else dist
            models[k] = distribution.fit(x=x, c=c_k, n=n, how=how)

        model = cls()
        model.causes = causes
        model.models = models
        return model

    @classmethod
    def fit_from_df(
        cls, df, x_col, e_col, c_col=None, n_col=None, dist=Weibull, how="MLE"
    ):
        """Fit from a DataFrame; see :meth:`fit`. ``x_col`` / ``e_col`` name
        the time and cause columns, with optional ``c_col`` / ``n_col``."""
        x = df[x_col].to_numpy()
        e = df[e_col].to_numpy(dtype=object)
        c = None if c_col is None else df[c_col].to_numpy()
        n = None if n_col is None else df[n_col].to_numpy()
        model = cls.fit(x, e, c=c, n=n, dist=dist, how=how)
        return model
