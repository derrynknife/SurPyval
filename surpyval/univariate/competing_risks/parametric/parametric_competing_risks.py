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
    A parametric competing-risks model: one distribution per cause, combined
    into cumulative-incidence functions. Build it in one step from data with
    :meth:`fit` / :meth:`fit_from_df`, or assemble it from already-fitted
    per-cause models -- each of any distribution family -- with
    :meth:`from_fitted`.
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
        # A density may be singular at 0 (e.g. LogNormal); the grid includes
        # 0, so silence the harmless evaluation there.
        with np.errstate(divide="ignore", invalid="ignore"):
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
        # Integrate out to where the all-cause incidence has essentially
        # converged, so the CIF grid stays concentrated where the mass is.
        upper = max(self._model_horizon(k) for k in self.causes)
        return self.cif(upper, event)

    def random(self, size, random_state=None):
        """
        Draw ``size`` samples of ``(time, cause)`` from the model, using the
        latent-failure-time representation: draw a latent time from each cause
        and take the earliest, recording its cause.

        Sampling is by numerical inversion of each cause's cumulative
        distribution, so it works for *any* per-cause model, including limited-
        failure (cure) models where some latent times are infinite. A unit
        whose every latent time is infinite never fails; it is returned with
        ``x = inf`` and cause ``None``.

        Returns a structured array with fields ``x`` and ``e``.
        """
        rng = np.random.default_rng(random_state)
        latent = np.column_stack(
            [self._latent_sample(k, size, rng) for k in self.causes]
        )
        idx = np.argmin(latent, axis=1)
        x = latent[np.arange(size), idx]
        e = np.array([self.causes[i] for i in idx], dtype=object)
        # A unit with no finite latent time never fails from any cause.
        never = ~np.isfinite(x)
        if never.any():
            e[never] = None
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

    def _model_horizon(self, k):
        # The time by which cause k has essentially played out, used as the
        # finite upper limit for CIF(inf). Its very high quantile if available;
        # otherwise (e.g. a limited-failure model, whose quantile is not
        # defined) grow a bound until the cause's incidence stops rising.
        model = self.models[k]
        try:
            q = float(np.ravel(model.qf(1.0 - 1e-6))[0])
            if np.isfinite(q) and q > 0:
                return q
        except Exception:
            pass
        t = 1.0
        try:
            m = float(model.mean())
            if np.isfinite(m) and m > 0:
                t = m
        except Exception:
            pass
        prev = -1.0
        for _ in range(200):
            val = float(np.ravel(model.ff(t))[0])
            if val - prev < 1e-10:
                break
            prev = val
            t *= 1.5
        return t

    def _latent_sample(self, k, size, rng):
        # A latent failure time for cause k by numerical inversion of its CDF.
        # Draws in the (possible) cure region above the model's attainable CDF
        # are mapped to infinity: that unit never fails from this cause.
        model = self.models[k]
        u = rng.uniform(size=size)
        upper = self._model_horizon(k)
        grid = np.linspace(0.0, upper, 8000)
        cdf = np.ravel(model.ff(grid))
        cdf_max = float(cdf[-1])
        t = np.interp(u, cdf, grid)
        return np.where(u <= cdf_max, t, np.inf)

    # -- construction -----------------------------------------------------

    @classmethod
    def from_fitted(cls, models):
        """
        Assemble a competing-risks model from already-fitted single-cause
        models -- one per cause -- instead of fitting them here.

        Each cause's model may be of a completely different family: a Weibull
        with a limited-failure (cure) fraction for one cause, a LogNormal for
        another, a discrete distribution for a third, and so on. The only
        requirement is that every model exposes the standard surpyval model
        interface (``sf`` / ``ff`` / ``df`` / ``hf`` / ``Hf``); the cumulative
        incidence, all-cause survival and sampling are then assembled from
        them exactly as for a :meth:`fit` model.

        This is the right entry point when each cause has been modelled
        separately -- for example fitted with its own distribution, offset,
        limited-failure or zero-inflated options -- and you want to combine
        them into one competing-risks object.

        Parameters
        ----------
        models : dict or sequence
            Either a ``{cause: model}`` mapping, or a sequence of models whose
            causes are taken to be their positions ``0, 1, 2, ...``.

        Returns
        -------
        ParametricCompetingRisks
            The assembled model.

        Notes
        -----
        Each per-cause model should be fitted to the *cause-specific* view of
        the data (that cause's events observed, every other cause's events and
        every censored unit treated as right-censored) for the assembled CIFs
        to be the competing-risks quantities. If the causes carry a cure
        fraction the all-cause survival need not fall to zero, so the cause
        probabilities need not sum to one -- some units never fail.
        """
        if isinstance(models, dict):
            mapping = dict(models)
        else:
            mapping = {i: m for i, m in enumerate(models)}
        if len(mapping) == 0:
            raise ValueError("At least one cause model is required.")
        for k, m in mapping.items():
            missing = [
                a
                for a in ("sf", "ff", "df", "hf", "Hf")
                if not callable(getattr(m, a, None))
            ]
            if missing:
                raise ValueError(
                    "Model for cause {!r} is missing the method(s) {}; it "
                    "does not look like a fitted surpyval model.".format(
                        k, missing
                    )
                )
        try:
            causes = sorted(mapping)
        except TypeError:
            causes = list(mapping)

        model = cls()
        model.causes = causes
        model.models = mapping
        return model

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
