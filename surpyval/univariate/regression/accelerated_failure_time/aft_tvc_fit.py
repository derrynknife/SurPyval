r"""
Time-varying-covariate *fitting* for the accelerated failure time family.

Unlike proportional / additive hazards -- whose cumulative hazard is additive
over disjoint intervals, so a time-varying-covariate subject factorises into
independent left-truncated rows that the ordinary MLE fits unchanged (see
``..tvc_fit.TVCFitMixin``) -- accelerated failure time rescales the *time
axis*. A covariate ``z`` on a segment contributes ``phi(z) = exp(beta'z)``
worth of *accelerated age* per unit real time, so the baseline is evaluated at
the subject's **accumulated** accelerated age

.. math::
    \psi(T) = \int_0^T e^{\beta' Z(u)}\,du = \sum_k e^{\beta' z_k}(b_k - a_k),

and the subject's likelihood is ``[phi(z_last) h0(psi)]^{event}
exp(-H0(psi))``. Because ``psi`` is a within-subject running sum of
``exp(beta'z_k)`` and the episode entry ages in that sum depend on ``beta``,
the episodes cannot be reshaped into independent rows with fixed truncation
points the way the additive families can. A bespoke negative log-likelihood is
therefore needed: on every optimiser step it re-accumulates each subject's
accelerated age before evaluating the baseline.

To keep the shared, well-tested machinery untouched this module does **not**
modify ``AFTFitter.fit`` or the shared ``regression_neg_ll``. It builds a fresh
``AFTFitter`` for the result (so every ordinary prediction function -- ``sf``,
``Hf``, ``sf_tvc`` -- is inherited unchanged) and overrides only its ``neg_ll``
with the accumulated-age likelihood on that single instance. The fitted
``ParametricRegressionModel`` therefore carries the *correct* likelihood, so
the generic confidence-bound path (a finite-difference Hessian of
``model.model.neg_ll(model.data, ...)``) is right without any change to that
code.
"""

import types

import numpy as np
from scipy.optimize import minimize

from surpyval.univariate.parametric.fitters import bounds_convert
from surpyval.utils.surpyval_data import SurpyvalData

from ..parametric_regression_model import ParametricRegressionModel


def _grouped_episodes(x, c, n, tl, ident):
    """
    From the (subject-contiguous, entry-sorted) episode arrays returned by
    ``handle_tvc``, derive the per-subject grouping the accumulated-age
    likelihood needs.

    Returns a dict of arrays: ``widths`` (b - a per episode), the group
    ``starts`` (first row index of each subject, ascending, for
    ``np.add.reduceat``), the terminal-episode index ``term`` per subject, the
    ``event`` flag per subject (its terminal row is an event), the subject
    ``weight`` (its terminal row's ``n``), and the subject exit times.
    """
    uniq, starts, counts = np.unique(
        ident, return_index=True, return_counts=True
    )
    term = starts + counts - 1
    return {
        "widths": (x - tl).astype(float),
        "starts": starts.astype(int),
        "term": term.astype(int),
        "event": (c[term] == 0),
        "weight": n[term].astype(float),
        "exit": x[term].astype(float),
        "term_c": c[term].astype(int),
        "n_subjects": int(uniq.shape[0]),
    }


def _aft_tvc_neg_ll(self, data, *params):
    """
    Negative log-likelihood of the accelerated-failure-time model along each
    subject's time-varying covariate path.

    Bound (per instance) onto the result's ``AFTFitter`` so it replaces the
    ordinary independent-rows ``neg_ll`` for this fit only. ``data`` is ignored
    -- the grouped episode arrays captured at fit time live on ``self._tvc`` --
    so the generic confidence-bound path, which re-calls this with the model's
    stored data, recomputes the accumulated-age likelihood correctly.
    """
    tvc = self._tvc
    k_dist = self.k_dist
    dist_params = params[:k_dist]
    beta = np.array(params[k_dist:], dtype=float)

    # Acceleration factor per episode, accumulated to each subject's total
    # accelerated age via a segment sum over its (contiguous) episode rows.
    phi_ep = np.exp(tvc["Zep"] @ beta)
    psi = np.add.reduceat(phi_ep * tvc["widths"], tvc["starts"])
    psi = np.maximum(psi, np.finfo(float).tiny)

    H0 = np.asarray(self.Hf_dist(psi, *dist_params), dtype=float)
    ll = -(tvc["weight"] * H0).sum()

    event = tvc["event"]
    if event.any():
        h0 = np.asarray(self.hf_dist(psi, *dist_params), dtype=float)
        phi_term = phi_ep[tvc["term"]]
        log_haz = np.log(
            np.maximum(phi_term[event] * h0[event], np.finfo(float).tiny)
        )
        ll = ll + (tvc["weight"][event] * log_haz).sum()

    return -ll


class AFTTVCFitMixin:
    """
    Adds time-varying-covariate fitting (``fit_tvc`` and friends) to the
    accelerated failure time fitter. Mixed into ``AFTFitter``; kept separate
    from the additive-hazard ``TVCFitMixin`` because AFT needs its own
    accumulated-age likelihood rather than a reshape-and-refit.
    """

    def fit_tvc(self, i, xl, xr, c, Z, n=None, fixed=None):
        """
        Fit the accelerated failure time model to start-stop (counting-process)
        time-varying-covariate data.

        Parameters
        ----------
        i : array_like
            Subject identifier per interval row.
        xl, xr : array_like
            The open-closed observation interval ``(xl, xr]`` of each row.
        c : array_like
            Status at ``xr`` in surpyval's convention: ``0`` terminal event,
            ``1`` right-censored interval end (covariate change / exit).
        Z : array_like
            Covariate row, constant on each interval.
        n : array_like, optional
            Count weight per subject (read from the terminal row). Default 1.
        fixed : dict, optional
            Parameters to hold fixed, by name.

        Returns
        -------
        ParametricRegressionModel
            The fitted model, carrying the accumulated-age likelihood so its
            confidence bounds are correct.
        """
        from ..proportional_hazards.tvc import handle_tvc
        from .aft_fitter import AFTFitter

        x, c_a, n_a, tl, Z_a, ident = handle_tvc(i, xl, xr, c, Z, n)
        return self._fit_tvc_arrays(
            x, c_a, n_a, tl, Z_a, ident, AFTFitter, fixed
        )

    def fit_tvc_timeline(self, i, x, Z, c, n=None, fixed=None):
        """
        Fit from a per-subject covariate *timeline* (one row per covariate
        change, terminal status on the last row) instead of explicit
        ``(xl, xr]`` intervals. See ``CoxPH.fit_tvc_timeline`` for the format.
        """
        from ..proportional_hazards.tvc import handle_tvc_timeline

        i2, xl, xr, c2, Z2, n2 = handle_tvc_timeline(i, x, Z, c, n)
        return self.fit_tvc(i2, xl, xr, c2, Z2, n=n2, fixed=fixed)

    def fit_tvc_from_df(
        self, df, id_col, xl_col, xr_col, c_col, Z_cols, n_col=None, fixed=None
    ):
        """
        ``fit_tvc`` from a start-stop ``DataFrame``. ``Z_cols`` may be a single
        column name or a list; ``feature_names`` is recorded on the model.
        """
        cols = [Z_cols] if isinstance(Z_cols, str) else list(Z_cols)
        n = None if n_col is None else df[n_col].values
        model = self.fit_tvc(
            df[id_col].values,
            df[xl_col].values,
            df[xr_col].values,
            df[c_col].values,
            df[cols].values,
            n=n,
            fixed=fixed,
        )
        model.feature_names = cols
        return model

    def _fit_tvc_arrays(self, x, c, n, tl, Z, ident, AFTFitter, fixed):
        if fixed is None:
            fixed = {}
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        if Z.shape[0] == 1 and x.shape[0] != 1:
            Z = Z.reshape(-1, 1)
        p = Z.shape[1]
        grp = _grouped_episodes(x, c, n, tl, ident)

        # Result fitter: a fresh AFTFitter (so all ordinary prediction
        # functions are inherited unchanged) with the accumulated-age
        # likelihood bound onto this one instance only.
        like = AFTFitter(self.dist)
        like._tvc = {**grp, "Zep": Z}
        like.neg_ll = types.MethodType(_aft_tvc_neg_ll, like)

        # Initial values: a plain distribution fit to the subject exit times,
        # regression coefficients at zero.
        init_data = SurpyvalData(
            grp["exit"], grp["term_c"], None, None, group_and_sort=False
        )
        ps = self.dist.fit_from_surpyval_data(init_data).params
        init = np.array([*ps, *np.zeros(p)])

        phi_param_map = {"beta_" + str(j): j for j in range(p)}
        bounds = (*self.bounds, *(((None, None),) * p))
        param_map = {
            **self.param_map,
            **{k: v + self.k_dist for k, v in phi_param_map.items()},
        }

        transform, inv_trans, const, fixed_idx, not_fixed = bounds_convert(
            grp["exit"], bounds, fixed, param_map
        )
        init = transform(init)[not_fixed]

        with np.errstate(all="ignore"):

            def fun(pars):
                return like.neg_ll(None, *inv_trans(const(pars)))

            res = minimize(
                fun, init, method="Nelder-Mead", options={"maxiter": 1000}
            )
            res2 = minimize(fun, res.x, method="TNC")
            res = res2 if res2.success else res

        params = inv_trans(const(res.x))

        _pm = phi_param_map

        class _PhiModel:
            name = "Log Linear [exp(beta'Z)]"
            phi_param_map = _pm

            def phi(self, Z, *pp):
                return np.exp(np.dot(Z, np.array(pp)))

        # Episode-level data container so generic consumers (repr, plotting)
        # have the usual attributes; the likelihood does not read it.
        edata = SurpyvalData(
            x,
            c,
            n,
            np.column_stack([tl, np.full(tl.shape[0], np.inf)]),
            group_and_sort=False,
        )
        edata.add_covariates(Z)

        model = ParametricRegressionModel()
        model.model = like
        model.reg_model = _PhiModel()
        model.kind = "Accelerated Failure Time"
        model.distribution = self.dist
        model.params = np.array(params)
        model.dist_params = np.array(params[: self.k_dist])
        model.phi_params = np.array(params[self.k_dist :])
        model.res = res
        model._neg_ll = res.fun
        model.fixed = fixed
        model.k_dist = self.k_dist
        model.k = len(bounds)
        model.data = edata
        model.is_tvc = True

        # Report information criteria on the *subject* count, not the episode
        # rows: the accumulated-age likelihood is one term per subject.
        n_subjects = float(grp["weight"].sum())
        n_events = float(grp["weight"][grp["event"]].sum())
        model.n_subjects = int(grp["n_subjects"])
        k = model.k
        if n_events > 0:
            model._bic = k * np.log(n_events) + 2 * res.fun
        if n_subjects - k - 1 > 0:
            model._aic_c = (2 * k + 2 * res.fun) + (2 * k**2 + 2 * k) / (
                n_subjects - k - 1
            )

        return model
