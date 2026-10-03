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
the generic confidence-bound path is right without any change to that code.
The likelihood is differentiable by autograd, so the fit ends as the ordinary
one does (``finish_search`` and ``keep_information``): a coefficient with no
finite maximum is warned of, and the covariance is the exact observed
information (#555), not a finite-difference Hessian of
``model.model.neg_ll(model.data, ...)``.
"""

from __future__ import annotations

import functools
import types
from typing import Any, Callable

import autograd.numpy as np
import numpy
import numpy.typing as npt
from autograd.extend import defvjp, primitive

from surpyval.univariate.information_criteria import ic_sample_size
from surpyval.univariate.parametric.fitters import bounds_convert
from surpyval.utils.surpyval_data import SurpyvalData

from .._kinds import ACCELERATED_FAILURE_TIME
from ..parametric_regression_model import ParametricRegressionModel
from ..tvc_fit import fit_tvc_df


def _validate_full_coverage(
    x: npt.NDArray, tl: npt.NDArray, ident: npt.NDArray
) -> None:
    """
    Require each subject's episodes to tile ``(0, T]`` completely: first
    entry at 0 and no gaps between consecutive episodes.

    The accumulated accelerated age ``psi(T) = sum_k phi_k (b_k - a_k)``
    implements the integral from 0, so uncovered time would silently
    contribute *zero* accelerated ageing — shifting every subject's window
    by +5 used to return bit-identical parameters (#258). Conditioning a
    delayed-entry AFT fit correctly needs ``psi`` over the unobserved
    pre-entry window, i.e. covariate values the data does not contain, so
    rather than guess them the fit refuses. Cox TVC (``CoxPH.fit_tvc``)
    handles delayed entry and gaps exactly, because its risk-set partial
    likelihood never needs the unobserved history.
    """
    uniq, starts, counts = np.unique(
        ident, return_index=True, return_counts=True
    )
    for u, f, cnt in zip(uniq, starts, counts):
        s_tl = tl[f : f + cnt]
        s_x = x[f : f + cnt]
        if s_tl[0] > 1e-12:
            raise ValueError(
                "subject {} enters observation at {} > 0: the accelerated "
                "failure time TVC likelihood integrates the covariate path "
                "from time 0, and the pre-entry covariates are unobserved. "
                "Start each subject's first interval at 0, or use Cox TVC "
                "(CoxPH.fit_tvc), which handles delayed entry "
                "exactly.".format(u, s_tl[0])
            )
        if cnt > 1 and np.any(np.abs(s_tl[1:] - s_x[:-1]) > 1e-9):
            raise ValueError(
                "subject {} has a gap between observation intervals: the "
                "accelerated failure time TVC likelihood needs the covariate "
                "path over the subject's whole life, and the in-gap "
                "covariates are unobserved. Make each subject's intervals "
                "contiguous, or use Cox TVC (CoxPH.fit_tvc), which handles "
                "gaps exactly.".format(u)
            )


def _grouped_episodes(
    x: npt.NDArray,
    c: npt.NDArray,
    n: npt.NDArray,
    tl: npt.NDArray,
    ident: npt.NDArray,
) -> dict:
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


@primitive
def _subject_sums(values: npt.NDArray, starts: npt.NDArray) -> npt.NDArray:
    """The sum of ``values`` over each subject's contiguous rows, the
    subjects' first rows at ``starts`` (ascending)."""
    return numpy.add.reduceat(values, starts)


def _subject_sums_vjp(
    ans: npt.NDArray, values: npt.NDArray, starts: npt.NDArray
) -> Callable:
    # Each row's sum is its subject's: the gradient of a row is that of
    # its subject's sum. (Indexing, so autograd differentiates it again
    # for the Hessian.)
    sizes = numpy.diff(numpy.append(starts, numpy.shape(values)[0]))
    subject = numpy.repeat(numpy.arange(len(starts)), sizes)
    return lambda g: g[subject]


defvjp(_subject_sums, _subject_sums_vjp)


def _aft_tvc_neg_ll(self: Any, data: Any, *params: float) -> float:
    """
    Negative log-likelihood of the accelerated-failure-time model along each
    subject's time-varying covariate path.

    Bound (per instance) onto the result's ``AFTFitter`` so it replaces the
    ordinary independent-rows ``neg_ll`` for this fit only. ``data`` is ignored
    -- the grouped episode arrays captured at fit time live on ``self._tvc`` --
    so the generic confidence-bound path, which re-calls this with the model's
    stored data, recomputes the accumulated-age likelihood correctly.

    Written in ``autograd.numpy``, with the per-subject sums an autograd
    primitive, so that the fit has its exact gradient and Hessian: the
    no-maximum check and the covariance read them (#555).
    """
    tvc = self._tvc
    k_dist = self.k_dist
    dist_params = params[:k_dist]
    beta = np.array(params[k_dist:])

    # Acceleration factor per episode, accumulated to each subject's total
    # accelerated age via a segment sum over its (contiguous) episode rows.
    phi_ep = np.exp(np.dot(tvc["Zep"], beta))
    psi = _subject_sums(phi_ep * tvc["widths"], tvc["starts"])
    psi = np.maximum(psi, numpy.finfo(float).tiny)

    H0 = self.Hf_dist(psi, *dist_params)
    ll = -(tvc["weight"] * H0).sum()

    event = tvc["event"]
    if event.any():
        h0 = self.hf_dist(psi, *dist_params)
        phi_term = phi_ep[tvc["term"]]
        log_haz = np.log(
            np.maximum(phi_term[event] * h0[event], numpy.finfo(float).tiny)
        )
        ll = ll + (tvc["weight"][event] * log_haz).sum()

    return -ll


from .._fit_skeleton import (  # noqa: E402
    Centring,
    LogLinearPhi,
    MirroredDistributionAttrs,
    alias_coefficients,
    assemble_regression_model,
    check_fixed_and_init,
    coefficient_floor,
    free_coefficients,
    judge_search,
    keep_information,
    optimise_nm_tnc,
    say_verdict,
)


class AFTTVCFitMixin(MirroredDistributionAttrs):
    """
    Adds time-varying-covariate fitting (``fit_tvc`` and friends) to the
    accelerated failure time fitter. Mixed into ``AFTFitter``; kept separate
    from the additive-hazard ``TVCFitMixin`` because AFT needs its own
    accumulated-age likelihood rather than a reshape-and-refit.
    """

    def fit_tvc(
        self,
        i: npt.ArrayLike,
        xl: npt.ArrayLike,
        xr: npt.ArrayLike,
        c: npt.ArrayLike,
        Z: npt.ArrayLike,
        n: "npt.ArrayLike | None" = None,
        fixed: "dict[str, float] | None" = None,
        center: bool = False,
    ) -> ParametricRegressionModel:
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
        center : bool, optional
            Report the baseline at the covariate means of the interval
            rows (stored as ``model.center``) instead of at ``Z = 0``, as
            for :meth:`fit`.

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
            x, c_a, n_a, tl, Z_a, ident, AFTFitter, fixed, center
        )

    def fit_tvc_timeline(
        self,
        i: npt.ArrayLike,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike,
        n: "npt.ArrayLike | None" = None,
        fixed: "dict[str, float] | None" = None,
        center: bool = False,
    ) -> ParametricRegressionModel:
        """
        Fit from a per-subject covariate *timeline* (one row per covariate
        change, terminal status on the last row) instead of explicit
        ``(xl, xr]`` intervals. See ``CoxPH.fit_tvc_timeline`` for the format.
        ``fixed`` and ``center`` are as for :meth:`fit_tvc`.
        """
        from ..proportional_hazards.tvc import handle_tvc_timeline

        i2, xl, xr, c2, Z2, n2 = handle_tvc_timeline(i, x, Z, c, n)
        return self.fit_tvc(
            i2, xl, xr, c2, Z2, n=n2, fixed=fixed, center=center
        )

    def fit_tvc_from_df(
        self,
        df: Any,
        i_col: str,
        xl_col: str,
        xr_col: str,
        c_col: str,
        Z_cols: "str | list[str] | None" = None,
        n_col: "str | None" = None,
        fixed: "dict[str, float] | None" = None,
        center: bool = False,
        formula: "str | None" = None,
    ) -> ParametricRegressionModel:
        """
        ``fit_tvc`` from a start-stop ``DataFrame``. The covariates are
        ``Z_cols``, a single column name or a list, or instead ``formula``,
        a ``formulaic`` formula as in ``fit_from_df``, which codes
        categorical columns; give exactly one. ``feature_names`` (and the
        ``formula`` and its encoding) are recorded on the model, so it
        predicts from a DataFrame with the same design. ``fixed`` and
        ``center`` are as for :meth:`fit_tvc`.
        """
        return fit_tvc_df(
            self.fit_tvc,
            df,
            {"i": i_col, "xl": xl_col, "xr": xr_col, "c": c_col},
            Z_cols,
            formula,
            n_col,
            fixed=fixed,
            center=center,
        )

    def fit_tvc_timeline_from_df(
        self,
        df: Any,
        i_col: str,
        x_col: str,
        Z_cols: "str | list[str] | None",
        c_col: str,
        n_col: "str | None" = None,
        fixed: "dict[str, float] | None" = None,
        center: bool = False,
        formula: "str | None" = None,
    ) -> ParametricRegressionModel:
        """
        :meth:`fit_tvc_timeline` from a covariate-timeline ``DataFrame``,
        as for the proportional hazards models and Cox.

        ``i_col``, ``x_col``, ``c_col`` and ``n_col`` name the columns
        passed to :meth:`fit_tvc_timeline` as ``i``, ``x``, ``c`` and
        ``n``. The covariates are ``Z_cols``, a column name or a list of
        them, or instead (pass ``Z_cols=None``) ``formula``, as for
        :meth:`fit_tvc_from_df`; the model records ``feature_names`` (and
        the ``formula`` and its encoding). ``fixed`` and ``center`` are as
        for :meth:`fit_tvc`. The timeline is the same data as its
        start-stop rows, so the fit is that of :meth:`fit_tvc_from_df`.

        Examples
        --------
        Each unit runs at its own stress ``z``, which steps up by 0.5 half
        way through its life; the last row of a unit gives its exit time
        and status:

        >>> import numpy as np
        >>> import pandas as pd
        >>> from surpyval import WeibullAFT
        >>> rng = np.random.default_rng(1)
        >>> z = rng.normal(size=100)
        >>> T = rng.weibull(2.0, 100) * 5 * np.exp(-0.5 * z)
        >>> df = pd.DataFrame({
        ...     "id": np.repeat(np.arange(100), 3),
        ...     "time": np.column_stack([0 * T, T / 2, T]).ravel(),
        ...     "z": np.column_stack([z, z + 0.5, z + 0.5]).ravel(),
        ...     "c": 0,
        ... })
        >>> model = WeibullAFT.fit_tvc_timeline_from_df(
        ...     df, "id", "time", "z", "c"
        ... )
        >>> model.feature_names
        ['z']
        """
        return fit_tvc_df(
            self.fit_tvc_timeline,
            df,
            {"i": i_col, "x": x_col, "c": c_col},
            Z_cols,
            formula,
            n_col,
            fixed=fixed,
            center=center,
        )

    def _fit_tvc_arrays(
        self,
        x: npt.NDArray,
        c: npt.NDArray,
        n: npt.NDArray,
        tl: npt.NDArray,
        Z: npt.NDArray,
        ident: npt.NDArray,
        AFTFitter: Any,
        fixed: "dict[str, float] | None",
        center: bool = False,
    ) -> ParametricRegressionModel:
        if fixed is None:
            fixed = {}
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        if Z.shape[0] == 1 and x.shape[0] != 1:
            Z = Z.reshape(-1, 1)
        p = Z.shape[1]
        _validate_full_coverage(x, tl, ident)
        grp = _grouped_episodes(x, c, n, tl, ident)
        phi_param_map = {"beta_" + str(j): j for j in range(p)}
        # A column the data cannot determine is held at 0 and reported as
        # nan, with one warning, as by the ordinary fit (#476).
        fixed = alias_coefficients(
            self, ACCELERATED_FAILURE_TIME, Z, n, fixed, phi_param_map
        )

        # Centred on the interval rows' means, as the ordinary AFT fit
        # (#463): exp(beta'z) on a covariate far from 0 overflows.
        centring = Centring.plan(
            self, ACCELERATED_FAILURE_TIME, Z, n, fixed, center
        )
        mean = np.zeros(p) if centring is None else centring.center

        # Result fitter: a fresh AFTFitter (so all ordinary prediction
        # functions are inherited unchanged) with the accumulated-age
        # likelihood bound onto this one instance only.
        like = AFTFitter(self.dist)
        like._tvc = {**grp, "Zep": Z - mean}
        like.neg_ll = types.MethodType(_aft_tvc_neg_ll, like)

        # Initial values: a plain distribution fit to the subject exit times,
        # regression coefficients at zero.
        init_data = SurpyvalData(
            grp["exit"], grp["term_c"], None, None, group_and_sort=False
        )
        ps = self.dist.fit_from_surpyval_data(init_data).params
        init = np.array([*ps, *np.zeros(p)])

        bounds = (*self.bounds, *(((None, None),) * p))
        param_map = {
            **self.param_map,
            **{k: v + self.k_dist for k, v in phi_param_map.items()},
        }

        check_fixed_and_init(fixed, None, param_map)
        transform, inv_trans, const, fixed_idx, not_fixed = bounds_convert(
            grp["exit"], bounds, fixed, param_map
        )
        init = transform(init)[not_fixed]
        coefs = free_coefficients(like, fixed, phi_param_map)
        # Each coefficient in its own covariate's units (#577)
        floor = coefficient_floor(len(init), coefs, Z)

        with np.errstate(all="ignore"):

            def fun(pars: npt.NDArray) -> float:
                return like.neg_ll(None, *inv_trans(const(pars)))

            # The same search as the ordinary AFT fit (the gradient ladder,
            # then Nelder-Mead and TNC), which says what it found below.
            res = optimise_nm_tnc(fun, init, quiet=True, floor=floor)
            # What it reached, polished where it was not a verified
            # maximum; said once the model is built. The likelihood is one
            # term per subject.
            verdict = judge_search(
                fun,
                res,
                coefs,
                init,
                float(np.sum(grp["weight"])),
                floor=floor,
            )
            res = verdict.res

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
        raw_neg_ll: "Callable | None" = None
        if centring is not None:
            # The baseline moved to Z = 0 when that is representable, as
            # for the ordinary fit; the likelihood of the data as given is
            # the check.
            centring.raw = edata
            raw = AFTFitter(self.dist)
            raw._tvc = {**grp, "Zep": Z}
            raw_neg_ll = functools.partial(_aft_tvc_neg_ll, raw, None)

        # The fitter carrying the accumulated-age likelihood is the
        # model's, so its bounds use that likelihood.
        model = assemble_regression_model(
            like,
            ACCELERATED_FAILURE_TIME,
            LogLinearPhi(LogLinearPhi.NAME_EXP, phi_param_map),
            edata,
            res,
            inv_trans(const(res.x)),
            bounds,
            phi_param_map,
            fixed,
            centring=centring,
            raw_neg_ll=raw_neg_ll,
        )
        model.is_tvc = True
        # One warning for what the search found, as for the ordinary fit
        # (#392, #555): a coefficient that runs off (a covariate level
        # with no events), or else a search that stopped short; and the
        # exact observed information for the covariance, which was a
        # numerical Hessian.
        say_verdict(verdict)
        model.maximum = verdict.maximum
        keep_information(
            model,
            verdict.no_maximum,
            verdict.derivatives,
            inv_trans,
            const,
            res.x,
            centring,
        )

        # Report information criteria on the *subjects*, not the episode
        # rows: the accumulated-age likelihood is one term per subject. The
        # sample size of bic and aic_c is the shared rule (ic_sample_size):
        # the subjects whose failure was observed, or all subjects when
        # none was. With none, bic used to fall back to the episode data.
        model.n_subjects = int(grp["n_subjects"])
        model._ic_n = ic_sample_size(
            np.where(grp["event"], 0, 1), grp["weight"]
        )
        return model
