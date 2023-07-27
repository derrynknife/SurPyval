"""
This code was created for and sponsored by Cartiga (www.cartiga.com).
Cartiga makes no representations or warranties in connection with the code
and waives any and all liability in connection therewith. Your use of the
code constitutes acceptance of these terms.

Copyright 2022 Cartiga LLC
"""

from copy import copy

import numpy as np
import numpy.ma as ma
import numpy_indexed as npi
from numba import njit
from numpy.linalg import inv, pinv
from scipy.optimize import root
from scipy.stats import norm

from ..nonparametric import (
    FlemingHarrington,
    KaplanMeier,
    NelsonAalen,
    Turnbull,
)
from ..utils import validate_coxph, validate_coxph_df_inputs
from .proportional_hazards import ProportionalHazardsModel
from .regression import Regression

nonparametric_dists = {
    "Nelson-Aalen": NelsonAalen,
    "Kaplan-Meier": KaplanMeier,
    "Fleming-Harrington": FlemingHarrington,
    "Turnbull": Turnbull,
}


@njit
def efron_jit(n_d, Ri, Di, out):
    for i in range(len(n_d)):
        val = 0.0
        if n_d[i] == 0:
            continue
        for j in range(int(n_d[i])):
            c = j / n_d[i]
            v = Ri[i] - (c * Di[i])
            val += np.log(v)[0]
        out[i] = val
    return out


# @njit
# def efron_jac_jit(n_d, Ri, ZRi, Di, ZDi, out):
#     for i in range(len(n_d)):
#         val = np.zeros(out.shape[1])

#         if n_d[i] == 0:
#             continue
#         for j in range(int(n_d[i])):
#             c = j / n_d[i]
#             val = val + ((ZRi[i] - (c * ZDi[i])) / (Ri[i] - (c * Di[i])))
#         out[i] = val
#     return out


def efron_jac(n_d, Ri, ZRi, Di, ZDi, masked_array):
    """
    Vectorised implementation of term two of the efron ll
    jacobian.

    This implementation runs the risk of large memory usage
    given the 3D array that is created.
    """
    r = masked_array / n_d

    denom = Ri - Di * r
    denom = np.expand_dims(denom, axis=-1)

    r = np.expand_dims(r, axis=-1)
    numer = np.expand_dims(ZDi, axis=1) * r
    numer = np.expand_dims(ZRi, axis=1) - numer

    out = numer / denom
    out = out.sum(axis=1)
    return out


@njit
def efron_hess_jit(n_d, Ri, ZRi, Z2Ri, Di, ZDi, Z2Di, out):
    # TODO: Vectorised implementation like for the jacobian above
    # Not sure my brain can handle that many dimensions.
    for i in range(len(n_d)):
        val = np.zeros(out.shape[1:])
        if n_d[i] == 0:
            continue
        for j in range(int(n_d[i])):
            c = j / n_d[i]
            dRD = Ri[i] - c * Di[i]
            a = ZRi[i] - c * ZDi[i]
            # If einsum ever supported by numba, remove this
            if a.shape == (1,):
                a2 = a * a
            else:
                a2 = a @ a.reshape(-1, 1)
            val += (
                +dRD * (Z2Ri[i] - c * Z2Di[i])
                # Einsum not supported by numba
                # Change if this ever is the case.
                # - np.einsum('i, j -> ij', a, a)
                - a2
            ) / dRD**2
        out[i] = val
    return out


def at_risk_beta_Z(arr, n, gb_x):
    R = gb_x.sum(n * arr)[1]
    # Get the reverse cumulative sum
    return R[::-1].cumsum(axis=0)[::-1]


class CoxPH_:
    """
    Best reference I can find that covers all the
    possibilities for estimating betas
    http://www-personal.umich.edu/~yili/lect4notes.pdf
    """

    def baseline(self, beta, x, c, n, Z):
        """
        Uses the Breslow method to compute the baseline hazard.
        """
        unique_x = np.unique(x)

        d = np.zeros_like(unique_x)
        r = np.zeros_like(unique_x)

        for i, tau_i in enumerate(unique_x):
            mask_d_i = (x == tau_i) & (c == 0)
            d[i] = n[mask_d_i].sum()

            mask_at_risk_i = x >= tau_i
            Z_ri = Z[mask_at_risk_i, :]

            r[i] = np.exp(Z_ri @ beta).sum()

        return unique_x, r, d

    def create_efron_ll_jac_hess(
        self, x, Z, c, n, tl, with_jac=True, with_hess=True
    ):
        """
        The reference used to compute the jacobian and hessian
        was https://mathweb.ucsd.edu/~rxu/math284/slect5.pdf
        """
        # TODO: Incorporate left-truncation... somehow.
        # left truncation allows for implementation of time-varying covariates

        # To do left truncation. All one needs to do is to adjust the risk set.
        # I am aiming to implement this by:
        # Ri - TRi.. that's it!

        # Groupby object for repeated use
        gb_x = npi.group_by(x)
        gb_tl = npi.group_by(tl)
        n_d_x = np.where(c == 0, n, 0)
        n_d = gb_x.sum(n_d_x)[1]
        n_d_x = n_d_x.reshape(-1, 1)
        n = n.reshape(-1, 1)

        max_n = n_d.max()

        x_ = gb_x.unique
        x_tl = gb_tl.unique
        idx = np.searchsorted(x_, x_tl, side="right")

        def log_like(beta):
            beta_z = Z @ beta

            S_d = gb_x.sum(n_d_x * beta_z.reshape(-1, 1))[1].reshape(-1, 1)
            e_beta_z = np.exp(beta_z).reshape(-1, 1)

            x_, Ri = gb_x.sum(n * e_beta_z)
            trunc = gb_tl.sum(n * e_beta_z)[1]

            Ri = Ri[::-1].cumsum(axis=0)[::-1]

            trunc = trunc[::-1].cumsum(axis=0)[::-1] - trunc

            TRi = np.ones_like(Ri) * np.inf

            TRi[idx] = trunc
            TRi = np.minimum.accumulate(TRi)
            # Subtract the truncated values from the others.
            Ri = Ri - TRi

            Di = gb_x.sum(n_d_x * e_beta_z)[1]

            efron_denom = np.zeros_like(x_)
            efron_denom = efron_jit(n_d, Ri, Di, efron_denom)

            like = S_d.sum() - efron_denom.sum()
            return -like

        S_d = gb_x.sum(n_d_x * Z)[1]

        arr = np.repeat([np.arange(max_n)], len(n_d), axis=0)
        mask = 1 - (arr < n_d.reshape(-1, 1)).astype(int)

        masked_array = ma.array(arr, mask=mask)

        def jac_hess(beta):
            # This line troubled me for longer than I care
            # to admit. I was using n, but it is only the
            # number of deaths at each point, n_d_x

            # Z = Z
            Z2 = np.einsum("ij, ik -> ijk", Z, Z)

            # Only call this once.. Yay.
            beta_z = Z @ beta

            e_beta_z = np.exp(beta_z).reshape(-1, 1)
            z_e_beta_z = Z * e_beta_z
            z2_e_beta_z = np.einsum("ijk, ij -> ijk", Z2, n * e_beta_z)

            Ri = at_risk_beta_Z(e_beta_z, n, gb_x)
            ZRi = at_risk_beta_Z(z_e_beta_z, n, gb_x)
            Z2Ri = gb_x.sum(z2_e_beta_z)[1]
            Z2Ri = Z2Ri[::-1].cumsum(axis=0)[::-1]

            trunc = gb_tl.sum(n * e_beta_z)[1]
            trunc = trunc[::-1].cumsum(axis=0)[::-1] - trunc

            z_trunc = gb_tl.sum(n * z_e_beta_z)[1]
            z_trunc = z_trunc[::-1].cumsum(axis=0)[::-1] - z_trunc

            z2_trunc = gb_tl.sum(z2_e_beta_z)[1]
            z2_trunc = z2_trunc[::-1].cumsum(axis=0)[::-1] - z2_trunc

            TRi = np.ones_like(Ri) * np.inf
            ZTRi = np.ones_like(ZRi) * np.inf
            Z2TRi = np.ones_like(Z2Ri) * np.inf

            TRi[idx] = trunc
            TRi = np.minimum.accumulate(TRi)

            ZTRi[idx] = z_trunc
            ZTRi = np.minimum.accumulate(ZTRi)

            Z2TRi[idx] = z2_trunc
            Z2TRi = np.minimum.accumulate(Z2TRi)

            # Subtract the truncated values from the non-truncated risk set.
            Ri -= TRi
            ZRi -= ZTRi
            Z2Ri -= Z2TRi

            Di = gb_x.sum(n_d_x * e_beta_z)[1]
            ZDi = gb_x.sum(n_d_x * z_e_beta_z)[1]

            expected_S_d = np.zeros_like(S_d)
            expected_S_d = efron_jac(
                n_d.reshape(-1, 1), Ri, ZRi, Di, ZDi, masked_array
            )

            diff = S_d - expected_S_d
            jacobian = -diff.sum(axis=0)

            # Something remains incorrect with the Hessian
            Z2Di = np.einsum("ijk, ij -> ijk", Z2, n_d_x * e_beta_z)
            Z2Di = gb_x.sum(Z2Di)[1]

            hess_matrix = np.zeros_like(Z2)
            hess_matrix = efron_hess_jit(
                n_d, Ri, ZRi, Z2Ri, Di, ZDi, Z2Di, hess_matrix
            )
            hess_matrix = -hess_matrix.sum(axis=0)

            return jacobian  # , hess_matrix

        return log_like, jac_hess, False

    def create_breslow_ll_jac_hess(self, x, Z, c, n, tl):
        """
        The reference used to compute the jacobian and hessian
        was https://mathweb.ucsd.edu/~rxu/math284/slect5.pdf
        """
        # TODO: Incorporate left-truncation... somehow.
        # left truncation allows for implementation of time-varying covariates

        gb_x = npi.group_by(x)
        gb_tl = npi.group_by(tl)
        n_d_x = np.where(c == 0, n, 0)
        n_d = gb_x.sum(n_d_x)[1]
        n_d_x = n_d_x.reshape(-1, 1)
        n = n.reshape(-1, 1)

        x_ = gb_x.unique
        x_tl = gb_tl.unique
        idx = np.searchsorted(x_, x_tl, side="right")

        # Create the log_like function for the data
        def log_like(beta):
            beta_z = Z @ beta
            di_beta_z = gb_x.sum(n_d_x * beta_z.reshape(-1, 1))[1].reshape(
                -1, 1
            )
            e_beta_z = np.exp(beta_z).reshape(-1, 1)
            Ri = at_risk_beta_Z(e_beta_z, n, gb_x)

            trunc = gb_tl.sum(n * e_beta_z)[1]
            trunc = trunc[::-1].cumsum(axis=0)[::-1] - trunc

            TRi = np.ones_like(Ri) * np.inf

            TRi[idx] = trunc
            TRi = np.minimum.accumulate(TRi)

            Ri -= TRi

            Ri = np.log(Ri)
            Ri = n_d.reshape(-1, 1) * Ri

            like = di_beta_z - Ri

            return -like.sum()

        S_d = gb_x.sum(n_d_x.reshape(-1, 1) * Z)[1]

        def jac_hess(beta):
            Z2 = np.einsum("ij, ik -> ijk", Z, Z)

            # Only call this once.. Yay.
            beta_z = Z @ beta

            e_beta_z = np.exp(beta_z).reshape(-1, 1)
            z_e_beta_z = Z * e_beta_z
            z2_e_beta_z = np.einsum("ijk, ij -> ijk", Z2, n * e_beta_z)

            Ri = at_risk_beta_Z(e_beta_z, n, gb_x)
            ZRi = at_risk_beta_Z(z_e_beta_z, n, gb_x)
            Z2Ri = gb_x.sum(z2_e_beta_z)[1]
            Z2Ri = Z2Ri[::-1].cumsum(axis=0)[::-1]

            trunc = gb_tl.sum(n * e_beta_z)[1]
            trunc = trunc[::-1].cumsum(axis=0)[::-1] - trunc

            z_trunc = gb_tl.sum(n * z_e_beta_z)[1]
            z_trunc = z_trunc[::-1].cumsum(axis=0)[::-1] - z_trunc

            z2_trunc = gb_tl.sum(z2_e_beta_z)[1]
            z2_trunc = z2_trunc[::-1].cumsum(axis=0)[::-1] - z2_trunc

            TRi = np.ones_like(Ri) * np.inf
            ZTRi = np.ones_like(ZRi) * np.inf
            Z2TRi = np.ones_like(Z2Ri) * np.inf

            TRi[idx] = trunc
            TRi = np.minimum.accumulate(TRi)

            ZTRi[idx] = z_trunc
            ZTRi = np.minimum.accumulate(ZTRi)

            Z2TRi[idx] = z2_trunc
            Z2TRi = np.minimum.accumulate(Z2TRi)

            # Subtract the truncated values from the non-truncated risk set.
            Ri -= TRi
            ZRi -= ZTRi
            Z2Ri -= Z2TRi

            EZ = ZRi / Ri
            EZ = n_d.reshape(-1, 1) * EZ

            jacobian = -(S_d - EZ).sum(axis=0)

            # calc term 1
            term_1 = np.einsum("ijk,ij->ijk", Z2Ri, 1.0 / Ri)

            # calc term 2
            term_2 = ZRi / Ri
            term_2 = np.einsum("ij, ik-> ijk", term_2, term_2)

            # Compute the Hessian matrix
            hess_matrix = term_1 - term_2
            hess_matrix = np.einsum("ijk,i->ijk", hess_matrix, n_d.flatten())
            hess_matrix = hess_matrix.sum(axis=0)

            return jacobian, hess_matrix

        return log_like, jac_hess, True

    def fit(self, x, Z, c=None, n=None, tl=None, method="breslow", tol=1e-10):
        """
        Need to add shape checking of x, c, n
        """
        x, c, n, tl, Z = validate_coxph(x, c, n, Z, tl, method)

        # Good initial guess assumes no impact
        beta_init = np.zeros(Z.shape[1])

        if method == "efron":
            func_generator = self.create_efron_ll_jac_hess
        elif method == "breslow":
            func_generator = self.create_breslow_ll_jac_hess
        # TODO: Cox-Exact
        # TODO: K&P
        # See: https://mathweb.ucsd.edu/~rxu/math284/slect5.pdf

        neg_ll, jac, hess = func_generator(x, Z, c, n, tl)

        # Have found that root finding is faster than minimization
        res = root(jac, beta_init, jac=hess, tol=tol)

        # The hessian is at [1] of the jac function
        if hess:
            # Finds the p-value for the null hypothesis
            # that the coefficient is 0.
            hessian_matrix = jac(res.x)[1]
            var = np.diag(inv(hessian_matrix))
            # Use the pseudo-inverse if the hessian does not have a
            # diagonal that is all positive.
            if np.any(var <= 0):
                var = np.diag(pinv(hessian_matrix))
            z_score = res.x / np.sqrt(var)
            p_values = 2 * (1 - norm.cdf(np.abs(z_score)))
        else:
            p_values = None

        model = Regression()
        model = ProportionalHazardsModel("Cox", "Semi-Parametric")
        model._neg_log_like = neg_ll(res.x)
        model.p_values = p_values
        model.neg_ll = neg_ll
        model.jac = jac
        model.hess = hess
        model.tie_method = method
        model.baseline_method = "breslow"
        model.res = res
        model.beta = copy(res.x)
        model.phi = lambda Z: np.exp(Z @ model.beta)
        model.params = res.x

        x, r, d = self.baseline(model.beta, x, c, n, Z)
        model.x = x
        model.r = r
        model.d = d
        model.tl = tl
        model.h0 = d / r
        model.H0 = model.h0.cumsum()

        return model

    def fit_from_df(
        self,
        df,
        x_col,
        Z_cols=None,
        c_col=None,
        n_col=None,
        formula=None,
        method="efron",
    ):
        x, c, n, Z, form = validate_coxph_df_inputs(
            df, x_col, c_col, n_col, Z_cols, formula
        )

        model = self.fit(x, Z, c, n, method=method)
        model.formula = form

        return model


CoxPH = CoxPH_()
