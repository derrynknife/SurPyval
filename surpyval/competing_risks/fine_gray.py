"""
This code was created for and sponsored by Cartiga (www.cartiga.com).
Cartiga makes no representations or warranties in connection with the code
and waives any and all liability in connection therewith. Your use of the
code constitutes acceptance of these terms.

Copyright 2022 Cartiga LLC
"""

import numpy as np

# import numpy_indexed as npi
from scipy.optimize import minimize

from surpyval.utils import validate_fine_gray_inputs

# from numba import njit
# from scipy.optimize import root

# TODO: Improve this so that it uses root and the jac and hess.
# TODO: Implement it with time-varying covariates


class FineGray_:
    # def create_breslow_ll_jac_hess(
    #     self, x, Z, c, n, e, mask, with_jac=True, with_hess=True
    # ):
    #     """
    #     The reference used to compute the jacobian and hessian
    #     was https://mathweb.ucsd.edu/~rxu/math284/slect5.pdf
    #     """
    #     # TODO: Incorporate left-truncation... somehow.
    #     #left truncation allows for implementation of time-varying covariates

    #     x_e, c_e, n_e = (arr[mask] for arr in (x, c, n))
    #     Z_e = Z[mask, :]

    #     x_fg, c_fg, n_fg = (arr[~mask] for arr in (x, c, n))
    #     Z_fg = Z[~mask, :]

    #     gb_x = npi.group_by(x_e)

    #     n_d_x = np.where(c_e == 0, n_e, 0)
    #     x_e_unique, n_d = gb_x.sum(n_d_x)
    #     n_d = n_d.reshape(-1, 1)
    #     n_e = n_e.reshape(-1, 1)

    #     # Create the log_like function for the data
    #     def log_like(beta):
    #         beta_z = Z_e @ beta
    #         di_beta_z = gb_x.sum(n_d_x * beta_z)[1].reshape(-1, 1)

    #         e_beta_z = np.exp(beta_z).reshape(-1, 1)
    #         Ri = gb_x.sum(n_e * e_beta_z)[1]
    #         Ri = Ri[::-1].cumsum()[::-1].reshape(-1, 1)
    #         # Add fine gray risk set
    #         Ri_e = n_fg * np.exp(Z_fg @ beta)

    #         Ri += Ri_e.sum()
    #         Ri = np.log(Ri)

    #         Ri = n_d * Ri

    #         like = di_beta_z - Ri

    #         return -like.sum()

    #     if with_jac:
    #         S_d = gb_x.sum(n_d_x.reshape(-1, 1) * Z)[1]

    #         def jac(beta):
    #             e_beta_z = np.exp(Z @ beta).reshape(-1, 1)
    #             z_e_beta_z = Z * e_beta_z

    #             Ri = gb_x.sum(n * e_beta_z)[1]
    #             Ri = Ri[::-1].cumsum(axis=0)[::-1]
    #             Ri += Ri_e

    #             ZRi = gb_x.sum(n * z_e_beta_z)[1]
    #             ZRi = ZRi[::-1].cumsum(axis=0)[::-1]
    #             ZRi += ZRi_e

    #             EZ = ZRi / Ri

    #             EZ = n_d * EZ

    #             out = S_d - EZ
    #             return -out.sum(axis=0)

    #     else:
    #         jac = None

    #     if with_hess:

    #         def hess(beta):
    #             # denominator for both terms
    #             e_beta_z = np.exp(Z @ beta).reshape(-1, 1)
    #             denom = at_risk_beta_Z(e_beta_z, n, gb_x)

    #             # calc term 1
    #             Z2 = np.einsum("ij, ik-> ijk", Z, Z)

    #             term_1 = np.einsum(
    #                 "ijk,ij->ijk", Z2, n.reshape(-1, 1) * e_beta_z
    #             )
    #             term_1 = gb_x.sum(term_1)[1]
    #             term_1 = term_1[::-1].cumsum(axis=0)[::-1]
    #             term_1 = np.einsum("ijk,ij->ijk", term_1, 1.0 / denom)

    #             # calc term 2
    #             z_beta_z = Z * e_beta_z
    #             term_2 = at_risk_beta_Z(z_beta_z, n, gb_x)
    #             term_2 = term_2 / denom
    #             term_2 = np.einsum("ij, ik-> ijk", term_2, term_2)

    #             # Compute the Hessian matrix
    #             hess_matrix = term_1 - term_2
    #             hess_matrix = np.einsum(
    #                 "ijk,i->ijk", hess_matrix, n_d.flatten()
    #             )
    #             hess_matrix = hess_matrix.sum(axis=0)
    #             return hess_matrix

    #     else:
    #         hess = None

    #     return log_like, jac, hess

    def fit(self, x, Z, e, c=None, n=None):
        x, Z, e, c, n = validate_fine_gray_inputs(x, Z, e, c, n)

        unique_e = list(set(e))
        if None in unique_e:
            unique_e.remove(None)

        # Best initial assumption is to assume there is no risk
        beta_init = np.zeros(Z.shape[1])

        results = []
        for i, event in enumerate(unique_e):
            mask = e == event
            neg_ll, jac, hess = self.create_breslow_ll_jac_hess(
                x, Z, c, n, e, mask, with_jac=False, with_hess=False
            )

            # Have found that root finding is faster than minimization!
            # res = root(jac, beta_init, jac=hess, tol=tol)
            # fun = lambda beta: -self.partial_log_like(beta, x, c_e, n, Z, e,
            # event)
            res = minimize(neg_ll, beta_init)

            results.append(res)

        return results, unique_e


FineGray = FineGray_()
