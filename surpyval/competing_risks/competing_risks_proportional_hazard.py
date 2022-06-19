# from cartiga import np
import numpy as np
from cartiga.competing_risks.fine_gray import FineGray
from cartiga.regression.cox_ph import CoxPH
from cartiga.utils import _get_idx
from cartiga.utils import validate_fine_gray_inputs, _scale
from scipy.optimize import minimize

class CompetingRiskProportionalHazard():
    """
    TODO: Time-Varying Implementation
    TODO: Change this to SemiParametricCompetingRiskProportionalHazard ??
    """

    def _f(self, arr, x, Z, event=None, interp='step'):
        idx, rev = _get_idx(self.x, x)

        if event is not None and event not in self.event_idx_map:
            raise ValueError("Unrecognised event type for this model")

        if event is None:
            return (arr.sum(axis=0)[idx] * self.phi(Z))[rev]
        else:
            e_i = self.event_idx_map.get(event, None)
            return (arr[e_i, idx] * self.phi_e(Z, e_i))[rev]

    def hf(self, x, Z, event=None, interp='step'):
        return self._f(self.h0_e, x, Z, event=event, interp=interp)

    def Hf(self, x, Z, event=None, interp='step'):
        return self._f(self.H0_e, x, Z, event=event, interp=interp)

    def sf(self, x, Z, event=None, interp='step'):
        return np.exp(-self.Hf(x, Z, event=event, interp=interp))

    def ff(self, x, Z, event=None, interp='step'):
        return 1 - self.sf(x, Z, event=event, interp=interp)

    def df(self, x, Z, event=None, interp='step'):
        return (self.hf(x, Z, event=event, interp=interp)
                * self.sf(x, Z, event=event, interp=interp))

    def cif(self, x, Z, event):
        # Index and reverse index
        # in case x is not in order.
        idx, rev = _get_idx(self.x, x)

        # CIF
        lambda_e = self.hf(self.x, Z, event)
        S = self.sf(self.x, Z)
        # iif = instantaneous incidence function
        iif = lambda_e * S
        cif = iif.cumsum()
        
        return cif[idx][rev]

    
    @classmethod
    def cox_risk_set_indices(cls, x_i, e_i, x, e):
        return (x >= x_i)

    @classmethod
    def fine_gray_risk_set_indices(cls, x_i, e_i, x, e):
        return ((x >= x_i) | (e != e_i))

    @classmethod
    def partial_log_like(cls, beta, x, c, n, Z, e, 
                         event, how='Cox', scale=False):
        """
        This is the Breslow implementation
        TODO: 
        - Efron, and
        - Kalbfleisch and Prentice (This is what we need!)
        """
        # print(beta)
        if how == 'Cox':
            risk_set_indices = cls.cox_risk_set_indices
        elif how == 'Fine-Gray':
            risk_set_indices = cls.fine_gray_risk_set_indices
        N = len(x)

        ll = np.zeros_like(x)
        idx = np.array(range(N))

        for i, x_i in enumerate(x):
            if c[i] == 1:
                continue
            elif e[i] != event:
                continue
            # This is the key insight: 
            # all non e failures are still at risk!
            at_risk = risk_set_indices(x_i, event, x, e)
            Z_ri = Z[at_risk, :]
            # Sum of 'Z's is sometimes also called 'S'
            # e.g. see https://myweb.uiowa.edu/pbreheny/7210/f15/notes/11-5.pdf
            Z_i = n[i] * Z[i, :]
            # Breslow log-like
            ll_i = (Z_i @ beta) - (n[i] * np.log(np.exp(Z_ri @ beta).sum()))
            ll = np.where(i == idx, ll_i, ll)

        return _scale(ll, n, scale)

    @classmethod
    def baseline(cls, beta, x, c, n, Z, e, event):
        unique_x = np.unique(x)

        d = np.zeros_like(unique_x)
        r = np.zeros_like(unique_x)
        for i, tau_i in enumerate(unique_x):
            mask_d_i = (x == tau_i) & (c == 0)
            d[i] = n[mask_d_i].sum()

            mask_at_risk_i = (x >= tau_i) | (e != event)
            Z_ri = Z[mask_at_risk_i, :]

            r[i] = np.exp(Z_ri @ beta).sum()

        return d / r

    @classmethod
    def fit_from_df(self, *args, **kwargs):
        # TODO: Finish this
        raise NotImplementedError("Not yet...")

    @classmethod
    def fit(cls, x, Z, e, c=None, n=None, how="Cox", tie_method="efron"):
        r"""
        This function aimed to have an API to mimic the simplicity 
        of the scipy API. That is, to use a simple :code:`fit()` call, 
        with as many or as few parameters as are needed.

        API is plaigiarised from surpyval (which I also authored :) )

        Parameters
        ----------

        x : array like
            Array of observations of the random variables.

        Z : ndarray like
            Array of covariates for each random variable, x.
        
        e : array like
            Array of events that corresponds to each x.

        c : array like, optional
            Array of censoring flag. -1 is left censored, 0 is observed, 1 is
            right censored, and 2 is intervally censored. Only right censored
            data is implemented in FineGray. In not provided assumes each x
            was fully observed, i.e. c is automatically set to 0 for all x.

        n : array like, optional
            Array of counts for each x. If data is proivded as counts, then
            this can be provided. If :code:`None` will assume each
            observation is 1.

        method : str, optional
            String which declares method which is used to estimate the 
            baseline survival function. Can be either 'Nelson-Aalen' or
            'Kaplan-Meier'. Default is 'Nelson-Aalen'.

        Returns
        -------

        model : CompetingRiskProportionalHazard
            A Competing Risk Proportional Hazard model with fitted params
            and helper methods using the fitted params.

        Examples
        --------
        >>> from cartiga import CRPH
        
        """
        x, Z, e, c, n = validate_fine_gray_inputs(x, Z, e, c, n)

        unique_e = set(e)
        if None in unique_e:
            unique_e.remove(None)

        n_event_types = len(unique_e)

        event_idx_map = {state : i for i, state in enumerate(unique_e)}

        betas = np.zeros((len(unique_e), Z.shape[1]))
        unique_x = np.unique(x)

        baselines = np.zeros((len(unique_e), len(unique_x)))
        # Best initial assumption is to assume there is no risk
        # beta_init = np.zeros(Z.shape[1])

        model = cls()
        model.n_event_types = n_event_types
        model.event_idx_map = event_idx_map
        model.how = how

        if how == 'Cox':
            # The Cox method is just assuming the other methods
            # are right censored.
            results = []
            for i, event in enumerate(unique_e):
                c_e = np.where(e == event, 0, 1)
            
                res = CoxPH.fit(x, Z, c_e, n,
                                method=tie_method,
                                with_hess=False).res
                                
                results.append(res)
                betas[i, :] = res.x
                baselines[i, :] = cls.baseline(res.x, x, c, n, Z, e, event)

        elif how == 'Fine-Gray':
            results, unique_e = FineGray.fit(x, Z, e, c, n)
            for i, res in enumerate(results):
                betas[i, :] = res.x
                baselines[i, :] = cls.baseline(res.x, x, c, n, Z, e, unique_e[i])
        else:
            raise ValueError("`how` must be either 'Cox' or 'Fine-Gray")

        model.results = results
        model.betas = betas
        model.beta = betas.sum(axis=0)
        model.phi_e = lambda Z, e_i: np.exp(Z @ model.betas[e_i, :])
        model.phi = lambda Z: np.exp(Z @ model.beta)
        model.h0_e = baselines
        model.H0_e = baselines.cumsum(axis=1)
        model.x = unique_x
        return model

CRPH = CompetingRiskProportionalHazard