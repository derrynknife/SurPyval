"""
This code was created for and sponsored by Cartiga (www.cartiga.com).
Cartiga makes no representations or warranties in connection with the code
and waives any and all liability in connection therewith. Your use of the
code constitutes acceptance of these terms.

Copyright 2022 Cartiga LLC
"""

import numpy as np

from surpyval.univariate.regression import CoxPH
from surpyval.utils import (
    _get_idx,
    validate_fine_gray_inputs,
    wrangle_and_check_form_and_Z_cols,
)

from .fine_gray import FineGray, _step


class CompetingRisksProportionalHazards:
    """
    Competing-risks proportional-hazards regression.

    Fits either a cause-specific proportional-hazards model (``how="Cox"``,
    one Cox model per cause with the other causes treated as censored) or a
    Fine-Gray subdistribution-hazards model (``how="Fine-Gray"``). The naming
    follows the package convention (compare ``CompetingRisks`` and
    ``ProportionalHazards``).

    TODO: Time-Varying Implementation
    """

    def _fg_model(self, event):
        # Resolve the per-cause Fine-Gray subdistribution model, requiring an
        # explicit cause (the Fine-Gray CIF is defined one cause at a time).
        if event is None:
            raise ValueError(
                "A Fine-Gray model predicts one cause at a time; pass `event`."
            )
        if event not in self._fg_models:
            raise ValueError("Unrecognised event type for this model")
        return self._fg_models[event]

    def _f(self, arr, x, Z, event=None, interp="step"):
        idx, rev = _get_idx(self.x, x)

        if event is not None:
            if event not in self.event_idx_map:
                raise ValueError("Unrecognised event type for this model")
            e_i = self.event_idx_map[event]
            return (arr[e_i, idx] * self.phi_e(Z, e_i))[rev]

        # All causes combined: each cause contributes with its OWN
        # coefficients, so the all-cause (cumulative) hazard is the sum of
        # H0_e(t) * exp(beta_e'Z), not a single summed-coefficient term.
        total = sum(
            arr[e_i, idx] * self.phi_e(Z, e_i)
            for e_i in self.event_idx_map.values()
        )
        return total[rev]

    def hf(self, x, Z, event=None, interp="step"):
        if self.how == "Fine-Gray":
            raise ValueError(
                "The Fine-Gray subdistribution hazard has no pointwise "
                "density from the step baseline; use `cif` or `Hf`."
            )
        return self._f(self.h0_e, x, Z, event=event, interp=interp)

    def Hf(self, x, Z, event=None, interp="step"):
        if self.how == "Fine-Gray":
            # Cumulative subdistribution hazard H0_k(x) * exp(beta'Z) = -log S.
            return -np.log(self.sf(x, Z, event=event))
        return self._f(self.H0_e, x, Z, event=event, interp=interp)

    def sf(self, x, Z, event=None, interp="step"):
        if self.how == "Fine-Gray":
            return self._fg_model(event).sf(x, Z)
        return np.exp(-self.Hf(x, Z, event=event, interp=interp))

    def ff(self, x, Z, event=None, interp="step"):
        if self.how == "Fine-Gray":
            return self.cif(x, Z, event)
        return 1 - self.sf(x, Z, event=event, interp=interp)

    def df(self, x, Z, event=None, interp="step"):
        if self.how == "Fine-Gray":
            raise ValueError(
                "The Fine-Gray subdistribution density has no pointwise form "
                "from the step baseline; use `cif`."
            )
        return self.hf(x, Z, event=event, interp=interp) * self.sf(
            x, Z, event=event, interp=interp
        )

    def cif(self, x, Z, event):
        if self.how == "Fine-Gray":
            # Direct subdistribution CIF: 1 - exp(-H0_k(x) exp(beta'Z)).
            return self._fg_model(event).cif(x, Z)

        # Cause-specific CIF: integrate this cause's hazard against the
        # all-cause survival. Index and reverse index in case x is unordered.
        idx, rev = _get_idx(self.x, x)

        lambda_e = self.hf(self.x, Z, event)
        S = self.sf(self.x, Z)
        # iif = instantaneous incidence function
        iif = lambda_e * S
        cif = iif.cumsum()

        return cif[idx][rev]

    @classmethod
    def fit_from_df(
        cls,
        df,
        x_col,
        e_col,
        Z_cols=None,
        c_col=None,
        n_col=None,
        formula=None,
        how="Cox",
        tie_method="efron",
    ):
        """
        Fit a competing-risks proportional-hazards model from a pandas
        DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The data.
        x_col : str
            Column of observed times.
        e_col : str
            Column of event-type (cause) labels. Use ``None`` (or a blank/NaN
            cell) for a censored observation.
        Z_cols : str or list of str, optional
            Covariate columns. Either ``Z_cols`` or ``formula`` must be given.
        c_col : str, optional
            Column of censoring flags (0 observed, 1 right-censored).
        n_col : str, optional
            Column of counts per row.
        formula : str, optional
            A patsy/formulaic formula for the covariates, as an alternative to
            ``Z_cols``.
        how : {'Cox', 'Fine-Gray'}, optional
            Cause-specific proportional hazards or Fine-Gray subdistribution
            hazards. Default 'Cox'.
        tie_method : {'efron', 'breslow'}, optional
            Tie handling for the ``how='Cox'`` path. Default 'efron'.

        Returns
        -------
        CompetingRisksProportionalHazards
            The fitted model. Predictions still take a covariate array ``Z``.
        """
        import pandas as pd

        Z, mask, form, feature_names, model_spec = (
            wrangle_and_check_form_and_Z_cols(Z_cols, formula, df)
        )
        sub = df.loc[mask]
        x = sub[x_col].values
        # A censored row's cause is ``None``; accept a blank/NaN cell for it.
        e = np.array(
            [None if pd.isna(v) else v for v in sub[e_col].values],
            dtype=object,
        )
        c = sub[c_col].values if c_col is not None else None
        n = sub[n_col].values if n_col is not None else None

        model = cls.fit(x, Z, e, c=c, n=n, how=how, tie_method=tie_method)
        model.formula = form
        model.feature_names = feature_names
        model._model_spec = model_spec
        return model

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

        model : CompetingRisksProportionalHazards
            A competing-risks proportional-hazards model with fitted params
            and helper methods using the fitted params.

        Examples
        --------
        >>> from surpyval.univariate.competing_risks import (
        ...     CompetingRisksProportionalHazards,
        ... )

        """
        x, Z, e, c, n = validate_fine_gray_inputs(x, Z, e, c, n)

        unique_e = set(e)
        if None in unique_e:
            unique_e.remove(None)

        n_event_types = len(unique_e)

        event_idx_map = {state: i for i, state in enumerate(unique_e)}

        betas = np.zeros((len(unique_e), Z.shape[1]))
        unique_x = np.unique(x)

        baselines = np.zeros((len(unique_e), len(unique_x)))
        # Best initial assumption is to assume there is no risk
        # beta_init = np.zeros(Z.shape[1])

        model = cls()
        model.n_event_types = n_event_types
        model.event_idx_map = event_idx_map
        model.how = how

        if how == "Cox":
            # Cause-specific proportional hazards: one Cox model per cause,
            # treating every other cause (and censoring) as right-censored.
            results = []
            for i, event in enumerate(unique_e):
                c_e = np.where(e == event, 0, 1)
                cox_model = CoxPH.fit(x, Z, c_e, n, method=tie_method)

                results.append(cox_model.res)
                betas[i, :] = cox_model.res.x
                # Cause-specific baseline hazard: reuse the fitted Cox model's
                # own Breslow baseline, which is built from c_e (the
                # cause-specific event indicator) and the standard risk set.
                # Map its cumulative hazard onto the shared unique_x grid and
                # store increments so H0_e = baselines.cumsum stays coherent.
                H_grid = _step(cox_model.x, cox_model.H0, unique_x, before=0.0)
                baselines[i, :] = np.diff(H_grid, prepend=0.0)

        elif how == "Fine-Gray":
            # Delegate to the IPCW Fine-Gray fitter, one subdistribution model
            # per cause. The authoritative predictions come from these models
            # (see ``_fg_models`` and the ``cif``/``sf`` branches below); the
            # ``baselines`` grid is filled with each cause's cumulative
            # subdistribution hazard for a coherent ``H0_e``.
            fg_models = {}
            results = []
            for i, event in enumerate(unique_e):
                fg = FineGray.fit(x, Z, e, c=c, n=n, cause=event)
                fg_models[event] = fg
                results.append(fg.res)
                betas[i, :] = fg.beta
                # Store increments so the shared ``H0_e = baselines.cumsum``
                # equals this cause's cumulative subdistribution hazard.
                H_grid = _step(fg._times, fg._cumhaz, unique_x, before=0.0)
                baselines[i, :] = np.diff(H_grid, prepend=0.0)
            model._fg_models = fg_models
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
