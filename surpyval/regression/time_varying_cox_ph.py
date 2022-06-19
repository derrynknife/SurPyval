from cartiga import np
from cartiga.utils import validate_tv_coxph_df_inputs
from . import CoxPH

class TimeVaryingCoxPH_():
    def fit_from_df(
        self, df, id_col, tl_col, x_col,
        Z_cols=None, c_col=None, n_col=None, 
        formula=None, method="efron"):

        tl, x, Z, id, c, n, form = validate_tv_coxph_df_inputs(
            df, id_col,
            tl_col, x_col,
            Z_cols,
            c_col, n_col,
            formula
        )

        model = CoxPH.fit(tl, x, Z, c, n, method=method)
        model.formula = form

        return model

TimeVaryingCoxPH = TimeVaryingCoxPH_()