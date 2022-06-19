from cartiga import np
import surpyval as surv

from cartiga.utils import (
    _get_idx,
    validate_cr_inputs,
    validate_cr_df_inputs,
    validate_event,
    validate_cif_event
)
import textwrap

from surpyval.nonparametric.nelson_aalen import na
from surpyval.nonparametric.kaplan_meier import km

class CompetingRisks():
    def __repr__(self):
        out = """\
        Competing Risk model with events:
        {events}
        """.format(events=list(self.event_idx_map.keys()))
        return textwrap.dedent(out)


    def _f(self, f, x, event):
        validate_event(self.event_idx_map, event)
        idx, rev = _get_idx(self.x, x)

        if f == 'h':
            arr = self.h0_e
        elif f == 'H':
            arr = self.H0_e
        elif f == "IIF":
            arr = self.IIF
        elif f =="CIF":
            arr = self.CIF

        if event is None:
            return arr.sum(axis=0)[idx][rev]
        else:
            e = self.event_idx_map[event]
            return arr[e, idx][rev]

    def hf(self, x, event=None):
        return self._f('h', x, event)

    def Hf(self, x, event=None):
        return self._f('H', x, event)

    def sf(self, x, event=None):
        return np.exp(-self.Hf(x, event=event))

    def ff(self, x, event=None):
        """
        A lot of commentary about this being difficult to interpret.
        In engineering this is not the case, eliminating the failure
        will result in the ff being gone.
        """
        return 1 - np.exp(-self.Hf(x, event=event))

    def df(self, x, event=None):
        return self.hf(x, event=event) * self.sf(x, event=event)

    def iif(self, x, event):
        """
        Instantaneous Incidence Function
        """
        validate_cif_event(event)
        return self._f('IIF', x, event)

    def cif(self, x, event):
        """
        Cumulative Incidence Function
        """
        validate_cif_event(event)

        return self._f('CIF', x, event)

    @classmethod
    def fit_from_df(cls, df, x_col, e_col, c_col=None, 
                    n_col=None, method="Nelson-Aalen"):

        x, c, n, e = validate_cr_df_inputs(df, x_col, e_col, 
                                           c_col, n_col)
        model = cls.fit(x, e, c, n, method)
        model.df = df
        return model

    @classmethod
    def fit(cls, x, e, c=None, n=None, method="Nelson-Aalen"):
        """
        Need to check that causes is the same length
        TODO: FlemingHarrington baseline.
        """
        x, c, n, e = validate_cr_inputs(x, c, n, e, method)

        # Get unique event types
        unique_e = set(e)
        # Remove None type, which relates to censored obs
        # np.unique doesn't work since it can't handle None
        if None in unique_e:
            unique_e.remove(None)

        # Count number of unique event types.
        # Ordering is stable over repeats due to sort.
        n_event_types = len(unique_e)
        event_idx_map = {state : i for i, state in enumerate(sorted(unique_e))}

        # Get the x, r, d format agnostic of event.
        unique_x, r, d = surv.xcn_to_xrd(x, c, n)

        # empty count array of occurrence (e) of amount (d) at time (x)
        d_e = np.zeros((n_event_types, len(unique_x)))

        # Counter for each occurrence
        for i, x_i in enumerate(x):
            if c[i] == 1:
                continue
            j = event_idx_map[e[i]]
            d_e[j, np.where(unique_x == x_i)] += n[i]

        if method == "Nelson-Aalen":
            S = na(r, d)
        elif method == "Kaplan-Meier":
            S = km(r, d)

        # Useful object to return to user
        model = cls()
        model.n_event_types = n_event_types
        model.event_idx_map = event_idx_map

        # Store relevant data to object
        model.x = unique_x
        model.d = d
        model.r = r
        model.h0 = d / r
        model.H0 = model.h0.cumsum()
        model.S = S
        model.d_e = d_e
        model.h0_e = d_e / r
        model.H0_e = model.h0_e.cumsum(axis=1)
        model.IIF = model.S * model.h0_e
        model.CIF = model.IIF.cumsum(axis=1)
        return model