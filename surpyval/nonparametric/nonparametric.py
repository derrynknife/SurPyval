import numpy as np
from surpyval import nonparametric as nonp
from scipy.stats import t, norm
from .kaplan_meier import KaplanMeier
from .nelson_aalen import NelsonAalen
from .fleming_harrington import FlemingHarrington_

from scipy.interpolate import interp1d
from autograd import jacobian

import matplotlib.pyplot as plt
import pandas as pd

def interp_function(x, y, kind):
    return interp1d(x, y, kind=kind,
                    bounds_error=False,
                    fill_value=np.nan)

class NonParametric():
    """
    Result of ``.fit()`` method for every non-parametric
    surpyval distribution. This means that each of the
    methods in this class can be called with a model created
    from the ``NelsonAalen``, ``KaplanMeier``,
    ``FlemingHarrington``, or ``Turnbull`` estimators.
    """
    def __repr__(self):
        out = ('Non-Parametric SurPyval Model'
               + '\n============================='
               + '\nModel            : {dist}'
               ).format(dist=self.model)

        if hasattr(self, 'data'):
            if 'estimator' in self.data:
                out += '\nEstimator        : {turnbull}'.format(
                    turnbull=self.data['estimator'])

        return out

    def sf(self, x, interp='step'):
        r"""

        Surival (or Reliability) function with the
        non-parametric estimates from the data.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which t
            he survival function will be calculated.

        Returns
        -------

        sf : scalar or numpy array
            The value(s) of the survival function at each x


        Examples
        --------
        >>> from surpyval import NelsonAalen
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> model = NelsonAalen.fit(x)
        >>> model.sf(2)
        array([0.63762815])
        >>> model.sf([1., 1.5, 2., 2.5])
        array([0.81873075, 0.81873075, 0.63762815, 0.63762815])
        """
        x = np.atleast_1d(x)
        idx = np.argsort(x)
        rev = np.argsort(idx)
        x = x[idx]
        if interp == 'step':
            idx = np.searchsorted(self.x, x, side='right') - 1
            R = self.R[idx]
            R = np.where(idx < 0, 1, R)
            R = np.where(np.isposinf(x), 0, R)
        else:
            # R = np.hstack([[1], self.R])
            # x_data = np.hstack([[0], self.x])
            # R = np.interp(x, x_data, R)
            R = interp_function(self.x, self.R, kind=interp)(x)
            # R[np.where(x > self.x.max())] = np.nan
        return R[rev]

    def ff(self, x, interp='step'):
        r"""

        CDF (failure or unreliability) function with the
        non-parametric estimates from the data

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which
            the survival function will be calculated.

        Returns
        -------

        ff : scalar or numpy array
            The value(s) of the failure function at each x


        Examples
        --------
        >>> from surpyval import NelsonAalen
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> model = NelsonAalen.fit(x)
        >>> model.ff(2)
        array([0.36237185])
        >>> model.ff([1., 1.5, 2., 2.5])
        array([0.18126925, 0.18126925, 0.36237185, 0.36237185])
        """
        return 1 - self.sf(x, interp=interp)

    def hf(self, x, interp='step'):
        r"""

        Instantaneous hazard function with the non-parametric
        estimates from the data. This is calculated using simply
        the difference between consecutive H(x).

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which
            the survival function will be calculated

        Returns
        -------

        hf : scalar or numpy array
            The value(s) of the failure function at each x


        Examples
        --------
        >>> from surpyval import NelsonAalen
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> model = NelsonAalen.fit(x)
        >>> model.ff(2)
        array([0.36237185])
        >>> model.ff([1., 1.5, 2., 2.5])
        array([0.18126925, 0.18126925, 0.36237185, 0.36237185])
        """
        idx = np.argsort(x)
        rev = np.argsort(idx)
        x = x[idx]
        hf = np.diff(np.hstack([self.Hf(x[0], interp=interp),
                                self.Hf(x, interp=interp)]))
        hf[0] = hf[1]
        hf = pd.Series(hf)
        hf[hf == 0] = np.nan
        hf = hf.ffill().values
        return hf[rev]

    def df(self, x, interp='step'):
        r"""

        Density function with the non-parametric estimates
        from the data. This is calculated using the relationship
        between the hazard function and the density:

        .. math::
            f(x) = h(x)e^{-H(x)}

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the
            survival function will be calculated

        Returns
        -------

        df : scalar or numpy array
            The value(s) of the density function at x


        Examples
        --------
        >>> from surpyval import NelsonAalen
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> model = NelsonAalen.fit(x)
        >>> model.df(2)
        array([0.28693267])
        >>> model.df([1., 1.5, 2., 2.5])
        array([0.16374615, 0.        , 0.15940704, 0.        ])
        """
        return self.hf(x, interp=interp) * np.exp(-self.Hf(x, interp=interp))

    def Hf(self, x, interp='step'):
        r"""

        Cumulative hazard rate with the non-parametric estimates
        from the data. This is calculated using the relationship
        between the hazard function and the density:

        .. math::
            H(x) = -\ln (R(x))

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the
            function will be calculated.

        Returns
        -------

        Hf : scalar or numpy array
            The value(s) of the density function at x

        Examples
        --------
        >>> from surpyval import NelsonAalen
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> model = NelsonAalen.fit(x)
        >>> model.Hf(2)
        array([0.45])
        >>> model.df([1., 1.5, 2., 2.5])
        model.Hf([1., 1.5, 2., 2.5])
        """
        return -np.log(self.sf(x, interp=interp))

    def cb(self, x, on='sf', bound='two-sided', interp='step',
           alpha_ci=0.05, bound_type='exp', dist='z'):
        r"""

        Confidence bounds of the ``on`` function at the
        ``alpa_ci`` level of significance. Can be the upper,
        lower, or two-sided confidence by changing value of ``bound``.
        Can change the bound type to be regular or exponential
        using either the 't' or 'z' statistic.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the confidence bounds
            will be calculated
        on : ('sf', 'ff', 'Hf'), optional
            The function on which the confidence bound will be calculated.
        bound : ('two-sided', 'upper', 'lower'), str, optional
            Compute either the two-sided, upper or lower confidence bound(s).
            Defaults to two-sided.
        interp : ('step', 'linear', 'cubic'), optional
            How to interpolate the values between observations. Survival
            statistics traditionally uses step functions, but can use
            interpolated values if desired. Defaults to step.
        alpha_ci : scalar, optional
            The level of significance at which the bound will be computed.
        bound_type : ('exp', 'regular'), str, optional
            The method with which the bounds will be calculated. Using regular
            will allow for the bounds to exceed 1 or be less than 0. Defaults
            to exp as this ensures the bounds are within 0 and 1.
        dist : ('t', 'z'), str, optional
            The statistic to use in calculating the bounds (student-t or
            normal). Defaults to the normal (z).

        Returns
        -------

        cb : scalar or numpy array
            The value(s) of the upper, lower, or both confidence bound(s) of
            the selected function at x

        Examples
        --------
        >>> from surpyval import NelsonAalen
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> model = NelsonAalen.fit(x)
        >>> model.cb([1., 1.5, 2., 2.5], bound='lower', dist='t')
        array([0.11434813, 0.11434813, 0.04794404, 0.04794404])
        >>> model.cb([1., 1.5, 2., 2.5])
        array([[0.97789387, 0.16706394],
               [0.97789387, 0.16706394],
               [0.91235117, 0.10996882],
               [0.91235117, 0.10996882]])

        References
        ----------

        http://reliawiki.org/index.php/Non-Parametric_Life_Data_Analysis

        """
        if on in ['df', 'hf']:
            raise ValueError("NonParametric cannot do confidence bounds on "
                             + "density or hazard rate functions. Try Hf, "
                             + "ff, or sf")

        old_err_state = np.seterr(all='ignore')

        cb = self.R_cb(x,
                       bound=bound,
                       interp=interp,
                       alpha_ci=alpha_ci,
                       bound_type=bound_type,
                       dist=dist
                       )

        if (on == 'ff') or (on == 'F'):
            cb = 1. - cb

        elif on == 'Hf':
            cb = -np.log(cb)

        np.seterr(**old_err_state)

        return cb

    def R_cb(self, x, bound='two-sided', interp='step', alpha_ci=0.05,
             bound_type='exp', dist='z'):

        if bound_type not in ['exp', 'normal']:
            return ValueError("'bound_type' must be in ['exp', 'normal']")
        if dist not in ['t', 'z']:
            return ValueError("'dist' must be in ['t', 'z']")

        confidence = 1. - alpha_ci

        old_err_state = np.seterr(all='ignore')

        x = np.atleast_1d(x)
        if bound in ['upper', 'lower']:
            if dist == 't':
                stat = t.ppf(1 - confidence, self.r - 1)
            else:
                stat = norm.ppf(1 - confidence, 0, 1)
            if bound == 'upper':
                stat = -stat
        elif bound == 'two-sided':
            if dist == 't':
                stat = t.ppf((1 - confidence) / 2, self.r - 1)
            else:
                stat = norm.ppf((1 - confidence) / 2, 0, 1)
            stat = np.array([-1, 1]).reshape(2, 1) * stat

        if bound_type == 'exp':
            # Exponential Greenwood confidence
            # print(self.greenwood)
            R_out = self.greenwood * 1. / (np.log(self.R)**2)
            R_out = np.log(-np.log(self.R)) - stat * np.sqrt(R_out)
            R_out = np.exp(-np.exp(R_out))
            R_out = np.where(self.greenwood == 0, 1, R_out)
        else:
            # Normal Greenwood confidence
            R_out = self.R + np.sqrt(self.greenwood * self.R**2) * stat

        # Allows for confidence bound to be estimated up to the last value.
        # only used in event that there is no right censoring.
        if bound == 'upper':
            R_out = np.where(np.isfinite(R_out), R_out, np.nanmin(R_out))
        elif bound == 'lower':
            R_out = np.where(np.isfinite(R_out), R_out, 0)
        else:
            R_out[0, :] = np.where(np.isfinite(R_out[0, :]), R_out[0, :], np.nanmin(R_out[0, :]))
            R_out[1, :] = np.where(np.isfinite(R_out[1, :]), R_out[1, :], 0)


        

        if interp == 'step':
            idx = np.searchsorted(self.x, x, side='right') - 1
            if bound == 'two-sided':
                R_out = R_out[:, idx]
                R_out = np.where(idx < 0, 1, R_out)
            else:
                R_out = R_out[idx]
                R_out = np.where(idx < 0, 1, R_out)

       
        else:
            if bound == 'two-sided':
                R1 = interp_function(self.x, R_out[0, :], kind=interp)(x)
                R2 = interp_function(self.x, R_out[1, :], kind=interp)(x)
                R_out = np.vstack([R1, R2])
            else:
                R_out = interp_function(self.x, R_out, kind=interp)(x)


        # The question remains. Should bounds above and below observed values
        # be calculable?...
        R_out = np.where((x < self.x.min()) | (x > self.x.max()), np.nan, R_out)

        if bound == 'two-sided':
            R_out = R_out.T

        np.seterr(**old_err_state)

        return R_out

    def random(self, size):
        return np.random.choice(self.x, size=size)

    def get_plot_data(self, **kwargs):
        y_scale_min = 0
        y_scale_max = 1

        # x-axis
        x_min = 0
        x_max = np.max(self.x)

        diff = (x_max - x_min) / 10
        x_scale_min = x_min
        x_scale_max = x_max + diff

        cbs = self.R_cb(self.x, **kwargs)

        return {
            'x_scale_min': x_scale_min,
            'x_scale_max': x_scale_max,
            'y_scale_min': y_scale_min,
            'y_scale_max': y_scale_max,
            'cbs': cbs,
            'x_': self.x,
            'R': self.R,
            'F': self.F
        }

    def plot(self, ax=None, **kwargs):
        r"""
        Creates a plot of the survival function.
        """
        if ax is None:
            ax = plt.gcf().gca()

        plot_bounds = kwargs.pop('plot_bounds', True)
        interp = kwargs.pop('interp', 'step')
        bound = kwargs.pop('bound', 'two-sided')
        alpha_ci = kwargs.pop('alpha_ci', 0.05)
        bound_type = kwargs.pop('bound_type', 'exp')
        dist = kwargs.pop('dist', 'z')

        d = self.get_plot_data(interp=interp,
                               bound=bound,
                               alpha_ci=alpha_ci,
                               bound_type=bound_type,
                               dist=dist)
        # MAKE THE PLOT
        # Set the y limits
        ax.set_ylim([d['y_scale_min'], d['y_scale_max']])

        # Label it
        ax.set_title('Model Survival Plot')
        ax.set_ylabel('R')
        if interp != 'step':
            ax.plot(d['x_'], d['R'])
            if plot_bounds:
                ax.plot(d['x_'], d['cbs'], color='r')
        else:
            ax.step(d['x_'], d['R'], where='post')
            if plot_bounds:
                ax.step(d['x_'], d['cbs'], color='r', where='post')
        
        return ax
