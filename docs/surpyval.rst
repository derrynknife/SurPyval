API
===

The reference for every public class and function in SurPyval, one page
per area. The theory behind each area is in the *Survival Analysis* pages
and worked examples are in the *SurPyval Modelling* pages; each reference
page below links to both.

**Reading these pages.** SurPyval follows one pattern throughout: a
*fitter* takes data through ``fit()`` (or ``fit_from_df()`` for a
DataFrame) and returns a fitted *model*, which you then ask for
survival probabilities, hazards, quantiles, bounds and plots. Most
fitters are exported as ready-made instances rather than classes, so
the reference documents their *class*, whose name usually ends in an
underscore: ``surpyval.Weibull`` is an instance of ``Weibull_``,
``surpyval.CoxPH`` of ``CoxPH_``, and so on. Call the methods on the
instance you import (``Weibull.fit(x)``), not on the class. The model a
fit returns has its own entry -- for example
:class:`~surpyval.univariate.parametric.parametric.Parametric` for every
univariate parametric fit -- and each fitter's ``Returns`` section names
it.

Fitters take data in a common form: event times ``x``, censoring flags
``c`` (0 observed, 1 right-censored, -1 left-censored, 2
interval-censored), counts ``n`` and truncation ``t`` (or ``tl`` /
``tr``). :doc:`Conventions` defines these exactly, and
:doc:`Data Wrangler Examples` shows how to build them from other
formats. Not every model supports every kind of censoring or
truncation; where one does not, its ``fit`` docstring says so.

.. toctree::
   :maxdepth: 2

   surpyval.nonparametric
   surpyval.parametric
   surpyval.regression
   surpyval.competing_risks
   surpyval.counting
   surpyval.degradation
   surpyval.multivariate
   surpyval.beta
   comparison_and_validation
   surpyval.serialisation
   surpyval.utilities
   surpyval.datasets
