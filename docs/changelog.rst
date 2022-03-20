Changelog
=========

v0.11.0 (planned)
-----------------

- General ALT fitter full release
- General PH fitter full release
- Formulas
- Add more than `Breslow <http://www-personal.umich.edu/~yili/lect4notes.pdf>`_ to the CoxPH methods.
- Parameter confidence bound
- Document the rationale behind using Fleming-Harrington as the default.
- Docs on how to integrate with Pandas
- Docs for CoxPH
- Docs for Accelerated Life fitters
- Create a ``RegressionFitter`` class. I keep copying code across the three fitters.
- Allow truncation with zi and lfp models.
- Allow truncation with regression

v0.10.1.0 (25 Mar 2022)
-----------------------

- Changed plot methods to now take 'Axis' object. This allows a user to pass in an existing axis.
- plot functions now return an Axis object instead of the Lines2D object. Allows for easy user update after plotting.
- Added fs_to_xcn as it was dropped in 10.0.1.
- Changed all imports for numpy to be done from the surpyval module. This will allow for easy maintenance in future in the event of deprecated autograd.

v0.10.0.1 (22 Nov 2021)
-----------------------

- Removed fsl_to_xcn function and replaced with fsli_to_xcn function that is able to take any combination of fsli.

v0.10.0 (9 Aug 2021)
--------------------

- Version snapshot for JOSS review

v0.9.0 (5 Aug 2021)
-------------------

- Better initial estimates in the ``_parameter_initialiser`` for the lfp data (use max F from nonp estimate...)
- `issue #13 <https://github.com/derrynknife/SurPyval/issues/13>`_ - Better failures when insufficient data provided.
- `issue #12 <https://github.com/derrynknife/SurPyval/issues/12>`_ - Created ``fsli_to_xcn`` helper function.
- Fixed bug in confidence bounds implementation for offset distributions. CBs were not using the offset and were therefore way out. Now fixed.
- Created a  ``NonParametric.cb()`` method to match ``Parametric`` API for confidence bounds.
- Cleaned up NonParametric code (removed some technical debt and duplicated code).
- Changed the ``__repr__`` function in ``NonParametric`` to be aligned to ``Parametric``
- Updated the docstring for ``fit()`` for ``NonParametric``
- Fixed bug in ``NonParametric`` that required the ``x`` input to be in order for the functions (e.g. ``df`` etc.).
- ``CoxPH`` released.
- General AL fitter in beta
- General PH fitter in beta
- Created ``Linear``, ``Power``, ``InversePower``, ``Exponential``, ``InverseExponential``, ``Eyring``, ``InverseEyring``, ``DualPower``, ``PowerExponential``, ``DualExponential`` life models.
- Created ``GeneralLogLinear`` life model for variable stress count input.
- For each combination of a SurPyval distribution and life model, there is an instance to use ``fit()``. For example there are ``WeibullDualExponential``, ``LogNormalPower``, ``ExponentialExponential`` etc.
- Docs Updates:
	- Add application examples to docs:
		- Reliability Engineering
		- Actuary / Demography
		- `Social Science/Criminology <https://link.springer.com/article/10.1007/s10940-021-09499-5>`_
		- Boston Housing
		- Medical science
		- `Economics <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0232615>`_
		- Biology - Ware, J.H., Demets, D.L.: Reanalysis of some baboon descent data. Biometrics 459–463 (1976).

v0.8.0 (27 July 2021)
---------------------

- Made backwards incompatible changes to ``LFP`` models, these are now created with the ``lfp=True`` keyword in the ``fit()`` method
- Created ability to fit zero-inflated models. Simply pass the ``zi=True`` option to the ``fit()`` method.
- Chanages to ``utils.xcnt_handler`` to ensure ``x``, ``xl``, and ``xr`` are handled consistently.
- changed the way ``__repr__`` displays a Parametric object.
- Changed the default for plotting to be ``Fleming-Harrington``. This was a result of seeing how poorly the ``Nelson-Aalen`` method fits zero inflated models. FH therefore offers the best performance of a Non-Parametric estimate at the low values of the survival function (as KM reaches 0 for fully observed data) and at high values (KM is good but NA is poor).
- Added a Fleming-Harrington method to the Turnbull class.
- Improved stability with dedicated ``log_sf``, ``log_ff``, and ``log_df`` functions. Less chance of overflows and therefore better convergence.
- Changed interpolation method of ``NonParametric``. Allows for use of cubic interpolation
- Changed ``from_params`` to accept lfp and zi (or any combo)
- Changed ``random()`` in ``Parametric`` so that lfp or zi models can be simulated!
- Improved the way surpyval fails
- Substantial docs updates.


v0.7.0 (19 July 2021)
---------------------

- Major changes to the confidence bounds for ``Parametric`` models. Now use the ``cb()`` method for every bound.
- Removed the ``OffsetParametric`` class and made ``Parametric`` class now work with (or without) an offset.
- Minor doc updates.