
Conventions
===========

Data Formats
------------

The conventional formats use in surpyval are:

- xcnt = x variables, with c as the censoring scheme, n as the counts, and t as the truncation
- xrd  = x variables, with the risk set, r, at x and the deaths, d, also at x
- xicnt = x variables, with c as the censoring scheme, n as the counts, and t as the truncation, and i as the item number

All functions in surpyval have default handling conditions for c and n. That is,
if these variables aren't passed, it is assumed that there was one observation
and it was a failure for every x. For recurrent event models, if i is not passed
it is assumed that it is all from the same item.

The xcnt format can be converted to the xrd format but not vice versa.

Surpyval fit() functions use the xcnt format. But the package has handlers for
other formats to rearrange it to the needed format. Other formats are:

Data sometimes comes in different formats. SurPyval has utilities available to convert
from these formats into the xcnt, xrd, and xicnt formats. These other formats are:

- fs = failure time array, f, and right censored time array, s
- fsl = fs format plus an array for left censored times.

Variable Names
--------------

Before discussing the formats, the conventions for vairable names needs to be clarified.

For single event survival models we use, the xcnt format. These mean:

- x  = The random variable (time, stress etc.) array
- xl = The random variable (time, stress etc.) array for the left interval of interval censored data.
- xr = The random variable (time, stress etc.) array for the right interval of interval censored data.
- c  = An array with the censor flag for each x
- n  = The count array associated with x
- t  = the truncation values for the left and right truncation at x (must be two dim, or use tl and tr instead)
- tl = one dimensional array or scalar value. If an array it is the value at which each value of x is left truncated. If a scalar all values of x are left truncated at the same value.
- tr = one dimensional array or scalar value. If an array it is the value at which each value of x is right truncated. If a scalar all values of x are right truncated at the same value.
- Z = the multi-dimensional array of covariates for each x.

Non-parametric models are better defined in the "xrd" format. These are taken to mean:

- x = array of random variables
- r = array with the number of items at risk for each value in x
- d = array with the number of failures/deaths at each value in x

For recurrent event data it is necessary to know which item has a subsequent event. For this we use all
the same as described above with the addition of:

- i = array with the item number/identifier for each value in x


Censoring Flag Conventions
--------------------------

For the censoring values, surpyval uses the convention used in Meeker and Escobar, that is:

- -1 = left
- 0 = failure / event
- 1 = right
- 2 = interval censoring. Must have left and right value in x

This convention gives an intuitive feel for the placement of the data on a timeline.


Function Conventions
--------------------

The conventions for single event SurPyval models are that each object returned from a :code:`fit()` call has the ability to compute the following:

- :code:`df()` - The density function
- :code:`ff()` - The CDF
- :code:`sf()` - The survival function, or reliability function
- :code:`hf()` - The (instantaneous) hazard function
- :code:`Hf()` - The cumulative hazard function

The conventions for recurrent event SurPyval models are that each object returned from a :code:`fit()` call has the ability to compute the following:

- :code:`iif()` - instantaneous intensity function
- :code:`cif()` - cumulative intensity function







