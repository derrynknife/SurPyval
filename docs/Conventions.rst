
Conventions
===========

Variable Names
--------------

Before discussing the formats, the conventions for vairable names needs to be clarified. These conventions are:

- x = The random variable (time, stress etc.) array
- c = Refers to the censoring data array associated with x
- n = The count array associated with x
- r = risk set at x
- d = failures/deaths at x
- t = two dimensional array of the truncation interval at x
- tl = one dimensional array of the value at which x is left truncated
- tr = one dimensional array of the value at which x is right truncated

Data Formats
------------

The conventional formats use in surpyval are:

- xcnt = x variables, with c as the censoring scheme, n as the counts, and t as the truncation
- xrd  = x variables, with the risk set, r, at x and the deaths, d, also at x

All functions in surpyval have default handling conditions for c and n. That is, if these variables aren't passed, it is assumed that there was one observation and it was a failure for every x.

Surpyval fit() functions use the xcn format. But the package has handlers for other formats to rearrange it to the needed format. Other formats are:

wranglers for formats:

- fs = failure time array, f, and right censored time array, s
- fsl = fs format plus an array for left censored times.

Censoring conventions
---------------------

For the censoring values, surpyval uses the convention used in Meeker and Escobar, that is:

- -1 = left
- 0 = failure / event
- 1 = right
- 2 = interval censoring. Must have left and right value in x

This convention gives an intuitive feel for the placement of the data on a timeline.