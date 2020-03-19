
Conventions
===========


Before discussing the formats, the conventions for vairable names needs to be clarified. These conventions are:

- x = The random variable (time, stress etc.) array
- c = Refers to the censoring data array associated with x
- n = The count array associated with x
- r = risk set at x
- d = deaths at x

usual format for data:

- xcn = x variables, with c as the censoring schemed and n as the counts
- xrd = x variables, with the risk set, r,  at x and the deaths, d, also at x

All functions in surpyval have default handling conditions for c and n. That is, if these variables aren't passed, it is assumed that there was one observation and it was a failure for every x.

Surpyval fit() functions use the xcn format. But the package has handlers for other formats to rearrange it to the needed format. Other formats are:

wranglers for formats:

- fs = failure time array, f, and right censored time array, s
- fsl = fs format plus an array for left censored times

For the censoring values, surpyval uses the convention seen in Meeker and Escobar, that is:

- -1 = left
- 0 = failure / event
- 1 = right
- 2 = interval censoring. Must have left and right coord

This convention gives an intuitive feel for the placement of the data on a timeline.