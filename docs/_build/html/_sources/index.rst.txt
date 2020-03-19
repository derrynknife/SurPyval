.. surpyval documentation master file, created by
   sphinx-quickstart on Thu Mar 19 20:15:57 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to surpyval's documentation!
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


surpyval
=====================================

*surpyval* is an implementation of survival analysis in Python. The intent of this was to see if I could actually make it, and therefore learn a lot about survival statistics along the way, but also so that each time a model is created, it can be reused by other planned projects for monte carlo simulations (used in reliability engineering) and optimisations.

Specifically, surpyval was designed to be used for the front end statistical analysis for the repyability package. The repyability package is a reliability engineering software package for engineers.

Conventions for surpyval package
-c = censoring
-x = random variable (time, stress etc.)
-n = counts
-r = risk set
-d = deaths

usual format for data:
xcn = x variables, with c as the censoring schemed and n as the counts
xrd = x variables, with the risk set, r,  at x and the deaths, d, also at x

wranglers for formats:
fs = failure times, f, and right censored times, s
fsl = fs format plus a vector for left censored times

df = Density Function
ff / F = Failure Function
sf / R = Survival Function
h = hazard rate
H = Cumulative hazard function

Censoring: -1 = left
0 = failure / event
1 = right
2 = interval censoring. Must have left and right coord
This is done to give an intuitive feel for when the 
event happened on the timeline.



Installation
------------------------------

Because someone beat me to putting 'surpyval' on pypi, i've settled with:

.. code-block:: console

    pip install reliafy-surpyval




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
