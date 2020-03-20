.. surpyval documentation master file, created by
   sphinx-quickstart on Thu Mar 19 20:15:57 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SurPyval
========

*surpyval* is an implementation of survival analysis in Python. The intent of this was to see if I could actually make it, and therefore learn a lot about survival statistics along the way, but also so that each time a model is created, it can be reused by other planned projects for monte carlo simulations (used in reliability engineering) and optimisations.

Specifically, surpyval was designed to be used for the front end statistical analysis for the repyability package. The repyability package is a reliability engineering software package for engineers.

One feature of surpyval that separates it from other survival analysis packages is the intuitive way with which you can pass data to the fit methods. There are many different formats that can be used for survival analysis; surpyval handles many of the conceivable ways you can have your data stored. This is discussed in the data format tab.

Surpyval is also unique in the way in which it lets you estimate the parameters. With surpyval, you can use any of the following methods to estimate the parameters of you distribution of interest:

- Maximum Likelihood
- Method of Moments
- Probability Plotting (least squares on linearised empirical CDF estimate)
- Maximum Product Spacing
- Least Squares (least squares on empirical CDF estimate)
- Expectation Maximisation (For Mixture Models Only)

Most other survival analysis packages focus on just using the MLE. This package grew out of replicating the historically used probability plotting method from engineering, and as it progressed, it was discovered that there are many many ways parameters of distributions can be estimated. The product spacing estimator is particularly useful for offset distributions.

Surpyval attempts to use the combination of these methods to make parameter estimation stable for any distribution and any censoring scheme.

Contents:
=========

.. toctree::
   :maxdepth: 1
   :caption: Quickstart & Intro

   Quickstart
   Conventions
   Data Wrangler Examples
   Datasets

.. toctree::
   :maxdepth: 1
   :caption: Non-Parametric Models

   Non-Parametric Models
   Kaplan-Meier Estimate
   Nelson-Aalen Estimate
   Fleming-Harrington Estimate

.. toctree::
   :maxdepth: 1
   :caption: Parametric Models

   Parametric Modelling
   Estimating Parameters
   Distributions



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
