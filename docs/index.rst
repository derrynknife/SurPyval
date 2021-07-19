.. surpyval documentation master file, created by
   sphinx-quickstart on Thu Mar 19 20:15:57 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SurPyval - Survival Analysis in Python
======================================

*surpyval* is an implementation of survival analysis in Python. The intent of this was to see if I could actually make it, and therefore learn a lot about survival statistics along the way, but also so that each time a model is created, it can be reused by other planned projects for monte carlo simulations (used in reliability engineering) and optimisations.

Specifically, surpyval was designed to be used for the front end statistical analysis for the repyability package. The repyability package is a reliability engineering software package for engineers.

One feature of surpyval that separates it from other survival analysis packages is the intuitive way with which you can pass data to the fit methods. There are many different formats that can be used for survival analysis; surpyval handles many of the conceivable ways you can have your data stored. This is discussed in the data format tab.

Surpyval is also unique in the way in which it lets you estimate the parameters. With surpyval, you can use any of the following methods to estimate the parameters of you distribution of interest:

.. list-table:: SurPyval Modelling Methods
   :header-rows: 1

   * - Method
     - Para/Non-Para
     - Observed
     - Censored
     - Truncated
   * - Maximum Likelihood (MLE)
     - Parametric
     - Yes
     - Yes
     - Yes
   * - Probability Plotting (MPP)
     - Parametric
     - Yes
     - Yes
     - Limited
   * - Mean Square Error (MSE)
     - Parametric
     - Yes
     - Yes
     - Limited
   * - Method of Moments (MOM)
     - Parametric
     - Yes
     - No
     - No
   * - Maximum Product Spacing (MPS)
     - Parametric
     - Yes
     - Yes
     - No (planned)
   * - Kaplan-Meier
     - Non-Parametric
     - Yes
     - Right only
     - Left only
   * - Nelson-Aalen
     - Non-Parametric
     - Yes
     - Right only
     - Left only
   * - Fleming-Harrington
     - Non-Parametric
     - Yes
     - Right only
     - Left only
   * - Turnbull
     - Non-Parametric
     - Yes
     - Yes
     - Yes

Most other survival analysis packages focus on just using the MLE, or maybe the Probability Plotting. This package grew out of replicating the historically used probability plotting method from engineering, and as it progressed, it was discovered that there are many many ways parameters of distributions can be estimated. The product spacing estimator is particularly useful for offset distributions or finitely bounded distributions.

SurPyval attempts to use the combination of these methods to make parameter estimation possible for any distribution with arbitrary combnations of observations, censoring, and truncation.

Becoming a competent survival analyst depends strongly on having a very strong understanding of censoring, truncation, and observations in conjunction with a solid understanding of different types of distributions. Knowing and being able to identify situations as being censored or truncated in real applications will ensure you do not make an errors in your analysis. This can be very difficult to do. This documention can be used as a reference to understand the types of censoring and truncation so that you can identify these situations in your work. Further, having a deep understanding of the types of distributions used in survival analysis will allow you to identify the process that is generating your data. This will then allow you to select an appropriate distribution, if any, to solve your problem. Survival analysis is an extremely powerful, and thoroughly interesting tool, so don't give up, or if you do give up, do the survival statistics on it.


Contents:
=========

.. toctree::
   :maxdepth: 1
   :caption: Quickstart & Intro

   Quickstart
   Types of Data
   Conventions
   Data Wrangler Examples
   Datasets


.. toctree::
   :maxdepth: 1
   :caption: Survival Analysis

   Non-Parametric Modelling
   Parametric Modelling
   SurPyval Models
   Distributions


.. toctree::
   :maxdepth: 1
   :caption: SurPyval

   surpyval

.. toctree::
   :maxdepth: 1
   :caption: Community Guidelines

   Support
   Report an Issue <https://github.com/derrynknife/SurPyval/issues>
   Contributing




Installation
------------------------------

*surpyval* can be installed easily with the pip command:

.. code-block:: bash

    $ pip install surpyval




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
