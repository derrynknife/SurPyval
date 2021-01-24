---
title: 'surpyval: Survival Analysis with Python'
tags:
  - survival analysis
  - parameter estimation
  - censored data
  - truncated data
  - maximum likelihood
  - product spacing estimation
  - method of moments
  - mean square error
  - probability plotting
  - probability plotting parameter estimation
authors:
 - name: Derryn Knife
   orcid: 0000-0003-0872-7098 
   affiliation: 1
affiliations:
 - name: Independent researcher
   index: 1
date: 23 Jan 2021
---

# Summary

Survival analysis is a tool that increasing numbers of scientist, data scientists, engineers, econometricians, and many more professions are using to solve their problems. *SurPyval* offers analysts several methods to estiamte parameters using either Maximum Likelihood, Mean Square Error, Probability Plotting, Minimum Product Spacing, and Method of Moments. Maximum Likelihood Estimation can be used for any arbitrary combination of censoring and truncation. Plotting and Mean Square Error can be used with arbitrarily censored data and limited truncation. The Minimum Spacing Estimator can be used with censored observations. The Method of Moment estimation can be used with just observed data. *SurPyval* also uses optimisations that enables fixing parameters and for 'offsets' to be calculated robustly. Fixing parameters is made clear and easy with the 'fixed' argument. Offsets are found robustly because *SurPyval* converts the bounds of an offset value from (-Inf, min(x)) to (-Inf, Inf) using a modified Exponential Linear Unit (ELU). This is inspired by the deep learning activation function, it enables the optimiser search values without there being a risk of 'nan' gradients and values. This is especially important as *SurPyval* uses autodifferentiation in the optimisation. Finally, *SurPyval* has robust nonparametric surpyval estimators that can be used to make visual comparisons between the parametric fit and nonparametric distributions. This has been used to create the plotting method that can create linearised probability plots for each distribution. *SurPyval* is an extremely powerful package with broad appeal for analysts using python in need of survival analysis tools.

# Statement of need

*SurPyval* fills a gap in the Python ecosystem of survival analysis. Other survival analysis packages, e.g. lifelines and reliability, offer excellent methods for most applications, but are limited many applications, for example, using offset values, fixing parameters, and arbitrary combinations of censoring and truncation. Further, *scipy* has yet to implement some basic features of survival analysis, i.e censoring. Therefore, there is a gap in the Python ecosystem for an api that is flexible to accomodate any arbitrary combination of observed failures (or deaths); left, right, or interval censored; and left or right truncated data with a single format. Commercial packages are well developed but can be expensive. R is excellent for survival analysis but many analysts now use python. Therefore there is a need to have an open source package available with flexibility to do survival analysis.

# Methods

Throughout this paper I use the Weibull distribution as an example. This distribution is used widely in survival analysis, in particular reliability engineering, all methods work with the other ditributions available in *SurPyval*.

An example:

```
from from surpyval import Weibull

# Weibull parameters
alpha = 10
beta  = 2

# Random samples
N = 30

x = Weibull.random(N, alpha, beta)

model = Weibull.fit(x)

print(model)
```

Unlike other survival analysis packages *SurPyval* allows users to fix parameters. This is similar to *scipy* which allows the location, shape, and scale parameters to be fixed, in *SurPyval* this is done using the fixed keyword argument. For example:

```
from from surpyval import Weibull

# Weibull parameters
alpha = 10
beta  = 2

# Random samples
N = 30

x = Weibull.random(N, alpha, beta)

model = Weibull.fit(x, fixed={'beta' : 2.})

print(model)
```

Second, SurPyval, inspired by lifelines, uses autograd to calculate the jacobians and hessians needed for optimisations to do parameter estimations. SurPyval uses lessons from deep learning to improve the stability of estimation. Concretely, SurPyval uses the elu function to do isomorphic transformations on bounded parameters to allow the optimisation process to be unrestricted during search. This substantially improves the stability of optimisation. This improvement enbales the ability to stably estimate 'offset' parameters for shifted distributions. 

Third, SurPyval allows a user to estimate parameters using several different parameter estimation methods, Maximum Likelihood Estimation, Method Of Moments, Method of Probability Plotting, Mean Square Error, and Minimum Product Spacing. This allows users to change the method they use to estimate the parameters of the distribution, this is useful depending on the needs of the analyst or if other methods do not work. For example, the three parameter Weibull has no solution for beta < 1, therefore another method is needed to estimate

Fourth, *SurPyval* provides extensive Non-Parametric estimation methods for arbitrarily censored or truncated data. Using these methods parametric methods can have their probability plots created to allow analysts to judge fits. This is an extraordinarily powerful feature of *SurPyval* as it allows a user to call plot 

*SurPyval* aims to be pure Python to make installation and maintenance simple. 



The central format used in *SurPyval* is the 'xcnt' format. This format is the variable (x), the censoring flag (c), the counts (n), and the truncation values (t). Using this format any arbitrary combination on the input can be used. The variable/failure time/time of death, x, is the measured variable that is being measured against the event.

*SurPyal* also provides utilities that assist users in wrangling their data to the 'xcnt' format.

*SurPyval* aims to be as close to the *scipy* api as is feasible. Additionally, SurPyval offers utilities to help users shape their data into the xcnt format. Using a single likelihood function all these combinations can be accomodated. This feature of SurPyval was to fill a gap in the Python ecosystem, *lifelines* is capable of some censoring but no truncation, *scipy* has no capability to do survival analysis with either type of data. *SurPyval* therefore fills a niche in the Python ecosystem.

# References
