---
title: 'SurPyval: Survival Analysis with Python'
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
bibliography: paper.bib
---

# Summary

Survival analysis is a tool that increasing numbers of scientist, data scientists, engineers, econometricians, and many more professions are using to solve their problems. Survival analysis is a unique set of tools that are used to estimate either the time to an event or the chance of an event happening. That is, survival analysis allows you to estimate how long something is likely to last or what risk there is of some event happening in future. This is vital for fields such as the medical sciences where we need to know how long someone with a particular diagnosis might live or if a treatment or intervention is successful at prolonging life. In engineering it is useful to understand the risk that fielded equipment might fail. In insurance it is necessary to help price policies and in economics it is useful for estimating the durations of recessions or the time to the next recession. Each of these applications are bugged with unique problems with data. In engineering components might not fail during the observation period, or it might fail between two inspections. In this case the data is said to be censored. In medical trials you might have subjects enter an experiment later than other subjects while for insurance claims are only lodged above the excess value on the policy. In these cases the data is said to be truncated. These considerations are unique to survival analysis and are critical to handle correctly to make appropriate predictions or find significant differences.

*SurPyval* is designed to be pure Python to make installation and maintenance simple. Further, *SurPyval* is a flexible and robust survival analysis package that can take as input an arbitrary combination of observed, censored, and truncated events over a wide number of distributions and their variations. For this reason *SurPyval* is likely to be of interest to a wide field of analysts in broad industries including finance, insurance, engineering, medical science, agricultural science, economics, and many others.

# Statement of need

*SurPyval* fills a gap in the Python ecosystem of survival analysis. Other survival analysis packages, e.g. *lifelines* and *reliability*, offer excellent methods for most applications, but are limited many applications, for example, using offset values, fixing parameters, and arbitrary combinations of censoring and truncation. Further, *scipy* has yet to implement some basic features of survival analysis; concretely, it does not offer censored data as an input. Therefore, there is a gap in the Python ecosystem for an API that is flexible to accomodate any arbitrary combination of observed failures (or deaths); left, right, or interval censored; and left or right truncated data with a single format. Further, *SurPyval* allows users to select an appropriate estimation method for their circumstances. MLE is used in most other applications, but the *SurPyval* implementation of Minimal Product Spacing, Method of Moments, Probability Plotting, Mean Square Error, and Expectation-Maximisation makes it a much more capable package than is currently available. Commercial packages are well developed but can be expensive. R is excellent for survival analysis but many analysts now use python as is explained in the *lifelines* paper. Therefore there is a need to have a flexible and open source python package to do survival analysis.

# Methods

*SurPyval* is grouped into two sections, these are parametric and non-parametric. For the parametric capability *SurPyval* offers several methods to estimate parameters; these are Maximum Likelihood, Mean Square Error, Probability Plotting, Minimum Product Spacing, Method of Moments, and Expectation-Maximisation. The EM is only used for mixture models. Support for data types and estimation methods can be seen in Table 1.

| Method | Observed | Censored | Truncated |
| ------ |-----|------|------|
| **MLE** | Yes | Yes | Yes |
| **MPP** | Yes | Yes | Limited |
| **MSE** | Yes | Yes | Limited |
| **MOM** | Yes | No | No |
| **MPS** | Yes | Yes | No |


Maximum Likelihood Estimation can be used for any arbitrary combination of censoring and truncation. Plotting and Mean Square Error can be used with arbitrarily censored data and limited truncation. Specifically, these methods are limited if the maximum and minimum of the observed data are truncated obserations. This is because the Turnbull NPMLE cannot assume the shape of the distribution and therefore cannot be used to estimate by how much the highest and lowest values are truncated. The Minimum Spacing Estimator can be used with censored observations. The Method of Moment estimation can be used with just observed data.

The central format used in *SurPyval* is the 'xcnt' format. This format is the variable (x), the censoring flag (c), the counts (n), and the truncation values (t). Using this format any arbitrary combination on the input can be used. The variable/failure time/time of death, x, is the measured variable that is being measured against the event.


*SurPyval*, inspired by lifelines, uses autodifferentiation to calculate the jacobians and hessians needed for optimisations in parametric analysis. SurPyval uses lessons from deep learning to improve the stability of estimation. Concretely, SurPyval uses the ELU function [@clevert2015fast] to transform bounded parameters to be unbounded. For example, the alpha parameter for a Weibull distribution is supported on the half-real line, (0, Inf). Using the ELU function this support is then the full real line (-Inf, Inf), this reduces the risk of optimisations failing because the numeric gradient might 'overshoot' and produce undefined gradients. The ELU is also useful for autodifferentiation because it is continuously differentiable making the transform stable. This transform works sufficiently well to allow *SurPyval* to robustly estimate offsets, i.e. the 'gamma' parameter, for half real-line supported distribtuions. This substantially improves the stability of optimisation and greatly expands the possible use in research and industry.

Finally, *SurPyval* has robust nonparametric surpyval estimators that can be used to make visual comparisons between the parametric fit and nonparametric distributions. This has been used to create the plotting method that can create linearised probability plots for each distribution. *SurPyval* is an extremely powerful package with broad appeal for analysts using python in need of survival analysis tools.

An example:

```python
from surpyval import Weibull

# Weibull parameters
alpha = 10
beta  = 2

# Random samples
N = 30

x = Weibull.random(N, alpha, beta)

model = Weibull.fit(x)
```

Unlike other survival analysis packages *SurPyval* allows users to fix parameters. This is similar to *scipy* which allows the location, shape, and scale parameters to be fixed, in *SurPyval* this is done using the fixed keyword argument. For example:


Fourth, *SurPyval* provides extensive Non-Parametric estimation methods for arbitrarily censored or truncated data. Using these methods parametric methods can have their probability plots created to allow analysts to judge fits. This is an extraordinarily powerful feature of *SurPyval* as it allows a user to call plot 

*SurPyal* also provides utilities that assist users in wrangling their data to the 'xcnt' format.

*SurPyval* aims to be as close to the *scipy* api as is feasible. Additionally, SurPyval offers utilities to help users shape their data into the xcnt format. Using a single likelihood function all these combinations can be accomodated. This feature of SurPyval was to fill a gap in the Python ecosystem, *lifelines* is capable of some censoring but no truncation, *scipy* has no capability to do survival analysis with either type of data. *SurPyval* therefore fills a niche in the Python ecosystem.

# References
