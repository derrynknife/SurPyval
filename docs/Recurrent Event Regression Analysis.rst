Recurrent Event Regression Analysis
====================================

In the same way that we want to understand the relationship between a
single event outcome and a set of covariates, we may want to understand the
relationship between a recurrent event process and a set of covariates. For example,
we may want to understand the relationship between the number of times an
item needs repairing based on the environment, the duty cycle, and the
types of inputs it receives. For example, an electric motor in a hot and humid
environment may need more repairs than one in a cool and dry environment.
Additionally, a motor that is used more often may need more repairs than one
that is used less often. Finally, a motor that is used in a factory that
produces a lot of dust may need more repairs than one that is used in a
cleaner environment. We can use recurrent event regression to understand
the relationship between these covariates and the number of repairs that
a motor needs.

We can start with the simplest version of recurrent event regression, a
Homogeneous Poisson Process Regression. In this case we have a regular
Homogeneous Poisson Process given by:

.. math::

    N(t) = \lambda t

But we extend this to include the impact that additional factors have on the
cumulative count. So the model becomes:

.. math::

    N(t) = \phi\left( X \right) \lambda t

In this case, just as was the case with single event proportional hazard
models, there is a factor that relates the covariates to the counting function.
In doing so we can now model jointly the cumulative event process with
factors that are likely to impact the rate at which the events occur. Again,
repeating the lessons from single event survival analysis, a logical choice 
for the phi function would be the exponential function. This is because
it ensures there is never a negative number and so will always provide a valid
rate even during optimisation. That is, our model will be:

.. math::

    N(t) = e^{X \beta} \lambda t


This is the proportional intensity HPP model. The log-linear (exponential)
link function :math:`e^{X\beta}` is the standard choice because it guarantees
a positive rate regardless of the sign of :math:`\beta`, mirrors the Cox model
for single events, and gives regression coefficients a direct multiplicative
interpretation on the rate.

A notable example within recurrent event regression is the Duane process,
which is particularly relevant in reliability engineering. The Duane model is 
a form of non-homogeneous Poisson process (NHPP) that describes the improvement 
in reliability of a system or component over time, typically as a result of 
learning effects or reliability growth. The model posits that the failure rate 
of a system decreases as a function of cumulative operating time, reflecting 
the notion that systems become more reliable through usage and corrective 
actions. In the Duane process, the cumulative number of failures is modeled as 
a function of time, providing a way to quantify reliability growth and 
forecast future performance. The convetntional parameterisation for the Duane
is:

.. math::

    N(t) = \alpha t^\beta

This can be interpreted as the number of events we can expect up to time t is
given by the result of the equation.



In addition to specific models like the Duane process, recurrent event
regression encompasses the broader class of proportional intensity models. 
These models, often used in the context of survival analysis, assume that 
the intensity function (or hazard function) for an individual's time to the next
event is proportional to a baseline intensity function, adjusted by the 
individual's covariates. This assumption of proportionality allows for the 
straightforward interpretation of covariate effects on the hazard of an event 
occurring. This makes the comparison of risks between different groups or 
under different conditions easy and interpretable.


1. The traditional Duane model's cumulative number of failures as a function of time:

.. math::

    N(t) = \alpha t^\beta

2. The failure intensity function derived from the Duane model:

.. math::

    \lambda(t) = \frac{dN(t)}{dt} = \alpha \beta t^{\beta - 1}

3. The Proportional Intensity model incorporating a covariate :math:`x`:

.. math::

    \lambda(t \mid x) = \lambda_0(t) \exp(\gamma x)

4. The adjusted failure intensity function with the covariate effect in the context of the Duane model:

.. math::

    \lambda(t \mid x) = \alpha \beta t^{\beta - 1} \exp(\gamma x)

These equations outline the framework for modelling the reliability growth of a
system, incorporating the effects of covariates on the failure intensity.

More generally, any of SurPyval's counting-process baselines can play the role
of :math:`\lambda_0(t)`. The proportional-intensity model multiplies that
baseline by the covariate factor :math:`e^{Z\beta}`:

.. math::

    \lambda(t \mid Z) = \lambda_0(t)\, e^{Z\beta},
    \qquad
    \Lambda(t \mid Z) = \Lambda_0(t)\, e^{Z\beta}.

With a constant baseline this is the proportional-intensity HPP; with a
power-law (Duane / Crow-AMSAA) or log-linear (Cox-Lewis) baseline it is the
proportional-intensity NHPP. Because the covariate factor scales the whole
cumulative intensity, the ratio of expected event counts between two covariate
settings is the constant :math:`e^{(Z_2 - Z_1)\beta}` at every time — the
recurrent-event analogue of a hazard ratio [Cook2007r]_, and the reason the
coefficients :math:`\beta` read directly as multiplicative effects on the event
rate.

The parameters are estimated jointly by maximum likelihood [Lawless1987]_,
maximising the NHPP log-likelihood with the covariate-scaled intensity
substituted in. The static, per-item covariates enter through
:math:`e^{Z\beta}`, so an item in a harsher environment simply accumulates
events faster. This proportional-intensity framing is standard in the
reliability-growth literature [Rigdon2000r]_.

Model checking
--------------

The fitted regression model carries the same diagnostics as the unconditional
intensity models, applied *per item* with each item's intensity scaled by its
own covariate factor. The time-rescaling residuals pool the rescaled
inter-arrival increments across items (i.i.d. Exp(1) under a well-specified
model), the trend test checks whether a time-varying baseline was warranted at
all, and the Cramér–von Mises test provides a bootstrapped goodness-of-fit
p-value. Confidence bounds on the fitted cumulative intensity at a covariate
setting come from the delta method via ``cif_cb``.

These methods form a comprehensive toolkit for researchers and practitioners
working with recurrent event data, enabling detailed analysis and prediction of
event occurrences.

References
----------

.. [Cook2007r] Cook, R.J. and Lawless, J.F., 2007. *The Statistical Analysis of
   Recurrent Events*. Springer.

.. [Lawless1987] Lawless, J.F., 1987. Regression methods for Poisson process
   data. *Journal of the American Statistical Association*, 82(399),
   pp.808-815.

.. [Rigdon2000r] Rigdon, S.E. and Basu, A.P., 2000. *Statistical Methods for the
   Reliability of Repairable Systems*. John Wiley & Sons.





