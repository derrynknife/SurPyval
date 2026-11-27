Hypoexponential Distribution
============================

The sum of independent Exponential stages (the generalised Erlang): the
lifetime of anything that must pass through several memoryless stages in
series -- a load-sharing group whose rate changes as members fail, a warm
or hot standby system. The number of stages is not fixed: ``from_params``
takes any number of rates, and the model carries that many parameters,
``lambda_1 ... lambda_m``. Rates may repeat; equal rates are the Erlang
case, a ``Gamma`` with an integer shape.

``fit(x, ..., stages=m)`` estimates the ``m`` rates from data, with every
kind of censoring and truncation, an offset, a limited failure population
and zero inflation, and reports them slowest stage first. The number of
stages is not estimated: compare fits with different numbers by their AIC.
A fit with more stages than the data support warns that a rate runs to
infinity.

.. autoclass:: surpyval.univariate.parametric.distributions.hypoexponential.Hypoexponential_
   :members:
   :inherited-members:
