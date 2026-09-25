Hypoexponential Distribution
============================

The sum of independent Exponential stages with distinct rates (the
generalised Erlang): the lifetime of anything that must pass through
several memoryless stages in series -- a load-sharing group whose rate
changes as members fail, a warm or hot standby system. The number of
stages is not fixed: ``from_params`` takes any number of rates and the
model carries that many parameters, ``lambda_1 ... lambda_m``. Equal
rates are the Erlang case and belong to ``Gamma``; rates too close to
each other are refused, since the closed form is ill-conditioned there.
It is not fitted from data (``fit`` raises ``NotImplementedError``) --
build it from known stage rates.

.. autoclass:: surpyval.univariate.parametric.distributions.hypoexponential.Hypoexponential_
   :members:
   :inherited-members:
