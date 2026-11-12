Parametric Model
================

The fitted model returned by every parametric distribution's ``fit``
(and by ``from_params``). It holds the fitted parameters ``params``
(with ``gamma``, ``lfp_p`` and ``f0`` for offset, limited-failure and
zero-inflated models), the data it was fitted to, and the parameter
covariance, and whether a maximum-likelihood fit reached a verified
maximum (``maximum``), and provides the distribution's functions at those
parameters (``sf``, ``ff``, ``df``, ``hf``, ``Hf``, ``qf``, ``cs``,
``mean``, ``var``, ``moment``, ``entropy``, ``random``, ``random_data``),
the same model with other parameters (``with_params``, ``extras``),
its bootstrap refits (``bootstrap``, a ``ParametricBootstrap``),
a maximum-likelihood fit's reduced bias adjustment (``rba``),
confidence bounds
on those functions (``cb``, with ``method="pivotal"`` for a
probability-plotting fit) and on the parameters (``param_cb``, and
the joint region of two, ``likelihood_contour``), the
restricted mean survival time with its bounds (``rmst``, ``rmst_cb``),
information criteria
(``aic``, ``aic_c``, ``bic``, ``neg_ll``), probability plots and
serialisation. How to use it is shown in
:doc:`../Parametric SurPyval Modelling`.

.. autoclass:: surpyval.univariate.parametric.parametric.Parametric
   :members:

.. autoclass:: surpyval.univariate.parametric.ParametricBootstrap
   :members:
