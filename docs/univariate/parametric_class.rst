Parametric Model
================

The fitted model returned by every parametric distribution's ``fit``
(and by ``from_params``). It holds the fitted parameters ``params``
(with ``gamma``, ``p`` and ``f0`` for offset, limited-failure and
zero-inflated models), the data it was fitted to, and the parameter
covariance, and provides the distribution's functions at those
parameters (``sf``, ``ff``, ``df``, ``hf``, ``Hf``, ``qf``, ``cs``,
``mean``, ``moment``, ``random``), confidence bounds on those functions
(``cb``) and on the parameters (``param_cb``), information criteria
(``aic``, ``aic_c``, ``bic``, ``neg_ll``), probability plots and
serialisation. How to use it is shown in
:doc:`../Parametric SurPyval Modelling`.

.. autoclass:: surpyval.univariate.parametric.parametric.Parametric
   :members:
