Example Applications
====================

The parametric API is essentially the exact same as the non-parametric API. All models are fit by a 
call to the ``fit()`` method. However, the parametric models have more options that are only applicable to parametric modelling. The inputs of ``x`` for the random variable, ``c`` for the censoring flag, ``n``
for count of each ``x``, ``xl`` and ``xr`` for intervally censored data (can't be used with ``x``) ``t``
for the truncation matrix, ``tl`` for the left truncation scalar or array, and ``tr`` for the right truncation scalar or array all remain.

Reliability Engineering
-----------------------

Demographics / Actuarial
------------------------

