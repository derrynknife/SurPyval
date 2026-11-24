Hazard Ratio
============

Every parametric model takes a hazard ratio ``hr``, a known number above
0, beside its offset ``gamma``, limited-failure proportion ``lfp_p`` and
zero-inflation fraction ``f0``. A model with ``hr`` has ``hr`` times the
hazard of the same model without it, at every age:

.. math::

    H(t) = hr \cdot H_0(t), \qquad R(t) = R_0(t)^{hr},

where :math:`H_0` and :math:`R_0` are the model's cumulative hazard and
survival without ``hr``. Use it for a stress, environment or quality
factor that multiplies a known hazard (a part run hotter than the fleet it
was fitted on, a handbook's environment factor), for derating, or to split
one life into parts: the models at ``hr = 1 - b`` and ``hr = b`` together
make the model, since :math:`R^{1-b} R^{b} = R`, which is how a group of
units sharing a common cause is built.

.. code-block:: python

    >>> import surpyval as surv
    >>> base = surv.Weibull.from_params([1000, 2])
    >>> hot = surv.Weibull.from_params([1000, 2], hr=1.5)
    >>> print(hot)
    Parametric SurPyval Model
    =========================
    Distribution        : Weibull
    Fitted by           : given parameters
    Hazard ratio (hr)   : 1.5
    Parameters          :
         alpha: 1000
          beta: 2
    >>> float(hot.sf(500)), float(base.sf(500) ** 1.5)
    (0.6872892787909722, 0.6872892787909722)

How it applies
--------------

- **Outermost.** ``hr`` applies over the offset, the limited failure
  population and the zero inflation: the whole model's survival is raised
  to ``hr``. A mass ``f0`` at 0 becomes ``1 - (1 - f0) ** hr``, which is
  also ``df(0)`` and ``hf(0)``; the proportion that never fails becomes
  ``(1 - lfp_p) ** hr``.
- **Every function.** ``sf``, ``ff``, ``Hf``, ``hf`` and ``df`` follow
  from the cumulative hazard, so each is precise far into both tails
  (``ff`` from ``-expm1(-hr * H0)``). A discrete model's hazard is
  ``1 - (1 - h0) ** hr``. ``qf(p)`` and ``isf(s)`` invert through the
  model's ``isf`` at :math:`R_0 = R^{1/hr}`, computed in log space, and
  ``random`` draws ``qf`` of its uniforms, so ``isf(sf(t))`` gives back
  ``t`` at a survival of 1e-300.
- **Closed forms stay exact.** A Weibull with ``hr`` is the Weibull with
  scale :math:`\alpha\, hr^{-1/\beta}`, and an exponential's rate is
  :math:`hr \cdot \lambda` (an offset and zero inflation carry across):
  their ``mean``, ``moment``, ``var``, ``entropy``, ``mrl`` and restricted
  mean are those families' closed forms. Other distributions integrate
  the survival function numerically for the mean, moments, variance, mean
  residual life and restricted mean, and refuse the entropy (integrate
  ``-df * log(df)`` instead). A discrete model's moments, and the
  defective moments of a limited failure population, are refused with
  what to use.
- **``hr = 1`` changes nothing.** Every function and every seeded draw is
  the model's without ``hr``, to the last bit.

``hr`` is not a parameter: it is not in ``params``, ``covariance_names``
or ``k``. It is in ``extras`` (so ``dist.from_params(params,
**model.extras)`` keeps it), ``with_params(..., hr=)`` sets it on any
model, and ``rescale_time`` keeps it (a hazard ratio has no unit).

Distributions that are probabilities of an event (``Bernoulli``,
``Binomial``, ``FixedEventProbability``, ``ExactEventTime``) refuse it:
scale the probability itself, ``q`` becoming ``1 - (1 - q) ** hr``.

Fitting with a known hazard ratio
---------------------------------

``fit(..., hr=)`` fits data that come from an environment with a known
hazard ratio: the likelihood is the hazard-scaled model's, and the fitted
model carries ``hr``. It is never estimated, since it is confounded with
the scale (a Weibull fitted with ``hr = 2`` is the plain fit with its
scale multiplied by :math:`2^{1/\beta}`). Maximum likelihood only.

.. code-block:: python

    >>> x = [10, 12, 15, 17, 20, 25, 31]
    >>> plain = surv.Weibull.fit(x)
    >>> hot = surv.Weibull.fit(x, hr=2.0)
    >>> hot.hr, hot.params.round(3)
    (2.0, array([26.451,  2.931]))
    >>> round(float(plain.params[0] * 2 ** (1 / plain.params[1])), 3)
    26.451

The bounds hold ``hr`` fixed, as a known constant:

- ``cb`` and ``quantile_cb`` with ``method="wald"`` (every ``on=``
  target), ``method="lr"`` and ``method="bootstrap"`` (each refit has the
  same ``hr``). The likelihood-ratio bound is the extreme of the
  hazard-scaled model's own function over the parameters' likelihood
  region, so a Weibull's is the plain Weibull fit's (the two models are
  one reparameterisation of the other);
- ``param_cb`` with every method, the likelihood-ratio bounds included
  (the profile likelihood is the hazard-scaled model's);
- refused, naming what to use instead: ``mean_cb``, ``mrl_cb``,
  ``rmst_cb`` and ``rmst`` (bound them from the bootstrap refits,
  ``model.bootstrap(...).models``, which carry ``hr``), and the pivotal
  bound, which is for a probability-plotting fit (``fit(..., hr=)`` is
  maximum likelihood only).

Saving
------

``to_dict`` writes ``"hr"`` only where it is not 1, and such a dict is
stamped schema 5, which SurPyval 0.28 and older refuse with a request to
upgrade rather than load the model without its hazard ratio. A model with
``hr = 1`` saves as before.
