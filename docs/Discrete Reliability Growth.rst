Discrete Reliability Growth
===========================

Some items do not run for hours: they either work or they don't, each
time they are used. A missile launches or not; an airbag fires or not; a
generator starts or not; a single-use medical device works first time or
not. Their reliability is the chance of success on a trial, and their
development is tested in stages (or *configurations*): a batch of trials,
fixes for what failed, then another batch. If the fixes work, the share
of successes rises from stage to stage.

The models here fit that rise to the failures and trials of each stage,
say how sure we can be of the reliability reached, and project the next
stage. They live in ``surpyval.recurrent``, beside the continuous growth
models (:class:`surpyval.CrowAMSAA`, :class:`surpyval.Duane`); planning a
programme before it starts is :doc:`Reliability Growth Planning`.

Which model?
------------

- :class:`~surpyval.recurrent.CrowAMSAADiscrete`, the AMSAA discrete
  model (MIL-HDBK-189C): Duane's learning curve counted in trials. The
  expected number of failures in the first ``T`` trials is
  :math:`\lambda T^\beta`; a :math:`\beta` below 1 means failures thin out
  as testing goes on. Configurations may have any number of trials, and
  the next configuration is projected for the trials it will have.
- :class:`~surpyval.recurrent.LloydLipow`: the reliability climbs to a
  ceiling, the gap closing as one over the stage number,
  :math:`R_k = R_\infty - \alpha / k`. Most of the gain comes early.
- :class:`~surpyval.recurrent.GompertzGrowth`: an S-shaped climb,
  :math:`R(T) = a\, b^{c^T}`, from :math:`a b` at stage 0 to the ceiling
  ``a``. :class:`~surpyval.recurrent.ModifiedGompertzGrowth` adds a floor,
  :math:`d + a\, b^{c^T}`, for growth that starts slowly and then speeds
  up.

Fitting
-------

Give the failures and trials of each stage. The fit is by maximum
likelihood: each stage's successes are binomial with that stage's
reliability.

.. code:: python

    from surpyval.recurrent import CrowAMSAADiscrete, LloydLipow

    # MIL-HDBK-189C's example: four configurations.
    model = CrowAMSAADiscrete.fit(failures=[5, 3, 4, 4],
                                  trials=[14, 19, 15, 20])
    model.params                    # lambda 0.595, beta 0.780
    model.reliability([1, 2, 3, 4]) # 0.667, 0.766, 0.794, 0.810
    model.next_reliability(20)      # a fifth configuration of 20 trials

These are the handbook's numbers (its table X). The handbook's own
approximate lower bound on the last configuration is
``model.handbook_lower_bound(alpha_ci)``.

Lloyd-Lipow and Gompertz are traditionally fitted by least squares to
the observed reliabilities; ``method="least_squares"`` does that, and it
is the only fit when you have reliabilities but not the counts behind
them (``reliability=[...]``, as fractions). Least squares weighs every
stage alike, however many trials it had; maximum likelihood weighs each
by its trials, and gives likelihood-ratio bounds.

.. code:: python

    from surpyval.recurrent import GompertzGrowth

    model = GompertzGrowth.fit(
        reliability=[0.58, 0.66, 0.725, 0.78, 0.82, 0.85],
        stage=[0, 1, 2, 3, 4, 5],          # months
    )
    model.params           # a 0.9422, b 0.6152, c 0.7321
    model.reliability(12)  # 0.931 at month 12
    model.limit            # 0.942, the most this growth reaches

The stage is 1, 2, ... by default; give ``stage`` for any other
numbering (Gompertz takes any increasing time, such as months).

How sure?
---------

``model.reliability_cb(stage, alpha_ci=0.1)`` gives two-sided bounds on
the reliability at a stage (``bound="lower"`` for a one-sided bound, as a
demonstration needs):

- ``method="wald"`` (the default) uses the standard error from the fit, on
  the logit scale so the bounds stay between 0 and 1;
- ``method="lr"``, for maximum likelihood fits, finds the reliabilities
  at which the profile likelihood has fallen by the chi-squared amount.
  It is the better bound with few trials or a reliability near 1.

``param_cb`` gives bounds on the parameters, and
``next_reliability_cb`` on the projection.

*Coverage.* In a simulation study (400 programmes each;
``surpyval/tests/calibration/test_coverage_discrete_growth.py``), the 90%
bounds on the last stage's reliability covered the truth in these
proportions:

.. list-table::
   :header-rows: 1

   * - Design
     - Wald
     - Likelihood ratio
   * - Lloyd-Lipow, 10 stages of 30 trials
     - 0.930
     - 0.933
   * - Gompertz, 10 stages of 30 trials
     - 0.930
     - 0.928
   * - AMSAA discrete, 6 configurations of 25 trials
     - 0.873
     - 0.860

Lloyd-Lipow and Gompertz bounds are a little wide at this size; the
AMSAA discrete bounds a little narrow (each within the study's
tolerance of 0.055). With few failures, treat a bound on the AMSAA
model's last configuration as optimistic, or read the more conservative
``handbook_lower_bound``.

Checked against
---------------

- MIL-HDBK-189C, section 6.2.4: the AMSAA discrete example's estimates
  and configuration reliabilities, to the three figures printed.
- Lloyd-Lipow's least squares has a closed form; the fit equals it to
  1e-12, and its covariance is ordinary least squares'.
- The Gompertz curves recover their parameters from noise-free
  reliabilities (to 1e-7 and 1e-5).
- Simulation: with 10 stages of 2000 trials, every model's estimates fall
  within four standard errors of the truth, and the truth's reliabilities
  within the fit's bounds.
- The worked examples of a commercial growth package's documentation:
  Lloyd-Lipow by least squares (:math:`R_\infty = 0.8104`,
  :math:`\alpha = 0.2207`) and by maximum likelihood
  (:math:`R_\infty = 0.7157`), Gompertz (above) and modified Gompertz
  (:math:`a = 0.6904, b = 0.0020, c = 0.4567, d = 0.3104`), all to the
  four figures printed.

References
----------

- Department of Defense (2011). *MIL-HDBK-189C, Reliability Growth
  Management*, section 6.2.4.
- Lloyd, D. K. and Lipow, M. (1962). *Reliability: Management, Methods,
  and Mathematics*. Prentice-Hall.
- Virene, E. P. (1968). Reliability growth and its upper limit.
  *Proceedings of the Annual Symposium on Reliability*, 265-270.
