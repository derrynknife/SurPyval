Maturity Projection (AMPM, AMPM-Stein, discrete AMPM-Stein)
===========================================================

A development test surfaces failure modes, and fixes then remove part of
each seen mode's rate: its fix-effectiveness factor (FEF), a fraction from
0 (no effect) to 1 (the mode is gone). Fixing what has been seen cannot
remove the rate of the modes the test has not surfaced yet. A maturity
projection estimates that hidden part from how often the seen modes
repeated, and so gives the reliability to expect once the fixes are in
(MIL-HDBK-189C sections 6.3.4 to 6.3.6).

In plain words: if most modes turned up several times, the test has
probably found most of what there is to find; if most turned up once,
much is still hidden. Each model treats the modes' rates (or, for one-shot
items, their chances of failing a trial) as a sample from a population,
in the limit of very many small modes, as in a complex system, and reads
that population's spread from the repeats.

- :func:`~surpyval.recurrent.ampm_projection`, the AMSAA maturity
  projection model (AMPM), uses the first occurrence of each mode to be
  fixed (a B mode). It does not matter when the fixes went in.
- :func:`~surpyval.recurrent.ampm_stein_projection`, AMPM-Stein, uses each
  mode's failure count. It shrinks every mode's observed rate towards the
  average (Stein's estimate), which also gives the modes not seen their
  share. No mode needs to be classed A (not to be fixed) or B, though
  ``a_modes`` can say which are A modes. Several systems tested to the same
  end are pooled.
- :func:`~surpyval.recurrent.adpm_stein_projection` is the same idea for
  one-shot items: the chance a trial succeeds once the seen modes are
  fixed.

Each refuses data from which the hidden part cannot be judged: no mode seen
more than once (the Stein models), or first occurrences that do not thin
out (the AMPM). Use :meth:`CrowAMSAA.projection
<surpyval.recurrent.parametric.crow_amsaa.CrowAMSAA.projection>` (the
AMSAA-Crow projection) there.

.. code:: python

    from surpyval.recurrent import ampm_projection, ampm_stein_projection

    x = [15, 42, 60, 98, 130, 171, 205, 260, 310, 345, 390, 400]
    modes = ["b1", "a1", "b2", "b1", "b3", "a2", "b2", "b4", "b1",
             "a1", "b5", None]
    c = [0] * 11 + [1]                      # the last row ends the test
    fef = {"b1": 0.8, "b2": 0.7, "b3": 0.75, "b4": 0.6, "b5": 0.7}

    ampm_projection(x, modes, fef, c=c).projected_mtbf        # 50.12
    ampm_stein_projection(x, modes, fef, c=c).projected_mtbf  # 52.2

Before the fixes the MTBF is 400 / 11 = 36.4; once the five B modes are
fixed both models project about 50, short of what removing those modes'
observed failures alone would suggest, because part of the rate is in
modes not seen yet (about 27% by the AMPM).

These are the many-modes forms, the ones the handbook recommends for
complex systems. The finite-number-of-modes forms, the method-of-moments
estimates and bounds on the projections are not built.

*Checked against:* the maximum of each model's own likelihood found by a
general optimiser (the AMPM's first-occurrence likelihood, to 1e-6; the
logarithmic-series likelihood of the counts behind AMPM-Stein's shrinkage,
to 1e-7; and, for the discrete model, the exact beta-binomial likelihood of
200,000 modes, to 1e-3), and simulations of the models' assumptions
(20,000 modes with gamma rates, or with beta chances, every seen mode
fixed): the projections average within 6% (continuous) and 0.01
(discrete reliability) of the true unseen part.

.. autofunction:: surpyval.recurrent.ampm_projection

.. autoclass:: surpyval.recurrent.AMPMProjection
   :members:

.. autofunction:: surpyval.recurrent.ampm_stein_projection

.. autoclass:: surpyval.recurrent.SteinProjection
   :members:

.. autofunction:: surpyval.recurrent.adpm_stein_projection

.. autoclass:: surpyval.recurrent.DiscreteSteinProjection
   :members:
