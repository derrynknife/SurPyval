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
complex systems. The finite-number-of-modes forms and the
method-of-moments estimates are not built.

*Checked against the handbook:* MIL-HDBK-189C's TABLE XV (56 failures in
400 hours), analysed by the AMPM with every BC and BD mode a B mode at the
BD modes' mean factor 0.72, as in MIL-HDBK-00189A section 7.6.8 (TABLE
XIX), gives the handbook's growth potential MTBF, 15.24, exactly, and a
projected MTBF of 10.68 against its 10.75 (0.7% apart). The handbook's
figure came from a program option for fixes made during the test whose
formula it does not give; the growth potential, which uses the same
:math:`\lambda_A`, :math:`\lambda_B` and mean factor, agrees.

*Checked against:* the maximum of each model's own likelihood found by a
general optimiser (the AMPM's first-occurrence likelihood, to 1e-6; the
logarithmic-series likelihood of the counts behind AMPM-Stein's shrinkage,
to 1e-7; and, for the discrete model, the exact beta-binomial likelihood of
200,000 modes, to 1e-3), and simulations of the models' assumptions
(20,000 modes with gamma rates, or with beta chances, every seen mode
fixed): the projections average within 6% (continuous) and 0.01
(discrete reliability) of the true unseen part.

Bounds
------

Each result has parametric bootstrap bounds:
``projected_intensity_cb`` and ``projected_mtbf_cb`` (and, for the AMPM,
``intensity_cb(t)`` and ``mtbf_cb(t)`` at any time, ``t=inf`` for the
growth potential), and ``projected_reliability_cb`` for one-shot items.
They take ``alpha_ci``, ``bound`` (``"two-sided"``, ``"lower"`` or
``"upper"``), ``n_boot`` and ``random_state`` as the package's other
bootstrap bounds do.

In plain words: the bootstrap runs the test again many times, each time on
a system drawn from the fitted mode population: which modes surface, how
often and (for the AMPM) when, each mode given a fix-effectiveness factor
drawn from the ones assessed. Each simulated test is then projected again
the same way, and the spread of those projections gives the bounds. The
assessed factors themselves are held fixed: they are engineering
judgements, not results of the test, so the bounds show the test's
sampling variation given those judgements, not doubt about the judgements.

There are two things a bound can be about, chosen with ``target``:

- ``target="realised"`` (the default): the intensity (or reliability) that
  this system will have once its seen modes are fixed. That value is
  itself uncertain, because nobody knows how large the unseen modes are,
  or exactly how large the seen ones are, so this is a prediction
  interval. Each simulated system's own realised value is drawn too, and
  the bounds come from the simulated projections' errors (on the log of
  the intensity, or of minus the log of the reliability).
- ``target="expected"``: the model's projected quantity, its average over
  tests like this one, the quantity the AMPM's formula estimates. These
  are the percentiles of the simulated projections.

The two behave differently as a test goes on. The Stein models learn each
seen mode's rate from its repeats, so their realised bounds close up
(above, from about 80% of the projection wide at 300 hours to 7% at
30,000). The expected bounds do not, because one test only ever shows one
draw of the mode population, and the population's average stays as
uncertain as that draw. The AMPM uses only first occurrences, not
repeats, so neither of its bounds closes up. Its estimate of the B modes'
total intensity comes from how quickly new modes appear, and a longer
test does not sharpen that. Where the failure counts are known,
AMPM-Stein gives the tighter answer.

.. code:: python

    result = ampm_stein_projection(x, modes, fef, c=c)
    result.projected_mtbf_cb(bound="lower", random_state=1)   # 24.4

A simulated test that cannot be projected in the usual way is projected at
its limit, not dropped. That happens when the first occurrences do not
thin out (the AMPM, which then shows no growth) or no mode is seen twice
(the Stein models, which then give no credit for the fixes). There are no
Wald or likelihood-ratio bounds: asking for one is refused. The
projection's estimates sit on the edge of their range for tests only a
little smaller than the one seen, which is exactly where those
approximations fail.

*Checked against:* simulated tests of 20,000 modes (factors 0.5, 0.7 or 0.9
drawn independently of the rates; about 20 B modes and 60 failures a
continuous test, 9 modes and 20 failures in 40 one-shot trials), 100 tests
of 200 replicates, seed 1126. The two-sided 90% bounds covered their
targets in 92% (AMPM, realised), 87% (AMPM, expected), 90% and 89%
(AMPM-Stein) and 90% and 92% (discrete) of tests. The unseen modes' draws
match their exact means and variances. The full study (400 tests of 400
replicates, one-sided bounds too) runs with ``--run-calibration``.

Reference
---------

.. autofunction:: surpyval.recurrent.ampm_projection

.. autoclass:: surpyval.recurrent.AMPMProjection
   :members:

.. autofunction:: surpyval.recurrent.ampm_stein_projection

.. autoclass:: surpyval.recurrent.SteinProjection
   :members:

.. autofunction:: surpyval.recurrent.adpm_stein_projection

.. autoclass:: surpyval.recurrent.DiscreteSteinProjection
   :members:
