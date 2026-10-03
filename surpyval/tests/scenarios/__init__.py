"""Practitioner scenario cards.

Each module is one *card*: an end-to-end study a reliability engineer
runs with SurPyval, on data simulated from a known truth, in the shape
the data arrive in the field (a maintenance system's date table, a
Nevada chart of warranty returns, readouts of an accelerated test ...).
The unit and conformance suites check that each function keeps its
contract; a card checks that the functions add up to an answer.

A card has

- a **persona** and a **question set** taken from the standard
  references for that kind of study (Meeker and Escobar, Nelson,
  Abernethy, MIL-HDBK-189C, IEC 61508), written in the card's docstring;
- a **generator** with a fixed seed and the **truth** it simulates from;
- **oracles** for the answers: the truth itself (within the fit's own
  confidence interval), an independent implementation of the same
  likelihood written in the card (the maximum the package reports must
  be the maximum), or an equivalent parameterisation that must give the
  same likelihood;
- **gaps**: each question the package cannot yet answer without the
  analyst writing it by hand is a strict xfail, its reason led by the
  issue (``"#578: ..."``). When the issue is fixed the test passes, the
  strict xfail turns the run red, and the card is updated to the API the
  fix chose.

A card should run in seconds. Studies that need thousands of repeats
(coverage, bias) belong in ``surpyval/tests/calibration``.

The cards run with ``--run-scenarios`` (nightly; see ``conftest.py`` and
``docs/Contributing.rst``).
"""
