"""
Model diagnostics: ``model.diagnose()`` (#867).

Every fitted regression model has one method, ``diagnose(alpha=0.05,
columns=None)``. It runs the checks its family's assumptions call for and
returns a :class:`Diagnosis`:

>>> from surpyval import CoxPH
>>> from surpyval.datasets import load_rossi_static
>>> df = load_rossi_static()
>>> x, c = df["week"].values, 1 - df["arrest"].values
>>> d = CoxPH.fit(x, df[["fin", "prio"]].values, c=c).diagnose()
>>> d.ok
True
>>> sorted(set(d.table["check"]))
['functional_form', 'influence', 'outliers', 'proportional_hazards']

``d.table`` has one row per check (test, statistic, p-value, verdict,
covariates), ``d.advice`` the remedies, each naming the SurPyval feature
that applies it, ``d.plot()`` draws the visual checks (matplotlib is
imported only then), ``d.raise_if_failed()`` raises for a pipeline and
``d.to_dict()`` is the machine-readable report.

Verdicts
--------
A row that is a test is judged by :func:`~surpyval.diagnostics.report.verdict`:
``"fail"`` when ``p < alpha``; ``"warn"`` when ``alpha <= p < 2 * alpha``
(borderline) or when the test could not be computed (``p`` is ``nan``,
and the row's ``detail`` says why); ``"ok"`` otherwise. A row that is a
screen, not a test (influence, outliers), sets ``"ok"`` or ``"warn"`` by
its family's documented rule and never fails. ``d.ok`` is True when no
row failed. Every row is judged at ``alpha`` on its own: with many rows
some fail by chance even when every assumption holds.

Families built so far: Cox proportional hazards (:mod:`.cox`, #868). The
other models' ``diagnose()`` refuses with ``NotImplementedError``, naming
the issue that tracks its family (:data:`.registry.PENDING`).

Adding a family
---------------
A family is one module that lists its checks and advice; the report, the
verdicts, the advice engine, the ``columns=`` and ``alpha`` handling and
the plot layout are shared. :mod:`surpyval.diagnostics.cox` is the model
to copy. The steps:

1. **Write ``surpyval/diagnostics/<family>.py``.** In it:

   - For each assumption, a :class:`~.registry.Check` with a snake-case
     ``name`` (the table's ``check`` and the name ``plot(which=...)``
     takes), a one-sentence ``assumption``, and ``rows(ctx)`` returning
     the table rows: dicts with ``test`` (the test's name), ``covariates``
     (a sequence of covariate names: one for a per-covariate test, all for
     a joint one, none for a model-wide one), ``statistic``, ``df``,
     ``p``, ``detail`` (a few words on what was found) and the family's
     own columns. Leave ``verdict`` out of a test's row (the shared rule
     sets it from ``p``); set it, to ``"ok"`` or ``"warn"``, only for a
     screen that is not a test, and document that screen's rule in the
     module docstring. Keys starting with ``_`` are not shown: they carry
     what the advice rules need (a covariate's kind, a row number).
   - ``panels(ctx)``, if the check has a plot: a list of
     :class:`~.registry.Panel`, each a title and ``draw(ax, options)``,
     which draws on one matplotlib axes. Import matplotlib (and
     :mod:`surpyval.plotting`) inside ``draw``, never at module level.
     Use ``options.rng`` and ``options.n_boot`` for any resampling and
     ``1 - options.alpha`` for a band's level. A check that is a plot
     only has ``rows=None``.
   - ``ctx`` is a :class:`~.registry.Context`: ``ctx.model``,
     ``ctx.alpha``, ``ctx.names`` (every covariate's name, as
     ``summary()`` names them), ``ctx.checked`` (``(position, name)`` of
     the covariates ``columns=`` chose), ``ctx.cached(key, compute)``
     (compute residuals once for all the checks and panels) and
     ``ctx.notes`` (append a note, e.g. for a covariate a check skipped).
   - The :class:`~.registry.AdviceRule` list: for a check's rows with a
     verdict in ``on`` (``("fail",)`` by default) for which
     ``when(row, ctx)`` holds, the advice ``message(row, ctx)``. Advise
     only SurPyval's own fixes, and list each one the message names in
     ``features``, spelt as the API spells it (``"CoxPH.fit(strata=)"``,
     ``"WeibullAFT.fit"``, or ``"summary(robust=)"`` for a method of the
     fitted model): ``tests/diagnostics/test_framework.py`` resolves every
     feature of every registered family and fails for one that does not
     exist. Never advise a statistic the report did not compute. A rule
     with ``check=None`` is asked once, with ``row=None``, about the
     model (the Cox family's clustered standard errors).
   - ``validate(model)``: raise, with a message that names the fix, for
     a model of the family the checks cannot handle (restored without
     its data, no finite maximum, an option not supported yet --
     ``NotImplementedError`` naming the issue for the last).
   - ``notes(ctx)``: the notes every report of the family carries (what
     the tests are, what is not checked and why).
   - Register it at the bottom: ``register(Family(name=..., issue=...,
     applies=..., checks=(...), advice=(...), validate=..., notes=...,
     columns=(...)))``. ``applies(model)`` is True for the family's
     fitted models; import their class inside it. ``columns`` names the
     family's own table columns (the Cox ``transform``; a ``cause`` for
     competing risks); they follow ``check``. ``options`` names any
     keyword options of the family's ``diagnose()`` besides ``alpha`` and
     ``columns`` (an accelerated life model's ``use_stress``): the checks
     read them from ``ctx.options``, and every other family refuses
     them.

2. **Add the module to** :data:`.registry.FAMILY_MODULES` **and delete
   the family's entries from** :data:`.registry.PENDING`.

3. **Make sure the model class has the method**: it inherits
   :class:`~.mixin.DiagnoseMixin` (every model with a ``PENDING`` entry
   already does). Nothing else in the model changes.

4. **Test it** in ``surpyval/tests/diagnostics/test_<family>.py``: the
   statistics against a reference (R, stored as a fixture by
   ``scripts/reference``, as ``test_cox.py`` does), each advice rule
   firing on a constructed failure and not on data that meet the
   assumption, each refusal, and ``plot()`` under the Agg backend. The
   shared report tests (``test_framework.py``) cover ``to_dict``,
   ``raise_if_failed``, ``columns=`` and ``alpha`` for every family.

5. **Document it** in ``docs/Model Diagnostics.rst`` (a section with a
   runnable example, its checks and their rules) and add a changelog
   entry.
"""

from typing import Any

__all__ = ["Diagnosis", "Advice", "DiagnosticsFailed", "diagnose"]


def __getattr__(name: str) -> Any:
    # The report needs pandas: loaded on first use, not by
    # ``import surpyval`` (the models import the mixin from here, #470).
    if name in ("Diagnosis", "Advice", "DiagnosticsFailed"):
        from . import report

        return getattr(report, name)
    if name == "diagnose":
        from .registry import diagnose

        return diagnose
    raise AttributeError(
        "module 'surpyval.diagnostics' has no attribute {!r}".format(name)
    )
