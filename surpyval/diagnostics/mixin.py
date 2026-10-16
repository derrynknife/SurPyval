"""``diagnose()`` on the fitted models (#867).

The mixin is all a model class needs: the method looks the model's family
up in :mod:`surpyval.diagnostics.registry` when called, so this module
imports nothing heavy and adding a family never touches a model class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from .report import Diagnosis


class DiagnoseMixin:
    """Gives a fitted model :meth:`diagnose`."""

    def diagnose(
        self, alpha: float = 0.05, columns: "Sequence[str] | None" = None
    ) -> "Diagnosis":
        """
        Check the model's assumptions, fit and influential points, and
        say what to do about any that fail.

        Runs the checks this kind of model calls for (for a Cox model:
        proportional hazards, the functional form of each continuous
        covariate, influence and outliers; see
        :doc:`/Model Diagnostics`) and returns a report: ``d.table``,
        one row per check with its test, statistic, p-value, verdict
        (``"ok"``, ``"warn"`` or ``"fail"``) and covariates; ``d.advice``,
        plain-English remedies that each name the SurPyval feature that
        applies them; ``d.plot()``, the visual checks (matplotlib is only
        imported then); ``d.ok``, True if nothing failed;
        ``d.raise_if_failed()`` for pipelines; and ``d.to_dict()``.

        A test fails when its p-value is below ``alpha``, and warns when
        it is below ``2 * alpha`` (borderline) or could not be computed.
        Each row is judged at ``alpha`` on its own, so with many rows
        some fail by chance even when every assumption holds.

        Parameters
        ----------
        alpha : float, optional
            The level each test is judged at. Default 0.05.
        columns : str or list of str, optional
            The covariates to check, named as ``summary()`` names them
            (the DataFrame or formula columns, else ``coef_0``, ...); all
            of them by default. A joint (``GLOBAL``) test still covers
            every covariate.

        Returns
        -------
        Diagnosis
            See :class:`surpyval.diagnostics.Diagnosis`.

        Raises
        ------
        NotImplementedError
            For a model whose family's diagnostics are not built yet,
            naming the issue that tracks them, and for an option of the
            model the checks do not support yet (a stratified Cox fit).
        ValueError
            For a model the checks cannot be run on (one restored with
            ``from_dict``, which has no data), saying what to do.

        Examples
        --------
        >>> from surpyval import CoxPH
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> df["censored"] = 1 - df["arrest"]
        >>> model = CoxPH.fit_from_df(
        ...     df, x_col="week", c_col="censored", Z_cols=["fin", "age"]
        ... )
        >>> d = model.diagnose()
        >>> d.ok
        False
        >>> d.table[["check", "transform", "covariates", "verdict"]].head(3)
                          check transform covariates verdict
        0  proportional_hazards        km     (fin,)      ok
        1  proportional_hazards        km     (age,)    fail
        2  proportional_hazards        km (fin, age)    fail
        """
        from .registry import diagnose

        return diagnose(self, alpha=alpha, columns=columns)
