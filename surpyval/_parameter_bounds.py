"""The public range of each parameter of a fitted model.

:class:`ParameterBoundsMixin` gives a model the read-only
``parameter_bounds``: one ``(low, high)`` pair per parameter, in the order
of the vector its ``with_params`` takes (``covariance_names``), ``None``
for an unbounded end. A model supplies the pairs through
``_declared_bounds()``, which by default is its private
``_parameter_bounds()`` (the ranges its confidence bounds and fits already
keep to).
"""

from __future__ import annotations

import math
from typing import Any


def _end(value: Any) -> float | None:
    """One end of a range: ``None`` where it is unbounded (``None`` or an
    infinity), else the number as a float."""
    if value is None:
        return None
    value = float(value)
    return None if math.isinf(value) else value


class ParameterBoundsMixin:
    """Adds :attr:`parameter_bounds` to a fitted model."""

    def _declared_bounds(self) -> list:
        return list(self._parameter_bounds())  # type: ignore[attr-defined]

    @property
    def parameter_bounds(self) -> list[tuple[float | None, float | None]]:
        """
        The range of each parameter: one ``(low, high)`` pair per entry of
        the vector :meth:`with_params` takes, in the order of
        ``covariance_names`` (``None`` for an end with no bound).

        A parameter set drawn inside these ranges is one ``with_params``
        does not refuse for being out of range, and one just outside a
        finite end is refused. Draw a parameter bounded on one side on the
        log scale of its distance from that end, and one bounded on both
        on the logit scale between them. Whether an end itself is allowed
        depends on the parameter (a probability may be 0 where a scale may
        not), so keep draws strictly inside.

        A few models also constrain their parameters jointly, which no
        per-parameter range can say, and ``with_params`` refuses a set
        that breaks the joint constraint, saying so: a mixture's weights
        sum to 1; a Uniform's ``a`` is below its ``b`` (as a Beta4's
        are); a Royston-Parmar spline does not fall; a Binomial's ``n`` is
        a whole number. The entry of a parameter ``with_params`` ignores
        (held fixed in the fit, or an accelerated life model's life
        parameter) gives the parameter's range all the same. An
        accelerated life model's life-model parameters can have ranges
        that depend on the stresses fitted to; a model loaded from a dict
        saved without its data gives them as unbounded.

        Examples
        --------
        >>> from surpyval import Weibull, WeibullPH
        >>> Weibull.from_params([10.0, 2.0]).parameter_bounds
        [(0.0, None), (0.0, None)]
        >>> model = WeibullPH.fit(
        ...     x=[1.0, 2, 3, 4, 5, 6, 7, 8], Z=[0.0, 1, 0, 1, 0, 1, 1, 0]
        ... )
        >>> model.covariance_names
        ['alpha', 'beta', 'coef_0']
        >>> model.parameter_bounds
        [(0.0, None), (0.0, None), (None, None)]
        """
        return [
            (_end(low), _end(high)) for low, high in self._declared_bounds()
        ]
