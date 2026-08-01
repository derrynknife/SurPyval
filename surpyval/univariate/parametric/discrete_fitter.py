"""Base class for the discrete lifetime distributions.

The discrete/continuous distinction used to live in scattered per-class
conventions: each discrete distribution individually set
``supports_mpp = False``, and nothing structurally stopped a
continuous-only estimation method from being asked of a discrete model.
This class is now the single home for that distinction.

A discrete distribution here is one whose mass sits on integers: ``df``
is the probability mass function ``P(T = k)`` (not a density), ``hf`` is
the discrete hazard ``P(T = k) / R(k - 1)``, and the survival ``sf(k)``
is ``P(T > k)``.
"""

from .parametric_fitter import ParametricFitter


class DiscreteParametricFitter(ParametricFitter):
    """A :class:`ParametricFitter` for distributions on the integers.

    Centralises what discreteness means for fitting:

    - ``discrete`` is ``True`` (``ParametricFitter`` declares ``False``),
      so callers can branch on the trait instead of keeping a list.
    - Probability plotting (``how="MPP"``) is rejected: it assumes a
      continuous, invertible CDF.
    - Maximum product of spacings (``how="MPS"``) is rejected: spacings
      are increments of a continuous CDF, and repeated integer
      observations make them degenerate.

    MLE, MSE (least squares against the nonparametric estimate, which is
    a step function anyway) and MOM (via each distribution's ``moment``)
    remain available.
    """

    discrete = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Kept as an instance attribute (not only the class trait) because
        # the shared ``_validate_fit_inputs`` reads it for every fitter.
        self.supports_mpp = False
