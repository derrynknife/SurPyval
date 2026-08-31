from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from autograd.numpy.numpy_boxes import ArrayBox

# The NHPP likelihoods differentiate these functions with autograd, so a
# parameter (and hence any value computed from one) may be a plain array,
# a float, or an autograd ArrayBox -- the same union the univariate
# distributions use. ``ArrayLike`` would be a trap here: it admits str
# and bytes, so the concrete models' arithmetic would not type check.
Boxable = npt.NDArray | float | ArrayBox


class CountingProcess(ABC):
    """
    Abstract base class for parametric counting-process intensity models.

    A counting process ``N(t)`` counts the number of events observed up to
    time ``t``. Every parametric counting process in surpyval is described
    by functions of time and its parameters:

    - ``iif`` the instantaneous intensity function (the rate of events),
    - ``cif`` the cumulative intensity function (the expected number of
      events by ``t``, i.e. the integral of ``iif`` from the origin), and
    - ``log_iif`` the natural logarithm of the instantaneous intensity,
      used directly in the log-likelihood.

    Concrete subclasses also expose a ``param_names`` attribute listing the
    names of the model's parameters. This shared base lets fitters such as
    :class:`ProportionalIntensityNHPP` verify, with a simple ``isinstance``
    check, that the intensity model handed to them really is a counting
    process.
    """

    #: Names of the model's parameters (see the class docstring).
    param_names: list

    @abstractmethod
    def iif(self, x: Boxable, *params: Boxable) -> Boxable:
        """Instantaneous intensity function (event rate) at ``x``."""
        ...

    @abstractmethod
    def cif(self, x: Boxable, *params: Boxable) -> Boxable:
        """Cumulative intensity (expected event count) by ``x``."""
        ...

    @abstractmethod
    def log_iif(self, x: Boxable, *params: Boxable) -> Boxable:
        """Natural logarithm of the instantaneous intensity at ``x``."""
        ...


class IntensityModel(CountingProcess):
    """
    Contract shared by the closed-form NHPP intensity baselines
    (:class:`Crow-AMSAA <surpyval.recurrent.parametric.crow_amsaa.CrowAMSAA_>`,
    :class:`Duane <surpyval.recurrent.parametric.duane.Duane_>`,
    :class:`Cox-Lewis <surpyval.recurrent.parametric.cox_lewis.CoxLewis_>`).

    These models are mathematically distinct but expose the *same* four
    functions of time and their parameters, plus a parameter initialiser. The
    contract -- and the docstrings describing each function -- lives here once
    so the concrete models only have to supply the maths:

    - ``cif(x, *params)`` cumulative intensity (expected event count) by ``x``,
      the integral of ``iif`` from the origin, with ``cif(0) == 0``;
    - ``iif(x, *params)`` instantaneous intensity (event rate) at ``x``;
    - ``log_iif(x, *params)`` natural logarithm of ``iif``, used directly in
      the log-likelihood for numerical stability;
    - ``inv_cif(N, *params)`` the time at which ``N`` events are expected to
      have occurred, i.e. the inverse of ``cif``;
    - ``parameter_initialiser(x)`` a starting parameter vector for the
      optimiser given the event times ``x``.

    All array arguments are evaluated elementwise.
    """

    @abstractmethod
    def inv_cif(self, N: Boxable, *params: Boxable) -> Boxable:
        """Time by which ``N`` events are expected; the inverse of ``cif``."""
        ...

    def parameter_initialiser(self, x: npt.ArrayLike) -> npt.NDArray:
        """Starting parameter vector for the optimiser given event times.

        All three baselines start the search from a vector of ones, so
        that is the default; a subclass whose parameters need a data-
        driven start overrides this.
        """
        return np.ones(len(self.param_names), dtype=float)
