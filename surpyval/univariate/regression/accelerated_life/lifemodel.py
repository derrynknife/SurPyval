from abc import ABC, abstractmethod
from collections.abc import Sequence

from autograd import numpy as np
from numpy import ndarray

from surpyval.utils.fitter_repr import FitterRepr

#: An absolute temperature below this, in kelvin (-73 degrees Celsius), is
#: far colder than any life test: every stress of a kelvin column below it
#: is most likely a temperature typed in degrees Celsius (#654).
KELVIN_WARNING_BELOW = 200.0


class LifeModel(FitterRepr, ABC):
    """
    Base class for the stress-life relationships used by
    ``AcceleratedLife``: a function :math:`L(Z)` giving the life parameter
    of a distribution at stress :math:`Z`.

    A subclass passes its ``name``, a ``phi_param_map`` from parameter
    name to position, and ``phi_bounds`` (one ``(lower, upper)`` pair per
    parameter, ``None`` for unbounded) to this constructor, and implements
    :meth:`phi` and :meth:`phi_init`. The built-in life models are
    instances of subclasses: ``Power``, ``InversePower``, ``Eyring``,
    ``InverseEyring``, ``ExponentialLifeModel``, ``InverseExponential``,
    ``Linear``, ``DualExponential``, ``DualPower``, ``PowerExponential``
    and ``GeneralLogLinear``.

    A life model whose parameters depend on the number of stress columns
    (``GeneralLogLinear``, one coefficient per column) sets
    ``n_stresses = None`` and overrides :meth:`resolve`, which the fit
    calls with the number of columns of ``Z`` to get the model with a
    fixed ``phi_param_map`` and ``phi_bounds``.

    Examples
    --------
    ``Power`` is one, with :math:`L(Z) = a Z^n`:

    >>> import numpy as np
    >>> from surpyval.life_models import LifeModel, Power
    >>> isinstance(Power, LifeModel)
    True
    >>> Power.phi_param_map
    {'a': 0, 'n': 1}
    >>> Power.phi(np.array([1.0, 2.0, 4.0]), 1000.0, -2.0)
    array([1000.  ,  250.  ,   62.5])
    """

    #: The ``repr``: ``Power: life model`` (#614)
    fitter_kind = "life model"

    #: The number of stress columns ``Z`` has (``None`` when it depends on
    #: the data, as for ``GeneralLogLinear``). Lets a single 1-D row
    #: ``[T, V]`` be read as one two-stress row rather than two stresses.
    n_stresses: "int | None" = 1
    #: Stress columns that must be strictly positive: the life model takes
    #: a power or logarithm of them (``Z**n``, ``log Z``), or reads them as
    #: an absolute temperature.
    positive_stress_columns: "tuple[int, ...]" = ()
    #: Stress columns read as an absolute temperature, in kelvin (the
    #: Arrhenius-type models): the fit refuses a value <= 0 there, naming
    #: kelvin, and warns when every value is below
    #: ``KELVIN_WARNING_BELOW`` (a temperature typed in degrees Celsius,
    #: #654).
    kelvin_stress_columns: "tuple[int, ...]" = ()
    #: Whether the fit warns when every value of a kelvin column is below
    #: ``KELVIN_WARNING_BELOW``; ``Eyring``, also used for a non-thermal
    #: stress, does not.
    warns_below_kelvin: bool = True
    #: Whether :meth:`phi` takes a 2-D array of stress rows and gives one
    #: life per row, as the built-in models do; the fit then finds every
    #: row's life in one call. ``False`` (the default, for a custom model
    #: written for a single stress) calls it once per distinct stress.
    phi_takes_rows: bool = False
    #: The parameters that multiply the life, a positive factor (``c`` of
    #: ``PowerExponential``, ``a`` of ``Power``): :meth:`log_life` takes
    #: each as its log, and the fit searches it on the log scale over its
    #: whole range (#634).
    log_scale_parameters: "tuple[str, ...]" = ()

    def __init__(
        self,
        name: str,
        phi_param_map: dict[str, int],
        phi_bounds: tuple[tuple[int | None, int | None], ...],
    ) -> None:
        self.name = name
        self.phi_param_map = phi_param_map
        self.phi_bounds = phi_bounds

    def resolve(self, n_stresses: int) -> "LifeModel":
        """
        The life model for ``n_stresses`` stress columns. A model with a
        fixed number of parameters is the same for any number, and
        returns itself (a wrong number of columns is refused by the fit);
        ``GeneralLogLinear`` returns the model with one coefficient per
        column.

        Examples
        --------
        >>> from surpyval.life_models import GeneralLogLinear, Power
        >>> Power.resolve(1) is Power
        True
        >>> GeneralLogLinear.resolve(2).phi_param_map
        {'c': 0, 'coef_0': 1, 'coef_1': 2}
        """
        return self

    @abstractmethod
    def phi(self, Z: ndarray, *params: float) -> ndarray:
        """
        The life :math:`L(Z)` at stress ``Z`` for the life-model
        parameters ``params``. Must be written with ``autograd.numpy`` so
        the fit can differentiate it.
        """

    def log_life(self, Z: ndarray, *params: float) -> ndarray:
        r"""
        The log of the life, :math:`\ln L(Z)`, with each of
        :attr:`log_scale_parameters` given as its log: for a life model
        whose log-life is linear in them (``PowerExponential``'s
        :math:`\ln c + a / U + n \ln V`), which its :meth:`phi` is the
        exponent of. The fit computes the life from it, on the log scale
        it searches those parameters on, so that the life stays finite
        where a factor alone would not (:math:`e^{a/U}` overflowing, or
        ``c`` underflowing, where their product is an ordinary life,
        #634). Only for a model with :attr:`log_scale_parameters`.
        """
        raise NotImplementedError(
            "{} has no log_life: it has no log_scale_parameters".format(
                self.name
            )
        )

    def _phi_from_log_life(self, Z: ndarray, params: tuple) -> ndarray:
        """:meth:`phi` as the exponent of :meth:`log_life` at the
        parameters ``params``, each of :attr:`log_scale_parameters` given
        as itself (its log taken here)."""
        logged = list(params)
        with np.errstate(divide="ignore"):
            for name in self.log_scale_parameters:
                i = self.phi_param_map[name]
                logged[i] = np.log(params[i])
        return np.exp(self.log_life(Z, *logged))

    @abstractmethod
    def phi_init(self, life: float, Z: ndarray) -> list[float]:
        """
        Starting values for the life-model parameters, given the life
        estimated separately at each distinct stress level (``life``, one
        value per row of ``Z``). Typically a least-squares fit of the
        linearised relationship.
        """

    def named(self, names: "Sequence[str]") -> "LifeModel":
        """
        This life model with the coefficients of :meth:`coefficient_columns`
        named ``names``, in column order (a fit's covariate columns,
        #614): ``GeneralLogLinear``'s are ``coef_0``, ``coef_1``, ...
        unless named. A model without such coefficients returns itself.

        Examples
        --------
        >>> from surpyval.life_models import GeneralLogLinear, Power
        >>> Power.named(["temp"]) is Power
        True
        >>> GeneralLogLinear.resolve(2).named(["temp", "volt"]).phi_param_map
        {'c': 0, 'temp': 1, 'volt': 2}
        """
        return self

    def coefficient_columns(self) -> "dict[str, int]":
        """
        The life-model parameters that are each the coefficient of one
        column of ``Z`` as it is (the log-life linear in that column), by
        name, with the column's number: the fit searches and judges each
        in its covariate's units (``coefficient_floor``, #612). None for a
        life model of a transformed stress (``log s``, ``1 / s``); one per
        column for ``GeneralLogLinear``.

        Examples
        --------
        >>> from surpyval.life_models import GeneralLogLinear, Power
        >>> Power.coefficient_columns()
        {}
        >>> GeneralLogLinear.resolve(2).coefficient_columns()
        {'coef_0': 0, 'coef_1': 1}
        """
        return {}

    def _stress_terms(
        self, Z: ndarray
    ) -> "tuple[ndarray, tuple[str, ...], bool] | None":
        """
        The terms of the stresses the log-life is linear in, for the
        check of which stress effects the data determine (#503): one
        column per stress column of ``Z`` (``log s`` for a power term,
        ``1 / s`` for an exponential one), the life-model parameter each
        multiplies, and whether the model has a free constant factor
        (an intercept on the log scale, which absorbs a constant
        stress). ``None`` for a life model with no such form, or with one
        stress, where one stress level is refused already.
        """
        return None
