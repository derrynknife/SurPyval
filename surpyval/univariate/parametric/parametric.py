from __future__ import annotations

import warnings
from collections import namedtuple
from copy import copy, deepcopy
from math import comb
from typing import TYPE_CHECKING, Any, Callable

import autograd.numpy as np
import numpy.typing as npt
from autograd import jacobian
from scipy.special import expit
from scipy.special import ndtri as z
from scipy.stats import uniform

import surpyval as surv
from surpyval import ParametricDistribution
from surpyval.serialisation import SerialisableMixin, stamp_schema, to_native
from surpyval.univariate.information_criteria import (
    InformationCriteriaMixin,
    ic_sample_size,
)
from surpyval.utils import fsli_to_xcnt, refuse_time_values
from surpyval.utils.data_summary import data_summary
from surpyval.utils.deprecation import renamed_arguments
from surpyval.utils.linalg import (
    cb_link,
    param_name,
    sf_link_bound,
    wald_undefined,
    warn_wald_undefined,
)
from surpyval.utils.rng import as_generator
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.surpyval_data import SurpyvalData
from surpyval.utils.validation import (
    BOUNDS,
    CB_ON,
    alpha_ci_error,
    check_option,
    no_covariance_error,
    option_error,
)

from ._likelihood_ratio import (
    _LN_MAX,
    _LN_TINY,
    LikelihoodRatioMixin,
    _central_gradient,
)
from .probability_plotting import (
    adjust_heuristic,
    curve_plot_data,
    draw_probability_plot,
    probability_plot_data,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# Shared inputs for the confidence-bound computations: the fitted parameter
# vector ``phi_hat`` (core params plus any LFP/ZI parameters), its covariance
# ``cov``, and ``n_core`` (the number of leading core parameters). These three
# always travel together, so they are bundled to keep the bound helpers'
# signatures small.
# Why a fitted model has no parameter covariance.
_NO_COVARIANCE_WHY = (
    "the Hessian was singular at the optimum, or the model was not fit by MLE"
)

_CBContext = namedtuple("_CBContext", ["phi_hat", "cov", "n_core"])

# The values of ``Parametric.maximum``: whether the log-likelihood a model
# reports is at a maximum (principles 12 and 13). The first three are what
# a maximum-likelihood fit reached, and agree with its warnings: a fit
# that warns "No finite maximum" is ``"no finite maximum"``, one that
# warns its search did not reach a verified maximum is ``"unverified"``,
# and a fit that warns neither is ``"verified"``.
MAXIMUM_STATES = (
    "verified",
    "unverified",
    "no finite maximum",
    "not applicable",
    "unknown",
)


def draw_state(random_state: Any = None) -> Any:
    """The ``random_state`` to give a numpy or scipy draw.

    ``None`` stays ``None``, numpy's global stream, so ``np.random.seed``
    reproduces a draw exactly as it did before ``random_state`` was an
    argument. Anything else is ``as_generator(random_state)``: a stream
    of its own, with an int seed meaning ``np.random.default_rng(seed)``
    (scipy's ``rvs`` would otherwise read an int as a legacy
    ``RandomState`` seed).
    """
    return None if random_state is None else as_generator(random_state)


def uniform_draws(
    size: int | tuple[int, ...], random_state: Any = None
) -> npt.NDArray:
    """Uniform draws on (0, 1) of shape ``size``: see :func:`draw_state`."""
    return uniform.rvs(size=size, random_state=draw_state(random_state))


def is_custom_distribution(dist: Any) -> bool:
    """Whether ``dist`` is, or discretizes, a ``CustomDistribution``."""
    # Imported here since the distributions import this module
    from .distributions.custom_distribution import CustomDistribution
    from .distributions.discretize import DiscretizedFitter

    if isinstance(dist, DiscretizedFitter):
        return is_custom_distribution(dist.dist)
    return isinstance(dist, CustomDistribution)


def resolve_distribution(name: str, custom: bool = False) -> Any:
    """The distribution a serialised model names, for ``from_dict``.

    - SurPyval's own distributions are looked up by name among the
      package's exports, and only there, so an untrusted dictionary cannot
      resolve arbitrary attributes.
    - ``"Discretize(<name>)"`` is rebuilt as ``Discretize`` of the
      distribution ``<name>`` resolves to.
    - A ``CustomDistribution`` holds user functions, which a dictionary
      cannot carry. Constructing one registers it under its name, so a
      model of it is read back in any session that has constructed the
      same distribution again (``custom`` says the dictionary came from
      one, and makes the registry take precedence over a built-in of the
      same name). Otherwise a ``ValueError`` says what to do.
    """
    from .distributions.custom_distribution import registered_custom
    from .distributions.discretize import Discretize
    from .parametric_fitter import ParametricFitter

    if name.startswith("Discretize(") and name.endswith(")"):
        return Discretize(resolve_distribution(name[11:-1], custom))
    if custom:
        dist = registered_custom(name)
        if dist is not None:
            return dist
        raise ValueError(
            f"'{name}' is a CustomDistribution, whose cumulative hazard "
            "cannot be stored in a dictionary. Construct it again with "
            f"CustomDistribution('{name}', ...) in this session, then "
            "read the dictionary back."
        )
    dist = getattr(surv, name, None)
    if isinstance(dist, ParametricFitter):
        return dist
    # A dictionary written before custom models were flagged
    dist = registered_custom(name)
    if dist is not None:
        return dist
    raise ValueError(f"Unknown distribution '{name}'")


class Parametric(
    LikelihoodRatioMixin,
    InformationCriteriaMixin,
    SerialisableMixin,
    ParametricDistribution,
):
    """
    Result of ``.fit()`` or ``.from_params()`` method for every parametric
    surpyval distribution.

    Instances of this class are very useful when a user needs the other
    functions of a distribution for plotting, optimizations, monte carlo
    analysis and numeric integration.

    Attributes
    ----------
    maximum : str
        Whether the fit's answer is a maximum of the likelihood, and so
        whether its log-likelihood, ``aic``, ``bic`` and ``aic_c`` and its
        standard errors mean what they say (principles 12 and 13):

        - ``"verified"``: a maximum-likelihood fit whose answer is
          verifiably a maximum (a zero gradient and a positive-definite
          Hessian), or exact (a closed form);
        - ``"unverified"``: a maximum-likelihood fit whose search did not
          reach a verified maximum; the parameters are the best point it
          found, and the fit warned so;
        - ``"no finite maximum"``: a maximum-likelihood fit to data on
          which the likelihood has no finite maximum (a parameter runs
          off to a limit of its range); the fit warned "No finite
          maximum", and the parameters are where the search stopped;
        - ``"not applicable"``: the parameters do not come from
          maximising the likelihood -- a fit by ``how="MPS"``, ``"MSE"``,
          ``"MPP"`` or ``"MOM"``, ``fit_from_ecdf``, or a model built
          with ``from_params``;
        - ``"unknown"``: a maximum-likelihood fit restored from a
          dictionary saved before the attribute existed.

        :func:`~surpyval.fit_best.fit_best` ranks a candidate by its criterion
        only when this is ``"verified"``, unless nothing else fitted.

    Examples
    --------
    >>> import surpyval as surv
    >>> model = surv.Weibull.fit([10, 12, 15, 17, 20, 25, 31])
    >>> type(model).__name__
    'Parametric'
    >>> model.params.round(3)
    array([20.881,  2.931])
    >>> model.sf([10, 20]).round(4)
    array([0.8909, 0.4143])
    >>> model.maximum
    'verified'

    A model built from parameters has the same functions. With a limited
    failure population, one unit in ten never fails, so the mean life is
    infinite:

    >>> lfp = surv.Weibull.from_params([20, 3], p=0.9)
    >>> float(lfp.sf(1000.0).round(4)), lfp.mean(), lfp.extras
    (0.1, inf, {'p': 0.9})
    >>> lfp.maximum
    'not applicable'
    """

    # Attributes populated after construction (by ``fit``, ``from_dict``
    # or ``from_params``). Declared here so static type checkers know
    # their types; the bare annotations do not create the attributes, so
    # the ``hasattr``/``getattr`` guards throughout still behave.
    params: npt.NDArray
    gamma: float
    p: float
    f0: float
    support: tuple[float, float]
    hess_inv: npt.NDArray
    cov_matrix: npt.NDArray
    surv_data: "SurpyvalData"
    fitting_info: dict[str, Any]
    optimizer: str
    maximum: str
    tl: Any
    # The printout's "Data" line of a model restored without its
    # data (#508)
    _data_summary: "str | None" = None
    tr: Any
    lfp_name: str
    _neg_ll: float
    _mean: float
    _bic: float
    _aic: float
    _aic_c: float

    def __init__(
        self,
        dist: Any,
        method: str,
        data: Any,
        offset: bool,
        lfp: bool,
        zi: bool,
    ) -> None:
        self.dist = dist
        self.k = copy(dist.k)
        self.method = method
        self.data = data
        self.offset = offset
        self.lfp = lfp
        self.zi = zi
        # Only a maximum-likelihood fit has a maximum to report; it sets
        # the status it reached (see ``MAXIMUM_STATES``).
        self.maximum = "unknown" if method == "MLE" else "not applicable"

        bounds = deepcopy(dist.bounds)
        param_map = dist.param_map.copy()

        if offset:
            if data is not None:
                x_min = np.asarray(data["x"])
                if zi:
                    # Exact zeros belong to the zero-inflation mass, so
                    # they must not cap the offset of the continuous part
                    x_min = x_min[x_min != 0]
                bounds = ((None, np.min(x_min)), *bounds)
            else:
                bounds = ((None, None), *bounds)

            param_map = {k: v + 1 for k, v in param_map.items()}
            param_map.update({"gamma": 0})
            self.k += 1
        else:
            self.gamma = 0

        # The limited-failure proportion is addressed as ``p`` (in
        # ``fixed``, ``param_cb`` and the repr) -- unless the distribution
        # has a parameter of its own called ``p`` (Geometric,
        # NegativeBinomial). Both keys then landed on the same
        # ``param_map`` entry, the proportion overwrote the distribution's
        # parameter, and the map came out one entry short of the bounds:
        # every such LFP fit died in a zip() length check, and
        # ``param_cb('p')`` read the distribution's ``p`` as the
        # proportion. The distribution keeps ``p`` and the proportion
        # becomes ``lfp_p``.
        self.lfp_name = "lfp_p" if "p" in dist.param_map else "p"
        if lfp:
            bounds = (*bounds, (0, 1))
            param_map.update({self.lfp_name: len(param_map)})
            self.k += 1
        else:
            self.p = 1

        if zi:
            bounds = (*bounds, (0, 1))
            param_map.update({"f0": len(param_map)})
            self.k += 1
        else:
            self.f0 = 0

        self.bounds = bounds
        self.param_map = param_map

    @classmethod
    def from_dict(cls, model_dict: dict) -> "Parametric":
        """
        Rebuild a model from the dictionary written by :meth:`to_dict`.

        Parameters
        ----------
        model_dict : dict
            A dictionary produced by :meth:`to_dict` (for example read
            back from JSON or a document store).

        Returns
        -------
        Parametric
            The restored model. Methods that need the original data
            (``plot``, likelihood-ratio bounds) work only if the
            dictionary was written with ``with_data=True``; ``aic``,
            ``bic`` and ``aic_c`` work from the stored likelihood and
            sample size either way.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> from surpyval.univariate.parametric.parametric import Parametric
        >>> model = Weibull.from_params([10, 2])
        >>> Parametric.from_dict(model.to_dict()).params
        array([10,  2])
        """
        if model_dict["parameterization"] != "parametric":
            raise ValueError(
                "Must create parametric model from parametric model dict"
            )

        dist = resolve_distribution(
            model_dict["distribution"], bool(model_dict.get("custom", False))
        )
        # A variable-arity distribution supplies the instance sized to
        # these parameters; every other distribution returns itself.
        dist = dist._for_params(model_dict["params"])
        if len(model_dict["params"]) != dist.k:
            raise ValueError(
                f"The dictionary holds {len(model_dict['params'])} "
                f"parameter(s) but the distribution '{dist.name}' has "
                f"{dist.k}."
            )
        how = model_dict["how"]
        if "data" in model_dict:
            # Coerce the JSON lists back to arrays so downstream users
            # (``bic``, ``aic_c``, re-serialisation with data) work on a
            # restored model just as on a fitted one (#261).
            data = {k: np.asarray(v) for k, v in model_dict["data"].items()}
        else:
            data = None
        offset = model_dict["offset"]
        lfp = model_dict["lfp"]
        zi = model_dict["zi"]
        out = cls(dist, how, data, offset, lfp, zi)

        if offset:
            out.gamma = model_dict["gamma"]

        if lfp:
            out.p = model_dict["p"]

        if zi:
            out.f0 = model_dict["f0"]

        if "hess_inv" in model_dict:
            out.hess_inv = np.array(model_dict["hess_inv"])

        if "cov_matrix" in model_dict:
            out.cov_matrix = np.array(model_dict["cov_matrix"])

        if "_neg_ll" in model_dict:
            out._neg_ll = model_dict["_neg_ll"]

        # The sample size of bic() and aic_c(), so they work -- and agree
        # with the fitted model -- without the data. Dicts written before
        # this key existed need the data for them.
        out._ic_n = cls._restored_ic_n(model_dict)

        # The parameters fixed at fit time are not estimated, so they do
        # not count towards the k of aic() and bic(); restoring them keeps
        # a round-tripped model's criteria equal to the fitted one's.
        # Dicts written before this key existed have none.
        fixed_names = model_dict.get("fixed") or []
        if fixed_names:
            out.fitting_info = {
                "fixed_idx": [out.param_map[name] for name in fixed_names]
            }

        out.params = np.array(model_dict["params"])

        # Dicts written before ``"maximum"`` existed keep the constructor's
        # value: "unknown" for a maximum-likelihood fit, "not applicable"
        # for any other.
        if "maximum" in model_dict:
            maximum = model_dict["maximum"]
            if maximum not in MAXIMUM_STATES:
                raise ValueError(
                    f"The dictionary's 'maximum' is {maximum!r}; it must be "
                    f"one of {list(MAXIMUM_STATES)}."
                )
            out.maximum = maximum
        out._data_summary = model_dict.get("data_summary")

        # Restore the support interval, which fit-time construction sets via
        # the fitter (#261).
        dist._set_support(out, offset)

        return out

    def to_dict(self, with_data: bool = False) -> dict:
        """
        Serialise the model to a dictionary of plain Python types.

        The dictionary holds the distribution name, the parameters, the
        offset / LFP / ZI settings, the names of any parameters fixed at fit
        time (``"fixed"``), whether the fit reached a maximum of the
        likelihood (``"maximum"``, see :class:`Parametric`) and, if
        available, the parameter covariance,
        fitted negative log-likelihood and the sample size of BIC and
        AIC_c (``"ic_n"``), so a restored model can compute confidence
        bounds, ``aic``, ``bic`` and ``aic_c``. Restore it with
        :meth:`from_dict` or ``surpyval.from_dict``.

        Parameters
        ----------
        with_data : bool, optional
            If :code:`True`, also store the ``x``, ``c``, ``n``, ``t`` data
            the model was fitted to, which ``plot`` and likelihood-ratio
            bounds need. Defaults to :code:`False`.
            ``to_json(path, with_data=True)`` writes this to a file.

        Returns
        -------
        dict
            A strict-JSON dictionary: non-finite values (such as the
            untruncated ``-inf``/``inf`` bounds in the data) are ``None``,
            recorded under ``"non_finite"`` and restored by
            :meth:`from_dict` (see :doc:`/surpyval.serialisation`).

        Examples
        --------
        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 2])
        >>> d = model.to_dict()
        >>> d["distribution"], d["params"]
        ('Weibull', [10, 2])
        """
        out: dict[str, Any] = {}
        out["parameterization"] = "parametric"
        out["distribution"] = self.dist.name
        if is_custom_distribution(self.dist):
            # Read back from the CustomDistribution registry rather than
            # from SurPyval's own distributions (see resolve_distribution).
            out["custom"] = True
        out["how"] = self.method
        out["param_names"] = self.dist.parameter_names

        data_dict: dict[str, Any] = {}
        if with_data:
            if self.data is not None:
                for ch in ["x", "c", "n", "t"]:
                    if self.data[ch] is None:
                        data_dict[ch] = []
                    else:
                        data_dict[ch] = self.data[ch].tolist()
                out["data"] = data_dict

        out["params"] = np.array(self.params).tolist()
        out["lfp"] = bool(self.lfp)

        if self.lfp:
            out["p"] = to_native(self.p)
        else:
            out["p"] = 1.0

        out["zi"] = bool(self.zi)
        if self.zi:
            out["f0"] = to_native(self.f0)
        else:
            out["f0"] = 0.0

        out["offset"] = bool(self.offset)
        if self.offset:
            out["gamma"] = to_native(self.gamma)
        else:
            out["gamma"] = 0.0

        if getattr(self, "hess_inv", None) is not None:
            out["hess_inv"] = self.hess_inv.tolist()
        if getattr(self, "cov_matrix", None) is not None:
            out["cov_matrix"] = self.cov_matrix.tolist()
        if hasattr(self, "_neg_ll"):
            out["_neg_ll"] = to_native(self._neg_ll)
        # Informational: a reader that predates it ignores it and restores
        # the same model, so it needs no newer schema.
        out["maximum"] = self.maximum
        # The printout's "Data" line (#508), informational like "maximum",
        # so a model restored without its data prints the same.
        if self._data_repr():
            out["data_summary"] = self._data_repr()
        ic_n = self._ic_sample_size_or_none()
        if ic_n is not None:
            out["ic_n"] = ic_n

        fixed_idx = sorted(self._user_fixed_idx())
        if fixed_idx:
            # Named, not indexed, so the entry reads on its own; from_dict
            # maps the names back through the rebuilt param_map.
            names = {i: name for name, i in self.param_map.items()}
            out["fixed"] = [names[i] for i in fixed_idx]

        return stamp_schema(out)

    @property
    def parameter_names(self) -> list[str]:
        """
        The names of ``params``, entry by entry: the distribution's
        ``parameter_names``. An offset ``gamma``, limited-failure
        proportion ``p`` and zero-inflation fraction ``f0`` are not in
        ``params`` and not named here (see :attr:`extras`).

        Examples
        --------
        >>> from surpyval import Weibull
        >>> Weibull.from_params([100, 2]).parameter_names
        ['alpha', 'beta']
        """
        return list(self.dist.parameter_names)

    @property
    def extras(self) -> dict[str, float]:
        """
        The offset, limited-failure proportion and zero-inflation fraction
        the model carries, as the keywords of ``from_params``.

        Only the ones the model has are included: ``"gamma"`` for an offset
        model, ``"p"`` for a limited-failure-population model (also for a
        ``Geometric`` or ``NegativeBinomial``, whose proportion is printed
        as ``lfp_p``: ``from_params`` takes it as ``p``) and ``"f0"`` for a
        zero-inflated one; a plain model gives an empty dict. So
        ``dist.from_params(params, **model.extras)`` rebuilds the model with
        other parameters, which :meth:`with_params` does. The dict is a
        copy; changing it does not change the model.

        Returns
        -------
        dict
            ``{name: value}`` for each of ``gamma``, ``p`` and ``f0`` the
            model has.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> Weibull.from_params([100, 2], gamma=5.0, p=0.9, f0=0.1).extras
        {'gamma': 5.0, 'p': 0.9, 'f0': 0.1}
        >>> Weibull.from_params([100, 2]).extras
        {}
        """
        out: dict[str, float] = {}
        # Keyed on the model's structure, as to_dict is, not on the values:
        # a model fitted with lfp=True keeps its p even where it came out
        # at 1.
        if self.offset:
            out["gamma"] = float(self.gamma)
        if self.lfp:
            out["p"] = float(self.p)
        if self.zi:
            out["f0"] = float(self.f0)
        return out

    def with_params(self, params: npt.ArrayLike) -> "Parametric":
        """
        The same model with other distribution parameters.

        The distribution, offset ``gamma``, limited-failure proportion
        ``p`` and zero-inflation fraction ``f0`` are kept (see
        :attr:`extras`); only the distribution's own parameters change. Use
        it to perturb or redraw a fitted model's parameters (sensitivity
        or uncertainty analyses): ``from_params(model.params)`` alone
        silently drops the offset, ``p`` and ``f0``.

        Parameters
        ----------
        params : array like
            The distribution's parameters, in the order of
            ``model.dist.parameter_names``. They are checked as
            ``from_params`` checks them.

        Returns
        -------
        Parametric
            A model built from parameters, as ``from_params`` builds it:
            it has no data, covariance or fitted likelihood, since those
            belong to the fit of the original parameters, so data-based
            methods (``plot``, confidence bounds, ``aic``) are not
            available on it.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([100, 2], gamma=5.0, p=0.9, f0=0.1)
        >>> other = model.with_params([120, 2])
        >>> other.params, other.gamma, other.p, other.f0
        (array([120,   2]), 5.0, 0.9, 0.1)
        """
        return self.dist.from_params(params, **self.extras)

    def __repr__(self) -> str:
        if hasattr(self, "params"):
            param_string = "\n".join(
                [
                    f"{name:>10}: {p}"
                    for p, name in zip(self.params, self.dist.parameter_names)
                ]
            )
            out = (
                "Parametric SurPyval Model"
                "\n========================="
                f"\nDistribution        : {self.dist.name}"
                f"\nFitted by           : {self.method}"
            )
            data_line = self._data_repr()
            if data_line:
                out += f"\nData                : {data_line}"
            if self.offset:
                out += f"\nOffset (gamma)      : {self.gamma}"

            if self.lfp:
                label = f"Max Proportion ({self.lfp_name})"
                out += f"\n{label:<20}: {self.p}"

            if self.zi:
                out += f"\nZero-Inflation (f0) : {self.f0}"

            out = out + "\nParameters          :\n" + param_string

            return out
        else:
            return "Unable to fit values"

    def _data_repr(self) -> str:
        """The data the model was fitted to, in one line, for the printout
        (#508): units weighted by ``n``, by kind of censoring and
        truncation. Empty for a model built from parameters; a model
        restored without its data gives the line it was saved with."""
        data = getattr(self, "data", None)
        if not isinstance(data, dict) or "c" not in data:
            return getattr(self, "_data_summary", None) or ""
        t = np.asarray(data.get("t", np.empty((0, 2))), dtype=float)
        lower, upper = np.asarray(self.support, dtype=float)
        x = data.get("x")
        if t.size == 0:
            return data_summary(data["c"], data.get("n"), x=x)
        return data_summary(
            data["c"], data.get("n"), t[:, 0], t[:, 1], lower, upper, x=x
        )

    def param_cb(
        self,
        name: str,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "wald",
    ) -> npt.NDArray:
        """
        Method to calculate the confidence bound on a parameter.

        Two interval methods are available via ``method``:

        - ``"wald"`` (default) -- a symmetric bound from the parameter's
          standard error, computed on a scale chosen from the parameter's
          support (log for a positive parameter, logit for a probability) so
          the interval stays valid.
        - ``"lr"`` -- a profile-likelihood (likelihood-ratio) bound. The
          interval is the set of values whose profile deviance
          :math:`2[\\ell(\\hat\\theta) - \\ell_p(\\theta)]` stays below the
          :math:`\\chi^2_1` critical value, with the remaining parameters
          re-optimised at each candidate. It is transformation-invariant,
          respects the parameter's boundary, and need not be symmetric about
          the estimate -- usually better small-sample coverage than Wald, and
          the reliability-engineering default. Aliases: ``"likelihood"``,
          ``"likelihood-ratio"``, ``"profile"``. The interval is the
          stretch around the estimate where the deviance stays below the
          critical value; where it stays below it out to the edge of the
          parameter's space (levelling off there, as a NegativeBinomial's
          ``r`` does as the model tends to a Poisson), the bound is that
          edge: 0, 1 or ``inf``. A side whose bound cannot be found is
          ``nan``, with a warning.

        A parameter fixed at fit time is known, so both methods give the
        degenerate interval at its value. A Wald bound does not exist
        where the parameter's variance (from the inverse observed
        information) is negative -- the information is not positive
        definite, typically because the estimate is at or near a boundary
        of the parameter space -- or where the estimate is on the edge of
        its support: it is ``nan`` there, with a warning saying why.

        Parameters
        ----------
        name : str
            The parameter, by name (e.g. ``"alpha"``; ``"p"`` for a
            limited-failure model, ``"f0"`` for a zero-inflated one). A
            distribution parameter named ``p`` (``Geometric``,
            ``NegativeBinomial``) keeps its name, and the limited-failure
            proportion of such a model is ``"lfp_p"``. The offset
            ``"gamma"`` has no confidence bound: it is a threshold
            parameter, whose likelihood is not regular, so no standard
            error is estimated for it.
        alpha_ci : float, optional
            The significance level: 0.05 (the default) gives a 95% bound.
        bound : str, optional
            ``"two-sided"`` (the default), ``"upper"`` or ``"lower"``.
        method : str, optional
            ``"wald"`` (the default) or ``"lr"``, as above.

        Returns
        -------
        numpy array
            ``[lower, upper]`` for a two-sided bound, else the one bound.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> np.random.seed(1)
        >>> model = Weibull.fit(Weibull.random(30, 10, 3))
        >>> model.param_cb("alpha")
        array([ 7.83345374, 10.56940099])
        >>> model.param_cb("beta", method="lr")
        array([1.82826755, 3.27740643])
        """
        if self._is_lr(method):
            return self._param_cb_lr(name, alpha_ci, bound)

        is_core, idx = self._resolve_param_name(name)
        if not is_core:
            cov = getattr(self, "cov_matrix", None)
            if cov is None:
                raise no_covariance_error(_NO_COVARIANCE_WHY)
            p_hat = self.f0 if name == "f0" else self.p
            var = cov[idx, idx]
            param_bounds: tuple[float | None, float | None] = (0, 1)
        else:
            p_hat = self.params[idx]
            hess_inv = getattr(self, "hess_inv", None)
            if hess_inv is None:
                raise no_covariance_error(_NO_COVARIANCE_WHY)
            var = hess_inv[idx, idx]
            param_bounds = self.dist.bounds[idx]

        check_option("bound", bound, BOUNDS)
        if bound == "two-sided":
            alpha = alpha_ci / 2
            bounds = np.array([-1, 1])
        elif bound == "lower":
            alpha = alpha_ci
            bounds = np.array([-1])
        else:
            alpha = alpha_ci
            bounds = np.array([1])

        # The edge only matters to the log and logit scales used below.
        edges = param_bounds if param_bounds in ((0, None), (0, 1)) else ()
        reason = wald_undefined(p_hat, var, *edges)
        if reason is not None:
            # Was nan with numpy's raw sqrt warning alone (#411).
            warn_wald_undefined(param_name(name), reason, stacklevel=2)
            return np.full(bounds.shape, np.nan)

        if param_bounds == (0, None):
            exponent = z(alpha) * np.sqrt(var) / p_hat
            bounds = -bounds * exponent
            return p_hat * np.exp(bounds)
        elif param_bounds == (0, 1):
            # Bounds on the logit keep the result within (0, 1)
            u_hat = np.log(p_hat / (1 - p_hat))
            diff = -bounds * z(alpha) * np.sqrt(var) / (p_hat * (1 - p_hat))
            return 1 / (1 + np.exp(-(u_hat + diff)))
        else:
            factor = z(alpha) * np.sqrt(var)
            bounds = -bounds * factor
            return p_hat + bounds

    def _resolve_param_name(self, name: str) -> tuple[bool, int]:
        """Locate the parameter ``name`` for a confidence bound.

        Returns ``(True, i)`` for the distribution's own ``i``-th
        parameter and ``(False, j)`` for the limited-failure proportion or
        the zero-inflation fraction, ``j`` being its index in the extended
        covariance ``cov_matrix`` (core parameters, then ``p``, then
        ``f0``). The distribution's parameters are looked up first, so a
        ``Geometric`` ``p`` is never mistaken for the LFP proportion (which
        is then ``lfp_p``, see ``__init__``). Anything else -- the offset,
        or a name the model does not have -- raises a ``ValueError``
        naming the valid choices rather than a bare ``KeyError``.
        """
        if name in self.dist.param_map:
            return True, self.dist.param_map[name]
        if name == self.lfp_name:
            if not self.lfp:
                raise ValueError(f"'{name}' is only estimated for lfp models")
            return False, len(self.params)
        if name == "f0":
            if not self.zi:
                raise ValueError("'f0' is only estimated for zi models")
            return False, len(self.params) + int(self.lfp)
        if name == "gamma":
            if not self.offset:
                raise ValueError("'gamma' is only estimated for offset models")
            # mle holds gamma out of the covariance: the threshold of an
            # offset model is non-regular (the likelihood's support moves
            # with it), so a Wald variance for it would be misleading.
            raise ValueError(
                "No confidence bound is available for the offset 'gamma': "
                "it is a threshold parameter whose likelihood is not "
                "regular, so no standard error is estimated for it."
            )
        valid = list(self.dist.parameter_names)
        if self.lfp:
            valid.append(self.lfp_name)
        if self.zi:
            valid.append("f0")
        raise ValueError(
            f"Unknown parameter {name!r}; expected one of {valid}"
        )

    def _ensure_surv_data(self) -> None:
        """Make ``surv_data`` available for the likelihood-ratio bounds.

        A fitted model holds it; a model restored from a dictionary
        written with ``to_dict(with_data=True)`` holds the same data as
        its ``x``, ``c``, ``n`` and ``t`` arrays, so the ``SurpyvalData``
        is rebuilt from those. Only without the data at all is there
        nothing to profile.
        """
        if hasattr(self, "surv_data"):
            return
        if self.data is None:
            raise ValueError(
                "Likelihood-ratio bounds need the original data, which this "
                "model does not have (it was built from parameters, or "
                "restored from a dict saved without its data -- save it "
                "with to_dict(with_data=True) to keep them); use "
                "method='wald' (which uses the stored covariance)."
            )
        self.surv_data = SurpyvalData(
            x=self.data["x"],
            c=self.data["c"],
            n=self.data["n"],
            t=self.data["t"],
        )

    def _is_fixed_param(self, name: str) -> bool:
        """Whether ``name`` was fixed at fit time."""
        # fixed_idx indexes param_map (which leads with gamma for an
        # offset model and ends with p / f0), so look the name up there.
        return self.param_map.get(name) in self._user_fixed_idx()

    def _user_fixed_idx(self) -> set:
        """``param_map`` indices of the parameters the user fixed at fit
        time (empty set for models without fitting info, e.g.
        ``from_params``). Without an offset these are the core-parameter
        indices; the likelihood-ratio bounds that read them as such reject
        offset models."""
        info = getattr(self, "fitting_info", None) or {}
        return set(info.get("fixed_idx", []) or [])

    def sf(self, x: npt.ArrayLike) -> npt.NDArray:
        r"""

        Survival (or Reliability) function for a distribution using the
        parameters found in the ``.params`` attribute.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the survival
            function will be calculated.

        Returns
        -------

        sf : scalar or numpy array
            The scalar value of the survival function of the distribution if
            a scalar was passed. If an array like object was passed then a
            numpy array is returned with the value of the survival function at
            each corresponding value in the input array.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 3])
        >>> model.sf(2)
        np.float64(0.9920319148370607)
        >>> model.sf([1, 2, 3, 4, 5])
        array([0.9990005 , 0.99203191, 0.97336124, 0.938005  , 0.8824969 ])
        """
        refuse_time_values(x, "x")
        x = np.asarray(x)
        xg = x - self.gamma  # type: ignore[operator]
        base_sf = self.dist.sf(xg, *self.params)
        # Below the (possibly offset) support the base distribution has not
        # started: clamp to R0 = 1 rather than evaluating the base function
        # at a negative argument (#256).
        s0 = getattr(self.dist, "support", (-np.inf, np.inf))[0]
        base_sf = np.where(xg < s0, 1.0, base_sf)
        out = 1 - self.p + (self.p - self.f0) * base_sf
        if self.f0 != 0:
            # The zero-inflation mass sits at 0, so before 0 nothing has
            # failed yet: R = 1 there, not 1 - f0.
            out = np.where(np.asarray(x) < 0, 1.0, out)[()]
        return out

    def ff(self, x: npt.ArrayLike) -> npt.NDArray:
        r"""

        The cumulative distribution function, or failure function, for a
        distribution using the parameters found in the ``.params`` attribute.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the failure function
            (CDF) will be calculated.

        Returns
        -------

        ff : scalar or numpy array
            The scalar value of the CDF of the distribution if a scalar was
            passed. If an array like object was passed then a numpy array is
            returned with the value of the CDF at each corresponding value in
            the input array.

        Examples
        --------

        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 3])
        >>> model.ff(2)
        np.float64(0.007968085162939372)
        >>> model.ff([1, 2, 3, 4, 5])
        array([0.0009995 , 0.00796809, 0.02663876, 0.061995  , 0.1175031 ])
        """
        refuse_time_values(x, "x")
        x = np.asarray(x)
        xg = x - self.gamma  # type: ignore[operator]
        base_ff = self.dist.ff(xg, *self.params)
        # Below the (possibly offset) support the base CDF is 0; evaluating
        # the base function at a negative argument gave F < 0 (#256).
        s0 = getattr(self.dist, "support", (-np.inf, np.inf))[0]
        base_ff = np.where(xg < s0, 0.0, base_ff)
        out = self.f0 + (self.p - self.f0) * base_ff
        if self.f0 != 0:
            # The zero-inflation mass f0 arrives at 0, not before it.
            out = np.where(np.asarray(x) < 0, 0.0, out)[()]
        return out

    def df(self, x: npt.ArrayLike, continuous: bool = False) -> npt.NDArray:
        r"""

        The density function for a distribution using the parameters found
        in the ``.params`` attribute.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the density function
            will be calculated.
        continuous : bool, optional
            Only matters for a zero-inflated model. If :code:`True`, return
            the density of the continuous part alone, ``p - f0`` times
            the base density at ``x - gamma``, with no point mass at 0:
            it integrates to
            ``p - f0``, so it is the one to integrate numerically (a
            convolution, the trapezoidal rule). Defaults to
            :code:`False`, which returns the point mass ``f0`` at
            exactly 0, as the likelihood uses it.

        Returns
        -------

        df : scalar or numpy array
            The scalar value of the density function of the distribution if
            a scalar was passed. If an array like object was passed then a
            numpy array is returned with the value of the density function at
            each corresponding value in the input array.

        Examples
        --------

        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 3])
        >>> model.df(2)
        np.float64(0.01190438297804473)
        >>> model.df([1, 2, 3, 4, 5])
        array([0.002997  , 0.01190438, 0.02628075, 0.04502424, 0.06618727])

        For a zero-inflated model, ``df(0)`` is the point mass ``f0`` (a
        probability, not a density); ``continuous=True`` leaves it out:

        >>> zi = Weibull.from_params([100, 2], f0=0.1)
        >>> zi.df(0.0), zi.df(0.0, continuous=True)
        (np.float64(0.1), np.float64(0.0))

        Notes
        -----
        A zero-inflated model is a mixture of a point mass ``f0`` at 0 and
        a continuous part of mass ``p - f0``. By default ``df`` returns the
        point mass itself at exactly ``x == 0``, since that is the
        probability an observation at 0 gets in the likelihood. A density
        integrated over a grid that starts at 0 then counts a spurious
        ``f0 * dx / 2`` (trapezoidal rule); use ``continuous=True`` for
        that, and add the mass ``f0`` at 0 separately if it is wanted.
        """
        refuse_time_values(x, "x")
        x = np.asarray(x)
        xg = x - self.gamma  # type: ignore[operator]
        base_df = self.dist.df(xg, *self.params)
        # Below the (possibly offset) support the density is 0 (#256).
        s0 = getattr(self.dist, "support", (-np.inf, np.inf))[0]
        base_df = np.where(xg < s0, 0.0, base_df)
        if self.f0 == 0 or continuous:
            # (p - f0) is p itself without zero inflation
            df = (self.p - self.f0) * base_df
        else:
            # The continuous part carries mass (p - f0) — the same constant
            # as sf/ff and the likelihood; (1 - f0) * p was inconsistent
            # with them for combined LFP + ZI models (#256). [()] makes a
            # scalar argument give a scalar, not a 0-d array.
            df = np.where(x == 0, self.f0, (self.p - self.f0) * base_df)[()]
        return df

    def hf(self, x: npt.ArrayLike) -> npt.NDArray:
        r"""
        The instantaneous hazard function for a distribution using the
        parameters found in the ``.params`` attribute.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the instantaneous
            hazard function will be calculated.

        Returns
        -------

        hf : scalar or numpy array
            The scalar value of the instantaneous hazard function of the
            distribution if a scalar was passed. If an array like object was
            passed then a numpy array is returned with the value of the
            instantaneous hazard function at each corresponding value in
            the input array.

        Examples
        --------

        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 3])
        >>> model.hf(2)
        np.float64(0.012000000000000002)
        >>> model.hf([1, 2, 3, 4, 5])
        array([0.003, 0.012, 0.027, 0.048, 0.075])
        """
        refuse_time_values(x, "x")
        x = np.asarray(x)
        if (self.p == 1) and (self.f0 == 0):
            xg = x - self.gamma  # type: ignore[operator]
            s0 = getattr(self.dist, "support", (-np.inf, np.inf))[0]
            out = np.where(xg < s0, 0.0, self.dist.hf(xg, *self.params))
            # ``np.where`` hands back a 0-d array for scalar input, so a
            # scalar argument used to come out as ``array(0.012)`` while
            # every sibling method returned a numpy scalar. ``[()]`` is a
            # no-op on a real array and unwraps the 0-d case.
            return out[()]
        elif self.dist.discrete:
            # A discrete hazard is conditioned on survival to the step
            # before, h(k) = P(T = k) / R(k - 1), as the distributions'
            # own hf; df / sf(k) disagreed with it for an LFP or ZI model.
            return self.df(x) / self.sf(np.asarray(x, dtype=float) - 1.0)
        else:
            return self.df(x) / self.sf(x)

    def Hf(self, x: npt.ArrayLike) -> npt.NDArray:
        """
        The cumulative hazard function for a distribution using the
        parameters found in the ``.params`` attribute.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the cumulative
            hazard function will be calculated

        Returns
        -------

        Hf : scalar or numpy array
            The scalar value of the cumulative hazard function of the
            distribution if a scalar was passed. If an array like object was
            passed then a numpy array is returned with the value of the
            cumulative hazard function at each corresponding value in the
            input array.

        Examples
        --------

        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 3])
        >>> model.Hf(2)
        np.float64(0.008000000000000002)
        >>> model.Hf([1, 2, 3, 4, 5])
        array([0.001, 0.008, 0.027, 0.064, 0.125])
        """
        refuse_time_values(x, "x")
        x = np.asarray(x)

        if (self.p == 1) and (self.f0 == 0):
            xg = x - self.gamma  # type: ignore[operator]
            s0 = getattr(self.dist, "support", (-np.inf, np.inf))[0]
            out = np.where(xg < s0, 0.0, self.dist.Hf(xg, *self.params))
            return out[()]
        else:
            # 0.0 - log(...) rather than -log(...): where sf is exactly 1
            # (before 0, or before the offset) the latter gave -0.0.
            return 0.0 - np.log(self.sf(x))

    def qf(self, p: npt.ArrayLike) -> npt.NDArray:
        r"""

        The quantile function for a distribution using the parameters found
        in the ``.params`` attribute.

        Parameters
        ----------
        p : array like or scalar
            The values, which must be between 0 and 1, at which the the
            quantile will be calculated

        Returns
        -------
        qf : scalar or numpy array
            The scalar value of the quantile of the distribution if a
            scalar was passed. If an array like object was passed then a
            numpy array is returned with the value of the quantile at each
            corresponding value in the input array.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 3])
        >>> model.qf(0.2)
        np.float64(6.06542793124108)
        >>> model.qf([.1, .2, .3, .4, .5])
        array([4.72308719, 6.06542793, 7.09181722, 7.99387877, 8.84997045])

        Notes
        -----
        For a model with a limited-failure (cure) fraction the failure
        function only reaches ``p`` in the limit, so any quantile at or above
        ``p`` is infinite (that proportion of the population never fails). For
        a zero-inflated model the mass ``f0`` sits at 0 (not at the offset),
        so quantiles at or below ``f0`` return 0. A probability outside
        [0, 1] gives NaN, as scipy's ``ppf`` does.
        """
        if isinstance(p, list):
            p = np.array(p)
        u = np.asarray(p, dtype=float)
        scalar = u.ndim == 0
        u = np.atleast_1d(u)

        # Invert the mixture failure function
        #   F(x) = f0 + (p - f0) F0(x - gamma):
        #   u <= f0    -> the zero-inflation mass, which sits at 0
        #   f0 < u < p -> gamma + F0^{-1}((u - f0) / (p - f0))
        #   u >= p     -> beyond the attainable proportion (cure), so infinite
        with np.errstate(divide="ignore", invalid="ignore"):
            base = (u - self.f0) / (self.p - self.f0)
            base = np.clip(base, 0.0, 1.0)
            # base == 1 (the u >= p region) makes the base quantile diverge;
            # it is overwritten with inf just below, so silence it here.
            q = self.gamma + self.dist.qf(base, *self.params)
        # The zero-inflation mass sits at 0 — consistent with df (mass at
        # x == 0), ff(0) = f0 and the likelihood — not at the offset (#256).
        # Only where there is such a mass: with none, a continuous
        # distribution's qf(0) is the start of its support (a Normal's
        # -inf; it was 0). A discrete one keeps 0 (Poisson's own qf(0) is
        # scipy's -1).
        at_zero = (self.f0 > 0) or getattr(self.dist, "discrete", False)
        q = np.where(at_zero & (u <= self.f0), 0.0, q)
        # Only with a cure fraction: otherwise qf(1) is the end of the
        # support (a Uniform's upper bound; it was inf).
        q = np.where((self.p < 1) & (u >= self.p), np.inf, q)
        # A probability outside [0, 1] has no quantile: NaN, as scipy's
        # ``ppf`` and ``CustomDistribution.qf`` (#437) give. It was inf
        # above 1 and 0 below 0, even for a Normal (#485).
        q = np.where((u < 0) | (u > 1), np.nan, q)
        q = np.asarray(q, dtype=float)
        return q[0] if scalar else q

    @renamed_arguments(X="given")
    def cs(self, x: npt.ArrayLike, given: npt.ArrayLike) -> npt.NDArray:
        r"""

        The conditional survival of the model; that is, the probability
        that an item that has survived to ``given`` survives a further ``x``:

        .. math::
            R(x, given) = \frac{R(x + given)}{R(given)}

        .. versionchanged:: 0.22
           The time already survived is ``given`` (it was ``X``, which
           still works until v0.23 with a ``DeprecationWarning``), the
           name the regression models' ``sf_tvc(..., given=)`` uses.

        Parameters
        ----------

        x : array like or scalar
            The further durations at which conditional survival is to be
            calculated.
        given : array like or scalar
            The value(s) at which it is known the item has survived

        Returns
        -------

        cs : array
            The conditional survival probability.

        Examples
        --------

        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 3])
        >>> model.cs(11, 10)
        np.float64(0.00025840046151723767)

        Notes
        -----
        The ratio is taken of the model's own :meth:`sf`, so a
        limited-failure proportion ``p``, a zero-inflation fraction ``f0``
        and an offset ``gamma`` all enter it: the never-failing units
        still count among the survivors at ``given``, and survival to an
        ``given`` before the offset is certain. Where :math:`R(given) = 0` the
        conditional survival is undefined and ``nan`` is returned.
        """
        x_arr = np.asarray(x, dtype=float)
        given_arr = np.asarray(given, dtype=float)
        given_g = given_arr - self.gamma
        s0 = getattr(self.dist, "support", (-np.inf, np.inf))[0]
        with np.errstate(all="ignore"):
            # The ratio of the model's own sf. Handing the shifted given to
            # ``dist.cs`` ignored p and f0 entirely (0.29 instead of 0.67
            # for p = 0.7) and, for a given before the offset, evaluated the
            # base sf at a negative time (1.0 or nan instead of 0.96).
            cs = np.asarray(
                self.sf(x_arr + given_arr) / self.sf(given_arr), dtype=float
            )
            if (self.p == 1) and (self.f0 == 0):
                # A plain model inside its support keeps the
                # distribution's own form, which is exact where the ratio
                # cancels in the far tail (the memoryless Exponential).
                inside = np.broadcast_to(given_g >= s0, cs.shape)
                if inside.any():
                    given_g_safe = np.where(given_g >= s0, given_g, s0)
                    own = np.asarray(
                        self.dist.cs(x_arr, given_g_safe, *self.params),
                        dtype=float,
                    )
                    cs = np.where(inside, own, cs)
        cs = np.where(cs > 1.0, 1.0, cs)
        return cs[()]

    def random(
        self,
        size: int | tuple[int, ...],
        a: float | None = None,
        b: float | None = None,
        *,
        random_state: Any = None,
    ) -> npt.NDArray:
        r"""

        A method to draw random lifetimes from the distribution using the
        parameters found in the ``.params`` attribute.

        Each draw is ``qf(u)`` for one uniform ``u``, for every model. With
        no ``random_state`` the uniforms come from numpy's global random
        generator: so ``np.random.seed`` makes the draws reproducible, and
        ``random(size)`` gives the same values as
        ``qf(np.random.random_sample(size))`` after the same seed. A unit
        of a limited-failure population that never fails (``p < 1``) is
        ``inf``, and one dead on arrival (``f0``) is exactly 0. To simulate
        a data set to fit, with the never-failing units right-censored,
        use :meth:`random_data`.

        Parameters
        ----------
        size : int or tuple of ints
            The number (or shape) of random samples to be drawn from the
            distribution.
        a: float or None
            The left truncated value if sampling from a truncated
            distribution
        b: float or None
            The right truncated value if sampling from a truncated
            distribution. Truncated sampling is not available for offset,
            limited-failure or zero-inflated models.
        random_state : int or numpy.random.Generator, optional
            Seed or generator for a reproducible draw of its own, which
            neither depends on nor advances numpy's global stream (an int
            is ``np.random.default_rng(seed)``). ``None`` (the default)
            draws from the global stream, as above.

        Returns
        -------
        random : numpy array
            An array of shape ``size`` of lifetimes drawn from the model.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 3])
        >>> np.random.seed(1)
        >>> model.random(1)
        array([8.14127103])
        >>> model.random(10)
        array([10.84103403,  0.48542084,  7.11387062,  5.41420125, 4.59286657,
                5.90703589,  7.5124326 ,  7.96575225,  9.18134126, 8.16000438])
        >>> lfp = Weibull.from_params([10, 3], p=0.8, f0=0.1)
        >>> np.random.seed(6)
        >>> lfp.random(5)
        array([       inf, 7.38380246,        inf, 0.        , 2.22387058])
        >>> bool(np.array_equal(model.random(3, random_state=1),
        ...                     model.random(3, random_state=1)))
        True
        """
        if ((a is not None) or (b is not None)) and (
            (self.p != 1) or (self.f0 != 0)
        ):
            raise NotImplementedError(
                "Truncated sampling not supported with LFP or ZI models"
            )
        elif ((a is not None) or (b is not None)) and self.offset:
            raise NotImplementedError(
                "Truncated sampling not supported with offset distributions"
            )

        if (self.p == 1) and (self.f0 == 0):
            if (a is None) and (b is None):
                if hasattr(self.dist, "qf"):
                    return (
                        self.dist.qf(
                            uniform_draws(size, random_state), *self.params
                        )
                        + self.gamma
                    )
                else:
                    return self.dist.random(
                        size, *self.params, random_state=random_state
                    )

            else:
                # Truncated sampling
                # F-1(u) = G-1[u(G(b) - G(a)) + G(a)]
                if a is None:
                    Fa = 0
                else:
                    Fa = self.dist.ff(a, *self.params)
                if b is None:
                    Fb = 1
                else:
                    Fb = self.dist.ff(b, *self.params)
                u = uniform_draws(size, random_state)
                return self.dist.qf((u * (Fb - Fa) + Fa), *self.params)

        # One uniform per draw through the model's quantile function: inf
        # for a unit that never fails, 0 for one dead on arrival (#403).
        # This used to return (x, c, n, t) survival data for an LFP model
        # (now random_data), and to draw a zero-inflated sample by a
        # binomial count and a shuffle, which qf(u) could not reproduce.
        return np.reshape(self.qf(uniform_draws(size, random_state)), size)

    def random_data(
        self,
        size: int,
        a: float | None = None,
        b: float | None = None,
        *,
        random_state: Any = None,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        r"""

        Draw a random survival data set from the model, in xcnt format, for
        simulate-and-refit studies (a data set to pass to ``fit``).

        The lifetimes are those of :meth:`random` (the same values after
        the same seed). A unit that never fails (``p < 1``) cannot be
        observed failing, so it is right-censored just after the last
        failure drawn, as if the test stopped there; every other draw is
        an observed failure, with the dead-on-arrival units (``f0``) at
        exactly 0. Repeated values are counted in ``n``.

        Parameters
        ----------
        size : int
            The number of units to draw.
        a: float or None
            The left truncation value if sampling from a truncated
            distribution; it is recorded as every row's left truncation.
        b: float or None
            The right truncation value if sampling from a truncated
            distribution; it is recorded as every row's right truncation.
        random_state : int or numpy.random.Generator, optional
            As for :meth:`random`: ``None`` (the default) draws from
            numpy's global stream, anything else from a stream of its own.

        Returns
        -------
        x, c, n, t : numpy arrays
            The draw in xcnt format: values, censoring flags (0 failed,
            1 right-censored), counts and ``[left, right]`` truncation.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 3], p=0.5)
        >>> np.random.seed(3)
        >>> x, c, n, t = model.random_data(8)
        >>> x
        array([ 6.61335008,  8.11938035,  9.55304821, 10.55304821])
        >>> c, n
        (array([0, 0, 0, 1]), array([1, 1, 1, 5]))
        """
        x = np.ravel(
            self.random(size, a, b, random_state=random_state)
        ).astype(float)
        finite = np.isfinite(x)
        f = x[finite]
        s = np.full(int(np.sum(~finite)), self._censor_time(f))
        xcnt = fsli_to_xcnt(f, s)
        if (a is not None) or (b is not None):
            t = xcnt[3]
            t[:, 0] = -np.inf if a is None else a
            t[:, 1] = np.inf if b is None else b
        return xcnt

    def _censor_time(self, f: npt.NDArray) -> float:
        """A censoring time beyond every drawn failure, valid even when the
        LFP draw produced no failures (``np.max`` of an empty array raised
        before, #256)."""
        if np.size(f):
            return float(np.max(f)) + 1.0
        return float(self.dist.qf(0.999, *self.params)) + self.gamma + 1.0

    def mean(self, defective: bool = False) -> float:
        r"""
        The mean of the distribution using the parameters found in the
        ``.params`` attribute.

        Parameters
        ----------
        defective : bool, optional
            Only matters for a limited-failure-population model
            (``p < 1``). If :code:`False` (the default), the mean
            lifetime, which is infinite there, since a fraction ``1 - p``
            never fails. If :code:`True`, the *defective* mean
            :math:`(p - f_0)\,\mathbb{E}[\gamma + X]`, the integral of
            :math:`t\,dF(t)` over the units that fail, with :math:`X` the
            base distribution; that is not the mean life of the units that
            fail, which is :math:`\gamma + \mathbb{E}[X]`.

        Returns
        -------
        mean : float
            Returns the mean of the distribution.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 3])
        >>> model.mean()
        np.float64(8.929795115692489)
        >>> lfp = Weibull.from_params([100, 2], p=0.9)
        >>> lfp.mean(), lfp.mean(defective=True)
        (inf, np.float64(79.76042329074821))

        Notes
        -----
        With ``p = 1`` the two agree, and give the mean lifetime of the
        model: for a zero-inflated model the mass ``f0`` at 0 contributes
        nothing, so it is :math:`(1 - f_0)\,\mathbb{E}[\gamma + X]`.
        """
        if self.p < 1 and not defective:
            # A fraction 1 - p never fails, so E[T] is infinite (#404).
            return np.inf
        if not hasattr(self, "_mean"):
            # Defective mean: the zero-inflated mass f0 sits at 0 and
            # contributes nothing, so the continuous part carries (p - f0)
            # — ``p`` alone ignored f0 (#256).
            self._mean = (self.p - self.f0) * (
                self.dist.mean(*self.params) + self.gamma
            )
        return self._mean

    def var(self, defective: bool = False) -> float:
        r"""
        The variance of the distribution using the parameters found in the
        ``.params`` attribute.

        Parameters
        ----------
        defective : bool, optional
            Only matters for a limited-failure-population model
            (``p < 1``). If :code:`False` (the default), the variance of
            the lifetime, which is infinite there (a fraction ``1 - p``
            never fails). If :code:`True`, ``moment(2, defective=True) -
            mean(defective=True)**2``, as in the Notes.

        Returns
        -------
        var : float
            Returns the variance of the distribution.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 3])
        >>> model.var()
        np.float64(10.533288486847923)

        Notes
        -----
        For a zero-inflated model (``p = 1``) this is the variance of the
        mixture, the mass ``f0`` sitting at 0. For a limited-failure model
        it is infinite, unless ``defective=True``, which scores the
        never-failing fraction ``1 - p`` at 0 as well (the *defective*
        convention of :meth:`mean` and :meth:`moment`). With
        :math:`q = p - f_0` the proportion failing through the base
        distribution :math:`X` (offset by :math:`\gamma`), both are

        .. math::
            \mathrm{Var}(T) = q\,\mathrm{Var}(X)
                + q(1 - q)\left(\gamma + \mathbb{E}[X]\right)^2,

        which reduces to :math:`\mathrm{Var}(X)` for a plain model (the
        offset does not change a variance). The defective variance of a
        limited-failure model is not a variance conditional on failure
        (fit without ``lfp`` for that).
        """
        if self.p < 1 and not defective:
            # A fraction 1 - p never fails: Var(T) is infinite (#404).
            return np.inf
        m1 = self.dist._moment(1, *self.params)
        m2 = self.dist._moment(2, *self.params)
        base_var = m2 - m1**2
        q = self.p - self.f0
        if q == 1:
            return base_var
        # Written as q Var(X) + q (1 - q) mu^2 rather than as
        # moment(2) - mean()**2, which subtracts two nearly equal numbers
        # once the offset is large. It used to return Var(X) whatever p
        # and f0 were, while mean() already applied the (p - f0) weight.
        return q * base_var + q * (1 - q) * (m1 + self.gamma) ** 2

    def moment(self, n: int, defective: bool = False) -> float:
        r"""

        The n-th moment of the distribution using the parameters found
        in the ``.params`` attribute.

        Parameters
        ----------
        n : integer
            The degree of the moment to be computed
        defective : bool, optional
            Only matters for a limited-failure-population model
            (``p < 1``). If :code:`False` (the default), the moment of the
            lifetime, which is infinite there for ``n >= 1`` (a fraction
            ``1 - p`` never fails). If :code:`True`, the *defective*
            moment described in the Notes.

        Returns
        -------
        moment[n] : float
            Returns the n-th moment of the distribution

        Examples
        --------
        >>> from surpyval import Normal
        >>> model = Normal.from_params([10, 3])
        >>> model.moment(1)
        10.0
        >>> model.moment(5)
        202150.0

        Notes
        -----
        For an offset or zero-inflated model this is the moment of the
        lifetime, consistent with :meth:`mean` (``moment(1) == mean()``):
        the offset shifts the failure times, and the zero-inflated mass
        sits at 0 and contributes nothing to a moment about zero. It is
        :math:`(p - f_0)\,\mathbb{E}\!\left[(\gamma + X)^n\right]` for
        :math:`X` the base distribution. For a limited-failure model the
        moment of the lifetime diverges (those units never fail);
        ``defective=True`` gives the same expression, in which the cured
        fraction ``1 - p`` contributes nothing.
        """
        if self.p < 1 and n >= 1 and not defective:
            # A fraction 1 - p never fails: E[T^n] is infinite (#404).
            return np.inf
        # Defective n-th moment E[(gamma + X)^n] weighted by the failing
        # proportion p; the binomial expansion recombines the base raw
        # moments. Reduces to the base moment for a plain model (gamma = 0,
        # p = 1) and to mean() for n = 1.
        base = [1.0] + [
            float(self.dist.moment(k, *self.params)) for k in range(1, n + 1)
        ]
        shifted = sum(
            comb(n, k) * self.gamma ** (n - k) * base[k] for k in range(n + 1)
        )
        # The zero-inflated mass f0 sits at 0 and contributes nothing to a
        # moment about zero, so the continuous part carries (p - f0) (#256).
        return float((self.p - self.f0) * shifted)

    def entropy(self) -> float:
        r"""
        The entropy of the distribution using the parameters found in
        the ``.params`` attribute.

        Returns
        -------

        entropy : float
            Returns entropy of the distribution

        Examples
        --------
        >>> from surpyval import Normal
        >>> model = Normal.from_params([10, 3])
        >>> model.entropy()
        np.float64(2.5175508218727822)

        Notes
        -----
        The (differential) entropy is translation-invariant, so an offset does
        not change it. It is only defined when the distribution has no
        probability atom; a limited-failure model places mass ``1 - p`` at
        infinity (the cured fraction) and a zero-inflated model places mass
        ``f0`` at 0, so a single differential entropy does not exist
        for those and a ``ValueError`` is raised. The entropy *conditional on
        failure* of a limited-failure model equals the entropy of the same
        model fitted without ``lfp``.
        """
        if self.p == 1 and self.f0 == 0:
            return self.dist.entropy(*self.params)
        raise ValueError(
            "Differential entropy is undefined for a distribution with a "
            "probability atom: a limited-failure model places mass 1 - p at "
            "infinity (the cured fraction never fails) and a zero-inflated "
            "model places mass f0 at 0. Fit without lfp / zi to take "
            "the entropy (which, for lfp, is the entropy conditional on "
            "failure)."
        )

    @keeps_query_shape
    def cb(
        self,
        x: npt.ArrayLike,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "wald",
    ) -> npt.NDArray:
        r"""
        Confidence bounds of the ``on`` function at the ``alpha_ci`` level of
        significance. Can be the upper, lower, or two-sided confidence by
        changing value of ``bound``.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the confidence bounds
            will be calculated
        on : ('sf', 'ff', 'Hf', 'hf', 'df'), optional
            The function on which the confidence bound will be calculated.
            The Wald bounds on ``sf``, ``ff`` and ``Hf`` come from one bound
            on the log cumulative hazard, ``log Hf = log(-log sf)`` (the
            "log-log" transform, on which a Weibull is a straight line in
            log time); those on ``hf`` and ``df`` are on the log scale (the
            logit scale for a discrete distribution, whose hazard and mass
            are probabilities), and are 0 where the rate is 0. Where the
            delta-method variance is negative (the covariance is not
            positive definite) a Wald bound is ``nan``, with a warning.
            The Wald band on ``sf`` and ``ff`` rises (or falls) with ``x``
            as the function does whenever the shape's own Wald interval
            excludes 0; with fewer failures than that it can turn back in a
            tail, and the likelihood-ratio band (``method="lr"``), which is
            always monotone, is the one to use.
        bound : ('two-sided', 'upper', 'lower'), str, optional
            Compute either the two-sided, upper or lower confidence bound(s).
            Defaults to two-sided.
        alpha_ci : scalar, optional
            The level of significance at which the bound will be computed.
        method : ('wald', 'lr'), str, optional
            ``"wald"`` (default) propagates the parameter covariance through
            the ``on`` function by the delta method. ``"lr"`` gives a
            profile-likelihood (likelihood-ratio) band: at each ``x`` the bound
            is the extreme value of the ``on`` function over the parameter
            confidence region ``{theta : 2[nll(theta) - nll_hat] <= chi2}``
            (the piece of it around the estimate), which is where the
            function's own profile deviance reaches ``chi2``; a band whose
            region reaches the edge of the function's range (0 or 1 for
            ``sf``) is that edge. The ``sf``, ``ff`` and ``Hf`` bands are
            one band, so they agree exactly.
            The likelihood-ratio band is transformation-invariant and does not
            rely on a quadratic approximation, so it is usually better in small
            samples (Meeker and Escobar recommend it there), but it is computed
            pointwise and so is slower, needs the original data (a model
            restored from ``to_dict(with_data=True)`` has it; one saved
            without it raises), and is not yet available for offset / LFP /
            ZI models. Where the constrained search cannot find a bound
            from any start, that bound is ``nan``, with a warning.

        Returns
        -------

        cb : scalar or numpy array
            The value(s) of the upper, lower, or both confidence bound(s) of
            the selected function at x. A two-sided bound has one row per
            ``x`` holding the ``[lower, upper]`` pair.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> np.random.seed(1)
        >>> x = Weibull.random(30, 10, 3)
        >>> model = Weibull.fit(x)
        >>> model.cb([5, 10], on="sf")
        array([[0.65083522, 0.88949979],
               [0.1629223 , 0.41352719]])
        >>> model.cb([5, 10], on="sf", bound="lower")
        array([0.67916426, 0.18042915])
        """
        t = np.atleast_1d(x)
        if self.method != "MLE":
            raise ValueError("Only MLE has confidence bounds")
        # Checked up front, as param_cb does: an unrecognised value (say
        # 'both') used to fall through to the lower-bound branch and
        # return one bound as if it were what was asked for.
        check_option("bound", bound, BOUNDS)
        if np.size(t) == 0 and on in CB_ON:
            # Nothing to bound (the Jacobian of no values fails).
            return np.empty((0, 2) if bound == "two-sided" else (0,))

        if self._is_lr(method):
            return self._cb_lr(t, on, alpha_ci, bound)

        ctx = self._cb_context()

        # ff, F and Hf are decreasing transforms of R; flip one-sided bounds
        if on in ["ff", "F", "Hf"] and bound == "lower":
            bound = "upper"
        elif on in ["ff", "F", "Hf"] and bound == "upper":
            bound = "lower"

        old_err_state = np.seterr(all="ignore")
        try:
            if (on == "ff") or (on == "F"):
                cb = 1.0 - self._cb_sf_bound(t, ctx, alpha_ci, bound)
            elif (on == "sf") or (on == "R"):
                cb = self._cb_sf_bound(t, ctx, alpha_ci, bound)
                if bound == "two-sided":
                    cb = np.fliplr(cb)
            elif on == "Hf":
                cb = -np.log(self._cb_sf_bound(t, ctx, alpha_ci, bound))
            elif on in ["hf", "df"]:
                cb = self._cb_rate_bound(t, ctx, alpha_ci, bound, on)
            else:
                raise option_error("on", on, CB_ON)
        finally:
            np.seterr(**old_err_state)

        return cb

    @keeps_query_shape
    def quantile_cb(
        self,
        p: npt.ArrayLike,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "wald",
    ) -> npt.NDArray:
        r"""
        Confidence bounds on the quantile ``qf(p)``: the B-life at ``p``
        (the B10 life is ``p = 0.1``), the time by which a fraction ``p``
        has failed.

        Parameters
        ----------

        p : array like or scalar
            The probabilities, in (0, 1), whose quantiles are bounded.
        alpha_ci : scalar, optional
            The level of significance at which the bound will be computed.
            Defaults to 0.05.
        bound : ('two-sided', 'upper', 'lower'), str, optional
            Compute either the two-sided, upper or lower confidence bound(s).
            Defaults to two-sided.
        method : ('wald', 'lr'), str, optional
            ``"wald"`` (default) is the delta method on the log of the
            quantile above the support's start (``log t_p`` for a lifetime,
            ``log(t_p - gamma)`` with an offset; the quantile itself for a
            distribution on the whole line), from the gradient of ``t_p``
            with respect to the parameters, ``-dF/dtheta / f`` at ``t_p``;
            for the Weibull, :math:`\log t_p = \log\alpha +
            \log(-\log(1 - p))/\beta`, the "Fisher matrix" bound on time
            of Nelson (1982) and of Meeker and Escobar (1998). ``"lr"`` is
            the likelihood-ratio bound: the extreme of ``qf(p)`` over the
            parameters' likelihood region, as ``cb(method="lr")`` is for
            a function of time. It is invariant to the parameterisation
            and better in small samples, slower, and, like ``cb``'s, not
            available for offset, limited-failure or zero-inflated models.

        Returns
        -------

        cb : scalar or numpy array
            The bound(s), shaped as ``p``; a two-sided bound adds a last
            ``[lower, upper]`` axis.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> np.random.seed(1)
        >>> x = Weibull.random(30, 10, 3)
        >>> model = Weibull.fit(x)
        >>> model.qf(0.1).round(3)
        np.float64(3.695)
        >>> model.quantile_cb(0.1).round(3)
        array([2.638, 5.175])
        >>> model.quantile_cb(0.1, bound="lower", method="lr").round(3)
        np.float64(2.648)

        Notes
        -----
        The nonparametric models' ``quantile_cb`` is the same bound for a
        step estimate (Brookmeyer and Crowley); :meth:`mean_cb` bounds the
        mean.
        """
        probs = np.asarray(p, dtype=float)
        self._check_summary_cb(alpha_ci, bound)
        if probs.size == 0:
            return np.empty((0, 2) if bound == "two-sided" else (0,))
        if not np.all((probs > 0) & (probs < 1)):
            raise ValueError(f"'p' must be in (0, 1); got {probs.tolist()}")
        if self.dist.discrete:
            self._is_lr(method)  # checks the name
            return self._quantile_cb_discrete(probs, alpha_ci, bound, method)
        if self._is_lr(method):
            fns = [
                lambda theta, p_i=p_i: self.dist.qf(np.array([p_i]), *theta)[0]
                for p_i in probs
            ]
            return self._summary_cb_lr(fns, alpha_ci, bound, "qf")
        return self._quantile_cb_wald(probs, alpha_ci, bound)

    def mean_cb(
        self,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "wald",
    ) -> Any:
        r"""
        Confidence bounds on the mean, :meth:`mean`.

        Parameters
        ----------

        alpha_ci : scalar, optional
            The level of significance at which the bound will be computed.
            Defaults to 0.05.
        bound : ('two-sided', 'upper', 'lower'), str, optional
            Compute either the two-sided, upper or lower confidence bound(s).
            Defaults to two-sided.
        method : ('wald', 'lr'), str, optional
            ``"wald"`` (default) is the delta method on the log of the mean
            above the support's start (on the mean itself for a
            distribution on the whole line); ``"lr"`` is the
            likelihood-ratio bound, the extreme of the mean over the
            parameters' likelihood region (not available for offset,
            limited-failure or zero-inflated models).

        Returns
        -------

        cb : numpy array or scalar
            The ``[lower, upper]`` interval, or the one bound asked for.
            With a limited failure population (``p < 1``) the mean is
            infinite, and so are its bounds.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> np.random.seed(1)
        >>> x = Weibull.random(30, 10, 3)
        >>> model = Weibull.fit(x)
        >>> model.mean().round(3)
        np.float64(8.073)
        >>> model.mean_cb().round(3)
        array([6.935, 9.398])

        Notes
        -----
        The nonparametric models' ``mean_cb`` bounds their (restricted)
        mean; :meth:`quantile_cb` bounds a quantile.
        """
        self._check_summary_cb(alpha_ci, bound)
        if self.p < 1:
            # A fraction 1 - p never fails: E[T] is infinite (#404).
            inf = np.inf
            return np.array([inf, inf]) if bound == "two-sided" else inf
        if self._is_lr(method):
            fns = [lambda theta: self.dist.mean(*theta)]
            out = self._summary_cb_lr(fns, alpha_ci, bound, "mean")
            return out[0]
        ctx = self._cb_context()

        def mean_of(phi: npt.NDArray) -> Any:
            core, p, f0 = self._cb_unpack(phi, ctx)
            return (p - f0) * (self.dist.mean(*core) + self.gamma)

        with np.errstate(all="ignore"):
            grad = _central_gradient(mean_of, ctx.phi_hat)
            var = np.atleast_1d(grad @ ctx.cov @ grad)
            value = np.atleast_1d(mean_of(ctx.phi_hat))
        # The mass f0 at 0 of a zero-inflated model is below any offset
        scale = self._summary_scale(zero_floor=bool(self.zi))
        out = self._summary_wald(value, var, scale, alpha_ci, bound, "mean")
        return out[0]

    @staticmethod
    def _is_lr(method: str) -> bool:
        m = method.lower()
        if m in ("lr", "likelihood", "likelihood-ratio", "profile"):
            return True
        if m != "wald":
            check_option(
                "method",
                method,
                ("wald", "lr"),
                "Case does not matter, and 'likelihood', "
                "'likelihood-ratio' and 'profile' also mean 'lr'.",
            )
        return False

    def _check_summary_cb(self, alpha_ci: float, bound: str) -> None:
        if self.method != "MLE":
            raise ValueError("Only MLE has confidence bounds")
        check_option("bound", bound, BOUNDS)
        if not 0 < alpha_ci < 1:
            raise alpha_ci_error(alpha_ci)

    def _summary_scale(self, zero_floor: bool = False) -> tuple:
        """The scale a quantile or the mean is bounded on, from the
        model's support ``(lo, hi)``: the logit of its position in the
        support when both ends are finite, the log of its distance above
        ``lo`` when only that end is, and its own scale otherwise (a
        distribution on the whole line, or one whose support ends are
        parameters). Returns ``(to_psi, from_psi, slope, ends)``: the
        transform, its inverse, its derivative and the ends of its range.
        ``zero_floor`` puts ``lo`` at 0 (the mean of a zero-inflated
        model, whose mass at 0 is below any offset).
        """
        s0, s1 = (float(v) for v in getattr(self.dist, "support", (0, 0)))
        lo = 0.0 if zero_floor else s0 + self.gamma
        hi = s1 + self.gamma
        if np.isfinite(lo) and np.isfinite(hi):
            width = hi - lo

            def to_psi(v: Any) -> Any:
                return np.log(v - lo) - np.log(hi - v)

            def from_psi(u: Any) -> Any:
                return lo + width * expit(u)

            def slope(v: Any) -> Any:
                return 1 / (v - lo) + 1 / (hi - v)

            return to_psi, from_psi, slope, (_LN_TINY, -_LN_TINY)
        if np.isfinite(lo):
            return (
                lambda v: np.log(v - lo),
                lambda u: lo + np.exp(u),
                lambda v: 1 / (v - lo),
                (_LN_TINY, _LN_MAX),
            )
        return (
            lambda v: v,
            lambda u: u,
            lambda v: np.ones_like(v),
            (-1e300, 1e300),
        )

    def _summary_wald(
        self,
        value: npt.NDArray,
        var: npt.NDArray,
        scale: tuple,
        alpha_ci: float,
        bound: str,
        what: str,
    ) -> npt.NDArray:
        """The Wald bound on ``value`` with delta-method variance ``var``,
        on the ``scale`` of ``_summary_scale``."""
        to_psi, from_psi, slope, _ = scale
        sd = self._cb_sd(var, value, what)
        if bound == "two-sided":
            k = z(1 - alpha_ci / 2) * np.array([-1.0, 1.0])
        elif bound == "lower":
            k = np.array([-z(1 - alpha_ci)])
        else:
            k = np.array([z(1 - alpha_ci)])
        with np.errstate(all="ignore"):
            psi = to_psi(value)
            se = sd * np.abs(slope(value))
            out = from_psi(psi[:, None] + k * se[:, None])
            # A value that is infinite (a quantile past a limited failure
            # population's p) or at an end of the scale is its own bound.
            edge = ~np.isfinite(psi) | (sd == 0)
        out = np.where(edge[:, None], value[:, None], out)
        return out if bound == "two-sided" else out[:, 0]

    def _quantile_cb_wald(
        self, p: npt.NDArray, alpha_ci: float, bound: str
    ) -> npt.NDArray:
        """The delta-method bound on ``qf(p)``. The gradient of the
        quantile ``t`` is implicit, from ``F(t; theta) = p``:
        ``dt/dtheta = -(dF/dtheta) / f(t)``, which needs only the
        distribution function, not a differentiable ``qf``."""
        ctx = self._cb_context()
        t = np.atleast_1d(np.asarray(self.qf(p), dtype=float))
        core, p_lfp, f0 = self._cb_unpack(ctx.phi_hat, ctx)
        finite = np.isfinite(t)
        t_eval = np.where(finite, t, self.gamma + 1.0)
        with np.errstate(all="ignore"):
            jac = np.atleast_2d(
                jacobian(lambda phi: self._cb_full_ff(t_eval, phi, ctx))(
                    ctx.phi_hat
                )
            )
            dens = (p_lfp - f0) * np.asarray(
                self.dist.df(t_eval - self.gamma, *core), dtype=float
            )
            grad = -jac / dens[:, None]
            var = np.einsum("ij,jk,ik->i", grad, ctx.cov, grad)
        var = np.where(finite, var, 0.0)
        return self._summary_wald(
            t, var, self._summary_scale(), alpha_ci, bound, "qf"
        )

    def _quantile_cb_discrete(
        self, p: npt.NDArray, alpha_ci: float, bound: str, method: str
    ) -> npt.NDArray:
        """The bound on a discrete quantile, the smallest ``k`` with
        ``F(k) >= p``: that ``k`` from the band on ``F`` instead of from
        ``F`` -- from its upper end for the lower bound on the quantile,
        and its lower end for the upper -- as the nonparametric
        ``quantile_cb`` inverts its band (Brookmeyer and Crowley). A
        discrete quantile is a step, with no gradient for the delta
        method. Each ``k`` is found by doubling and then bisection, so a
        heavy tail costs a few dozen evaluations of the band, not one per
        count; ``inf`` where the band does not reach ``p`` by ``2**40``.
        """
        start = float(getattr(self.dist, "support", (0, np.inf))[0])
        start = 0.0 if not np.isfinite(start) else start

        def band_end(k: float, end: str) -> float:
            # "upper" or "lower" end of the band on F at k: the one-sided
            # bound at alpha_ci for a one-sided bound on the quantile, the
            # end of the two-sided band otherwise
            if bound == "two-sided":
                b = self.cb(k, on="ff", alpha_ci=alpha_ci, method=method)
                value = b[1] if end == "upper" else b[0]
            else:
                value = self.cb(
                    k, on="ff", alpha_ci=alpha_ci, bound=end, method=method
                )
            value = float(value)
            return value if np.isfinite(value) else 0.0

        def first(level: float, end: str) -> float:
            # The smallest count at which the band's end reaches level
            if band_end(start, end) >= level:
                return start
            lo, step = start, 1.0
            while band_end(start + step, end) < level:
                lo = start + step
                step *= 2
                if step > 2.0**40:
                    return np.inf
            hi = start + step
            while hi - lo > 1:
                mid = np.floor((lo + hi) / 2)
                if band_end(mid, end) >= level:
                    hi = mid
                else:
                    lo = mid
            return hi

        with warnings.catch_warnings():
            # a warning of the band's is given once, not once per count
            warnings.simplefilter("ignore")
            lower = np.full(len(p), np.nan)
            upper = np.full(len(p), np.nan)
            if bound in ("two-sided", "lower"):
                lower = np.array([first(p_i, "upper") for p_i in p])
            if bound in ("two-sided", "upper"):
                upper = np.array([first(p_i, "lower") for p_i in p])
        if bound == "two-sided":
            return np.column_stack([lower, upper])
        return lower if bound == "lower" else upper

    def _cb_context(self) -> Any:
        """Assemble the parameter vector and covariance used by ``cb``.

        The variance is computed over the extended parameter vector
        ``(*params, p?, f0?)`` so that the uncertainty of the LFP and
        zero-inflation parameters widens the bounds. gamma is held fixed:
        the threshold parameter is non-regular, so it carries no Wald
        variance. Models deserialized without a ``cov_matrix`` fall back to
        treating p and f0 as fixed.
        """
        n_core = len(self.params)
        phi_hat = list(self.params)
        if self.lfp:
            phi_hat.append(self.p)
        if self.zi:
            phi_hat.append(self.f0)
        phi_hat = np.array(phi_hat)

        cov = getattr(self, "cov_matrix", None)
        if cov is None:
            hess_inv = getattr(self, "hess_inv", None)
            if hess_inv is None:
                raise no_covariance_error(_NO_COVARIANCE_WHY)
            cov = np.zeros((len(phi_hat), len(phi_hat)))
            cov[:n_core, :n_core] = np.copy(hess_inv)

        return _CBContext(phi_hat=phi_hat, cov=cov, n_core=n_core)

    def _cb_unpack(self, phi: npt.NDArray, ctx: Any) -> Any:
        """Split an extended parameter vector into ``(core, p, f0)``."""
        core = phi[: ctx.n_core]
        i = ctx.n_core
        if self.lfp:
            p = phi[i]
            i += 1
        else:
            p = 1.0
        f0 = phi[i] if self.zi else 0.0
        return core, p, f0

    def _cb_full_sf(self, x: Any, phi: npt.NDArray, ctx: Any) -> Any:
        """Survival function including the LFP and zero-inflation mass.

        Points below the (offset) support are clamped *before* the base sf is
        evaluated: a negative argument produces NaN (fractional powers), and
        one NaN poisons the whole autograd jacobian in the delta-method
        variance (#256).
        """
        core, p, f0 = self._cb_unpack(phi, ctx)
        s0 = getattr(self.dist, "support", (-np.inf, np.inf))[0]
        xg = x - self.gamma
        below = xg < s0
        if np.any(below):
            # Evaluate the unselected branch just inside the support so it
            # stays finite (np.where evaluates both branches under autograd).
            xg = np.where(below, s0 + 1e-10, xg)
        base_sf = np.where(below, 1.0, self.dist.sf(xg, *core))
        out = 1 - p + (p - f0) * base_sf
        if self.zi:
            # No zero-inflation mass before 0, matching ``sf``.
            out = np.where(x < 0, 1.0, out)
        return out

    def _cb_full_ff(self, x: Any, phi: npt.NDArray, ctx: Any) -> Any:
        """``1 - _cb_full_sf``, from the base ``ff``, so that it is
        accurate where it is small (in the left tail, where ``1 - sf``
        is not): ``f0 + (p - f0) F``."""
        core, p, f0 = self._cb_unpack(phi, ctx)
        s0 = getattr(self.dist, "support", (-np.inf, np.inf))[0]
        xg = x - self.gamma
        below = xg < s0
        if np.any(below):
            xg = np.where(below, s0 + 1e-10, xg)
        base_ff = np.where(below, 0.0, self.dist.ff(xg, *core))
        out = f0 + (p - f0) * base_ff
        if self.zi:
            out = np.where(x < 0, 0.0, out)
        return out

    def _cb_delta_var(self, func: Callable[..., Any], ctx: Any) -> Any:
        """First-order delta-method variance: ``Var(g) = J Sigma J^T``."""
        jac = np.atleast_2d(jacobian(func)(ctx.phi_hat))
        var = np.einsum("ij,jk,ik->i", jac, ctx.cov, jac)
        # Rounding can leave a zero variance (a flat direction, or a point
        # outside the support) a hair below zero; only a variance
        # negative beyond it says the covariance is not positive definite.
        scale = np.einsum("ij,jk,ik->i", abs(jac), abs(ctx.cov), abs(jac))
        return np.where((var < 0) & (var >= -1e-10 * scale), 0.0, var)

    def _cb_sd(self, var: Any, x: Any, on: str) -> Any:
        """The delta-method standard error, ``sqrt(var)``, with one
        warning where the variance is negative (#411): the covariance is
        not positive definite, so no Wald bound exists there, and the
        bound is nan. It used to be a silent nan."""
        bad = ~(var >= 0)
        if np.any(bad):
            where = np.broadcast_to(np.atleast_1d(x), np.shape(var))[bad]
            warn_wald_undefined(
                f"{on} at x = {where.tolist()}",
                "its delta-method variance is negative or not finite, so "
                "the parameter covariance is not positive definite (the "
                "estimate is at or near a boundary of the parameter space, "
                "or the likelihood is not regular there)",
                # _cb_sd -> the bound helper -> cb -> the query-shape
                # wrapper -> the caller
                stacklevel=5,
            )
        return np.sqrt(np.where(bad, np.nan, var))

    def _cb_sf_bound(
        self, x: npt.ArrayLike, ctx: Any, alpha_ci: float, bound: str
    ) -> Any:
        """Confidence bound on the survival function: a Wald bound on the
        scale on which the family is a straight line in (log) time -- its
        probability-plot scale -- mapped back to ``R``.

        The scale is the distribution's ``_cb_link``: ``log(-log R)`` (the
        log cumulative hazard) for the Weibull, Exponential, Rayleigh and
        Gumbel, the normal quantile of ``F`` for the Normal and LogNormal,
        and the logit for the Logistic and LogLogistic, and for every other
        family, which has no such scale. Each keeps the bound within
        ``(0, 1)``. On a family's straight-line scale the function is
        ``a s - b`` in ``s`` = (log) time, so the pointwise band is the
        envelope of the lines of the Wald ellipsoid for ``(a, b)``: it
        rises with ``s`` whenever the slope's (the shape's) own Wald
        interval excludes 0. The logit band used for every family before
        (#477) has no such property: on small samples it turned back in a
        tail for most Weibull, LogNormal and Normal fits -- a lower bound on
        ``F(10)`` of 0.00004 under one on ``F(5)`` of 0.39 -- where this one
        does only when the shape's interval reaches 0. The scales agree to
        first order, so large-sample bounds are essentially unchanged. A
        two-sided bound is ``[upper, lower]`` on ``R`` on the last axis,
        the layout the public ``cb`` method expects.
        """

        R_hat = self._cb_full_sf(x, ctx.phi_hat, ctx)
        F_hat = self._cb_full_ff(x, ctx.phi_hat, ctx)
        # The smaller of R and F, each accurate where it is small (1 - R
        # is not): the scales below are taken from it in each tail, and so
        # is the variance (Var R = Var F), relative to it, since its square
        # can underflow and the derivative of R where R is near 1 has lost
        # the digits that F's keeps.
        left = F_hat < 0.5
        small = np.where(left, F_hat, R_hat)
        unit = np.where(small > 0, small, 1.0)

        def sf_func(phi: npt.NDArray) -> Any:
            R = self._cb_full_sf(x, phi, ctx)
            F = self._cb_full_ff(x, phi, ctx)
            return np.where(left, -F, R) / unit

        sd_R = unit * self._cb_sd(self._cb_delta_var(sf_func, ctx), x, "sf")
        # On the family's scale (surpyval.utils.linalg.sf_link_bound, which
        # the degradation and regression bands share). At the boundary (R =
        # 0 or 1, e.g. t <= gamma) the transform degenerates to 0/0; the
        # bound there is the boundary itself (#256).
        R_cb = sf_link_bound(
            R_hat, sd_R, alpha_ci, bound, cb_link(self.dist), ff_hat=F_hat
        )
        # [upper, lower] on R for a two-sided bound: the layout the public
        # cb method expects (it flips it for sf).
        return R_cb[..., ::-1] if bound == "two-sided" else R_cb

    def _cb_rate_bound(
        self, t: Any, ctx: Any, alpha_ci: float, bound: str, on: str
    ) -> Any:
        """Confidence bound on the hazard (``hf``) or density (``df``).

        Both are non-negative, so the bound is computed on the log scale to
        keep the result positive. The delta method is applied directly to the
        rate function rather than differentiating the ``Hf`` bound curve.
        The hazard is the model's own: ``df(x) / sf(x)``, or for a discrete
        distribution ``df(k) / sf(k - 1)`` (#414). Where the rate is 0 --
        below the (offset) support, or at a discrete ``k`` with no mass --
        both bounds are 0 (#413).
        """
        s0 = getattr(self.dist, "support", (-np.inf, np.inf))[0]
        xg = t - self.gamma
        below = xg < s0
        # Evaluated just inside the support there (and then replaced), so
        # a negative argument cannot put a nan in the Jacobian (#256).
        xg = np.where(below, s0 + 1e-10, xg)

        def density(phi: npt.NDArray) -> Any:
            core, p, f0 = self._cb_unpack(phi, ctx)
            base = np.where(below, 0.0, self.dist.df(xg, *core))
            return (p - f0) * base

        if on == "hf":
            # The survival that conditions the hazard: to the step before
            # for a discrete distribution, as its hf is defined.
            t_sf = t - 1.0 if self.dist.discrete else t

            def func(phi: npt.NDArray) -> Any:
                return density(phi) / self._cb_full_sf(t_sf, phi, ctx)

        else:
            func = density

        g_hat = func(ctx.phi_hat)
        sd_g = self._cb_sd(self._cb_delta_var(func, ctx), t, on)

        if bound == "two-sided":
            diff = z(alpha_ci / 2) * np.array([1.0, -1.0]).reshape(2, 1)
        elif bound == "upper":
            diff = -z(alpha_ci)
        else:
            diff = z(alpha_ci)

        with np.errstate(divide="ignore", invalid="ignore"):
            if self.dist.discrete:
                # A discrete hazard and mass are probabilities: the logit
                # scale keeps their bounds in [0, 1], as for sf.
                exponent = -diff * sd_g / (g_hat * (1 - g_hat))
                cb = g_hat / (g_hat + (1 - g_hat) * np.exp(exponent))
                cb = np.where(np.broadcast_to(g_hat == 1.0, cb.shape), 1.0, cb)
            else:
                cb = g_hat * np.exp(diff * sd_g / g_hat)
        # Neither scale has a point at a rate of 0: the bounds are 0.
        cb = np.where(np.broadcast_to(g_hat == 0.0, cb.shape), 0.0, cb)
        if bound == "two-sided":
            cb = cb.T
        return cb

    # neg_ll/aic/bic/aic_c come from InformationCriteriaMixin. The aic_c
    # correction uses the same parameter count as the aic() penalty it
    # corrects — including gamma / p / f0 when fitted (#256).
    def _ic_k(self) -> int:
        """The number of *estimated* parameters, the ``k`` of AIC and BIC.

        ``self.k`` counts every parameter of the model -- the
        distribution's, plus gamma / p / f0 when fitted -- including any the
        user fixed. A fixed parameter is known, not estimated, so it costs
        no degree of freedom: counting it penalised a Weibull with its
        shape fixed as a two-parameter model, which is not the standard
        definition and biased every comparison against fixed fits.
        """
        return self.k - len(self._user_fixed_idx())

    def _require_data(self, what: str) -> None:
        if self.data is None:
            raise ValueError(
                "{} needs the data the model was fitted to, which this model "
                "does not have (it was built from parameters, or restored "
                "from a dict saved without its data -- save it with "
                "to_dict(with_data=True) to keep them)".format(what)
            )

    def _ic_sample_size_from_data(self) -> float:
        self._require_data("This information criterion")
        # The observed failures -- exact, left- or interval-censored --
        # falling back to the units when there is none; the rule every
        # model's BIC and AIC_c share (ic_sample_size).
        return ic_sample_size(self.data["c"], self.data["n"])

    def get_plot_data(
        self,
        heuristic: str = "Nelson-Aalen",
        alpha_ci: float = 0.05,
        method: str = "wald",
    ) -> dict:
        """

        A method to gather plot data

        Parameters
        ----------

        heuristic : {'Blom', 'Median', 'ECDF', 'Modal', 'Midpoint', 'Mean',\
            'Weibull', 'Benard', 'Beard', 'Hazen', 'Gringorten', 'None',\
            'Tukey', 'DPW', 'Fleming-Harrington', 'Kaplan-Meier',\
            'Nelson-Aalen', 'Filliben', 'Larsen', 'Turnbull'}, optional
            The method that the plotting point on the probability plot will
            be calculated. Default is "Nelson-Aalen".

        alpha_ci : float, optional
            The level of significance at which the confidence bounds, if
            able, will be calculated. Defaults to 0.05.

        method : ('wald', 'lr'), str, optional
            The method of the confidence band, as for :meth:`cb`. Defaults
            to ``"wald"``; ``"lr"``, the likelihood-ratio band, is slower
            but better in small samples, and always monotone.

        Returns
        -------

        data : dict
            Returns dictionary containing the data needed to do a plot.
            ``x_`` and ``F`` are every row of the plotting positions,
            suspensions included (a suspension's row carries the ``F`` of
            the failure before it); ``failed`` is a boolean mask of the
            same length, True where the row records a failure, and
            ``x_censored`` holds the suspension times. :meth:`plot` draws
            only the rows ``failed`` selects:
            ``d["x_"][d["failed"]], d["F"][d["failed"]]``.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> x = Weibull.random(100, 10, 3)
        >>> model = Weibull.fit(x)
        >>> data = model.get_plot_data()

        With suspensions, the rows to draw are the failures:

        >>> model = Weibull.fit([10, 20, 30, 40, 50, 60], [0, 0, 0, 0, 1, 1])
        >>> data = model.get_plot_data()
        >>> data["x_"][data["failed"]]
        array([10., 20., 30., 40.])
        >>> data["x_censored"]
        array([50., 60.])
        """
        self._require_data("get_plot_data()")
        cb_func: Callable[[Any], Any] | None
        if (
            hasattr(self, "hess_inv")
            and (self.method == "MLE")
            and (self.hess_inv is not None)
        ):

            def _cb_func(x_model: npt.NDArray) -> Any:
                return self.cb(
                    x_model, on="ff", alpha_ci=alpha_ci, method=method
                )

            cb_func = _cb_func
        else:
            cb_func = None

        return probability_plot_data(
            dist=self.dist,
            ff=self.ff,
            x=self.data["x"],
            c=self.data["c"],
            n=self.data["n"],
            t=self.data["t"],
            heuristic=heuristic,
            gamma=self.gamma,
            params=self.params,
            cb_func=cb_func,
        )

    def plot(
        self,
        heuristic: str = "Nelson-Aalen",
        plot_bounds: bool = True,
        alpha_ci: float = 0.05,
        ax: "Axes | None" = None,
        show_censored: bool = False,
        method: str = "wald",
        color: Any = None,
        label: "str | None" = None,
        **kwargs: Any,
    ) -> Axes:
        """
        A method to do a probability plot.

        The points are the failures, at their plotting positions (the
        rows :meth:`get_plot_data` marks ``failed``). A suspension
        (right-censored unit) moves the plotting positions of the
        failures after it but has no point of its own (Abernethy's *New
        Weibull Handbook*, Weibull++); ``show_censored=True`` marks each
        suspension time with a tick on the time axis. The time axis spans
        every time, suspensions included.

        A model without data (built with ``from_params``, or restored from
        a dict saved without its data) draws its CDF alone on the same
        axes, over its 1% to 99% quantiles, with no plotting points or
        bounds -- for comparing a specification with a fit.

        Parameters
        ----------

        heuristic : {'Blom', 'Median', 'ECDF', 'Modal', 'Midpoint', 'Mean', \
            'Weibull', 'Benard', 'Beard', 'Hazen', 'Gringorten', 'None',\
            'Tukey', 'DPW', 'Fleming-Harrington', 'Kaplan-Meier',\
            'Nelson-Aalen', 'Filliben', 'Larsen', 'Turnbull'}, optional
            The method that the plotting point on the probability plot will
            be calculated.

        plot_bounds : Boolean, optional
            A Boolean value to indicate whether you want the probability
            bounds to be calculated.

        alpha_ci : float, optional
            The level of significance at which the confidence bounds, if
            able, will be calculated. Defaults to 0.05.

        ax: matplotlib.axes.Axes, optional
            The axis onto which the plot will be created. Optional, if not
            provided a new axes will be created.

        show_censored : bool, optional
            Mark the suspension (right-censored) times with ticks along
            the time axis. Defaults to False.

        method : ('wald', 'lr'), str, optional
            The method of the confidence band, as for :meth:`cb`. Defaults
            to ``"wald"``; ``"lr"``, the likelihood-ratio band, is slower
            but better in small samples, and always monotone.

        color : matplotlib color, optional
            The colour of the points, the fitted line and its bounds. By
            default the next colour of the axes' colour cycle, so that
            models plotted on the same axes differ.

        label : str, optional
            The legend label of the fitted line.

        **kwargs
            Other keyword arguments for the fitted line (a
            ``matplotlib.lines.Line2D``: ``linestyle``, ``linewidth``, ...).

        Returns
        -------

        plot : matplotlib.axes.Axes
            the axes the probability plot was drawn onto. The x label is
            "Time" unless the axes already have one; change it with
            ``ax.set_xlabel``.

        Examples
        --------

        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> np.random.seed(1)
        >>> x = Weibull.random(100, 10, 3)
        >>> model = Weibull.fit(x)
        >>> model.plot()
        <Axes: title={'center': 'Weibull Probability Plot'}, xlabel='Time',
        ylabel='CDF'>

        Two populations on one plot, each in its own colour, with a
        legend:

        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> north = Weibull.fit(Weibull.random(30, 10, 3))
        >>> south = Weibull.fit(Weibull.random(30, 20, 2))
        >>> ax = north.plot(ax=ax, label="North")
        >>> ax = south.plot(ax=ax, label="South")
        >>> legend = ax.legend()
        >>> [text.get_text() for text in legend.get_texts()]
        ['North', 'South']
        >>> plt.close(fig)
        """
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gcf().gca()

        if not hasattr(self, "params"):
            raise ValueError("Can't plot model that failed to fit")

        if not (
            hasattr(self.dist, "mpp_y_transform")
            and hasattr(self.dist, "mpp_inv_y_transform")
        ):
            raise NotImplementedError(
                f"{self.dist.name} does not support probability plotting"
            )

        if self.data is None:
            # Built from parameters (or restored without its data): the
            # model's CDF on the same axes, with no points or bounds, to
            # compare a specification with a fit (#485).
            d = curve_plot_data(
                self.dist, self.ff, self.qf, self.gamma, self.params
            )
        else:
            heuristic = adjust_heuristic(
                self.data["c"], self.data["t"], heuristic
            )
            d = self.get_plot_data(
                heuristic=heuristic, alpha_ci=alpha_ci, method=method
            )

        return draw_probability_plot(
            ax,
            d,
            lambda x: self.dist.mpp_y_transform(x, *self.params),
            lambda x: self.dist.mpp_inv_y_transform(x, *self.params),
            title=f"{self.dist.name} Probability Plot",
            plot_bounds=plot_bounds,
            show_censored=show_censored,
            color=color,
            label=label,
            **kwargs,
        )
