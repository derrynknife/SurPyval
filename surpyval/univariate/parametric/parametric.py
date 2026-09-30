import warnings
from collections import namedtuple
from copy import copy, deepcopy
from math import comb
from typing import TYPE_CHECKING, Any, Callable

import numpy.typing as npt
from autograd import jacobian
from scipy.optimize import (
    brentq,
    minimize,
    minimize_scalar,
)
from scipy.special import expit
from scipy.special import ndtri as z
from scipy.stats import uniform

import surpyval as surv
from surpyval import ParametricDistribution, np
from surpyval.serialisation import SerialisableMixin, stamp_schema, to_native
from surpyval.univariate.information_criteria import (
    InformationCriteriaMixin,
    ic_sample_size,
)
from surpyval.utils import fsli_to_xcnt
from surpyval.utils.linalg import (
    param_name,
    wald_undefined,
    warn_wald_undefined,
)
from surpyval.utils.rng import as_generator
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.surpyval_data import SurpyvalData

from .probability_plotting import (
    adjust_heuristic,
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
_CBContext = namedtuple("_CBContext", ["phi_hat", "cov", "n_core"])

# The likelihood-ratio searches (#421) move each parameter in a coordinate
# that is unbounded over its space (see ``_LRCoord``). These are the
# coordinates' ends: past them a parameter is no longer a double distinct
# from the edge of its space.
_LN_MAX = float(np.log(np.finfo(float).max))  # 709.78: exp overflows
_LN_TINY = float(np.log(np.finfo(float).tiny))  # -708.40: exp underflows
_FLOAT_MAX = float(np.finfo(float).max)
# The profile deviance is solved to about 1e-8 (the searches' tolerances);
# this is the slack the likelihood-ratio walk allows it (``_lr_walk``).
_LR_NOISE = 1e-6
# A deviance that is not finite, or no search reaching the target at all,
# is a failure; a target beyond a data-derived edge (the Uniform's) is not
# reachable, and reads as this deviance, above any critical value.
_LR_UNREACHABLE = 1e6


class _LRCoord:
    """The coordinate a likelihood-ratio search moves one parameter in.

    Unbounded over the parameter's declared space ``(lo, hi)``: the log of
    its distance from a one-sided bound, the logit of its position within
    a finite interval, or the parameter itself when it has no bound -- as
    the fitter searches it, but a plain log throughout, which is the same
    at every scale. ``ends`` are the coordinate's values beyond which the
    parameter is no longer a double distinct from the edge of its space.
    """

    def __init__(self, lo: Any, hi: Any) -> None:
        self.lo = -np.inf if lo is None else float(lo)
        self.hi = np.inf if hi is None else float(hi)
        eps = np.finfo(float).eps

        def floor(edge: float) -> float:
            # The log of the smallest distance from ``edge`` that is still
            # a distinct double.
            return float(np.log(max(np.finfo(float).tiny, eps * abs(edge))))

        if np.isfinite(self.lo) and np.isfinite(self.hi):
            self.kind = "logit"
            log_width = float(np.log(self.hi - self.lo))
            # expit rounds to 1 above -log(eps): 36.04 at most.
            self.ends = (
                floor(self.lo) - log_width,
                min(-float(np.log(eps)), log_width - floor(self.hi)),
            )
        elif np.isfinite(self.lo):
            self.kind = "log"
            self.ends = (floor(self.lo), _LN_MAX)
        elif np.isfinite(self.hi):
            self.kind = "neglog"
            self.ends = (-_LN_MAX, -floor(self.hi))
        else:
            self.kind = "identity"
            self.ends = (-_FLOAT_MAX, _FLOAT_MAX)

    def to_u(self, theta: float) -> float:
        with np.errstate(all="ignore"):
            if self.kind == "log":
                return float(np.log(theta - self.lo))
            if self.kind == "neglog":
                return float(-np.log(self.hi - theta))
            if self.kind == "logit":
                return float(np.log(theta - self.lo) - np.log(self.hi - theta))
        return float(theta)

    def from_u(self, u: float) -> float:
        with np.errstate(all="ignore"):
            if self.kind == "log":
                return float(self.lo + np.exp(u))
            if self.kind == "neglog":
                return float(self.hi - np.exp(-u))
            if self.kind == "logit":
                return float(self.lo + (self.hi - self.lo) * expit(u))
        return float(u)

    def slope(self, theta: float) -> float:
        """d theta / d u at ``theta``."""
        if self.kind == "log":
            return float(theta - self.lo)
        if self.kind == "neglog":
            return float(self.hi - theta)
        if self.kind == "logit":
            return float(
                (theta - self.lo) * (self.hi - theta) / (self.hi - self.lo)
            )
        return 1.0

    def edge(self, direction: float) -> float:
        """The edge of the space the parameter goes to in ``direction``."""
        return self.hi if direction > 0 else self.lo


class _LRPath:
    """The points a profile search has solved, for continuation.

    A profile point is started from the solved point nearest to it and
    from the straight line through the two nearest, as well as from the
    estimate: the minimising nuisance parameters move along a curved
    valley (a NegativeBinomial ``p`` of ``1 - lambda / r`` as ``r`` grows;
    an ExpoWeibull ``alpha`` of 1e-28 at a ``beta`` of 0.05), which a
    search started from the estimate each time does not follow.
    """

    def __init__(self) -> None:
        self.w: list[float] = []
        self.u: list[npt.NDArray] = []
        # The minimum at each point (a negative log-likelihood).
        self.f: list[float] = []

    def add(self, w: float, u: npt.NDArray, f: float = np.nan) -> None:
        if np.isfinite(w) and np.all(np.isfinite(u)):
            self.w.append(float(w))
            self.u.append(np.array(u, dtype=float))
            self.f.append(float(f))

    def starts(self, w: float) -> list[npt.NDArray]:
        if not self.w:
            return []
        order = np.argsort(np.abs(np.asarray(self.w) - w))
        near = self.u[order[0]]
        out = [near]
        if len(order) > 1:
            w0, w1 = self.w[order[0]], self.w[order[1]]
            if w0 != w1:
                slope = (near - self.u[order[1]]) / (w0 - w1)
                out.append(near + slope * (w - w0))
        return out


def _lr_walk(
    deviance: Callable[[float], float],
    w_hat: float,
    step: float,
    direction: float,
    crit: float,
    end: float,
    stop: float | None = None,
    d_hat: float = 0.0,
) -> tuple[str, float]:
    """Walk out from ``w_hat`` to where ``deviance`` first reaches ``crit``.

    Steps of ``step``, growing by 1.6 each time, bracket the crossing; the
    bracket is walked again in eight equal steps (a long step can lose the
    valley the profile's minimum follows, and overstate the deviance: an
    ExpoWeibull ``mu`` bound of 1e-3 where the deviance falls to 0.3 at
    1e-4), and ``brentq`` solves the first of them to reach ``crit``.
    Returns ``("root", w)``; ``("stop", stop)`` when the walk reaches
    ``stop`` (a data-derived limit, beyond which the likelihood is 0)
    below ``crit``; ``("edge", end)`` when the deviance stays below
    ``crit`` to ``end`` (the end of the coordinate) or levels off below
    it; and ``("fail", nan)`` where the deviance is not finite, or the
    walk runs out of steps. ``d_hat`` is the deviance at ``w_hat``: 0 at
    the estimate.

    *Levelling off.* The deviance of a parameter that tends to a limiting
    model at the edge of its space (a NegativeBinomial ``r`` to infinity,
    the shifted Poisson) converges to that model's deviance. Where that is
    below ``crit`` no value of the parameter out to the edge is excluded,
    and the bound is the edge. The walk says so once, over its last three
    steps, the deviance has risen ever more slowly -- each rise per unit
    of ``w`` (a fall counting as no rise) no more than the one before, to
    within ``_LR_NOISE`` -- and a straight line from the latest point, at
    the latest of those slopes, stays below ``crit`` all the way to
    ``end``. A deviance rising ever more slowly lies below that line, so
    it cannot reach ``crit`` before the end of the representable range.
    A deviance rising at a steady or growing rate (a quadratic one, the
    usual case) fails the first test, and one rising at a slowing rate
    that would still reach ``crit`` before ``end`` fails the second: an
    ExpoWeibull ``alpha`` whose deviance rises by 0.024 per unit of
    ``log(alpha)`` at ``alpha`` = 7e-11 goes on to cross 3.84 at 5e-28,
    and the walk finds it there. The one profile the test misreads is one
    that falls for three steps and later climbs back above ``crit`` (a
    second, lower mode of the likelihood further out); the bound is then
    the edge, wider than it need be, never narrower.
    """
    ws, ds = [w_hat], [d_hat]
    # brentq starts from the bracket's ends, which the walk has solved.
    solved: dict[float, float] = {float(w_hat): float(d_hat)}

    def dev_at(v: float) -> float:
        v = float(v)
        if v not in solved:
            solved[v] = deviance(v)
        return solved[v]

    def f(v: float) -> float:
        d = dev_at(v)
        return (_LR_UNREACHABLE if d == np.inf else d) - crit

    def levels_off() -> bool:
        if len(ds) < 4:
            return False
        with np.errstate(all="ignore"):
            h = np.abs(np.diff(ws[-4:]))
            rate = np.maximum(np.diff(ds[-4:]), 0.0) / h
            slowing = np.all(rate[1:] <= rate[:-1] + _LR_NOISE / h[1:])
            reach = ds[-1] + rate[-1] * abs(end - ws[-1]) + _LR_NOISE
        return bool(slowing and reach < crit)

    for _ in range(80):
        w = ws[-1] + direction * step
        step *= 1.6
        at_stop = stop is not None and direction * (w - stop) >= 0
        at_end = direction * (w - end) >= 0
        if stop is not None and at_stop and direction * (stop - end) <= 0:
            # The data's limit comes before the end of the coordinate.
            w, at_end = float(stop), False
        elif at_end:
            w, at_stop = float(end), False
        dev = dev_at(w)
        if dev >= crit:
            # Walk the bracket again, in eight steps from its near end.
            for v in np.linspace(ws[-1], w, 9)[1:]:
                dev = dev_at(v)
                if dev >= crit:
                    if f(ws[-1]) >= 0:
                        # A start on the boundary (``d_hat`` at crit).
                        return "root", float(ws[-1])
                    a, b = sorted((ws[-1], v))
                    try:
                        root = brentq(f, a, b, xtol=1e-9, rtol=1e-9)
                    except ValueError:
                        return "fail", np.nan
                    return "root", float(root)
                if not np.isfinite(dev):
                    return "fail", np.nan
                if v != w:
                    ws.append(float(v))
                    ds.append(dev)
                    if levels_off():
                        return "edge", float(end)
            # The far end, solved again from the steps before it, is
            # below crit after all: the walk goes on from it.
        if not np.isfinite(dev):
            return "fail", np.nan
        if at_stop:
            return "stop", float(w)
        if at_end:
            return "edge", float(end)
        ws.append(w)
        ds.append(dev)
        if levels_off():
            return "edge", float(end)
    return "fail", np.nan


def _central_gradient(f: Callable[..., Any], u: npt.NDArray) -> npt.NDArray:
    """Central-difference gradient of ``f`` at ``u``."""
    grad = np.empty(len(u))
    for j in range(len(u)):
        h = 1e-6 * max(1.0, abs(u[j]))
        up, down = np.array(u, dtype=float), np.array(u, dtype=float)
        up[j] += h
        down[j] -= h
        grad[j] = (f(up) - f(down)) / (2 * h)
    return grad


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
    InformationCriteriaMixin, SerialisableMixin, ParametricDistribution
):
    """
    Result of ``.fit()`` or ``.from_params()`` method for every parametric
    surpyval distribution.

    Instances of this class are very useful when a user needs the other
    functions of a distribution for plotting, optimizations, monte carlo
    analysis and numeric integration.

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

    A model built from parameters has the same functions. With a limited
    failure population, one unit in ten never fails, so the mean life is
    infinite:

    >>> lfp = surv.Weibull.from_params([20, 3], p=0.9)
    >>> float(lfp.sf(1000.0).round(4)), lfp.mean(), lfp.extras
    (0.1, inf, {'p': 0.9})
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
    tl: Any
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

        # Restore the support interval, which fit-time construction sets via
        # the fitter (#261).
        dist._set_support(out, offset)

        return out

    def to_dict(self, with_data: bool = False) -> dict:
        """
        Serialise the model to a dictionary of plain Python types.

        The dictionary holds the distribution name, the parameters, the
        offset / LFP / ZI settings, the names of any parameters fixed at fit
        time (``"fixed"``) and, if available, the parameter covariance,
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
        out["param_names"] = self.dist.param_names

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
            ``model.dist.param_names``. They are checked as
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
                    for p, name in zip(self.params, self.dist.param_names)
                ]
            )
            out = (
                "Parametric SurPyval Model"
                "\n========================="
                f"\nDistribution        : {self.dist.name}"
                f"\nFitted by           : {self.method}"
            )
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
        if method.lower() in (
            "lr",
            "likelihood",
            "likelihood-ratio",
            "profile",
        ):
            return self._param_cb_lr(name, alpha_ci, bound)
        elif method.lower() != "wald":
            raise ValueError(
                f"Unknown confidence-bound method '{method}'; "
                "use 'wald' or 'lr'."
            )

        is_core, idx = self._resolve_param_name(name)
        if not is_core:
            cov = getattr(self, "cov_matrix", None)
            if cov is None:
                raise ValueError(
                    f"Model has no covariance for '{name}'; "
                    "it must be fit with the MLE method"
                )
            p_hat = self.f0 if name == "f0" else self.p
            var = cov[idx, idx]
            param_bounds = (0, 1)
        else:
            p_hat = self.params[idx]
            hess_inv = getattr(self, "hess_inv", None)
            if hess_inv is None:
                raise ValueError(
                    "Model carries no parameter covariance (the Hessian was "
                    "singular at the optimum, or the model was not fit by "
                    "MLE); confidence bounds are unavailable."
                )
            var = hess_inv[idx, idx]
            param_bounds = self.dist.bounds[idx]

        if bound == "two-sided":
            alpha = alpha_ci / 2
            bounds = np.array([-1, 1])
        elif bound == "lower":
            alpha = alpha_ci
            bounds = np.array([-1])
        elif bound == "upper":
            alpha = alpha_ci
            bounds = np.array([1])
        else:
            raise ValueError(
                "bound must be 'two-sided', 'lower' or 'upper'; got "
                f"{bound!r}"
            )

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
        valid = list(self.dist.param_names)
        if self.lfp:
            valid.append(self.lfp_name)
        if self.zi:
            valid.append("f0")
        raise ValueError(
            f"Unknown parameter {name!r} for this {self.dist.name} model; "
            f"expected one of {valid}"
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

    def _lr_neg_ll(self, theta: npt.NDArray) -> float:
        """The negative log-likelihood at core parameters ``theta``, as
        the likelihood-ratio searches see it.

        ``nan`` where it cannot be right: at parameters that are not
        finite (a search that has stepped off to nan; some likelihoods
        iterate to their limit on nan, a NegativeBinomial's incomplete
        beta for 2 s a call), and below the fit's own minimum by more
        than the fit's precision (``_LR_NOISE`` in deviance, or 1e-8 of
        the log-likelihood; the registry's fits are within 2e-11 of
        their profiles' minima). The fit is the maximum, so a likelihood
        above it is the likelihood failing at extreme parameters: an
        ExpoWeibull's at ``beta`` = 1e14 and ``mu`` = 1e-14 is a
        deviance of -1e21, and a search that reached it took it for the
        best point there.
        """
        if not np.all(np.isfinite(theta)):
            return np.nan
        nll = self._lr_raw_neg_ll(theta)
        params = np.asarray(self.params, dtype=float)
        kept = self.__dict__.get("_lr_nll_hat")
        if kept is None or kept[0] != params.tobytes():
            kept = (params.tobytes(), self._lr_raw_neg_ll(params))
            self.__dict__["_lr_nll_hat"] = kept
        slack = max(_LR_NOISE, 2e-8 * abs(kept[1]))
        if 2.0 * (nll - kept[1]) < -slack:
            return np.nan
        return nll

    def _lr_raw_neg_ll(self, theta: npt.NDArray) -> float:
        with np.errstate(all="ignore"):
            return float(
                self.dist._neg_ll_func(
                    self.surv_data, *theta, self.gamma, self.f0, self.p
                )
            )

    def _lr_coords(self) -> tuple[list[_LRCoord], list[tuple[Any, Any]]]:
        """Each core parameter's likelihood-ratio search coordinate
        (``_LRCoord``), and its box in that coordinate: the data-derived
        limits of ``_lr_limits``, ``None`` where the limit is the declared
        bound (which the coordinate maps to infinity)."""
        coords, boxes = [], []
        for (lo, hi), (l_lo, l_hi) in zip(
            self.dist.bounds, self._lr_limits(), strict=True
        ):
            coord = _LRCoord(lo, hi)
            coords.append(coord)
            boxes.append(
                (
                    None if l_lo == coord.lo else coord.to_u(l_lo),
                    None if l_hi == coord.hi else coord.to_u(l_hi),
                )
            )
        return coords, boxes

    @staticmethod
    def _lr_box(coords: list, limits: list) -> list[tuple[Any, Any]]:
        """The box a likelihood-ratio search runs in: the data-derived
        limits, and elsewhere the ends of each coordinate, so that a
        search cannot step off to where the parameter overflows (the
        identity coordinate's ends are the doubles' own)."""
        box = []
        for coord, (b_lo, b_hi) in zip(coords, limits, strict=True):
            if coord.kind != "identity":
                b_lo = coord.ends[0] if b_lo is None else b_lo
                b_hi = coord.ends[1] if b_hi is None else b_hi
            box.append((b_lo, b_hi))
        return box

    @staticmethod
    def _lr_start(x0: npt.NDArray, box: list) -> npt.NDArray:
        """``x0`` inside ``box``."""
        x0 = np.array(x0, dtype=float)
        for k, (b_lo, b_hi) in enumerate(box):
            if b_lo is not None:
                x0[k] = max(x0[k], b_lo)
            if b_hi is not None:
                x0[k] = min(x0[k], b_hi)
        return x0

    def _profile_neg_ll(
        self, idx: int, value: Any, path: _LRPath | None = None
    ) -> float:
        """Profile negative log-likelihood with core parameter ``idx`` fixed.

        Holds the ``idx``-th distribution parameter at ``value`` and
        minimises the negative log-likelihood over the remaining core
        parameters, each in its unbounded search coordinate
        (``_LRCoord``). The search starts from the fit and, given the
        ``path`` of the points already solved, from the nearest of them
        and the line through the two nearest (continuation); the lowest
        minimum is kept, and added to ``path``. The raw parameters within
        box limits, started from the fit every time, did not follow the
        minimum out along its valley: a NegativeBinomial profile deviance
        of 3.05 at ``p`` = 0.999999 where it is 2.35, and ExpoWeibull ones
        of 49 and 42 where they are 3.9 and 0.29 (#421).

        ``nan`` if every search fails; ``inf`` where the likelihood is 0
        from every start. ``gamma``, ``f0`` and ``p`` are held at their
        fitted values -- likelihood-ratio bounds for offset / LFP / ZI models
        are not yet supported, so the public entry point rejects them before
        this is reached.
        """
        theta = np.array(self.params, dtype=float)
        theta[idx] = value
        # Parameters the user fixed at fit time stay fixed during the
        # profile — re-freeing them makes the profile drop below the fitted
        # nll and silently inflates the interval (#255).
        user_fixed = self._user_fixed_idx()
        free = [
            j for j in range(len(theta)) if j != idx and j not in user_fixed
        ]
        if not free:
            # Single-parameter distribution: nothing left to profile over.
            return self._lr_neg_ll(theta)

        coords, limits = self._lr_coords()
        free_coords = [coords[j] for j in free]
        box = [limits[j] for j in free]
        bounds = box if any(b != (None, None) for b in box) else None
        # Starts are held inside the coordinates' ends as well.
        start_box = self._lr_box(free_coords, box)

        def obj(u: npt.NDArray) -> float:
            th = theta.copy()
            th[free] = [c.from_u(v) for c, v in zip(free_coords, u)]
            nll = self._lr_neg_ll(th)
            return nll if np.isfinite(nll) else np.inf

        u_hat = np.array([coords[j].to_u(self.params[j]) for j in free])
        w = coords[idx].to_u(value)
        starts = [] if path is None else path.starts(w)
        best, best_u, zero = np.inf, None, False
        with np.errstate(all="ignore"):
            for x0 in starts + [u_hat]:
                res = minimize(
                    obj,
                    self._lr_start(x0, start_box),
                    method="L-BFGS-B",
                    jac="3-point",
                    bounds=bounds,
                    options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 1000},
                )
                zero = zero or res.fun == np.inf
                if np.isfinite(res.fun) and res.fun < best:
                    best, best_u = float(res.fun), np.asarray(res.x)
            if best_u is None:
                # Every gradient search failed: derivative free from the
                # fit, as a last resort.
                res = minimize(
                    obj,
                    self._lr_start(u_hat, start_box),
                    method="Nelder-Mead",
                    bounds=bounds,
                )
                zero = zero or res.fun == np.inf
                if np.isfinite(res.fun):
                    best, best_u = float(res.fun), np.asarray(res.x)
        if best_u is None:
            return np.inf if zero else np.nan
        if path is not None:
            path.add(w, best_u, best)
        return best

    def _lr_limits(self) -> list[tuple[float, float]]:
        """``(lower, upper)`` of each core parameter for the
        likelihood-ratio searches: its declared bounds, and for a
        parameter that is an edge of the support (the Uniform's and the
        4-parameter Beta's ``a`` and ``b``) the data's extremes, beyond
        which the likelihood is 0. The searches could not follow that
        cliff: a Uniform band stalled at the estimate (#421)."""
        limits = [
            (-np.inf if lo is None else lo, np.inf if hi is None else hi)
            for lo, hi in self.dist.bounds
        ]
        support = np.asarray(
            getattr(self.dist, "support", (0.0, 0.0)), dtype=float
        )
        if np.any(np.isnan(support)):
            x = np.asarray(self.surv_data.x, dtype=float)
            x = x[np.isfinite(x)]
            i_lo, i_hi = self.dist.support_param_index
            if np.isnan(support[0]):
                lo, hi = limits[i_lo]
                limits[i_lo] = (lo, min(hi, float(x.min())))
            if np.isnan(support[1]):
                lo, hi = limits[i_hi]
                limits[i_hi] = (max(lo, float(x.max())), hi)
        return limits

    def _param_cb_lr(
        self, name: str, alpha_ci: float, bound: str
    ) -> npt.NDArray:
        """Profile-likelihood (likelihood-ratio) bound on a parameter.

        The bound(s) solve ``2[nll_p(v) - nll_hat] = c`` where ``nll_p`` is the
        profile negative log-likelihood, ``nll_hat`` the fitted value, and
        ``c`` the chi-squared critical value (``z**2``) at the requested level.
        The deviance is zero at the estimate, so each side walks out from it
        in the parameter's search coordinate (``_LRCoord``: its log, or
        logit, ...), in steps that start at the Wald standard error there,
        and the first crossing is solved by ``brentq`` (``_lr_walk``). The
        interval is thus the piece of the likelihood-ratio confidence set
        that contains the estimate. Where the deviance stays below ``c`` to
        the edge of the parameter's space, or levels off below it on the
        way, the bound is that edge (0, 1 or ``inf``): no value up to it is
        excluded. A data-derived limit (a Uniform's ``a`` at the smallest
        observation) reached first is the bound.
        """
        if self.method != "MLE":
            raise ValueError("Only MLE has confidence bounds")
        self._ensure_surv_data()
        if self.offset or self.lfp or self.zi:
            raise NotImplementedError(
                "Likelihood-ratio bounds are not yet available for offset, "
                "limited-failure-population or zero-inflated models; use "
                "method='wald'."
            )
        is_core, idx = self._resolve_param_name(name)
        if not is_core:
            raise NotImplementedError(
                "Likelihood-ratio bounds on 'p' / 'f0' are not yet "
                "available; use method='wald'."
            )

        if self._is_fixed_param(name):
            # A parameter fixed at fit time is known, not estimated: the
            # degenerate interval at its value, as the Wald method gives
            # (its variance is zero). This used to raise instead.
            value = float(self.params[idx])
            if bound == "two-sided":
                return np.array([value, value])
            if bound not in ("lower", "upper"):
                raise ValueError(
                    "bound must be 'two-sided', 'lower' or 'upper'"
                )
            return np.array([value])
        if bound == "two-sided":
            crit = z(1.0 - alpha_ci / 2.0) ** 2
        elif bound in ("lower", "upper"):
            crit = z(1.0 - alpha_ci) ** 2
        else:
            raise ValueError("bound must be 'two-sided', 'lower' or 'upper'")

        def solve_side(direction: Any) -> Any:
            value = self._lr_param_side(idx, crit, direction)
            if np.isnan(value):
                # Never fall back on the last candidate or the estimate:
                # an unfound bound is reported as such.
                side = "upper" if direction > 0 else "lower"
                warnings.warn(
                    f"The likelihood-ratio {side} bound on '{name}' could "
                    "not be found (the profile deviance never crossed the "
                    "critical value, or was not finite); nan is returned "
                    "for it. method='wald' gives a bound in its place.",
                    RuntimeWarning,
                    stacklevel=4,
                )
            return value

        if bound == "two-sided":
            return np.array([solve_side(-1), solve_side(1)])
        elif bound == "lower":
            return np.array([solve_side(-1)])
        else:
            return np.array([solve_side(1)])

    def _lr_key(self, idx: int, crit: float, direction: Any) -> tuple:
        """The key a parameter's likelihood-ratio side is kept under."""
        return (
            idx,
            float(crit),
            float(np.sign(direction)),
            np.asarray(self.params, dtype=float).tobytes(),
            id(self.surv_data),
        )

    def _lr_param_side(self, idx: int, crit: float, direction: Any) -> float:
        """One side of the likelihood-ratio interval on core parameter
        ``idx`` at the critical value ``crit``: the bound, the edge of the
        parameter's space, or ``nan`` where it cannot be found. See
        ``_param_cb_lr``.

        Each side has a path of its own, so a bound does not depend on
        whether the other side was asked for. A side is solved once per
        critical value and kept (a one-sided bound at ``alpha`` is the end
        of the two-sided one at ``2 alpha``, and ``cb`` reads the interval
        of every parameter).
        """
        theta_hat = float(self.params[idx])
        key = self._lr_key(idx, crit, direction)
        cache = self.__dict__.setdefault("_lr_sides", {})
        if key in cache:
            return cache[key]
        nll_hat = self._lr_neg_ll(np.asarray(self.params, dtype=float))

        hess_inv = getattr(self, "hess_inv", None)
        if (
            hess_inv is not None
            and np.ndim(hess_inv) == 2
            and np.isfinite(hess_inv[idx, idx])
            and hess_inv[idx, idx] > 0
        ):
            se = float(np.sqrt(hess_inv[idx, idx]))
        else:
            se = 0.5 * abs(theta_hat) if theta_hat != 0 else 1.0

        coords, limits = self._lr_coords()
        coord = coords[idx]
        w_hat = float(np.clip(coord.to_u(theta_hat), *coord.ends))
        # The first step: the standard error in the search coordinate.
        with np.errstate(all="ignore"):
            step = se / coord.slope(theta_hat)
        if not (np.isfinite(step) and step > 0):
            step = 1.0

        path = _LRPath()

        def deviance(w: float) -> float:
            nll = self._profile_neg_ll(idx, coord.from_u(w), path=path)
            return 2.0 * (nll - nll_hat)

        status, w = _lr_walk(
            deviance,
            w_hat,
            step,
            direction,
            crit,
            end=coord.ends[1] if direction > 0 else coord.ends[0],
            stop=limits[idx][1] if direction > 0 else limits[idx][0],
        )
        if status in ("root", "stop"):
            value = coord.from_u(w)
        elif status == "edge":
            value = float(coord.edge(direction))
        else:
            value = np.nan
        cache[key] = value
        # The points the walk solved inside the region, for the band's
        # searches to start from (see ``_cb_lr``).
        user_fixed = self._user_fixed_idx()
        free = [
            j
            for j in range(len(self.params))
            if j != idx and j not in user_fixed
        ]
        points = []
        for w_i, u_i, f_i in zip(path.w, path.u, path.f):
            if 2.0 * (f_i - nll_hat) <= crit:
                theta = np.array(self.params, dtype=float)
                theta[idx] = coord.from_u(w_i)
                theta[free] = [coords[j].from_u(v) for j, v in zip(free, u_i)]
                points.append(theta)
        self.__dict__.setdefault("_lr_points", {})[key] = points
        return value

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

    def cs(self, x: npt.ArrayLike, X: npt.ArrayLike) -> npt.NDArray:
        r"""

        The conditional survival of the model; that is, the probability
        that an item that has survived to ``X`` survives a further ``x``:

        .. math::
            R(x, X) = \frac{R(x + X)}{R(X)}

        Parameters
        ----------

        x : array like or scalar
            The further durations at which conditional survival is to be
            calculated.
        X : array like or scalar
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
        still count among the survivors at ``X``, and survival to an
        ``X`` before the offset is certain. Where :math:`R(X) = 0` the
        conditional survival is undefined and ``nan`` is returned.
        """
        x_arr = np.asarray(x, dtype=float)
        X_arr = np.asarray(X, dtype=float)
        Xg = X_arr - self.gamma
        s0 = getattr(self.dist, "support", (-np.inf, np.inf))[0]
        with np.errstate(all="ignore"):
            # The ratio of the model's own sf. Handing the shifted X to
            # ``dist.cs`` ignored p and f0 entirely (0.29 instead of 0.67
            # for p = 0.7) and, for an X before the offset, evaluated the
            # base sf at a negative time (1.0 or nan instead of 0.96).
            cs = np.asarray(
                self.sf(x_arr + X_arr) / self.sf(X_arr), dtype=float
            )
            if (self.p == 1) and (self.f0 == 0):
                # A plain model inside its support keeps the
                # distribution's own form, which is exact where the ratio
                # cancels in the far tail (the memoryless Exponential).
                inside = np.broadcast_to(Xg >= s0, cs.shape)
                if inside.any():
                    Xg_safe = np.where(Xg >= s0, Xg, s0)
                    own = np.asarray(
                        self.dist.cs(x_arr, Xg_safe, *self.params),
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
            on the logit of ``sf``; those on ``hf`` and ``df`` are on the log
            scale (the logit scale for a discrete distribution, whose hazard
            and mass are probabilities), and are 0 where the rate is 0. Where
            the delta-method variance is negative (the covariance is not
            positive definite) a Wald bound is ``nan``, with a warning.
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
            samples (the reliability-engineering default), but it is computed
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
        array([[0.65821001, 0.89149672],
               [0.17231083, 0.4256438 ]])
        >>> model.cb([5, 10], on="sf", bound="lower")
        array([0.68394304, 0.18735771])
        """
        t = np.atleast_1d(x)
        if self.method != "MLE":
            raise ValueError("Only MLE has confidence bounds")
        # Checked up front, as param_cb does: an unrecognised value (say
        # 'both') used to fall through to the lower-bound branch and
        # return one bound as if it were what was asked for.
        if bound not in ("two-sided", "lower", "upper"):
            raise ValueError(
                "bound must be 'two-sided', 'lower' or 'upper'; got "
                f"{bound!r}"
            )
        if np.size(t) == 0 and on in ("sf", "R", "ff", "F", "Hf", "hf", "df"):
            # Nothing to bound (the Jacobian of no values fails).
            return np.empty((0, 2) if bound == "two-sided" else (0,))

        if method.lower() in (
            "lr",
            "likelihood",
            "likelihood-ratio",
            "profile",
        ):
            return self._cb_lr(t, on, alpha_ci, bound)
        elif method.lower() != "wald":
            raise ValueError(
                f"Unknown confidence-bound method '{method}'; "
                "use 'wald' or 'lr'."
            )

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
                raise ValueError(
                    "'on' must be one of 'sf', 'R', 'ff', 'F', 'Hf', 'hf' "
                    f"or 'df'; got {on!r}"
                )
        finally:
            np.seterr(**old_err_state)

        return cb

    def _cb_lr_on_func(self, on: str) -> Any:
        """Return ``g(t, theta)`` for the requested ``on`` function.

        Evaluates the chosen distribution function at a single time for a
        candidate core-parameter vector, so the profile optimiser can push it
        to the edge of the likelihood region.
        """
        valid = ("sf", "R", "ff", "F", "Hf", "hf", "df")
        if on not in valid:
            raise ValueError(f"'on' must be one of {valid}")

        def g(t: Any, theta: npt.NDArray) -> Any:
            xt = np.atleast_1d(t) - self.gamma
            if on in ("sf", "R"):
                return self.dist.sf(xt, *theta)[0]
            if on in ("ff", "F"):
                return self.dist.ff(xt, *theta)[0]
            if on == "Hf":
                return self.dist.Hf(xt, *theta)[0]
            if on == "hf":
                return self.dist.hf(xt, *theta)[0]
            return self.dist.df(xt, *theta)[0]

        return g

    def _cb_lr(self, t: Any, on: str, alpha_ci: float, bound: str) -> Any:
        """Profile-likelihood (likelihood-ratio) band on a model function.

        At each time ``x`` the bound is the extreme value of the ``on``
        function over the parameter confidence region ``{theta :
        deviance(theta) <= crit}`` -- the piece of it that contains the
        estimate. It is found as ``param_cb`` finds a parameter's bound,
        with the function's value ``psi`` in the parameter's place: the
        bound is where the function's own profile deviance, ``2[min{
        nll(theta) : g(x, theta) = psi} - nll_hat]``, first reaches
        ``crit``. ``psi`` is on the scale the Wald band uses -- the logit
        of ``sf`` (from which the ``sf``, ``ff`` and ``Hf`` bands all
        come, so they agree exactly), the log of a continuous hazard or
        density, the logit of a discrete one.

        Each side is sought first directly (the extreme of ``psi`` over
        the region, by SLSQP, from the estimate and then from the points
        of the region farther out that the parameters' own walks found),
        and a result is taken only if it checks out as that crossing and
        is at least as far out as every point of the region known. Failing
        that, the profile of ``psi`` is walked out from the farthest point
        known (``_lr_walk``); where it stays below ``crit``, or levels off
        below it, to the end of the scale, the band reaches the edge of
        the function's range. Each end is solved once per level and kept.

        The search for the extreme from a warm start alone stopped wherever
        it first met the region's boundary: ExpoWeibull and
        NegativeBinomial bands that were ``nan`` where a search failed, or
        short of the region's far corners (a NegativeBinomial ``sf(8)``
        lower bound of 0.00918 for 0.00587), and 95% and 80% bands that
        were not nested (#421).
        """
        self._ensure_surv_data()
        if self.offset or self.lfp or self.zi:
            raise NotImplementedError(
                "Likelihood-ratio confidence bounds are not yet available "
                "for offset, limited-failure-population or zero-inflated "
                "models; use method='wald'."
            )
        if bound not in ("two-sided", "lower", "upper"):
            raise ValueError("bound must be 'two-sided', 'lower' or 'upper'")
        valid = ("sf", "R", "ff", "F", "Hf", "hf", "df")
        if on not in valid:
            raise ValueError(f"'on' must be one of {valid}")

        if bound == "two-sided":
            crit = z(1.0 - alpha_ci / 2.0) ** 2
        else:
            crit = z(1.0 - alpha_ci) ** 2

        t = np.atleast_1d(t).astype(float)
        theta_hat = np.array(self.params, dtype=float)
        user_fixed = self._user_fixed_idx()
        free = [j for j in range(len(theta_hat)) if j not in user_fixed]
        if not free:
            # Every parameter was fixed at fit time: the region is the
            # estimate, and so is the band.
            g = self._cb_lr_on_func(on)
            at = np.array([g(time, theta_hat) for time in t], dtype=float)
            if bound == "two-sided":
                return np.column_stack([at, at])
            return at
        if len(free) == 1:
            band = self._cb_lr_one_param(
                t, self._cb_lr_on_func(on), free[0], alpha_ci, bound
            )
            if band is not None:
                return band

        survival = on in ("sf", "R", "ff", "F", "Hf")
        # ff and Hf fall as sf rises: their lower bound is sf's upper.
        falling = on in ("ff", "F", "Hf")
        want_lower = bound in ("two-sided", "lower")
        want_upper = bound in ("two-sided", "upper")
        if falling:
            want_lower, want_upper = want_upper, want_lower

        if survival:

            def psi_of(time: Any, theta: npt.NDArray) -> float:
                x = np.atleast_1d(time) - self.gamma
                return float(
                    np.log(self.dist.sf(x, *theta)[0])
                    - np.log(self.dist.ff(x, *theta)[0])
                )

            ends = (_LN_TINY, -_LN_TINY)
            if on in ("sf", "R"):
                value: Callable[[Any], Any] = expit
            elif on in ("ff", "F"):
                value = lambda v: expit(-v)  # noqa: E731
            else:
                value = lambda v: np.logaddexp(0.0, -v)  # noqa: E731
        else:
            rate = self.dist.hf if on == "hf" else self.dist.df
            discrete = bool(self.dist.discrete)

            def psi_of(time: Any, theta: npt.NDArray) -> float:
                g = rate(np.atleast_1d(time) - self.gamma, *theta)[0]
                if discrete:
                    # A discrete hazard and mass are probabilities: the
                    # logit scale, as the Wald band's.
                    return float(np.log(g) - np.log1p(-g))
                return float(np.log(g))

            if discrete:
                ends = (_LN_TINY, -_LN_TINY)
                value = expit
            else:
                ends = (_LN_TINY, _LN_MAX)
                value = np.exp

        # A bound is solved once per function, time, level and side, and
        # kept: the sf, ff and Hf bands are one band, and a one-sided
        # bound at alpha is an end of the two-sided one at 2 alpha.
        kind = "survival" if survival else on
        cache = self.__dict__.setdefault("_lr_bands", {})
        region: list = []
        lower = np.full(t.shape, np.nan)
        upper = np.full(t.shape, np.nan)
        failed: list[float] = []
        with np.errstate(all="ignore"):
            for i, time in enumerate(t):
                key_lo = (kind, float(time), *self._lr_key(-1, crit, -1))
                key_hi = (kind, float(time), *self._lr_key(-1, crit, 1))
                need_lo = want_lower and key_lo not in cache
                need_hi = want_upper and key_hi not in cache
                if need_lo or need_hi:
                    if not region:
                        region.extend(self._lr_region(free, crit))
                    lo, hi = self._cb_lr_psi_bounds(
                        lambda theta: psi_of(time, theta),
                        free,
                        crit,
                        need_lo,
                        need_hi,
                        ends,
                        *region,
                    )
                    if need_lo:
                        cache[key_lo] = lo
                    if need_hi:
                        cache[key_hi] = hi
                lo = cache[key_lo] if want_lower else np.nan
                hi = cache[key_hi] if want_upper else np.nan
                if (want_lower and np.isnan(lo)) or (
                    want_upper and np.isnan(hi)
                ):
                    failed.append(float(time))
                lower[i], upper[i] = value(lo), value(hi)

        if failed:
            warnings.warn(
                "The likelihood-ratio bound could not be found at "
                f"t = {sorted(set(failed))} (the constrained optimiser "
                "failed from every start); nan is returned there. "
                "method='wald' gives a bound in its place.",
                RuntimeWarning,
                # _cb_lr -> cb -> the query-shape wrapper -> the caller
                stacklevel=4,
            )

        if falling:
            lower, upper = upper, lower
        if bound == "two-sided":
            return np.column_stack([lower, upper])
        elif bound == "lower":
            return lower
        else:
            return upper

    def _lr_region(
        self, free: list[int], crit: float
    ) -> tuple[list[tuple[Any, Any]], list[list[npt.NDArray]]]:
        """The box a likelihood-ratio band's searches run in, and the
        points they may start from (one list per walk), at the critical
        value ``crit``.

        The box is the one the parameters' own intervals at this level
        make: the region's extent in each parameter is that parameter's
        likelihood-ratio interval, so the box holds every point the band
        can come from, and it keeps a search for a value the function
        never takes from wandering off to where the likelihood is slow or
        not defined (a NegativeBinomial ``r`` of 1e-308, whose incomplete
        beta took 4.5 s a call). The points are those of the region that
        the intervals' walks passed through: they reach its far corners
        (an ExpoWeibull ``beta`` running off to infinity with ``alpha`` at
        the largest observation), which a search from the estimate does
        not find.
        """
        coords, limits = self._lr_coords()
        free_coords = [coords[j] for j in free]
        box = self._lr_box(free_coords, [limits[j] for j in free])
        with np.errstate(all="ignore"):
            for k, j in enumerate(free):
                b_lo, b_hi = box[k]
                lo_j = coords[j].to_u(self._lr_param_side(j, crit, -1))
                hi_j = coords[j].to_u(self._lr_param_side(j, crit, 1))
                if np.isfinite(lo_j):
                    b_lo = lo_j if b_lo is None else max(b_lo, lo_j)
                if np.isfinite(hi_j):
                    b_hi = hi_j if b_hi is None else min(b_hi, hi_j)
                box[k] = (b_lo, b_hi)
            # One list per walk.
            seeds = [
                [
                    np.array([coords[i].to_u(theta[i]) for i in free])
                    for theta in self.__dict__.get("_lr_points", {}).get(
                        self._lr_key(j, crit, d), []
                    )
                ]
                for j in free
                for d in (-1.0, 1.0)
            ]
        return box, seeds

    def _cb_lr_psi_bounds(
        self,
        psi_of: Callable[[npt.NDArray], float],
        free: list[int],
        crit: float,
        want_lower: bool,
        want_upper: bool,
        ends: tuple[float, float],
        box: list[tuple[Any, Any]],
        seeds: list[list[npt.NDArray]],
    ) -> tuple[float, float]:
        """The likelihood-ratio bounds on a function ``psi_of(theta)`` of
        the free core parameters, searched in ``box``: ``(lower,
        upper)``, ``nan`` for a side not asked for or not found, ``-inf``
        / ``inf`` for one at the edge of the scale. See ``_cb_lr``."""
        theta_hat = np.array(self.params, dtype=float)
        nll_hat = self._lr_neg_ll(theta_hat)
        coords, _ = self._lr_coords()
        free_coords = [coords[j] for j in free]
        bounds = box if any(b != (None, None) for b in box) else None
        u_hat = np.array([coords[j].to_u(theta_hat[j]) for j in free])

        def theta_of(u: npt.NDArray) -> npt.NDArray:
            theta = theta_hat.copy()
            theta[free] = [c.from_u(v) for c, v in zip(free_coords, u)]
            return theta

        def nll_of(u: npt.NDArray) -> float:
            nll = self._lr_neg_ll(theta_of(u))
            return nll if np.isfinite(nll) else np.inf

        def psi_u(u: npt.NDArray) -> float:
            if not np.all(np.isfinite(u)):
                return np.nan
            # Held to the ends of its scale where the function reaches the
            # edge of its range (a density that underflows to 0), so that
            # a search can still step there.
            return float(np.clip(psi_of(theta_of(u)), *ends))

        psi_hat = psi_of(theta_of(u_hat))
        if not np.isfinite(psi_hat):
            # The function is at the edge of its range at the estimate (a
            # rate of 0 below the support): the bounds are that value.
            return psi_hat, psi_hat

        def solve(
            target: float, starts: list[npt.NDArray]
        ) -> tuple[float, npt.NDArray | None]:
            # The least negative log-likelihood with psi at ``target``.
            best, best_u = np.inf, None
            constraint = {
                "type": "eq",
                "fun": lambda u: psi_u(u) - target,
                "jac": lambda u: _central_gradient(psi_u, u),
            }
            for x0 in starts:
                try:
                    res = minimize(
                        nll_of,
                        self._lr_start(x0, box),
                        method="SLSQP",
                        jac=lambda u: _central_gradient(nll_of, u),
                        bounds=bounds,
                        constraints=[constraint],
                        options={"ftol": 1e-10, "maxiter": 60},
                    )
                except (ValueError, np.linalg.LinAlgError):
                    continue
                if not np.all(np.isfinite(res.x)):
                    continue
                gap = abs(psi_u(res.x) - target)
                nll = nll_of(res.x)
                if gap <= 1e-7 * max(1.0, abs(target)) and nll < best:
                    best, best_u = nll, np.asarray(res.x)
            return best, best_u

        # The search is checked where its answer is known, at the
        # estimate: if it fails there, it cannot be trusted anywhere.
        _, u_start = solve(psi_hat, [u_hat])
        if u_start is None:
            return np.nan, np.nan

        # The first step: the Wald standard error on the psi scale.
        hess_inv = getattr(self, "hess_inv", None)
        step = np.nan
        if hess_inv is not None and np.ndim(hess_inv) == 2:
            slopes = np.array(
                [coords[j].slope(theta_hat[j]) for j in free], dtype=float
            )
            cov_u = np.asarray(hess_inv, dtype=float)[np.ix_(free, free)]
            cov_u = cov_u / np.outer(slopes, slopes)
            grad = _central_gradient(psi_u, u_hat)
            step = float(np.sqrt(grad @ cov_u @ grad))
        if not (np.isfinite(step) and step > 0):
            step = 1.0

        def dev_u(u: npt.NDArray) -> float:
            return 2.0 * (nll_of(u) - nll_hat)

        # The points of the region known so far, with their psi, and
        # those of each walk.
        known = [(psi_hat, u_start)]
        walks = []
        for group in seeds:
            walk = []
            for seed in group:
                seed = self._lr_start(seed, box)
                psi_seed = psi_u(seed)
                if np.isfinite(psi_seed) and dev_u(seed) <= crit:
                    walk.append((psi_seed, seed))
            known.extend(walk)
            if walk:
                walks.append(walk)

        def direct(direction: float, start: npt.NDArray) -> float | None:
            # The extreme of psi over the region, sought directly (SLSQP),
            # is taken when it checks out: its deviance at crit or below,
            # and none with psi a hair further out (1e-6 of it) at crit or
            # below. That is where the profile of psi crosses crit, which
            # the walk would find at many times the cost. Otherwise
            # ``None``.
            try:
                res = minimize(
                    lambda u: -direction * psi_u(u),
                    start,
                    method="SLSQP",
                    jac=lambda u: -direction * _central_gradient(psi_u, u),
                    bounds=bounds,
                    constraints=[
                        {
                            "type": "ineq",
                            "fun": lambda u: crit - dev_u(u),
                            "jac": lambda u: -_central_gradient(dev_u, u),
                        }
                    ],
                    options={"ftol": 1e-10, "maxiter": 100},
                )
            except (ValueError, np.linalg.LinAlgError):
                return None
            if not np.all(np.isfinite(res.x)):
                return None
            psi_star = psi_u(res.x)
            if not (
                np.isfinite(psi_star)
                and dev_u(res.x) <= crit + _LR_NOISE
                and direction * (psi_star - psi_hat) >= 0
            ):
                return None
            known.append((psi_star, np.asarray(res.x)))
            beyond = psi_star + direction * 1e-6 * max(1.0, abs(psi_star))
            nll, u = solve(beyond, [np.asarray(res.x), u_hat])
            if u is not None and 2.0 * (nll - nll_hat) < crit:
                return None
            return psi_star

        def solve_side(direction: float) -> float:
            # The search starts from the estimate and from the farthest
            # points of the two walks that reach farthest (the region can
            # have more than one local extreme: an ExpoWeibull hf(13) of
            # 0.108 on its near boundary at 99%, and 0.102 down the valley
            # of alpha -> 0, inside the 95% region already), and the most
            # extreme result that checks out is taken.
            tips = sorted(
                (max(walk, key=lambda k: direction * k[0]) for walk in walks),
                key=lambda k: -direction * k[0],
            )
            best = None
            for start in [u_start] + [k[1] for k in tips[:2]]:
                quick = direct(direction, start)
                if quick is not None and (
                    best is None or direction * quick > direction * best
                ):
                    best = quick
            far = max(direction * k[0] for k in known)
            if best is not None and direction * best >= far:
                return best
            # The bound is at least as far out as every point of the
            # region known: a search that stops short of one has stopped
            # at a local extreme, and is tried again from the points
            # beyond it, farthest first.
            tried: list[int] = [id(k[1]) for k in tips[:2]]
            for _ in range(3):
                beyond = [
                    k
                    for k in sorted(known, key=lambda k: -direction * k[0])
                    if id(k[1]) not in tried and k[1] is not u_start
                ]
                if not beyond:
                    break
                start = beyond[0][1]
                tried.append(id(start))
                quick = direct(direction, start)
                far = max(direction * k[0] for k in known)
                if quick is not None and direction * quick >= far:
                    return quick
            # Otherwise the profile of psi is walked out from the
            # farthest point known.
            far_psi, far_u = max(known, key=lambda k: direction * k[0])
            path = _LRPath()
            path.add(far_psi, far_u)

            def deviance(target: float) -> float:
                nll, u = solve(target, path.starts(target))
                if u is None:
                    nll, u = solve(target, [u_hat])
                if u is None:
                    # No start reaches the target: the function does not
                    # take that value near where the walk has come from
                    # (a NegativeBinomial df(5) above 0.195, its value
                    # in the Poisson limit; a Uniform hazard above
                    # 1 / (max(x) - x), which b >= max(x) caps). It reads
                    # as beyond any critical value.
                    return np.inf
                path.add(target, u)
                return 2.0 * (nll - nll_hat)

            status, w = _lr_walk(
                deviance,
                far_psi,
                step,
                direction,
                crit,
                end=ends[1] if direction > 0 else ends[0],
                d_hat=dev_u(far_u),
            )
            if status == "root":
                return w
            if status == "edge":
                return np.inf if direction > 0 else -np.inf
            return np.nan

        lower = solve_side(-1.0) if want_lower else np.nan
        upper = solve_side(1.0) if want_upper else np.nan
        return lower, upper

    def _cb_lr_one_param(
        self, t: Any, g: Any, j: int, alpha_ci: float, bound: str
    ) -> Any:
        """The likelihood-ratio band of a model with one free parameter.

        Its likelihood region is an interval, the profile bound on the
        parameter, so the band at each time is the extreme of ``g`` over
        that interval: at an end, or at an interior stationary point of
        ``g`` (a density at ``x`` peaks in the scale). A constrained
        search from a warm start found one end or the other, not always
        the more extreme: a Geometric df(5) lower bound of 0.0740 in a
        sweep but 0.0652 queried alone (#421). ``None`` (the general
        search is used instead) where the interval cannot be found.
        """
        name = self.dist.param_names[j]
        # A one-sided bound at alpha is an end of the two-sided region at
        # 2 alpha: the same chi-squared critical value.
        level = alpha_ci if bound == "two-sided" else 2.0 * alpha_ci
        if not 0 < level < 1:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ends = self._param_cb_lr(name, level, "two-sided")
        if not np.all(np.isfinite(ends)):
            return None
        lo, hi = (float(e) for e in ends)
        # The likelihood at a support edge (p = 0 or 1) is typically nan,
        # so g is evaluated just inside it, as the search is.
        lo_b, hi_b = self.dist.bounds[j]
        lo = 1e-10 if lo_b == 0 and lo == 0 else lo
        hi = hi - 1e-10 if hi_b == 1 and hi == 1 else hi
        theta = np.array(self.params, dtype=float)

        def at(time: Any, v: Any) -> Any:
            th = theta.copy()
            th[j] = v
            return g(time, th)

        lower = np.empty(t.shape)
        upper = np.empty(t.shape)
        with np.errstate(all="ignore"):
            for i, time in enumerate(t):
                values = [at(time, lo), at(time, hi), at(time, theta[j])]
                for sign in (1.0, -1.0):
                    res = minimize_scalar(
                        lambda v: sign * at(time, v),
                        bounds=(lo, hi),
                        method="bounded",
                    )
                    if np.isfinite(res.fun):
                        values.append(sign * res.fun)
                values = np.asarray(values, dtype=float)
                if not np.all(np.isfinite(values[:3])):
                    return None
                values = values[np.isfinite(values)]
                lower[i], upper[i] = values.min(), values.max()
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
                raise ValueError(
                    "Model carries no parameter covariance (the Hessian "
                    "was singular at the optimum, or the model was not fit "
                    "by MLE); confidence bounds are unavailable."
                )
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
        """Confidence bound on the survival function via a logit transform.

        Working on the logit of R keeps the bound within ``(0, 1)``. The
        returned array is the transpose of the per-point bounds, matching the
        layout the public ``cb`` method expects.
        """

        def sf_func(phi: npt.NDArray) -> Any:
            return self._cb_full_sf(x, phi, ctx)

        sd_R = self._cb_sd(self._cb_delta_var(sf_func, ctx), x, "sf")
        R_hat = self._cb_full_sf(x, ctx.phi_hat, ctx)
        if bound == "two-sided":
            diff = z(alpha_ci / 2) * sd_R * np.array([1.0, -1.0]).reshape(2, 1)
        elif bound == "upper":
            diff = z(alpha_ci) * sd_R
        else:
            diff = -z(alpha_ci) * sd_R

        with np.errstate(all="ignore"):
            exponent = diff / (R_hat * (1 - R_hat))
            R_cb = R_hat / (R_hat + (1 - R_hat) * np.exp(exponent))
        # At the boundary (R = 0 or 1, e.g. t <= gamma) the logit transform
        # degenerates to 0/0; the bound there is the boundary itself (#256).
        R_cb = np.where(np.broadcast_to(R_hat == 1.0, R_cb.shape), 1.0, R_cb)
        R_cb = np.where(np.broadcast_to(R_hat == 0.0, R_cb.shape), 0.0, R_cb)
        return R_cb.T

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
        self, heuristic: str = "Nelson-Aalen", alpha_ci: float = 0.05
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

        Returns
        -------

        data : dict
            Returns dictionary containing the data needed to do a plot.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> x = Weibull.random(100, 10, 3)
        >>> model = Weibull.fit(x)
        >>> data = model.get_plot_data()
        """
        self._require_data("get_plot_data()")
        cb_func: Callable[[Any], Any] | None
        if (
            hasattr(self, "hess_inv")
            and (self.method == "MLE")
            and (self.hess_inv is not None)
        ):

            def _cb_func(x_model: npt.NDArray) -> Any:
                return self.cb(x_model, on="ff", alpha_ci=alpha_ci)

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
    ) -> list:
        """
        A method to do a probability plot

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

        Returns
        -------

        plot : matplotlib.axes.Axes
            the axes the probability plot was drawn onto

        Examples
        --------

        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> np.random.seed(1)
        >>> x = Weibull.random(100, 10, 3)
        >>> model = Weibull.fit(x)
        >>> model.plot()
        <Axes: title={'center': 'Weibull Probability Plot'}, ylabel='CDF'>
        """
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gcf().gca()

        if not hasattr(self, "params"):
            raise ValueError("Can't plot model that failed to fit")

        if self.method == "given parameters":
            detail = "Can't plot model that was given parameters and no data"
            raise ValueError(detail)

        if not (
            hasattr(self.dist, "mpp_y_transform")
            and hasattr(self.dist, "mpp_inv_y_transform")
        ):
            raise NotImplementedError(
                f"{self.dist.name} does not support probability plotting"
            )

        self._require_data("plot()")
        heuristic = adjust_heuristic(self.data["c"], self.data["t"], heuristic)

        d = self.get_plot_data(heuristic=heuristic, alpha_ci=alpha_ci)

        return draw_probability_plot(
            ax,
            d,
            lambda x: self.dist.mpp_y_transform(x, *self.params),
            lambda x: self.dist.mpp_inv_y_transform(x, *self.params),
            title=f"{self.dist.name} Probability Plot",
            plot_bounds=plot_bounds,
        )
