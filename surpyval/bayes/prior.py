"""Prior beliefs for a Bayesian fit.

A prior says what is believed about a model before its data are seen.
SurPyval takes a prior on each parameter (any SurPyval distribution, or
the plain helpers :func:`around`, :func:`between`, :func:`uniform`,
:func:`flat` and :func:`fixed`), a joint prior on the parameters together
(:class:`MultivariateNormal`, which :func:`from_model` builds from an
earlier fit), a prior on the quantities people know (:func:`on_quantities`:
a B-life, a reliability at a time, the mean, or any function of the
fitted model by :func:`on_function`) and the old data themselves
(:func:`from_data`, a power prior; :func:`from_fit` for the data of an
earlier fit of any family).

Every prior is used on the parameter's unconstrained scale (the log of a
positive parameter, the logit of a proportion; see
:mod:`surpyval.bayes.transforms`), where the grid is laid. A prior given
as a density on the parameter itself is carried there by the change of
variables, so nothing is approximated.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

from surpyval.bayes.transforms import Transform
from surpyval.utils.aliases import canonical_name
from surpyval.utils.warnings import ignoring_warnings

__all__ = [
    "around",
    "between",
    "uniform",
    "flat",
    "fixed",
    "from_distribution",
    "from_model",
    "from_data",
    "from_params",
    "from_fit",
    "on_quantities",
    "on_function",
    "check",
    "Prior",
    "JointPrior",
    "NormalOnScale",
    "DistributionPrior",
    "Flat",
    "Fixed",
    "MultivariateNormal",
    "PowerPrior",
    "FitPowerPrior",
    "OnQuantities",
    "FunctionStatement",
    "PriorSpec",
]

#: A probability that counts as no mass at all, for support checks.
_NO_MASS = 1e-9


def _z(mass: float) -> float:
    """The standard normal quantile that leaves ``(1 - mass) / 2`` above."""
    return float(stats.norm.ppf(0.5 + 0.5 * mass))


def format_number(v: float) -> str:
    """A number to four significant figures, for the summaries: with
    thousands separators from 1,000 to below 10 million, in scientific
    notation beyond and below 0.001."""
    v = float(v)
    if not np.isfinite(v):
        return str(v)
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e7 or a < 1e-3:
        return f"{v:.4g}"
    if a >= 1000:
        digits = int(np.floor(np.log10(a))) + 1
        return f"{round(v, 4 - digits):,.0f}"
    return f"{v:.4g}"


# The name the summaries here use.
_fmt = format_number


def _check_mass(mass: float) -> float:
    mass = float(mass)
    if not 0 < mass < 1:
        raise ValueError(
            f"mass must be a probability between 0 and 1 (exclusive), the "
            f"chance the value lies in the interval; got {mass}."
        )
    return mass


# ---------------------------------------------------------------------------
# Priors on one parameter (or one quantity)
# ---------------------------------------------------------------------------
class Prior:
    """A belief about one parameter or one quantity.

    Subclasses give the log density with respect to the unconstrained
    value ``phi`` of the target once they are bound to the target's
    transform (:meth:`log_density_phi`), how to draw from it, and a plain
    description.

    Examples
    --------
    >>> import surpyval as sp
    >>> isinstance(sp.prior.around(2, between=(1.5, 3)), sp.prior.Prior)
    True
    """

    #: Whether the prior has a finite total mass (``flat()`` has not).
    proper: bool = True
    #: Whether it fixes the parameter (``fixed(v)``).
    is_fixed: bool = False
    #: The words the user gave, for the summary and the saved spec.
    said: str = ""

    def log_density_phi(self, phi: npt.NDArray, t: Transform) -> Any:
        raise NotImplementedError

    def centre_phi(self, t: Transform) -> float | None:
        """A typical value on the unconstrained scale, to start from."""
        return None

    def draw_phi(self, size: int, t: Transform, rng: Any) -> npt.NDArray:
        raise NotImplementedError

    def check_support(self, t: Transform, name: str) -> None:
        """Refuse a prior that puts mass outside the target's range."""

    def describe(self, t: Transform) -> str:
        return self.said

    def to_dict(self) -> dict:
        raise NotImplementedError


class NormalOnScale(Prior):
    """A normal belief on the target's unconstrained scale: a LogNormal on
    a positive parameter, a logit-normal on a proportion, a Normal on the
    real line. ``around`` and ``between`` build one from an interval,
    ``from_model`` and ``from_params`` from an earlier fit or numbers.

    Examples
    --------
    >>> from surpyval import prior
    >>> b = prior.around(2, between=(1.5, 3))
    >>> type(b).__name__
    'NormalOnScale'
    """

    def __init__(
        self,
        lower: float | None = None,
        upper: float | None = None,
        mass: float = 0.9,
        *,
        loc: float | None = None,
        scale: float | None = None,
        kind: str | None = None,
        said: str = "",
        stated: float | None = None,
    ) -> None:
        self.lower = None if lower is None else float(lower)
        self.upper = None if upper is None else float(upper)
        self.mass = _check_mass(mass)
        self.loc = None if loc is None else float(loc)
        self.scale = None if scale is None else float(scale)
        # The unconstrained scale the numbers are on, when they were given
        # there (from a model); None to take the target's.
        self.kind = kind
        self.said = said
        self.stated = None if stated is None else float(stated)
        if self.loc is None:
            if self.lower is None or self.upper is None:
                raise ValueError(
                    "Give the interval (lower, upper) the value is believed "
                    "to lie in."
                )
            if not self.lower < self.upper:
                raise ValueError(
                    f"between=(a, b) needs a < b; got ({self.lower}, "
                    f"{self.upper})."
                )
        elif self.scale is None or not self.scale > 0:
            raise ValueError(
                f"scale must be a positive spread; got {self.scale}."
            )

    def _resolved(self, t: Transform) -> tuple[float, float]:
        """``(loc, scale)`` on the target's unconstrained scale."""
        if self.loc is not None and self.scale is not None:
            if self.kind is not None and self.kind != t.kind:
                raise ValueError(
                    f"This prior was built on the {self.kind} scale, but "
                    f"its parameter is on the {t.kind} scale."
                )
            return self.loc, self.scale
        assert self.lower is not None and self.upper is not None
        for v, end in ((self.lower, "lower"), (self.upper, "upper")):
            if not (t.lower < v < t.upper):
                raise ValueError(
                    f"The {end} end of the interval, {v}, is outside the "
                    f"range of the value ({_range(t)})."
                )
        a, b = sorted(
            (float(t.to_phi(self.lower)), float(t.to_phi(self.upper)))
        )
        z = _z(self.mass)
        return 0.5 * (a + b), (b - a) / (2 * z)

    def log_density_phi(self, phi: npt.NDArray, t: Transform) -> Any:
        m, s = self._resolved(t)
        return stats.norm.logpdf(phi, m, s)

    def centre_phi(self, t: Transform) -> float | None:
        return self._resolved(t)[0]

    def draw_phi(self, size: int, t: Transform, rng: Any) -> npt.NDArray:
        m, s = self._resolved(t)
        return rng.normal(m, s, size)

    def check_support(self, t: Transform, name: str) -> None:
        self._resolved(t)

    def median(self, t: Transform) -> float:
        return float(t.to_theta(self._resolved(t)[0]))

    def describe(self, t: Transform) -> str:
        m, s = self._resolved(t)
        z = _z(0.9)
        lo, hi = sorted(
            (float(t.to_theta(m - z * s)), float(t.to_theta(m + z * s)))
        )
        built = {
            "log": "LogNormal",
            "logit": "logit-normal",
            "neglog": "normal on the log scale",
            "identity": "Normal",
        }[t.kind]
        words = self.said or "a normal belief"
        return (
            f"{words}, read as {built}(median "
            f"{_fmt(float(t.to_theta(m)))}, 90% between {_fmt(float(lo))} "
            f"and {_fmt(float(hi))})"
        )

    def to_dict(self) -> dict:
        return {
            "prior": "NormalOnScale",
            "lower": self.lower,
            "upper": self.upper,
            "mass": self.mass,
            "loc": self.loc,
            "scale": self.scale,
            "kind": self.kind,
            "said": self.said,
            "stated": self.stated,
        }


def _range(t: Transform) -> str:
    lo = "-inf" if not np.isfinite(t.lower) else _fmt(t.lower)
    hi = "inf" if not np.isfinite(t.upper) else _fmt(t.upper)
    return f"between {lo} and {hi}"


class DistributionPrior(Prior):
    """A prior given as any SurPyval distribution model (``Gamma.from_params
    ([2, 1])``, ``LogNormal.from_params(...)``, ``Uniform.from_params
    ([a, b])``), a density on the parameter itself.

    Examples
    --------
    >>> import surpyval as sp
    >>> p = sp.prior.DistributionPrior(sp.Gamma.from_params([2.0, 1.0]))
    >>> p.said
    'Gamma(2, 1)'
    """

    def __init__(self, model: Any, truncate: bool = False) -> None:
        if not hasattr(model, "dist") or not hasattr(model, "df"):
            raise ValueError(
                "A prior distribution must be a SurPyval model, e.g. "
                "surpyval.Gamma.from_params([2, 1]); got "
                f"{type(model).__name__}."
            )
        if getattr(model.dist, "discrete", False):
            raise ValueError(
                f"A prior on a parameter must be a continuous distribution;"
                f" {model.dist.name} is discrete."
            )
        self.model = model
        self.truncate = bool(truncate)
        params = ", ".join(_fmt(p) for p in np.atleast_1d(model.params))
        self.said = f"{model.dist.name}({params})"

    def _mass_inside(self, t: Transform) -> float:
        lo = float(self.model.ff(t.lower)) if np.isfinite(t.lower) else 0.0
        hi = float(self.model.sf(t.upper)) if np.isfinite(t.upper) else 0.0
        return 1.0 - lo - hi

    def check_support(self, t: Transform, name: str) -> None:
        outside = 1.0 - self._mass_inside(t)
        if outside > _NO_MASS and not self.truncate:
            raise ValueError(
                f"The prior {self.said} on {name} puts {outside:.1%} of its "
                f"mass outside the range of {name} ({_range(t)}). Use "
                "prior.around(...), a LogNormal or a Gamma for a positive "
                "parameter, a Beta for a proportion, or pass truncate=True "
                f"(prior.from_distribution(model, truncate=True)) to cut it "
                "at the range's ends."
            )

    def log_density_phi(self, phi: npt.NDArray, t: Transform) -> Any:
        theta = t.to_theta(phi)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.log(self.model.df(theta)) + t.log_jacobian(phi)
            if self.truncate:
                out = out - np.log(self._mass_inside(t))
        return np.where(np.isnan(out), -np.inf, out)

    def centre_phi(self, t: Transform) -> float | None:
        q = float(np.asarray(self.model.qf(0.5)))
        if t.lower < q < t.upper:
            return float(t.to_phi(q))
        return None

    def draw_phi(self, size: int, t: Transform, rng: Any) -> npt.NDArray:
        lo = float(self.model.ff(t.lower)) if np.isfinite(t.lower) else 0.0
        hi = 1.0 - (
            float(self.model.sf(t.upper)) if np.isfinite(t.upper) else 0.0
        )
        u = lo + (hi - lo) * rng.uniform(size=size)
        return t.to_phi(np.asarray(self.model.qf(u), dtype=float))

    def describe(self, t: Transform) -> str:
        return self.said + (" (cut at its range)" if self.truncate else "")

    def to_dict(self) -> dict:
        return {
            "prior": "DistributionPrior",
            "model": self.model.to_dict(),
            "truncate": self.truncate,
            "said": self.said,
        }


class Flat(Prior):
    """No opinion: flat on the parameter's unconstrained scale (the log
    scale for a positive parameter). Improper, so it is allowed only where
    the data alone determine the parameter.

    Examples
    --------
    >>> from surpyval import prior
    >>> prior.flat().proper
    False
    """

    proper = False

    def __init__(self, said: str = "") -> None:
        self.said = said or "no opinion"

    def log_density_phi(self, phi: npt.NDArray, t: Transform) -> Any:
        return np.zeros_like(np.asarray(phi, dtype=float))

    def draw_phi(self, size: int, t: Transform, rng: Any) -> npt.NDArray:
        raise ValueError(
            "A flat prior has no total mass, so nothing can be drawn from "
            "it: give a proper prior to see what it implies."
        )

    def describe(self, t: Transform) -> str:
        return f"flat on {t.scale_name} (the data alone identify it)"

    def to_dict(self) -> dict:
        return {"prior": "Flat", "said": self.said}


class Fixed(Prior):
    """The parameter is known: held at ``value`` (as ``fixed=`` in
    ``fit``).

    Examples
    --------
    >>> from surpyval import prior
    >>> prior.fixed(2.0).value
    2.0
    """

    is_fixed = True

    def __init__(self, value: float) -> None:
        self.value = float(value)
        if not np.isfinite(self.value):
            raise ValueError(f"fixed() needs a finite value; got {value}.")
        self.said = f"fixed at {_fmt(self.value)}"

    def log_density_phi(self, phi: npt.NDArray, t: Transform) -> Any:
        raise RuntimeError("a fixed parameter has no density")

    def check_support(self, t: Transform, name: str) -> None:
        if not (t.lower < self.value < t.upper) and not (
            t.lower <= self.value <= t.upper and t.kind == "logit"
        ):
            raise ValueError(
                f"{name} fixed at {self.value} is outside its range "
                f"({_range(t)})."
            )

    def describe(self, t: Transform) -> str:
        return self.said

    def to_dict(self) -> dict:
        return {"prior": "Fixed", "value": self.value}


# ---------------------------------------------------------------------------
# Joint priors: on several parameters at once
# ---------------------------------------------------------------------------
class JointPrior:
    """A prior on several parameters together. ``names`` are the
    parameters it covers (``None``: every free parameter of the fit).

    Examples
    --------
    >>> import surpyval as sp
    >>> belief = sp.prior.on_quantities(
    ...     sp.Weibull, B10=sp.prior.around(5000, between=(3000, 8000)),
    ...     beta=sp.prior.between(1.2, 4))
    >>> isinstance(belief, sp.prior.JointPrior)
    True
    """

    names: list[str] | None = None
    proper: bool = True
    #: A power prior multiplies the other priors (an extra likelihood)
    #: rather than covering its parameters.
    is_likelihood: bool = False

    def bind(self, fit: Any) -> Any:
        """A function of the free parameters' unconstrained values (an
        ``(N, d)`` array) giving the log density, once the fit's
        parameters are known (``fit`` is a :class:`BoundPriors`)."""
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError

    def to_dict(self) -> dict:
        raise NotImplementedError


class MultivariateNormal(JointPrior):
    """A joint normal prior on the parameters' unconstrained scale (log
    for a positive parameter, logit for a proportion): the shape of an
    earlier fit's estimates and their covariance, which
    :func:`from_model` builds.

    Parameters
    ----------
    names : list of str
        The parameters, in the order of ``mean``.
    mean : array like
        The prior mean of each on its unconstrained scale.
    cov : 2-D array like
        The prior covariance on that scale.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import prior
    >>> p = prior.MultivariateNormal(["alpha", "beta"],
    ...                              np.log([1200, 1.8]),
    ...                              [[0.01, 0.0], [0.0, 0.02]])
    >>> p.names
    ['alpha', 'beta']
    """

    def __init__(
        self,
        names: list[str],
        mean: npt.ArrayLike,
        cov: npt.ArrayLike,
        said: str = "",
        source: dict | None = None,
    ) -> None:
        self.names = list(names)
        self.mean = np.atleast_1d(np.asarray(mean, dtype=float))
        self.cov = np.atleast_2d(np.asarray(cov, dtype=float))
        k = len(self.names)
        if self.mean.shape != (k,) or self.cov.shape != (k, k):
            raise ValueError(
                f"mean needs {k} values and cov a {k} x {k} matrix, one per "
                f"name in {self.names}; got shapes {self.mean.shape} and "
                f"{self.cov.shape}."
            )
        if not np.allclose(self.cov, self.cov.T) or np.any(
            np.linalg.eigvalsh(0.5 * (self.cov + self.cov.T)) <= 0
        ):
            raise ValueError(
                "cov must be a symmetric, positive-definite covariance "
                "matrix."
            )
        self.said = said or "a joint normal prior"
        self.source = source

    def bind(self, fit: Any) -> Any:
        idx = fit.indices(self.names, "MultivariateNormal")
        dist = stats.multivariate_normal(self.mean, self.cov)

        def log_density(phi: npt.NDArray) -> Any:
            return np.atleast_1d(dist.logpdf(phi[:, idx]))

        return log_density

    def draw(self, size: int, rng: Any) -> npt.NDArray:
        return rng.multivariate_normal(self.mean, self.cov, size)

    def describe(self) -> str:
        return self.said

    def to_dict(self) -> dict:
        return {
            "prior": "MultivariateNormal",
            "names": self.names,
            "mean": self.mean.tolist(),
            "cov": self.cov.tolist(),
            "said": self.said,
            "source": self.source,
        }


class PowerPrior(JointPrior):
    """Old data as the prior: their likelihood raised to ``weight`` (a
    power prior; Ibrahim and Chen, 2000). Built by :func:`from_data`.

    It multiplies whatever other priors are given (a parameter with none
    is flat on its unconstrained scale), so with ``weight=1`` fitting the
    new data with it is the same as fitting the old and new data
    together.

    Examples
    --------
    >>> import surpyval as sp
    >>> p = sp.prior.from_data([100.0, 150, 210], c=[0, 0, 1], weight=0.5)
    >>> p.weight
    0.5
    """

    is_likelihood = True

    def __init__(
        self,
        data: Any,
        weight: float = 1.0,
        dist: Any = None,
        said: str = "",
    ) -> None:
        weight = float(weight)
        if not 0 < weight <= 1:
            raise ValueError(
                f"weight must be in (0, 1]: 1 counts the old data in full, "
                f"0.5 as half as much; got {weight}."
            )
        self.data = data
        self.weight = weight
        self.dist = dist
        self.said = said or (
            f"the old data ({_data_words(data)}) at weight {weight:g}"
        )

    def bind(self, fit: Any) -> Any:
        from surpyval.bayes.likelihood import LogLikelihood

        if self.dist is not None and canonical_name(
            self.dist
        ) != canonical_name(fit.dist):
            raise ValueError(
                f"The old data's prior was built for {self.dist.name}, but "
                f"the fit is {fit.dist.name}."
            )
        ll = LogLikelihood(
            fit.dist, self.data, offset=fit.offset, lfp=fit.lfp, zi=fit.zi
        )
        w = self.weight

        def log_density(phi: npt.NDArray) -> Any:
            return w * ll(fit.full_theta(phi))

        return log_density

    def describe(self) -> str:
        return self.said

    def to_dict(self) -> dict:
        return {
            "prior": "PowerPrior",
            "data": self.data.to_dict(),
            "weight": self.weight,
            "dist": None if self.dist is None else self.dist.name,
            "said": self.said,
        }


def _data_words(data: Any) -> str:
    n = np.asarray(data.n)
    c = np.asarray(data.c)
    failed = int(n[c != 1].sum())
    return f"{int(n.sum())} items, {failed} with failure information"


class OnQuantities(JointPrior):
    """Beliefs about quantities of the distribution (a B-life, the
    reliability at a time, the mean, the median, or a parameter by name)
    turned into a joint prior on its parameters. Built by
    :func:`on_quantities`.

    With as many statements as free parameters the change of variables
    is exact: the prior density of the quantities times the Jacobian of
    the map from the parameters to them (computed at every grid point).

    Examples
    --------
    >>> import surpyval as sp
    >>> belief = sp.prior.on_quantities(
    ...     sp.Weibull, R_at={2000: sp.prior.between(0.9, 0.99)},
    ...     beta=sp.prior.between(1.2, 4))
    >>> belief.said
    'R(2,000) 90% between 0.9 and 0.99; beta 90% between 1.2 and 4'
    """

    def __init__(
        self,
        dist: Any,
        statements: list[tuple[str, Any, Any]],
        instead_of: list[str] | None = None,
        functions: dict[str, Any] | None = None,
    ) -> None:
        self.dist = dist
        # (label, spec, prior): spec is ("q", p), ("R", t), ("mean",),
        # ("param", name), ("family", keyword, key) for a quantity the
        # fitted family names (an MTBF at a time), or ("fn", label,
        # scale) for a function of the fitted model, held in
        # ``functions`` by its label
        self.statements = statements
        #: The parameters the statements stand in place of priors on
        #: (``None``: every free parameter of the fit).
        self.instead_of = None if instead_of is None else list(instead_of)
        self.names = self.instead_of
        self.functions = dict(functions or {})
        self.said = "; ".join(
            f"{lab} {getattr(p, 'said', '') or 'as given'}"
            for lab, _, p in statements
        )

    def quantity(
        self, spec: tuple, theta: dict[str, Any], fit: Any = None
    ) -> Any:
        """The quantity ``spec`` at the parameter points ``theta``."""
        kind = spec[0]
        target = getattr(fit, "target", None)
        if kind == "param" and target is not None:
            return theta[spec[1]]
        if kind == "family":
            _, fn, _ = _family_quantity(target, spec)
            return fn(theta)
        if kind == "fn":
            fn = self.functions.get(spec[1])
            if fn is None:
                raise ValueError(
                    f"The prior on {spec[1]} was saved without its "
                    "function (a function is not saved); build the prior "
                    "again with prior.on_function to refit."
                )
            return _function_values(fn, target, fit, theta)
        if target is not None:
            raise ValueError(
                f"{spec[0]!r} statements (B-lives, R_at, mean, median, "
                "quantile) are quantities of a life distribution. For "
                f"{target.family}, state the belief with "
                "prior.on_function(lambda model: ..., belief, name=...), or "
                "one of the family's own quantities"
                + (
                    f" ({', '.join(target.quantities())})."
                    if target.quantities()
                    else "."
                )
            )
        return quantity_at(self.dist, spec, theta)

    def centre(self, fit: Any) -> npt.NDArray | None:
        """Parameters (unconstrained) at which every quantity is at its
        prior's centre, by least squares."""
        from scipy import optimize

        if self.instead_of is not None and len(self.instead_of) != fit.d:
            # the statements do not locate every free parameter
            return None
        if self.functions and any(v is None for v in self.functions.values()):
            return None
        ts = [self.transform_of(spec, fit) for _, spec, _ in self.statements]
        target = []
        for (_, _, p), t in zip(self.statements, ts):
            c = p.centre_phi(t)
            if c is None:
                return None
            target.append(c)
        target_arr = np.asarray(target)

        def resid(phi: npt.NDArray) -> npt.NDArray:
            theta = fit.theta_dict(phi[None, :])
            out = []
            for (_, spec, _), t in zip(self.statements, ts):
                with np.errstate(all="ignore"):
                    q = np.asarray(
                        self.quantity(spec, theta, fit), dtype=float
                    )
                    out.append(float(np.ravel(t.to_phi(q))[0]))
            r = np.asarray(out) - target_arr
            return np.where(np.isfinite(r), r, 1e6)

        best = None
        for start in (0.0, 1.0, -1.0, float(np.mean(target_arr))):
            x0 = np.full(fit.d, start)
            res = optimize.least_squares(resid, x0)
            cost = float(np.sum(res.fun**2))
            if best is None or cost < best[0]:
                best = (cost, res.x)
        assert best is not None
        return best[1] if best[0] < 1e-6 else None

    def transform_of(self, spec: tuple, fit: Any) -> Transform:
        kind = spec[0]
        if kind == "param":
            return fit.transforms[fit.names.index(spec[1])]
        if kind == "R":
            return Transform("logit", 0.0, 1.0)
        if kind in ("fn", "family"):
            scale = (
                spec[2]
                if kind == "fn"
                else _family_quantity(getattr(fit, "target", None), spec)[2]
            )
            if scale == "logit":
                return Transform("logit", 0.0, 1.0)
            if scale == "log":
                return Transform("log", 0.0, np.inf)
            return Transform("identity")
        lo = float(self.dist.support[0])
        if fit.offset or not np.isfinite(lo):
            return Transform("identity")
        return Transform("log", lo, np.inf)

    def bind(self, fit: Any) -> Any:
        free = fit.free_names
        if self.instead_of is None:
            if len(self.statements) != len(free):
                raise ValueError(
                    f"on_quantities has {len(self.statements)} statements "
                    f"for {len(free)} free parameters ({', '.join(free)}). "
                    "Give exactly one statement per free parameter (a "
                    "quantity, or a parameter by name, e.g. "
                    "beta=prior.between(1.2, 4)), or say which parameters "
                    "the statements stand in place of priors on "
                    "(instead_of=[...], one per statement) and give the "
                    "others their own priors."
                )
            cols_of = list(range(len(free)))
        else:
            if len(self.statements) != len(self.instead_of):
                raise ValueError(
                    f"on_quantities has {len(self.statements)} statements "
                    f"in place of {len(self.instead_of)} parameters "
                    f"({', '.join(self.instead_of)}): give one statement "
                    "for each parameter named in instead_of."
                )
            cols_of = fit.indices(self.instead_of, "on_quantities")
        for lab, spec, p in self.statements:
            if spec[0] == "param" and spec[1] not in free:
                raise ValueError(
                    f"on_quantities names {spec[1]}, which is not a free "
                    f"parameter of the fit ({', '.join(free)})."
                )
        if any(
            spec[0] == "fn" and self.functions.get(spec[1]) is None
            for _, spec, _ in self.statements
        ):
            # a saved prior without its function: it describes itself,
            # and refuses only where its density is asked for
            def unavailable(phi: npt.NDArray) -> Any:
                raise ValueError(
                    "A prior on a function of the model was saved without "
                    "the function; build it again with prior.on_function "
                    "to refit."
                )

            return unavailable
        ts = [self.transform_of(spec, fit) for _, spec, _ in self.statements]
        for (lab, spec, p), t in zip(self.statements, ts):
            p.check_support(t, lab)

        def q_phi(phi: npt.NDArray) -> npt.NDArray:
            theta = fit.theta_dict(phi)
            cols = []
            for (_, spec, _), t in zip(self.statements, ts):
                q = np.broadcast_to(
                    np.asarray(self.quantity(spec, theta, fit), dtype=float),
                    (phi.shape[0],),
                )
                cols.append(t.to_phi(q))
            return np.stack(cols, axis=1)

        def log_density(phi: npt.NDArray) -> Any:
            N = phi.shape[0]
            k = len(cols_of)
            psi = q_phi(phi)
            out = np.zeros(N)
            for j, ((_, _, p), t) in enumerate(zip(self.statements, ts)):
                out = out + p.log_density_phi(psi[:, j], t)
            # log |det d psi / d phi| by central differences, over the
            # parameters the statements stand in place of
            h = 1e-5
            J = np.empty((N, k, k))
            for i, col in enumerate(cols_of):
                step = np.zeros(phi.shape[1])
                step[col] = h
                J[:, :, i] = (q_phi(phi + step) - q_phi(phi - step)) / (2 * h)
            with np.errstate(divide="ignore", invalid="ignore"):
                sign, logdet = np.linalg.slogdet(J)
                out = out + np.where(sign != 0, logdet, -np.inf)
            return np.where(np.isfinite(out), out, -np.inf)

        return log_density

    def describe(self) -> str:
        return "on quantities: " + self.said

    def to_dict(self) -> dict:
        return {
            "prior": "OnQuantities",
            "dist": getattr(self.dist, "name", None),
            "statements": [
                [lab, list(spec), p.to_dict()]
                for lab, spec, p in self.statements
            ],
            "instead_of": self.instead_of,
        }


class FunctionStatement:
    """A belief about a function of the fitted model, for
    :func:`on_quantities`. Built by :func:`on_function`.

    Examples
    --------
    >>> import surpyval as sp
    >>> s = sp.prior.on_function(
    ...     lambda model: model.qf(0.1, [[0.0]]),
    ...     sp.prior.between(20, 40), name="B10 at Z = 0")
    >>> type(s).__name__, s.scale
    ('FunctionStatement', 'log')
    """

    def __init__(self, fn: Any, belief: Any, name: str, scale: str) -> None:
        if not callable(fn):
            raise ValueError(
                "on_function takes a function of the fitted model, e.g. "
                f"lambda model: model.mtbf(500.0); got {fn!r}."
            )
        if scale not in ("log", "logit", "identity"):
            raise ValueError(
                "scale must be 'log' (a positive quantity: a life, an "
                "MTBF), 'logit' (a probability) or 'identity' (any real "
                f"number); got {scale!r}."
            )
        self.fn = fn
        self.name = str(name)
        self.scale = scale
        self.belief = as_prior(belief, self.name)

    def __repr__(self) -> str:
        return f"FunctionStatement({self.name}: {self.belief.said})"


def on_function(
    fn: Any, belief: Any, name: str = "", scale: str = "log"
) -> FunctionStatement:
    """
    A belief about any function of the fitted model: the MTBF at the end
    of a test, a B10 life at a use stress, a remaining useful life.

    ``fn`` takes the family's ordinary model (what ``fit`` returns, at
    given parameters) and returns one number. The statement goes to
    :func:`on_quantities`, which carries the belief to the parameters by
    the change of variables.

    Parameters
    ----------
    fn : callable
        ``fn(model) -> float``.
    belief : prior
        What is believed about the function's value:
        :func:`around`, :func:`between` or a SurPyval distribution.
    name : str, optional
        What the function is, for the summary ("B10 at 50 C").
    scale : {'log', 'logit', 'identity'}, optional
        The quantity's range: positive (``'log'``, the default), a
        probability (``'logit'``) or any real number (``'identity'``).

    Examples
    --------
    >>> import surpyval as sp
    >>> s = sp.prior.on_function(
    ...     lambda model: model.qf(0.1, [[0.0]]),
    ...     sp.prior.between(20, 40), name="B10 at Z = 0")
    >>> s.name
    'B10 at Z = 0'
    """
    name = name or getattr(fn, "__name__", "a function of the model")
    return FunctionStatement(fn, belief, name, scale)


def _family_quantity(target: Any, spec: tuple) -> tuple:
    """``(label, function of the parameters' dict, scale)`` of a family's
    named quantity ``("family", keyword, key)``."""
    known = {} if target is None else target.quantities()
    if spec[1] not in known:
        family = "a life distribution" if target is None else target.family
        raise ValueError(
            f"on_quantities does not know {spec[1]!r} for {family}"
            + (f"; its quantities are {', '.join(known)}" if known else "")
            + ". State any other belief with prior.on_function."
        )
    out: tuple = known[spec[1]](spec[2])
    return out


def _function_values(
    fn: Any, target: Any, fit: Any, theta: dict[str, Any]
) -> npt.NDArray:
    """``fn`` of the family's model at each parameter point."""
    if target is None:
        raise ValueError(
            "prior.on_function is for the families whose fit_bayes builds "
            "the model at each parameter point (the regressions and the "
            "recurrent-event models). For a life distribution use the "
            "named quantities (B10=, R_at=, mean=, median=, quantile=)."
        )
    rows = np.stack([np.asarray(theta[n], dtype=float) for n in fit.names], 1)
    out = np.full(len(rows), np.nan)
    with ignoring_warnings(), np.errstate(all="ignore"):
        for i, row in enumerate(rows):
            if not np.all(np.isfinite(row)):
                continue
            try:
                out[i] = float(np.ravel(fn(target.with_params(row)))[0])
            except (ValueError, FloatingPointError, ZeroDivisionError):
                continue
    return out


class FitPowerPrior(JointPrior):
    """The data of an earlier fit, of any family, as the prior: their
    likelihood raised to ``weight`` (a power prior; Ibrahim and Chen,
    2000). Built by :func:`from_fit`.

    It multiplies whatever other priors are given, so with ``weight=1``
    fitting the new data with it is the same as fitting the old and the
    new data together.

    Examples
    --------
    >>> import numpy as np
    >>> import surpyval as sp
    >>> Z = np.random.default_rng(1).normal(size=(30, 1))
    >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
    >>> p = sp.prior.from_fit(sp.WeibullPH.fit(x, Z), weight=0.5)
    >>> type(p).__name__, p.weight
    ('FitPowerPrior', 0.5)
    """

    is_likelihood = True

    def __init__(self, target: Any, weight: float = 1.0, said: str = ""):
        weight = float(weight)
        if not 0 < weight <= 1:
            raise ValueError(
                f"weight must be in (0, 1]: 1 counts the old data in full, "
                f"0.5 as half as much; got {weight}."
            )
        self.target = target
        self.weight = weight
        self.said = said or (
            f"the data of an earlier {target.family} fit "
            f"({target.describe_data()}) at weight {weight:g}"
        )

    def bind(self, fit: Any) -> Any:
        new = getattr(fit, "target", None)
        if new is None or list(self.target.names) != list(fit.names):
            raise ValueError(
                f"The old data were fitted by {self.target.family} "
                f"(parameters {', '.join(self.target.names)}), and this "
                f"fit's parameters are {', '.join(fit.names)}: a power "
                "prior needs the same model for the old and the new data."
            )
        w = self.weight
        old = self.target

        def log_density(phi: npt.NDArray) -> Any:
            return w * old.log_lik(fit.full_theta(phi))

        return log_density

    def describe(self) -> str:
        return self.said

    def to_dict(self) -> dict:
        return {
            "prior": "FitPowerPrior",
            "target": self.target.to_dict(),
            "weight": self.weight,
            "said": self.said,
        }


def from_fit(model: Any, weight: float = 1.0) -> FitPowerPrior:
    """
    The data of an earlier fit as the prior, for any family with a
    Bayesian fit (a regression, a recurrent-event process): their exact
    likelihood raised to ``weight`` (a power prior; Ibrahim and Chen,
    2000).

    Better than :func:`from_model` when the earlier fit still holds its
    data, especially with few events, where a normal summary of them is
    poor. With ``weight=1`` it is the same as fitting the old and the new
    data together.

    Parameters
    ----------
    model : fitted model
        The earlier fit (maximum likelihood or Bayesian), holding its
        data.
    weight : float, optional
        How much of the old evidence to count, in (0, 1]. Defaults to 1.

    Examples
    --------
    >>> import numpy as np
    >>> import surpyval as sp
    >>> Z = np.random.default_rng(1).normal(size=(30, 1))
    >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
    >>> old = sp.WeibullPH.fit(x, Z)
    >>> p = sp.prior.from_fit(old, weight=0.5)
    >>> p.weight
    0.5
    """
    from surpyval.bayes.target import Target, target_for_model

    target = model if isinstance(model, Target) else None
    if target is None:
        target = getattr(model, "target", None)
    if target is None:
        target = target_for_model(model)
    if target is None or target.data is None:
        raise ValueError(
            "prior.from_fit takes an earlier fit that holds its data (a "
            "regression's or a recurrent-event process's, by fit or "
            f"fit_bayes); got {type(model).__name__}. For a life "
            "distribution's old data use prior.from_data(x, c, ...)."
        )
    return FitPowerPrior(target, weight)


def quantity_at(dist: Any, spec: tuple, theta: dict[str, Any]) -> Any:
    """A quantity of the model at parameter points ``theta`` (``{name:
    array}``): ``("q", p)`` the ``p`` quantile, ``("R", t)`` the
    reliability at ``t``, ``("F", t)`` the failure probability at ``t``
    (computed as itself, not as ``1 - R``, so it keeps its digits where it
    is small), ``("mean",)`` the mean, ``("param", name)`` a parameter;
    with the offset, limited-failure and zero-inflated forms.

    Examples
    --------
    >>> import numpy as np
    >>> import surpyval as sp
    >>> from surpyval.bayes.prior import quantity_at
    >>> theta = {"alpha": np.array([100.0]), "beta": np.array([2.0])}
    >>> quantity_at(sp.Weibull, ("q", 0.1), theta).round(3)
    array([32.459])
    """
    dp = [theta[n] for n in dist.parameter_names]
    gamma = theta.get("gamma", 0.0)
    lfp = theta.get("lfp_p", 1.0)
    f0 = theta.get("f0", 0.0)
    kind = spec[0]
    if kind == "param":
        return theta[spec[1]]
    if kind == "q":
        # F_mix(x) = f0 + (lfp - f0) F(x - gamma) = p
        u = (spec[1] - f0) / (lfp - f0)
        with np.errstate(invalid="ignore", divide="ignore"):
            inside = (u >= 0) & (u <= 1)
            q = gamma + dist.qf(np.clip(u, 0, 1), *dp)
        q = np.where(inside, q, np.nan)
        if "f0" in theta:
            # The zero-inflation mass f0 sits at 0 (not at the offset), so
            # a quantile at or below it is 0, as Parametric.qf has it.
            q = np.where((f0 > 0) & (0 <= spec[1]) & (spec[1] <= f0), 0.0, q)
        return q
    if kind == "R":
        r = 1 - lfp + (lfp - f0) * shifted(dist, "sf", spec[1] - gamma, dp)
        if "f0" in theta:
            # Before 0 nothing has failed, the mass at 0 included.
            r = np.where(np.asarray(spec[1]) < 0, 1.0, r)
        return r
    if kind == "F":
        f = f0 + (lfp - f0) * shifted(dist, "ff", spec[1] - gamma, dp)
        if "f0" in theta:
            f = np.where(np.asarray(spec[1]) < 0, 0.0, f)
        return f
    if kind == "mean":
        return gamma + lfp * _vector_mean(dist, dp)
    raise ValueError(f"Unknown quantity {spec!r}.")


def shifted(dist: Any, fname: str, xg: Any, dp: list) -> Any:
    """The base distribution's ``sf``, ``ff``, ``df`` or ``log_sf`` at
    ``xg``, the query less the offset, held to its support: below the
    support's lower edge the distribution has not started, so ``sf`` is 1
    and ``ff``, ``df`` and ``log_sf`` are 0 there (as ``Parametric``
    clamps them, #256), not the base function evaluated outside its
    support (NaN for most, more than 1 for an Exponential)."""
    with np.errstate(all="ignore"):
        v = getattr(dist, fname)(xg, *dp)
    s0 = getattr(dist, "support", (-np.inf, np.inf))[0]
    return np.where(np.asarray(xg) < s0, 1.0 if fname == "sf" else 0.0, v)


def _vector_mean(dist: Any, dp: list) -> Any:
    """The distribution's mean at many parameter points at once."""
    try:
        with np.errstate(all="ignore"):
            m = np.asarray(dist.mean(*dp), dtype=float)
        shape = np.broadcast(*[np.asarray(p) for p in dp]).shape
        if m.shape == shape:
            return m
    except (ValueError, TypeError):
        pass
    arrays = np.broadcast_arrays(*[np.asarray(p, dtype=float) for p in dp])
    flat = [a.ravel() for a in arrays]
    out = np.array(
        [float(dist.mean(*[f[i] for f in flat])) for i in range(flat[0].size)]
    )
    return out.reshape(arrays[0].shape)


# ---------------------------------------------------------------------------
# The helpers people type
# ---------------------------------------------------------------------------
def around(
    m: float, between: tuple[float, float], mass: float = 0.9
) -> NormalOnScale:
    """
    "About ``m``, almost surely between ``a`` and ``b``": a normal belief
    on the value's unconstrained scale with ``mass`` (default 90%) inside
    ``between``.

    For a positive value this is a LogNormal, for a proportion a
    logit-normal, for a value on the real line a Normal, with the
    interval's ends as its ``(1 - mass) / 2`` and ``(1 + mass) / 2``
    quantiles. Its median is then the interval's middle on that scale (the
    geometric middle for a positive value); where that is not ``m``, the
    summary says what was built, so the belief can be checked.

    Parameters
    ----------
    m : float
        The value believed most likely, as said. It must lie inside
        ``between``, or the call is refused.
    between : (float, float)
        The interval the value almost surely lies in.
    mass : float, optional
        The probability inside the interval. Defaults to 0.9.

    Returns
    -------
    NormalOnScale

    Examples
    --------
    >>> import surpyval as sp
    >>> from surpyval.bayes.transforms import Transform
    >>> b = sp.prior.around(2, between=(1.5, 3))
    >>> round(b.median(Transform.for_bounds((0, None))), 3)
    2.121
    """
    a, b = between
    built = NormalOnScale(
        a,
        b,
        mass,
        said=f"about {_fmt(m)} ({mass:.0%} between {_fmt(a)} and {_fmt(b)})",
        stated=m,
    )
    try:
        inside = bool(float(a) < float(m) < float(b))
    except (TypeError, ValueError):
        raise ValueError(f"`m` must be a number; got {m!r}.") from None
    if not inside:
        raise ValueError(
            f"prior.around(m={_fmt(m)}, between=({_fmt(a)}, {_fmt(b)})): "
            "the value believed most likely, `m`, is outside the interval "
            "`between` it is said almost surely to lie in. Widen "
            "`between` to take in `m`, correct `m`, or use "
            "prior.between(a, b) for a belief centred on the interval."
        )
    return built


def between(a: float, b: float, mass: float = 0.9) -> NormalOnScale:
    """
    "Somewhere between ``a`` and ``b``": as :func:`around`, centred on the
    interval's middle on the value's unconstrained scale, with ``mass``
    (default 90%) inside it.

    Examples
    --------
    >>> import surpyval as sp
    >>> sp.prior.between(1.2, 4).said
    '90% between 1.2 and 4'
    """
    return NormalOnScale(
        a, b, mass, said=f"{mass:.0%} between {_fmt(a)} and {_fmt(b)}"
    )


def uniform(a: float, b: float) -> DistributionPrior:
    """
    "Anything from ``a`` to ``b`` equally": a Uniform prior on the
    parameter itself.

    A uniform prior is not "no opinion": it says the value cannot lie
    outside ``(a, b)``, and it is not uniform on any other scale (a
    uniform on alpha from 0 to 1e6 says alpha is above 1,000 with
    probability 0.999). Use :func:`prior.check <check>` to see what it
    implies.

    Examples
    --------
    >>> import surpyval as sp
    >>> sp.prior.uniform(0.5, 5).said
    'Uniform(0.5, 5)'
    """
    from surpyval import Uniform

    return DistributionPrior(Uniform.from_params([float(a), float(b)]))


def flat() -> Flat:
    """
    No opinion: flat on the parameter's unconstrained scale (the log
    scale for a positive parameter, which is the standard reference
    choice for a scale and agrees with maximum likelihood in large
    samples). It is the default for a parameter given no prior, and is
    allowed only where the data alone determine the parameter; otherwise
    the fit refuses and asks for a prior.

    Examples
    --------
    >>> import surpyval as sp
    >>> sp.prior.flat().said
    'no opinion'
    """
    return Flat()


def fixed(value: float) -> Fixed:
    """
    "It is ``value``": the parameter is held at ``value``, as ``fixed=``
    does in ``fit``.

    Examples
    --------
    >>> import surpyval as sp
    >>> sp.prior.fixed(1.5).said
    'fixed at 1.5'
    """
    return Fixed(value)


def from_distribution(model: Any, truncate: bool = False) -> DistributionPrior:
    """
    A SurPyval distribution as the prior of one parameter (passing the
    model itself in ``priors`` does the same). ``truncate=True`` cuts it
    at the parameter's range, for example a Normal on a positive
    parameter.

    Examples
    --------
    >>> import surpyval as sp
    >>> p = sp.prior.from_distribution(sp.Normal.from_params([2, 1]),
    ...                                truncate=True)
    >>> p.truncate
    True
    """
    return DistributionPrior(model, truncate=truncate)


def _model_transforms(model: Any) -> tuple[list[str], list[Transform]]:
    """The names covered by a fitted model's covariance, and their
    transforms."""
    names = list(model.dist.parameter_names)
    bounds = list(model.dist.bounds)
    if getattr(model, "lfp", False):
        names.append("lfp_p")
        bounds.append((0, 1))
    if getattr(model, "zi", False):
        names.append("f0")
        bounds.append((0, 1))
    return names, [Transform.for_bounds(b) for b in bounds]


def from_model(
    model: Any,
    param: str | list[str] | None = None,
    weight: float = 1.0,
    widen: float = 1.0,
) -> Any:
    """
    An earlier fit as the prior: its estimates and how sure they were.

    For a maximum-likelihood fit (``Weibull.fit(...)``) the prior is a
    normal on the parameters' unconstrained scale (log for a positive
    parameter, logit for a proportion) with its mean at the estimates and
    the fit's covariance carried to that scale: the large-sample summary
    of the old data, so a fit from 60 failures counts for a lot and one
    from 5 for little. For an earlier Bayesian fit the prior is its own
    prior and its data, so nothing is approximated.

    Parameters
    ----------
    model : Parametric or BayesianParametric
        The earlier fit.
    param : str or list of str, optional
        Take only these parameters' prior from the model (the other
        parameters need their own priors, or are flat); by default every
        parameter the model estimated.
    weight : float, optional
        How much of the old evidence to count, in (0, 1]: 1 means the
        same population, 0.5 counts it as half as much (the covariance is
        divided by ``weight``; for a Bayesian model the old data's
        likelihood is raised to it). A judgement call: try two weights
        and compare. Defaults to 1.
    widen : float, optional
        Loosen the prior further by this factor on the spread (the
        standard deviations are multiplied by it), for a change the
        weight does not describe, such as a new site. Defaults to 1.

    Returns
    -------
    MultivariateNormal, NormalOnScale or list
        A joint prior, or for ``param="beta"`` a prior on that parameter
        alone, for the ``priors`` of ``fit_bayes``.

    Examples
    --------
    >>> import numpy as np
    >>> import surpyval as sp
    >>> np.random.seed(1)
    >>> old = sp.Weibull.fit(sp.Weibull.random(60, 1200, 1.8))
    >>> p = sp.prior.from_model(old, weight=0.5)
    >>> p.names
    ['alpha', 'beta']
    >>> b = sp.prior.from_model(old, param="beta", widen=2)
    >>> type(b).__name__
    'NormalOnScale'
    """
    weight = float(weight)
    widen = float(widen)
    if not 0 < weight <= 1:
        raise ValueError(
            f"weight must be in (0, 1]: 1 counts the old fit in full, 0.5 "
            f"as half as much; got {weight}."
        )
    if not widen >= 1:
        raise ValueError(
            f"widen must be at least 1 (it multiplies the prior's spread);"
            f" got {widen}."
        )
    from surpyval.bayes.model import BayesianParametric
    from surpyval.bayes.result import BayesianResult
    from surpyval.bayes.target import target_for_model

    if isinstance(model, (BayesianParametric, BayesianResult)):
        return model.as_prior(weight=weight, param=param, widen=widen)
    target = target_for_model(model)
    if not hasattr(model, "covariance") or not hasattr(model, "dist"):
        raise ValueError(
            "from_model takes a fitted SurPyval model (Weibull.fit(...)) "
            f"or a Bayesian fit; got {type(model).__name__}."
        )
    maximum = getattr(model, "maximum", None)
    if maximum != "verified":
        raise ValueError(
            f"The earlier fit's maximum is {maximum!r}, not 'verified', so "
            "its covariance cannot be trusted as a summary of its data. "
            "Pass the old data instead: prior.from_data(x_old, c_old, ...)."
        )
    try:
        cov = np.asarray(model.covariance(), dtype=float)
    except ValueError as e:
        raise ValueError(
            f"The earlier fit has no covariance ({e}). Give its numbers "
            "with prior.from_params(dist, params, within=...) instead."
        ) from None
    if target is not None:
        names = list(target.names)
        ts = [Transform.for_bounds(b) for b in target.bounds]
        values = [float(v) for v in target.estimates()]
        family = target.family
    else:
        names, ts = _model_transforms(model)
        values = list(np.atleast_1d(model.params))
        family = model.dist.name
    if target is None and getattr(model, "lfp", False):
        values.append(float(model.lfp_p))
    if target is None and getattr(model, "zi", False):
        values.append(float(model.f0))
    if cov.shape != (len(names), len(names)):
        raise ValueError(
            f"The earlier fit's covariance is {cov.shape[0]} x "
            f"{cov.shape[1]} for {len(names)} parameters "
            f"({', '.join(names)}), so it cannot be read as their prior. "
            "Use prior.from_fit(model) for its data, or give the numbers "
            "with prior.MultivariateNormal."
        )
    keep = [
        i
        for i in range(len(names))
        if np.isfinite(cov[i, i]) and cov[i, i] > 0
    ]
    v = np.array([values[i] for i in keep], dtype=float)
    ts_k = [ts[i] for i in keep]
    names_k = [names[i] for i in keep]
    mean = np.array([float(t.to_phi(x)) for t, x in zip(ts_k, v)])
    # d phi / d theta at the estimates
    grad = np.array(
        [float(np.exp(-t.log_jacobian(m))) for t, m in zip(ts_k, mean)]
    )
    cov_phi = cov[np.ix_(keep, keep)] * np.outer(grad, grad)
    cov_phi = cov_phi * (widen**2) / weight
    source = {
        "model": model.to_dict(),
        "weight": weight,
        "widen": widen,
        "param": param,
    }
    said = (
        f"from an earlier {family} fit "
        f"({', '.join(f'{n} {_fmt(x)}' for n, x in zip(names_k, v))}), "
        f"weight {weight:g}" + (f", widened x{widen:g}" if widen != 1 else "")
    )
    if param is None:
        return MultivariateNormal(names_k, mean, cov_phi, said, source)
    wanted = [param] if isinstance(param, str) else list(param)
    for w in wanted:
        if w not in names_k:
            raise ValueError(
                f"param={w!r} is not a parameter the earlier fit estimated "
                f"({', '.join(names_k)})."
            )
    idx = [names_k.index(w) for w in wanted]
    if len(idx) == 1:
        i = idx[0]
        return NormalOnScale(
            loc=mean[i],
            scale=float(np.sqrt(cov_phi[i, i])),
            kind=ts_k[i].kind,
            said=f"{wanted[0]} {said}",
        )
    return MultivariateNormal(
        wanted, mean[idx], cov_phi[np.ix_(idx, idx)], said, source
    )


def from_data(
    x: Any = None,
    c: Any = None,
    n: Any = None,
    t: Any = None,
    *,
    tl: Any = None,
    tr: Any = None,
    xl: Any = None,
    xr: Any = None,
    dist: Any = None,
    weight: float = 1.0,
) -> PowerPrior:
    """
    Old data as the prior: their exact likelihood raised to ``weight`` (a
    power prior; Ibrahim and Chen, 2000).

    Better than :func:`from_model` when the old data are at hand,
    especially with few old failures, where a normal summary of them is
    poor. With ``weight=1`` it is the same as fitting the old and the new
    data together.

    Parameters
    ----------
    x, c, n, t, tl, tr, xl, xr
        The old data, as for ``fit`` (``x`` may be a ``SurpyvalData``).
    dist : ParametricFitter, optional
        The distribution the old data are for; by default the one being
        fitted.
    weight : float, optional
        How much of the old evidence to count, in (0, 1]. Defaults to 1.

    Examples
    --------
    >>> import surpyval as sp
    >>> old = sp.prior.from_data([1500, 2100, 2600, 3000], c=[0, 0, 0, 1],
    ...                          weight=0.5)
    >>> shape = {"beta": sp.prior.between(1.5, 4)}
    >>> model = sp.Weibull.fit_bayes([1800, 2500, 2900], c=[0, 0, 1],
    ...                              priors=[old, shape])
    >>> model.method
    'grid'
    """
    from surpyval.bayes.fitting import bayes_data

    data = bayes_data(
        "prior.from_data", x, c=c, n=n, t=t, tl=tl, tr=tr, xl=xl, xr=xr
    )
    return PowerPrior(data, weight=weight, dist=dist)


def from_params(
    dist: Any, params: npt.ArrayLike, within: float = 0.2, mass: float = 0.9
) -> dict[str, NormalOnScale]:
    """
    A model known only by its numbers (a handbook, a data sheet) as the
    prior: independent beliefs that each parameter is within about
    ``within`` (a fraction, default 20%) of its value, with ``mass``
    (default 90%) inside that.

    For a positive parameter "within 20%" is a LogNormal with 90% between
    ``value / 1.2`` and ``value * 1.2``; for one on the real line a Normal
    with 90% within ``0.2 * |value|``.

    Returns
    -------
    dict
        ``{name: prior}``, for the ``priors`` of ``fit_bayes``.

    Examples
    --------
    >>> import surpyval as sp
    >>> p = sp.prior.from_params(sp.Weibull, [1200, 1.8], within=0.2)
    >>> sorted(p)
    ['alpha', 'beta']
    """
    within = float(within)
    if not within > 0:
        raise ValueError(f"within must be a positive fraction; got {within}.")
    values = np.atleast_1d(np.asarray(params, dtype=float))
    if values.size != dist.k:
        raise ValueError(
            f"{dist.name} has {dist.k} parameters "
            f"({', '.join(dist.parameter_names)}); got {values.size} values."
        )
    out: dict[str, NormalOnScale] = {}
    z = _z(mass)
    for name, b, v in zip(dist.parameter_names, dist.bounds, values):
        t = Transform.for_bounds(b)
        said = f"within {within:.0%} of {_fmt(v)}"
        if t.kind == "log" and t.lower == 0:
            s = np.log1p(within) / z
            out[name] = NormalOnScale(
                loc=float(np.log(v)), scale=s, kind="log", said=said
            )
        elif t.kind == "identity":
            if v == 0:
                raise ValueError(
                    f"{name} is 0, so 'within a fraction of it' says "
                    "nothing; give it a prior with prior.around(...)."
                )
            out[name] = NormalOnScale(
                loc=v, scale=within * abs(v) / z, kind="identity", said=said
            )
        else:
            lo, hi = v / (1 + within), v * (1 + within)
            out[name] = NormalOnScale(lo, hi, mass, said=said)
    return out


#: The words for the families' named quantities, in a prior's summary.
_FAMILY_LABELS = {
    "mtbf_at": "MTBF at",
    "intensity_at": "intensity at",
    "cif_at": "expected events by",
}


def _quantity_spec(key: str, value: Any, dist: Any) -> list[tuple]:
    """(label, spec, prior) entries for one keyword of on_quantities."""
    if key in ("R_at", "quantile"):
        if not isinstance(value, Mapping):
            raise ValueError(
                f"{key} takes a dict {{time or probability: prior}}; got "
                f"{type(value).__name__}."
            )
        out = []
        for k, p in value.items():
            if key == "R_at":
                out.append((f"R({_fmt(k)})", ("R", float(k)), p))
            else:
                if not 0 < float(k) < 1:
                    raise ValueError(
                        f"quantile keys are probabilities in (0, 1); got {k}."
                    )
                out.append((f"quantile {k:g}", ("q", float(k)), p))
        return out
    if key == "mean":
        return [("mean", ("mean",), value)]
    if key == "median":
        return [("median", ("q", 0.5), value)]
    if key.startswith("B") and key[1:].replace(".", "", 1).isdigit():
        p = float(key[1:]) / 100.0
        if not 0 < p < 1:
            raise ValueError(f"{key}: a B-life is between B0 and B100.")
        return [(key, ("q", p), value)]
    if key.endswith("_at") and isinstance(value, Mapping):
        # a quantity the fitted family names (an MTBF at a time)
        what = _FAMILY_LABELS.get(key, key[:-3].replace("_", " ") + " at")
        return [
            (f"{what} {_fmt(k)}", ("family", key, float(k)), p)
            for k, p in value.items()
        ]
    if not _is_life_distribution(dist):
        # a parameter of a family whose names the fit knows (a
        # coefficient, a process's beta): checked when the prior is bound
        return [(key, ("param", key), value)]
    names = list(dist.parameter_names) + ["gamma", "lfp_p", "f0"]
    if key in names:
        return [(key, ("param", key), value)]
    raise ValueError(
        f"on_quantities does not know {key!r}. It takes B-lives (B10=, "
        "B1.5=), quantile={p: ...}, R_at={t: ...}, mean=, median=, or a "
        f"parameter of {dist.name} by name ({', '.join(names)})."
    )


def _is_life_distribution(dist: Any) -> bool:
    """Whether ``dist`` is a univariate distribution (whose quantities
    ``quantity_at`` computes), not a regression or a process."""
    from surpyval.univariate.parametric.parametric_fitter import (
        ParametricFitter,
    )

    return isinstance(dist, ParametricFitter)


def on_quantities(
    dist: Any = None,
    *functions: Any,
    instead_of: list[str] | str | None = None,
    **statements: Any,
) -> OnQuantities:
    """
    A prior on the quantities people know, turned into a prior on the
    distribution's parameters.

    Say what is believed about two quantities of a two-parameter
    distribution (one per free parameter): a B-life (``B10=``), a
    reliability at a time (``R_at={2000: ...}``), the mean or median, any
    quantile (``quantile={0.5: ...}``), or a parameter by name
    (``beta=prior.between(1.2, 4)``, "it wears out"). The fit then works
    with exactly that belief: the prior density of the quantities is
    carried to the parameters by the change of variables, so nothing is
    approximated. :func:`check` shows what the prior implies.

    Each statement is a prior: :func:`around`, :func:`between` or a
    SurPyval distribution.

    For the other families with a Bayesian fit (a regression, a
    recurrent-event process) the quantities are functions of the fitted
    model, each stated with :func:`on_function` (a B10 life at a use
    stress, a remaining life), or one the family names (``mtbf_at={500:
    ...}`` for a process). There the statements usually stand in place
    of priors on some of the parameters only (``instead_of=["alpha"]``:
    one statement for each parameter named), and the other parameters
    keep their own priors.

    Parameters
    ----------
    dist : fitter
        The distribution (or regression, or process) the beliefs
        describe.
    *functions : FunctionStatement
        Beliefs about functions of the fitted model, from
        :func:`on_function`.
    instead_of : list of str, optional
        The parameters the statements stand in place of priors on, one
        for each statement. By default every free parameter of the fit.
    **statements
        The beliefs, as above.

    Examples
    --------
    >>> import surpyval as sp
    >>> belief = sp.prior.on_quantities(
    ...     sp.Weibull,
    ...     B10=sp.prior.around(5000, between=(3000, 8000)),
    ...     beta=sp.prior.between(1.2, 4),
    ... )
    >>> [label for label, _, _ in belief.statements]
    ['B10', 'beta']
    """
    if isinstance(dist, FunctionStatement):
        functions = (dist,) + functions
        dist = None
    if not statements and not functions:
        raise ValueError(
            "on_quantities needs at least one statement, e.g. "
            "B10=prior.around(5000, between=(3000, 8000))."
        )
    entries = []
    held: dict[str, Any] = {}
    for f in functions:
        if not isinstance(f, FunctionStatement):
            raise ValueError(
                "on_quantities takes the distribution, then statements "
                "from prior.on_function, then beliefs by keyword; got "
                f"{f!r}."
            )
        if f.name in held:
            raise ValueError(f"Two statements are named {f.name!r}.")
        held[f.name] = f.fn
        entries.append((f.name, ("fn", f.name, f.scale), f.belief))
    for key, value in statements.items():
        for lab, spec, p in _quantity_spec(key, value, dist):
            entries.append((lab, spec, as_prior(p, lab)))
    if isinstance(instead_of, str):
        instead_of = [instead_of]
    return OnQuantities(dist, entries, instead_of, held)


def as_prior(value: Any, name: str) -> Prior:
    """A one-parameter prior from what a user may pass: a prior, or a
    SurPyval distribution model."""
    if isinstance(value, Prior):
        return value
    if hasattr(value, "dist") and hasattr(value, "df"):
        return DistributionPrior(value)
    raise ValueError(
        f"The prior for {name} must be a prior (prior.around(...), "
        "prior.between(...), prior.flat(), prior.fixed(v)) or a SurPyval "
        "distribution such as surpyval.Gamma.from_params([2, 1]); got "
        f"{value!r}. To hold it at a value use prior.fixed(value)."
    )


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------
def prior_from_dict(d: dict) -> Any:
    """Rebuild a prior from its ``to_dict``."""
    from surpyval import from_dict as model_from_dict
    from surpyval.utils.surpyval_data import SurpyvalData

    kind = d["prior"]
    if kind == "NormalOnScale":
        return NormalOnScale(
            d.get("lower"),
            d.get("upper"),
            d.get("mass", 0.9),
            loc=d.get("loc"),
            scale=d.get("scale"),
            kind=d.get("kind"),
            said=d.get("said", ""),
            stated=d.get("stated"),
        )
    if kind == "DistributionPrior":
        return DistributionPrior(
            model_from_dict(d["model"]), truncate=d.get("truncate", False)
        )
    if kind == "Flat":
        return Flat(d.get("said", ""))
    if kind == "Fixed":
        return Fixed(d["value"])
    if kind == "MultivariateNormal":
        return MultivariateNormal(
            d["names"], d["mean"], d["cov"], d.get("said", ""), d["source"]
        )
    if kind == "PowerPrior":
        import surpyval

        dist = None if d.get("dist") is None else getattr(surpyval, d["dist"])
        return PowerPrior(
            SurpyvalData.from_dict(d["data"]),
            weight=d["weight"],
            dist=dist,
            said=d.get("said", ""),
        )
    if kind == "FitPowerPrior":
        from surpyval.bayes.target import target_from_dict

        return FitPowerPrior(
            target_from_dict(d["target"]), d["weight"], d.get("said", "")
        )
    if kind == "OnQuantities":
        import surpyval

        dist = (
            None
            if d.get("dist") is None
            else getattr(surpyval, d["dist"], None)
        )
        entries = [
            (lab, tuple(spec), prior_from_dict(p))
            for lab, spec, p in d["statements"]
        ]
        # a function is not saved: its statement describes itself, and
        # refuses where its density is asked for
        held = {spec[1]: None for _, spec, _ in entries if spec[0] == "fn"}
        return OnQuantities(dist, entries, d.get("instead_of"), held)
    if kind == "PriorSpec":
        return PriorSpec.from_dict(d)
    raise ValueError(f"Unknown prior {kind!r} in a saved prior.")


# ---------------------------------------------------------------------------
# A whole specification: what fit_bayes(priors=...) holds
# ---------------------------------------------------------------------------
class PriorSpec:
    """Every prior of a fit: one per parameter (``{name: prior}``) and any
    joint priors, as given. Plain JSON by :meth:`to_dict`.

    Examples
    --------
    >>> import surpyval as sp
    >>> spec = sp.prior.PriorSpec.parse({"beta": sp.prior.around(
    ...     2, between=(1.5, 3))})
    >>> list(spec.single)
    ['beta']
    """

    def __init__(
        self,
        single: dict[str, Prior] | None = None,
        joint: list[JointPrior] | None = None,
    ) -> None:
        self.single = dict(single or {})
        self.joint = list(joint or [])

    @classmethod
    def parse(cls, priors: Any) -> PriorSpec:
        if priors is None:
            return cls()
        if isinstance(priors, PriorSpec):
            return cls(priors.single, priors.joint)
        if isinstance(priors, JointPrior):
            return cls(joint=[priors])
        if isinstance(priors, Mapping):
            single: dict[str, Prior] = {}
            for k, v in priors.items():
                if not isinstance(k, str):
                    raise ValueError(
                        f"priors keys are parameter names; got {k!r}."
                    )
                single[k] = as_prior(v, k)
            return cls(single=single)
        if isinstance(priors, (list, tuple)):
            spec = cls()
            for part in priors:
                other = cls.parse(part)
                for k in other.single:
                    if k in spec.single:
                        raise ValueError(f"Two priors given for {k}.")
                spec.single.update(other.single)
                spec.joint += other.joint
            return spec
        raise ValueError(
            "priors must be a dict {parameter name: prior}, a joint prior "
            "(prior.from_model, prior.on_quantities, prior.from_data, "
            "prior.MultivariateNormal) or a list of these; got "
            f"{type(priors).__name__}."
        )

    def combined(self, other: PriorSpec) -> PriorSpec:
        out = PriorSpec(self.single, self.joint)
        for k in other.single:
            if k in out.single:
                raise ValueError(f"Two priors given for {k}.")
        out.single.update(other.single)
        out.joint += other.joint
        return out

    def fixed(self) -> dict[str, float]:
        return {
            k: p.value for k, p in self.single.items() if isinstance(p, Fixed)
        }

    def to_dict(self) -> dict:
        return {
            "prior": "PriorSpec",
            "single": {k: p.to_dict() for k, p in self.single.items()},
            "joint": [j.to_dict() for j in self.joint],
        }

    @classmethod
    def from_dict(cls, d: dict) -> PriorSpec:
        return cls(
            {k: prior_from_dict(v) for k, v in d.get("single", {}).items()},
            [prior_from_dict(j) for j in d.get("joint", [])],
        )


class BoundPriors:
    """A prior specification bound to a fit's parameters: which are free,
    their transforms, and the log prior density on the free parameters'
    unconstrained scale."""

    def __init__(
        self,
        spec: PriorSpec,
        dist: Any,
        names: list[str],
        bounds: list[tuple],
        offset: bool,
        lfp: bool,
        zi: bool,
        fixed: dict[str, float],
        target: Any = None,
    ) -> None:
        self.spec = spec
        #: The fitted family's target (:mod:`surpyval.bayes.target`),
        #: where the fit is not a life distribution's.
        self.target = target
        self.dist = dist
        self.names = list(names)
        self.offset, self.lfp, self.zi = offset, lfp, zi
        self.transforms = [Transform.for_bounds(b) for b in bounds]
        for k in list(spec.single) + [
            n for j in spec.joint for n in (j.names or [])
        ]:
            if k not in self.names:
                raise ValueError(
                    f"No parameter {k!r} in this fit; its parameters are "
                    f"{', '.join(self.names)}."
                )
        self.fixed = dict(fixed)
        for k, p in spec.single.items():
            if isinstance(p, Fixed):
                if k in self.fixed and self.fixed[k] != p.value:
                    raise ValueError(f"{k} is fixed twice, to two values.")
                self.fixed[k] = p.value
        for k, v in self.fixed.items():
            t = self.transforms[self.names.index(k)]
            Fixed(v).check_support(t, k)
        self.free_names = [n for n in self.names if n not in self.fixed]
        self.free_idx = [self.names.index(n) for n in self.free_names]
        self.d = len(self.free_names)
        self._factors: list[Any] = []
        covered: dict[str, str] = {}
        for k, p in spec.single.items():
            if p.is_fixed:
                continue
            if k in self.fixed:
                raise ValueError(f"{k} is fixed and also has a prior.")
            t = self.transforms[self.names.index(k)]
            p.check_support(t, k)
            covered[k] = p.describe(t)
        for j in spec.joint:
            if j.is_likelihood:
                continue
            jn = self.free_names if j.names is None else j.names
            for k in jn:
                if k in self.fixed:
                    raise ValueError(
                        f"{k} is fixed, but a joint prior covers it."
                    )
                if k in covered:
                    raise ValueError(f"Two priors given for {k}.")
                covered[k] = j.describe()
        self.flat_names = [n for n in self.free_names if n not in covered]
        #: Every free parameter with an improper prior: left out (flat)
        #: or given ``prior.flat()``.
        self.improper_names = [
            n
            for n in self.free_names
            if n in self.flat_names
            or (n in spec.single and not spec.single[n].proper)
        ]
        self.descriptions = covered
        self._bind()

    def _bind(self) -> None:
        for k, p in self.spec.single.items():
            if p.is_fixed:
                continue
            j = self.free_names.index(k)
            t = self.transforms[self.names.index(k)]
            self._factors.append(
                (lambda phi, p=p, j=j, t=t: p.log_density_phi(phi[:, j], t))
            )
        for jp in self.spec.joint:
            self._factors.append(jp.bind(self))

    @property
    def proper(self) -> bool:
        """Whether every free parameter has a proper prior."""
        if self.flat_names:
            return False
        if any(not p.proper for p in self.spec.single.values()):
            return False
        return not any(j.is_likelihood for j in self.spec.joint)

    @property
    def has_flat(self) -> bool:
        return bool(self.flat_names) or any(
            not p.proper and not p.is_fixed for p in self.spec.single.values()
        )

    def indices(self, names: list[str], who: str) -> list[int]:
        out = []
        for n in names:
            if n not in self.free_names:
                raise ValueError(
                    f"{who} covers {n}, which is not a free parameter of "
                    f"this fit ({', '.join(self.free_names)})."
                )
            out.append(self.free_names.index(n))
        return out

    def full_theta(self, phi: npt.NDArray) -> npt.NDArray:
        """Every parameter (fixed ones too) at the free ``phi`` points,
        shape ``(N, k)``."""
        phi = np.atleast_2d(phi)
        out = np.empty((phi.shape[0], len(self.names)))
        for j, i in enumerate(self.free_idx):
            out[:, i] = self.transforms[i].to_theta(phi[:, j])
        for k, v in self.fixed.items():
            out[:, self.names.index(k)] = v
        return out

    def theta_dict(self, phi: npt.NDArray) -> dict[str, npt.NDArray]:
        theta = self.full_theta(phi)
        return {n: theta[:, i] for i, n in enumerate(self.names)}

    def to_phi(self, theta: npt.NDArray) -> npt.NDArray:
        theta = np.atleast_2d(theta)
        return np.stack(
            [self.transforms[i].to_phi(theta[:, i]) for i in self.free_idx],
            axis=1,
        )

    def log_prior(self, phi: npt.NDArray) -> npt.NDArray:
        """The log prior density of the free parameters, on their
        unconstrained scale, at ``phi`` (shape ``(N, d)``); 0 for a flat
        parameter. A power prior's old likelihood is included."""
        phi = np.atleast_2d(phi)
        out = np.zeros(phi.shape[0])
        with ignoring_warnings():
            with np.errstate(all="ignore"):
                for f in self._factors:
                    out = out + np.asarray(f(phi), dtype=float)
        return np.where(np.isnan(out), -np.inf, out)

    def centre(self) -> npt.NDArray | None:
        """A start on the unconstrained scale from the priors' centres,
        or None where a parameter has none."""
        c = np.full(self.d, np.nan)
        for k, p in self.spec.single.items():
            if p.is_fixed:
                continue
            t = self.transforms[self.names.index(k)]
            v = p.centre_phi(t)
            if v is not None:
                c[self.free_names.index(k)] = v
        for j in self.spec.joint:
            if isinstance(j, MultivariateNormal):
                for n, m in zip(j.names or [], j.mean):
                    c[self.free_names.index(n)] = m
            elif isinstance(j, OnQuantities):
                found = j.centre(self)
                if found is not None:
                    c = found
            elif isinstance(j, FitPowerPrior):
                # the old data's own estimates locate the parameters
                # where the new data alone do not
                old = j.target.start()
                if old is not None and len(old) == len(self.names):
                    with np.errstate(all="ignore"):
                        at = self.to_phi(np.asarray(old)[None, :])[0]
                    c = np.where(np.isnan(c), at, c)
        return None if np.isnan(c).any() else c


def check(dist: Any, priors: Any = None, **fit_options: Any) -> Any:
    """
    What a prior says before any data: a table of the B1, B10 and B50
    lives, the mean and the median, with their 90% prior intervals.

    Run it before fitting, to catch a prior that says something nobody
    believes ("vague" priors often do: a LogNormal(0, 10) on alpha puts
    B10 anywhere from nanoseconds to geological time).

    Parameters
    ----------
    dist : ParametricFitter
        The distribution, e.g. ``surpyval.Weibull``.
    priors
        As for ``fit_bayes``. Every free parameter needs a proper prior.
    **fit_options
        ``offset``, ``lfp``, ``zi`` and ``fixed``, as for ``fit_bayes``.

    Returns
    -------
    PriorCheck
        Printable; its ``table`` is a dict of ``{quantity: (median,
        lower, upper)}`` and ``model`` the prior as a model, whose
        ``plot()`` draws what it implies.

    Examples
    --------
    >>> import surpyval as sp
    >>> belief = sp.prior.on_quantities(
    ...     sp.Weibull, B10=sp.prior.around(5000, between=(3000, 8000)),
    ...     beta=sp.prior.between(1.2, 4))
    >>> result = sp.prior.check(sp.Weibull, belief)
    >>> lo, hi = result.table["B10"][1:]
    >>> bool(2900 < lo < 3300 and 7400 < hi < 8300)
    True
    """
    from surpyval.bayes.fitting import prior_model

    return prior_model(dist, priors, **fit_options).prior_check()
