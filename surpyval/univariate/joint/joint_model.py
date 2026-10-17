"""The fitted joint model of a marker and a lifetime, :class:`JointModel`
(#891)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
from autograd import make_jvp
from scipy.stats import norm

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.univariate.information_criteria import (
    InformationCriteriaMixin,
    ic_sample_size,
)
from surpyval.utils.linalg import bound_signs, standard_errors_of
from surpyval.utils.no_maximum import maximum_entry, restored_maximum
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.validation import (
    BOUNDS,
    check_alpha_ci,
    check_option,
    no_covariance_error,
)

from ._likelihood import (
    GH_PREDICT,
    JointArrays,
    Layout,
    Model,
    PathFunction,
    _at,
    _path_params,
    adaptive_nodes,
    continued_modes,
    log_integrals,
    log_integrand,
    scatter,
)

if TYPE_CHECKING:
    import pandas as pd

    from .dynamic import DynamicPrediction

#: The functions ``cb`` bounds ('R' and 'F' are 'sf' and 'ff').
_CB_ON = ("sf", "R", "ff", "F")

#: The formula of each built-in path model, by its name, for the
#: printout; a path model of one's own may give its ``formula``.
FORMULAS = {
    "Linear": "a + b t",
    "Quadratic": "a + b t + c t^2",
    "Exponential": "a exp(b t)",
    "Power": "a t^b",
    "Logarithmic": "a + b log(t)",
    "Lloyd-Lipow": "a - b / t",
    "Gompertz": "a exp(-b exp(-c t))",
    "Offset Exponential": "a + b exp(c t)",
    "Michaelis-Menten": "a t / (b + t)",
}


def path_formula(path_model: Any) -> str:
    """The path's formula: its own ``formula`` where it has one, else the
    built-in one of its name, else its name's parameters; a path on
    links (``LinkedPathModel``) says which parameters are on a log link.
    """
    from surpyval.degradation.stress import LinkedPathModel

    if isinstance(path_model, LinkedPathModel):
        logged = [n for n, link in path_model.links.items() if link == "log"]
        formula = path_formula(path_model.base)
        return formula + "".join(f"; {n} on a log link" for n in logged)
    own = getattr(path_model, "formula", None)
    if isinstance(own, str) and own:
        return own
    return FORMULAS.get(
        path_model.name,
        "path(t; {})".format(", ".join(path_model.parameter_names)),
    )


def covariate_parts(
    path_names: list[str],
    covariates: list[str],
    path_covariates: Any = None,
    hazard_covariates: Any = None,
) -> tuple[dict[str, list[str]], list[str]]:
    """``(shifts, hazard)``: the covariates that shift each path
    parameter's mean, ``{path parameter: [covariate, ...]}``, and those
    in the hazard, each in the order of ``covariates``, from a fit's
    ``path_covariates`` and ``hazard_covariates``. ``None`` is every
    covariate; ``path_covariates`` may be a list (each path parameter) or
    a dict by path parameter (one left out has none). A name that is not
    a covariate, or a covariate in neither part, is refused."""

    def names_of(value: Any, what: str) -> list[str]:
        if isinstance(value, str):
            value = [value]
        try:
            given = [str(v) for v in value]
        except TypeError:
            raise ValueError(
                f"{what} is a list of covariate names; got {value!r}."
            ) from None
        unknown = [v for v in given if v not in covariates]
        if unknown:
            raise ValueError(
                f"{what} names {unknown!r}, which are not covariates of the "
                f"data; its covariates are {list(covariates)!r}."
            )
        return [c for c in covariates if c in given]

    if hazard_covariates is None:
        hazard = list(covariates)
    else:
        hazard = names_of(hazard_covariates, "hazard_covariates")
    if path_covariates is None:
        shifts = {name: list(covariates) for name in path_names}
    elif isinstance(path_covariates, Mapping):
        unknown = [k for k in path_covariates if k not in path_names]
        if unknown:
            raise ValueError(
                "path_covariates names the path parameters whose means the "
                f"covariates shift, of {list(path_names)!r}; got "
                f"{unknown!r}."
            )
        shifts = {
            name: names_of(
                path_covariates.get(name, []),
                f"path_covariates[{name!r}]",
            )
            for name in path_names
        }
    else:
        every = names_of(path_covariates, "path_covariates")
        shifts = {name: list(every) for name in path_names}
    used = set(hazard).union(*shifts.values())
    unused = [c for c in covariates if c not in used]
    if unused:
        raise ValueError(
            f"The covariates {unused!r} enter neither the hazard nor the "
            "path: name them in hazard_covariates or path_covariates, or "
            "leave them out of the data."
        )
    return shifts, hazard


def parameter_names(
    path_names: list[str],
    varying: list[str],
    covariates: list[str],
    shifts: dict[str, list[str]] | None = None,
    hazard: list[str] | None = None,
) -> list[str]:
    """The names of a joint model's parameters: each path parameter
    followed by its covariates' shifts of its mean (``a``, ``a_<cov>``,
    ``b``, ``b_<cov>``, ...); the spread of the varying ones (``sd_a``,
    ``sd_b``, ..., then ``corr_a_b``, ...); ``sd_noise``; the Weibull
    ``alpha`` and ``beta``; the hazard's covariate coefficients
    (``<cov>``); and ``marker_effect``. ``shifts`` and ``hazard`` (as
    :func:`covariate_parts` gives them) say which covariates enter which
    part; default every one in both."""
    if shifts is None:
        shifts = {name: list(covariates) for name in path_names}
    if hazard is None:
        hazard = list(covariates)
    out = []
    for name in path_names:
        out += [name] + [f"{name}_{cov}" for cov in shifts.get(name, [])]
    out += [f"sd_{name}" for name in varying]
    out += [
        f"corr_{a}_{b}"
        for i, a in enumerate(varying)
        for b in varying[i + 1 :]
    ]
    return out + ["sd_noise", "alpha", "beta", *hazard, "marker_effect"]


def reported_params(theta: dict, layout: Layout, refs: tuple) -> Any:
    """The reported parameters (of the estimated covariates) of the
    internal ``theta``; ``refs`` is ``(c, zbar, r)``: the marker's centre
    in the internal hazard, the covariate means, and the marker value
    at which ``alpha`` is reported. Differentiable by autograd."""
    c, zbar, r = refs
    zbar = np.asarray(zbar, dtype=float)
    P, p, q = layout.P, layout.p, layout.q
    parts = []
    for j in range(P):
        shift = theta["Delta"][j]
        parts.append(anp.reshape(theta["mu"][j] - anp.dot(shift, zbar), (1,)))
        kept = np.flatnonzero(layout.shifts[j])
        if kept.size == p:
            parts.append(anp.reshape(shift, (p,)))
        elif kept.size:
            parts.append(anp.stack([shift[k] for k in kept]))
    if q:
        L = theta["L"]
        D = anp.dot(L, anp.transpose(L))
        sd = anp.sqrt(anp.stack([D[i, i] for i in range(q)]))
        parts.append(sd)
        corr = [
            D[i, j] / (sd[i] * sd[j])
            for i in range(q)
            for j in range(i + 1, q)
        ]
        if corr:
            parts.append(anp.stack(corr))
    beta, gamma, zeta = theta["beta"], theta["gamma"], theta["zeta"]
    alpha = theta["alpha"] * anp.exp(
        (anp.dot(gamma, zbar) - zeta * (r - c)) / beta
    )
    parts.append(anp.stack([theta["sigma"], alpha, beta]))
    if layout.hazard.all():
        parts.append(anp.reshape(gamma, (p,)))
    elif layout.hazard.any():
        parts.append(
            anp.stack([gamma[k] for k in np.flatnonzero(layout.hazard)])
        )
    parts.append(anp.reshape(zeta, (1,)))
    return anp.concatenate(parts)


def internal_theta(params: Any, layout: Layout, refs: tuple) -> dict:
    """The internal parameters of the reported ``params`` (of the
    estimated covariates), the inverse of :func:`reported_params`;
    differentiable by autograd."""
    c, zbar, r = refs
    zbar = np.asarray(zbar, dtype=float)
    P, p, q = layout.P, layout.p, layout.q
    i = 0
    mus, shifts = [], []
    for j in range(P):
        n = int(layout.shifts[j].sum())
        shift = scatter(params[i + 1 : i + 1 + n], layout.shifts[j])
        mus.append(params[i] + anp.dot(shift, zbar))
        shifts.append(shift)
        i += 1 + n
    sd = params[i : i + q]
    i += q
    n_corr = q * (q - 1) // 2
    corr = params[i : i + n_corr]
    i += n_corr
    if q:
        rows = []
        k = 0
        upper = {}
        for a in range(q):
            for b in range(a + 1, q):
                upper[(a, b)] = corr[k]
                k += 1
        for a in range(q):
            row = []
            for b in range(q):
                if a == b:
                    row.append(sd[a] ** 2)
                else:
                    rho = upper[(min(a, b), max(a, b))]
                    row.append(rho * sd[a] * sd[b])
            rows.append(anp.stack(row))
        L = _cholesky(rows, q)
    else:
        L = anp.zeros((0, 0))
    sigma, alpha_r, beta = params[i], params[i + 1], params[i + 2]
    n_hazard = layout.n_hazard
    gamma = layout._gamma(params[i + 3 : i + 3 + n_hazard])
    zeta = params[i + 3 + n_hazard]
    return {
        "mu": anp.stack(mus),
        "Delta": anp.stack(shifts) if p else anp.zeros((P, 0)),
        "L": L,
        "sigma": sigma,
        "alpha": alpha_r
        * anp.exp(-(anp.dot(gamma, zbar) - zeta * (r - c)) / beta),
        "beta": beta,
        "gamma": gamma,
        "zeta": zeta,
    }


def _cholesky(rows: list, q: int) -> Any:
    """The lower Cholesky factor of the ``q x q`` matrix of ``rows``, by
    its recurrence (autograd's ``cholesky`` has no forward-mode
    derivative, which ``cb`` takes)."""
    L: list = [[0.0] * q for _ in range(q)]
    for i in range(q):
        for j in range(i + 1):
            acc = rows[i][j]
            for k in range(j):
                acc = acc - L[i][k] * L[j][k]
            L[i][j] = anp.sqrt(acc) if i == j else acc / L[j][j]
    zero = 0.0 * rows[0][0]
    return anp.stack(
        [
            anp.stack([L[i][j] if j <= i else zero for j in range(q)])
            for i in range(q)
        ]
    )


def _finite_number(name: str, value: Any) -> float:
    try:
        out = float(value) if np.ndim(value) == 0 else None
    except (TypeError, ValueError):
        out = None
    if out is None or not np.isfinite(out):
        raise ValueError(f"'{name}' must be one finite number; got {value!r}.")
    return out


class JointModel(InformationCriteriaMixin, SerialisableMixin):
    """
    A fitted joint model of a marker and a lifetime, returned by the
    :func:`~surpyval.univariate.joint.joint_ph.JointPH` fitters, such as
    ``WeibullJointPH``.

    Each item's marker follows the degradation path model (``LinearPath``
    for ``WeibullJointPH``; any ``PathModel``) with parameters of its
    own: those that vary between items (``varying``) are multivariate
    normal about a mean that the covariates shift, the others are common
    to all items. Each reading is the path plus normal noise; the hazard
    is Weibull proportional hazards in the path's current, noise-free
    value and the covariates:

    .. math::
        y_{ij} = m(t_{ij}; \\theta_i) + \\varepsilon_{ij}, \\qquad
        h_i(t) = \\frac{\\beta}{\\alpha}\\left(\\frac{t}{\\alpha}
        \\right)^{\\beta - 1} \\exp\\left(\\gamma' z_i + \\zeta
        \\left(m(t; \\theta_i) - r\\right)\\right),

    so ``alpha`` and ``beta`` are the baseline of an item with its
    marker at ``marker_reference`` (:math:`r`; the mean reading of the
    data unless the fit was given one) and its covariates at 0.

    ``params`` holds, in the order of ``parameter_names``: each path
    parameter's mean (by the path model's own names: ``a``, ``b``, ...)
    followed by its covariates' shifts (``a_<cov>``, ...); the spread of
    the varying ones between items (``sd_a``, ``sd_b``, ...,
    ``corr_a_b``, ...); ``sd_noise``; the Weibull ``alpha`` and
    ``beta``; the hazard's covariate coefficients (``<cov>``); and
    ``marker_effect`` (:math:`\\zeta`, the change in the log hazard per
    unit of the marker). An aliased covariate's entries are ``nan``
    (``aliased``). ``neg_ll()`` is the negative log-likelihood of the
    readings and the lifetimes together (so ``aic()`` and ``bic()``
    compare joint models of the same data).

    ``sf(x, Z, markers=..., given=...)`` predicts for one item: its
    covariates ``Z``, its readings so far and the age it was last seen
    running; ``sf(x, item)`` takes them from an ``ItemLifetime``.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import ItemLifetime, SurpyvalData, WeibullJointPH
    >>> rng = np.random.default_rng(3)
    >>> items = []
    >>> for k in range(60):
    ...     start, rate = rng.normal(2.0, 0.4), rng.normal(0.02, 0.005)
    ...     life = 100 * rng.weibull(2.0) * np.exp(-0.9 * (start - 2))
    ...     end = min(life, 120.0)
    ...     ages = np.arange(0.0, end, 10.0)
    ...     readings = start + rate * ages + rng.normal(0, 0.2, ages.size)
    ...     outcome = "observed" if life < 120 else "right_censored"
    ...     items.append(
    ...         ItemLifetime(k, **{outcome: end},
    ...                      markers=dict(zip(ages, readings)))
    ...     )
    >>> model = WeibullJointPH.fit(SurpyvalData.from_items(items))
    >>> model.parameter_names[:4]
    ['a', 'b', 'sd_a', 'sd_b']
    >>> model.sf([30, 40, 80], markers={0: 2.5, 30: 3.4}, given=40).shape
    (3,)
    """

    #: The baseline distribution (``Weibull``) and the marker's path model.
    dist: Any
    path_model: Any
    #: The names of the path parameters that vary between items.
    varying: list[str]
    #: The fitted parameters, in the order of ``parameter_names``.
    params: npt.NDArray
    parameter_names: list[str]
    #: The covariates' names (``coef_0``, ... where the data did not name
    #: them), and the columns whose coefficients the data cannot
    #: determine (``nan`` in ``params``).
    feature_names: list[str]
    aliased: npt.NDArray
    #: The covariates that shift each path parameter's mean, ``{path
    #: parameter: [covariate, ...]}``, and those in the hazard (every
    #: covariate in both, unless the fit was given ``path_covariates`` or
    #: ``hazard_covariates``).
    path_covariates: dict[str, list[str]] | None = None
    hazard_covariates: list[str] | None = None
    #: The marker value at which ``alpha`` is reported.
    marker_reference: float
    #: The number of estimated parameters, the ``k`` of the criteria.
    k: int
    #: What the fit reached (``surpyval.utils.no_maximum``).
    maximum: str = "unknown"
    #: The numbers of items, failures and readings fitted (count
    #: weighted).
    n_items: float = 0.0
    n_events: float = 0.0
    n_readings: float = 0.0
    #: The marker's centre in the internal hazard (the mean reading) and
    #: the covariate means: predictions are computed there.
    _centre: float = 0.0
    _zbar: npt.NDArray
    _covariance: npt.NDArray | None = None
    _neg_ll: float

    kind = "Joint (Weibull proportional hazards, current value)"

    # -- the parts -------------------------------------------------------

    @property
    def _kept(self) -> npt.NDArray:
        """The estimated covariate columns."""
        p = len(self.feature_names)
        return np.setdiff1d(np.arange(p), np.asarray(self.aliased, int))

    def _covariate_parts(self) -> tuple[dict[str, list[str]], list[str]]:
        """``(shifts, hazard)`` of every covariate (aliased ones too)."""
        names = list(self.path_model.parameter_names)
        shifts = self.path_covariates
        if shifts is None:
            shifts = {name: list(self.feature_names) for name in names}
        hazard = self.hazard_covariates
        if hazard is None:
            hazard = list(self.feature_names)
        return shifts, hazard

    @property
    def _layout(self) -> Layout:
        names = list(self.path_model.parameter_names)
        kept = [self.feature_names[k] for k in self._kept]
        shifts, hazard = self._covariate_parts()
        return Layout(
            len(names),
            len(kept),
            tuple(names.index(v) for v in self.varying),
            np.array(
                [[cov in shifts.get(n, []) for cov in kept] for n in names],
                dtype=bool,
            ).reshape(len(names), len(kept)),
            np.array([cov in hazard for cov in kept], dtype=bool),
        )

    @property
    def _estimated(self) -> npt.NDArray:
        """The entries of ``params`` that are estimated (not aliased)."""
        return ~np.isnan(np.asarray(self.params, dtype=float))

    @property
    def _refs(self) -> tuple:
        return (self._centre, self._zbar, self.marker_reference)

    def _theta(self, params: Any = None) -> dict:
        if params is None:
            params = np.asarray(self.params, dtype=float)[self._estimated]
        return internal_theta(params, self._layout, self._refs)

    def covariance(self) -> npt.NDArray:
        """The parameters' covariance, in the order of
        ``parameter_names``: the inverse of the observed information (the
        Hessian of the negative log-likelihood, by autograd) at the
        maximum, ``nan`` for an aliased coefficient. Raises a
        ``ValueError`` where the model has none."""
        if self._covariance is None:
            raise no_covariance_error()
        return np.array(self._covariance)

    def standard_errors(self) -> npt.NDArray:
        """The parameters' standard errors, the square roots of the
        diagonal of :meth:`covariance`, in the order of
        ``parameter_names``."""
        return standard_errors_of(self.covariance())

    def _ic_sample_size_from_data(self) -> float:
        return ic_sample_size(
            [0, 1], [self.n_events, self.n_items - self.n_events]
        )

    # -- the table -------------------------------------------------------

    def _parts(self) -> list[tuple[str, list[int]]]:
        """``(heading, positions in params)`` of each part."""
        names = list(self.path_model.parameter_names)
        shifts, hazard = self._covariate_parts()
        n_path = len(names) + sum(len(shifts.get(n, [])) for n in names)
        p = len(hazard)
        q = len(self.varying)
        path = list(range(n_path))
        spread = list(range(n_path, n_path + q + q * (q - 1) // 2))
        i = spread[-1] + 1 if spread else n_path
        return [
            (
                "path: {} ({})".format(
                    self.path_model.name, path_formula(self.path_model)
                ),
                path,
            ),
            ("spread between items", spread),
            ("noise", [i]),
            (
                "survival: {} PH, alpha at marker = {:.6g}".format(
                    self.dist.name, self.marker_reference
                ),
                list(range(i + 1, i + 3 + p)),
            ),
            ("link: current value", [i + 3 + p]),
        ]

    def summary(self, alpha_ci: float = 0.05) -> pd.DataFrame:
        """
        The parameter table: each parameter with its standard error, a
        two-sided ``1 - alpha_ci`` Wald interval and, for the
        coefficients, the Wald statistic ``z`` and its p-value. Indexed by
        ``(part, name)``, the parts headed by what they are: the path
        (its model and formula: its parameters' means and their
        covariates' shifts), the spread between items (``sd_<name>``,
        ``corr_<name>_<name>``), the noise (``sd_noise``), the survival
        part (the Weibull ``alpha`` -- the heading says at which marker
        value -- and ``beta``, and the covariate coefficients) and the
        link (``marker_effect``). ``exp(coef)`` is the hazard ratio of
        one unit of a covariate or of the marker. The intervals of the
        positive parameters (the sds, ``alpha``, ``beta``) are on the log
        scale and those of the correlations on Fisher's ``atanh`` scale,
        so each stays in its parameter's range.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ItemLifetime, SurpyvalData, WeibullJointPH
        >>> rng = np.random.default_rng(3)
        >>> items = []
        >>> for k in range(60):
        ...     start, rate = rng.normal(2.0, 0.4), rng.normal(0.02, 0.005)
        ...     life = 100 * rng.weibull(2.0) * np.exp(-0.9 * (start - 2))
        ...     end = min(life, 120.0)
        ...     ages = np.arange(0.0, end, 10.0)
        ...     y = start + rate * ages + rng.normal(0, 0.2, ages.size)
        ...     outcome = "observed" if life < 120 else "right_censored"
        ...     items.append(
        ...         ItemLifetime(k, **{outcome: end},
        ...                      markers=dict(zip(ages, y)))
        ...     )
        >>> model = WeibullJointPH.fit(SurpyvalData.from_items(items))
        >>> model.summary().index[0]
        ('path: Linear (a + b t)', 'a')
        """
        import pandas as pd

        from surpyval.univariate.regression._summary import (
            coefficient_table,
        )

        check_alpha_ci(alpha_ci)
        params = np.asarray(self.params, dtype=float)
        try:
            se = self.standard_errors()
        except ValueError:
            se = np.full(params.shape, np.nan)
        names = self.parameter_names
        q = norm.ppf(1 - alpha_ci / 2)
        level = f"{100 * (1 - alpha_ci):g}%"
        parts = self._parts()
        survival = parts[3][1]
        hazard = set(survival[2:]) | set(parts[4][1])
        positive = set(parts[2][1]) | set(survival[:2])
        n_sd = len(self.varying)
        positive |= set(parts[1][1][:n_sd])
        corr = set(parts[1][1][n_sd:])
        rows, index = [], []
        for heading, positions in parts:
            for j in positions:
                row = coefficient_table(
                    [names[j]], [params[j]], [se[j]], alpha_ci, exp=j in hazard
                )
                with np.errstate(all="ignore"):
                    if j in positive:
                        half = q * se[j] / params[j]
                        row["coef lower " + level] = params[j] * np.exp(-half)
                        row["coef upper " + level] = params[j] * np.exp(half)
                        row["z"] = row["p"] = np.nan
                    elif j in corr:
                        half = q * se[j] / (1 - params[j] ** 2)
                        centre = np.arctanh(params[j])
                        row["coef lower " + level] = np.tanh(centre - half)
                        row["coef upper " + level] = np.tanh(centre + half)
                        row["z"] = row["p"] = np.nan
                rows.append(row)
                index.append((heading, names[j]))
        table = pd.concat(rows)
        table.index = pd.MultiIndex.from_tuples(index, names=["part", "name"])
        return table

    def __repr__(self) -> str:
        from surpyval.univariate.regression._summary import format_table

        out = (
            "Joint SurPyval Model"
            "\n===================="
            f"\nMarker path         : {self.path_model.name}, "
            f"{path_formula(self.path_model)}; varying between items: "
            f"{', '.join(self.varying)}"
            f"\nSurvival            : {self.dist.name} proportional hazards "
            "in the marker's current value"
            f"\nData                : {self.n_items:g} items, "
            f"{self.n_events:g} failures, {self.n_readings:g} readings"
            f"\nFit                 : maximum {self.maximum}"
        )
        table = self.summary()
        columns = {
            "coef": "estimate",
            "se(coef)": "se",
            "coef lower 95%": "lower 95%",
            "coef upper 95%": "upper 95%",
        }
        for heading, positions in self._parts():
            if not positions:
                continue
            rows = table.loc[heading].rename(columns=columns)
            rows.index.name = None
            out += "\n{}\n{}".format(
                heading[0].upper() + heading[1:],
                format_table(rows, list(columns.values())),
            )
        return out

    # -- prediction ------------------------------------------------------

    def _question(
        self, Z: Any, markers: Any, given: Any
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, float]:
        """The covariates (of every column), readings and age the item
        was last seen running, from ``Z`` (or an ``ItemLifetime`` there),
        ``markers`` and ``given``, checked."""
        from surpyval.utils.item_history import covariate_row_in
        from surpyval.utils.item_lifetime import ItemLifetime

        p = len(self.feature_names)
        if isinstance(Z, ItemLifetime):
            item = Z
            if markers is not None or given is not None:
                raise ValueError(
                    f"Item {item.id!r} holds its readings and the age it "
                    "was last seen running; leave out markers and given "
                    "(or give Z, markers and given instead of the item)."
                )
            if item.outcome != "right_censored":
                raise ValueError(
                    f"Item {item.id!r} is {item.outcome}, not still "
                    "running: a prediction is for an item still running "
                    "at its right_censored age, given its readings so far."
                )
            if p == 0 and item.covariates is not None:
                raise ValueError(
                    f"Item {item.id!r} has covariates, and the model was "
                    "fitted without any; leave them out of the item."
                )
            if p and item.covariates is None:
                raise ValueError(
                    f"Item {item.id!r} has no covariates; the model was "
                    f"fitted on {self.feature_names!r}."
                )
            Z = None
            if p:
                have = item.covariate_names
                if have is not None:
                    if set(have) != set(self.feature_names):
                        raise ValueError(
                            f"Item {item.id!r} has covariates "
                            f"{list(have)!r}; the model was fitted on "
                            f"{self.feature_names!r}."
                        )
                    Z = dict(
                        zip(
                            self.feature_names,
                            covariate_row_in(item, self.feature_names),
                        )
                    )
                else:
                    Z = np.asarray(item.covariates, dtype=float)
            markers, given = item.markers, item.value
        z = self._covariate_row(Z)
        ages, values = self._readings(markers)
        if given is None:
            if ages.size:
                raise ValueError(
                    "Give 'given', the age the item was last seen running "
                    "(at or after its last reading, "
                    f"{ages.max():g}): the readings alone do not say it."
                )
            given_age = 0.0
        else:
            given_age = _finite_number("given", given)
            if given_age < 0:
                raise ValueError(
                    f"'given' is an age, 0 or more; got {given_age:g}."
                )
            late = ages[ages > given_age]
            if late.size:
                raise ValueError(
                    f"A reading at age {late.max():g} is after 'given', "
                    f"{given_age:g}, the age the item was last seen "
                    f"running: give given={late.max():g} or later."
                )
        if ages.size and np.all(np.isfinite(z)):
            self.path_model.check_data(ages, values)
        return z, ages, values, given_age

    def _covariate_row(self, Z: Any) -> npt.NDArray:
        """``Z`` as the row of every covariate column (a named mapping is
        read by name)."""
        names = self.feature_names
        p = len(names)
        if Z is None:
            if p:
                raise ValueError(
                    f"The model was fitted on covariates {names!r}: give "
                    "the item's, Z={...} by name or a row of "
                    f"{p} value(s)."
                )
            return np.zeros(0)
        if (
            isinstance(Z, Mapping)
            or hasattr(Z, "index")
            and hasattr(Z, "to_dict")
        ):
            given = dict(Z.items())
            if set(given) != set(names):
                raise ValueError(
                    f"Z names {sorted(given)!r}; the model was fitted on "
                    f"{names!r}."
                )
            return np.array([float(given[n]) for n in names])
        row = np.asarray(Z, dtype=float)
        if row.ndim == 2 and row.shape[0] == 1:
            row = row[0]
        if p == 0 and row.size == 0:
            return np.zeros(0)
        if row.ndim != 1 or row.size != p:
            raise ValueError(
                f"Z is one item's covariates, a row of {p} value(s) "
                f"({names!r}); got shape {np.shape(Z)}."
                + (
                    " The model was fitted without covariates: leave Z out."
                    if p == 0
                    else ""
                )
            )
        return row

    def _readings(self, markers: Any) -> tuple[npt.NDArray, npt.NDArray]:
        if markers is None:
            return np.zeros(0), np.zeros(0)
        if not isinstance(markers, Mapping):
            raise ValueError(
                "'markers' are the readings so far, a dict {age: value}."
            )
        ages, values = [], []
        for age, value in markers.items():
            age = _finite_number("markers' ages", age)
            if age < 0:
                raise ValueError(
                    f"A reading's age is 0 or more; got {age:g} in markers."
                )
            ages.append(age)
            values.append(_finite_number(f"the reading at age {age:g}", value))
        order = np.argsort(ages, kind="stable")
        return np.array(ages)[order], np.array(values)[order]

    def _conditional(
        self, x: npt.NDArray, z: npt.NDArray, ages: Any, values: Any, u: float
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """``(log_sf, log_hf)`` of one item at the ages ``x`` (each after
        ``u``; ``z`` its covariates, checked, ``ages`` and ``values`` its
        readings), at the fitted parameters: the log of its survival to
        each from ``u``, and of its hazard there given its readings and
        that it reached that age,

            h(x | y) = int p(y | theta) h(x | theta) S(x | theta) phi
                       / int p(y | theta) S(x | theta) phi,

        both integrals on the nodes of the second (centred at the mode of
        its integrand), and the survival as :meth:`sf` computes it."""
        kept = self._kept
        zc = (z[kept] - self._zbar).reshape(1, -1)
        times = np.concatenate([[u], np.asarray(x, dtype=float)])
        M = times.size
        many = JointArrays.build(
            np.ones(M),
            np.maximum(times, 1e-300),
            np.zeros(M),
            np.zeros(M),
            np.repeat(zc, M, axis=0),
            [ages] * M,
            [values] * M,
        )
        model = Model(
            self._layout, PathFunction(self.path_model), self._centre
        )
        theta = self._theta()
        with np.errstate(all="ignore"):
            mode, root = continued_modes(
                theta, model.layout, model.path, many, times, self._centre
            )
            nodes = adaptive_nodes(mode, root, GH_PREDICT[model.layout.q])
            log_I = log_integrals(
                theta, model, many, nodes, times, np.zeros(M), readings=True
            )
            log_J = log_integrals(
                theta, model, many, nodes, times, np.ones(M), readings=True
            )
        return log_I[1:] - log_I[0], (log_J - log_I)[1:]

    def _marker_moments(
        self, x: npt.NDArray, z: npt.NDArray, ages: Any, values: Any, u: float
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """The mean and standard deviation of one item's noise-free
        marker at the ages ``x``, over its path parameters' distribution
        given its readings and that it reached ``u`` (adaptive
        Gauss-Hermite quadrature, at the fitted parameters)."""
        kept = self._kept
        zc = (z[kept] - self._zbar).reshape(1, -1)
        one = JointArrays.build(
            np.ones(1),
            np.array([max(u, 1e-300)]),
            np.zeros(1),
            np.zeros(1),
            zc,
            [ages],
            [values],
        )
        model = Model(
            self._layout, PathFunction(self.path_model), self._centre
        )
        theta = self._theta()
        t = np.array([u])
        with np.errstate(all="ignore"):
            mode, root = continued_modes(
                theta, model.layout, model.path, one, t, self._centre
            )
            nodes = adaptive_nodes(mode, root, GH_PREDICT[model.layout.q])
            log_g = np.asarray(
                log_integrand(
                    theta, model, one, nodes, t, np.zeros(1), readings=True
                ),
                dtype=float,
            )[0]
            weights = np.exp(log_g - np.max(log_g))
            weights = weights / weights.sum()
            params = np.asarray(
                _path_params(theta, model.layout, one.Z, nodes[0]),
                dtype=float,
            )
            x = np.asarray(x, dtype=float)
            m = _at(model.path, params, x[None, :])[0]  # (K, M)
            m = np.asarray(m, dtype=float)
        mean = weights @ m
        var = weights @ (m - mean) ** 2
        return mean, np.sqrt(np.maximum(var, 0.0))

    def predict(
        self,
        Z: Any = None,
        markers: dict | None = None,
        given: float | None = None,
    ) -> DynamicPrediction:
        """
        The dynamic prediction of one item: its failure-age distribution
        given its covariates ``Z``, its readings so far ``markers`` and
        that it was still running at age ``given`` (an ``ItemLifetime``
        still running, as ``Z``, stands for all three), as an object with
        the conditional ``sf``, ``ff``, ``hf``, ``df`` and ``Hf``, ``qf``
        and the remaining life's quantiles, a Wald band from the
        parameters' uncertainty (``cb``), the item's smoothed and
        projected marker (``marker``, ``marker_interval``) and draws
        (``random``). Each new reading, given to ``predict`` again,
        updates it. :meth:`sf`, :meth:`ff` and :meth:`cb` give the same
        numbers for an array of ages.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ItemLifetime, SurpyvalData, WeibullJointPH
        >>> rng = np.random.default_rng(3)
        >>> items = []
        >>> for k in range(60):
        ...     start, rate = rng.normal(2.0, 0.4), rng.normal(0.02, 0.005)
        ...     life = 100 * rng.weibull(2.0) * np.exp(-0.9 * (start - 2))
        ...     end = min(life, 120.0)
        ...     ages = np.arange(0.0, end, 10.0)
        ...     y = start + rate * ages + rng.normal(0, 0.2, ages.size)
        ...     outcome = "observed" if life < 120 else "right_censored"
        ...     items.append(
        ...         ItemLifetime(k, **{outcome: end},
        ...                      markers=dict(zip(ages, y)))
        ...     )
        >>> model = WeibullJointPH.fit(SurpyvalData.from_items(items))
        >>> hot = model.predict(markers={0: 2.4, 20: 2.9, 40: 3.4}, given=40)
        >>> cool = model.predict(markers={0: 1.8, 20: 1.9, 40: 2.0}, given=40)
        >>> bool(hot.remaining_life(0.5) < cool.remaining_life(0.5))
        True
        >>> bool(hot.sf(60) == model.sf(60, markers=hot.markers, given=40))
        True
        """
        from .dynamic import DynamicPrediction

        return DynamicPrediction(self, Z, markers, given)

    def _log_sf_function(
        self, x: npt.NDArray, Z: Any, markers: Any, given: Any
    ) -> Any:
        """``(f, after, missing)``: ``f(params)``, the log of the item's
        survival to each age of ``x`` after ``given`` (``u``), as a
        function of the estimated parameters; the mask of those ages; and
        whether the item's covariates are missing (``nan``).

        The survival is ``I(x) / I(u)``, ``I(t) = int p(y | theta) S(t |
        theta) phi(theta_V) d theta_V``: each integral by adaptive
        Gauss-Hermite quadrature on nodes centred at the mode of its own
        integrand, placed at the fitted parameters and held while
        differentiating."""
        z, ages, values, u = self._question(Z, markers, given)
        after = np.isfinite(x) & (x > u)
        if not np.all(np.isfinite(z)):
            return None, after, True
        kept = self._kept
        zc = (z[kept] - self._zbar).reshape(1, -1)
        times = np.concatenate([[u], x[after]])
        M = times.size
        many = JointArrays.build(
            np.ones(M),
            np.maximum(times, 1e-300),
            np.zeros(M),
            np.zeros(M),
            np.repeat(zc, M, axis=0),
            [ages] * M,
            [values] * M,
        )
        model = Model(
            self._layout, PathFunction(self.path_model), self._centre
        )
        theta = self._theta()
        mode, root = continued_modes(
            theta, model.layout, model.path, many, times, self._centre
        )
        nodes = adaptive_nodes(mode, root, GH_PREDICT[model.layout.q])
        zero = np.zeros(M)

        def f(params: Any) -> Any:
            th = internal_theta(params, model.layout, self._refs)
            log_I = log_integrals(
                th, model, many, nodes, times, zero, readings=True
            )
            return log_I[1:] - log_I[0]

        return f, after, False

    @keeps_query_shape
    def sf(
        self,
        x: npt.ArrayLike,
        Z: Any = None,
        markers: dict | None = None,
        given: float | None = None,
    ) -> npt.NDArray:
        """
        The survival of one item to each age ``x``, given its covariates
        ``Z``, its readings so far ``markers`` and that it was still
        running at age ``given``:

        .. math::
            P(T > x \\mid T > u, y) =
            \\frac{\\int S(x \\mid \\theta) p(y \\mid \\theta)
            \\phi(\\theta) d\\theta}
            {\\int S(u \\mid \\theta) p(y \\mid \\theta) \\phi(\\theta)
            d\\theta},

        over its path's varying parameters, each integral by adaptive
        Gauss-Hermite quadrature (centred at the mode of its integrand).
        With no readings and no ``given`` it is a new item's survival,
        over the population's paths. An age at or before ``given`` gives
        1; ``nan`` gives ``nan``, and so does a missing covariate. The
        result has the shape of ``x``.

        Parameters
        ----------
        x : array_like
            The ages.
        Z : dict, array_like or ItemLifetime, optional
            The item's covariates: ``{name: value}`` or a row in the
            order of ``feature_names`` (needed where the model has
            covariates). An ``ItemLifetime`` (still running) stands for
            all three arguments: its covariates, its ``markers`` and its
            right censored age as ``given``.
        markers : dict, optional
            The readings so far, ``{age: value}``, at or before ``given``.
        given : float, optional
            The age the item was last seen running (needed with
            readings).

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ItemLifetime, SurpyvalData, WeibullJointPH
        >>> rng = np.random.default_rng(3)
        >>> items = []
        >>> for k in range(60):
        ...     start, rate = rng.normal(2.0, 0.4), rng.normal(0.02, 0.005)
        ...     life = 100 * rng.weibull(2.0) * np.exp(-0.9 * (start - 2))
        ...     end = min(life, 120.0)
        ...     ages = np.arange(0.0, end, 10.0)
        ...     y = start + rate * ages + rng.normal(0, 0.2, ages.size)
        ...     outcome = "observed" if life < 120 else "right_censored"
        ...     items.append(
        ...         ItemLifetime(k, **{outcome: end},
        ...                      markers=dict(zip(ages, y)))
        ...     )
        >>> model = WeibullJointPH.fit(SurpyvalData.from_items(items))
        >>> float(model.sf(40, markers={0: 2.0}, given=40))
        1.0
        >>> item = ItemLifetime("P", right_censored=40, markers={0: 2.0})
        >>> same = model.sf(60, markers={0: 2.0}, given=40)
        >>> bool(model.sf(60, item) == same)
        True
        """
        x = np.asarray(x, dtype=float)
        f, after, missing = self._log_sf_function(x, Z, markers, given)
        out = np.where(np.isnan(x), np.nan, 1.0)
        if missing:
            return np.full(x.shape, np.nan)
        if after.any():
            # far beyond the data the survival underflows to 0
            with np.errstate(all="ignore"):
                params = np.asarray(self.params, dtype=float)
                out[after] = np.exp(f(params[self._estimated]))
        return out

    @keeps_query_shape
    def ff(
        self,
        x: npt.ArrayLike,
        Z: Any = None,
        markers: dict | None = None,
        given: float | None = None,
    ) -> npt.NDArray:
        """
        The probability that one item fails by each age ``x``, given its
        covariates, readings and the age it was last seen running (as
        for :meth:`sf`): ``1 - sf(x, Z, markers, given)``.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ItemLifetime, SurpyvalData, WeibullJointPH
        >>> rng = np.random.default_rng(3)
        >>> items = []
        >>> for k in range(60):
        ...     start, rate = rng.normal(2.0, 0.4), rng.normal(0.02, 0.005)
        ...     life = 100 * rng.weibull(2.0) * np.exp(-0.9 * (start - 2))
        ...     end = min(life, 120.0)
        ...     ages = np.arange(0.0, end, 10.0)
        ...     y = start + rate * ages + rng.normal(0, 0.2, ages.size)
        ...     outcome = "observed" if life < 120 else "right_censored"
        ...     items.append(
        ...         ItemLifetime(k, **{outcome: end},
        ...                      markers=dict(zip(ages, y)))
        ...     )
        >>> model = WeibullJointPH.fit(SurpyvalData.from_items(items))
        >>> float(model.ff(20, markers={0: 2.0}, given=40))
        0.0
        """
        return 1.0 - self.sf(x, Z, markers, given)

    @keeps_query_shape
    def cb(
        self,
        x: npt.ArrayLike,
        Z: Any = None,
        markers: dict | None = None,
        given: float | None = None,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """
        Confidence bounds on one item's survival (``on="sf"``, the
        default) or failure probability (``on="ff"``) to each age ``x``,
        given its covariates, readings and the age it was last seen
        running (as for :meth:`sf`), from the uncertainty of the
        parameters: the delta method, with the parameters' covariance, on
        ``log(-log sf)`` (so the bounds stay in (0, 1)), the adaptive
        nodes held at the fitted parameters while differentiating. An age
        at or before ``given`` gives ``[1, 1]`` (``[0, 0]`` on ``ff``).

        Parameters
        ----------
        x : array_like
            The ages.
        Z, markers, given : optional
            The item, as for :meth:`sf` (an ``ItemLifetime`` as ``Z``
            stands for all three).
        on : str, optional
            ``"sf"`` (or ``"R"``) or ``"ff"`` (or ``"F"``).
        alpha_ci : float, optional
            The total tail probability: 0.05 (the default) gives 95%
            bounds.
        bound : str, optional
            ``"two-sided"`` (``[lower, upper]`` on a last axis),
            ``"lower"`` or ``"upper"``.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ItemLifetime, SurpyvalData, WeibullJointPH
        >>> rng = np.random.default_rng(3)
        >>> items = []
        >>> for k in range(60):
        ...     start, rate = rng.normal(2.0, 0.4), rng.normal(0.02, 0.005)
        ...     life = 100 * rng.weibull(2.0) * np.exp(-0.9 * (start - 2))
        ...     end = min(life, 120.0)
        ...     ages = np.arange(0.0, end, 10.0)
        ...     y = start + rate * ages + rng.normal(0, 0.2, ages.size)
        ...     outcome = "observed" if life < 120 else "right_censored"
        ...     items.append(
        ...         ItemLifetime(k, **{outcome: end},
        ...                      markers=dict(zip(ages, y)))
        ...     )
        >>> model = WeibullJointPH.fit(SurpyvalData.from_items(items))
        >>> model.cb([20, 60], markers={0: 2.0}, given=40).shape
        (2, 2)
        """
        check_alpha_ci(alpha_ci)
        check_option("on", on, _CB_ON)
        check_option("bound", bound, BOUNDS)
        cov = self.covariance()
        x = np.asarray(x, dtype=float)
        f, after, missing = self._log_sf_function(x, Z, markers, given)
        tail, signs = bound_signs(alpha_ci, bound)
        if on in ("ff", "F"):
            # a lower bound on ff is an upper one on sf
            signs = -signs
        q = norm.ppf(1 - tail)
        out = np.ones(x.shape + (signs.size,))
        out[np.isnan(x)] = np.nan
        if missing:
            out[:] = np.nan
        elif after.any():
            est = self._estimated
            params = np.asarray(self.params, dtype=float)[est]
            sub = cov[np.ix_(est, est)]

            def log_H(v: Any) -> Any:
                return anp.log(-f(v))

            # where the survival underflows to 0, the bound is nan
            with np.errstate(all="ignore"):
                jvp = make_jvp(log_H)(params)
                columns = [jvp(e)[1] for e in np.eye(params.size)]
                centre = log_H(params)
                J = np.stack(columns, axis=-1)
                se = np.sqrt(np.einsum("mi,ij,mj->m", J, sub, J))
                # an upper bound on log H is a lower bound on sf
                out[after] = np.exp(
                    -np.exp(centre[:, None] - signs[None, :] * q * se[:, None])
                )
        if on in ("ff", "F"):
            out = 1.0 - out
        return out if bound == "two-sided" else out[..., 0]

    # -- saving and loading ----------------------------------------------

    def to_dict(self) -> dict:
        """
        The model as a plain, JSON-serialisable ``dict``: its path model
        (by its built-in name) and which of its parameters vary, its
        parameters and their names, the covariates' names and the aliased
        ones, the marker reference, the centres its predictions are
        computed at, the covariance, the fitted negative log-likelihood
        and the counts of the data (not the data). :meth:`from_dict`
        rebuilds it, and it predicts and bounds as the original. A built-in
        path on links (``LinkedPathModel``) is saved as its name and its
        links. A path model of one's own cannot be rebuilt from a name,
        and is refused (``NotImplementedError``); the model pickles all
        the same.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval
        >>> from surpyval import ItemLifetime, SurpyvalData, WeibullJointPH
        >>> rng = np.random.default_rng(3)
        >>> items = []
        >>> for k in range(60):
        ...     start, rate = rng.normal(2.0, 0.4), rng.normal(0.02, 0.005)
        ...     life = 100 * rng.weibull(2.0) * np.exp(-0.9 * (start - 2))
        ...     end = min(life, 120.0)
        ...     ages = np.arange(0.0, end, 10.0)
        ...     y = start + rate * ages + rng.normal(0, 0.2, ages.size)
        ...     outcome = "observed" if life < 120 else "right_censored"
        ...     items.append(
        ...         ItemLifetime(k, **{outcome: end},
        ...                      markers=dict(zip(ages, y)))
        ...     )
        >>> model = WeibullJointPH.fit(SurpyvalData.from_items(items))
        >>> restored = surpyval.from_dict(model.to_dict())
        >>> bool(np.allclose(restored.params, model.params))
        True
        """
        from surpyval.degradation.path_models import (
            PATH_MODELS,
            path_model_key,
        )
        from surpyval.degradation.stress import LinkedPathModel

        path, links = self.path_model, None
        if isinstance(path, LinkedPathModel):
            links = dict(path.links)
            path = path.base
        key = path_model_key(path)
        # by its class: an unpickled built-in path is a copy of it
        if type(PATH_MODELS.get(key)) is not type(path):
            raise NotImplementedError(
                f"The {path.name!r} path model is not a built-in "
                "one, so it cannot be rebuilt from a name: keep the model "
                "with pickle."
            )
        out: dict[str, Any] = {
            "model": "JointModel",
            "distribution": self.dist.name,
            "path_model": key,
            "links": links,
            "varying": list(self.varying),
            "params": np.asarray(self.params, dtype=float).tolist(),
            "parameter_names": list(self.parameter_names),
            "feature_names": list(self.feature_names),
            "aliased": np.asarray(self.aliased, dtype=int).tolist(),
            "path_covariates": (
                None
                if self.path_covariates is None
                else {k: list(v) for k, v in self.path_covariates.items()}
            ),
            "hazard_covariates": (
                None
                if self.hazard_covariates is None
                else list(self.hazard_covariates)
            ),
            "marker_reference": float(self.marker_reference),
            "centre": float(self._centre),
            "covariate_means": np.asarray(self._zbar, dtype=float).tolist(),
            "n_items": float(self.n_items),
            "n_events": float(self.n_events),
            "n_readings": float(self.n_readings),
            "_neg_ll": float(self._neg_ll),
            **maximum_entry(self.maximum),
        }
        if self._covariance is not None:
            out["covariance"] = np.asarray(self._covariance).tolist()
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> JointModel:
        """
        Rebuild a model from a :meth:`to_dict` dictionary.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ItemLifetime, SurpyvalData, WeibullJointPH
        >>> from surpyval.univariate.joint import JointModel
        >>> rng = np.random.default_rng(3)
        >>> items = []
        >>> for k in range(60):
        ...     start, rate = rng.normal(2.0, 0.4), rng.normal(0.02, 0.005)
        ...     life = 100 * rng.weibull(2.0) * np.exp(-0.9 * (start - 2))
        ...     end = min(life, 120.0)
        ...     ages = np.arange(0.0, end, 10.0)
        ...     y = start + rate * ages + rng.normal(0, 0.2, ages.size)
        ...     outcome = "observed" if life < 120 else "right_censored"
        ...     items.append(
        ...         ItemLifetime(k, **{outcome: end},
        ...                      markers=dict(zip(ages, y)))
        ...     )
        >>> model = WeibullJointPH.fit(SurpyvalData.from_items(items))
        >>> restored = JointModel.from_dict(model.to_dict())
        >>> restored.parameter_names == model.parameter_names
        True
        """
        import surpyval
        from surpyval.degradation.path_models import PATH_MODELS

        require_model_tag(model_dict, "JointModel", "a joint model")
        if model_dict["distribution"] != "Weibull":
            raise ValueError(
                "A joint model has a Weibull baseline; the dict names "
                f"{model_dict['distribution']!r}."
            )
        key = model_dict["path_model"]
        if key not in PATH_MODELS:
            raise ValueError(
                f"Unknown path model {key!r}; the built-in ones are "
                f"{sorted(PATH_MODELS)}."
            )
        out = cls()
        out.dist = surpyval.Weibull
        out.path_model = PATH_MODELS[key]
        if model_dict.get("links"):
            from surpyval.degradation.stress import (
                LinkedPathModel,
                validate_links,
            )

            out.path_model = LinkedPathModel(
                out.path_model,
                validate_links(out.path_model, dict(model_dict["links"])),
            )
        out.varying = list(model_dict["varying"])
        out.params = np.array(model_dict["params"], dtype=float)
        out.parameter_names = list(model_dict["parameter_names"])
        out.feature_names = list(model_dict["feature_names"])
        out.aliased = np.array(model_dict.get("aliased", []), dtype=int)
        # saved before the parts' covariates could differ: every one in both
        shifts = model_dict.get("path_covariates")
        out.path_covariates = (
            None if shifts is None else {k: list(v) for k, v in shifts.items()}
        )
        hazard = model_dict.get("hazard_covariates")
        out.hazard_covariates = None if hazard is None else list(hazard)
        out.marker_reference = float(model_dict["marker_reference"])
        out._centre = float(model_dict["centre"])
        out._zbar = np.array(model_dict["covariate_means"], dtype=float)
        out.n_items = float(model_dict.get("n_items", 0.0))
        out.n_events = float(model_dict.get("n_events", 0.0))
        out.n_readings = float(model_dict.get("n_readings", 0.0))
        out._neg_ll = float(model_dict["_neg_ll"])
        out.k = int(np.sum(~np.isnan(out.params)))
        out.maximum = restored_maximum(model_dict)
        if "covariance" in model_dict:
            out._covariance = np.array(model_dict["covariance"], dtype=float)
        if len(out.parameter_names) != len(out.params):
            raise ValueError(
                "The dict's 'parameter_names' and 'params' differ in length."
            )
        return out
