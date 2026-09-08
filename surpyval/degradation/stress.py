"""Stress-dependent path parameters for accelerated degradation tests.

Stage 1 of accelerated degradation testing (``Z`` alone in
:meth:`DegradationAnalysis.fit`) lets stress act only on the pseudo
failure times, through the life regression. Stage 2 (``links``) models
the degradation *mechanism*: the path parameters themselves depend on
the stress a unit was tested at, with a random effect per unit on top.
Per unit ``i`` with stress row ``z_i``,

    eta_i = D(z_i) gamma + u_i,    u_i ~ MVN(0, Sigma),
    theta_i = h(eta_i),

where ``theta_i`` are the path parameters on their natural scale,
``eta_i`` the same parameters on a *link* scale (identity, or log for a
parameter that must stay positive and whose stress effect is
multiplicative -- a ``log`` link on a rate with ``Z = 1/T`` is the
Arrhenius relationship), ``D(z)`` the fixed-effects design that gives
every stress-dependent parameter the row ``[1, z']`` and every other
parameter an intercept only, and ``gamma`` the fixed effects.

Everything downstream works on the link scale: :class:`LinkedPathModel`
presents any path model as a path model in ``eta`` (so the existing
per-unit fits, the two-stage moments and the REML machinery apply
unchanged), and :func:`stress_design` builds ``D(z)``. Because the
population model is on the link scale, the random effect of a
``log``-linked parameter is log-normal on the natural scale.
"""

from typing import Any, Callable

import numpy as np
import numpy.typing as npt

from .path_models import PathModel

#: ``link name -> (h, h^-1, h')``: the natural-scale parameter as a
#: function of the link-scale one, its inverse, and its derivative.
_LINKS: dict[str, tuple[Callable, Callable, Callable]] = {
    "identity": (lambda e: e, lambda t: t, lambda e: np.ones_like(e)),
    "log": (np.exp, np.log, np.exp),
}

LINK_NAMES = tuple(_LINKS)


def validate_links(path_model: PathModel, links: Any) -> dict[str, str]:
    """
    Check a user ``links`` mapping against a path model.

    ``links`` names the *stress-dependent* path parameters and the link
    each is modelled on (``"identity"`` or ``"log"``). Returns the
    mapping in path-parameter order.
    """
    if not isinstance(links, dict) or len(links) == 0:
        raise ValueError(
            "links must be a non-empty dict mapping path parameter names "
            "to a link ('identity' or 'log'), e.g. {'b': 'log'}"
        )
    unknown = [name for name in links if name not in path_model.param_names]
    if unknown:
        raise ValueError(
            "links names parameter(s) {} that the {} path model does not "
            "have; its parameters are {}".format(
                unknown, path_model.name, path_model.param_names
            )
        )
    bad = {name: link for name, link in links.items() if link not in _LINKS}
    if bad:
        raise ValueError(
            "Unknown link(s) {}; each link must be one of {}".format(
                bad, list(LINK_NAMES)
            )
        )
    return {
        name: str(links[name])
        for name in path_model.param_names
        if name in links
    }


class LinkedPathModel(PathModel):
    """
    A path model reparameterised onto a link scale.

    Wraps ``base`` so that its parameters ``theta`` are replaced by
    ``eta`` with ``theta = h(eta)`` elementwise: ``h`` is the identity
    for most parameters and ``exp`` for those given a ``"log"`` link.
    ``path``, ``inv_path``, ``jacobian`` and ``fit`` all take and return
    link-scale parameters, so the wrapped model is a drop-in path model
    for the per-unit fits and the population (REML) machinery, which
    then estimate the population of ``eta``.

    Parameters
    ----------
    base : PathModel
        The path model on its natural scale.
    links : dict
        ``{parameter name: "identity" | "log"}`` for the parameters
        whose link is not the identity (a parameter left out gets the
        identity link).
    """

    def __init__(self, base: PathModel, links: dict[str, str]) -> None:
        self.base = base
        self.links = {
            name: links.get(name, "identity") for name in base.param_names
        }
        self.name = base.name
        self.param_names = [
            "log({})".format(name) if link == "log" else name
            for name, link in self.links.items()
        ]
        self.linear_in_parameters = base.linear_in_parameters and all(
            link == "identity" for link in self.links.values()
        )
        fns = [_LINKS[link] for link in self.links.values()]
        self._forward = [f[0] for f in fns]
        self._inverse = [f[1] for f in fns]
        self._deriv = [f[2] for f in fns]

    def to_natural(self, eta: npt.ArrayLike) -> npt.NDArray:
        """Natural-scale parameters ``theta = h(eta)``."""
        eta_arr = np.asarray(eta, dtype=float)
        return np.array([h(e) for h, e in zip(self._forward, eta_arr)])

    def to_link(self, theta: npt.ArrayLike) -> npt.NDArray:
        """Link-scale parameters ``eta = h^-1(theta)``."""
        theta_arr = np.asarray(theta, dtype=float)
        for name, link, t in zip(
            self.base.param_names, self.links.values(), theta_arr
        ):
            if link == "log" and not t > 0:
                raise ValueError(
                    "Path parameter {} = {:.6g} is not positive, so it "
                    "cannot be modelled on a log link".format(name, t)
                )
        return np.array([g(t) for g, t in zip(self._inverse, theta_arr)])

    def _link_derivative(self, eta: npt.NDArray) -> npt.NDArray:
        return np.array([d(e) for d, e in zip(self._deriv, eta)])

    def path(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        return self.base.path(x, *self.to_natural(params))

    def inv_path(self, y: npt.ArrayLike, *params: float) -> npt.NDArray:
        return self.base.inv_path(y, *self.to_natural(params))

    def jacobian(self, x: npt.ArrayLike, *params: float) -> npt.NDArray:
        # chain rule: d path / d eta = (d path / d theta) * h'(eta)
        eta = np.asarray(params, dtype=float)
        natural = np.asarray(
            self.base.jacobian(x, *self.to_natural(eta)), dtype=float
        )
        return natural * self._link_derivative(eta)[None, :]

    def check_data(self, x: npt.NDArray, y: npt.NDArray) -> None:
        self.base.check_data(x, y)

    def fit(self, x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray:
        """The base model's per-unit fit, returned on the link scale."""
        return self.to_link(self.base.fit(x, y))

    def __repr__(self) -> str:
        return "{} Degradation Path Model on links {}".format(
            self.base.name, self.links
        )


def stress_design(
    z: npt.ArrayLike, links: dict[str, str], param_names: list[str]
) -> npt.NDArray:
    """
    The fixed-effects design ``D(z)`` of one unit: a ``(p, m)`` matrix
    mapping the fixed effects ``gamma`` to the unit's link-scale path
    parameter means, ``eta_mean = D(z) gamma``.

    The columns are laid out parameter by parameter, in path order: an
    intercept column for every parameter, followed -- for the
    stress-dependent parameters named in ``links`` -- by one column per
    covariate carrying that unit's stress values. So
    ``m = p + q * len(links)`` with ``q`` covariates.
    """
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    q = z_arr.shape[0]
    m = len(param_names) + q * len(links)
    design = np.zeros((len(param_names), m))
    col = 0
    for row, name in enumerate(param_names):
        design[row, col] = 1.0
        col += 1
        if name in links:
            design[row, col : col + q] = z_arr
            col += q
    return design


def fixed_effect_names(
    linked_param_names: list[str],
    param_names: list[str],
    links: dict[str, str],
    n_cov: int,
) -> list[str]:
    """Labels for ``gamma`` matching :func:`stress_design`'s columns:
    the link-scale parameter name for each intercept and
    ``"<name>:Z<j>"`` for each stress coefficient."""
    names = []
    for linked_name, name in zip(linked_param_names, param_names):
        names.append(linked_name)
        if name in links:
            names.extend("{}:Z{}".format(linked_name, j) for j in range(n_cov))
    return names
