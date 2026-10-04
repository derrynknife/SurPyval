"""The one-line ``repr`` of a fitter (#614).

A fitter -- ``Weibull``, ``KaplanMeier``, ``WeibullAFT``, ``CoxPH``, ... --
printed as ``<surpyval...Weibull_ object at 0x7f...>`` in a notebook. Each
now prints its name and what it fits, with its baseline where it has one::

    Weibull: parametric fitter
    KaplanMeier: non-parametric fitter
    WeibullAFT: accelerated failure time fitter (Weibull baseline)
    CoxPH: semi-parametric proportional hazards fitter

The line is built from class attributes (:class:`FitterRepr`), not
written out for each fitter: a family's base class says what kind of
fitter it is (``fitter_kind``), and the name is the fitter's own where it
has one that reads as a name (``WeibullPH``, ``Galton``), else its class's.
"""

from typing import Any

from surpyval.utils.removed_names import RemovedNames

__all__ = ["FitterRepr", "baseline_name"]


class FitterRepr(RemovedNames):
    """
    A fitter's ``repr``: ``"<name>: <kind>"`` and, where the fitter has a
    baseline distribution (or other parts), ``" (<details>)"``.

    A family's base class sets :attr:`fitter_kind`; :meth:`_repr_name` and
    :meth:`_repr_details` give the name and the details, and a family
    overrides them where its name or baseline is held elsewhere.

    Examples
    --------
    >>> import surpyval as sp
    >>> sp.Weibull
    Weibull: parametric fitter
    >>> sp.KaplanMeier
    KaplanMeier: non-parametric fitter
    >>> sp.WeibullAFT
    WeibullAFT: accelerated failure time fitter (Weibull baseline)
    >>> sp.CoxPH
    CoxPH: semi-parametric proportional hazards fitter
    """

    #: What the fitter fits, printed after its name.
    fitter_kind: str = "fitter"

    def _repr_name(self) -> str:
        # The fitter's own name where it reads as one (``Weibull``,
        # ``WeibullPH``, ``Galton``); not a display name such as
        # "Crow-AMSAA" or "Homogeneous Poisson Process", where the class's
        # name, less the singleton's trailing underscore, is the public one.
        name = getattr(self, "name", None)
        if (
            isinstance(name, str)
            and name
            and not any(ch.isspace() or ch == "-" for ch in name)
        ):
            return name
        return type(self).__name__.rstrip("_")

    def _repr_details(self) -> "list[str]":
        return []

    def __repr__(self) -> str:
        details = self._repr_details()
        tail = " ({})".format(", ".join(details)) if details else ""
        return "{}: {}{}".format(self._repr_name(), self.fitter_kind, tail)


def baseline_name(fitter: Any) -> "list[str]":
    """``["<distribution> baseline"]`` for a fitter with a baseline
    distribution ``dist``, else ``[]``: the details of a regression
    fitter's ``repr``.

    Examples
    --------
    >>> import surpyval as sp
    >>> from surpyval.utils.fitter_repr import baseline_name
    >>> baseline_name(sp.WeibullPH), baseline_name(sp.Weibull)
    (['Weibull baseline'], [])
    """
    dist = getattr(fitter, "dist", None)
    name = getattr(dist, "name", None)
    return [] if not isinstance(name, str) else [name + " baseline"]
