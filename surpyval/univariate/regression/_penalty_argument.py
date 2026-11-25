"""
A penalty is an argument of ``fit``, not a method of its own.

A regression with a ridge, lasso or elastic-net penalty was fitted by a
twin of each fit (``fit_penalised``, ``fit_tvc_penalised``,
``fit_tvc_timeline_penalised``), and a Cox model with Firth's penalty by
``fit_firth``: ten fit methods on ``CoxPH`` alone, the data's shape times
its source times the penalty. Since v0.30 the penalty is an option of the
fit (Design Principles, principle 24): ``fit(..., penalizer=0.1,
l1_ratio=0.5)``, ``fit(..., firth=True)``.

:func:`penalty_argument` decorates a ``fit`` with those arguments. Without
them the fit is the fit it was; with them it is the penalised twin's,
called with every argument the caller gave, so a penalised call gives the
twin's model to the last digit. :func:`penalised_entry` keeps the twin's
old public name working, with a ``FutureWarning``, until v0.31.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, TypeVar

from surpyval.utils.deprecation import REMOVED_IN_NEXT, RenamedAttribute

F = TypeVar("F", bound=Callable[..., Any])

#: The penalty's arguments, each with its default (no penalty).
_PENALTY = {"penalizer": None, "l1_ratio": 0.0, "standardise": True}

_DOC = """
        penalizer : float or array_like, optional
            A ridge, lasso or elastic-net penalty on the covariate
            coefficients, :math:`\\lambda`: one value, or one per column
            of ``Z`` (0 leaves a coefficient unpenalised). ``None`` (the
            default) fits without a penalty. Any other value, 0 included,
            gives the penalised fit (until v0.30 ``{twin}``), whose model
            predicts as any other but has no standard errors or bounds:
            penalised estimates are biased towards 0 by design.
        l1_ratio : float, optional
            The penalty's mix, :math:`a`, from 0 (ridge, the default) to 1
            (lasso). Only with ``penalizer``.
        standardise : bool, optional
            ``True`` (the default) penalises the coefficients of the
            covariates divided by their standard deviations, so the penalty
            does not depend on their units. Only with ``penalizer``. The
            penalised fit's own options (its optimality tolerance ``tol``,
            and where the unpenalised fit lacks them, its weights ``w``)
            are given beside ``penalizer`` by name.
"""

_FIRTH_DOC = """
        firth : bool, optional
            ``True`` fits with Firth's penalty (until v0.30 ``fit_firth``):
            half the log-determinant of the information added to the
            log-partial-likelihood, which removes the first-order bias of
            the estimates and gives finite coefficients under separation.
            Defaults to ``False``. With it, ``alpha_ci`` and ``profile`` set
            the level and the kind of the intervals it reports.
"""


def _given(value: Any, default: Any) -> bool:
    """Whether an argument was given a value other than its default."""
    if value is default:
        return False
    try:
        return bool(value != default)
    except (TypeError, ValueError):
        # an array compared element-wise
        return True


def _with_parameters(doc: str | None, extra: str) -> str | None:
    """``doc`` with ``extra`` added at the end of its Parameters section
    (before Returns), or at the end."""
    if not doc:
        return doc
    for header in ("\n        Returns\n", "\n        Raises\n"):
        at = doc.find(header)
        if at >= 0:
            return doc[:at] + extra.rstrip("\n") + "\n" + doc[at:]
    return doc + extra


def penalty_argument(
    twin: str,
    firth: str | None = None,
    old_twin: str | None = None,
) -> Callable[[F], F]:
    """
    Decorate a ``fit`` (or ``fit_tvc``, ``fit_tvc_timeline``) so that it
    takes the penalty as arguments: ``penalizer``, ``l1_ratio`` and
    ``standardise`` (and, with ``firth``, Cox's ``firth``).

    Parameters
    ----------
    twin : str
        The private name of the penalised fit, ``"_fit_penalised"``.
    firth : str, optional
        The private name of the Firth-penalised fit, ``"_fit_firth"``.
    old_twin : str, optional
        The twin's old public name, for the documentation; ``twin``
        without its underscore by default.

    Notes
    -----
    Put it outermost, above every other decorator. Without a penalty the
    decorated fit is called with the arguments as given. With one, the
    twin is called with the arguments the caller gave, by name; an
    argument the twin does not take is refused, naming it, and so is an
    option only the twin takes (its ``tol``, its weights ``w``) given
    without a penalty.
    """
    old_name = old_twin or twin.lstrip("_")

    def decorate(fit: F) -> F:
        signature = inspect.signature(fit)
        names = list(signature.parameters)
        bound_method = names[0] in ("self", "cls")
        takes_any = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )

        def target_parameters(owner: Any, name: str) -> dict:
            return dict(inspect.signature(getattr(owner, name)).parameters)

        @functools.wraps(fit)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            owner = args[0] if bound_method else None
            penalty = {k: kwargs.pop(k, v) for k, v in _PENALTY.items()}
            use_firth = bool(kwargs.pop("firth", False)) if firth else False
            penalised = penalty["penalizer"] is not None
            where = f"{_owner_name(owner)}.{fit.__name__}"
            if not penalised:
                for k in ("l1_ratio", "standardise"):
                    if _given(penalty[k], _PENALTY[k]):
                        raise ValueError(
                            f"{where}: {k} sets the penalty, which needs a "
                            "penalizer: give penalizer=... (0.1, say), or "
                            f"leave {k} out."
                        )
            if penalised and use_firth:
                raise ValueError(
                    f"{where}: firth=True is Firth's penalty, not a ridge, "
                    "lasso or elastic net: give one of firth and penalizer."
                )
            extra_names = set(kwargs) - set(names)
            if not penalised and not use_firth:
                if not extra_names or takes_any or owner is None:
                    return fit(*args, **kwargs)
                # The fit refuses what it does not take, in its own words
                # (a WeightedRefusal for w); a plain unexpected keyword is
                # told it is an option of the penalised fit.
                try:
                    return fit(*args, **kwargs)
                except TypeError as error:
                    hint = _only_with_penalty(
                        owner, sorted(extra_names), twin, firth
                    )
                    if "unexpected keyword" not in str(error) or not hint:
                        raise
                    raise TypeError(f"{error}{hint}") from None
            target = firth if use_firth else twin
            assert target is not None
            bound = signature.bind_partial(
                *args, **{k: v for k, v in kwargs.items() if k in names}
            )
            call = {
                k: v
                for k, v in bound.arguments.items()
                if k not in ("self", "cls")
            }
            call.update({k: kwargs[k] for k in extra_names})
            accepted = target_parameters(owner, target)
            refused = sorted(k for k in call if k not in accepted)
            if refused:
                kind = "firth=True" if use_firth else "a penalizer"
                raise ValueError(
                    f"{where}: {', '.join(refused)} cannot be given with "
                    f"{kind}: the penalised fit does not take "
                    + ("it" if len(refused) == 1 else "them")
                    + ". Leave "
                    + ("it" if len(refused) == 1 else "them")
                    + " out, or fit without the penalty."
                )
            if not use_firth:
                call.update(penalty)
            return getattr(owner, target)(**call)

        params = list(signature.parameters.values())
        keyword = inspect.Parameter.KEYWORD_ONLY
        kept = [p for p in params if p.kind != inspect.Parameter.VAR_KEYWORD]
        tail = [p for p in params if p.kind == inspect.Parameter.VAR_KEYWORD]
        added = [
            inspect.Parameter(k, keyword, default=v)
            for k, v in _PENALTY.items()
            if k not in signature.parameters
        ]
        if firth:
            added.append(inspect.Parameter("firth", keyword, default=False))
        # keyword-only parameters come after the positional ones
        new_signature = signature.replace(parameters=kept + added + tail)
        wrapper.__signature__ = new_signature  # type: ignore[attr-defined]
        doc = _DOC.format(twin=old_name) + (_FIRTH_DOC if firth else "")
        wrapper.__doc__ = _with_parameters(fit.__doc__, doc)
        return wrapper  # type: ignore[return-value]

    return decorate


def _owner_name(owner: Any) -> str:
    if owner is None:
        return "fit"
    if isinstance(owner, type):
        return owner.__name__.rstrip("_")
    repr_name = getattr(owner, "_repr_name", None)
    if callable(repr_name):
        try:
            return str(repr_name())
        except Exception:
            pass
    name = getattr(owner, "name", None)
    if isinstance(name, str) and name and " " not in name and "-" not in name:
        return name
    return type(owner).__name__.rstrip("_")


def _only_with_penalty(
    owner: Any, names: list[str], twin: str, firth: str | None
) -> str:
    """The end of the error for an option only a penalised fit takes."""
    for target in (twin, firth):
        if target is None:
            continue
        accepted = inspect.signature(getattr(owner, target)).parameters
        if all(n in accepted for n in names):
            what = "firth=True" if target == firth else "a penalizer"
            return f": {', '.join(names)} is taken only with {what}."
    return ""


def penalised_entry(new: str, advice: str) -> RenamedAttribute:
    """
    The old public name of a penalised fit, now the private ``new``:
    ``fit_penalised`` and its kin, deprecated in v0.30 because the penalty
    is an argument of the fit. Declare it on the class under the old name,
    ``fit_penalised = penalised_entry("_fit_penalised", "call fit(...,
    penalizer=0.1)")``. Calling it warns with a ``FutureWarning`` that
    says what to call, and runs the private method, so the model is the
    same to the last digit until v0.31 removes it.
    """
    return _PenalisedEntry(new, REMOVED_IN_NEXT, FutureWarning, advice=advice)


class _PenalisedEntry(RenamedAttribute):
    """A :class:`RenamedAttribute` that names the fitter as users write it
    (``WeibullPH``, not its class ``ProportionalHazardsFitter``)."""

    def __get__(self, obj: Any, owner: type | None = None) -> Any:
        owner = type(obj) if owner is None else owner
        if obj is None and self.new not in dir(owner):
            return self
        import warnings

        from surpyval.utils.warnings import caller_stacklevel

        who = _owner_name(owner if obj is None else obj)
        warnings.warn(
            f"{who}.{self.old} is deprecated and will be removed in "
            f"v{self.removed_in}; {self.advice}.",
            self.category,
            stacklevel=caller_stacklevel(),
        )
        return getattr(owner if obj is None else obj, self.new)


__all__ = ["penalty_argument", "penalised_entry", "REMOVED_IN_NEXT"]
