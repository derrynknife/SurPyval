"""The check every degradation likelihood fit makes of its answer (#564).

The process and destructive fits search their likelihood with BFGS (and a
Nelder-Mead fallback) or a bounded scalar search, none of which says
whether it stopped at a maximum. :func:`verified_search` checks the answer
as every other maximum-likelihood fit does (principle 13): a zero gradient
and a positive-definite Hessian of the negative log-likelihood in the space
the fit searched, by central differences (``verify_or_polish(...,
numerical=True)``), polishing an answer that fails with BFGS and warning
(``warn_unverified``) if it still fails. The fitted model records what was
reached as its ``maximum``.

The bootstraps that refit a degradation model on resampled units count
what their refits reached with one :class:`RefitTally`, and say it in one
warning.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.univariate.parametric.fitters import verify_or_polish
from surpyval.utils.no_maximum import (
    NoFiniteMaximumError,
    warn_bootstrap_refits,
    warn_unverified,
)


def verified_search(
    fun: Callable[[npt.NDArray], Any],
    res: Any,
    n_obs: float,
    what: str,
) -> tuple[npt.NDArray, str]:
    """
    ``res.x``, the minimum of ``fun`` a fit's search found (polished where
    that was not a verified minimum), and its ``maximum`` state:
    ``"verified"``, or ``"unverified"`` after warning that ``what`` did not
    reach a verified maximum.

    ``fun`` is the negative log-likelihood in the space the fit searched,
    over ``n_obs`` observations (the per-observation scale of the gradient
    test), and ``res`` has the point ``x`` and its value ``fun``.
    """
    with np.errstate(all="ignore"):
        res, verified = verify_or_polish(fun, res, n_obs, numerical=True)
    if not verified:
        warn_unverified(what)
    state = "verified" if verified else "unverified"
    return np.asarray(res.x, dtype=float), state


class RefitTally:
    """
    The refits of a degradation bootstrap, counted for the one warning it
    gives about them (``warn_bootstrap_refits``): those kept, and of them
    the ones whose fit had no finite or an unverified maximum; and the
    resamples left out because their likelihood has no maximum at all. A
    refit that failed for any other reason is left out, and counted as
    failed from the others.

    Shared by the two-stage bootstrap of a degradation analysis
    (``DegradationModel.cb(method="bootstrap")``) and the bootstrap of a
    destructive degradation fit.

    Examples
    --------
    >>> from surpyval.degradation._maximum import RefitTally
    >>> from surpyval.utils.no_maximum import NoFiniteMaximumError
    >>> def no_maximum():
    ...     raise NoFiniteMaximumError("no failures")
    >>> tally = RefitTally()
    >>> tally.refit(no_maximum) is None
    True
    >>> tally.refit(lambda: "fit")
    'fit'
    >>> tally.keep("unverified")
    >>> tally.kept, tally.unverified, tally.no_estimate
    (1, 1, 1)
    """

    def __init__(self) -> None:
        self.kept = 0
        self.no_maximum = 0
        self.unverified = 0
        self.no_estimate = 0

    def refit(self, fit: Callable[..., Any], *args: Any) -> Any:
        """``fit(*args)``, or ``None`` where it raised: counted as left
        out with no estimate where the resample's likelihood has no maximum
        at all (``NoFiniteMaximumError``), as failed otherwise."""
        try:
            return fit(*args)
        except NoFiniteMaximumError:
            self.no_estimate += 1
        except Exception:
            pass
        return None

    def keep(self, maximum: Any) -> None:
        """Count a refit kept, by the ``maximum`` its fit reached."""
        self.kept += 1
        if maximum == "no finite maximum":
            self.no_maximum += 1
        elif maximum == "unverified":
            self.unverified += 1

    def warn(self, n_boot: int, advice: str) -> None:
        """The one warning about the ``n_boot`` refits, if any is due,
        ending with ``advice``."""
        warn_bootstrap_refits(
            n_boot,
            no_maximum=self.no_maximum,
            no_estimate=self.no_estimate,
            unverified=self.unverified,
            failed=n_boot - self.kept - self.no_estimate,
            advice=advice,
        )
