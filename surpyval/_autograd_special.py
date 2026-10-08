"""Load ``autograd.scipy.special`` without the rest of ``autograd.scipy``.

The distributions differentiate through autograd's special functions
(``gammaln``, ``betaln``, ``expit``, ...), but importing
``autograd.scipy.special`` runs the ``autograd.scipy`` package first,
which imports its ``integrate``, ``signal`` and ``stats`` modules and with
them ``scipy.stats`` and ``scipy.integrate``: about 0.45 s of every
``import surpyval`` (#470).

This loads autograd's ``scipy/special.py`` as the module
``autograd.scipy.special`` itself, so every ``from autograd.scipy.special
import ...`` finds it without running the package. Nothing else changes:
the functions are autograd's own, and a later ``import autograd.scipy``
(by SurPyval where it needs autograd's ``stats``, or by the user) runs the
package as usual and takes this module as its ``special``, so there is
one set of primitives. Where autograd is laid out differently, or the
package is already imported, nothing is done and the package is imported
the usual way.
"""

import importlib.util
import os
import sys

_NAME = "autograd.scipy.special"


def _load() -> None:
    if _NAME in sys.modules or "autograd.scipy" in sys.modules:
        return
    import autograd

    path = os.path.join(
        os.path.dirname(autograd.__file__), "scipy", "special.py"
    )
    if not os.path.isfile(path):
        return
    spec = importlib.util.spec_from_file_location(_NAME, path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        # Left to the usual import, which will say what is wrong
        del sys.modules[_NAME]


def _forward_rules() -> None:
    """``expit`` (the Logistic and LogLogistic distribution function) in
    forward mode: autograd defines its derivative only in reverse, and the
    Wald confidence bounds take their Jacobian in forward mode, one pass
    per parameter rather than one per point. The rule is the derivative
    autograd's reverse rule uses, ``s (1 - s)``."""
    try:
        from autograd.extend import defjvp
        from autograd.scipy import special
    except Exception:
        return
    if hasattr(special, "expit"):
        defjvp(special.expit, lambda g, ans, x: g * ans * (1.0 - ans))


_load()
_forward_rules()
