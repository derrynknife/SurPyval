"""
The one rule for turning a ``random_state`` (or ``seed``) argument into a
numpy ``Generator``.

``None`` draws the generator's seed from numpy's global RNG, so
``np.random.seed`` controls every draw SurPyval makes, as it already did
for ``Parametric.random`` and the recurrent simulations. An int or a
``Generator`` gives a stream that does not depend on (or advance) the
global state.

A plain ``np.random.default_rng(None)`` seeds from fresh OS entropy on
every call instead, so a draw could not be reproduced without passing a
seed to each call; code that seeds the global RNG once and calls
``model.random(n)`` generically got a different answer on every run for
the models built that way.
"""

from typing import Any

import numpy as np

__all__ = ["as_generator"]


def as_generator(random_state: Any = None) -> np.random.Generator:
    """
    A numpy ``Generator`` for ``random_state``.

    Parameters
    ----------
    random_state : None, int, array_like of ints, numpy.random.SeedSequence,
        numpy.random.BitGenerator or numpy.random.Generator
        ``None`` seeds the generator from numpy's global RNG, so
        ``np.random.seed`` makes the draw reproducible (and advances the
        global stream by one draw). Anything else is passed to
        ``numpy.random.default_rng``: a seed gives a reproducible stream
        independent of the global state, and a ``Generator`` is used as
        is.

    Returns
    -------
    numpy.random.Generator

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.utils.rng import as_generator
    >>> np.random.seed(0); a = as_generator().uniform(size=3)
    >>> np.random.seed(0); b = as_generator().uniform(size=3)
    >>> bool(np.array_equal(a, b))
    True
    >>> bool(np.array_equal(as_generator(1).uniform(size=3),
    ...                     np.random.default_rng(1).uniform(size=3)))
    True
    """
    if random_state is None:
        seed = np.random.randint(0, np.iinfo(np.int64).max, dtype=np.int64)
        return np.random.default_rng(int(seed))
    return np.random.default_rng(random_state)
