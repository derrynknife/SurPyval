import numpy as np
from numpy.typing import ArrayLike, NDArray

from .node import Node


class Tree:
    """
    TODO
    """

    def __init__(
        self,
        x: ArrayLike,
        Z: NDArray,
        c: ArrayLike,
        max_depth: int | float = float("inf"),
        min_leaf_samples: int = 6,
        bootstrap: bool = True,
    ):
        x = np.array(x)
        c = np.array(c)
        Z = np.array(Z)
        self._root = Node(
            x=x,
            Z=Z,
            c=c,
            curr_depth=0,
            max_depth=max_depth,
            min_leaf_samples=min_leaf_samples,
        )
