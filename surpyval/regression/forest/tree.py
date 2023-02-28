import numpy as np
from numpy.typing import ArrayLike, NDArray

from .node import Node


class Tree_:
    """
    TODO
    """

    def fit(
        self,
        x: ArrayLike,
        Z: NDArray,
        c: ArrayLike,
        max_depth: int | float = float("inf"),
    ):
        x = np.array(x)
        c = np.array(c)
        self._root = Node(x=x, Z=Z, c=c, curr_depth=0, max_depth=max_depth)

        return self


Tree = Tree_()
