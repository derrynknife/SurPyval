from numpy.typing import ArrayLike, NDArray

from surpyval import Weibull
from .log_rank_split import log_rank_split


class Node:
    def __init__(
        self,
        x: ArrayLike,
        Z: NDArray,
        c: ArrayLike,
        curr_depth: int,
        max_depth: int | float,
        min_leaf_samples: int
    ):
        # Choose the best feature-value pair
        (
            self.split_feature_index,
            self.split_value,
        ) = log_rank_split(x, Z, c, min_leaf_samples)
        self.left_node = LeafNode(x=x, Z=Z, c=c)
        self.right_node = LeafNode(x=x, Z=Z, c=c)


class LeafNode(Node):
    """
    Same as Node, but stores the model fitted on the leaf samples.
    """

    def __init__(
        self,
        x: ArrayLike,
        Z: NDArray,
        c: ArrayLike,
    ):
        self.model = Weibull.fit(x=x, c=c)
