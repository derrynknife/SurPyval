from surpyval.parametric import Weibull
from collections import defaultdict
import heapq


class GRP:
    __init__(self, dist):
        self.dist = dist


    def fit(self, units, x):
        units = np.array(units)
        x = np.array(x)

        if units.shape[0] != x.shape[0]:
            raise ValueError("units array must be same length as variables array")

        failure_data = defaultdict(lambda: [])
        for u, t in zip(units, times):
            heapq.heappush(failure_data[u], t)

        unique_units = failure_data.keys()
        ttff = np.zeros(len(unique_units))

        for i, k in enumerate(unique_units):
            ttff[i] = heapq.heappop(failure_data[k])

        remaining_failures = np.concatenate([np.fromiter(v, dtype=float) for v in failure_data.values()])


        self.underlying_distribution = dist.fit(ttff)

        














