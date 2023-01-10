from collections import defaultdict

import numpy as np


class Renewal:
    def __init__(self, model):
        self.model = model

    def simulate_over_interval(self, T, N=100_000):
        running = 0
        x = self.model.random(2 * N).tolist()
        aggregate_timeline = defaultdict(lambda: 0)
        # Put in values for times 0 and T
        # this ensures there is an entry at 0 and T
        aggregate_timeline[0] = 0
        aggregate_timeline[T] = 0
        counted = 0

        while counted < N:
            try:
                xi = x.pop()
            except Exception:
                # For now, assume 2 events in the window
                x = self.model.random(2 * N).tolist()
                xi = x.pop()

            running += xi

            if running > T:
                running = 0
                counted += 1
            else:
                aggregate_timeline[running] += 1

        timeline_arr = np.array(list(aggregate_timeline.items()))
        timeline_arr = timeline_arr[timeline_arr[:, 0].argsort()]

        timeline_arr[:, 1] = (timeline_arr[:, 1] / N).cumsum()
        timeline = timeline_arr[:, 0]
        exp_n = timeline_arr[:, 1]

        def cif(x):
            return np.interp(x, timeline, exp_n)

        self._cif = cif

    def reset(self):
        del self._cif

    def cif(self, x):
        return self._cif(x)
