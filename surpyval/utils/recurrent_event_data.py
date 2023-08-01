import numpy as np

from ..utils import xcnt_handler


class RecurrentEventData:
    def __init__(self, x, i, c, n):
        self.x = np.atleast_1d(x)
        self.i = np.atleast_1d(i)
        self.c = np.atleast_1d(c)
        self.n = np.atleast_1d(n)
        self.items = list(set(self.i))

        if self.x.ndim == 1:
            self.interarrival_times = self.find_interarrival_times(x, i)
        else:
            self.midpoints = self.x.mean(axis=1)
        self._index = 0

    def to_xrd(self, estimator="Nelson-Aalen"):
        if not hasattr(self, "xrd"):
            # find the total number of times an event occurs at each x
            if self.x.ndim == 2:
                x_out = self.midpoints
            else:
                x_out = self.x

            x_unique = np.unique(x_out)

            d = np.array(
                [
                    self.n[
                        (x_out == xi) & ((self.c == 0) | (self.c == 2))
                    ].sum()
                    for xi in x_unique
                ]
            )
            # count the number of items at their maximum x for each x_unique
            # find the maximum x for each item
            max_x = np.array(
                [self.x[self.i == item].max() for item in self.items]
            )
            # sum the number of items at each x in x_unique
            # that are at their max
            r = np.array([(max_x == xi).sum() for xi in x_unique])

            r = len(self.items) * np.ones_like(x_unique) - r.cumsum() + r

            self.xrd = x_unique, r, d
        return self.xrd

    def find_interarrival_times(self, x, i):
        interarrival_times = []
        prev_i = None
        prev_x = 0
        for xi, ii in zip(x, i):
            if ii != prev_i:
                interarrival_times.append(xi)
                prev_i = ii
            else:
                interarrival_times.append(xi - prev_x)

            prev_x = xi
        return np.atleast_1d(interarrival_times)

    def get_events_for_item(self, item):
        mask = self.i == item
        return self.x[mask], self.c[mask], self.n[mask]

    def get_times_to_first_events(self):
        x_ttff = np.array([self.x[self.i == item][0] for item in self.items])
        c_ttff = np.array([self.c[self.i == item][0] for item in self.items])
        n_ttff = np.array([self.n[self.i == item][0] for item in self.items])
        return xcnt_handler(x_ttff, c_ttff, n_ttff, as_surpyval_dataset=True)

    def __getitem__(self, index):
        return RecurrentEventData(
            self.x[index],
            self.i[index],
            self.c[index],
            self.n[index],
        )

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self.x):
            result = (
                self.x[self._index],
                self.i[self._index],
                self.c[self._index],
                self.n[self._index],
            )
            self._index += 1
            return result
        else:
            raise StopIteration

    def __repr__(self):
        return f"""
            RecurrentEventData(
    x={self.x},
    i={self.i},
    c={self.c},
    n={self.n}\n)
        """.strip()
