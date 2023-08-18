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
            self.interarrival_times = self.find_interarrival_times()
        else:
            x_midpoints = self.x.copy()
            x_midpoints[self.c == -1, 0] = 0
            self.midpoints = x_midpoints.mean(axis=1)
        self._index = 0

    def to_xrd(self, estimator="Nelson-Aalen"):
        if not hasattr(self, "xrd"):
            # find the total number of times an event occurs at each x
            if self.x.ndim == 2:
                x_out = self.midpoints
            else:
                x_out = self.x

            x_unique = np.unique(x_out)

            # TODO: consider having the presence of left-censored
            # data use the midpoints instead of the end value of the left
            # censored interval.

            d = np.array(
                [
                    self.n[
                        (x_out == xi)
                        & ((self.c == 0) | (self.c == 2) | (self.c == -1))
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

    def find_interarrival_times(self):
        _, idx = np.unique(self.i, return_index=True)
        arrival_times = np.split(self.x, idx)[1:]
        interarrival_times = [np.diff(arr, prepend=0) for arr in arrival_times]
        return np.concatenate(interarrival_times)

    def find_x_previous(self):
        unique_items = np.unique(self.i)
        x_previous = []
        for item in unique_items:
            mask_item = self.i == item
            x_item = self.x[mask_item]
            x_prev_item = np.roll(x_item, shift=1, axis=0)
            # Replace the first value with 0
            x_prev_item[0] = 0
            x_previous.append(x_prev_item)

        return np.concatenate(x_previous)

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
