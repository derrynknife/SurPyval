import numpy as np


class SurpyvalData:
    def __init__(self, x, c, n, t):
        self.x, self.c, self.n, self.t = x, c, n, t
        self.tl = self.t[:, 0]
        self.tr = self.t[:, 1]
        self.x_min, self.x_max = np.min(x), np.max(x)
        self.split_to_observation_types()
        self._index = 0

    def split_to_observation_types(self):
        self.x_o, self.n_o = self._split_by_mask(self.c == 0)
        self.x_r, self.n_r = self._split_by_mask(self.c == 1)
        self.x_l, self.n_l = self._split_by_mask(self.c == -1)
        if self.x.ndim == 2:
            interval_mask = self.c == 2
            self.x_il = self.x[interval_mask, 0]
            self.x_ir = self.x[interval_mask, 1]
            self.n_i = self.n[interval_mask]
        else:
            self.x_il, self.x_ir, self.n_i = [], [], []
        truncated_mask = np.isfinite(self.t).any(axis=1)
        self.x_tl, self.x_tr, self.n_t = (
            self.t[truncated_mask, 0],
            self.t[truncated_mask, 1],
            self.n[truncated_mask],
        )

    def _split_by_mask(self, mask, x=None):
        if x is None:
            x = self.x[mask] if self.x.ndim == 1 else self.x[mask, 0]
        return x, self.n[mask]

    def __getitem__(self, index):
        return SurpyvalData(
            self.x[index],
            self.c[index],
            self.n[index],
            self.t[index],
        )

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self.x):
            result = (
                self.x[self._index],
                self.c[self._index],
                self.n[self._index],
                self.t[self._index],
            )
            self._index += 1
            return result
        else:
            raise StopIteration

    def __repr__(self):
        return f"""
            SurpyvalData(\nx={self.x},\nc={self.c},\nn={self.n},\nt={self.t})
        """.strip()
