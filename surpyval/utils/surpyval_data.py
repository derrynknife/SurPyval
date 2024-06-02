import numpy as np

import surpyval
from surpyval.utils import xcnt_handler


class SurpyvalData:
    def __init__(
        self,
        x=None,
        c=None,
        n=None,
        t=None,
        xl=None,
        xr=None,
        tl=None,
        tr=None,
        group_and_sort=True,
        handle=True,
    ):
        """
        SurpyvalData class for single event survival data. Data arrays stored
        in the x, c, n, and t attributes of the object. Can be used to convert
        data to the xrd format, take subsets, sele

        Indexing is supported and will return a new SurpyvalData object with
        the indexed data.

        Uses the ``xcnt_handler`` method to validate the data.

        Parameters
        ----------

            x : numpy.ndarray
                The primary data array.
            c : numpy.ndarray
                The censoring flag array (0 = not censored, 1 = censored,
                -1 = left censored, 2 = interval censored) for each value
                in `x`.
            n : numpy.ndarray
                The number of occurrences for each value in `x`.
            t : numpy.ndarray
                The truncation information array. 2D array with the left and
                right truncation values for each value in `x`.
            xl: array or scalar, optional (default: None)
                array of the values of the left interval of interval censored
                data. Cannot be used with 'x' parameter, must be used with the
                'xr' parameter
            xr: array or scalar, optional (default: None)
                array of the values of the right interval of interval censored
                data. Cannot be used with 'x' parameter, must be used with the
                'xl' parameter
            tl: array or scalar, optional (default: None)
                array of values of the left value of truncation. If scalar, all
                values will be treated as left truncated by that value cannot
                be used with 't' parameter but can be used with the 'tr'
                parameter
            tr: array or scalar, optional (default: None)
                array of values of the right value of truncation. If scalar,
                all values will be treated as right truncated by that value
                cannot be used with 't' parameter but can be used with the 'tl'
                parameter
            group_and_sort: bool, optional (default: True)
                whether to group and sort the data. If False, the data will be
                returned in the order it was entered. This is useful for when
                validating survival data for which you also have covariates.
            handle: bool, optional (default: True)
                Whether to validate the data. If False, the data will not be
                validated. This is useful for when using data that has already
                been validated.

        Examples
        --------

        >>> import numpy as np
        >>> from surpyval import SurpyvalData
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> c = np.array([0, 0, 1, 1, 1])
        >>> n = np.array([1, 1, 1, 1, 1])
        >>> t = np.array([[0, 5], [0, 5], [0, 5], [0, 5], [0, 5]])
        >>> data = SurpyvalData(x, c, n, t)
        >>> data.to_xrd()
        (array([1, 2, 3, 4, 5]), array([5, 4, 3, 2, 1]), array(1, 1, 1, 1, 1]))
        >>> data[0:2]
        SurpyvalData(
            x=[1 2],
            c=[0 0],
            n=[1 1],
            t=[[0 5]
             [0 5]])
        """
        if handle:
            x, c, n, t = xcnt_handler(
                x, c, n, t, xl, xr, tl, tr, group_and_sort=group_and_sort
            )

        self.x, self.c, self.n, self.t = x, c, n, t
        self.tl = self.t[:, 0]
        self.tr = self.t[:, 1]
        self.x_min, self.x_max = np.min(x), np.max(x)
        self.split_to_observation_types()
        self._index = 0

    def add_covariates(self, Z):
        """
        Method to add covariates to the data. The covariates are stored in
        the Z attribute of the object. When doing regression survival analysis
        this method allows for the covariates to be added to the data in a
        consistent manner that also allows for the data to be converted to be
        passed to the fitters.

        Parameters
        ----------

        Z : numpy.ndarray
            The covariate array.
        """
        self.Z = Z
        self.Z_o = Z[self.c == 0]
        self.Z_r = Z[self.c == 1]
        self.Z_l = Z[self.c == -1]
        self.Z_i = Z[self.c == 2]

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

    def to_xrd(self, estimator="Nelson-Aalen"):
        """
        Converts the data into the xrd format. If the data has right truncated
        observations or left or interval censored observations, the data is
        converted to the xrd format using the Turnbull estimator. The
        ``estimator`` parameter will be used in the with, and only with,
        the Turnbull estimator.

        Parameters
        ----------

            estimator : str, optional
                The method for estimation if data requires the use of the
                Turnbull estimator to convert to xrd, defaults to
                "Nelson-Aalen".

        Returns
        -------

            tuple
                The xrd data.
        """
        if not hasattr(self, "xrd"):
            if (
                np.isfinite(self.t[:, 1]).any()
                | (2 in self.c)
                | (-1 in self.c)
            ):
                data = surpyval.univariate.nonparametric.turnbull(
                    self.x, self.c, self.n, self.t, estimator=estimator
                )
                xrd = data["x"], data["r"], data["d"]
            else:
                xrd = surpyval.utils.xcnt_to_xrd(
                    self.x, self.c, self.n, self.t
                )
            self.xrd = xrd
        return self.xrd

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
            handle=False,
        )

    def __iter__(self):
        # allows for unpacking in a function, i.e. fun(*data)
        return [self.x, self.c, self.n, self.t].__iter__()

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
            SurpyvalData(\n\tx={self.x},\n\tc={self.c},\n\tn={self.n},\n\tt={self.t})
        """.strip()
