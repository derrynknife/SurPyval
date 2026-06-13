import json
from numbers import Number
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

import surpyval
from surpyval.utils import xcnt_handler


class SurpyvalData:
    def __init__(
        self,
        x: ArrayLike | None = None,
        c: ArrayLike | None = None,
        n: ArrayLike | None = None,
        t: ArrayLike | None = None,
        xl: ArrayLike | None = None,
        xr: ArrayLike | None = None,
        tl: ArrayLike | Number | None = None,
        tr: ArrayLike | Number | None = None,
        Z: ArrayLike | None = None,
        group_and_sort: bool = True,
        handle: bool = True,
    ) -> None:
        """Initialize a SurpyvalData instance for survival analysis.

        Validates, sorts, and stores survival data in the xcnt format. Supports
        uncensored, right/left/interval-censored, and truncated observations.
        Can convert to xrd format and select subsets by censoring type.

        Parameters
        ----------
        x : array-like, optional
            The primary data array of failure/event times. When c is 2 the
            corresponding x entry is a 2-element array [left, right].
        c : array-like, optional
            Censoring flags for each value in x:
            * 0 = uncensored
            * 1 = right censored
            * -1 = left censored
            * 2 = interval censored
        n : array-like, optional
            Number of occurrences for each value in x.
        t : array-like, optional
            2D array of truncation bounds [left, right] for each value in x.
        xl : array-like, optional
            Left interval bounds for interval censored data.
            Cannot be used with 'x'. Must be paired with 'xr'.
        xr : array-like, optional
            Right interval bounds for interval censored data.
            Cannot be used with 'x'. Must be paired with 'xl'.
        tl : array-like or scalar, optional
            Left truncation bounds. Cannot be used with 't'.
            Must be paired with 'tr'.
        tr : array-like or scalar, optional
            Right truncation bounds. Cannot be used with 't'.
            Must be paired with 'tl'.
        group_and_sort : bool, default=True
            Whether to group and sort the data. Set False when using covariates
            to maintain data order.
        handle : bool, default=True
            Whether to validate and process the input data.
            Set False for pre-validated data.

        Examples
        --------
        Basic usage with uncensored data:
        >>> x = np.array([1, 2, 3])
        >>> data = SurpyvalData(x)

        Right censored data:
        >>> x = np.array([1, 2, 3])
        >>> c = np.array([0, 1, 1])  # 2 and 3 are censored
        >>> data = SurpyvalData(x, c)

        Interval censored data:
        >>> xl = np.array([1, 2, 3])
        >>> xr = np.array([2, 3, 4])
        >>> data = SurpyvalData(xl=xl, xr=xr)

        Interval Censored with nested 2 arrays:
        >>> x = [1, 2, [2, 5], 3, 6]
        >>> c = [0, 1, 2, 0, 0]
        >>> data = SurpyvalData(x=x, c=c)

        With truncation:
        >>> x = np.array([1, 2, 3])
        >>> t = np.array([[0, 5], [0, 5], [0, 5]])
        >>> data = SurpyvalData(x, t=t)
        """

        if Z is not None:
            group_and_sort = False

        if handle:
            x, c, n, t = xcnt_handler(
                x, c, n, t, xl, xr, tl, tr, group_and_sort=group_and_sort
            )

        x_arr: np.ndarray = np.asarray(x)
        c_arr: np.ndarray = np.asarray(c)
        n_arr: np.ndarray = np.asarray(n)
        t_arr: np.ndarray = np.asarray(t)

        self.x, self.c, self.n, self.t = x_arr, c_arr, n_arr, t_arr
        self.tl = self.t[:, 0]
        self.tr = self.t[:, 1]
        if x_arr.ndim == 2:
            self.x_min, self.x_max = np.min(x_arr[:, 0]), np.max(x_arr[:, 1])
        else:
            self.x_min, self.x_max = np.min(x_arr), np.max(x_arr)
        self._split_to_observation_types()
        if Z is not None:
            self.add_covariates(Z)
        else:
            self.Z: np.ndarray | None = None

    def add_covariates(self, Z: ArrayLike) -> None:
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
        Z_arr: np.ndarray = np.asarray(Z)
        self.Z = Z_arr
        self.Z_o = Z_arr[self.c == 0]
        self.Z_r = Z_arr[self.c == 1]
        self.Z_l = Z_arr[self.c == -1]
        self.Z_i = Z_arr[self.c == 2]
        # Covariates of the truncated subset, aligned with x_tl/x_tr/n_t so
        # that regression likelihoods can apply the truncation correction
        # using each truncated observation's own covariates.
        self.Z_t = Z_arr[self.truncated_mask]

    def _split_to_observation_types(self) -> None:
        self.x_o, self.n_o = self._split_by_mask(self.c == 0)
        self.x_r, self.n_r = self._split_by_mask(self.c == 1)
        self.x_l, self.n_l = self._split_by_mask(self.c == -1)
        if self.x.ndim == 2:
            interval_mask = self.c == 2
            self.x_il = self.x[interval_mask, 0]
            self.x_ir = self.x[interval_mask, 1]
            self.n_i = self.n[interval_mask]
        else:
            self.x_il = np.array([], dtype=float)
            self.x_ir = np.array([], dtype=float)
            self.n_i = np.array([], dtype=int)
        self.truncated_mask = np.isfinite(self.t).any(axis=1)
        self.x_tl, self.x_tr, self.n_t = (
            self.t[self.truncated_mask, 0],
            self.t[self.truncated_mask, 1],
            self.n[self.truncated_mask],
        )

    def to_xrd(self, estimator="Nelson-Aalen") -> tuple:
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
                or (2 in self.c)
                or (-1 in self.c)
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

    def _split_by_mask(
        self, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        x = self.x[mask] if self.x.ndim == 1 else self.x[mask, 0]
        return x, self.n[mask]

    def __getitem__(self, index):
        return SurpyvalData(
            np.atleast_1d(self.x[index]),
            np.atleast_1d(self.c[index]),
            np.atleast_1d(self.n[index]),
            np.atleast_2d(self.t[index]),
            handle=False,
        )

    def __len__(self) -> int:
        return len(self.x)

    def __iter__(self):
        return zip(self.x, self.c, self.n, self.t)

    def __repr__(self) -> str:
        return (
            f"SurpyvalData(\n"
            f"    x={self.x!r},\n"
            f"    c={self.c!r},\n"
            f"    n={self.n!r},\n"
            f"    t={self.t!r}\n"
            f")"
        )

    def to_json(self, filepath: str | Path | None = None) -> str | None:
        """
        Serialize SurpyvalData to JSON format.

        Parameters
        ----------
        filepath : str | Path, optional
            If provided, saves JSON to this file path

        Returns
        -------
        str | None
            JSON string if no filepath provided, None if saved to file
        """
        data = {
            "x": self.x.tolist(),
            "c": self.c.tolist(),
            "n": self.n.tolist(),
            "t": self.t.tolist(),
        }

        if self.Z is not None:
            data["Z"] = self.Z.tolist()

        if filepath:
            Path(filepath).write_text(json.dumps(data, indent=2))
            return None

        return json.dumps(data)

    @classmethod
    def from_json(cls, source: str | Path) -> "SurpyvalData":
        """
        Create SurpyvalData instance from JSON string or file path.

        Parameters
        ----------
        source : str | Path
            Pass a ``Path`` to load from a file; pass a ``str`` to parse as
            JSON text directly.

        Returns
        -------
        SurpyvalData
            New instance created from JSON data
        """
        if isinstance(source, Path):
            text = source.read_text()
        else:
            text = source

        data = json.loads(text)

        x = np.array(data["x"])
        c = np.array(data["c"])
        n = np.array(data["n"])
        t = np.array(data["t"])
        Z = np.array(data["Z"]) if "Z" in data else None

        return cls(x, c, n, t, Z=Z, handle=False)
