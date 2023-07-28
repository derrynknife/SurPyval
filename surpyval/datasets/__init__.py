import importlib.resources

import numpy as np
import pandas as pd

from surpyval.utils.recurrent_utils import handle_xicn

data_module = importlib.import_module("surpyval.datasets")


class BoforsSteel_:
    r"""
    A Class with a Pandas DataFrame containing the data of
    the tensile strenght of Bofors Steel from the Weibull paper [1]_.

    Attributes
    ----------
    df : DataFrame
        A Pandas DataFrame containing the data of
        the tensile strenght of Bofors Steel from the Weibull paper.

    Examples
    --------
    >>> from surpyval import BoforsSteel
    >>> df = BoforsSteel.df
    >>> df.head(5)

    .. list-table::
        :header-rows: 1

        * -
          - x
          - n
        * - 0
          - 40.800
          - 10
        * - 1
          - 42.075
          - 23
        * - 2
          - 43.350
          - 48
        * - 3
          - 44.625
          - 80
        * - 4
          - 45.900
          - 63

    References
    ----------
    .. [1] Weibull, W., A statistical distribution function
           of wide applicability, Journal of applied mechanics,
           Vol. 18, No. 3, pp 293-297 (1951).

    """

    def __init__(self):
        with importlib.resources.path(
            data_module, "bofors_steel.csv"
        ) as data_path:
            self.data = pd.read_csv(data_path, engine="python")

    def __repr__(self):
        return """
        Data from:
        Weibull, W., A Statistical Distribution Function
        of Wide Applicability, Journal of Applied Mechanics,
        Vol. 18, No. 3, pp 293-297 (1951).
        https://doi.org/10.1115/1.4010337
        """


class Boston_:
    def __init__(self):
        with importlib.resources.path(data_module, "boston.csv") as data_path:
            self.data = pd.read_csv(data_path, engine="python")


class Bearing_:
    def __init__(self):
        x = [
            17.88,
            28.92,
            33,
            41.52,
            42.12,
            45.6,
            48.4,
            51.84,
            51.96,
            54.12,
            55.56,
            67.8,
            68.64,
            68.64,
            68.88,
            84.12,
            93.12,
            98.64,
            105.12,
            105.84,
            127.92,
            128.04,
            173.4,
        ]
        self.data = pd.DataFrame({"Cycles to Failure (millions)": x})

    def __repr__(self):
        return """
        Data from:
        Lieblein, J. and Zelen, M. (1956) Statistical Investigation
        of the Fatigue Life of Deep-Groove Ball Bearings. Journal of
        Research of the National Bureau of Standards, 57, 273-315.
        http://dx.doi.org/10.6028/jres.057.033
        """


class Heart_:
    def __init__(self):
        with importlib.resources.path(data_module, "heart.csv") as data_path:
            self.data = pd.read_csv(data_path, engine="python")

    def __repr__(self):
        return """
    Data from:
    Heart transplant data
    """


class Lung_:
    def __init__(self):
        with importlib.resources.path(data_module, "lung.csv") as data_path:
            self.data = pd.read_csv(data_path, engine="python")

    def __repr__(self):
        return """
    Data from:
    Lung data...
    """


class Rossi_:
    def __init__(self):
        with importlib.resources.path(data_module, "rossi.csv") as data_path:
            self.data = pd.read_csv(data_path, engine="python")
        with importlib.resources.path(
            data_module, "rossi_tv.csv"
        ) as data_path:
            self.time_varying_data = pd.read_csv(data_path, engine="python")

    def __repr__(self):
        return """
    Data from:
    Rossi
    """


class Tires_:
    def __init__(self):
        with importlib.resources.path(data_module, "tires.csv") as data_path:
            self.data = pd.read_csv(data_path, engine="python")

    def __repr__(self):
        return """
    Data from:
    V.V Krivtsov, D.E Tananko, T.P Davis,
    Regression Approach to Tire Reliability Analysis,
    Reliability Engineering & System Safety,
    Volume 78, Issue 3, 2002,
    Pages 267-273,
    https://doi.org/10.1016/S0951-8320(02)00169-2.
    """


class RecurrentDataExample1_:
    def __init__(self):
        x = np.array(
            [
                2227.08,
                2733.229,
                3524.214,
                5568.634,
                5886.165,
                5946.301,
                6018.219,
                7202.724,
                8760,
                772.9542,
                1034.458,
                3011.114,
                3121.458,
                3624.158,
                3758.296,
                5000,
                900.9855,
                1289.95,
                2689.878,
                3928.824,
                4328.317,
                4704.24,
                5052.586,
                5473.171,
                6200,
                411.407,
                1122.74,
                1300,
                688.897,
                915.101,
                2650,
                105.824,
                500,
            ]
        )

        c = (
            [0] * 8
            + [1]
            + [0] * 6
            + [1]
            + [0] * 8
            + [1]
            + [0] * 2
            + [1]
            + [0] * 2
            + [1]
            + [0] * 1
            + [1]
        )
        i = [1] * 9 + [2] * 7 + [3] * 9 + [4] * 3 + [5] * 3 + [6] * 2
        n = np.ones_like(x)
        self.data = handle_xicn(x, i, c, n, as_recurrent_data=True)

    def __repr__(self):
        return """
        Data from:
        "Modeling and Analysis of Repairable Systems with General Repair."
        Mettas and Zhao (2005).
        """.strip()


class RecurrentDataExample2_:
    def __init__(self):
        x = np.array([3, 6, 11, 5, 16, 9, 19, 22, 37, 23, 31, 45]).cumsum()

        self.data = handle_xicn(x, as_recurrent_data=True)

    def __repr__(self):
        return """
        Data from:
        "G1-Renewal Process as Repairable System Model."
        Kaminskiy and Krivtsov (2010).
        """.strip()


BoforsSteel = BoforsSteel_()
Bearing = Bearing_()
Boston = Boston_()
Heart = Heart_()
Lung = Lung_()
Rossi = Rossi_()
Tires = Tires_()
RecurrentDataExample1 = RecurrentDataExample1_()
RecurrentDataExample2 = RecurrentDataExample2_()
