import importlib.resources

import numpy as np
import pandas as pd

from surpyval.utils.recurrent_utils import handle_xicn

data_module = importlib.import_module("surpyval.datasets")


def load_bofors_steel():
    """
    Returns a Pandas DataFrame containing the data of
    the tensile strength of Bofors Steel from [1]_.

    First 5 rows of the dataset:

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

    with importlib.resources.path(
        data_module, "bofors_steel.csv"
    ) as data_path:
        return pd.read_csv(data_path, engine="python")


def load_boston_housing():
    """
    The Boston house-price data of [2]_.

    This is a well-known data set used in machine learning. It can be
    analysed with survival analysis methods by considering the
    fact that the highest prices appear to be right censored.

    References
    ----------

    .. [2] Harrison, D. and Rubinfeld, D.L. (1978) Hedonic
              prices and the demand for clean air. J. Environ.
              Economics and Management 5, 81-102.

    """
    with importlib.resources.path(data_module, "boston.csv") as data_path:
        return pd.read_csv(data_path, engine="python")


def load_bearing_failures():
    """
    Data on the failure of bearings, from [3]_. "Cycles to Failure (millions)"
    is the number of cycles to failure in millions of cycles.

    References
    ----------
    .. [3] Lieblein, J. and Zelen, M. (1956) Statistical Investigation
           of the Fatigue Life of Deep-Groove Ball Bearings. Journal of
           Research of the National Bureau of Standards, 57, 273-315.
           http://dx.doi.org/10.6028/jres.057.033
    """

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
    return pd.DataFrame({"Cycles to Failure (millions)": x})


def load_heart_transplants():
    """
    Data on the survival of patients who may or may not have received a
    heart transplant, from [4]_.

    References
    ----------
    .. [4] Crowley, J. and Hu, M. (1977) Covariance analysis of heart
           transplant survival data. Journal of the American Statistical
           Association, 72, 27-36.
    """

    with importlib.resources.path(data_module, "heart.csv") as data_path:
        return pd.read_csv(data_path, engine="python")


def load_lung():
    """

    Data on the survival of patients with advanced lung cancer from [5]_.

    References
    ----------

    .. [5] Loprinzi CL. Laurie JA. Wieand HS. Krook JE. Novotny PJ.
           Kugler JW. Bartel J. Law M. Bateman M. Klatt NE. et al.
           Prospective evaluation of prognostic variables from
           patient-completed questionnaires. North Central Cancer
           Treatment Group. Journal of Clinical Oncology. 12(3):601-7, 1994.
    """

    with importlib.resources.path(data_module, "lung.csv") as data_path:
        return pd.read_csv(data_path, engine="python")


def load_rossi_static():
    """
    Data on the recidivism of released prisoners from [6]_. Uses only
    static covariates.

    References
    ----------

    .. [6] Rossi, P.H., R.A. Berk, and K.J. Lenihan (1980).
           Money, Work, and Crime: Some Experimental Results. New York:
           Academic Press. John Fox, Marilia Sa Carvalho (2012).
           The RcmdrPlugin.survival Package: Extending the R Commander
           Interface to Survival Analysis. Journal of Statistical Software,
           49(7), 1-32.
    """

    with importlib.resources.path(data_module, "rossi.csv") as data_path:
        return pd.read_csv(data_path, engine="python")


def load_rossi_time_varying():
    """
    Data on the recidivism of released prisoners from [6]_. Includes time
    varying covariates.

    References
    ----------

    .. [6] Rossi, P.H., R.A. Berk, and K.J. Lenihan (1980).
           Money, Work, and Crime: Some Experimental Results. New York:
           Academic Press. John Fox, Marilia Sa Carvalho (2012).
           The RcmdrPlugin.survival Package: Extending the R Commander
           Interface to Survival Analysis. Journal of Statistical Software,
           49(7), 1-32.
    """

    with importlib.resources.path(data_module, "rossi_tv.csv") as data_path:
        return pd.read_csv(data_path, engine="python")


def load_tires_data():
    """
    Data on the survival of tires from [7]_.

    References
    ----------

    .. [7] Krivtsov, V.V., Tananko, D.E., Davis, T.P. (2002).
           Regression approach to tire reliability analysis.
           Reliability Engineering and System Safety, 78(3), 267-273.
    """

    with importlib.resources.path(data_module, "tires.csv") as data_path:
        return pd.read_csv(data_path, engine="python")


def load_mettas_and_zhao():
    """
    Data on the survival of a repairable system from [8]_.

    References
    ----------
    .. [8] Mettas, A. and Zhao, Y.Q. (2005).
              Modeling and analysis of repairable systems with general repair.
                IEEE Transactions on Reliability, 54(1), 1-10.
    """
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

    return pd.DataFrame({"x": x, "i": i, "c": c})

    def __repr__(self):
        return """
        Data from:
        "Modeling and Analysis of Repairable Systems with General Repair."
        Mettas and Zhao (2005).
        """.strip()


def load_g1_kaminskiy_krivtsov():
    """
    Data on the survival of a repairable system from [9]_.

    References
    ----------

    .. [9] Kaminskiy, M.P. and Krivtsov, V.V. (2010).
              G1-renewal process as repairable system model.
              Reliability Engineering and System Safety, 95(1), 1-9.
    """
    x = np.array([3, 6, 11, 5, 16, 9, 19, 22, 37, 23, 31, 45]).cumsum()

    return pd.DataFrame({"x": x})
