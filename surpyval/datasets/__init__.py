import importlib.resources

import numpy as np
import pandas as pd

from surpyval.utils import fs_to_xcnt

data_module = importlib.import_module("surpyval.datasets")


def load_bearing_failures() -> pd.DataFrame:
    """
    Data on the failure of bearings, from [1]_. "Cycles to Failure (millions)"
    is the number of cycles to failure in millions of cycles.

    The 23 values are all exact failure times: there is no censoring
    column, so pass the column as ``x`` alone.

    Examples
    --------
    >>> import surpyval
    >>> from surpyval.datasets import load_bearing_failures
    >>> x = load_bearing_failures()["Cycles to Failure (millions)"]
    >>> surpyval.Weibull.fit(x).params.round(3)
    array([81.875,  2.102])

    References
    ----------
    .. [1] Lieblein, J. and Zelen, M. (1956) Statistical Investigation
           of the Fatigue Life of Deep-Groove Ball Bearings. Journal of
           Research of the National Bureau of Standards, 57, 273-315.
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


def load_bofors_steel() -> pd.DataFrame:
    """
    Returns a Pandas DataFrame containing the data of
    the tensile strength of Bofors Steel from [2]_.

    Grouped data in ten rows: ``x`` is the strength and ``n`` the number
    of specimens with that strength, so pass the columns as ``x`` and
    ``n``. Every value is an exact observation.

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
    .. [2] Weibull, W., A statistical distribution function
           of wide applicability, Journal of applied mechanics,
           Vol. 18, No. 3, pp 293-297 (1951).

    """

    data_path = importlib.resources.files(data_module) / "bofors_steel.csv"
    return pd.read_csv(data_path)


def load_boston_housing() -> pd.DataFrame:
    """
    The Boston house-price data of [3]_.

    This is a well-known data set used in machine learning. It can be
    analysed with survival analysis methods by considering the
    fact that the highest prices appear to be right censored.

    One row per census tract (506), with the thirteen covariates of the
    original data and ``medv``, the median house value in thousands of
    dollars. ``medv`` is capped at 50: the 16 tracts at 50.0 are right
    censored there. There is no censoring column; build one from the cap.

    References
    ----------

    .. [3] Harrison, D. and Rubinfeld, D.L. (1978) Hedonic
              prices and the demand for clean air. J. Environ.
              Economics and Management 5, 81-102.

    """
    data_path = importlib.resources.files(data_module) / "boston.csv"
    return pd.read_csv(data_path)


def load_g1_kaminskiy_krivtsov() -> pd.DataFrame:
    """
    Data on the survival of a repairable system from [4]_.

    One system with twelve failures. ``x`` is the *cumulative* time of
    each failure (the times between failures are 3, 6, 11, 5, 16, 9, 19,
    22, 37, 23, 31 and 45); every row is an observed failure and there is
    no item or censoring column.

    References
    ----------

    .. [4] Kaminskiy, M.P. and Krivtsov, V.V. (2010).
           G1-renewal process as repairable system model.
           Reliability Engineering and System Safety, 95(1), 1-9.
    """
    x = np.array([3, 6, 11, 5, 16, 9, 19, 22, 37, 23, 31, 45]).cumsum()

    return pd.DataFrame({"x": x})


def load_heart_transplants() -> pd.DataFrame:
    """
    Data on the survival of patients who may or may not have received a
    heart transplant, from [5]_.

    Start-stop form: each patient (``id``) has one row before a
    transplant and, if transplanted, one after (``transplant`` 1), over
    ``(start, stop]``; ``event`` is 1 for a death at ``stop`` (so
    ``c = 1 - event``). There are 172 rows for 103 patients, 75 of whom
    died. ``age`` is the age at acceptance minus 48 years, ``year`` the
    date of acceptance in years after 1 November 1967 and ``surgery`` 1
    for prior bypass surgery. ``Unnamed: 0`` is a row index.

    References
    ----------
    .. [5] Crowley, J. and Hu, M. (1977) Covariance analysis of heart
           transplant survival data. Journal of the American Statistical
           Association, 72, 27-36.
    """

    data_path = importlib.resources.files(data_module) / "heart.csv"
    return pd.read_csv(data_path)


def load_lung() -> pd.DataFrame:
    """

    Data on the survival of patients with advanced lung cancer from [6]_.

    ``time`` is the survival time in days. ``status`` is coded as
    SurPyval's censoring flag -- 0 for a death, 1 for a patient censored
    (alive at last follow-up) -- so use it directly as ``c``. There are
    228 patients, 165 of whom died. The other columns are the covariates
    of the original data (``inst``, ``age``, ``sex`` with 1 male and 2
    female, ``ph.ecog``, ``ph.karno``, ``pat.karno``, ``meal.cal`` and
    ``wt.loss``, several with missing values); ``Unnamed: 0`` is a row
    index.

    Examples
    --------
    >>> import surpyval
    >>> from surpyval.datasets import load_lung
    >>> df = load_lung()
    >>> km = surpyval.KaplanMeier.fit(df["time"], c=df["status"])
    >>> km.sf([365]).round(3)
    array([0.409])

    References
    ----------

    .. [6] Loprinzi CL. Laurie JA. Wieand HS. Krook JE. Novotny PJ.
           Kugler JW. Bartel J. Law M. Bateman M. Klatt NE. et al.
           Prospective evaluation of prognostic variables from
           patient-completed questionnaires. North Central Cancer
           Treatment Group. Journal of Clinical Oncology. 12(3):601-7, 1994.
    """

    data_path = importlib.resources.files(data_module) / "lung.csv"
    return pd.read_csv(data_path)


def load_mettas_and_zhao() -> pd.DataFrame:
    """
    Data on the survival of a repairable system from [7]_.

    Recurrent event data for six systems: ``x`` is the cumulative time of
    each event on system ``i``, and ``c`` is 1 on each system's last
    row, the end of its observation (0 for a failure).

    References
    ----------
    .. [7] Mettas, A. and Zhao, Y.Q. (2005).
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


def load_rossi_static() -> pd.DataFrame:
    """
    Data on the recidivism of released prisoners from [8]_. Uses only
    static covariates.

    One row per prisoner (432): ``week`` is the week of first arrest after
    release, or 52 for those not arrested in the year of follow-up, and
    the covariates are ``fin`` (financial aid), ``age``, ``race``,
    ``wexp`` (work experience), ``mar`` (married), ``paro`` (released on
    parole) and ``prio`` (number of prior convictions).

    Note that in this copy ``arrest`` is coded as SurPyval's censoring
    flag -- 1 for a prisoner *not* arrested (right-censored at week 52),
    0 for an arrest -- the reverse of the original data. Use it directly
    as ``c``. (The two leading ``Unnamed`` columns are row indices.)

    References
    ----------

    .. [8] Rossi, P.H., R.A. Berk, and K.J. Lenihan (1980).
           Money, Work, and Crime: Some Experimental Results. New York:
           Academic Press. John Fox, Marilia Sa Carvalho (2012).
           The RcmdrPlugin.survival Package: Extending the R Commander
           Interface to Survival Analysis. Journal of Statistical Software,
           49(7), 1-32.
    """

    data_path = importlib.resources.files(data_module) / "rossi.csv"
    return pd.read_csv(data_path)


def load_rossi_time_varying() -> pd.DataFrame:
    """
    Data on the recidivism of released prisoners from [9]_. Includes time
    varying covariates.

    Start-stop (counting-process) form: one row per prisoner-week, with
    ``id`` the prisoner, ``start`` and ``stop`` the week's interval,
    ``event`` 1 if the prisoner was arrested at the end of that week (so
    ``c = 1 - event``), and ``employed`` the time-varying covariate. Here
    ``arrest`` keeps the original coding (1 = arrested during follow-up),
    unlike :func:`load_rossi_static`.

    References
    ----------

    .. [9] Rossi, P.H., R.A. Berk, and K.J. Lenihan (1980).
           Money, Work, and Crime: Some Experimental Results. New York:
           Academic Press. John Fox, Marilia Sa Carvalho (2012).
           The RcmdrPlugin.survival Package: Extending the R Commander
           Interface to Survival Analysis. Journal of Statistical Software,
           49(7), 1-32.
    """

    data_path = importlib.resources.files(data_module) / "rossi_tv.csv"
    return pd.read_csv(data_path)


def load_tires_data() -> pd.DataFrame:
    """
    Data on the survival of tires from [10]_.

    One row per tire (34). ``Survival`` is the (normalised) time and
    ``Censoring`` is coded as SurPyval's censoring flag -- 1 for a tire
    that had not failed, 0 for a failure (11 of them) -- so use it
    directly as ``c``. The other seven columns are the covariates.

    References
    ----------

    .. [10] Krivtsov, V.V., Tananko, D.E., Davis, T.P. (2002).
           Regression approach to tire reliability analysis.
           Reliability Engineering and System Safety, 78(3), 267-273.
    """

    data_path = importlib.resources.files(data_module) / "tires.csv"
    return pd.read_csv(data_path)


def load_sae() -> pd.DataFrame:
    """
    Data on failures in automotive industry from [11]_.

    Features heavily (right) censored data: ``x`` is the time and ``c``
    SurPyval's censoring flag, with 10 failures (``c = 0``) and 21 right
    censored values (``c = 1``).

    References
    ----------

    .. [11] V.V. Krivtsov and J. W. Case (1999),
            Peculiarities of Censored Data Analysis in Automotive Industry
            Applications, SAE Technical Paper Series, # 1999-01-3220
    """

    f = [5248, 7454, 16890, 17200, 38700, 45000, 49390, 69040, 72280, 131900]
    s = [
        3961,
        4007,
        4734,
        6054,
        7298,
        10190,
        23060,
        27160,
        28690,
        37100,
        40060,
        45670,
        53000,
        67000,
        69630,
        77350,
        78470,
        91680,
        105700,
        106300,
        150400,
    ]

    x = f + s
    c = np.concatenate((np.zeros(len(f)), np.ones(len(s))))
    return pd.DataFrame({"x": x, "c": c})


def load_meeker_lfp() -> pd.DataFrame:
    """
    Data on failures of integrated circuits from [12]_.

    Very difficult for LFP calculations since the data is heavily
    right censored: 4,156 circuits were tested for 1,370 hours and 28
    failed. The data is in ``xcnt`` form: ``x`` the time in hours, ``c``
    the censoring flag and ``n`` the count, with the 4,128 survivors in
    one row right censored at 1,370.

    References
    ----------
    .. [12] Meeker, W.Q. (1987) Limited Failure Population Life Tests:
            Application to Integrated Circuit Reliability. Technometrics,
            29(1), 51-65.
    """

    f = np.array(
        [
            0.1,
            0.1,
            0.15,
            0.6,
            0.8,
            0.8,
            1.2,
            2.5,
            3,
            4,
            4,
            6,
            10,
            10,
            12.5,
            20,
            20,
            43,
            43,
            48,
            48,
            54,
            74,
            84,
            94,
            168,
            263,
            593,
        ]
    )
    total_tested = 4156
    s = np.ones(total_tested - len(f)) * 1370
    x, c, n, _ = fs_to_xcnt(f, s)
    return pd.DataFrame({"x": x, "c": c, "n": n})
