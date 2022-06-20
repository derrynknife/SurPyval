import pandas as pd
from pkg_resources import resource_filename


class BoforsSteel_():
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
        self.data = pd.read_csv(resource_filename("surpyval",
                                                "datasets/bofors_steel.csv"),
                              engine="python")
    
    def __repr__(self):
        return """
        Data from:
        Weibull, W., A Statistical Distribution Function
        of Wide Applicability, Journal of Applied Mechanics,
        Vol. 18, No. 3, pp 293-297 (1951).
        https://doi.org/10.1115/1.4010337
        """


class Boston_():
    def __init__(self):
        self.data = pd.read_csv(resource_filename("surpyval",
                                                "datasets/boston.csv"),
                              engine="python")


class Bearing_():
    def __init__(self):
        x = [17.88, 28.92, 33, 41.52, 42.12, 45.6, 48.4, 51.84,
             51.96, 54.12, 55.56, 67.8, 68.64, 68.64, 68.88, 84.12,
             93.12, 98.64, 105.12, 105.84, 127.92, 128.04, 173.4]
        self.data = pd.DataFrame({'Cycles to Failure (millions)': x})

    def __repr__(self):
        return """
        Data from:
        Lieblein, J. and Zelen, M. (1956) Statistical Investigation
        of the Fatigue Life of Deep-Groove Ball Bearings. Journal of
        Research of the National Bureau of Standards, 57, 273-315.
        http://dx.doi.org/10.6028/jres.057.033
        """

class Heart_():
  def __init__(self):
    self.data = pd.read_csv(resource_filename("surpyval",
                                                "datasets/heart.csv"),
                              engine="python")
  
  def __repr__(self):
    return """
    Data from:
    Heart transplant data
    """

class Lung_():
  def __init__(self):
    self.data = pd.read_csv(resource_filename("surpyval",
                                                "datasets/lung.csv"),
                              engine="python")
  
  def __repr__(self):
    return """
    Data from:
    Lung data...
    """

class Rossi_():
  def __init__(self):
    self.data = pd.read_csv(resource_filename("surpyval",
                                                "datasets/rossi.csv"),
                              engine="python")
    self.time_varying_data = pd.read_csv(resource_filename("surpyval",
                                                "datasets/rossi_tv.csv"),
                              engine="python")
  
  def __repr__(self):
    return """
    Data from:
    Rossi
    """

class Tires_():
  def __init__(self):
    self.data = pd.read_csv(resource_filename("surpyval",
                                              "datasets/tires.csv"),
                            engine="python")
  
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


BoforsSteel = BoforsSteel_()
Bearing = Bearing_()
Boston = Boston_()
Heart = Heart_()
Lung = Lung_()
Rossi = Rossi_()
Tires = Tires_()
