Datasets
========


Ball Bearing Failures
---------------------

.. code:: python

	>>> from surpyval.datasets import Bearing

	>>> Bearing.df.info()
	<class 'pandas.core.frame.DataFrame'>
	RangeIndex: 23 entries, 0 to 22
	Data columns (total 1 columns):
	 #   Column                        Non-Null Count  Dtype  
	---  ------                        --------------  -----  
	 0   Cycles to Failure (millions)  23 non-null     float64
	dtypes: float64(1)
	memory usage: 312.0 bytes

+---+----------------------------------+
|   | **Cycles to Failure (millions)** |
+---+----------------------------------+
| 0 |                          17.88   |
+---+----------------------------------+
| 1 |                          28.92   |
+---+----------------------------------+
| 2 |                          33      |
+---+----------------------------------+
| 3 |                          41.52   |
+---+----------------------------------+
| 4 |                          42.12   |
+---+----------------------------------+

.. code:: python

	