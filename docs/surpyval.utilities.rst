Utilities
=========

The data containers the fitters build from your arrays, the functions
that convert between SurPyval's data formats, and the abstract base
classes every model derives from. The formats themselves -- ``xcnt``
(times, censoring flags, counts, truncation), ``xrd`` (times, numbers at
risk, deaths) and ``fsli`` (lists of failures, suspensions, left- and
interval-censored values) -- are defined in :doc:`Conventions`, and
:doc:`Data Wrangler Examples` shows them in use.

Data Classes
------------

``SurpyvalData`` holds univariate ``xcnt`` data, split by censoring type;
every univariate fitter builds one, and ``fit_from_surpyval_data``
accepts one directly. ``RecurrentEventData`` holds recurrent-event data
(each row an event of an item); build it with :func:`handle_xicn
<surpyval.utils.recurrent_utils.handle_xicn>`.

.. autoclass:: surpyval.utils.surpyval_data.SurpyvalData
   :members:

.. autoclass:: surpyval.utils.recurrent_event_data.RecurrentEventData
   :members:

Data Wrangling Utilities
------------------------

Validate data in one format and convert it to another. Each of these is
importable directly from ``surpyval``.

.. autofunction:: surpyval.utils.xcnt_handler

.. autofunction:: surpyval.utils.fsli_handler

.. autofunction:: surpyval.utils.xrd_handler

.. autofunction:: surpyval.utils.recurrent_utils.handle_xicn

.. autofunction:: surpyval.utils.fs_to_xcnt

.. autofunction:: surpyval.utils.fsl_to_xcnt

.. autofunction:: surpyval.utils.fsli_to_xcnt

.. autofunction:: surpyval.utils.xcn_to_fs

.. autofunction:: surpyval.utils.fs_to_xrd

.. autofunction:: surpyval.utils.xcnt_to_xrd

.. autofunction:: surpyval.utils.xrd_to_xcnt

.. autofunction:: surpyval.utils.round_sig

Lower-level helpers in ``surpyval.utils``, used by the handlers above:

.. autofunction:: surpyval.utils.xcnt_sort

.. autofunction:: surpyval.utils.group_xcnt

.. autofunction:: surpyval.utils.coerce_xcnt_x

.. autofunction:: surpyval.utils.format_truncation

.. autofunction:: surpyval.utils.is_missing_event

.. autofunction:: surpyval.utils.resolve_cr_censoring

``surpyval.utils`` also holds the input validators the fitters call
(``check_*`` and ``validate_*`` functions such as ``validate_coxph`` and
``validate_cr_inputs``, and ``optional_column``, ``wrangle_Z``,
``validate_1d`` and ``validate_float_array``). They are internal: their
behaviour is covered by the fitters' own documentation, and they may
change without notice.

Abstract Base Classes
---------------------

Every model derives from one of these, exported from ``surpyval``; they
define the minimal interface a model provides and are useful for
``isinstance`` checks and for writing new model types.

.. autoclass:: surpyval.distribution.Distribution
   :members:

.. autoclass:: surpyval.distribution.ParametricDistribution
   :members:

.. autoclass:: surpyval.distribution.NonParametricDistribution
   :members:

.. autoclass:: surpyval.distribution.MultivariateDistribution
   :members:
