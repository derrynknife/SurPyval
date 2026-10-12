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
every univariate fitter builds one, and the parametric, non-parametric
and regression ``fit`` methods accept one in place of the arrays (``Weibull.fit(data)``). Its constructors
``from_observations`` and ``from_xrd`` read the ``fsli`` and ``xrd``
layouts, and ``to_xcnt``, ``to_observations`` and ``to_xrd`` write them. ``RecurrentEventData`` holds recurrent-event data
(each row an event of an item); build it from one
:class:`~surpyval.utils.item_history.ItemHistory` per item
(``RecurrentEventData.from_items``), from dictionaries keyed by item
(``from_histories``), or from arrays with :func:`handle_xicn
<surpyval.utils.recurrent_utils.handle_xicn>`.

.. autoclass:: surpyval.utils.surpyval_data.SurpyvalData
   :members:

.. autoclass:: surpyval.utils.recurrent_event_data.RecurrentEventData
   :members:

.. autoclass:: surpyval.utils.item_history.ItemHistory

.. autoclass:: surpyval.utils.item_lifetime.ItemLifetime

Data Wrangling Utilities
------------------------

Validate data in one format and convert it to another. The methods of
``SurpyvalData`` above are the way to do this; these functions are what
they use. Their top-level names (``surpyval.fs_to_xcnt`` and so on) are
deprecated and removed in the first release on or after 2027-04-12 (see
:doc:`Data Wrangler Examples` for the replacement of each);
``surpyval.utils`` keeps them.

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

Random draws follow one seeding rule (see :doc:`Conventions`); SurPyval
turns every ``random_state`` or ``seed`` argument into a generator with:

.. autofunction:: surpyval.utils.rng.as_generator

Lower-level helpers in ``surpyval.utils``, used by the handlers above:

.. autofunction:: surpyval.utils.xcnt_sort

.. autofunction:: surpyval.utils.group_xcnt

.. autofunction:: surpyval.utils.coerce_xcnt_x

.. autofunction:: surpyval.utils.format_truncation

.. autofunction:: surpyval.utils.is_missing_event

.. autofunction:: surpyval.utils.missing_events

.. autofunction:: surpyval.utils.resolve_cr_censoring

The functions above documented as ``surpyval.utils.<name>`` are what that
namespace promises (its ``__all__``). It also holds the input validators the fitters call
(``check_*`` and ``validate_*`` functions such as ``validate_coxph`` and
``validate_cr_inputs``, and ``optional_column``, ``validate_1d`` and
``validate_float_array``). They are internal: their
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
