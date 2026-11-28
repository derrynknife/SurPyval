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
layouts, and ``to_xcnt``, ``to_observations`` and ``to_xrd`` write them.

With each row's unit ``i``, a ``SurpyvalData`` also holds recurrent-event
data (each row an event of a unit, or the end of its observation), which
every recurrent ``fit`` takes. Build it from the rows
(``SurpyvalData(x, c, i=i)``), from one
:class:`~surpyval.utils.item_history.ItemHistory` per item
(``SurpyvalData.from_items``), from dictionaries keyed by item
(``SurpyvalData.from_histories``) or from a log in a DataFrame
(``SurpyvalData.from_dataframe``); ``to_first_events`` and ``to_gaps``
give lifetime data from it.

``EventHistoryData`` is the older class for the same data. It is
deprecated in v0.30 and removed in v0.31: building one yourself (its
constructor, ``from_items``, ``from_histories``, ``from_dataframe``,
``from_dict`` and ``from_json``, :func:`handle_xicn
<surpyval.utils.recurrent_utils.handle_xicn>` and
``Items.event_history_data()``) warns (``FutureWarning``) and names the
``SurpyvalData`` call that does the same. Its ``to_surpyval_data()`` gives
the same histories as a ``SurpyvalData``.

.. autoclass:: surpyval.utils.surpyval_data.SurpyvalData
   :members:

.. autoclass:: surpyval.utils.event_history_data.EventHistoryData
   :members:

.. autoclass:: surpyval.utils.item_history.ItemHistory
   :members:

.. autoclass:: surpyval.utils.item_lifetime.ItemLifetime
   :members:

``NevadaTable`` reads a warranty table of shipments by return period (a
Nevada chart) into lifetime data, writes it back from a list of returns,
and forecasts the survivors' returns (see "Warranty returns from a Nevada
table" in :doc:`Parametric SurPyval Modelling`).

.. autoclass:: surpyval.utils.nevada.NevadaTable
   :members:

``ConfidenceBounds`` is the array a two-sided bound comes back as where
its columns are labelled (``NonParametricCounting.mcf_cb`` and
``CauseSpecificMCF.mcf_cb``): ``[lower, upper]`` on the last axis, as
every two-sided bound (see :doc:`Conventions`), with the columns also
named, ``.lower`` and ``.upper``, and printed with their names and level.
Anything computed from it is a plain array.

.. autoclass:: surpyval.utils.bounds.ConfidenceBounds
   :members: wrap
   :no-inherited-members:
   :exclude-members: lower, upper

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

Dates to durations: start and end dates (a blank end date meaning the unit is
still running) to durations in a unit you choose and the censoring flags.

.. autofunction:: surpyval.utils.dates_to_durations

Unit labels (see :doc:`Conventions`): how a model stores the ``unit=`` it
was given, and whether two labels name the same unit.

.. autofunction:: surpyval.utils.normalise_unit

.. autofunction:: surpyval.utils.same_unit

Random draws follow one seeding rule (see :doc:`Conventions`); SurPyval
turns every ``random_state`` or ``seed`` argument into a generator with:

.. autofunction:: surpyval.utils.rng.as_generator

Helpers in ``surpyval.utils`` for the missing-event convention (see
:doc:`Conventions`). The handlers' internal steps (``xcnt_sort``,
``group_xcnt``, ``coerce_xcnt_x``, ``format_truncation`` and
``resolve_cr_censoring``) were public until v0.29; ``SurpyvalData`` runs
them:

.. autofunction:: surpyval.utils.is_missing_event

.. autofunction:: surpyval.utils.missing_events

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
