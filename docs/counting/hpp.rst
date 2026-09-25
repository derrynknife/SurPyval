HPP
===

The Homogeneous Poisson Process: events occur at a constant rate
:math:`\lambda` whatever the system's age, so the expected number of
events by time :math:`t` is :math:`\Lambda(t) = \lambda t`. It is the
recurrent-event counterpart of the Exponential distribution, and the
null model that the :doc:`trend tests <trend_tests>` test against.
``HPP.fit`` returns a
:doc:`ParametricRecurrenceModel <parametric_recurrence_model>`; the
methods below that take a ``rate`` are the process's functions at a
given rate.

.. autodata:: surpyval.recurrent.parametric.hpp.HPP
   :no-value:

   .. automethod:: surpyval.recurrent.parametric.hpp.HPP.fit
   .. automethod:: surpyval.recurrent.parametric.hpp.HPP.fit_from_recurrent_data
   .. automethod:: surpyval.recurrent.parametric.hpp.HPP.from_params
   .. automethod:: surpyval.recurrent.parametric.hpp.HPP.cif
   .. automethod:: surpyval.recurrent.parametric.hpp.HPP.iif
   .. automethod:: surpyval.recurrent.parametric.hpp.HPP.log_iif
   .. automethod:: surpyval.recurrent.parametric.hpp.HPP.inv_cif
