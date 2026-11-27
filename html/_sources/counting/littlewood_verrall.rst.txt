Littlewood-Verrall
==================

The Littlewood-Verrall software reliability model: the time to each
failure is exponential with a rate that is itself random (gamma), lower
on average after each fix, so a fix may make the software better or
worse. Fitted to one test campaign's times between failures, each of
which is then Pareto. It has no number of faults; it gives the
distribution of the time to the next failure and of the rate now.

A fit is refused where the gaps are no more scattered than exponential
times would be (``alpha`` then has no finite estimate).

.. autodata:: surpyval.recurrent.software.littlewood_verrall.LittlewoodVerrall
   :no-value:

   .. automethod:: surpyval.recurrent.software.littlewood_verrall.LittlewoodVerrall.fit
   .. automethod:: surpyval.recurrent.software.littlewood_verrall.LittlewoodVerrall.fit_from_df
   .. automethod:: surpyval.recurrent.software.littlewood_verrall.LittlewoodVerrall.from_params

.. autoclass:: surpyval.recurrent.software.littlewood_verrall.LittlewoodVerrallModel
   :members:
