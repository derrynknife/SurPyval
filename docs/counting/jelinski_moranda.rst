Jelinski-Moranda
================

The Jelinski-Moranda software reliability model: a whole number of
faults ``N``, each found at the same rate ``phi``, so the failure rate
steps down by ``phi`` at every failure. It is fitted to one test
campaign's failure times (or the times between them). It is not a
Poisson process: for the NHPP growth models see
:doc:`goel_okumoto` and its neighbours.

The estimate of ``N`` is a whole number, and its bounds are
likelihood-ratio sets of whole numbers; an upper bound of ``inf`` says
the data do not limit the number of faults. Data that show no
reliability growth have no finite estimate at all, and ``fit`` refuses
them, saying what was seen and what to fit instead.

.. autodata:: surpyval.recurrent.software.jelinski_moranda.JelinskiMoranda
   :no-value:

   .. automethod:: surpyval.recurrent.software.jelinski_moranda.JelinskiMoranda.fit
   .. automethod:: surpyval.recurrent.software.jelinski_moranda.JelinskiMoranda.from_params

.. autoclass:: surpyval.recurrent.software.jelinski_moranda.JelinskiMorandaModel
   :members:
