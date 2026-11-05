Joint Likelihood Contours
=========================

The joint likelihood-ratio confidence region of the two parameters of a
fit: ``model.likelihood_contour(alpha_ci, n_points)`` finds its boundary
and ``surpyval.plot_likelihood_contour`` draws one or more of them, at one
or several levels, with each estimate marked (for example, the regions of
two designs on one plot). Worked examples are in
:doc:`../Parametric SurPyval Modelling`.

.. autofunction:: surpyval.plot_likelihood_contour

.. autoclass:: surpyval.LikelihoodContour
   :members:

.. autoclass:: surpyval.LikelihoodContourPlot
