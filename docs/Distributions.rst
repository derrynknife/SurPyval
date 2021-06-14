
Distributions
=============

This section shows the distributions available in SurPyval. Please contact the developers if you would like another one added.

Weibull Distribution
--------------------

.. math::
	R(x) = e^{-{(\frac{x}{\alpha}})^{\beta}}

The Weibull distribution is parameterised by the scale parameter :math:`\alpha` and the shape parameter :math:`\beta`

Gumbel Distribution
--------------------

.. math::
	R(x) = 1 - e^{e^{-(x - \mu)/\sigma}}

The Gumbel distribution is parameterised by the scale parameter :math:`\mu` and the shape parameter :math:`\sigma`

Exponentiated Weibull Distribution
----------------------------------

.. math::
	R(x) = 1 - {(1 - e^{-{(\frac{x}{\alpha}})^{\beta}})}^{\mu}

The Exponentiated Weibull distribution is parameterised by the scale parameter :math:`\alpha` and the shape parameters :math:`\beta` and :math:`\mu`

Gamma Distribution
------------------

.. math::
	R(x) = 1 - \frac{1}{\Gamma(\alpha)}\int_{0}^{x}\beta^{\alpha}x^{\alpha - 1}e^{-\beta x} dx


The Gamma distribution is parameterised by the scale parameter :math:`\alpha` and the shape parameter :math:`\beta`

Beta Distribution
------------------

.. math::
	f(x) = \frac{1}{B(\alpha, \beta)}x^{\alpha - 1}(1 - x)^{\beta - 1}

The Beta distribution is parameterised by the two shape parameters :math:`\alpha` and :math:`\beta`

Logistic Distribution
---------------------

.. math::
	F(x) = \frac{1}{1 + e^{-(x - \mu)/\sigma}}

The Logistic distribution is parameterised by the scale parameter :math:`\mu` and the shape parameter :math:`\sigma`

LogLogistic Distribution
------------------------

.. math::
	F(x) = \frac{1}{1 + (x/\alpha)^{-\beta}}

The LogLogistic distribution is parameterised by the two scale parameter :math:`\alpha` and the shape parameter :math:`\beta`

Normal Distribution
-------------------

.. math::
	f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x - \mu}{\sigma})^{2}}

The Normal distribution is parameterised by the scale parameter :math:`\mu` and the shape parameter :math:`\sigma`

LogNormal Distribution
----------------------

.. math::
	f(x) = \frac{1}{x\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{\textrm{ln}(x) - \mu}{\sigma})^{2}}

The LogNormal distribution is parameterised by the scale parameter :math:`\mu` and the shape parameter :math:`\sigma`


Uniform Distribution
--------------------

.. math::
	F(x) = \frac{x - a}{b - a}

The LogNormal distribution is parameterised by the location parameters :math:`a` and :math:`b`

