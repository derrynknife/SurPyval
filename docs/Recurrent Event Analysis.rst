Counting Process Analysis
=========================

This section is about those events that occur but do not "kill" the subject. 
For example, in engineering an item may fail and be repaired instead of replaced.
In medicine, a patient may have multiple heart attacks which are not fatal.
In these cases, the event of interest is the time between events, which is the
subject of single event survival analysis. In single event survival analysis
one is interested in only the time to the first, and only, event. In counting
process analysis one is interested in the time to the first, second, third, etc.
event. We therefore need a way to describe how many events have occurred by time t.
To do this we use a counting process.

Recurrent events are also known as counting models or point processes. This is because
the model is based on the number of events that occur in a given time period. i.e. 
it counts the number of times an event of interest will repeatedly occur in a given time period.

This section is split into two subsections:

    1. Recurrent Event Modelling
    2. Renewal Modelling

These are both types of counting models but they are different in their methods.
It is therefore worth splitting them into two sections. In a recurrent event model
the inter-arrival time is based on some underlying hazrd rate, whereas for a renewal
model there is an underlying distribution where, after each event, the apparent
age of the subject is changed.

Recurrent Event Modelling
-------------------------

Recurrent event models aim to find a rate at which events occur. This is done by
estimating the hazard rate of the process. This can be done parametrically or
non-parametrically.

Non-Parametric - Mean Cumulative Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mean cumulative function is the average number of events that have occurred
by time t. It is defined as:

.. math::

    MCF(x) = \sum_{i:x_{i} \leq x}^{} \frac{d_{i} }{r_{i}}

This equation looks the same as the cumulative hazard estimate of the Nelson-Aalen estimator. 
That's because it is, the only difference is that when we do this for single events
the number of items at risk decreases after each failure. Whereas with recurrent
events the item continues to be observed until more events occur. Therefore the 
number of items in the risk set does not always decrease after each observed event.
Whereas when doing recurrent event analysis an item will remain in the at risk
set since it will be healed/repaired and returned to service/health to continue with life.

The MCF function is the non-parametric estimator of the number of events that
will occur up to x. So once we fit a model we can then estimate the expected
number of events that will occur up to time x by using the MCF function.

Non-Parametric estimation in counting processes has the same limitations as does
single event survival analysis. The main one being that it is not possible to
extrapolate an estimate of the MCF function beyond the last observed event. This
is because when doing non-parametric analysis we make no assumptions about the
shape of the curve and cannot therefore extrapolate beyond the last observed event.


Parametric Recurrent Event Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SurPyval has several parametric models for recurrent event analysis. These can 
be categorised as either:

    - Homogeneous Poisson Process, or
    - Non-Homogeneous Poisson Process

The Homogeneous Poisson Process is one where the rate of events is constant over time.
This is the simplest model. A Non-Homogeneous Poisson Process is one where the
rate of events is not constant over time.

SurPyval has a series of available Non-Homogeneous Poisson Process models available
for use. These are:

    - Duane
    - Cox-Lewis
    - AMSAA
    - Crow-AMSAA

See the API documentation to see the details of each of these models. The key
point is that each of these are a parametric representation of the hazard rate
of the process.

Parametric counting processes can be estimated in similar ways to single event
survival analysis. That is, we can use a simple mean square error estimation
or, more powerfully, we can use maximum likelihood estimation. The latter is
the default in SurPyval.

MLE for recurrent event models is done by understanding that the likelihood
between events is actually just the conditional likelihood of the next event
given the previous event. That is the equivalent to a left truncated
observation.

The benefit of parametric models is that they can be extrapolated beyond the
last observed event. This is because we have made assumptions about the shape
of the curve and can therefore extrapolate beyond the last observed event.

Renewal Modelling
-----------------

A renewal model is one where recurrent events occur after some intervention has
restored some of the life of the subject. This is different to a recurrent event
model where the events occur at some underlying rate. In a renewal model the
events occur at some underlying distribution. After each event the apparent age
of the subject is changed.

In engineering applications a renewal process is one where the item is repaired
and not replaced. A renewal model captures the effectiveness of the repair
process. In medicine, a renewal process is one where the patient is treated to
lessen the impact of some disease, in this case a renewal model captures the
effectiveness of the treatment.

In surpyval there are two types of renewal models, these are:

    - Generalised Renewal Process
    - G1 Renewal Process

Generalised Renewal Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Generalised Renewal Process (GRP) is one where the intervention can cause the
item to be:

    - as-good-as-new
    - as-bad-as-old
    - better-than-old-but-worse-than-new
    - worse-than-old

This is possible since by using a notion of "virtual age." The virtual age can
be understood to be the apparent age of the subject after the nth intervention.
An engineering example is the best to illustrate this point. Consider a pump
that has been operating for 1000 hours and then has a failure. After a technician
fixes the issue, the pump is returned to service.

If the pump was repaired
perfectly it could be considered as-good-as-new and the virtual age would be
zero. That is, the apparent age of the pump would effectively be as if were straight
out of the factory.

If the pump repair was only
minimal then the virtual age would be exactly 1000 since the pump can be treated
as if it were exactly the same as it was prior to the repair, just without the 
fault that was fixed. 

If the pump was well repaired, but not perfectly, then the apparent age of the
pump would be somewhere between 0 and 1000.

And finally, if the intervention
repaired the fault while doing some other damage, the age of the pump could be
above 1000.

A GRP has two ways that the virtual age can be calculated. These are known as
the Kijima-I and Kijima-II models.

The Kijima-I model assumes that the repair can only repair damage incurred
since the last repair. Mathematically, this is defined as:

.. math::

    V_{n} = V_{n-1} + qt_{n}

Where :math:`V_{n}` is the virtual age after the intervention, :math:`V_{n-1}`
is the virtual age just prior to the intervention, :math:`t_{n}` is the time
since the previous event, i.e. the inter-arrival time, and finally :math:`q` is
the effectiveness of the repair.

The Kijima-II model assumes that the repair can repair all damage incurred
since the item was new. Mathematically, this is defined as:

.. math::

    V_{n} = q(V_{n-1} + t_{n})

Each term here has the same meaning as the Kijima-I model.

Note that if :math:`q = 0` then the Kijima-I and Kijima-II models are
equivalent. That is the virtual age is always zero. This means that the
intervention is always perfect. If :math:`q = 1` then the Kijima-I and Kijima-II
models are also equivalent. That is the virtual age is always the same as the
actual age. This means that the intervention is always as-bad-as-old.

If :math:`q` is anything other than 0 or 1 then the models are different.
However, if :math:`0 < q < 1` then the interventionm according to both models
is better-than-old-but-worse-than-new. Finally if :math:`q > 1` then the
intervention make the item worse-than-old, that is "older" than it was when
it failed.

Both these options are available in SurPyval. But both have a shortcoming which
is that they cannot handle cases where the repair makes the item
better-than-new. To do this you will need to use the G1 Renewal Process.

G1 Renewal Process
~~~~~~~~~~~~~~~~~~

A G1 Renewal Process is one where the intervention can be better-than-new. The
G1 renewal process is able to do this by changing the life parameter of the
underlying distribution. This is in contrast to the GRP which alters the age
whereas the G1 process alters the remaining life.

The G1 Renewal process is able to do this by changing the life parameter of the
underlying distribution. This is done by using a transformation of the life
parameter after an event. The transformation is defined as:

.. math::

    \alpha_{i} = \alpha(1 + q)^{i - 1}

Where :math:`\alpha` is the life parameter of a location-scale distribution and
:math:`q` is the effectiveness of the intervention. Unlike what is possible 
with the G-Renewal process, if q is greater than zero the model captures the
behaviour of when the intervention can improves the life of the subject. If 
:math:`q = 0` then the repair is as-good-as-new. If :math:`q < 0` then the
intervention is restoring some life but not to make it as good as new. Note
that :math:`q` cannot be less than -1.

As an example, if the life parameter of an items was 100 hours and the repair
effectiveness was -0.2, then after the first repair the next time to an event
would have a life parameter of 80 hours. After the second repair the life
parameter would be 64 hours. After the third repair the life parameter would be
51.2 hours. And so on.

If the repair effectiveness was 0.2 then after the first repair the next time
to an event would have a life parameter of 120 hours. After the second repair
the life parameter would be 144 hours. After the third repair the life would be
172.8 hours. And so on.

The ability of the G1 Renewal Process to capture the behaviour of when the
intervention can improve the life of the subject is the reason why it is
a useful model to have available.


Parameter Estimation
--------------------

As with regular survival analysis there are several ways one can estimate the
parameters of the models. Mean Square Error (MSE) is quite straight forward
for both renewal and recurrence models. The complication with recurrence models
is that you have to do a monte carlo simulation for each set of parameters. This
can get quite time consuming and expensive. But it does work.

Maximum Likelihood Estimation also provides an excellent way to estimate
parameters. It is a relatively straight forward logical step from single 
event survival analysis to multiple events. For the first event, the likelihood
of that particular event is the same as for regular survival analysis. For 
recurrent events we tend to use the hazard rate, as this captures the full
nature of the circumstances, to define the model. The likelihood therefore is:

.. math::

    \ell = 