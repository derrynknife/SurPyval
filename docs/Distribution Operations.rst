Distribution Operations
=======================

In some cases you might be interested in some function of two outcomes. For example, in engineering you might know the distribution of possible stresses that an item will undergo and a distribution of strength of the item. In this case you want to know the probability that the strength will exceed the stress. In another case, you might want to know how long two steps in a process will take. In this case you want to know what the estimated time is for the two distributions, one distribution for step one and another distribution for step 2. SurPyval makes it extremely easy to get these distributions, as simple as arithmetic. For example:

.. code:: python

    import surpyval as surv

    stress = surv.Weibull.from_params([10., 2.])
    strength = surv.Weibull.from_params([14, 2.5])

    diff = strength - stress

    print(type(diff))
    print(diff.sf(0.))

.. code:: text

    <class 'surpyval.parametric.convolution.ConvolutionModel'>
    0.6942630843163992


In the above example we determined the probability that the stress would exceed the strength of the item. We calculated that the probability of survival (i.e. the stress not exceeding the strength) was approx 70%.

You can see that the regular `sf()` method worked on the 'ConvolutionModel' class. All the regular methods will work with this class, that is, the `df`, `ff`, `hf`, and `Hf` methods should all work.

The solution for the distributions is calculated using numerical integration (with the exception of the addition of two normal distributions). The results therefore do need to be checked to ensure that they are reasonable for the situattion. Scipy will through warnings in the event that there might have been some issues with convergence.

As another example, we might be interested in how long it takes to get through a queue. Alternately, an engineer might want to know how long it would take for a item, and a standby item to fail. In both cases it is the sum of two distributions.

.. code:: python

    import surpyval as surv

    item_1 = surv.Weibull.from_params([2., 0.5])
    item_2 = surv.Weibull.from_params([3, 2.5])

    both = item_1 + item_2

    print(both.sf(10.))

.. code:: text

    0.1499541692145563

Therefore we can say that there is a 15% chance that the queue will take longer than 10 (in whatever units) or that the system has a 15% chance of working past 10. Amazing!

In the theme of reliability we may also want to know the probability of survival for a group of items in "series". In series meaning that if one of them fails, the whole system will fail. We might also want to consider the probability of failure for items that work in parallel.

SurPyval has a very simple syntax to capture these situations. For example, to get a model of three items in series we can:

.. code:: python

    import surpyval as surv

    item_1 = surv.Weibull.from_params([2., 0.5])
    item_2 = surv.Weibull.from_params([3, 2.5])
    item_3 = surv.Weibull.from_params([5, 8.5])

    series = item_1 | item_2 | item_3

    print(series.sf(1.), item_1.sf(1.), item_2.sf(1.), item_3.sf(1.))

.. code:: text

    [0.46243098] 0.4930686913952398 0.9378642812813465 0.9999988551338509

We can see therefore that the chance of surviving to 1. for the whole system is approximately 46%. This is lower than the probability of each of the individual items. That is becauase the probability of survival at a given time is the product of the probability of survival to that time for all items in the series. This means that for each new item in the series it will increase the chance that the whole system will break.

If we want three items in parallel we can use the `&` operation instead.

.. code:: python

    parallel = item_1 & item_2 & item_3

    print(parallel.sf(1.), item_1.sf(1.), item_2.sf(1.), item_3.sf(1.))

.. code:: text

    [0.99999996] 0.4930686913952398 0.9378642812813465 0.9999988551338509

We can see therefore that the chance of surviving to 1 for the whole system is very close to 1. This is higher than the probability of each of the individual items. That is becauase the probability of failure at a given time is the product of the probability of failure at that time for all items that are in parallel. This means that the probability of failure of the whole system will decrease with each additional item in parallel.