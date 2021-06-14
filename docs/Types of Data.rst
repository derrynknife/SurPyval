
Types of Data
=============

Survival analysis is the statistics about durations. To understand durations, or time to events, we must have data that captures how long something lasts. This is the start of survival analysis where we have data in the form of a list of durations of some time to event. This time to event can be engineering failure data, health data on time to death from a given disease, economic data on the duration of a recession or time between recessions, or it could be race times for a group of athletes in a triathlon.

Survival analysis is unique in statistics because of the types of data that we encounter, specifically censoring and truncation. The purpose of this section is to explain these types of data and the scenarios under which they are generated so that you can understand when you might need to use the different flags in surpyval in your analysis.

Exactly Observed
----------------

The first type of data is exactly observed data. This is the type of data where we know exactly when the death or failure occurred. For example, if I run a test on how long some light bulbs will last, I get 5 and turn them on and watch them continuously. Then as each fail I record their failure times 983, 1321, 1889, 1923, and 2932 hours. Each of these times is exact because I saw the exact moment at which they failed. So the task is then understanding the distribution of these exact failures.

Censored Data
-------------

Say that I got bored of sitting and looking at light bulbs. And because of this boredome I stopped looking at the light bulbs at 1900 hours. I would therefore not have seen two of the light bulbs fail, the failures that would have occured at 1923 and 2932 hours. All we would know about these two light bulbs is that they failed sometime *after* 1,900 hours. That is, we know that these two light bulbs would have failed if they continued the test but that this faliure time is greater than 1,900. This is to say that the failure time has been *censored*. Specifically the failure has been right censored. 'Right' is used because if we consider a (horizontal) timeline with time progressing along the line from left to right, we know that the failure would have occured to the right of the time at which we stopped our observation. Hence, the observation is right censored.

If on the other hand, I also knew it would take some time for the bulbs to start failing so instead of waiting from the very start of the test I did not sit there for the first 1,000 hours. That is, the test continues to run but is not being observed for the first 1,000 hours, then after the first 1,000 hours I return to the test and start my observations. When I return I find that a bulb has failed. From the original data, I see that there was a bulb that failed at 983 hours. But if I was not observing for the first 1,000 hours all I would know about this failure is that it occurred sometime *before* 1,000 hours. Using the timeline concept again, I know that the failure would have occurred to the left of the 1,000 hour mark. Therefore, we say that the failure is *left* censored.

Finally, had I not been patient enough to sit down for any extended period of time and instead inspected the light bulbs at different times to see if any had failed. So say I inspect the bulbs every 100 hours from 1000 hours till 2,000 hours. The first and last failures would be left and right censored. But the middle failures would be known to fail between inspections. So the second failure would have occurred between the 1300 and 1400 hours inspections, the third between 1800 and 1900, and the second last failure would have happened between the 1900 and 2000 hours inspections. These failures are said to be *intervally* censored. That is because they are known to have happened in a given interval on a timeline.

Survival analysis has several methods for handling censored data in the parametric and non-parametric analysis. Surpyval is able to handle an input that has an arbitrary combination of observed and left, right, and intervally censored failure data. Although, not all methods can handle all types of data. This is covered in the sections on each of the estimation and fitting methods.

Surpyval uses a convention regarding censoring. Specifically, surpyval takes as input, with a list of failure times 'x', an optional censoring flag array 'c'. If no flaggin array is provided, it is assumed that all the data are exact observations, i.e. that they are not censored. But if the 'c' array is provided, it must have a value for each value in the x input. That is, they must be the same length. The possible values of c are -1, 0, 1, and 2. The convention tries to illustrate the concept of left, right, and interval censoring on the timeline. That is, -1 is the flag for left censoring because it is to the left of an observed failure. With an observed failure at 0. 1 is used to flag a value as right censored. Finally, 2 is used to flag a value as being intervally censored because it has 2 data points, a left and right point. In practice this will therefore look like:

.. code:: python

	import surpyval

	x = [3, 3, 3, 4, 4, [4, 6], [6, 8], 8]
	c = [-1, -1, -1, 0, 0, 2, 2, 1]

	model = surpyval.Weibull.fit(x=x, c=c)

This example shows the flexibility surpyval offers. It allows users to analyse data that has any arbitrary combination of the different types of censoring. The surpyval format is even more powerful, because the above example can be condensed even further through using the 'n' value.

.. code:: python

	import surpyval

	x = [3, 4, [4, 6], [6, 8], 8]
	c = [-1, 0, 2, 2, 1]
	n = [3, 2, 1, 1, 1]

	model = surpyval.Weibull.fit(x=x, c=c, n)

The first step of the fit method actually wrangles the input data into the densest form possible. So internally, the example without the n value, will be condensed to be the second example without you seeing it. But it shows the capability of how data can be input to surpyval if you have different formats. But we are getting away from data types...


Truncated Data
--------------

For my light bulb test, let's say I test a different manufacturers bulbs. This time, I know that the bulbs from this manufacturer have been tested for 500 hours prior to shipping them. This situation needs to be treated differently because we know that in this circumstance we only have the bulbs because they survived more than 500 hours. If there were any failures prior to 500 hours the bulb would not have been shipped and therefore would not be being tested by me. This is to say, that my observation of the distribution of the light bulb failures has been *truncated*. In this regime there is no way I can have any observation below 500 hours because of the testing then discarding done by the manufacturer. The astute reader might have observed that this data is in fact *left* truncated. This is because the truncation occurs to the left of the observation on a timeline. In this example, all the bulbs are left truncated at the 500 hour mark. 

In biostatistics left truncation is known as 'late-entry', this is because in clinical trials a participant can enter a trial later than other participant. Therefore this participant was at risk of not being present in the trial. This is because they could have died prior to entering the trial. Morbid, yes, but the estimate of the distribution needs to account for this risk otherwise the estimate will overestimate the true risk of the event.

Right truncated data is when you only observe a value because it happened below some time. For example, in the light bulb experiment, I received some of the bulbs that passed the burn in test. That is, I received some of the bulbs that survived the original 500 hours of testing. But if the failed bulbs were then given to an engineering team to investigate possible design changes that will improve reliability; they will have a series of failure times that must be below 500 hours. That is, from their perspective, they have data that is right truncated. There is one condition to this situation, they must not know how many other bulbs were tested. If they knew how many other bulbs were tested, they would know how many would fail after 500 hours. That is, they would know that all the other bulbs are right censored. So for our engineers investigating the failed bulbs, they must be ignorant of how many other bulbs were actually tested for the right truncation to work for them. In many applications we do know how many were under test and therefore right truncation become right censoring, but from our engineers circumstance, we can see that they are right censored.

Parametric and non-parametric analysis can both handle left truncated data. This is explained further in the estimation methods for both these methods. Right truncation can only be handles in surpyval with parametric analysis, specifically, with Maximum Likelihood Estimation and in limited cases with Minimum Product Spacing. This is also explained in their respective sections of these notes. 

In surpyval, passing truncated data to the fitting method looks like:

.. code:: python

	import surpyval

	x  = [674, 792, 1153, 1450, 1555, 1923, 2019]
	tl = [500, 500, 500, 500, 500, 500, 500]

	model = surpyval.Weibull.fit(x=x, tl=tl)


Concluding Points
-----------------

Having read through the above explanation you might be thinking how often these scenarios appear in real data, if ever. The vast majority of data used in survival analysis is observed or right censored. This is what happens when you observe a whole population but finish the observation before the event happens on all the items being observed. 

Right truncation is extremely rare because it only happens if you do not know the size of the whole population under test. It can happen with scientific instruments where say, a camera is limited in the frequencies of light it can capture. So if we were to try capture a distribution of light of an object, say a star, this distribution could be truncated above and below certain frequencies. Meeker and Escobar provide an example in their book on reliability statistics for warranty analysis, similar to the contrived example provided above. If you have some returns of products from the field, these are right-truncated because you do not know what has been bought and used in the field. A more realistic example could be the estimation of race finish times at a triathlon or marathon. If I arrive at the finish line of a race and record the times of participants as they cross the line during that window I will have truncated data. I do not know how many people started the race (presumably) and I only stay and watch for a given period of time, therefore all the observations I make are truncated within the window of my observation time. In conclusion though, right truncation in survival analysis is rare.

Left truncation is common in insurance studies. If an insurance company wants to estimate the distribution of losses due to property crime based on policy payouts they need to consider the impact of 'excess'. Excess is the cost of making a claim on an insurance policy. So if I have an insurance policy with an excess of $500, if I lose $20,000 worth of peoperty in a robbery I will have to pay $500 to be paid $20,000. Because of this, it is clear that if I lost $400 in a robbery I would not pay the $500 excess to make a claim. Therefore the distribution of property crime will be truncated by the value of the excesses on the policies. Actuaries need to consider this in their calculaitons of policy fees.

Insurance is also a good example of right censoring. An insurance policy will also have a maximum payout. So if calculating the distribution of the value of property crime an analyst will need to consider that those payouts that are at the maximum of that policy value are in fact censored. That is, the value of the loss or damage was greater than the actual payout and therefore the payout is a censored value. In the classic Boston housing pricing data there is censored data! A histogram of the values of houses shows that there is a large number of houses at the highest price. This can be understood because a limit was set on the highest possible value, therefore these house prices are actually censored, not exact observations.
