# SurPyval
Survival Statistics Package

Have decided to create a new package that contains the survival statistics elements of the RePyability repo. That repo will remain but will focus on developing tools specific to reliability engineering.

The surpyval package provides methods to create models of survival type scenarios.

These modes are broken into:
- [NonParametric](NonParametric.ipynb), and
- [Parametric](Parametric.ipynb).

The question is why is this package needed when we already have scipy?

Two reasons, firstly, scipy doesn't deal with 'censoring' very well (so far as I know). And secondy, perhaps the larger driving reason, is that I wanted to see if I could make a python package.

