Torch with SurPyval
===================

If you already train neural networks in `PyTorch <https://pytorch.org>`_,
``surpyval.torch`` gives your network SurPyval's likelihood for lifetime
data, and gives you SurPyval's tools for what it predicts. You keep your
own network, training loop, optimiser and device; SurPyval supplies the
loss and the glue at both ends.

In plain words: a censored or truncated lifetime cannot go into an
ordinary regression loss. A unit still running at 3,000 hours tells you
only that its life is longer than 3,000 hours; a unit found failed at an
inspection, only that it failed between two inspections; a unit you only
heard of because it survived to 1,000 hours, that it comes from a
population cut off at 1,000 hours. Each needs its own term in the
likelihood. ``surpyval.torch`` writes those terms as a differentiable
torch loss, exactly as SurPyval's ``fit`` counts them, for one
distribution per row or a mixture of several (the model of *Deep
Survival Machines*, Nagpal, Li and Dubrawski, 2021).

SurPyval does not install PyTorch and does not depend on it, not even as
an optional extra. ``import surpyval`` and ``import surpyval.torch`` never
import it; the helpers import it when you call them, and if it is not
installed they stop with a message saying to install it (``pip install
torch``, or the build for your machine from pytorch.org).

The helpers are:

- :func:`~surpyval.torch.targets`: lifetime data (failures, right, left and
  interval censoring, left and right truncation, counts), checked as
  ``fit`` checks it, as the tensors of a ``Targets``, one row per unit in
  the order given;
- :func:`~surpyval.torch.constrain`: a network's raw outputs as a
  distribution's parameters, each inside its range;
- :func:`~surpyval.torch.nll`: the exact negative log-likelihood, of one
  distribution or a mixture (``mixture=``), and ``NLLLoss``, the same as a
  ``torch.nn.Module``;
- :func:`~surpyval.torch.sf`: each row's survival curve as an ``(N, T)``
  tensor, what :func:`~surpyval.metrics.score_predictions` scores;
- :func:`~surpyval.torch.to_surpyval`: any row as an ordinary SurPyval
  model.

They support the Weibull, Exponential, LogNormal, Gamma, LogLogistic,
Normal, Gumbel, Logistic and Rayleigh distributions
(``surpyval.torch.SUPPORTED``, the same as the PyMC helpers'); any other is
refused with that list.

What SurPyval does not do
-------------------------

There are no training loops, network architectures, hyperparameter
searches, early stopping or GPU management here: those are yours, and
torch already has them. Nothing in the loss keeps a mixture's component
from collapsing (its weight going to 0, or its scale to a point):
mixtures train slowly and can do so, and how to train them (the learning
rate, the number of components, a prior or penalty of your own, where to
start) is your choice. Train in double precision (``.double()``) when you
want the loss to equal ``fit``'s likelihood: it does to 1e-10 there.
Single precision works, and is not held to that: a window far in a tail
can round to probability 0 in it, an infinite loss.

A mixture network, trained and scored
-------------------------------------

Two thousand units with two covariates fail in one of two ways: an early
mode, more common the larger the first covariate, whose scale also
depends on the second, and a wear-out mode. Each is censored at a random
time. A small network gives each unit a mixture of three Weibulls (one
more than the truth, as in practice you do not know how many), trained on
1,500 units with Adam, and scored on the other 500.

.. code:: python

    import numpy as np
    import torch
    import surpyval as sp
    import surpyval.torch as spt
    from surpyval.metrics import concordance_index, score_predictions

    # Simulated data: two failure modes, whose mix depends on the first
    # covariate, and an early mode whose scale depends on the second.
    rng = np.random.default_rng(2026)
    n = 2000
    X = rng.normal(size=(n, 2))
    early = rng.uniform(size=n) < 1 / (1 + np.exp(-1.5 * X[:, 0]))
    life = np.where(
        early,
        np.exp(1.8 + 0.4 * X[:, 1]) * rng.weibull(1.5, n),
        40.0 * rng.weibull(3.0, n),
    )
    censor = rng.uniform(10, 80, n)
    x, c = np.minimum(life, censor), (life > censor).astype(int)
    train, test = slice(0, 1500), slice(1500, None)

    # Your network: K = 3 weights and K Weibulls' two parameters per row.
    K = 3
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Linear(2, 32), torch.nn.ReLU(), torch.nn.Linear(32, K * 3)
    ).double()
    # Start the components apart: scales at the quartiles of the times,
    # shapes at 1.5 (the bias is the inverse of constrain's softplus).
    start = [np.quantile(x[train], q) for q in (0.25, 0.5, 0.75)]
    with torch.no_grad():
        net[2].bias[K:] = torch.tensor(
            np.log(np.expm1([v for a in start for v in (a, 1.5)]))
        )

    def predict(features):
        out = net(torch.as_tensor(features))
        logits, raw = out[:, :K], out[:, K:].reshape(-1, K, 2)
        return spt.constrain(sp.Weibull, raw), logits

    y = spt.targets(x=x[train], c=c[train])
    loss_fn = spt.NLLLoss(sp.Weibull)
    optimiser = torch.optim.Adam(net.parameters(), lr=0.01)
    for epoch in range(1000):
        optimiser.zero_grad()
        params, logits = predict(X[train])
        loss = loss_fn(y, params, mixture=logits)
        loss.backward()
        optimiser.step()
    print(f"loss per unit: {loss.item():.4f}")

    # Score the held-out rows with SurPyval's metrics.
    times = [5.0, 10.0, 20.0, 40.0]
    with torch.no_grad():
        params, logits = predict(X[test])
        S = spt.sf(sp.Weibull, times, params, mixture=logits).numpy()
    scores = score_predictions(x[test], c[test], S, times)
    print(scores.scores.round(3).to_string(index=False))
    print(concordance_index(x[test], c[test], 1 - S[:, 2]).summary())

    # Any row as an ordinary SurPyval model.
    unit = spt.to_surpyval(sp.Weibull, params, row=0, mixture=logits)
    print(unit.w.round(3), unit.params.round(2).tolist())
    print(unit.sf(times).round(4), S[0].round(4))

Its output, from one run with PyTorch 2.15 on a CPU (PyTorch is not
installed when this documentation is built, so the code on this page is
not run there; another version or machine can differ in the last digits):

.. code:: text

    loss per unit: 2.9315
    metric model  time  estimate    se  lower  upper
     brier model   5.0     0.163 0.010  0.144  0.182
     brier model  10.0     0.195 0.009  0.177  0.212
     brier model  20.0     0.197 0.009  0.180  0.214
     brier model  40.0     0.146 0.011  0.124  0.168
       auc model   5.0     0.752 0.024  0.706  0.799
       auc model  10.0     0.763 0.021  0.721  0.805
       auc model  20.0     0.769 0.022  0.726  0.812
       auc model  40.0     0.657 0.038  0.582  0.731
       ibs model   NaN     0.180 0.006  0.167  0.192
    Harrell's C
    ===========
    C                : 0.673536
    Std. error       : 0.014431
    95% CI           : (0.645252, 0.70182)
    Horizon tau      : none
    Subjects         : 500
    [0.162 0.06  0.777] [[4.93, 2.71], [21.08, 6.05], [39.95, 2.77]]
    [0.8928 0.8207 0.7001 0.2851] [0.8928 0.8207 0.7001 0.2851]

The first held-out unit's mixture puts 0.78 of its weight on a wear-out
mode (scale 40, shape 2.8; the truth's are 40 and 3) and 0.16 on an early
one (scale 4.9, shape 2.7); the third component is small, and its
survival function as a SurPyval model is the row of ``S``. The start
matters: from the network's random bias instead of the quartiles, the
same loop ended at 3.12 per unit, with two of that unit's three
components collapsed (weights below 1e-6, shapes near 0.03). That is for
the training to deal with, not the loss.

What each step does:

- The network ends in ``K * 3`` outputs per unit: ``K`` logits, whose
  softmax is the mixture's weights, and two raw values for each of the
  ``K`` Weibulls, reshaped to ``(N, K, 2)``. :func:`~surpyval.torch.constrain`
  maps those into each parameter's range (a softplus for a positive one),
  giving ``alpha`` and ``beta`` of shape ``(N, K)``.
- ``NLLLoss(sp.Weibull)(y, params, mixture=logits)`` is the negative
  log-likelihood per unit (``reduction="mean"``, divided by the total
  count). Each unit's term is the log of the weighted sum of the
  components' likelihoods of what was seen of it: a density for a
  failure, a survival probability for a unit still running, and the
  mixture's probability of a window for an interval. A truncation window
  divides by the mixture's probability of it.
- :func:`~surpyval.torch.sf` gives the held-out units' survival curves at
  the scoring times, an ``(N, T)`` array once detached:
  :func:`~surpyval.metrics.score_predictions` gives the Brier score at
  each time, the integrated Brier score and the time-dependent AUC, each
  with its standard error and interval, and
  :func:`~surpyval.metrics.concordance_index` Harrell's C of the risk
  ``1 - S(t)`` at a horizon, with its interval.
- :func:`~surpyval.torch.to_surpyval` turns the first held-out unit's
  prediction into a :class:`~surpyval.MixtureModel` (for a single
  distribution, ``dist.from_params(...)``): its ``sf``, ``ff``, ``hf``,
  ``qf``, ``mean`` and ``random`` are SurPyval's, and its survival function
  equals that row of ``sf``.

To draw one unit's curve:

.. code:: python

    import matplotlib.pyplot as plt

    t = np.linspace(0, 80, 200)
    plt.plot(t, unit.sf(t))
    plt.xlabel("time")
    plt.ylabel("probability of surviving")

Data and batches
----------------

``targets`` takes the data in any form ``fit`` does: ``x``, ``c``, ``n``,
``t`` or ``tl``/``tr``, ``xl``/``xr``, or a :class:`~surpyval.SurpyvalData`
(whose rows are its own, grouped and sorted unless built with
``group_and_sort=False``). Row ``i`` of the result is row ``i`` of your
features. A ``Targets`` is a named tuple, so it batches as a tuple does:

.. code:: python

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.as_tensor(X), *y),
        batch_size=256, shuffle=True,
    )
    for features, *fields in loader:
        params, logits = predict(features)
        loss = loss_fn(spt.Targets(*fields), params, mixture=logits)

A parameter can be one value per row (``(N,)``, or ``(N, K)`` in a
mixture), one value shared by every row (a scalar, or ``(K,)`` per
component in a mixture), or a mix: a shared shape and a scale per unit
is ``{"alpha": scale_per_row, "beta": shape_parameter}``. The loss's type
and device are those of the parameters.

One distribution per row is the same without ``mixture=``: ``raw`` of
shape ``(N, k)``, ``constrain`` gives ``(N,)`` parameters, and ``sf`` and
``to_surpyval`` take no logits. A mixture of one component is exactly
that.

The loss
--------

Each row adds one term, times its count ``n``: ``log f(x)`` for a failure,
``log S(x)`` right censored, ``log F(x)`` left censored and ``log[F(xr) -
F(xl)]`` for an interval; a row observed only inside the window ``(tl,
tr]`` subtracts ``log[F(tr) - F(tl)]``. For a mixture, each is the log of
the weighted sum of the components' (worked out in log space), so a
window is the mixture's ``F(b) - F(a)``. Every term is written in a form
that is stable in both tails, and each row is given only the formula of
its own kind: the loss and its gradient are finite wherever the
likelihood is, including where one component of a mixture gives a row
probability 0 and another explains it. The tests hold the loss to
SurPyval's own likelihood (``fit``'s, and ``MixtureModel``'s) to 1e-10 in
double precision, for every family, kind of row and truncation.

The gradient is first-order; for the Gamma distribution, whose
derivative in its shape is numerical, it cannot be differentiated again.
Because each kind of row is gathered on its own, the loss is for eager
torch, not ``torch.vmap`` or tracing.

API
---

.. autofunction:: surpyval.torch.targets

.. autofunction:: surpyval.torch.constrain

.. autofunction:: surpyval.torch.nll

.. autofunction:: surpyval.torch.sf

.. autofunction:: surpyval.torch.to_surpyval

``NLLLoss(dist, reduction="mean")`` is :func:`~surpyval.torch.nll` as a
``torch.nn.Module``: ``NLLLoss(dist)(target, params, mixture=None)`` is
``nll(dist, target, params, mixture=mixture)``. It is built when first
used, so that ``import surpyval.torch`` does not import torch.
