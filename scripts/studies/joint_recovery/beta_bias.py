"""How biased is the Weibull shape MLE alone with the study's size of data:
300 items, beta 2, censored at 730 days, alpha chosen so 74% fail (the
joint studies' 223 events of 300)? 2000 reps."""

import warnings

import numpy as np

from surpyval import Weibull

warnings.simplefilter("ignore")
rng = np.random.default_rng(7)
alpha = 730 / (-np.log(0.26)) ** 0.5
betas = []
for _ in range(2000):
    t = alpha * rng.weibull(2.0, 300)
    c = (t > 730).astype(int)
    x = np.minimum(t, 730.0)
    betas.append(Weibull.fit(x, c=c).params[1])
b = np.array(betas)
mcse = b.std(ddof=1) / np.sqrt(b.size) / 2
print(
    f"events {np.mean(t <= 730):.2f}; beta mean {b.mean():.4f}, rel bias "
    f"{b.mean() / 2 - 1:+.4f} (MC se {mcse:.4f})"
)
