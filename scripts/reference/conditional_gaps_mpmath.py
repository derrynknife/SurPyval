"""High-precision references for the mixture's cumulative hazard and hazard
rate in the tails, and for the residual-life draws past the quantile limit
(``conditional_gaps`` inverting ``H`` by Newton's method).

It prints the values typed into
``surpyval/tests/univariate/parametric/test_mixture_hazards.py`` and
``surpyval/tests/recurrent/test_conditional_gaps_newton.py``, computed
with mpmath at 50 significant digits:

* ``MIXTURE_TAIL``: ``Hf`` and ``hf`` of two mixtures at times from the
  body to where ``H`` is about 700 (the survival is 1e-304 there, so
  ``-log sf`` from the summed survivals has long since failed);
* ``ROOTS``: for a lifetime, a virtual age ``v`` and a uniform ``u``, the
  age ``x`` with ``H(x) = H(v) - log(u)``.

Every time and parameter is a double; the references are at those
doubles. Run from the repository root, with mpmath installed (it is not
a SurPyval dependency; use a separate environment)::

    python scripts/reference/conditional_gaps_mpmath.py
"""

from typing import Any

from mpmath import erfc, exp, findroot, log, mp, mpf, pi, sqrt

mp.dps = 50


def weibull(alpha: Any, beta: Any) -> Any:
    a, b = mpf(alpha), mpf(beta)

    def H(x: Any) -> Any:
        return (x / a) ** b

    def h(x: Any) -> Any:
        return b / a * (x / a) ** (b - 1)

    return H, h


def lognormal(mu: Any, sigma: Any) -> Any:
    m, s = mpf(mu), mpf(sigma)

    def S(x: Any) -> Any:
        return erfc((log(x) - m) / (s * sqrt(2))) / 2

    def H(x: Any) -> Any:
        return -log(S(x))

    def h(x: Any) -> Any:
        z = (log(x) - m) / s
        return exp(-z * z / 2) / (sqrt(2 * pi) * s * x) / S(x)

    return H, h


def normal(mu: Any, sigma: Any) -> Any:
    m, s = mpf(mu), mpf(sigma)

    def S(x: Any) -> Any:
        return erfc((x - m) / (s * sqrt(2))) / 2

    def H(x: Any) -> Any:
        return -log(S(x))

    def h(x: Any) -> Any:
        z = (x - m) / s
        return exp(-z * z / 2) / (sqrt(2 * pi) * s) / S(x)

    return H, h


def mixture(components: Any, weights: Any) -> Any:
    w = [mpf(v) for v in weights]

    def H(x: Any) -> Any:
        return -log(sum(wi * exp(-c[0](x)) for wi, c in zip(w, components)))

    def h(x: Any) -> Any:
        s = sum(wi * exp(-c[0](x)) for wi, c in zip(w, components))
        f = sum(wi * c[1](x) * exp(-c[0](x)) for wi, c in zip(w, components))
        return f / s

    return H, h


WEIBULL_MIX = mixture([weibull(60, 0.8), weibull(2000, 3.5)], [0.25, 0.75])
LOGNORMAL_MIX = mixture([lognormal(1, 0.5), lognormal(3, 1.2)], [0.4, 0.6])


def root(H: Any, h: Any, v: Any, u: Any) -> Any:
    target = H(mpf(v)) - log(mpf(u))
    x = mpf(v) * 1.5
    # Bracket, then Newton's method in mpmath.
    lo, hi = mpf(v), mpf(v) + max(abs(mpf(v)), 1)
    while H(hi) < target:
        lo, hi = hi, hi + 2 * (hi - mpf(v))
    x = findroot(lambda t: H(t) - target, (lo, hi), solver="anderson")
    x = findroot(lambda t: H(t) - target, x, solver="newton")
    return x


def main() -> None:
    print("MIXTURE_TAIL = [  # (name, x, Hf, hf)")
    tails = {
        "weibull": (
            WEIBULL_MIX,
            [1.0, 50.0, 1000.0, 5000.0, 2e4, 1e5, 2.2e5],
        ),
        "lognormal": (
            LOGNORMAL_MIX,
            [0.5, 3.0, 50.0, 1e4, 1e8, 1e12, 1e16, 1e20],
        ),
    }
    for name, ((H, h), xs) in tails.items():
        for x in xs:
            print(
                "    ({!r}, {!r}, {!r}, {!r}),".format(
                    name, x, float(H(mpf(x))), float(h(mpf(x)))
                )
            )
    print("]")
    print("ROOTS = [  # (name, v, u, x)")
    cases = [
        ("mixture", WEIBULL_MIX, 5000.0, [0.9, 0.3, 0.01]),
        ("lognormal", lognormal(5, 0.8), 40000.0, [0.9, 0.3, 0.01]),
        ("lognormal", lognormal(5, 0.8), 100000.0, [0.5]),
        ("normal", normal(10, 3), 40.0, [0.9, 0.3, 0.01]),
    ]
    for name, (H, h), v, us in cases:
        for u in us:
            print(
                "    ({!r}, {!r}, {!r}, {!r}),".format(
                    name, v, u, float(root(H, h, v, u))
                )
            )
    print("]")


if __name__ == "__main__":
    main()
