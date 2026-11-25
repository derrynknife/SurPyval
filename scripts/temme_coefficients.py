"""Taylor coefficients in eta of Temme's c_k(eta) (DLMF 8.12.8-8.12.11).

The table ``TEMME`` in
``surpyval/univariate/parametric/distributions/_temme.py`` (its first
eight rows, thirty columns). Run from anywhere; it writes
``temme_table.txt`` in the current directory, one row of c_k a line,
and checks that the negative powers of eta cancel in every c_k::

    python scripts/temme_coefficients.py
"""

from typing import Any

import mpmath as mp

mp.mp.dps = 80
K, N = 14, 30
M = N + 2 * K + 6  # working order


def mul(a: Any, b: Any) -> Any:
    out = [mp.mpf(0)] * M
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j in range(M - i):
            out[i + j] += ai * b[j]
    return out


def inv(a: Any) -> Any:  # 1 / series, a[0] != 0
    out = [mp.mpf(0)] * M
    out[0] = 1 / a[0]
    for n in range(1, M):
        out[n] = -sum(a[k] * out[n - k] for k in range(1, n + 1)) / a[0]
    return out


def compose(f: Any, g: Any) -> Any:  # f(g(eta)), g[0] = 0
    out = [mp.mpf(0)] * M
    for c in reversed(f):
        out = mul(out, g)
        out[0] += c
    return out


def sqrt_series(a: Any) -> Any:  # a[0] = 1
    out = [mp.mpf(0)] * M
    out[0] = mp.mpf(1)
    for n in range(1, M):
        out[n] = (a[n] - sum(out[k] * out[n - k] for k in range(1, n))) / 2
    return out


# s(t) = eta / t = sqrt(2 (t - log(1+t)) / t^2)
#      = sqrt(sum_{n>=0} 2 (-1)^n t^n / (n + 2))
inner = [mp.mpf(2) * (-1) ** n / (n + 2) for n in range(M)]
s = sqrt_series(inner)
# t(eta): t = eta / s(t), fixed point on series
t = [mp.mpf(0)] * M
t[1] = mp.mpf(1)
for _ in range(M + 2):
    st = compose(s, t)
    new = mul([mp.mpf(0), mp.mpf(1)] + [mp.mpf(0)] * (M - 2), inv(st))
    t = new
# 1/t as a Laurent series: 1/t = (1/eta) * 1/(t/eta)
tq = t[1:] + [mp.mpf(0)]
inv_tq = inv(tq)  # 1/t = inv_tq / eta  (coeffs of eta^{n-1})
# Stirling Gamma* coefficients:
# Gamma*(a) = exp(sum B_{2n} / (2n (2n-1) a^{2n-1}))
L = [mp.mpf(0)] * M
for n in range(1, M // 2):
    if 2 * n - 1 < M:
        L[2 * n - 1] = mp.bernoulli(2 * n) / (2 * n * (2 * n - 1))
# exp of series with L[0] = 0
E = [mp.mpf(0)] * M
E[0] = mp.mpf(1)
term = [mp.mpf(1)] + [mp.mpf(0)] * (M - 1)
for k in range(1, M):
    term = mul(term, L)
    term = [c / k for c in term]
    E = [e + c for e, c in zip(E, term)]
gam = E  # Gamma*(a) ~ sum gam[k] a^-k


# Laurent: represent as dict power->coeff, power from -1
def laurent_from_inv_t(scale: Any) -> Any:
    return {n - 1: scale * inv_tq[n] for n in range(M)}


c = []
c0 = laurent_from_inv_t(mp.mpf(1))
c0[-1] -= 1
c.append(c0)
for k in range(1, K + 1):
    prev = c[-1]
    # (1/eta) d/deta prev
    d: dict[int, Any] = {}
    for p, v in prev.items():
        if p != 0:
            d[p - 2] = d.get(p - 2, 0) + p * v
    extra = laurent_from_inv_t((-1) ** k * gam[k])
    for p, v in extra.items():
        d[p] = d.get(p, 0) + v
    c.append(d)
for k, ck in enumerate(c):
    neg = {p: v for p, v in ck.items() if p < 0 and abs(v) > mp.mpf(10) ** -60}
    assert not neg, (k, neg)
with open("temme_table.txt", "w") as fh:
    for k, ck in enumerate(c):
        row = [ck.get(n, mp.mpf(0)) for n in range(N)]
        fh.write(", ".join(mp.nstr(v, 20) for v in row) + "\n")
print([mp.nstr(c[0].get(n, 0), 10) for n in range(4)])
print([mp.nstr(c[1].get(n, 0), 10) for n in range(4)])
print([mp.nstr(gam[k], 10) for k in range(5)])
