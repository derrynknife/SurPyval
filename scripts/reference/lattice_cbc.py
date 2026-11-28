"""Build the rank-1 lattice rule of ``copula/_rectangle.py``.

The generating vector is built component by component (CBC) for the
Korobov space of smoothness 2 with product weights ``1 / sqrt(j + 1)``,
by the fast construction of Nuyens and Cools (2006): the search over each
component is a circular convolution over the multiplicative group of the
integers modulo the (prime) number of points, done by FFT.

    python scripts/reference/lattice_cbc.py 65521 14

prints the vector that ``_LATTICE_Z`` holds.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np


def _primitive_root(n: int) -> int:
    phi = n - 1
    factors, m, p = set(), phi, 2
    while p * p <= m:
        while m % p == 0:
            factors.add(p)
            m //= p
        p += 1
    if m > 1:
        factors.add(m)
    for g in range(2, n):
        if all(pow(g, phi // q, n) != 1 for q in factors):
            return g
    raise ValueError(f"{n} has no primitive root")


def cbc(n: int, dims: int) -> list:
    """The generating vector of ``dims`` components for ``n`` (prime)
    points."""

    def omega(x: Any) -> Any:
        return 2.0 * np.pi**2 * (x * x - x + 1.0 / 6.0)

    def gamma(j: int) -> float:
        return 1.0 / np.sqrt(1.0 + j)

    g = _primitive_root(n)
    perm = np.empty(n - 1, dtype=np.int64)
    perm[0] = 1
    for i in range(1, n - 1):
        perm[i] = perm[i - 1] * g % n
    fpsi = np.fft.fft(omega(perm / n))
    k = np.arange(n)
    z = [1]
    p = 1.0 + gamma(0) * omega(k / n)
    for s in range(1, dims):
        pp = p[perm]
        conv = np.real(np.fft.ifft(np.conj(np.fft.fft(pp)) * fpsi))
        error = (p[0] * (1 + gamma(s) * omega(0.0)) + pp.sum()) / n + (
            gamma(s) * conv / n
        )
        zs = int(perm[int(np.argmin(error))])
        z.append(zs)
        p = p * (1.0 + gamma(s) * omega((k * zs % n) / n))
    return z


if __name__ == "__main__":
    print(cbc(int(sys.argv[1]), int(sys.argv[2])))
