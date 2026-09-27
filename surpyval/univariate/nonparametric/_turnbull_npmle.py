r"""Does the truncated Turnbull NPMLE exist, and is it unique?

Decided from the data alone, before the EM runs (#327).

Setting
-------
The Turnbull pieces that can carry mass are the *atoms* ``0 .. K-1``: the
pieces inside at least one observation's support (mass anywhere else only
lowers the likelihood, and the EM keeps it at zero). Observation ``i``
has a support ``S_i = [lo_i, hi_i]``, the atoms its event may be in, inside
its truncation window ``W_i = [wl_i, wh_i]``, the atoms at which it could
have been observed at all; both are runs of atoms. An exact observation has
``lo_i == hi_i``; an untruncated one has ``W_i = [0, K-1]``. With masses
``p`` the log-likelihood is

.. math::

    \ell(p) = \sum_i n_i \left[\log p(S_i) - \log p(W_i)\right],

a sum of ratios, unchanged by rescaling ``p``. It is defined where every
``p(S_i) > 0``. The question is whether the supremum of :math:`\ell` is
attained there (the NPMLE *exists*), and if so whether the survival curve
it gives is unique.

Without truncation there are no denominators, :math:`\ell` is continuous on
the compact simplex and the NPMLE always exists. Under truncation the
supremum can sit where some ``p(W_i)`` falls to zero, where the ratio is a
limit ``0/0`` the simplex does not contain: the EM then climbs towards it
without settling, the fitted mass drifting onto the boundary (#308).

1. Existence (all data shapes)
------------------------------
Call a set ``A`` of atoms *closed* if every observation whose window meets
``A`` has a support that meets ``A``.

**Theorem 1.** If every non-empty closed set meets every window, the
NPMLE exists.

*Proof.* Take a maximising sequence and a limit point ``p*`` (the simplex
is compact); let ``A`` be its support. If some window met ``A`` while its
support did not, that ratio would tend to zero and the log-likelihood to
``-inf``, so ``A`` is closed. By assumption every window meets ``A``, so
every ``p*(W_i) > 0`` and every ``p*(S_i) > 0`` (``A`` is closed): the
likelihood is continuous at ``p*`` and attains the supremum there.

The condition, called E below, is checked exactly and cheaply. Its failure
is witnessed by a closed ``A`` missing some window ``W_j``; the atoms
reachable from ``S_j`` by the rule "if ``S_i`` lies inside, add ``W_i``"
stay outside ``A``, and since each added window overlaps the set built so
far (``S_i`` is inside both) they form a run ``[L, R]``. So E fails exactly
when some proper run ``[L, R]`` contains a support and is *stable*: every
support inside it has its window inside it too. :func:`_has_stable_run`
searches all ``R`` at once in ``O((N + K) log K)``.

2. Exact observations: Vardi (1985) and Wang (1991)
---------------------------------------------------
When every support is a single atom, E is the strong connectivity of
Vardi's graph (an edge from ``a`` to ``b`` when some observation observable
at ``a`` failed at ``b``): a closed set is a set no edge leaves. Split the
atoms into *blocks*, the runs over which windows overlap; observations in
different blocks share no atom, so the likelihood factorises and the mass
of one block relative to another is free. Their theorem then reads:

- every block satisfies E: the NPMLE exists, and is unique only if there is
  a single block (otherwise "not unique");
- some block fails E (is connected but not strongly connected): it does
  not exist. Scaling up the mass of a sink component, one that some edge
  enters, raises that edge's ratio and lowers none.

3. One-sided truncation (left *or* right), any censoring
--------------------------------------------------------
With left truncation only, every window is a suffix ``[w_i, K-1]``. Write
the likelihood in the discrete hazards ``h_k = p_k / p([k, K-1])``: an
observation contributes

.. math::

    \prod_{w_i \le k < lo_i} (1 - h_k) \times
    \Big[1 - \prod_{lo_i \le k \le hi_i} (1 - h_k)\Big],

a polynomial on the cube ``[0, 1]^K``, so a maximiser ``h*`` always exists.
The NPMLE in ``p`` exists iff some maximiser has ``h_k < 1`` for every
``k < max w``: at ``h_k = 1`` the survival is zero before a later entry.
Call ``k`` a *gap* if every observation that has entered by ``k`` has its
support starting at or before ``k`` (no one entered is known to survive
past it). At a non-gap some observation carries ``(1 - h_k)``, so
``h*_k < 1``; at a gap nothing penalises ``h_k`` and the likelihood is
non-decreasing in it. Gaps before the last entry are exactly the failures
of E (the atoms up to a gap form a closed set that misses the last entry's
window). Let ``L* = max lo`` (the last support start). Then:

- **no gap before the last entry**: E holds, the NPMLE exists;
- **a gap k < max w that some support containing it ends before L***:
  does not exist. At any candidate maximiser every ``h_j``, ``j < L*``, is
  below 1, so that support's bracket strictly increases in ``h_k`` while
  nothing decreases: raising ``h_k`` always helps;
- **otherwise**: exists but is not unique. Setting ``h_{L*} = 1`` costs
  nothing (no support starts after ``L*``), after which every support
  containing a gap also contains ``L*`` and the likelihood no longer
  depends on the gap hazards at all: any value below 1 gives a maximiser,
  each with a different survival curve after the gap.

Right truncation alone is the mirror image. The first case is a sample
counterpart of Woodroofe's (1985) condition that every time carrying mass
be observable with positive probability. The second is the collapse of
#308 (the six-point reproducer has a gap at the first piece after the first
entry, inside a left-censored support that ends at 2, long before the last
exact failure) and the Kaplan-Meier with delayed entry that drops to zero
when everyone at risk fails before the next unit enters. The third needs a
censored support running across the gap: a left- or interval-censored one,
or a right-censored one from a unit censored before the next entry, which
runs on through the gap (one unit entered at 0 and censored at 1, another
entered at 5 and failed at 6: nothing fixes how much probability lies in
``(1, 5]``; the Kaplan-Meier estimate with delayed entry is the maximiser
that puts none there).

4. Two-sided windows with censoring
-----------------------------------
The Vardi-Wang condition is stated for exact observations, and the hazard
argument needs one-sided windows. With both, existence can depend on the
counts ``n`` and not only on which sets contain which atoms, so no purely
structural rule decides every case. Blocks of this kind satisfying E exist
(Theorem 1). Otherwise one sufficient certificate of non-existence is
checked: an atom ``a`` that every window containing it also has in its
support (mass at ``a`` raises every ratio it enters and lowers none), inside
the support of an observation whose window minus support contains some
other observation's whole support (so that ratio is below 1 at every
admissible ``p``); adding mass at ``a`` then always helps. Remaining blocks
are reported as "undetermined".

What is not claimed: "exists" rules out a free relative scale between
parts of the data, the non-uniqueness these arguments are about. It does
not rule out the usual Turnbull ambiguity of where inside a piece the mass
sits (the ``R_upper`` and ``R_lower`` range), nor a ridge of maxima with
the same likelihood, which interval-censored data can have (Gentleman and
Geyer, 1994) and two of the 176 "exists" samples below did.

Verification
------------
The run search agrees with brute-force enumeration of closed sets on all
1,151 random small structures tried (left, right and two-sided windows).
Against what the EM does from several starts (drifting to the boundary,
the smallest window mass shrinking tenfold or more between 1,000 and
10,000 iterations; settling; or settling in different places with equal
likelihood): 240 samples of 30 left-truncated observations with all four
censoring types gave 63 "does not exist", all drifting; 1 "not unique",
settling in different places; 176 "exists", none drifting. 280 samples of
other shapes (Kaplan-Meier data, interval censoring, right and double
truncation) and 251 random small structures agreed the same way, except
that of 9 "undetermined" (two-sided windows with censoring) 6 drifted and
3 did not, which is why that case is left open. These samples were rerun
after the supports and windows were corrected (#368); 450 more small data
sets with every censoring type under left, right and double truncation,
run through an EM on a likelihood written directly from the data rather
than from these runs of pieces, agreed too: 169 "does not exist", all
drifting; 230 "exists" or "not unique", none drifting.

References
----------
Vardi, Y. (1985). Empirical distributions in selection bias models. Ann.
Statist. 13, 178-203.

Wang, M.-C. (1991). Nonparametric estimation from cross-sectional survival
data. J. Amer. Statist. Assoc. 86, 130-143.

Woodroofe, M. (1985). Estimating a distribution function with truncated
data. Ann. Statist. 13, 163-177.

Frydman, H. (1994). A note on nonparametric estimation of the distribution
function from interval-censored and truncated observations. J. R. Statist.
Soc. B 56, 71-74.

Alioum, A. and Commenges, D. (1996). A proportional hazards model for
arbitrarily censored and truncated data. Biometrics 52, 512-524.

Hudgens, M. G. (2005). On nonparametric maximum likelihood estimation with
interval censoring and left truncation. J. R. Statist. Soc. B 67, 573-587.

Gentleman, R. and Geyer, C. J. (1994). Maximum likelihood for interval
censored data: consistency and computation. Biometrika 81, 618-623.
"""

import numpy as np
import numpy.typing as npt

EXISTS = "exists"
NOT_UNIQUE = "not unique"
DOES_NOT_EXIST = "does not exist"
UNDETERMINED = "undetermined"

# Worst first: a data set's verdict is the worst of its blocks'.
_SEVERITY = {EXISTS: 0, NOT_UNIQUE: 1, UNDETERMINED: 2, DOES_NOT_EXIST: 3}


def _paint(
    start: npt.NDArray,
    end: npt.NDArray,
    value: npt.NDArray,
    size: int,
    op: np.ufunc,
    fill: int,
) -> npt.NDArray:
    """Reduce ``value`` with ``op`` over every run ``[start, end]``.

    Returns, for each position, ``op`` of the values of the runs that
    contain it (``fill`` where none does). Each run is written into the
    two power-of-two blocks that cover it, and the blocks are then pushed
    down one level at a time: ``O((N + size) log size)``.
    """
    out = np.full(size, fill, dtype=np.int64)
    if start.size == 0:
        return out
    levels = max(int(size).bit_length(), 1)
    table = np.full((levels, size), fill, dtype=np.int64)
    # frexp gives length = m * 2**e with m in [0.5, 1), exactly for ints.
    k = np.frexp((end - start + 1).astype(float))[1] - 1
    op.at(table, (k, start), value)
    op.at(table, (k, end - (1 << k) + 1), value)
    for level in range(levels - 1, 0, -1):
        half = 1 << (level - 1)
        table[level - 1] = op(table[level - 1], table[level])
        table[level - 1, half:] = op(
            table[level - 1, half:], table[level, :-half]
        )
    return table[0]


def _range_max(
    values: npt.NDArray, left: npt.NDArray, right: npt.NDArray
) -> npt.NDArray:
    """Maximum of ``values[left:right + 1]`` for each query (sparse table)."""
    table = [values]
    width = 1
    while 2 * width <= values.size:
        prev = table[-1]
        table.append(np.maximum(prev[:-width], prev[width:]))
        width *= 2
    k = np.frexp((right - left + 1).astype(float))[1] - 1
    out = np.empty(left.size, dtype=values.dtype)
    for level in np.unique(k):
        sel = k == level
        row = table[level]
        out[sel] = np.maximum(
            row[left[sel]], row[right[sel] - (1 << level) + 1]
        )
    return out


def _coverage(start: npt.NDArray, end: npt.NDArray, size: int) -> npt.NDArray:
    """How many of the runs ``[start, end]`` contain each position."""
    count = np.zeros(size + 1, dtype=np.int64)
    np.add.at(count, start, 1)
    np.add.at(count, end + 1, -1)
    return np.cumsum(count[:size])


def _has_stable_run(
    lo: npt.NDArray, hi: npt.NDArray, wl: npt.NDArray, wh: npt.NDArray, K: int
) -> bool:
    """Is there a proper stable run, i.e. does condition E fail?

    ``[L, R]`` is stable if every support inside it has its window inside
    it. For each ``R`` the admissible ``L`` form a range:

    - some support must lie inside: ``L <= b(R)``, the largest ``lo`` among
      supports ending by ``R``;
    - no support inside may have a window reaching past ``R``:
      ``L > a(R)``, the largest ``lo`` among supports ending by ``R`` whose
      window does not;
    - no support inside may have a window starting before ``L``: ``L`` must
      avoid every ``(wl, lo]`` of a support ending by ``R``. The first
      ``R`` at which each ``L`` is so excluded is ``first_excluded[L]``.

    The run is proper unless it is ``[0, K-1]``.
    """
    positions = np.arange(K)
    has = wl < lo
    first_excluded = _paint(wl[has] + 1, lo[has], hi[has], K, np.minimum, K)
    has = hi < wh
    a = _paint(hi[has], wh[has] - 1, lo[has], K, np.maximum, -1)
    b = np.full(K, -1, dtype=np.int64)
    np.maximum.at(b, hi, lo)
    b = np.maximum.accumulate(b)
    lower = a + 1
    lower[-1] = max(lower[-1], 1)
    ok = lower <= b
    if not ok.any():
        return False
    widest = _range_max(first_excluded, lower[ok], b[ok])
    return bool((widest > positions[ok]).any())


def _one_sided(
    lo: npt.NDArray, hi: npt.NDArray, w: npt.NDArray, K: int
) -> tuple[str, int]:
    """Verdict for left truncation only (windows ``[w, K-1]``).

    Returns the verdict and the gap it rests on (-1 if none).
    """
    last_entry = int(w.max())
    if last_entry == 0:
        return EXISTS, -1
    # The furthest support start among those entered by each position.
    started = np.full(K, -1, dtype=np.int64)
    np.maximum.at(started, w, lo)
    started = np.maximum.accumulate(started)
    k = np.arange(last_entry)
    gaps = k[started[:last_entry] <= k]
    if gaps.size == 0:
        return EXISTS, -1
    # The earliest end among supports containing each position.
    earliest_end = _paint(lo, hi, hi, K, np.minimum, K)
    fatal = gaps[earliest_end[gaps] < lo.max()]
    if fatal.size:
        return DOES_NOT_EXIST, int(fatal[0])
    return NOT_UNIQUE, int(gaps[0])


def _free_atom_certificate(
    lo: npt.NDArray, hi: npt.NDArray, wl: npt.NDArray, wh: npt.NDArray, K: int
) -> int:
    """An atom whose mass always raises the likelihood (-1 if none).

    See section 4 of the module docstring.
    """
    free = _coverage(wl, wh, K) == _coverage(lo, hi, K)
    # The earliest support end among supports starting at or after x.
    ends = np.full(K + 1, K, dtype=np.int64)
    np.minimum.at(ends, lo, hi)
    ends = np.minimum.accumulate(ends[::-1])[::-1]
    forced = ((lo > wl) & (ends[wl] <= lo - 1)) | (
        (hi < wh) & (ends[hi + 1] <= wh)
    )
    hit = free & (_coverage(lo[forced], hi[forced], K) > 0)
    return int(np.argmax(hit)) if hit.any() else -1


def _block(
    lo: npt.NDArray, hi: npt.NDArray, wl: npt.NDArray, wh: npt.NDArray, K: int
) -> tuple[str, int, str]:
    """Verdict for one block (its windows overlap into ``[0, K-1]``)."""
    left_only = bool((wh == K - 1).all())
    right_only = bool((wl == 0).all())
    if left_only and right_only:
        return EXISTS, -1, ""
    if left_only:
        return (*_one_sided(lo, hi, wl, K), "left")
    if right_only:
        # Mirror image: reverse the atoms and it is left truncation. The
        # gap is then the first atom after the free boundary.
        verdict, at = _one_sided(K - 1 - hi, K - 1 - lo, K - 1 - wh, K)
        return verdict, (K - 1 - at if at >= 0 else -1), "right"
    if not _has_stable_run(lo, hi, wl, wh, K):
        return EXISTS, -1, ""
    if (lo == hi).all():
        # Vardi (1985): a connected block that is not strongly connected.
        return DOES_NOT_EXIST, -1, "vardi"
    at = _free_atom_certificate(lo, hi, wl, wh, K)
    if at >= 0:
        return DOES_NOT_EXIST, at, "free"
    return UNDETERMINED, -1, ""


def npmle_existence(
    lo: npt.NDArray,
    hi: npt.NDArray,
    w_lo: npt.NDArray,
    w_hi: npt.NDArray,
    n: npt.NDArray,
    M: int,
) -> tuple[str, int, str]:
    """Classify the truncated Turnbull NPMLE from the data's structure.

    Parameters
    ----------
    lo, hi : ndarray of int
        Each observation's support, as inclusive indices into the ``M``
        Turnbull pieces; already inside the window.
    w_lo, w_hi : ndarray of int
        Each observation's truncation window, likewise (``0`` and ``M - 1``
        where untruncated).
    n : ndarray
        The counts; observations with no count are ignored.
    M : int
        The number of pieces.

    Returns
    -------
    verdict : str
        ``"exists"``, ``"not unique"``, ``"does not exist"`` or
        ``"undetermined"`` (see the module docstring).
    piece : int
        For a one-sided or certified verdict, a piece the verdict rests
        on (the gap, or the atom whose mass always helps); -1 otherwise.
    reason : str
        How the verdict was reached: ``"left"`` or ``"right"`` (a gap under
        one-sided truncation), ``"blocks"`` (windows in groups that do not
        overlap), ``"vardi"`` (exact observations), ``"free"`` (the
        free-atom certificate), or ``""``.
    """
    keep = np.asarray(n) > 0
    rows = np.unique(
        np.column_stack([lo[keep], hi[keep], w_lo[keep], w_hi[keep]]),
        axis=0,
    )
    if rows.shape[0] == 0:
        return EXISTS, -1, ""
    lo, hi, w_lo, w_hi = (rows[:, j].astype(np.int64) for j in range(4))
    # Re-index onto the atoms (pieces inside some support), and snap each
    # window to the atoms it contains.
    atom = _coverage(lo, hi, M) > 0
    before = np.concatenate([[0], np.cumsum(atom)])
    atoms = np.flatnonzero(atom)
    lo, hi = before[lo], before[hi]
    wl, wh = before[w_lo], before[w_hi + 1] - 1

    # Blocks: the runs of atoms over which the windows overlap.
    order = np.argsort(wl, kind="stable")
    reach = np.maximum.accumulate(wh[order])
    new = np.concatenate([[True], wl[order][1:] > reach[:-1]])
    block_of = np.empty(lo.size, dtype=np.int64)
    block_of[order] = np.cumsum(new) - 1
    starts = wl[order][new]
    ends = reach[np.concatenate([np.flatnonzero(new)[1:] - 1, [-1]])]

    # A block whose windows all span it has no truncation inside it and
    # needs no further work; only the others are visited one at a time.
    spans = (wl == starts[block_of]) & (wh == ends[block_of])
    trivial = np.ones(starts.size, dtype=bool)
    np.logical_and.at(trivial, block_of, spans)
    verdict, where, reason = EXISTS, -1, ""
    # Rows sorted by window start are grouped by block already.
    first = np.append(np.flatnonzero(new), lo.size)
    for b in np.flatnonzero(~trivial):
        sel = order[first[b] : first[b + 1]]
        s, size = starts[b], ends[b] - starts[b] + 1
        v, at, why = _block(
            lo[sel] - s, hi[sel] - s, wl[sel] - s, wh[sel] - s, size
        )
        if _SEVERITY[v] > _SEVERITY[verdict]:
            verdict, reason = v, why
            where = int(atoms[at + s]) if at >= 0 else -1
        if verdict == DOES_NOT_EXIST:
            break
    if verdict == EXISTS and starts.size > 1:
        verdict, reason = NOT_UNIQUE, "blocks"
    return verdict, where, reason
