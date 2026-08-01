"""Alpha-stage models.

The least mature tier of SurPyval: interfaces here are exploratory and
may change or disappear without notice. More mature (but still
pre-stable) models live in ``surpyval.beta``.

The former ``SeriesModel``/``ParallelModel`` reliability-block
composition was removed in v0.17.0 (#284): its nested composition
produced incorrect survival functions, and reliability block diagrams
are covered by the Repyability package.
"""
