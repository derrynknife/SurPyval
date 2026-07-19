"""Shared helpers for serialising fitted recurrent-event models.

The recurrence intensity models (``HPP``, ``CrowAMSAA``, ``Duane``,
``CoxLewis``) are stateless singletons whose maths is fully determined by
their identity, so a fitted model only needs to store its intensity model's
name alongside the fitted parameters. The display ``name`` of an intensity
model does not match its module attribute name, so reconstruction goes through
an explicit name map here.
"""


def intensity_dist_by_name(name):
    """
    Resolve a recurrence intensity model from its display ``name``.

    Restricted to the known intensity models so an untrusted dict cannot
    resolve an arbitrary attribute.
    """
    import surpyval.recurrent as recurrent

    mapping = {
        "Homogeneous Poisson Process": recurrent.HPP,
        "Crow-AMSAA": recurrent.CrowAMSAA,
        "Duane": recurrent.Duane,
        "Cox-Lewis": recurrent.CoxLewis,
    }
    if name not in mapping:
        raise ValueError(
            "Unknown recurrence intensity model '{}'; expected one of "
            "{}".format(name, sorted(mapping))
        )
    return mapping[name]
