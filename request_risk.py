"""Bounded certification propensities for the request-level IPW controller."""

def certification_probability(defeat_probability, budget, floor):
    """Return the actual Bernoulli propensity, including budget recovery.

    A negative budget forces certification. Without the upper bound, a draw
    still always certifies but its recorded propensity exceeds one, giving
    negative IPW feedback and an invalid variance diagnostic.
    """
    if not 0 < floor <= 1:
        raise ValueError('The exploration floor must be in (0, 1].')
    if budget <= 0:
        return 1.0
    if defeat_probability <= 0:
        return float(floor)
    return min(1.0, max(float(floor), 1.0-budget/defeat_probability))
