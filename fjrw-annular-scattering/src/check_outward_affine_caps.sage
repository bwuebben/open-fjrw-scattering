"""Exact affine gate for non-reflection outward caps.

The certificate proves only the affine/support part of the cap proposal.  It
does not assert extension of the polarization, slab rings, or pro-Kummer
coefficient charts over the compactifying divisors.
"""

CapRing = PolynomialRing(ZZ, names=("ell", "r"))
ell, r = CapRing.gens()


def outward_deck(width):
    """Affine deck map (x,r) -> (x+ell+width*r,r)."""

    return matrix(
        CapRing,
        [[1, width, ell], [0, 1, 0], [0, 0, 1]],
    )


for width in (1, 4, 16):
    deck = outward_deck(width)
    assert deck.det() == 1
    assert deck.inverse() == matrix(
        CapRing,
        [[1, -width, -ell], [0, 1, 0], [0, 0, 1]],
    )
    point = vector(CapRing, (0, r, 1))
    assert deck * point - point == vector(CapRing, (ell + width * r, 0, 0))
    for power in range(-8, 9):
        assert deck**power == matrix(
            CapRing,
            [
                [1, power * width, power * ell],
                [0, 1, 0],
                [0, 0, 1],
            ],
        )

    # The transverse component of every tangent vector is invariant.  Thus a
    # ray with positive outward component remains outward under all deck
    # translates and crosses each truncation r=constant at most once.
    tangent_deck = deck[:2, :2]
    a, b = vector(CapRing, (ell, r))
    assert (tangent_deck * vector(CapRing, (a, b)))[1] == b

# At integral truncation height n, the affine circumference is ell+w*n.  Its
# strict increase is the elementary convexity/properness feature of the
# flare, in contrast with the inward reflected collar ell-w*r.
for width in (1, 4, 16):
    circumference = ell + width * r
    assert circumference.subs({r: r + 1}) - circumference == width

# The two outer boundary holonomies have widths four and sixteen.
outer_holonomies = {
    "width_4": matrix(ZZ, [[1, -4], [0, 1]]),
    "width_16": matrix(ZZ, [[1, -16], [0, 1]]),
}
assert all(
    matrix(Integers(4), holonomy) == identity_matrix(Integers(4), 2)
    for holonomy in outer_holonomies.values()
)

# Every lifted focus monodromy is a primitive trace-two shear.  An oriented
# peripheral basis puts it in T or T^{-1}; reversing the peripheral
# orientation chooses the outward sign uniformly.
focus_holonomies = {
    "A": matrix(ZZ, [[-2, 9], [-1, 4]]),
    "B_0": matrix(ZZ, [[2, 1], [-1, 0]]),
    "B_1": matrix(ZZ, [[-14, 25], [-9, 16]]),
    "C": matrix(ZZ, [[-2, 1], [-9, 4]]),
}
focus_invariant_tangents = {
    "A": vector(ZZ, (3, 1)),
    "B_0": vector(ZZ, (1, -1)),
    "B_1": vector(ZZ, (5, 3)),
    "C": vector(ZZ, (1, 3)),
}
focus_normal_forms = {}
for label, holonomy in focus_holonomies.items():
    tangent = focus_invariant_tangents[label]
    assert holonomy.det() == 1
    assert holonomy.trace() == 2
    assert holonomy * tangent == tangent
    assert gcd((holonomy - identity_matrix(ZZ, 2)).list()) == 1
    basis = None
    for first in range(-6, 7):
        for second in range(-6, 7):
            candidate = matrix(
                ZZ,
                [[tangent[0], first], [tangent[1], second]],
            )
            if abs(candidate.det()) == 1:
                basis = candidate
                break
        if basis is not None:
            break
    assert basis is not None
    normal_form = basis.inverse() * holonomy * basis
    assert normal_form in (
        matrix(ZZ, [[1, -1], [0, 1]]),
        matrix(ZZ, [[1, 1], [0, 1]]),
    )
    focus_normal_forms[label] = normal_form

# The focus action on the ordered residue quotient remains the common
# involution already certified in the main lattice gate; the affine cap does
# not trivialize it.
P_residue = matrix(Integers(4), [[2, 1], [3, 0]])
for holonomy in focus_holonomies.values():
    assert holonomy.change_ring(Integers(4)) == P_residue

print("outward affine caps: PASS")
print("  outer widths:", [4, 16])
print("  focus widths:", [1, 1, 1, 1])
print("  focus normal forms:", focus_normal_forms)
print("  flare circumference: ell + width*r")
