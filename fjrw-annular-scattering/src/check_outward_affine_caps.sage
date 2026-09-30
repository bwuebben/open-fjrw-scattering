"""Exact checks for the outward affine flares of Section 4.

Checks the affine statements of the proposition on outward affine flares:
the flare action (x,r) -> (x+ell+w*r, r) for the two boundary widths 4 and
16, its powers, the invariance of the outward coordinate, and the agreement
of the flare action with the collar holonomy (x,y) -> (x+ell-w*y, y) under
y=-r, with ell=w*h from the development data of Appendix A.  It also checks
that the four lifted monodromies of Section 2 are primitive unipotent with a
common sign in oriented bases, and the action of the monodromies on the
residue lattice stated in Section 3.  No algebraic statement about the ends
is checked.

Usage: sage check_outward_affine_caps.sage
"""

CapRing = PolynomialRing(ZZ, names=("ell", "r"))
ell, r = CapRing.gens()


def outward_deck(width):
    """Affine deck map (x,r) -> (x+ell+width*r,r)."""

    return matrix(
        CapRing,
        [[1, width, ell], [0, 1, 0], [0, 0, 1]],
    )


for width in (4, 16):
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

    # The outward component of every tangent vector is invariant.  Thus a
    # ray with positive outward component remains outward under all deck
    # translates and crosses each level r=constant exactly once.
    tangent_deck = deck[:2, :2]
    a, b = vector(CapRing, (ell, r))
    assert (tangent_deck * vector(CapRing, (a, b)))[1] == b

# At height r the affine circumference is ell+w*r.
for width in (4, 16):
    circumference = ell + width * r
    assert circumference.subs({r: r + 1}) - circumference == width

# Development data of Appendix A: (ell, w, h) with ell = w*h.  Under y=-r the
# flare action is the collar holonomy (x,y) -> (x+ell-w*y, y), which fixes
# the apex at height y=h.
development = {0: (4, 4, QQ(1)), 1: (8, 16, QQ(1) / 2)}
X, Y = var("X Y")
for index, (length, width, height) in development.items():
    assert length == width * height
    collar = (X + length - width * Y, Y)
    flare_in_y = (X + length + width * (-Y), Y)
    assert bool(collar[0] == flare_in_y[0])
    assert bool(collar[0].subs({Y: height}) == X)

# The two boundary holonomies have widths four and sixteen and act trivially
# on the residue lattice (Z/4)^2 of Section 3.
outer_holonomies = {
    "width_4": matrix(ZZ, [[1, -4], [0, 1]]),
    "width_16": matrix(ZZ, [[1, -16], [0, 1]]),
}
assert all(
    matrix(Integers(4), holonomy) == identity_matrix(Integers(4), 2)
    for holonomy in outer_holonomies.values()
)

# Every lifted monodromy of Section 2 is a primitive trace-two shear.  In an
# oriented basis (invariant tangent, complement) with determinant +1 all four
# have the same normal form [[1,1],[0,1]].
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
            if candidate.det() == 1:
                basis = candidate
                break
        if basis is not None:
            break
    assert basis is not None
    normal_form = basis.inverse() * holonomy * basis
    assert normal_form == matrix(ZZ, [[1, 1], [0, 1]])
    focus_normal_forms[label] = normal_form

# Each of these monodromies acts on the residue lattice (Z/4)^2 by the
# matrix P_res of Section 3.
P_residue = matrix(Integers(4), [[2, 1], [3, 0]])
for holonomy in focus_holonomies.values():
    assert holonomy.change_ring(Integers(4)) == P_residue

print("outward affine flares: PASS")
print("  boundary widths:", [4, 16])
print("  development (ell, w, h):", development)
print("  lifted monodromies in oriented bases:", focus_normal_forms)
print("  flare circumference: ell + width*r")
