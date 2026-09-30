#!/usr/bin/env sage
"""Exact exit data of the singleton supports at the boundary of the disk.

The support of a shifted exponent q lies on the ray spanned by the cofactor
direction L*q.  For the two chords, and for every descendant exponent
q(a,b) = a(2,1) + b(1,2) with a, b >= 0, this checks through which facet of
conv{(-1,-1),(-1,1),(3,-1)} the positive and negative halves leave, and that
they leave transversely (nonzero pairing with the outward conormal).  This
is the transversality used in Section 4 for the supports meeting the two
boundary circles; the flare action preserves the outward coordinate.

Usage: sage check_outward_cap_valuations.sage
"""

L = matrix(ZZ, [[-2, 0], [1, 1]])

q = {
    "alpha": vector(ZZ, (2, 1)),
    "beta": vector(ZZ, (1, 2)),
}

# Primitive outward conormals of the three facets of
# conv{(-1,-1),(-1,1),(3,-1)}.
nu_left = vector(ZZ, (-1, 0))
nu_slope = vector(ZZ, (1, 2))
nu_bottom = vector(ZZ, (0, -1))

positive_facet = {
    "alpha": nu_left,
    "beta": nu_slope,
}

orders = {}
for name, exponent in q.items():
    d = L * exponent
    pos = positive_facet[name].dot_product(d)
    neg = nu_bottom.dot_product(d)
    assert pos > 0
    assert neg < 0
    # The negative half is traversed in direction -d, which is outward.
    assert nu_bottom.dot_product(-d) > 0
    orders[name] = (tuple(d), pos, neg)

# Every descendant direction q(a,b) leaves transversely: positive pairings
# with the left and slope conormals, negative pairing with the bottom one.
a, b = var("a b")
d_ab = L * vector(SR, (2*a + b, a + 2*b))
assert d_ab == vector(SR, (-4*a - 2*b, 3*a + 3*b))
assert nu_left.dot_product(d_ab) == 4*a + 2*b
assert nu_slope.dot_product(d_ab) == 2*a + 4*b
assert nu_bottom.dot_product(d_ab) == -3*a - 3*b

# The flare action preserves the outward coordinate of every vector.
w = var("w")
T = matrix(SR, [[1, w], [0, 1]])
x, r = var("x r")
assert (T * vector(SR, (x, r)))[1] == r

print("singleton exit data: PASS")
for name in ("alpha", "beta"):
    print(" ", name, "direction/positive/negative:", orders[name])
print("  general left pairing:", 4*a + 2*b)
print("  general slope pairing:", 2*a + 4*b)
print("  general bottom pairing:", -3*a - 3*b)
