#!/usr/bin/env sage
"""Exact asymptotic orders for the four standard cofactor chords."""

L = matrix(ZZ, [[-2, 0], [1, 1]])

q = {
    "alpha": vector(ZZ, (2, 1)),
    "delta": vector(ZZ, (4, 3)),
    "gamma": vector(ZZ, (3, 4)),
    "beta": vector(ZZ, (1, 2)),
}

# Primitive outward conormals of the three facets of
# conv{(-1,-1),(-1,1),(3,-1)}.
nu_left = vector(ZZ, (-1, 0))
nu_slope = vector(ZZ, (1, 2))
nu_bottom = vector(ZZ, (0, -1))

positive_facet = {
    "alpha": nu_left,
    "delta": nu_left,
    "gamma": nu_slope,
    "beta": nu_slope,
}

orders = {}
for name, exponent in q.items():
    d = L * exponent
    pos = positive_facet[name].dot_product(d)
    neg = nu_bottom.dot_product(d)
    assert pos > 0
    assert neg < 0
    # The negative endpoint is traversed in direction -d, which is outward.
    assert nu_bottom.dot_product(-d) > 0
    orders[name] = (tuple(d), pos, neg)

# Every outgoing completion direction q(a,b) has positive order at its
# unique outer end.  The same exponent on the negative half of a full chord
# has strictly negative bottom order.
a, b = var("a b")
d_ab = L * vector(SR, (2*a + b, a + 2*b))
assert d_ab == vector(SR, (-4*a - 2*b, 3*a + 3*b))
assert nu_left.dot_product(d_ab) == 4*a + 2*b
assert nu_slope.dot_product(d_ab) == 2*a + 4*b
assert nu_bottom.dot_product(d_ab) == -3*a - 3*b

# The flare holonomy preserves the normal coordinate of every exponent.
w = var("w")
T = matrix(SR, [[1, w], [0, 1]])
x, r = var("x r")
assert (T * vector(SR, (x, r)))[1] == r

print("outward cap valuations: PASS")
for name in ("alpha", "delta", "gamma", "beta"):
    print(" ", name, "direction/positive/negative:", orders[name])
print("  general left order:", 4*a + 2*b)
print("  general slope order:", 2*a + 4*b)
print("  general negative-bottom order:", -3*a - 3*b)
