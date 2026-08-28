#!/usr/bin/env sage
"""Exact degree/order bounds for the uniform Rees--Kummer cap chart."""

L = matrix(ZZ, [[-2, 0], [1, 1]])
kappa = vector(ZZ, (1, 1))
twists = {
    "alpha": vector(ZZ, (1, 0)),
    "delta": vector(ZZ, (3, 2)),
    "gamma": vector(ZZ, (2, 3)),
    "beta": vector(ZZ, (0, 1)),
}

# At the bottom end, u has outward order one and the character z^(Lq)
# has pole order equal to the second coordinate of Lq.
pole = {}
for name, tau in twists.items():
    q = tau + kappa
    d = L * q
    pole[name] = d[1]

assert pole == {"alpha": 3, "delta": 7, "gamma": 7, "beta": 3}
A = max(pole.values())
assert A == 7
residual = {name: A - value for name, value in pole.items()}
assert residual == {"alpha": 4, "delta": 0, "gamma": 0, "beta": 4}

# A nonzero Lie word on r distinct square-zero labels has exponent
# kappa + sum(tau_i).  Every standard twist has coordinate sum at most five.
# Thus its bottom pole is at most 5*r+2, while Rees homogenization followed
# by lambda=u^7*s contributes u^(7*r).
r = var("r")
assert bool((7*r - (5*r + 2) >= 0).subs(r=1))
for degree in range(1, 33):
    assert 7*degree - (5*degree + 2) >= 0

# The Kummer chart lambda=u^7*s is the toric monoid map
# N -> N^2, 1 |-> (7,1).  The image vector is primitive, so the group
# cokernel is torsion-free; both monoids are fine and saturated.
column = vector(ZZ, (7, 1))
assert gcd(column) == 1
assert matrix(ZZ, 2, 1, list(column)).elementary_divisors() == [1, 0]

print("Rees--Kummer cap: PASS")
print("  singleton bottom poles:", pole)
print("  minimal uniform Rees weight:", A)
print("  residual cap orders:", residual)
print("  nonzero Lie-word residual lower bound: 2*r - 2")
print("  monoid group cokernel: Z (torsion-free)")
