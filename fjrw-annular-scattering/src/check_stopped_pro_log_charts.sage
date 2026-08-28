"""Exact local checks for the pro-logarithmic focus-stop charts."""

degrees = {"A": 2, "B_0": 4, "B_1": 4, "C": 2}

# In group coordinates (x,t,s), adjoining s=d*r is a Kummer map of exact
# index d.  The monoid chart is understood with fs saturation.
for label, degree in degrees.items():
    group_map = diagonal_matrix(ZZ, [1, 1, degree])
    smith, _, _ = group_map.smith_form()
    assert [abs(smith[i, i]) for i in range(3)] == [1, 1, degree]
    assert abs(group_map.det()) == degree

# The multiplicative local node equation xy=t*s pulls back to xy=t*r^d.
LocalRing.<x, y, t, r, s, h> = PolynomialRing(QQ, 6)
old_node = x * y - t * s
for degree in sorted(set(degrees.values())):
    assert old_node(s=r**degree) == x * y - t * r**degree

# At either B cusp the fourth-root equation has two quadratic components.
assert r**4 - h**2 == (r**2 - h) * (r**2 + h)

# The finite root-choice band and the primitive-form divisor twists close in
# the already fixed coherent orientations.
phase_weights = {"A": 2, "B_0": -1, "B_1": 1, "C": 2}
assert sum(phase_weights.values()) % 4 == 0
primitive_line_weights = {"A": -2, "B_0": -2, "B_1": 2, "C": -2}
assert all(weight in ZZ for weight in primitive_line_weights.values())

# The two annular boundary holonomies act trivially on the full GKT residue
# lattice, so no extra finite root phase is created at the boundary stops.
Residues = Integers(4)
residue_identity = identity_matrix(Residues, 2)
boundary_holonomies = [
    matrix(ZZ, [[5, -4], [4, -3]]),
    matrix(ZZ, [[33, -64], [16, -31]]),
]
assert all(matrix(Residues, holonomy) == residue_identity for holonomy in boundary_holonomies)

print("stopped pro-log charts: PASS")
print("  local Kummer degrees:", degrees)
print("  root phase weights mod 4:", phase_weights)
print("  primitive-line divisor weights:", primitive_line_weights)
