"""Exact checks of root data used in Sections 2 to 4.

Checks the factorization of the fourth-root equation at the B cusps, the
line weights <N_V, kappa> of the lifted slab maps (Section 3), the root-free
form of the transported slab radicands used in Section 4 (the exponent of
g_V met by a character z^(2m), m in the parity sublattice, is divisible by
four), and the trivial action of the two boundary holonomies on the residue
lattice (Z/4)^2 (Section 3).

Usage: sage check_stopped_pro_log_charts.sage
"""

degrees = {"A": 2, "B_0": 4, "B_1": 4, "C": 2}
conormals = {
    "A": vector(ZZ, (1, -3)),
    "B_0": vector(ZZ, (-1, -1)),
    "B_1": vector(ZZ, (-3, 5)),
    "C": vector(ZZ, (-3, 1)),
}
tangents = {
    "A": vector(ZZ, (3, 1)),
    "B_0": vector(ZZ, (1, -1)),
    "B_1": vector(ZZ, (5, 3)),
    "C": vector(ZZ, (1, 3)),
}

# At either B cusp the fourth-root equation has two quadratic components.
LocalRing.<r, h> = PolynomialRing(QQ, 2)
assert r**4 - h**2 == (r**2 - h) * (r**2 + h)

# Line weights <N_V, kappa> with kappa = (1,1).
kappa = vector(ZZ, (1, 1))
line_weights = {label: N * kappa for label, N in conormals.items()}
assert line_weights == {"A": -2, "B_0": -2, "B_1": 2, "C": -2}

# Root-free transported radicands.  Each N_V annihilates its own tangent, all
# pairings <N_V, u_W> lie in 4Z, the parity monodromies preserve the
# sublattice {m1 = m2 mod 2} containing every u_W, and every N_V is even on
# that sublattice.  Hence a slab map multiplies z^(2m) by g_V^e with e
# divisible by 4, i.e. by an integral power of f_V = g_V^(d_V).
parity = [
    matrix(ZZ, [[2, 1], [-1, 0]]),
    matrix(ZZ, [[-1, 2], [-10, 19]]),
    matrix(ZZ, [[-2, 9], [-1, 4]]),
    matrix(ZZ, [[-14, 25], [-9, 16]]),
    matrix(ZZ, [[-20, 11], [-11, 6]]),
    matrix(ZZ, [[-2, 1], [-9, 4]]),
]
sublattice_gens = [vector(ZZ, (1, 1)), vector(ZZ, (2, 0))]
def in_sublattice(m):
    return (m[0] - m[1]) % 2 == 0
for label, N in conormals.items():
    assert N * tangents[label] == 0
    for other in tangents.values():
        assert (N * other) % 4 == 0
    for m in sublattice_gens:
        assert (N * m) % 2 == 0
        assert (2 * (N * m)) % 4 == 0
for u in tangents.values():
    assert in_sublattice(u)
for M in parity:
    for m in sublattice_gens:
        assert in_sublattice(M * m)
        assert in_sublattice(M.inverse() * m)

# The two boundary holonomies act trivially on the residue lattice.
Residues = Integers(4)
residue_identity = identity_matrix(Residues, 2)
boundary_holonomies = [
    matrix(ZZ, [[5, -4], [4, -3]]),
    matrix(ZZ, [[33, -64], [16, -31]]),
]
assert all(matrix(Residues, holonomy) == residue_identity for holonomy in boundary_holonomies)

print("root data and boundary residues: PASS")
print("  root degrees:", degrees)
print("  line weights <N_V,kappa>:", line_weights)
print("  pairings <N_V,u_W> mod 4:", {V: [ (N * u) % 4 for u in tangents.values()] for V, N in conormals.items()})
