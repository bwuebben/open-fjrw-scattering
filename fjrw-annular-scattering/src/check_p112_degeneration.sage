"""
Exact checks for the P(1,1,2) degeneration, the parity cover and the annular
lattice data (Sections 2 and 3 and Appendix A of the paper).

Run from this directory with:
    sage check_p112_degeneration.sage

Some blocks record auxiliary data that the paper does not use; their
comments say so.
"""

from sage.all import *


def det2(u, v):
    return matrix(ZZ, [[u[0], v[0]], [u[1], v[1]]]).det()


# Fan of P(1,1,2).  The coordinate divisors X=0, Y=0, Z=0 have
# primitive rays v_X, v_Y, v_Z and satisfy the weighted relation
# v_X + v_Y + 2 v_Z = 0.
v_X = vector(ZZ, (1, 0))
v_Y = vector(ZZ, (-1, -2))
v_Z = vector(ZZ, (0, 1))
assert v_X + v_Y + 2 * v_Z == 0

# The cone at [0:0:1] has index two, hence the coarse surface has one A_1
# singularity.  The other two cones are unimodular.
cone_indices = {
    "XY": abs(det2(v_X, v_Y)),
    "YZ": abs(det2(v_Y, v_Z)),
    "ZX": abs(det2(v_Z, v_X)),
}
assert cone_indices == {"XY": 2, "YZ": 1, "ZX": 1}

# Anticanonical reflexive polygon Delta = {m : <m,v_i> >= -1}.
Delta_vertices = [
    vector(ZZ, (-1, -1)),
    vector(ZZ, (-1, 1)),
    vector(ZZ, (3, -1)),
]
for m in Delta_vertices:
    assert all(m.dot_product(v) >= -1 for v in (v_X, v_Y, v_Z))

# All anticanonical lattice points.  The origin is the unique interior point;
# the other eight points lie on the boundary and have height one in the
# coherent star subdivision.
Delta_points = []
for a in range(-1, 4):
    for b in range(-1, 2):
        m = vector(ZZ, (a, b))
        if all(m.dot_product(v) >= -1 for v in (v_X, v_Y, v_Z)):
            Delta_points.append(m)
assert len(Delta_points) == 9
assert vector(ZZ, (0, 0)) in Delta_points


def cox_exponents(m):
    """Exponents of X,Y,Z in the anticanonical section indexed by m."""
    return tuple(ZZ(m.dot_product(v) + 1) for v in (v_X, v_Y, v_Z))


section_dictionary = {tuple(m): cox_exponents(m) for m in Delta_points}
assert section_dictionary[(0, 0)] == (1, 1, 1)       # XYZ
assert section_dictionary[(-1, -1)] == (0, 4, 0)     # Y^4
assert section_dictionary[(3, -1)] == (4, 0, 0)      # X^4
assert section_dictionary[(-1, 1)] == (0, 0, 2)      # Z^2
assert section_dictionary[(1, -1)] == (2, 2, 0)      # X^2 Y^2

height = {tuple(m): (0 if m == vector(ZZ, (0, 0)) else 1) for m in Delta_points}
assert sum(1 for h in height.values() if h == 0) == 1
assert sum(1 for h in height.values() if h == 1) == 8

# Its polar polygon has vertices the fan rays.  The XY edge has integral
# length two and unique interior lattice point m_*.
Delta_dual_vertices = [v_X, v_Z, v_Y]
edge_XY = v_X - v_Y
assert edge_XY == vector(ZZ, (2, 2))
edge_XY_primitive = vector(ZZ, (1, 1))
assert edge_XY == 2 * edge_XY_primitive
m_star = (v_X + v_Y) / 2
assert m_star == vector(ZZ, (0, -1))

# Natural boundary-charge map iota(e_1)=v_X, iota(e_2)=v_Y.
iota = matrix(ZZ, [[v_X[0], v_Y[0]], [v_X[1], v_Y[1]]])
assert abs(iota.det()) == 2
assert iota * vector(ZZ, (1, 1)) == 2 * m_star

# The vertex of Delta dual to the index-two XY cone.  In the
# degeneration-side intersection complex check B, the radial edge from the
# origin to this vertex carries the order-two singularity.
p_XY = vector(ZZ, (-1, 1))
assert p_XY in Delta_vertices
assert p_XY.dot_product(v_X) == -1
assert p_XY.dot_product(v_Y) == -1

# The direct map above lands in the mirror bounded cell Delta^vee.  The
# integral cofactor map L=det(iota)iota^(-T) gives the support directions in
# the plane of Delta (Section 2.2).  Only the lattice identities are checked
# here.
iota_cofactor = matrix(ZZ, iota.det() * iota.inverse().transpose())
assert iota_cofactor == matrix(ZZ, [[-2, 0], [1, 1]])
assert iota_cofactor * vector(ZZ, (1, 1)) == 2 * p_XY

# This cofactor is forced at the lattice level by the Hamiltonian/Legendre
# dictionary.  H_source sends a Hamiltonian exponent (a,b) to the GKT
# logarithmic vector (b,-a).  K_target sends a tangent vector to its conormal.
# Their composition around iota is precisely the cofactor map.
H_source = matrix(ZZ, [[0, 1], [-1, 0]])
K_target = matrix(ZZ, [[0, -1], [1, 0]])
assert K_target * iota * H_source == iota_cofactor
marginal_charge = vector(ZZ, (1, 1))
marginal_tangent = iota * H_source * marginal_charge
marginal_conormal = iota_cofactor * marginal_charge
assert marginal_tangent == edge_XY
assert marginal_conormal == 2 * p_XY
assert marginal_conormal.dot_product(marginal_tangent) == 0
assert iota_cofactor.transpose() * iota == iota.det() * identity_matrix(ZZ, 2)
# The paper states the first identity without the two rotations: with
# (a,b)^perp = (b,-a), it reads (L q)^perp = iota(q^perp); here
# H_source q = q^perp and K_target v = -v^perp.
def perp(v):
    return vector(ZZ, (v[1], -v[0]))
for q_test in (vector(ZZ, (1, 0)), vector(ZZ, (0, 1)), vector(ZZ, (2, 1))):
    assert perp(iota_cofactor * q_test) == iota * perp(q_test)

# An order-two focus-focus monodromy has the following conjugacy class.
# We display it in a basis whose first vector is tangent to the length-two
# edge, so that the invariant primitive vector is (1,1).
T2 = matrix(ZZ, [[1, -2], [0, 1]])
P = matrix(ZZ, [[1, 0], [1, 1]])
M2 = P * T2 * P.inverse()
assert M2 == matrix(ZZ, [[3, -2], [2, -1]])
assert M2.det() == 1 and M2.trace() == 2
assert M2 * edge_XY_primitive == edge_XY_primitive
assert M2 != identity_matrix(ZZ, 2)

# A conjugate representative on check B whose invariant vector is the radial
# direction p_XY.
P_check = matrix(ZZ, [[-1, 0], [1, 1]])
M2_check = P_check * T2 * P_check.inverse()
assert M2_check == matrix(ZZ, [[3, 2], [-2, -1]])
assert M2_check.det() == 1 and M2_check.trace() == 2
assert M2_check * p_XY == p_XY

# Reconstruct the straight-boundary chart at p_XY directly from CPS
# Construction 7.2.  The primitive boundary tangents on the two sides and
# the inward radial vector are unimodular bases.  Requiring the boundary
# tangents to become opposite and requiring agreement on the radial edge
# determines the two linear pieces uniquely.
p_left = vector(ZZ, (-1, -1))
p_right = vector(ZZ, (3, -1))
boundary_left = (p_left - p_XY) / 2
boundary_right = (p_right - p_XY) / 2
radial_inward = -p_XY
assert boundary_left == vector(ZZ, (0, -1))
assert boundary_right == vector(ZZ, (2, -1))
assert radial_inward == vector(ZZ, (1, -1))
basis_left = matrix(ZZ, [boundary_left, radial_inward]).transpose()
basis_right = matrix(ZZ, [boundary_right, radial_inward]).transpose()
assert abs(basis_left.det()) == 1
assert abs(basis_right.det()) == 1
target_left = matrix(ZZ, [[-1, 0], [0, 1]])
target_right = identity_matrix(ZZ, 2)
chart_left = target_left * basis_left.inverse()
chart_right = target_right * basis_right.inverse()
assert chart_left == matrix(ZZ, [[1, 1], [1, 0]])
assert chart_right == matrix(ZZ, [[1, 1], [-1, -2]])
assert chart_left * boundary_left == vector(ZZ, (-1, 0))
assert chart_right * boundary_right == vector(ZZ, (1, 0))
assert chart_left * radial_inward == vector(ZZ, (0, 1))
assert chart_right * radial_inward == vector(ZZ, (0, 1))
chart_transition = chart_right * chart_left.inverse()
assert chart_transition == matrix(ZZ, [[1, 0], [-2, 1]])
assert chart_right.inverse() * chart_left == M2_check

# Repeat the straight-boundary calculation at all three vertices of Delta.
# This checks the straight-boundary frames at every vertex, not only at B.  We
# use the cyclic order A--B--C and orient the two primitive boundary tangents
# at each vertex toward its two neighbors.  The dual monodromy is the
# transpose of the displayed check-B monodromy (equivalently inverse
# transpose after reversing loop orientation).
p_A = vector(ZZ, (-1, -1))
p_B = p_XY
p_C = vector(ZZ, (3, -1))


def straight_boundary_charts(boundary_before, boundary_after, radial):
    basis_before = matrix(ZZ, [boundary_before, radial]).transpose()
    basis_after = matrix(ZZ, [boundary_after, radial]).transpose()
    assert abs(basis_before.det()) == 1
    assert abs(basis_after.det()) == 1
    chart_before = matrix(ZZ, [[-1, 0], [0, 1]]) * basis_before.inverse()
    chart_after = identity_matrix(ZZ, 2) * basis_after.inverse()
    return matrix(ZZ, chart_before), matrix(ZZ, chart_after)


facet_vectors = {
    "A": (vector(ZZ, (1, 0)), vector(ZZ, (0, 1)), -p_A),
    "B": (vector(ZZ, (0, -1)), vector(ZZ, (2, -1)), -p_B),
    "C": (vector(ZZ, (-2, 1)), vector(ZZ, (-1, 0)), -p_C),
}
straight_charts = {
    label: straight_boundary_charts(*vectors)
    for label, vectors in facet_vectors.items()
}
assert straight_charts["A"] == (
    matrix(ZZ, [[-1, 1], [0, 1]]),
    matrix(ZZ, [[-1, 1], [1, 0]]),
)
assert straight_charts["B"] == (chart_left, chart_right)
assert straight_charts["C"] == (
    matrix(ZZ, [[-1, -3], [-1, -2]]),
    matrix(ZZ, [[-1, -3], [0, 1]]),
)

facet_monodromy_check = {
    label: matrix(ZZ, chart_after.inverse() * chart_before)
    for label, (chart_before, chart_after) in straight_charts.items()
}
facet_monodromy = {
    label: monodromy.transpose()
    for label, monodromy in facet_monodromy_check.items()
}
assert facet_monodromy_check == {
    "A": matrix(ZZ, [[0, 1], [-1, 2]]),
    "B": matrix(ZZ, [[3, 2], [-2, -1]]),
    "C": matrix(ZZ, [[4, 9], [-1, -2]]),
}
assert facet_monodromy == {
    "A": matrix(ZZ, [[0, -1], [1, 2]]),
    "B": matrix(ZZ, [[3, -2], [2, -1]]),
    "C": matrix(ZZ, [[4, -1], [9, -2]]),
}
assert facet_monodromy_check["B"] == M2_check
assert facet_monodromy["B"] == M2
facet_tangents = {
    "A": vector(ZZ, (1, -1)),
    "B": vector(ZZ, (1, 1)),
    "C": vector(ZZ, (1, 3)),
}
for label in facet_monodromy:
    assert facet_monodromy[label].det() == 1
    assert facet_monodromy[label].trace() == 2
    assert facet_monodromy[label] * facet_tangents[label] == facet_tangents[label]

# Only the marginal line can extend as a constant sub-local-system through a
# punctured neighborhood of the focus-focus point.  A full-rank constant map
# would force M2_check to be the identity.
fixed_lattice_matrix = M2_check - identity_matrix(ZZ, 2)
assert fixed_lattice_matrix.rank() == 1
assert fixed_lattice_matrix.right_kernel().rank() == 1
assert p_XY in fixed_lattice_matrix.right_kernel()
assert M2_check != identity_matrix(ZZ, 2)

# On the complement of a chosen monodromy cut the full cofactor map has two
# sectorial representatives related by the affine monodromy.  Their
# difference kills the marginal charge and has rank one.
iota_cofactor_plus = iota_cofactor
iota_cofactor_minus = M2_check * iota_cofactor_plus
assert iota_cofactor_minus == matrix(ZZ, [[-4, 2], [3, -1]])
assert (iota_cofactor_minus - iota_cofactor_plus).rank() == 1
assert (
    (iota_cofactor_minus - iota_cofactor_plus) * marginal_charge
    == vector(ZZ, (0, 0))
)

# The elliptic pencil's monodromy at s_eff=infinity has trace -2.  Since
# s_eff ~ q^(-2), a loop in q sees its square (or inverse square, depending
# on orientation), whose unipotent width is four.  This agrees with the I_4
# cycle obtained after resolving the stacky toric boundary.
M_ell_infinity = matrix(ZZ, [[1, -2], [2, -3]])
M_q = M_ell_infinity**2
assert M_q.det() == 1 and M_q.trace() == 2
assert gcd([abs(a) for a in (M_q - identity_matrix(ZZ, 2)).list()]) == 4

# Exact degeneration to the toric boundary.  For q != 0, completing the
# square returns the original quartic pencil with
# s_eff = s + 1/(4 q^2).  The code variable q plays the role of the
# degeneration parameter eta of Section 2.1 of the paper; the pencil is
# written here as Z^2 = X^4 + s X^2 Y^2 + Y^4, so the shift of s has the
# opposite sign to the one in the paper.
Rq = PolynomialRing(QQ, names=("q", "s"))
q, s = Rq.gens()
K = Rq.fraction_field()
S = PolynomialRing(K, names=("X", "Y", "Z", "Zp"))
X, Y, Z, Zp = S.gens()

F_q = X * Y * Z + q * (Z**2 - X**4 - s * X**2 * Y**2 - Y**4)
transformed = (F_q / q).subs({Z: Zp - X * Y / (2 * q)})
expected = Zp**2 - X**4 - (s + 1 / (4 * q**2)) * X**2 * Y**2 - Y**4
assert transformed == expected
assert (F_q - X * Y * Z) / q == Z**2 - X**4 - s * X**2 * Y**2 - Y**4

# Leading primary marginal GKT factor in the symmetric potential convention.
# The raw invariants satisfy nu_40 + nu_04 = -1/4, but a term with two
# identical (2,2) markings appears in W^{nu,sym} with factor (-1)^(2-1)/2!.
# The two extreme potentials therefore differ by
# (t_22^2/8)(Y^4-X^4), so |c|=1/32 in exp(c t_22^2 D).
aut_factor = -QQ(1) / factorial(2)
assert aut_factor == -QQ(1) / 2
c_gkt = QQ(1) / 32
assert c_gkt * vector(QQ, (4, -4)) == vector(QQ, (1, -1)) / 8

# The unresolved order-two slab has (1+z_rho)^2.  Its linear coefficient is
# two, but no equality with an FJRW deformation coefficient is asserted: the
# direct ray and the slab have different supports.
slab_linear_coefficient = ZZ(2)
assert (1 + 2 + 1) == 4

# CPS Theorem 8.2 gives, after removing the common degeneration factor,
# W_CPS=U+V+2V^(-1)+U^(-1)V^(-2).  A fiber W_CPS=w is the double cover with
# quartic (wV-V^2-2)^2-4.  Compute its binary-quartic j-invariant and verify
# that the naive parameter identification w=s does not recover the suspended
# GKT quartic pencil.
Rw = PolynomialRing(QQ, names=("w",))
(w,) = Rw.gens()
RV = PolynomialRing(Rw, names=("v",))
(v,) = RV.gens()
f_cps = (w * v - v**2 - 2)**2 - 4
f_coeffs = [f_cps[i] for i in range(5)]
e_cps, d_cps, c_cps, b_cps, a_cps = f_coeffs
I_cps = 12 * a_cps * e_cps - 3 * b_cps * d_cps + c_cps**2
J_cps = (
    72 * a_cps * c_cps * e_cps
    + 9 * b_cps * c_cps * d_cps
    - 27 * a_cps * d_cps**2
    - 27 * b_cps**2 * e_cps
    - 2 * c_cps**3
)
disc_cps = 4 * I_cps**3 - J_cps**2
assert I_cps == w**4 - 16 * w**2 + 16
assert disc_cps == 6912 * (w - 4) * (w + 4) * w**2
j_cps = Rw.fraction_field()(6912 * I_cps**3 / disc_cps)
j_cps_expected = Rw.fraction_field()(
    (w**4 - 16 * w**2 + 16)**3 / (w**2 * (w**2 - 16))
)
assert j_cps == j_cps_expected
j_gkt_naive = Rw.fraction_field()(16 * (w**2 + 12)**3 / (w**2 - 4)**2)
assert j_cps != j_gkt_naive

# The correct elliptic comparison varies the free middle coefficient a of the
# order-two slab 1+a*z+z^2 (the slab parameter, written a_rho in the paper)
# and takes the distinguished fiber W_a=0.  The
# resulting quartic is v^4+2*a*v^2+(a^2-4).  After h^2=a^2-4 and the standard
# quartic rescaling, its GKT parameter is s=2*a/h.  Verify the induced
# j-invariant identity without adjoining h, using s^2=4*a^2/(a^2-4).
Ra = PolynomialRing(QQ, names=("a",))
(a,) = Ra.gens()
Rva = PolynomialRing(Ra, names=("va",))
(va,) = Rva.gens()
f_slab_zero = (va**2 + a)**2 - 4
assert f_slab_zero == va**4 + 2 * a * va**2 + a**2 - 4
a4 = f_slab_zero[4]
a3 = f_slab_zero[3]
a2 = f_slab_zero[2]
a1 = f_slab_zero[1]
a0 = f_slab_zero[0]
I_slab = 12 * a4 * a0 - 3 * a3 * a1 + a2**2
J_slab = (
    72 * a4 * a2 * a0
    + 9 * a3 * a2 * a1
    - 27 * a4 * a1**2
    - 27 * a3**2 * a0
    - 2 * a2**3
)
disc_slab = 4 * I_slab**3 - J_slab**2
assert I_slab == 16 * (a**2 - 3)
assert J_slab == 128 * a * (a**2 - QQ(9) / 2)
assert disc_slab == 110592 * (a - 2) * (a + 2)
Ka = Ra.fraction_field()
j_slab = Ka(6912 * I_slab**3 / disc_slab)
j_slab_expected = Ka(256 * (a**2 - 3)**3 / (a**2 - 4))
assert j_slab == j_slab_expected
s_squared_via_a = Ka(4 * a**2 / (a**2 - 4))
j_gkt_via_a = Ka(
    16 * (s_squared_via_a + 12)**3 / (s_squared_via_a - 4)**2
)
assert j_slab == j_gkt_via_a

# The fourth-root Kummer cover of the variable B-slab changes type at the two
# cusps a=+/-2.  The generic class has order four by the simple zero
# valuations of 1+a*z+z^2.  At a cusp the slab is a square, and the degree-four
# equation splits into two quadratic components.  The valuation argument is
# given in the proof of the infinite Kummer orbit theorem (Section 2.5); this
# block checks the identities.
Rzg = PolynomialRing(QQ, names=("zroot", "groot"))
(zroot, groot) = Rzg.gens()
positive_cusp_slab = (1 + zroot) ** 2
negative_cusp_slab = (1 - zroot) ** 2
assert groot**4 - positive_cusp_slab == (
    groot**2 - (1 + zroot)
) * (groot**2 + (1 + zroot))
assert groot**4 - negative_cusp_slab == (
    groot**2 - (1 - zroot)
) * (groot**2 + (1 - zroot))

# Inverting the map gives a^2=4*s^2/(s^2-4).  Hence the unresolved value
# a=2 is the s=infinity cusp.  If s_eff=s0+1/(4*q^2), then
# a=2+64*q^4-512*s0*q^6+O(q^8).
Rs0 = PolynomialRing(QQ, names=("s0",))
(s0,) = Rs0.gens()
Kq = PowerSeriesRing(Rs0, names=("qsmall",), default_prec=10)
(qsmall,) = Kq.gens()
inverse_s_eff_series = 4 * qsmall**2 / (1 + 4 * s0 * qsmall**2)
a_series = 2 * (1 - 4 * inverse_s_eff_series**2) ** (-QQ(1) / 2)
assert a_series[0] == 2
assert a_series[4] == 64
assert a_series[6] == -512 * s0

# The marginal GKT torus is a change of endpoint trivialization at fixed
# elliptic/slab modulus.  On an edge polynomial A*X^4+B*X^2Y^2+C*Y^4,
# exp(tau*D) sends (A,B,C) to (e^(4tau)A,B,e^(-4tau)C), preserving
# B^2/(A*C).  For the two extreme gauges through t^2, the exact formal gauge
# parameter has leading coefficient -1/32.
Kt = PowerSeriesRing(QQ, names=("t",), default_prec=10)
(t,) = Kt.gens()
endpoint_correction = 1 + t**2 / 8
tau_extreme = -log(endpoint_correction) / 4
assert exp(4 * tau_extreme) * endpoint_correction == 1
assert exp(-4 * tau_extreme) == endpoint_correction
assert tau_extreme[2] == -QQ(1) / 32

# The normalized edge parameter is identical in the two extreme gauges.
edge_parameter_min_squared = t**2 / (endpoint_correction * 1)
edge_parameter_max_squared = t**2 / (1 * endpoint_correction)
assert edge_parameter_min_squared == edge_parameter_max_squared

# Gross-Hacking-Siebert Proposition 5.2 acts on a torus monomial z^m by a
# character t_sigma(m).  The character exp(4*f*m_1) has weights matching D on
# the quartic edge under the CPS monomial dictionary.
def endpoint_weight(m):
    return 4 * m[0]


assert endpoint_weight(v_X) == 4          # X^4
assert endpoint_weight(m_star) == 0       # X^2 Y^2
assert endpoint_weight(v_Y) == -4         # Y^4
assert endpoint_weight(v_Z) == 0          # the fourth CPS vertex term

# The quartic-edge dictionary is the rational exponent map F=iota/4:
# X^4, X^2*Y^2, Y^4 map to v_X, m_*, v_Y.  The first nonmarginal descendent
# generator t_(1,0,1) X_(1,0) sends X^4+Y^4 to 4*X^5-8*X*Y^4.  Its two
# exponents lie in one quarter-lattice coset, not in the ordinary CPS lattice.
# Thus the unextended Laurent ring cannot carry this derivation; a graded
# line-bundle/root-stack extension is required.
quartic_exponent_map = matrix(QQ, iota) / 4
assert quartic_exponent_map * vector(ZZ, (4, 0)) == vector(QQ, v_X)
assert quartic_exponent_map * vector(ZZ, (2, 2)) == vector(QQ, m_star)
assert quartic_exponent_map * vector(ZZ, (0, 4)) == vector(QQ, v_Y)

first_desc_shift = quartic_exponent_map * vector(ZZ, (1, 0))
first_desc_exponents = [
    quartic_exponent_map * vector(ZZ, (5, 0)),
    quartic_exponent_map * vector(ZZ, (1, 4)),
]
assert first_desc_shift == vector(QQ, (QQ(1) / 4, 0))
assert first_desc_exponents == [
    vector(QQ, (QQ(5) / 4, 0)),
    vector(QQ, (-QQ(3) / 4, -2)),
]
assert all(not all(c in ZZ for c in exponent) for exponent in first_desc_exponents)
assert first_desc_exponents[0] - first_desc_exponents[1] == vector(ZZ, (2, 2))

# The shifted boundary support remains integral under the cofactor map.  The
# obstruction is therefore in the monomial/section lattice, not wall support.
first_desc_boundary_charge = vector(ZZ, (2, 1))
assert iota_cofactor * first_desc_boundary_charge == vector(ZZ, (-4, 3))

second_desc_shift = quartic_exponent_map * vector(ZZ, (0, 1))
second_desc_exponents = [
    quartic_exponent_map * vector(ZZ, (4, 1)),
    quartic_exponent_map * vector(ZZ, (0, 5)),
]
assert second_desc_shift == vector(QQ, (-QQ(1) / 4, -QQ(1) / 2))
assert second_desc_exponents == [
    vector(QQ, (QQ(3) / 4, -QQ(1) / 2)),
    vector(QQ, (-QQ(5) / 4, -QQ(5) / 2)),
]
assert all(not all(c in ZZ for c in exponent) for exponent in second_desc_exponents)
assert second_desc_exponents[0] - second_desc_exponents[1] == vector(ZZ, (2, 2))
second_desc_boundary_charge = vector(ZZ, (1, 2))
assert iota_cofactor * second_desc_boundary_charge == vector(ZZ, (-2, 3))

# The cofactor directions have a canonical compact radial realization in the
# CPS triangle.  From the central lattice point, the two singleton rays hit
# the two boundary edges adjacent to p_B, while their first bracket points
# exactly toward p_B.  Every positive combination stays in this upper wedge.
support_origin = vector(QQ, (0, 0))
first_desc_support_direction = vector(
    QQ, iota_cofactor * first_desc_boundary_charge
)
second_desc_support_direction = vector(
    QQ, iota_cofactor * second_desc_boundary_charge
)
first_bracket_boundary_charge = vector(ZZ, (2, 2))
first_bracket_support_direction = vector(
    QQ, iota_cofactor * first_bracket_boundary_charge
)
first_desc_support_endpoint = support_origin + (
    QQ(1) / 4
) * first_desc_support_direction
second_desc_support_endpoint = support_origin + (
    QQ(1) / 4
) * second_desc_support_direction
first_bracket_support_endpoint = support_origin + (
    QQ(1) / 4
) * first_bracket_support_direction
assert first_desc_support_endpoint == vector(QQ, (-1, QQ(3) / 4))
assert second_desc_support_endpoint == vector(QQ, (-QQ(1) / 2, QQ(3) / 4))
assert first_bracket_support_endpoint == vector(QQ, p_B)


def lies_in_delta(point):
    return (
        point[0] >= -1
        and point[1] >= -1
        and point[0] + 2 * point[1] <= 1
    )


for endpoint in (
    first_desc_support_endpoint,
    second_desc_support_endpoint,
    first_bracket_support_endpoint,
):
    assert lies_in_delta(endpoint)
assert first_desc_support_endpoint[0] == -1
assert (
    second_desc_support_endpoint[0]
    + 2 * second_desc_support_endpoint[1]
    == 1
)
assert first_bracket_support_endpoint[0] == -1
assert (
    first_bracket_support_endpoint[0]
    + 2 * first_bracket_support_endpoint[1]
    == 1
)
assert first_desc_support_direction + second_desc_support_direction == (
    QQ(3) / 2
) * first_bracket_support_direction

# The singleton supports must be full chords through O, not positive
# half-rays: the local word at the joint (Section 3.3) crosses each incoming
# wall twice.
# Their negative halves meet the bottom facet at the following exact points.
first_desc_negative_endpoint = -QQ(1) / 3 * first_desc_support_direction
second_desc_negative_endpoint = -QQ(1) / 3 * second_desc_support_direction
assert first_desc_negative_endpoint == vector(QQ, (QQ(4) / 3, -1))
assert second_desc_negative_endpoint == vector(QQ, (QQ(2) / 3, -1))
assert lies_in_delta(first_desc_negative_endpoint)
assert lies_in_delta(second_desc_negative_endpoint)

# The monodromy-related minus cofactor chart has the same compact incidence
# pattern: the two singleton rays hit the two boundary edges and the diagonal
# ray still ends at p_B.
first_desc_support_direction_minus = vector(
    QQ, iota_cofactor_minus * first_desc_boundary_charge
)
second_desc_support_direction_minus = vector(
    QQ, iota_cofactor_minus * second_desc_boundary_charge
)
first_bracket_support_direction_minus = vector(
    QQ, iota_cofactor_minus * first_bracket_boundary_charge
)
assert first_desc_support_direction_minus == vector(QQ, (-6, 5))
assert second_desc_support_direction_minus == vector(QQ, (0, 1))
assert first_bracket_support_direction_minus == vector(QQ, (-4, 4))
first_desc_support_endpoint_minus = (
    QQ(1) / 6 * first_desc_support_direction_minus
)
second_desc_support_endpoint_minus = (
    QQ(1) / 2 * second_desc_support_direction_minus
)
first_bracket_support_endpoint_minus = (
    QQ(1) / 4 * first_bracket_support_direction_minus
)
assert first_desc_support_endpoint_minus == vector(QQ, (-1, QQ(5) / 6))
assert second_desc_support_endpoint_minus == vector(QQ, (0, QQ(1) / 2))
assert first_bracket_support_endpoint_minus == vector(QQ, p_B)
first_desc_negative_endpoint_minus = (
    -QQ(1) / 5 * first_desc_support_direction_minus
)
second_desc_negative_endpoint_minus = -second_desc_support_direction_minus
assert first_desc_negative_endpoint_minus == vector(QQ, (QQ(6) / 5, -1))
assert second_desc_negative_endpoint_minus == vector(QQ, (0, -1))
for endpoint in (
    first_desc_support_endpoint_minus,
    second_desc_support_endpoint_minus,
    first_bracket_support_endpoint_minus,
):
    assert lies_in_delta(endpoint)

# A positive cone coordinate (a,b) maps to
# (-4a-2b,3a+3b).  Its ray exits through x=-1 or x+2y=1 at finite positive
# time.  The two exit times coincide only on the bracket direction a=b.
SupportConeRing.<support_a, support_b> = PolynomialRing(QQ, 2)
general_support_direction = (
    support_a * vector(SupportConeRing, first_desc_support_direction)
    + support_b * vector(SupportConeRing, second_desc_support_direction)
)
assert general_support_direction == vector(
    SupportConeRing,
    (-4 * support_a - 2 * support_b, 3 * support_a + 3 * support_b),
)
left_edge_exit_denominator = -general_support_direction[0]
right_edge_exit_denominator = (
    general_support_direction[0] + 2 * general_support_direction[1]
)
assert left_edge_exit_denominator == 4 * support_a + 2 * support_b
assert right_edge_exit_denominator == 2 * support_a + 4 * support_b
assert left_edge_exit_denominator - right_edge_exit_denominator == (
    2 * (support_a - support_b)
)
assert general_support_direction[0] + general_support_direction[1] == (
    -support_a + support_b
)

# The two off-diagonal families are exchanged by an intrinsic integral
# reflection of the CPS triangle.  In charge coordinates this is a<->b; in
# the cofactor chart it is L*S*L^(-1).  The reflection fixes the B radial line,
# swaps A with C and AB with BC, and exchanges the two rational end cones.
charge_swap = matrix(ZZ, [[0, 1], [1, 0]])
support_reflection = iota_cofactor * charge_swap * iota_cofactor.inverse()
assert support_reflection == matrix(ZZ, [[-1, -2], [0, 1]])
assert support_reflection**2 == identity_matrix(ZZ, 2)
assert support_reflection.det() == -1
assert support_reflection * p_A == p_C
assert support_reflection * p_B == p_B
assert support_reflection * p_C == p_A
assert support_reflection * first_desc_support_direction == (
    second_desc_support_direction
)
assert support_reflection * second_desc_support_direction == (
    first_desc_support_direction
)
assert support_reflection * first_bracket_support_direction == (
    first_bracket_support_direction
)
assert support_reflection * first_desc_support_endpoint == (
    second_desc_support_endpoint
)

left_end_cone_generators = matrix(
    ZZ,
    [first_desc_support_direction, first_bracket_support_direction],
)
right_end_cone_generators = matrix(
    ZZ,
    [first_bracket_support_direction, second_desc_support_direction],
)
assert left_end_cone_generators.det() == -4
assert right_end_cone_generators.det() == -4
assert support_reflection * left_end_cone_generators.transpose() == matrix(
    ZZ,
    [second_desc_support_direction, first_bracket_support_direction],
).transpose()

# For a>b>=0 the ray meets the relative interior of AB at height
# 3(a+b)/(4a+2b); for b>a>=0 it meets the relative interior of BC at height
# 3(a+b)/(2a+4b).  These polynomial numerator identities certify that both
# heights lie in [3/4,1), with equality 1 only on the excluded diagonal.
left_end_height_denominator = 4 * support_a + 2 * support_b
right_end_height_denominator = 2 * support_a + 4 * support_b
assert (
    4 * 3 * (support_a + support_b)
    - 3 * left_end_height_denominator
) == 6 * support_b
assert (
    left_end_height_denominator - 3 * (support_a + support_b)
) == support_a - support_b
assert (
    4 * 3 * (support_a + support_b)
    - 3 * right_end_height_denominator
) == 6 * support_a
assert (
    right_end_height_denominator - 3 * (support_a + support_b)
) == support_b - support_a

# With A and C at the midpoints of their star edges, the straight A--C
# branch segment is y=-1/2.  It is disjoint from the outgoing singleton wedge,
# whose noncentral points have y>0, but each incoming singleton chord crosses
# it once.  In the presentation by crosswise gluing along this segment the
# lower chord half passes to the other copy of the disk, and each lift of O
# retains a full local line.  The segment carries no affine or coefficient
# transition.
barycentric_focus_A = QQ(1) / 2 * vector(QQ, p_A)
barycentric_focus_C = QQ(1) / 2 * vector(QQ, p_C)
assert barycentric_focus_A == vector(QQ, (-QQ(1) / 2, -QQ(1) / 2))
assert barycentric_focus_C == vector(QQ, (QQ(3) / 2, -QQ(1) / 2))
assert barycentric_focus_A[1] == barycentric_focus_C[1] == -QQ(1) / 2
assert all(endpoint[1] > 0 for endpoint in (
    first_desc_support_endpoint,
    second_desc_support_endpoint,
    first_bracket_support_endpoint,
))
first_chord_branch_crossing = -QQ(1) / 6 * first_desc_support_direction
second_chord_branch_crossing = -QQ(1) / 6 * second_desc_support_direction
assert first_chord_branch_crossing == vector(QQ, (QQ(2) / 3, -QQ(1) / 2))
assert second_chord_branch_crossing == vector(QQ, (QQ(1) / 3, -QQ(1) / 2))
for crossing in (first_chord_branch_crossing, second_chord_branch_crossing):
    assert barycentric_focus_A[0] < crossing[0] < barycentric_focus_C[0]

# The fractional cosets are not ad hoc: the minimal lattice containing every
# GKT exponent shift is M_tilde=F(Z^2).  It contains the original CPS lattice
# M with index eight and is preserved by the two order-two monodromy
# representatives.  Preservation by the other facets is tested below rather
# than assumed.
F = quartic_exponent_map
F_inverse = F.inverse()
assert F_inverse == matrix(ZZ, [[4, -2], [0, -2]])
assert abs(F_inverse.det()) == 8
M2_on_cover = F_inverse * M2 * F
M2_check_on_cover = F_inverse * M2_check * F
assert M2_on_cover == matrix(ZZ, [[2, 1], [-1, 0]])
assert M2_check_on_cover == matrix(ZZ, [[4, -9], [1, -2]])
assert M2_on_cover.det() == 1
assert M2_check_on_cover.det() == 1

# Conjugate the full GKT Hamiltonian algebra.  For k=(k1,k2), let
# q=k+(1,1) and n_source=(q2,-q1), so
# X_k(z^a)=<n_source,a> z^(a+k).  On M_tilde the exponent is F*k and the
# integral dual vector is F^(-T)*n_source.  The marginal vector gives exactly
# the GHS endpoint character (4,0).
def transported_normal(k):
    q = k + vector(ZZ, (1, 1))
    n_source = vector(ZZ, (q[1], -q[0]))
    return F_inverse.transpose() * n_source


marginal_normal = transported_normal(vector(ZZ, (0, 0)))
first_desc_normal = transported_normal(vector(ZZ, (1, 0)))
second_desc_normal = transported_normal(vector(ZZ, (0, 1)))
assert marginal_normal == vector(ZZ, (4, 0))
assert first_desc_normal == vector(ZZ, (4, 2))
assert second_desc_normal == vector(ZZ, (8, -2))
assert first_desc_normal.dot_product(v_X) == 4
assert first_desc_normal.dot_product(v_Y) == -8
assert second_desc_normal.dot_product(v_X) == 8
assert second_desc_normal.dot_product(v_Y) == -4

# Global facet obstruction.  The order-one monodromies at A and C do not
# preserve M_tilde, although the order-two monodromy at B does.  The smallest
# lattice containing M_tilde and stable under all three monodromies is
# (1/4)Z^2: F(e_1) supplies (1/4,0), and T_A F(e_1) supplies (0,1/4).
# On that larger lattice the first two descendent normals are not integral,
# because they pair to half-integers with (0,1/4).  Thus the unbranched
# index-eight torus cover cannot carry the two singleton walls globally as
# ordinary integral GHS automorphisms.
facet_monodromy_on_cover = {
    label: F_inverse * monodromy * F
    for label, monodromy in facet_monodromy.items()
}
assert facet_monodromy_on_cover["A"] == matrix(
    QQ, [[-QQ(1) / 2, QQ(9) / 2], [-QQ(1) / 2, QQ(5) / 2]]
)
assert facet_monodromy_on_cover["B"] == M2_on_cover
assert facet_monodromy_on_cover["C"] == matrix(
    QQ, [[-QQ(1) / 2, QQ(1) / 2], [-QQ(9) / 2, QQ(5) / 2]]
)
assert not all(entry in ZZ for entry in facet_monodromy_on_cover["A"].list())
assert all(entry in ZZ for entry in facet_monodromy_on_cover["B"].list())
assert not all(entry in ZZ for entry in facet_monodromy_on_cover["C"].list())
quarter_x = F * vector(ZZ, (1, 0))
quarter_y = facet_monodromy["A"] * quarter_x
assert quarter_x == vector(QQ, (QQ(1) / 4, 0))
assert quarter_y == vector(QQ, (0, QQ(1) / 4))
assert first_desc_normal.dot_product(quarter_y) == QQ(1) / 2
assert second_desc_normal.dot_product(quarter_y) == -QQ(1) / 2

# The parity double cover passes the lattice test.  On the double cover of
# the punctured affine base whose parity character is nontrivial around A and
# C and trivial around B, the lifted local monodromies are T_A^2, T_B, T_C^2.
# All three preserve M_tilde.  Only this necessary integral monodromy
# condition is checked here; the cover itself is constructed in Section 2.3.
facet_monodromy_on_base_double_cover = {
    "A": F_inverse * facet_monodromy["A"] ** 2 * F,
    "B": F_inverse * facet_monodromy["B"] * F,
    "C": F_inverse * facet_monodromy["C"] ** 2 * F,
}
assert facet_monodromy_on_base_double_cover == {
    "A": matrix(ZZ, [[-2, 9], [-1, 4]]),
    "B": matrix(ZZ, [[2, 1], [-1, 0]]),
    "C": matrix(ZZ, [[-2, 1], [-9, 4]]),
}

# The local-square test is not enough: the index-two subgroup also contains
# mixed even-parity loops.  With coset representatives {1,A}, a
# Reidemeister--Schreier generating set for the kernel of
# chi(A)=chi(C)=1, chi(B)=0 is represented by the following words.  Every one
# preserves M_tilde, so the full parity subgroup, not only the three local
# lifted monodromies, passes the lattice test.
T_A = facet_monodromy["A"]
T_B = facet_monodromy["B"]
T_C = facet_monodromy["C"]
parity_kernel_monodromies = {
    "B": T_B,
    "C*A^-1": T_C * T_A.inverse(),
    "A^2": T_A**2,
    "A*B*A^-1": T_A * T_B * T_A.inverse(),
    "A*C": T_A * T_C,
}
parity_kernel_on_cover = {
    label: F_inverse * monodromy * F
    for label, monodromy in parity_kernel_monodromies.items()
}
assert parity_kernel_on_cover == {
    "B": matrix(ZZ, [[2, 1], [-1, 0]]),
    "C*A^-1": matrix(ZZ, [[-1, 2], [-10, 19]]),
    "A^2": matrix(ZZ, [[-2, 9], [-1, 4]]),
    "A*B*A^-1": matrix(ZZ, [[-14, 25], [-9, 16]]),
    "A*C": matrix(ZZ, [[-20, 11], [-11, 6]]),
}
assert all(
    all(entry in ZZ for entry in monodromy.list())
    for monodromy in parity_kernel_on_cover.values()
)

# The completed parity cover has four singular points, not three: A and C
# each have one ramified lift, while the unbranched point B has two lifts.
# In one fixed F-basis the second B monodromy is the conjugate A*B*A^(-1).
# All four lifted monodromies are primitive unipotent on M_tilde.  B_0 and B_1
# are focus-focus points; the branch points over A and C are pullbacks of
# focus-focus germs under z -> z^2 (Theorem 2.3).
lifted_singularity_monodromies = {
    "A": facet_monodromy_on_base_double_cover["A"],
    "B_0": facet_monodromy_on_base_double_cover["B"],
    "B_1": parity_kernel_on_cover["A*B*A^-1"],
    "C": facet_monodromy_on_base_double_cover["C"],
}
lifted_rank_one_factorizations = {
    "A": (
        vector(ZZ, (3, 1)).column(),
        vector(ZZ, (-1, 3)).row(),
    ),
    "B_0": (
        vector(ZZ, (1, -1)).column(),
        vector(ZZ, (1, 1)).row(),
    ),
    "B_1": (
        vector(ZZ, (5, 3)).column(),
        vector(ZZ, (-3, 5)).row(),
    ),
    "C": (
        vector(ZZ, (1, 3)).column(),
        vector(ZZ, (-3, 1)).row(),
    ),
}
for label, monodromy in lifted_singularity_monodromies.items():
    column, row = lifted_rank_one_factorizations[label]
    assert monodromy - identity_matrix(ZZ, 2) == column * row
    assert gcd([abs(entry) for entry in column.list()]) == 1
    assert gcd([abs(entry) for entry in row.list()]) == 1

# The old primitive invariant directions become twice primitive in M_tilde.
# The corresponding cover-dual conormals are primitive elements of
# M_tilde^vee=F^(-T) Z^2, written here in the old rational coordinates.
lifted_primitive_directions = {
    label: F * lifted_rank_one_factorizations[label][0].column(0)
    for label in lifted_rank_one_factorizations
}
assert lifted_primitive_directions == {
    "A": vector(QQ, (QQ(1) / 2, -QQ(1) / 2)),
    "B_0": vector(QQ, (QQ(1) / 2, QQ(1) / 2)),
    "B_1": vector(QQ, (QQ(1) / 2, -QQ(3) / 2)),
    "C": vector(QQ, (-QQ(1) / 2, -QQ(3) / 2)),
}
lifted_primitive_conormals = {
    "A": vector(ZZ, (4, 4)),
    "B_0": vector(ZZ, (-4, 4)),
    "B_1": vector(ZZ, (-12, -4)),
    "C": vector(ZZ, (-12, 4)),
}
for label in lifted_primitive_directions:
    conormal = lifted_primitive_conormals[label]
    assert conormal.dot_product(lifted_primitive_directions[label]) == 0
    conormal_in_cover_basis = F.transpose() * conormal
    assert all(entry in ZZ for entry in conormal_in_cover_basis)
    assert gcd([abs(entry) for entry in conormal_in_cover_basis]) == 1

# The pulled-back CPS polarization is only quarter-integral for M_tilde.
# Multiplication by four is both sufficient and necessary to make the three
# old primitive facet conormals primitive in the cover-dual lattice.
old_facet_conormals = {
    "A": vector(ZZ, (1, 1)),
    "B": vector(ZZ, (-1, 1)),
    "C": vector(ZZ, (-3, 1)),
}
old_facet_conormals_in_cover_coordinates = {
    label: F.transpose() * conormal
    for label, conormal in old_facet_conormals.items()
}
assert old_facet_conormals_in_cover_coordinates == {
    "A": vector(QQ, (QQ(1) / 4, -QQ(3) / 4)),
    "B": vector(QQ, (-QQ(1) / 4, -QQ(1) / 4)),
    "C": vector(QQ, (-QQ(3) / 4, QQ(1) / 4)),
}
for coordinates in old_facet_conormals_in_cover_coordinates.values():
    assert not all(entry in ZZ for entry in coordinates)
    integral_coordinates = 4 * coordinates
    assert all(entry in ZZ for entry in integral_coordinates)
    assert gcd([abs(entry) for entry in integral_coordinates]) == 1

# Consequently the old slab transition attached to a primitive old conormal
# does not act on all M_tilde monomials.  Around the branched A and C points
# the spatial loop squares that transition, leaving denominator two; at either
# unbranched B lift the denominator remains four.  Preserving the old CPS
# transition therefore requires roots g_A^2=f_A, g_C^2=f_C and
# g_{B,+}^4=g_{B,-}^4=f_B.  These assertions record the exact denominators;
# existence and cocycle compatibility of the roots are geometric obligations.
slab_root_degrees = {"A": 2, "B_0": 4, "B_1": 4, "C": 2}
assert slab_root_degrees == {"A": 2, "B_0": 4, "B_1": 4, "C": 2}

# The finite quotient M_tilde/M packages the missing root characters.  In the
# F-basis, M_tilde is Z^2 and the old CPS lattice M is generated by the
# columns of F^(-1).  The quotient has invariant factors (2,4).  The two GKT
# singleton shifts are the classes of the two F-basis vectors; each has order
# four, while their sum (the grade of their first bracket) has order two.
# Gross--Siebert wall monomials include the common primitive-form shift
# rho=(1,1).  Their grades are therefore 2e_1+e_2 and e_1+2e_2; these again
# have order four, are exchanged by peripheral monodromy, and bracket into
# the same distinguished order-two grade e_1+e_2.
cover_coordinate_lattice = ZZ**2
old_lattice_in_cover_coordinates = cover_coordinate_lattice.submodule(
    F_inverse.columns()
)
sector_group = cover_coordinate_lattice / old_lattice_in_cover_coordinates
assert sector_group.invariants() == (2, 4)
sector_shift_first = sector_group(cover_coordinate_lattice.gen(0))
sector_shift_second = sector_group(cover_coordinate_lattice.gen(1))
sector_bracket = sector_shift_first + sector_shift_second
wall_grade_first_vector = vector(ZZ, (2, 1))
wall_grade_second_vector = vector(ZZ, (1, 2))
sector_wall_first = sector_group(wall_grade_first_vector)
sector_wall_second = sector_group(wall_grade_second_vector)
assert sector_shift_first.additive_order() == 4
assert sector_shift_second.additive_order() == 4
assert sector_wall_first.additive_order() == 4
assert sector_wall_second.additive_order() == 4
assert sector_bracket.additive_order() == 2
assert sector_wall_first + sector_wall_second == sector_bracket

# A single monodromy-invariant diagonal charge controls all scalar root
# phases.  The homomorphism s:D -> Z/4 is induced in the F-basis by
# s(x,y)=x+y mod 4.  It is well-defined because it kills both generators of
# the old CPS lattice.  Its kernel is the order-two anti-diagonal class, while
# the singleton shifts have charge one, their shifted wall monomials charge
# three, and their bracket charge two.
root_charge_ring = Integers(4)


def diagonal_root_charge(exponent):
    return root_charge_ring(exponent[0] + exponent[1])


for old_generator in old_lattice_in_cover_coordinates.gens():
    assert diagonal_root_charge(old_generator) == 0
sector_anti_diagonal = sector_shift_first - sector_shift_second
assert sector_anti_diagonal.additive_order() == 2
assert diagonal_root_charge(vector(ZZ, (1, -1))) == 0
assert diagonal_root_charge(vector(ZZ, (1, 0))) == 1
assert diagonal_root_charge(vector(ZZ, (0, 1))) == 1
assert diagonal_root_charge(wall_grade_first_vector) == 3
assert diagonal_root_charge(wall_grade_second_vector) == 3
assert diagonal_root_charge(vector(ZZ, (1, 1))) == 2
assert sector_group.cardinality() == 8

# GKT retain ordered twist pairs in (Z/4)^2.  The annular sector group D is
# their quotient by the diagonal order-two class (2,2).  Thus the sector
# double cover retains the alpha/beta swap but cannot restore the remaining
# diagonal parity.  The extension is nonsplit: (Z/4)^2 has four elements
# killed by two, whereas D x Z/2 would have eight.
gkt_twist_lattice = ZZ ** 2
gkt_twist_relations = gkt_twist_lattice.submodule(
    [4 * gkt_twist_lattice.gen(0), 4 * gkt_twist_lattice.gen(1)]
)
gkt_twist_group = gkt_twist_lattice / gkt_twist_relations
assert gkt_twist_group.invariants() == (4, 4)
gkt_to_annular_kernel_representatives = {
    (first_twist, second_twist)
    for first_twist in range(4)
    for second_twist in range(4)
    if sector_group(vector(ZZ, (first_twist, second_twist))) == 0
}
assert gkt_to_annular_kernel_representatives == {(0, 0), (2, 2)}
assert sector_group(vector(ZZ, (1, 0))) == sector_shift_first
assert sector_group(vector(ZZ, (0, 1))) == sector_shift_second
assert sector_group(vector(ZZ, (1, 1))) == sector_bracket
gkt_two_torsion = {
    gkt_twist_group(vector(ZZ, (first_twist, second_twist)))
    for first_twist in range(4)
    for second_twist in range(4)
    if 2 * gkt_twist_group(vector(ZZ, (first_twist, second_twist))) == 0
}
annular_sector_two_torsion = {
    sector_group(vector(ZZ, (first_twist, second_twist)))
    for first_twist in range(4)
    for second_twist in range(4)
    if 2 * sector_group(vector(ZZ, (first_twist, second_twist))) == 0
}
sector_times_sign_two_torsion_size = len(annular_sector_two_torsion) * 2
assert len(gkt_two_torsion) == 4
assert sector_times_sign_two_torsion_size == 8

# Thus 0 -> <alpha-beta> -> D -> Z/4 -> 0 is exact.  It does not split as a
# monodromy local system: the peripheral involution has no fixed charge-one
# class.  Representatives with coordinates in {0,1,2,3} cover D and make the
# finite check explicit.
charge_one_representatives = [
    vector(ZZ, (x_coordinate, y_coordinate))
    for x_coordinate in range(4)
    for y_coordinate in range(4)
    if diagonal_root_charge(vector(ZZ, (x_coordinate, y_coordinate))) == 1
]
assert charge_one_representatives
assert not any(
    all(
        sector_group(monodromy * representative) == sector_group(representative)
        for monodromy in lifted_singularity_monodromies.values()
    )
    for representative in charge_one_representatives
)

# The diagonal quotient is globally constant on the annular local system:
# every Reidemeister--Schreier generator preserves s, including the
# hyperbolic core generator.  This is stronger than checking only peripheral
# monodromies.
for monodromy in parity_kernel_on_cover.values():
    for generator in cover_coordinate_lattice.gens():
        assert diagonal_root_charge(monodromy * generator) == (
            diagonal_root_charge(generator)
        )

# The full sector monodromy has image generated by the singleton swap.  Its
# coinvariants impose e_1=e_2, so the canonical charge s is the maximal
# constant quotient and has group Z/4.  Its invariant subgroup has four
# elements, generated by the anti-diagonal class and 2*alpha.
singleton_swap = matrix(ZZ, [[0, 1], [1, 0]])
sector_coinvariant_relations = cover_coordinate_lattice.submodule(
    list(old_lattice_in_cover_coordinates.gens())
    + [cover_coordinate_lattice.gen(0) - cover_coordinate_lattice.gen(1)]
)
sector_coinvariants = cover_coordinate_lattice / sector_coinvariant_relations
assert sector_coinvariants.invariants() == (4,)
sector_elements = {
    sector_group(vector(ZZ, (x_coordinate, y_coordinate)))
    for x_coordinate in range(4)
    for y_coordinate in range(4)
}
assert len(sector_elements) == 8
sector_fixed_elements = {
    sector_group(representative)
    for x_coordinate in range(4)
    for y_coordinate in range(4)
    for representative in [vector(ZZ, (x_coordinate, y_coordinate))]
    if sector_group(singleton_swap * representative) == sector_group(representative)
}
sector_fixed_expected = {
    sector_group.zero(),
    sector_anti_diagonal,
    2 * sector_shift_first,
    sector_bracket,
}
assert sector_fixed_elements == sector_fixed_expected
assert len(sector_fixed_elements) == 4
assert all(element.additive_order() <= 2 for element in sector_fixed_elements)

# In the abstract splitting D=<alpha>+<delta>, peripheral transport fixes the
# Z/4 charge j and translates the anti-diagonal coordinate by j mod 2.  Hence
# the double cover nontrivial around each of the four punctures and trivial
# around the core is the minimal sector-trivializing cover.
sector_representatives = [
    vector(ZZ, (x_coordinate, y_coordinate))
    for x_coordinate in range(4)
    for y_coordinate in range(4)
]
for monodromy in lifted_singularity_monodromies.values():
    for representative in sector_representatives:
        charge_parity = ZZ(diagonal_root_charge(representative)) % 2
        assert sector_group(monodromy * representative) == (
            sector_group(representative) + charge_parity * sector_anti_diagonal
        )
sector_core_monodromy = parity_kernel_on_cover["C*A^-1"]
for representative in sector_representatives:
    assert sector_group(sector_core_monodromy * representative) == (
        sector_group(representative)
    )

# Riemann--Hurwitz for the compact extension: a double cover of an annulus
# branched at four interior points has Euler characteristic -4.  Both boundary
# monodromies are trivial, so there are four boundary components and genus 1.
sector_cover_euler_characteristic = 2 * 0 - 4
sector_cover_boundary_components = 4
sector_cover_genus = (
    2 - sector_cover_boundary_components - sector_cover_euler_characteristic
) // 2
assert sector_cover_euler_characteristic == -4
assert sector_cover_genus == 1

# With the coherent cover-dual conormal orientations fixed above, the four
# root-phase characters are 2s,-s,s,2s for A,B_0,B_1,C.  The factor two at
# A,C embeds their mu_2 phases into the common mu_4 band.  Their sum vanishes,
# so the finite-character part of the peripheral descent cocycle closes.
lifted_conormal_cover_coordinates = {
    label: vector(ZZ, F.transpose() * conormal)
    for label, conormal in lifted_primitive_conormals.items()
}
assert lifted_conormal_cover_coordinates == {
    "A": vector(ZZ, (1, -3)),
    "B_0": vector(ZZ, (-1, -1)),
    "B_1": vector(ZZ, (-3, 5)),
    "C": vector(ZZ, (-3, 1)),
}
root_phase_rows_mod_four = {
    "A": vector(root_charge_ring, 2 * lifted_conormal_cover_coordinates["A"]),
    "B_0": vector(root_charge_ring, lifted_conormal_cover_coordinates["B_0"]),
    "B_1": vector(root_charge_ring, lifted_conormal_cover_coordinates["B_1"]),
    "C": vector(root_charge_ring, 2 * lifted_conormal_cover_coordinates["C"]),
}
assert root_phase_rows_mod_four == {
    "A": vector(root_charge_ring, (2, 2)),
    "B_0": vector(root_charge_ring, (3, 3)),
    "B_1": vector(root_charge_ring, (1, 1)),
    "C": vector(root_charge_ring, (2, 2)),
}
assert sum(root_phase_rows_mod_four.values()) == vector(root_charge_ring, (0, 0))

# The common shift kappa=(1,1) in cover coordinates has diagonal charge two.
# It defines the chartwise monomial twist of the log-symplectic form.  The
# root-slab transition scales z^kappa by g_V^{<N_V,kappa>}; these four integer
# exponents are the transition weights of the missing primitive-form line.
kappa_cover_coordinates = vector(ZZ, (1, 1))
assert diagonal_root_charge(kappa_cover_coordinates) == 2
primitive_form_root_transition_weights = {
    label: conormal.dot_product(kappa_cover_coordinates)
    for label, conormal in lifted_conormal_cover_coordinates.items()
}
assert primitive_form_root_transition_weights == {
    "A": -2,
    "B_0": -2,
    "B_1": 2,
    "C": -2,
}

# Every old CPS slab tangent becomes twice primitive on the cover.  Pairing a
# lifted primitive conormal with any pulled-back old slab exponent is divisible
# by eight.  Hence cross-pullback by one root-slab shear acts on every other
# old slab monomial through an integral power of the shearing slab function;
# it creates no additional finite root phase.
lifted_direction_cover_coordinates = {
    label: vector(ZZ, factorization[0].column(0))
    for label, factorization in lifted_rank_one_factorizations.items()
}
old_slab_exponents_on_cover = {
    label: 2 * direction
    for label, direction in lifted_direction_cover_coordinates.items()
}
for exponent in old_slab_exponents_on_cover.values():
    assert exponent in old_lattice_in_cover_coordinates
cross_slab_pairings = matrix(
    ZZ,
    [
        [
            lifted_conormal_cover_coordinates[row_label].dot_product(
                old_slab_exponents_on_cover[column_label]
            )
            for column_label in ("A", "B_0", "B_1", "C")
        ]
        for row_label in ("A", "B_0", "B_1", "C")
    ],
)
assert cross_slab_pairings == matrix(
    ZZ,
    [
        [0, 8, -8, -16],
        [-8, 0, -16, -8],
        [-8, -16, 0, 24],
        [-16, -8, -24, 0],
    ],
)
assert all(entry % 8 == 0 for entry in cross_slab_pairings.list())

# Chartwise the transported GKT algebra is the Hamiltonian algebra of the
# kappa-twisted Poisson structure pi_kappa=z^(-kappa) pi_log.  For q=k+kappa,
# X_q^kappa has monomial exponent k and normal Jq=(q_2,-q_1).  The bracket
# closes with Hamiltonian exponent q+r-kappa and coefficient -det(q,r).
def cover_hamiltonian_normal(hamiltonian_exponent):
    return vector(
        ZZ,
        (hamiltonian_exponent[1], -hamiltonian_exponent[0]),
    )


for k_left in (
    vector(ZZ, (0, 0)),
    vector(ZZ, (1, 0)),
    vector(ZZ, (0, 1)),
    vector(ZZ, (1, 1)),
):
    q_left = k_left + kappa_cover_coordinates
    normal_left = cover_hamiltonian_normal(q_left)
    for k_right in (
        vector(ZZ, (0, 0)),
        vector(ZZ, (1, 0)),
        vector(ZZ, (0, 1)),
        vector(ZZ, (1, 1)),
    ):
        q_right = k_right + kappa_cover_coordinates
        normal_right = cover_hamiltonian_normal(q_right)
        twisted_bracket_normal = (
            normal_left.dot_product(k_right) * normal_right
            - normal_right.dot_product(k_left) * normal_left
        )
        determinant_coefficient = -matrix(
            ZZ, [q_left, q_right]
        ).det()
        target_normal = cover_hamiltonian_normal(
            q_left + q_right - kappa_cover_coordinates
        )
        assert twisted_bracket_normal == determinant_coefficient * target_normal

# A root-slab shear is conformally symplectic rather than symplectic for
# Omega_kappa.  Its formal logarithm is
#
#   S_V=(1/d_V)log(1+z^p) partial_N,
#
# where p=2u_V.  The r-th monomial in [S_V,X_q^kappa] has vector part
#
#   (-1)^(r+1)/d_V * ((<N,q-kappa>/r)Jq - <Jq,p>N).
#
# Its Omega_kappa-divergence is
#   (-1)^r/d_V * <Jq,p><N,kappa>.
# All eight first slab/singleton interactions have nonzero divergence.  Thus
# the chartwise kappa-Hamiltonian subalgebra is not normal under the slab
# group, forcing a Jacobi/Kirillov or homogeneous-Poisson enlargement for any
# global bracket.
singleton_hamiltonian_exponents = {
    "q_1": wall_grade_first_vector,
    "q_2": wall_grade_second_vector,
}
mixed_first_term_data = {}
for slab_label in ("A", "B_0", "B_1", "C"):
    slab_degree = slab_root_degrees[slab_label]
    slab_exponent = old_slab_exponents_on_cover[slab_label]
    slab_normal = lifted_conormal_cover_coordinates[slab_label]
    assert slab_normal.dot_product(slab_exponent) == 0
    for singleton_label, hamiltonian_exponent in singleton_hamiltonian_exponents.items():
        monomial_exponent = hamiltonian_exponent - kappa_cover_coordinates
        hamiltonian_normal = cover_hamiltonian_normal(hamiltonian_exponent)
        first_vector = vector(
            QQ,
            (
                slab_normal.dot_product(monomial_exponent) * hamiltonian_normal
                - hamiltonian_normal.dot_product(slab_exponent) * slab_normal
            )
            / slab_degree,
        )
        first_divergence = first_vector.dot_product(
            hamiltonian_exponent + slab_exponent
        )
        mixed_first_term_data[(slab_label, singleton_label)] = (
            first_vector,
            first_divergence,
        )
        assert first_divergence != 0
        for series_index in range(1, 5):
            series_coefficient = QQ((-1) ** (series_index + 1)) / slab_degree
            series_vector = series_coefficient * (
                QQ(slab_normal.dot_product(monomial_exponent))
                / series_index
                * hamiltonian_normal
                - hamiltonian_normal.dot_product(slab_exponent) * slab_normal
            )
            series_divergence = series_vector.dot_product(
                hamiltonian_exponent + series_index * slab_exponent
            )
            expected_divergence = (
                QQ((-1) ** series_index)
                / slab_degree
                * hamiltonian_normal.dot_product(slab_exponent)
                * slab_normal.dot_product(kappa_cover_coordinates)
            )
            assert series_divergence == expected_divergence

assert mixed_first_term_data == {
    ("A", "q_1"): (vector(QQ, (-QQ(1) / 2, 2)), QQ(2)),
    ("A", "q_2"): (vector(QQ, (-8, QQ(33) / 2)), QQ(10)),
    ("B_0", "q_1"): (vector(QQ, (QQ(5) / 4, 2)), QQ(3)),
    ("B_0", "q_2"): (vector(QQ, (1, QQ(7) / 4)), QQ(3)),
    ("B_1", "q_1"): (vector(QQ, (-QQ(9) / 4, 4)), QQ(1)),
    ("B_1", "q_2"): (vector(QQ, (13, -QQ(75) / 4)), QQ(-7)),
    ("C", "q_1"): (vector(QQ, (-QQ(33) / 2, 8)), QQ(-10)),
    ("C", "q_2"): (vector(QQ, (-2, QQ(1) / 2)), QQ(-2)),
}

# Every lifted monodromy about a singular point induces the same involution on
# the sector group: it exchanges the two singleton grades.  Their bracket
# grade is therefore invariant.  By contrast, the annulus-core generator
# C*A^(-1) acts trivially on the entire finite grading.
for monodromy in lifted_singularity_monodromies.values():
    assert sector_group(monodromy * cover_coordinate_lattice.gen(0)) == sector_shift_second
    assert sector_group(monodromy * cover_coordinate_lattice.gen(1)) == sector_shift_first
    assert sector_group(monodromy * (cover_coordinate_lattice.gen(0) + cover_coordinate_lattice.gen(1))) == sector_bracket
    assert sector_group(monodromy * wall_grade_first_vector) == sector_wall_second
    assert sector_group(monodromy * wall_grade_second_vector) == sector_wall_first

annulus_core_monodromy = parity_kernel_on_cover["C*A^-1"]
assert annulus_core_monodromy == matrix(ZZ, [[-1, 2], [-10, 19]])
assert annulus_core_monodromy.det() == 1
assert annulus_core_monodromy.trace() == 18
assert annulus_core_monodromy.charpoly() == polygen(ZZ) ** 2 - 18 * polygen(ZZ) + 1

# The hyperbolic core has no nonzero fixed lattice exponent.  More generally,
# its exact eigenvalues 9+/-4*sqrt(5) are not roots of unity, so no nonzero
# lattice exponent has finite orbit.  The finite power loop below is a guard
# on the implemented matrix convention; the argument for all periods is the
# hyperbolic core holonomy theorem of Section 3.
core_fixed_exponent_matrix = annulus_core_monodromy - identity_matrix(ZZ, 2)
assert core_fixed_exponent_matrix.det() == -16
assert core_fixed_exponent_matrix.rank() == 2
for core_period in range(1, 13):
    assert (annulus_core_monodromy**core_period - identity_matrix(ZZ, 2)).det() != 0

# The affine translation part of the core holonomy is a separate discrete
# datum.  Since I-K has Smith invariants (2,8), real affine conjugacy kills
# every translation, but integral translation conjugacy leaves a class in
# coker(I-K)=Z/2+Z/8.  The linear data alone therefore do not determine the
# developed annulus.  This block and the next three are auxiliary: the paper
# does not use the translation part on the character side.  (On the side of
# Delta, every transition is linear in coordinates centered at O; see
# Appendix A.2.)
core_radiance_matrix = matrix(
    ZZ,
    identity_matrix(QQ, 2) - annulus_core_monodromy,
)
assert core_radiance_matrix == matrix(ZZ, [[2, -2], [10, -18]])
assert core_radiance_matrix.det() == -16
core_radiance_smith, _, _ = core_radiance_matrix.smith_form()
assert core_radiance_smith[0, 1] == 0
assert core_radiance_smith[1, 0] == 0
assert [abs(core_radiance_smith[index, index]) for index in range(2)] == [2, 8]
core_translation_group = cover_coordinate_lattice / cover_coordinate_lattice.submodule(
    core_radiance_matrix.columns()
)
assert core_translation_group.invariants() == (2, 8)
assert core_radiance_matrix.inverse() == matrix(
    QQ,
    [[QQ(9) / 8, -QQ(1) / 8], [QQ(5) / 8, -QQ(1) / 8]],
)

# Auxiliary placement calculation, not used in the paper.  It applies the
# character monodromies T_A, T_C to radial points of the polygon, which mixes
# the two sides of the Legendre pair.  With the A and C singular points at the
# same radial fraction lambda of the polygon vertices, the affine core word
# C*A^(-1) then has cover translation 16*lambda*(1,1); for lambda=1/2 this
# translation is integrally conjugate to zero, with fixed point (8,4).
common_radial_fraction = QQ(1) / 2
affine_translation_A_inverse = (
    identity_matrix(QQ, 2) - T_A.inverse()
) * (common_radial_fraction * p_A)
affine_translation_C = (
    identity_matrix(QQ, 2) - T_C
) * (common_radial_fraction * p_C)
candidate_core_translation_old = (
    affine_translation_C + T_C * affine_translation_A_inverse
)
candidate_core_translation_cover = vector(
    ZZ,
    F_inverse * candidate_core_translation_old,
)
assert candidate_core_translation_old == vector(QQ, (0, -4))
assert candidate_core_translation_cover == vector(ZZ, (8, 8))
candidate_radiant_fixed_point = vector(
    ZZ,
    core_radiance_matrix.inverse() * candidate_core_translation_cover,
)
assert candidate_radiant_fixed_point == vector(ZZ, (8, 4))
assert (
    core_radiance_matrix * candidate_radiant_fixed_point
    == candidate_core_translation_cover
)
assert core_translation_group(candidate_core_translation_cover) == 0

# The zero radiance class does not put the lifted CPS cell in one hyperbolic
# sector.  The symmetrized Cartan form is K-invariant and its null cone is the
# union of the two irrational K-eigenlines.  Relative to the barycentric fixed
# point, the three lifted vertices have positive norm, while an interior point
# of the lifted A--C edge has negative norm.  Hence that edge crosses both
# eigenlines and the developed K-action is not proper on the whole cell.
hyperbolic_invariant_form = matrix(ZZ, [[10, -10], [-10, 2]])
assert (
    annulus_core_monodromy.transpose()
    * hyperbolic_invariant_form
    * annulus_core_monodromy
    == hyperbolic_invariant_form
)
lifted_polygon_vertices = [
    vector(ZZ, F_inverse * vertex)
    for vertex in (p_A, p_B, p_C)
]
assert lifted_polygon_vertices == [
    vector(ZZ, (-2, 2)),
    vector(ZZ, (-6, -2)),
    vector(ZZ, (14, 2)),
]


def hyperbolic_norm(point):
    displacement = point - candidate_radiant_fixed_point
    return displacement * hyperbolic_invariant_form * displacement


assert [hyperbolic_norm(vertex) for vertex in lifted_polygon_vertices] == [
    608,
    352,
    608,
]
lifted_ac_edge_test_point = vector(ZZ, (7, 2))
assert hyperbolic_norm(lifted_ac_edge_test_point) == -22

# Allow the A and C singular points to move independently along their radial
# segments.  For fractions alpha,gamma in [0,1], the core fixed point is
# (alpha+15*gamma, 3*alpha+5*gamma).  The shallow K-eigenline has slope
# 5-2*sqrt(5).  Its offset across the full placement square stays strictly
# inside the interval of offsets attained on the lifted triangle, so this
# eigenline meets the triangle for every admissible radial placement.
PlacementRing.<alpha_radial, gamma_radial> = PolynomialRing(QQ, 2)
placement_identity = identity_matrix(PlacementRing, 2)
placement_translation_A_inverse = (
    placement_identity - T_A.inverse()
) * (alpha_radial * vector(PlacementRing, p_A))
placement_translation_C = (
    placement_identity - T_C
) * (gamma_radial * vector(PlacementRing, p_C))
placement_core_translation_cover = matrix(
    PlacementRing, F_inverse
) * (
    placement_translation_C
    + matrix(PlacementRing, T_C) * placement_translation_A_inverse
)
placement_radiant_fixed_point = matrix(
    PlacementRing, core_radiance_matrix.inverse()
) * placement_core_translation_cover
assert placement_core_translation_cover == vector(
    PlacementRing,
    (-4 * alpha_radial + 20 * gamma_radial,
     -44 * alpha_radial + 60 * gamma_radial),
)
assert placement_radiant_fixed_point == vector(
    PlacementRing,
    (alpha_radial + 15 * gamma_radial,
     3 * alpha_radial + 5 * gamma_radial),
)

Eigenfield.<sqrt5> = QuadraticField(5)
shallow_eigenline_slope = 5 - 2 * sqrt5
alpha_offset_coefficient = shallow_eigenline_slope - 3
gamma_offset_coefficient = 15 * shallow_eigenline_slope - 5
assert alpha_offset_coefficient < 0
assert gamma_offset_coefficient > 0
placement_offset_interval = (
    alpha_offset_coefficient,
    gamma_offset_coefficient,
)
triangle_line_values_at_zero = [
    Eigenfield(vertex[1]) - shallow_eigenline_slope * Eigenfield(vertex[0])
    for vertex in lifted_polygon_vertices
]
assert triangle_line_values_at_zero == [
    12 - 4 * sqrt5,
    28 - 12 * sqrt5,
    -68 + 28 * sqrt5,
]
triangle_crossing_offset_interval = (
    -(12 - 4 * sqrt5),
    -(-68 + 28 * sqrt5),
)
assert triangle_crossing_offset_interval[0] < placement_offset_interval[0]
assert placement_offset_interval[1] < triangle_crossing_offset_interval[1]

# The core holonomy is a Coxeter element for the symmetrizable indefinite
# rank-two Cartan datum with off-diagonal entries 2 and 10.  This is an exact
# matrix factorization; no identification with a cluster scattering diagram
# is asserted.
coxeter_reflection_first = matrix(ZZ, [[-1, 2], [0, 1]])
coxeter_reflection_second = matrix(ZZ, [[1, 0], [10, -1]])
for reflection in (coxeter_reflection_first, coxeter_reflection_second):
    assert reflection**2 == identity_matrix(ZZ, 2)
    assert reflection.det() == -1
assert annulus_core_monodromy == coxeter_reflection_second * coxeter_reflection_first
cartan_matrix = matrix(ZZ, [[2, -2], [-10, 2]])
cartan_symmetrizer = diagonal_matrix(ZZ, [5, 1])
assert cartan_symmetrizer * cartan_matrix == matrix(ZZ, [[10, -10], [-10, 2]])
assert (cartan_symmetrizer * cartan_matrix).is_symmetric()
assert cartan_matrix.det() == -16
for generator in cover_coordinate_lattice.gens():
    assert sector_group(annulus_core_monodromy * generator) == sector_group(generator)

# The two shifted singleton wall exponents generate an infinite Pell-type
# transport tower under the hyperbolic core holonomy.  Simultaneous transport
# preserves their finite grades, primitivity, and determinant three.  By
# Cayley--Hamilton every orbit satisfies v_(n+2)=18v_(n+1)-v_n.
assert matrix(ZZ, [wall_grade_first_vector, wall_grade_second_vector]).det() == 3
singleton_wall_sublattice = cover_coordinate_lattice.submodule(
    [wall_grade_first_vector, wall_grade_second_vector]
)
assert cover_coordinate_lattice.index_in(singleton_wall_sublattice) == QQ(1) / 3
assert singleton_wall_sublattice.index_in(cover_coordinate_lattice) == 3
assert annulus_core_monodromy * wall_grade_first_vector not in singleton_wall_sublattice
assert annulus_core_monodromy * wall_grade_second_vector not in singleton_wall_sublattice
for exponent in (wall_grade_first_vector, wall_grade_second_vector):
    assert annulus_core_monodromy**2 * exponent == (
        18 * annulus_core_monodromy * exponent - exponent
    )
for power in range(-5, 6):
    transported_first = annulus_core_monodromy**power * wall_grade_first_vector
    transported_second = annulus_core_monodromy**power * wall_grade_second_vector
    transported_kappa = annulus_core_monodromy**power * kappa_cover_coordinates
    assert gcd([abs(entry) for entry in transported_first]) == 1
    assert gcd([abs(entry) for entry in transported_second]) == 1
    assert matrix(ZZ, [transported_first, transported_second]).det() == 3
    assert sector_group(transported_first) == sector_wall_first
    assert sector_group(transported_second) == sector_wall_second
    assert (
        transported_first + transported_second - transported_kappa
        == annulus_core_monodromy**power
        * (
            wall_grade_first_vector
            + wall_grade_second_vector
            - kappa_cover_coordinates
        )
    )

singleton_core_orbit_sample = {
    power: (
        annulus_core_monodromy**power * wall_grade_first_vector,
        annulus_core_monodromy**power * wall_grade_second_vector,
    )
    for power in range(-1, 3)
}
assert singleton_core_orbit_sample == {
    -1: (vector(ZZ, (36, 19)), vector(ZZ, (15, 8))),
    0: (vector(ZZ, (2, 1)), vector(ZZ, (1, 2))),
    1: (vector(ZZ, (0, -1)), vector(ZZ, (3, 28))),
    2: (vector(ZZ, (-2, -19)), vector(ZZ, (53, 502))),
}

# The two positive singleton tails approach opposite rays of the unstable
# eigenline.  Hence no pointed real cone contains both tails, ruling out one
# sharp toric-monoid completion for the full same-order Pell pair.
sqrt_five = AA(5).sqrt()
unstable_slope = 5 + 2 * sqrt_five
stable_slope = 5 - 2 * sqrt_five
unstable_eigenvalue = 9 + 4 * sqrt_five
unstable_eigenvector = vector(AA, (1, unstable_slope))
assert matrix(AA, annulus_core_monodromy) * unstable_eigenvector == (
    unstable_eigenvalue * unstable_eigenvector
)
unstable_coefficient_first = (
    AA(wall_grade_first_vector[1])
    - stable_slope * AA(wall_grade_first_vector[0])
) / (unstable_slope - stable_slope)
unstable_coefficient_second = (
    AA(wall_grade_second_vector[1])
    - stable_slope * AA(wall_grade_second_vector[0])
) / (unstable_slope - stable_slope)
assert unstable_coefficient_first < 0
assert unstable_coefficient_second > 0

# Although no single pointed cone contains both full positive Pell tails, the
# universal cover has a K-equivariant family C_n=K^n C_0.  Here C_0 is cut
# out by ell_1(x,y)=2x-y and ell_2(x,y)=2y-x.  The kappa-shifted bracket sends
# q,r to q+r-kappa.  Each cone coordinate therefore adds and loses exactly
# one.  If a coordinate could become negative, both inputs lie on the same
# boundary ray and their determinant (hence bracket) is zero.
singleton_cone_forms = matrix(ZZ, [[2, -1], [-1, 2]])
assert singleton_cone_forms * wall_grade_first_vector == vector(ZZ, (3, 0))
assert singleton_cone_forms * wall_grade_second_vector == vector(ZZ, (0, 3))
assert singleton_cone_forms * kappa_cover_coordinates == vector(ZZ, (1, 1))
for first_x in range(-10, 11):
    for first_y in range(-10, 11):
        first_exponent = vector(ZZ, (first_x, first_y))
        if any(value < 0 for value in singleton_cone_forms * first_exponent):
            continue
        for second_x in range(-10, 11):
            for second_y in range(-10, 11):
                second_exponent = vector(ZZ, (second_x, second_y))
                if any(
                    value < 0
                    for value in singleton_cone_forms * second_exponent
                ):
                    continue
                if matrix(ZZ, [first_exponent, second_exponent]).det() == 0:
                    continue
                bracket_exponent = (
                    first_exponent
                    + second_exponent
                    - kappa_cover_coordinates
                )
                assert all(
                    value >= 0
                    for value in singleton_cone_forms * bracket_exponent
                )

# Distinct Pell translates interact with hyperbolically growing determinant
# coefficients.  Every sequence det(q_i,K^d q_j) obeys the trace-18
# recurrence.  These are the second-order numerical factors which would
# appear if different winding translates meet in one local joint.
def pell_interaction_determinant(left_exponent, right_exponent, power):
    return matrix(
        ZZ,
        [left_exponent, annulus_core_monodromy**power * right_exponent],
    ).det()


pell_interaction_sequences = {
    "11": [
        pell_interaction_determinant(
            wall_grade_first_vector, wall_grade_first_vector, power
        )
        for power in range(8)
    ],
    "22": [
        pell_interaction_determinant(
            wall_grade_second_vector, wall_grade_second_vector, power
        )
        for power in range(8)
    ],
    "12": [
        pell_interaction_determinant(
            wall_grade_first_vector, wall_grade_second_vector, power
        )
        for power in range(8)
    ],
    "21": [
        pell_interaction_determinant(
            wall_grade_second_vector, wall_grade_first_vector, power
        )
        for power in range(8)
    ],
}
assert pell_interaction_sequences["11"][:6] == [0, -2, -36, -646, -11592, -208010]
assert pell_interaction_sequences["22"][:6] == [0, 22, 396, 7106, 127512, 2288110]
assert pell_interaction_sequences["12"][:6] == [3, 53, 951, 17065, 306219, 5494877]
assert pell_interaction_sequences["21"][:6] == [-3, -1, -15, -269, -4827, -86617]
for sequence in pell_interaction_sequences.values():
    for sequence_index in range(len(sequence) - 2):
        assert sequence[sequence_index + 2] == (
            18 * sequence[sequence_index + 1] - sequence[sequence_index]
        )
assert all(value != 0 for value in pell_interaction_sequences["12"])
assert all(value != 0 for value in pell_interaction_sequences["21"])
assert all(
    pell_interaction_sequences["22"][index]
    == -11 * pell_interaction_sequences["11"][index]
    for index in range(8)
)

# The chosen downstairs outer-boundary word and its conjugate give the two
# boundary lifts of the annulus.  Their affine holonomies are integral and
# unipotent on M_tilde, and both act trivially on the sector group.
outer_boundary_monodromy = F_inverse * (T_A * T_B * T_C) * F
other_boundary_monodromy = (
    F_inverse * T_A * (T_A * T_B * T_C) * T_A.inverse() * F
)
assert outer_boundary_monodromy == matrix(ZZ, [[5, -4], [4, -3]])
assert other_boundary_monodromy == matrix(ZZ, [[33, -64], [16, -31]])
for monodromy in (outer_boundary_monodromy, other_boundary_monodromy):
    assert monodromy.det() == 1 and monodromy.trace() == 2
    for generator in cover_coordinate_lattice.gens():
        assert sector_group(monodromy * generator) == sector_group(generator)

# The missing diagonal GKT residue is itself a lattice quotient.  In old CPS
# coordinates put M_GKT=iota(Z^2).  Since F=iota/4, its coordinates in the
# F-basis are exactly 4Z^2.  Thus
#
#   M_GKT subset M subset M_tilde,
#   M_tilde/M_GKT=(Z/4)^2,
#   M/M_GKT=<h>, h=(2,2),
#
# and quotienting by h recovers D=M_tilde/M.  This realizes the nonsplit
# diagonal lift geometrically rather than adjoining an abstract sign.
gkt_lattice_in_cover_coordinates = cover_coordinate_lattice.submodule(
    [4 * cover_coordinate_lattice.gen(0), 4 * cover_coordinate_lattice.gen(1)]
)
assert F_inverse * iota == 4 * identity_matrix(ZZ, 2)
assert all(
    generator in old_lattice_in_cover_coordinates
    for generator in gkt_lattice_in_cover_coordinates.gens()
)
assert abs(matrix(ZZ, old_lattice_in_cover_coordinates.basis()).det()) == 8
assert abs(matrix(ZZ, gkt_lattice_in_cover_coordinates.basis()).det()) == 16
assert gkt_lattice_in_cover_coordinates == gkt_twist_relations

diagonal_kernel_vector = vector(ZZ, (2, 2))
diagonal_kernel_class = gkt_twist_group(diagonal_kernel_vector)
assert diagonal_kernel_class.additive_order() == 2
assert old_lattice_in_cover_coordinates == cover_coordinate_lattice.submodule(
    list(gkt_lattice_in_cover_coordinates.gens()) + [diagonal_kernel_vector]
)

# On the full residue quotient Q_GKT, every peripheral monodromy is the same
# order-four matrix P.  Its square is the annular core action.  The two chosen
# boundary holonomies are trivial.  Modulo the fixed diagonal class h, P is
# exactly the singleton swap on D.  Therefore the full residue representation
# has cyclic image Z/4 and its mod-two quotient is the sector character.
residue_ring = Integers(4)
residue_identity = identity_matrix(residue_ring, 2)
residue_peripheral_generator = matrix(residue_ring, [[2, 1], [3, 0]])
residue_peripheral_matrices = {
    label: matrix(residue_ring, monodromy)
    for label, monodromy in lifted_singularity_monodromies.items()
}
assert all(
    monodromy == residue_peripheral_generator
    for monodromy in residue_peripheral_matrices.values()
)
assert residue_peripheral_generator**2 == matrix(
    residue_ring, [[3, 2], [2, 3]]
)
assert residue_peripheral_generator**2 != residue_identity
assert residue_peripheral_generator**4 == residue_identity
assert matrix(residue_ring, annulus_core_monodromy) == (
    residue_peripheral_generator**2
)
assert matrix(residue_ring, outer_boundary_monodromy) == residue_identity
assert matrix(residue_ring, other_boundary_monodromy) == residue_identity
assert residue_peripheral_generator * diagonal_kernel_vector == vector(
    residue_ring, diagonal_kernel_vector
)
residue_kappa = vector(residue_ring, (1, 1))
assert residue_peripheral_generator * residue_kappa == (
    residue_kappa + vector(residue_ring, diagonal_kernel_vector)
)
assert residue_peripheral_generator**2 * residue_kappa == residue_kappa
for representative in sector_representatives:
    assert sector_group(
        vector(ZZ, residue_peripheral_generator * representative)
    ) == sector_group(singleton_swap * representative)

# Linear monodromy is covariant for the family of shifted Hamiltonian
# algebras: L identifies the kappa algebra with the L*kappa algebra.  It is
# not an automorphism of the fixed-kappa algebra when L*kappa != kappa.  The
# exact determinant and shifted-output identities are checked on a generating
# sample for all four peripheral matrices and the core.
hamiltonian_covariance_sample = [
    vector(ZZ, (1, 1)),
    wall_grade_first_vector,
    wall_grade_second_vector,
    vector(ZZ, (3, 2)),
]
for monodromy in list(lifted_singularity_monodromies.values()) + [
    annulus_core_monodromy
]:
    transported_kappa = monodromy * kappa_cover_coordinates
    for left_exponent in hamiltonian_covariance_sample:
        for right_exponent in hamiltonian_covariance_sample:
            assert matrix(
                ZZ,
                [monodromy * left_exponent, monodromy * right_exponent],
            ).det() == matrix(ZZ, [left_exponent, right_exponent]).det()
            assert (
                monodromy
                * (
                    left_exponent
                    + right_exponent
                    - kappa_cover_coordinates
                )
                == monodromy * left_exponent
                + monodromy * right_exponent
                - transported_kappa
            )

# The degree-four monodromy cover is the minimal cover trivializing Q_GKT.
# It factors through the degree-two sector cover.  Its compact extension is
# fully ramified over the four foci and has four lifts of each boundary:
# chi=-12, b=8, hence g=3.  The remaining closed core loop is core^2; its
# lattice holonomy K^2 is still hyperbolic although it is trivial modulo four.
gkt_residue_cover_degree = 4
gkt_residue_cover_euler_characteristic = 4 * 0 - 4 * (4 - 1)
gkt_residue_cover_boundary_components = 2 * 4
gkt_residue_cover_genus = (
    2
    - gkt_residue_cover_boundary_components
    - gkt_residue_cover_euler_characteristic
) // 2
assert gkt_residue_cover_euler_characteristic == -12
assert gkt_residue_cover_boundary_components == 8
assert gkt_residue_cover_genus == 3
assert matrix(residue_ring, annulus_core_monodromy**2) == residue_identity
assert (annulus_core_monodromy**2).trace() == 322

# The GKT d=1 singleton orbit forced by P contains four allowed ordered twist
# types.  GKT Notation 2.30 gives m-rs=4(a+b), while GKT Definition 4.23 has the unique
# critical exponent (k1,k2)=(a,b) for each type.  Two orbit points contain a
# Ramond coordinate 3; they are retained by the full labelled ring A_I but
# omitted from the narrow symmetric potential ring A_{I,sym}.
gkt_singleton_twist_orbit = []
gkt_singleton_twist = vector(residue_ring, (1, 0))
for orbit_index in range(4):
    gkt_singleton_twist_orbit.append(tuple(ZZ(entry) for entry in gkt_singleton_twist))
    gkt_singleton_twist = residue_peripheral_generator * gkt_singleton_twist
assert gkt_singleton_twist_orbit == [(1, 0), (2, 3), (3, 2), (0, 1)]
assert gkt_singleton_twist == vector(residue_ring, (1, 0))
for twist_first, twist_second in gkt_singleton_twist_orbit:
    gkt_m_minus_rs = 4 * (twist_first + twist_second)
    gkt_critical_exponent = vector(ZZ, (twist_first, twist_second))
    assert 4 * sum(gkt_critical_exponent) == gkt_m_minus_rs
    assert gkt_critical_exponent[0] % 4 == twist_first
    assert gkt_critical_exponent[1] % 4 == twist_second
assert {
    twist
    for twist in gkt_singleton_twist_orbit
    if 3 in twist
} == {(2, 3), (3, 2)}

# Gross-Siebert wall functions use the torus Hamiltonian exponent
# q=k+(1,1), not the affine GKT shift k.  On the cover F*q is integral and
# orthogonal to the transported normal.  The two derivations differ by the
# single primitive-form monomial z^(-F(1,1)).  For k=0 this cancels the
# marginal wall exponent and leaves precisely the open-gluing gauge.
primitive_form_shift = F * vector(ZZ, (1, 1))


def transported_wall_exponent(k):
    return F * (k + vector(ZZ, (1, 1)))


for k in (
    vector(ZZ, (0, 0)),
    vector(ZZ, (1, 0)),
    vector(ZZ, (0, 1)),
    vector(ZZ, (1, 1)),
):
    wall_exponent = transported_wall_exponent(k)
    assert wall_exponent - primitive_form_shift == F * k
    assert transported_normal(k).dot_product(wall_exponent) == 0

assert primitive_form_shift == vector(QQ, (0, -QQ(1) / 2))
assert M2 * primitive_form_shift - primitive_form_shift == edge_XY_primitive
assert M2 * edge_XY_primitive == edge_XY_primitive
primitive_slab_conormal = K_target * edge_XY_primitive
assert primitive_slab_conormal == vector(ZZ, (-1, 1))
assert primitive_slab_conormal.dot_product(edge_XY_primitive) == 0
# This is primitive in the old Z^2 dual, but it is not integral in the dual of
# M_tilde.  The primitive cover-lattice conormal is four times larger.
assert not all(
    entry in ZZ for entry in F.transpose() * primitive_slab_conormal
)
primitive_cover_slab_conormal = 4 * primitive_slab_conormal
primitive_cover_slab_coordinates = F.transpose() * primitive_cover_slab_conormal
assert primitive_cover_slab_coordinates == vector(ZZ, (-1, -1))
assert gcd([abs(entry) for entry in primitive_cover_slab_coordinates]) == 1
assert primitive_cover_slab_conormal.dot_product(edge_XY_primitive) == 0
assert 4 * transported_wall_exponent(vector(ZZ, (1, 0))) == iota * first_desc_boundary_charge
assert 4 * transported_wall_exponent(vector(ZZ, (0, 1))) == iota * second_desc_boundary_charge

# The fixed primitive-form shift has half-edge defects at the two order-one
# facets and a full-edge defect at the order-two facet.  Squaring precisely
# the A and C monodromies converts all three defects to the chosen primitive
# old-lattice facet tangents.  This is why the parity double cover is the
# minimal cover on which the lattice M_tilde is preserved.
primitive_form_facet_defects = {
    label: monodromy * primitive_form_shift - primitive_form_shift
    for label, monodromy in facet_monodromy.items()
}
assert primitive_form_facet_defects == {
    "A": facet_tangents["A"] / 2,
    "B": facet_tangents["B"],
    "C": facet_tangents["C"] / 2,
}
primitive_form_double_cover_defects = {
    "A": facet_monodromy["A"] ** 2 * primitive_form_shift - primitive_form_shift,
    "B": facet_monodromy["B"] * primitive_form_shift - primitive_form_shift,
    "C": facet_monodromy["C"] ** 2 * primitive_form_shift - primitive_form_shift,
}
assert primitive_form_double_cover_defects == facet_tangents

# The first bracket is preserved.  In the standard monomial-derivation
# formula, [z^d d_n,z^e d_p] has covector <n,e>p-<p,d>n.  It equals -3 times
# the transported X_(1,1), matching the GKT determinant coefficient.
k_first = vector(ZZ, (1, 0))
k_second = vector(ZZ, (0, 1))
d_first = F * k_first
d_second = F * k_second
bracket_normal = (
    first_desc_normal.dot_product(d_second) * second_desc_normal
    - second_desc_normal.dot_product(d_first) * first_desc_normal
)
assert bracket_normal == -3 * transported_normal(k_first + k_second)

# The singleton-tower recursion of [Wue26] gives coefficient -1/2 for both
# d=1 singleton walls.  Check the two endpoint differences against the GKT
# vector-field actions:
#   X_(1,0)(W0)=4*X^5-8*X*Y^4,
#   X_(0,1)(W0)=8*X^4*Y-4*Y^5.
first_singleton_coefficient = -QQ(1) / 2
assert first_singleton_coefficient * vector(ZZ, (4, -8)) == vector(QQ, (-2, 4))
assert first_singleton_coefficient * vector(ZZ, (8, -4)) == vector(QQ, (-4, 2))
first_singleton_bracket_coefficient = (
    first_singleton_coefficient**2 * (-3)
)
assert first_singleton_bracket_coefficient == -QQ(3) / 4

# After shifting a Hamiltonian label q by kappa=(1,1), the singleton algebra
# is N^2-graded.  If E_(a,b)=X^kappa_(kappa+(a,b)), every bracket has bidegree
# (a+c,b+d).  This exact additive grading is the pronilpotent filtration used
# for the all-order local singleton scattering construction.
singleton_kappa = vector(ZZ, (1, 1))


def shifted_singleton_bracket(a, b, c, d):
    first_q = singleton_kappa + vector(ZZ, (a, b))
    second_q = singleton_kappa + vector(ZZ, (c, d))
    coefficient = -matrix(ZZ, [first_q, second_q]).det()
    output_q = first_q + second_q - singleton_kappa
    return coefficient, output_q


for a in range(4):
    for b in range(4):
        for c in range(4):
            for d in range(4):
                coefficient, output_q = shifted_singleton_bracket(a, b, c, d)
                assert output_q == singleton_kappa + vector(ZZ, (a + c, b + d))
                first_q = singleton_kappa + vector(ZZ, (a, b))
                second_q = singleton_kappa + vector(ZZ, (c, d))
                assert coefficient == -matrix(ZZ, [first_q, second_q]).det()

print("fan rays:", {"v_X": v_X, "v_Y": v_Y, "v_Z": v_Z})
print("cone indices:", cone_indices)
print("Delta vertices:", Delta_vertices)
print("Delta lattice-point/Cox dictionary:", section_dictionary)
print("Delta dual vertices:", Delta_dual_vertices)
print("length-two edge primitive:", edge_XY_primitive)
print("edge midpoint m_*:", m_star)
print("boundary-charge embedding matrix:")
print(iota)
print("index of iota(N_boundary):", abs(iota.det()))
print("iota(1,1):", iota * vector(ZZ, (1, 1)))
print("singular radial vertex p_XY:", p_XY)
print("lattice-level Legendre-dual cofactor matrix:")
print(iota_cofactor)
print("iota^!(1,1):", iota_cofactor * vector(ZZ, (1, 1)))
print("transported marginal Hamiltonian tangent:", marginal_tangent)
print("order-two monodromy in edge-adapted global coordinates:")
print(M2)
print("order-two monodromy in radial-edge-adapted coordinates:")
print(M2_check)
print("straight-boundary chart transition:")
print(chart_transition)
print("all dual-facet monodromies:", facet_monodromy)
print("q-loop elliptic monodromy (up to inverse/conjugacy):")
print(M_q)
print("completed-square quartic:")
print(transformed)
print("leading |GKT marginal coefficient|:", c_gkt)
print("CPS fiber j-invariant:", factor(j_cps))
print("naive CPS/GKT parameter identification:", "fails")
print("variable-slab w=0 j-invariant:", factor(j_slab))
print("successful mirror map:", "s^2=4*a^2/(a^2-4)")
print("B-slab fourth-root torsor at a=+/-2:", "splits into two quadratic components")
print("cusp expansion a(q):", a_series.add_bigoh(8))
print("extreme-gauge parameter tau(t):", tau_extreme.add_bigoh(6))
print(
    "GHS endpoint-character weights:",
    [endpoint_weight(m) for m in (v_X, m_star, v_Y, v_Z)],
)
print("first-descendent fractional shift:", first_desc_shift)
print("first-descendent exponent coset:", first_desc_exponents)
print(
    "first-descendent cofactor support:",
    iota_cofactor * first_desc_boundary_charge,
)
print("symmetric-descendent fractional shift:", second_desc_shift)
print("symmetric-descendent exponent coset:", second_desc_exponents)
print(
    "symmetric-descendent cofactor support:",
    iota_cofactor * second_desc_boundary_charge,
)
print(
    "compact singleton support endpoints (first, second, bracket):",
    first_desc_support_endpoint,
    second_desc_support_endpoint,
    first_bracket_support_endpoint,
)
print(
    "incoming singleton full-chord lower endpoints and branch crossings:",
    (first_desc_negative_endpoint, second_desc_negative_endpoint),
    (first_chord_branch_crossing, second_chord_branch_crossing),
)
print(
    "minus-chart compact support endpoints (first, second, bracket):",
    first_desc_support_endpoint_minus,
    second_desc_support_endpoint_minus,
    first_bracket_support_endpoint_minus,
)
print(
    "barycentric lower branch segment:",
    barycentric_focus_A,
    barycentric_focus_C,
)
print("index of CPS lattice in minimal GKT cover lattice:", abs(F_inverse.det()))
print("edge monodromy on cover lattice:")
print(M2_on_cover)
print("radial monodromy on cover lattice:")
print(M2_check_on_cover)
print(
    "transported normals (marginal, first descendants):",
    [marginal_normal, first_desc_normal, second_desc_normal],
)
print("primitive-form lattice shift:", primitive_form_shift)
print(
    "primitive-form monodromy defect:",
    M2 * primitive_form_shift - primitive_form_shift,
)
print("old-lattice rotation of the order-two defect:", primitive_slab_conormal)
print("primitive conormal in the dual cover lattice:", primitive_cover_slab_conormal)
print("primitive-form defects on all facets:", primitive_form_facet_defects)
print("facet monodromies on the index-eight cover:", facet_monodromy_on_cover)
print(
    "singleton-normal pairings with the forced global quarter vector:",
    [
        first_desc_normal.dot_product(quarter_y),
        second_desc_normal.dot_product(quarter_y),
    ],
)
print(
    "facet monodromies on the parity double cover:",
    facet_monodromy_on_base_double_cover,
)
print("parity-kernel monodromies on the cover lattice:", parity_kernel_on_cover)
print("four lifted singularity monodromies:", lifted_singularity_monodromies)
print("primitive lifted directions:", lifted_primitive_directions)
print("primitive lifted conormals:", lifted_primitive_conormals)
print(
    "old facet conormals in the cover-dual basis:",
    old_facet_conormals_in_cover_coordinates,
)
print("required slab-root degrees on the annular cover:", slab_root_degrees)
print("finite root-sector group M_tilde/M:", sector_group.invariants())
print(
    "GKT twist-pair quotient kernel:",
    sorted(gkt_to_annular_kernel_representatives),
)
print("full GKT residue peripheral generator:")
print(residue_peripheral_generator)
print(
    "full GKT residue cover (degree, genus, boundaries):",
    (
        gkt_residue_cover_degree,
        gkt_residue_cover_genus,
        gkt_residue_cover_boundary_components,
    ),
)
print("GKT singleton twist orbit:", gkt_singleton_twist_orbit)
print("global diagonal root charge s(e_1),s(e_2):", [1, 1], "mod 4")
print("peripheral root-phase rows mod 4:", root_phase_rows_mod_four)
print(
    "primitive-form root-transition weights:",
    primitive_form_root_transition_weights,
)
print("cross-slab conormal pairings:")
print(cross_slab_pairings)
print(
    "singleton-shift, wall-monomial, and bracket sector orders:",
    [
        sector_shift_first.additive_order(),
        sector_shift_second.additive_order(),
        sector_wall_first.additive_order(),
        sector_wall_second.additive_order(),
        sector_bracket.additive_order(),
    ],
)
print("annulus-core monodromy on M_tilde:")
print(annulus_core_monodromy)
print("core affine-translation class group:", core_translation_group.invariants())
print(
    "barycentric core translation/fixed point:",
    candidate_core_translation_cover,
    candidate_radiant_fixed_point,
)
print(
    "barycentric lifted-cell hyperbolic norms (vertices, edge test):",
    [hyperbolic_norm(vertex) for vertex in lifted_polygon_vertices],
    hyperbolic_norm(lifted_ac_edge_test_point),
)
print(
    "all radial placements cross the shallow eigenline;",
    "fixed point:",
    placement_radiant_fixed_point,
    "placement/crossing offset intervals:",
    placement_offset_interval,
    triangle_crossing_offset_interval,
)
print("indefinite Cartan datum for core Coxeter holonomy:")
print(cartan_matrix)
print("singleton core-orbit sample:", singleton_core_orbit_sample)
print(
    "annulus boundary holonomies:",
    [outer_boundary_monodromy, other_boundary_monodromy],
)
print(
    "transported wall exponents (marginal, first descendants):",
    [
        transported_wall_exponent(k)
        for k in (
            vector(ZZ, (0, 0)),
            vector(ZZ, (1, 0)),
            vector(ZZ, (0, 1)),
        )
    ],
)
print("first transported bracket coefficient:", -3)
print("first singleton wall coefficients:", [first_singleton_coefficient] * 2)
print("first singleton Lie-bracket coefficient:", first_singleton_bracket_coefficient)
print("singleton Hamiltonian grading:", "q=kappa+(a,b), bracket degree=(a+c,b+d)")
print("first mixed slab-singleton vector/divergence data:", mixed_first_term_data)
print(
    "singleton sectorial cone forms on q_1,q_2,kappa:",
    [
        singleton_cone_forms * exponent
        for exponent in (
            wall_grade_first_vector,
            wall_grade_second_vector,
            kappa_cover_coordinates,
        )
    ],
)
print("distinct Pell-translate determinant sequences:", pell_interaction_sequences)
print("P(1,1,2) degeneration, parity cover and annular lattice checks: PASS")
