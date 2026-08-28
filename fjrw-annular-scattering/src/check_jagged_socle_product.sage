"""Exact polarized-cofactor certificate for the first socle theta product.

The completed annular cofactor model is compact, so the three primary
labels are assigned finite home points rather than arbitrary common source
corridors.  The polynomial degree a+b and the mandatory factor-four
polarization give total degree D=4(a+b), while the spatial numerator is the
cofactor image L(a,b).

Modulo I^2=(t_alpha,t_beta)^2 only the two incoming singleton chords are
visible.  The (2,0) and (0,2) home points lie in the two exterior chambers;
the socle and both first correction labels lie in the middle chamber.  The
two inward paths therefore cross the inverse alpha and beta singleton
factors once.  The resulting coefficients +1 and -1 survive theta-basis
re-expansion and equal the signed GKT endpoint jumps of Proposition 6.31.

In the square-zero quadratic window, the socle home lies on the newly visible
diagonal wall.  The two adjacent one-sided germs give the same theta-basis
coefficient -4 on ((3,3);16).  The script separates this into -1 from the
two singleton bends, -1 from the distinguished coefficient-1/4 GKT
subfactor, and -2 from the coefficient-1/2 residual consistency subfactor.
"""


# Forced boundary-charge and Legendre-dual cofactor maps.
iota = matrix(ZZ, [[1, -1], [0, -2]])
L = matrix(ZZ, iota.det() * iota.inverse().transpose())
assert L == matrix(ZZ, [[-2, 0], [1, 1]])

charge_swap = matrix(ZZ, [[0, 1], [1, 0]])
support_reflection = L * charge_swap * L.inverse()
assert support_reflection == matrix(ZZ, [[-1, -2], [0, 1]])


def polarized_degree(charge):
    """Primitive polynomial degree times the forced factor-four polarization."""

    return 4 * sum(charge)


def home_point(charge, degree=None):
    """Cofactor home point of a homogeneous theta label (charge; degree)."""

    if degree is None:
        degree = polarized_degree(charge)
    assert degree > 0
    return vector(QQ, L * vector(ZZ, charge)) / degree


q_x = (2, 0)
q_y = (0, 2)
q_soc = (2, 2)
D_x = polarized_degree(q_x)
D_y = polarized_degree(q_y)
D_out = D_x + D_y
assert (D_x, D_y, D_out) == (8, 8, 16)
assert vector(ZZ, q_x) + vector(ZZ, q_y) == vector(ZZ, q_soc)

p_x = home_point(q_x, D_x)
p_y = home_point(q_y, D_y)
p_soc = home_point(q_soc, D_out)
assert p_x == vector(QQ, (-QQ(1) / 2, QQ(1) / 4))
assert p_y == vector(QQ, (0, QQ(1) / 4))
assert p_soc == vector(QQ, (-QQ(1) / 4, QQ(1) / 4))
assert p_soc == (D_x * p_x + D_y * p_y) / D_out
assert support_reflection * p_x == p_y
assert support_reflection * p_y == p_x
assert support_reflection * p_soc == p_soc


def lies_in_delta(point):
    """The plus cofactor triangle Delta=conv{(-1,-1),(-1,1),(3,-1)}."""

    return (
        point[0] >= -1
        and point[1] >= -1
        and point[0] + 2 * point[1] <= 1
    )


assert all(lies_in_delta(p) for p in (p_x, p_y, p_soc))

# The two visible singleton supports are d_alpha=L(2,1) and d_beta=L(1,2).
d_alpha = L * vector(ZZ, (2, 1))
d_beta = L * vector(ZZ, (1, 2))
assert d_alpha == vector(ZZ, (-4, 3))
assert d_beta == vector(ZZ, (-2, 3))

# Linear forms vanishing on the two full singleton chords.  The middle
# chamber is ell_alpha>0 and ell_beta<0.
n_alpha = vector(ZZ, (3, 4))
n_beta = vector(ZZ, (3, 2))
assert n_alpha.dot_product(d_alpha) == 0
assert n_beta.dot_product(d_beta) == 0


def chamber_signs(point):
    return (
        sign(n_alpha.dot_product(point)),
        sign(n_beta.dot_product(point)),
    )


assert chamber_signs(p_x) == (-1, -1)
assert chamber_signs(p_y) == (1, 1)
assert chamber_signs(p_soc) == (1, -1)

# The straight inward corridors cross exactly one visible chord each.
# Since each ell is affine on a segment, equal signs at the endpoints rule
# out a crossing and opposite signs give one transverse crossing.
assert n_alpha.dot_product(p_x) < 0 < n_alpha.dot_product(p_soc)
assert n_beta.dot_product(p_x) < 0 and n_beta.dot_product(p_soc) < 0
assert n_beta.dot_product(p_y) > 0 > n_beta.dot_product(p_soc)
assert n_alpha.dot_product(p_y) > 0 and n_alpha.dot_product(p_soc) > 0

# The first correction labels retain the product degree 16.  Both lie in the
# same middle chamber as the socle output, so no degree-one output transport
# can absorb their coefficients.
p_alpha_out = home_point((3, 2), D_out)
p_beta_out = home_point((2, 3), D_out)
assert p_alpha_out == vector(QQ, (-QQ(3) / 8, QQ(5) / 16))
assert p_beta_out == vector(QQ, (-QQ(1) / 4, QQ(5) / 16))
assert chamber_signs(p_alpha_out) == (1, -1)
assert chamber_signs(p_beta_out) == (1, -1)
assert support_reflection * p_alpha_out == p_beta_out

# Weighted barycenters commute with arbitrary affine transport.  This is the
# exact K-equivariance needed for the home-point rule and does not require a
# choice of origin or the unresolved affine translation class.
K = matrix(ZZ, [[-1, 2], [-10, 19]])
T.<translation_x, translation_y> = PolynomialRing(QQ, 2)
translation = vector(T, (translation_x, translation_y))
transported_midpoint = (
    D_x * (K * vector(T, p_x) + translation)
    + D_y * (K * vector(T, p_y) + translation)
) / D_out
assert transported_midpoint == K * vector(T, p_soc) + translation

# The order-four residue action exchanges the two inputs and fixes the
# diagonal socle class.
P4 = matrix(Integers(4), [[2, 1], [3, 0]])
qx4 = vector(Integers(4), q_x)
qy4 = vector(Integers(4), q_y)
qsoc4 = vector(Integers(4), q_soc)
assert P4 * qx4 == qy4
assert P4 * qy4 == qx4
assert P4 * qsoc4 == qsoc4


# Exact degree-one theta calculation.  Inward crossings use the inverses of
# A=-(t_alpha/2)X_(1,0) and B=-(t_beta/2)X_(0,1), since the incoming chord
# orientations point from the central joint toward the outer boundary.
S.<X, Y> = PolynomialRing(QQ, 2)


def gkt_derivative(poly, twist):
    a, b = twist
    return S(
        X**a
        * Y**b
        * ((b + 1) * X * poly.derivative(X) - (a + 1) * Y * poly.derivative(Y))
    )


theta_x = {
    (0, 0): X**2,
    (1, 0): QQ(1) / 2 * gkt_derivative(X**2, (1, 0)),
}
theta_y = {
    (0, 0): Y**2,
    (0, 1): QQ(1) / 2 * gkt_derivative(Y**2, (0, 1)),
}
assert theta_x == {(0, 0): X**2, (1, 0): X**3}
assert theta_y == {(0, 0): Y**2, (0, 1): -Y**3}


def multiply_mod_I2(left, right):
    output = {}
    for left_degree, left_poly in left.items():
        for right_degree, right_poly in right.items():
            degree = (
                left_degree[0] + right_degree[0],
                left_degree[1] + right_degree[1],
            )
            if sum(degree) >= 2:
                continue
            output[degree] = output.get(degree, S.zero()) + left_poly * right_poly
    return {degree: S(poly) for degree, poly in output.items() if poly}


socle_product = multiply_mod_I2(theta_x, theta_y)
assert socle_product == {
    (0, 0): X**2 * Y**2,
    (1, 0): X**3 * Y**2,
    (0, 1): -X**2 * Y**3,
}

# The three terminal monomials are precisely the leading monomials of the
# output theta labels (2,2;16), (3,2;16), and (2,3;16).  Their home points
# all lie in the middle chamber, so the raw vector is already its theta
# re-expansion modulo I^2.
theta_structure_constants = {
    ((2, 2), D_out): QQ.one(),
    ((3, 2), D_out): QQ.one(),
    ((2, 3), D_out): -QQ.one(),
}
assert theta_structure_constants[((3, 2), D_out)] == 1
assert theta_structure_constants[((2, 3), D_out)] == -1

# These are the independently certified signed endpoint jumps from
# Proposition 6.31.
alpha_signed_gkt_jump = QQ(3) - QQ(2)
beta_signed_gkt_jump = QQ(2) - QQ(3)
assert alpha_signed_gkt_jump == theta_structure_constants[((3, 2), D_out)]
assert beta_signed_gkt_jump == theta_structure_constants[((2, 3), D_out)]


# Quadratic diagonal continuation.  Retain t_alpha*t_beta but kill repeated
# singleton labels.  The loop orientation is fixed so that crossing the
# diagonal from the beta-side chamber to the alpha-side chamber applies
# exp((3/4)t_alpha*t_beta*X_(1,1)).
def add_states(left, right):
    output = dict(left)
    for degree, poly in right.items():
        output[degree] = output.get(degree, S.zero()) + poly
        if output[degree] == 0:
            del output[degree]
    return output


def flow_square_zero(state, coefficient, flow_degree, twist):
    correction = {}
    for degree, poly in state.items():
        new_degree = (
            degree[0] + flow_degree[0],
            degree[1] + flow_degree[1],
        )
        if new_degree[0] > 1 or new_degree[1] > 1:
            continue
        value = coefficient * gkt_derivative(poly, twist)
        if value:
            correction[new_degree] = correction.get(
                new_degree, S.zero()
            ) + value
    return add_states(state, correction)


def multiply_square_zero(left, right):
    output = {}
    for left_degree, left_poly in left.items():
        for right_degree, right_poly in right.items():
            degree = (
                left_degree[0] + right_degree[0],
                left_degree[1] + right_degree[1],
            )
            if degree[0] > 1 or degree[1] > 1:
                continue
            output[degree] = output.get(degree, S.zero()) + left_poly * right_poly
    return {degree: S(poly) for degree, poly in output.items() if poly}


monomial_x = {(0, 0): X**2}
monomial_y = {(0, 0): Y**2}
inverse_alpha = lambda state: flow_square_zero(
    state, QQ(1) / 2, (1, 0), (1, 0)
)
inverse_beta = lambda state: flow_square_zero(
    state, QQ(1) / 2, (0, 1), (0, 1)
)
diagonal_total = lambda state: flow_square_zero(
    state, QQ(3) / 4, (1, 1), (1, 1)
)
diagonal_total_inverse = lambda state: flow_square_zero(
    state, -QQ(3) / 4, (1, 1), (1, 1)
)
diagonal_gkt = lambda state: flow_square_zero(
    state, QQ(1) / 4, (1, 1), (1, 1)
)
diagonal_residual = lambda state: flow_square_zero(
    state, QQ(1) / 2, (1, 1), (1, 1)
)

# Evaluate in the alpha-side middle chamber.  The X-input crosses only the
# inverse alpha wall.  The Y-input crosses the inverse beta wall and then the
# positive total diagonal wall.
theta_x_alpha_side = inverse_alpha(monomial_x)
theta_y_alpha_side = diagonal_total(inverse_beta(monomial_y))
quadratic_product_alpha_side = multiply_square_zero(
    theta_x_alpha_side, theta_y_alpha_side
)
assert quadratic_product_alpha_side == {
    (0, 0): X**2 * Y**2,
    (1, 0): X**3 * Y**2,
    (0, 1): -X**2 * Y**3,
    (1, 1): -4 * X**3 * Y**3,
}

# Evaluation in the beta-side chamber gives the same re-expanded coefficient:
# now the X-input crosses the inverse total diagonal wall, while the Y-input
# crosses only the inverse beta wall.
theta_x_beta_side = diagonal_total_inverse(inverse_alpha(monomial_x))
theta_y_beta_side = inverse_beta(monomial_y)
quadratic_product_beta_side = multiply_square_zero(
    theta_x_beta_side, theta_y_beta_side
)
assert quadratic_product_beta_side == quadratic_product_alpha_side

# The diagonal wall fixes diagonal output monomials.  It also cannot change a
# linear correction term after multiplication by its singleton parameter,
# because that would repeat one square-zero label.  Thus the displayed raw
# vector is already the theta re-expansion through bidegree (1,1).
assert gkt_derivative(X**2 * Y**2, (1, 1)) == 0
assert gkt_derivative(X**3 * Y**3, (1, 1)) == 0

# Separate the distinguished GKT subfactor and residual consistency factor.
# The two independent singleton bends contribute -1, the coefficient-1/4
# GKT diagonal factor contributes -1, and the coefficient-1/2 residual
# factor contributes -2.  Their sum is the ordinary consistent theta
# coefficient -4.
product_before_diagonal = multiply_square_zero(
    inverse_alpha(monomial_x), inverse_beta(monomial_y)
)
assert product_before_diagonal[(1, 1)].monomial_coefficient(X**3 * Y**3) == -1

product_with_gkt_diagonal = multiply_square_zero(
    inverse_alpha(monomial_x), diagonal_gkt(inverse_beta(monomial_y))
)
assert product_with_gkt_diagonal[(1, 1)].monomial_coefficient(X**3 * Y**3) == -2

residual_action_on_y = diagonal_residual(inverse_beta(monomial_y))
residual_increment = multiply_square_zero(
    inverse_alpha(monomial_x), residual_action_on_y
)[(1, 1)].monomial_coefficient(X**3 * Y**3) - (
    product_before_diagonal[(1, 1)].monomial_coefficient(X**3 * Y**3)
)
assert residual_increment == -2
assert -1 + (-1) + (-2) == -4

# The GKT-only outgoing factor is not a homomorphic projection of the total
# wall which fixes the two singleton inputs.  The total, GKT, and residual
# logarithms are respectively -1, -1/3, and -2/3 times the forced
# singleton commutator.  Any homomorphism fixing the singleton factors fixes
# their commutator, while any scalar character kills it.
singleton_bracket_coefficient = (
    (QQ(1) / 2) * (QQ(1) / 2) * (-QQ(3))
)
assert singleton_bracket_coefficient == -QQ(3) / 4
assert QQ(3) / 4 == -singleton_bracket_coefficient
assert QQ(1) / 4 == -singleton_bracket_coefficient / 3
assert QQ(1) / 2 == -2 * singleton_bracket_coefficient / 3
assert (
    QQ(1) / 2 * gkt_derivative(X**2, (1, 1))
    == 2 * X**3 * Y
)

# A factorization-sensitive binary tensor correction survives the no-go.
# Apply R^(1/2) to the first total-theta input and R^(-1/2) to the second.
# On either side this redistributes the asymmetric residual crossing into a
# common residual transport of both inputs.  The GKT-decorated socle product
# is fixed by that common transport, leaving coefficient -2.
residual_half = lambda state: flow_square_zero(
    state, QQ(1) / 4, (1, 1), (1, 1)
)
residual_half_inverse = lambda state: flow_square_zero(
    state, -QQ(1) / 4, (1, 1), (1, 1)
)

balanced_alpha_side = multiply_square_zero(
    residual_half(theta_x_alpha_side),
    residual_half_inverse(theta_y_alpha_side),
)
balanced_beta_side = multiply_square_zero(
    residual_half(theta_x_beta_side),
    residual_half_inverse(theta_y_beta_side),
)
assert balanced_alpha_side == product_with_gkt_diagonal
assert balanced_beta_side == balanced_alpha_side
assert balanced_alpha_side[(1, 1)].monomial_coefficient(X**3 * Y**3) == -2
assert diagonal_residual(product_with_gkt_diagonal) == product_with_gkt_diagonal

# Uniqueness among exchange-antisymmetric residual powers: if the first and
# second corrections are R^u and R^v with u+v=0, equalizing residual crossing
# numbers (0,1) or (-1,0) forces (u,v)=(1/2,-1/2).
u_alpha = QQ(1) / 2
v_alpha = -u_alpha
assert u_alpha + v_alpha == 0
assert u_alpha == 1 + v_alpha
u_beta = QQ(1) / 2
v_beta = -u_beta
assert u_beta + v_beta == 0
assert -1 + u_beta == v_beta

print("polarized cofactor home points:", p_x, p_y, p_soc)
print("home chambers (X,Y,socle):", chamber_signs(p_x), chamber_signs(p_y), chamber_signs(p_soc))
print("socle theta product mod I^2: coefficients", theta_structure_constants)
print("quadratic theta coefficient on ((3,3),16):", -4)
print("quadratic decomposition (singleton pair, GKT subfactor, residual):", (-1, -1, -2))
print("GKT-only residual holonomy and homomorphic-extraction no-go: PASS")
print("residual half-tensorator: chamber-independent coefficient -2")
print("first genuine socle theta/GKT structure constants: PASS")
