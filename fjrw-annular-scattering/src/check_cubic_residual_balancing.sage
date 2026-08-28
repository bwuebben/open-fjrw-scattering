"""Cubic deformation-order test of residual balancing for the socle product.

The script treats both mixed three-label windows alpha^2 beta and
alpha beta^2, with all deformation labels individually square-zero.  It uses
the GKT/scattering/residual coefficients of Propositions 6.21--6.22 and the
cofactor homes of Proposition 6.33.  It compares the two one-sided socle
corridors at total-scattering, GKT-decorated, naive raywise-half, and
residual-torsor-midpoint levels without identifying deformation labels with
theta tensor factors.  All unipotent midpoint calculations are exact over
QQ on a dependency-closed finite monomial state space.
"""


S.<X, Y> = PolynomialRing(QQ, 2)


def jacobi_derivative(poly, twist):
    a, b = twist
    return S(
        X**a
        * Y**b
        * ((b + 1) * X * poly.derivative(X) - (a + 1) * Y * poly.derivative(Y))
    )


def add_states(left, right):
    output = dict(left)
    for support, poly in right.items():
        output[support] = output.get(support, S.zero()) + poly
        if output[support] == 0:
            del output[support]
    return output


def apply_generator(state, generator):
    support, twist, coefficient = generator
    output = {}
    for old_support, poly in state.items():
        if old_support & support:
            continue
        value = coefficient * jacobi_derivative(poly, twist)
        if value:
            new_support = old_support | support
            output[new_support] = output.get(new_support, S.zero()) + value
    return output


def flow(state, generator):
    return add_states(state, apply_generator(state, generator))


def chronological(state, generators):
    output = state
    for generator in generators:
        output = flow(output, generator)
    return output


def multiply(left, right):
    output = {}
    for left_support, left_poly in left.items():
        for right_support, right_poly in right.items():
            if left_support & right_support:
                continue
            support = left_support | right_support
            output[support] = output.get(support, S.zero()) + left_poly * right_poly
    return {support: S(poly) for support, poly in output.items() if poly}


def scaled(generator, scalar):
    support, twist, coefficient = generator
    return (support, twist, scalar * coefficient)


# Cofactor geometry and crossing order.
L = matrix(ZZ, [[-2, 0], [1, 1]])
p_x = vector(QQ, (-QQ(1) / 2, QQ(1) / 4))
p_y = vector(QQ, (0, QQ(1) / 4))
p_soc = vector(QQ, (-QQ(1) / 4, QQ(1) / 4))
epsilon = QQ(1) / 100
q_alpha = p_soc + vector(QQ, (-epsilon, 0))
q_beta = p_soc + vector(QQ, (epsilon, 0))


def segment_ray_parameter(source, target, direction):
    """Return (segment parameter, ray parameter), or None if no crossing."""

    # source+t(target-source)=s*direction
    matrix_system = matrix(QQ, [target - source, -direction]).transpose()
    if matrix_system.det() == 0:
        return None
    t, s = matrix_system.solve_right(-source)
    if 0 < t < 1 and s > 0:
        return (t, s)
    return None


directions = {
    "aa": vector(QQ, L * vector(ZZ, (3, 1))),
    "alpha": vector(QQ, L * vector(ZZ, (2, 1))),
    "aab": vector(QQ, L * vector(ZZ, (3, 2))),
    "diag": vector(QQ, L * vector(ZZ, (2, 2))),
    "beta": vector(QQ, L * vector(ZZ, (1, 2))),
}
assert directions == {
    "aa": vector(QQ, (-6, 4)),
    "alpha": vector(QQ, (-4, 3)),
    "aab": vector(QQ, (-6, 5)),
    "diag": vector(QQ, (-4, 4)),
    "beta": vector(QQ, (-2, 3)),
}


def crossed_in_order(source, target, ray_directions=directions):
    crossings = []
    for name, direction in ray_directions.items():
        parameter = segment_ray_parameter(source, target, direction)
        if parameter is not None:
            crossings.append((parameter[0], name))
    return tuple(name for _, name in sorted(crossings))


assert crossed_in_order(p_x, q_alpha) == ("aa", "alpha", "aab")
assert crossed_in_order(p_y, q_alpha) == ("beta", "diag")
assert crossed_in_order(p_x, q_beta) == ("aa", "alpha", "aab", "diag")
assert crossed_in_order(p_y, q_beta) == ("beta",)


# Three distinct square-zero deformation labels.
a1 = 1
a2 = 2
b = 4
aa = a1 | a2
a1b = a1 | b
a2b = a2 | b
aab = a1 | a2 | b

inverse_A1 = (a1, (1, 0), QQ(1) / 2)
inverse_A2 = (a2, (1, 0), QQ(1) / 2)
inverse_B = (b, (0, 1), QQ(1) / 2)

# GKT factors E, total scattering factors S, and residuals R=S-E.
E_aa = (aa, (2, 0), QQ(1) / 2)
R_aa = (aa, (2, 0), -QQ(1) / 2)
E_a1b = (a1b, (1, 1), QQ(1) / 4)
E_a2b = (a2b, (1, 1), QQ(1) / 4)
R_a1b = (a1b, (1, 1), QQ(1) / 2)
R_a2b = (a2b, (1, 1), QQ(1) / 2)
S_a1b = (a1b, (1, 1), QQ(3) / 4)
S_a2b = (a2b, (1, 1), QQ(3) / 4)
E_aab = (aab, (2, 1), -QQ(1) / 2)
R_aab = (aab, (2, 1), QQ(5) / 4)
S_aab = (aab, (2, 1), QQ(3) / 4)

zero_slot = chronological({0: X}, [E_aa, R_aa])
assert zero_slot == {0: X}
for enumerative, residual, total in (
    (E_a1b, R_a1b, S_a1b),
    (E_a2b, R_a2b, S_a2b),
    (E_aab, R_aab, S_aab),
):
    assert enumerative[2] + residual[2] == total[2]

monomial_x = {0: X**2}
monomial_y = {0: Y**2}

# Total-scattering theta inputs.  Decreasing-angle outgoing crossings use
# inverse factors; increasing-angle crossings use positive factors.
x_total_alpha = chronological(
    monomial_x,
    [inverse_A1, inverse_A2, scaled(S_aab, -1)],
)
y_total_alpha = chronological(
    monomial_y,
    [inverse_B, S_a1b, S_a2b],
)
product_total_alpha = multiply(x_total_alpha, y_total_alpha)

x_total_beta = chronological(
    monomial_x,
    [
        inverse_A1,
        inverse_A2,
        scaled(S_aab, -1),
        scaled(S_a1b, -1),
        scaled(S_a2b, -1),
    ],
)
y_total_beta = chronological(monomial_y, [inverse_B])
product_total_beta = multiply(x_total_beta, y_total_beta)

# Factorization-decorated GKT paths include the formal E_aa R_aa identity
# slot on the aa ray.  Retaining E and omitting R gives the GKT word.
x_gkt_alpha = chronological(
    monomial_x,
    [scaled(E_aa, -1), inverse_A1, inverse_A2, scaled(E_aab, -1)],
)
y_gkt_alpha = chronological(
    monomial_y,
    [inverse_B, E_a1b, E_a2b],
)
product_gkt_alpha = multiply(x_gkt_alpha, y_gkt_alpha)

x_gkt_beta = chronological(
    monomial_x,
    [
        scaled(E_aa, -1),
        inverse_A1,
        inverse_A2,
        scaled(E_aab, -1),
        scaled(E_a1b, -1),
        scaled(E_a2b, -1),
    ],
)
y_gkt_beta = chronological(monomial_y, [inverse_B])
product_gkt_beta = multiply(x_gkt_beta, y_gkt_beta)

# Naive coefficientwise extension of the quadratic half-tensorator.  The
# residual half-word is ordered by the cofactor-ray crossing order
# aa, aab, diag; its inverse word is applied to the second theta leg.
residual_half_first = [
    scaled(R_aa, QQ(1) / 2),
    scaled(R_aab, QQ(1) / 2),
    scaled(R_a1b, QQ(1) / 2),
    scaled(R_a2b, QQ(1) / 2),
]
residual_half_second = [
    scaled(R_aa, -QQ(1) / 2),
    scaled(R_aab, -QQ(1) / 2),
    scaled(R_a1b, -QQ(1) / 2),
    scaled(R_a2b, -QQ(1) / 2),
]

balanced_alpha = multiply(
    chronological(x_total_alpha, residual_half_first),
    chronological(y_total_alpha, residual_half_second),
)
balanced_beta = multiply(
    chronological(x_total_beta, residual_half_first),
    chronological(y_total_beta, residual_half_second),
)

# Re-expand beta-side vectors in the alpha-side output chart.  The chambers
# are separated only by the two labelled diagonal pair factors.
total_beta_in_alpha_chart = chronological(
    product_total_beta, [S_a1b, S_a2b]
)
gkt_beta_in_alpha_chart = chronological(
    product_gkt_beta, [E_a1b, E_a2b]
)
balanced_beta_in_alpha_chart_total = chronological(
    balanced_beta, [S_a1b, S_a2b]
)
balanced_beta_in_alpha_chart_gkt = chronological(
    balanced_beta, [E_a1b, E_a2b]
)

# Noncommutative torsor midpoint of the complete residual path holonomies.
# This replaces both the naive raywise half-word and the non-biequivariant
# arithmetic average of logarithms at cubic order.
def inverse_word(word):
    return [scaled(generator, -1) for generator in reversed(word)]


def relative_left_word(total_word, gkt_word):
    """Q with Op(total)=Q*Op(gkt), in chronological-list convention."""

    return inverse_word(gkt_word) + list(total_word)


word_x_total_alpha = [
    scaled(E_aa, -1),
    scaled(R_aa, -1),
    inverse_A1,
    inverse_A2,
    scaled(E_aab, -1),
    scaled(R_aab, -1),
]
word_x_gkt_alpha = [
    scaled(E_aa, -1),
    inverse_A1,
    inverse_A2,
    scaled(E_aab, -1),
]
word_y_total_alpha = [
    inverse_B,
    E_a1b,
    R_a1b,
    E_a2b,
    R_a2b,
]
word_y_gkt_alpha = [inverse_B, E_a1b, E_a2b]

word_x_total_beta = [
    scaled(E_aa, -1),
    scaled(R_aa, -1),
    inverse_A1,
    inverse_A2,
    scaled(E_aab, -1),
    scaled(R_aab, -1),
    scaled(E_a1b, -1),
    scaled(R_a1b, -1),
    scaled(E_a2b, -1),
    scaled(R_a2b, -1),
]
word_x_gkt_beta = [
    scaled(E_aa, -1),
    inverse_A1,
    inverse_A2,
    scaled(E_aab, -1),
    scaled(E_a1b, -1),
    scaled(E_a2b, -1),
]
word_y_total_beta = [inverse_B]
word_y_gkt_beta = [inverse_B]

assert chronological(monomial_x, word_x_total_alpha) == x_total_alpha
assert chronological(monomial_x, word_x_gkt_alpha) == x_gkt_alpha
assert chronological(monomial_y, word_y_total_alpha) == y_total_alpha
assert chronological(monomial_y, word_y_gkt_alpha) == y_gkt_alpha
assert chronological(monomial_x, word_x_total_beta) == x_total_beta
assert chronological(monomial_x, word_x_gkt_beta) == x_gkt_beta
assert chronological(monomial_y, word_y_total_beta) == y_total_beta
assert chronological(monomial_y, word_y_gkt_beta) == y_gkt_beta


# Build a finite faithful state space generated from both input monomials and
# their classical product under the labelled derivations in this window.
generator_templates = {
    (generator[0], generator[1])
    for generator in (
        inverse_A1,
        inverse_A2,
        inverse_B,
        E_aa,
        R_aa,
        E_a1b,
        R_a1b,
        E_a2b,
        R_a2b,
        E_aab,
        R_aab,
    )
}
# Also close the state space under the reflected alpha beta^2 generators,
# which are tested below with the same exact matrix machinery.
generator_templates.update(
    {
        (1, (1, 0)),
        (2, (0, 1)),
        (4, (0, 1)),
        (6, (0, 2)),
        (3, (1, 1)),
        (5, (1, 1)),
        (7, (1, 2)),
    }
)
basis_set = {(0, 2, 0), (0, 0, 2), (0, 2, 2)}
frontier = list(basis_set)
while frontier:
    support, m, n = frontier.pop()
    for generator_support, twist in generator_templates:
        if support & generator_support:
            continue
        a, b_twist = twist
        scalar = (b_twist + 1) * m - (a + 1) * n
        if scalar == 0:
            continue
        target = (support | generator_support, m + a, n + b_twist)
        if target not in basis_set:
            basis_set.add(target)
            frontier.append(target)

basis = tuple(sorted(basis_set))
basis_index = {term: index for index, term in enumerate(basis)}
dimension = len(basis)


def flow_matrix(generator):
    support_g, twist, coefficient_g = generator
    output = identity_matrix(QQ, dimension)
    a, b_twist = twist
    for column, (support, m, n) in enumerate(basis):
        if support & support_g:
            continue
        scalar = (b_twist + 1) * m - (a + 1) * n
        if scalar == 0:
            continue
        target = (support | support_g, m + a, n + b_twist)
        output[basis_index[target], column] += coefficient_g * scalar
    return output


def word_matrix(word):
    output = identity_matrix(QQ, dimension)
    for generator in word:
        output = flow_matrix(generator) * output
    return output


def state_vector(state):
    output = vector(QQ, dimension)
    for support, poly in state.items():
        for exponent, coefficient in poly.dict().items():
            output[basis_index[(support, exponent[0], exponent[1])]] += coefficient
    return output


def vector_state(value):
    output = {}
    for coefficient, (support, m, n) in zip(value, basis):
        if coefficient:
            output[support] = output.get(support, S.zero()) + coefficient * X**m * Y**n
    return {support: S(poly) for support, poly in output.items() if poly}


def nilpotent_log(matrix_value):
    identity = identity_matrix(QQ, dimension)
    nilpotent = matrix_value - identity
    output = zero_matrix(QQ, dimension)
    power = nilpotent
    for degree in range(1, 8):
        if power == 0:
            return output
        output += ((-1) ** (degree + 1)) * power / degree
        power *= nilpotent
    raise AssertionError("matrix logarithm did not truncate")


def nilpotent_exp(matrix_value):
    identity = identity_matrix(QQ, dimension)
    output = identity
    power = identity
    factorial = QQ.one()
    for degree in range(1, 8):
        power *= matrix_value
        factorial *= degree
        if power == 0:
            return output
        output += power / factorial
    raise AssertionError("matrix exponential did not truncate")


def torsor_midpoint(Q1, Q2):
    """The symmetric midpoint Q1*(Q1^-1*Q2)^(1/2) of a unipotent bitorsor."""

    relative_12 = Q1.inverse() * Q2
    relative_21 = Q2.inverse() * Q1
    H = Q1 * nilpotent_exp(nilpotent_log(relative_12) / 2)
    H_from_other_leg = Q2 * nilpotent_exp(nilpotent_log(relative_21) / 2)
    assert H == H_from_other_leg
    return H


def midpoint_correction(total_word_1, gkt_word_1, total_word_2, gkt_word_2):
    Q1 = word_matrix(relative_left_word(total_word_1, gkt_word_1))
    Q2 = word_matrix(relative_left_word(total_word_2, gkt_word_2))
    H = torsor_midpoint(Q1, Q2)
    C1 = H * Q1.inverse()
    C2 = H * Q2.inverse()
    return Q1, Q2, H, C1, C2


def assert_biequivariance(Q1, Q2, total_extension, gkt_extension):
    """Check H(TQ1W^-1,TQ2W^-1)=T H(Q1,Q2) W^-1 exactly."""

    W_inverse = gkt_extension.inverse()
    transformed_Q1 = total_extension * Q1 * W_inverse
    transformed_Q2 = total_extension * Q2 * W_inverse
    transformed_H = torsor_midpoint(transformed_Q1, transformed_Q2)
    expected_H = total_extension * torsor_midpoint(Q1, Q2) * W_inverse
    assert transformed_H == expected_H


def assert_corridor_transport(alpha_data, beta_data, total_word, gkt_word):
    """Check the full residual correction under an actual chamber change."""

    T = word_matrix(total_word)
    W = word_matrix(gkt_word)
    W_inverse = W.inverse()
    T_inverse = T.inverse()
    assert_biequivariance(beta_data["Q1"], beta_data["Q2"], T, W)
    for key in ("Q1", "Q2", "H"):
        assert alpha_data[key] == T * beta_data[key] * W_inverse
    for key in ("C1", "C2"):
        assert alpha_data[key] == T * beta_data[key] * T_inverse
    assert state_vector(alpha_data["gkt_product"]) == (
        W * state_vector(beta_data["gkt_product"])
    )
    assert state_vector(alpha_data["product"]) == (
        T * state_vector(beta_data["product"])
    )


def midpoint_balanced_product(
    total_1,
    gkt_1,
    total_word_1,
    gkt_word_1,
    total_2,
    gkt_2,
    total_word_2,
    gkt_word_2,
):
    Q1, Q2, H, C1, C2 = midpoint_correction(
        total_word_1, gkt_word_1, total_word_2, gkt_word_2
    )
    total_vector_1 = state_vector(total_1)
    total_vector_2 = state_vector(total_2)
    gkt_vector_1 = state_vector(gkt_1)
    gkt_vector_2 = state_vector(gkt_2)
    assert Q1 * gkt_vector_1 == total_vector_1
    assert Q2 * gkt_vector_2 == total_vector_2

    corrected_1 = vector_state(C1 * total_vector_1)
    corrected_2 = vector_state(C2 * total_vector_2)
    common_1 = vector_state(H * gkt_vector_1)
    common_2 = vector_state(H * gkt_vector_2)
    assert corrected_1 == common_1
    assert corrected_2 == common_2

    corrected_product = multiply(corrected_1, corrected_2)
    gkt_product = multiply(gkt_1, gkt_2)
    common_product = vector_state(H * state_vector(gkt_product))
    assert corrected_product == common_product
    return {
        "Q1": Q1,
        "Q2": Q2,
        "H": H,
        "C1": C1,
        "C2": C2,
        "product": corrected_product,
        "gkt_product": gkt_product,
        "common_product": common_product,
    }


midpoint_alpha = midpoint_balanced_product(
    x_total_alpha,
    x_gkt_alpha,
    word_x_total_alpha,
    word_x_gkt_alpha,
    y_total_alpha,
    y_gkt_alpha,
    word_y_total_alpha,
    word_y_gkt_alpha,
)
midpoint_beta = midpoint_balanced_product(
    x_total_beta,
    x_gkt_beta,
    word_x_total_beta,
    word_x_gkt_beta,
    y_total_beta,
    y_gkt_beta,
    word_y_total_beta,
    word_y_gkt_beta,
)
assert_corridor_transport(
    midpoint_alpha,
    midpoint_beta,
    [S_a1b, S_a2b],
    [E_a1b, E_a2b],
)
alpha_total_transport = word_matrix([S_a1b, S_a2b])
alpha_gkt_transport = word_matrix([E_a1b, E_a2b])
for alpha_word, beta_word in (
    (word_x_total_alpha, word_x_total_beta),
    (word_y_total_alpha, word_y_total_beta),
):
    assert word_matrix(alpha_word) == alpha_total_transport * word_matrix(beta_word)
for alpha_word, beta_word in (
    (word_x_gkt_alpha, word_x_gkt_beta),
    (word_y_gkt_alpha, word_y_gkt_beta),
):
    assert word_matrix(alpha_word) == alpha_gkt_transport * word_matrix(beta_word)

# Reflected mixed cubic window alpha beta^2.  Here the cubic GKT,
# scattering, and residual coefficients are -1/2, -3/4, and -1/4.
ra = 1
rb1 = 2
rb2 = 4
rbb = rb1 | rb2
rab1 = ra | rb1
rab2 = ra | rb2
rabb = ra | rb1 | rb2

r_inverse_A = (ra, (1, 0), QQ(1) / 2)
r_inverse_B1 = (rb1, (0, 1), QQ(1) / 2)
r_inverse_B2 = (rb2, (0, 1), QQ(1) / 2)
r_E_bb = (rbb, (0, 2), QQ(1) / 2)
r_R_bb = (rbb, (0, 2), -QQ(1) / 2)
r_E_ab1 = (rab1, (1, 1), QQ(1) / 4)
r_E_ab2 = (rab2, (1, 1), QQ(1) / 4)
r_R_ab1 = (rab1, (1, 1), QQ(1) / 2)
r_R_ab2 = (rab2, (1, 1), QQ(1) / 2)
r_E_abb = (rabb, (1, 2), -QQ(1) / 2)
r_R_abb = (rabb, (1, 2), -QQ(1) / 4)

r_directions = {
    "bb": vector(QQ, L * vector(ZZ, (1, 3))),
    "beta": directions["beta"],
    "abb": vector(QQ, L * vector(ZZ, (2, 3))),
    "diag": directions["diag"],
    "alpha": directions["alpha"],
}
assert r_directions == {
    "bb": vector(QQ, (-2, 4)),
    "beta": vector(QQ, (-2, 3)),
    "abb": vector(QQ, (-4, 5)),
    "diag": vector(QQ, (-4, 4)),
    "alpha": vector(QQ, (-4, 3)),
}
assert crossed_in_order(p_x, q_alpha, r_directions) == ("alpha",)
assert crossed_in_order(p_y, q_alpha, r_directions) == (
    "bb",
    "beta",
    "abb",
    "diag",
)
assert crossed_in_order(p_x, q_beta, r_directions) == ("alpha", "diag")
assert crossed_in_order(p_y, q_beta, r_directions) == (
    "bb",
    "beta",
    "abb",
)

r_word_x_total_alpha = [r_inverse_A]
r_word_x_gkt_alpha = [r_inverse_A]
r_word_y_total_alpha = [
    r_E_bb,
    r_R_bb,
    r_inverse_B1,
    r_inverse_B2,
    r_E_abb,
    r_R_abb,
    r_E_ab1,
    r_R_ab1,
    r_E_ab2,
    r_R_ab2,
]
r_word_y_gkt_alpha = [
    r_E_bb,
    r_inverse_B1,
    r_inverse_B2,
    r_E_abb,
    r_E_ab1,
    r_E_ab2,
]

r_word_x_total_beta = [
    r_inverse_A,
    scaled(r_E_ab1, -1),
    scaled(r_R_ab1, -1),
    scaled(r_E_ab2, -1),
    scaled(r_R_ab2, -1),
]
r_word_x_gkt_beta = [
    r_inverse_A,
    scaled(r_E_ab1, -1),
    scaled(r_E_ab2, -1),
]
r_word_y_total_beta = [
    r_E_bb,
    r_R_bb,
    r_inverse_B1,
    r_inverse_B2,
    r_E_abb,
    r_R_abb,
]
r_word_y_gkt_beta = [
    r_E_bb,
    r_inverse_B1,
    r_inverse_B2,
    r_E_abb,
]

r_x_total_alpha = chronological(monomial_x, r_word_x_total_alpha)
r_x_gkt_alpha = chronological(monomial_x, r_word_x_gkt_alpha)
r_y_total_alpha = chronological(monomial_y, r_word_y_total_alpha)
r_y_gkt_alpha = chronological(monomial_y, r_word_y_gkt_alpha)
r_product_total_alpha = multiply(r_x_total_alpha, r_y_total_alpha)
r_product_gkt_alpha = multiply(r_x_gkt_alpha, r_y_gkt_alpha)

r_x_total_beta = chronological(monomial_x, r_word_x_total_beta)
r_x_gkt_beta = chronological(monomial_x, r_word_x_gkt_beta)
r_y_total_beta = chronological(monomial_y, r_word_y_total_beta)
r_y_gkt_beta = chronological(monomial_y, r_word_y_gkt_beta)
r_product_total_beta = multiply(r_x_total_beta, r_y_total_beta)
r_product_gkt_beta = multiply(r_x_gkt_beta, r_y_gkt_beta)

r_S_ab1 = (
    rab1,
    (1, 1),
    r_E_ab1[2] + r_R_ab1[2],
)
r_S_ab2 = (
    rab2,
    (1, 1),
    r_E_ab2[2] + r_R_ab2[2],
)
assert r_product_total_alpha == chronological(
    r_product_total_beta, [r_S_ab1, r_S_ab2]
)
assert r_product_gkt_alpha == chronological(
    r_product_gkt_beta, [r_E_ab1, r_E_ab2]
)

r_midpoint_alpha = midpoint_balanced_product(
    r_x_total_alpha,
    r_x_gkt_alpha,
    r_word_x_total_alpha,
    r_word_x_gkt_alpha,
    r_y_total_alpha,
    r_y_gkt_alpha,
    r_word_y_total_alpha,
    r_word_y_gkt_alpha,
)
r_midpoint_beta = midpoint_balanced_product(
    r_x_total_beta,
    r_x_gkt_beta,
    r_word_x_total_beta,
    r_word_x_gkt_beta,
    r_y_total_beta,
    r_y_gkt_beta,
    r_word_y_total_beta,
    r_word_y_gkt_beta,
)
assert_corridor_transport(
    r_midpoint_alpha,
    r_midpoint_beta,
    [r_S_ab1, r_S_ab2],
    [r_E_ab1, r_E_ab2],
)
reflected_total_transport = word_matrix([r_S_ab1, r_S_ab2])
reflected_gkt_transport = word_matrix([r_E_ab1, r_E_ab2])
for alpha_word, beta_word in (
    (r_word_x_total_alpha, r_word_x_total_beta),
    (r_word_y_total_alpha, r_word_y_total_beta),
):
    assert word_matrix(alpha_word) == (
        reflected_total_transport * word_matrix(beta_word)
    )
for alpha_word, beta_word in (
    (r_word_x_gkt_alpha, r_word_x_gkt_beta),
    (r_word_y_gkt_alpha, r_word_y_gkt_beta),
):
    assert word_matrix(alpha_word) == (
        reflected_gkt_transport * word_matrix(beta_word)
    )

# The reflected full output (3,4;16) has its home in the beta-side chamber.
# The only lower output corridors cross walls whose deformation support
# overlaps the coefficient support, so beta-side raw GKT coefficients are
# already theta-basis coefficients.
r_output_charge = {
    0: (2, 2),
    ra: (3, 2),
    rb1: (2, 3),
    rb2: (2, 3),
    rbb: (2, 4),
    rab1: (3, 3),
    rab2: (3, 3),
    rabb: (3, 4),
}
r_output_crossings = {
    support: crossed_in_order(
        vector(QQ, L * vector(ZZ, charge)) / 16, q_beta, r_directions
    )
    for support, charge in r_output_charge.items()
}
assert r_output_crossings == {
    0: (),
    ra: ("diag",),
    rb1: (),
    rb2: (),
    rbb: ("abb",),
    rab1: (),
    rab2: (),
    rabb: (),
}
# The nonempty crossings are the diagonal for ra and the reflected abb ray
# for rbb, and both repeat a deformation label.
assert ra & rab1 and ra & rab2
assert rbb & rabb

r_expected_leading = {
    support: X**charge[0] * Y**charge[1]
    for support, charge in r_output_charge.items()
}
for support, poly in r_product_gkt_beta.items():
    assert len(poly.monomials()) == 1
    assert poly.monomials()[0] == r_expected_leading[support]
r_gkt_theta_constants_beta = {
    support: poly.monomial_coefficient(r_expected_leading[support])
    for support, poly in r_product_gkt_beta.items()
}
assert r_gkt_theta_constants_beta[rabb] == QQ(9) / 2
assert r_gkt_theta_constants_beta[rabb] != r_E_abb[2]

# Decompose the two mixed cubic theta coefficients.  The direct full-support
# GKT wall contributes +2 in each natural output chart.  Lower-support GKT
# paths contribute -5/2 in alpha^2 beta and +5/2 in alpha beta^2.
x_gkt_alpha_without_cubic = chronological(
    monomial_x,
    [scaled(E_aa, -1), inverse_A1, inverse_A2],
)
product_gkt_alpha_without_cubic = multiply(
    x_gkt_alpha_without_cubic, y_gkt_alpha
)
alpha_lower_support_contribution = (
    product_gkt_alpha_without_cubic.get(aab, S.zero())
    .monomial_coefficient(X**4 * Y**3)
)

r_y_gkt_beta_without_cubic = chronological(
    monomial_y,
    [r_E_bb, r_inverse_B1, r_inverse_B2],
)
r_product_gkt_beta_without_cubic = multiply(
    r_x_gkt_beta, r_y_gkt_beta_without_cubic
)
reflected_lower_support_contribution = (
    r_product_gkt_beta_without_cubic.get(rabb, S.zero())
    .monomial_coefficient(X**3 * Y**4)
)
reflected_direct_cubic_contribution = (
    r_gkt_theta_constants_beta[rabb] - reflected_lower_support_contribution
)
assert alpha_lower_support_contribution == -QQ(5) / 2
assert reflected_lower_support_contribution == QQ(5) / 2
assert reflected_direct_cubic_contribution == 2

# In the alpha-side GKT output chart, every lower output theta label is
# already its leading monomial at the coefficient support where it occurs.
# The only nonempty output corridors cross walls carrying a repeated
# deformation label, so their action vanishes in the labelled square-zero
# quotient.
output_charge = {
    0: (2, 2),
    a1: (3, 2),
    a2: (3, 2),
    b: (2, 3),
    aa: (4, 2),
    a1b: (3, 3),
    a2b: (3, 3),
    aab: (4, 3),
}
output_crossings = {
    support: crossed_in_order(
        vector(QQ, L * vector(ZZ, charge)) / 16, q_alpha
    )
    for support, charge in output_charge.items()
}
assert output_crossings == {
    0: (),
    a1: (),
    a2: (),
    b: ("diag",),
    aa: ("aab",),
    a1b: (),
    a2b: (),
    aab: (),
}
assert b & a1b and b & a2b
assert aa & aab

expected_leading_monomial = {
    support: X**charge[0] * Y**charge[1]
    for support, charge in output_charge.items()
}
for support, poly in product_gkt_alpha.items():
    assert len(poly.monomials()) == 1
    assert poly.monomials()[0] == expected_leading_monomial[support]

gkt_theta_constants_alpha = {
    support: poly.monomial_coefficient(expected_leading_monomial[support])
    for support, poly in product_gkt_alpha.items()
}
assert gkt_theta_constants_alpha[aab] == -QQ(1) / 2
assert gkt_theta_constants_alpha[aab] == E_aab[2]
alpha_direct_cubic_contribution = (
    gkt_theta_constants_alpha[aab] - alpha_lower_support_contribution
)
assert alpha_direct_cubic_contribution == 2

full = aab
print("cofactor crossings:")
print("  X -> alpha side", crossed_in_order(p_x, q_alpha))
print("  Y -> alpha side", crossed_in_order(p_y, q_alpha))
print("  X -> beta side ", crossed_in_order(p_x, q_beta))
print("  Y -> beta side ", crossed_in_order(p_y, q_beta))
print("full-support X^4Y^3 coefficients:")
for name, state in (
    ("total alpha", product_total_alpha),
    ("total beta", product_total_beta),
    ("GKT alpha", product_gkt_alpha),
    ("GKT beta", product_gkt_beta),
    ("balanced alpha", balanced_alpha),
    ("balanced beta", balanced_beta),
):
    print(" ", name, state.get(full, S.zero()).monomial_coefficient(X**4 * Y**3))
print("total chamber difference:", product_total_alpha.get(full, 0) - product_total_beta.get(full, 0))
print("GKT chamber difference:", product_gkt_alpha.get(full, 0) - product_gkt_beta.get(full, 0))
print("balanced chamber difference:", balanced_alpha.get(full, 0) - balanced_beta.get(full, 0))
print("common-output-chart equalities:")
print("  total via total diagonal", product_total_alpha == total_beta_in_alpha_chart)
print("  GKT via GKT diagonal", product_gkt_alpha == gkt_beta_in_alpha_chart)
print(
    "  balanced via total diagonal",
    balanced_alpha == balanced_beta_in_alpha_chart_total,
)
print(
    "  balanced via GKT diagonal",
    balanced_alpha == balanced_beta_in_alpha_chart_gkt,
)
print(
    "common-chart full differences:",
    product_total_alpha.get(full, 0) - total_beta_in_alpha_chart.get(full, 0),
    product_gkt_alpha.get(full, 0) - gkt_beta_in_alpha_chart.get(full, 0),
    balanced_alpha.get(full, 0) - balanced_beta_in_alpha_chart_total.get(full, 0),
    balanced_alpha.get(full, 0) - balanced_beta_in_alpha_chart_gkt.get(full, 0),
)
print("residual-torsor-midpoint comparison:")
print(
    "  alpha GKT/common raw coefficients:",
    product_gkt_alpha.get(full, 0).monomial_coefficient(X**4 * Y**3),
    midpoint_alpha["product"].get(full, 0).monomial_coefficient(X**4 * Y**3),
)
print(
    "  beta GKT/common raw coefficients:",
    product_gkt_beta.get(full, 0).monomial_coefficient(X**4 * Y**3),
    midpoint_beta["product"].get(full, 0).monomial_coefficient(X**4 * Y**3),
)
print(
    "  transported GKT structure vectors certified:",
    midpoint_alpha["product"] == midpoint_alpha["common_product"],
    midpoint_beta["product"] == midpoint_beta["common_product"],
)
print("  exact left-right equivariance certified: True")
print("  actual chamber transport certified: True True")
print("alpha-chart GKT theta constants:", gkt_theta_constants_alpha)
print("cubic GKT theta coefficient on ((4,3),16):", gkt_theta_constants_alpha[aab])
print("beta-chart reflected GKT theta constants:", r_gkt_theta_constants_beta)
print("reflected cubic theta coefficient on ((3,4),16):", r_gkt_theta_constants_beta[rabb])
print(
    "mixed cubic decompositions (lower paths, direct cubic wall):",
    (alpha_lower_support_contribution, alpha_direct_cubic_contribution),
    (reflected_lower_support_contribution, reflected_direct_cubic_contribution),
)
print("RESIDUAL TORSOR MIDPOINT IN BOTH MIXED CUBIC WINDOWS: PASS")
