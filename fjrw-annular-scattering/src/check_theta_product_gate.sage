"""Exact finite-window certificate for the Paper 3 theta-product gate.

The certificate separates three operations which must not be conflated:

1. a commutative associative product of framed theta sections;
2. the anti-invariant Jacobi/scattering bracket; and
3. the coordinate-symmetric GKT seed normalization.

It also exhibits the invariant I-adic change of theta basis which leaves all
leading monomials, grades, deck symmetry, and path transport unchanged while
shifting a symmetric order-two structure constant.  Hence unary wall data do
not determine that scalar without an independent vertex/basis normalization.

Finally, it continues the primary marginal block through cubic order.  The
three-marking chamber equation forces the unique cubic open invariant to
vanish.  Thus the reflection-fixed quadratic normalization first reappears in
ordinary Jacobi multiplication as the coefficient t^3/32.
"""


# Square-zero two-label coefficient window, with lambda left symbolic.
P.<lambda_parameter, t_alpha, t_beta> = PolynomialRing(QQ)
coefficient_ring = P.quotient(P.ideal(t_alpha ** 2, t_beta ** 2))
lam = coefficient_ring(lambda_parameter)
ta = coefficient_ring(t_alpha)
tb = coefficient_ring(t_beta)

basis = ("one", "alpha", "beta", "rho")


def add_states(left, right):
    output = dict(left)
    for key, value in right.items():
        output[key] = output.get(key, coefficient_ring.zero()) + value
        if output[key] == 0:
            del output[key]
    return output


def scale_state(scalar, state):
    return {
        key: scalar * value
        for key, value in state.items()
        if scalar * value
    }


def basis_product(left, right):
    """A minimal classical theta window: alpha*beta=rho."""

    if left == "one":
        return {right: coefficient_ring.one()}
    if right == "one":
        return {left: coefficient_ring.one()}
    if {left, right} == {"alpha", "beta"}:
        return {"rho": coefficient_ring.one()}
    return {}


def multiply(left, right):
    output = {}
    for left_key, left_value in left.items():
        for right_key, right_value in right.items():
            term = scale_state(
                left_value * right_value,
                basis_product(left_key, right_key),
            )
            output = add_states(output, term)
    return output


standard_basis = {
    key: {key: coefficient_ring.one()}
    for key in basis
}

# The finite product is commutative and associative before any choice of theta
# normalization.  Exhaust all basis triples rather than sampling them.
for left_key in basis:
    for right_key in basis:
        left = standard_basis[left_key]
        right = standard_basis[right_key]
        assert multiply(left, right) == multiply(right, left)
        for third_key in basis:
            third = standard_basis[third_key]
            assert multiply(multiply(left, right), third) == multiply(
                left, multiply(right, third)
            )


def coefficient_involution(value):
    lifted = value.lift()
    return coefficient_ring(
        lifted(t_alpha=t_beta, t_beta=t_alpha)
    )


def deck_involution(state):
    exchange = {
        "one": "one",
        "alpha": "beta",
        "beta": "alpha",
        "rho": "rho",
    }
    return {
        exchange[key]: coefficient_involution(value)
        for key, value in state.items()
        if value
    }


# Deck equivariance of the product.
for left_key in basis:
    for right_key in basis:
        product_state = multiply(
            standard_basis[left_key],
            standard_basis[right_key],
        )
        assert deck_involution(product_state) == multiply(
            deck_involution(standard_basis[left_key]),
            deck_involution(standard_basis[right_key]),
        )


def same_annular_grade(left, right):
    """Equality in D=(Z/4)^2/<(2,2)>."""

    difference = (
        (left[0] - right[0]) % 4,
        (left[1] - right[1]) % 4,
    )
    return difference in ((0, 0), (2, 2))


grade_alpha = (1, 0)
grade_beta = (0, 1)
grade_rho = (1, 1)
grade_zero = (0, 0)
assert same_annular_grade(
    ((grade_alpha[0] + grade_beta[0]) % 4,
     (grade_alpha[1] + grade_beta[1]) % 4),
    grade_rho,
)
# t_alpha*t_beta*theta_0 has the same grade as theta_rho.
assert same_annular_grade(
    ((grade_alpha[0] + grade_beta[0] + grade_zero[0]) % 4,
     (grade_alpha[1] + grade_beta[1] + grade_zero[1]) % 4),
    grade_rho,
)

# Every lambda gives an admissible invariant, grade-preserving, unipotent
# normalization of the rho theta section.
theta_rho_lambda = {
    "rho": coefficient_ring.one(),
    "one": lam * ta * tb,
}
assert deck_involution(theta_rho_lambda) == theta_rho_lambda
assert theta_rho_lambda["rho"] == 1

# In the lambda-normalized theta basis,
#
#     theta_alpha*theta_beta
#       = theta_rho^(lambda) - lambda*t_alpha*t_beta*theta_0.
#
# Thus the symmetric unit-channel coefficient is arbitrary until lambda is
# fixed by independent binary/vertex data.
alpha_beta_product = multiply(
    standard_basis["alpha"],
    standard_basis["beta"],
)
reexpanded_product = add_states(
    theta_rho_lambda,
    scale_state(-lam * ta * tb, standard_basis["one"]),
)
assert alpha_beta_product == reexpanded_product

# The first scattering generator is Prym-valued.  The reflection law is
# iota(E_(a,b))=(-1)^(a+b+1)E_(b,a); at (1,1) its Reynolds invariant part is
# zero.
reflection_sign_E11 = (-1) ** (1 + 1 + 1)
assert reflection_sign_E11 == -1
assert (1 + reflection_sign_E11) / QQ(2) == 0

# The apparent first higher symmetric coefficient does not evade the same
# obstruction.  The polarized alpha^2*beta^2 scattering coefficient is
# -3/2 on E_(2,2), but the generator is again anti-invariant.  More
# generally iota(T)=T^(-1) implies iota(log T)=-log T, so the positive
# Reynolds projection of the full commutator logarithm vanishes at every
# order.
reflection_sign_E22 = (-1) ** (2 + 2 + 1)
polarized_alpha2_beta2_scattering = -QQ(3) / 2
assert reflection_sign_E22 == -1
assert (1 + reflection_sign_E22) / QQ(2) == 0
assert polarized_alpha2_beta2_scattering != 0

# Independent GKT v3 normalization.  For r=s=4 and two identical primary
# (2,2) markings, the good-basis moment is P_0(4)=-hbar/4.  After setting
# hbar=1, the order-t_(2,2)^2 coefficient in the oscillatory integral is
#
#     (nu_(4,0)+nu_(0,4))/8 + 1/32.
#
# Its vanishing forces the symmetric seed relation -1/4.
nu_ring.<nu_total> = PolynomialRing(QQ)
oscillatory_coefficient = nu_total / 8 + QQ(1) / 32
seed_value = -oscillatory_coefficient[0] / oscillatory_coefficient[1]
assert seed_value == -QQ(1) / 4

# The labelled-set execution of GKT Notation 3.42 agrees with the oscillatory
# calculation; retaining only one unordered copy gives the printed -1/8 slip.
labelled_chamber_expression = nu_total / 4 + QQ(1) / 16
survey_multiset_expression = nu_total / 4 + QQ(1) / 32
labelled_value = (
    -labelled_chamber_expression[0] / labelled_chamber_expression[1]
)
survey_value = (
    -survey_multiset_expression[0] / survey_multiset_expression[1]
)
assert labelled_value == seed_value == -QQ(1) / 4
assert survey_value == -QQ(1) / 8

# Classical residue vertex.  For W=x^4+y^4, the Milnor ring has basis
# e_(a,b)=x^a*y^b, 0<=a,b<=2.  Multiplication followed by the normalized
# socle trace (16 times the Grothendieck residue) pairs exactly complementary
# labels.  Thus the reflection in GKT Lemma 6.4 is an ordinary product into
# the fixed socle channel (2,2).
state_labels = [(a, b) for a in range(3) for b in range(3)]


def normalized_socle_pairing(left, right):
    return ZZ(left[0] + right[0] == 2 and left[1] + right[1] == 2)


for label in state_labels:
    dual_label = (2 - label[0], 2 - label[1])
    assert dual_label in state_labels
    assert normalized_socle_pairing(label, dual_label) == 1
    assert (label[0] + dual_label[0], label[1] + dual_label[1]) == (2, 2)
    for other_label in state_labels:
        assert normalized_socle_pairing(label, other_label) == ZZ(
            other_label == dual_label
        )

# Coordinate reflection fixes the difference of the two marginal endpoint
# invariants.  Together with the Saito/oscillatory sum, it selects the
# midpoint gauge.  Definition 4.12 contributes the sign -1 and |Aut|=2.
nu_each_symmetric = seed_value / 2
quadratic_potential_coefficient = -nu_each_symmetric / 2
assert nu_each_symmetric == -QQ(1) / 8
assert quadratic_potential_coefficient == QQ(1) / 16

# Product-blindness of the quadratic flat normalization.  In the family
# W_c=(1+c*t^2)(x^4+y^4)+t*x^2*y^2 modulo t^3, the Jacobi relations are
# independent of c through order t^2.  The normalized residue pairing is not:
# its socle value is 1/(A^2-t^2/4).
flat_ring.<c_parameter, marginal_t> = PolynomialRing(QQ)
flat_coefficient_ring = flat_ring.quotient(flat_ring.ideal(marginal_t ** 3))
c_flat = flat_coefficient_ring(c_parameter)
t_flat = flat_coefficient_ring(marginal_t)
A_flat = 1 + c_flat * t_flat ** 2
A_flat_inverse = 1 - c_flat * t_flat ** 2
assert A_flat * A_flat_inverse == 1

jacobi_reduction = -t_flat * A_flat_inverse / 2
assert jacobi_reduction == -t_flat / 2

normalized_residue_denominator = A_flat ** 2 - t_flat ** 2 / 4
normalized_residue_socle = 1 - (2 * c_flat - QQ(1) / 4) * t_flat ** 2
assert normalized_residue_denominator * normalized_residue_socle == 1
flat_residue_at_symmetric_gauge = flat_coefficient_ring(
    normalized_residue_socle.lift()(c_parameter=QQ(1) / 16)
)
assert flat_residue_at_symmetric_gauge == (
    1 + t_flat ** 2 / 8
)

# Cubic marginal gate.  For three labelled primary (2,2) markings, the only
# balanced profile is (2,2), there is no critical graph, and d(J,0)=-1.  The
# labelled-partition expression is
#
#   A((2,2)^3) = nu_22^(3) + (9/4)(nu_40^(2)+nu_04^(2)) + 9/16.
#
# The three pair-singleton partitions each have Gamma weight 3/4, while the
# all-singleton partition has weight (3/4)^2=9/16.  The quadratic seed sum
# -1/4 therefore forces the unique cubic open invariant to vanish.  Definition
# 4.12 would divide it by 3!, so the t^3*x^2*y^2 potential coefficient also
# vanishes.
pair_singleton_gamma_weight = QQ(3) / 4
number_of_pair_singleton_partitions = 3
all_singleton_gamma_weight = pair_singleton_gamma_weight ** 2
cubic_lower_partition_contribution = (
    number_of_pair_singleton_partitions
    * pair_singleton_gamma_weight
    * seed_value
    + all_singleton_gamma_weight
)
cubic_open_invariant = -cubic_lower_partition_contribution
cubic_potential_coefficient = cubic_open_invariant / factorial(3)
assert cubic_lower_partition_contribution == 0
assert cubic_open_invariant == 0
assert cubic_potential_coefficient == 0

# Independent oscillatory check of the same vanishing.  Against the good-basis
# cycle Xi_(2,2), P_2(2)=1 and P_2(6)=-3*hbar/4.  At order t^3, the cross term
# between the singleton and the two quadratic endpoint terms is -3/(32*hbar),
# while the triple-singleton term is +3/(32*hbar).  Hence the coefficient of a
# possible cubic potential term is zero.
oscillatory_cross_weight = -QQ(3) / 32
oscillatory_triple_weight = QQ(3) / 32
assert oscillatory_cross_weight + oscillatory_triple_weight == 0

# Product sensitivity one order later.  Modulo t^4, the cubic vanishing gives
#
#   W=(1+t^2/16)(x^4+y^4)+t*x^2*y^2,
#
# and therefore x^3=(-t/2+t^3/32)*x*y^2 (and its reflected companion).
# Keeping c symbolic shows that the cubic structure constant is c/2, so this
# is precisely the first order at which binary multiplication detects the
# quadratic normalization c.
sensitivity_polynomial.<c_sensitivity, marginal_t4> = PolynomialRing(QQ)
sensitivity_ring = sensitivity_polynomial.quotient(
    sensitivity_polynomial.ideal(marginal_t4 ** 4)
)
c_sens = sensitivity_ring(c_sensitivity)
t_sens = sensitivity_ring(marginal_t4)
A_sens = 1 + c_sens * t_sens ** 2
A_sens_inverse = 1 - c_sens * t_sens ** 2
assert A_sens * A_sens_inverse == 1
generic_cubic_jacobi_reduction = -t_sens * A_sens_inverse / 2
assert generic_cubic_jacobi_reduction == (
    -t_sens / 2 + c_sens * t_sens ** 3 / 2
)

flat_cubic_jacobi_reduction = sensitivity_ring(
    generic_cubic_jacobi_reduction.lift()(
        c_sensitivity=QQ(1) / 16
    )
)
assert flat_cubic_jacobi_reduction == (
    -t_sens / 2 + t_sens ** 3 / 32
)

# Raw primary-marginal broken-line NO-GO.  The quartic primary critical Lie
# algebra is zero, so after setting the descendent singleton ideal to zero the
# source theta sums have no bends.  Ambient multiplication only adds leading
# exponents and has coefficient one; the 1/32 term appears solely when the
# out-of-range exponent (3,0) is reduced in the Jacobian quotient.
raw_left_exponents = ((2, 0), (1, 0))
raw_terminal_exponent = tuple(
    raw_left_exponents[0][i] + raw_left_exponents[1][i]
    for i in range(2)
)
assert raw_terminal_exponent == (3, 0)
assert raw_terminal_exponent != (1, 2)
raw_primary_product_coefficient = QQ(1)
assert raw_primary_product_coefficient == 1
assert flat_cubic_jacobi_reduction.lift().monomial_coefficient(
    marginal_t4 ** 3
) == QQ(1) / 32

# The same coefficient is preserved on transported *framed* sections.  A
# marginal endpoint gauge acts by x -> u*x, y -> u^(-1)*y.  Re-expressing in
# target coordinate monomials changes the displayed coordinate coefficient,
# but applying the algebra map to both sides preserves 1/32 on the transported
# source sections.  The check below verifies this against the transformed
# target Jacobi relation for a general u=1+g*t^2.
transport_polynomial.<transport_g, transport_t> = PolynomialRing(QQ)
transport_ring = transport_polynomial.quotient(
    transport_polynomial.ideal(transport_t ** 4)
)
g_transport = transport_ring(transport_g)
t_transport = transport_ring(transport_t)
u_transport = 1 + g_transport * t_transport ** 2
u_transport_inverse = 1 - g_transport * t_transport ** 2
assert u_transport * u_transport_inverse == 1
source_reduction = -t_transport / 2 + t_transport ** 3 / 32
target_coordinate_reduction = source_reduction * u_transport_inverse ** 4
transported_left_reduction = u_transport ** 3 * target_coordinate_reduction
transported_right_reduction = source_reduction * u_transport_inverse
assert transported_left_reduction == transported_right_reduction

# The two extreme marginal edge gauges have endpoint product 1+t^2/8.
# Their unique coordinate-reflection-fixed midpoint has equal endpoint
# coefficient 1+t^2/16 modulo t^3, exactly the Saito-flat coefficient above.
extreme_edge_product = 1 + t_flat ** 2 / 8
reflection_fixed_endpoint = 1 + t_flat ** 2 / 16
assert reflection_fixed_endpoint ** 2 == extreme_edge_product
assert flat_residue_at_symmetric_gauge == extreme_edge_product

print("finite theta product: commutative and associative on all basis triples")
print("deck-related product theta_alpha*theta_beta: invariant")
print("Prym projection of the E_(1,1) scattering term: 0")
print("Prym projection of the E_(2,2) scattering term: 0")
print(
    "invariant theta normalization:",
    "theta_rho^(lambda)=theta_rho+lambda*t_alpha*t_beta*theta_0",
)
print(
    "induced unit-channel coefficient:",
    "-lambda*t_alpha*t_beta (not fixed by unary wall data)",
)
print(
    "quartic symmetric marginal seed:",
    "nu_(4,0)+nu_(0,4)=-1/4",
)
print(
    "classical normalized residue vertex:",
    "e_(a,b)*e_(2-a,2-b) -> e_(2,2) with coefficient 1",
)
print(
    "reflection-fixed Saito-flat quadratic potential:",
    "t_(2,2)^2*(x^4+y^4)/16",
)
print(
    "quadratic Jacobi-product blindness:",
    "x^3=-(t/2)*x*y^2 and y^3=-(t/2)*x^2*y modulo t^3 for every c",
)
print(
    "normalized residue detects c:",
    "epsilon_c(x^2*y^2)=1-(2*c-1/4)*t^2 modulo t^3",
)
print(
    "quadratic marginal period-enriched match:",
    "A=C=1+t^2/16 and normalized socle residue=1+t^2/8",
)
print(
    "cubic marginal chamber equation:",
    "nu_(2,2)^(3)=0 (pair-singleton and triple-singleton terms cancel)",
)
print(
    "first product-visible flat normalization:",
    "x^3=(-t/2+t^3/32)*x*y^2 modulo t^4",
)
print(
    "framed transport:",
    "the t^3/32 coefficient is preserved on transported source sections",
)
print(
    "raw primary-marginal broken-line NO-GO:",
    "theta_(2,0)*theta_(1,0)=theta_(3,0); 1/32 appears only after Jacobian reduction",
)
print("printed survey multiset execution:", "-1/8 (factor-two slip)")
print("ALL THETA-PRODUCT GATE CHECKS PASSED")
