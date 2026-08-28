"""Exact finite-order certificate for the Paper 3 singleton Jacobi joint.

Write q=kappa+(a,b), with kappa=(1,1).  After absorbing the deformation
variables into u=t_1 z_1 and v=t_2 z_2, the Hamiltonian E_(a,b) acts by

    u^a v^b ((b+1) u d/du - (a+1) v d/dv).

The script forms the inverse commutator of the two coefficient -1/2
singleton factors and factors it uniquely by the shifted rays
R_{>=0}(a+1,b+1), using exact rational arithmetic.
"""

from collections import defaultdict
import os


MAX_LIE_DEGREE = ZZ(os.environ.get("SINGLETON_MAX_DEGREE", "10"))
assert MAX_LIE_DEGREE >= 2
MAX_POLYNOMIAL_DEGREE = MAX_LIE_DEGREE + 1

R.<u, v> = PolynomialRing(QQ, 2)


def truncate(poly, degree=MAX_POLYNOMIAL_DEGREE):
    """Discard monomials of total degree greater than ``degree``."""

    return R(
        {
            exponent: coefficient
            for exponent, coefficient in R(poly).dict().items()
            if sum(exponent) <= degree
        }
    )


def homogeneous_part(poly, degree):
    return R(
        {
            exponent: coefficient
            for exponent, coefficient in R(poly).dict().items()
            if sum(exponent) == degree
        }
    )


def jacobi_derivation(poly, terms):
    """Apply a finite sum of c*E_(a,b) to a truncated polynomial."""

    output = R.zero()
    for (a, b), coefficient in terms.items():
        if coefficient == 0:
            continue
        for (i, j), monomial_coefficient in R(poly).dict().items():
            structure_coefficient = (b + 1) * i - (a + 1) * j
            if structure_coefficient:
                output += (
                    coefficient
                    * monomial_coefficient
                    * structure_coefficient
                    * u ** (i + a)
                    * v ** (j + b)
                )
    return truncate(output)


def exponential_map(terms):
    """Return the exact truncated automorphism exp(sum c*E_(a,b))."""

    images = []
    for variable in (u, v):
        result = variable
        term = variable
        factorial = ZZ.one()
        step = 0
        while term:
            step += 1
            factorial *= step
            term = jacobi_derivation(term, terms)
            if term:
                result += term / factorial
        images.append(truncate(result))
    return tuple(images)


IDENTITY = (u, v)


def apply_map(automorphism, poly):
    return truncate(R(poly)(automorphism[0], automorphism[1]))


def multiply(first, second):
    """Operator product: (first*second)(f)=first(second(f))."""

    return (
        apply_map(first, second[0]),
        apply_map(first, second[1]),
    )


def product(factors):
    result = IDENTITY
    for factor in factors:
        result = multiply(result, factor)
    return result


def inverse_exponential(terms):
    return exponential_map({degree: -coefficient for degree, coefficient in terms.items()})


def operator_difference(automorphism, poly):
    """Apply ``automorphism - identity`` to a truncated polynomial."""

    return truncate(apply_map(automorphism, poly) - poly)


def fractional_power(automorphism, exponent):
    """Return a rational power of an I-adically unipotent automorphism."""

    images = []
    for variable in (u, v):
        result = variable
        difference_power = variable
        step = 0
        while difference_power:
            step += 1
            difference_power = operator_difference(
                automorphism,
                difference_power,
            )
            if difference_power:
                result += binomial(exponent, step) * difference_power
        images.append(truncate(result))
    return tuple(images)


def logarithm_map(automorphism):
    """Return the truncated logarithm of an I-adically unipotent map."""

    images = []
    for variable in (u, v):
        result = R.zero()
        difference_power = variable
        step = 0
        while difference_power:
            step += 1
            difference_power = operator_difference(
                automorphism,
                difference_power,
            )
            if difference_power:
                result += (-1) ** (step + 1) * difference_power / step
        images.append(truncate(result))
    return tuple(images)


def shifted_ray(a, b):
    divisor = gcd(a + 1, b + 1)
    return ((a + 1) // divisor, (b + 1) // divisor)


def ray_slope(ray):
    return QQ(ray[1]) / ray[0]


def ordered_factor_product(coefficients):
    ray_terms = defaultdict(dict)
    for degree, coefficient in coefficients.items():
        if coefficient:
            ray_terms[shifted_ray(*degree)][degree] = coefficient

    factors = []
    for ray in sorted(ray_terms, key=ray_slope):
        factors.append(exponential_map(ray_terms[ray]))
    return product(factors)


# Regression checkpoint from the exact degree-twelve run.  The default path
# recomputes these entries.  SINGLETON_FAST_START=1 may use the checkpoint to
# explore higher degrees without repeating every lower-order factorization.
CERTIFIED_COEFFICIENTS_THROUGH_12 = {
    (1, 1): QQ(3) / 4,
    (1, 2): -QQ(3) / 8,
    (2, 1): QQ(3) / 8,
    (1, 3): QQ(1) / 16,
    (2, 2): -QQ(3) / 8,
    (3, 1): QQ(1) / 16,
    (2, 3): -QQ(3) / 16,
    (3, 2): QQ(3) / 16,
    (2, 4): QQ(3) / 32,
    (3, 3): -QQ(25) / 64,
    (4, 2): QQ(3) / 32,
    (2, 5): -QQ(3) / 128,
    (3, 4): QQ(5) / 16,
    (4, 3): -QQ(5) / 16,
    (5, 2): QQ(3) / 128,
    (3, 5): QQ(3) / 256,
    (4, 4): QQ(9) / 64,
    (5, 3): QQ(3) / 256,
    (3, 6): -QQ(1) / 32,
    (4, 5): QQ(15) / 32,
    (5, 4): -QQ(15) / 32,
    (6, 3): QQ(1) / 32,
    (3, 7): QQ(13) / 1024,
    (4, 6): -QQ(69) / 256,
    (5, 5): QQ(357) / 640,
    (6, 4): -QQ(69) / 256,
    (7, 3): QQ(13) / 1024,
    (4, 7): QQ(9) / 256,
    (5, 6): -QQ(105) / 512,
    (6, 5): QQ(105) / 512,
    (7, 4): -QQ(9) / 256,
    (4, 8): QQ(39) / 4096,
    (5, 7): -QQ(75) / 256,
    (6, 6): QQ(75) / 512,
    (7, 5): -QQ(75) / 256,
    (8, 4): QQ(39) / 4096,
}


singleton_first = {(1, 0): -QQ(1) / 2}
singleton_second = {(0, 1): -QQ(1) / 2}

# C^{-1} for C=e^A e^B e^{-A} e^{-B}.
target = product(
    [
        exponential_map(singleton_second),
        exponential_map(singleton_first),
        inverse_exponential(singleton_second),
        inverse_exponential(singleton_first),
    ]
)

# Sanity check the inverse identity against the original commutator.
commutator = product(
    [
        exponential_map(singleton_first),
        exponential_map(singleton_second),
        inverse_exponential(singleton_first),
        inverse_exponential(singleton_second),
    ]
)
assert multiply(commutator, target) == IDENTITY
assert multiply(target, commutator) == IDENTITY

# The deck reflection exchanges the singleton inputs.  The resulting
# nonabelian norm defect is target=iota(AB)*(AB)^(-1).  Since iota(target)
# is target^(-1), the complete prounipotent group has a canonical half-cocycle.
# It gives a deck-invariant balanced norm, independent of the order AB versus
# BA.  Its logarithm has no mixed bidegree-(1,1) term: symmetrizing the ordered
# automorphism solves descent but cannot itself retain the first 3/4
# scattering interaction.
reflection = (-v, -u)


def conjugate_by_reflection(automorphism):
    return product([reflection, automorphism, reflection])


first_map = exponential_map(singleton_first)
second_map = exponential_map(singleton_second)
ordered_norm = product([first_map, second_map])
reflected_norm = conjugate_by_reflection(ordered_norm)
assert reflected_norm == product([second_map, first_map])
assert conjugate_by_reflection(target) == commutator

target_half = fractional_power(target, QQ(1) / 2)
target_inverse_half = fractional_power(target, -QQ(1) / 2)
assert multiply(target_half, target_half) == target
assert multiply(target_inverse_half, target_inverse_half) == commutator
assert conjugate_by_reflection(target_half) == target_inverse_half

balanced_norm = product([target_half, first_map, second_map])
balanced_norm_reversed = product(
    [target_inverse_half, second_map, first_map]
)
assert balanced_norm == balanced_norm_reversed
assert conjugate_by_reflection(balanced_norm) == balanced_norm

balanced_log = logarithm_map(balanced_norm)
assert balanced_log[0].monomial_coefficient(u ** 2 * v) == 0
assert balanced_log[1].monomial_coefficient(u * v ** 2) == 0

# The individual singleton flows and their inverse commutator have compact
# exact rational forms after rescaling x=u/2 and y=v/2.  These identities are
# independent of the truncation used below.
S.<x, y> = PolynomialRing(QQ, 2)
L = S.fraction_field()
x = L(x)
y = L(y)


def exact_multiply(first, second):
    return (
        second[0](x=first[0], y=first[1]),
        second[1](x=first[0], y=first[1]),
    )


exact_identity = (x, y)
exact_first = (x / (1 + x), y * (1 + x) ** 2)
exact_first_inverse = (x / (1 - x), y * (1 - x) ** 2)
exact_second = (x * (1 - y) ** 2, y / (1 - y))
exact_second_inverse = (x * (1 + y) ** 2, y / (1 + y))
exact_reflection = (-y, -x)


def conjugate_by_exact_reflection(transformation):
    return exact_multiply(
        exact_multiply(exact_reflection, transformation),
        exact_reflection,
    )


assert conjugate_by_exact_reflection(exact_first) == exact_second
assert conjugate_by_exact_reflection(exact_second) == exact_first
assert conjugate_by_exact_reflection(exact_first_inverse) == exact_second_inverse
assert conjugate_by_exact_reflection(exact_second_inverse) == exact_first_inverse

exact_target = exact_identity
for factor in (
    exact_second,
    exact_first,
    exact_second_inverse,
    exact_first_inverse,
):
    exact_target = exact_multiply(exact_target, factor)

compact_s = 1 + x * (1 - y) ** 2
compact_r = 1 - y + y * compact_s ** 2
compact_d = compact_s - x * compact_r ** 2
compact_target = (
    x * compact_r ** 2 / compact_d,
    y * compact_d ** 2 / compact_r,
)
assert exact_target == compact_target
compact_jacobian = matrix(
    L,
    [
        [compact_target[0].derivative(x), compact_target[0].derivative(y)],
        [compact_target[1].derivative(x), compact_target[1].derivative(y)],
    ],
).determinant()
assert compact_jacobian == 1


use_fast_start = os.environ.get("SINGLETON_FAST_START", "0") == "1"
if use_fast_start:
    assert MAX_LIE_DEGREE >= 12
    factor_coefficients = dict(CERTIFIED_COEFFICIENTS_THROUGH_12)
    first_degree_to_solve = 13
    checkpoint_product = ordered_factor_product(factor_coefficients)
    assert homogeneous_part(target[0] - checkpoint_product[0], 13) == 0
    assert homogeneous_part(target[1] - checkpoint_product[1], 13) == 0
else:
    factor_coefficients = {}
    first_degree_to_solve = 1

for lie_degree in range(first_degree_to_solve, MAX_LIE_DEGREE + 1):
    current = ordered_factor_product(factor_coefficients)
    difference_u = homogeneous_part(target[0] - current[0], lie_degree + 1)
    difference_v = homogeneous_part(target[1] - current[1], lie_degree + 1)

    for a in range(lie_degree + 1):
        b = lie_degree - a
        coefficient = difference_u.monomial_coefficient(u ** (a + 1) * v ** b) / (b + 1)
        coefficient_from_v = (
            -difference_v.monomial_coefficient(u ** a * v ** (b + 1)) / (a + 1)
        )
        assert coefficient == coefficient_from_v
        if coefficient:
            factor_coefficients[(a, b)] = coefficient

    current = ordered_factor_product(factor_coefficients)
    assert homogeneous_part(target[0] - current[0], lie_degree + 1) == 0
    assert homogeneous_part(target[1] - current[1], lie_degree + 1) == 0


factored_target = ordered_factor_product(factor_coefficients)
assert factored_target == target
assert factor_coefficients[(1, 1)] == QQ(3) / 4
for degree, expected_coefficient in CERTIFIED_COEFFICIENTS_THROUGH_12.items():
    if sum(degree) <= MAX_LIE_DEGREE:
        assert factor_coefficients[degree] == expected_coefficient

# Total degree four is the first time an outgoing completion factor returns
# to an initial shifted ray.  These are genuine nonzero resonances, so the
# geometric realization must merge them into the positive half of the
# corresponding incoming chord rather than draw a second coincident support.
if MAX_LIE_DEGREE >= 4:
    assert factor_coefficients[(3, 1)] == QQ(1) / 16
    assert factor_coefficients[(1, 3)] == QQ(1) / 16
    assert shifted_ray(3, 1) == shifted_ray(1, 0) == (2, 1)
    assert shifted_ray(1, 3) == shifted_ray(0, 1) == (1, 2)

# The commutator ideal contains both singleton variables, and Proposition 6.5
# confines every nonzero shifted direction to cone((2,1),(1,2)).
for (a, b), coefficient in factor_coefficients.items():
    assert coefficient
    assert a >= 1 and b >= 1
    assert 2 * (a + 1) - (b + 1) >= 0
    assert 2 * (b + 1) - (a + 1) >= 0

# Through the certified range, every shifted-cone bidegree occurs.  The
# reflection identity itself holds to all orders by conjugating with
# (u,v)->(-v,-u), which exchanges the two singleton inputs, reverses slopes,
# and sends E_(a,b) to (-1)^(a+b+1) E_(b,a).
for total_degree in range(2, MAX_LIE_DEGREE + 1):
    for a in range(1, total_degree):
        b = total_degree - a
        lies_in_shifted_cone = (
            2 * (a + 1) - (b + 1) >= 0
            and 2 * (b + 1) - (a + 1) >= 0
        )
        assert ((a, b) in factor_coefficients) == lies_in_shifted_cone
        if lies_in_shifted_cone:
            assert factor_coefficients[(b, a)] == (
                (-1) ** total_degree * factor_coefficients[(a, b)]
            )


# Boundary-adapted coordinates expose two exact one-variable shadows of the
# ordered factorization.  Put epsilon=1/x and w=x^2*y.  If
# delta=2*b-a, then the rescaled Hamiltonian 2^(a+b) E_(a,b) acts by
#
#   epsilon |-> -(b+1) epsilon^(delta+1) w^b,
#   w       |->  (delta+1) epsilon^delta w^(b+1).
#
# The delta=-1 terms form the extreme shifted ray.  Their total flow
# translates epsilon by a series H(w).  Removing that factor and restricting
# to epsilon=0 leaves the delta=0 layer, a one-variable formal diffeomorphism
# z(w).  The rational commutator gives the exact parametrization
#
#   w = z F(z)^3,        H = z(2-z)/F(z)^2,
#   F(z) = 1-z+4z^3-4z^4+z^5.

BOUNDARY_PRECISION = max(12, MAX_LIE_DEGREE + 3)
W.<ww> = PowerSeriesRing(QQ, default_prec=BOUNDARY_PRECISION)


def boundary_F(series):
    return 1 - series + 4 * series ** 3 - 4 * series ** 4 + series ** 5


boundary_z = ww
for _ in range(BOUNDARY_PRECISION + 1):
    boundary_z = (ww / boundary_F(boundary_z) ** 3).add_bigoh(BOUNDARY_PRECISION)

boundary_H = (
    boundary_z * (2 - boundary_z) / boundary_F(boundary_z) ** 2
).add_bigoh(BOUNDARY_PRECISION)
assert (
    boundary_z * boundary_F(boundary_z) ** 3 - ww
).add_bigoh(BOUNDARY_PRECISION) == 0
assert [boundary_H[degree] for degree in range(5)] == [0, 2, 9, 52, 300]

# Match H against every extreme-ray coefficient available at the requested
# two-variable truncation.  Here a=2*b+1 and the coefficient of w^b in H is
# (b+1) times the rescaled factor coefficient 2^(a+b)c_(a,b).
for b in range(1, (MAX_LIE_DEGREE - 1) // 3 + 1):
    a = 2 * b + 1
    scaled_coefficient = 2 ** (a + b) * factor_coefficients[(a, b)]
    assert boundary_H[b] == (b + 1) * scaled_coefficient

# Factor the delta=0 boundary diffeomorphism into its ordered monomial flows.
# For a=2*b, the time-one map on the boundary divisor is
#
#       phi_b(w)=w/(1-b*C_b*w^b)^(1/b),
#
# with C_b=2^(3b)c_(2b,b).  Increasing shifted slope makes the geometric
# composition z=phi_1 o phi_2 o ... .
boundary_layer_current = ww
boundary_layer_coefficients = []
for b in range(1, MAX_LIE_DEGREE // 3 + 1):
    layer_coefficient = (
        boundary_z[b + 1] - boundary_layer_current[b + 1]
    )
    expected_coefficient = 2 ** (3 * b) * factor_coefficients[(2 * b, b)]
    assert layer_coefficient == expected_coefficient
    boundary_layer_coefficients.append(layer_coefficient)
    layer_flow = (
        ww * (1 - b * layer_coefficient * ww ** b) ** (-QQ(1) / b)
    ).add_bigoh(BOUNDARY_PRECISION)
    boundary_layer_current = boundary_layer_current(layer_flow).add_bigoh(
        BOUNDARY_PRECISION
    )
    assert all(
        boundary_layer_current[degree] == boundary_z[degree]
        for degree in range(1, b + 2)
    )

# A finite delta=0 factorization would express z(w) by a finite tower of
# radicals.  The following exact Frobenius witnesses obstruct that possibility.
# For P(z)=zF(z)^3, the specialization P(z)-3 is squarefree and irreducible
# modulo 13 (cycle type 16), while modulo 83 its irreducible factor degrees are
# 13,1,1,1.  In the proof, the resulting 16- and 13-cycles, Jordan's theorem,
# and the odd parity of a 16-cycle force Galois group S_16.
ZQ.<q> = PolynomialRing(ZZ)
F_polynomial = 1 - q + 4 * q ** 3 - 4 * q ** 4 + q ** 5
P_polynomial = q * F_polynomial ** 3
specialized_polynomial = P_polynomial - 3
assert specialized_polynomial.degree() == 16
assert specialized_polynomial.discriminant() != 0

factor_degrees_by_prime = {}
for prime in (13, 83):
    finite_ring = PolynomialRing(GF(prime), "q")
    finite_polynomial = finite_ring(specialized_polynomial)
    assert gcd(finite_polynomial, finite_polynomial.derivative()) == 1
    factor_degrees_by_prime[prime] = sorted(
        [
            factor.degree()
            for factor, multiplicity in finite_polynomial.factor()
            for _ in range(multiplicity)
        ],
        reverse=True,
    )

assert factor_degrees_by_prime[13] == [16]
assert factor_degrees_by_prime[83] == [13, 1, 1, 1]


coefficients_by_degree = defaultdict(list)
coefficients_by_ray = defaultdict(list)
for degree, coefficient in factor_coefficients.items():
    coefficients_by_degree[sum(degree)].append((degree, coefficient))
    coefficients_by_ray[shifted_ray(*degree)].append((degree, coefficient))

print("maximum certified Lie degree:", MAX_LIE_DEGREE)
print("exact rescaled singleton flows:", exact_first, exact_second)
print(
    "compact inverse commutator:",
    "s=1+x*(1-y)^2, r=1-y+y*s^2, d=s-x*r^2;",
    "T(x)=x*r^2/d, T(y)=y*d^2/r, det(DT)=1",
)
print(
    "balanced norm gate:",
    "T^(1/2)*A*B=T^(-1/2)*B*A is deck invariant;",
    "mixed bidegree-(1,1) logarithmic term = 0",
)
print(
    "boundary algebraic shadow:",
    "w=z*F(z)^3, H=z*(2-z)/F(z)^2,",
    "F(z)=1-z+4*z^3-4*z^4+z^5",
)
print("extreme translation H(w):", boundary_H)
print("delta-zero layer coefficients:", boundary_layer_coefficients)
print("Galois witness factor degrees:", factor_degrees_by_prime)
if MAX_LIE_DEGREE >= 4:
    print(
        "first boundary-resonant completion factors:",
        "c_(3,1)=c_(1,3)=1/16 on singleton rays (2,1) and (1,2)",
    )
print("nonzero outgoing coefficients by degree:")
for degree in sorted(coefficients_by_degree):
    print(degree, sorted(coefficients_by_degree[degree]))
print("shifted-ray factors:")
for ray in sorted(coefficients_by_ray, key=ray_slope):
    print(ray, sorted(coefficients_by_ray[ray]))
