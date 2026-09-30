"""Exact finite-order checks for the scattering of the two initial
Hamiltonians (Section 3.2) and for the consistency of the diagram D at the
joint O (Section 3.3).

Statements checked: Proposition 3.5 (the ordered factorization, its support
in the cone spanned by the shifted exponents (2,1) and (1,2), the symmetry
c_(b,a) = c_(a,b), and the coefficients of Table 1) and the consistency at the
joint used in the proof of Theorem 3.7, by a direct loop test with the
coorientations of Section 3.3, which are those of the table in Section 7.

Conventions.  Write q = kappa + (a,b) with kappa = (1,1).  After absorbing the
deformation variables into u = t_1 z_1 and v = t_2 z_2, the Hamiltonian
E_(a,b) acts by

    u^a v^b ((b+1) u d/du - (a+1) v d/dv),

and A = -E_(1,0)/2, B = -E_(0,1)/2.  Automorphisms act on functions of (u, v)
and the rightmost factor of a product acts first.  The script first computes
the factorization of e^A e^B into shifted-ray factors in which the factor of
least slope acts first (Proposition 3.5(3)); this is the path-ordered product
along a path that crosses every wall through the joint positively, such as
the path gamma_+ of Sections 3 and 7.  It then derives the coefficients
c_(a,b) of the elements C_r and checks the identity of Proposition 3.5(1),

    e^A e^(-B) e^(-A) e^B = prod_r exp(C_r),   C_r = sum c_(a,b) E_(a,b),

with the slopes of the rays r increasing from left to right.

Coorientation convention (Section 3.3).  Every wall through the joint is
cooriented so that crossing it in the direction of increasing shifted slope
is the positive crossing.  The negative half-chords carry e^A and e^B, the
positive half-chords carry e^A exp(-C_r(2,1)) and e^B exp(-C_r(1,2)), and the
outgoing wall of a ray r with 1/2 < slope(r) < 2 carries exp(-C_r); the first
outgoing wall carries exp(-(3/4) E_(1,1)).

Run with Sage from this directory: sage check_singleton_scattering.sage
(about one minute at the default degree 10).  The environment variable
SINGLETON_MAX_DEGREE changes the degree; with SINGLETON_FAST_START=1 and a
degree of at least 13 the reference table below is used as a starting point.
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
    """Operator product: (first*second)(f) = first(second(f)); second acts first."""

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


def shifted_ray(a, b):
    divisor = gcd(a + 1, b + 1)
    return ((a + 1) // divisor, (b + 1) // divisor)


def ray_slope(ray):
    return QQ(ray[1]) / ray[0]


def ray_groups(coefficients):
    groups = defaultdict(dict)
    for degree, coefficient in coefficients.items():
        if coefficient:
            groups[shifted_ray(*degree)][degree] = coefficient
    return groups


def ordered_factor_product(coefficients, least_slope_first):
    """Product of the ray factors exp(sum c*E_(a,b)).

    least_slope_first=True:  F_max * ... * F_min (the least slope acts first);
    least_slope_first=False: F_min * ... * F_max (slopes increase from left to
    right, so the greatest slope acts first).
    """

    groups = ray_groups(coefficients)
    rays = sorted(groups, key=ray_slope)
    if least_slope_first:
        rays = rays[::-1]
    return product([exponential_map(groups[ray]) for ray in rays])


def in_shifted_cone(a, b):
    return 2 * (a + 1) - (b + 1) >= 0 and 2 * (b + 1) - (a + 1) >= 0


# Reference values c_(a,b) of Proposition 3.5 through total degree 12 (a >= b
# listed; the table is symmetric).  The default run recomputes every entry.
REFERENCE_COEFFICIENTS_THROUGH_12 = {}
for (a, b), value in {
    (1, 1): QQ(3) / 4,
    (2, 1): QQ(3) / 8,
    (3, 1): QQ(1) / 16,
    (2, 2): QQ(3) / 8,
    (3, 2): QQ(3) / 8,
    (4, 2): QQ(3) / 32,
    (3, 3): QQ(5) / 16,
    (5, 2): QQ(3) / 128,
    (4, 3): QQ(7) / 16,
    (5, 3): QQ(3) / 16,
    (4, 4): QQ(21) / 64,
    (6, 3): QQ(7) / 128,
    (5, 4): QQ(9) / 16,
    (7, 3): QQ(13) / 1024,
    (6, 4): QQ(19) / 64,
    (5, 5): QQ(63) / 160,
    (7, 4): QQ(3) / 32,
    (6, 5): QQ(99) / 128,
    (8, 4): QQ(147) / 4096,
    (7, 5): QQ(135) / 256,
    (6, 6): QQ(33) / 64,
}.items():
    REFERENCE_COEFFICIENTS_THROUGH_12[(a, b)] = value
    REFERENCE_COEFFICIENTS_THROUGH_12[(b, a)] = value


singleton_first = {(1, 0): -QQ(1) / 2}
singleton_second = {(0, 1): -QQ(1) / 2}
exp_A = exponential_map(singleton_first)
exp_B = exponential_map(singleton_second)
exp_minus_A = inverse_exponential(singleton_first)
exp_minus_B = inverse_exponential(singleton_second)

# e^A e^B, the product along the path below the joint (gamma_- of Section 7).
lower_word = product([exp_A, exp_B])
# The word of Proposition 3.5 and its inverse.
target = product([exp_A, exp_minus_B, exp_minus_A, exp_B])
target_inverse = product([exp_minus_B, exp_A, exp_B, exp_minus_A])
assert multiply(target, target_inverse) == IDENTITY
assert multiply(target_inverse, target) == IDENTITY

# The coordinate swap (u,v) -> (v,u) conjugates E_(a,b) to -E_(b,a); hence it
# sends A to -B, B to -A, and the target word to its inverse.  With
# uniqueness of the factorization this gives c_(b,a) = c_(a,b) at all orders.
swap = (v, u)
assert product([swap, exp_A, swap]) == exp_minus_B
assert product([swap, exp_B, swap]) == exp_minus_A
assert product([swap, target, swap]) == target_inverse


# Exact rational forms of the singleton flows after rescaling x=u/2, y=v/2.
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
exact_swap = (y, x)

assert exact_multiply(exact_first, exact_first_inverse) == exact_identity
assert exact_multiply(exact_second, exact_second_inverse) == exact_identity
assert exact_multiply(exact_multiply(exact_swap, exact_first), exact_swap) == exact_second_inverse

exact_target = exact_identity
for factor in (exact_first, exact_second_inverse, exact_first_inverse, exact_second):
    exact_target = exact_multiply(exact_target, factor)
exact_jacobian = matrix(
    L,
    [
        [exact_target[0].derivative(x), exact_target[0].derivative(y)],
        [exact_target[1].derivative(x), exact_target[1].derivative(y)],
    ],
).determinant()
assert exact_jacobian == 1


def truncated_expansion(rational_function):
    """Expand a rational function of (x,y), regular at 0, in u=2x, v=2y."""

    numerator = R(S(rational_function.numerator())(x=u / 2, y=v / 2))
    denominator = R(S(rational_function.denominator())(x=u / 2, y=v / 2))
    constant = denominator.constant_coefficient()
    assert constant != 0
    h = truncate(1 - denominator / constant)
    inverse = R.one()
    power = R.one()
    for _ in range(MAX_POLYNOMIAL_DEGREE):
        power = truncate(power * h)
        inverse += power
    return truncate(numerator * inverse / constant)


# The exact composite agrees with the truncated word to the working order.
assert truncate(2 * truncated_expansion(exact_target[0])) == target[0]
assert truncate(2 * truncated_expansion(exact_target[1])) == target[1]


# Factorization of e^A e^B with the least slope acting first.
use_fast_start = os.environ.get("SINGLETON_FAST_START", "0") == "1"
if use_fast_start:
    assert MAX_LIE_DEGREE >= 13
    increasing_coefficients = {(1, 0): -QQ(1) / 2, (0, 1): -QQ(1) / 2}
    for degree, value in REFERENCE_COEFFICIENTS_THROUGH_12.items():
        increasing_coefficients[degree] = -value
    first_degree_to_solve = 13
    checkpoint = ordered_factor_product(increasing_coefficients, True)
    for degree in range(2, 14):
        assert homogeneous_part(lower_word[0] - checkpoint[0], degree) == 0
        assert homogeneous_part(lower_word[1] - checkpoint[1], degree) == 0
else:
    increasing_coefficients = {}
    first_degree_to_solve = 1

for lie_degree in range(first_degree_to_solve, MAX_LIE_DEGREE + 1):
    current = ordered_factor_product(increasing_coefficients, True)
    difference_u = homogeneous_part(lower_word[0] - current[0], lie_degree + 1)
    difference_v = homogeneous_part(lower_word[1] - current[1], lie_degree + 1)

    for a in range(lie_degree + 1):
        b = lie_degree - a
        coefficient = difference_u.monomial_coefficient(u ** (a + 1) * v ** b) / (b + 1)
        coefficient_from_v = (
            -difference_v.monomial_coefficient(u ** a * v ** (b + 1)) / (a + 1)
        )
        assert coefficient == coefficient_from_v
        if coefficient:
            increasing_coefficients[(a, b)] = coefficient

    current = ordered_factor_product(increasing_coefficients, True)
    assert homogeneous_part(lower_word[0] - current[0], lie_degree + 1) == 0
    assert homogeneous_part(lower_word[1] - current[1], lie_degree + 1) == 0

assert ordered_factor_product(increasing_coefficients, True) == lower_word

# The chords carry A and B; every other increasing-slope coefficient is the
# negative of an attached coefficient c_(a,b).
assert increasing_coefficients[(1, 0)] == -QQ(1) / 2
assert increasing_coefficients[(0, 1)] == -QQ(1) / 2
factor_coefficients = {
    degree: -value
    for degree, value in increasing_coefficients.items()
    if degree not in ((1, 0), (0, 1))
}

# Proposition 3.5: e^A e^(-B) e^(-A) e^B = prod_r exp(C_r), slopes increasing
# from left to right.
assert ordered_factor_product(factor_coefficients, False) == target

assert factor_coefficients[(1, 1)] == QQ(3) / 4
for degree, expected_coefficient in REFERENCE_COEFFICIENTS_THROUGH_12.items():
    if sum(degree) <= MAX_LIE_DEGREE:
        assert factor_coefficients[degree] == expected_coefficient

# Total degree four is the first degree with a nonzero coefficient on an
# incoming ray; these factors are placed on the positive half-chords.
if MAX_LIE_DEGREE >= 4:
    assert factor_coefficients[(3, 1)] == QQ(1) / 16
    assert factor_coefficients[(1, 3)] == QQ(1) / 16
    assert shifted_ray(3, 1) == shifted_ray(1, 0) == (2, 1)
    assert shifted_ray(1, 3) == shifted_ray(0, 1) == (1, 2)

# Support, symmetry and positivity.  Every bidegree (a,b) with a, b >= 1 and
# shifted exponent in cone((2,1),(1,2)) occurs, and no other.
for (a, b), coefficient in factor_coefficients.items():
    assert coefficient > 0
    assert a >= 1 and b >= 1
    assert in_shifted_cone(a, b)
    assert factor_coefficients[(b, a)] == coefficient
for total_degree in range(2, MAX_LIE_DEGREE + 1):
    for a in range(total_degree + 1):
        b = total_degree - a
        expected = a >= 1 and b >= 1 and in_shifted_cone(a, b)
        assert ((a, b) in factor_coefficients) == expected


# Direct loop test at the joint O with the coorientations of Section 3.3,
# which are those of the table in Section 7.  Here L = [[-2,0],[1,1]] and
# (w_1,w_2)^perp = (w_2,-w_1).  Every wall through O in the direction +-L q is
# cooriented by (L q)^perp, so that crossing it in the direction of increasing
# shifted slope is the positive crossing.  The walls and their factors are
#   the negative half-chord of L(2,1) = (-4,3), normal (3,4): e^A;
#   the negative half-chord of L(1,2) = (-2,3), normal (3,2): e^B;
#   the positive half-chord of L(2,1): e^A exp(-C_r) for r the ray of (2,1);
#   the positive half-chord of L(1,2): e^B exp(-C_r) for r the ray of (1,2);
#   the outgoing wall R_{>=0} L q of a ray r with 1/2 < slope(r) < 2: exp(-C_r).
# The path gamma_+ runs along s_2 = 1/10 and gamma_- along s_2 = -1/10, from
# s_1 = -1 to s_1 = 1, joined to (-1,0) and (1,0) by vertical segments; the
# crossing points, their order and the crossing signs are computed from the
# supports.
def cofactor(q):
    return (-2 * q[0], q[0] + q[1])


def perpendicular(w):
    return (w[1], -w[0])


def pairing(first, second):
    return first[0] * second[0] + first[1] * second[1]


def negated(terms):
    return {degree: -value for degree, value in terms.items()}


factor_groups = ray_groups(factor_coefficients)
incoming_rays = ((2, 1), (1, 2))

# Walls through O: (direction of the half-line from O, normal, terms of the
# logarithm of the attached factor, slope of the shifted ray).
walls = []
for incoming_ray, singleton_terms in zip(incoming_rays, (singleton_first, singleton_second)):
    direction = cofactor(incoming_ray)
    normal = perpendicular(direction)
    walls.append(
        ((-direction[0], -direction[1]), normal, dict(singleton_terms), ray_slope(incoming_ray))
    )
    positive_terms = dict(singleton_terms)
    positive_terms.update(negated(factor_groups.get(incoming_ray, {})))
    walls.append((direction, normal, positive_terms, ray_slope(incoming_ray)))
for ray, terms in factor_groups.items():
    if ray in incoming_rays:
        continue
    assert QQ(1) / 2 < ray_slope(ray) < 2
    direction = cofactor(ray)
    walls.append((direction, perpendicular(direction), negated(terms), ray_slope(ray)))

# The normals are (3,4), (3,2) on the chords and proportional to (1,1) on the
# diagonal wall; the first outgoing wall carries exp(-(3/4) E_(1,1)).
assert perpendicular(cofactor((2, 1))) == (3, 4)
assert perpendicular(cofactor((1, 2))) == (3, 2)
diagonal_normal = perpendicular(cofactor((1, 1)))
assert diagonal_normal[0] == diagonal_normal[1] > 0
assert negated(factor_groups[(1, 1)])[(1, 1)] == -QQ(3) / 4

height = QQ(1) / 10
velocity = (1, 0)
crossings = {height: [], -height: []}
for direction, normal, terms, slope in walls:
    assert direction[0] != 0 and direction[1] != 0
    # The vertical segments s_1 = +-1, |s_2| <= height meet no support.
    assert abs(QQ(direction[1]) / direction[0]) > height
    level = height if direction[1] > 0 else -height
    abscissa = level / direction[1] * direction[0]
    assert -1 < abscissa < 1
    sign = 1 if pairing(velocity, normal) > 0 else -1
    crossings[level].append((abscissa, slope, sign, terms))


def path_word(path_crossings):
    """Path-ordered product: the first crossing acts first."""

    ordered = sorted(path_crossings, key=lambda c: c[0])
    factors = [
        exponential_map({degree: sign * value for degree, value in terms.items()})
        for abscissa, slope, sign, terms in ordered
    ]
    return product(factors[::-1])


for level in (height, -height):
    abscissae = [c[0] for c in crossings[level]]
    assert len(set(abscissae)) == len(abscissae)
    # Both paths cross every wall positively.
    assert all(c[2] == 1 for c in crossings[level])

# gamma_- meets the negative half-chords of (1,2) and then of (2,1).
assert [c[1] for c in sorted(crossings[-height], key=lambda c: c[0])] == [2, QQ(1) / 2]
# gamma_+ meets the positive (2,1)-half-chord, the outgoing walls in
# increasing slope, and the positive (1,2)-half-chord.
slopes_plus = [c[1] for c in sorted(crossings[height], key=lambda c: c[0])]
assert slopes_plus == sorted(slopes_plus)
assert slopes_plus[0] == QQ(1) / 2 and slopes_plus[-1] == 2
assert len(slopes_plus) == len(set(slopes_plus))

theta_plus = path_word(crossings[height])
theta_minus = path_word(crossings[-height])
assert theta_minus == lower_word
assert theta_plus == theta_minus


coefficients_by_degree = defaultdict(list)
coefficients_by_ray = defaultdict(list)
for degree, coefficient in factor_coefficients.items():
    coefficients_by_degree[sum(degree)].append((degree, coefficient))
    coefficients_by_ray[shifted_ray(*degree)].append((degree, coefficient))

print("maximum Lie degree checked:", MAX_LIE_DEGREE)
print("factored word: e^A e^(-B) e^(-A) e^B (Proposition 3.5), rightmost factor acting first")
print("exact rescaled singleton flows:", exact_first, exact_second)
print("exact composite: Jacobian determinant 1; expansion agrees with the truncated word")
print("swap symmetry: (u,v)->(v,u) inverts the word, so c_(b,a) = c_(a,b)")
print(
    "increasing-slope factorization of e^A e^B:",
    "chord coefficients -1/2, -1/2; all other coefficients -c_(a,b)",
)
print(
    "loop test at the joint (every wall crossed positively along gamma_+ and gamma_-;",
    "outgoing walls carry exp(-C_r)):",
    "Theta(gamma_+) = Theta(gamma_-) = e^A e^B through degree", MAX_LIE_DEGREE,
)
if MAX_LIE_DEGREE >= 4:
    print(
        "first factors on the incoming rays:",
        "c_(3,1) = c_(1,3) = 1/16 on the shifted rays (2,1) and (1,2)",
    )
print("coefficients c_(a,b) by total degree:")
for degree in sorted(coefficients_by_degree):
    print(degree, sorted(coefficients_by_degree[degree]))
print("shifted-ray factors:")
for ray in sorted(coefficients_by_ray, key=ray_slope):
    print(ray, sorted(coefficients_by_ray[ray]))
print("ALL SINGLETON SCATTERING CHECKS PASSED")
