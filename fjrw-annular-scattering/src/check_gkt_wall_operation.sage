"""Exact finite-window certificate for the GKT/Jacobi operation comparison.

The window has two distinct labels

    t1 = u_{i,1},  twist(i) = (1,0),
    t2 = u_{j,1},  twist(j) = (0,1),

so t1^2=t2^2=0.  The script checks the GKT chamber equation, the
boundary-to-boundary slope factorization, and its separation from the local
inverse commutator of the two singleton factors.  All arithmetic is exact.
"""


from math import gcd


S.<x, y> = PolynomialRing(QQ, 2)


def add_state(left, right):
    """Add coefficient states indexed by the exponents of (t1,t2)."""

    output = dict(left)
    for support, coefficient in right.items():
        output[support] = output.get(support, S.zero()) + coefficient
        if output[support] == 0:
            del output[support]
    return output


def jacobi_derivative(poly, k):
    """The quartic GKT generator X_k applied to a polynomial."""

    a, b = k
    return S(
        x**a * y**b
        * ((b + 1) * x * poly.derivative(x) - (a + 1) * y * poly.derivative(y))
    )


def apply_generator(state, generator):
    """Apply c*t1^e1*t2^e2*X_k in the square-zero labelled window."""

    support, k, coefficient = generator
    output = {}
    for old_support, poly in state.items():
        new_support = tuple(old_support[i] + support[i] for i in range(2))
        if any(exponent > 1 for exponent in new_support):
            continue
        value = coefficient * jacobi_derivative(poly, k)
        if value:
            output[new_support] = output.get(new_support, S.zero()) + value
    return output


def flow(state, generator):
    """Apply exp(generator); its square vanishes in this coefficient window."""

    return add_state(state, apply_generator(state, generator))


def chronological_product(state, generators):
    """Apply the listed generators from first time to last time."""

    output = state
    for generator in generators:
        output = flow(output, generator)
    return output


def operator_product(state, factors):
    """Apply a displayed operator product, whose rightmost factor acts first."""

    return chronological_product(state, reversed(factors))


def negate(generator):
    support, k, coefficient = generator
    return (support, k, -coefficient)


def gamma_ratio(top_numerator, base_numerator, denominator):
    """Gamma(top/denominator)/Gamma(base/denominator) in the integral-shift case."""

    assert top_numerator >= base_numerator
    assert (top_numerator - base_numerator) % denominator == 0
    output = QQ.one()
    for step in range((top_numerator - base_numerator) // denominator):
        output *= QQ(base_numerator + step * denominator) / denominator
    return output


def shifted_ray(k):
    a, b = k
    divisor = gcd(a + 1, b + 1)
    return ((a + 1) // divisor, (b + 1) // divisor)


# ---------------------------------------------------------------------------
# The two-marking GKT block and its chamber equation.
# ---------------------------------------------------------------------------

r = s = 4
markings = ((1, 0, 1), (0, 1, 1))
A_twist = sum(marking[0] for marking in markings)
B_twist = sum(marking[1] for marking in markings)
rJ = A_twist % r
sJ = B_twist % s
mJ = r * s + sum(
    s * a + r * b + r * s * (d - 1) for a, b, d in markings
)
dJ = (s * rJ + r * sJ - mJ) // (r * s) - 1
N = (A_twist - rJ) // r + (B_twist - sJ) // s - len(markings) + 1 + sum(
    marking[2] for marking in markings
)

assert (rJ, sJ, mJ, dJ, N) == (1, 1, 24, -2, 1)
balanced_profiles = tuple((rJ + p * r, sJ + (N - p) * s) for p in range(N + 1))
assert balanced_profiles == ((1, 5), (5, 1))

# The h=1 weight of nu_(1,5), and the h=2 weight of the product of the
# two surviving singleton coefficients in the minimum normal form.
weight_15 = gamma_ratio(1 + 1, 1 + rJ, r) * gamma_ratio(1 + 5, 1 + sJ, s)
weight_19 = gamma_ratio(1 + 1, 1 + rJ, r) * gamma_ratio(1 + 9, 1 + sJ, s)
assert (weight_15, weight_19) == (QQ(1) / 2, QQ(3) / 4)

nu_14_min = -4
nu_05_min = -2
nu_50_max = -2
nu_41_max = -4
nu_15_min = -weight_19 * nu_14_min * nu_05_min / weight_15
nu_51_max = -weight_19 * nu_50_max * nu_41_max / weight_15
assert (nu_15_min, nu_51_max) == (-12, -12)
assert weight_15 * nu_15_min + weight_19 * nu_14_min * nu_05_min == 0
assert weight_15 * nu_51_max + weight_19 * nu_50_max * nu_41_max == 0

# GKT's potential coefficient has the sign (-1)^(|J|-1), hence the mixed
# invariant -12 occurs in W with coefficient +12.
W_min = {
    (0, 0): x**4 + y**4,
    (1, 0): -4 * x * y**4,
    (0, 1): -2 * y**5,
    (1, 1): 12 * x * y**5,
}
W_max = {
    (0, 0): x**4 + y**4,
    (1, 0): -2 * x**5,
    (0, 1): -4 * x**4 * y,
    (1, 1): 12 * x**5 * y,
}


# ---------------------------------------------------------------------------
# The canonical GKT boundary-to-boundary product.
# ---------------------------------------------------------------------------

A = ((1, 0), (1, 0), -QQ(1) / 2)
B = ((0, 1), (0, 1), -QQ(1) / 2)
M = ((1, 1), (1, 1), QQ(1) / 4)

assert jacobi_derivative(x**4 + y**4, (1, 0)) == 4 * x**5 - 8 * x * y**4
assert jacobi_derivative(x**4 + y**4, (0, 1)) == 8 * x**4 * y - 4 * y**5
assert jacobi_derivative(x**4 + y**4, (1, 1)) == 8 * x**5 * y - 8 * x * y**5

after_A = chronological_product(W_min, [A])
assert after_A[(1, 0)] == -2 * x**5
assert after_A[(1, 1)] == 2 * x * y**5

# Increasing shifted slope is (2,1), (1,1), (1,2), so the chronological
# product is A, M, B and the displayed operator product is exp(B)exp(M)exp(A).
assert tuple(shifted_ray(generator[1]) for generator in (A, M, B)) == (
    (2, 1),
    (1, 1),
    (1, 2),
)
assert chronological_product(W_min, [A, M, B]) == W_max

# The two endpoint rows determine the middle coefficient independently.
c_from_left_row = QQ(2) / 8
c_from_right_row = QQ(12 - 10) / 8
assert c_from_left_row == c_from_right_row == QQ(1) / 4


# ---------------------------------------------------------------------------
# Separation from the local joint-consistency commutator.
# ---------------------------------------------------------------------------

def bracket_coefficient(k, ell):
    """Coefficient of X_(k+ell) in [X_k,X_ell]."""

    shifted_k = vector(ZZ, (k[0] + 1, k[1] + 1))
    shifted_ell = vector(ZZ, (ell[0] + 1, ell[1] + 1))
    return -matrix(ZZ, [shifted_k, shifted_ell]).det()


assert bracket_coefficient((1, 0), (0, 1)) == -3
intrinsic_bracket = A[2] * B[2] * bracket_coefficient(A[1], B[1])
inverse_commutator_coefficient = -intrinsic_bracket
assert intrinsic_bracket == -QQ(3) / 4
assert inverse_commutator_coefficient == QQ(3) / 4
assert M[2] != inverse_commutator_coefficient
assert 3 * M[2] == inverse_commutator_coefficient

# Direct operator check.  C=e^A e^B e^-A e^-B and its inverse have no
# higher terms because every further bracket repeats a label.
coordinate_x = {(0, 0): x}
coordinate_y = {(0, 0): y}
C_inverse_factors = [B, A, negate(B), negate(A)]
commutator_inverse_generator = ((1, 1), (1, 1), inverse_commutator_coefficient)
residual_consistency_generator = (
    (1, 1),
    (1, 1),
    inverse_commutator_coefficient - M[2],
)
assert residual_consistency_generator[2] == QQ(1) / 2
for coordinate in (coordinate_x, coordinate_y):
    assert operator_product(coordinate, C_inverse_factors) == chronological_product(
        coordinate, [commutator_inverse_generator]
    )
    assert chronological_product(coordinate, [M, residual_consistency_generator]) == (
        chronological_product(coordinate, [commutator_inverse_generator])
    )


# ---------------------------------------------------------------------------
# The marginal extension of the shifted Lie embedding.
# ---------------------------------------------------------------------------

# X_(0,0) maps to the shifted Hamiltonian with exponent kappa=(1,1).
# Its bracket with X_(a,b) is (a-b)X_(a,b), exactly the shifted determinant.
kappa = vector(ZZ, (1, 1))
for a in range(5):
    for b in range(5):
        k = vector(ZZ, (a, b))
        source = a - b
        target = -matrix(ZZ, [kappa, k + kappa]).det()
        assert source == target


print("quartic two-label block: d(J,d)=-2, balanced profiles (1,5),(5,1)")
print("extreme mixed invariants: nu_min(1,5)=nu_max(5,1)=-12")
print("canonical GKT product: exp(-1/2*t2*X_01) exp(1/4*t1*t2*X_11) exp(-1/2*t1*X_10)")
print("local inverse commutator factor: exp(3/4*t1*t2*X_11)")
print("operation separation: 1/4 != 3/4")
print("exact square-zero relation: GKT mixed factor = (inverse commutator)^(1/3)")
print("decorated outgoing wall: 1/4 GKT factor + 1/2 residual consistency = 3/4 total")
print("marginal shifted-bracket extension: checked")
print("ALL GKT/JACOBI OPERATION CHECKS PASSED")
