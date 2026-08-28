"""Exact first wall-bearing raw theta/GKT comparison for the quartic model.

Work with labelled square-zero parameters t=t_(2,2,0) and
u=t_(1,0,1).  The alpha descendent wall carries

    exp(-(u/2) X_(1,0)).

The script verifies three versions of the same coefficient:

1. the labelled-partition GKT equations give nu_min=3 and nu_max=2;
2. the mixed potential coefficient therefore jumps from -3 to -2; and
3. the two raw one-bend pairs in theta_(1,1)^2 contribute +1 to X^3 Y^2.

It also checks the complete minimum-to-maximum potential transport and the
coordinate-reflected beta identity.  All computations are exact over QQ.
"""


S.<X, Y> = PolynomialRing(QQ, 2)


def gamma_ratio(top_numerator, base_numerator, denominator=4):
    """Gamma(top/4)/Gamma(base/4) when the arguments differ integrally."""

    assert top_numerator >= base_numerator
    assert (top_numerator - base_numerator) % denominator == 0
    output = QQ.one()
    for step in range((top_numerator - base_numerator) // denominator):
        output *= QQ(base_numerator + step * denominator) / denominator
    return output


def gamma_weight(exponent, residue):
    return gamma_ratio(exponent[0] + 1, residue[0] + 1) * gamma_ratio(
        exponent[1] + 1, residue[1] + 1
    )


def jacobi_derivative(poly, twist):
    """The quartic GKT generator X_twist applied to a polynomial."""

    a, b = twist
    return S(
        X**a
        * Y**b
        * ((b + 1) * X * poly.derivative(X) - (a + 1) * Y * poly.derivative(Y))
    )


def add_state(left, right):
    output = dict(left)
    for support, coefficient in right.items():
        output[support] = output.get(support, S.zero()) + coefficient
        if output[support] == 0:
            del output[support]
    return output


def multiply_states(left, right):
    """Multiply modulo t^2=u^2=0; supports record the (t,u)-degree."""

    output = {}
    for left_support, left_poly in left.items():
        for right_support, right_poly in right.items():
            support = tuple(left_support[i] + right_support[i] for i in range(2))
            if any(exponent > 1 for exponent in support):
                continue
            output[support] = output.get(support, S.zero()) + left_poly * right_poly
    return {support: S(poly) for support, poly in output.items() if poly}


def wall_flow(state, twist):
    """Apply exp(-(u/2) X_twist) in the square-zero quotient."""

    correction = {}
    for support, poly in state.items():
        shifted_support = (support[0], support[1] + 1)
        if shifted_support[1] > 1:
            continue
        value = -QQ(1) / 2 * jacobi_derivative(poly, twist)
        if value:
            correction[shifted_support] = correction.get(
                shifted_support, S.zero()
            ) + value
    return add_state(state, correction)


def solve_mixed_endpoint(residue, product_exponent, descendent_invariant):
    """Solve A=nu_mixed+w*nu_primary*nu_descendent=0."""

    primary_invariant = QQ.one()
    weight = gamma_weight(product_exponent, residue)
    mixed_invariant = -weight * primary_invariant * descendent_invariant
    assert mixed_invariant + weight * primary_invariant * descendent_invariant == 0
    return mixed_invariant, weight


# Alpha: primary (2,2,0), descendent (1,0,1), full boundary profile (3,2).
alpha_residue = (3, 2)
alpha_min, alpha_min_weight = solve_mixed_endpoint(
    alpha_residue, (3, 6), -QQ(4)
)
alpha_max, alpha_max_weight = solve_mixed_endpoint(
    alpha_residue, (7, 2), -QQ(2)
)
assert (alpha_min_weight, alpha_max_weight) == (QQ(3) / 4, QQ.one())
assert (alpha_min, alpha_max) == (QQ(3), QQ(2))

W_alpha_min = {
    (0, 0): X**4 + Y**4,
    (1, 0): X**2 * Y**2,
    (0, 1): -4 * X * Y**4,
    (1, 1): -alpha_min * X**3 * Y**2,
}
W_alpha_max = {
    (0, 0): X**4 + Y**4,
    (1, 0): X**2 * Y**2,
    (0, 1): -2 * X**5,
    (1, 1): -alpha_max * X**3 * Y**2,
}
assert wall_flow(W_alpha_min, (1, 0)) == W_alpha_max

# A single theta_(1,1) has bend coefficient 1/2.  In its square, either
# labelled factor may bend, so the raw terminal coefficient is 1.
theta_11 = {(0, 0): X * Y}
transported_theta_11 = wall_flow(theta_11, (1, 0))
assert transported_theta_11 == {
    (0, 0): X * Y,
    (0, 1): QQ(1) / 2 * X**2 * Y,
}
raw_pair_product = multiply_states(transported_theta_11, transported_theta_11)
assert raw_pair_product == {
    (0, 0): X**2 * Y**2,
    (0, 1): X**3 * Y**2,
}
assert raw_pair_product == wall_flow({(0, 0): X**2 * Y**2}, (1, 0))
assert (-alpha_max) - (-alpha_min) == QQ.one()
assert raw_pair_product[(0, 1)].monomial_coefficient(X**3 * Y**2) == QQ.one()

# Beta is the reflected calculation.  The endpoint order reverses under
# X<->Y, and the common coefficient is -1.
beta_residue = (2, 3)
beta_min, beta_min_weight = solve_mixed_endpoint(
    beta_residue, (2, 7), -QQ(2)
)
beta_max, beta_max_weight = solve_mixed_endpoint(
    beta_residue, (6, 3), -QQ(4)
)
assert (beta_min_weight, beta_max_weight) == (QQ.one(), QQ(3) / 4)
assert (beta_min, beta_max) == (QQ(2), QQ(3))

W_beta_min = {
    (0, 0): X**4 + Y**4,
    (1, 0): X**2 * Y**2,
    (0, 1): -2 * Y**5,
    (1, 1): -beta_min * X**2 * Y**3,
}
W_beta_max = {
    (0, 0): X**4 + Y**4,
    (1, 0): X**2 * Y**2,
    (0, 1): -4 * X**4 * Y,
    (1, 1): -beta_max * X**2 * Y**3,
}
assert wall_flow(W_beta_min, (0, 1)) == W_beta_max

beta_theta_product = multiply_states(
    wall_flow(theta_11, (0, 1)), wall_flow(theta_11, (0, 1))
)
assert beta_theta_product == {
    (0, 0): X**2 * Y**2,
    (0, 1): -X**2 * Y**3,
}
assert (-beta_max) - (-beta_min) == -QQ.one()

print("alpha mixed invariants (min,max):", (alpha_min, alpha_max))
print("alpha raw/GKT potential jump:", QQ.one())
print("beta mixed invariants (min,max):", (beta_min, beta_max))
print("beta raw/GKT potential jump:", -QQ.one())
print("first wall-bearing raw theta/GKT comparison: PASS")
