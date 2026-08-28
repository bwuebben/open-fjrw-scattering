"""Exact mixed primary--descendent Jacobian/primitive-form NO-GO.

Work over A=QQ[t,u]/(t^2,u^2).  The alpha endpoint potentials from
Proposition 6.31 are

    W_min = W_t - 4*u*X*Y^4 - 3*t*u*X^3*Y^2,
    W_max = W_t - 2*u*X^5     - 2*t*u*X^3*Y^2,

where W_t=X^4+Y^4+t*X^2*Y^2.  This script proves exactly that:

1. W_min and W_max are pullbacks of W_t by explicit right equivalences;
2. both right equivalences carry the endpoint form dX wedge dY to the same
   calibrated base form (1+u*X)dX wedge dY;
3. the alpha wall intertwines those calibrations and becomes the identity on
   the common calibrated Jacobian algebra; and
4. a divergence-free t*u vector field shifts the mixed potential coefficient
   by one while preserving the primitive form.

Consequently no invariant of the primitive-form Jacobian algebra can recover
the endpoint coefficients -3,-2 (equivalently the open invariants 3,2).
All arithmetic is exact over QQ.
"""


S.<X, Y> = PolynomialRing(QQ, 2)


def clean(state):
    """Remove zero terms from a square-zero (t,u)-graded polynomial state."""

    return {
        support: S(poly)
        for support, poly in state.items()
        if poly and support[0] <= 1 and support[1] <= 1
    }


def add_states(left, right):
    output = dict(left)
    for support, poly in right.items():
        output[support] = output.get(support, S.zero()) + poly
    return clean(output)


def scale_state(state, scalar):
    return clean({support: scalar * poly for support, poly in state.items()})


def multiply_states(left, right):
    output = {}
    for left_support, left_poly in left.items():
        for right_support, right_poly in right.items():
            support = (
                left_support[0] + right_support[0],
                left_support[1] + right_support[1],
            )
            if support[0] > 1 or support[1] > 1:
                continue
            output[support] = output.get(support, S.zero()) + left_poly * right_poly
    return clean(output)


def apply_vector_field(state, vector_field):
    """Apply a coefficient-graded vector field to a coefficient state."""

    output = {}
    for old_support, poly in state.items():
        for vector_support, (p_component, q_component) in vector_field.items():
            support = (
                old_support[0] + vector_support[0],
                old_support[1] + vector_support[1],
            )
            if support[0] > 1 or support[1] > 1:
                continue
            value = (
                p_component * poly.derivative(X)
                + q_component * poly.derivative(Y)
            )
            output[support] = output.get(support, S.zero()) + value
    return clean(output)


def flow(state, vector_field):
    """Apply exp(V); every vector field below is u-divisible, so V^2=0."""

    return add_states(state, apply_vector_field(state, vector_field))


def divergence(vector_field):
    return clean(
        {
            support: p_component.derivative(X) + q_component.derivative(Y)
            for support, (p_component, q_component) in vector_field.items()
        }
    )


def normalized_base_residue(state):
    """Normalized residue for W_t modulo t^2, independently in each u-degree.

    At t=0 the normalized trace extracts [X^2 Y^2].  Expanding
    1/(d_X W_t d_Y W_t) to first t-order adds

        -(t/2)([X^4]+[Y^4]).
    """

    output = {}
    for (t_degree, u_degree), poly in state.items():
        socle = poly.monomial_coefficient(X**2 * Y**2)
        if socle:
            support = (t_degree, u_degree)
            output[support] = output.get(support, QQ.zero()) + socle
        if t_degree == 0:
            correction = -QQ(1) / 2 * (
                poly.monomial_coefficient(X**4)
                + poly.monomial_coefficient(Y**4)
            )
            if correction:
                support = (1, u_degree)
                output[support] = output.get(support, QQ.zero()) + correction
    return {support: value for support, value in output.items() if value}


W_t = {
    (0, 0): X**4 + Y**4,
    (1, 0): X**2 * Y**2,
}
W_min = add_states(
    W_t,
    {
        (0, 1): -4 * X * Y**4,
        (1, 1): -3 * X**3 * Y**2,
    },
)
W_max = add_states(
    W_t,
    {
        (0, 1): -2 * X**5,
        (1, 1): -2 * X**3 * Y**2,
    },
)

# W_endpoint=Phi_endpoint^* W_t.
V_min = {
    (0, 1): (S.zero(), -X * Y),
    (1, 1): (-QQ(1) / 4 * Y**2, S.zero()),
}
V_max = {
    (0, 1): (-QQ(1) / 2 * X**2, S.zero()),
    (1, 1): (-QQ(1) / 4 * Y**2, S.zero()),
}
assert flow(W_t, V_min) == W_min
assert flow(W_t, V_max) == W_max
assert divergence(V_min) == {(0, 1): -X}
assert divergence(V_max) == {(0, 1): -X}

# Both endpoint pairs (W_endpoint,dX wedge dY) are pullbacks of the same
# calibrated base pair (W_t,(1+u*X)dX wedge dY).
calibrated_density = {(0, 0): S.one(), (0, 1): X}
jacobian_min = add_states({(0, 0): S.one()}, divergence(V_min))
jacobian_max = add_states({(0, 0): S.one()}, divergence(V_max))
assert multiply_states(flow(calibrated_density, V_min), jacobian_min) == {
    (0, 0): S.one()
}
assert multiply_states(flow(calibrated_density, V_max), jacobian_max) == {
    (0, 0): S.one()
}

# The alpha wall is V_max-V_min=-u*X_(1,0)/2.  It is divergence-free,
# transports the full minimum potential to the maximum potential, and
# intertwines the two calibrated coordinate systems.
alpha_wall = {
    (0, 1): (-QQ(1) / 2 * X**2, X * Y),
}
assert divergence(alpha_wall) == {}
assert flow(W_min, alpha_wall) == W_max

coordinate_X = {(0, 0): X}
coordinate_Y = {(0, 0): Y}
X_min = flow(coordinate_X, V_min)
Y_min = flow(coordinate_Y, V_min)
X_max = flow(coordinate_X, V_max)
Y_max = flow(coordinate_Y, V_max)
assert X_min == {(0, 0): X, (1, 1): -QQ(1) / 4 * Y**2}
assert Y_min == {(0, 0): Y, (0, 1): -X * Y}
assert X_max == {
    (0, 0): X,
    (0, 1): -QQ(1) / 2 * X**2,
    (1, 1): -QQ(1) / 4 * Y**2,
}
assert Y_max == {(0, 0): Y}
assert flow(X_min, alpha_wall) == X_max
assert flow(Y_min, alpha_wall) == Y_max

# The common calibrated trace is nontrivial but universal: it gives u on
# X*Y^2 and does not remember the mixed endpoint coefficient.
calibrated_XY2 = multiply_states(
    {(0, 0): X * Y**2}, calibrated_density
)
assert normalized_base_residue(calibrated_XY2) == {(0, 1): QQ.one()}

# A divergence-free gauge shifts the t*u*X^3*Y^2 coefficient by one without
# changing any other displayed term or the primitive form.  Scalar multiples
# therefore shift it arbitrarily.
mixed_gauge = {
    (1, 1): (QQ(1) / 4 * Y**2, S.zero()),
}
assert divergence(mixed_gauge) == {}
W_min_shifted = flow(W_min, mixed_gauge)
assert W_min_shifted == add_states(
    W_t,
    {
        (0, 1): -4 * X * Y**4,
        (1, 1): -2 * X**3 * Y**2,
    },
)

# The corresponding fixed-monomial Jacobian coefficient is gauge-dependent.
# For W_min(C)=W_t-4*u*X*Y^4+C*t*u*X^3*Y^2,
#
#   X^3 = -(t/2)X*Y^2 + (-1/2-3C/4)t*u*X^2*Y^2.
def minimum_x3_coefficient(mixed_potential_coefficient):
    return -QQ(1) / 2 - QQ(3) / 4 * mixed_potential_coefficient


assert minimum_x3_coefficient(-3) == QQ(7) / 4
assert minimum_x3_coefficient(-2) == QQ.one()
assert minimum_x3_coefficient(-2) - minimum_x3_coefficient(-3) == -QQ(3) / 4

print("mixed endpoint right equivalences: verified")
print("common calibrated primitive form: (1+u*X)dX wedge dY")
print("alpha wall in calibrated Jacobian coordinates: identity")
print("universal calibrated trace epsilon(X*Y^2): u")
print("divergence-free mixed-coefficient shift: arbitrary")
print("mixed wall-bearing Jacobian/primitive-form recovery: NO-GO")
