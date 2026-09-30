"""Exact chain-level checks for Proposition 6.4(1) (Transported primary Frobenius
structure: transport of Brieskorn modules).

For a Hamiltonian wall field X_(a,b) of eq:quartic-vector-field

    D = X^a Y^b ((b+1) X d_X - (a+1) Y d_Y),

the square-zero flow phi=id+epsilon*D preserves dX wedge dY.  Pullback by
phi must intertwine the twisted de Rham complexes of W and phi(W):

    (hbar*d + d(phi W) wedge) phi = phi (hbar*d + dW wedge).

The script verifies both degrees 0->1 and 1->2 exactly over QQ for a grid of
wall generators and polynomial forms.  It also checks that the primitive
two-form is fixed and that the action is not scalar on polynomial
representatives.  Finally, for each lifted slab map
z^m -> g_V^<N_V,m> z^m with g_V^(d_V) = 1 + z^(p_V) and <N_V,p_V> = 0, it
checks that the transported form dU^dV equals g_V^<N_V,kappa> dz_1^dz_2,
the multiplier of eq:slab-conformal, with exponents (-2,-2,2,-2).

Run:  sage check_period_module_intertwiner.sage
"""


S.<hbar, X, Y> = PolynomialRing(QQ, 3)


def pair_add(left, right):
    return (left[0] + right[0], left[1] + right[1])


def pair_scale(value, scalar):
    return (scalar * value[0], scalar * value[1])


def pair_multiply(left, right):
    """Multiply modulo epsilon^2."""

    return (
        left[0] * right[0],
        left[0] * right[1] + left[1] * right[0],
    )


def pair_derivative(value, variable):
    return (value[0].derivative(variable), value[1].derivative(variable))


def derivation(poly, a, b):
    return S(
        X**a
        * Y**b
        * (
            (b + 1) * X * poly.derivative(X)
            - (a + 1) * Y * poly.derivative(Y)
        )
    )


def vector_field_components(a, b):
    return (
        (b + 1) * X ** (a + 1) * Y**b,
        -(a + 1) * X**a * Y ** (b + 1),
    )


def pullback_function(poly, a, b):
    return (poly, derivation(poly, a, b))


def pullback_one_form(p_component, q_component, a, b):
    """Pull back p dX+q dY by id+epsilon*D modulo epsilon^2."""

    dX_component, dY_component = vector_field_components(a, b)
    return (
        (
            p_component,
            derivation(p_component, a, b)
            + p_component * dX_component.derivative(X)
            + q_component * dY_component.derivative(X),
        ),
        (
            q_component,
            derivation(q_component, a, b)
            + p_component * dX_component.derivative(Y)
            + q_component * dY_component.derivative(Y),
        ),
    )


def divergence(a, b):
    p_component, q_component = vector_field_components(a, b)
    return p_component.derivative(X) + q_component.derivative(Y)


def pullback_two_form(coefficient, a, b):
    return (
        coefficient,
        derivation(coefficient, a, b) + coefficient * divergence(a, b),
    )


def twisted_d0(function_pair, potential_pair):
    """(hbar*d+dW wedge): Omega^0 -> Omega^1."""

    f_X = pair_derivative(function_pair, X)
    f_Y = pair_derivative(function_pair, Y)
    W_X = pair_derivative(potential_pair, X)
    W_Y = pair_derivative(potential_pair, Y)
    return (
        pair_add(pair_scale(f_X, hbar), pair_multiply(function_pair, W_X)),
        pair_add(pair_scale(f_Y, hbar), pair_multiply(function_pair, W_Y)),
    )


def twisted_d1(one_form_pair, potential_pair):
    """(hbar*d+dW wedge): Omega^1 -> Omega^2."""

    p_component, q_component = one_form_pair
    exterior_derivative = pair_add(
        pair_derivative(q_component, X),
        pair_scale(pair_derivative(p_component, Y), -1),
    )
    W_X = pair_derivative(potential_pair, X)
    W_Y = pair_derivative(potential_pair, Y)
    potential_wedge = pair_add(
        pair_multiply(W_X, q_component),
        pair_scale(pair_multiply(W_Y, p_component), -1),
    )
    return pair_add(pair_scale(exterior_derivative, hbar), potential_wedge)


potential = X**4 + Y**4 + X**2 * Y**2
functions = (
    S.one(),
    X,
    Y,
    X**2 * Y,
    X * Y**2,
    X**3 + 2 * X * Y + Y**3,
)
one_forms = (
    (S.one(), S.zero()),
    (S.zero(), S.one()),
    (X, Y),
    (X**2 * Y, X * Y**2),
    (X**3 + Y, Y**3 - X),
)


non_scalar_witnesses = []
for a in range(4):
    for b in range(4):
        assert divergence(a, b) == 0
        assert pullback_two_form(S.one(), a, b) == (S.one(), S.zero())

        potential_pullback = pullback_function(potential, a, b)
        for function in functions:
            old_d0 = twisted_d0((function, S.zero()), (potential, S.zero()))
            pulled_old_d0 = pullback_one_form(
                old_d0[0][0], old_d0[1][0], a, b
            )
            new_d0 = twisted_d0(
                pullback_function(function, a, b),
                potential_pullback,
            )
            assert pulled_old_d0 == new_d0

        for p_component, q_component in one_forms:
            old_d1 = twisted_d1(
                ((p_component, S.zero()), (q_component, S.zero())),
                (potential, S.zero()),
            )
            pulled_old_d1 = pullback_two_form(old_d1[0], a, b)
            new_d1 = twisted_d1(
                pullback_one_form(p_component, q_component, a, b),
                potential_pullback,
            )
            assert pulled_old_d1 == new_d1

        coordinate_action = pullback_function(X + Y, a, b)
        if coordinate_action[1] != 0:
            non_scalar_witnesses.append((a, b, coordinate_action[1]))

assert non_scalar_witnesses


print("Jacobi wall divergences on 16 generators: zero")
print("primitive two-form pullbacks: fixed")
print("twisted de Rham chain maps in degrees 0->1 and 1->2: PASS")
print("non-scalar coordinate-action witnesses:", len(non_scalar_witnesses))
# Slab multipliers: d_V and N_V from eq:root-data, p_V orthogonal to N_V.
z1, z2 = var("z1 z2")
slab_data = {
    "A": (2, (1, -3)),
    "B0": (4, (-1, -1)),
    "B1": (4, (-3, 5)),
    "C": (2, (-3, 1)),
}
multiplier_exponents = []
for name, (d_V, (n1, n2)) in slab_data.items():
    p1, p2 = n2, -n1
    assert n1 * p1 + n2 * p2 == 0
    g = (1 + z1**p1 * z2**p2) ** (1 / d_V)
    U = g**n1 * z1
    V = g**n2 * z2
    jac = diff(U, z1) * diff(V, z2) - diff(U, z2) * diff(V, z1)
    assert (jac / g ** (n1 + n2)).simplify_full() == 1, name
    multiplier_exponents.append(n1 + n2)
assert multiplier_exponents == [-2, -2, 2, -2]
print("slab multipliers dU^dV = g_V^<N_V,kappa> dz1^dz2, exponents", multiplier_exponents)
print("FRAMED BRIESKORN INTERTWINER: PASS")
