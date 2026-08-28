"""Exact four-type certificate for quartic GKT/Jacobi wall operations.

The four standard descendent-one singleton twists in the residue orbit are

    alpha = (1,0),  delta = (3,2),  gamma = (2,3),  beta = (0,1).

For every unordered pair of distinct labelled singletons, this script:

1. solves the two endpoint GKT chamber equations;
2. recovers the complete increasing-slope mixed GKT factorization;
3. computes the first local inverse-commutator factor; and
4. determines the unique residual factors which make the product equal to
   that local scattering correction.

The label variables are square-zero, so all computations are finite and all
arithmetic is exact over QQ.
"""


from itertools import combinations
from math import gcd


S.<x, y> = PolynomialRing(QQ, 2)


def gamma_ratio(top_numerator, base_numerator, denominator=4):
    """Gamma(top/4)/Gamma(base/4) when the arguments differ integrally."""

    assert top_numerator >= base_numerator
    assert (top_numerator - base_numerator) % denominator == 0
    output = QQ.one()
    for step in range((top_numerator - base_numerator) // denominator):
        output *= QQ(base_numerator + step * denominator) / denominator
    return output


def gamma_weight(exponent, residue):
    """The quartic Gamma weight of an exponent relative to its residue."""

    return gamma_ratio(exponent[0] + 1, residue[0] + 1) * gamma_ratio(
        exponent[1] + 1, residue[1] + 1
    )


def shifted_ray(k):
    """Primitive shifted Hamiltonian ray associated with the GKT exponent k."""

    divisor = gcd(k[0] + 1, k[1] + 1)
    return ((k[0] + 1) // divisor, (k[1] + 1) // divisor)


def slope(k):
    """Exact slope of the shifted Hamiltonian ray."""

    return QQ(k[1] + 1) / QQ(k[0] + 1)


def jacobi_derivative(poly, k):
    """The quartic GKT generator X_k applied to a polynomial."""

    a, b = k
    return S(
        x**a
        * y**b
        * ((b + 1) * x * poly.derivative(x) - (a + 1) * y * poly.derivative(y))
    )


def add_state(left, right):
    """Add coefficient states indexed by the two label exponents."""

    output = dict(left)
    for support, coefficient in right.items():
        output[support] = output.get(support, S.zero()) + coefficient
        if output[support] == 0:
            del output[support]
    return output


def apply_generator(state, generator):
    """Apply c*t1^e1*t2^e2*X_k in the labelled square-zero quotient."""

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
    """Apply exp(generator), which truncates after its linear term here."""

    return add_state(state, apply_generator(state, generator))


def chronological_product(state, generators):
    """Apply generators in chronological, hence increasing-slope, order."""

    output = state
    for generator in generators:
        output = flow(output, generator)
    return output


def singleton_data(twist):
    """Extreme invariants and canonical wall coefficient for one marking."""

    a, b = twist
    return {
        "twist": twist,
        "minimum_exponent": (a, b + 4),
        "minimum_invariant": -QQ(4) / (b + 1),
        "maximum_exponent": (a + 4, b),
        "maximum_invariant": -QQ(4) / (a + 1),
        "wall_coefficient": -QQ.one() / ((a + 1) * (b + 1)),
    }


def monomial(exponent):
    return x**exponent[0] * y**exponent[1]


def polynomial_vector(poly, profiles):
    """Coefficient vector on an ordered list of monomial profiles."""

    return vector(
        QQ,
        [poly.monomial_coefficient(monomial(profile)) for profile in profiles],
    )


def pair_block(name1, twist1, name2, twist2):
    """Solve one two-label endpoint transition and its residual decomposition."""

    first = singleton_data(twist1)
    second = singleton_data(twist2)
    A = twist1[0] + twist2[0]
    B = twist1[1] + twist2[1]
    residue = (A % 4, B % 4)
    carry = ((A - residue[0]) // 4, (B - residue[1]) // 4)

    # There are N_balanced+1 potential profiles and N_balanced critical
    # generators.  Equivalently, the critical-block parameter of
    # Proposition 6.19 is N_balanced-1=sum(carry).
    N_balanced = carry[0] + carry[1] + 1
    profiles = tuple(
        (residue[0] + 4 * p, residue[1] + 4 * (N_balanced - p))
        for p in range(N_balanced + 1)
    )
    critical = tuple(
        (residue[0] + 4 * p, residue[1] + 4 * (N_balanced - 1 - p))
        for p in range(N_balanced)
    )

    minimum_product_exponent = (
        first["minimum_exponent"][0] + second["minimum_exponent"][0],
        first["minimum_exponent"][1] + second["minimum_exponent"][1],
    )
    maximum_product_exponent = (
        first["maximum_exponent"][0] + second["maximum_exponent"][0],
        first["maximum_exponent"][1] + second["maximum_exponent"][1],
    )

    minimum_invariant = -(
        gamma_weight(minimum_product_exponent, residue)
        * first["minimum_invariant"]
        * second["minimum_invariant"]
        / gamma_weight(profiles[0], residue)
    )
    maximum_invariant = -(
        gamma_weight(maximum_product_exponent, residue)
        * first["maximum_invariant"]
        * second["maximum_invariant"]
        / gamma_weight(profiles[-1], residue)
    )

    # These are the two endpoint GKT chamber equations.  The potential sign
    # for the mixed block is (-1)^(2-1)=-1.
    assert (
        gamma_weight(profiles[0], residue) * minimum_invariant
        + gamma_weight(minimum_product_exponent, residue)
        * first["minimum_invariant"]
        * second["minimum_invariant"]
        == 0
    )
    assert (
        gamma_weight(profiles[-1], residue) * maximum_invariant
        + gamma_weight(maximum_product_exponent, residue)
        * first["maximum_invariant"]
        * second["maximum_invariant"]
        == 0
    )

    W_min = {
        (0, 0): x**4 + y**4,
        (1, 0): first["minimum_invariant"]
        * monomial(first["minimum_exponent"]),
        (0, 1): second["minimum_invariant"]
        * monomial(second["minimum_exponent"]),
        (1, 1): -minimum_invariant * monomial(profiles[0]),
    }
    W_max = {
        (0, 0): x**4 + y**4,
        (1, 0): first["maximum_invariant"]
        * monomial(first["maximum_exponent"]),
        (0, 1): second["maximum_invariant"]
        * monomial(second["maximum_exponent"]),
        (1, 1): -maximum_invariant * monomial(profiles[-1]),
    }

    singleton_generators = (
        ((1, 0), twist1, first["wall_coefficient"]),
        ((0, 1), twist2, second["wall_coefficient"]),
    )
    singleton_generators = tuple(
        sorted(singleton_generators, key=lambda generator: slope(generator[1]))
    )
    baseline = chronological_product(W_min, singleton_generators)

    # Every mixed generator acts only on the coefficient-free Fermat term.
    # Solve the resulting overdetermined tridiagonal system exactly.
    columns = tuple(
        polynomial_vector(jacobi_derivative(x**4 + y**4, k), profiles)
        for k in critical
    )
    matrix_columns = matrix(QQ, columns).transpose()
    target_difference = polynomial_vector(
        W_max[(1, 1)] - baseline.get((1, 1), S.zero()), profiles
    )
    gkt_coefficients = tuple(matrix_columns.solve_right(target_difference))
    gkt_by_exponent = dict(zip(critical, gkt_coefficients))

    mixed_generators = tuple(
        ((1, 1), k, gkt_by_exponent[k])
        for k in sorted(critical, key=slope)
    )
    all_generators = tuple(
        sorted(singleton_generators + mixed_generators, key=lambda generator: slope(generator[1]))
    )
    assert chronological_product(W_min, all_generators) == W_max

    # For the two incoming singleton lines, the inverse commutator is
    # det(q_low,q_high)*c_low*c_high times X_(twist1+twist2).  At labelled
    # order two this is the only ordinary scattering ray.  Mixed GKT factors
    # on every other critical ray must therefore be cancelled by their
    # residual factors.
    low, high = singleton_generators
    q_low = vector(ZZ, (low[1][0] + 1, low[1][1] + 1))
    q_high = vector(ZZ, (high[1][0] + 1, high[1][1] + 1))
    determinant = matrix(ZZ, [q_low, q_high]).det()
    assert determinant >= 0
    summed_exponent = (A, B)
    assert summed_exponent in critical
    scattering_coefficient = determinant * low[2] * high[2]
    scattering_by_exponent = {
        k: scattering_coefficient if k == summed_exponent else QQ.zero()
        for k in critical
    }
    residual_by_exponent = {
        k: scattering_by_exponent[k] - gkt_by_exponent[k] for k in critical
    }

    # The mixed ray groups commute in the labelled square-zero quotient, so
    # the enumerative and residual factors multiply exactly to the local
    # inverse commutator, not merely modulo higher order.
    coordinate_states = ({(0, 0): x}, {(0, 0): y})
    gkt_mixed = tuple(
        ((1, 1), k, gkt_by_exponent[k]) for k in sorted(critical, key=slope)
    )
    residual_mixed = tuple(
        ((1, 1), k, residual_by_exponent[k])
        for k in sorted(critical, key=slope)
        if residual_by_exponent[k]
    )
    scattering_mixed = (
        ((1, 1), summed_exponent, scattering_coefficient),
    )
    for coordinate in coordinate_states:
        assert chronological_product(
            chronological_product(coordinate, gkt_mixed), residual_mixed
        ) == chronological_product(coordinate, scattering_mixed)

    return {
        "names": (name1, name2),
        "twists": (twist1, twist2),
        "residue": residue,
        "profiles": profiles,
        "critical": tuple(sorted(critical, key=slope)),
        "minimum_invariant": minimum_invariant,
        "maximum_invariant": maximum_invariant,
        "gkt": gkt_by_exponent,
        "determinant": determinant,
        "scattering": scattering_by_exponent,
        "residual": residual_by_exponent,
    }


types = (
    ("alpha", (1, 0)),
    ("delta", (3, 2)),
    ("gamma", (2, 3)),
    ("beta", (0, 1)),
)

# The standard endpoint-frame singleton coefficients are not constant on the
# residue orbit.  This is an independent numerical witness to the distinction
# between standard retwisting and path-framed transport.
singleton_coefficients = {
    name: singleton_data(twist)["wall_coefficient"] for name, twist in types
}
assert singleton_coefficients == {
    "alpha": -QQ(1) / 2,
    "delta": -QQ(1) / 12,
    "gamma": -QQ(1) / 12,
    "beta": -QQ(1) / 2,
}

blocks = tuple(
    pair_block(name1, twist1, name2, twist2)
    for (name1, twist1), (name2, twist2) in combinations(types, 2)
)
repeated_blocks = tuple(
    pair_block(name, twist, name, twist) for name, twist in types
)

# Fixed regression table.  These assertions are deliberately separate from
# the chamber solver above: the certificate fails if any endpoint invariant,
# block rank, GKT factor, determinant, or residual coefficient changes.
expected = {
    ("alpha", "alpha"): {
        "profiles": ((2, 4), (6, 0)),
        "invariants": (-QQ(20), -QQ(7)),
        "gkt": {(2, 0): QQ(1) / 2},
        "determinant": 0,
        "scattering": {(2, 0): QQ.zero()},
        "residual": {(2, 0): -QQ(1) / 2},
    },
    ("alpha", "delta"): {
        "profiles": ((0, 10), (4, 6), (8, 2)),
        "invariants": (-QQ(4) / 3, -QQ(9) / 2),
        "gkt": {(0, 6): QQ(1) / 3, (4, 2): QQ(1) / 6},
        "determinant": 2,
        "scattering": {(0, 6): QQ.zero(), (4, 2): QQ(1) / 12},
        "residual": {(0, 6): -QQ(1) / 3, (4, 2): -QQ(1) / 12},
    },
    ("alpha", "gamma"): {
        "profiles": ((3, 7), (7, 3)),
        "invariants": (-QQ(8), -QQ(16) / 3),
        "gkt": {(3, 3): QQ(1) / 8},
        "determinant": 5,
        "scattering": {(3, 3): QQ(5) / 24},
        "residual": {(3, 3): QQ(1) / 12},
    },
    ("alpha", "beta"): {
        "profiles": ((1, 5), (5, 1)),
        "invariants": (-QQ(12), -QQ(12)),
        "gkt": {(1, 1): QQ(1) / 4},
        "determinant": 3,
        "scattering": {(1, 1): QQ(3) / 4},
        "residual": {(1, 1): QQ(1) / 2},
    },
    ("delta", "gamma"): {
        "profiles": ((1, 13), (5, 9), (9, 5), (13, 1)),
        "invariants": (-QQ(2) / 3, -QQ(2) / 3),
        "gkt": {
            (1, 9): QQ(1) / 12,
            (5, 5): QQ(1) / 16,
            (9, 1): QQ(1) / 12,
        },
        "determinant": 7,
        "scattering": {
            (1, 9): QQ.zero(),
            (5, 5): QQ(7) / 144,
            (9, 1): QQ.zero(),
        },
        "residual": {
            (1, 9): -QQ(1) / 12,
            (5, 5): -QQ(1) / 72,
            (9, 1): -QQ(1) / 12,
        },
    },
    ("delta", "delta"): {
        "profiles": ((2, 12), (6, 8), (10, 4), (14, 0)),
        "invariants": (-QQ(4) / 3, -QQ(1) / 4),
        "gkt": {
            (2, 8): QQ(1) / 9,
            (6, 4): QQ(1) / 12,
            (10, 0): QQ(1) / 16,
        },
        "determinant": 0,
        "scattering": {
            (2, 8): QQ.zero(),
            (6, 4): QQ.zero(),
            (10, 0): QQ.zero(),
        },
        "residual": {
            (2, 8): -QQ(1) / 9,
            (6, 4): -QQ(1) / 12,
            (10, 0): -QQ(1) / 16,
        },
    },
    ("delta", "beta"): {
        "profiles": ((3, 7), (7, 3)),
        "invariants": (-QQ(16) / 3, -QQ(8)),
        "gkt": {(3, 3): QQ(1) / 8},
        "determinant": 5,
        "scattering": {(3, 3): QQ(5) / 24},
        "residual": {(3, 3): QQ(1) / 12},
    },
    ("gamma", "beta"): {
        "profiles": ((2, 8), (6, 4), (10, 0)),
        "invariants": (-QQ(9) / 2, -QQ(4) / 3),
        "gkt": {(2, 4): QQ(1) / 6, (6, 0): QQ(1) / 3},
        "determinant": 2,
        "scattering": {(2, 4): QQ(1) / 12, (6, 0): QQ.zero()},
        "residual": {(2, 4): -QQ(1) / 12, (6, 0): -QQ(1) / 3},
    },
    ("gamma", "gamma"): {
        "profiles": ((0, 14), (4, 10), (8, 6), (12, 2)),
        "invariants": (-QQ(1) / 4, -QQ(4) / 3),
        "gkt": {
            (0, 10): QQ(1) / 16,
            (4, 6): QQ(1) / 12,
            (8, 2): QQ(1) / 9,
        },
        "determinant": 0,
        "scattering": {
            (0, 10): QQ.zero(),
            (4, 6): QQ.zero(),
            (8, 2): QQ.zero(),
        },
        "residual": {
            (0, 10): -QQ(1) / 16,
            (4, 6): -QQ(1) / 12,
            (8, 2): -QQ(1) / 9,
        },
    },
    ("beta", "beta"): {
        "profiles": ((0, 6), (4, 2)),
        "invariants": (-QQ(7), -QQ(20)),
        "gkt": {(0, 2): QQ(1) / 2},
        "determinant": 0,
        "scattering": {(0, 2): QQ.zero()},
        "residual": {(0, 2): -QQ(1) / 2},
    },
}

all_blocks = blocks + repeated_blocks
assert {block["names"] for block in all_blocks} == set(expected)
for block in all_blocks:
    target = expected[block["names"]]
    assert block["profiles"] == target["profiles"]
    assert (
        block["minimum_invariant"],
        block["maximum_invariant"],
    ) == target["invariants"]
    assert block["gkt"] == target["gkt"]
    assert block["determinant"] == target["determinant"]
    assert block["scattering"] == target["scattering"]
    assert block["residual"] == target["residual"]

print("standard singleton coefficients:", singleton_coefficients)
for block in all_blocks:
    print("PAIR", "-".join(block["names"]))
    print("  profiles:", block["profiles"])
    print(
        "  endpoint invariants:",
        block["minimum_invariant"],
        block["maximum_invariant"],
    )
    print(
        "  GKT:",
        tuple((k, block["gkt"][k]) for k in block["critical"]),
    )
    print("  determinant:", block["determinant"])
    print(
        "  scattering:",
        tuple((k, block["scattering"][k]) for k in block["critical"]),
    )
    print(
        "  residual:",
        tuple((k, block["residual"][k]) for k in block["critical"]),
    )

print("ALL FOUR-TYPE GKT/JACOBI OPERATION CHECKS PASSED")
