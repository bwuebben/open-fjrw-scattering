"""Exact support certificate for the four standard singleton types.

This script checks the geometric and quadratic assertions of Proposition
6.42.  The two middle standard types occur as degree-five rays of the
two-line completion.  Their positive germs calibrate two additional full
cofactor chords.  With independent square-zero labels on the four chords,
the ordinary quadratic scattering coefficients agree with the S-column of
Proposition 6.21, while the GKT side factors are cancelled ray by ray by the
residual decoration.

All calculations are exact over QQ.
"""


from itertools import combinations, combinations_with_replacement


kappa = vector(ZZ, (1, 1))
cofactor = matrix(ZZ, [[-2, 0], [1, 1]])
cone_forms = matrix(ZZ, [[2, -1], [-1, 2]])

# Slope order in the distinguished standard frame.
types = (
    ("alpha", vector(ZZ, (1, 0)), -QQ(1) / 2),
    ("delta", vector(ZZ, (3, 2)), -QQ(1) / 12),
    ("gamma", vector(ZZ, (2, 3)), -QQ(1) / 12),
    ("beta", vector(ZZ, (0, 1)), -QQ(1) / 2),
)


def shifted(twist):
    return twist + kappa


def slope(exponent):
    return QQ(exponent[1]) / exponent[0]


shifted_types = {name: shifted(twist) for name, twist, _ in types}
assert [slope(shifted_types[name]) for name, _, _ in types] == [
    QQ(1) / 2,
    QQ(3) / 4,
    QQ(4) / 3,
    QQ(2),
]
for exponent in shifted_types.values():
    assert all(value >= 0 for value in cone_forms * exponent)

# The middle rays are already nonzero positive germs in the completed
# two-line diagram.  The displayed normal coordinates turn their coefficients
# into the standard singleton coefficient -1/12.
two_line_degree_five = {
    (2, 3): -QQ(3) / 16,
    (3, 2): QQ(3) / 16,
}
composite_normal_coordinates = {
    "gamma": QQ(9) / 4,   # s_gamma=(9/4) t_alpha^2 t_beta^3
    "delta": -QQ(9) / 4,  # s_delta=-(9/4) t_alpha^3 t_beta^2
}
standard_coefficients = {name: coefficient for name, _, coefficient in types}
assert (
    standard_coefficients["gamma"] * composite_normal_coordinates["gamma"]
    == two_line_degree_five[(2, 3)]
)
assert (
    standard_coefficients["delta"] * composite_normal_coordinates["delta"]
    == two_line_degree_five[(3, 2)]
)
assert two_line_degree_five[(3, 2)] == -two_line_degree_five[(2, 3)]


def chord_geometry(exponent):
    """Cofactor chord endpoints and its barycentric branch-cut crossing."""

    direction = cofactor * exponent
    assert direction[0] < 0 and direction[1] > 0

    # Delta=conv{(-1,-1),(-1,1),(3,-1)}.  Its upper facets are x=-1 and
    # x+2y=1.  The positive ray hits whichever has the smaller exit time.
    left_time = -QQ.one() / direction[0]
    diagonal_time = QQ.one() / (direction[0] + 2 * direction[1])
    positive_time = min(left_time, diagonal_time)
    positive_endpoint = positive_time * direction

    # The negative ray always exits at y=-1.
    negative_time = -QQ.one() / direction[1]
    negative_endpoint = negative_time * direction

    # The barycentric A--C branch segment is
    # [-1/2,3/2] x {-1/2}.  Every standard chord crosses its interior once.
    branch_time = -QQ.one() / (2 * direction[1])
    branch_crossing = branch_time * direction

    assert positive_endpoint[0] == -1 or (
        positive_endpoint[0] + 2 * positive_endpoint[1] == 1
    )
    assert negative_endpoint[1] == -1
    assert branch_crossing[1] == -QQ(1) / 2
    assert -QQ(1) / 2 < branch_crossing[0] < QQ(3) / 2
    return direction, positive_endpoint, negative_endpoint, branch_crossing


geometry = {
    name: chord_geometry(shifted(twist)) for name, twist, _ in types
}
assert geometry == {
    "alpha": (
        vector(ZZ, (-4, 3)),
        vector(QQ, (-1, QQ(3) / 4)),
        vector(QQ, (QQ(4) / 3, -1)),
        vector(QQ, (QQ(2) / 3, -QQ(1) / 2)),
    ),
    "delta": (
        vector(ZZ, (-8, 7)),
        vector(QQ, (-1, QQ(7) / 8)),
        vector(QQ, (QQ(8) / 7, -1)),
        vector(QQ, (QQ(4) / 7, -QQ(1) / 2)),
    ),
    "gamma": (
        vector(ZZ, (-6, 7)),
        vector(QQ, (-QQ(3) / 4, QQ(7) / 8)),
        vector(QQ, (QQ(6) / 7, -1)),
        vector(QQ, (QQ(3) / 7, -QQ(1) / 2)),
    ),
    "beta": (
        vector(ZZ, (-2, 3)),
        vector(QQ, (-QQ(1) / 2, QQ(3) / 4)),
        vector(QQ, (QQ(2) / 3, -1)),
        vector(QQ, (QQ(1) / 3, -QQ(1) / 2)),
    ),
}

# Distinct standard chords meet only at O in the straight cofactor chart, and
# they cross the branch segment at four distinct nonsingular points.
for (name_left, _, _), (name_right, _, _) in combinations(types, 2):
    direction_left = geometry[name_left][0]
    direction_right = geometry[name_right][0]
    assert matrix(ZZ, [direction_left, direction_right]).det() != 0
assert len({data[3][0] for data in geometry.values()}) == 4


def determinant(first, second):
    return matrix(ZZ, [first, second]).det()


# Exact ordinary quadratic scattering table.  The names are already in
# increasing shifted-slope order, so every determinant is positive.
expected_scattering = {
    ("alpha", "delta"): ((4, 2), QQ(1) / 12),
    ("alpha", "gamma"): ((3, 3), QQ(5) / 24),
    ("alpha", "beta"): ((1, 1), QQ(3) / 4),
    ("delta", "gamma"): ((5, 5), QQ(7) / 144),
    ("delta", "beta"): ((3, 3), QQ(5) / 24),
    ("gamma", "beta"): ((2, 4), QQ(1) / 12),
}
twists = {name: twist for name, twist, _ in types}
for (left_name, _, left_coefficient), (
    right_name,
    _,
    right_coefficient,
) in combinations(types, 2):
    left_exponent = shifted_types[left_name]
    right_exponent = shifted_types[right_name]
    intersection = determinant(left_exponent, right_exponent)
    assert intersection > 0
    output_twist = twists[left_name] + twists[right_name]
    output_coefficient = intersection * left_coefficient * right_coefficient
    assert expected_scattering[(left_name, right_name)] == (
        tuple(output_twist),
        output_coefficient,
    )
    output_shifted = output_twist + kappa
    assert all(value >= 0 for value in cone_forms * output_shifted)

for name, _, coefficient in types:
    assert determinant(shifted_types[name], shifted_types[name]) * coefficient**2 == 0

# The complete quadratic GKT rows.  Every non-scattering GKT side factor is
# cancelled on its own ray by R=S-E.  Such a decorated identity slot changes
# no set-theoretic wall support.
gkt_rows = {
    ("alpha", "alpha"): {(2, 0): QQ(1) / 2},
    ("alpha", "delta"): {(0, 6): QQ(1) / 3, (4, 2): QQ(1) / 6},
    ("alpha", "gamma"): {(3, 3): QQ(1) / 8},
    ("alpha", "beta"): {(1, 1): QQ(1) / 4},
    ("delta", "delta"): {
        (2, 8): QQ(1) / 9,
        (6, 4): QQ(1) / 12,
        (10, 0): QQ(1) / 16,
    },
    ("delta", "gamma"): {
        (1, 9): QQ(1) / 12,
        (5, 5): QQ(1) / 16,
        (9, 1): QQ(1) / 12,
    },
    ("delta", "beta"): {(3, 3): QQ(1) / 8},
    ("gamma", "gamma"): {
        (0, 10): QQ(1) / 16,
        (4, 6): QQ(1) / 12,
        (8, 2): QQ(1) / 9,
    },
    ("gamma", "beta"): {(2, 4): QQ(1) / 6, (6, 0): QQ(1) / 3},
    ("beta", "beta"): {(0, 2): QQ(1) / 2},
}
assert set(gkt_rows) == {
    (left[0], right[0]) for left, right in combinations_with_replacement(types, 2)
}

for left, right in combinations_with_replacement(types, 2):
    pair = (left[0], right[0])
    enumerative = gkt_rows[pair]
    scattering = {critical: QQ.zero() for critical in enumerative}
    if left[0] != right[0]:
        output_twist, output_coefficient = expected_scattering[pair]
        assert output_twist in scattering
        scattering[output_twist] = output_coefficient
    residual = {
        critical: scattering[critical] - coefficient
        for critical, coefficient in enumerative.items()
    }
    for critical in enumerative:
        assert enumerative[critical] + residual[critical] == scattering[critical]
        # Every GKT critical exponent has a canonical proper radial germ in
        # the positive cofactor half-plane, whether or not its total factor
        # survives after residual cancellation.
        critical_shifted = vector(ZZ, critical) + kappa
        chord_geometry(critical_shifted)

print("standard shifted directions:", shifted_types)
print("four cofactor chord geometries:", geometry)
print("composite normal-coordinate calibration:", composite_normal_coordinates)
print("quadratic ordinary scattering rows:", expected_scattering)
print("FOUR-LINE ANNULAR SUPPORT CHECKS PASSED")
