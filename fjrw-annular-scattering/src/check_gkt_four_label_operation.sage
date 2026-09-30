"""Exact checks for the quartic labelled truncations of Proposition 5.5.

This script reuses (and therefore reruns) the recursive extremal-invariant
and factorization functions in check_gkt_three_label_operation.sage.  It treats the
five multisets of four distinct labels of twists alpha=(1,0), beta=(0,1),
including the two pure blocks where coordinate carry creates two critical GKT
rays.  The path-ordered product e^A e^B has only the summed-exponent
ray on the full support, so the comparison and residual are vector-valued on
the complete critical block.  One five-label carry block (alpha^4 beta) is
checked as well.  The engine is loaded from the directory of this file.
"""

import os

load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  "check_gkt_three_label_operation.sage"))


def run_quartic_case(name, labels):
    assert len(labels) == 4
    minimum = endpoint_system(labels, "minimum")
    maximum = endpoint_system(labels, "maximum")
    W_min = endpoint_potential(minimum)
    W_max = endpoint_potential(maximum)
    gkt, gkt_generators = solve_gkt_factorization(labels, W_min, W_max)
    word = increasing_slope_word(labels)
    scattering, scattering_generators = solve_scattering_factorization(labels, word)

    full_mask = (1 << len(labels)) - 1
    geometry = block_geometry(labels, full_mask)
    gkt_full = gkt[full_mask]
    summed_exponent = geometry["sum"]
    assert summed_exponent in geometry["critical"]
    scattering_full = {
        k: scattering[full_mask] if k == summed_exponent else QQ.zero()
        for k in geometry["critical"]
    }
    residual_full = {
        k: scattering_full[k] - gkt_full[k] for k in geometry["critical"]
    }

    # Full-support factors are central in this labelled quotient.  Check the
    # enumerative/residual product directly on both coordinate functions.
    gkt_mixed = tuple(
        (full_mask, k, gkt_full[k])
        for k in sorted(geometry["critical"], key=slope)
    )
    residual_mixed = tuple(
        (full_mask, k, residual_full[k])
        for k in sorted(geometry["critical"], key=slope)
        if residual_full[k]
    )
    scattering_mixed = tuple(
        (full_mask, k, scattering_full[k])
        for k in sorted(geometry["critical"], key=slope)
        if scattering_full[k]
    )
    for coordinate in ({0: x}, {0: y}):
        assert chronological_product(
            chronological_product(coordinate, gkt_mixed), residual_mixed
        ) == chronological_product(coordinate, scattering_mixed)

    return {
        "name": name,
        "labels": labels,
        "profiles": geometry["profiles"],
        "critical": tuple(sorted(geometry["critical"], key=slope)),
        "minimum_invariant": minimum[full_mask]["invariant"],
        "maximum_invariant": maximum[full_mask]["invariant"],
        "gkt": gkt_full,
        "scattering": scattering_full,
        "residual": residual_full,
        "gkt_generators": gkt_generators,
        "scattering_generators": scattering_generators,
    }


quartic_cases = (
    run_quartic_case("alpha^4", (alpha, alpha, alpha, alpha)),
    run_quartic_case("alpha^3 beta", (alpha, alpha, alpha, beta)),
    run_quartic_case("alpha^2 beta^2", (alpha, alpha, beta, beta)),
    run_quartic_case("alpha beta^3", (alpha, beta, beta, beta)),
    run_quartic_case("beta^4", (beta, beta, beta, beta)),
)

quartic_expected = {
    "alpha^4": {
        "profiles": ((0, 8), (4, 4), (8, 0)),
        "invariants": (-QQ(168), -QQ(495) / 2),
        "gkt": {(0, 4): QQ(42), (4, 0): QQ(9) / 4},
        "scattering": {(0, 4): QQ.zero(), (4, 0): QQ.zero()},
        "residual": {(0, 4): -QQ(42), (4, 0): -QQ(9) / 4},
    },
    "alpha^3 beta": {
        "profiles": ((3, 5), (7, 1)),
        "invariants": (-QQ(672), -QQ(360)),
        "gkt": {(3, 1): QQ(15) / 8},
        "scattering": {(3, 1): -QQ(3) / 8},
        "residual": {(3, 1): -QQ(9) / 4},
    },
    "alpha^2 beta^2": {
        "profiles": ((2, 6), (6, 2)),
        "invariants": (-QQ(504), -QQ(504)),
        "gkt": {(2, 2): QQ(3) / 2},
        "scattering": {(2, 2): -QQ(3) / 2},
        "residual": {(2, 2): -QQ(3)},
    },
    "alpha beta^3": {
        "profiles": ((1, 7), (5, 3)),
        "invariants": (-QQ(360), -QQ(672)),
        "gkt": {(1, 3): QQ(15) / 8},
        "scattering": {(1, 3): -QQ(3) / 8},
        "residual": {(1, 3): -QQ(9) / 4},
    },
    "beta^4": {
        "profiles": ((0, 8), (4, 4), (8, 0)),
        "invariants": (-QQ(495) / 2, -QQ(168)),
        "gkt": {(0, 4): QQ(9) / 4, (4, 0): QQ(42)},
        "scattering": {(0, 4): QQ.zero(), (4, 0): QQ.zero()},
        "residual": {(0, 4): -QQ(9) / 4, (4, 0): -QQ(42)},
    },
}

assert {case["name"] for case in quartic_cases} == set(quartic_expected)
for case in quartic_cases:
    target = quartic_expected[case["name"]]
    assert case["profiles"] == target["profiles"]
    assert (
        case["minimum_invariant"],
        case["maximum_invariant"],
    ) == target["invariants"]
    assert case["gkt"] == target["gkt"]
    assert case["scattering"] == target["scattering"]
    assert case["residual"] == target["residual"]

# Polarization of the unlabelled degree-four coefficients of the
# increasing-slope factorization, a_(3,1)=a_(1,3)=-1/16 and a_(2,2)=-3/8
# (the negatives of the coefficients of Proposition 3.5, computed
# independently): a term t_alpha^p*t_beta^q acquires p!*q! after its repeated
# variables are replaced by distinct square-zero labels.
assert quartic_expected["alpha^3 beta"]["scattering"][(3, 1)] == (
    factorial(3) * factorial(1) * (-QQ(1) / 16)
)
assert quartic_expected["alpha^2 beta^2"]["scattering"][(2, 2)] == (
    factorial(2) * factorial(2) * (-QQ(3) / 8)
)
assert quartic_expected["alpha beta^3"]["scattering"][(1, 3)] == (
    factorial(1) * factorial(3) * (-QQ(1) / 16)
)
# The residuals are symmetric under alpha <-> beta.
assert quartic_expected["alpha^3 beta"]["residual"][(3, 1)] == (
    quartic_expected["alpha beta^3"]["residual"][(1, 3)]
)

# The alpha^4*beta block is the first mixed quintic block with coordinate
# carry.  Its summed shifted ray (5,2) lies outside the cone spanned by (2,1)
# and (1,2), so its full-support scattering coefficient vanishes even though
# both critical GKT factors are nonzero.  This case checks the arbitrary-truncation
# statement of Proposition 5.5 beyond four labels.
quintic_labels = (alpha, alpha, alpha, alpha, beta)
quintic_minimum = endpoint_system(quintic_labels, "minimum")
quintic_maximum = endpoint_system(quintic_labels, "maximum")
quintic_W_min = endpoint_potential(quintic_minimum)
quintic_W_max = endpoint_potential(quintic_maximum)
quintic_gkt, _ = solve_gkt_factorization(
    quintic_labels, quintic_W_min, quintic_W_max
)
quintic_target = increasing_slope_word(quintic_labels)
quintic_scattering, _ = solve_scattering_factorization(
    quintic_labels, quintic_target
)
quintic_mask = (1 << len(quintic_labels)) - 1
quintic_geometry = block_geometry(quintic_labels, quintic_mask)
assert quintic_geometry["profiles"] == ((0, 9), (4, 5), (8, 1))
assert quintic_geometry["critical"] == ((0, 5), (4, 1))
assert quintic_minimum[quintic_mask]["invariant"] == -QQ(1428)
assert quintic_maximum[quintic_mask]["invariant"] == -QQ(2970)
assert quintic_gkt[quintic_mask] == {
    (0, 5): -QQ(189),
    (4, 1): -QQ(21) / 2,
}
assert quintic_scattering[quintic_mask] == 0

for case in quartic_cases:
    print("QUARTIC CASE", case["name"])
    print("  profiles:", case["profiles"])
    print(
        "  extremal invariants:",
        case["minimum_invariant"],
        case["maximum_invariant"],
    )
    print(
        "  GKT:",
        tuple((k, case["gkt"][k]) for k in case["critical"]),
    )
    print(
        "  scattering:",
        tuple((k, case["scattering"][k]) for k in case["critical"]),
    )
    print(
        "  residual:",
        tuple((k, case["residual"][k]) for k in case["critical"]),
    )

print("QUINTIC CARRY BLOCK alpha^4 beta")
print("  profiles:", quintic_geometry["profiles"])
print("  extremal invariants:", -QQ(1428), -QQ(2970))
print("  GKT:", quintic_gkt[quintic_mask])
print("  scattering:", quintic_scattering[quintic_mask])
print("ALL FOUR-LABEL GKT/JACOBI OPERATION CHECKS PASSED")
