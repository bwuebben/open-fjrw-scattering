"""
Exact checks for the labelled critical-algebra embedding (Theorem 5.1) and
the statement after Theorem 5.1 that transport along the ordered-residue
cover is not induced by a change of twists.

Run from the paper directory with:
    sage src/check_gkt_labelled_transport.sage
"""

from itertools import permutations as iter_permutations
from itertools import product as iter_product

from sage.all import *


quartic_order = 4
kappa = vector(ZZ, (1, 1))
residue_ring = Integers(quartic_order)
residue_generator = matrix(residue_ring, [[2, 1], [3, 0]])


def critical_exponents(twists, descendants):
    """GKT Definition 4.23 exponents for one labelled coefficient block."""
    assert len(twists) == len(descendants)
    assert twists
    assert all(0 <= a < 4 and 0 <= b < 4 for a, b in twists)
    assert all(d >= 0 for d in descendants)

    first_twist_sum = sum(a for a, _ in twists)
    second_twist_sum = sum(b for _, b in twists)
    first_residue = first_twist_sum % 4
    second_residue = second_twist_sum % 4
    critical_weight = sum(
        a + b + 4 * (d - 1)
        for (a, b), d in zip(twists, descendants)
    )

    exponents = []
    for first_exponent in range(max(critical_weight, 0) + 1):
        second_exponent = critical_weight - first_exponent
        exponent = vector(ZZ, (first_exponent, second_exponent))
        if min(exponent) < 0 or exponent == 0:
            continue
        if (
            first_exponent % 4 == first_residue
            and second_exponent % 4 == second_residue
        ):
            exponents.append(exponent)
    return exponents


def critical_block_rank_formula(twists, descendants):
    """Closed count obtained by separating the two residue carries."""
    first_twist_sum = sum(a for a, _ in twists)
    second_twist_sum = sum(b for _, b in twists)
    first_residue = first_twist_sum % 4
    second_residue = second_twist_sum % 4
    carry_degree = (
        (first_twist_sum - first_residue) // 4
        + (second_twist_sum - second_residue) // 4
        + sum(d - 1 for d in descendants)
    )
    if carry_degree < 0:
        return 0
    if first_residue == 0 and second_residue == 0 and carry_degree == 0:
        return 0
    return carry_degree + 1


def hamiltonian_bracket(first_k, second_k):
    """Coefficient and GKT output exponent under q=k+kappa."""
    first_q = first_k + kappa
    second_q = second_k + kappa
    coefficient = -matrix(ZZ, [first_q, second_q]).det()
    return coefficient, first_k + second_k


# Exhaust the small finite-labelled blocks used by the paper.  The count is
# invariant under relabelling, and every listed exponent satisfies the two
# residue conditions and the quartic Euler equation exactly.
twist_types = list(iter_product(range(4), repeat=2))
checked_blocks = 0
for marking_count in (1, 2, 3):
    for twists in iter_product(twist_types, repeat=marking_count):
        for descendants in iter_product(range(3), repeat=marking_count):
            exponents = critical_exponents(twists, descendants)
            assert len(exponents) == critical_block_rank_formula(
                twists, descendants
            )
            critical_weight = sum(
                a + b + 4 * (d - 1)
                for (a, b), d in zip(twists, descendants)
            )
            total_residue = vector(
                residue_ring,
                (
                    sum(a for a, _ in twists),
                    sum(b for _, b in twists),
                ),
            )
            for exponent in exponents:
                assert sum(exponent) == critical_weight
                assert vector(residue_ring, exponent) == total_residue

            for relabelling in iter_permutations(range(marking_count)):
                relabelled_twists = tuple(twists[index] for index in relabelling)
                relabelled_descendants = tuple(
                    descendants[index] for index in relabelling
                )
                assert critical_exponents(
                    relabelled_twists, relabelled_descendants
                ) == exponents
            checked_blocks += 1


# Disjoint labelled supports multiply.  The GKT critical equations are
# additive, so every possibly nonzero bracket lands in the union block.
bracket_checks = 0
small_blocks = [
    (((1, 0),), (1,)),
    (((0, 1),), (1,)),
    (((2, 3),), (1,)),
    (((3, 2),), (1,)),
    (((1, 0), (0, 1)), (1, 1)),
    (((2, 0),), (2,)),
]
for left_block in small_blocks:
    for right_block in small_blocks:
        left_twists, left_descendants = left_block
        right_twists, right_descendants = right_block
        union_twists = left_twists + right_twists
        union_descendants = left_descendants + right_descendants
        union_exponents = critical_exponents(union_twists, union_descendants)
        for left_exponent in critical_exponents(*left_block):
            for right_exponent in critical_exponents(*right_block):
                coefficient, output_exponent = hamiltonian_bracket(
                    left_exponent, right_exponent
                )
                assert output_exponent in union_exponents
                assert coefficient == -matrix(
                    ZZ,
                    [left_exponent + kappa, right_exponent + kappa],
                ).det()
                bracket_checks += 1


# Every lifted peripheral has residue P, while the core has residue P^2.
# Simultaneous transport of q, r, and kappa preserves both the determinant
# coefficient and the shifted bracket output.  This is exact SL_2 covariance,
# not an identification with the standard nonnegative GKT critical block in
# the target frame.
lifted_peripherals = {
    "A": matrix(ZZ, [[-2, 9], [-1, 4]]),
    "B_0": matrix(ZZ, [[2, 1], [-1, 0]]),
    "B_1": matrix(ZZ, [[-14, 25], [-9, 16]]),
    "C": matrix(ZZ, [[-2, 1], [-9, 4]]),
}
annulus_core = matrix(ZZ, [[-1, 2], [-10, 19]])
assert all(
    matrix(residue_ring, monodromy) == residue_generator
    for monodromy in lifted_peripherals.values()
)
assert matrix(residue_ring, annulus_core) == residue_generator**2

transport_sample = [
    vector(ZZ, (1, 1)),
    vector(ZZ, (2, 1)),
    vector(ZZ, (1, 2)),
    vector(ZZ, (3, 2)),
]
transport_checks = 0
for monodromy in list(lifted_peripherals.values()) + [annulus_core]:
    transported_kappa = monodromy * kappa
    for first_q in transport_sample:
        for second_q in transport_sample:
            assert matrix(
                ZZ, [monodromy * first_q, monodromy * second_q]
            ).det() == matrix(ZZ, [first_q, second_q]).det()
            assert monodromy * (first_q + second_q - kappa) == (
                monodromy * first_q
                + monodromy * second_q
                - transported_kappa
            )
            transport_checks += 1


# The four singleton residues are all legitimate d=1 GKT types, and each
# standard singleton block has rank one.
singleton_orbit = []
singleton_twist = vector(residue_ring, (1, 0))
for orbit_index in range(4):
    singleton_orbit.append(tuple(ZZ(entry) for entry in singleton_twist))
    singleton_twist = residue_generator * singleton_twist
assert singleton_orbit == [(1, 0), (2, 3), (3, 2), (0, 1)]
assert all(
    critical_exponents((twist,), (1,)) == [vector(ZZ, twist)]
    for twist in singleton_orbit
)


# Sharp NO-GO for a P-linearization of the full standard critical functor.
# Two primary alpha markings have one critical direction.  Retwisting both
# labels by P produces three.  A coefficient-preserving isomorphism cannot
# identify vector spaces of dimensions one and three.
two_alpha_twists = ((1, 0), (1, 0))
two_transported_twists = tuple(
    tuple(ZZ(entry) for entry in residue_generator * vector(residue_ring, twist))
    for twist in two_alpha_twists
)
two_primary_descendants = (1, 1)
two_alpha_block = critical_exponents(
    two_alpha_twists, two_primary_descendants
)
two_transported_block = critical_exponents(
    two_transported_twists, two_primary_descendants
)
assert two_transported_twists == ((2, 3), (2, 3))
assert two_alpha_block == [vector(ZZ, (2, 0))]
assert two_transported_block == [
    vector(ZZ, (0, 10)),
    vector(ZZ, (4, 6)),
    vector(ZZ, (8, 2)),
]
assert len(two_alpha_block) == 1
assert len(two_transported_block) == 3

# Exact peripheral transport of the source exponent is a Laurent/Jacobi
# exponent of the correct residue, but never one of the three standard
# nonnegative exponents of the retwisted block.  This is why the valid comparison must
# retain the path frame.
for monodromy in lifted_peripherals.values():
    transported_exponent = monodromy * two_alpha_block[0]
    assert vector(residue_ring, transported_exponent) == vector(
        residue_ring, (0, 2)
    )
    assert transported_exponent not in two_transported_block


print("finite GKT critical blocks checked:", checked_blocks)
print("disjoint-support bracket checks:", bracket_checks)
print("exact shifted-Hamiltonian transport checks:", transport_checks)
print("GKT singleton twist orbit:", singleton_orbit)
print("two-alpha critical block:", two_alpha_block)
print("P-retwisted two-alpha critical block:", two_transported_block)
print("change of twists by P_res:", "critical exponents 1 != 3")
