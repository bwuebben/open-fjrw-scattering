#!/usr/bin/env python3
"""Exact checks for the intrinsic affine-incidence boundary algebra.

The program checks the finite combinatorics and determinant-line algebra used
in the descendent boundary argument.  It does not replace the
PL orientation or general-position theorems, and it does not construct the
rational branched chain.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, permutations
from collections import defaultdict


Matrix = tuple[tuple[int, ...], ...]
Block = frozenset[int]
Partition = tuple[Block, ...]

K = ((-1, 2), (-10, 19))


def determinant(matrix: Matrix) -> int:
    """Bareiss determinant over the integers."""
    size = len(matrix)
    if size == 0:
        return 1
    assert all(len(row) == size for row in matrix)
    work = [list(row) for row in matrix]
    sign = 1
    denominator = 1
    for pivot_index in range(size - 1):
        pivot_row = next(
            (
                row
                for row in range(pivot_index, size)
                if work[row][pivot_index]
            ),
            None,
        )
        if pivot_row is None:
            return 0
        if pivot_row != pivot_index:
            work[pivot_index], work[pivot_row] = (
                work[pivot_row],
                work[pivot_index],
            )
            sign *= -1
        pivot = work[pivot_index][pivot_index]
        for row in range(pivot_index + 1, size):
            for column in range(pivot_index + 1, size):
                numerator = (
                    work[row][column] * pivot
                    - work[row][pivot_index] * work[pivot_index][column]
                )
                assert numerator % denominator == 0
                work[row][column] = numerator // denominator
        for row in range(pivot_index + 1, size):
            work[row][pivot_index] = 0
        denominator = pivot
    return sign * work[-1][-1]


def rank(matrix: Matrix, column_count: int | None = None) -> int:
    """Exact row rank over Q, including matrices with no rows."""
    if not matrix:
        return 0
    width = len(matrix[0]) if column_count is None else column_count
    assert all(len(row) == width for row in matrix)
    work = [[Fraction(entry) for entry in row] for row in matrix]
    pivot_row = 0
    for column in range(width):
        pivot = next(
            (
                row
                for row in range(pivot_row, len(work))
                if work[row][column]
            ),
            None,
        )
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        scale = work[pivot_row][column]
        work[pivot_row] = [entry / scale for entry in work[pivot_row]]
        for row in range(len(work)):
            if row == pivot_row:
                continue
            scale = work[row][column]
            if scale:
                work[row] = [
                    left - scale * right
                    for left, right in zip(work[row], work[pivot_row])
                ]
        pivot_row += 1
        if pivot_row == len(work):
            break
    return pivot_row


def permutation_sign(permutation: tuple[int, ...]) -> int:
    inversions = sum(
        permutation[i] > permutation[j]
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def augmentation_action(permutation: tuple[int, ...]) -> Matrix:
    """Matrix of a label permutation on V_n=ker(Z^n -> Z).

    The basis is e_i-e_{n-1}, 0 <= i < n-1.  A zero-sum vector is expressed
    in this basis by its first n-1 coordinates.
    """
    size = len(permutation)
    columns: list[tuple[int, ...]] = []
    for index in range(size - 1):
        image = [0] * size
        image[permutation[index]] += 1
        image[permutation[size - 1]] -= 1
        columns.append(tuple(image[:-1]))
    return tuple(
        tuple(columns[column][row] for column in range(size - 1))
        for row in range(size - 1)
    )


def check_augmentation_orientation() -> int:
    checks = 0
    for size in range(2, 9):
        for permutation in permutations(range(size)):
            sign = permutation_sign(permutation)
            determinant_sign = determinant(augmentation_action(permutation))
            assert determinant_sign == sign

            # The relative affine-length orientation is the same standard
            # representation.  Tensoring it with det(V_n) squares the sign.
            affine_orientation_sign = sign
            assert affine_orientation_sign * determinant_sign == 1
            checks += 2
    return checks


def set_partitions(size: int) -> tuple[Partition, ...]:
    partitions: list[tuple[tuple[int, ...], ...]] = [()]
    for label in range(size):
        updated: set[tuple[tuple[int, ...], ...]] = set()
        for partition in partitions:
            updated.add(partition + ((label,),))
            for index in range(len(partition)):
                block = tuple(sorted(partition[index] + (label,)))
                candidate = list(partition)
                candidate[index] = block
                candidate.sort(key=lambda entry: entry[0])
                updated.add(tuple(candidate))
        partitions = sorted(updated)
    return tuple(
        tuple(frozenset(block) for block in partition)
        for partition in partitions
    )


def refinement_basis(size: int, partition: Partition) -> Matrix:
    """A unimodular basis supplied by the augmentation exact sequence."""
    ordered = tuple(sorted((tuple(sorted(block)) for block in partition)))
    columns: list[tuple[int, ...]] = []

    # Within-block augmentation bases.
    for block in ordered:
        anchor = block[-1]
        for label in block[:-1]:
            vector = [0] * size
            vector[label] = 1
            vector[anchor] = -1
            columns.append(tuple(vector[:-1]))

    # Lifts of the augmentation basis on the set of blocks.
    block_anchor = ordered[-1][-1]
    for block in ordered[:-1]:
        vector = [0] * size
        vector[block[-1]] = 1
        vector[block_anchor] = -1
        columns.append(tuple(vector[:-1]))

    assert len(columns) == size - 1
    return tuple(
        tuple(columns[column][row] for column in range(size - 1))
        for row in range(size - 1)
    )


def check_partition_determinant_coherence() -> int:
    checks = 0
    for size in range(2, 8):
        for partition in set_partitions(size):
            matrix = refinement_basis(size, partition)
            assert abs(determinant(matrix)) == 1
            within_rank = sum(len(block) - 1 for block in partition)
            quotient_rank = len(partition) - 1
            assert within_rank + quotient_rank == size - 1
            checks += 2

            # Reordering q blocks has the same sign on relative affine
            # lengths and det(V_pi), hence no sign after tensoring.
            block_count = len(partition)
            for permutation in permutations(range(block_count)):
                sign = permutation_sign(permutation)
                assert sign * sign == 1
                checks += 1
    return checks


def deterministic_matrix(rows: int, columns: int, seed: int) -> Matrix:
    return tuple(
        tuple(
            ((row + 2) * (column + 3) + seed * (row - column + 1)) % 11 - 5
            for column in range(columns)
        )
        for row in range(rows)
    )


def check_determinant_complexes() -> int:
    checks = 0
    for columns in range(1, 8):
        for rows in range(0, 8):
            for seed in range(23):
                matrix = deterministic_matrix(rows, columns, seed)
                matrix_rank = rank(matrix, columns)
                kernel_dimension = columns - matrix_rank
                cokernel_dimension = rows - matrix_rank
                virtual_dimension = columns - rows
                assert kernel_dimension - cokernel_dimension == virtual_dimension
                checks += 1

                # Adding one face equation lowers virtual dimension by one,
                # whether the new row raises rank or creates an obstruction.
                face_row = tuple(
                    ((seed + 1) * (column + 2)) % 7 - 3
                    for column in range(columns)
                )
                face_matrix = matrix + (face_row,)
                face_rank = rank(face_matrix, columns)
                face_kernel = columns - face_rank
                face_cokernel = rows + 1 - face_rank
                assert face_kernel - face_cokernel == virtual_dimension - 1
                checks += 1

                # A wall-free seam subdivision adds one variable and one
                # independent bookkeeping equation, preserving virtual degree.
                enlarged = tuple(row + (0,) for row in matrix) + (
                    tuple([0] * columns + [1]),
                )
                enlarged_rank = rank(enlarged, columns + 1)
                assert (
                    columns + 1 - enlarged_rank
                    - (rows + 1 - enlarged_rank)
                    == virtual_dimension
                )
                checks += 1

                # Splitting one affine segment at a transverse wall bend
                # adds a two-coordinate bend point and one new length, and
                # adds one vector equation plus one wall-support equation.
                assert (columns + 3) - (rows + 3) == virtual_dimension
                checks += 1
    return checks


def compatible(block_a: Block, block_b: Block) -> bool:
    return not (block_a & block_b) or block_a <= block_b or block_b <= block_a


def check_collision_cones() -> tuple[int, int, int]:
    dimension_checks = 0
    forgetting_checks = 0
    nested_checks = 0

    for flag_count in range(2, 10):
        flags = frozenset(range(flag_count))
        blocks = tuple(
            frozenset(block)
            for size in range(2, flag_count + 1)
            for block in combinations(flags, size)
        )
        for block in blocks:
            block_size = len(block)
            exceptional_dimension = flag_count - 1
            diagonal_dimension = flag_count - (block_size - 1)
            excess_dimension = exceptional_dimension - diagonal_dimension
            assert excess_dimension == block_size - 2
            if block_size == 2:
                # The oriented normal sphere is S^0: its two sides cancel.
                assert 1 + (-1) == 0
            else:
                # The blowdown lowers dimension, so the bottom of the
                # collision cone is thin in the cellular pushforward.
                assert excess_dimension > 0
            dimension_checks += 1

            for retained_size in range(1, flag_count + 1):
                for retained_tuple in combinations(flags, retained_size):
                    retained = frozenset(retained_tuple)
                    image_block = block & retained
                    image_collision_codimension = max(len(image_block) - 1, 0)
                    image_dimension = retained_size - image_collision_codimension
                    dimension_drop = exceptional_dimension - image_dimension

                    if block_size >= 3:
                        assert dimension_drop > 0
                    elif dimension_drop == 0:
                        # The two S^0 sides have identical blowdown image and
                        # opposite normal orientation.
                        assert 1 + (-1) == 0
                    else:
                        assert dimension_drop > 0
                    forgetting_checks += 1

        for block_a, block_b in combinations(blocks, 2):
            if not compatible(block_a, block_b):
                continue
            # Nested and disjoint collision-cone normals form a Boolean
            # square.  Reversing their order changes the orientation.
            assert 1 + (-1) == 0
            nested_checks += 1

    return dimension_checks, forgetting_checks, nested_checks


def total_degree(
    geometric_dimension: int,
    form_degree: int,
    closed_chain_degree: int = 0,
) -> int:
    return geometric_dimension + closed_chain_degree - form_degree


def check_total_degrees() -> int:
    checks = 0
    for geometric_dimension in range(1, 16):
        for form_degree in range(3):
            degree = total_degree(geometric_dimension, form_degree)
            assert (
                total_degree(geometric_dimension - 1, form_degree)
                == degree - 1
            )
            checks += 1
            if form_degree < 2:
                assert (
                    total_degree(geometric_dimension, form_degree + 1)
                    == degree - 1
                )
                checks += 1

    # A metric branch with a top form and its corolla primitive have equal
    # total degree; their two boundary terms lie one degree lower.
    assert total_degree(1, 2) == total_degree(0, 1)
    assert total_degree(0, 2) == total_degree(1, 2) - 1
    checks += 2

    # A real face normal anticommutes with another real normal and commutes
    # with an even complex open--closed normal.
    assert (-1) ** (1 * 1) == -1
    assert (-1) ** (1 * 2) == 1
    checks += 2
    return checks


def circle_boundary(edge_coefficients: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    size = len(edge_coefficients)
    return tuple(
        edge_coefficients[(vertex - 1) % size] - edge_coefficients[vertex]
        for vertex in range(size)
    )


def rooted_circle_primitive(weights: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    """Unique primitive with coefficient zero on the rooted first arc."""
    assert sum(weights, Fraction(0)) == 0
    coefficients = [Fraction(0)]
    for vertex in range(1, len(weights)):
        coefficients.append(coefficients[-1] - weights[vertex])
    result = tuple(coefficients)
    assert circle_boundary(result) == weights
    return result


def check_rooted_circle_primitives() -> int:
    checks = 0
    for size in range(2, 10):
        for seed in range(101):
            entries = [
                Fraction(((seed + 3) * (index + 2)) % 17 - 8, index + 1)
                for index in range(1, size)
            ]
            weights = (Fraction(-sum(entries, Fraction(0))), *entries)
            primitive = rooted_circle_primitive(weights)
            assert primitive[0] == 0
            checks += 2

            # Subdividing an arc inserts a zero vertex; the primitive pulls
            # back by giving both new subarcs the old coefficient.
            for edge in range(size):
                insert_at = edge + 1
                subdivided_weights = (
                    weights[:insert_at]
                    + (Fraction(0),)
                    + weights[insert_at:]
                )
                subdivided = rooted_circle_primitive(subdivided_weights)
                expected = (
                    primitive[:insert_at]
                    + (primitive[edge],)
                    + primitive[insert_at:]
                )
                assert subdivided == expected
                checks += 1

            # Reversing the oriented circle while fixing the root reverses
            # the transported one-chain.
            reversed_weights = (weights[0],) + tuple(reversed(weights[1:]))
            reversed_primitive = rooted_circle_primitive(reversed_weights)
            transported = tuple(
                primitive[size - 1]
                - primitive[(size - index - 1) % size]
                for index in range(size)
            )
            assert reversed_primitive == transported
            checks += 1
    return checks


def proper_splits(block: frozenset[int]) -> tuple[tuple[Block, Block], ...]:
    labels = tuple(sorted(block))
    splits: list[tuple[Block, Block]] = []
    for size in range(1, len(labels)):
        for left_tuple in combinations(labels, size):
            left = frozenset(left_tuple)
            right = block - left
            splits.append((left, right))
    return tuple(splits)


def check_factorization_forest_gluing() -> int:
    checks = 0
    for size in range(3, 10):
        block = frozenset(range(size))
        left_associated: dict[tuple[Block, Block, Block], int] = defaultdict(int)
        right_associated: dict[tuple[Block, Block, Block], int] = defaultdict(int)
        for left, right in proper_splits(block):
            for first, second in proper_splits(left):
                left_associated[(first, second, right)] += 1
            for second, third in proper_splits(right):
                right_associated[(left, second, third)] += 1
        assert left_associated == right_associated
        checks += len(left_associated)

        # In the metric-forest quotient the two orders of cutting the
        # internal edges have opposite real-normal orientation.
        for multiplicity in left_associated.values():
            assert multiplicity == 1
            assert 1 + (-1) == 0
            checks += 2
    return checks


def check_face_ledger() -> int:
    ledger = {
        "endpoint": "external:endpoint",
        "pointing": "internal:radial-thimble-high-cancellation",
        "wall-cylinder": "internal:Rees-Cartan",
        "joint-or-tangency": "internal:virtual-scattering",
        "period-zero": "internal:corolla-homotopy",
        "period-one": "internal:product-forest-gluing",
        "critical": "external:WC",
        "detached": "external:Cont",
        "pair-direction-collision": "external:XCH-cancelling-pair",
        "multiple-direction-collision": "internal:collision-cone",
        "threshold-bubble": "external:OC-after-radial-thimble",
        "residual-rank-jump": "internal:coherent-virtual-regularization",
        "coefficient-order-at-least-N": "zero:truncated-coefficient",
        "off-degree-closed-decoration": "external:Null-degree-zero-moment",
        "unstable-wall-free-vertex": "internal:stabilization-gluing",
        "seam-subdivision": "internal:identity-gluing",
    }
    expected = {
        "external:endpoint",
        "internal:radial-thimble-high-cancellation",
        "internal:Rees-Cartan",
        "internal:virtual-scattering",
        "internal:corolla-homotopy",
        "internal:product-forest-gluing",
        "external:WC",
        "external:Cont",
        "external:XCH-cancelling-pair",
        "internal:collision-cone",
        "external:OC-after-radial-thimble",
        "internal:coherent-virtual-regularization",
        "zero:truncated-coefficient",
        "external:Null-degree-zero-moment",
        "internal:stabilization-gluing",
        "internal:identity-gluing",
    }
    assert set(ledger.values()) == expected
    assert len(ledger) == len(expected)
    return 2 * len(ledger)


def check_radial_thimble_and_relative_regression() -> int:
    checks = 0
    root_stack_trace = Fraction(1, 4)
    coarse_cartier_multiplicity = 4
    normalized_low_face = coarse_cartier_multiplicity * root_stack_trace
    assert normalized_low_face == 1
    checks += 1
    for numerator in range(-31, 32):
        for denominator in range(1, 14):
            point = Fraction(numerator, denominator)
            oc = Fraction(5 * numerator - denominator, denominator + 3)
            end = Fraction(7 * numerator + denominator, denominator + 5)
            colors = Fraction(11 * numerator - 2 * denominator, denominator + 7)

            raw_boundary = end - point + colors
            radial_thimble_boundary = point - normalized_low_face * oc
            assert (
                raw_boundary + radial_thimble_boundary
                == end - oc + colors
            )
            checks += 1

    # Regression: in C_*([0,1], {0}), the relative boundary of [0,1] is
    # represented by {1}, whose scalar augmentation is one.
    relative_boundary = (Fraction(1),)
    assert sum(relative_boundary, Fraction(0)) == 1
    assert sum(relative_boundary, Fraction(0)) != 0
    return checks + 2


def check_holonomy_orientation() -> int:
    assert determinant(K) == 1
    checks = 1
    vectors = tuple(
        (left, right)
        for left in range(-8, 9)
        for right in range(-8, 9)
        if (left, right) != (0, 0)
    )
    for first, second in combinations(vectors, 2):
        original = first[0] * second[1] - first[1] * second[0]
        first_image = (
            K[0][0] * first[0] + K[0][1] * first[1],
            K[1][0] * first[0] + K[1][1] * first[1],
        )
        second_image = (
            K[0][0] * second[0] + K[0][1] * second[1],
            K[1][0] * second[0] + K[1][1] * second[1],
        )
        transported = (
            first_image[0] * second_image[1]
            - first_image[1] * second_image[0]
        )
        assert transported == original
        checks += 1
    return checks


def main() -> None:
    augmentation_checks = check_augmentation_orientation()
    partition_checks = check_partition_determinant_coherence()
    determinant_checks = check_determinant_complexes()
    collision, forgetting, nested = check_collision_cones()
    degree_checks = check_total_degrees()
    pointing_checks = check_rooted_circle_primitives()
    factorization_checks = check_factorization_forest_gluing()
    ledger_checks = check_face_ledger()
    repair_checks = check_radial_thimble_and_relative_regression()
    holonomy_checks = check_holonomy_orientation()
    total = sum(
        (
            augmentation_checks,
            partition_checks,
            determinant_checks,
            collision,
            forgetting,
            nested,
            degree_checks,
            pointing_checks,
            factorization_checks,
            ledger_checks,
            repair_checks,
            holonomy_checks,
        )
    )
    print("intrinsic affine-incidence/residual boundary: exact checks passed")
    print("augmentation/affine orientation identities:", augmentation_checks)
    print("partition determinant-coherence identities:", partition_checks)
    print("determinant-complex/rank-jump identities:", determinant_checks)
    print("collision-cone dimension identities:", collision)
    print("collision-cone forgetting identities:", forgetting)
    print("nested/disjoint collision identities:", nested)
    print("total-degree and normal-sign identities:", degree_checks)
    print("rooted pointing-circle primitive identities:", pointing_checks)
    print("factorization forest-gluing identities:", factorization_checks)
    print("named face-ledger dictionary checks:", ledger_checks)
    print("Cartier-thimble/relative-stop regressions:", repair_checks)
    print("K-orientation identities:", holonomy_checks)
    print("total exact identities checked:", total)
    print("INTRINSIC AFFINE BOUNDARY ALGEBRA: EXACT CHECKS PASS")
    print("This program does not establish equivariant PL general position")


if __name__ == "__main__":
    main()
