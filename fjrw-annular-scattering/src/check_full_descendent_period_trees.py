"""Exact certificate for full-descendent quartic period forests.

The inputs are finite sets of distinct GKT markings with Neveu--Schwarz
twists (a_i,b_i) in {0,1,2}^2 and arbitrary descendent degrees d_i >= 0.
The script checks:

1. the balanced/critical dimension equations and d_GKT = -N - 1;
2. singleton-sourced inversion of every active endpoint system;
3. endpoint independence of every inactive forest output;
4. the arbitrary-descendent Fermat contraction identity;
5. Gamma-weight annihilation of every critical incidence column;
6. cofactor-state interleaving and homogeneous affine transport; and
7. closure of active supports under critical actions and brackets; and
8. the mixed-sign obstruction to the naive polarization lift beyond the
   two-line sector; and
9. the exact, partition-coherent repair by the leaf-normalized invertible
   factorization line.

The only active sources are the actual GKT singleton targets (-1)^d.  No open
endpoint coefficient or closed extended correlator is used as an input.  The
descendent-one alpha/beta specialization also recovers the fixed endpoint
values from the existing GKT/Jacobi certificates.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import combinations


Datum = tuple[int, int, int]
Exponent = tuple[int, int]

ALPHA: Datum = (1, 0, 1)
BETA: Datum = (0, 1, 1)
FERMAT_X: Exponent = (4, 0)
FERMAT_Y: Exponent = (0, 4)
COFACTOR = ((-2, 0), (1, 1))
CORE_LINEAR = ((-1, 2), (-10, 19))
CORE_TRANSLATION = (8, 8)


@lru_cache(maxsize=None)
def set_partitions(items: tuple[int, ...]) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if not items:
        return ((),)
    first = items[0]
    output: list[tuple[tuple[int, ...], ...]] = []
    for partition in set_partitions(items[1:]):
        output.append(((first,),) + partition)
        for position in range(len(partition)):
            enlarged = list(partition)
            enlarged[position] = (first,) + enlarged[position]
            output.append(tuple(enlarged))
    return tuple(output)


def product(values: tuple[Fraction, ...]) -> Fraction:
    output = Fraction(1)
    for value in values:
        output *= value
    return output


def add_exponents(exponents: tuple[Exponent, ...]) -> Exponent:
    return (
        sum(exponent[0] for exponent in exponents),
        sum(exponent[1] for exponent in exponents),
    )


def matrix_vector(matrix, vector) -> tuple[Fraction, Fraction]:
    return (
        Fraction(matrix[0][0]) * vector[0] + Fraction(matrix[0][1]) * vector[1],
        Fraction(matrix[1][0]) * vector[0] + Fraction(matrix[1][1]) * vector[1],
    )


def pochhammer_quarter(numerator: int, steps: int) -> Fraction:
    value = Fraction(1)
    for step in range(steps):
        value *= Fraction(numerator + 4 * step, 4)
    return value


def gamma_weight(exponent: Exponent, residue: Exponent) -> Fraction:
    differences = tuple(exponent[i] - residue[i] for i in range(2))
    assert all(difference >= 0 and difference % 4 == 0 for difference in differences)
    return pochhammer_quarter(residue[0] + 1, differences[0] // 4) * (
        pochhammer_quarter(residue[1] + 1, differences[1] // 4)
    )


def subsets(size: int):
    for cardinality in range(1, size + 1):
        yield from combinations(range(size), cardinality)


def geometry(data: tuple[Datum, ...], subset: tuple[int, ...]) -> dict[str, object]:
    total_a = sum(data[index][0] for index in subset)
    total_b = sum(data[index][1] for index in subset)
    total_d = sum(data[index][2] for index in subset)
    residue = (total_a % 4, total_b % 4)
    ell = (total_a - residue[0]) // 4
    emm = (total_b - residue[1]) // 4
    number = ell + emm - len(subset) + 1 + total_d
    gkt_d = len(subset) - ell - emm - total_d - 2
    assert gkt_d == -number - 1
    active = number >= 0
    profiles: tuple[Exponent, ...] = ()
    critical: tuple[Exponent, ...] = ()
    if active:
        profiles = tuple(
            (residue[0] + 4 * h, residue[1] + 4 * (number - h))
            for h in range(number + 1)
        )
        critical = tuple(
            (residue[0] + 4 * h, residue[1] + 4 * (number - 1 - h))
            for h in range(number)
        )

        # GKT Notation 2.29 and Propositions 2.30/2.37.
        gkt_m = 16 + sum(
            4 * data[index][0]
            + 4 * data[index][1]
            + 16 * (data[index][2] - 1)
            for index in subset
        )
        assert all(4 * sum(profile) == gkt_m for profile in profiles)
        assert all(4 * sum(exponent) == gkt_m - 16 for exponent in critical)

    return {
        "total_a": total_a,
        "total_b": total_b,
        "total_d": total_d,
        "residue": residue,
        "ell": ell,
        "emm": emm,
        "number": number,
        "gkt_d": gkt_d,
        "active": active,
        "profiles": profiles,
        "critical": critical,
    }


def endpoint_exponent(
    data: tuple[Datum, ...], subset: tuple[int, ...], endpoint: str
) -> Exponent:
    block = geometry(data, subset)
    assert block["active"]
    profiles = block["profiles"]
    assert endpoint in ("minimum", "maximum")
    return profiles[0] if endpoint == "minimum" else profiles[-1]  # type: ignore[index]


def active_target(data: tuple[Datum, ...], subset: tuple[int, ...]) -> Fraction:
    assert geometry(data, subset)["active"]
    if len(subset) == 1:
        return Fraction((-1) ** data[subset[0]][2])
    return Fraction(0)


def admissible_partitions(data: tuple[Datum, ...], subset: tuple[int, ...]):
    for partition in set_partitions(subset):
        if len(partition) < 2:
            continue
        if all(geometry(data, block)["active"] for block in partition):
            yield partition


def endpoint_solver(data: tuple[Datum, ...], endpoint: str):
    @lru_cache(maxsize=None)
    def coefficient(subset: tuple[int, ...]) -> Fraction:
        block = geometry(data, subset)
        assert block["active"]
        residue = block["residue"]
        parent = endpoint_exponent(data, subset, endpoint)
        parent_weight = gamma_weight(parent, residue)  # type: ignore[arg-type]
        value = active_target(data, subset) / parent_weight
        for partition in admissible_partitions(data, subset):
            child_exponent = add_exponents(
                tuple(endpoint_exponent(data, child, endpoint) for child in partition)
            )
            value -= gamma_weight(child_exponent, residue) / parent_weight * product(  # type: ignore[arg-type]
                tuple(coefficient(child) for child in partition)
            )
        return value

    return coefficient


def polarization_degree(exponent: Exponent) -> int:
    return 4 * sum(exponent)


def polarization_height(exponent: Exponent) -> int:
    return 8 * max(exponent)


def homogenized(exponent: Exponent) -> tuple[Fraction, Fraction, int]:
    image = matrix_vector(COFACTOR, exponent)
    return (image[0], image[1], polarization_degree(exponent))


def factorization_shift(
    data: tuple[Datum, ...], subset: tuple[int, ...], endpoint: str
) -> int:
    """Degree of the leaf-normalized invertible factorization line."""

    parent = endpoint_exponent(data, subset, endpoint)
    leaf_height = sum(
        polarization_height(endpoint_exponent(data, (index,), endpoint))
        for index in subset
    )
    return leaf_height - polarization_height(parent) - 32 * (len(subset) - 1)


def affine_homogeneous(
    charge: tuple[Fraction, Fraction, int],
) -> tuple[Fraction, Fraction, int]:
    vector = matrix_vector(CORE_LINEAR, (charge[0], charge[1]))
    return (
        vector[0] + charge[2] * CORE_TRANSLATION[0],
        vector[1] + charge[2] * CORE_TRANSLATION[1],
        charge[2],
    )


def check_vertex(
    data: tuple[Datum, ...],
    subset: tuple[int, ...],
    partition: tuple[tuple[int, ...], ...],
    endpoint: str,
) -> int:
    parent = endpoint_exponent(data, subset, endpoint)
    children = tuple(endpoint_exponent(data, child, endpoint) for child in partition)
    child_sum = add_exponents(children)
    difference = (child_sum[0] - parent[0], child_sum[1] - parent[1])
    assert difference[0] >= 0 and difference[0] % 4 == 0
    assert difference[1] >= 0 and difference[1] % 4 == 0
    contractions = (difference[0] // 4, difference[1] // 4)
    assert sum(contractions) == len(partition) - 1

    outputs = (parent,) + (FERMAT_X,) * contractions[0] + (FERMAT_Y,) * contractions[1]
    assert add_exponents(children) == add_exponents(outputs)

    child_charge = tuple(sum(homogenized(exponent)[i] for exponent in children) for i in range(3))
    output_charge = tuple(sum(homogenized(exponent)[i] for exponent in outputs) for i in range(3))
    assert child_charge == output_charge
    child_transport = tuple(
        sum(affine_homogeneous(homogenized(exponent))[i] for exponent in children)
        for i in range(3)
    )
    output_transport = tuple(
        sum(affine_homogeneous(homogenized(exponent))[i] for exponent in outputs)
        for i in range(3)
    )
    assert child_transport == output_transport

    # The bare polarization defect can have either sign.  After tensoring the
    # block edge by its leaf-normalized invertible grading line, the vertex is
    # exact.  This identity is independent of the chosen partition and is
    # therefore strictly coherent under all refinements.
    parent_augmented_height = (
        polarization_height(parent)
        + factorization_shift(data, subset, endpoint)
        + 32 * sum(contractions)
    )
    child_augmented_height = sum(
        polarization_height(exponent)
        + factorization_shift(data, child, endpoint)
        for child, exponent in zip(partition, children)
    )
    assert parent_augmented_height == child_augmented_height
    return 7


def check_endpoint(data: tuple[Datum, ...], endpoint: str) -> tuple[dict, int]:
    coefficient = endpoint_solver(data, endpoint)
    solved: dict[tuple[int, ...], Fraction] = {}
    checks = 0
    for subset in subsets(len(data)):
        block = geometry(data, subset)
        if not block["active"]:
            continue
        solved[subset] = coefficient(subset)
        residue = block["residue"]
        parent = endpoint_exponent(data, subset, endpoint)
        period = gamma_weight(parent, residue) * solved[subset]  # type: ignore[arg-type]
        for partition in admissible_partitions(data, subset):
            child_exponent = add_exponents(
                tuple(endpoint_exponent(data, child, endpoint) for child in partition)
            )
            period += gamma_weight(child_exponent, residue) * product(  # type: ignore[arg-type]
                tuple(solved[child] for child in partition)
            )
            checks += check_vertex(data, subset, partition, endpoint)
        assert period == active_target(data, subset)
        checks += 1
    return solved, checks


def inactive_output(
    data: tuple[Datum, ...],
    solved: dict[tuple[int, ...], Fraction],
    subset: tuple[int, ...],
    endpoint: str,
) -> Fraction:
    block = geometry(data, subset)
    assert not block["active"]
    residue = block["residue"]
    value = Fraction(0)
    for partition in admissible_partitions(data, subset):
        child_exponent = add_exponents(
            tuple(endpoint_exponent(data, child, endpoint) for child in partition)
        )
        value += gamma_weight(child_exponent, residue) * product(  # type: ignore[arg-type]
            tuple(solved[child] for child in partition)
        )
    return value


def check_inactive_outputs(
    data: tuple[Datum, ...],
    minimum: dict[tuple[int, ...], Fraction],
    maximum: dict[tuple[int, ...], Fraction],
) -> int:
    checks = 0
    for subset in subsets(len(data)):
        block = geometry(data, subset)
        if block["active"]:
            continue
        assert block["gkt_d"] >= 0
        assert inactive_output(data, minimum, subset, "minimum") == inactive_output(
            data, maximum, subset, "maximum"
        )
        checks += 2
    return checks


def check_activity_closure(data: tuple[Datum, ...]) -> int:
    all_subsets = tuple(subsets(len(data)))
    checks = 0
    for left in all_subsets:
        left_set = set(left)
        left_geometry = geometry(data, left)
        for right in all_subsets:
            if left_set.intersection(right):
                continue
            right_geometry = geometry(data, right)
            union = tuple(sorted(left + right))
            union_geometry = geometry(data, union)
            carry = (
                (left_geometry["residue"][0] + right_geometry["residue"][0]) // 4  # type: ignore[index]
                + (left_geometry["residue"][1] + right_geometry["residue"][1]) // 4  # type: ignore[index]
            )
            assert union_geometry["number"] == (
                left_geometry["number"] + right_geometry["number"] + carry - 1
            )
            if left_geometry["number"] >= 1 and right_geometry["number"] >= 0:
                assert union_geometry["active"]
            if left_geometry["number"] >= 1 and right_geometry["number"] >= 1:
                assert union_geometry["number"] >= 1
            checks += 3
    return checks


def check_critical_geometry(data: tuple[Datum, ...]) -> int:
    checks = 0
    for subset in subsets(len(data)):
        block = geometry(data, subset)
        if not block["active"]:
            continue
        profiles = block["profiles"]
        critical = block["critical"]
        residue = block["residue"]
        weights = tuple(gamma_weight(profile, residue) for profile in profiles)  # type: ignore[arg-type]
        for h, exponent in enumerate(critical):  # type: ignore[union-attr]
            u, v = exponent
            assert -4 * (u + 1) * weights[h] + 4 * (v + 1) * weights[h + 1] == 0

            profile_sum = sum(profiles[h])
            assert profile_sum == sum(profiles[h + 1])
            assert sum(exponent) == profile_sum - 4
            assert profile_sum > 0
            left_x = Fraction(-profiles[h][0], 2 * profile_sum)
            right_x = Fraction(-profiles[h + 1][0], 2 * profile_sum)
            crossing_x = Fraction(-(u + 1), 2 * (profile_sum - 2))
            assert right_x < crossing_x < left_x
            checks += 5
    return checks


def check_naive_polarization_no_go() -> int:
    def minimum_vertex_defect(data: tuple[Datum, Datum]) -> int:
        subset = (0, 1)
        partition = ((0,), (1,))
        parent = endpoint_exponent(data, subset, "minimum")
        children = tuple(
            endpoint_exponent(data, child, "minimum") for child in partition
        )
        difference = (
            sum(exponent[0] for exponent in children) - parent[0],
            sum(exponent[1] for exponent in children) - parent[1],
        )
        contractions = (difference[0] // 4, difference[1] // 4)
        outputs = (
            (parent,)
            + (FERMAT_X,) * contractions[0]
            + (FERMAT_Y,) * contractions[1]
        )
        return sum(map(polarization_height, outputs)) - sum(
            map(polarization_height, children)
        )

    # Both are valid two-block active GKT vertices.  Their opposite signs
    # rule out one fixed effective orientation for the naive extension.
    positive_data = ((2, 2, 0), (2, 1, 0))
    negative_data = ((2, 1, 0), (0, 1, 2))
    assert minimum_vertex_defect(positive_data) == 24
    assert minimum_vertex_defect(negative_data) == -8
    assert factorization_shift(positive_data, (0, 1), "minimum") == -24
    assert factorization_shift(negative_data, (0, 1), "minimum") == 8
    return 4


def run_window(data: tuple[Datum, ...], two_line_regression: bool = False) -> int:
    assert all(0 <= a <= 2 and 0 <= b <= 2 and d >= 0 for a, b, d in data)
    minimum, checks_minimum = check_endpoint(data, "minimum")
    maximum, checks_maximum = check_endpoint(data, "maximum")
    checks = (
        checks_minimum
        + checks_maximum
        + check_inactive_outputs(data, minimum, maximum)
        + check_critical_geometry(data)
        + check_activity_closure(data)
    )

    if two_line_regression:
        full = tuple(range(len(data)))
        if geometry(data, full)["active"]:
            p = sum(item == ALPHA for item in data)
            q = len(data) - p
            fixed = {
                (1, 0): (Fraction(-4), Fraction(-2)),
                (0, 1): (Fraction(-2), Fraction(-4)),
                (1, 1): (Fraction(-12), Fraction(-12)),
                (4, 0): (Fraction(-168), Fraction(-495, 2)),
                (3, 1): (Fraction(-672), Fraction(-360)),
                (2, 2): (Fraction(-504), Fraction(-504)),
                (4, 1): (Fraction(-1428), Fraction(-2970)),
            }
            if (p, q) in fixed:
                assert (minimum[full], maximum[full]) == fixed[(p, q)]
                checks += 1
    return checks


def main() -> None:
    total_checks = 0

    # Actual two-line specialization, including the first carry blocks.
    for size in range(1, 7):
        for p in range(size + 1):
            total_checks += run_window((ALPHA,) * p + (BETA,) * (size - p), True)

    # Representative all-NS-twist and arbitrary-descendent windows.  The
    # first two contain inactive supports with nontrivial forest outputs.
    windows = (
        ((0, 0, 0),),
        ((0, 0, 0), (0, 0, 0), (1, 0, 0), (0, 1, 0)),
        ((2, 2, 0), (1, 0, 1), (2, 1, 0), (0, 1, 2)),
        ((1, 0, 0), (0, 1, 0), (2, 2, 1), (2, 2, 2), (0, 0, 3)),
        ((1, 0, 1), (2, 2, 1), (2, 1, 1), (0, 1, 1)),
        tuple(((2 * i + 1) % 3, (3 * i + 2) % 3, i % 3) for i in range(6)),
    )
    for window in windows:
        total_checks += run_window(window)
    total_checks += check_naive_polarization_no_go()

    print("full-descendent period forests: exact quartic checks passed")
    print("data checked: all NS twists, descendants, active targets, and inactive outputs")
    print("geometry checked: Fermat excess, Gamma incidence, state interleaving, cone balance")
    print("closure checked: critical actions and brackets stay on active supports")
    print("polarization boundary: naive full-descendent effectivity has both signs")
    print("factorization repair: leaf-normalized invertible line is exactly coherent")
    print(f"identities checked: {total_checks}")
    print("two-line regression: fixed GKT endpoints through first carry blocks passed")
    print("ALL FULL-DESCENDENT PERIOD-FOREST CHECKS PASSED")


if __name__ == "__main__":
    main()
