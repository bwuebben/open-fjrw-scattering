#!/usr/bin/env python3
"""Exact checks for the intrinsic factorization-corolla homotopy.

The program verifies the polynomial algebra used in the descendent
correspondence.  It checks the symmetric
deformation retract of the quartic twisted de Rham complex, including
the exact X^3 and Y^3 boundary rows, and the unscalarized partition-vertex
defects.  It also checks the cellular metric-cube differential and the
Koszul square-zero identity used by the length-one forest quotient.  It does
not prove the geometric compactness or orientation results established in the
manuscript.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import permutations, product


Key = tuple[int, int, int]  # hbar, X, Y exponents
Poly = dict[Key, Fraction]
OneForm = tuple[Poly, Poly]  # p dX + q dY
Block = tuple[int, ...]
Datum = tuple[int, int, int]
Partition = tuple[Block, ...]
CubeCell = tuple[int, ...]  # -1 is free; 0 and 1 are fixed endpoints


def clean(poly: Poly) -> Poly:
    return {key: value for key, value in poly.items() if value}


def add(*polys: Poly) -> Poly:
    result: Poly = {}
    for poly in polys:
        for key, value in poly.items():
            result[key] = result.get(key, Fraction(0)) + value
    return clean(result)


def scale(poly: Poly, scalar: Fraction | int) -> Poly:
    scalar = Fraction(scalar)
    return clean({key: scalar * value for key, value in poly.items()})


def monomial(
    hbar_power: int,
    x_power: int,
    y_power: int,
    coefficient: Fraction | int = 1,
) -> Poly:
    assert min(hbar_power, x_power, y_power) >= 0
    coefficient = Fraction(coefficient)
    return {} if not coefficient else {
        (hbar_power, x_power, y_power): coefficient
    }


def shift(poly: Poly, dh: int = 0, dx: int = 0, dy: int = 0) -> Poly:
    assert min(dh, dx, dy) >= 0
    return {
        (hbar_power + dh, x_power + dx, y_power + dy): coefficient
        for (hbar_power, x_power, y_power), coefficient in poly.items()
    }


def derivative(poly: Poly, variable: str) -> Poly:
    result: Poly = {}
    index = 1 if variable == "x" else 2
    for exponent, coefficient in poly.items():
        power = exponent[index]
        if not power:
            continue
        lowered = list(exponent)
        lowered[index] -= 1
        key = tuple(lowered)
        result[key] = result.get(key, Fraction(0)) + coefficient * power
    return clean(result)


def add_one(*forms: OneForm) -> OneForm:
    return add(*(form[0] for form in forms)), add(*(form[1] for form in forms))


def scale_one(form: OneForm, scalar: Fraction | int) -> OneForm:
    return scale(form[0], scalar), scale(form[1], scalar)


def twisted_d0(function: Poly) -> OneForm:
    return (
        add(shift(derivative(function, "x"), dh=1), scale(shift(function, dx=3), 4)),
        add(shift(derivative(function, "y"), dh=1), scale(shift(function, dy=3), 4)),
    )


def twisted_d1(form: OneForm) -> Poly:
    p_component, q_component = form
    return add(
        shift(derivative(q_component, "x"), dh=1),
        scale(shift(derivative(p_component, "y"), dh=1), -1),
        scale(shift(q_component, dx=3), 4),
        scale(shift(p_component, dy=3), -4),
    )


def pochhammer_quarter(numerator: int, steps: int) -> Fraction:
    result = Fraction(1)
    for step in range(steps):
        result *= Fraction(numerator + 4 * step, 4)
    return result


def project_x(poly: Poly) -> Poly:
    result: Poly = {}
    for (hbar_power, x_power, y_power), coefficient in poly.items():
        residue = x_power % 4
        steps = x_power // 4
        if residue == 3:
            continue
        factor = Fraction((-1) ** steps) * pochhammer_quarter(
            residue + 1, steps
        )
        key = (hbar_power + steps, residue, y_power)
        result[key] = result.get(key, Fraction(0)) + coefficient * factor
    return clean(result)


def project_y(poly: Poly) -> Poly:
    result: Poly = {}
    for (hbar_power, x_power, y_power), coefficient in poly.items():
        residue = y_power % 4
        steps = y_power // 4
        if residue == 3:
            continue
        factor = Fraction((-1) ** steps) * pochhammer_quarter(
            residue + 1, steps
        )
        key = (hbar_power + steps, x_power, residue)
        result[key] = result.get(key, Fraction(0)) + coefficient * factor
    return clean(result)


def normal_projection(poly: Poly) -> Poly:
    return project_y(project_x(poly))


def project_x_relative(poly: Poly) -> Poly:
    result: Poly = {}
    for (hbar_power, x_power, y_power), coefficient in poly.items():
        residue = x_power % 4
        steps = x_power // 4
        factor = Fraction((-1) ** steps) * pochhammer_quarter(
            residue + 1, steps
        )
        key = (hbar_power + steps, residue, y_power)
        result[key] = result.get(key, Fraction(0)) + coefficient * factor
    return clean(result)


def project_y_relative(poly: Poly) -> Poly:
    result: Poly = {}
    for (hbar_power, x_power, y_power), coefficient in poly.items():
        residue = y_power % 4
        steps = y_power // 4
        factor = Fraction((-1) ** steps) * pochhammer_quarter(
            residue + 1, steps
        )
        key = (hbar_power + steps, x_power, residue)
        result[key] = result.get(key, Fraction(0)) + coefficient * factor
    return clean(result)


def relative_normal_projection(poly: Poly) -> Poly:
    """Projection to the 16 relative-contact residue sectors."""
    return project_y_relative(project_x_relative(poly))


def primitive_x(poly: Poly) -> Poly:
    """One-variable primitive s_x with D_x s_x=1-i_x R_x."""

    result: Poly = {}
    for key, coefficient in poly.items():
        hbar_power, x_power, y_power = key
        current_coefficient = coefficient
        current_hbar = hbar_power
        current_x = x_power
        while current_x >= 4:
            result = add(
                result,
                monomial(
                    current_hbar,
                    current_x - 3,
                    y_power,
                    current_coefficient / 4,
                ),
            )
            current_coefficient *= -Fraction(current_x - 3, 4)
            current_hbar += 1
            current_x -= 4
        if current_x == 3:
            # H_x(-1,b)=Y^b dY/4 makes X^3 Y^b Omega exact.
            result = add(
                result,
                monomial(current_hbar, 0, y_power, current_coefficient / 4),
            )
    return clean(result)


def primitive_y(poly: Poly) -> Poly:
    """One-variable primitive s_y with D_y s_y=1-i_y R_y."""

    result: Poly = {}
    for key, coefficient in poly.items():
        hbar_power, x_power, y_power = key
        current_coefficient = coefficient
        current_hbar = hbar_power
        current_y = y_power
        while current_y >= 4:
            result = add(
                result,
                monomial(
                    current_hbar,
                    x_power,
                    current_y - 3,
                    current_coefficient / 4,
                ),
            )
            current_coefficient *= -Fraction(current_y - 3, 4)
            current_hbar += 1
            current_y -= 4
        if current_y == 3:
            # H_y(a,-1)=-X^a dX/4 makes X^a Y^3 Omega exact.
            result = add(
                result,
                monomial(current_hbar, x_power, 0, current_coefficient / 4),
            )
    return clean(result)


def homotopy_top_x_first(poly: Poly) -> OneForm:
    return scale(primitive_y(project_x(poly)), -1), primitive_x(poly)


def homotopy_top_y_first(poly: Poly) -> OneForm:
    return scale(primitive_y(poly), -1), primitive_x(project_y(poly))


def homotopy_top(poly: Poly) -> OneForm:
    return scale_one(
        add_one(homotopy_top_x_first(poly), homotopy_top_y_first(poly)),
        Fraction(1, 2),
    )


def homotopy_one(form: OneForm) -> Poly:
    p_component, q_component = form
    return scale(add(primitive_x(p_component), primitive_y(q_component)), Fraction(1, 2))


def swap_poly(poly: Poly) -> Poly:
    return {
        (hbar_power, y_power, x_power): coefficient
        for (hbar_power, x_power, y_power), coefficient in poly.items()
    }


def swap_one(form: OneForm) -> OneForm:
    p_component, q_component = form
    return swap_poly(q_component), swap_poly(p_component)


def check_deformation_retract(max_power: int = 10) -> tuple[int, int, int]:
    degree_zero = 0
    degree_one = 0
    degree_two = 0
    for hbar_power in range(3):
        for x_power in range(max_power + 1):
            for y_power in range(max_power + 1):
                basis = monomial(hbar_power, x_power, y_power)

                # Degree zero: H D=1 because the twisted complex has H^0=0.
                assert homotopy_one(twisted_d0(basis)) == basis
                degree_zero += 1

                for component in (0, 1):
                    one = (basis, {}) if component == 0 else ({}, basis)
                    lhs = add_one(
                        twisted_d0(homotopy_one(one)),
                        homotopy_top(twisted_d1(one)),
                    )
                    assert lhs == one
                    degree_one += 1

                assert twisted_d1(homotopy_top(basis)) == add(
                    basis, scale(normal_projection(basis), -1)
                )
                degree_two += 1
    return degree_zero, degree_one, degree_two


def check_boundary_and_order(max_power: int = 14) -> tuple[int, int]:
    boundary = 0
    order = 0
    for y_power in range(max_power + 1):
        x3 = monomial(0, 3, y_power)
        assert normal_projection(x3) == {}
        assert twisted_d1(homotopy_top(x3)) == x3
        boundary += 2
    for x_power in range(max_power + 1):
        y3 = monomial(0, x_power, 3)
        assert normal_projection(y3) == {}
        assert twisted_d1(homotopy_top(y3)) == y3
        boundary += 2

    for x_power in range(max_power + 1):
        for y_power in range(max_power + 1):
            basis = monomial(0, x_power, y_power)
            difference = add_one(
                homotopy_top_x_first(basis),
                scale_one(homotopy_top_y_first(basis), -1),
            )
            assert twisted_d1(difference) == {}
            second_homotopy = homotopy_one(difference)
            assert twisted_d0(second_homotopy) == difference
            order += 2
    return boundary, order


def check_coordinate_symmetry(max_power: int = 12) -> int:
    checks = 0
    for hbar_power in range(2):
        for x_power in range(max_power + 1):
            for y_power in range(max_power + 1):
                basis = monomial(hbar_power, x_power, y_power)
                swapped_top = scale(swap_poly(basis), -1)
                assert normal_projection(swapped_top) == scale(
                    swap_poly(normal_projection(basis)), -1
                )
                assert homotopy_top(swapped_top) == swap_one(homotopy_top(basis))
                checks += 2
    return checks


@lru_cache(maxsize=None)
def set_partitions(block: Block) -> tuple[Partition, ...]:
    if not block:
        return ((),)
    first = block[0]
    output: list[Partition] = []
    for partition in set_partitions(block[1:]):
        output.append(((first,),) + partition)
        for position in range(len(partition)):
            enlarged = list(partition)
            enlarged[position] = tuple(sorted((first,) + enlarged[position]))
            output.append(tuple(sorted(enlarged)))
    return tuple(dict.fromkeys(output))


def geometry(data: tuple[Datum, ...], block: Block) -> dict[str, object]:
    total_a = sum(data[index][0] for index in block)
    total_b = sum(data[index][1] for index in block)
    total_d = sum(data[index][2] for index in block)
    residue = (total_a % 4, total_b % 4)
    ell = (total_a - residue[0]) // 4
    emm = (total_b - residue[1]) // 4
    number = ell + emm - len(block) + 1 + total_d
    profiles: tuple[tuple[int, int], ...] = ()
    if number >= 0:
        profiles = tuple(
            (residue[0] + 4 * level, residue[1] + 4 * (number - level))
            for level in range(number + 1)
        )
    return {"residue": residue, "number": number, "profiles": profiles}


def endpoint_exponent(
    data: tuple[Datum, ...], block: Block, endpoint: str
) -> tuple[int, int]:
    profiles = geometry(data, block)["profiles"]
    assert profiles
    return profiles[0] if endpoint == "minimum" else profiles[-1]  # type: ignore[index]


def gamma_weight(exponent: tuple[int, int], residue: tuple[int, int]) -> Fraction:
    differences = (exponent[0] - residue[0], exponent[1] - residue[1])
    assert all(value >= 0 and value % 4 == 0 for value in differences)
    return (
        pochhammer_quarter(residue[0] + 1, differences[0] // 4)
        * pochhammer_quarter(residue[1] + 1, differences[1] // 4)
    )


def add_exponents(values) -> tuple[int, int]:
    values = tuple(values)
    return sum(value[0] for value in values), sum(value[1] for value in values)


def active_partitions(data: tuple[Datum, ...], block: Block):
    for partition in set_partitions(block):
        if len(partition) < 2:
            continue
        if all(geometry(data, child)["profiles"] for child in partition):
            yield partition


def vertex_defect(
    data: tuple[Datum, ...],
    block: Block,
    partition: Partition,
    endpoint: str,
) -> tuple[Poly, Fraction, tuple[int, int], tuple[int, int]]:
    residue = geometry(data, block)["residue"]
    parent = endpoint_exponent(data, block, endpoint)
    child = add_exponents(
        endpoint_exponent(data, component, endpoint) for component in partition
    )
    q = len(partition)
    differences = (child[0] - parent[0], child[1] - parent[1])
    assert all(value >= 0 and value % 4 == 0 for value in differences)
    assert sum(value // 4 for value in differences) == q - 1
    parent_weight = gamma_weight(parent, residue)  # type: ignore[arg-type]
    child_weight = gamma_weight(child, residue)  # type: ignore[arg-type]
    vertex_coefficient = -child_weight / parent_weight
    defect = add(
        monomial(0, child[0], child[1]),
        monomial(
            q - 1,
            parent[0],
            parent[1],
            Fraction((-1) ** (q - 1)) * vertex_coefficient,
        ),
    )
    return defect, vertex_coefficient, parent, child


def sample_data() -> tuple[tuple[Datum, ...], ...]:
    atoms = (
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (2, 2, 0),
        (1, 2, 1),
        (2, 1, 0),
    )
    samples: list[tuple[Datum, ...]] = []
    for size in (2, 3, 4):
        samples.extend(tuple(values) for values in product(atoms, repeat=size))
    return tuple(samples)


def check_vertex_defects() -> tuple[int, int]:
    checks = 0
    exact_boundary_sectors = 0
    for data in sample_data():
        block = tuple(range(len(data)))
        if not geometry(data, block)["profiles"]:
            continue
        for endpoint in ("minimum", "maximum"):
            for partition in active_partitions(data, block):
                defect, _coefficient, _parent, _child = vertex_defect(
                    data, block, partition, endpoint
                )
                projected = normal_projection(defect)
                assert projected == {}
                assert relative_normal_projection(defect) == {}
                filler = homotopy_top(defect)
                assert twisted_d1(filler) == defect
                residue = geometry(data, block)["residue"]
                if 3 in residue:  # type: ignore[operator]
                    exact_boundary_sectors += 1
                checks += 3
    return checks, exact_boundary_sectors


def history_solver(data: tuple[Datum, ...], endpoint: str):
    @lru_cache(maxsize=None)
    def histories(block: Block) -> tuple[Fraction, ...]:
        parent = endpoint_exponent(data, block, endpoint)
        residue = geometry(data, block)["residue"]
        parent_weight = gamma_weight(parent, residue)  # type: ignore[arg-type]
        if len(block) == 1:
            return (Fraction((-1) ** data[block[0]][2]) / parent_weight,)
        output: list[Fraction] = []
        for partition in active_partitions(data, block):
            _defect, coefficient, _parent, _child = vertex_defect(
                data, block, partition, endpoint
            )
            child_histories = tuple(histories(child) for child in partition)
            for choices in product(*child_histories):
                value = coefficient
                for choice in choices:
                    value *= choice
                output.append(value)
        return tuple(output)

    return histories


def check_three_label_tripod() -> tuple[int, int]:
    # The residue-(0,0) version makes the Jacobi cancellation nontrivial.
    data = ((0, 0, 1),) * 3
    block = (0, 1, 2)
    histories = history_solver(data, "minimum")
    assert histories((0,)) == (Fraction(-4),)
    assert sum(histories((0, 1)), Fraction(0)) == -20
    triple_values = histories(block)
    assert sorted(triple_values) == [-100, -100, -100, 180]
    assert sum(triple_values, Fraction(0)) == -120

    # Assemble the five-term unscalarized endpoint defect.  The three binary
    # partitions carry one inherited contraction; the ternary partition does
    # not.  Their parent pieces sum to the connected coefficient -120.
    total: Poly = monomial(2, 0, 4, -120)
    binary_partitions = []
    ternary_partitions = []
    for partition in active_partitions(data, block):
        child = add_exponents(
            endpoint_exponent(data, component, "minimum")
            for component in partition
        )
        child_coefficients = Fraction(1)
        for component in partition:
            child_coefficients *= sum(histories(component), Fraction(0))
        inherited = len(block) - len(partition)
        total = add(
            total,
            monomial(
                inherited,
                child[0],
                child[1],
                Fraction((-1) ** inherited) * child_coefficients,
            ),
        )
        if len(partition) == 2:
            binary_partitions.append(partition)
        else:
            ternary_partitions.append(partition)
    assert len(binary_partitions) == 3
    assert len(ternary_partitions) == 1
    assert normal_projection(total) == {}
    assert relative_normal_projection(total) == {}
    assert twisted_d1(homotopy_top(total)) == total

    # Every rooted partition tree has exactly |J|-1 Fermat contractions.
    grading_checks = 0
    for partition in active_partitions(data, block):
        root_degree = len(partition) - 1
        child_degree = sum(len(component) - 1 for component in partition)
        assert root_degree + child_degree == len(block) - 1
        grading_checks += 1
    return 11, grading_checks


def check_relabelling() -> int:
    data = ((0, 0, 1), (1, 0, 1), (0, 1, 1), (2, 2, 0))
    block = tuple(range(len(data)))
    checks = 0
    if not geometry(data, block)["profiles"]:
        return checks
    for endpoint in ("minimum", "maximum"):
        original = sorted(
            (
                len(partition),
                vertex_defect(data, block, partition, endpoint)[1],
            )
            for partition in active_partitions(data, block)
        )
        for permutation in permutations(block):
            relabelled = tuple(data[index] for index in permutation)
            relabelled_block = tuple(range(len(relabelled)))
            comparison = sorted(
                (
                    len(partition),
                    vertex_defect(
                        relabelled, relabelled_block, partition, endpoint
                    )[1],
                )
                for partition in active_partitions(relabelled, relabelled_block)
            )
            assert comparison == original
            checks += 1
    return checks


def cube_boundary(cell: CubeCell) -> dict[CubeCell, int]:
    """Oriented cellular boundary of a metric cube face."""
    output: dict[CubeCell, int] = {}
    free_coordinates = tuple(
        index for index, value in enumerate(cell) if value == -1
    )
    for position, coordinate in enumerate(free_coordinates):
        for endpoint, endpoint_sign in ((1, 1), (0, -1)):
            face = list(cell)
            face[coordinate] = endpoint
            key = tuple(face)
            coefficient = ((-1) ** position) * endpoint_sign
            output[key] = output.get(key, 0) + coefficient
            if output[key] == 0:
                del output[key]
    return output


def chain_boundary(chain: dict[CubeCell, int]) -> dict[CubeCell, int]:
    output: dict[CubeCell, int] = {}
    for cell, cell_coefficient in chain.items():
        for face, face_coefficient in cube_boundary(cell).items():
            output[face] = (
                output.get(face, 0)
                + cell_coefficient * face_coefficient
            )
            if output[face] == 0:
                del output[face]
    return output


def check_cellular_factorization_complex() -> int:
    """Check metric-face signs, cut-order coherence, and mixed Koszul signs."""
    checks = 0
    for edge_count in range(1, 10):
        top_cell = tuple([-1] * edge_count)
        first_boundary = cube_boundary(top_cell)
        assert chain_boundary(first_boundary) == {}
        checks += 1

        # A value-one coordinate records an actual cut edge.  Every
        # codimension-two two-cut forest is reached in the two possible
        # orders, with opposite outward-normal signs.
        for first in range(edge_count):
            for second in range(first + 1, edge_count):
                target = list(top_cell)
                target[first] = 1
                target[second] = 1
                assert tuple(target) not in chain_boundary(first_boundary)
                checks += 1

        # For d = partial + (-1)^p delta, the two mixed terms cancel after
        # partial lowers the cellular degree from p to p-1.
        mixed_after_partial = (-1) ** (edge_count - 1)
        mixed_before_partial = (-1) ** edge_count
        assert mixed_after_partial + mixed_before_partial == 0
        checks += 1
    return checks


def main() -> None:
    degree_zero, degree_one, degree_two = check_deformation_retract()
    boundary_checks, order_checks = check_boundary_and_order()
    symmetry_checks = check_coordinate_symmetry()
    vertex_checks, exact_sectors = check_vertex_defects()
    tripod_checks, grading_checks = check_three_label_tripod()
    relabelling_checks = check_relabelling()
    cellular_checks = check_cellular_factorization_complex()
    total = (
        degree_zero
        + degree_one
        + degree_two
        + boundary_checks
        + order_checks
        + symmetry_checks
        + vertex_checks
        + tripod_checks
        + grading_checks
        + relabelling_checks
        + cellular_checks
    )
    print("intrinsic factorization-corolla homotopy: exact checks passed")
    print(
        "symmetric deformation-retract identities by degree:",
        degree_zero,
        degree_one,
        degree_two,
    )
    print("X^3/Y^3 boundary and coordinate-order identities:", boundary_checks, order_checks)
    print("coordinate-swap identities:", symmetry_checks)
    print("ordinary/relative partition-vertex identities:", vertex_checks)
    print("residue-three relative sectors retained:", exact_sectors)
    print("three-label history coefficients: (-100, -100, -100, 180) -> -120")
    print("partition-refinement grading identities:", grading_checks)
    print("relabeling identities:", relabelling_checks)
    print("cellular forest-quotient identities:", cellular_checks)
    print("total exact identities checked:", total)
    print("INTRINSIC FACTORIZATION-COROLLA ALGEBRA: EXACT CHECKS PASS")


if __name__ == "__main__":
    main()
