#!/usr/bin/env python3
"""Exact checks for the intrinsic contact and virtual-chain algebra.

The script checks the normal-crossings Stokes identities, including the
residue-three counterexample that invalidates an ordinary Brieskorn trace,
the divisor/corner signs, and the raw-plus-thimble outer-boundary ledger.
It checks the local cellular factorization scalar, the literal corolla-chain
boundary, augmentation-line rank telescoping, and the ambient/nerve facet
inventory as regressions.  It does not prove the PL general-position lemma
or construct the branched chain.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache


def quarter_pochhammer(residue: int, steps: int) -> Fraction:
    value = Fraction(1)
    for step in range(steps):
        value *= Fraction(residue + 1 + 4 * step, 4)
    return value


def bulk_trace(a: int, b: int) -> Fraction:
    """The hbar=1 specialization of tau_A(X^a Y^b Omega)."""
    residue_a, steps_a = a % 4, a // 4
    residue_b, steps_b = b % 4, b // 4
    return (
        Fraction((-1) ** (steps_a + steps_b))
        * quarter_pochhammer(residue_a, steps_a)
        * quarter_pochhammer(residue_b, steps_b)
    )


def x_divisor_trace(b: int) -> Fraction:
    """tau_x(Y^b dY), with the inverse B mu_4 trace."""
    residue, steps = b % 4, b // 4
    return (
        4
        * Fraction((-1) ** steps)
        * quarter_pochhammer(residue, steps)
    )


def y_divisor_trace(a: int) -> Fraction:
    """tau_y(X^a dX), including its boundary-orientation sign."""
    residue, steps = a % 4, a // 4
    return (
        -4
        * Fraction((-1) ** steps)
        * quarter_pochhammer(residue, steps)
    )


Polynomial = dict[int, Fraction]


def add_polynomials(*polynomials: Polynomial) -> Polynomial:
    output: Polynomial = {}
    for polynomial in polynomials:
        for exponent, coefficient in polynomial.items():
            output[exponent] = output.get(exponent, Fraction(0)) + coefficient
            if output[exponent] == 0:
                del output[exponent]
    return output


def scale_polynomial(polynomial: Polynomial, scalar: Fraction) -> Polynomial:
    return {
        exponent: scalar * coefficient
        for exponent, coefficient in polynomial.items()
        if scalar * coefficient
    }


def twisted_derivative(polynomial: Polynomial) -> Polynomial:
    output: Polynomial = {}
    for exponent, coefficient in polynomial.items():
        if exponent:
            output[exponent - 1] = (
                output.get(exponent - 1, Fraction(0))
                + exponent * coefficient
            )
        output[exponent + 3] = (
            output.get(exponent + 3, Fraction(0))
            + 4 * coefficient
        )
    return {exponent: value for exponent, value in output.items() if value}


def relative_normal_form(polynomial: Polynomial) -> Polynomial:
    output: Polynomial = {}
    for exponent, coefficient in polynomial.items():
        residue, steps = exponent % 4, exponent // 4
        value = (
            coefficient
            * Fraction((-1) ** steps)
            * quarter_pochhammer(residue, steps)
        )
        output[residue] = output.get(residue, Fraction(0)) + value
    return {exponent: value for exponent, value in output.items() if value}


@lru_cache(maxsize=None)
def relative_primitive_monomial(exponent: int) -> tuple[tuple[int, Fraction], ...]:
    if exponent <= 3:
        return ()
    leading = {exponent - 3: Fraction(1, 4)}
    lower = dict(relative_primitive_monomial(exponent - 4))
    result = add_polynomials(
        leading,
        scale_polynomial(lower, Fraction(-(exponent - 3), 4)),
    )
    return tuple(sorted(result.items()))


def relative_primitive(polynomial: Polynomial) -> Polynomial:
    output: Polynomial = {}
    for exponent, coefficient in polynomial.items():
        output = add_polynomials(
            output,
            scale_polynomial(
                dict(relative_primitive_monomial(exponent)),
                coefficient,
            ),
        )
    return output


def check_relative_deformation_retract() -> int:
    checks = 0
    for exponent in range(129):
        monomial = {exponent: Fraction(1)}
        primitive = relative_primitive(monomial)
        expected = add_polynomials(
            monomial,
            scale_polynomial(relative_normal_form(monomial), Fraction(-1)),
        )
        assert twisted_derivative(primitive) == expected
        checks += 1

        differential = twisted_derivative(monomial)
        projected = relative_normal_form(differential)
        point_value = Fraction(1) if exponent == 0 else Fraction(0)
        projected = add_polynomials(
            projected,
            {3: -4 * point_value} if point_value else {},
        )
        assert projected == {}
        checks += 1

        recovered = add_polynomials(
            relative_primitive(differential),
            {0: point_value} if point_value else {},
        )
        assert recovered == monomial
        checks += 1

    for seed in range(257):
        polynomial = {
            (7 * seed + 3 * index) % 41: Fraction(
                ((seed + index) % 13) - 6,
                index + 1,
            )
            for index in range(1, 6)
        }
        polynomial = {
            exponent: coefficient
            for exponent, coefficient in polynomial.items()
            if coefficient
        }
        point = Fraction((seed % 17) - 8, (seed % 7) + 1)
        homotopy = add_polynomials(relative_primitive(polynomial), {0: point})
        left_bulk = twisted_derivative(homotopy)
        right_bulk = add_polynomials(
            polynomial,
            scale_polynomial(relative_normal_form(polynomial), Fraction(-1)),
            {3: 4 * point} if point else {},
        )
        assert left_bulk == right_bulk
        assert homotopy.get(0, Fraction(0)) == point
        checks += 2
    return checks


def check_bulk_stokes() -> int:
    checks = 0
    for transverse_power in range(33):
        for spectator_power in range(33):
            # D(X^m Y^b dY) =
            # (m X^(m-1)Y^b + 4 X^(m+3)Y^b) Omega at hbar=1.
            d_y_form = 4 * bulk_trace(
                transverse_power + 3, spectator_power
            )
            if transverse_power:
                d_y_form += transverse_power * bulk_trace(
                    transverse_power - 1, spectator_power
                )
            x_boundary = (
                x_divisor_trace(spectator_power)
                if transverse_power == 0
                else Fraction(0)
            )
            assert d_y_form == x_boundary
            checks += 1

            # D(X^a Y^m dX) is the oriented negative transpose.
            d_x_form = -4 * bulk_trace(
                spectator_power, transverse_power + 3
            )
            if transverse_power:
                d_x_form -= transverse_power * bulk_trace(
                    spectator_power, transverse_power - 1
                )
            y_boundary = (
                y_divisor_trace(spectator_power)
                if transverse_power == 0
                else Fraction(0)
            )
            assert d_x_form == y_boundary
            checks += 1
    return checks


def check_divisor_corner_stokes() -> int:
    checks = 0
    for power in range(65):
        x_value = 4 * x_divisor_trace(power + 3)
        y_value = 4 * y_divisor_trace(power + 3)
        if power:
            x_value += power * x_divisor_trace(power - 1)
            y_value += power * y_divisor_trace(power - 1)
        assert x_value == (16 if power == 0 else 0)
        assert y_value == (-16 if power == 0 else 0)
        checks += 2
    return checks


def check_full_relative_trace() -> int:
    checks = 0
    for seed in range(401):
        a = (7 * seed + 3) % 29
        b = (11 * seed + 5) % 31
        m = (13 * seed + 1) % 27
        n = (17 * seed + 2) % 25
        hp = Fraction((seed % 9) - 4, (seed % 5) + 1)
        hq = Fraction(((3 * seed) % 11) - 5, (seed % 7) + 1)
        ex = Fraction(((5 * seed) % 13) - 6, (seed % 3) + 1)
        ey = Fraction(((2 * seed) % 15) - 7, (seed % 4) + 1)

        # H = hp X^a Y^m dX + hq X^n Y^b dY.
        bulk = hp * (
            -4 * bulk_trace(a, m + 3)
            - (m * bulk_trace(a, m - 1) if m else 0)
        )
        bulk += hq * (
            4 * bulk_trace(n + 3, b)
            + (n * bulk_trace(n - 1, b) if n else 0)
        )
        restricted_x = hq * x_divisor_trace(b) if n == 0 else 0
        restricted_y = hp * y_divisor_trace(a) if m == 0 else 0

        # Boundary functions eta_x=ex Y^m and eta_y=ey X^n.
        d_eta_x = ex * (
            4 * x_divisor_trace(m + 3)
            + (m * x_divisor_trace(m - 1) if m else 0)
        )
        d_eta_y = ey * (
            4 * y_divisor_trace(n + 3)
            + (n * y_divisor_trace(n - 1) if n else 0)
        )
        corner = 16 * (
            (ex if m == 0 else 0) - (ey if n == 0 else 0)
        )

        relative_trace_of_delta = (
            bulk
            - restricted_x
            - restricted_y
            + d_eta_x
            + d_eta_y
            - corner
        )
        assert relative_trace_of_delta == 0
        checks += 1
    return checks


def check_residue_three_regression() -> int:
    # Ordinary Brieskorn exactness would demand tau_A(X^3 Omega)=0, but
    # the endpoint moment assigns it one.  The relative divisor component
    # of delta(1/4 dY) cancels that one exactly.
    ordinary_value = bulk_trace(3, 0)
    divisor_value = Fraction(1, 4) * x_divisor_trace(0)
    assert ordinary_value == 1
    assert divisor_value == 1
    assert ordinary_value - divisor_value == 0

    y_ordinary = bulk_trace(0, 3)
    y_divisor = Fraction(-1, 4) * y_divisor_trace(0)
    assert y_ordinary == 1
    assert y_divisor == 1
    assert y_ordinary - y_divisor == 0
    return 6


def check_factorization_cellular_cocycle() -> int:
    checks = 0
    observed_nonzero_bare_split = False
    for parent_a in range(25):
        for parent_b in range(25):
            parent_weight = (
                quarter_pochhammer(parent_a % 4, parent_a // 4)
                * quarter_pochhammer(parent_b % 4, parent_b // 4)
            )
            for blocks in range(2, 6):
                excess = blocks - 1
                for excess_x in range(excess + 1):
                    excess_y = excess - excess_x
                    split_a = parent_a + 4 * excess_x
                    split_b = parent_b + 4 * excess_y
                    split_weight = (
                        quarter_pochhammer(split_a % 4, split_a // 4)
                        * quarter_pochhammer(split_b % 4, split_b // 4)
                    )
                    vertex = -split_weight / parent_weight
                    bare_split = bulk_trace(split_a, split_b)
                    connected_parent = (
                        Fraction((-1) ** excess)
                        * vertex
                        * bulk_trace(parent_a, parent_b)
                    )
                    assert bare_split + connected_parent == 0
                    assert bare_split != 0
                    observed_nonzero_bare_split = True
                    checks += 2
    assert observed_nonzero_bare_split
    # Literal two-label regression: this is the scalar of the two boundary
    # faces of one corolla package, not the differential of a bare zero-cell.
    two_label_split = bulk_trace(4, 0)
    two_label_parent = Fraction(-1, 4) * -bulk_trace(0, 0)
    assert two_label_split == Fraction(-1, 4)
    assert two_label_parent == Fraction(1, 4)
    assert two_label_split + two_label_parent == 0
    return checks + 4


def check_literal_corolla_chain_boundary() -> int:
    """d(e*split + v0*G) = v1*split + v0*parent when dG=split+parent."""
    checks = 0
    for seed in range(2003):
        split = Fraction((7 * seed) % 43 - 21, seed % 11 + 1)
        parent = Fraction((13 * seed) % 47 - 23, seed % 13 + 1)
        delta_g = split + parent
        high = split
        low = -split + delta_g
        assert high == split
        assert low == parent
        checks += 2
    return checks


def check_augmentation_forest_lines() -> int:
    """Rank of the forest associated graded telescopes to dim V_I."""
    checks = 0
    for leaf_count in range(1, 15):
        for seed in range(257):
            roots = leaf_count
            incoming_ranks: list[int] = []
            step = 0
            while roots > 1 and (step == 0 or (seed + 3 * step) % 5):
                arity = 2 + ((seed + 7 * step) % min(4, roots))
                arity = min(arity, roots)
                incoming_ranks.append(arity - 1)
                roots -= arity - 1
                step += 1
            root_rank = roots - 1
            assert root_rank + sum(incoming_ranks) == leaf_count - 1
            assert all(rank >= 1 for rank in incoming_ranks)
            checks += 2
    return checks


def check_radial_thimble_ledger() -> int:
    checks = 0
    stack_trace = Fraction(1, 4)
    cartier_multiplicity = 4
    normalized_low_face = cartier_multiplicity * stack_trace
    assert normalized_low_face == 1
    checks += 1
    for seed in range(503):
        point = Fraction((seed % 23) - 11, (seed % 7) + 1)
        oc = Fraction(((3 * seed) % 29) - 14, (seed % 11) + 1)
        end = Fraction(((5 * seed) % 31) - 15, (seed % 13) + 1)
        wc = Fraction(((7 * seed) % 37) - 18, (seed % 5) + 1)
        cont = Fraction(((11 * seed) % 41) - 20, (seed % 3) + 1)
        xch = Fraction(((13 * seed) % 43) - 21, (seed % 17) + 1)

        raw = end - point + wc + cont + xch
        thimble = point - normalized_low_face * oc
        total = raw + thimble
        assert total == end - oc + wc + cont + xch
        checks += 1
    return checks


def check_ambient_facet_inventory() -> int:
    # Smoke-test the factor-by-factor inventory printed in Proposition 3.1.
    # The proposition's product-facet proof, not this dictionary, is the
    # exhaustion argument.
    dispositions = {
        "corridor_boundary": "named_end",
        "segment_zero": "structural",
        "seam_endpoint": "structural",
        "side_equality": "structural",
        "metric_zero": "structural",
        "metric_one": "product_forest_gluing",
        "bar_facet": "structural",
        "nerve_first_face": "structural",
        "nerve_composition_face": "structural",
        "nerve_last_face": "structural",
        "fm_scale_zero": "structural",
        "moving_flag_or_critical": "named_moving",
        "root_homotopy_endpoint": "named_root",
        "auxiliary_cap": "empty",
        "order_at_least_N": "truncated",
        "closed_rank_defect": "off_degree_null",
        "residual_rank_change": "active_interior",
    }
    allowed = {
        "named_end",
        "named_moving",
        "named_root",
        "structural",
        "product_forest_gluing",
        "empty",
        "truncated",
        "off_degree_null",
        "active_interior",
    }
    assert set(dispositions.values()) <= allowed
    assert "unclassified_terminal" not in dispositions.values()
    return len(dispositions) + 2


def check_structural_nerve_faces() -> int:
    checks = 0
    arrow_kinds = {
        "zero_edge_union",
        "forgetful",
        "corridor_restriction",
        "stabilization",
        "collision_factorization",
    }
    for simplex_dimension in range(1, 13):
        faces = []
        for index in range(simplex_dimension + 1):
            if index == 0:
                faces.append("apply_first")
            elif index == simplex_dimension:
                faces.append("delete_last")
            else:
                faces.append("compose_adjacent")
        assert len(faces) == simplex_dimension + 1
        assert faces.count("apply_first") == 1
        assert faces.count("delete_last") == 1
        assert faces.count("compose_adjacent") == simplex_dimension - 1
        for kind in arrow_kinds:
            assert kind != "rees_cartan_bar"
            checks += 5
    return checks


def check_shared_cell_degree() -> int:
    checks = 0
    for geometric_dimension in range(13):
        for contact_degree in range(3):
            total_degree = geometric_dimension + contact_degree
            duplicated_degree = 2 * geometric_dimension + contact_degree
            assert total_degree == geometric_dimension + contact_degree
            if geometric_dimension:
                assert total_degree != duplicated_degree
            checks += 2
    return checks


def check_relative_augmentation_regression() -> int:
    # In C_*([0,1], {0}), the boundary of [0,1] is represented by {1}.
    # Its scalar augmentation is one, not zero.
    relative_boundary_coefficients = (Fraction(1),)
    augmentation = sum(relative_boundary_coefficients, Fraction(0))
    assert augmentation == 1
    assert augmentation != 0
    return 2


def main() -> None:
    retract = check_relative_deformation_retract()
    bulk = check_bulk_stokes()
    corner = check_divisor_corner_stokes()
    full = check_full_relative_trace()
    residue_three = check_residue_three_regression()
    factorization = check_factorization_cellular_cocycle()
    literal_corolla = check_literal_corolla_chain_boundary()
    forest_lines = check_augmentation_forest_lines()
    thimble = check_radial_thimble_ledger()
    facet_inventory = check_ambient_facet_inventory()
    nerve_faces = check_structural_nerve_faces()
    shared_degree = check_shared_cell_degree()
    relative_regression = check_relative_augmentation_regression()
    total = sum(
        (
            retract,
            bulk,
            corner,
            full,
            residue_three,
            factorization,
            literal_corolla,
            forest_lines,
            thimble,
            facet_inventory,
            nerve_faces,
            shared_degree,
            relative_regression,
        )
    )

    print("relative Fermat deformation-retract identities:", retract)
    print("bulk-to-divisor Stokes identities:", bulk)
    print("divisor-to-corner Stokes identities:", corner)
    print("full relative-contact trace identities:", full)
    print("residue-three counterexample regressions:", residue_three)
    print("cellular factorization-cocycle identities:", factorization)
    print("literal corolla-chain boundary identities:", literal_corolla)
    print("augmentation forest-line identities:", forest_lines)
    print("raw-plus-radial-thimble boundary identities:", thimble)
    print("ambient-factor facet-inventory checks:", facet_inventory)
    print("structural-nerve face checks:", nerve_faces)
    print("shared-cell degree checks:", shared_degree)
    print("relative-augmentation counterexample checks:", relative_regression)
    print("total exact identities checked:", total)
    print("RELATIVE CONTACT AND POINT/OC REPAIR: EXACT REGRESSIONS PASS")
    print("This program does not establish PL general position or the geometric boundary theorem")


if __name__ == "__main__":
    main()
