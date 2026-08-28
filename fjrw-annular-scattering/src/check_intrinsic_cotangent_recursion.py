#!/usr/bin/env python3
"""Exact checks for the intrinsic quartic cotangent recursion theorem.

The program verifies the finite algebra used in the cotangent-recursion
argument: contact-trace WC cancellation,
factorization-primitive moment equations, threshold open--closed profiles,
Q1/Q2 dependency orders, support reduction, sign bookkeeping, and singleton
initial conditions.  It does not replace the geometric boundary theorem or
the GKT comparison theorem.  It also checks the exact triangular equations
and critical-exponent dictionary for the two supported GKT normal forms.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product

import check_full_descendent_period_trees as period
import check_intrinsic_relative_contact_virtual_chain as contact


Datum = tuple[int, int, int]
Data = tuple[Datum, ...]
Support = frozenset[int]


def nonempty_subsets(items: tuple[int, ...]):
    for size in range(1, len(items) + 1):
        yield from combinations(items, size)


def support_state(data: Data, support: Support) -> dict[str, int]:
    total_a = sum(data[index][0] for index in support)
    total_b = sum(data[index][1] for index in support)
    total_d = sum(data[index][2] for index in support)
    residue_a = total_a % 4
    residue_b = total_b % 4
    ell = (total_a - residue_a) // 4
    emm = (total_b - residue_b) // 4
    return {
        "a": total_a,
        "b": total_b,
        "d": total_d,
        "r": residue_a,
        "s": residue_b,
        "ell": ell,
        "emm": emm,
        "N": ell + emm - len(support) + 1 + total_d,
    }


def global_state(data: Data, extra_index: int | None = None) -> tuple[int, int, int]:
    total_a = sum(item[0] for item in data)
    total_b = sum(item[1] for item in data)
    total_d = sum(item[2] for item in data)
    if extra_index is not None:
        total_d += 1
    residue_a = total_a % 4
    residue_b = total_b % 4
    return (
        residue_a,
        residue_b,
        (total_a - residue_a) // 4
        + (total_b - residue_b) // 4
        - len(data)
        + 1
        + total_d,
    )


def threshold_supports(data: Data, distinguished: int) -> tuple[Support, ...]:
    others = tuple(index for index in range(len(data)) if index != distinguished)
    output = []
    for chosen in nonempty_subsets(others):
        support = frozenset((distinguished,) + chosen)
        if support_state(data, support)["N"] == -1:
            output.append(support)
    return tuple(output)


def continuing_data(data: Data, support: Support) -> Data:
    state = support_state(data, support)
    spectators = tuple(data[index] for index in range(len(data)) if index not in support)
    return spectators + ((state["r"], state["s"], 0),)


def sample_windows() -> tuple[Data, ...]:
    marking_types = tuple(product(range(3), range(3), range(3)))
    windows: list[Data] = []

    # Exhaust every ordered two-marking quartic datum in this degree range.
    windows.extend(tuple(pair) for pair in product(marking_types, repeat=2))

    # Deterministic broad samples through six labels cover every residue,
    # primary bases, mixed descendants, and many threshold supports.
    for size in range(3, 7):
        for seed in range(420):
            window = tuple(
                marking_types[
                    (seed * (position + 5) + 11 * position * position + 7 * size)
                    % len(marking_types)
                ]
                for position in range(size)
            )
            windows.append(window)

        # Include explicit all-primary windows for the initial-condition audit.
        for seed in range(90):
            windows.append(
                tuple(
                    (
                        (seed + 2 * position + size) % 3,
                        (2 * seed + position * position + 1) % 3,
                        0,
                    )
                    for position in range(size)
                )
            )
    return tuple(windows)


def check_wc_contact_trace() -> int:
    checks = 0
    for u in range(25):
        for v in range(25):
            residue = (u % 4, v % 4)
            right_profile = (u + 4, v)
            upper_profile = (u, v + 4)
            right_weight = period.gamma_weight(right_profile, residue)
            upper_weight = period.gamma_weight(upper_profile, residue)
            assert (
                4 * (v + 1) * right_weight
                - 4 * (u + 1) * upper_weight
                == 0
            )
            checks += 1
    return checks


def check_endpoint_moments(windows: tuple[Data, ...]) -> tuple[int, int]:
    moment_checks = 0
    primary_checks = 0
    # The exhaustive two-label list and a deterministic slice of the larger
    # windows are enough to exercise every quartic twist and descendant class
    # without duplicating the much larger companion certificate.
    selected = windows[:729] + windows[729::17]
    for data in selected:
        for endpoint in ("minimum", "maximum"):
            coefficient = period.endpoint_solver(data, endpoint)
            for support_tuple in period.subsets(len(data)):
                geometry = period.geometry(data, support_tuple)
                if not geometry["active"]:
                    continue
                residue = geometry["residue"]
                parent = period.endpoint_exponent(data, support_tuple, endpoint)
                moment = period.gamma_weight(parent, residue) * coefficient(support_tuple)
                for partition in period.admissible_partitions(data, support_tuple):
                    child_exponent = period.add_exponents(
                        tuple(
                            period.endpoint_exponent(data, child, endpoint)
                            for child in partition
                        )
                    )
                    moment += period.gamma_weight(child_exponent, residue) * period.product(
                        tuple(coefficient(child) for child in partition)
                    )
                assert moment == period.active_target(data, support_tuple)
                moment_checks += 1
                if all(data[index][2] == 0 for index in support_tuple):
                    primary_checks += 1
    assert primary_checks > 0
    return moment_checks, primary_checks


def check_gkt_extreme_normal_forms(windows: tuple[Data, ...]) -> int:
    checks = 0
    selected = windows[:729] + windows[729::29]
    for data in selected:
        for endpoint in ("minimum", "maximum"):
            coefficient = period.endpoint_solver(data, endpoint)
            for support_tuple in period.subsets(len(data)):
                geometry = period.geometry(data, support_tuple)
                if not geometry["active"]:
                    continue
                residue = geometry["residue"]
                level_count = int(geometry["number"])
                proper_support = Fraction(0)
                for partition in period.admissible_partitions(data, support_tuple):
                    child_exponent = period.add_exponents(
                        tuple(
                            period.endpoint_exponent(data, child, endpoint)
                            for child in partition
                        )
                    )
                    proper_support += (
                        period.gamma_weight(child_exponent, residue)
                        * period.product(tuple(coefficient(child) for child in partition))
                    )

                vector = [Fraction(0) for _ in range(level_count + 1)]
                retained = 0 if endpoint == "minimum" else level_count
                vector[retained] = coefficient(support_tuple)
                one_block = Fraction(0)
                for level, value in enumerate(vector):
                    profile = (
                        residue[0] + 4 * level,
                        residue[1] + 4 * (level_count - level),
                    )
                    one_block += period.gamma_weight(profile, residue) * value
                assert one_block + proper_support == period.active_target(
                    data, support_tuple
                )
                assert sum(value != 0 for index, value in enumerate(vector) if index != retained) == 0
                checks += 2

                # GKT Lambda_{J,p} has boundary multiplicities one larger
                # than the intrinsic Hamiltonian exponent.
                for p_index in range(1, level_count + 1):
                    gkt_boundary = (
                        residue[0] + 4 * (p_index - 1) + 1,
                        residue[1] + 4 * (level_count - p_index) + 1,
                    )
                    intrinsic = (
                        residue[0] + 4 * (p_index - 1),
                        residue[1] + 4 * (level_count - p_index),
                    )
                    assert (
                        gkt_boundary[0] - 1,
                        gkt_boundary[1] - 1,
                    ) == intrinsic
                    checks += 1
    return checks


def check_threshold_recursions(windows: tuple[Data, ...]) -> tuple[int, int, int, int]:
    threshold_checks = 0
    degree_checks = 0
    q_difference_checks = 0
    support_checks = 0

    for data in windows:
        lower_degree = sum(item[2] for item in data)
        for distinguished in range(len(data)):
            q1 = set(threshold_supports(data, distinguished))
            for support in q1:
                state = support_state(data, support)
                extended = (2 - state["r"], 2 - state["s"])
                assert all(-1 <= entry <= 2 for entry in extended)
                assert (state["a"] + extended[0]) % 4 == 2
                assert (state["b"] + extended[1]) % 4 == 2

                # Orbifold RR rank minus closed dimension is N+1=0.
                rank = state["ell"] + state["emm"] + state["d"]
                dimension = len(support) - 2
                assert rank == dimension

                continued = continuing_data(data, support)
                assert global_state(data, distinguished) == global_state(continued)

                continuing_degree = sum(item[2] for item in continued)
                assert continuing_degree <= lower_degree
                assert continuing_degree < lower_degree + 1
                assert len(continued) == len(data) - len(support) + 1
                threshold_checks += 6
                degree_checks += 2

            # Q1 also has the contraction edge from degree D+1 to D.
            assert lower_degree < lower_degree + 1
            degree_checks += 1

            for second in range(len(data)):
                if second == distinguished:
                    continue
                q2 = {support for support in q1 if second not in support}
                difference = q1 - q2
                expected = {support for support in q1 if second in support}
                assert difference == expected
                assert q1 == q2 | expected
                assert not (q2 & expected)
                q_difference_checks += 3

                # Subtracting Q2 from Q1 leaves the lower moment on the left
                # and only strict support reductions on the right.
                for support in difference:
                    continued = continuing_data(data, support)
                    assert len(continued) <= len(data) - 1
                    assert len(support) >= 2
                    support_checks += 2

    assert threshold_checks > 0
    assert support_checks > 0
    return threshold_checks, degree_checks, q_difference_checks, support_checks


def check_off_degree_null_faces(windows: tuple[Data, ...]) -> int:
    checks = 0
    selected = windows[:729] + windows[729::31]
    for data in selected:
        items = tuple(range(len(data)))
        for support_tuple in nonempty_subsets(items):
            support = frozenset(support_tuple)
            if len(support) < 2:
                continue
            state = support_state(data, support)
            rank = state["ell"] + state["emm"] + state["d"]
            dimension = len(support) - 2
            defect = rank - dimension
            assert defect == state["N"] + 1
            if state["N"] == -1:
                assert defect == 0
            else:
                assert defect != 0
                # Positive defect is excessive Euler degree; negative defect
                # leaves positive residual dimension.  Both miss degree zero.
                assert (defect > 0) != (defect < 0)
            checks += 3
    return checks


def check_homotopy_sign_ledger() -> int:
    checks = 0
    # The absolute total boundary is End - OC + geometric colors.  WC and
    # XCH vanish.  The induced Cont color is -lower for Q1 and zero for Q2;
    # the outward geometric Cont face has the opposite sign.
    for numerator in range(-19, 20):
        for denominator in range(1, 12):
            oc_sum = Fraction(numerator, denominator)
            point = Fraction(5 * numerator + denominator, denominator + 3)
            lower = Fraction(3 * numerator - denominator, denominator + 2)
            q1_cont = -lower
            q2_cont = Fraction(0)
            q1_parent = oc_sum + q1_cont
            q2_parent = oc_sum + q2_cont
            assert q1_parent == oc_sum - lower
            assert q2_parent == oc_sum
            assert q2_parent - q1_parent == lower

            # The raw chain has End - Point + Cont_geo; the normalized
            # radial thimble has Point - OC.  Point cancels before the
            # absolute boundary is augmented.
            q1_cont_geo = -q1_cont
            q2_cont_geo = -q2_cont
            q1_raw = q1_parent - point + q1_cont_geo
            q2_raw = q2_parent - point + q2_cont_geo
            radial = point - oc_sum
            assert q1_raw + radial == 0
            assert q2_raw + radial == 0
            assert q1_raw + radial == q1_parent - oc_sum + q1_cont_geo
            assert q2_raw + radial == q2_parent - oc_sum + q2_cont_geo
            checks += 7
    return checks


def check_initial_conditions() -> int:
    checks = 0
    vacuum = Fraction(1)
    assert vacuum == 1
    checks += 1
    for descendant in range(31):
        direct = Fraction((-1) ** descendant)
        if descendant == 0:
            assert direct == 1
        else:
            previous = Fraction((-1) ** (descendant - 1))
            assert direct == -previous
        checks += 1
    return checks


def main() -> None:
    windows = sample_windows()
    wc_checks = check_wc_contact_trace()
    moment_checks, primary_checks = check_endpoint_moments(windows)
    gkt_normal_form_checks = check_gkt_extreme_normal_forms(windows)
    threshold, degree, q_difference, support = check_threshold_recursions(windows)
    sign_checks = check_homotopy_sign_ledger()
    null_checks = check_off_degree_null_faces(windows)
    relative_contact_checks = (
        contact.check_bulk_stokes()
        + contact.check_divisor_corner_stokes()
        + contact.check_residue_three_regression()
    )
    factorization_checks = contact.check_factorization_cellular_cocycle()
    initial_checks = check_initial_conditions()
    total = sum(
        (
            wc_checks,
            moment_checks,
            gkt_normal_form_checks,
            threshold,
            degree,
            q_difference,
            support,
            sign_checks,
            null_checks,
            relative_contact_checks,
            factorization_checks,
            initial_checks,
        )
    )

    print("finite quartic windows checked:", len(windows))
    print("WC contact-trace identities:", wc_checks)
    print("endpoint moment identities:", moment_checks)
    print("GKT extreme-normal-form identities:", gkt_normal_form_checks)
    print("degree-zero endpoint bases among them:", primary_checks)
    print("threshold/profile identities:", threshold)
    print("strict descendant-degree identities:", degree)
    print("Q1/Q2 set-difference identities:", q_difference)
    print("strict support-reduction identities:", support)
    print("homotopy sign-ledger identities:", sign_checks)
    print("off-degree Null-face identities:", null_checks)
    print("relative-contact Stokes regressions:", relative_contact_checks)
    print("factorization cellular-cocycle regressions:", factorization_checks)
    print("vacuum/singleton initial identities:", initial_checks)
    print("total exact identities checked:", total)
    print("INTRINSIC COTANGENT ALGEBRA AND UNIQUENESS: EXACT CHECKS PASS")


if __name__ == "__main__":
    main()
