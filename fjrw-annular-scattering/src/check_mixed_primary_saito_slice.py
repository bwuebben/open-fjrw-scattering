#!/usr/bin/env python3
"""Exact check of the mixed primary slice, eq:mixed-primary-potential (Section 6).

Set u=t_(1,1,0) and t=t_(2,2,0).  The chamber-index equations A(J,0,nu)=0 of
GKT Definition 3.47 determine the first mixed open invariants of the
reflection-fixed chamber index.  After inserting them into the symmetric
potential of GKT Definition 4.12, direct good-basis integration checks the
primitive-form conditions of GKT Definition 4.3 for all nine cycles through
total parameter degree three: no positive powers of hbar, hbar^0 coefficient
delta_(mu,0), and hbar^(-1) coefficients equal to the flat coordinates u, t.

It uses the A-invariant engine src/a_invariants.py at the top of the
repository.  Run with Python 3 and SymPy (the working directory does not
matter):

    python3 -B check_mixed_primary_saito_slice.py
"""


from math import factorial
from pathlib import Path
import sys

import sympy as sp


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from a_invariants import A_invariant, d_of, part_boundaries  # noqa: E402


U_MARK = (1, 1, 0)
T_MARK = (2, 2, 0)


def mixed_block_values():
    """Solve the two new cubic primary blocks from A(J,0,nu)=0."""

    nu_utt, nu_uut = sp.symbols("nu_utt nu_uut")
    known = {
        ((T_MARK, T_MARK), 4, 0): -sp.Rational(1, 8),
        ((T_MARK, T_MARK), 0, 4): -sp.Rational(1, 8),
        ((U_MARK, T_MARK, T_MARK), 1, 1): nu_utt,
        ((U_MARK, U_MARK, T_MARK), 0, 0): nu_uut,
    }

    def nu(part_key, k1, k2):
        return known.get((part_key, k1, k2))

    utt = [U_MARK, T_MARK, T_MARK]
    uut = [U_MARK, U_MARK, T_MARK]
    assert d_of(utt, 4, 4) == d_of(uut, 4, 4) == -1
    assert part_boundaries(utt, 4, 4) == [(1, 1)]
    assert part_boundaries(uut, 4, 4) == [(0, 0)]

    relation_utt = sp.expand(A_invariant(utt, 4, 4, nu))
    relation_uut = sp.expand(A_invariant(uut, 4, 4, nu))
    assert relation_utt == nu_utt + sp.Rational(1, 8)
    assert relation_uut == nu_uut + sp.Rational(1, 16)

    value_utt = sp.solve(sp.Eq(relation_utt, 0), nu_utt)
    value_uut = sp.solve(sp.Eq(relation_uut, 0), nu_uut)
    assert value_utt == [-sp.Rational(1, 8)]
    assert value_uut == [-sp.Rational(1, 16)]
    return value_utt[0], value_uut[0]


def good_basis_moment(cycle_index, exponent, hbar):
    """Integral of x^exponent on the normalized quartic cycle Xi_cycle."""

    difference = exponent - cycle_index
    if difference < 0 or difference % 4:
        return sp.Integer(0)
    steps = difference // 4
    return (
        (-hbar) ** steps
        * sp.rf(sp.Rational(cycle_index + 1, 4), steps)
    )


def truncate_parameter_degree(expression, variables, max_degree):
    polynomial = sp.Poly(sp.expand(expression), *variables)
    output = sp.Integer(0)
    for powers, coefficient in polynomial.terms():
        if sum(powers) <= max_degree:
            monomial = sp.prod(variable**power for variable, power in zip(variables, powers))
            output += coefficient * monomial
    return sp.expand(output)


def oscillatory_integral(cycle, perturbation, variables, hbar, max_degree):
    """Integrate exp(perturbation/hbar) on one product good-basis cycle."""

    x, y = sp.symbols("x y")
    expansion = sp.Integer(0)
    for order in range(max_degree + 1):
        expansion += perturbation**order / (factorial(order) * hbar**order)
    expansion = truncate_parameter_degree(expansion, variables, max_degree)

    polynomial = sp.Poly(sp.expand(expansion), x, y, *variables)
    output = sp.Integer(0)
    for powers, coefficient in polynomial.terms():
        x_power, y_power = powers[:2]
        parameter_powers = powers[2:]
        moment = good_basis_moment(cycle[0], x_power, hbar)
        moment *= good_basis_moment(cycle[1], y_power, hbar)
        parameter_monomial = sp.prod(
            variable**power for variable, power in zip(variables, parameter_powers)
        )
        output += coefficient * parameter_monomial * moment
    return sp.expand(output)


def coefficient_of_hbar_minus_one(expression, hbar):
    """Extract the coefficient of hbar^(-1) from a finite Laurent sum."""

    return sp.expand(sp.residue(expression, hbar, 0))


def hbar_coefficients(expression, hbar):
    """All coefficients of a finite Laurent polynomial in hbar."""

    numerator, denominator = sp.fraction(sp.together(sp.expand(expression)))
    shift = sp.Poly(denominator, hbar).degree()
    unit = sp.Poly(denominator, hbar).LC()
    polynomial = sp.Poly(sp.expand(numerator / unit), hbar)
    return {
        power - shift: sp.expand(coefficient)
        for (power,), coefficient in polynomial.terms()
    }


def main():
    nu_utt, nu_uut = mixed_block_values()

    x, y, u, t, hbar = sp.symbols("x y u t hbar")

    # GKT Definition 4.12 contributes sign (+) at cardinality three and divides
    # by the order-two automorphism of each repeated marking.  Therefore the
    # two open invariants give -u*t^2*x*y/16 and -u^2*t/32.
    assert nu_utt / factorial(2) == -sp.Rational(1, 16)
    assert nu_uut / factorial(2) == -sp.Rational(1, 32)
    perturbation = (
        u * x * y
        + t * x**2 * y**2
        + t**2 * (x**4 + y**4) / 16
        - u * t**2 * x * y / 16
        - u**2 * t / 32
    )

    hbar_minus_one = {}
    for a in range(3):
        for b in range(3):
            integral = oscillatory_integral(
                (a, b), perturbation, (u, t), hbar, 3
            )
            hbar_minus_one[(a, b)] = coefficient_of_hbar_minus_one(
                integral, hbar
            )
            coefficients = hbar_coefficients(integral, hbar)
            # no positive powers of hbar, and hbar^0 coefficient delta_(mu,0)
            for power, coefficient in coefficients.items():
                if power > 0:
                    assert truncate_parameter_degree(
                        coefficient, (u, t), 3
                    ) == 0, ((a, b), power, coefficient)
            constant = truncate_parameter_degree(
                coefficients.get(0, sp.Integer(0)), (u, t), 3
            )
            assert constant == (1 if (a, b) == (0, 0) else 0), ((a, b), constant)
            assert hbar_minus_one[(a, b)] == truncate_parameter_degree(
                coefficients.get(-1, sp.Integer(0)), (u, t), 3
            )

    assert hbar_minus_one[(1, 1)] == u
    assert hbar_minus_one[(2, 2)] == t
    assert all(
        coefficient == 0
        for cycle, coefficient in hbar_minus_one.items()
        if cycle not in {(1, 1), (2, 2)}
    )

    # The period-normalized Kodaira--Spencer basis is obtained by
    # differentiating the flat potential.  Its period matrix is the identity.
    phi_u = sp.diff(perturbation, u)
    phi_t = sp.diff(perturbation, t)
    assert sp.expand(
        phi_u - ((1 - t**2 / 16) * x * y - u * t / 16)
    ) == 0
    assert sp.expand(
        phi_t
        - (
            x**2 * y**2
            + t * (x**4 + y**4) / 8
            - u * t * x * y / 8
            - u**2 / 32
        )
    ) == 0
    flat_period_vector = sp.Matrix(
        [hbar_minus_one[(1, 1)], hbar_minus_one[(2, 2)]]
    )
    flat_period_matrix = flat_period_vector.jacobian((u, t))
    assert flat_period_matrix == sp.eye(2)

    # Expose the two mixed cancellations separately.
    direct_xy_correction = -u * t**2 / 16
    endpoint_xy_cross = -u * t**2 / 16
    triple_xy_term = u * t**2 / 8
    assert direct_xy_correction + endpoint_xy_cross + triple_xy_term == 0

    direct_constant_correction = -u**2 * t / 32
    triple_constant_term = u**2 * t / 32
    assert direct_constant_correction + triple_constant_term == 0

    print("mixed open blocks: nu_(u t^2;1,1)=", nu_utt)
    print("                   nu_(u^2 t;0,0)=", nu_uut)
    print("no positive hbar powers; hbar^0 coefficients delta_(mu,0): PASS")
    print("flat hbar^-1 period coefficients:", hbar_minus_one)
    print("period-normalized basis: Phi_u=", phi_u, "Phi_t=", phi_t)
    print("MIXED PRIMARY SAITO SLICE THROUGH CUBIC ORDER: PASS")


if __name__ == "__main__":
    main()
