#!/usr/bin/env python3
"""Recursive primary-marginal GKT/Saito series for x^4+y^4.

For n repeated labelled primary markings of twist (2,2), the full block has
boundary support (4,0),(0,4) when n is even and (2,2) when n is odd.  The
labelled-partition equation A(J,0,nu)=0 determines the new reflection-fixed
invariant from all lower blocks.  This script imports the independently
verified A-invariant engine and computes the resulting potential series.

Run with Sage's Python so SymPy and the project dependencies are available:

    sage -python fjrw-annular-scattering/src/check_marginal_saito_series.py
"""

from math import factorial
from pathlib import Path
import sys

import sympy as sp


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from a_invariants import A_invariant, d_of, part_boundaries  # noqa: E402


def marginal_block_values(max_order=8):
    values = {
        1: {(2, 2): sp.Integer(1)},
    }
    relations = {}

    for order in range(2, max_order + 1):
        J = [(2, 2, 0)] * order
        assert d_of(J, 4, 4) < 0
        expected_boundaries = (
            [(0, 4), (4, 0)] if order % 2 == 0 else [(2, 2)]
        )
        assert part_boundaries(J, 4, 4) == expected_boundaries

        unknown = sp.Symbol(f"nu_{order}")

        def nu(part_key, k1, k2):
            block_order = len(part_key)
            boundary = (k1, k2)
            if block_order == order:
                if boundary in expected_boundaries:
                    return unknown
                return None
            return values[block_order].get(boundary)

        relation = sp.expand(A_invariant(J, 4, 4, nu))
        solutions = sp.solve(sp.Eq(relation, 0), unknown)
        assert len(solutions) == 1, (order, relation, solutions)
        value = sp.simplify(solutions[0])
        values[order] = {
            boundary: value for boundary in expected_boundaries
        }
        relations[order] = relation

    return values, relations


def potential_coefficients(values):
    endpoint = {0: sp.Integer(1)}
    central = {}
    for order, boundary_values in sorted(values.items()):
        coefficient = (
            sp.Integer((-1) ** (order - 1))
            * next(iter(boundary_values.values()))
            / factorial(order)
        )
        if order % 2 == 0:
            endpoint[order] = sp.simplify(coefficient)
        else:
            central[order] = sp.simplify(coefficient)
    return endpoint, central


def oscillatory_periods(alpha, beta, max_order):
    """Good-basis periods F_0 and F_2 for the quartic marginal slice.

    Here alpha multiplies x^4+y^4 relative to the Fermat potential and beta
    multiplies x^2*y^2.  The moment recursions give P_0(4k)=
    (-hbar)^k(1/4)_k and P_2(4k+2)=(-hbar)^k(3/4)_k.  All hbar powers cancel
    in F_0; F_2 is the coefficient of hbar^-1.
    """

    F0 = sp.Integer(0)
    F2 = sp.Integer(0)
    for i in range(max_order + 1):
        for j in range(max_order + 1):
            sign = sp.Integer((-1) ** (i + j))
            for ell in range(max_order + 1):
                if 2 * (i + j + ell) <= max_order:
                    F0 += (
                        sign
                        * alpha ** (i + j)
                        * beta ** (2 * ell)
                        * sp.rf(sp.Rational(1, 4), i + ell)
                        * sp.rf(sp.Rational(1, 4), j + ell)
                        / (
                            factorial(i)
                            * factorial(j)
                            * factorial(2 * ell)
                        )
                    )
                if 2 * (i + j + ell) + 1 <= max_order:
                    F2 += (
                        sign
                        * alpha ** (i + j)
                        * beta ** (2 * ell + 1)
                        * sp.rf(sp.Rational(3, 4), i + ell)
                        * sp.rf(sp.Rational(3, 4), j + ell)
                        / (
                            factorial(i)
                            * factorial(j)
                            * factorial(2 * ell + 1)
                        )
                    )
    return sp.expand(F0), sp.expand(F2)


def main():
    values, relations = marginal_block_values()
    endpoint, central = potential_coefficients(values)

    assert values[2][(4, 0)] == -sp.Rational(1, 8)
    assert values[2][(0, 4)] == -sp.Rational(1, 8)
    assert values[3][(2, 2)] == 0
    assert endpoint[2] == sp.Rational(1, 16)
    assert central[1] == 1
    assert central[3] == 0

    t = sp.Symbol("t")
    alpha = sum(coefficient * t**order for order, coefficient in endpoint.items()) - 1
    beta = sum(coefficient * t**order for order, coefficient in central.items())
    F0, F2 = oscillatory_periods(sp.Symbol("alpha"), sp.Symbol("beta"), 8)
    substituted_F0 = sp.series(
        F0.subs({"alpha": alpha, "beta": beta}), t, 0, 9
    ).removeO().expand()
    substituted_F2 = sp.series(
        F2.subs({"alpha": alpha, "beta": beta}), t, 0, 9
    ).removeO().expand()
    assert substituted_F0 == 1
    assert substituted_F2 == t

    alpha_symbol = sp.Symbol("alpha")
    beta_symbol = sp.Symbol("beta")
    linear_F0_alpha = sp.diff(F0, alpha_symbol).subs(
        {alpha_symbol: 0, beta_symbol: 0}
    )
    linear_F2_beta = sp.diff(F2, beta_symbol).subs(
        {alpha_symbol: 0, beta_symbol: 0}
    )
    assert linear_F0_alpha == -sp.Rational(1, 2)
    assert linear_F2_beta == 1

    print("reflection-fixed full-block invariants:")
    for order in sorted(values):
        print(f"  n={order}: {next(iter(values[order].values()))}")
    print("endpoint series A(t):", endpoint)
    print("central series B(t):", central)
    print("oscillatory inverse equations F_0=1, F_2=t: PASS")
    print("labelled-partition relations certified:", len(relations))
    print("PRIMARY MARGINAL SAITO SERIES THROUGH ORDER 8: PASS")


if __name__ == "__main__":
    main()
