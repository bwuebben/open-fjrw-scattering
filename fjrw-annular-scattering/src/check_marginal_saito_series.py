#!/usr/bin/env python3
"""Exact checks for Theorem E.3 (Marginal period calibration) of Appendix E.

For n labelled primary markings of twist (2,2), the balanced graphs have
boundary degrees (4,0),(0,4) when n is even and (2,2) when n is odd.  The
chamber-index equation A(J,0,nu)=0 of GKT Definition 3.47 determines the
reflection-fixed invariants from all lower ones.  The script computes the
resulting potential series lambda(t), mu(t) (the series frak a(t), frak b(t)
of eq:A-series and eq:B-series), checks the period equations
varpi_0 = 1, varpi_2 = t (eq:period-inverse), the residue normalization (eq:residue-gauge and
eq:KS-gauge), and the hypergeometric form (eq:NY-form).  It uses the
A-invariant engine src/a_invariants.py at the top of the repository.

Run with Python 3 and SymPy (the working directory does not matter):

    python3 -B check_marginal_saito_series.py
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
    """Good-basis periods varpi_0 and varpi_2 (eq:F0, eq:F2) of the marginal slice.

    Here alpha = lambda-1 multiplies x^4+y^4 relative to the Fermat potential
    and beta = mu multiplies x^2*y^2.  The moment recursions give P_0(4k)=
    (-hbar)^k(1/4)_k and P_2(4k+2)=(-hbar)^k(3/4)_k.  All hbar powers cancel
    in varpi_0; varpi_2 is the coefficient of hbar^-1.
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

    # eq:residue-gauge and eq:KS-gauge for W_c = (1+ct^2)(x^4+y^4) + t x^2y^2:
    # x^4 = -t/(2(1+ct^2)) x^2y^2 and eps(x^2y^2) = 4/(4a^2-b^2).
    c = sp.Symbol("c")
    a_c, b_c = 1 + c * t**2, t
    eps_socle = 4 / (4 * a_c**2 - b_c**2)
    assert sp.simplify(eps_socle - 1 / ((1 + c * t**2) ** 2 - t**2 / 4)) == 0
    dtW_socle = sp.diff(b_c, t) - sp.diff(a_c, t) * b_c / a_c
    eta_c = sp.series(dtW_socle * eps_socle, t, 0, 4).removeO()
    assert sp.expand(eta_c - (1 + (sp.Rational(1, 4) - 4 * c) * t**2)) == 0
    assert sp.solve(eta_c.coeff(t, 2), c) == [sp.Rational(1, 16)]
    A_ser = sum(coefficient * t**order for order, coefficient in endpoint.items())
    B_ser = sum(coefficient * t**order for order, coefficient in central.items())
    eta_flat = sp.series(
        (sp.diff(B_ser, t) - sp.diff(A_ser, t) * B_ser / A_ser)
        * 4 / (4 * A_ser**2 - B_ser**2), t, 0, 8
    ).removeO()
    assert sp.expand(eta_flat) == 1

    # eq:NY-form: A^(1/2) = 2F1(1/4,1/4;1/2;s^2/4) and
    # t = s 2F1(3/4,3/4;3/2;s^2/4) / 2F1(1/4,1/4;1/2;s^2/4), s = B/A.
    def hyp(p, q, r, z, n):
        return sum(
            sp.rf(p, k) * sp.rf(q, k) / (sp.rf(r, k) * sp.factorial(k)) * z**k
            for k in range(n + 1)
        )

    s_ser = sp.series(B_ser / A_ser, t, 0, 9).removeO()
    z_ser = s_ser**2 / 4
    h0 = hyp(sp.Rational(1, 4), sp.Rational(1, 4), sp.Rational(1, 2), z_ser, 4)
    h1 = hyp(sp.Rational(3, 4), sp.Rational(3, 4), sp.Rational(3, 2), z_ser, 4)
    assert sp.expand(
        sp.series(sp.sqrt(A_ser) - h0, t, 0, 9).removeO()
    ) == 0
    assert sp.expand(sp.series(s_ser * h1 / h0, t, 0, 9).removeO() - t) == 0
    print("residue normalization eta(d_tW_c,1) = 1+(1/4-4c)t^2+...: c = 1/16: PASS")
    print("flat lambda(t), mu(t): eta(d_tW,1) = 1 through t^7: PASS")
    print("hypergeometric form of the marginal flat coordinate through t^8: PASS")

    print("reflection-fixed full-block invariants:")
    for order in sorted(values):
        print(f"  n={order}: {next(iter(values[order].values()))}")
    print("x^4 coefficient series lambda(t):", endpoint)
    print("x^2y^2 coefficient series mu(t):", central)
    print("oscillatory inverse equations varpi_0=1, varpi_2=t: PASS")
    print("chamber-index relations verified:", len(relations))
    print("PRIMARY MARGINAL SAITO SERIES THROUGH ORDER 8: PASS")


if __name__ == "__main__":
    main()
