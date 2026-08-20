#!/usr/bin/env python3
r"""
The first PRIMARY diagonal of x^5+y^5 carrying two walls on two distinct rays.

N_2 is the least number of primary insertions admitting a diagonal with N >= 2.
For r = s = 5, N_2 = 5 and is realized by the multiset

    J = { (3,3) x 5 },   d = 0.

This is the lowest primary degree for x^5+y^5 at which one global
Lemma-3.39 normal-form family would have to carry two rays at once. The
stagewise realization theorem separates the rays by using one such family per
stage. Nothing in the earlier computed sectors reaches this obstruction: every
diagonal there has N <= 1.

Computed here, exactly:
  (1) the structure of the diagonal -- N = 2, the three balanced profiles, the
      two critical graphs and their rays, and that those rays are distinct;
  (2) the two wall vectors w_p = a_p e_p - b_p e_{p-1};
  (3) the chamber-index condition, which is A(J,0,nu) = 0 since d(J,0) < 0, as a
      Gamma-weighted linear relation on the three balanced coefficients with
      right-hand side supplied by the lower orders;
  (4) the resulting extreme value nu^min_{(0,10)}.

Lower-order input, in the minimal gauge of Thm 1.3 (only the e_0 coefficient
survives on each diagonal), all previously verified in this repository:
    |J| = 1   nu_{(3,3)}  = 1        (Thm 5.1 normalization)
    |J| = 2   nu_{(1,6)}  = -2/5,  nu_{(6,1)} = 0
    |J| = 3   nu_{(4,4)}  = 1/5     (N = 0; gauge independent)
    |J| = 4   nu_{(2,7)}  = -12/25, nu_{(7,2)} = 0

Run: ./venv/bin/python src/first_two_ray_diagonal.py
"""
from fractions import Fraction

import sympy as sp

from a_invariants import (A_invariant, d_of, m_of, nu_from_dict, part_boundaries,
                          r_of, s_of)

R = S = 5
T = (3, 3, 0)          # the twist (3,3), no descendent


def diagonal_structure(J, r, s):
    """(N, balanced profiles, critical graphs with their rays)."""
    sa = sum(a for a, _, _ in J)
    sb = sum(b for _, b, _ in J)
    rJ, l1 = sa % r, sa // r
    sJ, l2 = sb % s, sb // s
    N = l1 + l2 - len(J) + 1 + sum(d for _, _, d in J)
    bal = [(rJ + p * r, sJ + (N - p) * s) for p in range(N + 1)]
    crit = [(rJ + (p - 1) * r + 1, sJ + (N - p) * s + 1) for p in range(1, N + 1)]
    return N, bal, crit


def check_lower_orders():
    """Re-derive the |J| = 2, 3, 4 values with the SAME engine and the same gauge.

    If the engine reproduces the published -2/5, 1/5 and -12/25 from the
    chamber-index conditions alone, the order-5 value computed below rests on a
    validated pipeline rather than on transcribed inputs.
    """
    print("=" * 68)
    print("[0] self-check: re-derive the lower orders from the conditions")
    print("=" * 68)
    known = {((T,), 3, 3): sp.Integer(1)}

    # |J| = 2 : two balanced points, one relation -> the first-wall sum
    J2 = [T] * 2
    a, b = sp.symbols("a b")
    bnd2 = part_boundaries(J2, R, S)
    k = dict(zip(bnd2, (a, b)))
    for pt, sym in k.items():
        known[((T,) * 2, pt[0], pt[1])] = sym
    e2 = sp.expand(A_invariant(J2, R, S, nu_from_dict(known)))
    got_sum = sp.nsimplify(sp.solve(e2.subs(b, sp.Symbol("S_") - a), sp.Symbol("S_"))[0])
    print(f"  |J|=2  profiles {bnd2}:  A = 0  =>  sum = {got_sum}")
    assert got_sum == sp.Rational(-2, 5), got_sum
    # minimal gauge: e_1 = 0
    v_min2 = sp.nsimplify(sp.solve(e2.subs(b, 0), a)[0])
    assert v_min2 == sp.Rational(-2, 5), v_min2
    known[((T,) * 2, bnd2[0][0], bnd2[0][1])] = sp.Rational(-2, 5)
    known[((T,) * 2, bnd2[1][0], bnd2[1][1])] = sp.Integer(0)
    print(f"         minimal gauge nu_{bnd2[0]} = {v_min2}   [paper Ex. 5.1]  OK")

    # |J| = 3 : N = 0, a single balanced point, absolutely determined
    J3 = [T] * 3
    bnd3 = part_boundaries(J3, R, S)
    c = sp.Symbol("c")
    known[((T,) * 3, bnd3[0][0], bnd3[0][1])] = c
    e3 = sp.expand(A_invariant(J3, R, S, nu_from_dict(known)))
    v3 = sp.nsimplify(sp.solve(e3, c)[0])
    print(f"  |J|=3  profile  {bnd3}:  nu = {v3}   [paper eq. (3)]  OK")
    assert v3 == sp.Rational(1, 5), v3
    known[((T,) * 3, bnd3[0][0], bnd3[0][1])] = v3

    # |J| = 4 : two balanced points; minimal gauge kills e_1
    J4 = [T] * 4
    bnd4 = part_boundaries(J4, R, S)
    p, q = sp.symbols("p q")
    known[((T,) * 4, bnd4[0][0], bnd4[0][1])] = p
    known[((T,) * 4, bnd4[1][0], bnd4[1][1])] = q
    e4 = sp.expand(A_invariant(J4, R, S, nu_from_dict(known)))
    v4 = sp.nsimplify(sp.solve(e4.subs(q, 0), p)[0])
    print(f"  |J|=4  profiles {bnd4}:  nu^min_{bnd4[0]} = {v4}   [paper Rem. 4.2]  OK")
    assert v4 == sp.Rational(-12, 25), v4
    print("  engine reproduces every published value on this tower.")
    print()


def main():
    check_lower_orders()
    J5 = [T] * 5
    N, bal, crit = diagonal_structure(J5, R, S)

    print("=" * 68)
    print("[1] the diagonal  J = {(3,3) x 5},  d = 0,  at r = s = 5")
    print("=" * 68)
    print(f"  m(J,0)      = {m_of(J5, R, S)}      (Euler weight w = m - rs = {m_of(J5,R,S) - R*S})")
    print(f"  r(J), s(J)  = {r_of(J5, R)}, {s_of(J5, S)}")
    print(f"  N           = {N}")
    print(f"  balanced    = {bal}")
    assert bal == part_boundaries(J5, R, S), "balanced set must match the engine"
    print(f"  critical    = {crit}   (boundary degrees = rays)")

    # the two rays, as primitive directions
    from math import gcd
    rays = [(k1 // gcd(k1, k2), k2 // gcd(k1, k2)) for k1, k2 in crit]
    print(f"  rays        = {rays}")
    assert N == 2 and len(set(rays)) == 2, "expected two walls on two distinct rays"
    print("  -> TWO walls on TWO DISTINCT rays: the first such primary diagonal.")

    # sanity: the dimension constraint s k1 + r k2 = w on the LIE bidegree
    w = m_of(J5, R, S) - R * S
    for (k1, k2) in crit:
        assert S * (k1 - 1) + R * (k2 - 1) == w
    print(f"  dimension constraint s k1 + r k2 = w = {w}: OK for both walls")

    print()
    print("=" * 68)
    print("[2] wall vectors  w_p = a_p e_p - b_p e_{p-1}")
    print("=" * 68)
    for p, (k1, k2) in enumerate(crit, start=1):
        a_p, b_p = R * k2, S * k1
        print(f"  Lambda_{p}: boundary {(k1,k2)}  ->  w_{p} = {a_p} e_{p} - {b_p} e_{p-1}")
    print("  (triangular, hence independent: 3 coefficients - 2 torsor directions")
    print("   = 1 invariant combination, matching Thm 1.4)")

    print()
    print("=" * 68)
    print("[3] the chamber-index condition")
    print("=" * 68)
    dJ = d_of(J5, R, S)
    print(f"  d(J,0) = {dJ} < 0, so the condition is  A(J,0,nu) = 0  [GKT Def 3.46]")

    # lower-order data, minimal gauge
    known = {
        ((T,), 3, 3): sp.Integer(1),
        ((T,) * 2, 1, 6): sp.Rational(-2, 5),
        ((T,) * 2, 6, 1): sp.Integer(0),
        ((T,) * 3, 4, 4): sp.Rational(1, 5),
        ((T,) * 4, 2, 7): sp.Rational(-12, 25),
        ((T,) * 4, 7, 2): sp.Integer(0),
    }
    v0, v1, v2 = sp.symbols("v0 v1 v2")          # nu at bal[0], bal[1], bal[2]
    for (k1, k2), sym in zip(bal, (v0, v1, v2)):
        known[((T,) * 5, k1, k2)] = sym

    expr = sp.expand(A_invariant(J5, R, S, nu_from_dict(known)))
    print(f"  A(J,0,nu) = {expr}")
    lhs = sp.expand(expr - expr.subs({v0: 0, v1: 0, v2: 0}))
    rhs = -expr.subs({v0: 0, v1: 0, v2: 0})
    print(f"  relation:   {sp.simplify(lhs)}  =  {sp.nsimplify(rhs)}")

    print()
    print("=" * 68)
    print("[4] the minimal normal form")
    print("=" * 68)
    sol = sp.solve(expr.subs({v1: 0, v2: 0}), v0)
    assert len(sol) == 1
    nu_min = sp.nsimplify(sol[0])
    print(f"  Thm 1.3 minimal gauge kills e_1, e_2; the condition then gives")
    print(f"      nu^min_({bal[0][0]},{bal[0][1]}) = {nu_min}")
    print(f"  i.e.  <tau_0^(3,3)^5 sigma_1^{bal[0][0]} sigma_2^{bal[0][1]} sigma_12>^min = {nu_min}")

    # cross-check: the maximal gauge kills e_0, e_1
    sol_max = sp.solve(expr.subs({v0: 0, v1: 0}), v2)
    print(f"  and symmetrically nu^max_({bal[2][0]},{bal[2][1]}) = {sp.nsimplify(sol_max[0])}")

    print()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
