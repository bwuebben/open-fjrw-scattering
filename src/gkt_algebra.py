#!/usr/bin/env python3
r"""
gkt_algebra.py -- verification of the Landau-Ginzburg wall-crossing algebra of
[GKT] = arXiv:2203.02435v2 (Definition 4.22), and of the fact that a scattering
diagram for open FJRW theory must be
seeded off GKT's actual Landau-Ginzburg wall-crossing Lie algebra.

The definitive objects, from [GKT] = arXiv:2203.02435v2:

  * Def 4.22:  g^{r,s}_{A_I,(k1,k2)} is spanned by  u_{J,d} X_{k1,k2}  with
        k1 = r(J) mod r,   k2 = s(J) mod s,   s*k1 + r*k2 = m(J,d) - rs,
    where  X_{k1,k2} = x^{k1} y^{k2} ((k2+1) x d_x - (k1+1) y d_y)  (Hamiltonian of
    x^{k1+1}y^{k2+1} for dx^dy),  m(J,d) = rs + sum_j (s a_j + r b_j + rs(d_j - 1)),
    r(J) = sum a_j mod r in {0..r-1}, s(J) = sum b_j mod s.
  * Prop 4.24: closure under bracket <=> additivity of  m(J,d) - rs.
  * Notation 2.38: for fixed (J,d), writing sum a_j = r(J) + l1*r, sum b_j = s(J) + l2*s,
    N := l1 + l2 - |J| + 1 + sum d_j:
      Balanced(J,d) = {Gamma_{J,p} : 0<=p<=N},  boundary degrees k1 = r(J)+p*r,
                                                                 k2 = s(J)+(N-p)*s;
      Crit(J,d)     = {Lambda_{J,p} : 1<=p<=N}, boundary degrees k1 = r(J)+(p-1)*r+1,
                                                                 k2 = s(J)+(N-p)*s+1,
    and (Prop 2.37) the Lie-algebra degrees of Lambda_{J,p} are (#r-points - 1,
    #s-points - 1) -- the (1,1)-shift = removing the single fully-twisted point sigma_12
    that balanced graphs carry and critical graphs do not.

Certified here (exact arithmetic, no floats):
  (1) additivity of the Def 4.22 constraint (bracket closure), symbolically over
      random multisets;
  (2) the naive seeds t_{2,3}X_{1,2}, t_{3,2}X_{2,1} (r=s=5) and t_{2,2}X_{1,1}
      (r=s=4) VIOLATE Def 4.22; the naive leading wall 5 t_{2,3}t_{3,2} X_{3,3} too;
  (3) x^4+y^4 has NO d=0 critical graphs at all (all primary deformations have
      non-positive weight; only (2,2) is marginal), so primary open invariants of
      x^4+y^4 do not wall-cross; first walls are descendent-seeded (t_{a,b,1}X_{a,b});
  (4) x^5+y^5: first primary (d=0) wall is the critical graph J={(3,3),(3,3)}, N=1,
      Lie degrees (1,1): element t_{3,3}^2 X_{1,1}; enumeration of all d=0 walls to
      t-order 4;
  (5) the balanced/critical (N+1 vs N) count of Notation 2.38 reproduced by brute
      enumeration of the congruence + dimension constraints;
  (6) g^{r,s}.W_0 is period-invisible: for v = u X_{k1,k2} in g^{r,s},
      v(W_0) = u (r(k2+1) x^{k1+r} y^{k2} - s(k1+1) x^{k1} y^{k2+s})  has vanishing
      good-basis periods, via P_a(p) = -hbar(p-r+1)/r * P_a(p-r)  (Lemma 4.1);
      verified symbolically in hbar;
  (7) the balanced support (potential monomials) IS the E-homogeneity line
      s*k1 + r*k2 = m(J,d) (Thm 0.2/(0.6)); the naive target monomial
      t_{2,3}t_{3,2} x^{r-1}y^{s-1} violates it (its legal home at r=s=5 is
      t_{3,3}^3 x^4 y^4, a single-balanced-graph N=0 coefficient with NO wall).

Run: ./venv/bin/python src/gkt_algebra.py
"""

from itertools import combinations_with_replacement
from fractions import Fraction
import random

import sympy as sp


def m_minus_rs(r, s, J, d):
    """m(J,d) - rs = sum_j (s a_j + r b_j + rs(d_j-1)) for J = [(a,b),...], d = [d_j]."""
    return sum(s * a + r * b + r * s * (dj - 1) for (a, b), dj in zip(J, d))


def rJ(r, J):
    return sum(a for a, _ in J) % r


def sJ(s, J):
    return sum(b for _, b in J) % s


def lie_degrees(r, s, J, d):
    """All (k1,k2), k1,k2>=0, (k1,k2)!=(0,0), satisfying Def 4.22 for (J,d)."""
    M = m_minus_rs(r, s, J, d)
    out = []
    if M < 0:
        return out
    for k1 in range(M // s + 1):
        rem = M - s * k1
        if rem % r:
            continue
        k2 = rem // r
        if (k1, k2) == (0, 0):
            continue
        if k1 % r == rJ(r, J) and k2 % s == sJ(s, J):
            out.append((k1, k2))
    return out


def balanced_degrees(r, s, J, d):
    """All (k1,k2) with s k1 + r k2 = m(J,d) and the congruences: potential support."""
    M = m_minus_rs(r, s, J, d) + r * s
    out = []
    if M < 0:
        return out
    for k1 in range(M // s + 1):
        rem = M - s * k1
        if rem % r:
            continue
        k2 = rem // r
        if k1 % r == rJ(r, J) and k2 % s == sJ(s, J):
            out.append((k1, k2))
    return out


def notation238_counts(r, s, J, d):
    """(N, balanced list, crit list) per Notation 2.38."""
    sum_a = sum(a for a, _ in J)
    sum_b = sum(b for _, b in J)
    l1 = (sum_a - rJ(r, J)) // r
    l2 = (sum_b - sJ(s, J)) // s
    N = l1 + l2 - len(J) + 1 + sum(d)
    bal = [(rJ(r, J) + p * r, sJ(s, J) + (N - p) * s) for p in range(N + 1)] if N >= 0 else []
    crit = [(rJ(r, J) + (p - 1) * r + 1, sJ(s, J) + (N - p) * s + 1) for p in range(1, N + 1)]
    return N, bal, crit


def check_additivity(trials=500, seed=1):
    """Prop 4.24: the constraint quantity m(J,d)-rs is additive under disjoint union."""
    rng = random.Random(seed)
    for _ in range(trials):
        r = rng.randint(2, 9)
        s = rng.randint(2, 9)
        J1 = [(rng.randint(0, r - 2), rng.randint(0, s - 2)) for _ in range(rng.randint(1, 4))]
        J2 = [(rng.randint(0, r - 2), rng.randint(0, s - 2)) for _ in range(rng.randint(1, 4))]
        d1 = [rng.randint(0, 3) for _ in J1]
        d2 = [rng.randint(0, 3) for _ in J2]
        lhs = m_minus_rs(r, s, J1 + J2, d1 + d2)
        rhs = m_minus_rs(r, s, J1, d1) + m_minus_rs(r, s, J2, d2)
        assert lhs == rhs, (r, s, J1, J2)
    return True


def check_naive_seeds_violate():
    """The naive seeding is not in g^{r,s} (Def 4.22)."""
    viol = []
    # r=s=5: t_{2,3} X_{1,2}, t_{3,2} X_{2,1}, and the naive leading wall t23 t32 X_{3,3}
    assert (1, 2) not in lie_degrees(5, 5, [(2, 3)], [0]);  viol.append(("5,5", "t23*X_{1,2}"))
    assert (2, 1) not in lie_degrees(5, 5, [(3, 2)], [0]);  viol.append(("5,5", "t32*X_{2,1}"))
    assert (3, 3) not in lie_degrees(5, 5, [(2, 3), (3, 2)], [0, 0]); viol.append(("5,5", "t23*t32*X_{3,3}"))
    # in fact the (J,d)=({(2,3)},0) graded piece is EMPTY (marginal deformation):
    assert lie_degrees(5, 5, [(2, 3)], [0]) == []
    assert lie_degrees(5, 5, [(2, 3), (3, 2)], [0, 0]) == []
    # r=s=4: naive first wall t_{2,2}(^2) X_{1,1}
    assert (1, 1) not in lie_degrees(4, 4, [(2, 2)], [0]); viol.append(("4,4", "t22*X_{1,1}"))
    assert lie_degrees(4, 4, [(2, 2), (2, 2)], [0, 0]) == []
    return viol


def enumerate_d0_walls(r, s, max_l):
    """All d=0 graded pieces (J multiset, Lie degrees) with nonempty g^{r,s}, |J|<=max_l."""
    twists = [(a, b) for a in range(r - 1) for b in range(s - 1)]
    found = []
    for l in range(1, max_l + 1):
        for J in combinations_with_replacement(twists, l):
            degs = lie_degrees(r, s, list(J), [0] * l)
            if degs:
                found.append((tuple(J), degs))
    return found


def check_x4y4_no_primary_walls():
    # every twist (a,b), a,b<=2 has s a + r b - rs = 4a+4b-16 <= 0, marginal iff (2,2)
    assert all(4 * a + 4 * b - 16 <= 0 for a in range(3) for b in range(3))
    assert enumerate_d0_walls(4, 4, 6) == []
    # first walls are descendent-seeded: t_{a,b,1} X_{a,b} for every twist != (0,0)
    for a in range(3):
        for b in range(3):
            degs = lie_degrees(4, 4, [(a, b)], [1])
            if (a, b) == (0, 0):
                # weight-0 monomial: sk1+rk2 = 0 has no admissible (k1,k2);
                # the unit-twist walls first appear at d=2 on the boundary rays
                assert degs == []
                assert lie_degrees(4, 4, [(0, 0)], [2]) == [(0, 4), (4, 0)]
            else:
                assert degs == [(a, b)], ((a, b), degs)
    return True


def check_x5y5_first_wall():
    walls = enumerate_d0_walls(5, 5, 4)
    # the l=2 walls: {(2,3),(3,3)} -> (0,1) [axis!], {(3,2),(3,3)} -> (1,0) [axis!],
    # {(3,3),(3,3)} -> (1,1).  NOTE k1=0 or k2=0 occurs: GKT's algebra is NOT confined
    # to the open cone k1,k2>=1 the paper assumed.
    l2 = [(J, degs) for J, degs in walls if len(J) == 2]
    assert l2 == [
        (((2, 3), (3, 3)), [(0, 1)]),
        (((3, 2), (3, 3)), [(1, 0)]),
        (((3, 3), (3, 3)), [(1, 1)]),
    ], l2
    J = [(3, 3), (3, 3)]
    assert lie_degrees(5, 5, J, [0, 0]) == [(1, 1)]
    N, bal, crit = notation238_counts(5, 5, J, [0, 0])
    assert N == 1 and bal == [(1, 6), (6, 1)] and crit == [(2, 2)]
    # Lie degrees = crit boundary degrees - (1,1)  (Prop 2.37: remove the sigma_12 shift)
    assert [(k1 - 1, k2 - 1) for k1, k2 in crit] == lie_degrees(5, 5, J, [0, 0])
    return walls


def check_notation238_vs_bruteforce(trials=300, seed=2):
    rng = random.Random(seed)
    for _ in range(trials):
        r = rng.randint(2, 8)
        s = rng.randint(2, 8)
        l = rng.randint(1, 4)
        J = [(rng.randint(0, r - 2), rng.randint(0, s - 2)) for _ in range(l)]
        d = [rng.randint(0, 2) for _ in J]
        N, bal, crit = notation238_counts(r, s, J, d)
        assert sorted(bal) == sorted(balanced_degrees(r, s, J, d)), (r, s, J, d)
        lie = lie_degrees(r, s, J, d)
        shifted = {(k1 - 1, k2 - 1) for (k1, k2) in crit}
        # (1,1)-shifted critical boundary degrees = Def 4.22 solutions, except that a
        # critical graph of Lie degree (0,0) (E-weight-0; the excluded torus direction
        # x d_x - y d_y, kept out of g^{r,s} for nilpotency) carries no group summand.
        assert shifted - {(0, 0)} == set(lie), (r, s, J, d, sorted(shifted), lie)
    return True


def check_period_invisibility():
    """v(W_0) has vanishing good-basis periods, symbolically in hbar.

    P_a(p): p<=r-2: delta_{ap}; p=r-1: 0; else -hbar(p-r+1)/r * P_a(p-r).
    v = X_{k1,k2}: v(W_0) = r(k2+1) x^{k1+r} y^{k2} - s(k1+1) x^{k1} y^{k2+s}.
    Period against Xi_{a,b}: r(k2+1) P_a(k1+r) P_b(k2) - s(k1+1) P_a(k1) P_b(k2+s)
      = r(k2+1) * (-hbar(k1+1)/r) P_a(k1) P_b(k2)
        - s(k1+1) * (-hbar(k2+1)/s) P_a(k1) P_b(k2)  ... wait sign: P_b(k2+s) =
        -hbar(k2+1)/s P_b(k2).  Sum = -hbar(k1+1)(k2+1) P P + hbar(k1+1)(k2+1) P P = 0.
    Certified numerically-symbolically below for all k1,k2 <= 12, all (a,b), r,s in
    {3,4,5,6}.
    """
    hbar = sp.Symbol("hbar")

    def P(r, a, p):
        if p < 0:
            return sp.Integer(0)
        if p <= r - 2:
            return sp.Integer(1) if p == a else sp.Integer(0)
        if p == r - 1:
            return sp.Integer(0)
        return -hbar * sp.Rational(p - r + 1, r) * P(r, a, p - r)

    for r in (3, 4, 5, 6):
        s = r  # and one asymmetric case below
        for k1 in range(13):
            for k2 in range(13):
                if (k1, k2) == (0, 0):
                    continue
                for a in range(r - 1):
                    for b in range(s - 1):
                        expr = r * (k2 + 1) * P(r, a, k1 + r) * P(s, b, k2) \
                             - s * (k1 + 1) * P(r, a, k1) * P(s, b, k2 + s)
                        assert sp.simplify(expr) == 0, (r, s, k1, k2, a, b)
    r, s = 3, 5
    for k1 in range(10):
        for k2 in range(10):
            if (k1, k2) == (0, 0):
                continue
            for a in range(r - 1):
                for b in range(s - 1):
                    expr = r * (k2 + 1) * P(r, a, k1 + r) * P(s, b, k2) \
                         - s * (k1 + 1) * P(r, a, k1) * P(s, b, k2 + s)
                    assert sp.simplify(expr) == 0, (r, s, k1, k2, a, b)
    return True


def check_naive_target_monomial():
    """t_{2,3}t_{3,2} x^4 y^4 violates E-homogeneity; the legal home of x^4y^4 at
    r=s=5 is t_{3,3}^3 (N=0, no wall)."""
    # naive target: J={(2,3),(3,2)}, monomial (4,4): needs 5*4+5*4 = m(J,0) = 25
    assert 5 * 4 + 5 * 4 != m_minus_rs(5, 5, [(2, 3), (3, 2)], [0, 0]) + 25
    # legal balanced support of that J: (5,0),(0,5)
    assert sorted(balanced_degrees(5, 5, [(2, 3), (3, 2)], [0, 0])) == [(0, 5), (5, 0)]
    # x^4y^4 lives at J={(3,3)}^3, d=0, with N=0 (single balanced graph, no critical):
    J3 = [(3, 3)] * 3
    assert (4, 4) in balanced_degrees(5, 5, J3, [0, 0, 0])
    N, bal, crit = notation238_counts(5, 5, J3, [0, 0, 0])
    assert (N, bal, crit) == (0, [(4, 4)], [])
    return True


def main():
    print("== GKT Def 4.22 wall-crossing algebra: verification ==")
    assert check_additivity()
    print("[1] bracket closure (additivity of m(J,d)-rs): OK (500 random cases)")
    viol = check_naive_seeds_violate()
    print(f"[2] naive seeds violate Def 4.22: {', '.join(w for _, w in viol)}  -- all confirmed NOT in g^(r,s)")
    assert check_x4y4_no_primary_walls()
    print("[3] x^4+y^4: NO d=0 critical graphs (primary invariants do not wall-cross);")
    print("    first walls are descendent-seeded t_{a,b,1}X_{a,b}")
    walls = check_x5y5_first_wall()
    print("[4] x^5+y^5 d=0 walls to t-order 4:")
    for J, degs in walls:
        print(f"      J={list(J)}  ->  X degrees {degs}")
    print("    first wall: t_{3,3}^2 X_{1,1}; Notation 2.38: N=1, balanced {(1,6),(6,1)}, crit {(2,2)};")
    print("    Lie degrees = crit boundary degrees - (1,1)   [the sigma_12 / primitive-form shift]")
    assert check_notation238_vs_bruteforce()
    print("[5] Notation 2.38 classification == brute-force constraint enumeration: OK (300 random cases)")
    assert check_period_invisibility()
    print("[6] g^{r,s}.W_0 is period-invisible (symbolic in hbar; r=s=3..6 and (r,s)=(3,5)): OK")
    assert check_naive_target_monomial()
    print("[7] naive l=3 target t_23 t_32 x^4y^4 violates E-homogeneity; legal home of x^4y^4")
    print("    at r=s=5 is t_{3,3}^3 (N=0: single balanced graph, NO wall)")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
