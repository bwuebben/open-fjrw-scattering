#!/usr/bin/env python3
r"""
mixed_sector.py -- the t23-t33 sector of x^5+y^5 through order t^4: the first true
scattering vertex of the corrected theory (noncommuting walls; the commutator of the
order-2 walls feeds the (1,2)-direction wall), and the anchor test with bending-generated
N=0 slots.

Sector diagonals (d=0, variables a := t_{2,3}, b := t_{3,3}), all A-conditions reduce to
vanishings (closed values die by fractional degree or the string equation):
  order 2: ab  (N=1, wall X_{0,1} ray (1,2)): (2/5)nu06+(1/5)nu51 = -2/25   [known]
           b^2 (N=1, wall X_{1,1} ray (1,1)): nu16+nu61 = -2/5              [known]
           a^2 (marginal, N=0, invariant): nu41 = -2/5  [NEW absolute invariant
               <tau0^{(2,3)}tau0^{(2,3)} sigma_1^4 sigma_2 sigma_12> = -2/5]
  order 3: b^3 (N=0, jump-safe): nu44 = 1/5 [known]; a b^2 (N=0, (3,4)-slot, JUMPS);
           a^2 b (N=0, (2,4)-slot); a^3 (marginal, (1,4)-slot)
  order 4: a b^3 (N=1: (1,7),(6,2); wall X_{1,2} ray (2,3));
           a^2 b^2 (N=1: (0,7),(5,2); wall X_{0,2} ray (1,3));
           b^4 (N=1: known, wall coefficient 0); a^3 b ((4,2)); a^4 ((3,2), marginal)

Certified here:
 [1] the A-conditions solved on every sector diagonal in the min and max gauges
     (labeled-convention engine); in particular the new absolute invariant nu41 = -2/5;
 [2] the canonical boundary-ray factorization
     g^{min->max} = theta_(1,3) o theta_(1,2) o theta_(2,3) o theta_(1,1)
     (slope-ordered, bottom-up: rays (1,1) < (2,3) < (1,2) < (1,3)),
     with the two order-4 wall coefficients (c12 on ray (2,3), c02 on ray (1,3))
     solved from g(W^min) = W^max -- an overdetermined nonlinear system whose consistency
     tests the framework INCLUDING the commutator [theta_(1,2), theta_(1,1)] contribution;
 [3] (deferred) the ANCHOR with own-direction-chamber seed values; an earlier apparent
     "open inconsistency" was traced to a bug in this test script (|Aut|=2 dressing
     missing on a b^2; a^3 b and a^4 slots omitted) -- framework fully consistent.

Run: ./venv/bin/python src/mixed_sector.py
"""

import os
import sys

import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a_invariants import A_invariant, d_of, nu_from_dict, part_boundaries  # noqa: E402

x, y, a, b = sp.symbols("x y a b")  # a = t23, b = t33
T23, T33 = (2, 3, 0), (3, 3, 0)


def X(k1, k2, f):
    return sp.expand(x**k1 * y**k2 * ((k2 + 1) * x * sp.diff(f, x)
                                      - (k1 + 1) * y * sp.diff(f, y)))


def trunc(f):
    f = sp.expand(f)
    out = sp.Integer(0)
    for term in f.as_ordered_terms():
        if sp.degree(term, a) + sp.degree(term, b) <= 4:
            out += term
    return out


def exp_action(v, f):
    out, term, fact = sp.Integer(0), f, 1
    for n in range(4):
        out += term / fact
        term = trunc(v(term))
        fact *= (n + 1)
        if term == 0:
            break
    return trunc(sp.expand(out))


def solve_diagonal(J, numap, unknowns, rhs=0):
    A = A_invariant(J, 5, 5, nu_from_dict(numap))
    sols = sp.solve(sp.Eq(A, rhs), unknowns, dict=True)
    return A, sols


def main():
    print("== The t23-t33 sector through t^4: scattering vertex + anchor ==")

    # ---------------- [1] A-condition solves --------------------------------
    # a^2 (marginal, single slot (4,1)):
    nu41 = sp.Symbol("nu41")
    A, _ = solve_diagonal([T23, T23], {((T23, T23), 4, 1): nu41}, [nu41])
    v41 = sp.solve(sp.Eq(A, 0), nu41)[0]
    assert v41 == sp.Rational(-2, 5), A
    print(f"[1] NEW absolute invariant <tau0^(2,3)^2 s1^4 s2 s12> = {v41}  (marginal)")

    # a b^2, (3,4)-slot (jumps; per-gauge value from its vanishing condition):
    nu34 = sp.Symbol("nu34")
    known_min = {((T23, T33), 0, 6): sp.Rational(-1, 5), ((T23, T33), 5, 1): 0,
                 ((T33, T33), 1, 6): sp.Rational(-2, 5), ((T33, T33), 6, 1): 0}
    known_max = {((T23, T33), 0, 6): 0, ((T23, T33), 5, 1): sp.Rational(-2, 5),
                 ((T33, T33), 1, 6): 0, ((T33, T33), 6, 1): sp.Rational(-2, 5)}
    J_ab2 = [T23, T33, T33]
    assert part_boundaries(J_ab2, 5, 5) == [(3, 4)]
    A, _ = solve_diagonal(J_ab2, {**known_min, (tuple(sorted(J_ab2)), 3, 4): nu34}, [nu34])
    v34_min = sp.solve(sp.Eq(A, 0), nu34)[0]
    A, _ = solve_diagonal(J_ab2, {**known_max, (tuple(sorted(J_ab2)), 3, 4): nu34}, [nu34])
    v34_max = sp.solve(sp.Eq(A, 0), nu34)[0]
    print(f"    a b^2 (3,4)-slot: min {v34_min}, max {v34_max}  (jumps, as predicted)")

    # a^2 b, (2,4)-slot:
    nu24 = sp.Symbol("nu24")
    J_a2b = [T23, T23, T33]
    assert part_boundaries(J_a2b, 5, 5) == [(2, 4)]
    A, _ = solve_diagonal(J_a2b, {**known_min, ((T23, T23), 4, 1): v41,
                                  (tuple(sorted(J_a2b)), 2, 4): nu24}, [nu24])
    v24_min = sp.solve(sp.Eq(A, 0), nu24)[0]
    A, _ = solve_diagonal(J_a2b, {**known_max, ((T23, T23), 4, 1): v41,
                                  (tuple(sorted(J_a2b)), 2, 4): nu24}, [nu24])
    v24_max = sp.solve(sp.Eq(A, 0), nu24)[0]
    print(f"    a^2 b (2,4)-slot: min {v24_min}, max {v24_max}")

    # a^3 ((1,4)-slot, marginal):
    nu14 = sp.Symbol("nu14")
    J_a3 = [T23] * 3
    assert part_boundaries(J_a3, 5, 5) == [(1, 4)]
    A, _ = solve_diagonal(J_a3, {((T23, T23), 4, 1): v41,
                                 (tuple(sorted(J_a3)), 1, 4): nu14}, [nu14])
    v14 = sp.solve(sp.Eq(A, 0), nu14)[0]
    print(f"    a^3 (1,4)-slot (marginal, absolute): {v14}")

    # order-4 wall diagonals:
    def wall_diagonal(J, slots, known):
        n0, n1 = sp.symbols("n0 n1")
        numap = {**known, (tuple(sorted(J)), *slots[0]): n0,
                 (tuple(sorted(J)), *slots[1]): n1}
        A = A_invariant(J, 5, 5, nu_from_dict(numap))
        rel = sp.expand(A)
        vmin = sp.solve(sp.Eq(rel.subs(n1, 0), 0), n0)[0]   # min: kill larger-k1 slot
        vmax = sp.solve(sp.Eq(rel.subs(n0, 0), 0), n1)[0]
        return rel, vmin, vmax

    km3 = {**known_min, ((T23, T23), 4, 1): v41,
           (tuple(sorted(J_a3)), 1, 4): v14,
           (tuple(sorted(J_ab2)), 3, 4): v34_min,
           (tuple(sorted(J_a2b)), 2, 4): v24_min,
           ((T33, T33, T33), 4, 4): sp.Rational(1, 5)}
    kx3 = {**known_max, ((T23, T23), 4, 1): v41,
           (tuple(sorted(J_a3)), 1, 4): v14,
           (tuple(sorted(J_ab2)), 3, 4): v34_max,
           (tuple(sorted(J_a2b)), 2, 4): v24_max,
           ((T33, T33, T33), 4, 4): sp.Rational(1, 5)}
    J_ab3 = [T23, T33, T33, T33]
    assert part_boundaries(J_ab3, 5, 5) == [(1, 7), (6, 2)]
    _, v17_min, v62_max = (lambda r: r)(None) or (None, None, None)
    rel, v17_min, v62_max = wall_diagonal(J_ab3, [(1, 7), (6, 2)], km3)
    rel_x, v17_min_x, v62_max_x = wall_diagonal(J_ab3, [(1, 7), (6, 2)], kx3)
    v62_max = v62_max_x
    J_a2b2 = [T23, T23, T33, T33]
    assert part_boundaries(J_a2b2, 5, 5) == [(0, 7), (5, 2)]
    rel2, v07_min, v52_max_ = wall_diagonal(J_a2b2, [(0, 7), (5, 2)], km3)
    rel2x, _, v52_max = wall_diagonal(J_a2b2, [(0, 7), (5, 2)], kx3)
    print(f"    a b^3: min nu17 = {v17_min}; max nu62 = {v62_max}")
    print(f"    a^2b^2: min nu07 = {v07_min}; max nu52 = {v52_max}")

    # a^3 b ((4,2)-slot, N=0) and a^4 ((3,2)-slot, marginal) -- omitted before:
    nu42, nu32 = sp.symbols("nu42 nu32")
    J_a3b = [T23, T23, T23, T33]
    assert part_boundaries(J_a3b, 5, 5) == [(4, 2)]
    A, _ = solve_diagonal(J_a3b, {**km3, (tuple(sorted(J_a3b)), 4, 2): nu42}, [nu42])
    v42_min = sp.solve(sp.Eq(A, 0), nu42)[0]
    A, _ = solve_diagonal(J_a3b, {**kx3, (tuple(sorted(J_a3b)), 4, 2): nu42}, [nu42])
    v42_max = sp.solve(sp.Eq(A, 0), nu42)[0]
    J_a4 = [T23] * 4
    assert part_boundaries(J_a4, 5, 5) == [(3, 2)]
    A, _ = solve_diagonal(J_a4, {**km3, (tuple(sorted(J_a3)), 1, 4): v14,
                                 (tuple(sorted(J_a4)), 3, 2): nu32}, [nu32])
    v32_min = sp.solve(sp.Eq(A, 0), nu32)[0]
    A, _ = solve_diagonal(J_a4, {**kx3, (tuple(sorted(J_a3)), 1, 4): v14,
                                 (tuple(sorted(J_a4)), 3, 2): nu32}, [nu32])
    v32_max = sp.solve(sp.Eq(A, 0), nu32)[0]
    assert v32_min == v32_max  # marginal diagonal: gauge-independent (soft check)
    print(f"    a^3 b (4,2)-slot: min {v42_min}, max {v42_max};  a^4 (3,2): {v32_min} (invariant)")

    # ---------------- [2] the canonical factorization -----------------------
    def dress(l, aut, nu):
        return sp.Integer(-1) ** (l - 1) * nu / aut

    Wmin = trunc(
        x**5 + y**5 + a * x**2 * y**3 + b * x**3 * y**3
        + sp.Rational(1, 5) * a * b * y**6                      # -(nu06) y^6, nu06=-1/5
        + sp.Rational(1, 5) * b**2 * x * y**6                   # -(1/2)nu16, nu16=-2/5
        + sp.Rational(1, 5) * a**2 * x**4 * y                   # -(1/2)nu41, nu41=-2/5
        + dress(3, 2, v34_min) * a * b**2 * x**3 * y**4
        + dress(3, 2, v24_min) * a**2 * b * x**2 * y**4
        + dress(3, 6, v14) * a**3 * x * y**4
        + sp.Rational(1, 30) * b**3 * x**4 * y**4               # nu44/6, nu44=1/5
        + dress(4, 6, v17_min) * a * b**3 * x * y**7
        + dress(4, 4, v07_min) * a**2 * b**2 * y**7
        + dress(4, 6, v42_min) * a**3 * b * x**4 * y**2
        + dress(4, 24, v32_min) * a**4 * x**3 * y**2
        + sp.Rational(1, 50) * b**4 * x**2 * y**7)              # -(nu27)/24, nu27=-12/25
    Wmax = trunc(
        x**5 + y**5 + a * x**2 * y**3 + b * x**3 * y**3
        + sp.Rational(2, 5) * a * b * x**5 * y                  # -(nu51), nu51=-2/5
        + sp.Rational(1, 5) * b**2 * x**6 * y
        + sp.Rational(1, 5) * a**2 * x**4 * y                   # marginal: invariant
        + dress(3, 2, v34_max) * a * b**2 * x**3 * y**4
        + dress(3, 2, v24_max) * a**2 * b * x**2 * y**4
        + dress(3, 6, v14) * a**3 * x * y**4
        + sp.Rational(1, 30) * b**3 * x**4 * y**4
        + dress(4, 6, v62_max) * a * b**3 * x**6 * y**2
        + dress(4, 4, v52_max) * a**2 * b**2 * x**5 * y**2
        + dress(4, 6, v42_max) * a**3 * b * x**4 * y**2
        + dress(4, 24, v32_max) * a**4 * x**3 * y**2
        + sp.Rational(1, 50) * b**4 * x**7 * y**2)

    c12, c02 = sp.symbols("c12 c02")
    th11 = lambda f: exp_action(lambda g: trunc(sp.Rational(1, 50) * b**2 * X(1, 1, g)), f)
    thX12 = lambda f: exp_action(lambda g: trunc(c12 * a * b**3 * X(1, 2, g)), f)
    th01 = lambda f: exp_action(lambda g: trunc(sp.Rational(1, 25) * a * b * X(0, 1, g)), f)
    thX02 = lambda f: exp_action(lambda g: trunc(c02 * a**2 * b**2 * X(0, 2, g)), f)

    G = thX02(th01(thX12(th11(Wmin))))
    diff = sp.expand(G - Wmax)
    eqs = [sp.simplify(cf) for cf in sp.Poly(diff, x, y, a, b).coeffs()]
    eqs = [e for e in eqs if e != 0]
    sol = sp.solve([sp.Eq(e, 0) for e in eqs], [c12, c02], dict=True)
    assert len(sol) == 1, (eqs, sol)
    c12v, c02v = sol[0][c12], sol[0][c02]
    resid = sp.expand(diff.subs({c12: c12v, c02: c02v}))
    assert resid == 0, resid
    print(f"[2] canonical factorization CONSISTENT (all slots, incl. the commutator")
    print(f"    [theta_(1,2), theta_(1,1)] feeding ray (2,3)):  c_(1,2) = {c12v},  c_(0,2) = {c02v}")
    print("    (the earlier apparent 'open inconsistency' was a bug in this test script --")
    print("     |Aut|=2 dressing missing on the a b^2 diagonal + omitted a^3 b, a^4 slots;")
    print("     the invisible-diagonal A-weights are VINDICATED)")

    # ---------------- [3] the ANCHOR: own-direction-chamber seeding ---------
    from fractions import Fraction

    def mk(c):
        return lambda f: exp_action(lambda g: trunc(c[0] * c[1] * X(*c[2], g)), f)

    wall_data = [  # bottom-up: (coefficient, t-monomial, X-degrees, ray slope)
        (sp.Rational(1, 50), b**2, (1, 1), Fraction(1)),
        (c12v, a * b**3, (1, 2), Fraction(3, 2)),
        (sp.Rational(1, 25), a * b, (0, 1), Fraction(2)),
        (c02v, a**2 * b**2, (0, 2), Fraction(3)),
    ]
    ups = [mk((c, u, k)) for c, u, k, _ in wall_data]
    downs = [mk((-c, u, k)) for c, u, k, _ in wall_data]
    slopes = [sl for _, _, _, sl in wall_data]

    Wc = [Wmin]
    for f in ups:
        Wc.append(trunc(f(Wc[-1])))
    assert sp.expand(Wc[4] - Wmax) == 0  # sanity: top chamber = max

    def transport(term, h, c):
        if c > h:
            for i in range(h, c):
                term = ups[i](term)
        else:
            for i in range(h - 1, c - 1, -1):
                term = downs[i](term)
        return trunc(term)

    # slot list: every (x,y,a,b)-monomial appearing in any chamber potential
    slots = set()
    for W in Wc:
        for mono in sp.Poly(W, x, y, a, b).monoms():
            slots.add(mono)

    def home_of(mono):
        k1, k2 = mono[0], mono[1]
        sl = None if k1 == 0 else Fraction(k2, k1)
        if k1 == 0:
            return 4
        return sum(1 for r in slopes if r < sl)

    T = [sp.Integer(0)] * 5
    n_seeds = 0
    for mono in slots:
        h = home_of(mono)
        coeff = Wc[h].coeff(x**mono[0] * y**mono[1] * a**mono[2] * b**mono[3])             if False else sp.Poly(Wc[h], x, y, a, b).coeff_monomial(mono)
        if coeff == 0:
            continue
        n_seeds += 1
        seed_term = coeff * x**mono[0] * y**mono[1] * a**mono[2] * b**mono[3]
        for c in range(5):
            T[c] += transport(seed_term, h, c)
    ok = all(sp.expand(T[c] - Wc[c]) == 0 for c in range(5))
    assert ok, [sp.expand(T[c] - Wc[c]) for c in range(5)]
    zero_seeds = len(slots) - n_seeds
    print(f"[3] ANCHOR HOLDS in the mixed sector through t^4: {n_seeds} nonzero seeds")
    print(f"    (one per slot, valued in its own-direction chamber; {zero_seeds} slots have")
    print(f"    ZERO home value = fully bending-generated, incl. all wall-diagonal extremes)")
    print("    T(c) = W(c) for ALL five chambers -- the broken-line/transport potential")
    print("    reproduces the entire chamber family from own-chamber seed data")
    print("ALL CHECKS PASSED")

if __name__ == "__main__":
    main()
