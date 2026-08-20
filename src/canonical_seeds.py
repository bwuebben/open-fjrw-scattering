#!/usr/bin/env python3
r"""
canonical_seeds.py -- (1) the census re-run with the diagonal-side on-ray convention;
(2) the first off-ray far-side case D = {(0,0,2),(2,1,0)}, solved completely; and
(3) THEOREM 1.5: the transport representation -- seeds are canonical chamber-independent
constants V := W|_D - F, and with seed := V the transport identity T = W holds by
divisibility induction.  The naive own-chamber seed rule computes V exactly on
foreign-free chambers (the classes of paper Thms 7.4/7.6) and fails otherwise -- as the off-ray
case shows, where the EXTREME slot (7,1) carries the nonzero canonical seed -10/3
hidden behind a normal-form zero (0 = seed + far-side bend 10/3).

Run: ./venv/bin/python src/canonical_seeds.py
"""

import os
import sys
from itertools import product as iproduct

import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a_invariants import part_boundaries  # noqa: E402
from backscatter import bend_choices, cross, submultisets  # noqa: E402

x, y, u, v = sp.symbols("x y u v")  # u = t_{0,0,2}, v = t_{2,1,0}


# ---------------------------------------------------------------- [1] corrected census
def candidates2(r, s, D):
    """Foreign own-chamber candidates under the DIAGONAL-SIDE on-ray convention."""
    out = []
    slots = [tuple(t) for t in part_boundaries(list(D), r, s)]
    for F, rem in submultisets(D):
        seeds = [(r, 0), (0, s)] if not F else \
            [tuple(t) for t in part_boundaries(list(F), r, s)]
        for vs in bend_choices(rem, r, s):
            tot = (sum(w[0] for w in vs), sum(w[1] for w in vs))
            for w in seeds:
                m = (w[0] + tot[0], w[1] + tot[1])
                if m not in slots:
                    continue
                good = True
                for vv in vs:
                    beta = (vv[0] + 1, vv[1] + 1)
                    cw, cm = cross(beta, w), cross(beta, m)
                    if cw == 0:                     # seed on ray: zero coefficient
                        good = False
                        break
                    if cm == 0:                     # m on this ray: diagonal side
                        if m[0] > m[1]:             # below diag -> chamber above ray
                            if cw >= 0:             # crossed only from below
                                good = False
                                break
                        elif m[0] < m[1]:           # above diag -> chamber below ray
                            if cw <= 0:
                                good = False
                                break
                        else:                       # on-diagonal: fully protected
                            good = False
                            break
                    elif cw * cm >= 0:              # ray not strictly between
                        good = False
                        break
                if good:
                    out.append((F, vs, w, m))
    return out


def census():
    T23, T33 = (2, 3, 0), (3, 3, 0)
    tot_sector = 0
    for i in range(5):
        for j in range(5):
            if 2 <= i + j <= 4:
                D = tuple(sorted([T23] * i + [T33] * j))
                tot_sector += len(candidates2(5, 5, D))
    assert tot_sector == 0
    found = []
    twists = [(a, b) for a in range(4) for b in range(4)]
    for (a1, b1), d1 in iproduct(twists, range(4)):
        for (a2, b2), d2 in iproduct(twists, range(4)):
            if (a1, b1, d1) > (a2, b2, d2):
                continue
            D = tuple(sorted([(a1, b1, d1), (a2, b2, d2)]))
            if not part_boundaries(list(D), 5, 5):
                continue
            found.extend((D,) + c for c in candidates2(5, 5, D))
    return tot_sector, found


# ------------------------------------------------- [2] the off-ray far-side case
def X(k1, k2, f):
    return sp.expand(x**k1 * y**k2 * ((k2 + 1) * x * sp.diff(f, x)
                                      - (k1 + 1) * y * sp.diff(f, y)))


def trunc(f):
    f = sp.expand(f)
    return sum(t for t in f.as_ordered_terms()
               if sp.degree(t, u) <= 1 and sp.degree(t, v) <= 1)


def exp_action(vfun, f):
    out, term, fact = sp.Integer(0), f, 1
    for n in range(3):
        out += term / fact
        term = trunc(vfun(term))
        fact *= (n + 1)
        if term == 0:
            break
    return trunc(sp.expand(out))


def wall(c, mono, k1, k2):
    return lambda f: exp_action(lambda g: trunc(c * mono * X(k1, k2, g)), f)


def offray_case():
    """D = {(0,0,2),(2,1,0)}: slots (2,6),(7,1); rays (6,1) [axis], (3,2) [own],
    (1,6) [axis]."""
    cD = sp.Symbol("cD")
    c_ax = sp.Rational(5, 6)
    W0 = (x**5 + y**5 + v * x**2 * y
          + sp.Rational(25, 6) * u * y**10
          + sp.Rational(35, 6) * u * v * x**2 * y**6)   # (2,6); (7,1) = 0 (min form)
    th = [wall(c_ax, u, 5, 0), wall(cD, u * v, 2, 1), wall(c_ax, u, 0, 5)]
    Wc = [sp.expand(W0)]
    for f in th:
        Wc.append(sp.expand(f(Wc[-1])))
    sol = sp.solve(sp.Eq(sp.expand(Wc[3]).coeff(u * v * x**2 * y**6), 0), cD)
    assert sol == [1], sol
    Wc = [sp.expand(W.subs(cD, 1)) for W in Wc]
    vec = [(W.coeff(u * v * x**2 * y**6), W.coeff(u * v * x**7 * y)) for W in Wc]
    expect = [(sp.Rational(35, 6), 0), (sp.Rational(35, 6), sp.Rational(-10, 3)),
              (sp.Rational(-55, 6), sp.Rational(20, 3)), (0, sp.Rational(20, 3))]
    assert vec == expect, vec
    # A(D) = 0 in all four chambers (weights: h1 (2/5, 3/5); h2 (14/25, 6/25, 24/25))
    for i, W in enumerate(Wc):
        ax = (W.coeff(u * y**10), W.coeff(u * x**5 * y**5), W.coeff(u * x**10))
        A = -sp.Rational(2, 5) * vec[i][0] - sp.Rational(3, 5) * vec[i][1] \
            + sp.Rational(14, 25) * ax[0] + sp.Rational(6, 25) * ax[1] \
            + sp.Rational(24, 25) * ax[2]
        assert sp.simplify(A) == 0, (i, A)

    # transports
    ths = [wall(c_ax, u, 5, 0), wall(1, u * v, 2, 1), wall(c_ax, u, 0, 5)]
    inv = [wall(-c_ax, u, 5, 0), wall(-1, u * v, 2, 1), wall(-c_ax, u, 0, 5)]

    def transport(term, h, c):
        if c > h:
            for i in range(h, c):
                term = ths[i](term)
        else:
            for i in range(h - 1, c - 1, -1):
                term = inv[i](term)
        return trunc(term)

    def T_of(seed71):
        seeds = [(x**5, 0), (y**5, 3), (v * x**2 * y, 1),
                 (-25 * u * x**5 * y**5, 2),
                 (sp.Rational(-55, 6) * u * v * x**2 * y**6, 2),
                 (seed71 * u * v * x**7 * y, 0)]
        return [sum(transport(s, h, c) for s, h in seeds) for c in range(4)]

    # naive own-chamber rule: seed(7,1) = W(c0)-value = 0  ->  FAILS
    T_naive = T_of(sp.Integer(0))
    defects = [sp.expand(Wc[c] - T_naive[c]).coeff(u * v * x**7 * y) for c in range(4)]
    assert defects == [sp.Rational(-10, 3)] * 4, defects
    # canonical seed V = W - F: constant -10/3 (F = (10/3, 0, 10, 10) per chamber)
    T_can = T_of(sp.Rational(-10, 3))
    for c in range(4):
        assert sp.expand(Wc[c] - T_can[c]) == 0, c
    return vec


def main():
    print("== Census re-run ((1,1)-side rule) + the off-ray case + the canonical seed (Thm 1.5) ==")
    tot, found = census()
    print(f"[1] corrected census: computed sectors: {tot} candidates (unchanged);")
    print(f"    probe (l = 2, twists <= (3,3), d <= 3): {len(found)} candidates")
    print("    first entries:")
    for D, F, vs, w, m in found[:5]:
        print(f"      D = {list(D)}: seed {w} (on {list(F) if F else 'W0'}) "
              f"--{list(vs)}--> {m}")
    vec = offray_case()
    print("[2] off-ray case D = {(0,0,2),(2,1,0)} SOLVED: own wall forced c_D = 1;")
    print(f"    chamber vectors {vec}; A(D) = 0 in ALL FOUR chambers;")
    print("    naive own-chamber seeding FAILS (constant defect -10/3 at slot (7,1),")
    print("    an EXTREME slot: its normal-form zero = seed + far-side bend);")
    print("    canonical seed V = W - F = -10/3 (chamber-independent) -> T = W EXACTLY")
    print("[3] THEOREM 1.5: V := W|_D - F is chamber-independent (same-jump argument),")
    print("    so seed := V makes the transport identity T = W hold on every diagonal")
    print("    by divisibility induction -- the residual lemma DISSOLVES.")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
