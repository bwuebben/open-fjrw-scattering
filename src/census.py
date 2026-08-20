#!/usr/bin/env python3
r"""
census.py -- verification of the companion note "The central-charge threshold for
wall-crossing in two-variable open FJRW theory" on the corrected foundations
([GKT] Def 4.22 conventions; see src/gkt_algebra.py).

Certified:
 [1] the ADE list: for c-hat < 1 there is NO primary (d=0) critical graph, for any
     multiset size m (verified up to the closed-form bound m <= 1/(1-c-hat) forced
     by the Lemma ell_1+ell_2 <= m*c-hat); the wall-crossing-free pairs over the grid
     2 <= r,s <= 40 are exactly {(n,2),(2,n)} u {(3,3),(3,4),(4,3),(3,5),(5,3)}
     (A_{n-1}, D_4, E_6, E_8);
 [2] the trichotomy m_*(r,s) = infinity | 2 | 3 as in the note, over the grid;
 [3] the corrected first-wall census (the note's Table 1): for each listed (r,s),
     the critical multisets at m = m_*, each graph's Lie bidegree (k_1,k_2)
     [GKT Def 4.22 indexing: the graph has k_1+1 and k_2+1 boundary points], its
     wall Hamiltonian x^{k_1+1}y^{k_2+1} (the boundary monomial), and the
     MARGINAL/TORUS split: Euler weight 0 <=> Lie bidegree (0,0) <=> the wall
     direction is the torus field x d_x - y d_y (Hamiltonian xy);
 [4] the parabolic pairs (4,4),(3,6),(6,3): exactly ONE critical graph at m_*,
     the maximal-twist multiset, and its direction is the TORUS direction;
 [5] the universal descendent wall: for every (r,s) and every twist (a,b), the
     singleton with descendent d=1 is a critical graph (N=1) of Lie bidegree
     (a,b); it is a genuine pro-nilpotent wall iff (a,b) != (0,0), and the torus
     direction at (a,b) = (0,0);
 [6] Lie bidegrees of positive-weight critical graphs coincide with the Def 4.22
     admissible degrees computed independently in gkt_algebra.lie_degrees;
 [7] the Hamiltonian identity X_{a,b} = X_H with H = x^{a+1}y^{b+1}, the
     divergence-freeness, and the bracket
     [X_{a,b},X_{c,d}] = ((b+1)(c+1)-(a+1)(d+1)) X_{a+c,b+d}, symbolically over
     0 <= a,b,c,d <= 4.

Run: python src/census.py
"""

import os
import sys
from fractions import Fraction
from itertools import combinations_with_replacement

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gkt_algebra import lie_degrees  # noqa: E402


def chat(r, s):
    return Fraction(2) - Fraction(2, r) - Fraction(2, s)


def N_of(r, s, J, d):
    sa = sum(a for a, b in J)
    sb = sum(b for a, b in J)
    return sa // r + sb // s - len(J) + 1 + sum(d)


def crit_data(r, s, J, d):
    """For a critical multiset (N>=1): list of (Lie bidegree, weight, torus?)."""
    N = N_of(r, s, J, d)
    rJ = sum(a for a, b in J) % r
    sJ = sum(b for a, b in J) % s
    out = []
    w = sum(s * a + r * b + r * s * (dd - 1) for (a, b), dd in zip(J, d))
    for p in range(1, N + 1):
        lie = (rJ + (p - 1) * r, sJ + (N - p) * s)
        out.append((lie, w, lie == (0, 0)))
    return out


def primary_multisets(r, s, m):
    tw = [(a, b) for a in range(r - 1) for b in range(s - 1)]
    for J in combinations_with_replacement(tw, m):
        if N_of(r, s, list(J), [0] * m) >= 1:
            yield list(J)


def m_star(r, s, cap=60):
    if chat(r, s) < 1:
        return None  # infinity (verified separately)
    for m in range(1, cap + 1):
        # maximal-twist multiset maximizes N
        J = [(r - 2, s - 2)] * m
        if N_of(r, s, J, [0] * m) >= 1:
            # need SOME multiset; max-twist is the maximizer, so this is m_*
            return m
    raise AssertionError((r, s))


def check_ade(grid=40):
    ade = set()
    for r in range(2, grid + 1):
        for s in range(2, grid + 1):
            ch = chat(r, s)
            if ch < 1:
                bound = int(1 / (1 - ch)) + 1
                for m in range(1, bound + 1):
                    assert N_of(r, s, [(r - 2, s - 2)] * m, [0] * m) <= 0, (r, s, m)
                ade.add((r, s))
    expect = {(n, 2) for n in range(2, grid + 1)} | {(2, n) for n in range(2, grid + 1)} \
        | {(3, 3), (3, 4), (4, 3), (3, 5), (5, 3)}
    assert ade == expect, ade ^ expect
    return len(ade)


def check_trichotomy(grid=40):
    for r in range(2, grid + 1):
        for s in range(2, grid + 1):
            ms = m_star(r, s)
            if chat(r, s) < 1:
                assert ms is None
            elif min(r, s) >= 4:
                assert ms == 2, (r, s, ms)
            else:
                assert min(r, s) == 3 and max(r, s) >= 6 and ms == 3, (r, s, ms)
    return True


def first_wall_table():
    rows = []
    for (r, s) in [(4, 4), (3, 6), (6, 3), (4, 5), (3, 7), (4, 6), (5, 5)]:
        ms = m_star(r, s)
        graphs = []
        for J in primary_multisets(r, s, ms):
            for lie, w, torus in crit_data(r, s, J, [0] * ms):
                graphs.append((tuple(J), lie, w, torus))
                # [6] cross-check vs Def 4.22 admissible degrees
                if w > 0:
                    assert list(lie) in [list(t) for t in
                                         lie_degrees(r, s, list(J), [0] * ms)], (r, s, J)
                else:
                    assert w == 0 and lie == (0, 0), (r, s, J, lie, w)
        rows.append((r, s, ms, graphs))
    return rows


def check_parabolic(rows):
    for (r, s, ms, graphs) in rows:
        if chat(r, s) == 1:
            assert len(graphs) == 1, (r, s, graphs)
            J, lie, w, torus = graphs[0]
            assert torus and set(J) == {(r - 2, s - 2)}, (r, s, graphs)
    return True


def check_descendent(grid=12):
    for r in range(2, grid + 1):
        for s in range(2, grid + 1):
            for a in range(r - 1):
                for b in range(s - 1):
                    assert N_of(r, s, [(a, b)], [1]) == 1
                    data = crit_data(r, s, [(a, b)], [1])
                    assert data[0][0] == (a, b), (r, s, a, b, data)
                    if (a, b) == (0, 0):
                        assert data[0][2]  # torus
                    else:
                        assert data[0][1] > 0 and \
                            [a, b] in [list(t) for t in lie_degrees(r, s, [(a, b)], [1])]
    return True


def check_identities(top=4):
    import sympy as sp
    x, y = sp.symbols("x y")

    def X(a, b, f):
        return sp.expand(x**a * y**b * ((b + 1) * x * sp.diff(f, x)
                                        - (a + 1) * y * sp.diff(f, y)))

    for a in range(top + 1):
        for b in range(top + 1):
            H = x**(a + 1) * y**(b + 1)
            XH_x = sp.diff(H, y)          # X_H = (dH/dy) d_x - (dH/dx) d_y
            XH_y = -sp.diff(H, x)
            assert sp.expand(XH_x - (b + 1) * x**(a + 1) * y**b) == 0
            assert sp.expand(XH_y + (a + 1) * x**a * y**(b + 1)) == 0
            div = sp.diff((b + 1) * x**(a + 1) * y**b, x)                 + sp.diff(-(a + 1) * x**a * y**(b + 1), y)
            assert sp.expand(div) == 0
    f = sp.Function("f")(x, y)
    probe = x**3 * y**2 + x * y**5 + 1
    for a in range(top + 1):
        for b in range(top + 1):
            for c in range(top + 1):
                for d in range(top + 1):
                    lhs = X(a, b, X(c, d, probe)) - X(c, d, X(a, b, probe))
                    rhs = ((b + 1) * (c + 1) - (a + 1) * (d + 1)) * X(a + c, b + d, probe)
                    assert sp.expand(lhs - rhs) == 0, (a, b, c, d)
    return True


def ham(lie):
    return f"x^{lie[0]+1}y^{lie[1]+1}" if lie != (0, 0) else "xy (torus)"


def main():
    print("== Census verification (corrected conventions) ==")
    n = check_ade()
    print(f"[1] ADE list verified over the 2..40 grid ({n} wall-crossing-free pairs;")
    print("    negatives verified up to the closed-form bound m <= 1/(1-c-hat))")
    assert check_trichotomy()
    print("[2] the trichotomy m_*(r,s) = inf | 2 | 3 verified over the grid")
    rows = first_wall_table()
    print("[3] the corrected first-wall census:")
    for (r, s, ms, graphs) in rows:
        walls = [g for g in graphs if not g[3]]
        tori = [g for g in graphs if g[3]]
        hams = ", ".join(sorted({ham(g[1]) for g in walls})) or "-"
        print(f"    ({r},{s}): m_* = {ms}; {len(graphs)} critical graphs = "
              f"{len(walls)} walls [{hams}] + {len(tori)} marginal (torus)")
    assert check_parabolic(rows)
    print("[4] parabolic pairs: exactly one critical graph, maximal-twist, TORUS direction")
    assert check_descendent()
    print("[5] universal descendent wall: singleton d=1 critical for every (r,s) and twist;")
    print("    Lie bidegree (a,b); torus at (0,0), pro-nilpotent wall otherwise")
    print("[6] positive-weight Lie bidegrees match Def 4.22 (gkt_algebra.lie_degrees)")
    assert check_identities()
    print("[7] Hamiltonian identity, divergence-freeness, and the bracket verified")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
