#!/usr/bin/env python3
"""
Where does a PRIMARY diagonal first carry N >= 2 walls?

Why it matters. In the primary quotient a marking subset A determines a single
diagonal (A, 0). The jumps of a canonical homotopy of [GKT, Lem. 3.39] are indexed
by marking subsets, so the jump at t_A is supported on that one diagonal. If that
diagonal carries at most ONE wall, the jump is automatically supported on a single
ray. Thus one unstaged Lemma-3.39 normal-form family suffices below this degree:

    every primary diagonal has N <= 1   ==>   no ray staging is needed.

The full finite-order primary realization is now proved by separate ray stages.
This script verifies N <= 1 everywhere <=> chat_W <= 1 (simple and parabolic),
the exact maximal-twist formula, and the complete piecewise classification of
the first |J| at which N >= 2 in the hyperbolic regime.
"""
from fractions import Fraction
from itertools import combinations_with_replacement


def chat(r, s):
    return Fraction(2) - Fraction(2, r) - Fraction(2, s)


def ceil_div(a, b):
    return -(-a // b)


def maxN_at(r, s, m):
    """Max N over primary multisets of size m (N is maximal at maximal twist)."""
    return m + 1 - ceil_div(2 * m, r) - ceil_div(2 * m, s)


def first_N2(r, s, mmax=400):
    for m in range(3, mmax + 1):
        if ceil_div(2 * m, r) + ceil_div(2 * m, s) <= m - 1:
            return m
    return None


def classified_N2(r, s):
    """Exact piecewise N_2, with None denoting the simple/parabolic cases."""
    r, s = sorted((r, s))
    if chat(r, s) <= 1:
        return None
    if r == 3:
        if s == 7:
            return 21
        if s == 8:
            return 12
        if s <= 11:
            return 9
        return 6
    if r == 4:
        if s == 5:
            return 10
        if s <= 7:
            return 6
        return 4
    if r == 5:
        return 5 if s <= 7 else 4
    return 3


def brute_check(r, s, mmax):
    """Confirm the closed form against full enumeration of multisets."""
    twists = [(a, b) for a in range(r - 1) for b in range(s - 1)]
    for m in range(1, mmax + 1):
        best = -10**9
        for J in combinations_with_replacement(twists, m):
            sa = sum(a for a, _ in J)
            sb = sum(b for _, b in J)
            best = max(best, sa // r + sb // s - m + 1)
        assert best == maxN_at(r, s, m), (r, s, m, best, maxN_at(r, s, m))
    return True


if __name__ == "__main__":
    print("=== closed form vs brute force (primary) ===")
    for (r, s) in [(4, 4), (5, 5), (4, 5), (3, 7), (3, 3), (5, 2)]:
        brute_check(r, s, 6)
        print(f"  r={r},s={s}: closed form verified to |J|=6")

    print("\n=== is  N <= 1 everywhere  <=>  chat_W <= 1 ? ===")
    bad = []
    for r in range(2, 41):
        for s in range(2, 41):
            n2 = first_N2(r, s)
            le1 = (n2 is None)
            if le1 != (chat(r, s) <= 1):
                bad.append((r, s, chat(r, s), n2))
    print(f"  counterexamples over 2<=r,s<=40: {bad if bad else 'NONE'}")

    print("\n=== exact piecewise classification of N_2 ===")
    bad = []
    for r in range(2, 101):
        for s in range(2, 101):
            scanned = first_N2(r, s)
            classified = classified_N2(r, s)
            if scanned != classified:
                bad.append((r, s, scanned, classified))
    print(f"  counterexamples over 2<=r,s<=100: {bad if bad else 'NONE'}")

    print("\n=== hyperbolic: first |J| carrying a primary diagonal with N>=2 ===")
    rows = []
    for (r, s) in [(5, 5), (4, 5), (4, 6), (3, 7), (3, 8), (4, 7), (5, 6),
                   (6, 6), (3, 12), (10, 10), (40, 40)]:
        c = chat(r, s)
        rows.append((r, s, c, first_N2(r, s)))
    print(f"  {'(r,s)':>10} {'chat_W':>10} {'1/(chat-1)':>12} {'first |J| with N>=2':>22}")
    for r, s, c, m in rows:
        bound = float(1 / (c - 1)) if c > 1 else float("inf")
        print(f"  {f'({r},{s})':>10} {str(c):>10} {bound:>12.3f} {str(m):>22}")

    print("\n=== the boundary cases, explicitly ===")
    for (r, s) in [(4, 4), (3, 6), (6, 3), (3, 3), (3, 5), (7, 2)]:
        c = chat(r, s)
        maxes = [maxN_at(r, s, m) for m in range(1, 13)]
        kind = "simple" if c < 1 else ("parabolic" if c == 1 else "hyperbolic")
        print(f"  ({r},{s}) {kind:>10}  chat={c}  max N for |J|=1..12: {maxes}")
