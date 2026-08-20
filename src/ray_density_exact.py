#!/usr/bin/env python3
"""
The accumulation of Prop. 7.19, made exact, and where it bites.

CLAIM A (structure of a singleton descendent tower). For J = {(a,b)} and descendent d,
  N = d  and the critical graphs have boundary degrees (rays)
      K_p = ( a + (p-1)r + 1 ,  b + (d-p)s + 1 ),   p = 1..d.
  Setting j = p-1 >= 0, k = d-p >= 0 (so j+k = d-1), the ray set over ALL d >= 1 is
      R(a,b) = { ( a + jr + 1 , b + ks + 1 ) : j,k >= 0 }.
  Every one of these sits at total t-degree ONE (the single variable t_{a,b,d}).

CLAIM B (density). The slopes of R(a,b) are dense in (0, infinity), for every (r,s)
  with r,s >= 2 -- including the simple (ADE) models.

CLAIM C (the primary sector is different). For d = 0, each t-degree m carries only
  finitely many rays, so the PRIMARY diagram is locally finite order-by-order.
"""
from itertools import combinations_with_replacement
from math import gcd


def crit_rays(r, s, J, d):
    sa, sb = sum(a for a, _ in J), sum(b for _, b in J)
    rJ, l1 = sa % r, sa // r
    sJ, l2 = sb % s, sb // s
    N = l1 + l2 - len(J) + 1 + sum(d)
    return [(rJ + (p - 1) * r + 1, sJ + (N - p) * s + 1) for p in range(1, N + 1)]


def check_claim_A(r, s, a, b, dmax=40):
    direct, formula = set(), set()
    for d in range(1, dmax + 1):
        direct.update(crit_rays(r, s, [(a, b)], [d]))
        for j in range(d):
            formula.add((a + j * r + 1, b + (d - 1 - j) * s + 1))
    return direct == formula


def density_witness(r, s, a, b, targets, jmax=20000):
    """For each target slope t, best |slope - t| over the tower's rays with j <= jmax."""
    out = []
    for t in targets:
        best = float("inf")
        for j in range(jmax):
            K1 = a + j * r + 1
            # want (b + k s + 1)/K1 ~ t  ->  k ~ (t*K1 - 1 - b)/s
            kstar = (t * K1 - 1 - b) / s
            for k in {max(0, int(kstar)), max(0, int(kstar) + 1)}:
                best = min(best, abs((b + k * s + 1) / K1 - t))
        out.append((t, best))
    return out


def primary_rays_by_degree(r, s, mmax):
    twists = [(a, b) for a in range(r - 1) for b in range(s - 1)]
    per_degree = {}
    for m in range(1, mmax + 1):
        rays = set()
        for J in combinations_with_replacement(twists, m):
            rays.update(crit_rays(r, s, list(J), [0] * m))
        per_degree[m] = rays
    return per_degree


if __name__ == "__main__":
    print("=== CLAIM A: tower ray set = { (a+jr+1, b+ks+1) } ===")
    for (r, s, a, b) in [(5, 5, 0, 0), (5, 5, 3, 3), (4, 4, 2, 2), (3, 2, 1, 0), (2, 2, 0, 0)]:
        ok = check_claim_A(r, s, a, b)
        print(f"  r={r} s={s} twist=({a},{b}):  {'OK' if ok else 'MISMATCH'}")

    print("\n=== CLAIM B: slopes dense in (0, inf) -- incl. the ADE models ===")
    targets = [0.1, 0.5, 1.0, 1.6180339887, 3.14159265, 7.0]
    for (r, s, a, b, name) in [(5, 5, 0, 0, "x^5+y^5  hyperbolic"),
                               (4, 4, 2, 2, "x^4+y^4  PARABOLIC"),
                               (3, 3, 1, 1, "x^3+y^3  D_4 (SIMPLE)"),
                               (3, 2, 1, 0, "x^3+y^2  A_2 (SIMPLE)")]:
        w = density_witness(r, s, a, b, targets)
        worst = max(e for _, e in w)
        print(f"  {name:22s}  worst approx error over 6 targets (j<=20000): {worst:.3e}")

    print("\n=== CLAIM C: primary (d=0) rays per t-degree ===")
    for (r, s) in [(4, 4), (5, 5), (4, 5), (3, 7)]:
        per = primary_rays_by_degree(r, s, 7)
        counts = {m: len(v) for m, v in per.items()}
        cum = set()
        cumcounts = []
        for m in sorted(per):
            cum |= per[m]
            cumcounts.append(len(cum))
        print(f"  r={r},s={s}:  per-degree {counts}   cumulative {cumcounts}")

    print("\n=== THE PILOT, isolated ===")
    per = primary_rays_by_degree(4, 4, 10)
    allr = set().union(*per.values())
    print(f"  x^4+y^4 primary rays, |J| <= 10:  {sorted(allr)}")
    print("  -> a single ray; the sector has exactly TWO chambers.")
