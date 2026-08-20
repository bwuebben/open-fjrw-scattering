#!/usr/bin/env python3
"""
Do the rays of the scattering diagram D^{r,s} accumulate?  (paper, Prop. 7.19)

Ray of a critical graph Lambda_{J,p} = its BOUNDARY degree (= Lie bidegree + (1,1)):
    K = ( r(J) + (p-1)r + 1 ,  s(J) + (N-p)s + 1 ),   1 <= p <= N
with N = l1 + l2 - |J| + 1 + sum d_j,  sum a = r(J) + l1 r,  sum b = s(J) + l2 s.

Reports, separately for the primary (d=0) sector and for descendent towers:
  * number of distinct primitive ray directions,
  * the largest angular gap between consecutive rays in the open sector,
  * how both behave as the enumeration deepens.

The distinction needed by the geometric picture is whether the ray set is
LOCALLY FINITE, so that chambers are open regions, or ACCUMULATING. The primary
diagram is locally finite at each deformation order; dense descendent transport
is therefore formulated coefficientwise by one-sided slope cuts.
"""
from fractions import Fraction
from itertools import combinations_with_replacement
from math import atan2, gcd, pi


def crit_rays(r, s, J, d):
    """Boundary degrees (= ray directions) of the critical graphs over (J, d)."""
    sa, sb = sum(a for a, _ in J), sum(b for _, b in J)
    rJ, l1 = sa % r, sa // r
    sJ, l2 = sb % s, sb // s
    N = l1 + l2 - len(J) + 1 + sum(d)
    return [(rJ + (p - 1) * r + 1, sJ + (N - p) * s + 1) for p in range(1, N + 1)]


def primitive(v):
    g = gcd(v[0], v[1])
    return (v[0] // g, v[1] // g)


def max_gap(rays):
    """Largest angular gap (radians) between consecutive rays inside the sector."""
    angs = sorted({atan2(k2, k1) for k1, k2 in rays})
    if len(angs) < 2:
        return pi / 2
    # include the two sector boundaries (the axes) as the outer limits
    bounds = [0.0] + angs + [pi / 2]
    return max(b - a for a, b in zip(bounds, bounds[1:]))


def primary_sector(r, s, mmax):
    """d = 0. Enumerate multisets of NS twists up to size mmax."""
    twists = [(a, b) for a in range(r - 1) for b in range(s - 1)]
    print(f"\n=== PRIMARY (d=0), r={r}, s={s} ===")
    allrays = set()
    for m in range(1, mmax + 1):
        for J in combinations_with_replacement(twists, m):
            allrays.update(primitive(v) for v in crit_rays(r, s, list(J), [0] * m))
        print(f"  |J| <= {m:2d}:  {len(allrays):5d} distinct rays,"
              f"  max angular gap = {max_gap(allrays):.4f} rad"
              if allrays else f"  |J| <= {m:2d}:  no walls")
    return allrays


def descendent_tower(r, s, a, b, dmax):
    """Singleton J = {(a,b)} with descendent d: t-DEGREE 1, but N = d walls."""
    print(f"\n=== SINGLETON DESCENDENT TOWER t_{{{a},{b},d}}, r={r}, s={s} ===")
    print("     (every one of these sits at total t-degree 1)")
    allrays = set()
    for d in range(1, dmax + 1):
        allrays.update(primitive(v) for v in crit_rays(r, s, [(a, b)], [d]))
        if d in (1, 2, 3, 5, 10, 20, dmax):
            print(f"  d <= {d:3d}:  {len(allrays):5d} distinct rays,"
                  f"  max angular gap = {max_gap(allrays):.4f} rad")
    return allrays


if __name__ == "__main__":
    r = s = 5
    prim = primary_sector(r, s, 6)
    tow = descendent_tower(r, s, 0, 0, 60)

    print("\n=== VERDICT INPUT ===")
    print(f"primary rays (|J|<=6):            {len(prim)}")
    print(f"one descendent tower (d<=60):     {len(tow)}   at t-degree 1")
    both = prim | tow
    print(f"union max angular gap:            {max_gap(both):.5f} rad")

    # x^4+y^4, the parabolic pilot
    print("\n\n########## PILOT: x^4+y^4 (parabolic) ##########")
    p4 = primary_sector(4, 4, 8)
    print(f"\n  primary rays for x^4+y^4, |J|<=8: {sorted(p4)}")
    t4 = descendent_tower(4, 4, 2, 2, 20)
    print(f"  descendent rays (t_{{2,2,d}}, d<=20): {len(t4)}")
