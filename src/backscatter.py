#!/usr/bin/env python3
r"""
backscatter.py -- the no-backwards-bending theorem (paper Thm 7.5): the primitive-form
shift is the monotone-bending mechanism; plus the exhaustive far-side census.

The no-backwards-bending theorem (paper Thm 7.5): a foreign transported seed
contributes to slot m in m's OWN chamber only along paths whose bend rays lie strictly
between dir(seed) and dir(m).  The exponent identity m + B(1,1) = w_sigma + sum(beta_f)
(beta_f = boundary directions on the crossed rays, B = number of bend factors >= 1),
crossed with m, forces
    B * (m1 - m2)  =  cross(m, w_sigma) + sum_f cross(m, beta_f),
whose right side has a definite sign for monotone paths: from below (all directions
shallower than m) it is negative, so m1 < m2 is forced; from above, m1 > m2.  Hence:
  * slots with m1 >= m2 receive NOTHING from below;
  * slots with m1 <= m2 receive NOTHING from above;
  * slots ON the diagonal direction (m1 = m2) receive NOTHING AT ALL -- the
    own-chamber cancellation lemma holds unconditionally for them.
Every survivor arrives from the (1,1)-side of the target's own ray.  The seed need NOT
lie beyond the (1,1)-direction: same-side configurations (seed and target on one side of
the diagonal) occur, and are the far-side bends of paper Ex 7.18.

This script:
 [1] verifies the sign identity on random configurations (the algebraic content of the
     theorem);
 [2] exhaustively enumerates candidate foreign own-chamber contributions in the two
     computed sectors of x^5+y^5 (all diagonals dividing t23^i t33^j, i+j <= 4):
     result: NONE (so the verified anchors are fully explained by Thms 7.3 + 7.5 + 7.7);
 [3] probes for the smallest far-side candidates over composite diagonals with
     descendents (l <= 2, d <= 3, r = s = 5) and checks each against the theorem:
     every candidate arrives from the (1,1)-side of its target's ray.  These are the
     far-side bends absorbed by the canonical seeds (paper Thm 1.5; canonical_seeds.py).

Run: ./venv/bin/python src/backscatter.py
"""

import os
import random
import sys
from itertools import combinations, product as iproduct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a_invariants import part_boundaries  # noqa: E402
from gkt_algebra import lie_degrees  # noqa: E402
from sympy.utilities.iterables import multiset_partitions  # noqa: E402


def cross(u, v):
    return u[0] * v[1] - u[1] * v[0]


# ---------------------------------------------------------------- [1] the sign identity
def check_sign_identity(trials=20000, seed=11):
    """m + B(1,1) = w + sum(beta) => B(m1-m2) = cross(m,w) + sum cross(m,beta):
    verify the identity and the same-side sign forcing on random data."""
    rng = random.Random(seed)
    for _ in range(trials):
        B = rng.randint(1, 5)
        betas = [(rng.randint(0, 9), rng.randint(0, 9)) for _ in range(B)]
        w = (rng.randint(0, 12), rng.randint(0, 12))
        m = (w[0] + sum(b[0] for b in betas) - B, w[1] + sum(b[1] for b in betas) - B)
        lhs = B * (m[0] - m[1])
        rhs = cross(m, w) + sum(cross(m, b) for b in betas)
        assert lhs == rhs, (m, w, betas)
        # same-side forcing: if all of w, beta strictly shallower than m => rhs < 0
        if all(cross(m, b) < 0 for b in betas) and cross(m, w) < 0:
            assert m[0] < m[1]
        if all(cross(m, b) > 0 for b in betas) and cross(m, w) > 0:
            assert m[0] > m[1]
    return trials


# ------------------------------------------------- candidate enumeration machinery
def submultisets(D):
    seen = set()
    n = len(D)
    for k in range(n):  # F proper (rem nonempty); F may be empty
        for idx in combinations(range(n), k):
            F = tuple(sorted(D[i] for i in idx))
            rem = tuple(sorted(D[i] for i in range(n) if i not in idx))
            if (F, rem) in seen:
                continue
            seen.add((F, rem))
            yield F, rem


def bend_choices(rem, r, s):
    """All ways to split rem into wall factors with a Lie-degree choice per factor."""
    for parts in multiset_partitions(list(rem)):
        opts = []
        ok = True
        for part in parts:
            J = [(a, b) for a, b, d in part]
            dd = [d for a, b, d in part]
            degs = lie_degrees(r, s, J, dd)
            if not degs:
                ok = False
                break
            opts.append([tuple(v) for v in degs])
        if not ok:
            continue
        yield from iproduct(*opts)


def candidates(r, s, D):
    """Foreign own-chamber contribution candidates for diagonal D (list of triples).

    Necessary conditions imposed: exponent identity; every bend ray strictly between
    dir(seed) and dir(m) (with the on-ray conventions: seed on a bend ray => zero
    coefficient => skip; m on a bend ray => counted for from-above only)."""
    out = []
    slots = [tuple(t) for t in part_boundaries(list(D), r, s)]
    for F, rem in submultisets(D):
        seeds = [(r, 0), (0, s)] if not F else \
            [tuple(t) for t in part_boundaries(list(F), r, s)]
        for vs in bend_choices(rem, r, s):
            tot = (sum(v[0] for v in vs), sum(v[1] for v in vs))
            for w in seeds:
                m = (w[0] + tot[0], w[1] + tot[1])
                if m not in slots:
                    continue
                good = True
                for v in vs:
                    beta = (v[0] + 1, v[1] + 1)
                    cw, cm = cross(beta, w), cross(beta, m)
                    if cw == 0:          # seed on the bend ray: zero coefficient
                        good = False
                        break
                    if cw > 0:           # seed above the ray: from-above crossing
                        if cm > 0:       # m also above: ray not between
                            good = False
                            break
                    else:                # seed below: from-below crossing
                        if cm <= 0:      # m below-or-on (strict-below convention)
                            good = False
                            break
                if good:
                    out.append((F, vs, w, m))
    return out


def main():
    print("== The no-backwards-bending theorem (Thm 7.5) and the far-side census ==")
    n = check_sign_identity()
    print(f"[1] sign identity B(m1-m2) = cross(m,w)+sum cross(m,beta): {n} random checks OK;")
    print("    sign forcing confirmed: from below only into slots above the diagonal,")
    print("    from above only into slots below; diagonal slots (m1 = m2) fully protected")

    # [2] the two computed sectors
    T23, T33 = (2, 3, 0), (3, 3, 0)
    total = 0
    for i in range(5):
        for j in range(5):
            if not (2 <= i + j <= 4):
                continue
            D = tuple(sorted([T23] * i + [T33] * j))
            cs = candidates(5, 5, D)
            total += len(cs)
            assert not cs, (D, cs)
    print("[2] computed sectors (all diagonals dividing t23^i t33^j, i+j <= 4):")
    print(f"    foreign own-chamber candidates: {total} -- NONE.  The verified anchors")
    print("    are fully explained by Thms 7.3 + 7.5 + 7.7.")

    # [3] the probe: composite diagonals with descendents
    found = []
    twists = [(a, b) for a in range(4) for b in range(4)]
    for (a1, b1), d1 in iproduct(twists, range(4)):
        for (a2, b2), d2 in iproduct(twists, range(4)):
            if (a1, b1, d1) > (a2, b2, d2):
                continue
            D = tuple(sorted([(a1, b1, d1), (a2, b2, d2)]))
            if not part_boundaries(list(D), 5, 5):
                continue
            cs = candidates(5, 5, D)
            for F, vs, w, m in cs:
                # the theorem's check: arrival from the (1,1)-side of the
                # target's own ray -- seed strictly, bend rays weakly (equality
                # exactly for an on-ray target, counted one-sidedly)
                sgn = m[0] - m[1]
                assert sgn != 0, (D, F, vs, w, m)
                assert cross(m, w) * sgn > 0, (D, F, vs, w, m)
                for v in vs:
                    beta = (v[0] + 1, v[1] + 1)
                    assert cross(m, beta) * sgn >= 0, (D, F, vs, w, m)
                found.append((D, F, vs, w, m))
    print(f"[3] probe (l = 2, twists <= (3,3), d <= 3, r = s = 5): "
          f"{len(found)} far-side candidates, every one arriving from the (1,1)-side")
    for D, F, vs, w, m in found[:8]:
        print(f"      D = {list(D)}: seed {w} (on {list(F) if F else 'W0'}) "
              f"--bends {list(vs)}--> slot {m}")
    if found:
        print("    these are the far-side bends of Ex 7.18; the canonical seeds of")
        print("    Thm 1.5 absorb them (canonical_seeds.py)")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
