#!/usr/bin/env python3
r"""
Are BOTH walls of the first two-ray primary diagonal actually crossed?

Prop 7.23 is conditional: a family of simple canonical homotopies cannot give
ray-supported jumps "once two of a diagonal's walls are actually crossed", since
all walls of a diagonal share the one time t_J. This script discharges that
hypothesis in the smallest case, turning the conditional into a concrete
impossibility with a witness.

Setting: r = s = 5, J = {(3,3) x 5}, d = 0 -- the diagonal of Ex 7.24, N = 2,
balanced profiles (0,10),(5,5),(10,0), walls of boundary degree (1,6),(6,1) on the
two distinct rays (1,6),(6,1).

Two things are computed.

(1) THE TRANSITION FORCES BOTH COEFFICIENTS. Solve
        nu^max - nu^min = c_1 w_1 + c_2 w_2,   w_p = a_p e_p - b_p e_{p-1},
    with a_p = r k_2(Lambda_p), b_p = s k_1(Lambda_p). If BOTH c_p are nonzero, the
    u_J-component of the transition needs both walls, so both zero counts Delta_p
    are nonzero -- and they fire at the same time t_J.

(2) NO LOWER-ORDER TERM CAN ABSORB EITHER. The u_J-component of log(g) also
    receives Baker-Campbell-Hausdorff contributions from products of wall elements
    of PROPER sub-multisets, and higher terms of the exponentials. If any existed,
    (1) would not by itself force the Delta's. Enumerate every way a product of
    wall-carrying diagonals of the pure t_{3,3} tower can land on t_{3,3}^5; brackets
    need DISJOINT marking sets (GKT Prop 4.24), so the multiplicities must add.

Run: ./venv/bin/python src/two_ray_forced.py
"""
from fractions import Fraction
from itertools import combinations_with_replacement

R = S = 5
T = (3, 3)


def diagonal(m):
    """(N, balanced profiles, critical boundary degrees) for J = {(3,3)^m}."""
    sa = sb = 3 * m
    rJ, l1 = sa % R, sa // R
    sJ, l2 = sb % S, sb // S
    N = l1 + l2 - m + 1
    bal = [(rJ + p * R, sJ + (N - p) * S) for p in range(N + 1)]
    crit = [(rJ + (p - 1) * R + 1, sJ + (N - p) * S + 1) for p in range(1, N + 1)]
    return N, bal, crit


def main():
    print("=" * 70)
    print("[1] does the transition force BOTH wall coefficients?")
    print("=" * 70)
    m = 5
    N, bal, crit = diagonal(m)
    assert N == 2 and bal == [(0, 10), (5, 5), (10, 0)] and crit == [(1, 6), (6, 1)]

    # nu^min keeps only e_0, nu^max only e_N (Thm 1.3); the value is Ex 7.24's.
    v = Fraction(34, 625)
    nu_min = [v, Fraction(0), Fraction(0)]
    nu_max = [Fraction(0), Fraction(0), v]
    target = [b - a for a, b in zip(nu_min, nu_max)]
    print(f"  nu^min = {nu_min}")
    print(f"  nu^max = {nu_max}   (equal by the x<->y symmetry of Ex 7.24)")
    print(f"  nu^max - nu^min = {target}")

    # wall vectors w_p = a_p e_p - b_p e_{p-1}, boundary degrees from crit
    W = []
    for p, (k1, k2) in enumerate(crit, start=1):
        a_p, b_p = R * k2, S * k1
        col = [Fraction(0)] * (N + 1)
        col[p] += a_p
        col[p - 1] -= b_p
        W.append(col)
        print(f"  w_{p} = {a_p} e_{p} - {b_p} e_{p-1}")

    # solve target = c1 w1 + c2 w2 (overdetermined: 3 equations, 2 unknowns)
    c1 = target[0] / Fraction(W[0][0])           # e_0 sees only w_1 (coefficient -b_1)
    c2 = target[2] / Fraction(W[1][2])           # e_2 sees only w_2
    mid = c1 * W[0][1] + c2 * W[1][1]
    print(f"\n  from the e_0 row: c_1 = {c1}")
    print(f"  from the e_2 row: c_2 = {c2}")
    print(f"  consistency on the e_1 row: {mid} (must be {target[1]})")
    assert mid == target[1], "overdetermined system inconsistent"
    assert c1 != 0 and c2 != 0
    print(f"\n  BOTH nonzero: c_1 = c_2 = {c1}")
    print("  -> the u_J-component of the transition needs BOTH walls.")

    print()
    print("=" * 70)
    print("[2] can any lower-order term supply either wall instead?")
    print("=" * 70)
    print("  Wall-carrying diagonals of the pure t_(3,3) tower, |J| = 1..5:")
    carriers = []
    for k in range(1, 6):
        Nk, _, _ = diagonal(k)
        mark = "carries walls" if Nk >= 1 else "no wall (N <= 0)"
        print(f"    |J| = {k}:  N = {Nk:2d}   {mark}")
        if Nk >= 1:
            carriers.append(k)
    print(f"  carriers: |J| in {carriers}")

    # brackets need DISJOINT marking sets, so multiplicities add to 5;
    # higher exp terms of a single element land on multiples of its own monomial.
    hits = []
    for size in range(2, 6):
        for combo in combinations_with_replacement(carriers, size):
            if sum(combo) == m:
                hits.append(combo)
    print(f"\n  products of carriers whose multiplicities sum to {m}: "
          f"{hits if hits else 'NONE'}")
    assert not hits
    print("  -> no bracket and no higher exponential term reaches t_(3,3)^5.")
    print("     Hence  [log g]_{u_J}  is EXACTLY the sum of this diagonal's own")
    print("     wall elements, with the coefficients found in [1].")

    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("  Both zero counts Delta_1, Delta_2 on this diagonal are nonzero, and by")
    print("  [GKT Lem 3.39(3)(a)] both walls fire at the single time t_J. So the jump")
    print("  at t_J is supported on the two distinct rays (1,6) and (6,1).")
    print()
    print("  Therefore NO family of simple canonical homotopies as in [GKT Lem 3.39]")
    print("  has all jumps ray-supported for x^5+y^5 modulo m^6. Prop 7.23's hypothesis")
    print("  is discharged in the smallest case, with an explicit witness.")
    print()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
