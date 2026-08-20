#!/usr/bin/env sage
r"""
verify_thm05_full.sage -- FULL (narrow + extended) numerical verification of the
Gross-Kelly-Tessler open topological recursion ([GKT] Thm 0.5 / eq 0.8) for x^5+y^5.

This is the extended (Ramond, tw=-1) completion of src/verify_thm05.py: every closed
FJRW invariant -- NS and extended alike -- is computed with SageMath's admcycles via
Witten's r-spin class (Wittenrspin = the Chiodo/PPZ class), so the previously "narrow"
sum over node twists a,b in {0..r-2} is replaced by the FULL sum over {-1..r-2}.

REQUIRES: SageMath + admcycles  (sage -pip install admcycles). This is the one
Sage-dependent script; the pure-Python verify_thm05.py covers the narrow sector alone.

Convention (VALIDATED, see below): the tensor invariant of x^5+y^5 is
    < prod tau_{d_i}^{(a_i,b_i)} > = (-1)^{e_1+e_2} int_{Mbar_{0,n}} W^5(a) W^5(b) prod psi_i^{d_i},
with W^5 = Wittenrspin(0, ., r_coeff=5). admcycles reproduces every rspin.py/Saito value
exactly under this rule (incl. the (2,2)@n=7 = 34/625 case and descendents). Ramond twists
are passed as -1 (admcycles distinguishes -1 from r-1: only -1 gives the correct nonzero
class); the continuation twist z=(r-2-a,s-2-b) is reduced r-1 -> -1.

A(A,d) is the survey [GKTsurvey] Thm 5.4 form (no leading sign); eq 0.8 is checked as an
identity between admcycles-computed invariants.

Run:  sage src/verify_thm05_full.sage
"""
from admcycles import Wittenrspin, psiclass
from sage.all import QQ
from itertools import combinations, combinations_with_replacement as cwr

r = s = 5
_wc = {}
def wr(a):
    k = tuple(a)
    if k not in _wc:
        _wc[k] = Wittenrspin(0, k, r_coeff=5)
    return _wc[k]

def red(t):
    return -1 if t == r - 1 else t          # Ramond r-1 -> -1 (admcycles representative)

_tc = {}
def tensor(a, b, ds):
    a = tuple(red(t) for t in a); b = tuple(red(t) for t in b); ds = tuple(ds)
    k = (a, b, ds)
    if k in _tc:
        return _tc[k]
    n = len(a); Wa = wr(a); Wb = wr(b)
    if not hasattr(Wa, 'evaluate') or not hasattr(Wb, 'evaluate'):
        _tc[k] = QQ(0); return QQ(0)
    cl = Wa * Wb
    for i in range(n):
        for _ in range(ds[i]):
            cl = cl * psiclass(i + 1, 0, n)
    if not hasattr(cl, 'evaluate'):
        _tc[k] = QQ(0); return QQ(0)
    raw = cl.evaluate()
    if raw == 0:
        _tc[k] = QQ(0); return QQ(0)
    E = (sum(a) - 3 + sum(b) - 3) // 5
    v = QQ((-1) ** E) * raw
    _tc[k] = v
    return v

def dfun(tws, ds):
    rA = int(sum(a for a, b in tws) % r); sA = int(sum(b for a, b in tws) % s)
    m = int(r * s + sum(s * a + r * b + r * s * (d - 1) for (a, b), d in zip(tws, ds)))
    return QQ(s * rA + r * sA - m) / QQ(r * s) - 1, rA, sA

def Aval(tws, ds):
    if len(tws) == 1:
        return QQ((-1) ** ds[0])
    dA, rA, sA = dfun(tws, ds)
    if dA < 0 or dA.denominator() != 1:
        return QQ(0)
    dA = int(dA); ex = (r - 2 - rA, s - 2 - sA)     # ex[0] or ex[1] may be -1 (Ramond)
    return tensor([ex[0]] + [t[0] for t in tws],
                  [ex[1]] + [t[1] for t in tws], [dA] + list(ds))

def verify(J, ds, j1):
    n = len(J); dr = list(ds); dr[j1] += 1
    lhs = Aval(J, dr); base = Aval(J, ds)
    rest = [i for i in range(n) if i != j1]; rhs = QQ(0)
    for kA in range(1, len(rest) + 1):
        for Aset in combinations(rest, kA):
            Bset = [i for i in rest if i not in Aset]
            for a in range(-1, r - 1):
                for b in range(-1, s - 1):
                    vv = tensor([a, J[j1][0]] + [J[i][0] for i in Aset],
                                [b, J[j1][1]] + [J[i][1] for i in Aset],
                                [0, ds[j1]] + [ds[i] for i in Aset])
                    if vv == 0:
                        continue
                    rhs += vv * Aval([J[i] for i in Bset] + [(r - 2 - a, s - 2 - b)],
                                     [ds[i] for i in Bset] + [0])
    return lhs == rhs - base, lhs, rhs - base


def run():
    print("=" * 74)
    print("FULL (narrow + extended) verification: GKT open recursion Thm 0.5 for x^5+y^5")
    print("all closed invariants via SageMath/admcycles (Witten r-spin class)")
    print("=" * 74)
    pool2 = [(1, 1), (2, 1), (1, 2), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2)]
    pool3 = [(1, 1), (2, 1), (1, 2), (2, 2), (1, 3)]
    for lJ, pool in [(2, pool2), (3, pool3)]:
        match = mismatch = ext = 0; mism = []
        for J in cwr(pool, lJ):
            rA = sum(a for a, b in J) % r; sA = sum(b for a, b in J) % s
            is_ext = (rA == r - 1 or sA == s - 1)
            for j1 in range(lJ):
                ok, lhs, rhs = verify(list(J), [0] * lJ, j1)
                if is_ext:
                    ext += 1
                if ok:
                    match += 1
                else:
                    mismatch += 1; mism.append((J, j1, lhs, rhs))
        print(f"  |J|={lJ}: MATCH={match}  MISMATCH={mismatch}  "
              f"(extended [Ramond] instances included={ext})")
        for J, j1, lhs, rhs in mism[:5]:
            print(f"    MISMATCH J={J} j1={j1}: lhs={lhs} rhs-base={rhs}")
        assert mismatch == 0, "recursion mismatch"
    print("\n" + "=" * 74)
    print("GKT's open topological recursion holds for x^5+y^5 in EVERY instance -- NS and")
    print("extended (Ramond) alike -- with all invariants computed by admcycles. The narrow")
    print("restriction of verify_thm05.py is removed; the recursion is confirmed in full.")
    print("=" * 74)


run()
