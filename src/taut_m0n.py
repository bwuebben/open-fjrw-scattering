"""Genus-0 tautological intersection numbers on \\bar M_{0,n}, self-contained (no Sage/admcycles).

    integrate(marks, psi, bdys)
      = int_{\\bar M_{0,marks}} prod_i psi_i^{psi[i]} * prod_{S in bdys} delta_S

`marks` is any finite label set (the n markings); `psi` maps a marking to its psi-power; `bdys`
is a list of boundary divisors, each given as one side S (a subset of marks; the divisor is the
partition {S, marks\\S}). Repeated S encodes self-intersection.

Method: split at a boundary divisor S = {A, B}; psi-classes and the other divisors restrict to the
two factors M_{0, A+node} x M_{0, B+node}; the KEY self-intersection relation is
    delta_S^2 = -psi_{node_A} - psi_{node_B}
and two boundary divisors that "cross" (neither side of one lies inside a side of the other) do not
meet. The base case is the genus-0 Witten-Kontsevich number <prod psi^{d_i}> = (n-3)!/prod d_i!.

VERIFIED (run `verify()` / `python src/taut_m0n.py`) against:
  * Witten-Kontsevich on \\bar M_{0,5}, \\bar M_{0,6} (pure psi);
  * the \\bar M_{0,5} = dP5 Petersen intersection form (delta^2=-1, disjoint=+1, share-one=0).
"""
from fractions import Fraction as F
from math import factorial


def wk(powers):
    """Genus-0 Witten-Kontsevich: <prod tau_{d_i}>_{0,n} = (n-3)!/prod d_i!  (0 unless sum d = n-3)."""
    n = len(powers)
    if n == 3:
        return F(1) if all(d == 0 for d in powers) else F(0)
    if n < 3 or any(d < 0 for d in powers) or sum(powers) != n - 3:
        return F(0)
    num = factorial(n - 3)
    for d in powers:
        num //= factorial(d)
    return F(num)


_ctr = [-1]


def _fresh():
    _ctr[0] -= 1
    return _ctr[0]


def integrate(marks, psi, bdys):
    marks = frozenset(marks)
    psi = {m: psi.get(m, 0) for m in marks}
    if not bdys:
        return wk([psi[m] for m in marks])
    S = frozenset(bdys[0]) & marks
    A, B = S, marks - S
    if len(A) < 2 or len(B) < 2:
        return F(0)                       # not a stable boundary divisor
    rest = bdys[1:]
    nA, nB = _fresh(), _fresh()
    marksA, marksB = A | {nA}, B | {nB}
    psiA = {m: psi[m] for m in A}; psiA[nA] = 0
    psiB = {m: psi[m] for m in B}; psiB[nB] = 0
    bdysA, bdysB = [], []
    k_self = 0
    for T in rest:
        T = frozenset(T) & marks
        Tc = marks - T
        if T == A or T == B:              # same divisor as S -> self-intersection copy
            k_self += 1
            continue
        if T <= A or Tc <= A:
            bdysA.append(T if T <= A else Tc)
        elif T <= B or Tc <= B:
            bdysB.append(T if T <= B else Tc)
        else:
            return F(0)                   # crossing divisors do not meet
    total = F(0)                          # (-psi_nA - psi_nB)^{k_self} over the two node psi-classes
    for j in range(k_self + 1):
        coeff = F((-1) ** k_self) * F(factorial(k_self) // (factorial(j) * factorial(k_self - j)))
        pA = dict(psiA); pA[nA] += j
        pB = dict(psiB); pB[nB] += (k_self - j)
        total += coeff * integrate(marksA, pA, bdysA) * integrate(marksB, pB, bdysB)
    return total


def verify():
    """Independent self-check; returns True iff every known value is reproduced."""
    checks = [
        # pure psi = Witten-Kontsevich
        (integrate({1, 2, 3, 4, 5}, {1: 2}, []), F(1)),
        (integrate({1, 2, 3, 4, 5}, {1: 1, 2: 1}, []), F(2)),
        (integrate({1, 2, 3, 4, 5, 6}, {1: 3}, []), F(1)),
        (integrate({1, 2, 3, 4, 5, 6}, {1: 2, 2: 1}, []), F(3)),
        (integrate({1, 2, 3, 4, 5, 6}, {1: 1, 2: 1, 3: 1}, []), F(6)),
        # M_{0,5} = dP5 Petersen intersection form
        (integrate({1, 2, 3, 4, 5}, {}, [{1, 2}, {1, 2}]), F(-1)),
        (integrate({1, 2, 3, 4, 5}, {}, [{1, 2}, {3, 4}]), F(1)),
        (integrate({1, 2, 3, 4, 5}, {}, [{1, 2}, {1, 3}]), F(0)),
    ]
    return all(got == exp for got, exp in checks)


if __name__ == "__main__":
    print("taut_m0n self-verification (WK + dP5 Petersen):", verify())
