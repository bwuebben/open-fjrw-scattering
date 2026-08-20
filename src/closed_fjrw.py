#!/usr/bin/env python3
"""
closed_fjrw.py -- the closed FJRW invariants of the Landau-Ginzburg A-model
(x^r + y^s, mu_r x mu_s), genus zero. These are the VERTEX WEIGHTS of the
Gross-Kelly-Tessler open topological recursion (eqn (6.5) of arXiv:2203.02435v2),
hence the data needed to test Framing 2 (open invariants = broken-line counts) on
a case with a genuine 2-D scattering vertex -- the smallest such being x^5+y^5
(hyperbolic, central charge chat = 6/5, N=2 vertices exist).

WHAT IS COMPUTED
----------------
1. The genus-0 3-point (primary) invariants = the Frobenius algebra. For W=x^r+y^s
   with the maximal group mu_r x mu_s, the closed FJRW theory is the Saito-Givental
   B-model of the same polynomial, whose Frobenius algebra is the Jacobi ring
        J = C[x,y]/(x^{r-1}, y^{s-1}),   basis  phi_{(i,j)} = x^i y^j, 0<=i<=r-2, 0<=j<=s-2,
   with product phi_{(i,j)}*phi_{(k,l)} = phi_{(i+k,j+l)} (=0 if i+k>r-2 or j+l>s-2),
   unit phi_{(0,0)}, socle phi_{(r-2,s-2)}, and residue pairing
        eta(phi_{(i,j)}, phi_{(k,l)}) = [i+k=r-2][j+l=s-2].
   (Topological normalization: dual pairs pair to 1; the Saito residue gives an
   extra 1/(rs). Both are reported.) The 3-point invariant is
        < phi_a phi_b phi_c > = eta(phi_a * phi_b, phi_c)
                              = [i_a+i_b+i_c = r-2][j_a+j_b+j_c = s-2].
   This is the Sebastiani-Thom tensor square of the A_{r-1} (r-spin) Frobenius algebra.

2. The genus-0 SELECTION RULES that decide which higher/descendent invariants
   < prod tau_{d_i}^{(a_i,b_i)} >_0 are nonzero (a_i,b_i the Jacobi indices):
        e_1 := (sum a_i - (r-2))/r in Z_{>=0},   e_2 := (sum b_i - (s-2))/s in Z_{>=0},
        dimension:  sum d_i + e_1 + e_2 = n - 3.
   (Derived from the r-spin line-bundle degree; Obs 2.24 of [GKT].)

The primary Frobenius data is EXACT and verified (associativity, Frobenius symmetry,
unit). Values of invariants with e_1+e_2>0 or descendents require the r-spin
intersection recursion (Witten/FSZ) and are flagged, not fabricated.

Run:  python3 closed_fjrw.py
"""

from fractions import Fraction as F
from itertools import product as iproduct


# --------------------------------------------------------------------------
# The Frobenius algebra of FJRW(x^r+y^s, mu_r x mu_s), genus-0 primary.
# --------------------------------------------------------------------------

def basis(r, s):
    """Jacobi-ring basis indices (i,j), 0<=i<=r-2, 0<=j<=s-2."""
    return [(i, j) for i in range(r - 1) for j in range(s - 1)]


def ring_product(a, b, r, s):
    """phi_a * phi_b in C[x,y]/(x^{r-1},y^{s-1}); returns (i,j) or None if 0."""
    i, j = a[0] + b[0], a[1] + b[1]
    if i <= r - 2 and j <= s - 2:
        return (i, j)
    return None


def pairing(a, b, r, s):
    """eta(phi_a, phi_b) = [i_a+i_b=r-2][j_a+j_b=s-2] (topological normalization)."""
    return 1 if (a[0] + b[0] == r - 2 and a[1] + b[1] == s - 2) else 0


def three_point(a, b, c, r, s):
    """< phi_a phi_b phi_c >_0 = eta(phi_a*phi_b, phi_c)."""
    ab = ring_product(a, b, r, s)
    if ab is None:
        return 0
    return pairing(ab, c, r, s)


def dual(a, r, s):
    """The eta-dual sector: phi_a pairs to 1 with phi_{dual(a)}."""
    return (r - 2 - a[0], s - 2 - a[1])


# --------------------------------------------------------------------------
# Genus-0 selection rules (Jacobi indices).
# --------------------------------------------------------------------------

def witten_ranks(twists, r, s):
    """e_1,e_2 = (sum a_i-(r-2))/r, (sum b_i-(s-2))/s. Return (e1,e2) as Fractions
    (integers iff the x-/y-integrality selection rule holds)."""
    sa = sum(t[0] for t in twists)
    sb = sum(t[1] for t in twists)
    return F(sa - (r - 2), r), F(sb - (s - 2), s)


def selection_ok(twists, descendents, r, s):
    """Genus-0: e1,e2 in Z_{>=0} and sum d_i + e1 + e2 = n-3."""
    e1, e2 = witten_ranks(twists, r, s)
    if e1.denominator != 1 or e2.denominator != 1 or e1 < 0 or e2 < 0:
        return False
    n = len(twists)
    return sum(descendents) + int(e1) + int(e2) == n - 3


# --------------------------------------------------------------------------
# Verification of the Frobenius-algebra axioms.
# --------------------------------------------------------------------------

def _check(name, cond):
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


def verify_frobenius(r, s):
    B = basis(r, s)
    unit = (0, 0)
    socle = (r - 2, s - 2)
    # unit axiom
    _check("unit phi_{(0,0)}: phi_a * 1 = phi_a",
           all(ring_product(a, unit, r, s) == a for a in B))
    # pairing is symmetric and non-degenerate (each a has a unique dual in B)
    _check("pairing symmetric", all(pairing(a, b, r, s) == pairing(b, a, r, s)
                                    for a in B for b in B))
    _check("pairing non-degenerate (unique dual in basis)",
           all(dual(a, r, s) in B and pairing(a, dual(a, r, s), r, s) == 1 for a in B))
    # 3-point fully symmetric
    sym = all(three_point(a, b, c, r, s) == three_point(p, q, t, r, s)
              for a in B for b in B for c in B
              for (p, q, t) in [(a, c, b), (b, a, c), (c, b, a)])
    _check("3-point < abc > totally symmetric", sym)
    # Frobenius / associativity: eta(ab,c) = eta(a,bc)  (=> associative algebra)
    def eta_prod(a, b, c):
        ab = ring_product(a, b, r, s)
        return pairing(ab, c, r, s) if ab else 0
    assoc = all(eta_prod(a, b, c) == eta_prod(a2, b2, c2)
                for a in B for b in B for c in B
                for (a2, b2, c2) in [(a, b, c)]  # eta(ab,c)
                if True) and \
        all(eta_prod(a, b, c) == (pairing(a, ring_product(b, c, r, s), r, s)
                                  if ring_product(b, c, r, s) else 0)
            for a in B for b in B for c in B)
    _check("Frobenius: eta(ab,c) = eta(a,bc)  (associative Frobenius algebra)", assoc)
    # unit 2-point: <1,a,b> = eta(a,b)
    _check("<1,a,b> = eta(a,b)  (unit / metric compatibility)",
           all(three_point(unit, a, b, r, s) == pairing(a, b, r, s) for a in B for b in B))
    return B


# --------------------------------------------------------------------------
# Sebastiani-Thom: the algebra is the tensor square of the A_{r-1} r-spin algebra.
# --------------------------------------------------------------------------

def verify_tensor_structure(r, s):
    # A-type (1 variable x^r): basis {0..r-2}, 3-point [i+j+k=r-2], pairing [i+j=r-2].
    def A_three(i, j, k, m):
        return 1 if i + j + k == m - 2 else 0
    ok = True
    for (i, j) in basis(r, s):
        for (k, l) in basis(r, s):
            for (p, q) in basis(r, s):
                lhs = three_point((i, j), (k, l), (p, q), r, s)
                rhs = A_three(i, k, p, r) * A_three(j, l, q, s)  # tensor product
                if lhs != rhs:
                    ok = False
    _check("FJRW(x^r+y^s) 3-point = A_{r-1}(x) tensor A_{s-1}(y)  (Sebastiani-Thom)", ok)


# --------------------------------------------------------------------------
# Report + the x^5+y^5 testbed data.
# --------------------------------------------------------------------------

def run(r=5, s=5):
    print("=" * 74)
    print(f"Closed FJRW invariants of (x^{r} + y^{s}, mu_{r} x mu_{s}), genus 0")
    print("=" * 74)
    B = verify_frobenius(r, s)
    print(f"\n  state space dim = (r-1)(s-1) = {(r-1)*(s-1)}   basis phi_(i,j), "
          f"0<=i<={r-2}, 0<=j<={s-2}")
    print(f"  unit = phi_(0,0)   socle = phi_({r-2},{s-2})   "
          f"central charge chat = {F(2)-F(2,r)-F(2,s)}")
    verify_tensor_structure(r, s)

    # the 3-point invariants (topological normalization; Saito residue = /(rs))
    nz3 = [(a, b, c) for a in B for b in B for c in B if three_point(a, b, c, r, s)]
    print(f"\n  nonzero genus-0 3-point invariants: {len(nz3)} triples")
    print(f"     each = 1 (topological norm)  =  1/{r*s} (Saito residue norm)")
    print(f"     nonzero iff  i_a+i_b+i_c = {r-2}  and  j_a+j_b+j_c = {s-2}")
    print("     examples:")
    for (a, b, c) in [((0,0),(0,0),(r-2,s-2)),  # <1, 1, socle>
                      ((0,0),(1,1),(r-3,s-3)),
                      ((1,1),(1,1),(r-4,s-4)) if r>=4 and s>=4 else ((0,0),(0,0),(r-2,s-2))]:
        print(f"       < phi_{a} phi_{b} phi_{c} > = {three_point(a,b,c,r,s)}")

    # selection rules: enumerate nonzero PRIMARY (d=0) n-point invariants' existence
    print(f"\n  genus-0 selection rules (Jacobi indices):")
    print(f"     e1=(sum a_i-{r-2})/{r} in Z>=0,  e2=(sum b_i-{s-2})/{s} in Z>=0,")
    print(f"     sum d_i + e1 + e2 = n-3.")
    # count nonzero primary 3- and 4-point invariants by selection rule
    for n in (3, 4):
        cnt = 0
        for tw in iproduct(B, repeat=n):
            if selection_ok(tw, [0]*n, r, s):
                cnt += 1
        e_dist = {}
        for tw in iproduct(B, repeat=n):
            if selection_ok(tw, [0]*n, r, s):
                e1, e2 = witten_ranks(tw, r, s)
                e_dist[(int(e1), int(e2))] = e_dist.get((int(e1), int(e2)), 0) + 1
        print(f"     n={n}, all d=0: {cnt} selection-allowed twist-tuples; "
              f"(e1,e2) multiplicities {dict(sorted(e_dist.items()))}")

    # x^5+y^5 2-D vertex data: the vertex of the recursion (eqn 6.5) is a closed
    # extended invariant. The e1=e2=0 ones are the Frobenius 3-points above (value 1);
    # the e1+e2>0 ones need the r-spin recursion.
    print(f"\n  --- x^{r}+y^{s} 2-D-vertex testbed note ---")
    print(f"  The recursion vertex < tau_0^(a,b) prod tau_{{d_i}}^(a_i,b_i) >^ext is a")
    print(f"  closed invariant. Those with e1=e2=0 and all d=0 ARE the Frobenius")
    print(f"  3-points above (value 1). The scattering-vertex example X_(1,2),X_(2,1)")
    print(f"  -> X_(3,3) (from src/scattering.py) needs the vertex weights with the")
    print(f"  twists of those critical graphs; the e1=e2=0 / primary ones are given")
    print(f"  here EXACTLY. Invariants with e1+e2>0 or descendents (d>0) require the")
    print(f"  5-spin (Witten/Faber-Shadrin-Zvonkine) intersection recursion -- the")
    print(f"  next computational step, NOT fabricated here.")

    print("\n" + "=" * 74)
    print("Frobenius algebra EXACT and verified. For the SINGLE-VARIABLE 5-spin (A_4)")
    print("building blocks (primaries with a Witten class + descendents), see the")
    print("validated computation in src/rspin.py (needs sympy / the project venv).")
    print("=" * 74)


def run_5spin_extension():
    """The A_4 (5-spin) single-variable genus-0 invariants -- the building blocks for
    x^5+y^5 = A_4(x) (x) A_4(y). Computed and validated in rspin.py (Saito potential +
    string/dilaton/TRR). The 3-point of x^5+y^5 factorizes (verified above); the full
    tensor CohFT (invariants mixing x- and y-Witten classes, e_1,e_2 both > 0) is the
    remaining step -- the tensor-product / Givental R-matrix computation."""
    print("\n" + "=" * 74)
    print("5-SPIN (A_4) SINGLE-VARIABLE BUILDING BLOCKS  (src/rspin.py, validated)")
    print("=" * 74)
    try:
        from rspin import SaitoAr, enumerate_primaries, HAVE_SYMPY
        if not HAVE_SYMPY:
            raise ImportError
    except Exception:
        print("  (sympy unavailable -- run with ./venv/bin/python to see the values;")
        print("   they are validated in src/rspin.py: A_2 magnitude 1/3, WDVV A_2..A_4,")
        print("   r=2 = Witten-Kontsevich. Sign convention (-1)^e.)")
        return
    S5 = SaitoAr(5)
    print("  genus-0 primaries < tau_0^{a_1} ... >_0 of A_4 (sign convention (-1)^e):")
    for fields in enumerate_primaries(5, 6):
        e = (sum(fields) - 3) // 5
        print(f"     < {' '.join('t'+str(a) for a in fields)} > (e={e}) = {S5.primary(fields)}")
    from rspin import TensorSebastianiThom
    T = TensorSebastianiThom(5, 5)
    print("\n  x^5+y^5 TENSOR correlators, FACTORIZING part (e_1=0 or e_2=0), via A_4:")
    for ins in [[((2,1),0),((2,1),0),((2,1),0),((2,0),0)],   # e1=1,e2=0
                [((1,3),0),((1,0),0),((3,0),0),((3,0),0)],   # e1=1,e2=0
                [((3,2),0),((0,2),0),((0,2),0),((0,2),0)]]:  # e1=0,e2=1
        e1, e2 = T.e1e2(ins)
        desc = ' '.join(f"t({a},{b})" for (a, b), d in ins)
        print(f"     (e1,e2)=({e1},{e2}) < {desc} > = {T.correlator(ins)}")
    print("\n  x^5+y^5 FULLY-MIXED primary correlators, min(e_1,e_2)=1 (one Witten class a divisor),")
    print("  via genus-0 boundary reconstruction on M_{0,n} (self-contained taut_m0n; psi-gated):")
    for a, b in [((2,2,2,1,1),(1,1,2,2,2)),               # n=5 (1,1)
                 ((2,2,1,1,1,1),(3,3,3,2,1,1)),           # n=6 (1,2)
                 ((2,2,1,1,1,1,0),(3,3,3,3,3,2,1))]:      # n=7 (1,3)
        ins = [((a[i], b[i]), 0) for i in range(len(a))]
        print(f"     n={len(a)} < {' '.join(f't({a[i]},{b[i]})' for i in range(len(a)))} > = {T.correlator(ins)}")
    print("\n  x^5+y^5 SYMMETRIC-MIDDLE mixed e_1=e_2=2 (n=7), via the codim-2 stratum")
    print("  intersection form p^x M^+ p^y (dP5 method one dimension up; slow ~1-2 min):")
    a, b = (3,2,2,2,2,1,1), (1,1,2,2,2,2,3)
    v = T.correlator([((a[i], b[i]), 0) for i in range(7)])
    print(f"     < {' '.join(f't({a[i]},{b[i]})' for i in range(7))} > = {v}"
          f"   (permutation-invariant; expect 34/625)")
    print("  Remaining: min(e_1,e_2)>=2 with e_1!=e_2, or e_1=e_2>=3 -> None (higher-codim recon).")
    print("=" * 74)


if __name__ == "__main__":
    run(5, 5)
    run_5spin_extension()
