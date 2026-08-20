#!/usr/bin/env python3
"""
scattering.py -- leading-order BCH calculation in the abstract affine
Hamiltonian vector-field algebra, computed by explicit automorphism composition.
Two generators scatter, and a single correction at the sum direction restores
consistency to leading order, as in the tropical-vertex baby calculation.

This is retained as a negative control for naive ungraded seeding. The vector
fields below carry no deformation monomials and therefore do not by themselves
define walls of the GKT algebra: ``gkt_algebra.py`` checks the missing degree
constraint, and ``canonical_diagram.py`` computes the legal first GKT walls.

Generators (Section 5):  X_{a,b} = x^a y^b((b+1)x d_x - (a+1)y d_y),
bracket [X_{a,b},X_{c,d}] = ((b+1)(c+1)-(a+1)(d+1)) X_{a+c,b+d}  (verified).

An automorphism theta = exp(V) (V a vector field, nilpotent mod truncation) acts
on the ring by  theta(f) = sum_n V^n(f)/n!  = f(theta(x), theta(y)); it is a ring
homomorphism, so composition is substitution. We form the group commutator
    Phi = exp(u) exp(v) exp(-u) exp(-v)
of the two wall automorphisms and show its leading nontrivial term is [u,v]
applied, i.e. Phi = exp([u,v] + higher). Hence adding one wall exp(-[u,v]) at the
sum direction cancels the obstruction: exp(-[u,v]) . Phi = id to leading order.
The scattering coefficient IS the Section-5 structure constant.

Everything exact (fractions.Fraction); truncated at total degree <= N.
Run:  python3 scattering.py
"""

from fractions import Fraction as F


# ---- polynomial arithmetic, truncated at total degree <= N ----------------

def deg_ok(k, N):
    return k[0] + k[1] <= N


def add(*ps):
    r = {}
    for p in ps:
        for k, v in p.items():
            r[k] = r.get(k, 0) + v
    return {k: v for k, v in r.items() if v != 0}


def scale(p, c):
    return {k: c * v for k, v in p.items() if c * v != 0}


def mul(p, q, N):
    r = {}
    for (i, j), a in p.items():
        for (k, l), b in q.items():
            key = (i + k, j + l)
            if deg_ok(key, N):
                r[key] = r.get(key, 0) + a * b
    return {k: v for k, v in r.items() if v != 0}


def dx(p):
    return {(i - 1, j): v * i for (i, j), v in p.items() if i != 0}


def dy(p):
    return {(i, j - 1): v * j for (i, j), v in p.items() if j != 0}


def X(a, b):
    # X_{a,b} = (b+1) x^{a+1} y^b d_x - (a+1) x^a y^{b+1} d_y
    return ({(a + 1, b): b + 1}, {(a, b + 1): -(a + 1)})


def apply_vf(V, f, N):
    (P, Q) = V
    r = add(mul(P, dx(f), N), mul(Q, dy(f), N))
    return {k: v for k, v in r.items() if deg_ok(k, N)}


def bracket(V, W, N):
    (P1, Q1), (P2, Q2) = V, W
    Bx = add(mul(P1, dx(P2), N), mul(Q1, dy(P2), N),
             scale(add(mul(P2, dx(P1), N), mul(Q2, dy(P1), N)), -1))
    By = add(mul(P1, dx(Q2), N), mul(Q1, dy(Q2), N),
             scale(add(mul(P2, dx(Q1), N), mul(Q2, dy(Q1), N)), -1))
    return (Bx, By)


# ---- exp(V) as an automorphism (image of x and y), and composition --------

def exp_on(V, f, N):
    """exp(V)(f) = sum_n V^n(f)/n!, truncated at total degree <= N."""
    term = dict(f)
    total = dict(f)
    n = 0
    while term:
        n += 1
        term = scale(apply_vf(V, term, N), F(1, n))
        total = add(total, term)
        if n > 4 * N:
            break
    return total


def auto(V, N):
    return (exp_on(V, {(1, 0): 1}, N), exp_on(V, {(0, 1): 1}, N))


def subst(g, A, N):
    """g(Ax, Ay), truncated. Ax = A[0], Ay = A[1]."""
    Ax, Ay = A
    # precompute powers of Ax, Ay up to degree N
    powx = {0: {(0, 0): 1}}
    powy = {0: {(0, 0): 1}}
    for e in range(1, N + 1):
        powx[e] = mul(powx[e - 1], Ax, N)
        powy[e] = mul(powy[e - 1], Ay, N)
    r = {}
    for (i, j), c in g.items():
        if i < 0 or j < 0:
            raise ValueError("negative exponent in subst")
        term = mul(powx[i], powy[j], N)
        r = add(r, scale(term, c))
    return r


def compose(A, B, N):
    """(A o B) as automorphisms: (A o B)(x) = subst(Bx, A)."""
    return (subst(B[0], A, N), subst(B[1], A, N))


def is_identity(A):
    return A[0] == {(1, 0): 1} and A[1] == {(0, 1): 1}


def diff_from_id(A):
    dxr = add(A[0], scale({(1, 0): 1}, -1))
    dyr = add(A[1], scale({(0, 1): 1}, -1))
    return dxr, dyr


def min_deg(p):
    return min((k[0] + k[1] for k in p), default=None)


# ---------------------------------------------------------------------------

def _check(name, cond):
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


def run():
    print("=" * 74)
    print("Leading-order BCH calculation in the affine Hamiltonian algebra")
    print("=" * 74)

    N = 11
    # two abstract generators: u on direction (1,2), v on direction (2,1)
    u = X(1, 2)
    v = X(2, 1)
    mu = scale(u[0], -1), scale(u[1], -1)   # -u
    mv = scale(v[0], -1), scale(v[1], -1)   # -v

    print("\nGenerators: u = X_{1,2} (dir (1,2)),  v = X_{2,1} (dir (2,1))")
    br = bracket(u, v, N)
    print("Section-5 bracket [u,v] = 5 X_{3,3}; check as vector field:")
    _check("[X_{1,2},X_{2,1}] == 5 X_{3,3}",
           br[0] == scale(X(3, 3)[0], 5) and br[1] == scale(X(3, 3)[1], 5))

    # automorphisms
    tu = auto(u, N)
    tv = auto(v, N)
    tmu = auto(mu, N)
    tmv = auto(mv, N)

    # group commutator Phi = exp(u) exp(v) exp(-u) exp(-v)
    Phi = compose(tu, compose(tv, compose(tmu, tmv, N), N), N)
    dphix, dphiy = diff_from_id(Phi)
    md = min(min_deg(dphix) or 99, min_deg(dphiy) or 99)
    print(f"\nCommutator Phi = [exp(u),exp(v)] deviates from identity first at "
          f"total degree {md}")
    _check("Phi != identity (the two walls are inconsistent)", not is_identity(Phi))
    _check("leading deviation of Phi is at degree 7 (the X_{3,3} wall)", md == 7)

    # the leading term of Phi should equal [u,v] applied to x,y (BCH order 1)
    # exp([u,v])(x) - x  vs  Phi(x) - x, matched at degree 7
    exp_br = auto(br, N)
    ebx, eby = diff_from_id(exp_br)
    deg7_match_x = {k: val for k, val in dphix.items() if k[0] + k[1] == 7} == \
                   {k: val for k, val in ebx.items() if k[0] + k[1] == 7}
    deg7_match_y = {k: val for k, val in dphiy.items() if k[0] + k[1] == 7} == \
                   {k: val for k, val in eby.items() if k[0] + k[1] == 7}
    _check("Phi's degree-7 part == exp([u,v])'s degree-7 part (BCH leading order)",
           deg7_match_x and deg7_match_y)

    # CANCELLATION: add the correction wall exp(-[u,v]) at direction (3,3).
    corr = auto((scale(br[0], -1), scale(br[1], -1)), N)   # exp(-[u,v])
    Psi = compose(corr, Phi, N)
    dpx, dpy = diff_from_id(Psi)
    md2 = min(min_deg(dpx) or 99, min_deg(dpy) or 99)
    print(f"\nAdd one wall exp(-[u,v]) at (3,3): the product closes up.")
    _check("exp(-[u,v]) . Phi has NO degree-7 obstruction (leading order consistent)",
           all(k[0] + k[1] != 7 for k in dpx) and all(k[0] + k[1] != 7 for k in dpy))
    md2_str = f"degree {md2}" if md2 < 99 else f"beyond truncation N={N}"
    print(f"      residual obstruction now first appears at {md2_str}: the NEXT")
    print(f"      scattering order. Its directions are the next algebraic Lie directions:")
    # identify the directions of the residual (grade = function-degree minus the base)
    # a residual term c*x^i y^j in delta(x) comes from a wall X_{m} with m=(i-1,j)
    resid_dirs = set()
    for (i, j) in list(dpx.keys()):
        resid_dirs.add((i - 1, j))
    for (i, j) in list(dpy.keys()):
        resid_dirs.add((i, j - 1))
    resid_dirs = sorted(d for d in resid_dirs if d[0] >= 1 and d[1] >= 1)
    print(f"      residual wall directions (a,b): {resid_dirs}")
    print(f"      note (1,2)+2*(2,1)=(5,4) and 2*(1,2)+(2,1)=(4,5): the degree-10 walls.")

    print("\n" + "=" * 74)
    print("ALGEBRA CHECK: the two generators scatter, and a correction at (3,3)")
    print("restores consistency to leading order. The scattering coefficient is the")
    print("Section-5 structure constant 5. This is not a GKT wall seeding without")
    print("the deformation-degree constraint checked in gkt_algebra.py.")
    print("=" * 74)


if __name__ == "__main__":
    run()
