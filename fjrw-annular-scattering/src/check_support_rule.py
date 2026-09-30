"""Exact checks for the wall-direction paragraph of Section 2.2.

Standard coordinates: the exponent lattice M (the fan lattice of P(1,1,2))
and its dual M^vee (the plane of the polygon Delta, coordinates (s1, s2)),
paired by the dot product.  Bidegrees (a, b) of monomials x^a y^b form Z^2.
The script verifies:

1. L = det(iota) iota^{-T} = -1/2 F^{-T}, hence <Lq, F(q')> = -1/2 q.q';
2. Lq = d psi(F(q)) for the quadratic function psi(F(a,b)) = -(a^2+b^2)/4,
   whose gradient is -8 (iota iota^T)^{-1}, of determinant 16;
3. at each singular point V the star edge through V is fixed by the
   monodromy of M^vee and annihilated by the invariant character u_V of the
   monodromy of M, which is tangent to the dual bounded edge;
4. the exchange of x and y acts on M by S = F sigma F^{-1}, which fixes v_Z
   and exchanges v_X, v_Y, and on the plane of Delta by S^T, which preserves
   Delta, fixes p_B and B and exchanges p_A, p_C and A, C;
5. L sigma = S^T L, whereas the kernel rule q -> F^{-T} q^perp (support in
   the annihilator of the exponent) satisfies K sigma = -S^T K;
6. the linear maps P with P sigma = S^T P are exactly F^{-T}(alpha Id +
   beta sigma); each is the composite of F with the gradient of the
   quadratic function F(a,b) -> (alpha(a^2+b^2) + 2 beta ab)/2, is
   orientation-preserving in the sense det(P F^{-1}) > 0 exactly when
   |beta| < |alpha|, and sends (1,1) into the line through O and B;
7. the diagonal support: L(1,1) runs from O to B, while the kernel rule
   sends (1,1) to the line s2 = 0, which meets Delta in the segment from
   (-1,0) to (1,0) and misses A, B and C.

Run with Python 3 and SymPy:
    python3 fjrw-annular-scattering/src/check_support_rule.py
"""

import sympy as sp

checks = 0


def check(cond, msg):
    global checks
    if not cond:
        raise AssertionError(msg)
    checks += 1


iota = sp.Matrix([[1, -1], [0, -2]])
F = iota / 4
L = iota.det() * iota.inv().T
FmT = F.inv().T
sigma = sp.Matrix([[0, 1], [1, 0]])
q1, q2, r1, r2, al, be = sp.symbols("q1 q2 r1 r2 alpha beta")
q = sp.Matrix([q1, q2])
r = sp.Matrix([r1, r2])


def perp(v):
    return sp.Matrix([v[1], -v[0]])


# 1. the cofactor is -1/2 F^{-T}
check(L == sp.Matrix([[-2, 0], [1, 1]]), "cofactor matrix")
check(L == -FmT / 2, "L = -1/2 F^{-T}")
check(sp.expand((L * q).dot(F * r) + (q.dot(r)) / 2) == 0,
      "<Lq, F(q')> = -1/2 q.q'")

# 2. Legendre transform of a quadratic function
y1, y2 = sp.symbols("y1 y2")
yv = sp.Matrix([y1, y2])
ab = F.inv() * yv
psi = -(ab[0] ** 2 + ab[1] ** 2) / 4
grad = sp.Matrix([sp.diff(psi, y1), sp.diff(psi, y2)])
G = -8 * (iota * iota.T).inv()
check(sp.simplify(grad - G * yv) == sp.zeros(2, 1), "grad psi = -8(ii^T)^{-1}")
check(G == G.T and G.det() == 16, "symmetric, determinant 16")
check(sp.simplify(G * F * q - L * q) == sp.zeros(2, 1), "Lq = dpsi(F(q))")

# 3. cuts and slabs
vX, vY, vZ = sp.Matrix([1, 0]), sp.Matrix([-1, -2]), sp.Matrix([0, 1])
check(iota * sp.Matrix([1, 0]) == vX and iota * sp.Matrix([0, 1]) == vY,
      "iota(e1) = v_X, iota(e2) = v_Y")
check(F * sp.Matrix([-2, -2]) == vZ, "F(-2,-2) = v_Z")
mono = {"A": sp.Matrix([[0, -1], [1, 2]]),
        "B": sp.Matrix([[3, -2], [2, -1]]),
        "C": sp.Matrix([[4, -1], [9, -2]])}
pV = {"A": sp.Matrix([-1, -1]), "B": sp.Matrix([-1, 1]), "C": sp.Matrix([3, -1])}
dual_edge = {"A": vX - vZ, "B": vX - vY, "C": vZ - vY}
for V, T in mono.items():
    ker = (T - sp.eye(2)).nullspace()
    check(len(ker) == 1, f"transvection at {V}")
    u = ker[0]
    check(sp.Matrix.hstack(u, dual_edge[V]).det() == 0,
          f"u_{V} tangent to the dual edge")
    check(u.dot(pV[V]) == 0, f"u_{V} annihilates the star edge through {V}")
    check(T.inv().T * pV[V] == pV[V], f"star edge through {V} is invariant")

# 4. the exchange of x and y
S = F * sigma * F.inv()
check(S == sp.Matrix([[-1, 0], [-2, 1]]), "S")
check(S * vZ == vZ and S * vX == vY and S * vY == vX, "S on the rays")
St = S.T
check(St * pV["B"] == pV["B"] and St * pV["A"] == pV["C"]
      and St * pV["C"] == pV["A"], "S^T preserves Delta")
mid = {V: pV[V] / 2 for V in pV}
check(St * mid["B"] == mid["B"] and St * mid["A"] == mid["C"],
      "S^T fixes B and exchanges A, C")
check((St - sp.eye(2)).nullspace()[0].T * sp.Matrix([[0, 1], [-1, 0]])
      * pV["B"] == sp.zeros(1, 1), "fixed line of S^T is the line O p_B")

# 5. equivariance of the cofactor, anti-equivariance of the kernel rule
check(L * sigma == St * L, "L sigma = S^T L")
Kr = FmT * sp.Matrix([[0, 1], [-1, 0]])  # q -> F^{-T} q^perp
check(sp.simplify(Kr * q - FmT * perp(q)) == sp.zeros(2, 1), "kernel rule")
check(sp.expand((Kr * q).dot(F * q)) == 0, "kernel rule annihilates F(q)")
check(Kr * sigma == -St * Kr, "K sigma = -S^T K")

# 6. all equivariant linear rules
p = sp.symbols("p0:4")
P = sp.Matrix(2, 2, p)
sol = sp.solve(list(P * sigma - St * P), p, dict=True)
check(len(sol) == 1, "equivariance is linear")
Pg = P.subs(sol[0])
Qg = F.T * Pg  # P = F^{-T} Q
free = sorted(Qg.free_symbols, key=str)
check(len(free) == 2, "two-parameter family")
check(Qg[0, 0] == Qg[1, 1] and Qg[0, 1] == Qg[1, 0], "Q = alpha Id + beta sigma")
Q = al * sp.eye(2) + be * sigma
Pab = FmT * Q
Gab = Pab * F.inv()
check(sp.simplify(Gab - Gab.T) == sp.zeros(2, 2), "P F^{-1} is symmetric")
psiab = (al * (ab[0] ** 2 + ab[1] ** 2) + 2 * be * ab[0] * ab[1]) / 2
gradab = sp.Matrix([sp.diff(psiab, y1), sp.diff(psiab, y2)])
check(sp.simplify(gradab - Gab * yv) == sp.zeros(2, 1),
      "P = d psi_(alpha,beta) o F")
check(sp.factor(Gab.det()) == sp.factor(64 * (al ** 2 - be ** 2)),
      "det(P F^{-1}) = 64(alpha^2 - beta^2)")
diag = Pab * sp.Matrix([1, 1])
check(sp.simplify(diag[0] + diag[1]) == 0, "diagonal into the line O p_B")
check(Pab.subs({al: sp.Rational(-1, 2), be: 0}) == L, "cofactor is (-1/2, 0)")

# 7. the diagonal support
d = L * sp.Matrix([1, 1])
check(d == sp.Matrix([-2, 2]) and d / 4 == mid["B"], "L(1,1) runs from O to B")
k = Kr * sp.Matrix([1, 1])
check(k == sp.Matrix([4, 0]), "kernel rule sends (1,1) to (4,0)")
# Delta = {s1 >= -1, s2 >= -1, s1 + 2 s2 <= 1}; on s2 = 0: -1 <= s1 <= 1
check(all(mid[V][1] != 0 for V in mid), "s2 = 0 misses A, B, C")

print(f"check_support_rule: all {checks} checks passed")
