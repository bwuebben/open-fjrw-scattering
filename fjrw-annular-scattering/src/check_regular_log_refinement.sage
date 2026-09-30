"""Exact local checks for the regularity statements of Appendix C
(Proposition C.1 and Theorem C.2).

Part 1 checks the multiple-root obstruction of Proposition C.1.  Write the
B_0 slab function as 1 + a*x + x^2 = (1 + lam*x)(1 + x/lam) with
lam != +-1, and put u = (1 + x_A^(-1))^4, the factor by which transport
about the branch point A~ multiplies x_B.  The transported radicands are

    rho_n = f_B(u^n x_B) = rho_n^+ rho_n^-,
    rho_n^(+-) = 1 + lam^(+-1) u^n x_B.

At p = (x_A, x_B) = (-1/2, -1/lam) one has u(p) = 1 and f_A(p) = 1/2, every
rho_n^+ vanishes with gradient (-16n, lam), and rho_n^- = 1 - lam^(-2) is a
unit.  The Kummer cover with the fourth roots of rho_0, rho_1, rho_2 has
Jacobian rank two at the point over p, hence a tangent space of dimension
three over a surface; adjoining a further root through p does not change
this.  The cusps lam = +-1 are checked as well.

Part 2 checks the B-cusp case of the local root formula in the proof of
Theorem C.2, and Part 3 the integrality of the exponents 4a/d for the root
degrees d = 2, 4 after the fourth root of every component x_j = y_j^4 of a
simple normal crossings divisor is adjoined.

Usage: sage check_regular_log_refinement.sage
"""


# ---------------------------------------------------------------------------
# 1. Multiple-root obstruction with the transported B_0 radicands.
# ---------------------------------------------------------------------------

K.<lam> = FunctionField(QQ)
A.<x_A, x_B, r0, r1, r2> = PolynomialRing(K, 5)
Q = A.fraction_field()
f_A = 1 + x_A
u = (1 + 1 / Q(x_A)) ** 4
indices = [0, 1, 2]
roots = [r0, r1, r2]
plus = [1 + lam * u ** n * x_B for n in indices]
minus = [1 + lam ** -1 * u ** n * x_B for n in indices]
radicands = [plus[j] * minus[j] for j in range(3)]
slab_parameter = lam + 1 / lam
for j, n in enumerate(indices):
    t = u ** n * x_B
    assert radicands[j] == 1 + slab_parameter * t + t ** 2
    # for n >= 0 the radicand is a Laurent polynomial: no poles on the torus
    assert radicands[j].denominator() in [x_A ** k for k in range(0, 17)]
equations = [Q(roots[j]) ** 4 - radicands[j] for j in range(3)]

point = {x_A: -1 / 2, x_B: -1 / lam, r0: 0, r1: 0, r2: 0}
assert u.subs(point) == 1
assert f_A.subs(point) == 1 / 2
assert all(eq.subs(point) == 0 for eq in equations)
for j, n in enumerate(indices):
    assert plus[j].subs(point) == 0
    assert minus[j].subs(point) == 1 - lam ** -2

jacobian_at_point = matrix(
    K, [[eq.derivative(v).subs(point) for v in A.gens()] for eq in equations]
)
assert jacobian_at_point.rank() == 2
cover_dimension = 2
tangent_dimension = len(A.gens()) - jacobian_at_point.rank()
assert tangent_dimension == 3 > cover_dimension

branch_gradients = []
for E in plus:
    branch_gradients.append(
        (E.derivative(x_A).subs(point), E.derivative(x_B).subs(point))
    )
assert branch_gradients == [(0, lam), (-16, lam), (-32, lam)]
for i in range(3):
    for j in range(i + 1, 3):
        assert matrix(K, [branch_gradients[i], branch_gradients[j]]).det() != 0
# R_0^+ and R_1^+ are local parameters at p, so O_p/(R_0^+, R_1^+) = k.
assert matrix(K, branch_gradients[:2]).det() == 16 * lam

# A further root through p (degree 2 or 4) raises the number of equations
# and of variables by one and the rank by at most one.
for lv in [2, 3, -5, QQ(1) / 3, QQ(7) / 2]:
    P.<a1, b1, s0, s1, s2, s3> = PolynomialRing(QQ, 6)
    F = P.fraction_field()
    uq = (1 + 1 / F(a1)) ** 4
    Rq = [(1 + lv * uq ** n * b1) * (1 + uq ** n * b1 / lv) for n in indices]
    for degree in (2, 4):
        extra = (1 + lv * b1) + (a1 + 1 / 2)
        eqs = [F(s0) ** 4 - Rq[0], F(s1) ** 4 - Rq[1], F(s2) ** 4 - Rq[2],
               F(s3) ** degree - extra]
        at = {a1: -1 / 2, b1: -1 / lv, s0: 0, s1: 0, s2: 0, s3: 0}
        assert all(q.subs(at) == 0 for q in eqs)
        J = matrix(QQ, [[q.derivative(v).subs(at) for v in P.gens()] for q in eqs])
        assert J.rank() <= 3
        assert len(P.gens()) - J.rank() >= 3

# The cusps lam = +-1: fourth roots of (R_n^+)^2 and square roots of R_n^+.
C.<a1, b1, s0, s1, s2> = PolynomialRing(QQ, 5)
CF = C.fraction_field()
uc = (1 + 1 / CF(a1)) ** 4
for sign in (1, -1):
    E = [1 + sign * uc ** n * b1 for n in indices]
    at = {a1: -1 / 2, b1: -sign, s0: 0, s1: 0, s2: 0}
    for n in indices:
        assert (E[n].derivative(a1).subs(at), E[n].derivative(b1).subs(at)) == (-16 * n, sign)
    for degree, rad in ((4, [e ** 2 for e in E]), (2, E)):
        eqs = [CF(s0) ** degree - rad[0], CF(s1) ** degree - rad[1],
               CF(s2) ** degree - rad[2]]
        assert all(q.subs(at) == 0 for q in eqs)
        J = matrix(QQ, [[q.derivative(v).subs(at) for v in C.gens()] for q in eqs])
        assert len(C.gens()) - J.rank() >= 3


# ---------------------------------------------------------------------------
# 2. The B-cusp and the fourth-root formula.
# ---------------------------------------------------------------------------

B.<h, g, y> = PolynomialRing(QQ, 3)
naive_cusp = g ** 4 - h ** 2
assert naive_cusp == (g ** 2 - h) * (g ** 2 + h)
assert vector([naive_cusp.derivative(h), naive_cusp.derivative(g)]).subs(
    {h: 0, g: 0}
) == vector([0, 0])

# On the fourth root of the reduced component, h = y^4, the required fourth
# root of h^2 is g = eta*y^2 with eta^4 = 1.
assert (y ** 2) ** 4 == (y ** 4) ** 2


# ---------------------------------------------------------------------------
# 3. Integrality of the local root formula for the root degrees 2 and 4.
# ---------------------------------------------------------------------------

L = 4
for d in [2, 4]:
    assert L % d == 0
    for a1_ in range(6):
        for a2_ in range(6):
            e1 = L * a1_ // d
            e2 = L * a2_ // d
            assert d * e1 == L * a1_
            assert d * e2 == L * a2_

print("regularity checks: PASS")
print("  multiple-root Jacobian rank:", jacobian_at_point.rank())
print("  multiple-root tangent dimension:", tangent_dimension)
print("  branch gradients at (-1/2,-1/lam):", branch_gradients)
print("  cusp chart: h=y^4, g=eta*y^2")
print("  common root order of the boundary components:", L)
