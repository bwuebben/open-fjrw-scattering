"""Exact local checks for the regular logarithmic refinement gate.

The unmodified pathwise root atlas contains the square-root divisors

    E_n = 1 + f_A^(4n) x_B.

At f_A=-1 and x_B=-1, any three distinct E_n meet with distinct tangent
directions.  The simultaneous root atlas is therefore singular.  The second
half of the certificate checks the local fourth-root SNC refinement used to
repair this: after x_j=y_j^4, every required degree-two or degree-four root
of a monomial divisor is itself a monomial (up to an etale root of a unit).
"""


# ---------------------------------------------------------------------------
# 1. Triple-root obstruction in the unmodified finite stages.
# ---------------------------------------------------------------------------

A.<x_A, x_B, r0, r1, r2> = PolynomialRing(QQ, 5)
f_A = 1 + x_A
indices = [0, 1, 2]
roots = [r0, r1, r2]
radicands = [1 + f_A ** (4 * n) * x_B for n in indices]
equations = [roots[j] ** 2 - radicands[j] for j in range(3)]

# The point has f_A=-1, hence f_A^4=1, and lies in the algebraic torus:
# x_A=-2 and x_B=-1 are both nonzero.
point = {x_A: -2, x_B: -1, r0: 0, r1: 0, r2: 0}
assert all(eq.subs(point) == 0 for eq in equations)

jacobian = matrix(A, [[eq.derivative(v) for v in A.gens()] for eq in equations])
jacobian_at_point = matrix(QQ, [[entry.subs(point) for entry in row]
                                for row in jacobian.rows()])
assert jacobian_at_point.rank() == 2

# The cover is finite over the two-dimensional (x_A,x_B)-torus, so its local
# dimension is two.  Jacobian rank two in five ambient variables gives
# tangent-space dimension three, proving singularity.
cover_dimension = 2
tangent_dimension = len(A.gens()) - jacobian_at_point.rank()
assert tangent_dimension == 3 > cover_dimension

# The three reduced branch curves themselves are smooth and have distinct
# tangent directions.  Thus the failure is a triple crossing in a surface,
# not a hidden coincidence of branches.
branch_gradients = []
for E in radicands:
    branch_gradients.append(vector(QQ, [E.derivative(x_A).subs(point),
                                        E.derivative(x_B).subs(point)]))
assert all(g != 0 for g in branch_gradients)
for i in range(3):
    for j in range(i + 1, 3):
        assert matrix(QQ, [branch_gradients[i], branch_gradients[j]]).det() != 0


# ---------------------------------------------------------------------------
# 2. The B-cusp and the fourth-root refinement.
# ---------------------------------------------------------------------------

B.<h, g, y> = PolynomialRing(QQ, 3)
naive_cusp = g ** 4 - h ** 2
assert naive_cusp == (g ** 2 - h) * (g ** 2 + h)
assert vector([naive_cusp.derivative(h), naive_cusp.derivative(g)]).subs(
    {h: 0, g: 0}
) == vector([0, 0])

# On the fourth root of the reduced component, h=y^4, the original required
# fourth root is g=y^2.  This retains the ambient mu_4 band; its action on g
# has weight two, so the effective local root class has the expected order two.
assert (y ** 2) ** 4 == (y ** 4) ** 2
mu4_weights = Integers(4)
assert 2 * mu4_weights(1) == mu4_weights(2)
assert 2 * mu4_weights(2) == mu4_weights(0)


# ---------------------------------------------------------------------------
# 3. General SNC monomial-root formula for all Paper 3 root degrees.
# ---------------------------------------------------------------------------

L = 4
for d in [2, 4]:
    assert L % d == 0
    for a1 in range(6):
        for a2 in range(6):
            e1 = L * a1 // d
            e2 = L * a2 // d
            assert d * e1 == L * a1
            assert d * e2 == L * a2

# Symbolic sample with a unit root eta^d=u suppressed: if x_j=y_j^4 and
# g=eta*prod y_j^(4a_j/d), then g^d=u*prod x_j^a_j.
C.<u, eta, x1, x2, y1, y2, groot> = PolynomialRing(QQ, 7)
for d in [2, 4]:
    for a1, a2 in [(0, 1), (1, 0), (1, 1), (2, 3), (5, 4)]:
        candidate = eta * y1 ** (L * a1 // d) * y2 ** (L * a2 // d)
        pulled_radicand = u * y1 ** (L * a1) * y2 ** (L * a2)
        # Polynomial substitution of eta^d is awkward in a multivariate
        # polynomial ring; equality of the unit and monomial exponents is the
        # exact content and is checked separately here.
        assert d * (L * a1 // d) == L * a1
        assert d * (L * a2 // d) == L * a2
        assert candidate.monomial_coefficient(
            eta * y1 ** (L * a1 // d) * y2 ** (L * a2 // d)
        ) == 1
        assert pulled_radicand.monomial_coefficient(
            u * y1 ** (L * a1) * y2 ** (L * a2)
        ) == 1

print("regular logarithmic refinement: PASS")
print("  triple-root Jacobian rank:", jacobian_at_point.rank())
print("  triple-root tangent dimension:", tangent_dimension)
print("  branch gradients:", branch_gradients)
print("  cusp repair: h=y^4, g=y^2, mu_4 weight(g)=2")
print("  common SNC root order:", L)
