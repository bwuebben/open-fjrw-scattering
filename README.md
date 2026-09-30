# Open FJRW wall crossing and annular scattering

**Bernd Johannes Wuebben**: three manuscripts and their exact verification code.

Gross, Kelly and Tessler constructed genus-zero open FJRW invariants for the
Landau--Ginzburg model

```math
(W,G)=(x^r+y^s,\mu_r\times\mu_s)
```

and a wall-crossing group describing their dependence on canonical boundary
conditions. These papers study the first wall-crossing threshold, factor the
boundary transport by slope, and, for the quartic theory, construct an
integral-affine annulus carrying a consistent scattering diagram and compare
it with GKT wall-crossing. References to GKT use
version 3 of [*Open FJRW theory and mirror
symmetry*](https://arxiv.org/abs/2203.02435), published in *Geometry &
Topology* 30 (2026), 2779--2960.

## Manuscripts

### Paper 1: The central-charge threshold

[`central-charge-threshold/central-charge-threshold.pdf`](central-charge-threshold/central-charge-threshold.pdf)
proves that the primary invariants are independent of canonical boundary
conditions precisely when the central charge is less than one, equivalently
for the two-variable Fermat simple singularities. It gives the exact first-wall
trichotomy, exhibits a universal one-insertion descendent wall, and identifies
each wall vector field with a Hamiltonian vector field in a graded subalgebra
of the tropical vertex Lie algebra.

### Paper 2: Boundary-ray factorization

[`boundary-ray-factorization/boundary-ray-factorization.pdf`](boundary-ray-factorization/boundary-ray-factorization.pdf)
proves unique increasing-slope factorization of primary transitions and
coefficientwise factorization for descendents on finite divisor-closed
coefficient sets. It constructs the two extremal systems, reconstructs every
open potential from its initial coefficients, and realizes every finite
factorization by a concatenation of GKT canonical homotopies. Its exact
`x^5+y^5` calculations include a complete Sage/admcycles check of the relevant
open topological recursion.

Paper 2 does not prove that the finite realizing homotopies can be chosen
compatibly as the coefficient set grows. Paper 3 proves this for finite
systems closed under the boundary and component dependencies of the GKT
construction (its Appendix B).

### Paper 3: Annular scattering in quartic open FJRW theory

[`fjrw-annular-scattering/fjrw-annular-scattering.pdf`](fjrw-annular-scattering/fjrw-annular-scattering.pdf)
shows that the first descendent generators of the quartic theory have
fractional exponents, which force an index-eight enlargement of the exponent
lattice of the Carl--Pumperla--Siebert base. The smallest cover
of the natural disk-shaped base that preserves it is an integral-affine
annulus with four singular points and hyperbolic core holonomy of trace 18.
On this annulus the paper constructs a coefficient system of Kummer
extensions that no finite root stack contains, a Jacobi--Kirillov bracket on a
twisted volume line, and a scattering diagram that is consistent at every
finite order; the walls leave through the two
boundary circles, and the finite root-stack presentations have a regular
dominating presentation.

It then compares this structure with GKT open FJRW theory: the GKT critical
Lie algebra embeds in the wall algebra without change of coefficients, but the
diagram is not GKT wall-crossing (on the first mixed ray its coefficient is
-3/4, against GKT's 1/4), each wall factor splits into a GKT factor and a
residual factor, the finite GKT
homotopies can be chosen compatibly, and the primary Frobenius structure,
including Saito's primitive form, is transported through the annular charts.
For positive descendents the paper computes the first truncations: the
consistent diagram does not carry one extremal GKT potential to the other, a
maximal normal form removes the difference, and its geometric realization is
stated as an open problem. Theta functions on the annulus and a logarithmic
compactification of the finite stages are also left open.

## Repository layout

```text
central-charge-threshold/      Paper 1 source and PDF
boundary-ray-factorization/    Paper 2 source and PDF
fjrw-annular-scattering/       Paper 3 source, PDF, and exact checks
src/                            shared verification programs (Papers 1 and 2;
                                two Paper 3 checks import src/a_invariants.py)
```

Each manuscript builds independently with `latexmk -pdf main.tex` from its
source directory. Paper 3 has its own command list in
[`fjrw-annular-scattering/README.md`](fjrw-annular-scattering/README.md).

## Verification

The Python programs use exact integer, rational, and symbolic arithmetic. With
Python 3 and SymPy installed:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
for f in src/*.py; do .venv/bin/python "$f"; done
```

Paper 2's full Neveu--Schwarz and Ramond recursion check additionally requires
SageMath and [admcycles](https://pypi.org/project/admcycles/):

```sh
sage -pip install admcycles
sage src/verify_thm05_full.sage
```

That check reports 72/72 matching two-insertion instances and 105/105 matching
three-insertion instances.

## References

- M. Gross, T. L. Kelly, and R. J. Tessler, [*Open FJRW theory and mirror
  symmetry*](https://arxiv.org/abs/2203.02435), *Geom. Topol.* 30 (2026),
  2779--2960.
- M. Gross, T. L. Kelly, and R. J. Tessler, [*Open enumerative geometries for
  Landau--Ginzburg models*](https://arxiv.org/abs/2602.12707).
- R. Maher, [*Predictions in open Fan--Jarvis--Ruan--Witten theory via mirror
  symmetry, modularity, and wall-crossing*](https://etheses.bham.ac.uk/id/eprint/15556/),
  Ph.D. thesis, University of Birmingham, 2024.

## License

The manuscript sources and PDFs are licensed under CC BY 4.0. The verification
code is licensed under the MIT License. See [`LICENSE`](LICENSE).
