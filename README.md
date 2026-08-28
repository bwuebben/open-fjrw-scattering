# Open FJRW wall crossing and annular scattering

**Bernd Johannes Wuebben** — three manuscripts and their exact verification code.

Gross, Kelly and Tessler constructed genus-zero open FJRW invariants for the
Landau--Ginzburg model

```math
(W,G)=(x^r+y^s,\mu_r\times\mu_s)
```

and a wall-crossing group describing their dependence on canonical boundary
conditions. These papers study the first wall-crossing threshold, factor the
boundary transport by slope, and construct an annular scattering and
logarithmic framework for the quartic descendent theory. References to GKT use
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

The chosen finite realizing homotopies are not proved compatible in the
inverse limit. The paper therefore does not claim a completed geometric
homotopy for the dense descendent ray set or a Gross--Siebert/SYZ theta theory
from the un-enriched continuation data alone.

### Paper 3: Annular scattering and quartic descendents

[`fjrw-annular-scattering/fjrw-annular-scattering.pdf`](fjrw-annular-scattering/fjrw-annular-scattering.pdf)
constructs a compact rational PL bordism class over finite diagrams of open
and closed quartic W-spin moduli spaces. A relative residue trace turns its
exterior boundary into the two GKT cotangent-line recursions and identifies the
two annular ends with the least- and greatest-slope canonical systems. The
supporting geometry consists of the forced annular parity cover, its finite
pro-Kummer stages, a Jacobi theta-algebra local system, and separated
fine-saturated logarithmic compactifications.

The construction is coefficientwise over finite windows. It does not produce
an analytic infinite-window virtual fundamental class, identify a coefficient
in an arbitrary GKT chamber, or select a distinguished toric degeneration of
the original quartic pair.

## Repository layout

```text
central-charge-threshold/      Paper 1 source and descriptive PDF
boundary-ray-factorization/    Paper 2 source and descriptive PDF
fjrw-annular-scattering/       Paper 3 reader source, PDF, and exact checks
src/                            shared verification programs for Papers 1--3
```

Each manuscript builds independently with `latexmk -pdf main.tex` from its
source directory. Paper 3 has a more detailed command map in
[`fjrw-annular-scattering/README.md`](fjrw-annular-scattering/README.md).

## Verification

The Python programs use exact integer, rational, and symbolic arithmetic. With
Python 3 and SymPy installed:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
for f in src/*.py; do .venv/bin/python "$f"; done
for f in fjrw-annular-scattering/src/*.py; do .venv/bin/python "$f"; done
```

The Sage programs can be run from the repository root:

```sh
for f in fjrw-annular-scattering/src/*.sage; do sage "$f"; done
```

Paper 2's full Neveu--Schwarz and Ramond recursion check additionally requires
[admcycles](https://pypi.org/project/admcycles/):

```sh
sage -pip install admcycles
sage src/verify_thm05_full.sage
```

That check reports 72/72 matching two-insertion instances and 105/105 matching
three-insertion instances. Paper 3's four final Python checks verify 116,311
factorization-corolla, 33,225 relative-contact, 654,670 affine-boundary, and
404,875 cotangent-recursion identities.

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
