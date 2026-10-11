# Open FJRW wall crossing and annular scattering

**Bernd Johannes Wuebben**: three manuscripts and their exact verification code.

Gross, Kelly and Tessler constructed genus-zero open FJRW invariants for the
Landau--Ginzburg model

```math
(W,G)=(x^r+y^s,\mu_r\times\mu_s)
```

and a wall-crossing group describing their dependence on canonical boundary
conditions. These papers determine where wall-crossing begins, factor the
wall-crossing transformations along rays of boundary degrees, and, for the
quartic theory, construct an integral-affine annulus carrying a wall
structure and compare it with GKT wall-crossing. References to GKT use
version 3 of [*Open FJRW theory and mirror
symmetry*](https://arxiv.org/abs/2203.02435), published in *Geometry &
Topology* 30 (2026), 2779--2960.

## Manuscripts

### Paper 1: The central-charge threshold

[`central-charge-threshold/central-charge-threshold.pdf`](central-charge-threshold/central-charge-threshold.pdf)
proves that the genus-zero primary invariants are independent of the choice of
symmetric canonical boundary conditions if and only if the central charge is
less than one, that is, exactly for the Fermat simple singularities, as GKT
predicted. It
determines the least number of interior insertions at which dependence
appears, shows that a single descendent insertion produces dependence in
every case, and writes each wall generator as the Hamiltonian vector field
of a monomial for `dx∧dy`. It also shows that the wall-crossing group acts
simply transitively on systems of invariants once the torus vector field
`x∂x−y∂y` is adjoined to its Lie algebra; without it, transitivity fails
whenever a critical graph of weight zero occurs.

### Paper 2: Slope-ordered factorization of wall-crossing

[`slope-ordered-factorization/slope-ordered-factorization.pdf`](slope-ordered-factorization/slope-ordered-factorization.pdf)
proves that the wall term of each critical graph lies on the ray spanned by its
numbers of boundary marked points, and that the transformation between the
two extremal systems of invariants factors uniquely into automorphisms
supported on these rays, ordered by slope. With descendents the
factorization holds in every finite truncation, where it is realized by a
homotopy of boundary conditions whose jumps occur ray by ray; in general a
single homotopy in the normal form of the GKT extension lemma does not
achieve this. In each finite truncation the potential between consecutive
rays is obtained by transport across the rays from initial constants, and
for `x^5+y^5` nonzero constants occur beyond `x^r`, `y^s` and the terms with
one internal marking. Exact `x^5+y^5` calculations give the
first ray factors, and the closed correlators are checked against the open
topological recursion with Sage and admcycles.

Paper 2 does not prove that the finite realizing homotopies can be chosen
compatibly as the truncation grows. For `x^4+y^4`, Paper 3 proves this for
finite systems closed under the boundary and component dependencies of the
GKT construction (its Appendix B).

### Paper 3: Annular scattering in quartic open FJRW theory

[`fjrw-annular-scattering/fjrw-annular-scattering.pdf`](fjrw-annular-scattering/fjrw-annular-scattering.pdf)
treats `x^4+y^4`. With `x^4` and `y^4` matched to the coordinate divisors of
weight one of `P(1,1,2)`, the two first-descendent generators of a single
marking of twist (1,0) or (0,1) have fractional exponents, and the smallest cover of the
Carl--Pumperla--Siebert disk on which monodromy preserves the lattice they
span with the integral exponents is an integral-affine annulus with four
singular points and hyperbolic holonomy along its core loop. On the annulus
the slab functions need roots generating an infinite Kummer extension, the
volume form glues only in a line bundle carrying a Jacobi bracket, and there
is a wall structure consistent at its two joints at every finite order, in a
sense weaker than that of Gross and Siebert.

The paper then compares this wall structure with GKT wall-crossing. On the
two generators they agree; on the first wall carrying one marking of each,
the coefficient is -3/4 against GKT's 1/4, a difference that leaves periods
unchanged and is removed by a normal form whose geometric origin is left
open. The paper also shows that the GKT homotopies realizing the transition
between the extremal boundary conditions can be chosen compatibly as labels
are added, and transports the primary Frobenius structure, including Saito's
primitive form, through the annular charts.

## Repository layout

```text
central-charge-threshold/      Paper 1 source and PDF
slope-ordered-factorization/   Paper 2 source and PDF
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
  symmetry, modularity and wall-crossing*](https://etheses.bham.ac.uk/id/eprint/15556/),
  Ph.D. thesis, University of Birmingham, 2024.

## License

The manuscript sources and PDFs are licensed under CC BY 4.0. The verification
code is licensed under the MIT License. See [`LICENSE`](LICENSE).
