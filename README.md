# Open FJRW boundary-ray factorization

**Bernd Johannes Wuebben** — two manuscripts and their exact verification code.

Gross, Kelly and Tessler constructed genus-zero open FJRW invariants for the
Landau--Ginzburg model

```math
(W,G)=(x^r+y^s,\mu_r\times\mu_s)
```

and a wall-crossing group describing their dependence on canonical boundary
conditions. This repository studies the first walls, reorganizes their
automorphisms by boundary direction, and relates the resulting algebraic
factorization to actual homotopies of GKT boundary conditions. References to
GKT use version 3 of [*Open FJRW theory and mirror
symmetry*](https://arxiv.org/abs/2203.02435).

## Manuscripts

### The central-charge threshold

[`companion/central-charge-threshold.pdf`](companion/central-charge-threshold.pdf)
proves that the primary invariants are independent of canonical boundary
conditions precisely when the central charge is less than one, equivalently
for the two-variable Fermat simple singularities. It also gives the exact
first-wall trichotomy, exhibits a universal one-insertion descendent wall, and
identifies each wall vector field with a Hamiltonian vector field in a graded
subalgebra of the tropical vertex Lie algebra. At central charge one, the
first wall is the torus direction
`x d/dx - y d/dy`.

### Boundary-ray factorization and geometric wall-crossing

[`paper/boundary-ray-factorization.pdf`](paper/boundary-ray-factorization.pdf)
proves:

1. unique increasing-slope factorization of primary transitions over the rays
   spanned by the two boundary-marking counts, and coefficientwise
   factorization for descendents on finite divisor-closed coefficient sets;
2. canonical extremal systems and a canonical boundary-to-boundary
   factorization between them;
3. reconstruction of every open potential from unique initial coefficients by
   successive ray transport;
4. realization, for every finite primary or descendent coefficient set, by an
   actual concatenation of GKT canonical homotopies whose nonzero projected
   jumps occur on one boundary ray at a time, in increasing slope;
5. exact first factors and invariant values for `x^5+y^5`, together with a
   complete Sage/admcycles check of the relevant open topological recursion;
6. a projective obstruction: the punctured boundary-charge sector records
   slope transport but supplies neither origin monodromy nor a singular
   integral-affine position space.

The finite algebraic factorizations are compatible as the coefficient set
grows. The chosen realizing homotopies are **not** proved compatible in that
limit. Thus the paper does not construct one completed geometric homotopy for
the dense descendent ray set, nor does it claim a Gross--Siebert/SYZ theta
theory from the un-enriched continuation data alone.

## Repository layout

```text
paper/       boundary-ray paper: LaTeX source and descriptive PDF
companion/   central-charge note: LaTeX source and descriptive PDF
src/         exact verification programs
```

## Verification

The Python programs use exact integer, rational, and symbolic arithmetic. With
Python 3 and SymPy installed:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
for f in src/*.py; do .venv/bin/python "$f"; done
```

Equivalently, a SageMath installation can run the same programs with
`sage -python`. The full Neveu--Schwarz and Ramond recursion check additionally
requires [admcycles](https://pypi.org/project/admcycles/):

```sh
sage -pip install admcycles
sage src/verify_thm05_full.sage
```

The full check reports 72/72 matching two-insertion instances and 105/105
matching three-insertion instances.

### Code map

| Programs | Mathematical content |
|---|---|
| `census.py`, `gkt_algebra.py` | critical-graph census, central-charge trichotomy, wall algebra |
| `canonical_diagram.py`, `diagonal_wall_count.py`, `first_two_ray_diagonal.py`, `two_ray_forced.py` | canonical factors, diagonal structure, and the first forced two-ray example |
| `anchor_induction.py`, `backscatter.py`, `canonical_seeds.py`, `farside_test.py`, `seed_chain_formula.py`, `transport_engine.py`, `mixed_sector.py` | seed inversion and boundary-ray transport |
| `ray_accumulation.py`, `ray_density_exact.py` | primary local finiteness and descendent ray accumulation |
| `mirror_periods.py`, `oscillatory_check.py`, `a_invariants.py` | period functionals and exact open-invariant relations |
| `rspin.py`, `taut_m0n.py`, `closed_fjrw.py` | closed FJRW input and tautological intersections |
| `verify_thm05.py`, `verify_thm05_full.sage` | narrow and full open-topological-recursion checks |
| `scattering.py` | Hamiltonian bracket and leading BCH calculation |

## References

- M. Gross, T. L. Kelly, and R. J. Tessler, [*Open FJRW theory and mirror
  symmetry*](https://arxiv.org/abs/2203.02435).
- M. Gross, T. L. Kelly, and R. J. Tessler, [*Open enumerative geometries for
  Landau--Ginzburg models*](https://arxiv.org/abs/2602.12707).
- R. Maher, *Predictions in open Fan--Jarvis--Ruan--Witten theory via mirror
  symmetry, modularity, and wall-crossing*, Ph.D. thesis, University of
  Birmingham, 2024.

## License

The manuscript sources and PDFs are licensed under CC BY 4.0. The verification
code is licensed under the MIT License. See [`LICENSE`](LICENSE).
