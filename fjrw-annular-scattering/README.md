# Annular scattering and quartic open FJRW descendents

This directory contains the reader-facing source and descriptive PDF for
Paper 3.

- [`fjrw-annular-scattering.pdf`](fjrw-annular-scattering.pdf) is the compiled
  manuscript.
- `main.tex`, `references.tex`, `sections/`, and `figures/` are the complete
  LaTeX source.
- `src/` contains the publication-relevant exact checks for the affine,
  scattering, logarithmic, primary, and descendent calculations.

Build the manuscript from this directory with:

```sh
latexmk -pdf main.tex
```

From the repository root, run the dependency-free final descendent checks
with:

```sh
python3 fjrw-annular-scattering/src/check_intrinsic_factorization_corolla_homotopy.py
python3 fjrw-annular-scattering/src/check_intrinsic_relative_contact_virtual_chain.py
python3 fjrw-annular-scattering/src/check_intrinsic_affine_residual_boundary.py
python3 fjrw-annular-scattering/src/check_intrinsic_cotangent_recursion.py
```

The two Saito-series programs use SymPy and import the shared exact engine in
`src/a_invariants.py`. The remaining `.sage` programs use SageMath:

```sh
.venv/bin/python fjrw-annular-scattering/src/check_marginal_saito_series.py
.venv/bin/python fjrw-annular-scattering/src/check_mixed_primary_saito_slice.py
for f in fjrw-annular-scattering/src/*.sage; do sage "$f"; done
```

The exact programs verify finite algebraic and combinatorial identities. The
geometric PL transversality and boundary theorems are proved in the manuscript,
not delegated to computation.
