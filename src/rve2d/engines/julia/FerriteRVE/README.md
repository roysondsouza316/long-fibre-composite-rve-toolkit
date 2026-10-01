# FerriteRVE.jl

The Julia engine of rve2d: RVE solvers on [Ferrite.jl](https://ferrite-fem.github.io/) for
2D and 3D meshes of long-fibre composites.

- **Linear homogenization** (`solver.engine: julia`): effective stiffness from one sparse
  Cholesky factorization and one solve per unit macro strain, for plane stress, plane
  strain, generalized plane strain (three displacement components on a 2D mesh, full 6×6
  stiffness) and 3D solids, with periodic or zero-boundary-fluctuation (affine) boundary
  conditions.
- **Nonlinear solve** (`nonlinear.engine: julia`): small-strain J2 plasticity in the matrix
  and fibres and zero-thickness cohesive fibre/matrix interfaces with
  [DiffCohesive.jl](../DiffCohesive) laws, under periodic (or affine) boundary conditions
  with mixed macro strain/stress control, for 2D plane strain, 2D generalized plane strain
  and 3D solids.

It is the counterpart of the Python engine (`rve2d.engines.tensormesh`): same constraints,
interface insertion, Newton strategy and output files. Both engines agree to round-off:
the linear effective stiffness to about 1e-16 and, on the nonlinear examples, identical
increments and Newton iterations with macro stresses equal to 1e-14.

| File | Content |
|---|---|
| `src/mesh.jl` | read the mesh arrays written by rve2d, orient cells, duplicate interface nodes, build oriented cohesive elements |
| `src/constraints.jl` | periodic node matching per interface side (or zero boundary fluctuation), interior rigid-body anchor |
| `src/homogenization.jl` | linear homogenization: assembly, factorization, average stresses, face tractions, VTU |
| `src/material.jl` | J2 radial return and consistent tangent (Tensors.jl) |
| `src/system.jl` | Ferrite `DofHandler`/`CellValues` bulk assembly, cohesive elements, bordered tangent |
| `src/solver.jl` | incremental Newton with line search and increment cutting |
| `src/output.jl`, `src/driver.jl` | response CSV, VTU (Ferrite `VTKGridFile`, WriteVTK), TOML entry point |
| `src/precompile.jl` | PrecompileTools workload (both tasks, 2D and 3D) |

rve2d runs it for you (`--engine julia`): it writes the mesh arrays and a TOML input, runs
`run.jl` in this package's environment and reads the TOML result. The environment is set
up on first use, or ahead of time with `rve2d doctor --setup-julia`. By hand, from
`src/rve2d/engines/julia/`:

```bash
julia --project=FerriteRVE -e 'using Pkg; Pkg.instantiate()'   # Julia 1.11+
julia --project=FerriteRVE run.jl INPUT.toml                    # task = "homogenization" or "nonlinear"
julia --project=FerriteRVE -e 'using Pkg; Pkg.test()'
```

The tests (structured meshes, no gmsh) cover interface insertion, patch tests with stiff
interfaces in plane strain, generalized plane strain and 3D, the analytical
elastic-plastic uniaxial curve, the global tangent against finite differences, debonding,
and for the linear homogenization the recovery of a single material's stiffness, symmetry
and the bounds between periodic, affine, Voigt and Reuss estimates.
