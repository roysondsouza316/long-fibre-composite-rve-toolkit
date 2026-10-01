# NonlinearRVE.jl

Nonlinear RVE solver on [Ferrite.jl](https://ferrite-fem.github.io/): small-strain J2
plasticity in the matrix and fibres and zero-thickness cohesive fibre/matrix interfaces with
[DiffCohesive.jl](../DiffCohesive) laws, under periodic (or affine) boundary conditions with
mixed macro strain/stress control, for 2D plane strain, 2D generalized plane strain and 3D
solids. It is the Julia backend of the rve2d nonlinear solve (`nonlinear.backend: julia`)
and the counterpart of the Python `rve2d.nonlinear` package: same interface insertion,
constraints, Newton strategy and output files. On the repository examples the two give
identical increments and Newton iterations and macro stresses equal to 1e-14.

| File | Content |
|---|---|
| `src/mesh.jl` | read the bridge's mesh arrays, orient cells, duplicate interface nodes, build oriented cohesive elements |
| `src/constraints.jl` | periodic node matching per interface side (or zero boundary fluctuation), rigid-body anchor |
| `src/material.jl` | J2 radial return and consistent tangent (Tensors.jl) |
| `src/system.jl` | Ferrite `DofHandler`/`CellValues` bulk assembly, cohesive elements, bordered tangent |
| `src/solver.jl` | incremental Newton with line search and increment cutting |
| `src/output.jl`, `src/driver.jl` | response CSV, VTU (Ferrite `VTKGridFile`, WriteVTK), TOML entry point |

```bash
julia --project=julia/NonlinearRVE -e 'using Pkg; Pkg.instantiate()'   # Julia 1.11+
julia --project=julia/NonlinearRVE julia/ferrite_nonlinear_rve.jl INPUT.toml
julia --project=julia/NonlinearRVE -e 'using Pkg; Pkg.test()'
```

`INPUT.toml` is written by `rve2d.nonlinear.julia_bridge.write_julia_input`; normally the
rve2d CLI writes it and runs the solver for you. The tests (structured meshes, no gmsh)
cover interface insertion, patch tests with stiff interfaces in plane strain, generalized
plane strain and 3D, the analytical elastic-plastic uniaxial curve, the global tangent
against finite differences and debonding.
