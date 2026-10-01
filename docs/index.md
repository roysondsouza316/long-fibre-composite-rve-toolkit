# Documentation

- [Configuration reference](configuration.md) — every YAML field with type,
  default, and notes.
- [Linear homogenization](homogenization.md) — formulation, kinematics
  (including generalized plane strain), boundary conditions, the Voigt convention,
  engineering constants and verification.
- [Nonlinear RVE solve](nonlinear.md) — J2 plasticity and cohesive
  fibre/matrix interfaces: configuration, formulation, verification.
- [Examples picker](../examples/README.md) — pick a starting YAML by
  dimension and input mode.

## Typical workflow

1. Write a YAML or JSON config.
2. `rve2d validate-config your_config.yaml`
3. `rve2d build your_config.yaml` (geometry + mesh + metadata)
4. Open the resulting `.msh` or `.xdmf` in your FEM workflow, **or**
5. `rve2d build-and-solve your_config.yaml` to also run the linear
   homogenization (`solver` section) on the Python engine, or on the Julia
   engine (Ferrite.jl) with `engine: julia` / `--engine julia`.
6. Use `rve2d batch-study config_a.yaml config_b.yaml --output-dir
   outputs/study` to aggregate engineering constants across multiple RVEs.
7. Add a `nonlinear` section (and `pip install -e ".[nonlinear]"`) to run a
   plasticity + interface-debonding solve on the same mesh, either within
   `build-and-solve` or with `rve2d solve-nonlinear config.yaml mesh.msh
   --output-dir out/`.

## Notes

- The `gmsh` Python module is required only for meshing.
- Periodic boundary output (`periodic_pairs.json`) is solver-agnostic metadata for
  external solvers; the bundled engines match the periodic nodes of the mesh
  themselves.
- The image pipeline expects a segmented mask where fibre pixels are
  foreground (`invert: true` for dark fibres).
- Visualization output is written as `.vtu` and is consumable by ParaView /
  VisIt.
- `dimension: 3` switches both geometry generation and the solves to the 3D
  tetrahedral path.
- `rve2d doctor` reports what is installed (gmsh, h5py, PyTorch, Julia) and what
  each part is needed for.
