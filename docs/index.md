# Documentation

- [Configuration reference](configuration.md) — every YAML field with type,
  default, and notes.
- [Homogenization notes](homogenization.md) — what the Ferrite.jl solve does,
  the Voigt convention, and how engineering constants are derived.
- [Examples picker](../examples/README.md) — pick a starting YAML by
  dimension and input mode.

## Typical workflow

1. Write a YAML or JSON config.
2. `rve2d validate-config your_config.yaml`
3. `rve2d build your_config.yaml` (geometry + mesh + metadata)
4. Open the resulting `.msh` or `.xdmf` in your FEM workflow, **or**
5. `rve2d build-and-solve your_config.yaml` to also run the Ferrite.jl
   homogenization stage.
6. Use `rve2d batch-study config_a.yaml config_b.yaml --output-dir
   outputs/study` to aggregate engineering constants across multiple RVEs.

## Notes

- The `gmsh` Python module is required only for meshing.
- Periodic boundary output is solver-agnostic metadata, not solver-specific
  constraints. The bundled Ferrite path *does* consume it to build periodic
  constraints.
- The image pipeline assumes a binary segmented mask where fibre pixels are
  foreground.
- Visualization output is written as `.vtu` and is consumable by ParaView /
  VisIt.
- `dimension: 3` switches both geometry generation and the downstream Ferrite
  solver to the 3D tetrahedral path.
