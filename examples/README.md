# Examples

Each example is a complete YAML config that can be passed straight to the
`rve2d` CLI. They are organised so the **dimension** and the **input mode**
are obvious from the path:

```
examples/
├── 2d/
│   ├── synthetic/   # randomly placed circular fibres in a rectangle
│   ├── image/       # binary segmented mask -> 2D polygons
│   └── sem/         # raw SEM micrograph -> mask or fitted circles/ellipses
├── 3d/
│   ├── synthetic/   # randomly placed cylindrical fibres in a box
│   └── image/       # 2D mask extruded into a 3D box
└── pipelines/       # RVE -> ply -> laminates (`rve2d laminate`)
    ├── tensormesh/  # on the TensorMesh engine
    └── julia/       # the same configs on the Julia engine
```

## Pick your starting point

| Goal                                             | Use                                                 |
| ------------------------------------------------ | --------------------------------------------------- |
| Quick 2D smoke test                              | `2d/synthetic/basic.yaml`                           |
| 2D periodic homogenization                       | `2d/synthetic/periodic_solve.yaml`                  |
| 2D rotated orthotropic phase materials           | `2d/synthetic/orthotropic_solve.yaml`               |
| 2D periodic RVE at Vf 0.60 (relaxation packing)  | `2d/synthetic/high_vf_periodic.yaml`                |
| Build 2D RVE from a binary mask                  | `2d/image/basic_solve.yaml`                         |
| Same with rotated orthotropic phase materials    | `2d/image/oriented_solve.yaml`                      |
| Convert a raw SEM image into circular fibres     | `2d/sem/to_synthetic.yaml`                          |
| Just import the SEM micrograph as polygons       | `2d/sem/import.yaml`                                |
| 3D smoke test                                    | `3d/synthetic/basic.yaml`                           |
| 3D periodic homogenization (solid)               | `3d/synthetic/periodic_solve.yaml`                  |
| 3D periodic RVE at Vf 0.60 (relaxation packing)  | `3d/synthetic/high_vf_periodic.yaml`                |
| 3D rotated orthotropic phases (explicit angles)  | `3d/synthetic/orthotropic_solve.yaml`               |
| 3D rotated orthotropic phases (workflow angles)  | `3d/synthetic/workflow_oriented_solve.yaml`         |
| 3D RVE from an extruded 2D mask                  | `3d/image/extruded.yaml`                            |
| Same with periodic homogenization                | `3d/image/extruded_periodic_solve.yaml`             |
| 2D plasticity + fibre/matrix debonding (nonlinear) | `2d/synthetic/nonlinear_cohesive_plastic.yaml`    |
| 3D plasticity + fibre/matrix debonding (nonlinear) | `3d/synthetic/nonlinear_cohesive_plastic.yaml`    |
| Laminate stiffness of several stacking sequences | `pipelines/<engine>/laminate_stiffness_2d.yaml` (or `_3d`) |
| Laminate tensile test: stress, strain, elongation, force | `pipelines/<engine>/laminate_tensile_2d.yaml` (or `_3d`) |

## Run from the repository root

```bash
rve2d validate-config examples/2d/synthetic/basic.yaml
rve2d build           examples/2d/synthetic/basic.yaml
rve2d build-and-solve examples/2d/synthetic/periodic_solve.yaml
rve2d build-and-solve examples/2d/synthetic/periodic_solve.yaml --engine julia   # Ferrite.jl
rve2d build-and-solve examples/2d/synthetic/nonlinear_cohesive_plastic.yaml   # needs .[nonlinear]
rve2d laminate        examples/pipelines/tensormesh/laminate_stiffness_2d.yaml
rve2d laminate        examples/pipelines/julia/laminate_tensile_2d.yaml
```

Every solve runs on the TensorMesh engine unless the config sets `engine: julia` (in the
`solver` or `nonlinear` section) or `--engine julia` is given; both engines give the same
results. The Julia engine needs Julia 1.11+ (`rve2d doctor --setup-julia` prepares it).

The nonlinear examples use mm / MPa / N/mm units. On a 4-core CPU the 2D example solves in
about 40 s on either engine and the 3D example in about 60 s with the TensorMesh engine
and 40 s with the Julia engine. See [`docs/nonlinear.md`](../docs/nonlinear.md).

## Laminate pipelines

`pipelines/tensormesh/` and `pipelines/julia/` hold the same four configs; only `engine`
and the output folder differ, and both give the same results:

| Config | Pipeline |
| ------ | -------- |
| `laminate_stiffness_2d.yaml` | 2D RVE → ply stiffness → ABD matrix, engineering constants and 3D stiffness of 7 stacking sequences |
| `laminate_stiffness_3d.yaml` | the same from a 3D RVE |
| `laminate_tensile_2d.yaml` | 2D RVE → ply stiffness and transverse/shear curves (nonlinear RVE) → stiffness and tensile test of 5 stacking sequences: stress–strain curve, elongation, force |
| `laminate_tensile_3d.yaml` | the same from a 3D RVE |

To try other stacking sequences, edit `stacking_sequences` and rerun with
`--ply OUTPUT/ply/ply_properties.json`: the RVE is not solved again. See
[`docs/laminate.md`](../docs/laminate.md).

`image_path` and `output_dir` in each config are relative to the directory
where you launch `rve2d`, *not* relative to the YAML file. Run from the repo
root to pick up the bundled `examples/2d/image/sample_mask.pgm` automatically.

## SEM examples need your own image

The SEM configs expect a greyscale micrograph at:

```
examples/2d/sem/sem_sample.png
```

Drop your own SEM image there (or change `image_path`). See
[`2d/sem/README.md`](2d/sem/README.md) for details.
