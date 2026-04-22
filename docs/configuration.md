# Configuration reference

Configs are YAML or JSON. Top-level keys:

```yaml
mode: synthetic | image | sem_to_synthetic
dimension: 2 | 3            # default: 2
synthetic: { ... }          # required if mode == synthetic
image: { ... }              # required if mode == image
sem_to_synthetic: { ... }   # required if mode == sem_to_synthetic (2D only)
mesh: { ... }
export: { ... }
solver: { ... }             # optional; runs Ferrite.jl when enabled: true
```

---

## `synthetic`

Random-placement circular (2D) or cylindrical (3D) fibres in a rectangular /
box domain.

| Field                               | Type     | Default  | Notes                                                            |
| ----------------------------------- | -------- | -------- | ---------------------------------------------------------------- |
| `domain_width`                      | float    | required | Domain extent in x                                               |
| `domain_height`                     | float    | required | Domain extent in y                                               |
| `domain_depth`                      | float    | `None`   | Required for 3D                                                  |
| `fibre_radius`                      | float    | required | Fibre cross-section radius                                       |
| `target_volume_fraction`            | float    | required | In `(0, 1)`. Used when `fibre_count` is unset                    |
| `fibre_count`                       | int?     | `None`   | Overrides `target_volume_fraction` when set                      |
| `min_spacing`                       | float    | `0.0`    | Minimum gap between fibre surfaces                               |
| `edge_clearance`                    | float    | `0.0`    | Minimum gap between fibre and domain edge                        |
| `random_seed`                       | int      | `0`      | RNG seed for reproducibility                                     |
| `max_attempts`                      | int      | `100000` | Random-placement attempt cap                                     |
| `periodic_compatible`               | bool     | `false`  | Emit periodic boundary metadata + node-matched mesh              |
| `orientation_deg` *(2D)*            | float    | `0.0`    | In-plane fibre orientation; falls through to solver as `nuxy`    |
| `matrix_orientation_angle_*_deg`    | float?   | `None`   | 3D matrix rotation; piped to Ferrite via geometry metadata       |
| `fibre_orientation_angle_*_deg`     | float?   | `None`   | 3D fibre rotation; piped to Ferrite via geometry metadata        |

---

## `image`

Binary segmented mask → 2D polygons or 3D extruded volumes.

| Field                  | Type     | Default | Notes                                                           |
| ---------------------- | -------- | ------- | --------------------------------------------------------------- |
| `image_path`           | string   | req.    | Path readable by `skimage.io.imread`                            |
| `pixel_size`           | float    | `1.0`   | Physical edge length of one pixel                               |
| `extrusion_depth`      | float    | `None`  | Required for 3D                                                 |
| `threshold`            | float    | `0.5`   | Foreground threshold on grayscale image                         |
| `invert`               | bool     | `false` | Treat dark pixels as foreground when true                       |
| `min_artifact_area_px` | int      | `16`    | Drop connected components smaller than this                     |
| `clear_border`         | bool     | `false` | Remove regions touching the image border                        |
| `simplify_tolerance`   | float    | `1.0`   | Polygon simplification tolerance, in pixels                     |
| `domain_width`         | float?   | `None`  | Override derived domain width                                   |
| `domain_height`        | float?   | `None`  | Override derived domain height                                  |
| `periodic_compatible`  | bool     | `false` | Emit periodic boundary metadata                                 |
| `matrix/fibre_orientation_angle_*_deg` | float? | `None` | Per-phase rotations piped to Ferrite via geometry metadata |

---

## `sem_to_synthetic` (2D only)

Raw SEM micrograph → segment + watershed-split + circle/ellipse fit → clean
synthetic-style RVE.

Key fields (see [`src/rve2d/config.py`](../src/rve2d/config.py) for the full
list and validation rules):

- `image_path`, `pixel_size`
- `smoothing_sigma`, `threshold`, `invert`, `clear_border`
- `separate_touching_fibres`, `separation_min_distance_px`,
  `separation_peak_threshold_px`
- `min_region_area_px`, `min_artifact_area_px`
- `ellipse_axis_ratio_threshold`, `ellipse_sample_points`
- `max_candidate_overlap_fraction`, `drop_boundary_fibres`
- `domain_width`, `domain_height`, `periodic_compatible`

---

## `mesh`

| Field              | Type   | Default |
| ------------------ | ------ | ------- |
| `element_size_min` | float  | `0.01`  |
| `element_size_max` | float  | `0.05`  |
| `algorithm`        | int    | `6`     |
| `mesh_order`       | int    | `1`     |
| `recombine`        | bool   | `false` |
| `verbosity`        | int    | `2`     |

---

## `export`

| Field                 | Type      | Default               |
| --------------------- | --------- | --------------------- |
| `output_dir`          | string    | `outputs`             |
| `basename`            | string    | `rve2d`               |
| `formats`             | string[]  | `["msh", "xdmf"]`     |
| `write_phase_json`    | bool      | `true`                |
| `write_quality_json`  | bool      | `true`                |
| `write_periodic_json` | bool      | `true`                |

---

## `solver` (Ferrite.jl)

Run only when `enabled: true`. Materials default to a sensible PEEK-like
matrix and a glass-fibre-like inclusion; override per phase.

| Field                                      | Type    | Default        | Notes                                                  |
| ------------------------------------------ | ------- | -------------- | ------------------------------------------------------ |
| `enabled`                                  | bool    | `false`        | Master switch                                          |
| `boundary_condition`                       | enum    | `dirichlet`    | `dirichlet` or `periodic`                              |
| `kinematics`                               | enum    | `plane_strain` | `plane_strain`, `plane_stress`, or `solid` (3D req'd) |
| `matrix_material_model` / `fibre_material_model` | enum | `isotropic` | `isotropic` or `orthotropic`                          |
| `matrix_youngs_modulus` / `matrix_poisson_ratio` | float | 3.5e9 / 0.35 | Used for isotropic matrix                            |
| `fibre_youngs_modulus`  / `fibre_poisson_ratio`  | float | 70e9  / 0.20 | Used for isotropic fibre                             |
| `matrix_e1, e2, e3, g12, g13, g23, nu12, nu13, nu23` | float? | `None` | Required for orthotropic matrix (3D needs 3-axis set) |
| `fibre_e1, e2, e3, g12, g13, g23, nu12, nu13, nu23`  | float? | `None` | Required for orthotropic fibre (3D needs 3-axis set)  |
| `matrix_material_angle_deg` / `fibre_material_angle_deg`         | float? | `None` | 2D rotation                          |
| `matrix_material_angle_x/y/z_deg` / `fibre_material_angle_x/y/z_deg` | float? | `None` | 3D rotation; cannot mix with the 2D form |
| `matrix_cellset`, `fibre_cellset`          | string  | `matrix`/`fibre` | Cell-set names handed to Ferrite                    |
| `matrix_phase_id`, `fibre_phase_id`        | int     | `1`/`2`        | gmsh physical ids                                     |
| `write_vtk`                                | bool    | `true`         | Write `.vtu` for ParaView                             |

### Validation rules (enforced by `load_config`)

- Periodic solves require a config marked `periodic_compatible: true`.
- `kinematics: plane_strain` does not currently support orthotropic phases.
- 3D Ferrite solves require `kinematics: solid`.
- Mixing `material_angle_deg` with `material_angle_x/y/z_deg` is rejected.
- Mesh element sizes must be positive and `min <= max`.
