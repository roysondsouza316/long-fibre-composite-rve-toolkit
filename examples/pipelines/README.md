# Laminate pipelines

RVE → ply → laminates with `rve2d laminate`. The two folders hold the same pipelines, one
per engine; only `engine` and the output folder differ, and both give the same results:

```
pipelines/
├── tensormesh/   # engine: tensormesh (NumPy/SciPy; PyTorch + diffcohesive for the curves)
└── julia/        # engine: julia (Ferrite.jl; DiffCohesive.jl for the curves)
```

| Config                       | Pipeline | TensorMesh | Julia |
| ---------------------------- | -------- | ---------- | ----- |
| `laminate_stiffness_2d.yaml` | 2D RVE → ply stiffness → stiffness of 7 stacking sequences | 2 s | 5 s |
| `laminate_stiffness_3d.yaml` | the same from a 3D RVE | 2.5 s | 5 s |
| `laminate_damage_2d.yaml`    | E-glass/epoxy 2D RVE → ply stiffness, curves and strengths → stiffness of 6 stacking sequences, tension and compression tests with 4 damage models | 4.6 min | 3.7 min |
| `laminate_damage_3d.yaml`    | the same from a 3D RVE | 7.1 min | 5.4 min |

Wall times on a 4-core CPU (Julia times include about 4 s of start-up per solve). The
damage pipelines spend most of it in the three nonlinear RVE solves (transverse tension,
transverse compression, shear); the 48 coupon tests take under a minute, which is also the
time of a rerun with `--ply`. Both engines give the same results: ply stiffness to 3e-16,
ply curves to 3e-11, coupon tests to 2e-11.

## Run

```bash
rve2d laminate examples/pipelines/tensormesh/laminate_stiffness_2d.yaml
rve2d laminate examples/pipelines/julia/laminate_damage_2d.yaml
```

The stiffness pipelines need gmsh only (and Julia for the `julia/` copies); the damage
pipelines also need the `nonlinear` extra on the TensorMesh engine (`pip install -e
".[gmsh,nonlinear]"`). Outputs go to `outputs/pipelines/<engine>/<config name>/`.

## Change the stacking sequence, the damage models or the tests

Edit `laminate.stacking_sequences` (e.g. `["[0/90]2s", "[0/±45/90]s", "[0_2/±30]s",
"[(0/90)_2/45]T"]`), `damage_models` or `coupon_tests`, and rerun with the ply properties
of the first run, so the RVE is not solved again:

```bash
rve2d laminate examples/pipelines/tensormesh/laminate_damage_2d.yaml \
    --ply outputs/pipelines/tensormesh/laminate_damage_2d/ply/ply_properties.json
```

The notation (`s`, `2s`, `T`, `±`, subscripts, groups, odd-symmetric `sb`) is in
[`docs/laminate.md`](../../docs/laminate.md#stacking-sequence-notation). Strengths and
fracture energies set in the config replace those stored with the ply, so a measured
`in_plane_shear` strength, say, can be tried without solving the RVE again.

## Stiffness

`laminates/summary.csv` has one row per stacking sequence: thickness, membrane constants
(`ex`, `ey`, `gxy`, `nuxy`, `nuyx`), flexural constants, `symmetric` / `balanced` and the
through-thickness constants of the 3D effective stiffness (`ez_3d`, `gxz_3d`, ...). Each
laminate folder (`laminates/0_pm45_90-s/`, ...) holds `abd.csv`,
`effective_3d_stiffness.csv` and `constants.json` (all constants, coupling coefficients).

## Damage models and coupon tests

The damage pipelines take an E-glass/epoxy ply (the E-glass/LY556 system of the first
World-Wide Failure Exercise, whose fibre, matrix and lamina data are published together)
from the RVE to strain-controlled coupon tests of six laminates:

```yaml
laminate:
  damage_models: [rve_curves, max_stress, hashin, continuum_damage]
  strengths:                    # MPa: Xt, Xc are inputs; Yt, Yc, S12 from the RVE
    longitudinal_tension: 1140.0
    longitudinal_compression: 570.0
  fracture_energies: {fibre_tension: 40.0, fibre_compression: 20.0,
                      matrix_tension: 0.3, matrix_compression: 1.0}
  characteristic_length: 0.5
  coupon_tests:                 # any names; every test runs with every model
    tension:     {direction: x, max_strain: 0.03, steps: 300, gauge_length: 150.0, width: 25.0}
    compression: {direction: x, max_strain: -0.02, steps: 200, gauge_length: 150.0, width: 25.0}
```

The four models, from the simplest:

| Model | Failure | After failure |
|---|---|---|
| `max_stress` | a stress component reaches its strength | ply discount |
| `hashin` | Hashin's four interactive modes | mode-dependent residual stiffness |
| `continuum_damage` | Hashin's modes on the effective stress | linear softening that dissipates the fracture energy over a crack band |
| `rve_curves` | the peaks of the RVE curves | the RVE curves (matrix plasticity, debonding), fibre failure at Xt / Xc |

Outputs:

- `laminates/coupon_tests.csv`: one row per laminate, model and test: initial modulus,
  peak stress and strain, first ply failure (event, strain, stress), first fibre failure,
  elongation and force at the peak, warnings;
- `laminates/<sequence>/<model>/<test>.csv`: `strain`, `stress`, mid-plane strains,
  curvatures, damaged and fibre-failed plies, `elongation`, `force` at every step, and
  `<test>_summary.json` with the events;
- `laminates/coupon_tests_<model>.png` (all laminates, one model) and
  `laminates/<sequence>/coupon_tests.png` (all models, one laminate);
- `pipeline_summary.json`: ply constants, the strengths used and their sources (`input` or
  `rve`), the curves, every laminate and test, and warnings (e.g. an RVE curve that stopped
  at the snap-back after its peak).

What the 2D example gives (3D in brackets) against the WWFE-I lamina data:

| | WWFE-I | RVE, 2D (3D) |
|---|---|---|
| E1, E2, G12 (GPa) | 53.48, 17.7, 5.83 | 50.3, 14.6, 5.51 (50.0, 15.0, 5.55) |
| Yt, Yc, S12 (MPa) | 35, 114, 72 | 37.2, 135.1, 55.6 (41.2, 127.5, 58.5) |

Yt and Yc land within 6–18 %; S12 is about 20 % low, because the RVE's shear strength is
bounded by the matrix's own shear strength (about 57 MPa here) while the measured lamina
value is a third above it. Set `strengths.in_plane_shear: 72.0` (and rerun with `--ply`)
to use the measured value. The RVE curves stop where damage localises after their peaks;
`pipeline_summary.json` lists this as warnings, and `rve_curves` holds the last stress
beyond the end of a curve.

The models, the reference data and what is assumed are described in
[`docs/laminate.md`](../../docs/laminate.md).
