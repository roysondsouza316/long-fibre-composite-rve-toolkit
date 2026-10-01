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
| `laminate_stiffness_2d.yaml` | 2D RVE → ply stiffness → stiffness of 7 stacking sequences | 2 s | 4 s |
| `laminate_stiffness_3d.yaml` | the same from a 3D RVE | 2 s | 5 s |
| `laminate_tensile_2d.yaml`   | 2D RVE → ply stiffness and curves → stiffness and tensile test of 5 stacking sequences | 4.7 min | 4.8 min |
| `laminate_tensile_3d.yaml`   | the same from a 3D RVE | 6.4 min | 4.5 min |

Wall times on a 4-core CPU (Julia times include about 4 s of start-up per solve). The
tensile pipelines spend almost all of it in the two nonlinear RVE solves (transverse and
shear curves); the laminates themselves take about 10 s. Both engines give the same
results: ply stiffness to 4e-16, ply curves to 5e-13, laminate results to 4e-15.

## Run

```bash
rve2d laminate examples/pipelines/tensormesh/laminate_stiffness_2d.yaml
rve2d laminate examples/pipelines/julia/laminate_tensile_2d.yaml
```

The stiffness pipelines need gmsh only (and Julia for the `julia/` copies); the tensile
pipelines also need the `nonlinear` extra on the TensorMesh engine (`pip install -e
".[gmsh,nonlinear]"`). Outputs go to `outputs/pipelines/<engine>/<config name>/`.

## Change the stacking sequence

Edit `laminate.stacking_sequences`, e.g. `["[0/90]2s", "[0/±45/90]s", "[0_2/±30]s",
"[(0/90)_2/45]T"]`, and rerun with the ply properties of the first run, so the RVE is not
solved again (about 10 s):

```bash
rve2d laminate examples/pipelines/tensormesh/laminate_tensile_2d.yaml \
    --ply outputs/pipelines/tensormesh/laminate_tensile_2d/ply/ply_properties.json
```

The notation (`s`, `2s`, `T`, `±`, subscripts, groups, odd-symmetric `sb`) is in
[`docs/laminate.md`](../../docs/laminate.md#stacking-sequence-notation).

## Stiffness

`laminates/summary.csv` has one row per stacking sequence: thickness, membrane constants
(`ex`, `ey`, `gxy`, `nuxy`, `nuyx`), flexural constants, `symmetric` / `balanced` and the
through-thickness constants of the 3D effective stiffness (`ez_3d`, `gxz_3d`, ...). Each
laminate folder (`laminates/0_pm45_90-s/`, ...) holds `abd.csv`,
`effective_3d_stiffness.csv` and `constants.json` (all constants, coupling coefficients).

## Stress and elongation

The tensile pipelines add a strain-controlled coupon test of every laminate
(`laminate.tensile_test`):

```yaml
tensile_test:
  direction: x          # or y, or xy (in-plane shear)
  max_strain: 0.03      # negative for compression
  steps: 300
  gauge_length: 150.0   # mm: elongation = strain x gauge length
  width: 25.0           # mm: force = stress x laminate thickness x width
```

- `laminates/<sequence>/tensile_test.csv`: `strain`, `stress`, `elongation`, `force`,
  mid-plane strains, curvatures and failed plies at every step;
- `laminates/<sequence>/tensile_test_summary.json`: initial modulus, peak stress and its
  strain, first transverse and shear damage, first fibre failure;
- `laminates/summary.csv`: `test_peak_stress`, `test_strain_at_peak`,
  `elongation_at_peak`, `force_at_peak` and the strains of the first events;
- `laminates/tensile_tests.png`: the stress–strain curves of all laminates, with the
  elongation on the top axis.

The ply curves come from the nonlinear RVE (matrix plasticity and fibre/matrix debonding,
`nonlinear` section); the fibre strengths `longitudinal_tensile_strength` and
`longitudinal_compressive_strength` are inputs. Keep `transverse_max_strain` and
`shear_max_strain` above the ply strains of the test: a `curve_exceeded_strain` in the
summary means a ply went past the end of its curve (its stress was held at the last value).
