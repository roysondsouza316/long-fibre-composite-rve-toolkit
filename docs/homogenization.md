# Linear homogenization

The `solver` config section computes the **effective (homogenized) stiffness**
$\bar{C}$ of the RVE in Voigt notation and the **engineering constants** derived from it.
Two engines implement the same formulation and write the same files:

- `engine: python` (default): NumPy/SciPy, vectorised assembly and one sparse LU
  factorization (`src/rve2d/engines/python/homogenization.py`);
- `engine: julia`: Ferrite.jl `DofHandler`/`CellValues` assembly and one sparse Cholesky
  factorization (`FerriteRVE.homogenize` in
  `src/rve2d/engines/julia/FerriteRVE/src/homogenization.jl`).

## Formulation

The displacement is split into a macro part and a fluctuation,
$u = \bar{\varepsilon}\,x + w$. For each unit macro strain $\bar{\varepsilon}^{(j)}$
(one per active Voigt component) the fluctuation solves

$$K\,w^{(j)} = -F\,\bar{\varepsilon}^{(j)},\qquad
K = \sum_e V_e\,B_e^\top D_e B_e,\qquad F = \sum_e V_e\,B_e^\top D_e,$$

on linear triangles (2D) or tetrahedra (3D), with $D_e$ the stiffness of the cell's phase
(optionally rotated). All load cases share one factorization of $K$. The $j$-th column of
the effective stiffness is the volume-averaged stress,

$$\bar{C}_{:,j} = \bar{\sigma}^{(j)} = \frac{1}{V}\sum_e V_e\,D_e\left(\bar{\varepsilon}^{(j)} + B_e w^{(j)}\right).$$

## Kinematics

| `kinematics` | Mesh | Displacement | Active strains | $\bar{C}$ |
|---|---|---|---|---|
| `plane_stress` | 2D | $u_x, u_y$ | $xx, yy, xy$ | 3×3, from the reduced stiffness $Q$ of each phase |
| `plane_strain` | 2D | $u_x, u_y$ | $xx, yy, xy$ ($\varepsilon_{zz} = 0$) | 3×3, rows/columns $xx, yy, xy$ of the 3D stiffness |
| `generalized_plane_strain` | 2D | $u_x, u_y, u_z$ | all six | 6×6 |
| `solid` | 3D | $u_x, u_y, u_z$ | all six | 6×6 |

**Generalized plane strain** is the natural setting for the cross-section of a
unidirectional ply with the fibres along $z$: the fields do not vary along $z$, but all
six macro strains are applied (the fluctuation has an out-of-plane component $w_z(x, y)$
for the longitudinal shear cases). It gives the full 6×6 stiffness, and so the
longitudinal modulus, the longitudinal shear moduli and the true transverse Young's
moduli, from a 2D mesh. It is exact for z-invariant microstructures: on the same
cross-section, a 3D periodic solve of the extruded mesh gives the same $\bar{C}$ to
round-off (tested). Plane strain gives the in-plane block of the generalized-plane-strain
stiffness (also tested).

## Voigt convention

Order $(xx, yy, zz, yz, xz, xy)$ with engineering shear strains,
$(\varepsilon_{xx}, \varepsilon_{yy}, \varepsilon_{zz}, 2\varepsilon_{yz}, 2\varepsilon_{xz}, 2\varepsilon_{xy})$
and $(\sigma_{xx}, \sigma_{yy}, \sigma_{zz}, \sigma_{yz}, \sigma_{xz}, \sigma_{xy})$,
restricted to the active components of the kinematics: $(xx, yy, xy)$ for plane stress and
plane strain. `homogenization_summary.json` lists them as `voigt_components`.

## Engineering constants

From the compliance $\bar{S} = \bar{C}^{-1}$ (indices in the Voigt order above):

**6×6 (`solid`, `generalized_plane_strain`):**

$$E_x = 1/\bar{S}_{11},\ E_y = 1/\bar{S}_{22},\ E_z = 1/\bar{S}_{33},\quad
G_{yz} = 1/\bar{S}_{44},\ G_{xz} = 1/\bar{S}_{55},\ G_{xy} = 1/\bar{S}_{66},$$
$$\nu_{ij} = -\bar{S}_{ij}\,E_i\quad\text{(e.g. } \nu_{xy} = -\bar{S}_{12}/\bar{S}_{11}\text{)}.$$

**Plane stress** (3×3 in $xx, yy, xy$): $E_x = 1/\bar{S}_{11}$, $E_y = 1/\bar{S}_{22}$,
$G_{xy} = 1/\bar{S}_{33}$, $\nu_{xy} = -\bar{S}_{12}/\bar{S}_{11}$,
$\nu_{yx} = -\bar{S}_{12}/\bar{S}_{22}$: the in-plane constants of a thin lamina.

**Plane strain:** the same formulas applied to the plane-strain stiffness give
*plane-strain moduli*, not Young's moduli (for an isotropic material
$E^{ps} = E/(1-\nu^2)$ and $\nu^{ps} = \nu/(1-\nu)$). They are reported as
`ex_plane_strain`, `ey_plane_strain`, `gxy_plane_strain`, `nuxy_plane_strain` and
`nuyx_plane_strain` so they cannot be mistaken for engineering constants. Use
`generalized_plane_strain` for the transverse Young's moduli and Poisson's ratios of a UD
ply.

The constants are written to `engineering_constants.csv` and `homogenization_summary.json`.

## Boundary conditions

- `periodic`: the fluctuation is periodic, $w(x^+) = w(x^-)$ for every node on an image
  face (right, top, back) and its mirror node on the opposite face. Corner and edge nodes
  are chained to a single master, so the constraints never conflict, and one interior node
  is fixed to remove the rigid-body translation (the problem is otherwise singular). This
  needs a node-matched periodic mesh: set `periodic_compatible: true` in the geometry
  section. The result does not depend on which node is fixed.
- `dirichlet`: zero fluctuation on the whole boundary, i.e. the boundary displacement is
  the affine field $\bar{\varepsilon}\,x$ (kinematic uniform boundary conditions). This
  over-constrains the RVE: $\bar{C}$ is an upper bound of the periodic result and
  converges to it as the RVE grows.

For any RVE, in the sense of the strain energy
$\bar{\varepsilon}^\top \bar{C} \bar{\varepsilon}$,

$$C_\text{Reuss} \le \bar{C}_\text{periodic} \le \bar{C}_\text{dirichlet} \le C_\text{Voigt},$$

which the test suite checks in 2D and 3D.

## Materials and rotations

Phases are assigned from the gmsh physical ids (`matrix_phase_id`, `fibre_phase_id`).

- **Isotropic:** `youngs_modulus`, `poisson_ratio`.
- **Orthotropic:** plane stress needs the in-plane constants `e1, e2, g12, nu12`; every
  other kinematics needs all nine 3D constants
  `e1, e2, e3, g12, g13, g23, nu12, nu13, nu23`. The compliance in material axes is
  built with $\nu_{ji} = \nu_{ij} E_j / E_i$ and must be positive definite (checked when
  the config is loaded).
- **Rotations:** `material_angle_deg` is a rotation about $z$;
  `material_angle_x/y/z_deg` rotate the material axes by $R = R_z R_y R_x$, and the
  stiffness becomes $T C T^\top$ with the stress transformation $T$ of $R$. Plane stress
  allows in-plane rotations only. With fibres along $z$ (generalized plane strain, or 3D
  cylinders along $z$), an orthotropic fibre whose axis 1 is the fibre direction needs
  `fibre_material_angle_y_deg: -90` (axis 1 onto $z$).
- **Precedence:** explicit `solver` angles, then the geometry metadata
  `phase_orientation_rotations_deg` (written by the 3D generators), then, for the fibres
  only, the 2D geometry's `orientation_deg` as a rotation about $z$. `rve2d solve` and
  `solve-ferrite` read the metadata from `geometry_summary.json` next to the mesh, so they
  apply the same rotations as `build-and-solve`.

## Verification

All of these run in the test suite (`tests/test_homogenization.py`, structured meshes, no
gmsh needed); the engine-parity tests run when the Julia engine is set up.

| Check | Result (both engines) |
|---|---|
| Single material: $\bar{C}$ equals the material stiffness, every kinematics and BC | max relative error 5e-15 |
| Generalized plane strain vs a 3D periodic solve of the extruded mesh | 1.8e-15 |
| Plane strain vs the in-plane block of generalized plane strain | 1.3e-16 |
| Reuss ≤ periodic ≤ dirichlet ≤ Voigt; $\bar{C}$ symmetric | holds |
| Periodic solution: $u(x^+) - u(x^-) = \bar{\varepsilon}\,(x^+ - x^-)$ at every node pair | holds to 1e-9 |
| Python engine vs Julia engine: $\bar{C}$, tractions, nodal displacements | agree to 1e-10 or better (about 1e-16 on the examples) |

## Performance

Wall time of `rve2d solve` including writing `homogenization.vtu` (median of three runs,
4-core Xeon at 2.1 GHz) on the high-fibre-fraction examples and refined versions of them;
in brackets the Julia engine's time without starting Julia (about 4 s).

| Mesh | Unknowns | Python engine | Julia engine |
|---|---|---|---|
| 2D example (generalized plane strain) | 16,758 | 1.9 s | 5.4 s (0.9 s) |
| 2D refined | 118,152 | 15.5 s | 10.5 s (6.2 s) |
| 3D example | 16,599 | 5.1 s | 6.9 s (3.0 s) |
| 3D refined | 150,258 | 208 s | 43 s (38 s) |

The Python engine factorizes with SciPy's SuperLU (symmetric mode), which is fast for
small and medium meshes; the Julia engine uses CHOLMOD's supernodal Cholesky
factorization, which scales much better on large 3D meshes: prefer `engine: julia` for
large 3D RVEs (5× faster at 150,000 unknowns here, while Python was faster at 17,000).

## Numerical notes

- Elements are linear (constant strain), so stresses are piecewise constant; refine the
  mesh (`mesh.element_size_*`, `elements_per_circle`) for converged local fields. The
  effective stiffness converges much faster than the local stresses.
- The solve needs a positive-definite $K$: valid material constants and either periodic
  constraints with the anchor node or the Dirichlet boundary. A singular system is reported
  as an error rather than returning a meaningless stiffness.
- Linear meshes only (`mesh.mesh_order: 1`, no `recombine`); the config is rejected
  otherwise.
