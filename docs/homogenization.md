# Homogenization notes

The Ferrite.jl bridge runs a unit-cell linear-elastic homogenization on the
generated mesh. It returns the **homogenized stiffness** $\bar{C}$ in Voigt
notation along with derived **engineering constants**.

## What the solver does

For each independent macro strain $\bar{\varepsilon}^{(j)}$ (3 cases in 2D, 6
in 3D), the solver:

1. Builds a per-cell constitutive matrix `D` (matrix or fibre, optionally
   rotated by the configured angles).
2. Assembles $K \, u = -B^\top D \, \bar{\varepsilon}^{(j)}$ on a constant-strain
   triangle (2D) or tetrahedron (3D) mesh.
3. Applies the requested boundary condition (Dirichlet or strong periodic
   constraints with one anchor node in 3D).
4. Computes the volume-averaged stress $\bar{\sigma}^{(j)}$ over the unit cell.

The j-th column of the homogenized stiffness matrix is then
$\bar{C}_{:, j} = \bar{\sigma}^{(j)}$.

## Voigt convention

**2D (3 components):** $(\varepsilon_{xx},\, \varepsilon_{yy},\, 2\varepsilon_{xy})$
and $(\sigma_{xx},\, \sigma_{yy},\, \sigma_{xy})$.

**3D (6 components):** $(\varepsilon_{xx},\, \varepsilon_{yy},\, \varepsilon_{zz},\, 2\varepsilon_{yz},\, 2\varepsilon_{xz},\, 2\varepsilon_{xy})$
and $(\sigma_{xx},\, \sigma_{yy},\, \sigma_{zz},\, \sigma_{yz},\, \sigma_{xz},\, \sigma_{xy})$.

## Engineering constants

Computed from $\bar{S} = \bar{C}^{-1}$:

**2D:**
$$E_x = 1/\bar{S}_{11},\ E_y = 1/\bar{S}_{22},\ G_{xy} = 1/\bar{S}_{33},\ \nu_{xy} = -\bar{S}_{12}/\bar{S}_{22}.$$

**3D:**
$$E_x = 1/\bar{S}_{11},\ E_y = 1/\bar{S}_{22},\ E_z = 1/\bar{S}_{33},$$
$$G_{yz} = 1/\bar{S}_{44},\ G_{xz} = 1/\bar{S}_{55},\ G_{xy} = 1/\bar{S}_{66},$$
$$\nu_{ij} = -\bar{S}_{ij} \cdot E_i.$$

These are reported in `engineering_constants.csv` and inside
`ferrite_homogenization_summary.json`.

## Boundary conditions

- `dirichlet`: zero displacement on all outer boundaries; the resulting
  $\bar{C}$ is a **stiff upper bound** because outer boundaries are clamped.
  Useful for quick sanity checks but not physically homogenizing.
- `periodic`: strong periodic constraints between matched boundary node
  pairs, with one anchor in 3D to remove rigid-body translation. Requires
  `periodic_compatible: true` in the geometry config so gmsh emits a
  node-matched periodic mesh.

## Numerical limits

- Elements are constant-strain (linear Lagrange); for accurate stress fields
  consider denser meshes or higher-order elements (not currently shipped).
- Cholesky is used on the symmetric system; this assumes a positive-definite
  $K$ (true for valid material constants and a well-constrained problem).
- The 3D `solid` orthotropic compliance is built from
  $E_1, E_2, E_3, G_{12}, G_{13}, G_{23}, \nu_{12}, \nu_{13}, \nu_{23}$ via
  the standard symmetry $\nu_{ji} = \nu_{ij} E_j / E_i$ and rotated by
  $R_z R_y R_x$ before being inverted to the global stiffness.
