"""
    FerriteRVE

The Julia engine of rve2d, on [Ferrite.jl](https://ferrite-fem.github.io/):

- **linear homogenization**: effective stiffness of the RVE from unit macro strains under
  periodic or affine boundary conditions, for 2D plane stress, plane strain and generalized
  plane strain (full 6x6 stiffness of a unidirectional ply from its cross-section) and 3D
  solids, with anisotropic (rotated orthotropic) phases;
- **nonlinear RVE solve**: small-strain J2 plasticity in matrix and fibres and zero-thickness
  cohesive fibre/matrix interfaces with DiffCohesive.jl traction-separation laws, under
  periodic (or affine) boundary conditions with mixed macro strain/stress control.

It mirrors the Python engine (`rve2d.engines.tensormesh`): same formulations, interface
insertion, constraints, Newton strategy and output files, so the two engines are
interchangeable (`engine: julia` in an rve2d config) and give the same results to round-off.
The rve2d bridge writes a TOML input and runs

    julia --project=<rve2d>/engines/julia/FerriteRVE <rve2d>/engines/julia/run.jl INPUT.toml
"""
module FerriteRVE

using DelimitedFiles: readdlm
using DiffCohesive: DiffCohesive, BilinearMixedMode, TractionSeparationLaw, shape_law,
    state_length, traction, traction_tangent
using Ferrite
using LinearAlgebra: LinearAlgebra, Symmetric, cholesky, cross, dot, lu, norm
using PrecompileTools: @compile_workload, @setup_workload
using Printf: @printf, @sprintf
using SparseArrays: SparseMatrixCSC, sparse
using StaticArrays: MMatrix, SMatrix, SVector
using TOML: TOML
using WriteVTK: WriteVTK

export homogenize, run_input, main

include("mesh.jl")
include("constraints.jl")
include("homogenization.jl")
include("material.jl")
include("system.jl")
include("solver.jl")
include("output.jl")
include("driver.jl")
include("precompile.jl")

end # module
