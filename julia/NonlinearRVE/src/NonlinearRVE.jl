"""
    NonlinearRVE

Nonlinear RVE solver on Ferrite.jl: small-strain J2 plasticity in the matrix and the fibres
and zero-thickness cohesive fibre/matrix interfaces with DiffCohesive.jl traction-separation
laws, under periodic (or affine) boundary conditions with mixed macro strain/stress control,
for 2D plane strain, 2D generalized plane strain and 3D solids.

It is the Julia counterpart of the Python `rve2d.nonlinear` solver: same formulation,
interface insertion, constraints, Newton strategy and output files, so the two can be used
interchangeably (`nonlinear.backend: julia` in an rve2d config) and cross-checked.

Run it on the inputs written by the rve2d bridge with

    julia --project=julia/NonlinearRVE julia/ferrite_nonlinear_rve.jl INPUT.toml
"""
module NonlinearRVE

using DelimitedFiles: readdlm
using DiffCohesive: DiffCohesive, BilinearMixedMode, TractionSeparationLaw, shape_law,
    state_length, traction, traction_tangent
using Ferrite
using LinearAlgebra: LinearAlgebra, cross, dot, lu, norm
using PrecompileTools: @compile_workload, @setup_workload
using Printf: @printf, @sprintf
using SparseArrays: SparseMatrixCSC, sparse
using StaticArrays: MMatrix, SMatrix, SVector
using TOML: TOML
using WriteVTK: WriteVTK

export run_input, main

include("mesh.jl")
include("constraints.jl")
include("material.jl")
include("system.jl")
include("solver.jl")
include("output.jl")
include("driver.jl")
include("precompile.jl")

end # module
