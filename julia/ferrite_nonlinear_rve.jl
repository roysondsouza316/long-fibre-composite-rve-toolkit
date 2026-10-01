# Nonlinear RVE solve on Ferrite.jl (J2 plasticity + cohesive fibre/matrix interfaces with
# DiffCohesive.jl laws) for the inputs written by the rve2d bridge
# (`nonlinear.backend: julia`):
#
#     julia --project=julia/NonlinearRVE julia/ferrite_nonlinear_rve.jl INPUT.toml
#
# The first run instantiates the NonlinearRVE environment (Ferrite, DiffCohesive, ...).
import Pkg

try
    using NonlinearRVE
catch
    Pkg.instantiate()
    using NonlinearRVE
end

exit(NonlinearRVE.main(ARGS))
