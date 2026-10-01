# Entry point of the rve2d Julia engine (FerriteRVE on Ferrite.jl + DiffCohesive.jl): runs
# the linear homogenization or nonlinear solve described by a TOML input written by the
# rve2d bridge.
#
#     julia --project=<this folder>/FerriteRVE <this folder>/run.jl INPUT.toml
#
# The rve2d bridge instantiates the environment before the first run; if it is missing (for
# example when this script is run by hand), it is instantiated here.
import Pkg

try
    using FerriteRVE
catch
    Pkg.instantiate()
    using FerriteRVE
end

exit(FerriteRVE.main(ARGS))
