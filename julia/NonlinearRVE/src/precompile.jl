# Precompilation workload: solve tiny 2D and 3D RVEs through the same entry point the rve2d
# bridge uses, so a fresh `julia` process does not spend its first half-minute compiling.

function workload_mesh(dir::AbstractString, dim::Int; n::Int = 4)
    xs = range(0.0, 1.0; length = n + 1)
    if dim == 2
        points = [(x, y) for y in xs for x in xs]
        cells = NTuple{3, Int}[]
        for j in 0:(n - 1), i in 0:(n - 1)
            a = j * (n + 1) + i + 1
            push!(cells, (a, a + 1, a + n + 2), (a, a + n + 2, a + n + 1))
        end
    else
        points = [(x, y, z) for z in xs for y in xs for x in xs]
        node(i, j, k) = (k * (n + 1) + j) * (n + 1) + i + 1
        cells = NTuple{4, Int}[]
        for k in 0:(n - 1), j in 0:(n - 1), i in 0:(n - 1)
            v = [node(i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1)) for b in 0:7]
            for (first, second) in ((1, 2), (2, 1), (1, 4), (4, 1), (2, 4), (4, 2))
                push!(cells, (v[1], v[first + 1], v[(first | second) + 1], v[8]))
            end
        end
    end
    tags = [all(0.3 < sum(points[c][i] for c in cell) / length(cell) < 0.7 for i in 1:2) ? 2 : 1
            for cell in cells]
    write_rows(path, rows) = open(io -> foreach(r -> println(io, join(r, ",")), rows), path, "w")
    write_rows(joinpath(dir, "nodes.csv"), points)
    write_rows(joinpath(dir, "cells.csv"), cells)
    write_rows(joinpath(dir, "tags.csv"), tags)
    return nothing
end

function workload_input(dir::AbstractString, dim::Int, viscosity::Float64)
    workload_mesh(dir, dim)
    active = dim == 2 ? [1, 2, 3, 6] : collect(1:6)
    input = Dict{String, Any}(
        "dimension" => dim,
        "kinematics" => dim == 2 ? "generalized_plane_strain" : "solid",
        "boundary_condition" => "periodic",
        "nodes" => joinpath(dir, "nodes.csv"),
        "cells" => joinpath(dir, "cells.csv"),
        "cell_tags" => joinpath(dir, "tags.csv"),
        "output_dir" => joinpath(dir, "out"),
        "stress_scale" => 3500.0,
        "matrix" => Dict("youngs_modulus" => 3500.0, "poisson_ratio" => 0.35,
            "yield_stress" => 20.0, "hardening_modulus" => 300.0),
        "fibre" => Dict("youngs_modulus" => 70000.0, "poisson_ratio" => 0.2),
        "interface" => Dict("law" => "bilinear_mixed_mode", "penalty_stiffness" => 1.0e6,
            "normal_strength" => 10.0, "mode_i_toughness" => 0.05, "viscosity" => viscosity),
        "load" => Dict("prescribed_components" => [1], "prescribed_values" => [0.01],
            "fixed_zero" => setdiff(1:6, active),
            "stress_components" => active[2:end], "stress_values" => zeros(length(active) - 1)),
        "newton" => Dict("steps" => 2),
    )
    path = joinpath(dir, "input.toml")
    open(io -> TOML.print(io, input), path, "w")
    return path
end

@setup_workload begin
    @compile_workload begin
        for (dim, viscosity) in ((2, 0.0), (2, 0.01), (3, 0.0))
            mktempdir() do dir
                run_input(workload_input(dir, dim, viscosity); verbose = false)
            end
        end
    end
end
