# Entry point: run the solve described by a TOML input file (written by the rve2d bridge,
# see rve2d.nonlinear.julia_bridge) and write the result files.

function phase_material(table::AbstractDict)
    return PhaseMaterial(;
        youngs_modulus = Float64(table["youngs_modulus"]),
        poisson_ratio = Float64(table["poisson_ratio"]),
        yield_stress = Float64(get(table, "yield_stress", Inf)),
        hardening_modulus = Float64(get(table, "hardening_modulus", 0.0)),
    )
end

"DiffCohesive law for the `[interface]` table (keys as in the rve2d `nonlinear.interface`)."
function interface_law(table::AbstractDict)
    name = table["law"]
    stiffness = Float64(table["penalty_stiffness"])
    normal = Float64(table["normal_strength"])
    mode_i = Float64(table["mode_i_toughness"])
    viscosity = Float64(get(table, "viscosity", 0.0))
    if name == "bilinear_mixed_mode"
        return BilinearMixedMode(;
            stiffness,
            shear_stiffness = Float64(get(table, "shear_penalty_stiffness", stiffness)),
            normal_strength = normal,
            shear_strength = Float64(get(table, "shear_strength", normal)),
            mode_i_toughness = mode_i,
            mode_ii_toughness = Float64(get(table, "mode_ii_toughness", mode_i)),
            exponent = Float64(get(table, "bk_exponent", 1.45)),
            criterion = Symbol(get(table, "mixed_mode_criterion", "bk")),
            viscosity,
        )
    end
    viscosity > 0 && error("viscous regularisation is available for bilinear_mixed_mode only")
    return shape_law(name; stiffness, strength = normal, toughness = mode_i)
end

function load_path(table::AbstractDict)
    components(keys, values) = [Int(k) => Float64(v) for (k, v) in zip(keys, values)]
    return LoadPath(
        components(table["prescribed_components"], table["prescribed_values"]),
        Int.(table["fixed_zero"]),
        components(table["stress_components"], table["stress_values"]),
        Bool(get(table, "unload", false)),
    )
end

function newton_settings(table::AbstractDict)
    return NewtonSettings(;
        max_iterations = Int(get(table, "max_iterations", 25)),
        tolerance = Float64(get(table, "tolerance", 1.0e-8)),
        steps = Int(get(table, "steps", 40)),
        max_cuts = Int(get(table, "max_cuts", 8)),
        growth = Float64(get(table, "growth", 1.5)),
        line_search = Bool(get(table, "line_search", true)),
    )
end

"""
    run_input(path) -> Dict

Run the solve described by the TOML file at `path`. Writes `nonlinear_response.csv`,
`nonlinear_final.vtu` (+ `_interface.vtu`), optional per-step fields and
`julia_nonlinear_result.toml` to the input's `output_dir`; returns the result table.
"""
function run_input(path::AbstractString; verbose::Bool = true)
    started = time()
    input = TOML.parsefile(path)
    dim = Int(input["dimension"])
    output_dir = input["output_dir"]
    mkpath(output_dir)
    matrix_id = Int(get(input, "matrix_phase_id", 1))
    fibre_id = Int(get(input, "fibre_phase_id", 2))
    interface = get(input, "interface", nothing)

    points, cells, tags = read_rve_arrays(input["nodes"], input["cells"], input["cell_tags"], dim)
    mesh = build_rve_mesh(points, cells, tags, matrix_id, fibre_id;
        cohesive = interface !== nothing)
    matrix, fibre = phase_material(input["matrix"]), phase_material(input["fibre"])
    materials = [phase == fibre_id ? fibre : matrix for phase in mesh.phase]
    law = interface === nothing ? nothing : interface_law(interface)
    integration = interface === nothing ? "nodal" : get(interface, "integration", "nodal")
    sys = RVESystem(mesh, materials, law, input["boundary_condition"], integration)
    load = load_path(input["load"])
    output_every = Int(get(input, "output_every", 0))
    if verbose
        @printf("MESH nodes=%d bulk_cells=%d cohesive=%d unknowns=%d\n", n_nodes(mesh),
            n_cells(mesh), n_cohesive(mesh), n_reduced(sys) + length(load.stress_controlled))
    end
    outcome, field_files = run_solve(sys, load, newton_settings(input["newton"]),
        Float64(input["stress_scale"]), output_dir, output_every, verbose)

    response = write_response_csv(joinpath(output_dir, "nonlinear_response.csv"),
        outcome.records)
    append!(field_files, write_fields(output_dir, "nonlinear_final", sys, outcome.state,
        outcome.last_evaluation))
    result = Dict{String, Any}(
        "completed" => outcome.completed,
        "reached_time" => outcome.reached_time,
        "steps" => length(outcome.records) - 1,
        "newton_iterations" => outcome.total_iterations,
        "increment_cuts" => outcome.cuts,
        "messages" => outcome.messages,
        "nodes" => n_nodes(mesh),
        "bulk_cells" => n_cells(mesh),
        "cohesive_elements" => n_cohesive(mesh),
        "unknowns" => n_reduced(sys) + length(load.stress_controlled),
        "linear_solver" => "umfpack",
        "julia_version" => string(VERSION),
        "runtime_seconds" => round(time() - started; digits = 2),
        "response_csv" => response,
        "field_files" => field_files,
    )
    open(joinpath(output_dir, "julia_nonlinear_result.toml"), "w") do io
        TOML.print(io, result)
    end
    return result
end

# Function barrier: everything below specialises on the concrete system type.
function run_solve(sys::RVESystem, load, settings, stress_scale, output_dir, output_every,
        verbose)
    field_files = String[]
    step_count = Ref(0)
    function on_step(step, state, evaluation)
        step_count[] += 1
        if verbose
            @printf("STEP %d t=%.5f E=%.6g S=%.6g it=%d max_damage=%.3f debonded=%.3f yielded=%.3f\n",
                step_count[], step.time, maximum(abs, step.macro_strain),
                step.macro_stress[argmax(abs.(step.macro_strain))], step.iterations,
                step.max_damage, step.damaged_fraction, step.yielded_fraction)
            flush(stdout)
        end
        if output_every > 0 && step_count[] % output_every == 0
            stem = @sprintf("nonlinear_step_%04d", step_count[])
            append!(field_files, write_fields(output_dir, stem, sys, state, evaluation))
        end
    end
    outcome = solve_load_path(sys, load, settings, initial_state(sys), stress_scale; on_step)
    return outcome, field_files
end

"Command-line entry: `main([input_toml])`; returns the process exit code."
function main(args::Vector{String} = ARGS)
    if length(args) != 1
        println(stderr, "usage: julia --project=julia/NonlinearRVE julia/ferrite_nonlinear_rve.jl INPUT.toml")
        return 2
    end
    result = run_input(args[1])
    println("COMPLETED=", result["completed"])
    return 0
end
