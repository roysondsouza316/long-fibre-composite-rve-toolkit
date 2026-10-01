# Result files: response CSV (same columns as the Python solver), VTU fields for the bulk
# (Ferrite VTKGridFile) and for the cohesive interface (WriteVTK).

const COMPONENTS = ("xx", "yy", "zz", "yz", "xz", "xy")

function write_response_csv(path::AbstractString, records::Vector{StepRecord})
    header = vcat(["step", "time", "load_factor"], ["e_$c" for c in COMPONENTS],
        ["s_$c" for c in COMPONENTS],
        ["iterations", "max_damage", "debonded_fraction", "mean_damage", "yielded_fraction",
            "mean_eqps_matrix", "mean_eqps_fibre", "work_density", "mean_bulk_damage"])
    open(path, "w") do io
        println(io, join(header, ","))
        for (index, r) in enumerate(records)
            row = Any[index - 1, r.time, r.load_factor, r.macro_strain..., r.macro_stress...,
                r.iterations, r.max_damage, r.damaged_fraction, r.mean_damage,
                r.yielded_fraction, r.mean_eqps_matrix, r.mean_eqps_fibre, r.work_density,
                r.mean_bulk_damage]
            println(io, join(string.(row), ","))
        end
    end
    return path
end

"Total displacement `E x + w` at every node."
function total_displacement(sys::RVESystem{dim}, state::SolverState) where {dim}
    w_full = expand(sys, state.w)
    grad = macro_tensor(state.macro_strain)
    return [Vec{dim}(ntuple(i -> w_full[sys.node_dofs[i, node]] +
                                 sum(grad[i, j] * sys.mesh.points[node][j] for j in 1:dim), dim))
            for node in 1:n_nodes(sys.mesh)]
end

function write_fields(output_dir::AbstractString, stem::AbstractString, sys::RVESystem{dim},
        state::SolverState, evaluation::Evaluation) where {dim}
    mesh = sys.mesh
    displacement = total_displacement(sys, state)
    bulk_path = joinpath(output_dir, stem)
    VTKGridFile(bulk_path, sys.grid) do vtk
        write_node_data(vtk, displacement, "displacement")
        write_cell_data(vtk, Float64.(mesh.phase), "phase")
        write_cell_data(vtk, [p.equivalent_plastic_strain for p in state.plastic],
            "equivalent_plastic_strain")
        write_cell_data(vtk, von_mises.(evaluation.stress), "von_mises")
        write_cell_data(vtk, [bulk_damage(p.equivalent_plastic_strain, m)
                              for (p, m) in zip(state.plastic, sys.materials)], "bulk_damage")
        for (k, name) in enumerate(COMPONENTS)
            i, j = VOIGT_INDEX[k]
            write_cell_data(vtk, [σ[i, j] for σ in evaluation.stress], "stress_$name")
        end
    end
    written = [bulk_path * ".vtu"]
    if n_cohesive(mesh) > 0
        points = zeros(3, n_nodes(mesh))
        for (node, p) in enumerate(mesh.points), i in 1:dim
            points[i, node] = p[i]
        end
        cell_type = dim == 2 ? WriteVTK.VTKCellTypes.VTK_LINE : WriteVTK.VTKCellTypes.VTK_TRIANGLE
        cells = [WriteVTK.MeshCell(cell_type, mesh.cohesive[1:dim, e]) for e in 1:n_cohesive(mesh)]
        u = zeros(3, n_nodes(mesh))
        for (node, d) in enumerate(displacement), i in 1:dim
            u[i, node] = d[i]
        end
        δ = evaluation.delta_local
        nq = size(δ, 1)
        interface_path = joinpath(output_dir, stem * "_interface")
        WriteVTK.vtk_grid(interface_path, points, cells) do vtk
            vtk["displacement", WriteVTK.VTKPointData()] = u
            vtk["damage", WriteVTK.VTKCellData()] = vec(sum(evaluation.damage; dims = 1)) ./ nq
            vtk["max_damage", WriteVTK.VTKCellData()] = vec(maximum(evaluation.damage; dims = 1))
            vtk["normal_opening", WriteVTK.VTKCellData()] =
                [sum(δ[q, e][1] for q in 1:nq) / nq for e in axes(δ, 2)]
            vtk["shear_opening", WriteVTK.VTKCellData()] =
                [sum(norm(δ[q, e][2:end]) for q in 1:nq) / nq for e in axes(δ, 2)]
        end
        push!(written, interface_path * ".vtu")
    end
    return written
end
