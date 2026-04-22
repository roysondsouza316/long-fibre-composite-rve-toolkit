using Ferrite
using DelimitedFiles
using LinearAlgebra
using SparseArrays

function main()
    if length(ARGS) != 17
        error(
            "Usage: ferrite_homogenization.jl NODES TRIANGLES CELL_TAGS OUTPUT_DIR BC "
            * "KINEMATICS Dm Di MATRIX_SET FIBRE_SET MATRIX_ID FIBRE_ID "
            * "XMIN XMAX YMIN YMAX WRITE_VTK",
        )
    end

    nodes_path = ARGS[1]
    triangles_path = ARGS[2]
    cell_tags_path = ARGS[3]
    output_dir = ARGS[4]
    boundary_condition = ARGS[5]
    kinematics = ARGS[6]
    matrix_matrix_path = ARGS[7]
    fibre_matrix_path = ARGS[8]
    matrix_set = ARGS[9]
    fibre_set = ARGS[10]
    matrix_phase_id = parse(Int, ARGS[11])
    fibre_phase_id = parse(Int, ARGS[12])
    xmin = parse(Float64, ARGS[13])
    xmax = parse(Float64, ARGS[14])
    ymin = parse(Float64, ARGS[15])
    ymax = parse(Float64, ARGS[16])
    write_vtk = lowercase(ARGS[17]) == "true"

    mkpath(output_dir)
    println("STEP=grid_start")
    grid = build_grid(
        nodes_path,
        triangles_path,
        cell_tags_path,
        matrix_set,
        fibre_set,
        matrix_phase_id,
        fibre_phase_id,
        xmin,
        xmax,
        ymin,
        ymax,
    )
    println("STEP=grid_done")
    ip = Lagrange{RefTriangle, 1}()^2

    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)
    println("STEP=dofs_done")

    ch = create_constraint_handler(grid, dh, boundary_condition)
    K = boundary_condition == "periodic" ? allocate_matrix(dh, ch) : allocate_matrix(dh)
    println("STEP=constraints_done")

    Em = readdlm(matrix_matrix_path, ',', Float64)
    Ef = readdlm(fibre_matrix_path, ',', Float64)
    macro_strains = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    rhs = assemble_system!(K, dh, macro_strains, Em, Ef, fibre_set)
    rhsdata = get_rhs_data(ch, K)
    apply!(K, ch)
    println("STEP=assembly_done")

    solutions = Vector{Vector{Float64}}()
    for i in 1:size(rhs, 2)
        rhs_i = copy(@view rhs[:, i])
        apply_rhs!(rhsdata, rhs_i, ch)
        u_i = cholesky(Symmetric(K)) \ rhs_i
        apply!(u_i, ch)
        push!(solutions, u_i)
    end
    println("STEP=solve_done")

    average_stresses = SymmetricTensor{2, 2}[]
    boundary_tractions = Vector{Dict{String, Vector{Float64}}}()
    for i in 1:length(macro_strains)
        sigma_bar = compute_stress(dh, solutions[i], macro_strains[i], Em, Ef, fibre_set)
        push!(average_stresses, sigma_bar)
        push!(
            boundary_tractions,
            compute_boundary_tractions(dh, solutions[i], macro_strains[i], Em, Ef, fibre_set),
        )
    end
    println("STEP=stress_done")

    # Voigt stiffness assembled column-by-column from the three macro-strain load cases:
    # column j of C is the average stress vector produced by macro strain εM_j.
    # Voigt component order is (xx, yy, xy).
    homogenized_stiffness = [
        [average_stresses[col][1, 1] for col in 1:length(macro_strains)],
        [average_stresses[col][2, 2] for col in 1:length(macro_strains)],
        [average_stresses[col][1, 2] for col in 1:length(macro_strains)],
    ]
    matrix_volume_fraction = compute_matrix_volume_fraction(grid, fibre_set)
    if write_vtk
        write_visualization_data(output_dir, solutions, macro_strains)
    end

    println("BOUNDARY_CONDITION=", boundary_condition)
    println("KINEMATICS=", kinematics)
    println("MATRIX_CELLSET=", matrix_set)
    println("FIBRE_CELLSET=", fibre_set)
    println("MATRIX_VOLUME_FRACTION=", matrix_volume_fraction)
    println("FIBRE_VOLUME_FRACTION=", 1.0 - matrix_volume_fraction)
    for (index, row) in enumerate(homogenized_stiffness)
        println("C_ROW_", index, "=", join(row, ","))
    end
    for (index, sigma) in enumerate(average_stresses)
        println("SIGMA_BAR_", index, "=", join([sigma[1, 1], sigma[2, 2], sigma[1, 2]], ","))
    end
    for (case_index, traction_dict) in enumerate(boundary_tractions)
        for boundary in ["right", "left", "top", "bottom"]
            traction = traction_dict[boundary]
            println(
                "TRACTION_",
                case_index,
                "_",
                uppercase(boundary),
                "=",
                join(traction, ","),
            )
        end
    end
    vtk_path = write_vtk ? joinpath(output_dir, "ferrite_homogenization.vtu") : "none"
    println("VTK_PATH=", vtk_path)
end

function build_grid(
    nodes_path,
    triangles_path,
    cell_tags_path,
    matrix_set,
    fibre_set,
    matrix_phase_id,
    fibre_phase_id,
    xmin,
    xmax,
    ymin,
    ymax,
)
    node_table = readdlm(nodes_path, ',', Float64)
    triangle_table = Int.(readdlm(triangles_path, ',', Float64))
    tag_table = Int.(vec(readdlm(cell_tags_path, ',', Float64)))

    nodes = [Node((node_table[i, 1], node_table[i, 2])) for i in 1:size(node_table, 1)]
    cells = [
        Triangle((triangle_table[i, 1], triangle_table[i, 2], triangle_table[i, 3]))
        for i in 1:size(triangle_table, 1)
    ]
    grid = Grid(cells, nodes)

    addcellset!(grid, fibre_set, Set(findall(==(fibre_phase_id), tag_table)))
    addcellset!(grid, matrix_set, Set(findall(==(matrix_phase_id), tag_table)))

    tol = max(1.0e-9, max(xmax - xmin, ymax - ymin) * 1.0e-6)
    addfacetset!(grid, "left", x -> abs(x[1] - xmin) <= tol)
    addfacetset!(grid, "right", x -> abs(x[1] - xmax) <= tol)
    addfacetset!(grid, "bottom", x -> abs(x[2] - ymin) <= tol)
    addfacetset!(grid, "top", x -> abs(x[2] - ymax) <= tol)
    return grid
end

function create_constraint_handler(grid, dh, boundary_condition)
    ch = ConstraintHandler(dh)
    if boundary_condition == "dirichlet"
        dirichlet = Dirichlet(
            :u,
            union(getfacetset.(Ref(grid), ["left", "right", "top", "bottom"])...),
            (x, t) -> [0.0, 0.0],
            [1, 2],
        )
        add!(ch, dirichlet)
    elseif boundary_condition == "periodic"
        width = maximum(node.x[1] for node in grid.nodes) - minimum(node.x[1] for node in grid.nodes)
        height = maximum(node.x[2] for node in grid.nodes) - minimum(node.x[2] for node in grid.nodes)
        periodic_faces = collect_periodic_facets(
            grid,
            "left",
            "right",
            x -> x - Vec{2}((width, 0.0)),
        )
        collect_periodic_facets!(
            periodic_faces,
            grid,
            "bottom",
            "top",
            x -> x - Vec{2}((0.0, height)),
        )
        periodic = PeriodicDirichlet(:u, periodic_faces, [1, 2])
        add!(ch, periodic)
    else
        error("Unsupported boundary condition: $boundary_condition")
    end
    close!(ch)
    update!(ch, 0.0)
    return ch
end

function assemble_system!(K, dh, macro_strains, Em, Ef, fibre_set)
    f = zeros(ndofs(dh), length(macro_strains))
    assembler = start_assemble(K)
    fibre_cells = getcellset(dh.grid, fibre_set)

    for cell in CellIterator(dh)
        D = cellid(cell) in fibre_cells ? Ef : Em
        B, area = triangle_kinematics(getcoordinates(cell))
        Ke = area * (B' * D * B)
        fe = hcat([-(area * (B' * D * εM)) for εM in macro_strains]...)
        cdofs = celldofs(cell)
        assemble!(assembler, cdofs, Ke)
        f[cdofs, :] .+= fe
    end
    return f
end

function compute_stress(dh, u, macro_strain, Em, Ef, fibre_set)
    sigma_bar_integral = zeros(3)
    volume = 0.0
    fibre_cells = getcellset(dh.grid, fibre_set)
    for cell in CellIterator(dh)
        D = cellid(cell) in fibre_cells ? Ef : Em
        B, area = triangle_kinematics(getcoordinates(cell))
        ue = u[celldofs(cell)]
        sigma = D * (macro_strain + B * ue)
        volume += area
        sigma_bar_integral += sigma * area
    end
    sigma_bar = sigma_bar_integral / volume
    return SymmetricTensor{2, 2}([sigma_bar[1] sigma_bar[3]; sigma_bar[3] sigma_bar[2]])
end

function compute_boundary_tractions(dh, u, macro_strain, Em, Ef, fibre_set)
    boundary_names = ["right", "left", "top", "bottom"]
    traction_integrals = Dict(name => zeros(2) for name in boundary_names)
    boundary_lengths = Dict(name => 0.0 for name in boundary_names)
    fibre_cells = getcellset(dh.grid, fibre_set)

    for boundary in boundary_names
        normal = boundary_normal(boundary)
        for facet in getfacetset(dh.grid, boundary)
            cell_index, local_facet = facet.idx
            coordinates = getcoordinates(dh.grid, cell_index)
            D = cell_index in fibre_cells ? Ef : Em
            B, _ = triangle_kinematics(coordinates)
            cell_dofs = celldofs(dh, cell_index)
            ue = u[cell_dofs]
            sigma = D * (macro_strain + B * ue)
            traction = voigt_to_tensor(sigma) * normal
            facet_length = triangle_facet_length(coordinates, local_facet)
            traction_integrals[boundary] .+= traction * facet_length
            boundary_lengths[boundary] += facet_length
        end
    end

    return Dict(
        boundary => traction_integrals[boundary] / boundary_lengths[boundary]
        for boundary in boundary_names
    )
end

function compute_matrix_volume_fraction(grid, fibre_set)
    total = 0.0
    matrix = 0.0
    fibre_cells = getcellset(grid, fibre_set)
    for cell in CellIterator(grid)
        _, area = triangle_kinematics(getcoordinates(cell))
        is_matrix = !(cellid(cell) in fibre_cells)
        total += area
        if is_matrix
            matrix += area
        end
    end
    return matrix / total
end

function triangle_kinematics(coords)
    x1, y1 = coords[1][1], coords[1][2]
    x2, y2 = coords[2][1], coords[2][2]
    x3, y3 = coords[3][1], coords[3][2]
    detJ = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = abs(detJ) / 2.0
    b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
    c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1
    B = (1.0 / detJ) * [
        b1 0.0 b2 0.0 b3 0.0
        0.0 c1 0.0 c2 0.0 c3
        c1 b1 c2 b2 c3 b3
    ]
    return B, area
end

function triangle_facet_length(coords, local_facet)
    node_pairs = ((1, 2), (2, 3), (3, 1))
    node_a, node_b = node_pairs[local_facet]
    ax, ay = coords[node_a][1], coords[node_a][2]
    bx, by = coords[node_b][1], coords[node_b][2]
    return hypot(bx - ax, by - ay)
end

function boundary_normal(boundary)
    if boundary == "right"
        return [1.0, 0.0]
    elseif boundary == "left"
        return [-1.0, 0.0]
    elseif boundary == "top"
        return [0.0, 1.0]
    elseif boundary == "bottom"
        return [0.0, -1.0]
    end
    error("Unsupported boundary name: $boundary")
end

function voigt_to_tensor(sigma)
    return [sigma[1] sigma[3]; sigma[3] sigma[2]]
end

function write_visualization_data(output_dir, solutions, macro_strains)
    for i in 1:length(solutions)
        writedlm(joinpath(output_dir, "ferrite_solution_case_$i.csv"), solutions[i], ',')
        writedlm(joinpath(output_dir, "ferrite_macro_strain_case_$i.csv"), macro_strains[i], ',')
    end
end

main()
