using DelimitedFiles
using Ferrite
using LinearAlgebra
using SparseArrays

function main()
    if length(ARGS) != 19
        error(
            "Usage: ferrite_homogenization_3d.jl NODES CELLS CELL_TAGS OUTPUT_DIR BC "
            * "KINEMATICS Dm Di MATRIX_SET FIBRE_SET MATRIX_ID FIBRE_ID "
            * "XMIN XMAX YMIN YMAX ZMIN ZMAX WRITE_VTK",
        )
    end

    nodes_path = ARGS[1]
    cells_path = ARGS[2]
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
    zmin = parse(Float64, ARGS[17])
    zmax = parse(Float64, ARGS[18])
    write_vtk = lowercase(ARGS[19]) == "true"

    if kinematics != "solid"
        error("3D Ferrite homogenization requires kinematics='solid'.")
    end

    mkpath(output_dir)
    println("STEP=grid_start")
    grid = build_grid(
        nodes_path,
        cells_path,
        cell_tags_path,
        matrix_set,
        fibre_set,
        matrix_phase_id,
        fibre_phase_id,
        xmin,
        xmax,
        ymin,
        ymax,
        zmin,
        zmax,
    )
    println("STEP=grid_done")
    ip = Lagrange{RefTetrahedron, 1}()^3

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
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
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

    average_stresses = Vector{Vector{Float64}}()
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

    homogenized_stiffness = [
        [average_stresses[col][row] for col in 1:length(macro_strains)]
        for row in 1:6
    ]
    matrix_volume_fraction = compute_matrix_volume_fraction(grid, fibre_set)
    if write_vtk
        write_visualization_data(output_dir, solutions, macro_strains)
    end
    println("STEP=vtk_data_done")

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
        println("SIGMA_BAR_", index, "=", join(sigma, ","))
    end
    for (case_index, traction_dict) in enumerate(boundary_tractions)
        for boundary in ["right", "left", "back", "front", "top", "bottom"]
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
    cells_path,
    cell_tags_path,
    matrix_set,
    fibre_set,
    matrix_phase_id,
    fibre_phase_id,
    xmin,
    xmax,
    ymin,
    ymax,
    zmin,
    zmax,
)
    node_table = readdlm(nodes_path, ',', Float64)
    cell_table = Int.(readdlm(cells_path, ',', Float64))
    tag_table = Int.(vec(readdlm(cell_tags_path, ',', Float64)))

    nodes = [Node((node_table[i, 1], node_table[i, 2], node_table[i, 3])) for i in 1:size(node_table, 1)]
    cells = [
        Tetrahedron((cell_table[i, 1], cell_table[i, 2], cell_table[i, 3], cell_table[i, 4]))
        for i in 1:size(cell_table, 1)
    ]
    grid = Grid(cells, nodes)

    addcellset!(grid, fibre_set, Set(findall(==(fibre_phase_id), tag_table)))
    addcellset!(grid, matrix_set, Set(findall(==(matrix_phase_id), tag_table)))

    tol = max(1.0e-9, max(xmax - xmin, ymax - ymin, zmax - zmin) * 1.0e-6)
    addnodeset!(
        grid,
        "anchor",
        x -> abs(x[1] - xmin) <= tol && abs(x[2] - ymin) <= tol && abs(x[3] - zmin) <= tol,
    )
    addfacetset!(grid, "left", x -> abs(x[1] - xmin) <= tol)
    addfacetset!(grid, "right", x -> abs(x[1] - xmax) <= tol)
    addfacetset!(grid, "front", x -> abs(x[2] - ymin) <= tol)
    addfacetset!(grid, "back", x -> abs(x[2] - ymax) <= tol)
    addfacetset!(grid, "bottom", x -> abs(x[3] - zmin) <= tol)
    addfacetset!(grid, "top", x -> abs(x[3] - zmax) <= tol)
    return grid
end

function create_constraint_handler(grid, dh, boundary_condition)
    ch = ConstraintHandler(dh)
    if boundary_condition == "dirichlet"
        dirichlet = Dirichlet(
            :u,
            union(getfacetset.(Ref(grid), ["left", "right", "front", "back", "top", "bottom"])...),
            (x, t) -> [0.0, 0.0, 0.0],
            [1, 2, 3],
        )
        add!(ch, dirichlet)
    elseif boundary_condition == "periodic"
        width = maximum(node.x[1] for node in grid.nodes) - minimum(node.x[1] for node in grid.nodes)
        height = maximum(node.x[2] for node in grid.nodes) - minimum(node.x[2] for node in grid.nodes)
        depth = maximum(node.x[3] for node in grid.nodes) - minimum(node.x[3] for node in grid.nodes)
        periodic_faces = collect_periodic_facets(
            grid,
            "left",
            "right",
            x -> x - Vec{3}((width, 0.0, 0.0)),
        )
        collect_periodic_facets!(
            periodic_faces,
            grid,
            "front",
            "back",
            x -> x - Vec{3}((0.0, height, 0.0)),
        )
        collect_periodic_facets!(
            periodic_faces,
            grid,
            "bottom",
            "top",
            x -> x - Vec{3}((0.0, 0.0, depth)),
        )
        periodic = PeriodicDirichlet(:u, periodic_faces, [1, 2, 3])
        add!(ch, periodic)
        anchor = Dirichlet(:u, getnodeset(grid, "anchor"), (x, t) -> [0.0, 0.0, 0.0], [1, 2, 3])
        add!(ch, anchor)
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
        B, volume = tetrahedron_kinematics(getcoordinates(cell))
        Ke = volume * (B' * D * B)
        fe = hcat([-(volume * (B' * D * εM)) for εM in macro_strains]...)
        cdofs = celldofs(cell)
        assemble!(assembler, cdofs, Ke)
        f[cdofs, :] .+= fe
    end
    return f
end

function compute_stress(dh, u, macro_strain, Em, Ef, fibre_set)
    sigma_bar_integral = zeros(6)
    volume_total = 0.0
    fibre_cells = getcellset(dh.grid, fibre_set)
    for cell in CellIterator(dh)
        D = cellid(cell) in fibre_cells ? Ef : Em
        B, volume = tetrahedron_kinematics(getcoordinates(cell))
        ue = u[celldofs(cell)]
        sigma = D * (macro_strain + B * ue)
        volume_total += volume
        sigma_bar_integral += sigma * volume
    end
    return sigma_bar_integral / volume_total
end

function compute_boundary_tractions(dh, u, macro_strain, Em, Ef, fibre_set)
    boundary_names = ["right", "left", "back", "front", "top", "bottom"]
    traction_integrals = Dict(name => zeros(3) for name in boundary_names)
    boundary_areas = Dict(name => 0.0 for name in boundary_names)
    fibre_cells = getcellset(dh.grid, fibre_set)

    for boundary in boundary_names
        normal = boundary_normal(boundary)
        for facet in getfacetset(dh.grid, boundary)
            cell_index, local_facet = facet.idx
            coordinates = getcoordinates(dh.grid, cell_index)
            D = cell_index in fibre_cells ? Ef : Em
            B, _ = tetrahedron_kinematics(coordinates)
            cell_dofs = celldofs(dh, cell_index)
            ue = u[cell_dofs]
            sigma = D * (macro_strain + B * ue)
            traction = voigt_to_tensor_3d(sigma) * normal
            facet_area = tetrahedron_facet_area(coordinates, local_facet)
            traction_integrals[boundary] .+= traction * facet_area
            boundary_areas[boundary] += facet_area
        end
    end

    return Dict(
        boundary => traction_integrals[boundary] / boundary_areas[boundary]
        for boundary in boundary_names
    )
end

function compute_matrix_volume_fraction(grid, fibre_set)
    total = 0.0
    matrix = 0.0
    fibre_cells = getcellset(grid, fibre_set)
    for cell in CellIterator(grid)
        _, volume = tetrahedron_kinematics(getcoordinates(cell))
        is_matrix = !(cellid(cell) in fibre_cells)
        total += volume
        if is_matrix
            matrix += volume
        end
    end
    return matrix / total
end

function tetrahedron_kinematics(coords)
    C = [
        1.0 coords[1][1] coords[1][2] coords[1][3]
        1.0 coords[2][1] coords[2][2] coords[2][3]
        1.0 coords[3][1] coords[3][2] coords[3][3]
        1.0 coords[4][1] coords[4][2] coords[4][3]
    ]
    invC = inv(C)
    grads = invC[2:4, :]
    volume = abs(det(C)) / 6.0
    B = zeros(6, 12)
    for i in 1:4
        dx = grads[1, i]
        dy = grads[2, i]
        dz = grads[3, i]
        base = 3 * (i - 1)
        B[1, base + 1] = dx
        B[2, base + 2] = dy
        B[3, base + 3] = dz
        B[4, base + 2] = dz
        B[4, base + 3] = dy
        B[5, base + 1] = dz
        B[5, base + 3] = dx
        B[6, base + 1] = dy
        B[6, base + 2] = dx
    end
    return B, volume
end

function tetrahedron_facet_area(coords, local_facet)
    node_triples = ((1, 2, 3), (1, 2, 4), (2, 3, 4), (1, 3, 4))
    node_a, node_b, node_c = node_triples[local_facet]
    a = collect(coords[node_a])
    b = collect(coords[node_b])
    c = collect(coords[node_c])
    return norm(cross(b - a, c - a)) / 2.0
end

function boundary_normal(boundary)
    if boundary == "right"
        return [1.0, 0.0, 0.0]
    elseif boundary == "left"
        return [-1.0, 0.0, 0.0]
    elseif boundary == "back"
        return [0.0, 1.0, 0.0]
    elseif boundary == "front"
        return [0.0, -1.0, 0.0]
    elseif boundary == "top"
        return [0.0, 0.0, 1.0]
    elseif boundary == "bottom"
        return [0.0, 0.0, -1.0]
    end
    error("Unsupported boundary name: $boundary")
end

function voigt_to_tensor_3d(sigma)
    return [
        sigma[1] sigma[6] sigma[5]
        sigma[6] sigma[2] sigma[4]
        sigma[5] sigma[4] sigma[3]
    ]
end

function write_visualization_data(output_dir, solutions, macro_strains)
    for i in 1:length(solutions)
        writedlm(joinpath(output_dir, "ferrite_solution_case_$i.csv"), solutions[i], ',')
        writedlm(joinpath(output_dir, "ferrite_macro_strain_case_$i.csv"), macro_strains[i], ',')
    end
end

main()
