

check_normalize!(m) = m ./= sqrt.(sum(m .^ 2, dims=1))

function Init_m(mesh::Mesh, init::Vector{Float32})

    if mesh.nz == 1
        m0 = CUDA.zeros(Float32, 3, mesh.nx, mesh.ny)
    else
        m0 = CUDA.zeros(Float32, 3, mesh.nx, mesh.ny, mesh.nz)
    end

    m0[1, ..] .= init[1]
    m0[2, ..] .= init[2]
    m0[3, ..] .= init[3]

    check_normalize!(m0)

    return m0
end

# FIXME: This should be changed for the 2D case
function Init_m(mesh::Mesh, init::Function)

    m0 = zeros(Float32, 3, mesh.nx, mesh.ny, mesh.nz)

    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz

    for i in CartesianIndices((nx, ny, nz))
        m0[:, i] = init(i[1] * dx, i[2] * dy, i[3] * dz)
    end

    m0 = CuArray{Float32}(m0)

    check_normalize!(m0)

    return m0
end