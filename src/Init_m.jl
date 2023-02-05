

check_normalize!(m) = m ./= sqrt.(sum(m .^ 2, dims=1))

function Init_m(mesh::Mesh, init::T) where T<:Vector

    if mesh.nz == 1
        m = CUDA.zeros(Float32, 3, mesh.nx, mesh.ny)
    else
        m = CUDA.zeros(Float32, 3, mesh.nx, mesh.ny, mesh.nz)
    end

    m[1, ..] .= init[1]
    m[2, ..] .= init[2]
    m[3, ..] .= init[3]

    check_normalize!(m)

    return m
end

# FIXME: This should be changed for the 2D case
function Init_m(mesh::Mesh, init::T) where T<:Function

    m = zeros(Float32, 3, mesh.nx, mesh.ny, mesh.nz)

    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz

    for i in CartesianIndices((nx, ny, nz))
        m[:, i] = init(i[1] * dx, i[2] * dy, i[3] * dz)
    end

    m = CuArray{Float32}(m)

    check_normalize!(m)

    return m
end