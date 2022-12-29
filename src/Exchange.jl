function neumann_bc(x, n)

    if x == n + 1
        return x - 1
    elseif x == 0
        return x + 1
    else
        return x
    end

end

# TODO: Periodic boundary conditions


function Exchange_kernel!(Heff, M, exch, nx, ny, nz, dx, dy, dz)

    c = blockIdx().x 
    i = blockIdx().y
    j = threadIdx().x
    k = threadIdx().y

    ip1 = neumann_bc(i + 1, nx)
    im1 = neumann_bc(i - 1, nx)

    jp1 = neumann_bc(j + 1, ny)
    jm1 = neumann_bc(j - 1, ny)

    kp1 = neumann_bc(k + 1, nz)
    km1 = neumann_bc(k - 1, nz)

    laplace = (M[c, ip1, j, k] - 2 * M[c, i, j, k] + M[c, im1, j, k]) / (dx*dx) +
                        (M[c, i, jp1, k] - 2 * M[c, i, j, k] + M[c, i, jm1, k]) / (dy*dy) + 
                        (M[c, i, j, kp1] - 2 * M[c, i, j, k] + M[c, i, j, km1]) / (dz*dz)

    laplace *= exch 

    Heff[c, i, j, k] += laplace

    nothing
end 

function Exchange!(Heff::CuArray{Float32, 4}, m::CuArray{Float32, 4}, mesh::Mesh, exch::Float64, t::Float64)

    nx = mesh.nx
    ny = mesh.ny
    nz = mesh.nz

    # FIXME: Will error if nx >= 1024
    @cuda blocks=(3, nx) threads=(ny, nz) Exchange_kernel!(Heff, m, exch, nx, ny, nz, mesh.dx, mesh.dy, mesh.dz)

    nothing 
end




    
