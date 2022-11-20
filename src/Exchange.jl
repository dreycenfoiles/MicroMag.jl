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


function Exchange_kernel(H_eff, M, exch, dx, dy, dz)

    # TODO: Generalize to 3D

    nc, nx, ny, nz = size(M)

    c = blockIdx().x 
    j = blockIdx().y
    i = threadIdx().x

    ip1 = neumann_bc(i + 1, nx)
    im1 = neumann_bc(i - 1, nx)
    jp1 = neumann_bc(j + 1, ny)
    jm1 = neumann_bc(j - 1, ny)


    laplace = (M[c, ip1, j, 1] - 2 * M[c, i, j, 1] + M[c, im1, j, 1]) / dx / dx +
              (M[c, i, jp1, 1] - 2 * M[c, i, j, 1] + M[c, i, jm1, 1]) / dy / dy

    laplace *= exch 

    H_eff[c, i, j, 1] += laplace

    nothing
end 

function Exchange!(H_eff, M, exch, dx, dy, dz)

    nc, nx, ny, nz = size(M)

    # FIXME: Will error if nx >= 1024
    @cuda blocks=(nc, ny) threads=nx Exchange_kernel(H_eff, M, exch, dx, dy, dz)

    return
end

    
