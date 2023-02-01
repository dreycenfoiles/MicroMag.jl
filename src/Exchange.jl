
struct Exchange{T,U<:Int,V<:Float64} <: AbstractField
    ExchangeCoefficient::T
    nx::U
    ny::U
    nz::U
    dx::V
    dy::V
    dz::V
end

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


function Exchange_kernel!(H_eff::T, m::T, nx, ny, nz, dx, dy, dz) where {T<:CuDeviceArray{Float32,3}}

    c = blockIdx().x
    i = blockIdx().y
    j = threadIdx().x

    ip1 = neumann_bc(i + 1, nx)
    im1 = neumann_bc(i - 1, nx)

    jp1 = neumann_bc(j + 1, ny)
    jm1 = neumann_bc(j - 1, ny)

    ∇²m = (m[c, ip1, j] - 2 * m[c, i, j] + m[c, im1, j]) / (dx * dx) +
          (m[c, i, jp1] - 2 * m[c, i, j] + m[c, i, jm1]) / (dy * dy)

    H_eff[c, i, j] += ∇²m

    nothing
end


function Exchange_kernel!(H_eff::T, m::T, nx, ny, nz, dx, dy, dz) where {T<:CuDeviceArray{Float32,4}}

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

    ∇²m = (m[c, ip1, j, k] - 2 * m[c, i, j, k] + m[c, im1, j, k]) / (dx * dx) +
          (m[c, i, jp1, k] - 2 * m[c, i, j, k] + m[c, i, jm1, k]) / (dy * dy) +
          (m[c, i, j, kp1] - 2 * m[c, i, j, k] + m[c, i, j, km1]) / (dz * dz)

    H_eff[c, i, j, k] += ∇²m

    nothing
end

function (exch::Exchange)(H_eff::T, m::T, t) where {T<:CuArray{Float32}}

    nx = exch.nx
    ny = exch.ny
    nz = exch.nz

    dx = exch.dx
    dy = exch.dy
    dz = exch.dz

    # FIXME: Will error if nx >= 1024
    @cuda blocks=(3,nx) threads=ny Exchange_kernel!(H_eff, m, nx, ny, nz, dx, dy, dz)

    H_eff .*= exch.ExchangeCoefficient

    nothing
end


