using Statistics
using CUDA
using CUDA.CUFFT
using DifferentialEquations
using Plots
using LinearAlgebra
using PreallocationTools

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

CUDA.allowscalar(false)

nx = 166; # number of cells on x direction
ny = 42;
nz = 1;
dx = 3; # cell volume = dd x dd x dd
dy = 3; # cell volume = dd x dd x dd
dz = 3; # cell volume = dd x dd x dd

mu_0 = pi*4e-7/1e9; # vacuum permeability, = 4 * pi / 10
dt = 1e-4; # timestep in nanosecond

Mx_pad = CUDA.zeros(Float32, nx * 2, ny * 2, nz * 2);
My_pad = CUDA.zeros(Float32, nx * 2, ny * 2, nz * 2);
Mz_pad = CUDA.zeros(Float32, nx * 2, ny * 2, nz * 2);

Mx_fft = CUDA.zeros(ComplexF32, nx + 1, ny * 2, nz * 2);
My_fft = CUDA.zeros(ComplexF32, nx + 1, ny * 2, nz * 2)
Mz_fft = CUDA.zeros(ComplexF32, nx + 1, ny * 2, nz * 2)

plan = plan_rfft(Mx_pad);
iplan = plan_irfft(Mx_fft, 2 * nx);

Hx_demag = CUDA.zeros(Float32, nx * 2, ny * 2, nz * 2);
Hy_demag = CUDA.zeros(Float32, nx * 2, ny * 2, nz * 2);
Hz_demag = CUDA.zeros(Float32, nx * 2, ny * 2, nz * 2);

H_eff = CUDA.zeros(Float32, 3, nx, ny, nz);

MxHx = CUDA.zeros(Float32, nx, ny, nz);
MxHy = CUDA.zeros(Float32, nx, ny, nz);
MxHz = CUDA.zeros(Float32, nx, ny, nz);


include("Demag.jl")
include("Exchange.jl")

Kxx_fft, Kyy_fft, Kzz_fft, Kxy_fft, Kxz_fft, Kyz_fft = Demag_Kernel(nx, ny, nz, dx, dy, dz)


H_exch = CUDA.zeros(3, nx, ny, nz);

function LLG_loop!(dm, m0, p, t)

    Ms, A, alpha = p

    prefactor1 = -2.221e5 * dt / (1 + alpha * alpha)
    prefactor2 = prefactor1 * alpha / Ms
    exch = 2 * A / mu_0 / Ms / Ms

    Hx_eff = @views H_eff[1, :, :, :]
    Hy_eff = @views H_eff[2, :, :, :]
    Hz_eff = @views H_eff[3, :, :, :]

    #################
    ## Demag Field ##
    #################

    Mx = @views m0[1, :, :, :]
    My = @views m0[2, :, :, :]
    Mz = @views m0[3, :, :, :]

    fill!(Mx_pad, 0)
    fill!(My_pad, 0)
    fill!(Mz_pad, 0)

    Mx_pad[1:nx, 1:ny, 1:nz] = Mx
    My_pad[1:nx, 1:ny, 1:nz] = My
    Mz_pad[1:nx, 1:ny, 1:nz] = Mz

    mul!(Mx_fft, plan, Mx_pad)
    mul!(My_fft, plan, My_pad)
    mul!(Mz_fft, plan, Mz_pad)

    mul!(Hx_demag, iplan,
        Mx_fft .* Kxx_fft .+
        My_fft .* Kxy_fft .+
        Mz_fft .* Kxz_fft
    ) # calc demag field with fft

    mul!(Hy_demag, iplan,
        Mx_fft .* Kxy_fft .+
        My_fft .* Kyy_fft .+
        Mz_fft .* Kyz_fft
    )

    mul!(Hz_demag, iplan,
        Mx_fft .* Kxz_fft .+
        My_fft .* Kyz_fft .+
        Mz_fft .* Kzz_fft
    )

    Hx_eff .= real(Hx_demag[nx:(2*nx-1), ny:(2*ny-1), nz:(2*nz-1)]) # truncation of demag field
    Hy_eff .= real(Hy_demag[nx:(2*nx-1), ny:(2*ny-1), nz:(2*nz-1)])
    Hz_eff .= real(Hz_demag[nx:(2*nx-1), ny:(2*ny-1), nz:(2*nz-1)])

    ####################
    ## Exchange Field ##
    ####################

    Exchange!(H_exch, m0, exch, dx, dy, dz)

    H_eff .+= H_exch

    if t < 500
        Hx_eff .+= .1/1e18/mu_0 # apply a saturation field to get S-state
        Hy_eff .+= .1/1e18/mu_0
        Hz_eff .+= .1/1e18/mu_0
    elseif t > 2500
        Hx_eff .+= -24.6e-3/1e18/mu_0 # apply the reverse field
        Hy_eff .+= +4.3e-3/1e18/mu_0
        alpha = 0.02
        prefactor1 = -2.221e5 * dt / (1 + alpha * alpha)
        prefactor2 = prefactor1 * alpha / Ms
    end
    # apply LLG equation
    @. MxHx = My * Hz_eff - Mz * Hy_eff # = M cross H
    @. MxHy = Mz * Hx_eff - Mx * Hz_eff
    @. MxHz = Mx * Hy_eff - My * Hx_eff

    # @show t * dt

    @. dm[1, :, :, :] = prefactor1 * MxHx + prefactor2 * (My * MxHz - Mz * MxHy)
    @. dm[2, :, :, :] = prefactor1 * MxHy + prefactor2 * (Mz * MxHx - Mx * MxHz)
    @. dm[3, :, :, :] = prefactor1 * MxHz + prefactor2 * (Mx * MxHy - My * MxHx)

    # @show dm[1, :, :, :]
    nothing

end

# function check_normalize!(m,Ms)
#     nc,nx,ny,nz = size(m)

#     for index in CartesianIndices((nx,ny,nz))
#         current_m = m[:, index]
#         if norm

#         end

#     end
# end


end_point = 10000
tspan = (0, end_point)

alpha = 0.5; # damping constant to relax system to S-state
A = 1.3E-11/1e9; # nanometer/nanosecond units
Ms = 8e5/1e9; # saturation magnetization
p = (Ms, A, alpha)

m0 = CUDA.zeros(Float32, 3, nx, ny, nz)
m0[2, :, :, :] .= Ms


prob = ODEProblem(LLG_loop!, m0, tspan, p);
SS = TerminateSteadyState(1e-5, 1e-5)
# saveat=2000 if memory becomes an issue
sol = solve(prob, BS3(), progress=true, progress_steps=500, abstol=1e-10);


# # The '...' is absolutely necessary here. It's called splatting and I don't know 
# # how it works.
cpu_sol = cat([Array(x) for x in sol.u]..., dims=5)

mx_vals = cpu_sol[1, 1:nx, 1:ny, 1:nz, :]
my_vals = cpu_sol[2, 1:nx, 1:ny, 1:nz, :]
mz_vals = cpu_sol[3, 1:nx, 1:ny, 1:nz, :]

mx_avg = mean(mx_vals, dims=[1, 2, 3])[1, 1, 1, :]
my_avg = mean(my_vals, dims=[1, 2, 3])[1, 1, 1, :]
mz_avg = mean(mz_vals, dims=[1, 2, 3])[1, 1, 1, :]

m_norm = sqrt.(mx_avg .^ 2 + my_avg .^ 2 + mz_avg .^ 2)

plot(sol.t, mx_avg, label="mx")
plot!(sol.t, my_avg, label="my", color="orange")
plot!(sol.t, mz_avg, label="mz")
# plot!(sol.t, m_norm, label="norm")