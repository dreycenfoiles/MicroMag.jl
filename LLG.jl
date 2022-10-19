using Statistics
using CUDA
using CUDA.CUFFT
using DifferentialEquations
using Plots
using LinearAlgebra
using PreallocationTools
using Memoize
using LoopVectorization

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

const mu_0 = pi * 4e-7 / 1e9; # vacuum permeability, = 4 * pi / 10
const gamma = 2.221e5

M_pad = CUDA.zeros(Float32, 3, nx * 2, ny * 2, nz * 2);

M_fft = CUDA.zeros(ComplexF32, 3, nx + 1, ny * 2, nz * 2);

H_demag_fft = CUDA.zeros(ComplexF32, 3, nx + 1, ny * 2, nz * 2)

plan = plan_rfft(M_pad, [2, 3, 4]);
iplan = plan_irfft(M_fft, 2 * nx, [2, 3, 4]);

H_demag = CUDA.zeros(Float32, 3, nx * 2, ny * 2, nz * 2);

H_eff = CUDA.zeros(Float32, 3, nx, ny, nz);

M_x_H = CUDA.zeros(Float32, 3, nx, ny, nz);


include("Demag.jl")
include("Exchange.jl")

Kxx_fft, Kyy_fft, Kzz_fft, Kxy_fft, Kxz_fft, Kyz_fft = Demag_Kernel(nx, ny, nz, dx, dy, dz)


H_exch = CUDA.zeros(3, nx, ny, nz);

function LLG_loop!(dm, m0, p, t)

    Hx_eff = @view H_eff[1, :, :, :]
    Hy_eff = @view H_eff[2, :, :, :]
    Hz_eff = @view H_eff[3, :, :, :]

    Ms, A, alpha = p

    prefactor1 = -gamma / (1 + alpha * alpha)
    prefactor2 = prefactor1 * alpha
    exch = 2 * A / mu_0 / Ms

    #################
    ## Demag Field ##
    #################

    fill!(M_pad, 0)

    ind = CartesianIndices(m0)

    @inbounds M_pad[ind] = m0 .* Ms

    mul!(M_fft, plan, M_pad)

    Mx_fft = @view M_fft[1, :, :, :]
    My_fft = @view M_fft[2, :, :, :]
    Mz_fft = @view M_fft[3, :, :, :]

    @. @views H_demag_fft[1, :, :, :] = Mx_fft * Kxx_fft + My_fft * Kxy_fft + Mz_fft * Kxz_fft # calc demag field with fft
    @. @views H_demag_fft[2, :, :, :] = Mx_fft * Kxy_fft + My_fft * Kyy_fft + Mz_fft * Kyz_fft
    @. @views H_demag_fft[3, :, :, :] = Mx_fft * Kxz_fft + My_fft * Kyz_fft + Mz_fft * Kzz_fft

    mul!(H_demag, iplan, H_demag_fft)

    Hx_demag = @view H_demag[1, :, :, :]
    Hy_demag = @view H_demag[2, :, :, :]
    Hz_demag = @view H_demag[3, :, :, :]

    @inbounds @views Hx_eff .= real(Hx_demag[nx:(2*nx-1), ny:(2*ny-1), nz:(2*nz-1)]) # truncation of demag field
    @inbounds @views Hy_eff .= real(Hy_demag[nx:(2*nx-1), ny:(2*ny-1), nz:(2*nz-1)])
    @inbounds @views Hz_eff .= real(Hz_demag[nx:(2*nx-1), ny:(2*ny-1), nz:(2*nz-1)])

    ####################
    ## Exchange Field ##
    ####################

    Exchange!(H_exch, m0, exch, dx, dy, dz)

    Hx_eff .+= @views H_exch[1, :, :, :]
    Hy_eff .+= @views H_exch[2, :, :, :]
    Hz_eff .+= @views H_exch[3, :, :, :]

    if t < 0.05
        Hx_eff .+= 0.1 / 1e18 / mu_0 # apply a saturation field to get S-state
        Hy_eff .+= 0.1 / 1e18 / mu_0
        Hz_eff .+= 0.1 / 1e18 / mu_0
    elseif t > 0.25
        Hx_eff .+= -24.6e-3 / 1e18 / mu_0 # apply the reverse field
        Hy_eff .+= +4.3e-3 / 1e18 / mu_0
        alpha = 0.02
        prefactor1 = -gamma / (1 + alpha * alpha)
        prefactor2 = prefactor1 * alpha
    end

    # apply LLG equation
    cross!(M_x_H, m0, H_eff)
    cross!(dm, m0, M_x_H)

    dm .*= prefactor2
    @. dm += prefactor1 * M_x_H

    nothing

end


function cross!(product, A, B)

    Ax = @views A[1, :, :, :]
    Ay = @views A[2, :, :, :]
    Az = @views A[3, :, :, :]

    Bx = @views B[1, :, :, :]
    By = @views B[2, :, :, :]
    Bz = @views B[3, :, :, :]

    @. @views product[1, :, :, :] = Ay * Bz - Az * By
    @. @views product[2, :, :, :] = Az * Bx - Ax * Bz
    @. @views product[3, :, :, :] = Ax * By - Ay * Bx
end

function check_normalize!(m)
    nc, nx, ny, nz = size(m)

    @simd for index in CartesianIndices((nx, ny, nz))
        current_m = @views m[:, index]
        if norm(current_m) != 1
            normalize!(current_m)
        end

    end
end


end_point = 1
tspan = (0, end_point)
t_points = range(0, end_point, length=200)

alpha = 0.5; # damping constant to relax system to S-state
A = 1.3E-11 / 1e9; # nanometer/nanosecond units
Ms = 8e5 / 1e9; # saturation magnetization
p = (Ms, A, alpha)

m0 = CUDA.zeros(Float32, 3, nx, ny, nz)
m0[1, :, :, :] .= 1
m0[2, :, :, :] .= 1
m0[3, :, :, :] .= 1
check_normalize!(m0)

function Relax(m0)
    prob = SteadyStateProblem(LLG_loop!, m0, p)
    # saveat=2000 if memory becomes an issue
    sol = solve(prob, DynamicSS(OwrenZen3()), abstol=1e-2, reltol=1e-2)
    return sol
end


prob = ODEProblem(LLG_loop!, m0, tspan, p);
# saveat=2000 if memory becomes an issue
sol = solve(prob, OwrenZen3(), progress=true, progress_steps=500, abstol=1e-3, reltol=1e-3, saveat=t_points);

# sol = solve(prob, OwrenZen3(), progress=true, progress_steps=500, abstol=1e-3, reltol=1e-3, callback=SS, saveat=2000);

# @profview solve(prob, BS3(), progress=true, progress_steps=500, abstol=1e-10);
# @btime solve(prob, OwrenZen3(), progress=true, progress_steps=500, abstol=1e-3, reltol=1e-3);


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