using Statistics
using CUDA
using CUDA.CUFFT
using DifferentialEquations
using Plots
using LinearAlgebra

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

CUDA.allowscalar(false)

nx = 166; # number of cells on x direction
ny = 42;
nz = 1;
dd = 3; # cell volume = dd x dd x dd
dt = 5E-6; # timestep in nanoseconds
# dt = CuArray(dt); # copy data from CPU to GPU
timesteps = 150000;
alpha = 0.5; # damping constant to relax system to S-state
# alpha = CuArray(alpha);
exchConstant = 1.3E-11 * 1E18; # nanometer/nanosecond units
mu_0 = 1.256636; # vacuum permeability, = 4 * pi / 10
Ms = 800; # saturation magnetization
# Ms = CuArray(Ms);
exch = 2 * exchConstant / mu_0 / Ms / Ms;
# exch = CuArray(exch);
prefactor1 = (-0.221) * dt / (1 + alpha * alpha);
prefactor2 = prefactor1 * alpha / Ms;
# prefactor1 = CuArray(prefactor1);
# prefactor2 = CuArray(prefactor2);
Mx = ones(nx, ny, nz) .* Ms; # magnetization on x direction 
My = zeros(nx, ny, nz); # magnetization on y direction
Mz = zeros(nx, ny, nz); # magnetization on z direction

Mx = CuArray(Mx);
My = CuArray(My);
Mz = CuArray(Mz);

deltaMx = CUDA.zeros(nx, ny, nz);
deltaMy = CUDA.zeros(nx, ny, nz);
deltaMz = CUDA.zeros(nx, ny, nz);
mag = CUDA.zeros(nx, ny, nz);

Mx_pad = CUDA.zeros(Float64, nx * 2, ny * 2, nz * 2);
My_pad = CUDA.zeros(Float64, nx * 2, ny * 2, nz * 2);
Mz_pad = CUDA.zeros(Float64, nx * 2, ny * 2, nz * 2);

Mx_fft = CUDA.zeros(ComplexF64, nx + 1, ny * 2, nz * 2);
My_fft = CUDA.zeros(ComplexF64, nx + 1, ny * 2, nz * 2)
Mz_fft = CUDA.zeros(ComplexF64, nx + 1, ny * 2, nz * 2)

Hx_pad = CUDA.zeros(Float64, nx * 2, ny * 2, nz * 2);
Hy_pad = CUDA.zeros(Float64, nx * 2, ny * 2, nz * 2);
Hz_pad = CUDA.zeros(Float64, nx * 2, ny * 2, nz * 2);

MxHx = CUDA.zeros(Float64, nx, ny, nz);
MxHy = CUDA.zeros(Float64, nx, ny, nz);
MxHz = CUDA.zeros(Float64, nx, ny, nz);

Kxx = zeros(Float64, nx * 2, ny * 2, nz * 2); # Initialization of demagnetization tensor
Kxy = zeros(Float64, nx * 2, ny * 2, nz * 2);
Kxz = zeros(Float64, nx * 2, ny * 2, nz * 2);
Kyy = zeros(Float64, nx * 2, ny * 2, nz * 2);
Kyz = zeros(Float64, nx * 2, ny * 2, nz * 2);
Kzz = zeros(Float64, nx * 2, ny * 2, nz * 2);
prefactor = 1 / 4 / 3.14159265;

include("Demag.jl")
include("Exchange.jl")


H_exch = CUDA.zeros(3, nx, ny, nz);

H0 = CUDA.zeros(3, nx, ny, nz);
H1 = CUDA.zeros(3, nx, ny, nz);
H2 = CUDA.zeros(3, nx, ny, nz);
H3 = CUDA.zeros(3, nx, ny, nz);

function LLG_loop!(dm, m0, p, t)

    global alpha
    global prefactor1
    global prefactor2
    global mag

    Mx = @views m0[1, :, :, :]
    My = @views m0[2, :, :, :]
    Mz = @views m0[3, :, :, :]

    fill!(Mx_pad, 0)
    fill!(My_pad, 0)
    fill!(Mz_pad, 0)

    fill!(H_exch, 0)

    Mx_pad[1:nx, 1:ny, 1:nz] .= Mx
    My_pad[1:nx, 1:ny, 1:nz] .= My
    Mz_pad[1:nx, 1:ny, 1:nz] .= Mz

    mul!(Mx_fft, plan, Mx_pad)
    mul!(My_fft, plan, My_pad)
    mul!(Mz_fft, plan, Mz_pad)

    mul!(Hx_pad, iplan,
        Mx_fft .* Kxx_fft .+
        My_fft .* Kxy_fft .+
        Mz_fft .* Kxz_fft
    ) # calc demag field with fft

    mul!(Hy_pad, iplan,
        Mx_fft .* Kxy_fft .+
        My_fft .* Kyy_fft .+
        Mz_fft .* Kyz_fft
    )

    mul!(Hz_pad, iplan,
        Mx_fft .* Kxz_fft .+
        My_fft .* Kyz_fft .+
        Mz_fft .* Kzz_fft
    )


    Hx = real(Hx_pad[nx:(2*nx-1), ny:(2*ny-1), nz:(2*nz-1)]) # truncation of demag field
    Hy = real(Hy_pad[nx:(2*nx-1), ny:(2*ny-1), nz:(2*nz-1)])
    Hz = real(Hz_pad[nx:(2*nx-1), ny:(2*ny-1), nz:(2*nz-1)])

    Exchange!(H_exch, m0, exch, H0, H1, H2, H3, dd)
    Hx_exch = @view H_exch[1, :, :, :]
    Hy_exch = @view H_exch[2, :, :, :]
    Hz_exch = @view H_exch[3, :, :, :]

    Hx += Hx_exch
    Hy += Hy_exch
    Hz += Hz_exch

    if t < 4000
        Hx .+= 100 # apply a saturation field to get S-state
        Hy .+= 100
        Hz .+= 100
    elseif t < 6000
        Hx .+= (6000 - t) / 20 # gradually diminish the field
        Hx .+= (6000 - t) / 20
        Hx .+= (6000 - t) / 20
    elseif t > 50000
        Hx .+= -19.576 # apply the reverse field
        Hy .+= +3.422
        alpha = 0.02
        prefactor1 = (-0.221) * dt / (1 + alpha * alpha)
        prefactor2 = prefactor1 * alpha / Ms
    end
    # apply LLG equation
    @. MxHx = My * Hz - Mz * Hy # = M cross H
    @. MxHy = Mz * Hx - Mx * Hz
    @. MxHz = Mx * Hy - My * Hx

    @. dm[1, :, :, :] = prefactor1 * MxHx + prefactor2 * (My * MxHz - Mz * MxHy)
    @. dm[2, :, :, :] = prefactor1 * MxHy + prefactor2 * (Mz * MxHx - Mx * MxHz)
    @. dm[3, :, :, :] = prefactor1 * MxHz + prefactor2 * (Mx * MxHy - My * MxHx)

end
# close(outFile);

end_point = 200000
tspan = (0, end_point)
t_range = range(0, end_point, length=300)

m0 = CUDA.zeros(3, nx, ny, nz)
m0[1, :, :, :] .= Mx
m0[2, :, :, :] .= My
m0[3, :, :, :] .= Mz

p = ()


prob = ODEProblem(LLG_loop!, m0, tspan, p)
# @profview sol = solve(prob, OwrenZen3(), progress=true, progress_steps=100)
# @profview sol = solve(prob, OwrenZen3(), progress=true, progress_steps=100)
sol = solve(prob, BS3(), progress=true, progress_steps=200)



# The '...' is absolutely necessary here. It's called splatting and I don't know 
# how it works.
cpu_sol = cat([Array(x) for x in sol.u]..., dims=5)

mx_vals = cpu_sol[1, 1:nx, 1:ny, 1:nz, :]
my_vals = cpu_sol[2, 1:nx, 1:ny, 1:nz, :]
mz_vals = cpu_sol[3, 1:nx, 1:ny, 1:nz, :]

mx_avg = mean(mx_vals, dims=[1, 2, 3])[1, 1, 1, :]
my_avg = mean(my_vals, dims=[1, 2, 3])[1, 1, 1, :]
mz_avg = mean(mz_vals, dims=[1, 2, 3])[1, 1, 1, :]

m_norm = sqrt.(mx_avg .^ 2 + my_avg .^ 2 + mz_avg .^ 2)

plot(sol.t, mx_avg, label="mx")
plot!(sol.t, my_avg, label="my", color="black")
plot!(sol.t, mz_avg, label="mz")
plot!(sol.t, m_norm, label="norm")