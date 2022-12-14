using MicroMag
using Statistics
using CUDA
using CUDA.CUFFT
using DifferentialEquations
using Plots
using LinearAlgebra
using PreallocationTools
using Memoize


nx = 166 # number of cells on x direction
ny = 42
nz = 1
dx = 3. # cell volume = dd x dd x dd
dy = 3. # cell volume = dd x dd x dd
dz = 3. # cell volume = dd x dd x dd

# spacing = (dx, dy, dz)


# indices = (input_indices, output_indices)

# M_pad = CUDA.zeros(Float32, 3, nx * 2, ny * 2, nz * 2)
# M_fft = CUDA.zeros(ComplexF32, 3, nx + 1, ny * 2, nz * 2)

# H_demag = CUDA.zeros(Float32, 3, nx * 2, ny * 2, nz * 2)
# H_demag_fft = CUDA.zeros(ComplexF32, 3, nx + 1, ny * 2, nz * 2)

# H_eff = CUDA.zeros(Float32, 3, nx, ny, nz)
# M_x_H = CUDA.zeros(Float32, 3, nx, ny, nz)

# arrays = (M_pad, M_fft, H_demag, H_demag_fft, H_eff, M_x_H)

# plan = plan_rfft(M_pad, [2, 3, 4])


# Kxx_fft, Kyy_fft, Kzz_fft, Kxy_fft, Kxz_fft, Kyz_fft = Demag_Kernel(nx, ny, nz, dx, dy, dz)

# Mx_kernels = (Kxx_fft, Kxy_fft, Kxz_fft)
# My_kernels = (Kxy_fft, Kyy_fft, Kyz_fft)
# Mz_kernels = (Kxz_fft, Kyz_fft, Kzz_fft)


alpha = 0.5; # damping constant to relax system to S-state
A = 1.3E-11 / 1e9; # nanometer/nanosecond units
Ms = 8e5 / 1e9; # saturation magnetization

parameters = (Ms, A, alpha)

# p = (arrays, fft_plans, indices, parameters, Mx_kernels, My_kernels, Mz_kernels, spacing)


m0 = CUDA.zeros(Float32, 3, nx, ny, nz)
m0[1, :, :, :] .= 1
m0[2, :, :, :] .= 1
m0[3, :, :, :] .= 1

p = Init_sim(m0, dx, dy, dz, Ms, A, alpha)
@profview t, cpu_sol = Run(m0, 1., p)

# cb = DiscreteCallback(condition, normalize_llg!)

# prob = ODEProblem(LLG_loop!, m0, tspan, p);
# sol = solve(prob, OwrenZen3(), progress=true, progress_steps=500, abstol=1e-3, reltol=1e-3, saveat=t_points);


# # The '...' is absolutely necessary here
# cpu_sol = cat([Array(x) for x in sol.u]..., dims=5)

mx_vals = cpu_sol[1, 1:nx, 1:ny, 1:nz, :]
my_vals = cpu_sol[2, 1:nx, 1:ny, 1:nz, :]
mz_vals = cpu_sol[3, 1:nx, 1:ny, 1:nz, :]

mx_avg = mean(mx_vals, dims=[1, 2, 3])[1, 1, 1, :]
my_avg = mean(my_vals, dims=[1, 2, 3])[1, 1, 1, :]
mz_avg = mean(mz_vals, dims=[1, 2, 3])[1, 1, 1, :]

m_norm = sqrt.(mx_avg .^ 2 + my_avg .^ 2 + mz_avg .^ 2)

plot(t, mx_avg, label="mx")
plot!(t, my_avg, label="my", color="orange")
plot!(t, mz_avg, label="mz")