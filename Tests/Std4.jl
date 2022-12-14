using MicroMag
using Statistics
using CUDA
using CUDA.CUFFT
using Plots
using LinearAlgebra
using PreallocationTools

nx = 166 # number of cells on x direction
ny = 42
nz = 1
dx = 3. # cell volume = dd x dd x dd
dy = 3. # cell volume = dd x dd x dd
dz = 3. # cell volume = dd x dd x dd

alpha = 0.5; # damping constant to relax system to S-state
A = 1.3E-11 / 1e9; # nanometer/nanosecond units
Ms = 8e5 / 1e9; # saturation magnetization

parameters = (Ms, A, alpha)

m0 = CUDA.zeros(Float32, 3, nx, ny, nz)
m0[1, :, :, :] .= 1
m0[2, :, :, :] .= 1
m0[3, :, :, :] .= 1

p = Init_sim(m0, dx, dy, dz, Ms, A, alpha)

Relax(m0, p)

t, cpu_sol = Run(m0, 1., p)

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