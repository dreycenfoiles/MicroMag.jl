using MicroMag
using Statistics
using CUDA
using Plots
using LinearAlgebra
using DataFrames
using CSV 
using EllipsisNotation


nx = 166 # number of cells on x direction
ny = 42
nz = 1
dx = 3. # cell volume = dd x dd x dd
dy = 3. # cell volume = dd x dd x dd
dz = 3. # cell volume = dd x dd x dd

alpha = 0.02; # damping constant to relax system to S-state
A = 1.3E-11 / 1e9; # nanometer/nanosecond units
Ms = 8e5 / 1e9; # saturation magnetization

B_ext = [-24.6e-3, 4.3e-3, 0.0]

init_m = [1f0, .1f0, 0f0]

mesh = Mesh(nx, ny, nz, dx, dy, dz)
params = Params(A=A, Ms=Ms, α=alpha, B_ext=B_ext)

p = InitSim(init_m, mesh, params)

Relax(p)

t, cpu_sol = Run(p, 3.)

mx_vals = cpu_sol[1, ..]
my_vals = cpu_sol[2, ..]
mz_vals = cpu_sol[3, ..]

mx_avg = mean(mx_vals, dims=[1, 2])[1, 1, :]
my_avg = mean(my_vals, dims=[1, 2])[1, 1, :]
mz_avg = mean(mz_vals, dims=[1, 2])[1, 1, :]

plot(t, mx_avg, label="mx")
plot!(t, my_avg, label="my")
plot!(t, mz_avg, label="mz")

mumax3 = CSV.File("Tests/table.txt") |> DataFrame

mumax3_t = mumax3[!, "# t (s)"] * 1e9

mumax3_mx = mumax3[!, "mx ()"]
mumax3_my = mumax3[!, "my ()"]
mumax3_mz = mumax3[!, "mz ()"]

plot!(mumax3_t, mumax3_mx, label="mumax3 mx")
plot!(mumax3_t, mumax3_my, label="mumax3 my")
plot!(mumax3_t, mumax3_mz, label="mumax3 mz")