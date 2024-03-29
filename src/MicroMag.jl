module MicroMag

using Statistics
using CUDA
using CUDA.CUFFT
using DifferentialEquations
using LinearAlgebra
using PreallocationTools
using Memoize
using BenchmarkTools
using EllipsisNotation

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

CUDA.allowscalar(false)

export Relax!
export InitSim
export Run
export Mesh

const μ₀::Float32 = pi * 4e-7 / 1e9 # vacuum permeability, = 4 * pi / 10
const γ::Float32 = 2.221e5

abstract type AbstractField end

struct Mesh{T<:Int,U<:Number}
    nx::T
    ny::T
    nz::T
    dx::U
    dy::U
    dz::U
end

struct Sim
    m::CuArray{Float32}
    mesh::Mesh
    H_eff::CuArray{Float32}
    M_x_H::CuArray{Float32}
    Interactions::Vector{AbstractField}
    prefactor1::Number
    prefactor2::Number
    relax_prefactor1::Number
    relax_prefactor2::Number
end

include("Demag.jl")
include("Exchange.jl")
include("LLG.jl")
include("Zeeman.jl")
include("Init_m.jl")


VectorOrFunction = Union{Vector{Number},Function}
ScalarOrFunction = Union{Number,Function}

function InitSim(mesh::Mesh, m0; Aex=0.0, Ms=0., α=0., Bext=[0., 0., 0.])

    exch = 2 * Aex / μ₀ / Ms
    prefactor1 = -γ / (1 + α * α)
    prefactor2 = prefactor1 * α
    relax_α = 0.5
    relax_prefactor1 = -γ / (1 + relax_α * relax_α)
    relax_prefactor2 = relax_prefactor1 * relax_α

    ### Initialize Mesh ###

    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz

    m = Init_m(mesh, m0)

    if nz == 1

        input_indices = CartesianIndices((3, nx, ny))
        output_indices = CartesianIndices((3, nx:(2*nx-1), ny:(2*ny-1)))

        ########################

        ### Initialize Empty Arrays ###

        M_pad = CUDA.zeros(Float32, 3, nx * 2, ny * 2)
        M_fft = CUDA.zeros(ComplexF32, 3, nx + 1, ny * 2)

        H_demag = CUDA.zeros(Float32, 3, nx * 2, ny * 2)
        H_demag_fft = CUDA.zeros(ComplexF32, 3, nx + 1, ny * 2)

        H_eff = CUDA.zeros(Float32, 3, nx, ny)
        M_x_H = CUDA.zeros(Float32, 3, nx, ny)

        ################################

        ### Initialize Demag Kernel ###

        plan = plan_rfft(M_pad, [2, 3])

        demag = Demag(Ms, Demag_Kernel(nx, ny, nz, dx, dy, dz)..., plan, M_pad, M_fft, H_demag_fft, input_indices, output_indices)

        ################################

    else

        input_indices = CartesianIndices((3, nx, ny, nz))
        output_indices = CartesianIndices((3, nx:(2*nx-1), ny:(2*ny-1), nz:(2*nz-1)))

        ########################

        ### Initialize Empty Arrays ###

        M_pad = CUDA.zeros(Float32, 3, nx * 2, ny * 2, nz * 2)
        M_fft = CUDA.zeros(ComplexF32, 3, nx + 1, ny * 2, nz * 2)

        H_demag = CUDA.zeros(Float32, 3, nx * 2, ny * 2, nz * 2)
        H_demag_fft = CUDA.zeros(ComplexF32, 3, nx + 1, ny * 2, nz * 2)

        H_eff = CUDA.zeros(Float32, 3, nx, ny, nz)
        M_x_H = CUDA.zeros(Float32, 3, nx, ny, nz)

        ################################

        ### Initialize Demag Kernel ###

        plan = plan_rfft(M_pad, [2, 3, 4])

        demag = Demag(Demag_Kernel(nx, ny, nz, dx, dy, dz)..., plan, M_pad, M_fft, H_demag_fft, input_indices, output_indices)

        ################################

    end

    sim = Sim(m, mesh, H_eff, M_x_H, [], prefactor1, prefactor2, relax_prefactor1, relax_prefactor2)

    push!(sim.Interactions, demag)
    push!(sim.Interactions, Exchange(exch, nx, ny, nz, dx, dy, dz))
    push!(sim.Interactions, Zeeman(Bext))
    
    return sim
end

function Relax!(sim::Sim)
    prob = SteadyStateProblem(LLG_relax!, sim.m, sim)
    sol = solve(prob, DynamicSS(OwrenZen3(), abstol=.1), dt=1e-3)
    
    sim.m .= sol.u

    nothing
end

function Run(sim::Sim, t)



    # TODO: Convert to nice units
    end_point = t
    tspan = (0, end_point)
    t_points = range(0, end_point, length=300)

    prob = ODEProblem(LLG_run!, sim.m, tspan, sim)
    sol = solve(prob, OwrenZen3(), progress=true, progress_steps=1000, saveat=t_points, dt=1e-3)

    last_dim = length(size(sim.m))

    cpu_sol = cat([Array(x) for x in sol.u]..., dims=last_dim + 1)

    return sol.t, cpu_sol
end

end # module MicroMag.jl
