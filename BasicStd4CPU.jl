using DifferentialEquations
using LinearAlgebra # Used for cross product
using Plots
using Statistics # Used for mean 
using DSP # Used for convolution
using FFTW
using Unitful

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

nx = 166; # number of cells on x direction
ny = 42; # number of cells on y direction
nz = 1;



dx = 3e-9u"nm"; # cell size on x direction (nm)
dy = 3e-9u"nm"; # cell size on y direction (nm)
dz = 3e-9u"nm"; # cell size on z direction (nm)

alpha = 0.5
A = 1.3e-11u"J/m"
mu0 = pi * 4e-7u"H/m"
Ms = 8e5u"A/m"
gamma = 2.21e5u"m/(A*s)"


Kxx = zeros(2 * nx, 2 * ny, 2 * nz) # Initialization of demagnetization tensors
Kxy = zeros(2 * nx, 2 * ny, 2 * nz)
Kxz = zeros(2 * nx, 2 * ny, 2 * nz)
Kyy = zeros(2 * nx, 2 * ny, 2 * nz)
Kyz = zeros(2 * nx, 2 * ny, 2 * nz)
Kzz = zeros(2 * nx, 2 * ny, 2 * nz)

# Calculate demagnetization tensor
for index in CartesianIndices((range(-nx + 1, step=1, nx - 1), range(-ny + 1, step=1, ny - 1), range(-nz + 1, step=1, nz - 1)))

    I, J, K = Tuple(index)

    if I == 0 && J == 0 && K == 0
        continue
    end

    L = I + nx # shift the indices, b/c no negative index allowed in Julia
    M = J + ny
    N = K + nz

    for i = 0:1 # helper indices
        for j = 0:1
            for k = 0:1

                r = sqrt((I + i - 0.5) * (I + i - 0.5) * dx * dx + (J + j - 0.5) * (J + j - 0.5) * dy * dy + (K + k - 0.5) * (K + k - 0.5) * dz * dz)

                Kxx[L, M, N] += 1 / 4 / pi * (-1)^(i + j + k) * atan((K + k - 0.5) * (J + j - 0.5) * dy * dz / r / (I + i - 0.5))
                Kyy[L, M, N] += 1 / 4 / pi * (-1)^(i + j + k) * atan((I + i - 0.5) * (K + k - 0.5) * dx * dz / r / (J + j - 0.5))
                Kzz[L, M, N] += 1 / 4 / pi * (-1)^(i + j + k) * atan((J + j - 0.5) * (I + i - 0.5) * dx * dy / r / (K + k - 0.5))

                Kxy[L, M, N] += -1 / 4 / pi * (-1)^(i + j + k) * log((K + k - 0.5) * dz + r)
                Kxz[L, M, N] += -1 / 4 / pi * (-1)^(i + j + k) * log((J + j - 0.5) * dy + r)
                Kyz[L, M, N] += -1 / 4 / pi * (-1)^(i + j + k) * log((I + i - 0.5) * dz + r)

            end
        end
    end
end

fft_Kxx = fft(Kxx);
fft_Kxy = fft(Kxy);
fft_Kxz = fft(Kxz);
fft_Kyy = fft(Kxy);
fft_Kyz = fft(Kyz);
fft_Kzz = fft(Kzz);

# calculation of demag tensor done

# Demag tensors are larger so they can have zero padding
Hx_demag = zeros(2 * nx, 2 * ny, 2 * nz)
Hy_demag = zeros(2 * nx, 2 * ny, 2 * nz)
Hz_demag = zeros(2 * nx, 2 * ny, 2 * nz)

mx_demag_buffer = zeros(2 * nx, 2 * ny, 2 * nz)
my_demag_buffer = zeros(2 * nx, 2 * ny, 2 * nz)
mz_demag_buffer = zeros(2 * nx, 2 * ny, 2 * nz)

# TODO: Plan FFT 
# TODO: Use optional Julia FFT implementation for autodifferentiation
function convolve_kernel(fft_kernel1, fft_kernel2, fft_kernel3, mag1, mag2, mag3)

    fft_mag1 = fft(mag1)
    fft_mag2 = fft(mag2)
    fft_mag3 = fft(mag3)

    fft_demag1 = fft_mag1 .* fft_kernel1
    fft_demag2 = fft_mag2 .* fft_kernel2
    fft_demag3 = fft_mag3 .* fft_kernel3

    return real(ifft(fft_demag1 + fft_demag2 + fft_demag3))

end


function LLG_loop(dm, m, p, t)

    A, mu0, Ms, alpha, gamma, dx, dy, dz = p

    H_ext = [1u"T", 1u"T", 1u"T"] / mu0 
    precess=1

    prefactor1 = -gamma * (1 + alpha^2)
    prefactor2 = alpha/Ms

    mx = m[1, :, :, :]
    my = m[2, :, :, :]
    mz = m[3, :, :, :]

    for index in CartesianIndices((nx, ny, nz))
        mx_demag_buffer[index] = mx[index]
        my_demag_buffer[index] = my[index]
        mz_demag_buffer[index] = mz[index]
    end

    Hx_demag = convolve_kernel(fft_Kxx, fft_Kxy, fft_Kxz, mx_demag_buffer, my_demag_buffer, mz_demag_buffer)
    Hy_demag = convolve_kernel(fft_Kxy, fft_Kyy, fft_Kyz, mx_demag_buffer, my_demag_buffer, mz_demag_buffer)
    Hz_demag = convolve_kernel(fft_Kxz, fft_Kyz, fft_Kzz, mx_demag_buffer, my_demag_buffer, mz_demag_buffer)

    for index in CartesianIndices((nx, ny, nz))

        i, j, k = Tuple(index)

        # TODO: Periodic boundary conditions


        #TODO: Add 3rd dimsion to exchange field calulation
        #TODO: Use ParallelStencil.jl to speed up the calculation
        #TODO: Implement boundary conditions for truly 3D system

        # Neumann boundary conditions
        if (i == 1 || i == nx) && !(j == 1 || j == ny)

            mx_diff = (m[1, i, j+1] - 2 * m[1, i, j] + m[1, i, j-1]) / dy / dy
            my_diff = (m[2, i, j+1] - 2 * m[2, i, j] + m[1, i, j-1]) / dy / dy
            mz_diff = (m[3, i, j+1] - 2 * m[3, i, j] + m[1, i, j-1]) / dy / dy

        elseif (j == 1 || j == ny) && !(i == 1 || i == nx)
            mx_diff = (m[1, i+1, j] - 2 * m[1, i, j] + m[1, i-1, j]) / dx / dx
            my_diff = (m[2, i+1, j] - 2 * m[2, i, j] + m[2, i-1, j]) / dx / dx
            mz_diff = (m[3, i+1, j] - 2 * m[3, i, j] + m[3, i-1, j]) / dx / dx

        elseif (i == 1 || i == nx) && (j == 1 || j == ny)
            mx_diff = 0
            my_diff = 0
            mz_diff = 0

        else
            mx_diff = (m[1, i+1, j] - 2 * m[1, i, j] + m[1, i-1, j]) / dx / dx + (m[1, i, j+1] - 2 * m[1, i, j] + m[1, i, j-1]) / dy / dy
            my_diff = (m[2, i+1, j] - 2 * m[2, i, j] + m[2, i-1, j]) / dx / dx + (m[2, i, j+1] - 2 * m[2, i, j] + m[1, i, j-1]) / dy / dy
            mz_diff = (m[3, i+1, j] - 2 * m[3, i, j] + m[3, i-1, j]) / dx / dx + (m[3, i, j+1] - 2 * m[3, i, j] + m[1, i, j-1]) / dy / dy

        end

        #TODO: Use harmonic mean for spatially dependent exchange field
        H_exch = 2 * A / (Ms * mu0) .* [mx_diff, my_diff, mz_diff]

        #TODO: Make more concise vector expression
        H_effx = Hx_demag[i, j, k] + H_exch[1] + H_ext[1] # effective field
        H_effy = Hy_demag[i, j, k] + H_exch[2] + H_ext[2] # effective field
        H_effz = Hz_demag[i, j, k] + H_exch[3] + H_ext[3] # effective field

        H_eff = [H_effx, H_effy, H_effz]

        # TODO: Add anisotropy term 
        # TODO: Add DMI term

        current_m = [mx[i, j, k], my[i, j, k], mz[i, j, k]]

        # TODO: Spatially dependent damping
        dm[:, i, j, k] = prefactor1 .* (precess * cross(current_m, H_eff) +
                                        prefactor2 .* cross(current_m, cross(current_m, H_eff))) / 1e9

        # println(dm[2, i, j, k])

    end # end of LLG loop
end

function normalize!(m,Ms)
    
    mx = m[1, :, :, :]
    my = m[2, :, :, :]
    mz = m[3, :, :, :]

    mag_length = sqrt.(mx.^2 + my.^2 + mz.^2)
    mx ./= mag_length .* Ms 
    my ./= mag_length .* Ms
    mz ./= mag_length .* Ms

end 


m0 = zeros(3, nx, ny, nz)
m0[1, :, :, :] .= Ms

p = (A, mu0, Ms, alpha, gamma, dx, dy, dz)

function condition(m, t, integrator)

    mx = m[1, :, :, :]
    my = m[2, :, :, :]
    mz = m[3, :, :, :]

    A, mu0, Ms, alpha, gamma, dx, dy, dz = integrator.p
    return mean(sqrt.(mx .^ 2 + my .^ 2 + mz .^ 2)) != Ms
end

function affect!(integrator)

    A, mu0, Ms, alpha, gamma, dx, dy, dz = integrator.p

    mx = integrator.u[1, :, :, :]
    my = integrator.u[2, :, :, :]
    mz = integrator.u[3, :, :, :]

    mag_length = sqrt.(mx .^ 2 + my .^ 2 + mz .^ 2)
    # @info("Magnetization length: %f", mx ./ mag_length .* Ms)
    integrator.u[1, :, :, :] ./= mag_length .* Ms
    integrator.u[2, :, :, :] ./= mag_length .* Ms
    integrator.u[3, :, :, :] ./= mag_length .* Ms

    @info("Magnetization length: %f", integrator.u[1, :, :, :])
end

cb = DiscreteCallback(condition,affect!)

end_point = 30e-9
tspan = (0, end_point)
t_range = range(0, end_point, length=300)

# norm_manifold = ManifoldProjection(norm_residual)

# prob = SteadyStateProblem(LLG_loop, m0, p)
prob = ODEProblem(LLG_loop, m0, tspan, p)

@time sol = solve(prob, Tsit5(), progress=true, progress_steps=100,dt=1e-12)


mx_vals = sol(t_range)[1, :, :, :, :]
my_vals = sol(t_range)[2, :, :, :, :]
mz_vals = sol(t_range)[3, :, :, :, :]

mx_avg = mean(mx_vals, dims=[1, 2, 3])[1, 1, 1, :]
my_avg = mean(my_vals, dims=[1, 2, 3])[1, 1, 1, :]
mz_avg = mean(mz_vals, dims=[1, 2, 3])[1, 1, 1, :]

mag = sqrt.(mx_avg .^ 2 + my_avg .^ 2 + mz_avg .^ 2)

plot(t_range, mx_avg, label="mx")
plot!(t_range, my_avg, label="my")
plot!(t_range, mz_avg, label="mz")
plot!(t_range, mag, label="norm")
