using DifferentialEquations
using LinearAlgebra # Used for cross product
using Plots 
using Statistics # Used for mean 
using DSP # Used for convolution
using FFTW
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

# TODO: Scale problem?
nx = 6; # number of cells on x direction
ny = 6; # number of cells on y direction
nz = 1;

dx = 3; # cell size on x direction (nm)
dy = 3; # cell size on y direction (nm)
dz = 3; # cell size on z direction (nm)

alpha = 0.5 
A = 1.3e-11 * 1e18
mu0 = pi * 4e-7 / 10
Ms = 800
gamma = .221


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
Hx_demag = zeros(2*nx, 2*ny, 2*nz)
Hy_demag = zeros(2*nx, 2*ny, 2*nz)
Hz_demag = zeros(2*nx, 2*ny, 2*nz)

mx_demag_buffer = zeros(2*nx, 2*ny, 2*nz)
my_demag_buffer = zeros(2*nx, 2*ny, 2*nz)
mz_demag_buffer = zeros(2*nx, 2*ny, 2*nz)

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


    A, mu0, Ms, alpha, gamma, precess, dx, dy, dz = p

    prefactor1 = -gamma / (1 + alpha * alpha)
    prefactor2 = alpha / Ms

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
        # TODO: Implement boundary conditions for truly 3D system
        # Neumann boundary conditions
        if i == 1 || j == 1 || i == nx || j == ny
            dm[1, i, j, k] = 0
            dm[2, i, j, k] = 0
            dm[3, i, j, k] = 0
            continue
        end

        #TODO: Add 3rd dimsion to exchange field calulation
        #TODO: Use ParallelStencil.jl to speed up the calculation
        mx_diff = (m[1, i+1, j] - 2 * m[1, i, j] + m[1, i-1, j]) / dx / dx + (m[1, i, j+1] - 2 * m[1, i, j] + m[1, i, j-1]) / dy / dy
        my_diff = (m[2, i+1, j] - 2 * m[2, i, j] + m[2, i-1, j]) / dx / dx + (m[2, i, j+1] - 2 * m[2, i, j] + m[1, i, j-1]) / dy / dy
        mz_diff = (m[3, i+1, j] - 2 * m[3, i, j] + m[3, i-1, j]) / dx / dx + (m[3, i, j+1] - 2 * m[3, i, j] + m[1, i, j-1]) / dy / dy

        #TODO: Use harmonic mean for spatially dependent exchange field
        H_exch = 2 * A / (mu0 * Ms * Ms) .* [mx_diff, my_diff, mz_diff]

        #TODO: Make this a passed variable
        # H_ext = [-24.6e-3, 4.3e-3, 0] * 1e-9 * 1e-9 / mu0 # External field in 
        # H_ext = [1, 1, 1]
        H_ext = [100, 100, 100] 


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
                                        prefactor2 .* cross(current_m, cross(current_m, H_eff))) / 200000000



    end # end of LLG loop
end


m0 = zeros(3, nx, ny, nz)
m0[1, :, :, :] .= Ms
# m0[1, :, :, :] .= .1 * Ms # initial magnetization
# m0[2, :, :, :] .= 0.1 * Ms # initial magnetization

# dm_dummy = zeros(3, nx, ny, nz)
p = (A, mu0, Ms, alpha, gamma, 1, dx, dy, dz)

end_point = .75
tspan = (0, end_point)

# prob = SteadyStateProblem(LLG_loop, m0, p)
# @time sol = solve(prob, DynamicSS(BS3()))

# p = (A, mu0, Ms, alpha, gamma, 1, dx, dy, dz)

prob = ODEProblem(LLG_loop, m0, tspan, p)
@time sol = solve(prob, BS3(), progress=true,progress_steps=100)

t_range = range(0, end_point, length=1000)

mx_vals = sol(t_range)[1, :, :, :, :]
my_vals = sol(t_range)[2, :, :, :, :] 
mz_vals = sol(t_range)[3, :, :, :, :] 

mx_avg = mean(mx_vals, dims=[1, 2, 3])[1, 1, 1, :]
my_avg = mean(my_vals, dims=[1, 2, 3])[1, 1, 1, :]
mz_avg = mean(mz_vals, dims=[1, 2, 3])[1, 1, 1, :]

norm = sqrt.(mx_avg.^ 2 + my_avg.^2 + mz_avg.^2)

plot(t_range, mx_avg./norm, label="mx")
plot!(t_range, my_avg./norm, label="my")
plot!(t_range, mz_avg./norm, label="mz")