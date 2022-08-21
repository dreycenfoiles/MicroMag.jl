using DifferentialEquations
using LinearAlgebra # Used for cross product
using Plots
using Statistics # Used for mean 
using DSP # Used for convolution
using FFTW
using ModelingToolkit

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

nx = 30; # number of cells on x direction
ny = 20; # number of cells on y direction
nz = 1;



dx = 3; # cell size on x direction (nm)
dy = 3 # cell size on y direction (nm)
dz = 3; # cell size on z direction (nm)

alpha = 0.5
A = 1.3e-11 * 1e18
mu0 = 1.256
Ms = 800
gamma = 0.221


Kxx = zeros(2 * nx, 2 * ny, 2 * nz) # Initialization of demagnetization tensors
Kxy = zeros(2 * nx, 2 * ny, 2 * nz)
Kxz = zeros(2 * nx, 2 * ny, 2 * nz)
Kyy = zeros(2 * nx, 2 * ny, 2 * nz)
Kyz = zeros(2 * nx, 2 * ny, 2 * nz)
Kzz = zeros(2 * nx, 2 * ny, 2 * nz)


#Numerical Micromagnetics: Finite Difference Methods, Jacques E. Miltat1 and Michael J. Donahue. Page 14.

function newell_f(x, y, z)
    x2 = x * x
    y2 = y * y
    z2 = z * z
    R = sqrt(x2 + y2 + z2)
    if R == 0.0
        return 0.0
    end

    f = 1.0 / 6 * (2 * x2 - y2 - z2) * R

    if x2 > 0
        f -= x * y * z * atan(y * z / (x * R))
    end

    if x2 + z2 > 0
        f += 0.5 * y * (z2 - x2) * asinh(y / (sqrt(x2 + z2)))
    end

    if x2 + y2 > 0
        f += 0.5 * z * (y2 - x2) * asinh(z / (sqrt(x2 + y2)))
    end
    return f
end


function newell_g(x, y, z)
    x2 = x * x
    y2 = y * y
    z2 = z * z

    R = sqrt(x2 + y2 + z2)
    if R == 0.0
        return 0.0
    end

    g = -1.0 / 3 * x * y * R

    if z2 > 0
        g -= 1.0 / 6 * z2 * z * atan(x * y / (z * R))
    end
    if y2 > 0
        g -= 0.5 * y2 * z * atan(x * z / (y * R))
    end
    if x2 > 0
        g -= 0.5 * x2 * z * atan(y * z / (x * R))
    end

    if x2 + y2 > 0
        g += x * y * z * asinh(z / (sqrt(x2 + y2)))
    end

    if y2 + z2 > 0
        g += 1.0 / 6 * y * (3 * z2 - y2) * asinh(x / (sqrt(y2 + z2)))
    end

    if x2 + z2 > 0
        g += 1.0 / 6 * x * (3 * z2 - x2) * asinh(y / (sqrt(x2 + z2)))
    end

    return g
end


function demag_tensor_diag(x, y, z, dx, dy, dz)

    tensor = 8.0 * newell_f(x, y, z)

    tensor -= 4.0 * newell_f(x + dx, y, z)
    tensor -= 4.0 * newell_f(x - dx, y, z)
    tensor -= 4.0 * newell_f(x, y - dy, z)
    tensor -= 4.0 * newell_f(x, y + dy, z)
    tensor -= 4.0 * newell_f(x, y, z - dz)
    tensor -= 4.0 * newell_f(x, y, z + dz)

    tensor += 2.0 * newell_f(x + dx, y + dy, z)
    tensor += 2.0 * newell_f(x + dx, y - dy, z)
    tensor += 2.0 * newell_f(x - dx, y - dy, z)
    tensor += 2.0 * newell_f(x - dx, y + dy, z)
    tensor += 2.0 * newell_f(x + dx, y, z + dz)
    tensor += 2.0 * newell_f(x + dx, y, z - dz)
    tensor += 2.0 * newell_f(x - dx, y, z + dz)
    tensor += 2.0 * newell_f(x - dx, y, z - dz)
    tensor += 2.0 * newell_f(x, y - dy, z - dz)
    tensor += 2.0 * newell_f(x, y - dy, z + dz)
    tensor += 2.0 * newell_f(x, y + dy, z + dz)
    tensor += 2.0 * newell_f(x, y + dy, z - dz)

    tensor -= newell_f(x + dx, y + dy, z + dz)
    tensor -= newell_f(x + dx, y + dy, z - dz)
    tensor -= newell_f(x + dx, y - dy, z + dz)
    tensor -= newell_f(x + dx, y - dy, z - dz)
    tensor -= newell_f(x - dx, y + dy, z + dz)
    tensor -= newell_f(x - dx, y + dy, z - dz)
    tensor -= newell_f(x - dx, y - dy, z + dz)
    tensor -= newell_f(x - dx, y - dy, z - dz)

    return tensor / (4.0 * pi * dx * dy * dz)

end

function demag_tensor_off_diag(x, y, z, dx, dy, dz)

    tensor = 8.0 * newell_g(x, y, z)

    tensor -= 4.0 * newell_g(x + dx, y, z)
    tensor -= 4.0 * newell_g(x - dx, y, z)
    tensor -= 4.0 * newell_g(x, y - dy, z)
    tensor -= 4.0 * newell_g(x, y + dy, z)
    tensor -= 4.0 * newell_g(x, y, z - dz)
    tensor -= 4.0 * newell_g(x, y, z + dz)


    tensor += 2.0 * newell_g(x + dx, y + dy, z)
    tensor += 2.0 * newell_g(x + dx, y - dy, z)
    tensor += 2.0 * newell_g(x - dx, y - dy, z)
    tensor += 2.0 * newell_g(x - dx, y + dy, z)
    tensor += 2.0 * newell_g(x + dx, y, z + dz)
    tensor += 2.0 * newell_g(x + dx, y, z - dz)
    tensor += 2.0 * newell_g(x - dx, y, z + dz)
    tensor += 2.0 * newell_g(x - dx, y, z - dz)
    tensor += 2.0 * newell_g(x, y - dy, z - dz)
    tensor += 2.0 * newell_g(x, y - dy, z + dz)
    tensor += 2.0 * newell_g(x, y + dy, z + dz)
    tensor += 2.0 * newell_g(x, y + dy, z - dz)

    tensor -= newell_g(x + dx, y + dy, z + dz)
    tensor -= newell_g(x + dx, y + dy, z - dz)
    tensor -= newell_g(x + dx, y - dy, z + dz)
    tensor -= newell_g(x + dx, y - dy, z - dz)
    tensor -= newell_g(x - dx, y + dy, z + dz)
    tensor -= newell_g(x - dx, y + dy, z - dz)
    tensor -= newell_g(x - dx, y - dy, z + dz)
    tensor -= newell_g(x - dx, y - dy, z - dz)

    return tensor / (4.0 * pi * dx * dy * dz)

end

# Calculate demagnetization tensor
for index in CartesianIndices((range(-nx + 1, step=1, nx - 1), range(-ny + 1, step=1, ny - 1), range(-nz + 1, step=1, nz - 1)))

    I, J, K = Tuple(index)

    if I == 0 && J == 0 && K == 0
        continue
    end

    x = I * dx
    y = J * dy
    z = K * dz

    L = I + nx # shift the indices, b/c no negative index allowed in Julia
    M = J + ny
    N = K + nz

    Kxx[L, M, N] += demag_tensor_diag(x, y, z, dx, dy, dz)
    Kyy[L, M, N] += demag_tensor_diag(y, z, x, dx, dy, dz)
    Kzz[L, M, N] += demag_tensor_diag(z, x, y, dx, dy, dz)

    Kxy[L, M, N] += demag_tensor_off_diag(x, y, z, dx, dy, dz)
    Kxz[L, M, N] += demag_tensor_off_diag(y, z, x, dx, dy, dz)
    Kyz[L, M, N] += demag_tensor_off_diag(z, x, y, dx, dy, dz)

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

function neumann_bc(x, n)

    if x == n + 1
        return x - 1
    elseif x == 0
        return x + 1
    else
        return x
    end
end


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


function LLG_loop(dm, m0, p, t)

    A, mu0, Ms, alpha, gamma, dx, dy, dz = p

    H_ext = [100, 100, 100]
    precess = 1

    prefactor1 = -gamma * (1 + alpha^2)
    prefactor2 = alpha / Ms

    mx = m0[1, :, :, :]
    my = m0[2, :, :, :]
    mz = m0[3, :, :, :]

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

        ip1 = neumann_bc(i + 1, nx)
        im1 = neumann_bc(i - 1, nx)
        jp1 = neumann_bc(j + 1, ny)
        jm1 = neumann_bc(j - 1, ny)

        # Laplacian operator
        m_diff = [
            (m0[comp, ip1, j] - 2 * m0[comp, i, j] + m0[comp, im1, j]) / dx / dx +
            (m0[comp, i, jp1] - 2 * m0[comp, i, j] + m0[comp, i, jm1]) / dy / dy for comp in 1:3
        ]


        #TODO: Use harmonic mean for spatially dependent exchange field
        H_exch = 2 * A / (Ms^2 * mu0) .* m_diff

        #TODO: Make more concise vector expression
        H_effx = Hx_demag[i, j, k] + H_exch[1] + H_ext[1] # effective field
        H_effy = Hy_demag[i, j, k] + H_exch[2] + H_ext[2]
        H_effz = Hz_demag[i, j, k] + H_exch[3] + H_ext[3]

        H_eff = [H_effx, H_effy, H_effz]

        # TODO: Add anisotropy term 
        # TODO: Add DMI term

        current_m = [mx[i, j, k], my[i, j, k], mz[i, j, k]]

        # TODO: Spatially dependent damping
        dm[:, i, j, k] = prefactor1 .* (precess * cross(current_m, H_eff) +
                                        prefactor2 .* cross(current_m, cross(current_m, H_eff)))

        # dm[4, i, j, k] = sqrt.(mx.^2 + my.^2 + mz.^2) - Ms

    end # end of LLG loop
end


p = (A, mu0, Ms, alpha, gamma, dx, dy, dz)

end_point = 0.5
tspan = (0, end_point)

m0 = zeros(3, nx, ny, nz)
m0[1, :, :, :] .= Ms

prob = ODEProblem(LLG_loop, m0, tspan, p)

dt = 1e-3
t_range = range(0, end_point, length=100)


@time sol = solve(prob, Tsit5(), progress=true, progress_steps=100)


mx_vals = sol(t_range)[1, :, :, :, :]
my_vals = sol(t_range)[2, :, :, :, :]
mz_vals = sol(t_range)[3, :, :, :, :]

mx_avg = mean(mx_vals, dims=[1, 2, 3])[1, 1, 1, :]
my_avg = mean(my_vals, dims=[1, 2, 3])[1, 1, 1, :]
mz_avg = mean(mz_vals, dims=[1, 2, 3])[1, 1, 1, :]

m_norm = sqrt.(mx_avg .^ 2 + my_avg .^ 2 + mz_avg .^ 2)

plot(t_range, mx_avg, label="mx")
plot!(t_range, my_avg, label="my")
plot!(t_range, mz_avg, label="mz")
plot!(t_range, m_norm, label="norm")
