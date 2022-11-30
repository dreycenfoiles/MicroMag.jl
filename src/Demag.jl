

# Yoshinobu Nakatani et al 1989 Jpn. J. Appl. Phys. 28 2485

@memoize function Demag_Kernel(nx, ny, nz, dx, dy, dz)

    prefactor = 1 / 4 / pi

    Kxx_cpu = zeros(2 * nx, 2 * ny, 2 * nz)
    Kxy_cpu = zeros(2 * nx, 2 * ny, 2 * nz)
    Kxz_cpu = zeros(2 * nx, 2 * ny, 2 * nz)
    Kyy_cpu = zeros(2 * nx, 2 * ny, 2 * nz)
    Kyz_cpu = zeros(2 * nx, 2 * ny, 2 * nz)
    Kzz_cpu = zeros(2 * nx, 2 * ny, 2 * nz)

    @inbounds @simd for K = -nz+1:nz-1 # Calculation of Demag tensor
        for J = -ny+1:ny-1
            for I = -nx+1:nx-1
                L = I + nx # shift the indices, b/c no negative index allowed in Julia
                M = J + ny
                N = K + nz
                for i = 0:1 # helper indices
                    for j = 0:1
                        for k = 0:1
                            r = sqrt((I + i - 0.5) * (I + i - 0.5) * dx * dx + (J + j - 0.5) * (J + j - 0.5) * dy * dy + (K + k - 0.5) * (K + k - 0.5) * dz * dz)

                            Kxx_cpu[L, M, N] += (-1)^(i + j + k) * atan((K + k - 0.5) * (J + j - 0.5) * dy * dz / (r * (I + i - 0.5) * dx))
                            Kyy_cpu[L, M, N] += (-1)^(i + j + k) * atan((I + i - 0.5) * (K + k - 0.5) * dx * dz / (r * (J + j - 0.5) * dy))
                            Kzz_cpu[L, M, N] += (-1)^(i + j + k) * atan((J + j - 0.5) * (I + i - 0.5) * dy * dx / (r * (K + k - 0.5) * dz))

                            Kxy_cpu[L, M, N] += (-1)^(i + j + k) * log((K + k - 0.5) * dz + r)
                            Kxz_cpu[L, M, N] += (-1)^(i + j + k) * log((J + j - 0.5) * dy + r)
                            Kyz_cpu[L, M, N] += (-1)^(i + j + k) * log((I + i - 0.5) * dx + r)
                        end
                    end
                end

                Kxx_cpu[L, M, N] *= prefactor
                Kzz_cpu[L, M, N] *= prefactor
                Kyy_cpu[L, M, N] *= prefactor

                Kxy_cpu[L, M, N] *= -prefactor
                Kxz_cpu[L, M, N] *= -prefactor
                Kyz_cpu[L, M, N] *= -prefactor
            end
        end
    end
    Kxx = CuArray{Float32}(Kxx_cpu)
    Kxy = CuArray{Float32}(Kxy_cpu)
    Kxz = CuArray{Float32}(Kxz_cpu)
    Kyy = CuArray{Float32}(Kyy_cpu)
    Kyz = CuArray{Float32}(Kyz_cpu)
    Kzz = CuArray{Float32}(Kzz_cpu)

    Kxx_fft = rfft(Kxx) # fast fourier transform of demag tensor
    Kxy_fft = rfft(Kxy) # needs to be done only one time
    Kxz_fft = rfft(Kxz)
    Kyy_fft = rfft(Kyy)
    Kyz_fft = rfft(Kyz)
    Kzz_fft = rfft(Kzz)

    return Kxx_fft, Kyy_fft, Kzz_fft, Kxy_fft, Kxz_fft, Kyz_fft
end

function Demag!(H_eff, M_pad, M_fft, H_demag, H_demag_fft, fft_plans, Mx_kernels, My_kernels, Mz_kernels, output_indices)

    Kxx_fft, Kxy_fft, Kxz_fft = Mx_kernels
    Kxy_fft, Kyy_fft, Kyz_fft = My_kernels
    Kxz_fft, Kyz_fft, Kzz_fft = Mz_kernels

    plan, iplan = fft_plans


    mul!(M_fft, plan, M_pad)

    Mx_fft = @view M_fft[1, :, :, :]
    My_fft = @view M_fft[2, :, :, :]
    Mz_fft = @view M_fft[3, :, :, :]

    @. H_demag_fft[1, :, :, :] = Mx_fft * Kxx_fft + My_fft * Kxy_fft + Mz_fft * Kxz_fft
    @. H_demag_fft[2, :, :, :] = Mx_fft * Kxy_fft + My_fft * Kyy_fft + Mz_fft * Kyz_fft
    @. H_demag_fft[3, :, :, :] = Mx_fft * Kxz_fft + My_fft * Kyz_fft + Mz_fft * Kzz_fft

    mul!(H_demag, iplan, H_demag_fft)

    @inbounds H_eff .= real(H_demag[output_indices]) # truncation of demag field

    nothing
end


