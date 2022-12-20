using FFTW
using Memoize

# Yoshinobu Nakatani et al 1989 Jpn. J. Appl. Phys. 28 2485


# 3D Kernel
@memoize function Demag_Kernel(nx::Int, ny::Int, nz::Int, dx::Float64, dy::Float64, dz::Float64)

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

    Kxx_fft_cpu = rfft(Kxx_cpu) # fast fourier transform of demag tensor
    Kxy_fft_cpu = rfft(Kxy_cpu) # needs to be done only one time
    Kxz_fft_cpu = rfft(Kxz_cpu)
    Kyy_fft_cpu = rfft(Kyy_cpu)
    Kyz_fft_cpu = rfft(Kyz_cpu)
    Kzz_fft_cpu = rfft(Kzz_cpu)

    Kxx_fft = CuArray{ComplexF32}(Kxx_fft_cpu)
    Kxy_fft = CuArray{ComplexF32}(Kxy_fft_cpu)
    Kxz_fft = CuArray{ComplexF32}(Kxz_fft_cpu)
    Kyy_fft = CuArray{ComplexF32}(Kyy_fft_cpu)
    Kyz_fft = CuArray{ComplexF32}(Kyz_fft_cpu)
    Kzz_fft = CuArray{ComplexF32}(Kzz_fft_cpu)

    return Kxx_fft, Kyy_fft, Kzz_fft, Kxy_fft, Kxz_fft, Kyz_fft

end

# 2D Kernel
@memoize function Demag_Kernel(nx::Int, ny::Int, dx::Float64, dy::Float64, dz::Float64)

    prefactor = 1 / 4 / pi

    Kxx_cpu = zeros(2 * nx, 2 * ny, 2 * nz)
    Kxy_cpu = zeros(2 * nx, 2 * ny, 2 * nz)
    Kyy_cpu = zeros(2 * nx, 2 * ny, 2 * nz)
    Kzz_cpu = zeros(2 * nx, 2 * ny, 2 * nz)

    K = 0
    @inbounds @simd for J = -ny+1:ny-1
        for I = -nx+1:nx-1
            L = I + nx # shift the indices, b/c no negative index allowed in Julia
            M = J + ny
            for i = 0:1 # helper indices
                for j = 0:1
                    for k = 0:1
                        r = sqrt((I + i - 0.5) * (I + i - 0.5) * dx * dx + (J + j - 0.5) * (J + j - 0.5) * dy * dy + (K + k - 0.5) * (K + k - 0.5) * dz * dz)

                        Kxx_cpu[L, M, N] += (-1)^(i + j + k) * atan((K + k - 0.5) * (J + j - 0.5) * dy * dz / (r * (I + i - 0.5) * dx))
                        Kyy_cpu[L, M, N] += (-1)^(i + j + k) * atan((I + i - 0.5) * (K + k - 0.5) * dx * dz / (r * (J + j - 0.5) * dy))
                        Kzz_cpu[L, M, N] += (-1)^(i + j + k) * atan((J + j - 0.5) * (I + i - 0.5) * dy * dx / (r * (K + k - 0.5) * dz))

                        Kxy_cpu[L, M, N] += (-1)^(i + j + k) * log((K + k - 0.5) * dz + r)
                    end
                end
            end

            Kxx_cpu[L, M, N] *= prefactor
            Kzz_cpu[L, M, N] *= prefactor
            Kyy_cpu[L, M, N] *= prefactor

            Kxy_cpu[L, M, N] *= -prefactor
        end
    end

    Kxx_fft_cpu = rfft(Kxx_cpu) # fast fourier transform of demag tensor
    Kxy_fft_cpu = rfft(Kxy_cpu) # needs to be done only one time
    Kyy_fft_cpu = rfft(Kyy_cpu)
    Kzz_fft_cpu = rfft(Kzz_cpu)

    Kxx_fft = CuArray{ComplexF32}(Kxx_fft_cpu)
    Kxy_fft = CuArray{ComplexF32}(Kxy_fft_cpu)
    Kyy_fft = CuArray{ComplexF32}(Kyy_fft_cpu)
    Kzz_fft = CuArray{ComplexF32}(Kzz_fft_cpu)

    return Kxx_fft, Kyy_fft, Kzz_fft, Kxy_fft

end


function Demag!(fields::Fields, demag::Demag, mesh::Mesh)

    mul!(fields.M_fft, demag.fft, fields.M_pad)

    Mx_fft = @view fields.M_fft[1, :, :, :]
    My_fft = @view fields.M_fft[2, :, :, :]
    Mz_fft = @view fields.M_fft[3, :, :, :]

    @. fields.H_demag_fft[1, :, :, :] = Mx_fft * demag.Kxx_fft + My_fft * demag.Kxy_fft + Mz_fft * demag.Kxz_fft
    @. fields.H_demag_fft[2, :, :, :] = Mx_fft * demag.Kxy_fft + My_fft * demag.Kyy_fft + Mz_fft * demag.Kyz_fft
    @. fields.H_demag_fft[3, :, :, :] = Mx_fft * demag.Kxz_fft + My_fft * demag.Kyz_fft + Mz_fft * demag.Kzz_fft

    ldiv!(fields.M_pad, demag.fft, fields.H_demag_fft)

    @inbounds fields.H_eff .= fields.M_pad[mesh.out] # truncation of demag field

    nothing
end


