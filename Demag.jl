#Numerical Micromagnetics: Finite Difference Methods, Jacques E. Miltat1 and Michael J. Donahue. Page 14.
@time begin

    nx = 15
    ny = 15
    nz = 15

    # Size of the padded arrays
    lnx = 2 * nx
    lny = 2 * ny
    lnz = 2 * nz

    dx = 3e-9
    dy = 3e-9
    dz = 3e-9

    Kxx_short = zeros(nx, ny, nz); # Initialization of demagnetizatten
    Kxy_short = zeros(nx, ny, nz);
    Kxz_short = zeros(nx, ny, nz);
    Kyy_short = zeros(nx, ny, nz);
    Kyz_short = zeros(nx, ny, nz);
    Kzz_short = zeros(nx, ny, nz);

    Kxx_long = zeros(lnx, lny, lnz); # Initialization of demagnetizatten
    Kxy_long = zeros(lnx, lny, lnz);
    Kxz_long = zeros(lnx, lny, lnz);
    Kyy_long = zeros(lnx, lny, lnz);
    Kyz_long = zeros(lnx, lny, lnz);
    Kzz_long = zeros(lnx, lny, lnz);

    # #Spins
    # mx = zeros(lnx, lny, lnz)
    # my = zeros(lnx, lny, lnz)
    # mz = zeros(lnx, lny, lnz)

    # #Fourier transform of the magnetization
    # Mx = zeros(Complex{Float64}, nx, lny, lnz)
    # My = zeros(Complex{Float64}, nx, lny, lnz)
    # Mz = zeros(Complex{Float64}, nx, lny, lnz)

    # m_plan = FFTW.plan_rfft(mx)


    # Hx_demag = zeros(Complex{Float64}, nx, ny, nz)
    # Hx_demag = zeros(Complex{Float64}, nx, ny, nz)
    # Hx_demag = zeros(Complex{Float64}, nx, ny, nz)

    # h_plan = FFTW.plan_irfft(Hx_demag, lnx)


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

    # First for loop calculates the unique elements of the tensor 
    for index in CartesianIndices((1:nx, 1:ny, 1:nz))

        I, J, K = Tuple(index)

        if I == 0 && J == 0 && K == 0
            continue
        end

        x = (I - 1) * dx
        y = (J - 1) * dy
        z = (K - 1) * dz


        Kxx_short[index] = demag_tensor_diag(x, y, z, dx, dy, dz)
        Kyy_short[index] = demag_tensor_diag(y, z, x, dx, dy, dz)
        Kzz_short[index] = demag_tensor_diag(z, x, y, dx, dy, dz)

        Kxy_short[index] = demag_tensor_off_diag(x, y, z, dx, dy, dz)
        Kxz_short[index] = demag_tensor_off_diag(y, z, x, dx, dy, dz)
        Kyz_short[index] = demag_tensor_off_diag(z, x, y, dx, dy, dz)

    end

    # # Second for loop pads the larger part of the tensor 
    # for index in CartesianIndices((1:lnx, 1:lny, 1:lnz))

    #     I, J, K = Tuple(index)

    #     # If the index is within 
    #     if (I <= nx) && (J <= ny) && (K <= nz)
    #         continue

    #         # elseif (I == nx + 1) ||
    #         #    (J == ny + 1) ||
    #         #    (K == nz + 1)
    #         #     continue

    #     else
    #         x = (I <= nx) ? I : lnx - I + 2
    #         y = (J <= ny) ? J : lny - J + 2
    #         z = (K <= nz) ? K : lnz - K + 2
    #         Kxx[index] = Kxx[x, y, z]
    #     end
    # end


    function fill_demag_tensors(long_tensor, tensor)
        lnx, lny, lnz = size(long_tensor)
        nx, ny, nz = size(tensor)
        for i = 1:lnx, j = 1:lny, k = 1:lnz
            if (lnx % 2 == 0 && i == nx + 1) || (lny % 2 == 0 && j == ny + 1) || (lnz % 2 == 0 && k == nz + 1)
                continue
            end
            x = (i <= nx) ? i : lnx - i + 2
            y = (j <= ny) ? j : lny - j + 2
            z = (k <= nz) ? k : lnz - k + 2
            long_tensor[i, j, k] = tensor[x, y, z];
        end
    end

    fill_demag_tensors(Kxx_long, Kxx_short);
    fill_demag_tensors(Kyy_long, Kyy_short);
    fill_demag_tensors(Kzz_long, Kzz_short);
    fill_demag_tensors(Kxy_long, Kxy_short);
    fill_demag_tensors(Kxz_long, Kxz_short);
    fill_demag_tensors(Kyz_long, Kyz_short);

    @show Kxx_short

    fft_Kxx = rfft(Kxx_long);
    fft_Kxy = rfft(Kxy_long);
    fft_Kxz = rfft(Kxz_long);
    fft_Kyy = rfft(Kxy_long);
    fft_Kyz = rfft(Kyz_long);
    fft_Kzz = rfft(Kzz_long);


end
