for K = -nz+1:nz-1 # Calculation of Demag tensor
    for J = -ny+1:ny-1
        for I = -nx+1:nx-1
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

                        Kxx[L, M, N] += (-1) .^ (i + j + k) * atan((K + k - 0.5) * (J + j - 0.5)*dy*dz / r / ((I + i - 0.5)*dx))
                        Kyy[L, M, N] += (-1) .^ (i + j + k) * atan((I + i - 0.5) * (K + k - 0.5)*dx*dz / r / ((J + j - 0.5)*dy))
                        Kzz[L, M, N] += (-1) .^ (i + j + k) * atan((J + j - 0.5) * (I + i - 0.5)*dy*dx / r / ((K + k - 0.5)*dz))

                        Kxy[L, M, N] += (-1) .^ (i + j + k) * log((K + k - 0.5) * dz + r)
                        Kxz[L, M, N] += (-1) .^ (i + j + k) * log((J + j - 0.5) * dy + r)
                        Kyz[L, M, N] += (-1) .^ (i + j + k) * log((I + i - 0.5) * dx + r)
                    end
                end
            end
            Kxx[L, M, N] *= prefactor
            Kxy[L, M, N] *= -prefactor
            Kxz[L, M, N] *= -prefactor
            Kyy[L, M, N] *= prefactor
            Kyz[L, M, N] *= -prefactor
            Kzz[L, M, N] *= prefactor
        end
    end
end # calculation of demag tensor done



Kxx = zeros(2*nx,2*ny,2*nz)

function newell_f(x,y,z)

    R = sqrt(x^2 + y^2 + z^2)

    return .5 * y * (z^2 - x^2) * asinh(y / sqrt(x^2 + z^2)) + 
           .5 * z * (y^2 - x^2) * asinh(z / sqrt(x^2 + y^2)) - 
           x*y*z * atan(y*z/x/R) - 1/6 * (2*x^2 - y^2 - z^2) * R 
end


function newell_g(x,y,z)
    R = sqrt(x^2 + y^2 + z^2)
    
    return x*y*z * asinh(z / sqrt(y^2 + z^2)) + 
           1/6 * y * (3*z^2 - y^2) * asinh(x/sqrt(y^2 + z^2)) + 
           1/6 * x * (3*z^2 - x^2) * asinh(y/sqrt(x^2 + z^2)) - 
           1/2 * y^2*z * atan(x*z/y/R) -
           1/2 * x^2*z * atan(y*z/x/R) - 
           1/6 * z^3 * atan(x*y/z/R) - 
           x*y*R/3
end


function Demag_Tensor(nx,ny,nz,dx,dy,dz)

    A = [[dx, 0, 0], [0, dy, 0], [0, 0, dz]]
    B = [[dx, dy, 0], [dx, 0, dz], [0, dy, dz]]
    C = [[dx, dy, dz]]
    for i in CartesianIndices((nx,ny,nz))
        
    


Kxx = CuArray(Kxx);
Kxy = CuArray(Kxy);
Kxz = CuArray(Kxz);
Kyy = CuArray(Kyy);
Kyz = CuArray(Kyz);
Kzz = CuArray(Kzz);

Kxx_fft = rfft(Kxx); # fast fourier transform of demag tensor
Kxy_fft = rfft(Kxy); # needs to be done only one time
Kxz_fft = rfft(Kxz);
Kyy_fft = rfft(Kyy);
Kyz_fft = rfft(Kyz);
Kzz_fft = rfft(Kzz);

plan = plan_rfft(Kxx);
iplan = plan_irfft(Kxx_fft, 2 * nx);