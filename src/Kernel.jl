function padSize(size, periodic) 
	
    padded = zeros(Int, 3)

	for i in 1:length(size)
		if periodic[i] != 0 
			padded[i] = size[i]
			continue
        end

        SMALL_N = 5

		if i != 3 || size[i] > SMALL_N  # for some reason it only works for Z, perhaps we assume even FFT size elsewhere?
			# large N: zero pad * 2 for FFT performance
			padded[i] = size[i] * 2
		else 
			#  small N: minimal zero padding for memory/performance
			padded[i] = size[i]*2 #- 1
        end
    end
	
	return padded
end


function wrap(number, max_val) 
	while number < 0
        number += max_val
    end  
		
	while number >= max_val 
		number -= max_val
    end
	return number
end

function kernelRanges(size, pbc) 

    r1 = zeros(Float64, 3)
    r2 = zeros(Float64, 3)

	for c in 1:3 
		if pbc[c] == 0 
			r1[c], r2[c] = -ceil(Int,(size[c])/2), ceil(Int,(size[c])/2)
		else
			r1[c], r2[c] = -(size[c]*pbc[c]), (size[c]*pbc[c]) 
            # no /2 here, or we would take half right and half left image
        end
    end
	# support for 2D simulations (thickness 1)

	if size[3] == 1 && pbc[3] == 0
		r2[3] = 0
    end

	return r1, r2
end

function delta(d)
	if d < 0 
		d = -d
    end

	if d > 0 
		d -= 1
    end

	return d
end


function CalcDemagKernel(inputSize, pbc, cellsize, accuracy)

    size = padSize(inputSize, pbc)

    # TODO: Only allocate upper diagonal 
    K = zeros(3, 3, reverse(size)...)

    L = cellsize[1]

    # Field (destination) loop ranges
	r1, r2 = kernelRanges(size, pbc)

    if cellsize[2] < L
        L = cellsize[2]
    end

    if cellsize[3] < L
        L = cellsize[3]
    end

    for s in 1:3
     
        u, v, w = s, s%3 + 1, (s+1)%3 + 1

        R = zeros(3)
        R2 = zeros(3)
        pole = zeros(3)
        points = zeros(3)
    
        for z in r1[3]:r2[3]+1

            zw = round(Int,wrap(z, size[3])) + 1
            R[3] = z * cellsize[3]

            for y in r1[2]:r2[2]+1

                yw = round(Int,wrap(y, size[2])) + 1

                if yw > size[2]/2
                   continue
                end

                R[2] = y * cellsize[2]

                for x in r1[1]:r2[1]+1

                    xw = round(Int,wrap(x, size[1])) + 1

                    if xw > size[1]/2
                        continue
                    end

                    R[1] = x * cellsize[1]

                    dx = delta(x)*cellsize[1]
                    dy = delta(y)*cellsize[2]
                    dz = delta(z)*cellsize[3]

                    d = sqrt(dx^2 + dy^2 + dz^2)

                    if d == 0 
                        d = L 
                    end 

                    maxSize = d / accuracy

                    nv = ceil(Int, max(cellsize[v]/maxSize, 1) + 0.5)
                    nw = ceil(Int, max(cellsize[w]/maxSize, 1) + 0.5)
                    nx = ceil(Int, max(cellsize[1]/maxSize, 1) + 0.5)
                    ny = ceil(Int, max(cellsize[2]/maxSize, 1) + 0.5)
                    nz = ceil(Int, max(cellsize[3]/maxSize, 1) + 0.5)

                    nv *= 2 
                    nw *= 2

                    scale = 1 / (nv * nw * nx * ny * nz)
                    surface = cellsize[v] * cellsize[w] 

                    charge = surface * scale 
                    
                    pu1 = cellsize[u]  / 2. 
                    pu2 = -pu1 

                    B = zeros(Float64, 3)

                    for i in 0:nv-1 

                        pv = -(cellsize[v] / 2.) + cellsize[v]/(2*nv) +
                             i * (cellsize[v] / nv)
                            
                        pole[v] = pv 

                        for j in 0:nw-1 

                            pw = -(cellsize[w] / 2.) + cellsize[w]/(2*nw) +
                                 j * (cellsize[w] / nw)

                            pole[w] = pw 

                            for α in 0:nx-1

                                rx = R[1] - cellsize[1]/2 + cellsize[1]/(2*nx) +
                                     α * (cellsize[1] / nx)
                                

                                for β in 0:ny-1

                                    ry = R[2] - cellsize[2]/2 + cellsize[2]/(2*ny) +
                                         β * (cellsize[2] / ny)

                                    for γ in 0:nz-1

                                        rz = R[3] - cellsize[3]/2 + cellsize[3]/(2*nz) +
                                            γ * (cellsize[3] / nz)

                                        points .+= 1

                                        pole[u] = pu1
                                        R2[1] = rx - pole[1]
                                        R2[2] = ry - pole[2]
                                        R2[3] = rz - pole[3]

                                        r = sqrt(R2[1]^2 + R2[2]^2 + R2[3]^2)

                                        qr = -charge / (4*π*r*r*r)

                                        bx = R2[1] * qr
                                        by = R2[2] * qr
                                        bz = R2[3] * qr

                                        pole[u] = pu2
                                        R2[1] = rx - pole[1]
                                        R2[2] = ry - pole[2]
                                        R2[3] = rz - pole[3]
                                        r = sqrt(R2[1]^2 + R2[2]^2 + R2[3]^2)
                                        qr = -charge / (4*π*r*r*r)

                                        B[1] += bx + R2[1] * qr
                                        B[2] += by + R2[2] * qr
                                        B[3] += bz + R2[3] * qr

                                    end
                                end
                            end
                        end
                    end
                    for d in s:3
                        K[s, d, zw, yw, xw] += B[d]
                    end
                end
            end
        end
    end

    for z in 1:size[3]
        for y in 1:size[2]
            for x in round(Int,size[1]/2+1):size[1]
                x2 = size[1] - x + 1
                K[1, 1, z, y, x] = K[1, 1, z, y, x2]
                K[1, 2, z, y, x] = -K[1, 2, z, y, x2]
                K[1, 3, z, y, x] = -K[1, 3, z, y, x2]
                K[2, 2, z, y, x] = K[2, 2, z, y, x2]
                K[2, 3, z, y, x] = K[2, 3, z, y, x2]
                K[3, 3, z, y, x] = K[3, 3, z, y, x2]
            end
        end
    end

    for z in 1:size[3]
        for y in round(Int,size[2]/2+1):size[2]
            y2 = size[2] - y + 1
            for x in 1:size[1]
                K[1, 1, z, y, x] = K[1, 1, z, y2, x]
                K[1, 2, z, y, x] = -K[1, 2, z, y2, x]
                K[1, 3, z, y, x] = K[1, 3, z, y2, x]
                K[2, 2, z, y, x] = K[2, 2, z, y2, x]
                K[2, 3, z, y, x] = -K[2, 3, z, y2, x]
                K[3, 3, z, y, x] = K[3, 3, z, y2, x]
            end
        end
    end

    for z in round(Int,size[3]/2+1):size[3]
        z2 = size[3] - z + 1
        for y in 1:size[2]
            for x in 1:size[1]
                K[1, 1, z, y, x] = K[1, 1, z2, y, x]
                K[1, 2, z, y, x] = K[1, 2, z2, y, x]
                K[1, 3, z, y, x] = -K[1, 3, z2, y, x]
                K[2, 2, z, y, x] = K[2, 2, z2, y, x]
                K[2, 3, z, y, x] = -K[2, 3, z2, y, x]
                K[3, 3, z, y, x] = K[3, 3, z2, y, x]
            end
        end
    end

    if size[3] == 1
        K[1,3,:,:,:] = 0
        K[2,3,:,:,:] = 0
    end

    K[2,1,:,:,:] = K[1,2,:,:,:]
    K[3,1,:,:,:] = K[1,3,:,:,:]
    K[3,2,:,:,:] = K[2,3,:,:,:]

    return K
                                        
    
end


size = (2, 2, 2)
cellsize = (1e-9, 1e-9, 1e-9)
pbc = (0,0,0)
accuracy = 6 

K = CalcDemagKernel(size, pbc, cellsize, accuracy)