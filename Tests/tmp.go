// Calculates the magnetostatic kernel by brute-force integration
// of magnetic charges over the faces and averages over cell volumes.
func CalcDemagKernel(inputSize, pbc [3]int, cellsize [3]float64, accuracy float64) (kernel [3][3]*data.Slice) {

	// Add zero-padding in non-PBC directions
	size := padSize(inputSize, pbc)

	// Sanity check
	// {
	// 	util.Assert(size[Z] > 0 && size[Y] > 0 && size[X] > 0)
	// 	util.Assert(cellsize[X] > 0 && cellsize[Y] > 0 && cellsize[Z] > 0)
	// 	util.Assert(pbc[X] >= 0 && pbc[Y] >= 0 && pbc[Z] >= 0)
	// 	util.Assert(accuracy > 0)
	// }

	// Allocate only upper diagonal part. The rest is symmetric due to reciprocity.
	var array [3][3][][][]float32
	for i := 0; i < 3; i++ {
		for j := i; j < 3; j++ {
			kernel[i][j] = data.NewSlice(1, size)
			array[i][j] = kernel[i][j].Scalars()
		}
	}

	// Field (destination) loop ranges
	r1, r2 := kernelRanges(size, pbc)

	// smallest cell dimension is our typical length scale
	L := cellsize[X]
	{
		if cellsize[Y] < L {
			L = cellsize[Y]
		}
		if cellsize[Z] < L {
			L = cellsize[Z]
		}
	}

	// progress, progmax := 0, (1+(r2[Y]-r1[Y]))*(1+(r2[Z]-r1[Z])) // progress bar
	// done := make(chan struct{}, 3)                              // parallel calculation of one component done?

	// Start brute integration
	// 9 nested loops, does that stress you out?
	// Fortunately, the 5 inner ones usually loop over just one element.

for s := 0; s < 3; s++ { // source index Ksdxyz (parallelized over)
		go func(s int) {
			u, v, w := s, (s+1)%3, (s+2)%3 // u = direction of source (s), v & w are the orthogonal directions
			var (
				R, R2  [3]float64 // field and source cell center positions
				pole   [3]float64 // position of point charge on the surface
				points int        // counts used integration points
			)

			for z := r1[Z]; z <= r2[Z]; z++ {
				zw := wrap(z, size[Z])
				// skip one half, reconstruct from symmetry later
				// check on wrapped index instead of loop range so it also works for PBC
				// if zw > size[Z]/2 {
				// 	if s == 0 {
				// 		progress += (1 + (r2[Y] - r1[Y]))
				// 	}
				// 	continue
				// }
				R[Z] = float64(z) * cellsize[Z]

				for y := r1[Y]; y <= r2[Y]; y++ {

					// if s == 0 { // show progress of only one component
					// 	progress++
					// 	util.Progress(progress, progmax, "Calculating demag kernel")
					// }

					yw := wrap(y, size[Y])
					if yw > size[Y]/2 {
						continue
					}
					R[Y] = float64(y) * cellsize[Y]

					for x := r1[X]; x <= r2[X]; x++ {
						xw := wrap(x, size[X])
						if xw > size[X]/2 {
							continue
						}
						R[X] = float64(x) * cellsize[X]

						// choose number of integration points depending on how far we are from source.
						dx, dy, dz := delta(x)*cellsize[X], delta(y)*cellsize[Y], delta(z)*cellsize[Z]
						d := math.Sqrt(dx*dx + dy*dy + dz*dz)
						if d == 0 {
							d = L
						}
						maxSize := d / accuracy // maximum acceptable integration size

						nv := int(math.Max(cellsize[v]/maxSize, 1) + 0.5)
						nw := int(math.Max(cellsize[w]/maxSize, 1) + 0.5)
						nx := int(math.Max(cellsize[X]/maxSize, 1) + 0.5)
						ny := int(math.Max(cellsize[Y]/maxSize, 1) + 0.5)
						nz := int(math.Max(cellsize[Z]/maxSize, 1) + 0.5)
						// Stagger source and destination grids.
						// Massively improves accuracy, see note.
						nv *= 2
						nw *= 2

						// util.Assert(nv > 0 && nw > 0 && nx > 0 && ny > 0 && nz > 0)

						scale := 1 / float64(nv*nw*nx*ny*nz)
						surface := cellsize[v] * cellsize[w] // the two directions perpendicular to direction s
						charge := surface * scale
						pu1 := cellsize[u] / 2. // positive pole center
						pu2 := -pu1             // negative pole center

						// Do surface integral over source cell, accumulate  in B
						var B [3]float64
						for i := 0; i < nv; i++ {
							pv := -(cellsize[v] / 2.) + cellsize[v]/float64(2*nv) + float64(i)*(cellsize[v]/float64(nv))
							pole[v] = pv
							for j := 0; j < nw; j++ {
								pw := -(cellsize[w] / 2.) + cellsize[w]/float64(2*nw) + float64(j)*(cellsize[w]/float64(nw))
								pole[w] = pw

								// Do volume integral over destination cell
								for α := 0; α < nx; α++ {
									rx := R[X] - cellsize[X]/2 + cellsize[X]/float64(2*nx) + (cellsize[X]/float64(nx))*float64(α)

									for β := 0; β < ny; β++ {
										ry := R[Y] - cellsize[Y]/2 + cellsize[Y]/float64(2*ny) + (cellsize[Y]/float64(ny))*float64(β)

										for γ := 0; γ < nz; γ++ {
											rz := R[Z] - cellsize[Z]/2 + cellsize[Z]/float64(2*nz) + (cellsize[Z]/float64(nz))*float64(γ)
											points++

											pole[u] = pu1
											R2[X], R2[Y], R2[Z] = rx-pole[X], ry-pole[Y], rz-pole[Z]
											r := math.Sqrt(R2[X]*R2[X] + R2[Y]*R2[Y] + R2[Z]*R2[Z])
											qr := charge / (4 * math.Pi * r * r * r)
											bx := R2[X] * qr
											by := R2[Y] * qr
											bz := R2[Z] * qr

											pole[u] = pu2
											R2[X], R2[Y], R2[Z] = rx-pole[X], ry-pole[Y], rz-pole[Z]
											r = math.Sqrt(R2[X]*R2[X] + R2[Y]*R2[Y] + R2[Z]*R2[Z])
											qr = -charge / (4 * math.Pi * r * r * r)
											B[X] += (bx + R2[X]*qr) // addition ordered for accuracy
											B[Y] += (by + R2[Y]*qr)
											B[Z] += (bz + R2[Z]*qr)

										}
									}
								}
							}
						}
						for d := s; d < 3; d++ { // destination index Ksdxyz
							array[s][d][zw][yw][xw] += float32(B[d]) // += needed in case of PBC
						}
					}
				}
			}
			done <- struct{}{} // notify parallel computation of this component is done
		}(s)
	}
	// wait for all 3 components to finish
	<-done
	<-done
	<-done

	// Reconstruct skipped parts from symmetry (X)
	for z := 0; z < size[Z]; z++ {
		for y := 0; y < size[Y]; y++ {
			for x := size[X]/2 + 1; x < size[X]; x++ {
				x2 := size[X] - x
				array[X][X][z][y][x] = array[X][X][z][y][x2]
				array[X][Y][z][y][x] = -array[X][Y][z][y][x2]
				array[X][Z][z][y][x] = -array[X][Z][z][y][x2]
				array[Y][Y][z][y][x] = array[Y][Y][z][y][x2]
				array[Y][Z][z][y][x] = array[Y][Z][z][y][x2]
				array[Z][Z][z][y][x] = array[Z][Z][z][y][x2]
			}
		}
	}

	// Reconstruct skipped parts from symmetry (Y)
	for z := 0; z < size[Z]; z++ {
		for y := size[Y]/2 + 1; y < size[Y]; y++ {
			y2 := size[Y] - y
			for x := 0; x < size[X]; x++ {
				array[X][X][z][y][x] = array[X][X][z][y2][x]
				array[X][Y][z][y][x] = -array[X][Y][z][y2][x]
				array[X][Z][z][y][x] = array[X][Z][z][y2][x]
				array[Y][Y][z][y][x] = array[Y][Y][z][y2][x]
				array[Y][Z][z][y][x] = -array[Y][Z][z][y2][x]
				array[Z][Z][z][y][x] = array[Z][Z][z][y2][x]

			}
		}
	}

	// Reconstruct skipped parts from symmetry (Z)
	for z := size[Z]/2 + 1; z < size[Z]; z++ {
		z2 := size[Z] - z
		for y := 0; y < size[Y]; y++ {
			for x := 0; x < size[X]; x++ {
				array[X][X][z][y][x] = array[X][X][z2][y][x]
				array[X][Y][z][y][x] = array[X][Y][z2][y][x]
				array[X][Z][z][y][x] = -array[X][Z][z2][y][x]
				array[Y][Y][z][y][x] = array[Y][Y][z2][y][x]
				array[Y][Z][z][y][x] = -array[Y][Z][z2][y][x]
				array[Z][Z][z][y][x] = array[Z][Z][z2][y][x]
			}
		}
	}

	// for 2D these elements are zero:
	if size[Z] == 1 {
		kernel[X][Z] = nil
		kernel[Y][Z] = nil
	}
	// make result symmetric for tools that expect it so.
	kernel[Y][X] = kernel[X][Y]
	kernel[Z][X] = kernel[X][Z]
	kernel[Z][Y] = kernel[Y][Z]
	return kernel
}