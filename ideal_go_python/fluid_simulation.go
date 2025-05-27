package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
)

type FluidFlowSimulation struct {
	// Grid dimensions
	nx, ny int
	// Simulation parameters
	vInf    float64
	maxIter int
	tol     float64

	// Domain setup
	x, y             []float64
	dx, dy           float64
	X, Y             [][]float64
	psi, phi         [][]float64
	mask             [][]bool
	u, v             [][]float64
	p                [][]float64
	cylinder         *Circle
	airfoil          *Airfoil
	rotatingCylinder *RotatingCylinder
}

// Circle represents a circular object in the flow
type Circle struct {
	centerX, centerY float64
	radius           float64
}

// RotatingCylinder represents a cylinder with "circulation" in the flow
type RotatingCylinder struct {
	centerX, centerY float64
	radius           float64
	circulation      float64
}

// Airfoil represents an airfoil shape in the flow
type Airfoil struct {
	centerX, centerY float64
	chord            float64
	angleOfAttack    float64 // in degrees
	useCustomProfile bool    // whether to use custom profile or NACA equation
	circulation      float64 // circulation parameter (constant psi)
	// Parameters for NACA equation
	thicknessRatio    float64
	camber, camberPos float64
	// For custom airfoil profile
	profile [][2]float64 // normalized coordinates
}

// NewFluidFlowSimulation creates a new fluid flow simulation
func NewFluidFlowSimulation(nx, ny int, vInf float64, maxIter int, tol float64) *FluidFlowSimulation {
	sim := &FluidFlowSimulation{
		nx:      nx,
		ny:      ny,
		vInf:    vInf,
		maxIter: maxIter,
		tol:     tol,
	}

	// Setup domain
	sim.x = linspace(0, 10, nx)
	sim.y = linspace(-5, 5, ny)
	sim.dx = sim.x[1] - sim.x[0]
	sim.dy = sim.y[1] - sim.y[0]

	// Create 2D grids
	sim.X = make([][]float64, ny)
	sim.Y = make([][]float64, ny)
	sim.psi = make([][]float64, ny)
	sim.phi = make([][]float64, ny)
	sim.mask = make([][]bool, ny)
	sim.u = make([][]float64, ny)
	sim.v = make([][]float64, ny)
	sim.p = make([][]float64, ny)

	for j := 0; j < ny; j++ {
		sim.X[j] = make([]float64, nx)
		sim.Y[j] = make([]float64, nx)
		sim.psi[j] = make([]float64, nx)
		sim.phi[j] = make([]float64, nx)
		sim.mask[j] = make([]bool, nx)
		sim.u[j] = make([]float64, nx)
		sim.v[j] = make([]float64, nx)
		sim.p[j] = make([]float64, nx)

		// Initialize mask as all fluid
		for i := 0; i < nx; i++ {
			sim.mask[j][i] = true
		}

		// Create meshgrid
		for i := 0; i < nx; i++ {
			sim.X[j][i] = sim.x[i]
			sim.Y[j][i] = sim.y[j]
		}
	}

	// Set boundary conditions
	sim.setBoundaryConditions()

	return sim
}

// Sets initial boundary conditions for stream function
func (sim *FluidFlowSimulation) setBoundaryConditions() {
	// Inlet BC (x=0): u = v_inf, v = 0
	// Stream function varies linearly with y
	for j := 0; j < sim.ny; j++ {
		sim.psi[j][0] = sim.vInf * sim.y[j]
	}

	// Top boundary (y=ymax): constant psi
	for i := 0; i < sim.nx; i++ {
		sim.psi[0][i] = sim.vInf * sim.y[0]
	}

	// Bottom boundary (y=ymin): constant psi
	for i := 0; i < sim.nx; i++ {
		sim.psi[sim.ny-1][i] = sim.vInf * sim.y[sim.ny-1]
	}

	// Outlet boundary (x=xmax): zero gradient
	for j := 0; j < sim.ny; j++ {
		sim.psi[j][sim.nx-1] = sim.psi[j][sim.nx-2]
	}

	// Initialize velocity potential (phi) - uniform flow in x direction
	for i := 0; i < sim.nx; i++ {
		for j := 0; j < sim.ny; j++ {
			sim.phi[j][i] = sim.vInf * sim.x[i]
		}
	}
}

// Adds a circular object to the flow field
func (sim *FluidFlowSimulation) AddCylinder(centerX, centerY, radius float64) {
	// Mask solid points
	for i := 0; i < sim.nx; i++ {
		for j := 0; j < sim.ny; j++ {
			if math.Pow(sim.x[i]-centerX, 2)+math.Pow(sim.y[j]-centerY, 2) <= math.Pow(radius, 2) {
				sim.mask[j][i] = false // Mark as solid
			}
		}
	}

	// Store for visualization
	sim.cylinder = &Circle{
		centerX: centerX,
		centerY: centerY,
		radius:  radius,
	}
}

// Adds a circular object to the flow field with "circulation" the constant psi
func (sim *FluidFlowSimulation) AddRotatingCylinder(centerX, centerY, radius, circulation float64) {
	// Mask solid points same as regular cylinder
	for i := 0; i < sim.nx; i++ {
		for j := 0; j < sim.ny; j++ {
			if math.Pow(sim.x[i]-centerX, 2)+math.Pow(sim.y[j]-centerY, 2) <= math.Pow(radius, 2) {
				sim.mask[j][i] = false // Mark as solid
			}
		}
	}

	// Initialize stream function with circulation control (same as airfoil)
	for j := 0; j < sim.ny; j++ {
		for i := 0; i < sim.nx; i++ {
			if !sim.mask[j][i] {
				// Inside cylinder: set constant stream function value
				sim.psi[j][i] = circulation
			} else {
				// Outside cylinder: initialize with free stream flow
				sim.psi[j][i] = sim.vInf * sim.y[j]
			}
		}
	}

	// Reset boundary conditions
	sim.setBoundaryConditions()

	// Store for visualization and calculation
	sim.rotatingCylinder = &RotatingCylinder{
		centerX:     centerX,
		centerY:     centerY,
		radius:      radius,
		circulation: circulation,
	}

	fmt.Printf("Initialized cylinder stream function with circulation parameter: %.2f\n", circulation)
}

func (sim *FluidFlowSimulation) initializeAirfoilStreamFunction(centerX, centerY, circulation float64) {
	for j := 0; j < sim.ny; j++ {
		for i := 0; i < sim.nx; i++ {
			if !sim.mask[j][i] {
				// Inside airfoil: set constant stream function value (like Python's C parameter)
				sim.psi[j][i] = circulation
			} else {
				// Outside airfoil: initialize with free stream flow
				sim.psi[j][i] = sim.vInf * sim.y[j]
			}
		}
	}

	// Set boundary conditions
	// Inlet boundary
	for j := 0; j < sim.ny; j++ {
		sim.psi[j][0] = sim.vInf * sim.y[j]
	}

	// Top and bottom boundaries
	for i := 0; i < sim.nx; i++ {
		sim.psi[0][i] = sim.vInf * sim.y[0]
		sim.psi[sim.ny-1][i] = sim.vInf * sim.y[sim.ny-1]
	}

	// Outlet boundary
	for j := 0; j < sim.ny; j++ {
		sim.psi[j][sim.nx-1] = sim.psi[j][sim.nx-2]
	}

	fmt.Printf("Initialized airfoil stream function with circulation parameter: %.2f\n", circulation)
}

// AddAirfoil adds a cambered NACA 4-digit airfoil to the mask
func (sim *FluidFlowSimulation) AddAirfoil(centerX, centerY, chord, angleOfAttack, circulation float64, useCustomProfile bool, thicknessRatio, camber, camberPos float64) {
	// Convert angle from degrees to radians
	// Positive AOA means nose up, which should rotate counterclockwise in our coordinate system
	angleRad := -angleOfAttack * math.Pi / 180.0 // Negative sign added to reverse direction
	cosAngle := math.Cos(angleRad)
	sinAngle := math.Sin(angleRad)

	// Initialize custom profile if needed
	var profile [][2]float64
	if useCustomProfile {
		// NACA 24012 airfoil profile from NACAs website
		profile = [][2]float64{
			{1.000034, 0.001260}, {0.998499, 0.001517}, {0.993901, 0.002286}, {0.986271, 0.003554},
			{0.975654, 0.005302}, {0.962115, 0.007503}, {0.945737, 0.010126}, {0.926621, 0.013135},
			{0.904884, 0.016488}, {0.880660, 0.020143}, {0.854096, 0.024054}, {0.825357, 0.028176},
			{0.794619, 0.032461}, {0.762070, 0.036859}, {0.727912, 0.041322}, {0.692353, 0.045797},
			{0.655613, 0.050232}, {0.617917, 0.054570}, {0.579496, 0.058755}, {0.540587, 0.062726},
			{0.501429, 0.066422}, {0.462262, 0.069781}, {0.423325, 0.072741}, {0.384859, 0.075241},
			{0.347100, 0.077226}, {0.310278, 0.078646}, {0.274563, 0.079453}, {0.239831, 0.079489},
			{0.206317, 0.078497}, {0.174345, 0.076299}, {0.144248, 0.072811}, {0.116354, 0.068044},
			{0.090970, 0.062103}, {0.068368, 0.055171}, {0.048771, 0.047489}, {0.032351, 0.039329},
			{0.019219, 0.030969}, {0.009432, 0.022669}, {0.002997, 0.014646}, {-0.000123, 0.007059},
			{0.000000, 0.000000}, {0.003205, -0.006286}, {0.009315, -0.011612}, {0.018198, -0.016059},
			{0.029725, -0.019740}, {0.043770, -0.022790}, {0.060222, -0.025349}, {0.078992, -0.027560},
			{0.100013, -0.029550}, {0.123240, -0.031426}, {0.148645, -0.033265}, {0.176207, -0.035103},
			{0.205898, -0.036930}, {0.237671, -0.038675}, {0.271447, -0.040202}, {0.307039, -0.041310},
			{0.343883, -0.041880}, {0.381695, -0.041935}, {0.420240, -0.041514}, {0.459279, -0.040660},
			{0.498571, -0.039420}, {0.537872, -0.037842}, {0.576938, -0.035976}, {0.615529, -0.033871},
			{0.653404, -0.031573}, {0.690330, -0.029128}, {0.726079, -0.026578}, {0.760428, -0.023965},
			{0.793166, -0.021330}, {0.824091, -0.018710}, {0.853011, -0.016145}, {0.879746, -0.013673},
			{0.904133, -0.011331}, {0.926019, -0.009155}, {0.945270, -0.007183}, {0.961765, -0.005448},
			{0.975403, -0.003980}, {0.986099, -0.002808}, {0.993787, -0.001953}, {0.998419, -0.001434},
			{0.999966, -0.001260},
		}
	}

	// For custom profile, find the bounding box
	minX, maxX, minY, maxY := 0.0, 0.0, 0.0, 0.0
	if useCustomProfile {
		minX, maxX = profile[0][0], profile[0][0]
		minY, maxY = profile[0][1], profile[0][1]

		for _, point := range profile {
			if point[0] < minX {
				minX = point[0]
			}
			if point[0] > maxX {
				maxX = point[0]
			}
			if point[1] < minY {
				minY = point[1]
			}
			if point[1] > maxY {
				maxY = point[1]
			}
		}
	}

	// Loop through grid points
	for i := 0; i < sim.nx; i++ {
		for j := 0; j < sim.ny; j++ {
			// Convert to airfoil-centered coordinates
			xRel := sim.x[i] - centerX
			yRel := sim.y[j] - centerY

			// Rotate coordinates based on angle of attack
			xRot := xRel*cosAngle + yRel*sinAngle
			yRot := -xRel*sinAngle + yRel*cosAngle

			// Check if point is within airfoil shape
			if useCustomProfile {
				// For custom airfoil profile
				if xRot >= 0 && xRot <= chord {
					// Normalize to profile coordinates
					xNorm := xRot / chord

					// Find the corresponding y values by linear interpolation
					var yUpper, yLower float64

					// Find upper and lower profile points at this x position
					foundUpper, foundLower := false, false

					for k := 0; k < len(profile)-1; k++ {
						// Check if this segment contains our x value
						x1, y1 := profile[k][0], profile[k][1]
						x2, y2 := profile[k+1][0], profile[k+1][1]

						if xNorm >= x1 && xNorm <= x2 || xNorm >= x2 && xNorm <= x1 {
							// Linear interpolation
							t := (xNorm - x1) / (x2 - x1)
							yInterp := y1 + t*(y2-y1)

							// Determine if this is upper or lower surface
							if yInterp >= 0 && !foundUpper {
								yUpper = yInterp * chord
								foundUpper = true
							} else if yInterp <= 0 && !foundLower {
								yLower = yInterp * chord
								foundLower = true
							}

							// Exit if we found both surfaces
							if foundUpper && foundLower {
								break
							}
						}
					}

					// Check if the point is inside the airfoil
					if foundUpper && foundLower && yRot >= yLower && yRot <= yUpper {
						sim.mask[j][i] = false
					}
				}
			} else {
				// For NACA 4-digit airfoil using the equation method
				if xRot >= 0 && xRot <= chord {
					xNorm := xRot / chord

					// Camber line z_c and its slope dzc_dx
					var zC, dzcDx float64
					if xNorm < camberPos {
						zC = (camber / math.Pow(camberPos, 2)) * (2*camberPos*xNorm - math.Pow(xNorm, 2))
						dzcDx = (2 * camber / math.Pow(camberPos, 2)) * (camberPos - xNorm)
					} else {
						zC = (camber / math.Pow(1-camberPos, 2)) * ((1 - 2*camberPos) + 2*camberPos*xNorm - math.Pow(xNorm, 2))
						dzcDx = (2 * camber / math.Pow(1-camberPos, 2)) * (camberPos - xNorm)
					}

					theta := math.Atan(dzcDx)

					// Thickness distribution
					yt := 5 * thicknessRatio * chord * (0.2969*math.Sqrt(xNorm) -
						0.1260*xNorm -
						0.3516*math.Pow(xNorm, 2) +
						0.2843*math.Pow(xNorm, 3) -
						0.1015*math.Pow(xNorm, 4))

					// Upper and lower surface positions
					yUpper := zC + yt*math.Cos(theta)
					yLower := zC - yt*math.Cos(theta)

					if yRot >= yLower && yRot <= yUpper {
						sim.mask[j][i] = false
					}
				}
			}
		}
	}
	// Initialize stream function with circulation control
	sim.initializeAirfoilStreamFunction(centerX, centerY, circulation)

	sim.airfoil = &Airfoil{
		centerX:          centerX,
		centerY:          centerY,
		chord:            chord,
		angleOfAttack:    angleOfAttack,
		circulation:      circulation,
		useCustomProfile: useCustomProfile,
		thicknessRatio:   thicknessRatio,
		camber:           camber,
		camberPos:        camberPos,
		profile:          profile,
	}
}

// SolveStreamFunction solves the Laplace equation for the stream function
func (sim *FluidFlowSimulation) SolveStreamFunction() {
	psiOld := make([][]float64, sim.ny)
	for j := 0; j < sim.ny; j++ {
		psiOld[j] = make([]float64, sim.nx)
		copy(psiOld[j], sim.psi[j])
	}

	// Use multiple goroutines for parallelization
	numCPU := runtime.NumCPU()

	// Relaxation parameter for SOR method (optimized for faster convergence)
	omega := 1.8

	for iterCount := 0; iterCount < sim.maxIter; iterCount++ {
		// Use parallel processing for the interior points
		var wg sync.WaitGroup
		rowsPerGoroutine := (sim.ny - 2) / numCPU
		if rowsPerGoroutine < 1 {
			rowsPerGoroutine = 1
		}

		for cpu := 0; cpu < numCPU; cpu++ {
			wg.Add(1)
			startRow := 1 + cpu*rowsPerGoroutine
			endRow := startRow + rowsPerGoroutine
			if cpu == numCPU-1 {
				endRow = sim.ny - 1 // ensure we cover all rows
			}
			if endRow > sim.ny-1 {
				endRow = sim.ny - 1
			}

			go func(startRow, endRow int) {
				defer wg.Done()
				// Gauss-Seidel iteration with SOR for interior points
				for j := startRow; j < endRow; j++ {
					for i := 1; i < sim.nx-1; i++ {
						if sim.mask[j][i] {
							psiNew := 0.25 * (sim.psi[j][i+1] + sim.psi[j][i-1] +
								sim.psi[j+1][i] + sim.psi[j-1][i])
							sim.psi[j][i] = (1-omega)*sim.psi[j][i] + omega*psiNew
						}
					}
				}
			}(startRow, endRow)
		}
		wg.Wait()

		// Outlet boundary condition (zero gradient)
		for j := 0; j < sim.ny; j++ {
			sim.psi[j][sim.nx-1] = sim.psi[j][sim.nx-2]
		}

		// Check convergence
		maxDiff := 0.0
		for j := 0; j < sim.ny; j++ {
			for i := 0; i < sim.nx; i++ {
				diff := math.Abs(sim.psi[j][i] - psiOld[j][i])
				if diff > maxDiff {
					maxDiff = diff
				}
				psiOld[j][i] = sim.psi[j][i]
			}
		}

		if maxDiff < sim.tol {
			fmt.Printf("Stream function converged after %d iterations\n", iterCount)
			break
		}

		// Print progress for long computations
		if iterCount%1000 == 0 {
			fmt.Printf("Iteration: %d, Max difference: %e\n", iterCount, maxDiff)
		}
	}

	// Set inlet velocity
	for j := 0; j < sim.ny; j++ {
		sim.u[j][0] = sim.vInf
		sim.v[j][0] = 0
	}

	// Calculate velocity field
	sim.calculateVelocityField()
}

// Calculates velocity from stream function using central differences
func (sim *FluidFlowSimulation) calculateVelocityField() {
	// Use goroutines for parallel processing
	var wg sync.WaitGroup
	numCPU := runtime.NumCPU()
	rowsPerGoroutine := (sim.ny - 2) / numCPU
	if rowsPerGoroutine < 1 {
		rowsPerGoroutine = 1
	}

	for cpu := 0; cpu < numCPU; cpu++ {
		wg.Add(1)
		startRow := 1 + cpu*rowsPerGoroutine
		endRow := startRow + rowsPerGoroutine
		if cpu == numCPU-1 {
			endRow = sim.ny - 1 // ensure we cover all rows
		}
		if endRow > sim.ny-1 {
			endRow = sim.ny - 1
		}

		go func(startRow, endRow int) {
			defer wg.Done()
			// Calculate velocity for interior points
			for j := startRow; j < endRow; j++ {
				for i := 1; i < sim.nx-1; i++ {
					if sim.mask[j][i] {
						// u = dpsi/dy
						sim.u[j][i] = (sim.psi[j+1][i] - sim.psi[j-1][i]) / (2 * sim.dy)
						// v = -dpsi/dx
						sim.v[j][i] = -(sim.psi[j][i+1] - sim.psi[j][i-1]) / (2 * sim.dx)
					}
				}
			}
		}(startRow, endRow)
	}
	wg.Wait()

	// One-sided differences for walls
	for j := 1; j < sim.ny-1; j++ {
		if sim.mask[j][sim.nx-1] {
			// Forward difference for u
			sim.u[j][sim.nx-1] = sim.u[j][sim.nx-2]
			// Backward difference for v
			sim.v[j][sim.nx-1] = -(sim.psi[j][sim.nx-1] - sim.psi[j][sim.nx-2]) / sim.dx
		}
	}

	// Calculate pressure distribution using Bernoulli's equation
	vInfSquared := sim.vInf * sim.vInf
	pInf := 0.0 // reference pressure
	rho := 1.0  // density (assumed constant)

	for j := 0; j < sim.ny; j++ {
		for i := 0; i < sim.nx; i++ {
			vSquared := sim.u[j][i]*sim.u[j][i] + sim.v[j][i]*sim.v[j][i]
			// P/ρ + V²/2 = P_inf/ρ + V_inf²/2
			sim.p[j][i] = pInf + 0.5*rho*(vInfSquared-vSquared)
		}
	}
}

// Calculates both lift and drag coefficients by integrating the pressure distribution around the airfoil surface
func (sim *FluidFlowSimulation) CalculateAerodynamicCoefficients() (float64, float64) {
	// Only proceed if we have an airfoil
	if sim.airfoil == nil {
		fmt.Println("No airfoil present in simulation")
		return 0.0, 0.0
	}

	// Reference values
	rho := 1.0                              // Density (assumed constant)
	qInf := 0.5 * rho * sim.vInf * sim.vInf // Dynamic pressure
	chordLength := sim.airfoil.chord

	// Get angle of attack in radians
	angleRad := sim.airfoil.angleOfAttack * math.Pi / 180.0

	// Initialize lift and drag forces
	liftForce := 0.0
	dragForce := 0.0

	// Find surface points around airfoil (points adjacent to solid cells)
	surfacePoints := [][3]float64{} // [x, y, pressure]

	// Loop through grid points to find surface points
	for j := 1; j < sim.ny-1; j++ {
		for i := 1; i < sim.nx-1; i++ {
			// Check if this is a fluid point adjacent to a solid point
			if sim.mask[j][i] && (!sim.mask[j+1][i] || !sim.mask[j-1][i] ||
				!sim.mask[j][i+1] || !sim.mask[j][i-1]) {
				// This is a fluid point adjacent to the airfoil surface
				// Get pressure at this point
				pressure := sim.p[j][i]

				// Store coordinates and pressure
				surfacePoints = append(surfacePoints, [3]float64{sim.x[i], sim.y[j], pressure})
			}
		}
	}

	// Sort surface points clockwise around the airfoil center
	centerX, centerY := sim.airfoil.centerX, sim.airfoil.centerY
	sort.Slice(surfacePoints, func(i, j int) bool {
		// Calculate angles from center to each point
		angleI := math.Atan2(surfacePoints[i][1]-centerY, surfacePoints[i][0]-centerX)
		angleJ := math.Atan2(surfacePoints[j][1]-centerY, surfacePoints[j][0]-centerX)
		return angleI < angleJ
	})

	// Integrate pressure around the airfoil surface
	for i := 0; i < len(surfacePoints); i++ {
		p1 := surfacePoints[i]
		p2 := surfacePoints[(i+1)%len(surfacePoints)]

		// Calculate pressure force on this segment
		avgPressure := (p1[2] + p2[2]) / 2.0

		// Calculate normal vector to the segment
		dx := p2[0] - p1[0]
		dy := p2[1] - p1[1]
		length := math.Sqrt(dx*dx + dy*dy)

		if length > 0 {
			// Normal vector (perpendicular to segment, pointing outward)
			nx := dy / length
			ny := -dx / length

			// Force is pressure times area (length in 2D)
			force := avgPressure * length

			liftForce -= force * (nx*math.Sin(angleRad) - ny*math.Cos(angleRad))
			dragForce -= force * (nx*math.Cos(angleRad) + ny*math.Sin(angleRad))
		}
	}

	// Calculate lift coefficient
	Cl := liftForce / (qInf * chordLength)
	Cd := dragForce / (qInf * chordLength)

	fmt.Printf("Calculated lift coefficient (Cl): %.4f at %.1f° angle of attack\n",
		Cl, sim.airfoil.angleOfAttack)
	fmt.Printf("Calculated drag coefficient (Cd): %.4f\n", Cd)
	fmt.Printf("Lift-to-drag ratio (L/D): %.2f\n", Cl/Cd)

	return Cl, Cd
}

// Saves plots of simulation results
func (sim *FluidFlowSimulation) SaveResults() {
	// Create data directory if it doesn't exist
	dataDir := "data"
	err := os.MkdirAll(dataDir, 0755)
	if err != nil {
		panic(err)
	}

	// Stream function plot
	sim.savePlot("stream_function", sim.psi, "Stream Function")

	// Velocity magnitude
	velMag := make([][]float64, sim.ny)
	for j := 0; j < sim.ny; j++ {
		velMag[j] = make([]float64, sim.nx)
		for i := 0; i < sim.nx; i++ {
			velMag[j][i] = math.Sqrt(sim.u[j][i]*sim.u[j][i] + sim.v[j][i]*sim.v[j][i])
		}
	}
	sim.savePlot("velocity_magnitude", velMag, "Velocity Magnitude")

	// Pressure plot
	sim.savePlot("pressure", sim.p, "Pressure Field")

	// Create velocity field quiver plot
	sim.saveVelocityPlot("velocity_field")
	// Save mask data
	sim.saveMaskData("mask")
}

// Saves the fluid/solid mask (1 = fluid, 0 = solid)
func (sim *FluidFlowSimulation) saveMaskData(filename string) {
	dataDir := "data"
	filepath := filepath.Join(dataDir, filename+".data")

	f, err := os.Create(filepath)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	for j := 0; j < sim.ny; j++ {
		for i := 0; i < sim.nx; i++ {
			maskVal := 0.0
			if sim.mask[j][i] {
				maskVal = 1.0
			}
			fmt.Fprintf(f, "%f %f %f\n", sim.x[i], sim.y[j], maskVal)
		}
		fmt.Fprintln(f)
	}

	fmt.Printf("Mask data saved to %s.data\n", filename)
}

// Saves a contour plot for a given field
func (sim *FluidFlowSimulation) savePlot(filename string, field [][]float64, title string) {
	// Save data to a text file instead
	dataDir := "data"
	filepath := filepath.Join(dataDir, filename+".data")

	f, err := os.Create(filepath)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	// Write field data
	for j := 0; j < sim.ny; j++ {
		for i := 0; i < sim.nx; i++ {
			fmt.Fprintf(f, "%f %f %f\n", sim.x[i], sim.y[j], field[j][i])
		}
		fmt.Fprintln(f)
	}

	fmt.Printf("Data for %s saved to %s.data\n", title, filename)
}

// Saves vectors to a data file
func (sim *FluidFlowSimulation) saveVelocityPlot(filename string) {
	dataDir := "data"
	filepath := filepath.Join(dataDir, filename+".data")

	f, err := os.Create(filepath)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	// Write velocity vector data
	skip := 4 // Skip some points for better visualization
	for j := 0; j < sim.ny; j += skip {
		for i := 0; i < sim.nx; i += skip {
			if sim.mask[j][i] {
				fmt.Fprintf(f, "%f %f %f %f\n",
					sim.X[j][i], sim.Y[j][i],
					sim.u[j][i], sim.v[j][i])
			}
		}
	}

	fmt.Printf("Velocity data saved to %s.data\n", filename)
}

// Utility functions
func linspace(start, end float64, num int) []float64 {
	result := make([]float64, num)
	step := (end - start) / float64(num-1)
	for i := range result {
		result[i] = start + float64(i)*step
	}
	return result
}

// Executes the Python visualization script
func runVisualization() {
	// Check if Python is installed
	pythonCmd := "python"
	if _, err := exec.LookPath(pythonCmd); err != nil {
		// Try python3 if python command not found
		pythonCmd = "python3"
		if _, err := exec.LookPath(pythonCmd); err != nil {
			fmt.Println("Error: Python not found. Please install Python and try again.")
			return
		}
	}

	// Set up the command
	cmd := exec.Command(pythonCmd, "visualize_data.py")

	// Get the output
	output, err := cmd.CombinedOutput()
	outputStr := string(output)

	if err != nil {
		fmt.Printf("Error running visualization: %s\n", err)

		// Check for specific common errors and provide helpful instructions
		if strings.Contains(outputStr, "No module named 'numpy'") ||
			strings.Contains(outputStr, "No module named 'matplotlib'") {
			fmt.Println("\nIt looks like you're missing some required Python packages.")
			fmt.Println("Please install the required packages with the following command:")
			fmt.Println("\npip install numpy matplotlib")
			fmt.Println("\nOr if you're using Python 3:")
			fmt.Println("pip3 install numpy matplotlib")
			fmt.Println("\nAfter installing the packages, run the visualization manually with:")
			fmt.Println("python visualize_data.py")
		} else {
			fmt.Println("Error output:", outputStr)
		}
		return
	}

	fmt.Println("Visualization completed successfully!")
	fmt.Println(outputStr)
}

// Shows the simulation options and gets user selection
func displayMenu() (int, bool, string, int, float64) {
	fmt.Println("\n=== Fluid Flow Simulation Options ===")
	fmt.Println("1. Regular Cylinder")
	fmt.Println("2. Rotating Cylinder (with circulation)")
	fmt.Println("3. NACA 24012 Airfoil (custom profile)")
	fmt.Println("5. Exit")
	fmt.Println("\nAdditional parameters:")
	fmt.Println("- Add 'auto' to automatically run visualization after simulation")
	fmt.Println("- For airfoil options (3-4), you can specify angle of attack in degrees")
	fmt.Println("- Add 'res=N' to set grid resolution (e.g., 'res=100' for 100x100 grid)")
	fmt.Println("  Default resolution is 150. Higher values give better accuracy but take longer.")
	fmt.Println("- Add 'psi=N' to set circulation parameter for airfoils and rotating cylinders")
	fmt.Println("  Default circulation is -3 for rotating cylinder, 0 for airfoils")
	fmt.Println("\nExamples:")
	fmt.Println("  '3 auto 10 res=200' - NACA 24012 at 10 degrees with 200x200 grid and auto visualization")
	fmt.Println("  '2 psi=-5 auto' - Rotating cylinder with circulation -5 and auto visualization")
	fmt.Println("  '3 auto 10 res=600 psi=-3' - NACA 24012 at 10 degrees, 600x600 grid, circulation -3")
	fmt.Print("\nEnter your choice: ")

	reader := bufio.NewReader(os.Stdin)
	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(input)

	// Parse the input
	parts := strings.Fields(input)
	if len(parts) == 0 {
		fmt.Println("Invalid selection. Please try again.")
		return displayMenu() // recursively call until valid input
	}

	// Check if auto option is included
	autoVisualize := false
	params := ""
	resolution := 150  // Default resolution
	circulation := 0.0 // Default circulation (will be set based on object type)

	// Get the choice first
	choiceStr := parts[0]
	choice, err := strconv.Atoi(choiceStr)
	if err != nil || choice < 1 || choice > 5 {
		fmt.Println("Invalid selection. Please enter a number between 1 and 5.")
		return displayMenu() // recursively call until valid input
	}

	// Process remaining parts
	for i := 1; i < len(parts); i++ {
		part := strings.ToLower(parts[i])

		if part == "auto" {
			autoVisualize = true
		} else if strings.HasPrefix(part, "res=") {
			// Extract resolution value
			resStr := strings.TrimPrefix(part, "res=")
			res, err := strconv.Atoi(resStr)
			if err == nil && res > 0 {
				resolution = res
			} else {
				fmt.Printf("Invalid resolution format: '%s'. Using default (150x150).\n", part)
			}
		} else if strings.HasPrefix(part, "psi=") {
			// Extract circulation value
			psiStr := strings.TrimPrefix(part, "psi=")
			psi, err := strconv.ParseFloat(psiStr, 64)
			if err == nil {
				circulation = psi
			} else {
				fmt.Printf("Invalid circulation format: '%s'. Using default circulation.\n", part)
			}
		} else {
			// Collect additional parameters (like angle of attack)
			if params != "" {
				params += " "
			}
			params += parts[i]
		}
	}

	return choice, autoVisualize, params, resolution, circulation
}

func main() {
	// Display menu and get user choice
	choice, autoVisualize, params, resolution, circulation := displayMenu()

	// Exit if user chose option 5
	if choice == 5 {
		fmt.Println("Exiting program.")
		return
	}

	// Set default circulation values based on object type if not specified by user
	circulationSet := false
	for _, part := range strings.Fields(strings.ToLower(fmt.Sprintf("%v", params))) {
		if strings.Contains(part, "psi=") {
			circulationSet = true
			break
		}
	}

	// Check if circulation was set via psi= parameter in the original input
	if circulation == 0.0 && !circulationSet {
		// Set default circulation based on object type
		switch choice {
		case 2: // Rotating cylinder
			circulation = -3.0
		case 3, 4: // Airfoils
			circulation = 0.0
		default:
			circulation = 0.0
		}
	}

	// Create simulation with appropriate parameters
	fmt.Printf("Creating simulation with %dx%d grid resolution...\n", resolution, resolution)
	// Calculate max iterations based on resolution (higher resolution needs more iterations)
	maxIter := 50000
	if resolution > 150 {
		// Scale iterations up for higher resolutions
		maxIter = int(float64(maxIter) * math.Pow(float64(resolution)/150.0, 1.5))
	}

	sim := NewFluidFlowSimulation(resolution, resolution, 1.0, maxIter, 1e-8)

	angleOfAttack := 0.0
	if (choice == 3 || choice == 4) && params != "" {
		angle, err := strconv.ParseFloat(params, 64)
		if err == nil {
			angleOfAttack = angle
			fmt.Printf("Setting angle of attack to %.1f degrees\n", angleOfAttack)
		} else {
			fmt.Printf("Could not parse angle of attack '%s', using default (0 degrees)\n", params)
		}
	}

	// Setup based on user selection
	switch choice {
	case 1:
		fmt.Println("Running simulation with regular cylinder...")
		sim.AddCylinder(5.0, 0.0, 1.0)
	case 2:
		fmt.Printf("Running simulation with rotating cylinder (circulation: %.2f)...\n", circulation)
		// Parameters: centerX, centerY, radius, circulation
		sim.AddRotatingCylinder(5.0, 0.0, 1.0, circulation)
	case 3:
		fmt.Printf("Running simulation with NACA airfoil at %.1f° angle of attack (circulation: %.2f)...\n", angleOfAttack, circulation)
		// Parameters: centerX, centerY, chord, angleOfAttack, circulation, useCustomProfile, thickness, camber, camberPos
		// Note: Positive AOA means nose up (leading edge higher than trailing edge)
		sim.AddAirfoil(2.5, 0.0, 7.0, angleOfAttack, circulation, true, 0.12, 0.02, 0.4)
	}

	// Solve and visualize
	fmt.Printf("Solving stream function with up to %d iterations. This may take a few minutes...\n", maxIter)
	sim.SolveStreamFunction()
	fmt.Println("Saving results...")
	sim.SaveResults()

	// Calculate lift coefficient if this is an airfoil simulation
	if choice == 3 || choice == 4 {
		cl, cd := sim.CalculateAerodynamicCoefficients()
		fmt.Printf("Lift coefficient (Cl): %.4f\n", cl)
		fmt.Printf("Drag coefficient (Cd): %.4f\n", cd)
	}

	fmt.Println("\nSimulation completed. Results saved to data directory.")

	// Automatically run visualization if requested
	if autoVisualize {
		fmt.Println("Automatically running visualization...")
		runVisualization()
	} else {
		fmt.Println("Run the Python visualizer with: python visualize_data.py")
		fmt.Println("\nNote: The visualization requires numpy, scipy, tqdm and matplotlib.")
		fmt.Println("If you haven't installed them, run:")
		fmt.Println("pip install numpy matplotlib scipy tqdm")
	}
}
