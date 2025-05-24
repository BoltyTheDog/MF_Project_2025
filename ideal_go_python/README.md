# Fluid Flow Simulation

A 2D computational fluid dynamics (CFD) simulation written in Go that solves potential flow around various objects including cylinders, rotating cylinders, and NACA airfoils.

## Features

- **Multiple simulation types:**
  - Regular cylinder flow
  - Rotating cylinder with circulation (Magnus effect)
  - NACA airfoil analysis
  - Custom NACA 24012 airfoil profile

## Requirements

### Go Dependencies
```bash
go mod init fluid_simulation
go get gonum.org/v1/plot
go get gonum.org/v1/plot/plotter
go get gonum.org/v1/plot/vg
```

### Python Dependencies (for visualization)
```bash
pip install numpy matplotlib scipy tqdm
```

## Quick Start

1. **Compile and run the simulation:**
```bash
go run fluid_simulation.go
```

2. **Choose simulation type from the menu:**
   - `1` - Regular cylinder
   - `2` - Rotating cylinder 
   - `3` - NACA airfoil
   - `4` - NACA 24012 airfoil
   - `5` - Exit

3. **Optional parameters:**
   - Add `auto` to automatically run visualization
   - Specify angle of attack for airfoils (e.g., `3 5` for 5° AOA)
   - Set grid resolution with `res=N` (e.g., `res=200`)

4. **Run visualization:**
```bash
python visualize_data.py
```

## Examples

```bash
# NACA airfoil at 10° angle of attack with auto visualization
3 auto 10

# High-resolution NACA 24012 at 5° AOA
4 5 res=300

# Rotating cylinder with automatic visualization
2 auto
```

## Output

The simulation generates:
- **Data files** in `data/` directory (stream function, velocity, pressure)
- **Visualizations** in `visuals/` directory:
  - `velocity_magnitude.png` - Flow speed distribution
  - `pressure_field.png` - Pressure contours
  - `pressure_coefficient.png` - Dimensionless pressure coefficient
  - `stream_function.png` - Stream function contours
  - `velocity_vectors.png` - Velocity vector field
  - `streamlines.png` - Flow streamlines

## License

Open source - feel free to modify and distribute.