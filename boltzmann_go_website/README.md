# Boltzmann CFD Simulator

A professional web-based Computational Fluid Dynamics simulator using the Lattice Boltzmann Method to simulate fluid flows.

## Requirements

- Go 1.18 or higher
- Node.js 14.x or higher
- npm


### On Windows:
```batch
# Clone the repository
git clone https://github.com/yourusername/boltzmann-cfd-simulator.git
cd boltzmann-cfd-simulator

# Run the setup script
setup.bat

# Start the server
npm start
```

Then open your browser and navigate to `http://localhost:3000`

## Project Structure

```
boltzmann-cfd-simulator/
├── main.go                  # Go simulation code
├── server.js                # Node.js Express backend server
├── package.json             # Node.js dependencies and scripts
├── index.html               # Main HTML file with modern UI
├── default_config.json      # Default configuration
├── setup.bat                # Setup script for Windows
├── README.md                # Project documentation
└── public/                  # Static files (created at runtime)
    ├── index.html           # Copied from root
    └── cfd_simulation.gif   # Generated simulation output
```

## Simulation Parameters

The simulator allows adjustment of the following parameters:

- **Canvas Dimensions**: Width and height of the visualization
- **Simulation Grid**: Resolution of the simulation (higher values = more detail, slower simulation)
- **Physics Parameters**: 
  - Velocity: Initial fluid velocity
  - Viscosity: Fluid "stickiness"
- **Simulation Control**:
  - Time Steps Per Frame: How many simulation steps between each GIF frame
  - Total Time Steps: Total length of the simulation
  - GIF Delay: Frame delay in the output GIF


## How It Works

The Lattice Boltzmann Method (LBM) is a computational fluid dynamics technique used to simulate fluid flows. Instead of solving the Navier-Stokes equations directly, LBM models fluid as a collection of particles and simulates their collisions and streaming.

The simulation consists of three main steps:
1. **Collision**: Particles at each lattice site collide, resulting in redistribution
2. **Streaming**: After collision, particles move to neighboring lattice sites
3. **Boundary Conditions**: Special rules handle obstacles and inlet/outlet conditions

## License

Open source - feel free to modify and distribute.
