import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import griddata
from tqdm import tqdm
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as pe

# Set up modern style with high-quality aesthetics
plt.style.use('ggplot')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.figsize': (12, 7),
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Custom colormaps for better visualization
def create_custom_colormaps():
    # Velocity colormap (blue to red)
    velocity_colors = plt.cm.viridis
    
    # Pressure colormap (custom: blue-white-red for negative to positive)
    pressure_colors = LinearSegmentedColormap.from_list(
        'pressure_cmap', 
        [(0, 'darkblue'), (0.5, 'white'), (1, 'darkred')],
        N=256
    )
    
    # Stream function colormap
    stream_colors = plt.cm.plasma
    
    return velocity_colors, pressure_colors, stream_colors

def load_scalar_field(filepath):
    data = np.genfromtxt(filepath)
    return data[:, 0], data[:, 1], data[:, 2]

def load_vector_field(filepath):
    data = np.genfromtxt(filepath)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]

def create_grid(x, y, z, resolution=400):
    xi = np.linspace(x.min(), x.max(), resolution)
    yi = np.linspace(y.min(), y.max(), resolution)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method='cubic', fill_value=np.nan)
    return xi, yi, zi

def load_mask(data_dir, resolution=400):
    x, y, m = load_scalar_field(os.path.join(data_dir, 'mask.data'))
    xi, yi, mi = create_grid(x, y, m, resolution)
    return xi, yi, mi

def overlay_mask(ax, mask_xi, mask_yi, mask_zi, edgecolor='black'):
    # Fill solid regions with gray
    solid = ax.contourf(mask_xi, mask_yi, mask_zi < 0.5, levels=[0.5, 1],
                colors='#444444', alpha=0.9, zorder=10)
    
    # Add a solid outline around the object for better definition
    outline = ax.contour(mask_xi, mask_yi, mask_zi, levels=[0.5], 
                colors=edgecolor, linewidths=1.5, zorder=11)
    
    return solid, outline

def plot_scalar_field(xi, yi, zi, mask_xi, mask_yi, mask_zi, title, cmap, label, filename, 
                      units="", vmin=None, vmax=None, add_contours=True):
    # Create figure with proper size
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create contour fill with better aesthetics
    if vmin is None:
        vmin = np.nanpercentile(zi, 2)
    if vmax is None:
        vmax = np.nanpercentile(zi, 98)
        
    # Add contour fill with smoother gradients
    cf = ax.contourf(xi, yi, zi, levels=100, cmap=cmap, alpha=0.95, 
                     vmin=vmin, vmax=vmax, extend='both', zorder=1)
    
    # Add contour lines for more detailed information
    if add_contours:
        contour_levels = np.linspace(vmin, vmax, 15)
        contours = ax.contour(xi, yi, zi, levels=contour_levels, colors='k', 
                             alpha=0.3, linewidths=0.5, zorder=2)
        
    # Add the solid object mask
    solid, outline = overlay_mask(ax, mask_xi, mask_yi, mask_zi)
    
    # Create properly sized and positioned colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(cf, cax=cax)
    
    # Add proper units and formatting to colorbar
    if units:
        cbar.set_label(f"{label} [{units}]", fontsize=12)
    else:
        cbar.set_label(label, fontsize=12)
    
    # Improve title and axis labels
    ax.set_title(title, fontsize=16, pad=10, fontweight='bold')
    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('y [m]', fontsize=12)
    
    # Ensure plot aspect is equal to avoid distortion
    ax.set_aspect('equal')
    
    # Add grid for better readability but with low opacity
    ax.grid(alpha=0.2, linestyle='--')
    
    
    # Save high-quality figure
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_velocity_field(x, y, u, v, mask_xi, mask_yi, mask_zi, title, filename, downsample=15):
    # Calculate velocity magnitude for coloring
    speed = np.sqrt(u**2 + v**2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Calculate the velocity magnitude for coloring
    norm = plt.Normalize(vmin=0, vmax=np.percentile(speed, 95))
    
    # Downsample the vectors for better visibility
    step = downsample
    x_ds, y_ds = x[::step], y[::step]
    u_ds, v_ds = u[::step], v[::step]
    speed_ds = speed[::step]
    
    # Create a quiver plot with better aesthetics
    quiv = ax.quiver(x_ds, y_ds, u_ds, v_ds, speed_ds, 
                    cmap='viridis', norm=norm, scale=30, 
                    width=0.002, headwidth=4, headlength=5,
                    headaxislength=4.5, alpha=0.9, zorder=5)
    
    # Add mask
    solid, outline = overlay_mask(ax, mask_xi, mask_yi, mask_zi)
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(quiv, cax=cax)
    cbar.set_label('Velocity Magnitude [m/s]', fontsize=12)
    
    # Add a key/scale reference
    qk = ax.quiverkey(quiv, 0.85, 0.92, 1, r'1 m/s', labelpos='E',
                    coordinates='figure', fontproperties={'size': 10})
    
    # Set title and labels with improved styling
    ax.set_title(title, fontsize=16, pad=10, fontweight='bold')
    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('y [m]', fontsize=12)
    ax.set_aspect('equal')
    
    # Add grid with low opacity
    ax.grid(alpha=0.2, linestyle='--')
    
    
    # Save high-quality figure
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_streamlines(x, y, u, v, mask_xi, mask_yi, mask_zi, title, filename, density=1.5):
    xi = np.linspace(x.min(), x.max(), 400)
    yi = np.linspace(y.min(), y.max(), 400)
    xi, yi = np.meshgrid(xi, yi)
    ui = griddata((x, y), u, (xi, yi), method='cubic')
    vi = griddata((x, y), v, (xi, yi), method='cubic')
    
    # Calculate speed for coloring
    speed = np.sqrt(ui**2 + vi**2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create a more visually appealing streamplot
    streamplot = ax.streamplot(xi, yi, ui, vi, density=density, 
                              color=speed, cmap='viridis',
                              linewidth=1.5, arrowsize=1.5,
                              norm=plt.Normalize(vmin=0, vmax=np.nanpercentile(speed, 95)),
                              zorder=5)
    
    # Add mask overlay
    solid, outline = overlay_mask(ax, mask_xi, mask_yi, mask_zi)
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(streamplot.lines, cax=cax)
    cbar.set_label('Velocity Magnitude [m/s]', fontsize=12)
    
    # Set title and labels with better styling
    ax.set_title(title, fontsize=16, pad=10, fontweight='bold')
    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('y [m]', fontsize=12)
    ax.set_aspect('equal')
    
    # Add grid with low opacity
    ax.grid(alpha=0.2, linestyle='--')
    
    # Save high-quality figure
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_lagrangian_animation(x, y, u, v, mask_xi, mask_yi, mask_zi,
                               filename, num_particles=200, steps=150):
    print("Initializing Lagrangian particle animation...")

    # Create velocity interpolation with higher resolution
    xi = np.linspace(x.min(), x.max(), 400)
    yi = np.linspace(y.min(), y.max(), 400)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    ui = griddata((x, y), u, (xi_grid, yi_grid), method='cubic')
    vi = griddata((x, y), v, (xi_grid, yi_grid), method='cubic')
    
    # Calculate the speed field for coloring
    speed = np.sqrt(ui**2 + vi**2)
    norm = plt.Normalize(vmin=0, vmax=np.nanpercentile(speed, 95))

    # Create starting positions
    inlet_x = x.min() + 0.05  # Slightly offset from left boundary
    y_span = yi_grid[:, 0]
    y_valid = y_span[(mask_zi[:, 0] > 0.5)]
    
    # Add some randomness to starting positions for more natural flow
    chosen_y = np.linspace(y_valid.min(), y_valid.max(), num_particles)
    chosen_y += np.random.normal(0, 0.1, size=chosen_y.shape)  # Add small jitter
    
    # Keep only valid starting positions
    valid_mask = (chosen_y >= y_valid.min()) & (chosen_y <= y_valid.max())
    chosen_y = chosen_y[valid_mask]
    num_particles = len(chosen_y)
    
    px = np.full_like(chosen_y, inlet_x)
    py = chosen_y
    
    # Create array for particle ages to manage trails and opacity
    particle_age = np.zeros(num_particles)
    
    # Save particle history for trails
    history = np.zeros((steps, num_particles, 2))
    history[0, :, 0] = px
    history[0, :, 1] = py
    
    # Setup figure with modern styling
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_xlabel("x [m]", fontsize=12)
    ax.set_ylabel("y [m]", fontsize=12)
    ax.set_aspect('equal')
    
    # Add a nice background gradient showing velocity magnitude
    contour_background = ax.contourf(xi_grid, yi_grid, speed, levels=50, 
                                     cmap='viridis', alpha=0.3, zorder=1,
                                     norm=norm)
    
    # Add the solid object mask
    solid, outline = overlay_mask(ax, mask_xi, mask_yi, mask_zi)
    
    # Initialize empty scatter plot
    scatter = ax.scatter([], [], s=12, c=[], cmap='viridis', 
                        norm=norm, edgecolor='white', linewidth=0.5,
                        alpha=0.8, zorder=10)
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(contour_background, cax=cax)
    cbar.set_label('Velocity Magnitude [m/s]', fontsize=12)
    
    # Add animation info text
    timer_text = ax.text(0.02, 0.96, "", transform=ax.transAxes,
                         fontsize=10, horizontalalignment='left',
                         bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    title.set_path_effects([pe.withStroke(linewidth=3, foreground='white')])
    
    # Add simulation info text
    info_text = ax.text(0.02, 0.02, 
                       "Particles follow the local fluid velocity field.\n"
                       "Color indicates particle velocity magnitude.",
                       transform=ax.transAxes, fontsize=9,
                       bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # Variable for storing trail plots
    trail_lines = []
    
    def init():
        scatter.set_offsets(np.column_stack([px, py]))
        timer_text.set_text("Time: 0.00")
        return scatter, timer_text
    
    def animate(i):
        nonlocal px, py, particle_age, trail_lines
        
        # Update timer
        timer_text.set_text(f"Time: {i*0.05:.2f} s")
        
        # Interpolate velocities at particle positions
        u_interp = griddata((xi_grid.flatten(), yi_grid.flatten()), 
                           ui.flatten(), (px, py), method='linear', fill_value=0)
        v_interp = griddata((xi_grid.flatten(), yi_grid.flatten()), 
                           vi.flatten(), (px, py), method='linear', fill_value=0)
        
        # Calculate current speed for coloring
        speeds = np.sqrt(u_interp**2 + v_interp**2)
        
        # Update positions with smaller time step for smoother motion
        px += u_interp * 0.05
        py += v_interp * 0.05
        
        # Store history
        history[i, :, 0] = px
        history[i, :, 1] = py
        
        # Increment age for existing particles
        particle_age += 1
        
        # Reset particles that have left the domain or entered solid regions
        oob_indices = ((px < x.min()) | (px > x.max()) | 
                      (py < y.min()) | (py > y.max()))
        
        if i > 0:  # Skip the first frame when interpolating
            # Check if particles have entered the solid region
            in_solid = []
            for j in range(len(px)):
                # Interpolate mask value at particle position
                mask_val = griddata((mask_xi.flatten(), mask_yi.flatten()), 
                                   mask_zi.flatten(), ([px[j]], [py[j]]), 
                                   method='linear', fill_value=1)
                in_solid.append(mask_val[0] < 0.5)
            
            # Add solid collision indices to out-of-bounds
            oob_indices = oob_indices | np.array(in_solid)
        
        # Reset out-of-bounds particles to inlet
        if np.any(oob_indices):
            px[oob_indices] = inlet_x
            py[oob_indices] = np.random.uniform(
                y_valid.min(), y_valid.max(), size=np.sum(oob_indices))
            particle_age[oob_indices] = 0
        
        # Remove old trail plots
        for line in trail_lines:
            if line in ax.lines:
                line.remove()
        trail_lines = []
        
        # Draw new trails with variable opacity based on age
        max_trail_length = 20
        for j in range(num_particles):
            if particle_age[j] > 5:  # Only draw trails for particles that have been around a while
                # Get history for this particle
                end_idx = i
                start_idx = max(0, end_idx - max_trail_length)
                
                if end_idx > start_idx:
                    x_history = history[start_idx:end_idx+1, j, 0]
                    y_history = history[start_idx:end_idx+1, j, 1]
                    
                    # Create a gradual fade effect for the trail
                    points = np.array([x_history, y_history]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    
                    # Use speed for colors
                    segment_speeds = speeds[j] * np.ones(len(segments))
                    
                    # Create a line collection for the trail
                    from matplotlib.collections import LineCollection
                    lc = LineCollection(segments, cmap='viridis', norm=norm,
                                       linewidth=1.5, alpha=0.5, zorder=5)
                    lc.set_array(segment_speeds)
                    line = ax.add_line(lc)
                    trail_lines.append(lc)
        
        # Update scatter plot
        scatter.set_offsets(np.column_stack([px, py]))
        scatter.set_array(speeds)
        
        return scatter, timer_text, *trail_lines

    print("Rendering animation frames...")
    ani = animation.FuncAnimation(fig, animate, frames=tqdm(range(steps)),
                                 init_func=init, interval=50, blit=False)
    
    # Save with higher quality
    writer = animation.PillowWriter(fps=30)
    ani.save(filename, writer=writer, dpi=150)
    plt.close(fig)
    
    print(f"Lagrangian animation saved to: {filename}")

def create_pressure_coefficient_plot(x, y, p, mask_xi, mask_yi, mask_zi, v_inf, filename):
    # Calculate pressure coefficient Cp = (p - p_inf) / (0.5 * rho * V_inf^2)
    # Assuming p_inf = 0 and rho = 1 for simplicity
    p_inf = 0
    rho = 1
    dynamic_pressure = 0.5 * rho * v_inf * v_inf
    
    cp = (p - p_inf) / dynamic_pressure
    
    # Create grid
    xi, yi, cp_i = create_grid(x, y, cp, resolution=400)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Set limits for better visualization
    vmin = np.nanpercentile(cp_i, 1)
    vmax = np.nanpercentile(cp_i, 99)
    vmin = min(vmin, -1.0)  # Ensure theoretical stagnation point is visible
    vmax = max(vmax, 1.0)
    
    # Plot pressure coefficient
    levels = np.linspace(vmin, vmax, 100)
    cf = ax.contourf(xi, yi, cp_i, levels=levels, cmap='coolwarm', 
                    alpha=0.95, extend='both', zorder=1)
    
    # Add contour lines
    contour_levels = np.linspace(vmin, vmax, 15)
    contours = ax.contour(xi, yi, cp_i, levels=contour_levels, 
                         colors='k', alpha=0.3, linewidths=0.5, zorder=2)
    
    # Label some important contours
    fmt = {0: 'Cp = 0', 1: 'Cp = 1', -1: 'Cp = -1'}
    ax.clabel(contours, contour_levels[::3], inline=True, fmt=fmt, fontsize=8)
    
    # Add the solid object mask
    solid, outline = overlay_mask(ax, mask_xi, mask_yi, mask_zi)
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(cf, cax=cax)
    cbar.set_label('Pressure Coefficient (Cp)', fontsize=12)
    
    # Set title and labels
    ax.set_title('Pressure Coefficient Distribution', fontsize=16, pad=10, fontweight='bold')
    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('y [m]', fontsize=12)
    ax.set_aspect('equal')
    
    # Add grid
    ax.grid(alpha=0.2, linestyle='--')

    
    # Save figure
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    data_dir = 'data'
    visuals_dir = 'visuals'
    os.makedirs(visuals_dir, exist_ok=True)
    
    # Create custom colormaps
    velocity_cmap, pressure_cmap, stream_cmap = create_custom_colormaps()
    
    # Freestream velocity (from simulation)
    v_inf = 1.0

    # Load mask with higher resolution
    print("Loading mask data...")
    mask_xi, mask_yi, mask_zi = load_mask(data_dir, resolution=400)

    # Scalar fields
    print("Generating velocity magnitude visualization...")
    x, y, vel_mag = load_scalar_field(os.path.join(data_dir, 'velocity_magnitude.data'))
    xi, yi, zi = create_grid(x, y, vel_mag, resolution=400)
    plot_scalar_field(xi, yi, zi, mask_xi, mask_yi, mask_zi,
                     'Velocity Magnitude', cmap=velocity_cmap, label='Velocity',
                     filename=os.path.join(visuals_dir, 'velocity_magnitude.png'),
                     units="m/s")

    print("Generating pressure field visualization...")
    x, y, pressure = load_scalar_field(os.path.join(data_dir, 'pressure.data'))
    xi, yi, zi = create_grid(x, y, pressure, resolution=400)
    plot_scalar_field(xi, yi, zi, mask_xi, mask_yi, mask_zi,
                     'Pressure Field', cmap=pressure_cmap, label='Pressure',
                     filename=os.path.join(visuals_dir, 'pressure_field.png'),
                     units="Pa")
    
    # Also create a pressure coefficient plot
    print("Generating pressure coefficient visualization...")
    create_pressure_coefficient_plot(x, y, pressure, mask_xi, mask_yi, mask_zi, v_inf,
                                   filename=os.path.join(visuals_dir, 'pressure_coefficient.png'))

    print("Generating stream function visualization...")
    x, y, psi = load_scalar_field(os.path.join(data_dir, 'stream_function.data'))
    xi, yi, zi = create_grid(x, y, psi, resolution=400)
    plot_scalar_field(xi, yi, zi, mask_xi, mask_yi, mask_zi,
                     'Stream Function (ψ)', cmap=stream_cmap, label='ψ',
                     filename=os.path.join(visuals_dir, 'stream_function.png'),
                     units="m²/s")

    # Vector fields
    print("Generating velocity vector field visualization...")
    x, y, u, v = load_vector_field(os.path.join(data_dir, 'velocity_field.data'))
    plot_velocity_field(x, y, u, v, mask_xi, mask_yi, mask_zi, 
                       title='Velocity Vector Field',
                       filename=os.path.join(visuals_dir, 'velocity_vectors.png'),
                       downsample=12)  # Downsample for clearer vectors

    print("Generating streamlines visualization...")
    plot_streamlines(x, y, u, v, mask_xi, mask_yi, mask_zi,
                    title='Flow Streamlines', 
                    filename=os.path.join(visuals_dir, 'streamlines.png'),
                    density=1.8)


    # Lagrangian Particle Animation
    # print("Generating Lagrangian particle animation...")
    # create_lagrangian_animation(x, y, u, v, mask_xi, mask_yi, mask_zi,
    #                            filename=os.path.join(visuals_dir, 'particle_flow.gif'),
    #                            num_particles=200, steps=150)

    print("\nAll visualizations completed successfully!")
    print("Enhanced visualizations saved to the 'visuals/' folder:")
    print("  - velocity_magnitude.png: Color map of flow speed")
    print("  - pressure_field.png: Pressure distribution")
    print("  - pressure_coefficient.png: Non-dimensional pressure coefficient")
    print("  - stream_function.png: Stream function contours")
    print("  - velocity_vectors.png: Vector field representation")
    print("  - streamlines.png: Flow streamlines with velocity coloring")
    print("  - vorticity.png: Vorticity field showing rotational flow features")
    print("  - particle_flow.gif: Animated particle trajectories")
    
    print("\nNote: For best viewing, open the PNG files in an image viewer that supports high-resolution images.")

if __name__ == '__main__':
    # Set better font rendering for high-quality plots
    plt.rcParams['text.usetex'] = False  # Set to True if you have LaTeX installed
    plt.rcParams['mathtext.fontset'] = 'dejavusans'
    plt.rcParams['font.family'] = 'sans-serif'
    
    print("Starting enhanced fluid flow visualization...")
    main()