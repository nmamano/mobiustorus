from shape_utils import *

def plot_pointy_cylinder(ax1, ax2, R, r):
    """Generate a cylinder with a pointy cross-section (one sharp corner)"""
    # Parameters
    cylinder_radius = R  # Main radius of the cylinder
    circle_radius = r * 0.7  # Radius of the circular part
    square_size = r * 0.7  # Further reduced size to move the point even closer to center
    height = r / 2  # Reduced height to half of the original
    
    # Create the special cross-section
    theta = np.linspace(0, 2*np.pi, 100)
    cross_x = np.zeros_like(theta)
    cross_y = np.zeros_like(theta)
    
    for i, angle in enumerate(theta):
        # Start with a circle
        x = circle_radius * np.cos(angle)
        y = circle_radius * np.sin(angle)
        
        # Check if we're in the top-right quadrant (for the sharp corner)
        if x > 0 and y > 0:
            # Make a sharp corner in the top-right quadrant, but closer to center
            cross_x[i] = square_size
            cross_y[i] = square_size
        else:
            # For other quadrants, keep the circle shape
            cross_x[i] = x
            cross_y[i] = y
    
    # Generate the cylinder
    z_values = np.linspace(-height/2, height/2, 50)
    
    # Define colors for each face
    wall_color = 'lightblue'
    top_color = 'lightgreen'
    bottom_color = 'coral'
    
    # Plot on both axes
    for ax in [ax1, ax2]:
        # Create a surface for the cylinder wall
        theta_grid, z_grid = np.meshgrid(theta, z_values)
        x_grid = np.zeros_like(theta_grid)
        y_grid = np.zeros_like(theta_grid)
        
        for i in range(len(theta)):
            x_grid[:, i] = cross_x[i]
            y_grid[:, i] = cross_y[i]
        
        # Plot the cylinder wall
        surface = ax.plot_surface(x_grid, y_grid, z_grid, color=wall_color, alpha=0.9)
        
        # Create top and bottom faces using plot_surface instead of fill
        face_colors = [bottom_color, top_color]  # Bottom, Top
        for idx, z in enumerate([-height/2, height/2]):
            # Create a grid for the circular face
            r_grid = np.linspace(0, 1, 10)  # Radial grid from center to edge
            theta_grid, r_grid = np.meshgrid(theta, r_grid)
            
            # Calculate coordinates for each point in the grid
            face_x = np.zeros_like(theta_grid)
            face_y = np.zeros_like(theta_grid)
            face_z = np.ones_like(theta_grid) * z
            
            # For each angle, interpolate from center to edge
            for i in range(len(theta)):
                face_x[:, i] = r_grid[:, i] * cross_x[i]
                face_y[:, i] = r_grid[:, i] * cross_y[i]
            
            # Plot the face as a surface with its own color
            ax.plot_surface(face_x, face_y, face_z, color=face_colors[idx], alpha=0.9)
            
            # Draw the outline of the face
            outline_color = 'darkblue' if idx == 0 else 'darkgreen'
            ax.plot(cross_x, cross_y, np.ones_like(theta) * z, color=outline_color, linewidth=2)
    
    limit = max(np.max(cross_x), np.max(cross_y)) * 1.2
    set_common_plot_properties(ax1, ax2, "Pointy Cylinder", limit) 
