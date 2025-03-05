from shape_utils import *

def plot_twisted_pointy_torus(ax1, ax2, R, r):
    """Generate a torus with a cross-section that's a circle with one sharp corner that rotates"""
    # Parameters
    circle_radius = r * 0.7  # Radius of the inscribed circle
    point_scale = 0.7  # Scale factor to move the point closer to center
    
    def pointy_cross_section(angle):
        # Start with a circle
        x = circle_radius * np.cos(angle)
        y = circle_radius * np.sin(angle)
        
        # Check if we're in the top-right quadrant (for the sharp corner)
        if x > 0 and y > 0:
            # Make a sharp corner in the top-right quadrant
            return r * point_scale, r * point_scale
        
        # For other quadrants, keep the circle shape
        return x, y
    
    def linear_twist(u_angle):
        return u_angle  # Full rotation as we go around the torus
    
    # Generate parametric coordinates
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, 2*np.pi, 100)
    u, v = np.meshgrid(u, v)
    
    # Create cross-section coordinates
    cross_x = np.zeros_like(v)
    cross_y = np.zeros_like(v)
    
    # Apply cross-section function with rotation
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            # Get rotation angle from twist function
            rotation = linear_twist(u[i,j])
            
            # Get base cross-section coordinates
            base_x, base_y = pointy_cross_section(v[i,j])
            
            # Apply rotation
            cos_rot = np.cos(rotation)
            sin_rot = np.sin(rotation)
            cross_x[i,j] = base_x * cos_rot - base_y * sin_rot
            cross_y[i,j] = base_x * sin_rot + base_y * cos_rot
    
    # Generate the torus
    x = (R + cross_x) * np.cos(u)
    y = (R + cross_x) * np.sin(u)
    z = cross_y
    
    # Plot on both axes
    for ax in [ax1, ax2]:
        ax.plot_surface(x, y, z, color='lightgreen', alpha=0.9)
    
    limit = (R + r) * 1.2
    set_common_plot_properties(ax1, ax2, "Pointy Torus (Rotating Sharp Corner)", limit) 
