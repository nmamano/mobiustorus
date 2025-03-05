from shape_utils import *

def plot_edged_torus(ax1, ax2, R, r):
    """Generate a lenticular torus with different colors for top and bottom halves"""
    inner_hole_size = 0.05  # Controls the size of the inner hole (0.1 = small hole, 1.0 = large hole)
    
    def lenticular_cross_section(angle):
        # Create a lens shape with sharp points at inner and outer edges
        # Sharpen the edges by using a power function
        EDGE_SHARPNESS = 0.0000099  # Lower = sharper points
        
        # Calculate base coordinates for a circle
        base_x = np.cos(angle)
        base_y = np.sin(angle)
        
        # Create lens shape with sharp points
        x = base_x
        # Make the outer part more round by reducing the sharpness effect for the outer half
        OUTER_HALF_START = -np.pi/2
        OUTER_HALF_END = np.pi/2
        if OUTER_HALF_START < angle < OUTER_HALF_END:  # Outer half
            # Use a milder sharpness for the outer part to make it more round
            OUTER_SHARPNESS = 0.3  # Higher value = more round
            y = base_y * np.abs(np.sin(angle)) ** OUTER_SHARPNESS
        else:
            # Keep the inner part sharp
            y = base_y * np.abs(np.sin(angle)) ** EDGE_SHARPNESS
        
        # Make the inner edge reach toward the center
        INNER_HALF_START = np.pi/2
        INNER_HALF_END = 3*np.pi/2
        if INNER_HALF_START < angle < INNER_HALF_END:  # Inner half
            # Scale x to reach toward the center
            MIN_INNER_SCALE = 0.1  # Reduced minimum scale factor for sharper inner edge
            MAX_INNER_SCALE = 0.7  # Additional scale factor based on angle
            # Use a higher exponent for sharper inner edge
            INNER_SHARPNESS = 0.00001  # Lower value = sharper inner edge
            x = x * (MIN_INNER_SCALE + MAX_INNER_SCALE * np.abs(np.sin(angle)) ** INNER_SHARPNESS)
        
        # Make the cross-section wider to reduce the inner hole size
        # Scale the x coordinate based on the inner_hole_size parameter
        if angle > INNER_HALF_START and angle < INNER_HALF_END:
            # For the inner part, make it reach further inward
            x = x * (2.0 - inner_hole_size)  # Larger value = smaller hole
        else:
            # For the outer part, make it wider
            x = x * inner_hole_size  # Smaller value = wider outer edge
        
        return r * x, r * y
    
    # Generate parametric coordinates
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, 2*np.pi, 100)
    u, v = np.meshgrid(u, v)
    
    # Create cross-section coordinates
    cross_x = np.zeros_like(v)
    cross_y = np.zeros_like(v)
    
    # Apply cross-section function
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            cross_x[i,j], cross_y[i,j] = lenticular_cross_section(v[i,j])
    
    # Generate the torus
    x = (R + cross_x) * np.cos(u)
    y = (R + cross_x) * np.sin(u)
    z = cross_y
    
    # Create color array based on z-coordinate
    colors = np.zeros((*z.shape, 4))  # RGBA array
    colors[z >= 0] = np.array([0.8, 0.9, 1.0, 0.9])  # lightblue with alpha
    colors[z < 0] = np.array([1.0, 0.5, 0.4, 0.9])   # coral with alpha
    
    # Plot on both axes
    for ax in [ax1, ax2]:
        surface = ax.plot_surface(x, y, z, facecolors=colors)
    
    PLOT_MARGIN = 1.2  # Extra space around the shape (20% margin)
    MAX_CROSS_SECTION_SCALE = 2.0  # Maximum scale factor for the outer part
    limit = (R + r * (MAX_CROSS_SECTION_SCALE - inner_hole_size)) * PLOT_MARGIN  # Adjust limit based on the wider cross-section
    set_common_plot_properties(ax1, ax2, "Edged Torus", limit) 
