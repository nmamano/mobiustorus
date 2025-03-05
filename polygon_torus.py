from shape_utils import *

def gcd(a, b):
    """Find the greatest common divisor"""
    if b == 0:
        return a
    return gcd(b, a % b)

def plot_polygon_torus(ax1, ax2, R, r, k, twist_multiplier, points_per_side):
    """Generate a torus with a polygonal cross-section with optional twist"""
    # Calculate the base twist angle
    base_twist = 2 * np.pi / k
    total_twist = base_twist * twist_multiplier
    
    # Calculate the greatest common divisor between k and twist_multiplier
    # If twist_multiplier is 0, use k as the number of colors
    num_colors = gcd(k, twist_multiplier) if twist_multiplier > 0 else k
    
    # Generate colors for each face based on GCD
    colors = plt.cm.rainbow(np.linspace(0, 1, num_colors))
    
    # Generate vertices of the regular polygon
    angles = np.linspace(0, 2*np.pi, k+1)
    # For k=4, rotate the polygon by 45 degrees (π/4 radians)
    rotation_offset = np.pi/4 if k == 4 else 0
    vertices_x = r * np.cos(angles + rotation_offset)
    vertices_y = r * np.sin(angles + rotation_offset)
    
    # Create points between vertices
    polygon_x = []
    polygon_y = []
    face_colors = []
    
    for i in range(k):
        x1, y1 = vertices_x[i], vertices_y[i]
        x2, y2 = vertices_x[i+1], vertices_y[i+1]
        
        t = np.linspace(0, 1, points_per_side)
        polygon_x.extend(x1 + (x2 - x1) * t)
        polygon_y.extend(y1 + (y2 - y1) * t)
        
        # Assign color based on modulo of GCD
        color_index = i % num_colors
        face_colors.extend([colors[color_index]] * points_per_side)
    
    polygon_x = np.array(polygon_x)
    polygon_y = np.array(polygon_y)
    face_colors = np.array(face_colors)
    
    # Create the meshgrid
    theta = np.linspace(0, 2*np.pi, 100)
    theta_grid, t_grid = np.meshgrid(theta, np.arange(len(polygon_x)))
    
    # Apply twist
    twist_angle = total_twist * theta_grid / (2*np.pi)
    
    # Rotate the polygon coordinates
    rotated_x = polygon_x[t_grid.astype(int)] * np.cos(twist_angle) - polygon_y[t_grid.astype(int)] * np.sin(twist_angle)
    rotated_y = polygon_x[t_grid.astype(int)] * np.sin(twist_angle) + polygon_y[t_grid.astype(int)] * np.cos(twist_angle)
    
    # Generate coordinates
    x = (R + rotated_x) * np.cos(theta_grid)
    y = (R + rotated_x) * np.sin(theta_grid)
    z = rotated_y
    
    # Create color array for the entire surface
    colors_2d = np.zeros((x.shape[0], x.shape[1], 4))
    for i in range(x.shape[0]):
        colors_2d[i, :] = face_colors[i]
    
    # Plot on both axes
    for ax in [ax1, ax2]:
        surface = ax.plot_surface(x, y, z, facecolors=colors_2d, alpha=0.9)
    
    limit = (R + r) * 1.2
    
    if twist_multiplier == 0:
        twist_text = ""
    else:
        twist_text = f" with {twist_multiplier} face{'s' if twist_multiplier > 1 else ''} twist"
    
    # Add GCD information to the title
    gcd_text = f"GCD({k}, {twist_multiplier}) = {num_colors} colors" if twist_multiplier > 0 else f"{k} colors"
    title = f'Torus with {k}-sided cross-section{twist_text}\n{gcd_text}'
    
    set_common_plot_properties(ax1, ax2, title, limit) 
