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
    
    create_torus(ax1, ax2, R, r, pointy_cross_section, linear_twist,
                 colors='lightgreen',
                 title="Pointy Torus (Rotating Sharp Corner)")
