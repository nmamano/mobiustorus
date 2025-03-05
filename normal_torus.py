from shape_utils import *

def plot_normal_torus(ax1, ax2, R, r):
    """Generate a normal (circular) torus"""
    def circle_cross_section(angle):
        return r * np.cos(angle), r * np.sin(angle)
    
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
            cross_x[i,j], cross_y[i,j] = circle_cross_section(v[i,j])
    
    # Generate the torus
    x = (R + cross_x) * np.cos(u)
    y = (R + cross_x) * np.sin(u)
    z = cross_y
    
    # Plot on both axes
    for ax in [ax1, ax2]:
        ax.plot_surface(x, y, z, color='lightblue', alpha=0.9)
    
    limit = (R + r) * 1.2
    set_common_plot_properties(ax1, ax2, "Normal Torus", limit) 
