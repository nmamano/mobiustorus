from shape_utils import *

def plot_edged_ball(ax1, ax2, r):
    """Create a flattened sphere with two differently colored hemispheres"""
    # Generate the sphere coordinates
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    
    # Create the sphere with flattening at the poles
    flattening_factor = 0.8  # Controls how flattened the sphere is (1.0 = perfect sphere)
    
    # Create separate arrays for upper and lower hemispheres
    # Upper hemisphere (0 to pi/2)
    v_upper = np.linspace(0, np.pi/2, 25)
    x_upper = r * np.outer(np.cos(u), np.sin(v_upper))
    y_upper = r * np.outer(np.sin(u), np.sin(v_upper))
    z_upper = r * flattening_factor * np.outer(np.ones(np.size(u)), np.cos(v_upper))
    
    # Lower hemisphere (pi/2 to pi)
    v_lower = np.linspace(np.pi/2, np.pi, 25)
    x_lower = r * np.outer(np.cos(u), np.sin(v_lower))
    y_lower = r * np.outer(np.sin(u), np.sin(v_lower))
    z_lower = r * flattening_factor * np.outer(np.ones(np.size(u)), np.cos(v_lower))
    
    # Plot on both axes
    for ax in [ax1, ax2]:
        # Plot the upper hemisphere (z >= 0)
        upper_hemisphere = ax.plot_surface(
            x_upper, y_upper, z_upper, 
            rstride=1, cstride=1, 
            color='lightblue', alpha=0.9,
            linewidth=0, edgecolor='none',
            clip_on=False,
            zorder=1
        )
        
        # Plot the lower hemisphere (z < 0)
        lower_hemisphere = ax.plot_surface(
            x_lower, y_lower, z_lower,  # Lower hemisphere already has correct z values
            rstride=1, cstride=1, 
            color='lightgreen', alpha=0.9,
            linewidth=0, edgecolor='none',
            clip_on=False,
            zorder=0
        )
    
    limit = r * 1.5
    set_common_plot_properties(ax1, ax2, "Edged Ball", limit) 
