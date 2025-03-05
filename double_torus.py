from shape_utils import *

def plot_double_torus(ax1, ax2, R, r):
    """Generate a torus with a doughnut cross-section (double torus)"""
    # Parameters for the double torus
    r_outer = r  # Outer radius of the cross-section
    r_inner = r/2  # Inner radius of the cross-section
    
    def double_torus_components(u, v, R, r):
        # This function returns multiple components for the double torus
        # Outer torus
        x_outer = (R + r_outer * np.cos(v)) * np.cos(u)
        y_outer = (R + r_outer * np.cos(v)) * np.sin(u)
        z_outer = r_outer * np.sin(v)
        
        # Inner torus (smaller doughnut inside the cross-section)
        x_inner = (R + r_inner * np.cos(v + np.pi)) * np.cos(u)
        y_inner = (R + r_inner * np.cos(v + np.pi)) * np.sin(u)
        z_inner = r_inner * np.sin(v + np.pi)
        
        return [
            {'x': x_outer, 'y': y_outer, 'z': z_outer},
            {'x': x_inner, 'y': y_inner, 'z': z_inner}
        ]
    
    # Generate parametric coordinates
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, 2*np.pi, 100)
    u, v = np.meshgrid(u, v)
    
    components = double_torus_components(u, v, R, r)
    
    # Plot all components on both axes
    for ax in [ax1, ax2]:
        for idx, component in enumerate(components):
            color = 'lightblue' if idx == 0 else 'coral'
            alpha = 0.6 if idx == 0 else 0.9
            ax.plot_surface(
                component['x'], component['y'], component['z'],
                color=color, alpha=alpha
            )
    
    # Calculate limit based on maximum extent
    max_x = max(np.max(np.abs(comp['x'])) for comp in components)
    max_y = max(np.max(np.abs(comp['y'])) for comp in components)
    max_z = max(np.max(np.abs(comp['z'])) for comp in components)
    limit = max(max_x, max_y, max_z) * 1.2
    
    set_common_plot_properties(ax1, ax2, "Double Torus (Torus with Doughnut Cross-section)", limit) 
