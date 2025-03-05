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
    
    # Mark the function as multi-component
    double_torus_components.is_multi_component = True
    
    create_torus(
        ax1, ax2, R, r, 
        double_torus_components, 
        colors=[{'color': 'lightblue', 'alpha': 0.6}, {'color': 'coral', 'alpha': 0.9}],
        title="Double Torus (Torus with Doughnut Cross-section)"
    )
