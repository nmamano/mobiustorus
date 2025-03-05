import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def set_common_plot_properties(ax1, ax2, title, limit):
    """Set common properties for both plot axes"""
    for ax in [ax1, ax2]:
        ax.set_axis_off()
        ax.set_box_aspect([1,1,1])
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
    
    # Set initial viewing angles
    ax1.view_init(elev=20, azim=45)  # front view
    ax2.view_init(elev=-20, azim=225)  # back view
    
    plt.gcf().suptitle(f'{title}\nClick and drag to rotate!', y=0.95)
    plt.gcf().canvas.draw_idle() 


# Generalized function to create a torus from any cross-section
def create_torus(ax1, ax2, R, r, cross_section_func, twist_func=None, colors=None, title="Custom Torus", additional_info=None):
    """
    Creates a torus with a custom cross-section and optional twisting.
    
    Parameters:
    - ax1, ax2: The two axes to plot on (front and back views)
    - R: Major radius of the torus
    - r: Scale factor for the cross-section
    - cross_section_func: Function that takes angle v and returns (x,y) coordinates of cross-section
    - twist_func: Optional function that takes angle u and returns rotation angle
    - colors: Color or list of colors for the surfaces
    - title: Title for the plot
    - additional_info: Optional text to add below the title
    
    Returns:
    - Dictionary containing the x, y, z coordinates and other data for later use if needed
    """
    # Generate parametric coordinates
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, 2*np.pi, 100)
    u, v = np.meshgrid(u, v)
    
    # Handle multiple components if the cross-section function returns them
    if hasattr(cross_section_func, 'is_multi_component') and cross_section_func.is_multi_component:
        components = cross_section_func(u, v, R, r)
        
        # Plot all components on both axes
        for ax in [ax1, ax2]:
            for idx, component in enumerate(components):
                color = colors[idx] if isinstance(colors, list) and idx < len(colors) else 'lightblue'
                alpha = 0.9 if not isinstance(color, dict) else color.get('alpha', 0.9)
                c = color if not isinstance(color, dict) else color.get('color', 'lightblue')
                
                ax.plot_surface(
                    component['x'], component['y'], component['z'],
                    color=c, alpha=alpha
                )
                
            # Standard view setup
            ax.set_axis_off()
            ax.set_box_aspect([1,1,1])
            
            # Calculate limit based on maximum extent
            max_x = max(np.max(np.abs(comp['x'])) for comp in components)
            max_y = max(np.max(np.abs(comp['y'])) for comp in components)
            max_z = max(np.max(np.abs(comp['z'])) for comp in components)
            limit = max(max_x, max_y, max_z) * 1.2
            
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_zlim(-limit, limit)
    else:
        # Create cross-section coordinates
        cross_x = np.zeros_like(v)
        cross_y = np.zeros_like(v)
        
        # Apply cross-section function with optional rotation
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                if twist_func:
                    # Get rotation angle from twist function
                    rotation = twist_func(u[i,j])
                    
                    # Get base cross-section coordinates
                    base_x, base_y = cross_section_func(v[i,j])
                    
                    # Apply rotation
                    cos_rot = np.cos(rotation)
                    sin_rot = np.sin(rotation)
                    cross_x[i,j] = base_x * cos_rot - base_y * sin_rot
                    cross_y[i,j] = base_x * sin_rot + base_y * cos_rot
                else:
                    # No rotation, just get coordinates
                    cross_x[i,j], cross_y[i,j] = cross_section_func(v[i,j])
        
        # Generate the torus
        x = (R + cross_x) * np.cos(u)
        y = (R + cross_x) * np.sin(u)
        z = cross_y
        
        # Handle special case for polygon torus with face colors
        if hasattr(cross_section_func, 'has_face_colors') and cross_section_func.has_face_colors:
            face_colors = cross_section_func.get_face_colors(u, v)
            
            # Plot on both axes
            for ax in [ax1, ax2]:
                surface = ax.plot_surface(x, y, z, facecolors=face_colors, alpha=0.9)
                ax.set_axis_off()
                ax.set_box_aspect([1,1,1])
                
                limit = (R + np.max(np.abs(cross_x))) * 1.2
                ax.set_xlim(-limit, limit)
                ax.set_ylim(-limit, limit)
                ax.set_zlim(-limit, limit)
        else:
            # Regular single-color surface
            color = colors if colors else 'lightblue'
            
            # Plot on both axes
            for ax in [ax1, ax2]:
                surface = ax.plot_surface(x, y, z, color=color, alpha=0.9)
                ax.set_axis_off()
                ax.set_box_aspect([1,1,1])
                
                limit = (R + np.max(np.abs(cross_x))) * 1.2
                ax.set_xlim(-limit, limit)
                ax.set_ylim(-limit, limit)
                ax.set_zlim(-limit, limit)
    
    # Set initial viewing angles
    ax1.view_init(elev=20, azim=45)  # front view
    ax2.view_init(elev=-20, azim=225)  # back view
    
    # Create title with optional additional info
    full_title = f'{title}\nClick and drag to rotate!'
    if additional_info:
        full_title += f'\n{additional_info}'
    
    # Get the figure from one of the axes and set the title
    plt.gcf().suptitle(full_title, y=0.95)
    plt.gcf().canvas.draw_idle()
    
    # Return generated data for potential reuse
    if 'x' in locals():
        return {'x': x, 'y': y, 'z': z, 'cross_x': cross_x, 'cross_y': cross_y}
    else:
        return components
