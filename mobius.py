import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import sys
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import json

# Constants
SETTINGS_FILE = 'shape_settings.json'

# Load last selected shape from file
def load_last_shape():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
                return settings.get('last_shape', 'Polygon Torus')
    except:
        pass
    return 'Polygon Torus'

# Save selected shape to file
def save_selected_shape(shape):
    try:
        settings = {
            "_comment": "This is an auto-generated settings file for the Mobius Torus visualization program. DO NOT MODIFY MANUALLY.",
            "_purpose": "This file stores the last selected shape to persist user preferences between program restarts.",
            "last_shape": shape,
            "last_modified": str(np.datetime64('now'))  # Add timestamp for reference
        }
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=4)  # Use pretty printing for better readability
    except:
        pass

# File monitoring setup
# This code sets up automatic program termination and restart when the source file is modified.
# It's useful during development to ensure the program restarts with the latest code changes.
# When the script is modified and saved, the program will detect the change, exit,
# and automatically start a new instance with the updated code.
class FileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        # Ignore changes to the settings file
        if event.src_path == os.path.abspath(SETTINGS_FILE):
            return
            
        if event.src_path == os.path.abspath(__file__):
            print("\nFile changed. Restarting the program...")
            # Start new process before exiting
            subprocess.Popen([sys.executable, __file__])
            os._exit(0)

def start_file_monitoring():
    event_handler = FileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(os.path.abspath(__file__)), recursive=False)
    observer.start()
    
# Start file monitoring in a separate thread
monitoring_thread = threading.Thread(target=start_file_monitoring, daemon=True)
monitoring_thread.start()

# Custom rotation handler to link both views
def on_mouse_move(event):
    if event.inaxes == ax1:
        ax = ax1
        other_ax = ax2
    elif event.inaxes == ax2:
        ax = ax2
        other_ax = ax1
    else:
        return

    if event.button == 1:  # Left mouse button
        if hasattr(ax, '_mouse_init'):
            # Calculate rotation
            dx = event.xdata - ax._mouse_init[0]
            dy = event.ydata - ax._mouse_init[1]
            
            # Get current view
            elev = ax.elev
            azim = ax.azim
            
            # Update primary view
            ax.view_init(elev - dy, azim - dx)
            
            # Update opposite view (viewed from back)
            other_ax.view_init(elev - dy, (azim - dx + 180) % 360)
            
            # Flip the elevation for the back view
            if other_ax == ax2:  # if this is the back view
                other_ax.elev = -other_ax.elev
            
            ax.figure.canvas.draw_idle()
    
    ax._mouse_init = event.xdata, event.ydata

def on_button_press(event):
    if event.inaxes in [ax1, ax2]:
        event.inaxes._mouse_init = event.xdata, event.ydata

def on_button_release(event):
    if event.inaxes in [ax1, ax2]:
        event.inaxes._mouse_init = None

# Function to find the greatest common divisor
def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)

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
        
    fig.suptitle(full_title, y=0.95)
    fig.canvas.draw_idle()
    
    # Return generated data for potential reuse
    if 'x' in locals():
        return {'x': x, 'y': y, 'z': z, 'cross_x': cross_x, 'cross_y': cross_y}
    else:
        return components

def plot_normal_torus(ax1, ax2, R, r):
    """Generate a normal (circular) torus"""
    def circle_cross_section(angle):
        return r * np.cos(angle), r * np.sin(angle)
    
    create_torus(ax1, ax2, R, r, circle_cross_section, 
                 colors='lightblue', 
                 title="Normal Torus")

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

def plot_pointy_torus(ax1, ax2, R, r):
    """Generate a torus with a cross-section that's a circle with one sharp corner"""
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
    
    create_torus(ax1, ax2, R, r, pointy_cross_section, 
                 colors='lightgreen',
                 title="Pointy Torus (One Sharp Corner)")

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
        ax.set_axis_off()
        ax.set_box_aspect([1,1,1])
        
        limit = (R + r) * 1.2
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
    
    # Set initial viewing angles
    ax1.view_init(elev=20, azim=45)  # front view
    ax2.view_init(elev=-20, azim=225)  # back view
    
    if twist_multiplier == 0:
        twist_text = ""
    else:
        twist_text = f" with {twist_multiplier} face{'s' if twist_multiplier > 1 else ''} twist"
    
    # Add GCD information to the title
    gcd_text = f"GCD({k}, {twist_multiplier}) = {num_colors} colors" if twist_multiplier > 0 else f"{k} colors"
    fig.suptitle(f'Torus with {k}-sided cross-section{twist_text}\n{gcd_text}\nClick and drag to rotate!', y=0.95)
    fig.canvas.draw_idle()

def plot_cube(ax1, ax2, r):
    # Generate a cube
    # Define the vertices of the cube
    cube_size = r * 1.5  # Size of the cube
    vertices = np.array([
        [-cube_size, -cube_size, -cube_size],  # 0
        [cube_size, -cube_size, -cube_size],   # 1
        [cube_size, cube_size, -cube_size],    # 2
        [-cube_size, cube_size, -cube_size],   # 3
        [-cube_size, -cube_size, cube_size],   # 4
        [cube_size, -cube_size, cube_size],    # 5
        [cube_size, cube_size, cube_size],     # 6
        [-cube_size, cube_size, cube_size]     # 7
    ])
    
    # Define the faces of the cube
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
        [vertices[0], vertices[3], vertices[7], vertices[4]]   # left
    ]
    
    # Define colors for each face
    face_colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple']
    
    # Plot on both axes
    for ax in [ax1, ax2]:
        # Plot each face of the cube
        for i, face in enumerate(faces):
            face = np.array(face)
            # Create x, y, z coordinates for the face
            x = face[:, 0]
            y = face[:, 1]
            z = face[:, 2]
            
            # Plot the face as a filled polygon
            ax.plot_surface(
                np.array([[x[0], x[1]], [x[3], x[2]]]),
                np.array([[y[0], y[1]], [y[3], y[2]]]),
                np.array([[z[0], z[1]], [z[3], z[2]]]),
                color=face_colors[i],
                alpha=0.9
            )
        
        ax.set_axis_off()
        ax.set_box_aspect([1,1,1])
        
        limit = cube_size * 1.5
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
    
    # Set initial viewing angles
    ax1.view_init(elev=20, azim=45)  # front view
    ax2.view_init(elev=-20, azim=225)  # back view
    
    fig.suptitle('Cube\nClick and drag to rotate!', y=0.95)
    fig.canvas.draw_idle()

def plot_pointy_cylinder(ax1, ax2, R, r):
    # Generate a cylinder with a pointy cross-section (one sharp corner)
    # This matches the cross-section of the pointy torus
    
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
        
        ax.set_axis_off()
        ax.set_box_aspect([1,1,1])
        
        limit = max(np.max(cross_x), np.max(cross_y)) * 1.2
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-height*1.5, height*1.5)  # Adjusted z-limits to match new height
    
    # Set initial viewing angles
    ax1.view_init(elev=20, azim=45)  # front view
    ax2.view_init(elev=-20, azim=225)  # back view
    
    fig.suptitle('Pointy Cylinder\nClick and drag to rotate!', y=0.95)
    fig.canvas.draw_idle()

def plot_octahedron(ax1, ax2, r):
    # Generate an octahedron (8-faced polyhedron)
    # Define the vertices of the octahedron
    size = r * 1.5  # Size of the octahedron
    vertices = np.array([
        [0, 0, size],      # top
        [size, 0, 0],      # right
        [0, size, 0],      # front
        [-size, 0, 0],     # left
        [0, -size, 0],     # back
        [0, 0, -size]      # bottom
    ])
    
    # Define the faces of the octahedron (each face is a triangle)
    faces = [
        [vertices[0], vertices[1], vertices[2]],  # top-right-front
        [vertices[0], vertices[2], vertices[3]],  # top-front-left
        [vertices[0], vertices[3], vertices[4]],  # top-left-back
        [vertices[0], vertices[4], vertices[1]],  # top-back-right
        [vertices[5], vertices[1], vertices[2]],  # bottom-right-front
        [vertices[5], vertices[2], vertices[3]],  # bottom-front-left
        [vertices[5], vertices[3], vertices[4]],  # bottom-left-back
        [vertices[5], vertices[4], vertices[1]]   # bottom-back-right
    ]
    
    # Define colors for each face - using more distinct colors for better visibility
    face_colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan', 'magenta']
    
    # Plot on both axes
    for ax in [ax1, ax2]:
        # Clear any existing content
        ax.clear()
        
        # Sort faces by depth for proper transparency rendering
        # Calculate the centroid (average position) of each face
        centroids = np.array([np.mean(face, axis=0) for face in faces])
        
        # Get the current view direction
        view_direction = np.array([0, 0, -1])  # Default view looks along negative z-axis
        
        # Calculate the dot product to determine depth
        depths = np.dot(centroids, view_direction)
        
        # Sort faces by depth (furthest first for proper transparency)
        sorted_indices = np.argsort(depths)
        sorted_faces = [faces[i] for i in sorted_indices]
        sorted_colors = [face_colors[i] for i in sorted_indices]
        
        # Create a Poly3DCollection with transparency
        collection = Poly3DCollection(
            sorted_faces,
            facecolors=sorted_colors,
            linewidths=0,  # No edge lines
            alpha=0.9  # Transparency
        )
        
        # Enable proper depth sorting
        collection.set_sort_zpos(0)
        
        # Add the collection to the axis
        ax.add_collection3d(collection)
        
        ax.set_axis_off()
        ax.set_box_aspect([1,1,1])
        
        limit = size * 1.5
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
    
    # Set initial viewing angles
    ax1.view_init(elev=20, azim=45)  # front view
    ax2.view_init(elev=-20, azim=225)  # back view
    
    fig.suptitle('Octahedron\nClick and drag to rotate!', y=0.95)
    fig.canvas.draw_idle()

def plot_edged_ball(ax1, ax2, r):
    # Create a flattened sphere with two differently colored hemispheres
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
        
        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 1])
        
        limit = r * 1.5
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
    
    # Set initial viewing angles
    ax1.view_init(elev=20, azim=45)  # front view
    ax2.view_init(elev=-20, azim=225)  # back view
    
    fig.suptitle('Edged Ball\nClick and drag to rotate!', y=0.95)
    fig.canvas.draw_idle()

def plot_tetrahedron(ax1, ax2, r):
    # Generate a tetrahedron (4-faced polyhedron)
    # Define the vertices of the tetrahedron
    size = r * 1.5  # Size of the tetrahedron
    
    # Define the vertices (regular tetrahedron)
    vertices = np.array([
        [1, 1, 1],              # top
        [1, -1, -1],            # right
        [-1, 1, -1],            # left
        [-1, -1, 1]             # front
    ]) * size / np.sqrt(3)      # Normalize to desired size
    
    # Define the faces of the tetrahedron (each face is a triangle)
    faces = [
        [vertices[0], vertices[1], vertices[2]],  # top-right-left
        [vertices[0], vertices[2], vertices[3]],  # top-left-front
        [vertices[0], vertices[3], vertices[1]],  # top-front-right
        [vertices[1], vertices[3], vertices[2]]   # right-front-left (bottom)
    ]
    
    # Define colors for each face - using more distinct colors
    face_colors = ['#FF5555', '#55FF55', '#5555FF', '#FFFF55']  # Brighter red, green, blue, yellow
    
    # Plot on both axes
    for ax in [ax1, ax2]:
        # Clear any existing content
        ax.clear()
        
        # Sort faces by depth for proper transparency rendering
        # Calculate the centroid (average position) of each face
        centroids = np.array([np.mean(face, axis=0) for face in faces])
        
        # Get the current view direction
        view_direction = np.array([0, 0, -1])  # Default view looks along negative z-axis
        
        # Calculate the dot product to determine depth
        depths = np.dot(centroids, view_direction)
        
        # Sort faces by depth (furthest first for proper transparency)
        sorted_indices = np.argsort(depths)
        sorted_faces = [faces[i] for i in sorted_indices]
        sorted_colors = [face_colors[i] for i in sorted_indices]
        
        # Create a Poly3DCollection with transparency
        collection = Poly3DCollection(
            sorted_faces,
            facecolors=sorted_colors,
            linewidths=0,  # No edges
            edgecolors='none',  # No edge colors
            alpha=0.8  # Add transparency
        )
        
        # Enable proper depth sorting
        collection.set_sort_zpos(0)
        
        # Add the collection to the axis
        ax.add_collection3d(collection)
        
        # Improve the 3D rendering
        ax.set_axis_off()
        ax.set_box_aspect([1,1,1])
        
        # Set wider limits to avoid clipping
        limit = size * 1.8
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
    
    # Set initial viewing angles
    ax1.view_init(elev=20, azim=45)  # front view
    ax2.view_init(elev=-20, azim=225)  # back view
    
    fig.suptitle('Tetrahedron\nClick and drag to rotate!', y=0.95)
    fig.canvas.draw_idle()

def plot_thick_triangle(ax1, ax2, r):
    # Generate a thick triangle
    # - equilateral triangle with varying thickness
    # - thickness is 0 at base and top vertex
    # - thickness increases and decreases in between
    
    # Parameters for the triangle
    size = r * 5  # Scale factor for the triangle size
    height = size * np.sqrt(3) / 2  # Height of equilateral triangle
    
    # Define the front face vertices of the triangle (centered at origin)
    top_vertex = np.array([0, height * 2/3, 0])         # top vertex
    bottom_left = np.array([-size/2, -height/3, 0])      # bottom left
    bottom_right = np.array([size/2, -height/3, 0])      # bottom right
    
    # Function to calculate thickness at a given height
    # We want thickness=0 at y=-height/3 (base) and y=height*2/3 (top)
    # and maximum thickness in between
    def thickness_at_height(y):
        # Normalize y between 0 and 1 (0 at base, 1 at top)
        y_norm = (y - (-height/3)) / (height)
        
        # Parabolic function that's 0 at y=0 and y=1
        return 5 * r * y_norm * (1 - y_norm)
    
    # Number of segments for the curved sides
    n_segments = 30
    
    # Create points along the left and right edges with varying thickness
    left_edge_front = []
    left_edge_back = []
    right_edge_front = []
    right_edge_back = []
    
    # Generate points along the edges
    t_values = np.linspace(0, 1, n_segments)
    for t in t_values:
        # Left edge (top to bottom-left)
        left_point = top_vertex * (1-t) + bottom_left * t
        
        # Right edge (top to bottom-right)
        right_point = top_vertex * (1-t) + bottom_right * t
        
        # Calculate thickness at this height
        left_thick = thickness_at_height(left_point[1])
        right_thick = thickness_at_height(right_point[1])
        
        # Add front and back points with appropriate thickness
        left_edge_front.append(left_point + np.array([0, 0, left_thick/2]))
        left_edge_back.append(left_point + np.array([0, 0, -left_thick/2]))
        
        right_edge_front.append(right_point + np.array([0, 0, right_thick/2]))
        right_edge_back.append(right_point + np.array([0, 0, -right_thick/2]))
    
    # Create the faces
    faces = []
    face_colors = []
    
    # Front face - triangulate between the front edges
    for i in range(n_segments - 1):
        # Add triangles to fill the front face
        faces.append([left_edge_front[i], right_edge_front[i], left_edge_front[i+1]])
        faces.append([right_edge_front[i], right_edge_front[i+1], left_edge_front[i+1]])
        face_colors.extend(['lightblue', 'lightblue'])
    
    # Back face - triangulate between the back edges
    for i in range(n_segments - 1):
        # Add triangles to fill the back face
        faces.append([left_edge_back[i], left_edge_back[i+1], right_edge_back[i]])
        faces.append([right_edge_back[i], left_edge_back[i+1], right_edge_back[i+1]])
        face_colors.extend(['lightgreen', 'lightgreen'])
    
    # Left side face - connect left front and back edges
    for i in range(n_segments - 1):
        # Add quad (as two triangles) for each segment
        faces.append([left_edge_front[i], left_edge_front[i+1], left_edge_back[i]])
        faces.append([left_edge_back[i], left_edge_front[i+1], left_edge_back[i+1]])
        face_colors.extend(['coral', 'coral'])
    
    # Right side face - connect right front and back edges
    for i in range(n_segments - 1):
        # Add quad (as two triangles) for each segment
        faces.append([right_edge_front[i], right_edge_back[i], right_edge_front[i+1]])
        faces.append([right_edge_back[i], right_edge_back[i+1], right_edge_front[i+1]])
        face_colors.extend(['gold', 'gold'])
    
    # Plot on both axes
    for ax in [ax1, ax2]:
        # Clear any existing content
        ax.clear()
        
        # Create a Poly3DCollection with the faces
        collection = Poly3DCollection(
            faces,
            facecolors=face_colors,
            linewidths=0.05,
            edgecolors='black',
            alpha=0.95
        )
        
        # Enable proper depth sorting
        collection.set_sort_zpos(0)
        
        # Add the collection to the axis
        ax.add_collection3d(collection)
        
        # Set up the axes
        ax.set_axis_off()
        ax.set_box_aspect([1,1,1])
        
        # Make the shape appear larger by significantly reducing the axis limits
        # This effectively zooms in on the shape
        limit = size * 0.7  # Much smaller limit to zoom in on the shape
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
    
    # Set initial viewing angles
    ax1.view_init(elev=20, azim=45)  # front view
    ax2.view_init(elev=-20, azim=225)  # back view
    
    fig.suptitle('Thick Triangle\nClick and drag to rotate!', y=0.95)
    fig.canvas.draw_idle()

def plot_edged_torus(ax1, ax2, R, r):
    """
    Generate a lenticular torus with different colors for top and bottom halves
    
    Parameters:
    - ax1, ax2: The two axes to plot on (front and back views)
    - R: Major radius of the torus
    - r: Scale factor for the cross-section
    - inner_hole_size: Controls the size of the inner hole (0.1 = small hole, 1.0 = large hole)
    """
    inner_hole_size=0.05
    
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
        ax.set_axis_off()
        ax.set_box_aspect([1,1,1])
        
        PLOT_MARGIN = 1.2  # Extra space around the shape (20% margin)
        MAX_CROSS_SECTION_SCALE = 2.0  # Maximum scale factor for the outer part
        limit = (R + r * (MAX_CROSS_SECTION_SCALE - inner_hole_size)) * PLOT_MARGIN  # Adjust limit based on the wider cross-section
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
    
    # Set initial viewing angles
    ax1.view_init(elev=20, azim=45)  # front view
    ax2.view_init(elev=-20, azim=225)  # back view
    
    fig.suptitle('Edged Torus\nClick and drag to rotate!', y=0.95)
    fig.canvas.draw_idle()

def update(val):
    k = int(k_slider.val)
    twist_multiplier = int(twist_slider.val)
    
    ax1.clear()
    ax2.clear()
    
    # Get the current selected shape
    selected_shape = shape_selector.value_selected
    
    if selected_shape == 'Normal Torus':
        plot_normal_torus(ax1, ax2, R, r)
        return
    
    elif selected_shape == 'Double Torus':
        plot_double_torus(ax1, ax2, R, r)
        return
    
    elif selected_shape == 'Pointy Torus':
        plot_pointy_torus(ax1, ax2, R, r)
        return
    
    elif selected_shape == 'Twisted Pointy Torus':
        plot_twisted_pointy_torus(ax1, ax2, R, r)
        return
    
    elif selected_shape == 'Pointy Cylinder':
        plot_pointy_cylinder(ax1, ax2, R, r)
        return
    
    elif selected_shape == 'Cube':
        plot_cube(ax1, ax2, r)
        return
    
    elif selected_shape == 'Octahedron':
        plot_octahedron(ax1, ax2, r)
        return
    
    elif selected_shape == 'Edged Ball':
        plot_edged_ball(ax1, ax2, r)
        return
    
    elif selected_shape == 'Tetrahedron':
        plot_tetrahedron(ax1, ax2, r)
        return
    
    elif selected_shape == 'Thick Triangle':
        plot_thick_triangle(ax1, ax2, r)
        return
    
    elif selected_shape == 'Edged Torus':
        plot_edged_torus(ax1, ax2, R, r)
        return

    plot_polygon_torus(ax1, ax2, R, r, k, twist_multiplier, points_per_side)

def update_twist_slider(val):
    k = int(k_slider.val)
    twist_slider.valmax = k
    twist_slider.ax.set_xlim(0, k)
    if twist_slider.val > k:
        twist_slider.set_val(0)
    update(val)

def reset(event):
    k_slider.reset()
    twist_slider.reset()

def on_shape_select(label):
    # Show/hide sliders based on selection
    if label == 'Polygon Torus':
        k_slider_ax.set_visible(True)
        twist_slider_ax.set_visible(True)
        k_slider.set_active(True)
        twist_slider.set_active(True)
    else:
        k_slider_ax.set_visible(False)
        twist_slider_ax.set_visible(False)
        k_slider.set_active(False)
        twist_slider.set_active(False)
    
    # Save the selected shape
    save_selected_shape(label)
    
    # Redraw the figure to update visibility
    fig.canvas.draw_idle()
    update(None)

# Create figure with two subplots and space for sliders and controls
fig = plt.figure(figsize=(10, 7))  # Increased figure height

# Adjust subplot positions to make room for controls
plt.subplots_adjust(top=0.85, bottom=0.25)  # Increased bottom margin

ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# Parameters for the torus
R = 4
r = 2.5
points_per_side = 10

# Add sliders with adjusted positions and reduced width
k_slider_ax = plt.axes([0.75, 0.12, 0.20, 0.03])  # Moved to right, reduced width
twist_slider_ax = plt.axes([0.75, 0.07, 0.20, 0.03])  # Moved to right, reduced width

k_slider = Slider(
    ax=k_slider_ax,
    label='Number of Sides',
    valmin=3,
    valmax=12,
    valinit=4,
    valstep=1
)

twist_slider = Slider(
    ax=twist_slider_ax,
    label='Twist (faces)',
    valmin=0,
    valmax=4,
    valinit=0,
    valstep=1
)

# Get the last selected shape
last_shape = load_last_shape()

# Replace the shape selector with adjusted position and size
shape_selector_ax = plt.axes([0.02, 0.02, 0.20, 0.18])  # Increased height, moved to bottom left
shape_options = ('Polygon Torus', 'Normal Torus', 'Double Torus', 'Pointy Torus', 
     'Twisted Pointy Torus', 'Pointy Cylinder', 'Cube', 'Octahedron', 'Edged Ball', 'Tetrahedron', 'Thick Triangle', 'Edged Torus')
shape_selector = RadioButtons(
    shape_selector_ax, 
    shape_options,
    active=shape_options.index(last_shape)  # Set active based on last selected shape.
)

# Connect the callback to the radio buttons
shape_selector.on_clicked(on_shape_select)

# Connect the sliders to the update function
k_slider.on_changed(update_twist_slider)
twist_slider.on_changed(update)

# Connect the mouse handlers
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
fig.canvas.mpl_connect('button_press_event', on_button_press)
fig.canvas.mpl_connect('button_release_event', on_button_release)

# Initial plot
update(4)

plt.show()
