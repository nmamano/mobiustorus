import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

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

def plot_normal_torus(ax1, ax2, R, r):
    # Generate a normal (circular) torus
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, 2*np.pi, 100)
    u, v = np.meshgrid(u, v)
    
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    
    # Plot on both axes
    for ax in [ax1, ax2]:
        surface = ax.plot_surface(x, y, z, color='lightblue', alpha=0.9)
        ax.set_axis_off()
        ax.set_box_aspect([1,1,1])
        
        limit = (R + r) * 1.2
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
    
    # Set initial viewing angles
    ax1.view_init(elev=20, azim=45)  # front view
    ax2.view_init(elev=-20, azim=225)  # back view
    
    fig.suptitle('Normal Torus\nClick and drag to rotate!', y=0.95)
    fig.canvas.draw_idle()

def plot_double_torus(ax1, ax2, R, r):
    # Generate a torus with a doughnut cross-section (double torus)
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, 2*np.pi, 100)
    u, v = np.meshgrid(u, v)
    
    # Parameters for the double torus
    R_main = R  # Main torus radius
    r_outer = r  # Outer radius of the cross-section
    r_inner = r/2  # Inner radius of the cross-section
    
    # Outer torus
    x_outer = (R_main + r_outer * np.cos(v)) * np.cos(u)
    y_outer = (R_main + r_outer * np.cos(v)) * np.sin(u)
    z_outer = r_outer * np.sin(v)
    
    # Inner torus (smaller doughnut inside the cross-section)
    # We create a smaller torus within the cross-section
    x_inner = (R_main + r_inner * np.cos(v + np.pi)) * np.cos(u)
    y_inner = (R_main + r_inner * np.cos(v + np.pi)) * np.sin(u)
    z_inner = r_inner * np.sin(v + np.pi)
    
    # Plot on both axes
    for ax in [ax1, ax2]:
        # Plot outer torus with transparency
        outer_surface = ax.plot_surface(x_outer, y_outer, z_outer, color='lightblue', alpha=0.6)
        # Plot inner torus
        inner_surface = ax.plot_surface(x_inner, y_inner, z_inner, color='coral', alpha=0.9)
        
        ax.set_axis_off()
        ax.set_box_aspect([1,1,1])
        
        limit = (R_main + r_outer) * 1.2
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
    
    # Set initial viewing angles
    ax1.view_init(elev=20, azim=45)  # front view
    ax2.view_init(elev=-20, azim=225)  # back view
    
    fig.suptitle('Double Torus (Torus with Doughnut Cross-section)\nClick and drag to rotate!', y=0.95)
    fig.canvas.draw_idle()

def plot_pointy_torus(ax1, ax2, R, r):
    # Generate a torus with a cross-section that's a circle inscribed in a square
    # with one corner of the square still sharp
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, 2*np.pi, 100)
    u, v = np.meshgrid(u, v)
    
    # Parameters
    square_size = r  # Size of the square
    circle_radius = r * 0.7  # Radius of the inscribed circle
    point_scale = 0.7  # Scale factor to move the point closer to center (0.7 matches the cylinder)
    
    # Function to create the special cross-section
    def cross_section(angle):
        # Start with a circle
        x = circle_radius * np.cos(angle)
        y = circle_radius * np.sin(angle)
        
        # Check if we're in the top-right quadrant (for the sharp corner)
        if x > 0 and y > 0:
            # Make a sharp corner in the top-right quadrant, but closer to center
            return square_size * point_scale, square_size * point_scale
        
        # For other quadrants, keep the circle shape
        return x, y
    
    # Apply the cross-section function to each angle
    cross_x = np.zeros_like(v)
    cross_y = np.zeros_like(v)
    
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            cross_x[i,j], cross_y[i,j] = cross_section(v[i,j])
    
    # Generate the torus
    x = (R + cross_x) * np.cos(u)
    y = (R + cross_x) * np.sin(u)
    z = cross_y
    
    # Plot on both axes
    for ax in [ax1, ax2]:
        surface = ax.plot_surface(x, y, z, color='lightgreen', alpha=0.9)
        ax.set_axis_off()
        ax.set_box_aspect([1,1,1])
        
        limit = (R + square_size) * 1.2
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
    
    # Set initial viewing angles
    ax1.view_init(elev=20, azim=45)  # front view
    ax2.view_init(elev=-20, azim=225)  # back view
    
    fig.suptitle('Pointy Torus (One Sharp Corner)\nClick and drag to rotate!', y=0.95)
    fig.canvas.draw_idle()

def plot_twisted_pointy_torus(ax1, ax2, R, r):
    # Generate a torus with a cross-section that's a circle inscribed in a square
    # with one corner of the square still sharp
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, 2*np.pi, 100)
    u, v = np.meshgrid(u, v)
   
    # Parameters
    square_size = r  # Size of the square
    circle_radius = r * 0.7  # Radius of the inscribed circle
    point_scale = 0.7  # Scale factor to move the point closer to center (0.7 matches the cylinder)
   
    # Function to create the special cross-section
    def cross_section(angle, rotation):
        # Start with a circle
        x = circle_radius * np.cos(angle)
        y = circle_radius * np.sin(angle)
       
        # Apply rotation to the angle to determine which quadrant gets the sharp corner
        rotated_angle = angle + rotation
        rotated_x = circle_radius * np.cos(rotated_angle)
        rotated_y = circle_radius * np.sin(rotated_angle)
       
        # Check if we're in the top-right quadrant after rotation (for the sharp corner)
        if rotated_x > 0 and rotated_y > 0:
            # Calculate the sharp corner position
            sharp_x = square_size * point_scale
            sharp_y = square_size * point_scale
           
            # Rotate the sharp corner back to the original coordinate system
            cos_rot = np.cos(-rotation)
            sin_rot = np.sin(-rotation)
            final_x = sharp_x * cos_rot - sharp_y * sin_rot
            final_y = sharp_x * sin_rot + sharp_y * cos_rot
           
            return final_x, final_y
       
        # For other quadrants, keep the circle shape
        return x, y
   
    # Apply the cross-section function to each angle
    cross_x = np.zeros_like(v)
    cross_y = np.zeros_like(v)
   
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            # Make the cross-section rotate as we go around the torus
            # u[i,j] ranges from 0 to 2π, so we use it as the rotation angle
            cross_x[i,j], cross_y[i,j] = cross_section(v[i,j], u[i,j])
   
    # Generate the torus
    x = (R + cross_x) * np.cos(u)
    y = (R + cross_x) * np.sin(u)
    z = cross_y
   
    # Plot on both axes
    for ax in [ax1, ax2]:
        surface = ax.plot_surface(x, y, z, color='lightgreen', alpha=0.9)
        ax.set_axis_off()
        ax.set_box_aspect([1,1,1])
       
        limit = (R + square_size) * 1.2
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
   
    # Set initial viewing angles
    ax1.view_init(elev=20, azim=45)  # front view
    ax2.view_init(elev=-20, azim=225)  # back view
   
    fig.suptitle('Pointy Torus (Rotating Sharp Corner)\nClick and drag to rotate!', y=0.95)
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

def plot_polygon_torus(ax1, ax2, R, r, k, twist_multiplier, points_per_side):
    # Calculate the base twist angle
    base_twist = 2 * np.pi / k
    total_twist = base_twist * twist_multiplier
    
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
    
    # Calculate the greatest common divisor between k and twist_multiplier
    # If twist_multiplier is 0, use k as the number of colors
    num_colors = gcd(k, twist_multiplier) if twist_multiplier > 0 else k
    
    # Generate different colors for each face based on GCD
    colors = plt.cm.rainbow(np.linspace(0, 1, num_colors))

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
    ax2.view_init(elev=-20, azim=225)  # back view (note the negative elevation)
    
    if twist_multiplier == 0:
        twist_text = ""
    else:
        twist_text = f" with {twist_multiplier} face{'s' if twist_multiplier > 1 else ''} twist"
    
    # Add GCD information to the title
    gcd_text = f"\nGCD({k}, {twist_multiplier}) = {num_colors} colors" if twist_multiplier > 0 else f"\n{k} colors"
    fig.suptitle(f'Torus with {k}-sided cross-section{twist_text}{gcd_text}\nClick and drag to rotate!', y=0.95)
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
    
    # Define colors for each face
    face_colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan', 'magenta']
    
    # Plot on both axes
    for ax in [ax1, ax2]:
        # Plot each face of the octahedron
        for i, face in enumerate(faces):
            face = np.array(face)
            # Create x, y, z coordinates for the face
            x = face[:, 0]
            y = face[:, 1]
            z = face[:, 2]
            
            # Create a triangular grid for the face
            tri = np.array([[0, 1, 2]])
            
            # Plot the face as a triangular surface
            ax.plot_trisurf(x, y, z, triangles=tri, color=face_colors[i], alpha=0.9)
            
            # Draw the edges of the face
            for j in range(3):
                ax.plot([x[j], x[(j+1)%3]], [y[j], y[(j+1)%3]], [z[j], z[(j+1)%3]], 
                        color='black', linewidth=1.5)
        
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

# Function to update the plot when parameters change
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

# Replace the shape selector with adjusted position and size
shape_selector_ax = plt.axes([0.02, 0.02, 0.20, 0.18])  # Increased height, moved to bottom left
shape_selector = RadioButtons(
    shape_selector_ax, 
    ('Polygon Torus', 'Normal Torus', 'Double Torus', 'Pointy Torus', 
     'Twisted Pointy Torus', 'Pointy Cylinder', 'Cube', 'Octahedron'),
    active=0  # Default to Polygon Torus
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
