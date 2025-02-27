import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import math

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

# Function to update the plot when parameters change
def update(val):
    k = int(k_slider.val)
    base_twist = 2 * np.pi / k
    twist_multiplier = int(twist_slider.val)
    total_twist = base_twist * twist_multiplier
    
    ax1.clear()
    ax2.clear()
    
    if normal_torus_button.label.get_text() == 'Normal Torus':
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
        return
    
    elif normal_torus_button.label.get_text() == 'Double Torus':
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
        return
    
    elif normal_torus_button.label.get_text() == 'Square-Circle Torus':
        # Generate a torus with a cross-section that's a circle inscribed in a square
        # with one corner of the square still sharp
        u = np.linspace(0, 2*np.pi, 100)
        v = np.linspace(0, 2*np.pi, 100)
        u, v = np.meshgrid(u, v)
        
        # Parameters
        square_size = r  # Size of the square
        circle_radius = r * 0.7  # Radius of the inscribed circle
        
        # Function to create the special cross-section
        def cross_section(angle):
            # Start with a circle
            x = circle_radius * np.cos(angle)
            y = circle_radius * np.sin(angle)
            
            # Check if we're in the top-right quadrant (for the sharp corner)
            if x > 0 and y > 0:
                # Make a sharp corner in the top-right quadrant
                return square_size, square_size
            
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
        
        fig.suptitle('Square-Circle Torus (One Sharp Corner)\nClick and drag to rotate!', y=0.95)
        fig.canvas.draw_idle()
        return
    
    elif normal_torus_button.label.get_text() == 'Pie':
        # Generate a pie/fat cylinder shape
        u = np.linspace(0, 2*np.pi, 100)
        # Make the height match the torus cross-section (2r is the diameter, so r is the radius)
        v = np.linspace(-r/3, r/3, 100)  # Further reduced height
        u, v = np.meshgrid(u, v)
        
        # Create a fat cylinder/pie shape
        x = R * np.cos(u)
        y = R * np.sin(u)
        z = v
        
        # Plot on both axes
        for ax in [ax1, ax2]:
            # Plot the cylinder surface
            surface = ax.plot_surface(x, y, z, color='yellow', alpha=0.9)
            
            # Add the top and bottom circular faces
            theta = np.linspace(0, 2*np.pi, 100)
            for height in [-r/3, r/3]:  # Adjusted heights
                circle_x = R * np.cos(theta)
                circle_y = R * np.sin(theta)
                circle_z = np.ones_like(theta) * height
                ax.plot(circle_x, circle_y, circle_z, color='orange', linewidth=2)
                
                # Fill the circular faces
                face_x = np.zeros((2, 100))
                face_y = np.zeros((2, 100))
                face_z = np.zeros((2, 100))
                
                # Create a filled circle by connecting points to the center
                for i in range(100):
                    face_x[0, i] = 0
                    face_y[0, i] = 0
                    face_z[0, i] = height
                    face_x[1, i] = circle_x[i]
                    face_y[1, i] = circle_y[i]
                    face_z[1, i] = circle_z[i]
                    
                    # Draw radial lines to fill the circle
                    ax.plot(face_x[:, i], face_y[:, i], face_z[:, i], color='orange', linewidth=0.5)
            
            ax.set_axis_off()
            ax.set_box_aspect([1,1,1])
            
            limit = R * 1.2
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_zlim(-r*1.2, r*1.2)  # Keep the z-limits the same as before for consistency
        
        # Set initial viewing angles
        ax1.view_init(elev=20, azim=45)  # front view
        ax2.view_init(elev=-20, azim=225)  # back view
        
        fig.suptitle('Pie (Fat Cylinder)\nClick and drag to rotate!', y=0.95)
        fig.canvas.draw_idle()
        return
    
    elif normal_torus_button.label.get_text() == 'Cube':
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
        return

    # Generate vertices of the regular polygon
    angles = np.linspace(0, 2*np.pi, k+1)
    vertices_x = r * np.cos(angles)
    vertices_y = r * np.sin(angles)

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

def toggle_torus(event):
    if normal_torus_button.label.get_text() == 'Normal Torus':
        normal_torus_button.label.set_text('Double Torus')
        k_slider.set_active(False)
        twist_slider.set_active(False)
    elif normal_torus_button.label.get_text() == 'Double Torus':
        normal_torus_button.label.set_text('Square-Circle Torus')
        k_slider.set_active(False)
        twist_slider.set_active(False)
    elif normal_torus_button.label.get_text() == 'Square-Circle Torus':
        normal_torus_button.label.set_text('Pie')
        k_slider.set_active(False)
        twist_slider.set_active(False)
    elif normal_torus_button.label.get_text() == 'Pie':
        normal_torus_button.label.set_text('Cube')
        k_slider.set_active(False)
        twist_slider.set_active(False)
    elif normal_torus_button.label.get_text() == 'Cube':
        normal_torus_button.label.set_text('Polygon Torus')
        k_slider.set_active(True)
        twist_slider.set_active(True)
    else:  # 'Polygon Torus'
        normal_torus_button.label.set_text('Normal Torus')
        k_slider.set_active(False)
        twist_slider.set_active(False)
    update(None)

# Create figure with two subplots and space for sliders
fig = plt.figure(figsize=(10, 6))  # Reduced from (20, 12) to (16, 9)

# Adjust subplot positions to reduce white space
plt.subplots_adjust(top=0.9, bottom=0.15)

ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# Parameters for the torus
R = 4
r = 2.5
points_per_side = 10

# Add sliders
k_slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
twist_slider_ax = plt.axes([0.2, 0.02, 0.6, 0.03])

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

# Add toggle button for normal torus
normal_button_ax = plt.axes([0.85, 0.02, 0.12, 0.05])
normal_torus_button = Button(normal_button_ax, 'Polygon Torus')
normal_torus_button.on_clicked(toggle_torus)

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
