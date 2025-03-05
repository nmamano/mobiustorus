import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import sys
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import json
from matplotlib.animation import FuncAnimation

# Import all shape plotting functions
from normal_torus import plot_normal_torus
from double_torus import plot_double_torus
from pointy_torus import plot_pointy_torus
from twisted_pointy_torus import plot_twisted_pointy_torus
from polygon_torus import plot_polygon_torus
from pointy_cylinder import plot_pointy_cylinder
from cube import plot_cube
from octahedron import plot_octahedron
from edged_ball import plot_edged_ball
from tetrahedron import plot_tetrahedron
from thick_triangle import plot_thick_triangle
from edged_torus import plot_edged_torus

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

def update(val):
    k = int(k_slider.val)
    twist_multiplier = int(twist_slider.val)
    
    ax1.clear()
    ax2.clear()
    
    # Get the current selected shape and implementation
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

# Add these variables before creating the figure
rotation_active = True
anim = None
time_counter = 0  # Add this to track time for smooth oscillation

def rotate(frame):
    if not rotation_active:
        return
    
    global time_counter
    time_counter += 1
    
    # Rotate both views laterally (constant speed)
    ax1.azim = (ax1.azim + 1) % 360
    ax2.azim = (ax2.azim + 1) % 360
    
    # Smooth sinusoidal vertical motion
    # Using sin function to oscillate between -45 and 45 degrees
    elevation = 45 * np.sin(time_counter * 0.02)  # 0.02 controls speed
    ax1.elev = elevation
    ax2.elev = -elevation  # Keep the back view mirrored
    
    fig.canvas.draw_idle()

def toggle_rotation(event):
    global rotation_active
    rotation_active = not rotation_active
    rotation_button.label.set_text('Start Rotation' if not rotation_active else 'Stop Rotation')
    plt.draw()

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

# Set initial visibility of sliders based on last selected shape
k_slider_ax.set_visible(last_shape == 'Polygon Torus')
twist_slider_ax.set_visible(last_shape == 'Polygon Torus')
k_slider.set_active(last_shape == 'Polygon Torus')
twist_slider.set_active(last_shape == 'Polygon Torus')

# Connect the callback to the radio buttons
shape_selector.on_clicked(on_shape_select)

# Connect the sliders to the update function
k_slider.on_changed(update_twist_slider)
twist_slider.on_changed(update)

# Connect the mouse handlers
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
fig.canvas.mpl_connect('button_press_event', on_button_press)
fig.canvas.mpl_connect('button_release_event', on_button_release)

# Add this after creating the shape selector but before the initial plot
rotation_button_ax = plt.axes([0.02, 0.22, 0.20, 0.04])  # Position above shape selector
rotation_button = Button(rotation_button_ax, 'Stop Rotation')
rotation_button.on_clicked(toggle_rotation)

# Initial plot
update(4)

# Add this right before plt.show()
anim = FuncAnimation(fig, rotate, interval=50)  # 50ms interval = 20fps

plt.show()
