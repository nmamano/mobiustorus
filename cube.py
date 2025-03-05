from shape_utils import *

def plot_cube(ax1, ax2, r):
    """Generate a cube with colored faces"""
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
    
    limit = cube_size * 1.5
    set_common_plot_properties(ax1, ax2, "Cube", limit) 
