from shape_utils import *

def plot_octahedron(ax1, ax2, r):
    """Generate an octahedron (8-faced polyhedron)"""
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
    
    limit = size * 1.5
    set_common_plot_properties(ax1, ax2, "Octahedron", limit) 
