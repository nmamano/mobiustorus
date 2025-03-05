from shape_utils import *

def plot_tetrahedron(ax1, ax2, r):
    """Generate a tetrahedron (4-faced polyhedron)"""
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
    
    # Set wider limits to avoid clipping
    limit = size * 1.8
    set_common_plot_properties(ax1, ax2, "Tetrahedron", limit) 
