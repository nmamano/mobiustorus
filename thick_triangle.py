from shape_utils import *

def plot_thick_triangle(ax1, ax2, r):
    """Generate a thick triangle with varying thickness"""
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
    
    # Make the shape appear larger by significantly reducing the axis limits
    # This effectively zooms in on the shape
    limit = size * 0.7  # Much smaller limit to zoom in on the shape
    set_common_plot_properties(ax1, ax2, "Thick Triangle", limit) 
