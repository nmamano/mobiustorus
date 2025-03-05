from shape_utils import *

def plot_normal_torus(ax1, ax2, R, r):
    """Generate a normal (circular) torus"""
    def circle_cross_section(angle):
        return r * np.cos(angle), r * np.sin(angle)
    
    create_torus(ax1, ax2, R, r, circle_cross_section, 
                 colors='lightblue', 
                 title="Normal Torus")
