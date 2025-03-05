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
