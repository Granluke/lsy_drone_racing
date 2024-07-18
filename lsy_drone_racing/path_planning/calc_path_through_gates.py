import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, pathpatch_2d_to_3d
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev
from math import sqrt

OBSTACLE_DIMENSIONS = {'shape': 'cylinder', 'height': 1.05, 'radius': 0.05}


def create_gate(self, x, y, yaw, gate_type):
    if gate_type == 0:
        height = 1.0
    else:
        height = 0.525
    
    edge_length = 0.45
    z_bottom = height - 0.45/2 
    z_top = height + 0.45/2
    
    # Define the square in the XZ plane
    points = np.array([
        [-edge_length / 2, 0, z_bottom],
        [edge_length / 2, 0, z_bottom],
        [edge_length / 2, 0, z_top],
        [-edge_length / 2, 0, z_top]
    ])

    # Apply yaw rotation
    yaw_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    points[:, :2] = np.dot(points[:, :2], yaw_matrix[:2, :2].T)
    
    # Adjust position to account for x, y
    points[:, 0] += x
    points[:, 1] += y
    
    return points


def create_cylinder(self, x, y, z, height, radius):
    z_values = np.linspace(0, height, 30)
    theta = np.linspace(0, 2 * np.pi, 30)
    theta_grid, z_grid = np.meshgrid(theta, z_values)
    x_grid = radius * np.cos(theta_grid) + x
    y_grid = radius * np.sin(theta_grid) + y
    z_grid += z
    return x_grid, y_grid, z_grid

   





if __name__ == '__main__':
    pass