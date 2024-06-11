import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, pathpatch_2d_to_3d
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev
from math import sqrt

obstacle_dimensions = {'shape': 'cylinder', 'height': 1.05, 'radius': 0.05}

def create_gate(x, y, yaw, gate_type):
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


def create_cylinder(x, y, z, height, radius):
    z_values = np.linspace(0, height, 30)
    theta = np.linspace(0, 2 * np.pi, 30)
    theta_grid, z_grid = np.meshgrid(theta, z_values)
    x_grid = radius * np.cos(theta_grid) + x
    y_grid = radius * np.sin(theta_grid) + y
    z_grid += z
    return x_grid, y_grid, z_grid

def point_in_gate(point, gate_center, yaw, height, buffer):
    gate_width = 0.45 + 2 * buffer
    gate_height = height + 0.45 / 2 + 2 * buffer
    rel_point = point - gate_center

    yaw_matrix = np.array([
        [np.cos(-yaw), -np.sin(-yaw), 0],
        [np.sin(-yaw), np.cos(-yaw), 0],
        [0, 0, 1]
    ])
    
    local_point = np.dot(yaw_matrix, rel_point)

    return (-gate_width / 2 <= local_point[0] <= gate_width / 2 and
            -gate_height / 2 <= local_point[2] <= gate_height / 2)

def line_intersects_plane(p1, p2, plane_point, plane_normal):
    line_vec = p2 - p1
    plane_point_vec = plane_point - p1
    dot_product = np.dot(plane_normal, line_vec)
    
    if np.abs(dot_product) < 1e-6:
        return False, None
    
    t = np.dot(plane_normal, plane_point_vec) / dot_product
    
    if 0 <= t <= 1:
        intersection_point = p1 + t * line_vec
        return True, intersection_point
    
    return False, None

def create_waypoints(gates, start_point):
    waypoints = [start_point]
    buffer = 0.2
    before_after_points = []
    go_around_points = []
    intersection_points = []
    avoidance_distance = 0.15

    for i, gate in enumerate(gates):
        x, y, z_center, roll, pitch, yaw, gate_type = gate
        if gate_type == 0:
            height = 1.0
        else:
            height = 0.525
        z = height

        tmp_wp_1 = [x - buffer * np.sin(yaw), y + buffer * np.cos(yaw), z]
        tmp_wp_2 = [x + buffer * np.sin(yaw), y - buffer * np.cos(yaw), z]

        prev_wp = waypoints[-1]
        dist_tmp_wp_1 = np.linalg.norm([prev_wp[0] - tmp_wp_1[0], prev_wp[1] - tmp_wp_1[1]])
        dist_tmp_wp_2 = np.linalg.norm([prev_wp[0] - tmp_wp_2[0], prev_wp[1] - tmp_wp_2[1]])

        if dist_tmp_wp_2 < dist_tmp_wp_1:
            after_wp = tmp_wp_1
            before_wp = tmp_wp_2
        else:
            before_wp = tmp_wp_1
            after_wp = tmp_wp_2
        
        waypoints.append(before_wp)
        waypoints.append([x, y, z])
        waypoints.append(after_wp)

        before_after_points.append((before_wp, after_wp))
        
        # Check if the line to the next gate goes through the current gate
        if i < len(gates) - 1:
            next_gate = gates[i + 1]
            next_x, next_y, next_z_center, next_roll, next_pitch, next_yaw, next_gate_type = next_gate
            line_start = np.array([after_wp[0], after_wp[1], after_wp[2]])
            line_end = np.array([next_x, next_y, next_z_center])
            plane_point = np.array([x, y, z])
            plane_normal = np.array([np.sin(yaw), -np.cos(yaw), 0])

            intersects, intersection_point = line_intersects_plane(line_start, line_end, plane_point, plane_normal)
            if intersects:
                intersection_points.append(intersection_point)
            
            if intersects and point_in_gate(intersection_point, np.array([x, y, z]), yaw, height, buffer):
                print(f"Gate {i} intersects with line to gate {i + 1}")
               # Calculate direction to next gate
                # Calculate direction from gate center to intersection point
                intersection_vector = intersection_point - np.array([x, y, z_center])
                intersection_vector /= np.linalg.norm(intersection_vector)

                # Calculate the avoidance point
                avoidance_wp = plane_point + intersection_vector * (0.45/sqrt(2) + avoidance_distance)

                waypoints.append(avoidance_wp)
                go_around_points.append((avoidance_wp))
            else:
                go_around_points.append(([]))
        else:
            go_around_points.append(([]))

    return np.array(waypoints), before_after_points, go_around_points, intersection_points

def plot_gates_and_cylinders(gates, cylinders, path_before_obstacle_avoidance, path_after_obstacle_avoidance, waypoints, before_after_points, go_around_points, intersection_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    all_points = []
    
    for i, gate in enumerate(gates):
        x, y, z_center, roll, pitch, yaw, gate_type = gate
        points = create_gate(x, y, yaw, gate_type)
        all_points.append(points)
        verts = [list(zip(points[:, 0], points[:, 1], points[:, 2]))]
        poly = Poly3DCollection(verts, alpha=0.5, linewidths=1, edgecolors='r')
        poly.set_facecolor([0.5, 0.5, 1])
        ax.add_collection3d(poly)
        ax.text(x, y, z_center, str(i), color='black')

        # Plot before and after waypoints
        before_wp, after_wp = before_after_points[i]
        ax.scatter(*before_wp, color='blue', s=50)
        ax.scatter(*after_wp, color='red', s=50)

        go_around_wp1 = go_around_points[i]
        if len(go_around_wp1) > 0:
            ax.scatter(*go_around_wp1, color='green', s=50)
           # ax.scatter(*go_around_wp2, color='green', s=50)
        
    
    for cylinder in cylinders:
        x, y, z, roll, pitch, yaw = cylinder
        height = obstacle_dimensions['height']
        radius = obstacle_dimensions['radius']
        x_grid, y_grid, z_grid = create_cylinder(x, y, z, height, radius)
        ax.plot_surface(x_grid, y_grid, z_grid, color='b', alpha=0.5)
        all_points.append(np.column_stack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten())))

    for intersection_point in intersection_points:
        if intersection_point is not None:
            ax.scatter(*intersection_point, color='purple', s=50)

    # Convert all_points to a single numpy array for easy limit calculation
    all_points = np.concatenate(all_points, axis=0)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set limits
    ax.set_xlim([all_points[:, 0].min() - 1, all_points[:, 0].max() + 1])
    ax.set_ylim([all_points[:, 1].min() - 1, all_points[:, 1].max() + 1])
    ax.set_zlim([all_points[:, 2].min() - 1, all_points[:, 2].max() + 1])

    # Plot the path
    ax.plot(path_before_obstacle_avoidance[0], path_before_obstacle_avoidance[1], path_before_obstacle_avoidance[2], 'g', label='Path')
    ax.plot(path_after_obstacle_avoidance[0], path_after_obstacle_avoidance[1], path_after_obstacle_avoidance[2], 'r', label='Path')
    
    # Plot waypoints
    for wp in waypoints:
        ax.scatter(*wp, color='black', s=50)

    plt.legend()
    plt.show()

def check_path_collision(path, cylinders, buffer=0.15):
    height = obstacle_dimensions['height']
    radius = obstacle_dimensions['radius']
    radius = radius + buffer
    for cylinder in cylinders:
        x_c, y_c, z_c, _, _, _ = cylinder
        for p in path.T:
            x, y, z = p
            if (x - x_c) ** 2 + (y - y_c) ** 2 <= radius ** 2 and 0 <= z <= height:
                print(f"Collision at {x}, {y}, {z}")
                return True
    return False

def adjust_waypoints(waypoints, cylinders, buffer=0.1):
    extra_buffer = 0.2
    adjusted_waypoints = []
    height = obstacle_dimensions['height']
    radius = obstacle_dimensions['radius']
    radius = radius + buffer
    for waypoint in waypoints:
        x, y, z = waypoint
        collision = False
        for cylinder in cylinders:
            x_c, y_c, z_c, _, _, _ = cylinder
            if (x - x_c) ** 2 + (y - y_c) ** 2 <= radius ** 2 and 0 <= z <= height:
                collision = True
                print(f"Collision at {x}, {y}, {z}")
                # Adjust the waypoint by moving it away from the cylinder
                angle = np.arctan2(y - y_c, x - x_c)
                x += np.cos(angle) * extra_buffer
                y += np.sin(angle) * extra_buffer
                print(f"Adjusted to {x}, {y}, {z}")
        adjusted_waypoints.append([x, y, z])
    return np.array(adjusted_waypoints)

def adjust_path(path, cylinders, buffer=0.15):
    extra_buffer = 0.3
    adjusted_path = [[], [], []]
    height = obstacle_dimensions['height']
    radius = obstacle_dimensions['radius']
    radius = radius + buffer
    r_squared = radius ** 2
    for x, y, z in zip(*path):
        collision = False
        for cylinder in cylinders:
            x_c, y_c, z_c, _, _, _ = cylinder
            distance_calc = (x - x_c) ** 2 + (y - y_c) ** 2
            if distance_calc <= r_squared and 0 <= z <= height:
                collision = True
                # Adjust the point by moving it away from the cylinder
                angle = np.arctan2(y - y_c, x - x_c)
                distance = np.sqrt(distance_calc)
                # Apply a quadratic adjustment factor
                adjustment_factor = np.clip(((radius - distance) / radius) ** 2, 0, 1)
                x += np.cos(angle) * extra_buffer * adjustment_factor
                y += np.sin(angle) * extra_buffer * adjustment_factor
        adjusted_path[0].append(x)
        adjusted_path[1].append(y)
        adjusted_path[2].append(z)
    
    return np.array(adjusted_path)

def calc_best_path(gates, cylinders, start_point, t, plot=True):
    print("Starting point", start_point)
    waypoints, before_after_points, go_around_points, intersection_points = create_waypoints(gates, start_point)
    tck, u = splprep(waypoints.T, s=0.1)
    
    path1 = splev(t, tck)
    path = path1
    if check_path_collision(np.array(path), cylinders):
        print("Path collides with obstacles, adjusting waypoints...")
        path = adjust_path(path, cylinders)
        # recompute splprep and path with splev
        tck, u = splprep(path, s=0.1)
        path3 = splev(t, tck)
        path = path3

    if plot:
        plot_gates_and_cylinders(gates, cylinders, path1, path, waypoints, before_after_points, go_around_points, intersection_points)


    return path, waypoints




GATES = [
    [0.45, -1.0, 0, 0, 0, 2.35, 1],  # Example gate, z_center adjusted for low obstacle
    [1.0, -1.55, 0, 0, 0, -0.78, 0], 
    [0.0, 0.5, 0, 0, 0, 0, 1], 
    [-0.5, -0.5, 0, 0, 0, 3.14, 0]
]

OBSTACLES = [
    [1.0, -0.5, 0, 0, 0, 0],
    [0.5, -1.5, 0, 0, 0, 0],
    [-0.5, 0, 0, 0, 0, 0],
    [0, 1.0, 0, 0, 0, 0]
]
duration = 10
CTRL_FREQ = 30
T = np.linspace(0, 1, int(duration * CTRL_FREQ))

START_POINT = [0.9339007658017658, 0.9934538571454128, 0.3]

#calc_best_path(GATES, OBSTACLES, START_POINT, T, plot=True)