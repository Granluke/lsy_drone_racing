"""Write your control strategy.

Then run:

    $ python scripts/sim --config config/getting_started.yaml

Tips:
    Search for strings `INSTRUCTIONS:` and `REPLACE THIS (START)` in this file.

    Change the code between the 5 blocks starting with
        #########################
        # REPLACE THIS (START) ##
        #########################
    and ending with
        #########################
        # REPLACE THIS (END) ####
        #########################
    with your own code.

    They are in methods:
        1) __init__
        2) compute_control
        3) step_learn (optional)
        4) episode_learn (optional)

"""

from __future__ import annotations  # Python 3.10 type hints

import numpy as np
from scipy import interpolate

from lsy_drone_racing.command import Command
from lsy_drone_racing.controller import BaseController
from lsy_drone_racing.utils import draw_trajectory
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
    z_bottom = height - 0.45
    z_top = height
    
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
    gate_height = height + 2 * buffer
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
        z = height - 0.45 / 2

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

def plot_gates_and_cylinders(gates, cylinders, path, before_after_points, go_around_points, intersection_points):
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
    ax.plot(path[0], path[1], path[2], 'g', label='Path')
    ax.scatter(path[0], path[1], path[2], color='r')  # Waypoints

    plt.legend()
    plt.show()

def check_path_collision(path, cylinders, buffer=0.2):
    for cylinder in cylinders:
        x_c, y_c, z_c, _, _, _ = cylinder
        height = obstacle_dimensions['height']
        radius = obstacle_dimensions['radius']
        radius = radius + buffer
        for p in path.T:
            x, y, z = p
            if (x - x_c) ** 2 + (y - y_c) ** 2 <= radius ** 2 and 0 <= z <= height:
                return True
    return False

def adjust_waypoints(waypoints, cylinders):
    buffer = 0.2
    adjusted_waypoints = []
    for waypoint in waypoints:
        x, y, z = waypoint
        collision = False
        for cylinder in cylinders:
            x_c, y_c, z_c, _, _, _ = cylinder
            height = obstacle_dimensions['height']
            radius = obstacle_dimensions['radius']
            radius = radius + buffer
            if (x - x_c) ** 2 + (y - y_c) ** 2 <= radius ** 2 and 0 <= z <= height:
                collision = True
                # Adjust the waypoint by moving it away from the cylinder
                angle = np.arctan2(y - y_c, x - x_c)
                x += np.cos(angle) * buffer
                y += np.sin(angle) * buffer
        adjusted_waypoints.append([x, y, z])
    return np.array(adjusted_waypoints)

def calc_best_path(gates, cylinders, start_point, plot=True):
    waypoints, before_after_points, go_around_points, intersection_points = create_waypoints(gates, start_point)
    waypoints = adjust_waypoints(waypoints, cylinders)
    tck, u = splprep(waypoints.T, s=0)
    unew = np.linspace(0, 1, 1000)
    path = splev(unew, tck)
    if check_path_collision(np.array(path), cylinders):
        print("Path collides with obstacles, adjusting waypoints...")
        waypoints = adjust_waypoints(waypoints, cylinders)
        tck, u = splprep(waypoints.T, s=0)
    if plot:
        path = splev(unew, tck)
        plot_gates_and_cylinders(gates, cylinders, path, before_after_points, go_around_points, intersection_points)
    return waypoints, tck



class Controller(BaseController):
    """Template controller class."""

    def __init__(
        self,
        initial_obs: np.ndarray,
        initial_info: dict,
        buffer_size: int = 100,
        verbose: bool = False,
    ):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori
            infromation contained in dictionary `initial_info`. Use this method to initialize
            constants, counters, pre-plan trajectories, etc.

        Args:
            initial_obs: The initial observation of the quadrotor's state
                [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            initial_info: The a priori information as a dictionary with keys 'symbolic_model',
                'nominal_physical_parameters', 'nominal_gates_pos_and_type', etc.
            buffer_size: Size of the data buffers used in method `learn()`.
            verbose: Turn on and off additional printouts and plots.
        """
        super().__init__(initial_obs, initial_info, buffer_size, verbose)
        # Save environment and control parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size

        # Store a priori scenario information.
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]

        # Reset counters and buffers.
        self.reset()
        self.episode_reset()

        #########################
        # REPLACE THIS (START) ##
        #########################

        # Example: Hard-code waypoints through the gates. Obviously this is a crude way of
        # completing the challenge that is highly susceptible to noise and does not generalize at
        # all. It is meant solely as an example on how the drones can be controlled
        waypoints = []
        start_point = [self.initial_obs[0], self.initial_obs[2], 0.3] 
        print(initial_obs)
        gates = self.NOMINAL_GATES
        


        self.waypoints, tck = calc_best_path(gates, self.NOMINAL_OBSTACLES, start_point, plot=False)
        duration = 10
        t = np.linspace(0, 1, int(duration * self.CTRL_FREQ))
        self.ref_x, self.ref_y, self.ref_z = interpolate.splev(t, tck)
        assert max(self.ref_z) < 2.5, "Drone must stay below the ceiling"

        if self.VERBOSE:
            # Draw the trajectory on PyBullet's GUI.
            draw_trajectory(initial_info, self.waypoints, self.ref_x, self.ref_y, self.ref_z)

        self._take_off = False
        self._setpoint_land = False
        self._land = False
        #########################
        # REPLACE THIS (END) ####
        #########################

    def compute_control(
        self,
        ep_time: float,
        obs: np.ndarray,
        reward: float | None = None,
        done: bool | None = None,
        info: dict | None = None,
    ) -> tuple[Command, list]:
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        INSTRUCTIONS:
            Re-implement this method to return the target position, velocity, acceleration,
            attitude, and attitude rates to be sent from Crazyswarm to the Crazyflie using, e.g., a
            `cmdFullState` call.

        Args:
            ep_time: Episode's elapsed time, in seconds.
            obs: The quadrotor's Vicon data [x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0].
            reward: The reward signal.
            done: Wether the episode has terminated.
            info: Current step information as a dictionary with keys 'constraint_violation',
                'current_target_gate_pos', etc.

        Returns:
            The command type and arguments to be sent to the quadrotor. See `Command`.
        """
        iteration = int(ep_time * self.CTRL_FREQ)

        #########################
        # REPLACE THIS (START) ##
        #########################

        # Handcrafted solution for getting_stated scenario.

        if not self._take_off:
            command_type = Command.TAKEOFF
            args = [0.3, 2]  # Height, duration
            self._take_off = True  # Only send takeoff command once
        else:
            step = iteration - 2 * self.CTRL_FREQ  # Account for 2s delay due to takeoff
            if ep_time - 2 > 0 and step < len(self.ref_x):
                target_pos = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
                target_vel = np.zeros(3)
                target_acc = np.zeros(3)
                target_yaw = 0.0
                target_rpy_rates = np.zeros(3)
                command_type = Command.FULLSTATE
                args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates, ep_time]
            # Notify set point stop has to be called every time we transition from low-level
            # commands to high-level ones. Prepares for landing
            elif step >= len(self.ref_x) and not self._setpoint_land:
                command_type = Command.NOTIFYSETPOINTSTOP
                args = []
                self._setpoint_land = True
            elif step >= len(self.ref_x) and not self._land:
                command_type = Command.LAND
                args = [0.0, 2.0]  # Height, duration
                self._land = True  # Send landing command only once
            elif self._land:
                command_type = Command.FINISHED
                args = []
            else:
                command_type = Command.NONE
                args = []

        #########################
        # REPLACE THIS (END) ####
        #########################

        return command_type, args

    def step_learn(
        self,
        action: list,
        obs: np.ndarray,
        reward: float | None = None,
        done: bool | None = None,
        info: dict | None = None,
    ):
        """Learning and controller updates called between control steps.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions,
            observations, rewards, done flags, and information dictionaries to learn, adapt, and/or
            re-plan.

        Args:
            action: Most recent applied action.
            obs: Most recent observation of the quadrotor state.
            reward: Most recent reward.
            done: Most recent done flag.
            info: Most recent information dictionary.

        """
        #########################
        # REPLACE THIS (START) ##
        #########################

        # Store the last step's events.
        self.action_buffer.append(action)
        self.obs_buffer.append(obs)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        self.info_buffer.append(info)

        # Implement some learning algorithm here if needed

        #########################
        # REPLACE THIS (END) ####
        #########################

    def episode_learn(self):
        """Learning and controller updates called between episodes.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions,
            observations, rewards, done flags, and information dictionaries to learn, adapt, and/or
            re-plan.

        """
        #########################
        # REPLACE THIS (START) ##
        #########################

        _ = self.action_buffer
        _ = self.obs_buffer
        _ = self.reward_buffer
        _ = self.done_buffer
        _ = self.info_buffer

        #########################
        # REPLACE THIS (END) ####
        #########################
