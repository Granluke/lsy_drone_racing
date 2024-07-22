"""Helper functions to randomly initialize the drone for the training."""
import numpy as np

def find_closest_traj_point(waypoints:np.ndarray, X_GOAL:np.ndarray) -> int:
    """Find the closest point on the trajectory for each waypoint
    Args:
        waypoints (np.ndarray): The waypoints of the trajectory.
        X_GOAL (np.ndarray): The current state of the drone.
    Returns:
        int: The index of the closest point on the trajectory."""

    waypoints = waypoints[:,None] # create a tensor of shape (N,1,3)
    dist = np.linalg.norm(waypoints - X_GOAL[:, :3], axis=2) # Do a tensor operation
    closest_idx = np.argmin(dist, axis=1) # Find the index of the minimum distance
    return closest_idx

def find_closest_gate(X_GOAL:np.ndarray, initial_info:dict, waypoint_idx:np.array) -> dict:
    """Find the coming gate for each waypoint. Only a hardcoded version is available.
    Args:
        X_GOAL (np.ndarray): The current state of the drone.
        initial_info (dict): The initial information of the environment.
        waypoint_idx (np.array): The index of the waypoints.
    Returns:
        dict: The index of the coming gate for each waypoint.
    """

    wp_gate_match = {i:-1 for i in range(len(waypoint_idx))}
    wp_gate_match[0] = 0
    wp_gate_match[1] = 0
    wp_gate_match[2] = 0
    
    wp_gate_match[3] = 1
    wp_gate_match[4] = 1
    wp_gate_match[5] = 1
    
    wp_gate_match[6] = 2
    wp_gate_match[7] = 2
    wp_gate_match[8] = 2
    wp_gate_match[9] = 2
    
    wp_gate_match[10] = 3
    wp_gate_match[11] = 3
    wp_gate_match[12] = 3
    wp_gate_match[13] = 3
    wp_gate_match[14] = -1
    # wp_gate_match[15] = -1
    return wp_gate_match