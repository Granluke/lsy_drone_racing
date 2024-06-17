import numpy as np
from scipy import interpolate

def create_waypoints_old(quadrotor_config: dict):
    CTRL_FREQ = quadrotor_config["ctrl_freq"]
    CTRL_TIMESTEP = 1 / CTRL_FREQ
    initial_obs = quadrotor_config["init_state"]
    init_x = initial_obs["init_x"]
    init_y = initial_obs["init_y"]
    # Store a priori scenario information.
    NOMINAL_GATES = quadrotor_config["gates"]
    NOMINAL_OBSTACLES = quadrotor_config["obstacles"]

    waypoints = []
    waypoints.append([init_x, init_y, 0.3])
    gates = NOMINAL_GATES
    z_low = 0.525
    z_high = 1.0
    waypoints.append([1, 0, z_low])
    waypoints.append([gates[0][0] + 0.2, gates[0][1] + 0.1, z_low])
    waypoints.append([gates[0][0] + 0.1, gates[0][1], z_low])
    waypoints.append([gates[0][0] - 0.1, gates[0][1], z_low])
    waypoints.append(
        [
            (gates[0][0] + gates[1][0]) / 2 - 0.7,
            (gates[0][1] + gates[1][1]) / 2 - 0.3,
            (z_low + z_high) / 2,
        ]
    )
    waypoints.append(
        [
            (gates[0][0] + gates[1][0]) / 2 - 0.5,
            (gates[0][1] + gates[1][1]) / 2 - 0.6,
            (z_low + z_high) / 2,
        ]
    )
    waypoints.append([gates[1][0] - 0.3, gates[1][1] - 0.2, z_high])
    waypoints.append([gates[1][0] + 0.2, gates[1][1] + 0.2, z_high])
    waypoints.append([gates[2][0], gates[2][1] - 0.4, z_low])
    waypoints.append([gates[2][0], gates[2][1] + 0.2, z_low])
    waypoints.append([gates[2][0], gates[2][1] + 0.2, z_high + 0.2])
    waypoints.append([gates[3][0], gates[3][1] + 0.1, z_high])
    waypoints.append([gates[3][0], gates[3][1] - 0.1, z_high + 0.1])
    waypoints = np.array(waypoints)

    tck, u = interpolate.splprep([waypoints[:, 0], waypoints[:, 1], waypoints[:, 2]], s=0.1)
    duration = 12
    t = np.linspace(0, 1, int(duration * CTRL_FREQ))
    ref_x, ref_y, ref_z = interpolate.splev(t, tck)
    assert max(ref_z) < 2.5, "Drone must stay below the ceiling"
    x_goal = np.zeros((ref_x.shape[0], 12))
    x_goal[:,0] = ref_x
    x_goal[:,1] = ref_y
    x_goal[:,2] = ref_z
    return x_goal

def create_waypoints(initial_obs: np.ndarray, initial_info: dict):
    CTRL_TIMESTEP = initial_info["ctrl_timestep"]
    CTRL_FREQ = initial_info["ctrl_freq"]

    # Store a priori scenario information.
    NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
    NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]
    waypoints = []
    ## Be careful with initial obs, what I get is directly from the firmware wrapper 
    ## which has the shape, x, x_dot, y, y_dot, z, z_dot,...
    waypoints.append([initial_obs[0], initial_obs[2], 0.3])
    gates = NOMINAL_GATES
    z_low = initial_info["gate_dimensions"]["low"]["height"]
    z_high = initial_info["gate_dimensions"]["tall"]["height"]
    waypoints.append([1, 0, z_low])
    waypoints.append([gates[0][0] + 0.2, gates[0][1] + 0.1, z_low])
    waypoints.append([gates[0][0] + 0.1, gates[0][1], z_low])
    waypoints.append([gates[0][0] - 0.1, gates[0][1], z_low])
    waypoints.append(
        [
            (gates[0][0] + gates[1][0]) / 2 - 0.7,
            (gates[0][1] + gates[1][1]) / 2 - 0.3,
            (z_low + z_high) / 2,
        ]
    )
    waypoints.append(
        [
            (gates[0][0] + gates[1][0]) / 2 - 0.5,
            (gates[0][1] + gates[1][1]) / 2 - 0.6,
            (z_low + z_high) / 2,
        ]
    )
    waypoints.append([gates[1][0] - 0.3, gates[1][1] - 0.2, z_high])
    waypoints.append([gates[1][0] + 0.2, gates[1][1] + 0.2, z_high])
    waypoints.append([gates[2][0], gates[2][1] - 0.4, z_low])
    waypoints.append([gates[2][0], gates[2][1] + 0.2, z_low])
    waypoints.append([gates[2][0], gates[2][1] + 0.2, z_high + 0.2])
    waypoints.append([gates[3][0], gates[3][1] + 0.1, z_high])
    waypoints.append([gates[3][0], gates[3][1] - 0.1, z_high + 0.1])
    waypoints.append(
        [
            initial_info["x_reference"][0],
            initial_info["x_reference"][2],
            initial_info["x_reference"][4],
        ]
    )
    waypoints.append(
        [
            initial_info["x_reference"][0],
            initial_info["x_reference"][2] - 0.2,
            initial_info["x_reference"][4],
        ]
    )
    waypoints = np.array(waypoints)

    tck, u = interpolate.splprep([waypoints[:, 0], waypoints[:, 1], waypoints[:, 2]], s=0.1)
    duration = 12
    t = np.linspace(0, 1, int(duration * CTRL_FREQ))
    ref_x, ref_y, ref_z = interpolate.splev(t, tck)
    assert max(ref_z) < 2.5, "Drone must stay below the ceiling"
    x_goal = np.zeros((ref_x.shape[0], 12))
    x_goal[:,0] = ref_x
    x_goal[:,1] = ref_y
    x_goal[:,2] = ref_z
    return x_goal, waypoints