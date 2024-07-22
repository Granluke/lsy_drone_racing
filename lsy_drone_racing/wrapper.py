"""Wrapper to make the environment compatible with the gymnasium API.

The drone simulator does not conform to the gymnasium API, which is used by most RL frameworks. This
wrapper can be used as a translation layer between these modules and the simulation.

RL environments are expected to have a uniform action interface. However, the Crazyflie commands are
highly heterogeneous. Users have to make a discrete action choice, each of which comes with varying
additional arguments. Such an interface is impractical for most standard RL algorithms. Therefore,
we restrict the action space to only include FullStateCommands.

We also include the gate pose and range in the observation space. This information is usually
available in the info dict, but since it is vital information for the agent, we include it directly
in the observation space.

Warning:
    The RL wrapper uses a reduced action space and a transformed observation space!
"""

from __future__ import annotations

import os
from contextlib import redirect_stdout
from copy import deepcopy
import logging
from typing import Any

import numpy as np
from gymnasium import Env, Wrapper, ActionWrapper
from gymnasium.error import InvalidAction
from gymnasium.spaces import Box
from safe_control_gym.controllers.firmware.firmware_wrapper import FirmwareWrapper

from lsy_drone_racing.rotations import map2pi
from lsy_drone_racing.create_waypoints import find_closest_traj_point, find_closest_gate
from lsy_drone_racing.path_planning import calc_best_path
from lsy_drone_racing.global_parameters import *

logger = logging.getLogger(__name__)

enum = ['waypoints', 'gates', 'None']


class DroneRacingWrapper(Wrapper):
    """Drone racing firmware wrapper to make the environment compatible with the gymnasium API.

    In contrast to the underlying environment, this wrapper only accepts FullState commands as
    actions.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: FirmwareWrapper, terminate_on_lap: bool = True, train_random_state: bool = False):
        """Initialize the wrapper.

        Args:
            env: The firmware wrapper.
            terminate_on_lap: Stop the simulation early when the drone has passed the last gate.
        """
        if not isinstance(env, FirmwareWrapper):
            raise TypeError(f"`env` must be an instance of `FirmwareWrapper`, is {type(env)}")
        super().__init__(env)
        # Patch the FirmwareWrapper to add any missing attributes required by the gymnasium API.
        self.env = env
        # Unwrapped attribute is required for the gymnasium API. Some packages like stable-baselines
        # use it to check if the environment is unique. Therefore, we cannot use None, as None is
        # None returns True and falsely indicates that the environment is not unique. Lists have
        # unique id()s, so we use lists as a dummy instead.
        self.env.unwrapped = []
        self.env.render_mode = None

        # Gymnasium env required attributes
        # Action space:
        # [x, y, z, yaw]
        # x, y, z)  The desired position of the drone in the world frame.
        # yaw)      The desired yaw angle.
        # All values are scaled to [-1, 1]. Transformed back, x, y, z values of 1 correspond to 5m.
        # The yaw value of 1 corresponds to pi radians.
        #region Initialize spaces
        self.action_scale = np.array([1, 1, 1, np.pi])
        self.learned_action_scale = 0.2 # Contribution of the learned action
        self.fixed_action_scale = 1 - self.learned_action_scale
        print(f'Fixed Action Scale: {self.fixed_action_scale}')
        print(f'Learned Action Scale: {self.learned_action_scale}')
        act_low = np.array([-1, -1, -1])
        self.action_space = Box(act_low*self.learned_action_scale, -act_low*self.learned_action_scale, dtype=np.float32)
        self.action_space_total = Box(-1, 1, shape=(4,), dtype=np.float32)
        self.if_chunk = 'gates'
        assert self.if_chunk in enum, "The chunk must be either waypoints or None"
        # If chunk is for aiming the trajectory points or the waypoints

        # Observation space:
        # [drone_xyz, drone_rpy, drone_vxyz, drone vrpy, gates_xyz_yaw, gates_in_range,
        # obstacles_xyz, obstacles_in_range, gate_id]
        # drone_xyz)  Drone position in meters.
        # drone_rpy)  Drone orientation in radians.
        # drone_vxyz)  Drone velocity in m/s.
        # drone_vrpy)  Drone angular velocity in rad/s.
        # gates_xyz_yaw)  The pose of the gates. Positions are in meters and yaw in radians. The
        #       length is dependent on the number of gates. Ordering is [x0, y0, z0, yaw0, x1,...].
        # gates_in_range)  A boolean array indicating if the drone is within the gates' range. The
        #       length is dependent on the number of gates.
        # obstacles_xyz)  The pose of the obstacles. Positions are in meters. The length is
        #       dependent on the number of obstacles. Ordering is [x0, y0, z0, x1,...].
        # obstacles_in_range)  A boolean array indicating if the drone is within the obstacles'
        #       range. The length is dependent on the number of obstacles.
        # gate_id)  The ID of the current target gate. -1 if the task is completed.
        n_gates = env.env.NUM_GATES
        n_obstacles = env.env.n_obstacles
        # Velocity limits are set to 10 m/s for the drone and 10 rad/s for the angular velocity.
        # While drones could go faster in theory, it's not safe in practice and we don't allow it in
        # sim either.
        drone_limits = [5, 5, 5, np.pi, np.pi, np.pi, 10, 10, 10, 10, 10, 10]
        obs_limits_high = np.array(drone_limits)
        obs_limits_low = -np.array(drone_limits)
        ## Add Goal States to Observation Space
        self.obs_goal_horizon = self.env.env.obs_goal_horizon
        if self.obs_goal_horizon > 0:
            traj_limits = [5, 5, 5]
            if self.if_chunk == 'waypoints' or self.if_chunk == 'gates':
                temp_h = np.array(traj_limits*2)
                temp_l = -np.array(traj_limits*2)
            else:
                temp_h = np.array(traj_limits*self.obs_goal_horizon)
                temp_l = -np.array(traj_limits*self.obs_goal_horizon)
            # One for the next state and the other is for goal of that chunk.
            obs_limits_high = np.concatenate((obs_limits_high, temp_h), axis=0)
            obs_limits_low = np.concatenate((obs_limits_low, temp_l), axis=0)
        self.observation_space = Box(obs_limits_low, obs_limits_high, dtype=np.float32)
        #endregion
        self.pyb_client_id: int = env.env.PYB_CLIENT
        # Config and helper flags
        self.terminate_on_lap = terminate_on_lap
        self._reset_required = False
        # The original firmware wrapper requires a sim time as input to the step function. This
        # breaks the gymnasium interface. Instead, we keep track of the sim time here. On each step,
        # it is incremented by the control time step. On env reset, it is reset to 0.
        self._sim_time = 0.0
        self._drone_pose = None
        # The firmware quadrotor env requires the rotor forces as input to the step function. These
        # are zero initially and updated by the step function. We automatically insert them to
        # ensure compatibility with the gymnasium interface.
        # TODO: It is not clear if the rotor forces are even used in the firmware env. Initial tests
        #       suggest otherwise.
        self._f_rotors = np.zeros(4)
        self.gates = None  # Gate Positions
        self.obstacles = None  # Obstacle Positions
        self.start_point = None  # Start Point
        self.X_GOAL = None  # Goal States
        self.X_GOAL_crop = None  # Cropped Goal States, for random training
        self.X_GOAL_distances = None  # Distance between each step//Only for a specific reward
        self.train_random_state = train_random_state # Randomly choose the starting waypoint
        self._wrap_ctr_step = None  # Counter for the current control step
        self.waypoints = None  # Waypoints
        self.wp_traj_idx = None  # Waypoint to Trajectory Index (waypoint corresponds to what trajectory point)
        self.wp_gate_dict = None  # Waypoint to Gate Dictionary (waypoint corresponds to what gate)
        self._current_gate_idx = None  # Current Gate Index
        self.chunk_goal = None  # Chunk Goal
        self.idx_chunk_goal = None  # Index of the Chunk Goal
        self.chunk_distance = None  # Distance to the Chunk Goal
        self.max_iter_chunk_switch = None  # Maximum Iteration for the Chunk Goal
        self.ctr_chunk_switch = None  # Counter for the Chunk Goal Switch
    
    @property
    def time(self) -> float:
        """Return the current simulation time in seconds."""
        return self._sim_time
    
    #region RESET
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment.

        Args:
            seed: The random seed to use for the environment. Not used in this wrapper.
            options: Additional options to pass to the environment. Not used in this wrapper.

        Returns:
            The initial observation and info dict of the next episode.
        """
        print('Resetting Environment!')
        self._reset_required = False
        self._sim_time = 0.0
        self._f_rotors[:] = 0.0
        obs, info = self.env.reset()
        
        #region Path Planning RESET
        ## Extract data for the path planning
        self.gates = deepcopy(info["nominal_gates_pos_and_type"])
        self._current_gate_idx = deepcopy(info["current_gate_id"])
        self.obstacles = deepcopy(info["nominal_obstacles_pos"])
        self.start_point = [obs[0], obs[2], obs[4]]
        freq = self.env.ctrl_freq
        ## Define times steps for the trajectory
        self.t = np.linspace(0, 1, int(DURATION * freq))
        ## Path Planning
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull):
                self.X_GOAL, self.waypoints = calc_best_path(gates=self.gates, obstacles=self.obstacles,
                                                             start_point=self.start_point, t=self.t, plot=False)
        # Spawn the drone on randomly chosen waypoint
        if self.train_random_state:
            ## Find the trajectory point closest to the waypoints
            self.wp_traj_idx = find_closest_traj_point(self.waypoints, self.X_GOAL)
            ## Find the gate corresponding to the waypoints
            self.wp_gate_dict = find_closest_gate(self.X_GOAL, info, self.wp_traj_idx)
            start_ind = 7
            self.START_IND = start_ind
            start_pose = self.waypoints[start_ind,:3]
            start_indwp = self.wp_traj_idx[start_ind]
            self.waypoints = self.waypoints[start_ind:,:] # Crop the waypoints
            self.wp_traj_idx = self.wp_traj_idx[start_ind:] # Crop the waypoint indices
            self.X_GOAL_crop = self.X_GOAL[start_indwp:,:] # Crop the goal states
            self.X_GOAL_distances = self.X_GOAL_distances[start_indwp:] # Crop the distances
            buffer = (self.env.env.INIT_X, self.env.env.INIT_Y, self.env.env.INIT_Z)
            self.env.env.INIT_X = start_pose[0] # Manual adjustment
            self.env.env.INIT_Y = start_pose[1]
            self.env.env.INIT_Z = start_pose[2]
            ## Reset again to change the env to the initial state
            obs, info = self.env.reset()
            self.start_point = [obs[0], obs[2], obs[4]]
            self.env.env.INIT_X, self.env.env.INIT_Y, self.env.env.INIT_Z = buffer
            # env.reset() cannot reset in the exact position, which leads to a mismatch.
            self.env.env.current_gate = self.wp_gate_dict[start_ind]
            self._current_gate_idx = self.wp_gate_dict[start_ind]
        else:
            assert np.linalg.norm(self.start_point[2] - TAKEOFF_HEIGHT) < 0.1, "Takeoff height must be the start height for training!"
            self.START_IND = 0
            self.X_GOAL_crop = deepcopy(self.X_GOAL[self.START_IND:])
            self._current_gate_idx = info["current_gate_id"]
            print(f'Current Gate {self._current_gate_idx} with position {self.gates[self._current_gate_idx]}')
        #endregion
        
        # Keep the horizon included in the obs
        assert self.obs_goal_horizon > 0, "The horizon must be greater than 0"
        self._wrap_ctr_step = 0
        
        #region Observation Transform RESET
        obs = self.observation_transform(obs, info, self.X_GOAL_crop, self._wrap_ctr_step, self.obs_goal_horizon).astype(np.float32)
        self._drone_pose = obs[[0, 1, 2, 5]]
        # Determine what to follow, either waypoints or gates, or None
        if self.if_chunk == 'waypoints':
            self.idx_chunk_goal = 1
            self.chunk_goal = self.waypoints[self.idx_chunk_goal,:]
            self.chunk_distance = np.linalg.norm(obs[:3] - self.chunk_goal[:3], axis=0)
            obs = np.concatenate([obs[:15], self.chunk_goal])
            self.max_iter_chunk_switch = 100
            self.ctr_chunk_switch = 0
        elif self.if_chunk == 'gates':
            self.chunk_goal = self.gates[self._current_gate_idx, :3]
            self.chunk_goal[2] = 1 if self.gates[self._current_gate_idx,-1] == 0 else 0.525
            self.chunk_distance = np.linalg.norm(obs[:3] - self.chunk_goal[:3], axis=0)
            obs = np.concatenate([obs[:15], self.chunk_goal], dtype=np.float32)
            self.max_iter_chunk_switch = 100
            self.ctr_chunk_switch = 0
        else:
            # For directly following the trajectory
            pass
        #endregion
        
        return obs, info
    #endregion
    
    #region STEP
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment.

        Args:
            action: The action to take in the environment. See action space for details.

        Returns:
            The next observation, the reward, the terminated and truncated flags, and the info dict.
        """
        assert not self._reset_required, "Environment must be reset before taking a step"
        if action not in self.action_space_total:
            # Wrapper has a reduced action space compared to the firmware env to make it compatible
            # with the gymnasium interface and popular RL libraries.
            raise InvalidAction(f"Invalid action: {action}")
        action = self._action_transform(action)
        assert action.shape[-1] == 4, "Action must have shape (..., 4)"
        # The firmware does not use the action input in the step function
        zeros = np.zeros(3)
        self.env.sendFullStateCmd(action[:3], zeros, zeros, action[3], zeros, self._sim_time)
        # The firmware quadrotor env requires the sim time as input to the step function. It also
        # returns the desired rotor forces. Both modifications are not part of the gymnasium
        # interface. We automatically insert the sim time and reuse the last rotor forces.
        obs, reward, done, info, f_rotors = self.env.step(self._sim_time, action=self._f_rotors)
        self._f_rotors[:] = f_rotors
        # We set truncated to True if the task is completed but the drone has not yet passed the
        # final gate. We set terminated to True if the task is completed and the drone has passed
        # the final gate.
        terminated, truncated = False, False
        if info["task_completed"] and info["current_gate_id"] != -1:
            truncated = True
        elif self.terminate_on_lap and info["current_gate_id"] == -1:
            print('Task Completed!')
            info["task_completed"] = True
            terminated = True
        elif self.terminate_on_lap and done:  # Done, but last gate not passed -> terminate
            terminated = True
        # Increment the sim time after the step if we are not yet done.
        if not terminated and not truncated:
            self._sim_time += self.env.ctrl_dt
        
        #region Observation Transform STEP
        obs = self.observation_transform(obs, info, self.X_GOAL_crop, self._wrap_ctr_step, self.obs_goal_horizon).astype(np.float32)
        self._wrap_ctr_step += 1
        ## Update Chunk Goal and Change Observation
        if self.if_chunk == 'waypoints':
            obs = np.concatenate([obs[:15], self.chunk_goal])
            ## Integrate Reward Wrapper inside
            reward = self._compute_reward(obs, reward, terminated, truncated, info)
            ## Increase the counter for the chunk switch
            self.ctr_chunk_switch += 1
            ## If the drone is sufficiently close to the waypoint, switch to the next one
            if self.chunk_distance < 0.2:
                reward += 1 # REWARD for the switch
                self.idx_chunk_goal += 1 if self.idx_chunk_goal < self.waypoints.shape[0]-1 else 0
                self.chunk_goal = self.waypoints[self.idx_chunk_goal,:]
                ## Update the distance for the reward calculation
                self.chunk_distance = np.linalg.norm(obs[:3] - self.chunk_goal[:3], axis=0)
                ## Reset the counter for the chunk switch
                self.ctr_chunk_switch = 0
                ## Append the chunk goal to the observation
                obs = np.concatenate([obs[:15], self.chunk_goal])
                print(f'Chunk Switched to {self.idx_chunk_goal}')
           
            ## Recalculating the trajectory if the gate position is changed
            if self._current_gate_idx != -1:
                recalc = False if np.linalg.norm(np.array(info['gates_pose'][self._current_gate_idx,:2])-np.array(self.gates[self._current_gate_idx, :2])) < 0.01 else True
                if recalc:
                    print(f"Gate {self._current_gate_idx} at {self.gates[self._current_gate_idx]} is not the same as {info['gates_pose'][self._current_gate_idx]}")    
                    ## Update the gates
                    self.gates[self._current_gate_idx, :-1] = info['gates_pose'][self._current_gate_idx]
                    with open(os.devnull, 'w') as fnull:
                        with redirect_stdout(fnull):
                            self.X_GOAL, self.waypoints = calc_best_path(self.gates, self.obstacles, 
                                                                         self.start_point, t=self.t, plot=False)
                    self.X_GOAL_crop = deepcopy(self.X_GOAL[self.START_IND:])
                    obs = self.observation_transform(obs, info, self.X_GOAL_crop, self._wrap_ctr_step, 
                                                     self.obs_goal_horizon).astype(np.float32)
                    ## Waypoints change with the gates
                    self.chunk_goal = self.waypoints[self.idx_chunk_goal+1,:]
                    ## Calculate the distance for the reward calculation
                    self.chunk_distance = np.linalg.norm(obs[:3] - self.chunk_goal[:3], axis=0)
                    ## Append the chunk goal to the observation
                    obs = np.concatenate([obs[:15], self.chunk_goal])
                    assert max(self.X_GOAL[:,2]) < 2.5, "Drone must stay below the ceiling"
        elif self.if_chunk == 'gates':
            obs = np.concatenate([obs[:15], self.chunk_goal], dtype=np.float32)
            ## Integrate Reward Wrapper inside
            reward = self._compute_reward(obs, reward, terminated, truncated, info)
            ## Recalculating the trajectory if the gate position is changed
            recalc = False if np.linalg.norm(np.array(info['gates_pose'][self._current_gate_idx,:2])-np.array(self.gates[self._current_gate_idx, :2])) < 0.01 else True
            if recalc:
                print(f"Gate {self._current_gate_idx} at {self.gates[self._current_gate_idx]} is not the same as {info['gates_pose'][self._current_gate_idx]}")    
                ## Update the gates
                self.gates[self._current_gate_idx, :-1] = info['gates_pose'][self._current_gate_idx]
                with open(os.devnull, 'w') as fnull:
                    with redirect_stdout(fnull):
                        self.X_GOAL, self.waypoints = calc_best_path(self.gates, self.obstacles, self.start_point, 
                                                                     t=self.t, plot=False)
                self.X_GOAL_crop = deepcopy(self.X_GOAL[self.START_IND:])
                obs = self.observation_transform(obs, info, self.X_GOAL_crop, self._wrap_ctr_step, 
                                                 self.obs_goal_horizon).astype(np.float32)
                ## Update the chunk goal since the gates are the goals
                self.chunk_goal = self.gates[self._current_gate_idx, :3]
                self.chunk_goal[2] = 1 if self.gates[self._current_gate_idx,-1] == 0 else 0.525
                ## Update the distance for the reward calculation
                self.chunk_distance = np.linalg.norm(obs[:3] - self.chunk_goal[:3], axis=0)
                ## Append the chunk goal to the observation
                obs = np.concatenate([obs[:15], self.chunk_goal], dtype=np.float32)
                assert max(self.X_GOAL[:,2]) < 2.5, "Drone must stay below the ceiling"
        else:
            ## Integrate Reward Wrapper inside
            reward = self._compute_reward(obs, reward, terminated, truncated, info)
        #endregion

        ## Recalculate for the obstacles
        if np.linalg.norm(self.obstacles[:,:2] - info["obstacles_pose"][:,:2]) > 0.01:
            self.obstacles = deepcopy(info["obstacles_pose"])
            with open(os.devnull, 'w') as fnull:
                    with redirect_stdout(fnull):
                        self.X_GOAL, self.waypoints = calc_best_path(self.gates, self.obstacles, self.start_point, 
                                                                     t=self.t, plot=False)
            obs = DroneRacingWrapper.observation_transform(obs, info, self.X_GOAL, self._wrap_ctr_step, 
                                                           self.obs_goal_horizon).astype(np.float32)
            ## Since the obstacles are changed, the trajectory also changed but chunk goal remains the same.
            obs = np.concatenate([obs[:15], self.chunk_goal], dtype=np.float32)

        self._drone_pose = obs[[0, 1, 2, 5]]
        ## Check if the drone is in the limits
        if obs not in self.observation_space:
            pos_space = Box(self.observation_space.low[:3], self.observation_space.high[:3], dtype=np.float32)
            if obs[:3] not in pos_space:
                print(f'Observation {obs[:3]} is not in the position space!')
                terminated = True
        ## Check counter for the chunk switch to enforce progress to chunk goals
        if self.ctr_chunk_switch > self.max_iter_chunk_switch:
            print('Chunk Switch Counter Exceeded!')
            terminated = True
            reward -= 10   
        ## Check if the gate passed
        if info["current_gate_id"] != self._current_gate_idx:
            ## Update the gate index
            self._current_gate_idx = info["current_gate_id"]
            print(f'Gate Passed! Current Gate Index: {self._current_gate_idx}')
            ## If the drone is following the gates, update the chunk goal
            if self.if_chunk == 'gates':
                ## Update the chunk goal
                self.chunk_goal = self.gates[self._current_gate_idx, :3]
                self.chunk_goal[2] = 1 if self.gates[self._current_gate_idx,-1] == 0 else 0.525
                ## Calculate the distance for the reward calculation
                self.chunk_distance = np.linalg.norm(obs[:3] - self.chunk_goal[:3], axis=0)
                ## Update the observartion since the chunk goal is changed
                obs = np.concatenate([obs[:15], self.chunk_goal], dtype=np.float32)
                ## Give reward for the gate pass
                reward += 2.5
        self._reset_required = terminated or truncated
        return obs, reward, terminated, truncated, info
    #endregion

    def _action_transform(self, action: np.ndarray) -> np.ndarray:
        """Transform the action to the format expected by the firmware env.

        Args:
            action: The action to transform.

        Returns:
            The transformed action.
        """
        action = self._drone_pose + (action * self.action_scale)
        action[3] = map2pi(action[3])  # Ensure yaw is in [-pi, pi]
        return action

    def render(self):
        """Render the environment.

        Used for compatibility with the gymnasium API. Checks if PyBullet was launched with an
        active GUI.

        Raises:
            AssertionError: If PyBullet was not launched with an active GUI.
        """
        assert self.pyb_client_id != -1, "PyBullet not initialized with active GUI"

    @staticmethod
    def observation_transform(obs: np.ndarray, info: dict[str, Any], X_GOAL: np.ndarray=None, crnt_step: int=-1, obs_goal_horizon: int=0) -> np.ndarray:
        """Transform the observation to include the trajecotry points in the horizon.

        Args:
            obs: The observation to transform.
            info: Additional information to include in the observation.
            X_GOAL: The goal states from the trajectory.
            crnt_step: The current step in the trajectory.
            obs_goal_horizon: The horizon for the goal states.

        Returns:
            The transformed observation.
        """
        drone_pos = obs[0:6:2]
        drone_vel = obs[1:6:2]
        drone_rpy = obs[6:9]
        drone_ang_vel = obs[9:12]
        obs = np.concatenate(
            [
                drone_pos,
                drone_rpy,
                drone_vel,
                drone_ang_vel,
            ]
        )
        ## Append the goal states to the observation
        wp_idx = [min(crnt_step + i, X_GOAL.shape[0]-1) 
                for i in range(obs_goal_horizon)]
        goal_state = X_GOAL[wp_idx].flatten()
        obs = np.concatenate([obs, goal_state], dtype=np.float32)
        return obs

    #region REWARD
    def _compute_reward(
        self, obs: np.ndarray, reward: float, terminated: bool, truncated: bool, info: dict
    ) -> float:
        """Compute the reward for the current step.

        Args:
            obs: The current observation.
            reward: The reward from the environment.
            terminated: True if the episode is terminated.
            truncated: True if the episode is truncated.
            info: Additional information from the environment.

        Returns:
            The computed reward.
        """
        reward = 0
        drone_pos = obs[:3]
        current_chunk_distance = np.linalg.norm(drone_pos - self.chunk_goal[:3], axis=0)
        reward += self.chunk_distance - current_chunk_distance
        self.chunk_distance = current_chunk_distance
        ## Crash Penality
        crash_penality = -10 if self.env.env.currently_collided else 0
        ## Constraint Violation Penality
        cstr_penalty = -10 if self.env.env.cnstr_violation else 0
        ## Gate Passing Reward
        # It is outside of the reward function
        ## Waypoint Passing Reward
        # It is outside of the reward function
        task_rew = 10 if info["task_completed"] else 0
        return reward+crash_penality+cstr_penalty+task_rew
    #endregion
        
class DroneRacingObservationWrapper:
    """Wrapper to transform the observation space the firmware wrapper.

    This wrapper matches the observation space of the DroneRacingWrapper. See its definition for
    more details. While we want to transform the observation space, we do not want to change the API
    of the firmware wrapper. Therefore, we create a separate wrapper for the observation space.

    Note:
        This wrapper is not a subclass of the gymnasium ObservationWrapper because the firmware is
        not compatible with the gymnasium API.
    """

    def __init__(self, env: FirmwareWrapper):
        """Initialize the wrapper.

        Args:
            env: The firmware wrapper.
        """
        if not isinstance(env, FirmwareWrapper):
            raise TypeError(f"`env` must be an instance of `FirmwareWrapper`, is {type(env)}")
        self.env = env
        self.pyb_client_id: int = env.env.PYB_CLIENT
        self.X_GOAL = None
        self.waypoints = None
        self.obs_goal_horizon = self.env.env.obs_goal_horizon
        self._wrap_ctr_step = None
        self.if_chunk = 'gates' #  waypoints or None or gates
        assert self.if_chunk in enum, "The chunk must be either waypoints or traj or None"

    def __getattribute__(self, name: str) -> Any:
        """Get an attribute from the object.

        If the attribute is not found in the wrapper, it is fetched from the firmware wrapper.

        Args:
            name: The name of the attribute.

        Returns:
            The attribute value.
        """
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.env, name)
    #region RESET
    def reset(self, *args: Any, **kwargs: dict[str, Any]) -> tuple[np.ndarray, dict]:
        """Reset the environment.

        Args:
            args: Positional arguments to pass to the firmware wrapper.
            kwargs: Keyword arguments to pass to the firmware wrapper.

        Returns:
            The transformed observation and the info dict.
        """
        obs, info = self.env.reset(*args, **kwargs)
        #region Path Planning
        ## Extract data for the path planning
        self.gates = deepcopy(info["nominal_gates_pos_and_type"])
        self._current_gate_idx = deepcopy(info["current_gate_id"])
        self.obstacles = deepcopy(info["nominal_obstacles_pos"])
        self.start_point = [obs[0], obs[2], TAKEOFF_HEIGHT]
        freq = self.env.ctrl_freq
        self.t = np.linspace(0, 1, int(DURATION * freq))
        with open(os.devnull, 'w') as fnull:
                    with redirect_stdout(fnull):
                        self.X_GOAL, self.waypoints = calc_best_path(gates=self.gates, obstacles=self.obstacles, start_point=self.start_point, 
                                                                     t=self.t, plot=0)
        #endregion
        #region Observation Transform RESET
        assert self.obs_goal_horizon > 0, "The horizon must be greater than 0"
        self._wrap_ctr_step = 0
        ## Append the goal states to the observation accoring to the horizon
        obs = DroneRacingWrapper.observation_transform(obs, info, self.X_GOAL, self._wrap_ctr_step, 
                                                       self.obs_goal_horizon).astype(np.float32)
        ## If the drone follows the waypoints
        if self.if_chunk == 'waypoints':
            ## Initialize the chunk goal to the second waypoint
            self.idx_chunk_goal = 1
            self.chunk_goal = self.waypoints[self.idx_chunk_goal,:]
            ## Calculate the distance for the chunk switch
            self.chunk_distance = np.linalg.norm(obs[:3] - self.chunk_goal[:3], axis=0)
            ## Append the chunk goal to the observation
            obs = np.concatenate([obs[:15], self.chunk_goal])
        ## If the drone follows the gates
        elif self.if_chunk == 'gates':
            ## Assign the current gate as the chunk goal
            self.chunk_goal = self.gates[self._current_gate_idx, :3]
            self.chunk_goal[2] = 1 if self.gates[self._current_gate_idx,-1] == 0 else 0.525
            ## Append the chunk goal to the observation
            obs = np.concatenate([obs[:15], self.chunk_goal])
        #endregion
        return obs, info
    #endregion
    #region STEP
    def step(
        self, *args: Any, **kwargs: dict[str, Any]
    ) -> tuple[np.ndarray, float, bool, dict, np.ndarray]:
        """Take a step in the current environment.

        Args:
            args: Positional arguments to pass to the firmware wrapper.
            kwargs: Keyword arguments to pass to the firmware wrapper.

        Returns:
            The transformed observation and the info dict.
        """
        obs, reward, done, info, action = self.env.step(*args, **kwargs)
        curr_time = args[0] # Current Time
        
        #region Observation Transform
        obs = DroneRacingWrapper.observation_transform(obs, info, self.X_GOAL, self._wrap_ctr_step, 
                                                       self.obs_goal_horizon).astype(np.float32)
        ## To make it compatibel with the controller, wait until the drone takes off
        self._wrap_ctr_step += 1 if curr_time >= TAKEOFF_DURATION else 0
        if self.if_chunk == 'waypoints':
            ## we first calculate the chunk distance
            self.chunk_distance = np.linalg.norm(obs[:3] - self.chunk_goal[:3], axis=0)
            ## If we are sufficiently close to the waypoint, switch to the next one
            if self.chunk_distance < 0.2:
                ## Assign the next waypoint as the chunk goal
                self.idx_chunk_goal += 1 if self.idx_chunk_goal < self.waypoints.shape[0]-1 else 0
                self.chunk_goal = self.waypoints[self.idx_chunk_goal,:]
                ## Calculate the distance for the switch
                self.chunk_distance = np.linalg.norm(obs[:3] - self.chunk_goal[:3], axis=0)
                print(f'Chunk Idx: {self.idx_chunk_goal}')
            if self._current_gate_idx != -1:
                ## Do a recalculation if the gate position is changed
                recalc = False if np.linalg.norm(np.array(info['gates_pose'][self._current_gate_idx,:2])-np.array(self.gates[self._current_gate_idx, :2])) < 0.01 else True
                if recalc:
                    print(f"Gate {self._current_gate_idx} at {self.gates[self._current_gate_idx]} is not the same as {info['gates_pose'][self._current_gate_idx]}")    
                    ## Update the gates
                    self.gates[self._current_gate_idx, :-1] = info['gates_pose'][self._current_gate_idx]
                    with open(os.devnull, 'w') as fnull:
                        with redirect_stdout(fnull):
                            self.X_GOAL, self.waypoints = calc_best_path(self.gates, self.obstacles, self.start_point, 
                                                                         t=self.t, plot=False)
                    obs = DroneRacingWrapper.observation_transform(obs, info, self.X_GOAL, self._wrap_ctr_step, 
                                                                   self.obs_goal_horizon).astype(np.float32)
                    ## Update the chunk goal
                    self.chunk_goal = self.waypoints[self.idx_chunk_goal,:]
                    assert max(self.X_GOAL[:,2]) < 2.5, "Drone must stay below the ceiling"
        elif self.if_chunk == 'gates':
            ## Do a recalculation if the gate position is changed
            recalc = False if np.linalg.norm(np.array(info['gates_pose'][self._current_gate_idx,:2])-np.array(self.gates[self._current_gate_idx, :2])) < 0.01 else True
            if recalc:
                print(f"Gate {self._current_gate_idx} at {self.gates[self._current_gate_idx]} is not the same as {info['gates_pose'][self._current_gate_idx]}")    
                ## Update the gates
                self.gates[self._current_gate_idx, :-1] = info['gates_pose'][self._current_gate_idx]
                with open(os.devnull, 'w') as fnull:
                    with redirect_stdout(fnull):
                        self.X_GOAL, self.waypoints = calc_best_path(self.gates, self.obstacles, self.start_point, 
                                                                     t=self.t, plot=False)
                obs = DroneRacingWrapper.observation_transform(obs, info, self.X_GOAL, self._wrap_ctr_step, 
                                                               self.obs_goal_horizon).astype(np.float32)
                ## Update the chunk goal with the new gate
                self.chunk_goal = self.gates[self._current_gate_idx, :3]
                self.chunk_goal[2] = 1 if self.gates[self._current_gate_idx,-1] == 0 else 0.525
                assert max(self.X_GOAL[:,2]) < 2.5, "Drone must stay below the ceiling"
        
        ## Obstacle Check
        if np.linalg.norm(self.obstacles[:,:2] - info["obstacles_pose"][:,:2]) > 0.01:
            ## Update the obstacles
            self.obstacles = deepcopy(info["obstacles_pose"])
            with open(os.devnull, 'w') as fnull:
                    with redirect_stdout(fnull):
                        self.X_GOAL, self.waypoints = calc_best_path(self.gates, self.obstacles, self.start_point, 
                                                                     t=self.t, plot=False)
            obs = DroneRacingWrapper.observation_transform(obs, info, self.X_GOAL, self._wrap_ctr_step, 
                                                           self.obs_goal_horizon).astype(np.float32)
        #endregion
        ## Update the current gate if the gate is passed
        if info["current_gate_id"] != self._current_gate_idx:
            ## Update the gate index
            self._current_gate_idx = info["current_gate_id"]
            print(f'Gate Passed! Current Gate Index: {self._current_gate_idx}')
            ## If the drone is following the gates, update the chunk goal
            if self.if_chunk == 'gates':
                self.chunk_goal = self.gates[self._current_gate_idx, :3]
                self.chunk_goal[2] = 1 if self.gates[self._current_gate_idx,-1] == 0 else 0.525
        ## Append the chunk goal to the observation
        obs = np.concatenate([obs[:15], self.chunk_goal])
        return obs, reward, done, info, action
#region Action Wrapper
class ActionWrapper(ActionWrapper):
    """Wrapper to modify the action based on the current observation for the Shared Control Paradigm.
    This Wrapper is used only designed for the training of the drone."""
    
    def __init__(self, env):
        """Initialize the wrapper. Define the fixed and learned action scales.
        Args:
            env: The DroneRacingWrapper environment to wrap."""
        super().__init__(env)
        self.fas = env.fixed_action_scale
        self.las = env.learned_action_scale
        self.current_obs = None

    def reset(self, **kwargs):
        """Reset the environment and store the current observation.
        Returns:
            obs: The initial observation of the environment.
            info: Additional information from the environment."""

        self.current_obs, info = self.env.reset(**kwargs)
        return self.current_obs, info

    def step(self, action):
        """Modify the action based on the current observation and take a step in the environment.
        Args:
            action: The action to take in the environment. See action space for details.
        Returns:
            obs: The next observation.
            reward: The reward from the environment.
            terminated: True if the episode is terminated.
            truncated: True if the episode is truncated.
            info: Additional information from the environment."""
        ## Modify the action based on the current observation
        modified_action = self.modify_action(action, self.current_obs)
        ## Take a step in the environment
        obs, reward, terminated, truncated, info = self.env.step(modified_action)
        self.current_obs = obs
        return obs, reward, terminated, truncated, info

    def modify_action(self, action, obs):
        """Modify the action based on the current observation.
        Args:
            action: The action to modify.
            obs: The current observation.
        Returns:
            action: The modified action."""
        # Modify the action based on the current observation
        current_pose = obs[:3]
        current_yaw = obs[5]
        ## Extract the goal pose from the observation
        goal_pose = obs[12:15]
        ## Fixed action is determined by the difference between the goal and current pose
        action_fix = goal_pose - current_pose
        yaw = 0.0
        ## add fixed action to the action, which comes from the policy
        ## Since the action is coming from the action space is normalized with
        ## the learned action scale, we don't need to scale it again.
        action[:3] = action[:3] + self.fas * action_fix
        # action = action_fix
        action = np.concatenate((action,[yaw]))
        check_mask1 = action > 1
        check_mask2 = action < -1
        if check_mask1.any():
            action[check_mask1] = 1
        if check_mask2.any():
            action[check_mask2] = -1
        return np.float32(action)
#endregion
class MultiProcessingWrapper(Wrapper):
    """Wrapper to enable multiprocessing for vectorized environments.

    The info dict returned by the firmware wrapper contains CasADi models. These models cannot be
    pickled and therefore cannot be passed between processes. This wrapper removes the CasADi models
    from the info dict to enable multiprocessing.
    """

    def __init__(self, env: Env):
        """Initialize the wrapper.

        Args:
            env: The firmware wrapper.
        """
        super().__init__(env)

    def reset(self, *args: Any, **kwargs: dict[str, Any]) -> tuple[np.ndarray, dict]:
        """Reset the environment.

        Returns:
            The initial observation of the next episode.
        """
        obs, info = self.env.reset(*args, **kwargs)
        return obs, self._remove_non_serializable(info)

    def _remove_non_serializable(self, info: dict[str, Any]) -> dict[str, Any]:
        """Remove non-serializable objects from the info dict."""
        # CasADi models cannot be pickled and therefore cannot be passed between processes
        info.pop("symbolic_model", None)
        info.pop("symbolic_constraints", None)
        return info
