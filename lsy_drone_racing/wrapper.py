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
from safe_control_gym.envs.benchmark_env import Cost, Task

from lsy_drone_racing.rotations import map2pi
from lsy_drone_racing.create_waypoints import create_waypoints, find_closest_traj_point, find_closest_gate, set_buffer_warmup
from lsy_drone_racing.path_planning import calc_best_path
from lsy_drone_racing.global_parameters import *

logger = logging.getLogger(__name__)

enum = ['waypoints', 'traj', 'None']


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
        self.if_chunk = 'waypoints' # 'waypoints or traj or None'
        assert self.if_chunk in enum, "The chunk must be either waypoints or traj or None"
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
        ## We may want to choose if we want to include the gate and obstacle observations
        if False:
            gate_limits = [5, 5, 5, np.pi] * n_gates + [1] * n_gates  # Gate poses and range mask
            obstacle_limits = [5, 5, 5] * n_obstacles + [1] * n_obstacles  # Obstacle pos and range mask
            obs_limits = drone_limits + gate_limits + obstacle_limits + [n_gates]  # [1] for gate_id
            obs_limits_high = np.array(obs_limits)
            obs_limits_low = np.concatenate([-obs_limits_high[:-1], [-1]])
        else:
            obs_limits_high = np.array(drone_limits)
            obs_limits_low = -np.array(drone_limits)
        ## Add Goal States to Observation Space
        self.obs_goal_horizon = self.env.env.obs_goal_horizon
        if self.obs_goal_horizon > 0:
            traj_limits = [5, 5, 5]
            if self.if_chunk == 'traj' or self.if_chunk == 'waypoints':
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
        self.X_GOAL = None # Goal States
        self.X_GOAL_crop = None # Cropped Goal States, for random training
        self.X_GOAL_distances = None # Distance between each step//Only for a specific reward
        self.train_random_state = train_random_state # Randomly choose the starting waypoint
        self._wrap_ctr_step = None # Counter for the current control step
        self.waypoints = None # Waypoints
        self.wp_traj_idx = None # Waypoint to Trajectory Index (waypoint corresponds to what trajectory point)
        self.wp_gate_dict = None # Waypoint to Gate Dictionary (waypoint corresponds to what gate)
        self._current_gate_idx = None # Current Gate Index
        self.chunk_goal = None # Chunk Goal
        self.idx_chunk_goal = None # Index of the Chunk Goal
        self.chunk_distance = None # Distance to the Chunk Goal
    
    @property
    def time(self) -> float:
        """Return the current simulation time in seconds."""
        return self._sim_time
    
    #region Reset
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
        #region Path Planning
        ## Extract data for the path planning
        self.gates = info["nominal_gates_pos_and_type"]
        self._current_gate_idx = info["current_gate_id"]
        self.obstacles = info["nominal_obstacles_pos"]
        self.start_point = [obs[0], obs[2], obs[4]]
        freq = self.env.ctrl_freq
        ## Define times steps for the trajectory
        self.t = np.linspace(0, 1, int(DURATION * freq))
        ## Path Planning
        # It is only for not printing
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull):
                self.X_GOAL, self.waypoints = calc_best_path(gates=self.gates, obstacles=self.obstacles, start_point=self.start_point, t=self.t, plot=False)
        ## Find the trajectory point closest to the waypoints
        self.wp_traj_idx = find_closest_traj_point(self.waypoints, self.X_GOAL)
        ## Find the gate corresponding to the waypoints
        self.wp_gate_dict = find_closest_gate(self.X_GOAL, info, self.wp_traj_idx)
        ## Calculate the distance between each step
        self.X_GOAL_distance = np.mean(np.linalg.norm(self.X_GOAL[1:,:] - self.X_GOAL[:-1,:], axis=1))
        if self.train_random_state: # Spawn the drone on randomly chosen waypoint
            # start_ind = np.random.randint(1, self.waypoints.shape[0]-2)
            start_ind = 7
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
            self.env.env.INIT_X, self.env.env.INIT_Y, self.env.env.INIT_Z = buffer
            # env.reset() cannot reset in the exact position, which leads to a mismatch.
            self.env.env.current_gate = self.wp_gate_dict[start_ind]
            self._current_gate_idx = self.wp_gate_dict[start_ind]
        else:
            assert np.linalg.norm(self.start_point[2] - TAKEOFF_HEIGHT) < 0.1, "Takeoff height must be the start height for training!"
            self.X_GOAL_crop = deepcopy(self.X_GOAL)
            self._current_gate_idx = info["current_gate_id"]
        #endregion
        # Keep the horizon included in the obs
        assert self.obs_goal_horizon > 0, "The horizon must be greater than 0"
        self._wrap_ctr_step = 0
        #region Observation Transform
        obs = self.observation_transform(obs, info, self.X_GOAL_crop, self._wrap_ctr_step, self.obs_goal_horizon).astype(np.float32)
        self._drone_pose = obs[[0, 1, 2, 5]]
        # Determine the point chunk to follow, either waypoints or trajectory, or None
        # traj locks onto the last point of the horizon until it disappears in the obs
        # waypoints locks onto the waypoints and switches to the next one when the drone reaches the index
        if self.if_chunk == 'traj' and False:
            self.idx_chunk_goal = self.obs_goal_horizon
            self.chunk_goal = obs[12+3*(self.idx_chunk_goal-1):]
            obs = np.concatenate([obs[:15], self.chunk_goal])
            # print(f'In RESET Chunk Idx: {self.idx_chunk_goal}')
            # print(f'In RESET Current Chunk Goal: {self.chunk_goal}')
            self.chunk_distance = np.linalg.norm(obs[:3] - self.chunk_goal[:3], axis=0)
        elif self.if_chunk == 'waypoints':
            self.idx_chunk_goal = 1
            self.chunk_goal = self.waypoints[self.idx_chunk_goal,:]
            self.idx_wp2traj = self.wp_traj_idx[self.idx_chunk_goal]
            self.chunk_distance = np.linalg.norm(obs[:3] - self.chunk_goal[:3], axis=0)
            obs = np.concatenate([obs[:15], self.chunk_goal])
        else:
            pass
        #endregion
        return obs, info
    #endregion
    #region Step
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
        #region Observation Transform
        obs = self.observation_transform(obs, info, self.X_GOAL_crop, self._wrap_ctr_step, self.obs_goal_horizon).astype(np.float32)
        self._wrap_ctr_step += 1
        ## Update Chunk Goal and Change Observation
        if self.if_chunk == 'traj' and False:
            self.idx_chunk_goal -= 1
            self.chunk_goal = obs[12+3*(self.idx_chunk_goal-1):3*(self.idx_chunk_goal)]
            obs = np.concatenate([obs[:15], self.chunk_goal])
            ## Integrate Reward Wrapper inside
            reward = self._compute_reward(obs, reward, terminated, truncated, info)
            self.idx_chunk_goal = self.obs_goal_horizon+1 if self.idx_chunk_goal == 1 else self.idx_chunk_goal
            # we set it to idx+1 because in the step() we will decrease it right away. The inconvenience 
            # comes from initializing it directly in reset().
        elif self.if_chunk == 'waypoints':
            obs = np.concatenate([obs[:15], self.chunk_goal])
            ## Integrate Reward Wrapper inside
            reward = self._compute_reward(obs, reward, terminated, truncated, info)
            if self._wrap_ctr_step == self.idx_wp2traj and False: # If we arrive the waypoint index
                dist = np.linalg.norm(obs[:3] - self.chunk_goal[:3], axis=0)
                m=0.4
                reward += -1/m*dist+1 # REWARD for the switch
                # print('Waypoint Passed with local dist:', dist)
                # print('Waypoint Passed with local reward:', -1/m*dist+1)
                self.idx_chunk_goal += 1 if self.idx_chunk_goal < self.waypoints.shape[0]-1 else 0
                self.chunk_goal = self.waypoints[self.idx_chunk_goal,:]
                self.idx_wp2traj = self.wp_traj_idx[self.idx_chunk_goal]
                # Update the distance to the next chunk
                self.chunk_distance = np.linalg.norm(obs[:3] - self.chunk_goal[:3], axis=0)
        else:
            ## Integrate Reward Wrapper inside
            reward = self._compute_reward(obs, reward, terminated, truncated, info)
        #endregion
        self._drone_pose = obs[[0, 1, 2, 5]]
        # Check Observation
        if obs not in self.observation_space:
            pos_space = Box(self.observation_space.low[:3], self.observation_space.high[:3], dtype=np.float32)
            if obs[:3] not in pos_space:
                print(f'Observation {obs[:3]} is not in the position space!')
                terminated = True
            else:
                print(f'Observation {obs[9:12]} is not in the observation space!')
        self._reset_required = terminated or truncated
        # Check if the gate passed
        if info["current_gate_id"] != self._current_gate_idx:
            self._current_gate_idx = info["current_gate_id"]
            print(f'Gate Passed! Current Gate Index: {self._current_gate_idx}')
        ## Recalculating the trajectory if the gate position is changed
        #region Recalculate Trajectory
        if self._current_gate_idx != -1:
            recalc = False if (info['gates_pose'][self._current_gate_idx,:2] == self.gates[self._current_gate_idx, :2]).all() else True
            # self.gates is from nominal gate pose, info['gates_pose'] is the current gate pose
            if recalc and False:
                print(f"Gate {self._current_gate_idx} at {self.gates[self._current_gate_idx]} is not the same as {info['gates_pose'][self._current_gate_idx]}")    
                self.gates[self._current_gate_idx, :-1] = info['gates_pose'][self._current_gate_idx]
                self.X_GOAL, self.waypoints = calc_best_path(self.gates, self.obstacles, self.start_point, t=self.t, plot=False)
                assert max(self.X_GOAL[:,2]) < 2.5, "Drone must stay below the ceiling"
                self.wp_traj_idx = find_closest_traj_point(self.waypoints, self.X_GOAL)
        #endregion
        print(f'Distance to waypoint: {self.chunk_distance} with Step: {self._wrap_ctr_step}')
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
        """Transform the observation to include additional information.

        Args:
            obs: The observation to transform.
            info: Additional information to include in the observation.

        Returns:
            The transformed observation.
        """
        drone_pos = obs[0:6:2]
        drone_vel = obs[1:6:2]
        drone_rpy = obs[6:9]
        drone_ang_vel = obs[9:12]
        # obs = np.concatenate(
        #     [
        #         drone_pos,
        #         drone_rpy,
        #         drone_vel,
        #         drone_ang_vel,
        #         info["gates_pose"][:, [0, 1, 2, 5]].flatten(),
        #         info["gates_in_range"],
        #         info["obstacles_pose"][:, :3].flatten(),
        #         info["obstacles_in_range"],
        #         [info["current_gate_id"]],
        #     ]
        # )
        obs = np.concatenate(
            [
                drone_pos,
                drone_rpy,
                drone_vel,
                drone_ang_vel,
            ]
        )
        if obs_goal_horizon > 0:
            wp_idx = [min(crnt_step + i, X_GOAL.shape[0]-1) 
                    for i in range(obs_goal_horizon)]
            goal_state = X_GOAL[wp_idx].flatten()
            obs = np.concatenate([obs, goal_state], dtype=np.float32)
        return obs

    #region Reward
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
        goal_state = obs[-3*self.obs_goal_horizon:].reshape(self.obs_goal_horizon, 3)
        goal_x, goal_y, goal_z = goal_state[:, 0], goal_state[:, 1], goal_state[:, 2]
        goal_pos_all = np.array([goal_x, goal_y, goal_z]).T
        # Don't know if it is necessary to enforce closeness to the whole horizon
        if False:
            dist = np.linalg.norm(drone_pos[None,:] - goal_pos_all, axis=1)
            expected_dist = np.linalg.norm(goal_pos_all[1:,:] - goal_pos_all[:-1,:], axis=1)
            disc_factor = [0.9**i for i in range(self.obs_goal_horizon)]
            reward += (np.exp(-2*(dist-expected_dist))-0.3).dot(disc_factor)
        elif False:
            dist = np.linalg.norm(drone_pos[None,:] - goal_pos_all, axis=1)
            disc_factor = [0.9**i for i in range(self.obs_goal_horizon)]
            rew_std = [0.05 for i in range(1,self.obs_goal_horizon+1)]
            reward += np.exp(-1/2*(dist/rew_std)**2).dot(disc_factor)
        elif True:
            # if self.idx_chunk_goal == self.obs_goal_horizon:
            #     ## This if case is for switching to the next chunk.
            #     self.chunk_distance = np.linalg.norm(drone_pos - self.chunk_goal[:3], axis=0)
            #     reward += 0.05
            current_chunk_distance = np.linalg.norm(drone_pos - self.chunk_goal[:3], axis=0)
            reward += self.chunk_distance - current_chunk_distance
            self.chunk_distance = current_chunk_distance
            # print(f'Reward: {reward}')
        elif False:
            dist = np.linalg.norm(drone_pos - self.chunk_goal[:3], axis=0)
            # Chunk_goal is the same as the last observation
            reward += np.exp(-2*dist)-0.2
            print(f'Reward: {reward}')
        else:
            dist = np.linalg.norm(drone_pos[None,:] - goal_pos_all, axis=1)
            dist = np.min(dist)
            ind = self._wrap_ctr_step if self._wrap_ctr_step < len(self.X_GOAL_distances) else -1
            dist_comp = self.X_GOAL_distances[ind]
            b = 0.2
            m = -1/b
            reward += m*(dist-dist_comp)+1
            print(reward)
            
        ## Body Rate Penalty
        body_rate = obs[9:12]
        body_rate_penalty = -0.01*np.linalg.norm(body_rate, axis=0)
        ## Crash Penality
        crash_penality = -10 if self.env.env.currently_collided else 0
        ## Constraint Violation Penality
        cstr_penalty = -10 if self.env.env.cnstr_violation else 0
        ## Gate Passing Reward
        # gate_rew = 2.5 if self.env.env.stepped_through_gate else 0
        ## Waypoint Passing Reward
        # It is outside of the reward function
        task_rew = 10 if info["task_completed"] else 0
        return reward+crash_penality+cstr_penalty+task_rew+body_rate_penalty
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
        self.if_chunk = 'waypoints' # traj or waypoints or None
        assert self.if_chunk in enum, "The chunk must be either waypoints or traj or None"
        self.warmup_length = None
        self.warmed_up = False

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
    #region Reset
    def reset(self, *args: Any, **kwargs: dict[str, Any]) -> tuple[np.ndarray, dict]:
        """Reset the environment.

        Args:
            args: Positional arguments to pass to the firmware wrapper.
            kwargs: Keyword arguments to pass to the firmware wrapper.

        Returns:
            The transformed observation and the info dict.
        """
        obs, info = self.env.reset(*args, **kwargs)
        ## Path Planning
        #region Path Planning
        self.gates = info["nominal_gates_pos_and_type"]
        self._current_gate_idx = info["current_gate_id"]
        self.obstacles = info["nominal_obstacles_pos"]
        self.start_point = [obs[0], obs[2], TAKEOFF_HEIGHT]
        freq = self.env.ctrl_freq
        self.t = np.linspace(0, 1, int(DURATION * freq))
        self.X_GOAL, self.waypoints = calc_best_path(gates=self.gates, obstacles=self.obstacles, start_point=self.start_point, t=self.t, plot=False)
        self.wp_traj_idx = find_closest_traj_point(self.waypoints, self.X_GOAL)
        self.warmup_length = 15
        self.X_GOAL = set_buffer_warmup(self.X_GOAL, self.warmup_length)
        #endregion
        #region Observation Transform
        assert self.obs_goal_horizon > 0, "The horizon must be greater than 0"
        self._wrap_ctr_step = 0
        obs = DroneRacingWrapper.observation_transform(obs, info, self.X_GOAL, self._wrap_ctr_step, self.obs_goal_horizon).astype(np.float32)
        if self.if_chunk == 'traj' and False:
            self.idx_chunk_goal = self.obs_goal_horizon
            self.chunk_goal = obs[12*self.idx_chunk_goal:]
            obs = np.concatenate([obs[:15], self.chunk_goal])
            # print(f'In RESET Chunk Idx: {self.idx_chunk_goal}')
            # print(f'In RESET Current Chunk Goal: {self.chunk_goal}')
            self.chunk_distance = np.linalg.norm(obs[:3] - self.chunk_goal[:3], axis=0)
        elif self.if_chunk == 'waypoints':
            self.idx_chunk_goal = 1
            self.chunk_goal = self.waypoints[self.idx_chunk_goal,:]
            self.idx_wp2traj = self.wp_traj_idx[self.idx_chunk_goal]
            self.chunk_distance = np.linalg.norm(obs[:3] - self.chunk_goal[:3], axis=0)
            obs = np.concatenate([obs[:15], self.chunk_goal])
        #endregion
        return obs, info
    #endregion

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
        if not self.warmed_up:
            ## Warmup the observation (Return the goal of the warmuplength step.)
            obs = DroneRacingWrapper.observation_transform(obs, info, self.X_GOAL, self.warmup_length, 
                                                           self.obs_goal_horizon).astype(np.float32)
            if self._wrap_ctr_step == self.warmup_length-1:
                self.warmed_up = True
                self._wrap_ctr_step = -1 # Reset the counter to -1 to start from 0
        else:
            obs = DroneRacingWrapper.observation_transform(obs, info, self.X_GOAL, self._wrap_ctr_step, 
                                                       self.obs_goal_horizon).astype(np.float32)
        self._wrap_ctr_step += 1 if curr_time > TAKEOFF_DURATION else 0 # X seconds for takeoff
        if self.if_chunk == 'traj' and False:
            self.idx_chunk_goal = self.obs_goal_horizon
            self.chunk_goal = obs[12*self.idx_chunk_goal:]
            obs = np.concatenate([obs[:15], self.chunk_goal])
            # print(f'In RESET Chunk Idx: {self.idx_chunk_goal}')
            # print(f'In RESET Current Chunk Goal: {self.chunk_goal}')
            self.chunk_distance = np.linalg.norm(obs[:3] - self.chunk_goal[:3], axis=0)
        elif self.if_chunk == 'waypoints':
            obs = np.concatenate([obs[:15], self.chunk_goal])
            self.chunk_distance = np.linalg.norm(obs[:3] - self.chunk_goal[:3], axis=0)
            if self._wrap_ctr_step == self.idx_wp2traj: # If we arrive the waypoint index
                self.idx_chunk_goal += 1 if self.idx_chunk_goal < self.waypoints.shape[0]-1 else 0
                self.chunk_goal = self.waypoints[self.idx_chunk_goal,:]
                self.idx_wp2traj = self.wp_traj_idx[self.idx_chunk_goal]
                self.chunk_distance = np.linalg.norm(obs[:3] - self.chunk_goal[:3], axis=0)
                print(f'Distance to waypoint: {self.chunk_distance} with Step: {self._wrap_ctr_step}')
        #endregion
        #region Recalculate Trajectory
        if self._current_gate_idx != -1:
            recalc = False if (info['gates_pose'][self._current_gate_idx,:2] == self.gates[self._current_gate_idx, :2]).all() else True
            if False:
                print(f"Gate {self._current_gate_idx} at {self.gates[self._current_gate_idx]} is not the same as {info['gates_pose'][self._current_gate_idx]}")    
                self.gates[self._current_gate_idx, :-1] = info['gates_pose'][self._current_gate_idx]
                self.X_GOAL, self.waypoints = calc_best_path(self.gates, self.obstacles, self.start_point, t=self.t, plot=False)
                # convert path resulted from splev to x,y,z points
                assert max(self.X_GOAL[:,2]) < 2.5, "Drone must stay below the ceiling"
                self.wp_traj_idx = find_closest_traj_point(self.waypoints, self.X_GOAL)
        #endregion
        return obs, reward, done, info, action
#region Action Wrapper
class ActionWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.env = env
        self.fas = env.fixed_action_scale
        self.las = env.learned_action_scale
        self.current_obs = None

    def reset(self, **kwargs):
        self.current_obs, info = self.env.reset(**kwargs)
        return self.current_obs, info

    def step(self, action):
        modified_action = self.modify_action(action, self.current_obs)
        obs, reward, terminated, truncated, info = self.env.step(modified_action)
        self.current_obs = obs
        return obs, reward, terminated, truncated, info

    def modify_action(self, action, obs):
        # Modify the action based on the current observation
        # Example: setting action to 0 if a specific condition is met in the observation
        current_pose = obs[:3]
        current_yaw = obs[5]
        goal_pose = obs[12:15]
        action_fix = goal_pose - current_pose
        # goal_yaw = np.arctan2(-(goal_pose[1]-current_pose[1]), (goal_pose[0]-current_pose[0]))
        # yaw = (goal_yaw - current_yaw)/np.pi
        yaw = 0.0
        # action[:3] = action[:3] + self.fas * action_fix
        action = action_fix
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

class RewardWrapper(Wrapper):
    """Wrapper to alter the default reward function from the environment for RL training."""

    def __init__(self, env: Env):
        """Initialize the wrapper.

        Args:
            env: The firmware wrapper.
        """
        super().__init__(env)
        self._last_gate = None

    def reset(self, *args: Any, **kwargs: dict[str, Any]) -> np.ndarray:
        """Reset the environment.

        Args:
            args: Positional arguments to pass to the firmware wrapper.
            kwargs: Keyword arguments to pass to the firmware wrapper.

        Returns:
            The initial observation of the next episode.
        """
        obs, info = self.env.reset(*args, **kwargs)
        self._last_gate = info["current_gate_id"]
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment.

        Args:
            action: The action to take in the environment. See action space for details.

        Returns:
            The next observation, the reward, the terminated and truncated flags, and the info dict.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = self._compute_reward(obs, reward, terminated, truncated, info)
        return obs, reward, terminated, truncated, info

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
        gate_id = info["current_gate_id"]
        gate_reward = np.exp(-np.linalg.norm(info["gates_pose"][gate_id, :3] - obs[:3]))
        gate_passed_reward = 0 if gate_id == self._last_gate else 0.1
        crash_penality = -1 if terminated and not info["task_completed"] else 0
        return gate_reward + crash_penality + gate_passed_reward
