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

from copy import deepcopy
import logging
from typing import Any

import numpy as np
from gymnasium import Env, Wrapper
from gymnasium.error import InvalidAction
from gymnasium.spaces import Box
from safe_control_gym.controllers.firmware.firmware_wrapper import FirmwareWrapper
from safe_control_gym.envs.benchmark_env import Cost, Task

from lsy_drone_racing.rotations import map2pi
from lsy_drone_racing.create_waypoints import create_waypoints

logger = logging.getLogger(__name__)


class DroneRacingWrapper(Wrapper):
    """Drone racing firmware wrapper to make the environment compatible with the gymnasium API.

    In contrast to the underlying environment, this wrapper only accepts FullState commands as
    actions.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env: FirmwareWrapper, terminate_on_lap: bool = True, train_random_state: bool = False, inc_gate_obs: bool = True):
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
        self.action_scale = np.array([1, 1, 1, np.pi])
        self.action_space = Box(-1, 1, shape=(4,), dtype=np.float32)

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
        self.inc_gate_obs = inc_gate_obs
        drone_limits = [5, 5, 5, np.pi, np.pi, np.pi, 10, 10, 10, 10, 10, 10]
        ## We may want to choose if we want to include the gate and obstacle observations
        if self.inc_gate_obs:
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
            traj_limits = [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            temp_h = np.array(traj_limits*self.obs_goal_horizon)
            temp_l = -np.array(traj_limits*self.obs_goal_horizon)
            obs_limits_high = np.concatenate((obs_limits_high, temp_h), axis=0)
            obs_limits_low = np.concatenate((obs_limits_low, temp_l), axis=0)
        self.observation_space = Box(obs_limits_low, obs_limits_high, dtype=np.float32)

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
        self.X_GOAL = None
        self.X_GOAL_crop = None
        self.train_random_state = train_random_state
        self._wrap_dist = None
        self._wrap_ctr_step = None
    @property
    def time(self) -> float:
        """Return the current simulation time in seconds."""
        return self._sim_time
    
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
        self._reset_required = False
        self._sim_time = 0.0
        self._f_rotors[:] = 0.0
        obs, info = self.env.reset()
        self.X_GOAL = create_waypoints(obs, info)[0]
        if self.train_random_state:
            start_ind = np.random.randint(1, self.X_GOAL.shape[0]/2)
            start_pose = self.X_GOAL[start_ind,:3]
            self.X_GOAL_crop = self.X_GOAL[start_ind:,:]
            self.env.env.INIT_X = start_pose[0] + 0.05 # Manual adjustment
            self.env.env.INIT_Y = start_pose[1]
            self.env.env.INIT_Z = start_pose[2]
            if start_ind > self.X_GOAL.shape[0]/4 and False:
                next_pose = self.X_GOAL[start_ind+1,:3]
                self.env.env.INIT_X_DOT = (next_pose[0] - start_pose[0])*30
                self.env.env.INIT_Y_DOT = (next_pose[1] - start_pose[1])*30
                self.env.env.INIT_Z_DOT = (next_pose[2] - start_pose[2])*30
        else:
            self.X_GOAL_crop = deepcopy(self.X_GOAL)
        ## Reset again to change the env to the initial state
        obs, info = self.env.reset()
        # env.reset() cannot reset in the exact position, which leads to a mismatch.
        self.X_GOAL_crop[0,:3] = obs[:6:2]
        # Store obstacle height for observation expansion during env steps.
        if self.obs_goal_horizon > 0:
            self._wrap_ctr_step = 0
            obs = self.observation_transform(obs, info, self.X_GOAL_crop, self._wrap_ctr_step, self.obs_goal_horizon, self.inc_gate_obs).astype(np.float32)
            self.goal_pos = obs[-12:]
            self._wrap_dist = np.linalg.norm(obs[:3] - self.goal_pos[:3], axis=0)
        else:
            raise NotImplementedError
            obs = self.observation_transform(obs, info).astype(np.float32)
        self._drone_pose = obs[[0, 1, 2, 5]]
        # print(f'Reset Position: {obs[:3]}')
        # print(f'Reset Goal Position: {obs[12:15]}')
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment.

        Args:
            action: The action to take in the environment. See action space for details.

        Returns:
            The next observation, the reward, the terminated and truncated flags, and the info dict.
        """
        assert not self._reset_required, "Environment must be reset before taking a step"
        if action not in self.action_space:
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
            info["task_completed"] = True
            terminated = True
        elif self.terminate_on_lap and done:  # Done, but last gate not passed -> terminate
            terminated = True
        # Increment the sim time after the step if we are not yet done.
        if not terminated and not truncated:
            self._sim_time += self.env.ctrl_dt
        
        if self.obs_goal_horizon > 0:
            self._wrap_ctr_step += 1
            obs = self.observation_transform(obs, info, self.X_GOAL_crop, self._wrap_ctr_step, self.obs_goal_horizon, self.inc_gate_obs).astype(np.float32)
            #### Integrate Reward Wrapper inside
            reward = self._compute_reward(obs, reward, terminated, truncated, info)
        else:
            raise NotImplementedError
            obs = self.observation_transform(obs, info).astype(np.float32)
        
        self._drone_pose = obs[[0, 1, 2, 5]]
        if obs not in self.observation_space:
            terminated = True
            reward = -1
        self._reset_required = terminated or truncated
        # print(f'Step Position: {obs[:3]}')
        # print(f'Step Goal Position: {obs[12:15]}')
        return obs, reward, terminated, truncated, info

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
    def observation_transform(obs: np.ndarray, info: dict[str, Any], X_GOAL: np.ndarray=None, crnt_step: int=-1, obs_goal_horizon: int=0, inc_gate_obs: bool=True) -> np.ndarray:
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
        if inc_gate_obs:
            obs = np.concatenate(
                [
                    drone_pos,
                    drone_rpy,
                    drone_vel,
                    drone_ang_vel,
                    info["gates_pose"][:, [0, 1, 2, 5]].flatten(),
                    info["gates_in_range"],
                    info["obstacles_pose"][:, :3].flatten(),
                    info["obstacles_in_range"],
                    [info["current_gate_id"]],
                ]
            )
        else:
            obs = np.concatenate(
                [
                    drone_pos,
                    drone_rpy,
                    drone_vel,
                    drone_ang_vel,
                ]
            )
        if obs_goal_horizon > 0:
            wp_idx = [min(crnt_step + 1 + i, X_GOAL.shape[0]-1) 
                    for i in range(obs_goal_horizon)]
            goal_state = X_GOAL[wp_idx].flatten()
            obs = np.concatenate([obs, goal_state])
        return obs

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
        goal_state = obs[-12*self.obs_goal_horizon:].reshape(self.obs_goal_horizon, 12)
        goal_x, goal_y, goal_z = goal_state[:, 0], goal_state[:, 1], goal_state[:, 2]
        goal_pos_all = np.array([goal_x, goal_y, goal_z]).T
        # Don't know if it is necessary to enforce closeness to the whole horizon
        if False:
            goal_pos = goal_pos_all[0,:]
            dist = np.sum(np.linalg.norm(drone_pos - goal_pos, axis=0))
            reward += np.exp(-dist)
        else:
            dist = np.linalg.norm(drone_pos[None,:] - goal_pos_all, axis=1)
            disc_factor = [0.6**i for i in range(self.obs_goal_horizon)]
            dist = np.sum(dist*disc_factor)
            rew_std = 0.01
            reward += np.exp(-(dist**2)/(rew_std**2))
        ## Enforce Progress
        if self._wrap_ctr_step >= 1:
            goal_pos_old = goal_pos_all[-2,:] # Previous Goal Position
            dist_old = np.linalg.norm(drone_pos - goal_pos_old, axis=0) # Distance to previous goal
            # self._wrap_dist is the distance to goal in previous iteration
            reward += (self._wrap_dist - dist_old) # Progress Reward
            print(f'Progress Reward: {self._wrap_dist - dist_old}')
            ## UPdate Distance
            self.goal_pos = goal_pos_all[-1,:]
            self._wrap_dist = np.linalg.norm(drone_pos - self.goal_pos, axis=0)
        ## Crash Penality
        crash_penality = -1 if self.env.env.currently_collided else 0
        ## Constraint Violation Penality
        cstr_penalty = -1 if self.env.env.cnstr_violation else 0
        ## Gate Passing Reward
        gate_rew = 1 if self.env.env.stepped_through_gate else 0
        return reward+crash_penality+cstr_penalty+gate_rew

        
class DroneRacingObservationWrapper:
    """Wrapper to transform the observation space the firmware wrapper.

    This wrapper matches the observation space of the DroneRacingWrapper. See its definition for
    more details. While we want to transform the observation space, we do not want to change the API
    of the firmware wrapper. Therefore, we create a separate wrapper for the observation space.

    Note:
        This wrapper is not a subclass of the gymnasium ObservationWrapper because the firmware is
        not compatible with the gymnasium API.
    """

    def __init__(self, env: FirmwareWrapper, inc_gate_obs: bool=True):
        """Initialize the wrapper.

        Args:
            env: The firmware wrapper.
        """
        if not isinstance(env, FirmwareWrapper):
            raise TypeError(f"`env` must be an instance of `FirmwareWrapper`, is {type(env)}")
        self.env = env
        self.pyb_client_id: int = env.env.PYB_CLIENT
        self.X_GOAL = None
        self.obs_goal_horizon = self.env.env.obs_goal_horizon
        self._wrap_ctr_step = None
        self.inc_gate_obs = inc_gate_obs
    
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

    def reset(self, *args: Any, **kwargs: dict[str, Any]) -> tuple[np.ndarray, dict]:
        """Reset the environment.

        Args:
            args: Positional arguments to pass to the firmware wrapper.
            kwargs: Keyword arguments to pass to the firmware wrapper.

        Returns:
            The transformed observation and the info dict.
        """
        obs, info = self.env.reset(*args, **kwargs)
        # Just need waypoints for drawing trajectory
        self.X_GOAL, self.waypoints = create_waypoints(obs, info)
        if self.obs_goal_horizon > 0:
            self._wrap_ctr_step = 0
            obs = DroneRacingWrapper.observation_transform(obs, info, self.X_GOAL, self._wrap_ctr_step, self.obs_goal_horizon, self.inc_gate_obs).astype(np.float32)
            self.goal_pos = obs[-12:]
            self._wrap_dist = np.linalg.norm(obs[:3] - self.goal_pos[:3], axis=0)
        else:
            raise NotImplementedError
            obs = DroneRacingWrapper.observation_transform(obs, info).astype(np.float32)
        return obs, info

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
        if self.obs_goal_horizon > 0:
            self._wrap_ctr_step += 1
            obs = DroneRacingWrapper.observation_transform(obs, info, self.X_GOAL, self._wrap_ctr_step, self.obs_goal_horizon, self.inc_gate_obs).astype(np.float32)
            print(f'Old Goal Position: {self.goal_pos}')
            print(f'Obs -2: {obs[-24:-12]}')
            print(f'Previous Distance: {self._wrap_dist}')
            print(f'Current Distance: {np.linalg.norm(obs[:3] - obs[-24:-21], axis=0)}')
            print(f'New Goal Position: {obs[-12:]}')
            print(f'New Distance: {np.linalg.norm(obs[:3] - obs[-12:-9], axis=0)}')
            self.goal_pos = obs[-12:]
            self._wrap_dist = np.linalg.norm(obs[:3] - self.goal_pos[:3], axis=0)
        else:
            raise NotImplementedError
            obs = DroneRacingWrapper.observation_transform(obs, info).astype(np.float32)
        return obs, reward, done, info, action


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
