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

from copy import deepcopy
import numpy as np
from scipy import interpolate
from stable_baselines3 import PPO

from lsy_drone_racing.command import Command
from lsy_drone_racing.controller import BaseController
from lsy_drone_racing.utils import draw_trajectory
from lsy_drone_racing.rotations import map2pi
from lsy_drone_racing.global_parameters import *


class Controller(BaseController):
    """Template controller class."""

    def __init__(
        self,
        initial_obs: np.ndarray,
        initial_info: dict,
        buffer_size: int = 100,
        verbose: bool = False,
        X_GOAL: np.ndarray = None,
        waypoints: list = None,
    ):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori
            infromation contained in dictionary `initial_info`. Use this method to initialize
            constants, counters, pre-plan trajectories, etc.

        Args:
            initial_obs: The initial observation of the environment's state. Consists of
                [drone_xyz_yaw, gates_xyz_yaw, gates_in_range, obstacles_xyz, obstacles_in_range,
                gate_id]
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
        self.iter_counter = 0
        assert X_GOAL is not None and waypoints is not None, "X_GOAL and waypoints must be provided"
        self.agent = PPO.load("./models/ppo_wp_lvl1_7s22.zip")
        self.las = self.agent.action_space.high[0]
        self.fas = 1 - self.las
        print(f'LAS: {self.las}')
        self.action_scale = np.array([1, 1, 1, np.pi])
        self.X_GOAL = X_GOAL
        self.waypoints = waypoints
        self.RL = True
        self.ref_x = self.X_GOAL[:, 0]
        self.ref_y = self.X_GOAL[:, 1]  
        self.ref_z = self.X_GOAL[:, 2]
        # Example: Hard-code waypoints through the gates. Obviously this is a crude way of
        # completing the challenge that is highly susceptible to noise and does not generalize at
        # all. It is meant solely as an example on how the drones can be controlled
        self._take_off = False
        self._setpoint_land = False
        self._land = False
        
        if self.VERBOSE:
            # Draw the trajectory on PyBullet's GUI.
            draw_trajectory(initial_info, self.waypoints, self.ref_x, self.ref_y, self.ref_z)

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
            obs: The environment's observation [drone_xyz_yaw, gates_xyz_yaw, gates_in_range,
                obstacles_xyz, obstacles_in_range, gate_id].
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
            args = [TAKEOFF_HEIGHT, TAKEOFF_DURATION]  # Height, duration
            self._take_off = True  # Only send takeoff command once
        else:
            step = iteration - TAKEOFF_DURATION * self.CTRL_FREQ  # Account for 2s delay due to takeoff
            if ep_time - TAKEOFF_DURATION > 0 and step < len(self.ref_x):
                command_type = Command.FULLSTATE
                self.iter_counter += 1
                action, _states = self.agent.predict(observation=obs)
                action = self.action_scale * action
                # Adding the first point in the horizon
                pos = self.action_scale[:-1] * obs[12:15]
                if False:#self.RL:
                    pos = (self.las*obs[:3] + action[:3]) + self.fas*pos
                else:
                    pos = 0.0*(self.las*obs[:3] + action[:3]) + 1.0*pos
                yaw = np.arctan2(-(pos[1]-obs[1]), (pos[0]-obs[0]))
                # yaw = 0.0
                args = [pos, np.zeros(3), np.zeros(3), yaw, np.zeros(3), ep_time]
            elif step >= len(self.ref_x) and not self._setpoint_land and info["task_completed"] == False:
                print("Task not completed but reached the end of the path ins teps, continue to last reference point")
                target_pos = np.array([self.ref_x[-1], self.ref_y[-1], self.ref_z[-1]])
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
