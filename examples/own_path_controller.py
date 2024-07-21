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
from lsy_drone_racing.path_planning import PIDController, PathPlanning
import csv
import pandas as pd
import os
import json
# added by me for not using reference to dict
import copy

TARGET_DURATION = 6.5 # seconds # 5.1 is best for Level 1
START_TO_HEIGHT = 0.1 # meters

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
        self.initial_info = initial_info
        self.VERBOSE = False
        self.BUFFER_SIZE = buffer_size

        # Store a priori scenario information.
        self.NOMINAL_GATES = copy.deepcopy(initial_info["nominal_gates_pos_and_type"])
        self.ACTUAL_GATES = copy.deepcopy(initial_info["nominal_gates_pos_and_type"])
        self.NOMINAL_OBSTACLES = copy.deepcopy(initial_info["nominal_obstacles_pos"])


        # Reset counters and buffers.
        self.reset()
        self.episode_reset()

        #########################
        # REPLACE THIS (START) ##
        #########################

        # Example: Hard-code waypoints through the gates. Obviously this is a crude way of
        # completing the challenge that is highly susceptible to noise and does not generalize at
        # all. It is meant solely as an example on how the drones can be controlled

        

        self.pid = PIDController()

        start_point = [self.initial_obs[0], self.initial_obs[2], START_TO_HEIGHT] 
        gates = self.NOMINAL_GATES
        t = np.linspace(0, 1, int(TARGET_DURATION * self.CTRL_FREQ))

        path_planning = PathPlanning(gates, self.NOMINAL_OBSTACLES, start_point, t=t, plot=False)

        path, waypoints = path_planning.calc_best_path()
        self.waypoints = waypoints

        self.append_new_path_and_gates_to_csv(path, gates, 'paths_gates.csv', obstacles=self.NOMINAL_OBSTACLES, waypoints=waypoints)
        # convert path resulted from splev to x,y,z points
        self.ref_x, self.ref_y, self.ref_z = path
        assert max(self.ref_z) < 2.5, "Drone must stay below the ceiling"

        if self.VERBOSE:
            # Draw the trajectory on PyBullet's GUI.
            draw_trajectory(initial_info, self.waypoints, self.ref_x, self.ref_y, self.ref_z)
        self.step_offset = 0
        self._take_off = False
        self._setpoint_land = False
        self._land = False
        #########################
        # REPLACE THIS (END) ####
        #########################
    
    # Funktion zum Speichern der aktualisierten Pfade und Tore
    def append_new_path_and_gates_to_csv(self, new_path, actual_gates, file_name, obstacles=None, waypoints=None):
        # Überprüfen, ob die CSV-Datei bereits existiert
        if os.path.isfile(file_name) and obstacles is None:
            # Lesen der existierenden CSV-Datei in ein DataFrame
            df = pd.read_csv(file_name)
        else:
            # Erstellen eines neuen DataFrame, wenn die Datei nicht existiert
            df = pd.DataFrame()
            df["Obstacles"] = pd.Series([json.dumps(obstacles)])

        
        df["Last Waypoints"] = pd.Series([json.dumps(waypoints.tolist())])

        # Hinzufügen der neuen Daten als Spalten
        x = new_path[0].tolist()
        y = new_path[1].tolist()
        z = new_path[2].tolist()
        
        column_index = len(df.columns) // 2


        # save x, y, z path as path_i_x, path_i_y, path_i_z
        df[f'Path_{column_index}_x'] = pd.Series([json.dumps(x)])
        df[f'Path_{column_index}_y'] = pd.Series([json.dumps(y)])
        df[f'Path_{column_index}_z'] = pd.Series([json.dumps(z)])
        
        print(actual_gates, "Actual Gates")
        actual_gates_new = [[float(e) for e in ele] for ele in actual_gates]
        df[f'Gates_{column_index}'] = pd.Series([json.dumps(actual_gates_new)])

        # Speichern des aktualisierten DataFrames zurück in die CSV-Datei
        df.to_csv(file_name, index=False)

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

        if not self._take_off:
            command_type = Command.TAKEOFF
            args = [START_TO_HEIGHT, 0.2]  # Height, duration
            self._take_off = True  # Only send takeoff command once
        else:
            step = iteration - self.CTRL_FREQ  # Account for 1s delay due to takeoff
            if step < 0:
                if self.step_offset == 0:
                    self.step_offset = step
                #self.step_offset += 1
            
            step -= self.step_offset # step offset is negative
            if ep_time - 0.2 > 0 and step < len(self.ref_x):

                current_target_gate_id = copy.deepcopy(info["current_target_gate_id"])
                current_target_gate_pos = copy.deepcopy(info["current_target_gate_pos"])
                current_target_gate_type = copy.deepcopy(info["current_target_gate_type"])
                #append type to the current_target_gate_pos
                current_target_gate_pos.append(current_target_gate_type)
                nominal_gate_pos = self.NOMINAL_GATES[current_target_gate_id]

                if current_target_gate_pos == nominal_gate_pos:
                    target_pos = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
                    control_output = self.pid.compute_control(target_pos, obs)
                    #print("Control Output 1:", control_output)
                    #print("Target Pos:", target_pos)
                    target_vel = np.zeros(3)
                    target_acc = np.zeros(3)
                    target_yaw = 0.0
                    target_rpy_rates = np.zeros(3)
                    command_type = Command.FULLSTATE
                    args = [control_output, target_vel, target_acc, target_yaw, target_rpy_rates, ep_time]
                else:
                    print(f"Gate {current_target_gate_id} at {nominal_gate_pos} is not the same as {current_target_gate_pos}")
                    
                    start_point = [self.initial_obs[0], self.initial_obs[2], START_TO_HEIGHT] 
                    
                    self.ACTUAL_GATES[current_target_gate_id] = current_target_gate_pos
                    t = np.linspace(0, 1, int(TARGET_DURATION * self.CTRL_FREQ))
                    path_planning = PathPlanning(self.ACTUAL_GATES, self.NOMINAL_OBSTACLES, start_point, t=t, plot=False)

                    path, waypoints = path_planning.calc_best_path()
                    self.append_new_path_and_gates_to_csv(path, self.ACTUAL_GATES, 'paths_gates.csv', obstacles=None, waypoints=waypoints)
                    self.waypoints = waypoints
                    # convert path resulted from splev to x,y,z points
                    self.ref_x, self.ref_y, self.ref_z = path
                    assert max(self.ref_z) < 2.5, "Drone must stay below the ceiling"
                    #print(info, "Info")

                    target_pos = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
                    control_output = self.pid.compute_control(target_pos, obs)


                    #print("Control Output:", control_output)
                    target_vel = np.zeros(3)
                    target_acc = np.zeros(3)
                    target_yaw = 0.0
                    target_rpy_rates = np.zeros(3)
                    command_type = Command.FULLSTATE
                    args = [control_output, target_vel, target_acc, target_yaw, target_rpy_rates, ep_time]
                    #if self.VERBOSE:
                        # Draw the trajectory on PyBullet's GUI.
                        #draw_trajectory(self.initial_info, self.waypoints, self.ref_x, self.ref_y, self.ref_z)


              
                #print(info, "Info")
                #print(args, "Args")
                #print(obs, "Obs")

            # Notify set point stop has to be called every time we transition from low-level
            # commands to high-level ones. Prepares for landing
            elif step >= len(self.ref_x) and not self._setpoint_land and info["task_completed"] == False:
                print("Task not completed but reached the end of the path ins teps, continue to last reference point")
                target_pos = np.array([self.ref_x[-1], self.ref_y[-1], self.ref_z[-1]])
                target_vel = np.zeros(3)
                target_acc = np.zeros(3)
                target_yaw = 0.0
                target_rpy_rates = np.zeros(3)
                command_type = Command.FULLSTATE
                args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates, ep_time]
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
