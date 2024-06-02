"""Example training script using the stable-baselines3 library.

Note:
    This script requires you to install the stable-baselines3 library.
"""

from __future__ import annotations

from copy import deepcopy
import logging
from functools import partial
from pathlib import Path

import numpy as np
import fire
from safe_control_gym.utils.registration import make
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from scipy import interpolate

from lsy_drone_racing.constants import FIRMWARE_FREQ
from lsy_drone_racing.utils import load_config, load_controller
from lsy_drone_racing.wrapper import DroneRacingWrapper

logger = logging.getLogger(__name__)


def create_race_env(config_path: Path, gui: bool = False) -> DroneRacingWrapper:
    """Create the drone racing environment."""
    # Load configuration and check if firmare should be used.
    config = load_config(config_path)
    # Overwrite config options
    config.quadrotor_config.gui = gui
    CTRL_FREQ = config.quadrotor_config["ctrl_freq"]
    # Create environment
    assert config.use_firmware, "Firmware must be used for the competition."
    pyb_freq = config.quadrotor_config["pyb_freq"]
    assert pyb_freq % FIRMWARE_FREQ == 0, "pyb_freq must be a multiple of firmware freq"
    config.quadrotor_config["ctrl_freq"] = FIRMWARE_FREQ
    x_goal = create_waypoints(config.quadrotor_config)
    
    env_factory = partial(make, "quadrotor",**config.quadrotor_config)
    firmware_env = make("firmware", env_factory, FIRMWARE_FREQ, CTRL_FREQ)
    firmware_env.env.X_GOAL = x_goal
    # goal state is of shape (N,12), where N is the number of waypoints and 12 is the states of the drone
    # x,dx,y,dy,z,dz,phi,theta,psi,p,q,r
    # We need to define something for the missing states since we only have x,y,z
    # Obey the order of the states
    #TODO:
    return DroneRacingWrapper(firmware_env, terminate_on_lap=True)


def main(config: str = "config/getting_started.yaml", gui: bool = False):
    """Create the environment, check its compatibility with sb3, and run a PPO agent."""
    logging.basicConfig(level=logging.INFO)
    config_path = Path(__file__).resolve().parents[1] / config # resolve() returns the absolute path, parents[1] /config adds the config
    env = create_race_env(config_path=config_path, gui=gui)
    ## Get initial state and info
    # obs, info = env.reset()
    ## Load the controller module
    # controller = "examples/controller.py"
    # ctrl_path = Path(__file__).parents[1] / controller
    # ctrl_class = load_controller(ctrl_path)
    # ctrl = ctrl_class(obs, info)
    check_env(env)  # Sanity check to ensure the environment conforms to the sb3 API
    if True:
        model = PPO("MlpPolicy", env, verbose=1, n_epochs=40)
        model.learn(total_timesteps=4096)
        model.save("ppo_drone_racing")

def create_waypoints(quadrotor_config: dict):
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
    x_goal[:,2] = ref_y
    x_goal[:,4] = ref_z
    return x_goal
if __name__ == "__main__":
    fire.Fire(main)
