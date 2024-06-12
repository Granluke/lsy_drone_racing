"""Example training script using the stable-baselines3 library.

Note:
    This script requires you to install the stable-baselines3 library.
"""

from __future__ import annotations

from copy import deepcopy
import logging
from functools import partial
from pathlib import Path
import time

import numpy as np
import fire
from safe_control_gym.utils.registration import make

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from scipy import interpolate

from lsy_drone_racing.constants import FIRMWARE_FREQ
from lsy_drone_racing.utils import load_config, load_controller
from lsy_drone_racing.wrapper import DroneRacingWrapper, RewardWrapper

logger = logging.getLogger(__name__)


def create_race_env(config_path: Path, gui: bool = False) -> DroneRacingWrapper:
    """Utility function for multiprocessed env."""
    
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
    # firmware_env = RewardWrapper(firmware_env)
    # goal state is of shape (N,12), where N is the number of waypoints and 12 is the states of the drone
    # x,dx,y,dy,z,dz,phi,theta,psi,p,q,r
    # We need to define something for the missing states since we only have x,y,z
    # Obey the order of the states
    return DroneRacingWrapper(firmware_env, terminate_on_lap=True)


def main(config: str = "config/getting_started_train.yaml", gui: bool = False):
    """Create the environment, check its compatibility with sb3, and run a PPO agent."""

    logging.basicConfig(level=logging.INFO)
    config_path = Path(__file__).resolve().parents[1] / config # resolve() returns the absolute path, parents[1] /config adds the config
    PROCESSES_TO_TEST = 1 # Number of vectorized environments to train
    NUM_EXPERIMENTS = 1  # RL algorithms can often be unstable, so we run several experiments (see https://arxiv.org/abs/1709.06560)
    TRAIN_STEPS = 10000  # Number of training steps
    EVAL_EPS = 5 # Number of episodes for evaluation
    ALGO = PPO
    if_validate = False
    reward_averages = []
    reward_std = []
    training_times = []
    train_env = create_race_env(config_path=config_path, gui=gui)
    check_env(train_env)
    vec_train_env = make_vec_env(lambda: create_race_env(config_path=config_path, gui=gui), n_envs=PROCESSES_TO_TEST)
    # check_env(vec_train_env)
    train_env = vec_train_env
    eval_env = create_race_env(config_path=config_path, gui=gui)
    check_env(eval_env)
    rewards = []
    times = []
    for experiment in range(NUM_EXPERIMENTS):
        # it is recommended to run several experiments due to variability in results
        obs = train_env.reset()[0]
        model = ALGO("MlpPolicy", train_env, verbose=1, tensorboard_log="./logs", n_steps=TRAIN_STEPS,
                     learning_rate=0.0003, ent_coef=0.01, device='auto', n_epochs=10)
        start = time.time()
        model.learn(total_timesteps=TRAIN_STEPS, progress_bar=True, tb_log_name='./logs')
        times.append(time.time() - start)
        if if_validate:
            mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPS)
            rewards.append(mean_reward)
    model.save("ppo_drone_racing0003")
    train_env.close()
    eval_env.close()
    if if_validate:
        reward_averages.append(np.mean(rewards))
        reward_std.append(np.std(rewards))
        training_times.append(np.mean(times))

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
    x_goal[:,1] = ref_y
    x_goal[:,2] = ref_z
    return x_goal


if __name__ == "__main__":
    fire.Fire(main)
