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
    env_factory = partial(make, "quadrotor",**config.quadrotor_config)
    firmware_env = make("firmware", env_factory, FIRMWARE_FREQ, CTRL_FREQ)
    inc_gate_obs = config.quadrotor_config["inc_gate_obs"]
    # goal state is of shape (N,12), where N is the number of waypoints and 12 is the states of the drone
    # x,dx,y,dy,z,dz,phi,theta,psi,p,q,r
    # We need to define something for the missing states since we only have x,y,z
    # Obey the order of the states
    return DroneRacingWrapper(firmware_env, terminate_on_lap=True, train_random_state=True, inc_gate_obs=inc_gate_obs)


def main(config: str = "config/getting_started_train.yaml", gui: bool = False):
    """Create the environment, check its compatibility with sb3, and run a PPO agent."""

    logging.basicConfig(level=logging.INFO)
    config_path = Path(__file__).resolve().parents[1] / config # resolve() returns the absolute path, parents[1] /config adds the config
    PROCESSES_TO_TEST = 1 # Number of vectorized environments to train
    NUM_EXPERIMENTS = 1  # RL algorithms can often be unstable, so we run several experiments (see https://arxiv.org/abs/1709.06560)
    TRAIN_STEPS = 2**15  # Number of training steps
    EVAL_EPS = 5 # Number of episodes for evaluation
    ALGO = PPO
    if_validate = False
    reward_averages = []
    reward_std = []
    training_times = []
    train_env = create_race_env(config_path=config_path, gui=gui)
    check_env(train_env)
    vec_train_env = make_vec_env(lambda: create_race_env(config_path=config_path, gui=gui), n_envs=PROCESSES_TO_TEST)
    train_env = vec_train_env
    if if_validate:
        eval_env = create_race_env(config_path=config_path, gui=gui)
        check_env(eval_env)
    rewards = []
    times = []
    for experiment in range(NUM_EXPERIMENTS):
        # it is recommended to run several experiments due to variability in results
        n_steps = 2**10
        batch_size = n_steps * PROCESSES_TO_TEST // 2
        model = ALGO("MlpPolicy", train_env, verbose=1, tensorboard_log="./logs", n_steps=n_steps,
                     learning_rate=0.0003, ent_coef=0.01, device='auto', n_epochs=10, batch_size=batch_size,
                     clip_range=0.1, gae_lambda=0.95)
        start = time.time()
        model.learn(total_timesteps=TRAIN_STEPS, progress_bar=False, tb_log_name='./logs')
        times.append(time.time() - start)
        if if_validate:
            mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPS)
            rewards.append(mean_reward)
    model.save("ppo_gaus_random2")
    train_env.close()
    if if_validate:
        eval_env.close()
        reward_averages.append(np.mean(rewards))
        reward_std.append(np.std(rewards))
        training_times.append(np.mean(times))

if __name__ == "__main__":
    fire.Fire(main)
