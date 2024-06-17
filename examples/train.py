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
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

from scipy import interpolate

from lsy_drone_racing.constants import FIRMWARE_FREQ
from lsy_drone_racing.utils import load_config, load_controller
from lsy_drone_racing.wrapper import DroneRacingWrapper, MultiProcessingWrapper

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
    ## Training parameters
    PROCESSES_TO_TEST = 2 # Number of vectorized environments to train
    NUM_EXPERIMENTS = 1  # RL algorithms can often be unstable, so we run several experiments (see https://arxiv.org/abs/1709.06560)
    TRAIN_STEPS = 2**18  # Number of training steps
    EVAL_EPS = 5 # Number of episodes for evaluation
    ALGO = PPO
    n_steps = 2**11
    batch_size = n_steps * PROCESSES_TO_TEST // 2
    ## Create Environments
    load_model = False
    if_validate = True
    train_env = create_race_env(config_path=config_path, gui=gui)
    check_env(train_env)
    if PROCESSES_TO_TEST > 1:
        train_env = MultiProcessingWrapper(train_env)
        vec_train_env = make_vec_env(lambda: MultiProcessingWrapper(create_race_env(config_path=config_path, gui=gui)),
                                     n_envs=PROCESSES_TO_TEST, vec_env_cls=SubprocVecEnv)
        train_env = vec_train_env
    k = 1 # The learning iteration
    save_path = './models/'
    save_name = './ppo_gaus_random' + str(k)
    load_path = save_path
    load_name = './ppo_gaus_random' + str(k-1) + '.zip'
    tb_log_name = './PPO_GREW_' + str(k)
    checkpoint_callback = CheckpointCallback(save_freq=2**11, save_path=save_path+save_name,
                                         name_prefix='rl_model')
    if if_validate:
        eval_env = create_race_env(config_path=config_path, gui=gui)
        check_env(eval_env)
        eval_callback = EvalCallback(eval_env, best_model_save_path=save_path+save_name+'_best',
                                 log_path='./logs/', eval_freq=2**14, deterministic=True, render=False)
    callback_list = [checkpoint_callback, eval_callback] if if_validate else [checkpoint_callback]
    callback_list = CallbackList(callback_list)
    for experiment in range(NUM_EXPERIMENTS):
        # it is recommended to run several experiments due to variability in results
        if not load_model:
            print(f'Creating model...')
            model = ALGO("MlpPolicy", train_env, verbose=1, tensorboard_log="./logs", n_steps=n_steps,
                        learning_rate=0.0003, ent_coef=0.01, device='auto', n_epochs=10, batch_size=batch_size,
                        clip_range=0.1, gae_lambda=0.95)
        else:
            print(f'Loading model from {load_path+load_name}')
            model = ALGO.load(load_path+load_name, env=train_env)
        print(f'Starting experiment...')
        print(f'Log Name: {tb_log_name}')
        model.learn(total_timesteps=TRAIN_STEPS, progress_bar=True, tb_log_name=tb_log_name, callback=callback_list)
        # if if_validate:
        #     mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPS)
    model.save(save_path+save_name)
    train_env.close()
    if if_validate:
        eval_env.close()

if __name__ == "__main__":
    fire.Fire(main)
