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


from lsy_drone_racing.constants import FIRMWARE_FREQ
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.wrapper import DroneRacingWrapper, MultiProcessingWrapper, ActionWrapper

logger = logging.getLogger(__name__)


def create_race_env(config_path: Path, gui: bool = False, random_train=False) -> DroneRacingWrapper:
    """Utility function for multiprocessed env.
    Args:
        config_path (Path): Path to the configuration file.
        gui (bool): Whether to show the GUI.
        random_train (bool): Whether to randomize the training environment.
    Returns:
        DroneRacingWrapper: The drone racing environment.
    """
    # Create the drone racing environment
    # Load configuration and check if firmare should be used
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
    drone_env = DroneRacingWrapper(firmware_env, terminate_on_lap=True, train_random_state=random_train)
    return ActionWrapper(drone_env)


def main(config: str = "config/level2_train.yaml", gui: bool = False):
    """Create the environment, check its compatibility with sb3, and run a PPO agent.
    Args:
        config (str): Path to the configuration file.
        gui (bool): Whether to show the GUI.
    """

    logging.basicConfig(level=logging.INFO)
    config_path = Path(__file__).resolve().parents[1] / config # resolve() returns the absolute path, parents[1] /config adds the config
    ## Training parameters
    PROCESSES_TO_TEST = 4 # Number of vectorized environments to train
    NUM_EXPERIMENTS = 1  # RL algorithms can often be unstable, so we run several experiments (see https://arxiv.org/abs/1709.06560)
    TRAIN_STEPS = 2**19  # Number of training steps
    EVAL_EPS = 5 # Number of episodes for evaluation
    ALGO = PPO
    n_steps = 2**10 # Number of steps to run for each environment per update
    batch_size = n_steps // 2**4
    ## Create Environments
    load_model = False ## Load a model from a previous training session
    if_validate = True ## Validate the model during training
    random_train = False ## Randomize the initialization of the training environment
    train_env = create_race_env(config_path=config_path, gui=gui, random_train=random_train)
    check_env(train_env)
    if PROCESSES_TO_TEST > 1:
        train_env = MultiProcessingWrapper(train_env)
        vec_train_env = make_vec_env(lambda: MultiProcessingWrapper(create_race_env(config_path=config_path, gui=gui, random_train=random_train)),
                                     n_envs=PROCESSES_TO_TEST, vec_env_cls=SubprocVecEnv)
        train_env = vec_train_env
    k = 1 # The learning iteration
    ## Save and Load Paths
    save_path = './models'
    save_name = '/ppo_lvl2_5_5sgate_iter' + str(k)
    load_path = save_path
    load_name = '/ppo_lvl2_5_5sgate_iter' + str(k-1) + '.zip'
    tb_log_name = save_name.split('/')[-1]
    ## Checkpoint and Evaluation Callbacks for saving and evaluating the model
    checkpoint_callback = CheckpointCallback(save_freq=2**15, save_path=save_path+save_name,
                                         name_prefix='rl_model')
    if if_validate:
        eval_env = create_race_env(config_path=config_path, gui=gui)
        check_env(eval_env)
        eval_callback = EvalCallback(eval_env, best_model_save_path=save_path+save_name+'_best',
                                 log_path='./logs/', eval_freq=2**14, deterministic=True, render=False)
    callback_list = [checkpoint_callback, eval_callback] if if_validate else [checkpoint_callback]
    callback_list = CallbackList(callback_list)
    ## Whether to load a model or create a new one
    if not load_model:
        print(f'Creating model...')
        model = ALGO("MlpPolicy", train_env, verbose=1, tensorboard_log="./logs", n_steps=n_steps,
                    learning_rate=0.0003, ent_coef=0.02, device='auto', n_epochs=10, batch_size=batch_size,
                    clip_range=0.2, gae_lambda=0.95)
    else:
        print(f'Loading model from {load_path+load_name}')
        model = ALGO.load(load_path+load_name, env=train_env)

    print(f'Starting experiment...')
    print(f'Log Name: {tb_log_name}')
    model.learn(total_timesteps=TRAIN_STEPS, progress_bar=True, tb_log_name=tb_log_name, callback=callback_list)
    model.save(save_path+save_name)
    train_env.close()
    if if_validate:
        eval_env.close()

if __name__ == "__main__":
    fire.Fire(main)
