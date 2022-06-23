from __future__ import absolute_import
from __future__ import division
#from __future__ import google_type_annotations
from __future__ import print_function
from distutils import command
from distutils.log import debug

import argparse
from copy import deepcopy

import numpy as np
import os
import random
import time

import torch 



import os
import inspect
from urllib import robotparser

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
# from gym_A1.envs.a1_env_v1 import A1_env_v1
from gym_A1.envs.a1_env_v2_3 import A1_env_v1
from sb3_callbacks import VideoRecorderCallback, TensorboardCallback
import gym 

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_checker import check_env

from typing import Callable


def make_env(env_config, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        
        env = A1_env_v1(**env_config)
        env = Monitor(env, './results/env_logs')
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    _min = 1e-9
    start = initial_value
    decay = 2e-7
    total_steps = 20e6

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
    
        step = int(total_steps - progress_remaining*total_steps)
        return _min + (start - _min) * np.exp(-decay * step)

    return func


def main():

    env_config = dict(
            usePhaseObs = True,
            # useTargetTrajectory = True,
            useContactStates = True,
            terrain_type='uniform',  # initial terrain overide; flat, uniform, perlin, gap, bump, None
            terrain_probability="fixed",  # uniform, fixed

            action_upper_bound= .05,
            action_lower_bound=-.05,
            alpha=0.1,  #0.1 init (0, 1] - 1=no lpf, alpha->0 old action
            visionEnabled=False,
            rayLength=0.8,
            vision_dim=(28, 28),
            enable_rendering=0,  # 0: disabled, 1: render, 2: target + direction
            enable_recording=0,  # 0: disabled, 1: built-in render, 2: moviepy
            enable_rays=0,  # 0: disabled, 1: balls, 2: balls + rays

            MPC_warmup_time=0.,
            max_time_limit=60,  # secs, in realtime

            autoencoder_filename="lidar_28x28_0.8m_clipped.pth",
            kernel_dir = "../kernel/results/walking_cmd/checkpoint_best.pt"
        )

    eval_config = deepcopy(env_config)
    eval_config["enable_rendering"] = 0

    eval_env = A1_env_v1(**eval_config)
    eval_env = Monitor(eval_env, './results/eval_env_logs')


    policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                        net_arch=[dict(pi=[256, 256, 256], vf=[256, 256, 256])])


    num_cpu = 5  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_config, i) for i in range(num_cpu)])

    eval_callback = EvalCallback(env, best_model_save_path='./results/model/v2_3/',
                                log_path='./results/logs/', eval_freq=5e5/num_cpu, # eg every 100,000  
                                deterministic=True, render=False, n_eval_episodes = 5)
                                
    callback = CallbackList([eval_callback, TensorboardCallback(num_procs=num_cpu)])

    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./results/reward_shaping/", learning_rate=linear_schedule(0.0001), batch_size=int(5e3), n_steps =int(25e3/num_cpu), n_epochs=200, device='cuda')

    model.learn(total_timesteps=10e6, tb_log_name="v2_3", callback=callback)

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5, deterministic=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    #TODO check basevel also may be good to add angular vel into obs




if __name__ == '__main__':
    main()