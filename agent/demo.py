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


from agent.gym_A1.envs.a1_env_v2_4 import A1_env_v1

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from collections import namedtuple
from agent.utils.lr_scheduler import LRScheduler

from typing import Callable

args = {}
args['gravity'] = 1
args['distance'] = 1
args['linear'] = 1
args['angular'] = 1
# args['g_weight'] = trial.suggest_int("rollouts", 5000, 20000, step=5000)
# args['d_weight'] = trial.suggest_int("num_epochs", 1, 19, step=3)
# args['l_weight'] = trial.suggest_categorical("net_depth", ['net_small', 'net_med', 'net_large'])
# args['l_weight'] = trial.suggest_categorical("net_depth", ['net_small', 'net_med', 'net_large'])

args['terrain_difficulty'] = 0.04
args['target_distance'] = 2.5

args = namedtuple("ObjectName", args.keys())(*args.values())


env_config = dict(
        usePhaseObs = True,
        useContactStates = True,
        # useTargetTrajectory = True,
        terrain_type='uniform',  # initial terrain overide; flat, uniform, perlin, gap, bump, None
        terrain_probability="fixed",  # uniform, fixed

        action_upper_bound= .05,
        action_lower_bound=-.05,
        alpha=0.1,  #0.1 init (0, 1] - 1=no lpf, alpha->0 old action
        visionEnabled=False,
        rayLength=0.8,
        vision_dim=(28, 28),
        enable_rendering=1,  # 0: disabled, 1: render, 2: target + direction
        enable_recording=0,  # 0: disabled, 1: built-in render, 2: moviepy
        enable_rays=0,  # 0: disabled, 1: balls, 2: balls + rays

        MPC_warmup_time=0.,
        max_time_limit=60,  # secs, in realtime

        autoencoder_filename="lidar_28x28_0.8m_clipped.pth",
        kernel_dir = "walking_cmd/checkpoint_best.pt",
        reward_params = args,
        terrain_difficulty = args.terrain_difficulty,
        target_distance = args.target_distance
    )

eval_config = deepcopy(env_config)
eval_config["enable_rendering"] = 1

policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                        net_arch=[256,256,256,dict(pi=[128], vf=[64])])

env = A1_env_v1(**env_config)
env = Monitor(env, './results/env_logs', info_keywords=("is_success",) )

lr_sched = LRScheduler(1e-8, 0.001, 0.001)

model = PPO.load("./optimization/results/model_rewards_confirmation/0_215/best_model.zip", custom_objects={'learning_rate':lr_sched.schedule(), 'lr_schedule':lr_sched})

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
