from __future__ import absolute_import
from __future__ import division
#from __future__ import google_type_annotations
from __future__ import print_function
from distutils import command
from distutils.log import debug


from copy import deepcopy
import re
from joblib import parallel_backend

import numpy as np
import os
import random
import time
from typing import Callable

import torch 

import os
import inspect
from urllib import robotparser



currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import gym 
import argparse
from collections import namedtuple
import optuna

import gc
gc.enable()


from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_checker import check_env
from callbacks.sb3_callbacks import TensorboardCallback
from utils.lr_scheduler import LRScheduler


def get_label(hp_params):

    s = f'{hp_params.id}_{reward_params.seed}'
    return s

def main(hp_params):

    # parser = argparse.ArgumentParser()  
    # parser.add_argument('--log_dir', type=str)
    # parser.add_argument('--save_dir', type=str)

    # log_args = parser.parse_args()

    args = {'angular': 7.47, 'distance': 0.74, 'gravity': 2.35, 'linear': 18.42}

    args['terrain_difficulty'] = 0.03
    args['target_distance'] = 2.5


    args['lr'] = hp_params.lr
    args['lr_decay'] = hp_params.lr_decay
    args['entropy_coef'] = hp_params.entropy_coef
    args['batch_size'] = hp_params.batch_size
    args['roll_length'] = hp_params.roll_length
    args['num_epochs'] = hp_params.num_epochs
    args['network'] = hp_params.network
 

    args = namedtuple("ObjectName", args.keys())(*args.values())

 

    networks = {
        'large':dict(activation_fn=torch.nn.Tanh, net_arch=[256,256,256,dict(pi=[128], vf=[64])]),
        'medium':dict(activation_fn=torch.nn.Tanh, net_arch=[256,256,dict(pi=[128], vf=[64])]),
        'small':dict(activation_fn=torch.nn.Tanh, net_arch=[128,128,dict(pi=[128], vf=[64])]),
    }

    label = get_label(hp_params)

 
    #### run experiments


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
            kernel_dir = "walking_cmd/checkpoint_best.pt",
            reward_params = args,
            terrain_difficulty = args.terrain_difficulty,
            target_distance = args.target_distance
        )

  
    policy_kwargs = networks[args.network]

    num_cpu = 5

    env = SubprocVecEnv([make_env(env_config, i) for i in range(num_cpu)])
    # env = gym.make("gym_A1:A1_all_terrains-v24", **env_config)
    # env = Monitor(env)
    

    eval_callback = EvalCallback(env, best_model_save_path=f'./results/hp_confirmation_models/{label}/',
                                log_path='./results/logs/', eval_freq=5e5/num_cpu, # eg every 100,000 per cpu
                                deterministic=True, render=False, n_eval_episodes = 5)

    output = {'mean_target_count':0}                       
    callback = CallbackList([eval_callback, TensorboardCallback(None, args, output, num_procs=num_cpu)])

    lr_sched = LRScheduler(1e-8, args.lr, args.lr_decay)


    model = PPO("MlpPolicy", 
                env, 
                policy_kwargs=policy_kwargs, 
                verbose=1, 
                tensorboard_log="./results/hp_confirmation/", 
                learning_rate=lr_sched.schedule(), 
                batch_size=int(args.batch_size), 
                n_steps =int(args.roll_length/num_cpu), 
                n_epochs=args.num_epochs, 
                ent_coef=args.entropy_coef,
                seed=reward_params.seed,
                device='cuda')

    model.learn(total_timesteps=5e6, tb_log_name=f'{label}', callback=callback)

    print(output['mean_target_count'])
    return output['mean_target_count']

    #TODO save each policy under a different name




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
        
        env = gym.make("gym_A1:A1_all_terrains-v24", **env_config)
        env = Monitor(env)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


if __name__ == '__main__':


    parser = argparse.ArgumentParser()  
    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_decay', type=float)
    parser.add_argument('--entropy_coef', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--roll_length', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--network', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--id', type=int)

    reward_params = parser.parse_args()

    print(reward_params)

    main(reward_params)

    
