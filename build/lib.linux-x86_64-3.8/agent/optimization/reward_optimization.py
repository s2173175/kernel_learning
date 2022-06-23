from __future__ import absolute_import
from __future__ import division
#from __future__ import google_type_annotations
from __future__ import print_function
from distutils import command
from distutils.log import debug


from copy import deepcopy
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




def main(trial):

    args = {}
    args['gravity'] = trial.suggest_float("gravity", 1,4)
    args['distance'] = trial.suggest_float("distance", 1,4)
    args['linear'] = trial.suggest_float("linear",  15, 20)
    args['agular'] = trial.suggest_float("angular", 5,10)
    # args['g_weight'] = trial.suggest_int("rollouts", 5000, 20000, step=5000)
    # args['d_weight'] = trial.suggest_int("num_epochs", 1, 19, step=3)
    # args['l_weight'] = trial.suggest_categorical("net_depth", ['net_small', 'net_med', 'net_large'])
    # args['l_weight'] = trial.suggest_categorical("net_depth", ['net_small', 'net_med', 'net_large'])
 
    args['env_dificulty'] = 0.04
    args['target_distance'] = 2

    args['lr'] = 0.001
    args['lr_decay'] = 1e-7
    args['entropy_coef'] = 0.000015
    args['batch_size'] = 2000
    args['roll_length'] = 20000
    args['num_epochs'] = 10


    args = namedtuple("ObjectName", args.keys())(*args.values())

 



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
            # visionEnabled=False,
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
            terrain_difficulty = args.env_dificulty,
            target_distance = args.target_distance
            

        )

  
    policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                        net_arch=[256,256,256,dict(pi=[128], vf=[64])])


    num_cpu = 1

    # env = SubprocVecEnv([make_env(env_config, i) for i in range(num_cpu)])
    env = gym.make("gym_A1:A1_all_terrains-v24", **env_config)
    env = Monitor(env)
    

    eval_callback = EvalCallback(env, best_model_save_path=f'./results/model_reward/trial_{trial._trial_id}/',
                                log_path='./results/logs/', eval_freq=5e5/num_cpu, # eg every 100,000 per cpu
                                deterministic=True, render=False, n_eval_episodes = 5)

    output = {'mean_target_count':0}                       
    callback = CallbackList([eval_callback, TensorboardCallback(trial, args, output, num_procs=num_cpu)])

    lr_sched = LRScheduler(1e-8, args.lr, args.lr_decay)


    model = PPO("MlpPolicy", 
                env, 
                policy_kwargs=policy_kwargs, 
                verbose=1, 
                tensorboard_log="./results/reward_opt/", 
                learning_rate=lr_sched.schedule(), 
                batch_size=int(args.batch_size), 
                n_steps =int(args.roll_length/num_cpu), 
                n_epochs=args.num_epochs, 
                ent_coef=args.entropy_coef,
                device='cuda')

    model.learn(total_timesteps=3e6, tb_log_name=f'{trial._trial_id}_id', callback=callback)

    print(output['mean_target_count'])
    return output['mean_target_count'] #target_count and rollout reward

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
        
        env = gym.make("gym_A1:A1_all_terrains-v2-4", **env_config)
        env = Monitor(env)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


if __name__ == '__main__':

    storage = "sqlite:///demo.db"
    study = optuna.create_study(storage=storage, directions=["maximize"])

    with parallel_backend('multiprocessing'):  # Overrides `prefer="threads"` to use multi-processing.
        study.optimize(main, n_trials=100, n_jobs=1)


    study.best_params 

