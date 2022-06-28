from __future__ import absolute_import
from __future__ import division
#from __future__ import google_type_annotations
from __future__ import print_function
from distutils import command
from distutils.log import debug
from distutils.util import strtobool

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
from env_configs import ENV_CONFIGS


def get_label(reward_params):

    s = f'{reward_params.id}_{reward_params.seed}'
    return s

def main(args, env_config):

    label = get_label(args)

    env_config = dict(
            usePhaseObs = args.use_phase,
            useTargetTrajectories = args.use_targets,
            useContactStates = args.use_contacts,
            visionEnabled=args.use_vision,

            action_upper_bound= .05,
            action_lower_bound=-.05,
            alpha=0.1,  #0.1 init (0, 1] - 1=no lpf, alpha->0 old action
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
         
            enable_external_force = env_config.enable_external_force,
            force_range = env_config.force_range,
            force_frequency = env_config.force_frequency,
            target_distance = env_config.target_distance,
            terrain_probability = env_config.terrain_probability,
            terrain_uniform_range = env_config.terrain_uniform_range,
            perlin_params = env_config.perlin_params,

        )

  
    num_cpu = 5
    output = {'mean_target_count':0}                       
    policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                        net_arch=[256,256,256,dict(pi=[128], vf=[64])])


    if args.train:
        env = SubprocVecEnv([make_env(env_config, i) for i in range(num_cpu)])   

        eval_callback = EvalCallback(env, best_model_save_path=f'./results/{args.save_dir}/{label}/',
                                    log_path='./results/logs/', eval_freq=5e5/num_cpu, # eg every 100,000 per cpu
                                    deterministic=True, render=False, n_eval_episodes = 5)

        callback = CallbackList([eval_callback, TensorboardCallback(None, args, output, num_procs=num_cpu)])

        lr_sched = LRScheduler(1e-8, args.lr, args.lr_decay)

        model = PPO("MlpPolicy", 
                    env, 
                    policy_kwargs=policy_kwargs, 
                    verbose=1, 
                    tensorboard_log=f'./results/{args.log_dir}/', 
                    learning_rate=lr_sched.schedule(), 
                    batch_size=int(args.batch_size), 
                    n_steps =int(args.rollout_length/num_cpu), 
                    n_epochs=args.num_epochs, 
                    ent_coef=args.entropy_coef,
                    seed=args.seed,
                    device='cuda')

        model.learn(total_timesteps=args.max_timesteps, tb_log_name=f'{label}', callback=callback)

    elif args.eval:
         
        if args.model == 'rl':
            model = PPO.load(f'./results/results/{args.model_dir}/best_model.zip', 
                        custom_objects={'learning_rate':lr_sched.schedule(), 
                        'lr_schedule':lr_sched})

            env = SubprocVecEnv([make_env(env_config, i) for i in range(num_cpu)])   
            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)
        pass

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
        
        env = gym.make("gym_A1:A1_multi_task_training-v1", **env_config)
        env = Monitor(env)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


if __name__ == '__main__':

    parser = argparse.ArgumentParser()  
    parser.add_argument('--id', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--save_dir', type=str)

    parser.add_argument('--gravity', type=float)
    parser.add_argument('--distance', type=float)
    parser.add_argument('--linear', type=float)
    parser.add_argument('--angular', type=float)

    parser.add_argument('--lr', type=float)
    parser.add_argument('--lr_decay', type=float)
    parser.add_argument('--entropy_coef', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--rollout_length', type=int)
    parser.add_argument('--num_epochs', type=int)

    parser.add_argument('--use_contacts',type=lambda x:bool(strtobool(x)),nargs='?', const=True, default=False)
    parser.add_argument('--use_targets', type=lambda x:bool(strtobool(x)),nargs='?', const=True, default=False)
    parser.add_argument('--use_phase',  type=lambda x:bool(strtobool(x)),nargs='?', const=True, default=False)
    parser.add_argument('--use_vision',  type=lambda x:bool(strtobool(x)),nargs='?', const=True, default=False)

    parser.add_argument('--max_timesteps', type=int)

    parser.add_argument('--env', type=str)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--model', default='rl', type=str, choices=['mpc', 'kernel', 'rl'])
    parser.add_argument('--model_dir', default=None, type=str)


    args = parser.parse_args()
    env_config = ENV_CONFIGS[args.env]

    print("args: ",type(args))
    print("args: ",type(env_config))
    print("args: ",vars(args))
    print("env config: ",env_config._asdict())
  


    main(args, env_config)

    
