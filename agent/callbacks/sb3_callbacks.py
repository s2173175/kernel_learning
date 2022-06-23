from stable_baselines3.common.callbacks import BaseCallback

# check list

# tensorboard logging and view 
# periodic evaluation - to tensor board


# TENSOR BOARD CALL BACK 
import gym
import torch as th
import time
import numpy as np
from typing import Any, Dict
import random
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
from stable_baselines3.common.logger import TensorBoardOutputFormat

import optuna


class TensorboardCallback(BaseCallback):
    #TODO test multiple cpus
    """
    Custom callback for plotting additional values in tensorboard.
    """



    def __init__(self, trial, opt_args, output, verbose=0, num_procs=1):
        super(TensorboardCallback, self).__init__(verbose)
        self.opt_args = opt_args
        self.trial = trial
        self.output = output
        self.num_procs = num_procs
        self.successes = []
        self.gravity_ratios = []
        self.angular_ratios = []
        self.linear_ratios = []
        self.target_counts = []
        self.episode_rewards = []
        self.episode_count = 0
        self.start_time = time.time()
        

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        output_formats = self.logger.output_formats
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))
      
        with self.tb_formatter.writer as w:
            w.add_hparams(dict(self.opt_args._asdict()),{'loss':11}, run_name='.')
            w.flush()

        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        if len(self.episode_rewards)>30:
            self.episode_rewards = self.episode_rewards[-30:]
        
        self.output['mean_target_count'] = sum(self.target_counts)/len(self.target_counts)
        pass

    def calc_reward_dist(self, dist):
        # {"gravity":0, "angular":0, "linear":0, "total":0}
        self.gravity_ratios.append(dist['gravity'])
        self.angular_ratios.append(dist['angular'])
        self.linear_ratios.append(dist['linear'])

        return

    def _on_step(self) -> bool:
        #  {"is_success": is_success,"universal_reward_measure": self.universal_reward, "target_count": self.num_targets_reached, "reward_dist":self.reward_proportions}
       
        for i in range(self.num_procs):
            if(self.locals['dones'][i] == True):
                self.successes.append(1 if self.locals['infos'][i]['is_success'] == True else 0)
                self.target_counts.append(self.locals['infos'][i]['target_count'])
                self.episode_rewards.append(self.locals['infos'][i]['reward_dist']['total'])
                self.calc_reward_dist(self.locals['infos'][i]['reward_dist'])
                self.episode_count += 1
               
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        length = len(self.successes)
        assert length  == len(self.gravity_ratios) == len(self.angular_ratios) == len(self.linear_ratios) == len(self.target_counts)

        if length > 30:
            self.successes = self.successes[-30:]
            self.gravity_ratios = self.gravity_ratios[-30:]
            self.angular_ratios = self.angular_ratios[-30:]
            self.linear_ratios = self.linear_ratios[-30:]
            self.target_counts = self.target_counts[-30:]
            self.episode_rewards = self.episode_rewards[-30:]

        length = len(self.successes)

        if length > 0:
            self.logger.record('rollout/success_rate', sum(self.successes)/length)
            self.logger.record('rollout/gravity', sum(self.gravity_ratios)/length)
            self.logger.record('rollout/angular', sum(self.angular_ratios)/length)
            self.logger.record('rollout/linear', sum(self.linear_ratios)/length)
            self.logger.record('rollout/target_count', sum(self.target_counts)/length)
            self.logger.record('rollout/mean_reward', sum(self.episode_rewards)/length)

            self.trial.report(sum(self.target_counts)/length, self.num_timesteps)

            # Handle pruning based on the intermediate value.
            # if self.trial.should_prune():
            #     raise optuna.exceptions.TrialPruned()

        else:
            self.logger.record('rollout/success_rate', 0)
            self.logger.record('rollout/gravity', 0)
            self.logger.record('rollout/angular', 0)
            self.logger.record('rollout/linear', 0)
            self.logger.record('rollout/target_count', 0)
            self.logger.record('rollout/mean_reward', 0)

        self.logger.record('rollout/time_elapsed', time.time()-self.start_time)



        pass


class TimeCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0):
        super(TimeCallback, self).__init__(verbose)

        self.update_start = 0
        self.update_end = 0
        self.update_time = []

        self.rollout_start = 0
        self.rollout_end = 0
        self.rollout_time = []


    def _on_step(self) -> bool:
        
        return True

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        self.update_end = time.time()
        self.update_time.append(self.update_start - self.update_end)
        print("updates: ", self.update_time)

        self.rollout_start = time.time()

        return

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """

        self.rollout_end = time.time()
        self.rollout_time.append(self.rollout_end - self.rollout_start)
        print("rollouts: ", self.rollout_time)



        self.update_start = time.time()
        return



class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                screen = self._eval_env.render(mode="rgb_array")
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            self.logger.record(
                "trajectory/video",
                Video(th.ByteTensor([screens]), fps=40),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True