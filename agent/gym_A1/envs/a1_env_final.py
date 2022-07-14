from __future__ import absolute_import
from __future__ import division
#from __future__ import google_type_annotations
from __future__ import print_function
from distutils import command
from distutils.log import debug

import os
import inspect
from urllib import robotparser

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from absl import app
from absl import flags
import scipy.interpolate
import numpy as np
import pybullet_data as pd
from pybullet_utils import bullet_client

import time
import pybullet 
import random
import math
import collections

from mpc_controller import com_velocity_estimator
from mpc_controller import gait_generator as gait_generator_lib
from mpc_controller import locomotion_controller
from mpc_controller import openloop_gait_generator
from mpc_controller import raibert_swing_leg_controller
from mpc_controller import torque_stance_leg_controller

#uncomment the robot of choice
#from mpc_controller import laikago_sim as robot_sim
from mpc_controller import a1_sim as robot_sim

# from motion_imitation.envs.sensors import space_utils
# from motion_imitation.envs.sensors import sensor
# from motion_imitation.robots import robot_config
# from motion_imitation.envs import locomotion_gym_config
# from motion_imitation.utilities import pose3d
# from motion_imitation.utilities import motion_util

from pybullet_utils import transformations
from gym.utils import seeding
from gym import spaces
import gym

#TODO i have changed this but this may break it fix later
from gym_A1.envs.vision_v2 import Vision
from gym_A1.envs.vision_encoder import A1_vision_encoder as VisionEncoder
# from gym_A1.util.logger import Logger


from kernel.models.denseNN import DenseNN
import torch

import gc
gc.enable()

__cwd__ = os.path.realpath( os.path.join(os.getcwd(), os.path.dirname(__file__)))


FLAGS = flags.FLAGS
'''
Physicx: 1kHz
MPC: 200Hz
PD: 1kHz
'''


_STANCE_DURATION_SECONDS = [
    0.3
] * 4  # For faster trotting (v > 1.5 ms reduce this to 0.13s).


# Trotting
_DUTY_FACTOR = [0.6] * 4
_INIT_PHASE_FULL_CYCLE = [0.9, 0, 0, 0.9]
_MAX_TIME_SECONDS = 150

_INIT_LEG_STATE = (
    gait_generator_lib.LegState.SWING,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.SWING,
)


class A1_env_v1(gym.Env):
    """The gym environment for the locomotion tasks."""
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 100
    }

    def __init__(self,
                 
                 # terrain params -------------------------
                 enable_external_force = False,
                 force_range = (250,500),
                 force_frequency = (4,10), 
                 terrain_type=None,  # initial terrain overide; uniform, perlin
                 terrain_probability="uniform",
                 terrain_uniform_range = (0, 0.045),
                 target_distance = 2.5,
                 perlin_params = [
                    {'scale':180,'octaves':4, 'persistence':0.2},
                    {'scale':200,'octaves':6, 'persistence':0.3},
                 ],
          
                 time_step=0.001,
                 max_time_limit=50,

                 # action params ------------------------
                 action_upper_bound= 0.05,  # cm
                 action_lower_bound=-0.05,  # cm
                 alpha=0.02, 
                 reward_params = None,

                # state params ---------------------------
                 kernel_dir = None,
                 usePhaseObs = False,
                 useContactStates = False,
                 useTargetTrajectories = False,

                 visionEnabled=False,
                 rayLength=0.8,
                 vision_dim=(28, 28),
                 autoencoder_filename=None,
                 
                 # util params -----------------------
                 FoV_V_Max=np.deg2rad(120),
                 FoV_V_offset=np.deg2rad(0),
                 FoV_H_Max=np.deg2rad(110),
                 FoV_H_offset=np.deg2rad(0),
                 MPC_warmup_time = 0.0,
                 enable_rendering=0, # 0: disabled, 1: render, 2: target + direction
                 enable_recording=0,
                 enable_rays=0,  # 0: disabled, 1: balls, 2: balls + rays
                 video_name="test",
                 
                 
                 ):



        self.seed(10)

        config = {
            "learning_rate":1e-3,
            "dropout_prob":0.2,
            "l2":0,
            "max_epoch":20,
            "data_dir": ["./data/sets/dense_standing_walk3_x.csv", "./data/sets/dense_standing_walk3_y.csv"],
            "batch_size":100,
            "save_dir": "./kernel/results/q_q_dot",
            "log_file": "./kernel/results/q_q_dot/training_logs.out",
            "model_file": kernel_dir,
            "device": "cpu"
        }

        self.kernel = DenseNN(7, 12, (256,256,256), **config)
        self.kernel.load_model()
        self.kernel.eval()

        self.reward_params = reward_params
        self.perlin_params = perlin_params

        self.reward_proportions = {"gravity":0, "angular":0, "linear":0, "total":0}


        # Initialize internal variables
        self._world_dict = {}
        self._observation = None
        self._vision_input_size = 10
        self.terrain_uniform_range = terrain_uniform_range
        self.ground_data = None


        #state init
        self._usePhaseObs = usePhaseObs
        self._useContactStates = useContactStates
        self._useTargetTrajectories = useTargetTrajectories

        #force vars
        self.enable_force = enable_external_force
        self.force_range = force_range
        self.force_frequency = force_frequency
        self.force_last_applied = 0
        self.force_wait_time = np.random.uniform(force_frequency[0], force_frequency[1])

        
        # Set up Vision module
        self.visionEnabled = visionEnabled
        self._vision_dim = vision_dim
        self._raynum = self._vision_dim[0] * self._vision_dim[1]
        self.lidar = Vision(FoV_V_Max, FoV_V_offset, FoV_H_Max, FoV_H_offset, rayLength, vision_dim[0], vision_dim[1])

        self.lidar_offset = [0, 0, -0.07]
        self.lidar_balls = []
        self.GREENCOLOR = [0, 1, 0, 0.8]
        self.REDCOLOR = [1, 0, 0, 0.8]
        self.BLUECOLOR = [0, 0, 1, 0.8]

        # Set up VisionEncoder
        try: 
            assert autoencoder_filename, "Must provide autoencoder_filename"
            self.autoencoder_filename = autoencoder_filename
            self._vision_encoder = VisionEncoder(self.autoencoder_filename)  # "lidar_28x28_0.8m_clipped.pth"
        except:
            print("\nCouln't load Autoencoder!\n")

        ############################
        self.rendering_enabled = enable_rendering
        self.enable_rays = enable_rays
        self.record_video = enable_recording
        self._p = pybullet
        if self.record_video == 1:
            # self._p.connect(self._p.GUI, options="--width=1280 --height=720 --mp4=\"test.mp4\" --mp4fps=100")
            self._p.connect(self._p.GUI, options=f"--width=1280 --height=720 --mp4=\"{video_name}.mp4\" --mp4fps=25")
            # self._p.connect(self._p.GUI, options=f"--width=960 --height=720 --mp4=\"{video_name}.mp4\" --mp4fps=100")
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 0)
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI, 0)
        elif self.record_video == 0 and self.rendering_enabled:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        elif self.record_video == 2:
            # self._p.connect(self._p.GUI)
            self._p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
        else:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)

        self._p.setAdditionalSearchPath(pd.getDataPath())

        self._max_time_limit = max_time_limit
        self._time_step = time_step
        self.setupPhysicsParmeters()

        self.terrain_names = ["uniform", "perlin"]
        self.terrain_probability = terrain_probability
        self.terrain_type = None 
        
        self.loadTerrain(terrain_overide=terrain_type)

        self.loadRobot()
        self.joint_upper_bound = self._robot.joint_upper_bound
        self.joint_lower_bound = self._robot.joint_lower_bound

        # Start making observation space
        obs_upper_bound = np.concatenate((
            self.joint_upper_bound,     # 12 - GetJointAngles
            np.full((12, ), np.inf),    # 12 - GetJointVelocities
            np.full((3, ), 1.0),        # 3  - GetGravityVector
            np.array([np.pi/4]),          # 1  - angular speed command
            np.full((1, ), 0.5),     # 1  - forward speed command
            np.full((1, ), 0.2),     # 1  - side speed command
            np.full((3, ), 2.0),        # 3  - GetBaseVelocity
            np.full((3, ), 2.0),        # 3  - GetBaseAngularVelocity
            np.full((1, ), 1.0),        # 3  - max target distance
          
        ))
        obs_lower_bound = np.concatenate((
            self.joint_lower_bound,    # 12 - GetJointAngles
            np.full((12, ), -np.inf),  # 12 - GetJointVelocities
            np.full((3, ), -1.0),      # 3  - GetGravityVector
            np.array([-np.pi/4]),        # 1  - GetYawError
            np.full((1, ), -0.5),     # 1  - forward speed command
            np.full((1, ), 0.2),     # 1  - side speed command
            np.full((3, ), -2.0),      # 3  - GetBaseVelocity
            np.full((3, ), -2.0),        # 3  - GetBaseAngularVelocity
            np.full((1, ), 0.0),        # 3  - min target distance
        ))
        if self.visionEnabled:
            obs_upper_bound = np.concatenate((
                obs_upper_bound, np.full((10, ), np.inf)))    # 10 - GetVisionObservation
            obs_lower_bound = np.concatenate((
                obs_lower_bound, np.full((10, ), -np.inf)))  # 10 - GetVisionObservation

        if self._usePhaseObs:
            obs_upper_bound = np.concatenate((
                obs_upper_bound, np.full((4, ), 2.)))  # 1 - mpc leg phase
            obs_lower_bound = np.concatenate((
                obs_lower_bound, np.full((4, ), 0.)))  # 1 - mpc leg phase

        if self._useContactStates:
            obs_upper_bound = np.concatenate((
                obs_upper_bound, np.full((4, ), 1.0)))  # 1 - mpc leg phase
            obs_lower_bound = np.concatenate((
                obs_lower_bound, np.full((4, ), 0.0)))  # 1 - mpc leg phase

        if self._useTargetTrajectories:
            obs_upper_bound = np.concatenate((
                obs_upper_bound, np.full((12, ), 0.3)))  # 1 - mpc leg phase
            obs_lower_bound = np.concatenate((
                obs_lower_bound, np.full((12, ), -0.3)))

        
        obs_upper_bound = obs_upper_bound.astype(np.float32)
        obs_lower_bound = obs_lower_bound.astype(np.float32)

        
        self.observation_space = spaces.Box(low=obs_lower_bound.min(), high=obs_upper_bound.max(), shape=obs_upper_bound.shape, dtype=np.float32)
        
        # Start making action space
        # Old residual actions at joint level:
        # NOTE: New residual is 4x(x,y,z) foot trajectory residual
        self.action_upper_bound = action_upper_bound  # cm
        self.action_lower_bound = action_lower_bound  # cm

        self.action_space = spaces.Box(low=self.action_lower_bound, high=self.action_upper_bound, shape=(12,), dtype=np.float32)
  


        
        # reset viewpoint
        # Set the default render options.
        self._camera_dist = 1.5
        self._camera_yaw = 0
        # self._camera_yaw_offset = 25
        self._camera_pitch = -25
        self._render_width = 960 # 480
        self._render_height = 720 # 360
        if self.record_video:
            self._p.resetDebugVisualizerCamera(self._camera_dist, self._camera_yaw, self._camera_pitch, [0, 0, 0])

        ############################
        # Set up MPC controller
        self.gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
            self._robot,
            stance_duration=_STANCE_DURATION_SECONDS,
            duty_factor=_DUTY_FACTOR,
            initial_leg_phase=_INIT_PHASE_FULL_CYCLE,   
            initial_leg_state=_INIT_LEG_STATE)
        self.gait_generator.reset(0)

        self.alpha = alpha
        self._MPC_warmup_time = MPC_warmup_time
        self.resetResidualCommands()
        ############################
        # Set up the task
        self._target_reached_threshold = 0.5  # m
        self._minimum_target_distance = target_distance  # m
        self.target = self.generateNewTarget()  # this must go after loading the robot

        self.ball_urdf = __cwd__ + "/data/ball_vision.urdf"
        if self.record_video == 2 or self.rendering_enabled > 0:
            self.target_indicator = self._p.loadURDF(self.ball_urdf, basePosition=self.target, baseOrientation=[0, 0, 0, 1],
                                                     useFixedBase=True, globalScaling=0.5)
            self._p.changeVisualShape(self.target_indicator, -1, rgbaColor=self.GREENCOLOR)
        # ############################
        # Set up Vision visualisation
        # Initiallising the LiDAR
        lidar_points_from = [self.lidar_offset] * self._raynum
        lidar_points_to = self.lidar.getRayBatchLocal()
        results = self._p.rayTestBatch(lidar_points_from, lidar_points_to, parentObjectUniqueId=self._robot.quadruped, parentLinkIndex=-1)

        lidar_points_final = []
        for i in range(self._raynum):
            hitObjectUid = results[i][0]
            if (hitObjectUid < 0):
                lidar_points_final.append(lidar_points_to[i, :])
            else:
                hitPosition = results[i][3]
                lidar_points_final.append(hitPosition)

        self.lidar_balls = []
        if self.rendering_enabled or self.enable_rays > 0:
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 0)
            # Visualising the sheres
            # ball_urdf = self.dir_path + "ball_vision.urdf"
            for i in range(self._raynum):
                ball = self._p.loadURDF(self.ball_urdf, basePosition=lidar_points_final[i], baseOrientation=[0, 0, 0, 1],
                                  useFixedBase=True, globalScaling=0.10)

                # self._p.setCollisionFilterGroupMask(ball, -1, b'10101011', ) 
                for jointNo in range(self._p.getNumJoints(self._robot.quadruped)):
                    self._p.setCollisionFilterPair(ball, self._robot.quadruped, -1, jointNo, -1)  # disable ball-robot collision

                self._p.changeVisualShape(ball, -1, rgbaColor=self.REDCOLOR)
                self.lidar_balls.append(ball)

        # Lidar device indecator is always on
        base_pos, base_quat = self._p.getBasePositionAndOrientation(self._robot.quadruped)
        R = quat_to_rot(base_quat)
        lidar_origin_world = base_pos + R @ np.array(lidar_points_from[0])
        self.lidar_ball_origin = self._p.loadURDF(self.ball_urdf, basePosition=lidar_origin_world, baseOrientation=[0, 0, 0, 1],
                                            useFixedBase=True, globalScaling=0.35)
        self._p.changeVisualShape(self.lidar_ball_origin, -1, rgbaColor=self.BLUECOLOR)

        self.reset_num = 0
        self.reset()
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)
        # self._p.configureDebugVisualizer(self._p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

        self.tmp_time_start = 0
        self.tmp_time_end = 0
        

    def seed(self, seed=None):
        self.np_random, self.np_random_seed = seeding.np_random(seed)
        return [self.np_random_seed]

    def hard_reset_func(self, terrain_overide=None):
      
        self._p.resetSimulation()

        self._world_dict = {}
        self._observation = None
        # self._vision_input_size = 10
        self._env_step_counter = 0  # this attribute is not used
        ############################
        # Set up Vision module
        self.lidar_balls = []
        ############################
        # Set up VisionEncoder
        try:
            self._vision_encoder = VisionEncoder(self.autoencoder_filename)  # "lidar_28x28_0.8m_clipped.pth"
        except:
            print("\nCouln't load Autoencoder!\n")
        ############################
        self.setupPhysicsParmeters()

        self.loadTerrain(terrain_overide=terrain_overide)
        self.loadRobot()

        if self.record_video:
            self._p.resetDebugVisualizerCamera(self._camera_dist, self._camera_yaw, self._camera_pitch, [0, 0, 0])

        ############################
        # Set up MPC controller
        self.gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
            self._robot,
            stance_duration=_STANCE_DURATION_SECONDS,
            duty_factor=_DUTY_FACTOR,
            initial_leg_phase=_INIT_PHASE_FULL_CYCLE,   
            initial_leg_state=_INIT_LEG_STATE)
        self.gait_generator.reset(0)

        self.resetResidualCommands()
        # self._MPC_warmup_time = MPC_warmup_time
        ############################
        # Set up the task
        self._target_reached_threshold = 0.5  # m
        self._minimum_target_distance = 3.0  # m
        self.target = self.generateNewTarget()  # this must go after loading the robot
        ############################
        # Set up Vision visualisation
        # Initiallising the LiDAR
        lidar_points_from = [self.lidar_offset] * self._raynum
        lidar_points_to = self.lidar.getRayBatchLocal()
        results = self._p.rayTestBatch(lidar_points_from, lidar_points_to, parentObjectUniqueId=self._robot.quadruped, parentLinkIndex=-1)

        lidar_points_final = []
        for i in range(self._raynum):
            hitObjectUid = results[i][0]
            if (hitObjectUid < 0):
                lidar_points_final.append(lidar_points_to[i, :])
            else:
                hitPosition = results[i][3]
                lidar_points_final.append(hitPosition)

        self.ball_urdf = __cwd__ + "/data/ball_vision.urdf"
        self.lidar_balls = []
        if self.enable_rays > 0:
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 0)
            # Visualising the sheres
            # ball_urdf = self.dir_path + "ball_vision.urdf"
            for i in range(self._raynum):
                ball = self._p.loadURDF(self.ball_urdf, basePosition=lidar_points_final[i], baseOrientation=[0, 0, 0, 1],
                                        useFixedBase=True, globalScaling=0.10)

                # self._p.setCollisionFilterGroupMask(ball, -1, b'10101011', )
                for jointNo in range(self._p.getNumJoints(self._robot.quadruped)):
                    self._p.setCollisionFilterPair(ball, self._robot.quadruped, -1, jointNo, -1)  # disable ball-robot collision

                self._p.changeVisualShape(ball, -1, rgbaColor=self.REDCOLOR)
                self.lidar_balls.append(ball)

        # Lidar device indecator is always on
        base_pos, base_quat = self._p.getBasePositionAndOrientation(self._robot.quadruped)
        R = quat_to_rot(base_quat)
        lidar_origin_world = base_pos + R @ np.array(lidar_points_from[0])
        self.lidar_ball_origin = self._p.loadURDF(self.ball_urdf, basePosition=lidar_origin_world, baseOrientation=[0, 0, 0, 1],
                                                  useFixedBase=True, globalScaling=0.35)
        self._p.changeVisualShape(self.lidar_ball_origin, -1, rgbaColor=self.BLUECOLOR)

        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)
        # self._p.configureDebugVisualizer(self._p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

    

    def reset(self,
              terrain_overide=None,
              initial_motor_angles=None,
              reset_duration=0.0,
              reset_visualization_camera=True):
        """Resets the robot's position in the world or rebuild the sim world.

        The simulation world will be rebuilt if self._hard_reset is True.

        Args:
          initial_motor_angles: A list of Floats. The desired joint angles after
            reset. If None, the robot will use its built-in value.
          reset_duration: Float. The time (in seconds) needed to rotate all motors
            to the desired initial values.
          reset_visualization_camera: Whether to reset debug visualization camera on
            reset.

        Returns:
          A numpy array contains the initial observation after reset.
        """
      
        self._robot.ResetTime() 
        self.gait_generator.reset(0)
        self._env_step_counter = 0
        self.ang_speed_cmd = 0
        self.lin_speed_cmd = [0,0]
        self.phases = [0,0,0,0]
        self.num_targets_reached = 0
        self.force_wait_time = np.random.uniform(self.force_frequency[0], self.force_frequency[1])
        self.force_last_applied = 0
       
        self.reward_proportions = {"gravity":0, "angular":0, "linear":0, "total":0}
        
        if self.reset_num > 5:  # hard reset
            # print(f"\t\treseting...Hard reset...")
            self.hard_reset_func(terrain_overide=terrain_overide)
            self.reset_num = 0
            # print(f"\t\treseting...Hard reset...Done")
        else:
            self.loadTerrain(terrain_overide=terrain_overide)

        
        # print(f"\treseting...start reset loops")
        need_reset = True
        reset_count = 0
        while need_reset:
            reset_count += 1
            self.updateNominalPoseAndTarget()
            self._p.resetBasePositionAndOrientation(self._robot.quadruped, self.base_pos_nom, self.base_orn_nom)
            self._p.resetBaseVelocity(self._robot.quadruped, linearVelocity=0.0)
            self._robot.ResetPose()
            self._robot._SettleDownForReset(reset_time=1.0)
            self._robot.ReceiveObservation()
            self._robot.ResetTime()
            
            self.gait_generator.reset(0)
            
            temp_done = False
            while not temp_done and self._robot.GetTimeSinceReset() < self._MPC_warmup_time:
                _, _, temp_done, _ = self.step(np.zeros_like(self.action_space))
            
            need_reset = self._termination()
            self._env_step_counter = 0  # this attribute is not used

        # print(f"\treseting...Done")

        
            
        # print("reset done")
        self.reset_num += 1
       
        self._observation = self._get_observation()
        return self._observation

    def step(self, action): #, lin_speed_cmd=None, ang_speed_cmd=None):

        # self.render()
        if (self.rendering_enabled):
            self.render()

        return self.stepPD(action)
            
    def _ClipResidualAction(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)

    def stepPD(self, action):
        # print(action)
        """Step forward the simulation, given the action.

        Args:
        action: Can be a list of desired motor angles for all motors when the
            robot is in position control mode; A list of desired motor torques. Or a
            list of tuples (q, qdot, kp, kd, tau) for hybrid control mode. The
            action must be compatible with the robot's motor control mode. Also, we
            are not going to use the leg space (swing/extension) definition at the
            gym level, since they are specific to Minitaur.

        Returns:
        observations: The observation dictionary. The keys are the sensor names
            and the values are the sensor readings.
        reward: The reward for the current state-action pair.
        done: Whether the episode has ended.
        info: A dictionary that stores diagnostic information.

        Raises:
        ValueError: The action dimension is not the same as the number of motors.
        ValueError: The magnitude of actions is out of bounds.
        """
        # add residual here
        #----- residual (action) must be merged in some other way, average, max, lpf
        
        residual_old = self.residual.copy()
        self.residual = self._ClipResidualAction((1.-self.alpha)*residual_old + self.alpha*action)
      
        self.target_positions += list(self.residual)

        mID0, pos0 =self._robot.ComputeMotorAnglesFromFootLocalPosition(0,self.target_positions[0:3]) #RF
        mID1, pos1 =self._robot.ComputeMotorAnglesFromFootLocalPosition(1,self.target_positions[3:6]) #LF
        mID2, pos2 =self._robot.ComputeMotorAnglesFromFootLocalPosition(2,self.target_positions[6:9]) #RB
        mID3, pos3 =self._robot.ComputeMotorAnglesFromFootLocalPosition(3,self.target_positions[9:]) #LB

        motorsComm = np.hstack((pos0,pos1,pos2,pos3))

        for i in range(5):    
            self._robot._StepInternal(motorsComm,1)

        self._robot._step_counter += 5

        self._robot.ReceiveObservation()
        
        # if self.rendering_enabled:
        #     # pass
        #     # self._p.configureDebugVisualizer(self._p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
        #     s = ''

        self._env_step_counter += 1
        
        self.check_target_reached()
        self._observation = self._get_observation()

        reward = self._reward()

        ##apply force after getting the reward, then recover the new observation
        
        if self._robot.GetTimeSinceReset() - self.force_last_applied > self.force_wait_time:
            self.apply_force()
            self._observation = self._get_observation()
    
        done = self._termination()
       
        if done:
            is_success = self.check_success()
        else:
            is_success = False

        if np.isnan(self._observation).any() or np.isnan(reward):
            print(f"NaN detected!!! \nself._observation:{self._observation} \nreward:{reward}")
            self._observation, reward, done = np.zeros(self.observation_space.shape), 0.0, True

 
        self.tmp_time_end = time.time()
        return self._observation, reward, done, {"is_success": is_success, "target_count": self.num_targets_reached, "reward_dist":self.reward_proportions}


    def render(self, mode='human'):
        if mode == 'human':
           
            if not self.rendering_enabled and not self.record_video:
                assert False, "please start with correct rendering mode"
                
            self._p.resetBasePositionAndOrientation(self.target_indicator, self.target, [0, 0, 0, 1])

            base_pos = self.GetBasePosition()
            base_pos[2] = 0.3
            base_rot = self.GetBaseOrientationEuler()
            if self.record_video:
                # Also keep the previous orientation of the camera set by the user.
                [yaw, pitch, dist] = self._p.getDebugVisualizerCamera()[8:11]
                self._p.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)
           
            if self._robot.GetTimeSinceReset() / self._time_step % 1000:  # 1/10 frequency
                self._p.resetBasePositionAndOrientation(self.target_indicator, self.target, [0, 0, 0, 1])
                # if self.rendering_enabled > 1:
                #     print("draw_circle --------------------")
                #     drawCircle(self._p, self.target, self._target_reached_threshold)
                #     # render the target
                #     p0 = base_pos.copy()
                #     p0[2] = 0.
                #     p1 = self.target.copy()
                #     p1[2] = 0.
                #     self._p.addUserDebugLine(p0, p1, np.array([1, 0, 0]), 4, self._time_step*10)

        elif mode == 'rgb_array':
            if self._robot.GetTimeSinceReset() / self._time_step % 100:  # 1/10 frequency
                base_pos = self.GetBasePosition()
                drawCircle(self._p, self.target, self._target_reached_threshold)
                # render the target
                p0 = base_pos.copy()
                p0[2] = 0.
                p1 = self.target.copy()
                p1[2] = 0.
                self._p.addUserDebugLine(p0, p1, np.array([1, 0, 0]), 4, self._time_step*10)

            if self.record_video == 2:  #  or self.rendering_enabled > 0
                self._p.resetBasePositionAndOrientation(self.target_indicator, self.target, [0, 0, 0, 1])
            base_pos = self.GetBasePosition()
            base_pos[2] = 0.3
            view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=base_pos,
                distance=self._camera_dist,
                yaw=self._camera_yaw,
                pitch=self._camera_pitch,
                roll=0,
                upAxisIndex=2)
            proj_matrix = self._p.computeProjectionMatrixFOV(
                fov=60,
                aspect=float(self._render_width) / self._render_height,
                nearVal=0.1,
                farVal=100.0)
            (_, _, px, _, _) = self._p.getCameraImage(
                width=self._render_width,
                height=self._render_height,
                renderer=self._p.ER_BULLET_HARDWARE_OPENGL,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix)
            rgb_array = np.array(px)
            rgb_array = rgb_array[:, :, :3]
            return rgb_array
        else:
            super(MyEnv, self).render(mode=mode) # just raise an exception

    def check_success(self):
        return self.num_targets_reached >= 2 and not self.check_contact_fail()

    def check_target_limit(self):
        return self.num_targets_reached == 5

    def check_target_reached(self):
        if self.GetTargetDistance() < self._target_reached_threshold:
            self.num_targets_reached += 1
            self.generateNewTarget()
            return True
        return False

    def check_contact_fail(self):
        foot_links = self._robot.GetFootLinkIDs()
        leg_links = self._robot.GetLegLinkIDs()
        allowed_contact_links = foot_links + leg_links  + self._robot._motor_link_ids
        ground = self.get_ground()

        # sometimes the robot can be initialized with some ground penetration
        # so do not check for contacts until after the first env step.
        if self.env_step_counter > 0:
            robot_ground_contacts = self._p.getContactPoints(
                bodyA=self.robot.quadruped, bodyB=ground)

            for contact in robot_ground_contacts:
                if contact[3] not in allowed_contact_links:
                    
                    return True
        return False

    def check_contact_danger(self):
        #TODO ensure this works for perlin
        warning_contact_links = self._robot._motor_link_ids
        ground = self.get_ground()

        # sometimes the robot can be initialized with some ground penetration
        # so do not check for contacts until after the first env step.
        if self.env_step_counter > 0:
            robot_ground_contacts = self._p.getContactPoints(
                bodyA=self.robot.quadruped, bodyB=ground)

            for contact in robot_ground_contacts:
                if contact[3] not in warning_contact_links:
                    
                    return 1
        return 0

    def check_timeout(self):
        # time_out = False
        if self._robot.GetTimeSinceReset() > self._max_time_limit:  # in secs
            return True
        return False

    def _termination(self):
        return self.check_target_limit() \
            or self.check_contact_fail() \
            or self.check_timeout() 


    def _reward(self):

        #gravity reward - creates stability
        gravity_vec = self.GetGravityVector()
        gravity_error = np.array([0, 0, -1]) - gravity_vec
        gravity_error = np.linalg.norm(gravity_error)  
        gravity_reward = np.exp(-gravity_error**2*self.reward_params.gravity)

        # angular reward = rotational
        desired_ang_vel  = np.array([self.ang_speed_cmd])
        ang_vel = self._robot.GetBaseRollPitchYawRate()[2]
        ang_reward = np.exp(-np.linalg.norm(desired_ang_vel - ang_vel)**2*self.reward_params.angular)

        #linear reward = speed and direction
        desired_lin_com = np.array([self.lin_speed_cmd[0],self.lin_speed_cmd[1]])
        lin_vel = self._robot.GetBaseVelocity()[:2]
        linear_reward = np.exp(-np.linalg.norm(desired_lin_com - lin_vel)**2*self.reward_params.linear)

        #distance reward = speed and direction
        target_dist = np.clip(self.GetTargetDistance(), -1,1)
        distance_reward = np.exp(-np.linalg.norm(0 - target_dist)**2*self.reward_params.distance)
        
        reward = gravity_reward + ang_reward + linear_reward + distance_reward


        self.reward_proportions['gravity'] += gravity_reward 
        self.reward_proportions['angular'] += ang_reward 
        self.reward_proportions['linear'] += linear_reward 
        self.reward_proportions['total'] += reward * 0.01
        

        return reward*0.01

    def _get_observation(self):
        """Get observation of this environment from a list of sensors.

        Returns:
        observations: sensory observation in the numpy array format
        """
        # first we need to set up class params for the kernal as well
        # phase, ang_cmd, forward_cmd, side_cmd
        r = self.GetTargetDistance()
        pos = self.GetBasePose()
        robotOrn = self._p.getEulerFromQuaternion(pos[1])
        theta_ = np.arctan2(self.target[1]-pos[0][1],self.target[0]-pos[0][0]) - robotOrn[2] 
        theta = self.ModNearest(math.degrees(theta_))
        # theta = self.GetYawError()
        lin_cmd = [r*np.cos(theta),r*np.sin(theta),0]
        ang_cmd = 0.3*theta

        #this needs to update gradually
        # if (self._robot.GetTimeSinceReset()): #freq = 20
        self.lin_speed_cmd[0] += max( min( lin_cmd[0] - self.lin_speed_cmd[0], 0.005) , -0.005)
        self.lin_speed_cmd[1] += max( min( lin_cmd[1] - self.lin_speed_cmd[1], 0.005) , -0.005)
        self.ang_speed_cmd += max( min( ang_cmd - self.ang_speed_cmd, 0.005) , -0.005)

        # limit overall maximum (only affects the command when the limit is actually exceeded)
        manhatDist = abs(self.lin_speed_cmd[0]) + abs(self.lin_speed_cmd[1])
        if manhatDist > 0.5:
            factor = 1 #1 means maximum performance (<1 means safe parameters)
            norm   = manhatDist / (0.5*factor)
            self.lin_speed_cmd[0] /= norm
            self.lin_speed_cmd[1] /= norm

        self.lin_speed_cmd[0] = np.clip (self.lin_speed_cmd[0],-0.5,0.5)
        self.lin_speed_cmd[1] = np.clip (self.lin_speed_cmd[1],-0.2,0.2)
        
        self.ang_speed_cmd = np.clip (self.ang_speed_cmd ,-np.pi/4,np.pi/4)

        self.gait_generator.update(self._robot.GetTimeSinceReset())
        norm_phase = self.gait_generator.normalized_phase
        states = self.gait_generator.desired_leg_state
        phase_states = list(zip(norm_phase, states))
        self.phases = list(map(lambda x: x[0] + 1 if (x[1] == gait_generator_lib.LegState.SWING) else x[0], phase_states))

        x_input = torch.Tensor(  [self.ang_speed_cmd] + list(self.lin_speed_cmd) + self.phases).unsqueeze(0).float()
        self.target_positions = self.kernel(x_input).squeeze().detach().numpy()


        observations = []
        observations.append(self.GetJointAngles())          # 12D - action_space
        observations.append(self.GetJointVelocities())      # 12D - pm inf rad/s
        observations.append(self.GetGravityVector())        # 3 - pm1
        # observations.append(self.theta_env)        # 3 - pm1
        # observations.append(self.r_env)        # 3 - pm1
        observations.append(self.ang_speed_cmd)             # 1 - pm1
        observations.append(self.lin_speed_cmd[0])       # 1 - pm1
        observations.append(self.lin_speed_cmd[1])       # 1 - pm1
        observations.append(self.GetBaseVelocity())         # 3 - 2 m/s
        observations.append(self._robot.GetBaseRollPitchYawRate())         # 3 - 2 m/s
        # observations.append(self.GetTargetDistance())         # 3 - 2 m/s
        observations.append(np.clip(self.GetTargetDistance(), -1,1))
        if self.visionEnabled:
            observations.append(self.GetVisionObservation())    # 10 - pm1
        if self._usePhaseObs:
            observations.append(self.phases)    # 1 - [0, 1]
        if self._useContactStates:
            observations.append(self.GetContactState())    # 1 - [0, 1]
        if self._useTargetTrajectories:
            observations.append(self.target_positions)

        self.check_observation = np.hstack(np.array(observations))
        return np.hstack(np.array(observations))










############################################################### UTILS

    def apply_force(self):
        #force range 300 - 500
        base_pos = self._p.getBasePositionAndOrientation(self._robot.quadruped)
        base_orn = self._p.getEulerFromQuaternion(base_pos[1])

        force = np.random.uniform(low=self.force_range[0], high=self.force_range[1])
        side = np.random.randn()

        if side > 0:
            force_direction = np.array([1/4,1,1/4])
        else:
            force_direction = np.array([-1/4,-1,1/4])

        force_scaling = force / np.linalg.norm(force_direction)
        force_vec = force_direction * force_scaling

        hit_point = np.array([random.uniform(-0.25/2, 0.25/2), random.uniform(-0.18/2,0), random.uniform(-0.11/2, 0.11/2)])

        if self.rendering_enabled > 0:
            debug_id = self._p.addUserDebugLine(base_pos[0]+hit_point,base_pos[0]-force_vec, [0,1,1] , 50)
            self._p.addUserDebugLine([0,0,0], [0,0,100], self.REDCOLOR[:3], 10)
    
        for i in range(30):
            self._p.applyExternalForce(self._robot.quadruped, -1, force_vec, hit_point, flags=self._p.LINK_FRAME )
            self._p.stepSimulation()

        if self.rendering_enabled  > 0:
            self._p.removeUserDebugItem(debug_id)


        self.force_last_applied = self._robot.GetTimeSinceReset()
        self.force_wait_time = np.random.uniform(self.force_frequency[0], self.force_frequency[1])


        return




    def GetContactState(self):
        return np.array(self._robot.GetFootContacts()).astype(int)



    def GetJointAngles(self):  # 12
        return self._robot.GetTrueMotorAngles()

    def GetJointVelocities(self):  # 12
        return self._robot.GetTrueMotorVelocities()

    def GetGravityVector(self):  # 3
        # This may not be the Pelvis CoM position _exactly_ but should be fine, otherwise can apply local transformation
        base_pos, base_quat = self.GetBasePose()
        gravity = np.array([0, 0, -1])
        gravity_quat = self._p.getQuaternionFromEuler([0, 0, 0])
        invBasePos, invBaseQuat = self._p.invertTransform([0, 0, 0], base_quat)
        gravityPosInBase, gravityQuatInBase = self._p.multiplyTransforms(invBasePos, invBaseQuat, gravity, gravity_quat)
        gravityPosInBase = np.array(gravityPosInBase)
        
        return gravityPosInBase

    def GetYawError(self): #1
        # returning yaw error = 0.0 rads when a target is reached
        if self.check_target_reached():
            return 0.0
        base_pos = self.GetBasePosition()
        base_orn = self.GetBaseOrientationEuler()
        target_pos = self.GetTargetDirection()
        theta = np.arctan2(target_pos[1] - base_pos[1], target_pos[0] - base_pos[0]) - base_orn[2]
        theta = self.ModNearest(np.rad2deg(theta)) # ModNearest takes [deg] as input, gives [rad] as output!
        theta = (theta + np.pi) % (2*np.pi) - np.pi # wrap angle to interval [-pi,pi) rad
        return theta

    def GetTargetError(self): # auxilary function
        error = self.target - self.GetBasePosition()
        return error[:2]

    def GetTargetDirection(self):  
        return self.GetTargetError() / self.GetTargetDistance()
    
    def GetTargetDistance(self):  # 1
        return np.linalg.norm(self.GetTargetError())

    def GetBaseVelocity(self):  # 3
        return self._robot.GetBaseVelocity()

    def GetVisionObservation(self):  # 10
        base_pos, base_quat = self._p.getBasePositionAndOrientation(self._robot.quadruped)
        R = quat_to_rot(base_quat)

        # Initiallising the LiDAR
        lidar_points_from = [self.lidar_offset] * self._raynum
        lidar_points_to = self.lidar.getRayBatchLocal()
        results = self._p.rayTestBatch(lidar_points_from, lidar_points_to, parentObjectUniqueId=self._robot.quadruped, parentLinkIndex=-1)
        lidar_origin_world = base_pos + R @ np.array(lidar_points_from[0])

        # get the vision observations
        visionObservation = np.zeros(self._raynum)

        # extracting the lidar measurements
        for i in range(self._raynum):
            visionObservation[i] = results[i][2]

        visionObservation = - visionObservation * 2 + 1  # Normalise to [-1, +1], which maps to ray [0, full len]

        if self.rendering_enabled or self.enable_rays > 0:

            self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 0)
            # remove debug lines if rendering
            # p.removeAllUserDebugItems()
            # update the ball-indicators for vision, if rendering
            for i in range(self._raynum):
                hitObjectUid = results[i][0]
                ball = self.lidar_balls[i]
                if (hitObjectUid < 0):
                    default_ball_pos = results[i][3]
                    if self.enable_rays > 0:
                        self._p.resetBasePositionAndOrientation(ball, default_ball_pos, [0, 0, 0, 1])
                        self._p.changeVisualShape(ball, -1, rgbaColor=self.REDCOLOR)
                    if self.enable_rays > 1:
                        self._p.addUserDebugLine(lidar_origin_world, default_ball_pos, self.REDCOLOR[:3], 2., 0.01)
                else:
                    hitPosition = results[i][3]
                    if self.enable_rays > 0:
                        self._p.resetBasePositionAndOrientation(ball, hitPosition, [0, 0, 0, 1])
                        self._p.changeVisualShape(ball, -1, rgbaColor=self.GREENCOLOR)
                        # self._p.addUserDebugLine(lidar_origin_world, hitPosition, lineColorRGB=self.GREENCOLOR[:3], lineWidth=2.5, parentObjectUniqueId=self.r, parentLinkIndex=0)
                    if self.enable_rays > 1:
                        self._p.addUserDebugLine(lidar_origin_world, hitPosition, self.GREENCOLOR[:3], 2.5, 0.01)

            # reset the position of the lidar origin
            self._p.resetBasePositionAndOrientation(self.lidar_ball_origin, lidar_origin_world, [0, 0, 0, 1])
            self._p.changeVisualShape(self.lidar_ball_origin, -1, rgbaColor=self.BLUECOLOR)

            self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)

        visionObservation = visionObservation.reshape(self._vision_dim)
        self.visionObservation = visionObservation  # for saving logs
        extracted_feature = self._vision_encoder(visionObservation)
        assert extracted_feature.shape == (self._vision_input_size, ), \
            f"extracted_feature should have dim {(self._vision_input_size, )}, but got {extracted_feature.shape}"

        return extracted_feature

    def GetBaseHeading(self): 
        base_pos, base_quat = self.GetBasePose()
        base_orn = self._p.getEulerFromQuaternion(base_quat)
        base_pos_vel = np.array([1,0,0])
        base_pos_vel.resize(1, 3)
        Rz = rotZ(base_orn[2])
        Rz_i = np.linalg.inv(Rz)
        # heading
        base_heading_xy = np.transpose(Rz_i @ base_pos_vel.transpose())  # base velocity in adjusted yaw frame
        return base_heading_xy

    def get_ref_base_rotation(self):
        direction = self.target - self.GetBasePosition()
        direction[2] = 0
        direction = direction / np.linalg.norm(direction)
        return direction
    
    def _get_default_root_rotation(self):
        """Get default root rotation."""
        motion = self.get_active_motion()
        root_rot = motion.get_frame_root_rot(self._default_pose)
        return root_rot

    def _generate_example_linear_angular_speed(self, t):
        """Creates an example speed profile based on time for demo purpose."""
        vx = 0.6 * robot_sim.MPC_VELOCITY_MULTIPLIER
        vy = 0.2 * robot_sim.MPC_VELOCITY_MULTIPLIER
        wz = 0.8 * robot_sim.MPC_VELOCITY_MULTIPLIER

        time_points = (0, 5, 10, 15, 20, 25, 30)
        speed_points = ((0, 0, 0, 0), (0, 0, 0, wz), (vx, 0, 0, 0), (0, 0, 0, -wz), (0, -vy, 0, 0),
                        (0, 0, 0, 0), (0, 0, 0, wz))

        speed = scipy.interpolate.interp1d(
            time_points,
            speed_points,
            kind="previous",
            fill_value="extrapolate",
            axis=0)(
                t)

        return speed[0:3], speed[3]

    def _generate_example_linear_angular_speed_from_keyboard(self, lin_cmd=None, ang_cmd=None):
        """Creates an example speed profile based on time for demo purpose."""
        vx = 0.6 * robot_sim.MPC_VELOCITY_MULTIPLIER
        vy = 0.2 * robot_sim.MPC_VELOCITY_MULTIPLIER
        wz = 0.8 * robot_sim.MPC_VELOCITY_MULTIPLIER

        linear_velocity = [0.0, 0.0, 0.0]
        if lin_cmd:
            directions = collections.defaultdict(lambda: 0.0)
            directions['f'] = 0.0
            directions['b'] = np.pi
            directions['l'] = np.pi/2
            directions['r'] = -np.pi/2
            directions['fl'] = np.pi/4
            directions['fr'] = -np.pi/4
            directions['bl'] = 3*np.pi/4
            directions['br'] = -3*np.pi/4
            theta = directions[lin_cmd]
            linear_velocity = [vx * np.cos(theta), vx * np.sin(theta), 0]

        angular_velocity = 0.0
        if ang_cmd:
            angular_velocity = wz if ang_cmd == 'tl' else -wz

        return linear_velocity, angular_velocity



    def _get_key_pressed(self, client, relevant=None):
        p = client
        pressed_keys = []
        events = p.getKeyboardEvents()
        key_codes = events.keys()
        for key in key_codes:
            pressed_keys.append(key)
        return pressed_keys

    def _parse_keyboard_input(self, pressed_keys):
        # parse keys into linear, angular velocity commands
        key_map = collections.defaultdict(lambda: 0b0000)
        key_map[65295] = 0b000001  # 'left',
        key_map[65296] = 0b000010  # 'right',
        key_map[65297] = 0b000100  # 'up',
        key_map[65298] = 0b001000  # 'down'
        key_map[44] = 0b010000  # 'turn-left'
        key_map[46] = 0b100000  # 'turn-right'

        key_comp = sum([key_map[k] for k in pressed_keys])

        directions = collections.defaultdict(lambda: None)
        directions[0b000001] = 'l'  # left
        directions[0b000010] = 'r'  # right
        directions[0b000100] = 'f'  # front
        directions[0b001000] = 'b'  # back
        directions[0b000101] = 'fl'  # front-left
        directions[0b000110] = 'fr'  # front-right
        directions[0b001001] = 'bl'  # back-left
        directions[0b001010] = 'br'  # back-right

        directions[0b010000] = 'tl'  # turn-left
        directions[0b100000] = 'tr'  # turn-right

        #      linear velocity command          angular velocity command
        return directions[key_comp & 0b001111], directions[key_comp & 0b110000]

    def setupPhysicsParmeters(self):
        self._p.setTimeStep(self._time_step)
        self._p.setGravity(0, 0, -9.8)

        num_bullet_solver_iterations = 30
        self._p.setPhysicsEngineParameter(numSolverIterations=num_bullet_solver_iterations)
        self._p.setPhysicsEngineParameter(enableConeFriction=0)
        self._p.setPhysicsEngineParameter(numSolverIterations=30)


    def updateNominalPoseAndTarget(self):
        if self.terrain_type == 'gap':
            base_pos_nom, base_orn_nom = np.array([0.0, 0.0, 0.3]), np.random.uniform(-np.pi, np.pi, (3, ))
            base_orn_nom[0] = 0
            base_orn_nom[1] = 0
            base_orn_nom = self._p.getQuaternionFromEuler(base_orn_nom)
            self.base_pos_nom = base_pos_nom
            self.base_orn_nom = base_orn_nom

            self.target = self.generateNewTarget(currXY=base_pos_nom)
        elif self.terrain_type == "stair":
            base_pos_nom = [-1.5, 0, 0.3]
            base_orn_nom = self._p.getQuaternionFromEuler(np.r_[0, 0, np.random.uniform(-np.pi, np.pi)])
            self.base_pos_nom = base_pos_nom
            self.base_orn_nom = base_orn_nom
            self.target = self.generateNewTarget(currXY=base_pos_nom)
        else: # flat, uniform-noise
            base_pos_nom, base_orn_nom = np.random.uniform(-1, 1, (3, )), np.random.uniform(-np.pi, np.pi, (3, ))

            ground_height = np.max(self.ground_data)
         
            #TODO checking perlin remove me
            base_pos_nom[2] = 0.3+ground_height
            base_orn_nom[0] = 0 
            base_orn_nom[1] = 0
            base_orn_nom = self._p.getQuaternionFromEuler(base_orn_nom)
            self.base_pos_nom = base_pos_nom
            self.base_orn_nom = base_orn_nom

            self.target = self.generateNewTarget()

    def resetResidualCommands(self):
        self.residual = np.zeros(self.action_space.shape[0])

    def loadRobot(self):
        quadruped = self._p.loadURDF(robot_sim.URDF_NAME, robot_sim.START_POS)
        self._robot = robot_sim.SimpleRobot(self._p, quadruped, simulation_time_step=self._time_step)
        self.base_pos_nom = np.r_[0,0,0.3]
        self.base_orn_nom = np.r_[0,0,0,1]

    def loadTerrain(self, terrain_overide=None):
        self.old_terrain = self.terrain_type
        if terrain_overide:
            self.terrain_type = terrain_overide
        elif self.terrain_probability == "uniform":
            terrain_id = np.random.randint(0, len(self.terrain_names))
            self.terrain_type = self.terrain_names[terrain_id]
        elif self.terrain_probability == "fixed":
            pass  # don't change terrain type
        else:
            raise Exception(f"Not supported \'self.terrain_probability\' {self.terrain_probability}")
        
        try:
            self._p.removeBody(self.get_ground())
        except:
            pass
        
        if self.terrain_type == "flat":  # "flat", "perlin", "uniform", "stair", "gap"
            self.set_ground(self._p.loadURDF("plane.urdf"))
        else:
            self.set_ground(self.create_uneven_terrain(self._p))

    def create_uneven_terrain(self, p, heightPerturbationRange=0.06):

        heightPerturbationRange = np.random.uniform(low=self.terrain_uniform_range[0], high=self.terrain_uniform_range[1])
        numHeightfieldRows = 256
        numHeightfieldColumns = 256
        meshScale = [.06, .06, 1.6]
        nominalPos = [0, 0, -heightPerturbationRange]
        p.configureDebugVisualizer( p.COV_ENABLE_RENDERING, 1) #it will not be reenabled in this function
            
        if self.terrain_type == "uniform":
            heightfieldData = [0] *  numHeightfieldRows *  numHeightfieldColumns
            self.ground_data = np.zeros((256,256))
            for j in range(int( numHeightfieldColumns / 2)):
                for i in range(int( numHeightfieldRows / 2)):
                    height = random.uniform(0, heightPerturbationRange)
                    heightfieldData[2 * i + 2 * j *  numHeightfieldRows] = height
                    heightfieldData[2 * i + 1 + 2 * j *  numHeightfieldRows] = height
                    heightfieldData[2 * i + (2 * j + 1) *  numHeightfieldRows] = height
                    heightfieldData[2 * i + 1 + (2 * j + 1) *  numHeightfieldRows] = height
        elif self.terrain_type == "perlin":
            try: 
                import noise
            except: 
                "Missing noise module, try 'pip install noise'"
                sys.exit()
            heightfieldData = np.zeros((numHeightfieldRows, numHeightfieldColumns)) #[0]*numHeightfieldRows*numHeightfieldColumns 

            shape = (50,50)
            lacunarity = 5.0
           
            p_params = np.random.choice(self.perlin_params)

            for i in range (int(numHeightfieldColumns)):
                for j in range (int(numHeightfieldRows) ):
                    heightfieldData[i,j] = noise.pnoise2(i/p_params['scale'], 
                                            j/p_params['scale'], 
                                            octaves=p_params['octaves'], 
                                            persistence=p_params['persistence'], 
                                            lacunarity=lacunarity, 
                                            repeatx=1024, 
                                            repeaty=1024, 
                                            base=42)

            self.ground_data = heightfieldData
           
            heightfieldData = heightfieldData.flatten()

        elif self.terrain_type == "gap":
            # Overriding settings
            gapScale = 0.25
            meshScale = np.array([.06 * gapScale, .06 * gapScale, 1.6])
            heightPerturbationRange = 0.08
            nominalPos = [0, 0, -heightPerturbationRange]

            heightfieldData = np.full((numHeightfieldRows, numHeightfieldColumns), 0.0)
            gap_centers = []
            gap_remaining = 60
            gap_minimumSeparation = 25
            while gap_remaining > 0:
                centre = np.random.randint(0, numHeightfieldColumns, (2, ))
                valid = True
                for c in gap_centers:
                    if np.linalg.norm(c - centre) < gap_minimumSeparation:
                        valid = False
                        break
                if not valid or np.linalg.norm(centre) < (numHeightfieldRows * 0.25 * 0.75):
                    continue
                # centre = numHeightfieldRows // 2
                # c = centre, centre # in y, x direction
                c = centre
                if np.random.random() > 0.5:
                    s = 15, 2  # in y, x direction
                else:
                    s = 2, 15  # in y, x direction
                heightfieldData[c[0]-s[0]:c[0]+s[0], c[1]-s[1]:c[1]+s[1]] = -heightPerturbationRange
                gap_centers.append(centre)
                gap_remaining -= 1

            self.ground_data = heightfieldData
            heightfieldData = heightfieldData.flatten()
            
        elif self.terrain_type == "stair":
            # Overriding settings
            stairScale = 0.25
            meshScale = np.array([.06 * stairScale, .06 * stairScale, 1.])

            heightfieldData = np.full((numHeightfieldRows, numHeightfieldColumns), 0.0)
            stair_remaining = 13
            stair_height_diff = 0.05
            stair_shape = 40, 10  # in y, x direction
            centre = np.array([numHeightfieldRows//2, numHeightfieldRows//2-40])
            stair_height = stair_height_diff
            while stair_remaining > 0:
                c = centre
                s = stair_shape
                heightfieldData[c[0]-s[0]:c[0]+s[0], c[1]-s[1]:c[1]+s[1]] = stair_height

                centre[1] = centre[1] + stair_shape[1] * 2
                stair_height += stair_height_diff
                stair_remaining -= 1

            stair_height -= stair_height_diff
            nominalPos = [0, 0, 0.15]

            self.ground_data = heightfieldData
            heightfieldData = heightfieldData.flatten()
        ### unevenTerrainShape generated

        

        unevenTerrainShape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=meshScale,
            heightfieldTextureScaling=( numHeightfieldRows - 1)/ 2,
            heightfieldData=heightfieldData,
            numHeightfieldRows= numHeightfieldRows,
            numHeightfieldColumns= numHeightfieldColumns)

        newTerrain = p.createMultiBody(0, unevenTerrainShape)
     
        p.resetBasePositionAndOrientation(newTerrain, nominalPos, [0, 0, 0, 1])
        p.changeDynamics(newTerrain, -1, lateralFriction=1.0)
        tiletextureId = p.loadTexture(__cwd__ + "/data/tile.png")
        p.changeVisualShape(newTerrain, -1, textureUniqueId=tiletextureId, rgbaColor=[1, 1, 1, 1], specularColor=[1, 1, 0])
        
        return newTerrain
        
    def ModNearest(self, a, b=360.0):
        if (a<0): 
            a+=360

        theta = a - b * np.round(a / b)
        if (theta>180):
            theta -= 360
    
        return math.radians(theta);  

    def generateNewTarget(self, currXY=None, rx=3, ry=3, minDistance=None):
        if not self._robot:
            print("Do this after loading the robot.")
            return np.zeros((3, ))
        
        if minDistance is None:
            minDistance = self._minimum_target_distance
            
        if (currXY is None):
            currXY = self.GetBasePosition()[:2]
        elif len(currXY) > 2:
            currXY = currXY[:2]

        if self.terrain_type == 'stair':
            newtarget = np.array([2., 0, .5])
        else:
            x0, y0 = currXY
            xn = np.random.uniform(-rx, rx)
            yn = np.random.uniform(-ry, ry)

            newtarget = np.array([xn, yn, 0])
                
            gap_terrain_max_size = 2.5
            r = np.linalg.norm(np.array([x0, y0]) - newtarget[:2])
            if self.terrain_type == "gap":
                validTarget = np.all(np.abs(newtarget[:2]) <= gap_terrain_max_size)
            else:
                validTarget = True
            while (r <= minDistance) or not validTarget:
                xn = np.random.uniform(-rx, rx)
                yn = np.random.uniform(-ry, ry)
                newtarget = np.array([xn, yn, 0])
                r = np.linalg.norm(np.array([x0, y0]) - newtarget[:2])
                if self.terrain_type == "gap":
                    validTarget = np.all(newtarget[:2] <= gap_terrain_max_size)
                else:
                    validTarget = True

        self.target = newtarget
        return newtarget
    
        
    def GetBasePose(self):
        pos, quat = self._p.getBasePositionAndOrientation(self._robot.quadruped)
        return np.array(pos), np.array(quat) #np.array(self._p.getEulerFromQuaternion(quat))
    
    def GetBasePosition(self):
        return self.GetBasePose()[0]
        
    def GetBaseOrientation(self): 
        return self.GetBasePose()[1]
        
    def GetBaseOrientationEuler(self): 
        return np.array(self._p.getEulerFromQuaternion(self.GetBasePose()[1]))
    
    def GetTrueMotorAngles(self):
        return self._robot.GetTrueMotorAngles()


    def get_ground(self):
        """Get simulation ground model."""
        return self._world_dict['ground']


    def set_ground(self, ground_id):
        """Set simulation ground model."""
        self._world_dict['ground'] = ground_id

    @property
    def world_dict(self):
        return self._world_dict.copy()

    @world_dict.setter
    def world_dict(self, new_dict):
        self._world_dict = new_dict.copy()

    @property
    def last_base_position(self):
        return self._last_base_position

    @property
    def pybullet_client(self):
        return self._p

    @property
    def robot(self):
        return self._robot

    @property
    def env_step_counter(self): # this is 5 times more than the agent's step_counter
        return self._env_step_counter

    @property
    def plt_vision(self):
        if hasattr(self, "visionObservation"):
            return self.visionObservation
        else:
            raise Exception("Please call this after step")

    @property
    def plt_mpc(self):
        if hasattr(self, "hybrid_action"):
            return self.hybrid_action
        else:
            raise Exception("Please call this after step")

    @property
    def gaitPhase(self):
        if hasattr(self, "gait_generator"):


            #CHECK - Note i think i should remove the update in step
            self.gait_generator.update(self._robot.GetTimeSinceReset())
            norm_phase = self.gait_generator.normalized_phase
            states = self.gait_generator.desired_leg_state
            phase_states = list(zip(norm_phase, states))
            phases = list(map(lambda x: x[0] + 1 if (x[1] == gait_generator_lib.LegState.SWING) else x[0], phase_states))
        
            return np.array(phases)
        else:
            raise Exception("Please call this after setting up the MPC controller")

    
    ### End of the Gym env.



def _run_example(max_time=_MAX_TIME_SECONDS):
    """Runs the locomotion controller example."""

    from collections import namedtuple
    args = {}
    args['gravity'] = 1
    args['distance'] = 1
    args['linear'] = 1
    args['angular'] = 1
    # args['g_weight'] = trial.suggest_int("rollouts", 5000, 20000, step=5000)
    # args['d_weight'] = trial.suggest_int("num_epochs", 1, 19, step=3)
    # args['l_weight'] = trial.suggest_categorical("net_depth", ['net_small', 'net_med', 'net_large'])
    # args['l_weight'] = trial.suggest_categorical("net_depth", ['net_small', 'net_med', 'net_large'])
 

    args['target_distance'] = 2

    args['lr'] = 0.001
    args['lr_decay'] = 1e-7
    args['entropy_coef'] = 0.000015
    args['batch_size'] = 2000
    args['roll_length'] = 20000
    args['num_epochs'] = 10


    args = namedtuple("ObjectName", args.keys())(*args.values())
    env_config = dict(
            # usePhaseObs = True,
            useTargetTrajectories = True,
            useContactStates = True,
            terrain_type=None,  # initial terrain overide; flat, uniform, perlin, gap, bump, None
            terrain_probability="uniform",  # uniform, fixed

            action_upper_bound= .05,
            action_lower_bound=-.05,
            alpha=0.1,  #0.1 init (0, 1] - 1=no lpf, alpha->0 old action
            visionEnabled=True,
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
            target_distance = args.target_distance,
            perlin_params = [
                {'scale':180,'octaves':4, 'persistence':0.2},
                {'scale':200,'octaves':6, 'persistence':0.3},
            ]
            

        )
    env = A1_env_v1(**env_config)  
    start = time.time()
    for i in range(100000):
        _,_, done, _ = env.step(np.array([0,0,0,0,0,0,0,0,0,0,0,0]))  
        if done:
            print("terminations")
            input()

    

       
def drawCircle(p, center, radius, color=[0, 1, 1], thickness=5, lifeTime=0.01):
    nPoint = 15
    for i in range(nPoint+1):
        p1 = [center[0]+radius*np.sin(i*2*np.pi/nPoint), center[1]+radius*np.cos(i*2*np.pi/nPoint), 0]
        j = i+1
        p2 = [center[0]+radius*np.sin(j*2*np.pi/nPoint), center[1]+radius*np.cos(j*2*np.pi/nPoint), 0]
        p.addUserDebugLine(p1, p2, color, thickness, lifeTime)
        # p.addUserDebugLine([center[0],center[1],0], [center[0]+radius*np.sin(i*2*np.pi/10),center[1]+radius*np.cos(i*2*np.pi/10),0], [1,0,1], 5)


def drawRectangle(self, p, A, B, C, D, color=[0, 1, 1], thickness=5):
    p.addUserDebugLine(A, B, color, thickness, 0.05)
    p.addUserDebugLine(B, C, color, thickness, 0.05)
    p.addUserDebugLine(C, D, color, thickness, 0.05)
    p.addUserDebugLine(D, A, color, thickness, 0.05)


def drawRectangle(self, p, origin, w, h, color=[0, 1, 1], thickness=5):
    A = origin
    B = A + np.array([w, 0, 0])
    C = B + np.array([0, h, 0])
    D = C - np.array([w, 0, 0])

    p.addUserDebugLine(A, B, color, thickness, 0.05)
    p.addUserDebugLine(B, C, color, thickness, 0.05)
    p.addUserDebugLine(C, D, color, thickness, 0.05)
    p.addUserDebugLine(D, A, color, thickness, 0.05)


def translate(value, old_range, new_range):
    value = np.array(value)
    old_range = np.array(old_range)
    new_range = np.array(new_range)

    OldRange = float(old_range[1][:] - old_range[0][:])
    NewRange = float(new_range[1][:] - new_range[0][:])
    NewValue = (value - old_range[0][:]) * NewRange / float(OldRange) + new_range[0][:]
    return NewValue


def rotX(theta):
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(theta), -math.sin(theta)],
        [0.0, math.sin(theta), math.cos(theta)]])
    return R


def rotY(theta):
    R = np.array([
        [math.cos(theta), 0.0, math.sin(theta)],
        [0.0, 1.0, 0.0],
        [-math.sin(theta), 0.0, math.cos(theta)]])
    return R


def rotZ(theta):
    R = np.array([
        [math.cos(theta), -math.sin(theta), 0.0],
        [math.sin(theta), math.cos(theta), 0.0],
        [0.0, 0.0, 1.0]])
    return R


def quat_to_rot(qs):  # transform quaternion into rotation matrix
    qx = qs[0]
    qy = qs[1]
    qz = qs[2]
    qw = qs[3]

    x2 = qx + qx
    y2 = qy + qy
    z2 = qz + qz
    xx = qx * x2
    yy = qy * y2
    wx = qw * x2
    xy = qx * y2
    yz = qy * z2
    wy = qw * y2
    xz = qx * z2
    zz = qz * z2
    wz = qw * z2

    m = np.empty([3, 3])
    m[0, 0] = 1.0 - (yy + zz)
    m[0, 1] = xy - wz
    m[0, 2] = xz + wy
    m[1, 0] = xy + wz
    m[1, 1] = 1.0 - (xx + zz)
    m[1, 2] = yz - wx
    m[2, 0] = xz - wy
    m[2, 1] = yz + wx
    m[2, 2] = 1.0 - (xx + yy)

    return m


def euler_to_quat(roll, pitch, yaw):  # rad
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    w = cy * cr * cp + sy * sr * sp
    x = cy * sr * cp - sy * cr * sp
    y = cy * cr * sp + sy * sr * cp
    z = sy * cr * cp - cy * sr * sp

    return [x, y, z, w]


def rescale(value, old_range, new_range):
    value = np.array(value)
    old_range = np.array(old_range)
    new_range = np.array(new_range)

    OldRange = old_range[1][:] - old_range[0][:]
    NewRange = new_range[1][:] - new_range[0][:]
    NewValue = (value - old_range[0][:]) * NewRange / OldRange + new_range[0][:]
    return NewValue


def clip_increment(ref_value, target_value, inc_bound):
    min_bound = ref_value-inc_bound
    max_bound = ref_value+inc_bound
    new_value = np.clip(target_value, min_bound, max_bound)
    return new_value



def main(argv):
    del argv
    _run_example()


if __name__ == "__main__":
    app.run(main)
