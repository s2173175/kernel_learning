from __future__ import absolute_import
from __future__ import division
#from __future__ import google_type_annotations
from __future__ import print_function
from distutils import command

import os
import inspect
from urllib import robotparser
from numpy import dtype

from requests import get
from torch import save

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
from copy import deepcopy
import csv 

from mpc_controller import com_velocity_estimator
from mpc_controller import gait_generator as gait_generator_lib
from mpc_controller import locomotion_controller
from mpc_controller import openloop_gait_generator
from mpc_controller import raibert_swing_leg_controller
from mpc_controller import torque_stance_leg_controller
from mpc_controller.raibert_swing_leg_controller import _gen_swing_foot_trajectory as get_tracjectory

#uncomment the robot of choice
#from mpc_controller import laikago_sim as robot_sim
from mpc_controller import a1_sim as robot_sim

import optuna

__cwd__ = os.path.realpath( os.path.join(os.getcwd(), os.path.dirname(__file__)))


##########################################################################################

#### speeds : 0.3 --- 0.20 --- 0.15
#### step_heights : 0.24 --- 0.18 --- 0.12

"""
0.3 - 0.24 - 0.6
0.3 - 0.18 - 0.82
0.3 - 0.12 - 0.83

0.2 - 0.24 - 0.53
0.2 - 0.18 - 0.56 
0.2 - 0.12 -      --------------- cannot make this work 

0.15 - 0.24 - 0.5
0.15 - 0.18 - 0.7
0.15 - 0.12 - 0.7

"""

##########################################################################################


"""
This has not worked, 
Idea use bayesian optmization to find stable parameters 

eg, i can predefine step height, and a target frequency range to find an optimal duty factor.

can i just use the RL reward function to compare 

"""

##########################################################################################

_INIT_PHASE_FULL_CYCLE = [0.9, 0, 0, 0.9]
_INIT_LEG_STATE = (
    gait_generator_lib.LegState.SWING,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.SWING,
)


test_params = [(0.3, 0.24, 0.6), (0.3, 0.18, 0.82), (0.3, 0.12, 0.83),
               (0.2, 0.24, 0.53), (0.2, 0.18, 0.56),
               (0.15, 0.24, 0.5), (0.15, 0.18, 0.7), (0.15, 0.12, 0.7)]

num_gaits = len(test_params)

num_targets = 5


def reset(robot, robot_uid, controller, init_pose, p):

  p.resetBasePositionAndOrientation(robot_uid,init_pose[0],init_pose[1])
  robot.ResetPose()
  robot.ResetTime()
  controller.reset()


  current_time = robot.GetTimeSinceReset()
  lin_speed_cmd = np.array([0.0,0.0,0.0,0.0]) 
  lin_speed = np.array([0.0,0.0,0.0,0.0]) 
  ang_speed = 0.0

  return current_time, lin_speed_cmd, lin_speed, ang_speed

def _setup_controller(robot, duty_factor, step_height, frequency):
  """Demonstrates how to create a locomotion controller."""
  desired_speed = (0, 0)
  desired_twisting_speed = 0
  desired_height = step_height

  gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
      robot,
      stance_duration=frequency,
      duty_factor=duty_factor,
      initial_leg_phase=_INIT_PHASE_FULL_CYCLE,
      initial_leg_state=_INIT_LEG_STATE)
  state_estimator = com_velocity_estimator.COMVelocityEstimator(robot,
                                                                window_size=20)
  sw_controller = raibert_swing_leg_controller.RaibertSwingLegController(
      robot,
      gait_generator,
      state_estimator,
      desired_speed=desired_speed,
      desired_twisting_speed=desired_twisting_speed,
      desired_height=desired_height, #robot_sim.MPC_BODY_HEIGHT,
      foot_clearance=0.01)

  st_controller = torque_stance_leg_controller.TorqueStanceLegController(
      robot,
      gait_generator,
      state_estimator,
      desired_speed=desired_speed,
      desired_twisting_speed=desired_twisting_speed,
      desired_body_height=robot_sim.MPC_BODY_HEIGHT,
      body_mass=robot_sim.MPC_BODY_MASS,
      body_inertia=robot_sim.MPC_BODY_INERTIA)

  controller = locomotion_controller.LocomotionController(
      robot=robot,
      gait_generator=gait_generator,
      state_estimator=state_estimator,
      swing_leg_controller=sw_controller,
      stance_leg_controller=st_controller,
      clock=robot.GetTimeSinceReset)
  return controller

def _update_controller_params(controller, lin_speed, ang_speed):
  controller.swing_leg_controller.desired_speed = lin_speed
  controller.swing_leg_controller.desired_twisting_speed = ang_speed
  controller.stance_leg_controller.desired_speed = lin_speed
  controller.stance_leg_controller.desired_twisting_speed = ang_speed

def ModNearest(a, b=360.0):

  if (a<0): 
    a+=360

  theta = a - b * np.round(a / b)
  if (theta>180):
      theta -= 360
  
  return math.radians(theta); 

def check_contact_fail(_robot, _p, ground_id):
    foot_links = _robot.GetFootLinkIDs()
    leg_links = _robot.GetLegLinkIDs()
    allowed_contact_links = foot_links + leg_links
    ground = ground_id

  # sometimes the robot can be initialized with some ground penetration
  # so do not check for contacts until after the first env step.
  
    robot_ground_contacts = _p.getContactPoints(
        bodyA=_robot.quadruped, bodyB=ground)

    for contact in robot_ground_contacts:
        if contact[3] not in allowed_contact_links:
            return True

    return False

def setup_env():

 
    p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
            
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
    
    p.setAdditionalSearchPath(pd.getDataPath())
    
    num_bullet_solver_iterations = 30

    p.setPhysicsEngineParameter(numSolverIterations=num_bullet_solver_iterations)
    
    
    p.setPhysicsEngineParameter(enableConeFriction=0)
    p.setPhysicsEngineParameter(numSolverIterations=30)
    simulation_time_step = 0.001

    p.setTimeStep(simulation_time_step)
    
    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(enableConeFriction=0)
    p.setAdditionalSearchPath(pd.getDataPath())

    
    ground_id = p.loadURDF("plane.urdf")    
    p.changeDynamics(ground_id, -1, lateralFriction=5.0)
  

    return p, ground_id

def generateNewTarget(currentTarget,rx=2,ry=2,minDistance=2):

    x0,y0,_ = currentTarget
    xn = np.random.uniform(-rx,rx)
    yn = np.random.uniform(-ry,ry)
  
    newtarget = np.array([xn,yn,0])

    r  = np.linalg.norm(np.array([x0,y0,0])- newtarget)
    while (r <= minDistance):      
        xn = np.random.uniform(-rx,rx)
        yn = np.random.uniform(-ry,ry)
        newtarget = np.array([xn,yn,0])
        r  = np.linalg.norm(np.array([x0,y0,0])- newtarget)
    
    return newtarget


def get_gravity( p, robot):  # 3
    # This may not be the Pelvis CoM position _exactly_ but should be fine, otherwise can apply local transformation
    pos, quat = p.getBasePositionAndOrientation(robot.quadruped)
    base_pos, base_quat = np.array(pos), np.array(quat)
    gravity = np.array([0, 0, -1])
    gravity_quat = p.getQuaternionFromEuler([0, 0, 0])
    invBasePos, invBaseQuat = p.invertTransform([0, 0, 0], base_quat)
    gravityPosInBase, gravityQuatInBase = p.multiplyTransforms(invBasePos, invBaseQuat, gravity, gravity_quat)
    gravityPosInBase = np.array(gravityPosInBase)
    
    return gravityPosInBase


def reward_function(p, robot, lin_speed_cmd, ang_speed_cmd):
    gravity_vec = get_gravity(p, robot)
    gravity_error = np.array([0, 0, -1]) - gravity_vec
    gravity_error = np.linalg.norm(gravity_error)  
    gravity_reward = np.exp(-gravity_error**2*2.5)

    # angular reward = rotational
    desired_ang_vel  = np.array([ang_speed_cmd])
    ang_vel = robot.GetBaseRollPitchYawRate()[2]
    ang_reward = np.exp(-np.linalg.norm(desired_ang_vel - ang_vel)**2*7.5)

    #linear reward = speed and direction
    desired_lin_com = np.array([lin_speed_cmd[0],lin_speed_cmd[1]])
    lin_vel = robot.GetBaseVelocity()[:2]
    linear_reward = np.exp(-np.linalg.norm(desired_lin_com - lin_vel)**2*18.5)

    #### TODO maybe add a distance error -- may also require increasing num time stesp
    #### TODO i think it would be good to add swing target error 

    return gravity_reward + ang_reward + linear_reward



def _run_example(step_height, frequency, duty):
  """Runs the locomotion controller example."""
  step_height = step_height
  num_targets=5
  frequency = frequency
  duty_factor = duty

  print('Starting study -----------------')
  print(frequency)
  print(duty_factor)
  

  p, ground_id = setup_env()
  
  robot_uid = p.loadURDF(robot_sim.URDF_NAME, robot_sim.START_POS)
  robot = robot_sim.SimpleRobot(p, robot_uid, simulation_time_step=0.001)

  init_pose = p.getBasePositionAndOrientation(robot_uid)
  controller = _setup_controller(robot, [duty_factor]*4, step_height, [frequency]*4 )
  controller.reset()
  
  p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

  current_time = robot.GetTimeSinceReset()
  targetPos = np.array([2.0,2.0,0])
  ball_urdf = __cwd__ + "/data/ball_vision.urdf"
  target_indicator = p.loadURDF(ball_urdf, basePosition=targetPos, baseOrientation=[0, 0, 0, 1],
                                                     useFixedBase=True, globalScaling=0.5)
  
  lin_speed_cmd = np.array([0.0,0.0,0.0,0.0]) 
  lin_speed = np.array([0.0,0.0,0.0,0.0]) 
  ang_speed = 0.0

  t0 = current_time
  lastDrawing = current_time

  freq  = 20

  total_reward = 0
  total_steps = 0

  for goals_reached in range(num_targets):
    num_steps = 0
    while True:

        num_steps += 1
        if num_steps > 2000:
            break

        p.submitProfileTiming("loop")
        robotPose = p.getBasePositionAndOrientation(robot_uid)

        robotOrn = p.getEulerFromQuaternion(robotPose[1])
        r = np.linalg.norm(robotPose[0][:2]-targetPos[:2])
        theta = np.arctan2(targetPos[1]-robotPose[0][1],targetPos[0]-robotPose[0][0]) - robotOrn[2] 
        theta = ModNearest(math.degrees(theta))
        
        command = [r*np.cos(theta),r*np.sin(theta),0]

        if (r<0.5):
            break


        if check_contact_fail(robot, p, ground_id):
            raise optuna.exceptions.TrialPruned()


        lin_speed[0] = command[0]
        lin_speed[1] = command[1]
        lin_speed[2] = 0.0
        lin_speed[3] = theta

        if (current_time - t0 > 1 / freq):
            lin_speed_cmd[0] += max( min( lin_speed[0] - lin_speed_cmd[0], 0.005) , -0.005)
            lin_speed_cmd[1] += max( min( lin_speed[1] - lin_speed_cmd[1], 0.005) , -0.005)
            lin_speed_cmd[2] += max( min( lin_speed[2] - lin_speed_cmd[2], 0.005) , -0.005)
            lin_speed_cmd[3] += max( min( lin_speed[3] - lin_speed_cmd[3], 0.005) , -0.005)
            t0 = current_time

            # limit overall maximum (only affects the command when the limit is actually exceeded)
            manhatDist = abs(lin_speed_cmd[0]) + abs(lin_speed_cmd[1]) + abs(lin_speed_cmd[3])
            if manhatDist > 0.5:
                factor = 1 #1 means maximum performance (<1 means safe parameters)
                norm   = manhatDist / (0.5*factor)

                lin_speed_cmd[0] /= norm
                lin_speed_cmd[1] /= norm

        lin_speed_cmd[0] = np.clip (lin_speed_cmd[0],-0.5,0.5)  
        lin_speed_cmd[1] = np.clip (lin_speed_cmd[1],-0.2,0.2)
        lin_speed_cmd[2] = np.clip (lin_speed_cmd[2],-0.2,0.2)
        ang_speed        = np.clip (lin_speed_cmd[3],-np.pi/4,np.pi/4)
        
        _update_controller_params(controller, lin_speed_cmd[:3], ang_speed)

        controller.update()
        hybrid_action, info = controller.get_action()
        robot.Step(hybrid_action)

        total_reward += reward_function(p, robot, lin_speed_cmd, ang_speed)

        current_time = robot.GetTimeSinceReset()
        p.submitProfileTiming()
        time.sleep(0.01)

    total_steps += num_steps
    targetPos=generateNewTarget(targetPos)
    p.resetBasePositionAndOrientation(target_indicator, targetPos, [0, 0, 0, 1])

  print('mean reward:', total_reward/total_steps)
  return total_reward/total_steps



if __name__ == '__main__':

    step_height = 0.20
    frequency = 0.2135728
    duty = 0.6406803

    _run_example(step_height, frequency, duty)



