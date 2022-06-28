
from __future__ import absolute_import
from __future__ import division
#from __future__ import google_type_annotations
from __future__ import print_function
from copy import deepcopy

import os
import inspect
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

from mpc_controller import com_velocity_estimator
from mpc_controller import gait_generator as gait_generator_lib
from mpc_controller import locomotion_controller
from mpc_controller import openloop_gait_generator
from mpc_controller import raibert_swing_leg_controller
from mpc_controller import torque_stance_leg_controller

#uncomment the robot of choice
#from mpc_controller import laikago_sim as robot_sim
from mpc_controller import a1_sim as robot_sim

FLAGS = flags.FLAGS


_NUM_SIMULATION_ITERATION_STEPS = 300


######################### Trotting

# _STANCE_DURATION_SECONDS = [
#     0.3
# ] * 4  # For faster trotting (v > 1.5 ms reduce this to 0.13s).


# # Trotting
# _DUTY_FACTOR = [0.8] * 4
# _INIT_PHASE_FULL_CYCLE = [0, 0.5, 0.75, 0.25]
# _MAX_TIME_SECONDS = 50

# _STEP_HEIGHT = 0.18

# _INIT_LEG_STATE = (
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.SWING,
#     gait_generator_lib.LegState.STANCE,
# )

_STANCE_DURATION_SECONDS = [
    0.3
] * 4  # For faster trotting (v > 1.5 ms reduce this to 0.13s).


# Trotting
_DUTY_FACTOR = [0.8] * 4
_INIT_PHASE_FULL_CYCLE = [0., 0.3, 0.6, 0.]
_MAX_TIME_SECONDS = 50

_STEP_HEIGHT = 0.24

_INIT_LEG_STATE = (
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.SWING,
)



def _generate_example_linear_angular_speed(t, lin, ang):
  """Creates an example speed profile based on time for demo purpose."""
  vx = 0.6 * robot_sim.MPC_VELOCITY_MULTIPLIER
  vy = 0.2 * robot_sim.MPC_VELOCITY_MULTIPLIER
  wz = 0.8 * robot_sim.MPC_VELOCITY_MULTIPLIER
  
  time_points = (0, 5, 10, 15, 20, 25,30)
  speed_points = ((0, 0, 0, 0), (0, 0, 0, wz), (vx, 0, 0, 0), (0, 0, 0, -wz), (0, -vy, 0, 0),
                  (0, 0, 0, 0), (0, 0, 0, wz))

  speed = scipy.interpolate.interp1d(
      time_points,
      speed_points,
      kind="previous",
      fill_value="extrapolate",
      axis=0)(
          t)

  step = 0.005
  target_lin = np.array([0.1,0,0])
  norm = np.linalg.norm(target_lin)

  if norm > 0.5:
    target_lin = target_lin * 0.5/norm
  target_ang = [0.2]

  anf_new = deepcopy(ang)
  lin_new = list(deepcopy(lin))

  if ang[0] < target_ang[0]:
      anf_new = np.array([ang[0] + step])

  if lin[0] < target_lin[0]:
      
      lin_new[0] = lin[0]+step

  if lin[1] < target_lin[1]:
      lin_new[1] = lin[1]+step

  print(lin_new, anf_new)

  return np.array(lin_new), anf_new#speed[0:3], speed[3]


def _setup_controller(robot):
  """Demonstrates how to create a locomotion controller."""
  desired_speed = (0, 0)
  desired_twisting_speed = 0
  desired_height = _STEP_HEIGHT

  gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
      robot,
      stance_duration=_STANCE_DURATION_SECONDS,
      duty_factor=_DUTY_FACTOR,
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
      desired_height=desired_height,#robot_sim.MPC_BODY_HEIGHT,
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


def _run_example(max_time=_MAX_TIME_SECONDS):
  """Runs the locomotion controller example."""
  
  #recording video requires ffmpeg in the path
  record_video = False
  if record_video:
    p = pybullet
    p.connect(p.GUI, options="--width=1280 --height=720 --mp4=\"test.mp4\" --mp4fps=100")
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
  else:
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
  
  #random.seed(10)
  #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
  heightPerturbationRange = 0.06
  
  plane = True
  if plane:
    new_terrain = p.loadURDF("plane.urdf")
    # planeShape = p.createCollisionShape(shapeType = p.GEOM_PLANE)
    # ground_id  = p.createMultiBody(0, planeShape)
  else:
    numHeightfieldRows = 256
    numHeightfieldColumns = 256
    heightfieldData = [0]*numHeightfieldRows*numHeightfieldColumns 
    for j in range (int(numHeightfieldColumns/2)):
      for i in range (int(numHeightfieldRows/2) ):
        height = random.uniform(0,heightPerturbationRange)
        heightfieldData[2*i+2*j*numHeightfieldRows]=height
        heightfieldData[2*i+1+2*j*numHeightfieldRows]=height
        heightfieldData[2*i+(2*j+1)*numHeightfieldRows]=height
        heightfieldData[2*i+1+(2*j+1)*numHeightfieldRows]=height
    
    terrainShape = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, meshScale=[.05,.05,1], heightfieldTextureScaling=(numHeightfieldRows-1)/2, heightfieldData=heightfieldData, numHeightfieldRows=numHeightfieldRows, numHeightfieldColumns=numHeightfieldColumns)
    ground_id  = p.createMultiBody(0, terrainShape)

  p.changeDynamics(new_terrain, -1, lateralFriction=5.0)
  #p.resetBasePositionAndOrientation(ground_id,[0,0,0], [0,0,0,1])
  
  #p.changeDynamics(ground_id, -1, lateralFriction=1.0)
  
  robot_uid = p.loadURDF(robot_sim.URDF_NAME, robot_sim.START_POS)

  robot = robot_sim.SimpleRobot(p, robot_uid, simulation_time_step=simulation_time_step)
  
  controller = _setup_controller(robot)
  controller.reset()
  
  p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
  # p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
  #while p.isConnected():
  #  pos,orn = p.getBasePositionAndOrientation(robot_uid)
  #  print("pos=",pos)
  #  p.stepSimulation()
  #  time.sleep(1./240)  
  current_time = robot.GetTimeSinceReset()
  #logId = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "mpc.json")
  lin_speed, ang_speed = np.array([0,0,0]), np.array([0])
  while current_time < max_time:
    #pos,orn = p.getBasePositionAndOrientation(robot_uid)
    #print("pos=",pos, " orn=",orn)
    # p.submitProfileTiming("loop")
    print('here')
 
    # Updates the controller behavior parameters.
    lin_speed, ang_speed = _generate_example_linear_angular_speed(current_time, lin_speed, ang_speed)
    #lin_speed, ang_speed = (0., 0., 0.), 0.
    _update_controller_params(controller, lin_speed, ang_speed)

    # Needed before every call to get_action().
    controller.update()
    hybrid_action, info = controller.get_action()
    
    robot.Step(hybrid_action)
    
    if record_video:
      p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)

    #time.sleep(0.003)
    current_time = robot.GetTimeSinceReset()
    # p.submitProfileTiming()
  #p.stopStateLogging(logId)
  #while p.isConnected():
    time.sleep(0.01)

def main(argv):
  del argv
  _run_example()


if __name__ == "__main__":
  app.run(main)
