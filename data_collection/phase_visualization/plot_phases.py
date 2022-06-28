from __future__ import absolute_import
from __future__ import division
#from __future__ import google_type_annotations
from __future__ import print_function
from distutils import command

import os
import inspect
from turtle import color
from urllib import robotparser
from numpy import dtype

from requests import get
from torch import save

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import pybullet_data as pd
from pybullet_utils import bullet_client
import pybullet

from absl import app
from absl import flags

from mpc_controller import gait_generator as gait_generator_lib
from mpc_controller import openloop_gait_generator
from mpc_controller import a1_sim as robot_sim

import numpy as np


import time
import matplotlib.pyplot as plt

_STANCE_DURATION_SECONDS = [
    0.6
] * 4  # For faster trotting (v > 1.5 ms reduce this to 0.13s).


# Trotting
_DUTY_FACTOR = [0.5] * 4
_INIT_PHASE_FULL_CYCLE = [0.9, 0.0, 0.0, 0.9]
_MAX_TIME_SECONDS = 50

_STEP_HEIGHT = 0.24

_INIT_LEG_STATE = (
    gait_generator_lib.LegState.SWING,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.SWING,
)




def plot_phase():

    p = pybullet
    p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
    p.setAdditionalSearchPath(pd.getDataPath())

    robot_uid = p.loadURDF(robot_sim.URDF_NAME, robot_sim.START_POS)

    robot = robot_sim.SimpleRobot(p, robot_uid, simulation_time_step=0.001)

    gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
      robot,
      stance_duration=_STANCE_DURATION_SECONDS,
      duty_factor=_DUTY_FACTOR,
      initial_leg_phase=_INIT_PHASE_FULL_CYCLE,
      initial_leg_state=_INIT_LEG_STATE)

    phase_hist = []
    time = []
    for i in range(1,300):
        print(i)
        gait_generator.update(i*0.01)
        norm_phase = gait_generator.normalized_phase
        states = gait_generator.desired_leg_state
        phase_states = list(zip(norm_phase, states))
        phases = list(map(lambda x: x[0] + 1 if (x[1] == gait_generator_lib.LegState.STANCE) else x[0], phase_states))
        phase_hist.append(phases)
        time.append(i*0.09)

    phase_hist = np.array(phase_hist)-1
        
    plt.plot(time, phase_hist[:,0], color='b')

    # plt.show()

    return

def plot_sin_phase():

    p = pybullet
    p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
    p.setAdditionalSearchPath(pd.getDataPath())

    robot_uid = p.loadURDF(robot_sim.URDF_NAME, robot_sim.START_POS)

    robot = robot_sim.SimpleRobot(p, robot_uid, simulation_time_step=0.001)

    gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
      robot,
      stance_duration=_STANCE_DURATION_SECONDS,
      duty_factor=_DUTY_FACTOR,
      initial_leg_phase=_INIT_PHASE_FULL_CYCLE,
      initial_leg_state=_INIT_LEG_STATE)

    phase_hist = []
    time = []
    for i in range(1,300):
        print(i)
        gait_generator.update(i*0.01)
        norm_phase = gait_generator.normalized_phase
        states = gait_generator.desired_leg_state
        phase_states = list(zip(norm_phase, states))
        phases = list(map(lambda x: x[0] + 1 if (x[1] == gait_generator_lib.LegState.SWING) else x[0], phase_states))
        phase_hist.append(phases)
        time.append(i*0.09)

    phase_hist = np.sin(np.array(phase_hist)*np.pi)
        
    plt.plot(time, phase_hist[:,0], color='r')

    # plt.show()

    return

plot_sin_phase()
plot_phase()
plt.show()