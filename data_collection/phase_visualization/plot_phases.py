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
import torch
from torchsummary import summary
from kernel.models.denseNN_v2 import DenseNN

import time
import matplotlib.pyplot as plt


from mpl_toolkits import mplot3d



_STANCE_DURATION_SECONDS = [
    0.3
] * 4  # For faster trotting (v > 1.5 ms reduce this to 0.13s).


# Trotting
_DUTY_FACTOR = [0.6] * 4
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
        phases = list(map(lambda x: x[0] + 1 if (x[1] == gait_generator_lib.LegState.SWING) else x[0], phase_states))
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

    phase_hist = -np.sin(np.array(phase_hist)*np.pi)
    phase_hist2 = np.where(phase_hist<0, phase_hist, phase_hist*1.5)
        
    plt.plot(time, phase_hist[:,0], color='r')
    plt.plot(time, phase_hist2[:,0], color='g')

    # plt.show()

    return


def viz_sin_phase_preds():

    config = {
        "learning_rate":1e-3,
        "dropout_prob":0.2,
        "l2":0,
        "max_epoch":20,
        "data_dir": ["./data/sets/walking_cmd_v2_x.csv", "./data/sets/walking_cmd_v2_y.csv"],
        "batch_size":1000,
        "save_dir": "./kernel/results/walking_sin_phase",
        "log_file": "./kernel/results/walking_sin_phase/training_logs.out",
        "model_file": "../../kernel/results/walking_sin_phase/checkpoint_best.pt",
        "device": 'cpu',
        "mode":"training",
        "seed":0,
        "decay_rate":0,
        'depth':3,
        'width':256
    }
    model = DenseNN(11, 12, **config)
    model.eval()
    model.load_model()

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
    for i in range(1,1000):
        gait_generator.update(i*0.001)
        norm_phase = gait_generator.normalized_phase
        states = gait_generator.desired_leg_state
        phase_states = list(zip(norm_phase, states))
        phases = list(map(lambda x: x[0] + 1 if (x[1] == gait_generator_lib.LegState.SWING) else x[0], phase_states))
        phase_hist.append(phases)
        time.append(i*0.001)

    phase_hist = -np.sin(np.array(phase_hist)*np.pi)
    phase_hist2 = np.where(phase_hist<0, phase_hist, phase_hist*1.5)

    plt.plot(time, phase_hist2, 'r')
    plt.plot(time, phase_hist, 'g')

    lookback = 1
    stacked = np.array([])
    for i in range(len(phase_hist)):
  
        z = phase_hist[(i-lookback) if (i-lookback) > 0 else 0 : i+1 ]
        if len(z) < lookback + 1:
            z = np.repeat(z, 2, axis=0)
            
        p1 = z[:,0]
        p2 = z[:,1]
        p3 = z[:,2]
        p4 = z[:,3]
        
        final = np.hstack((p1,p2,p3,p4))
        if len(stacked) == 0:
            stacked = final
        else:
            stacked = np.vstack((stacked,final))

    lookback = 1
    stacked2 = np.array([])
    for i in range(len(phase_hist2)):
  
        z = phase_hist2[(i-lookback) if (i-lookback) > 0 else 0 : i+1 ]
        if len(z) < lookback + 1:
            z = np.repeat(z, 2, axis=0)
            
        p1 = z[:,0]
        p2 = z[:,1]
        p3 = z[:,2]
        p4 = z[:,3]
        
        final = np.hstack((p1,p2,p3,p4))
        if len(stacked2) == 0:
            stacked2 = final
        else:
            stacked2 = np.vstack((stacked2,final))

    
    commands = np.zeros((len(stacked),3))
    commands2 = np.zeros((len(stacked2),3))

    full_input = np.hstack((commands, stacked))
    full_input2 = np.hstack((commands2, stacked2))

    print(full_input.shape)

    predictions = model.forward(torch.Tensor(full_input))
    predictions2 = model.forward(torch.Tensor(full_input2))

    print(predictions.size())

    leg1 = predictions[:,:3].detach().numpy()
    leg12 = predictions2[:,:3].detach().numpy()

    


    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(leg1[:,0], leg1[:,1], leg1[:,2], 'gray')
    ax.plot3D(leg12[:,0], leg12[:,1], leg12[:,2], 'b')

    plt.show()

    return

viz_sin_phase_preds()
# plot_sin_phase()
# plot_phase()
# plt.show()