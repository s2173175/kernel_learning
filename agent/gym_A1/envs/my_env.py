import pybullet 
import pybullet_data as pd
from pybullet_utils import bullet_client
import torch
import gym
import random
import os

from mpc_controller import a1_sim as robot_sim
from mpc_controller import openloop_gait_generator
from kernel.models.denseNN import DenseNN

import inspect
from urllib import robotparser

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)


__cwd__ = os.path.realpath( os.path.join(os.getcwd(), os.path.dirname(__file__)))

import gc
gc.enable()

class MyEnv(gym.Env):


    def __init__(self, config):

        self.rendering = True

        #### internal vars --- requires def
        self.sim_time_step = 0.001
        self.sim_time_limit = 100
        self.terrain_probability = 0
        self.terrain_difficulty = 0.035
        self.terrain_type = 'uniform'
        self.terrain_names = ["flat", "uniform"] 

        ##### internal vars
        self.terrain_id = None


        ##### creating simulation env
        self._p = pybullet

        if self.rendering:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)

        self.setupPhysicsParmeters()
        self.loadTerrain()
        self.loadRobot()








        ##### loading the kernel ----------
        self.gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
            self._robot,
            stance_duration=[0.3]*4,
            duty_factor=[0.6] * 4,
            initial_leg_phase=[0.9, 0, 0, 0.9],   
            initial_leg_state=(gait_generator_lib.LegState.SWING,gait_generator_lib.LegState.STANCE,gait_generator_lib.LegState.STANCE,gait_generator_lib.LegState.SWING,)
            )
        self.gait_generator.reset(0)

        config = {
            "learning_rate":1e-3,
            "dropout_prob":0.2,
            "l2":0,
            "max_epoch":20,
            "data_dir": ["./data/sets/dense_standing_walk3_x.csv", "./data/sets/dense_standing_walk3_y.csv"],
            "batch_size":100,
            "save_dir": "./kernel/results/q_q_dot",
            "log_file": "./kernel/results/q_q_dot/training_logs.out",
            "model_file": "walking_cmd/checkpoint_best.pt",
            "device": "cpu"
        }
        self.kernel = DenseNN(7, 12, (256,256,256), **config)
        self.kernel.load_model()
        self.kernel.eval()
        
        input()

        return


    def setupPhysicsParmeters(self):
        self._p.setTimeStep(self.sim_time_step)
        self._p.setGravity(0, 0, -9.8)

        num_bullet_solver_iterations = 30
        self._p.setPhysicsEngineParameter(numSolverIterations=num_bullet_solver_iterations)
        self._p.setPhysicsEngineParameter(enableConeFriction=0)
        self._p.setPhysicsEngineParameter(numSolverIterations=30)

    def loadRobot(self):
        print(robot_sim.URDF_NAME, robot_sim.START_POS)
        quadruped = self._p.loadURDF(robot_sim.URDF_NAME, robot_sim.START_POS)
        self._robot = robot_sim.SimpleRobot(self._p, quadruped, simulation_time_step=self.sim_time_step)
        self.base_pos_nom = np.r_[0,0,0.3]
        self.base_orn_nom = np.r_[0,0,0,1]

    def loadTerrain(self):
        if self.terrain_type == "flat":  # "flat", "perlin", "uniform", "stair", "gap"
            self.terrain_id = self._p.loadURDF("plane.urdf")
        else:
            self.terrain_id = self.create_uneven_terrain(self._p, heightPerturbationRange=self.terrain_difficulty)

    def create_uneven_terrain(self, p, heightPerturbationRange=0.02):
        numHeightfieldRows = 256
        numHeightfieldColumns = 256
        meshScale = [.06, .06, 1.6]
        nominalPos = [0, 0, -heightPerturbationRange]
        p.configureDebugVisualizer( p.COV_ENABLE_RENDERING, 1) #it will not be reenabled in this function
            
        if self.terrain_type == "uniform":
            heightfieldData = [0] *  numHeightfieldRows *  numHeightfieldColumns
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
            scale = 300.0
            octaves = 6
            persistence = 0.2
            lacunarity = 5.0
            for i in range (int(numHeightfieldColumns)):
                for j in range (int(numHeightfieldRows) ):
                    heightfieldData[i,j] = noise.pnoise2(i/scale, 
                                            j/scale, 
                                            octaves=octaves, 
                                            persistence=persistence, 
                                            lacunarity=lacunarity, 
                                            repeatx=1024, 
                                            repeaty=1024, 
                                            base=42)
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

    def step(self):

        return


    
    def reset(self):

        return


    def get_observation(self):

        return








if __name__ == '__main__':

   

    x = MyEnv({})
