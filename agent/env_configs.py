from collections import namedtuple
from unicodedata import name

TRAINING_ENV_CONFIG = {
    "enable_external_force":True,
    "force_range": (250,500),
    "force_frequency": (4,10),
    "target_distance":2.5,
    "terrain_type":None,
    "terrain_probability":"uniform",
    "terrain_uniform_range":(0, 0.045),
    "target_distance": 2.5,
    "perlin_params": [
                {'scale':180,'octaves':4, 'persistence':0.2},
                {'scale':200,'octaves':6, 'persistence':0.3},
            ]
}

"""
eval envs 
episode = 50
controller type = 'mpc', 'kernel', 'rl'

type

flat: 0.02, 0.04, 0.06,
perlin: (
    {'scale':150,'octaves':4, 'persistence':0.2},
    {'scale':170,'octaves':6, 'persistence':0.3}
)

forces: 350, 450, 550 -> per 4 seconds

total eval ens = 15
"""


# ------------- flat 0.02 

EVAL_ENV_CONFIG_1 = {
    "enable_external_force":True,
    "force_range": (350,350),
    "force_frequency": (4,4),
    "target_distance":2.5,
    "terrain_type":"uniform",
    "terrain_probability":"fixed",
    "terrain_uniform_range":(0.02, 0.02),
    "perlin_params": None
}

EVAL_ENV_CONFIG_2 = {
    "enable_external_force":True,
    "force_range": (450,450),
    "force_frequency": (4,4),
    "target_distance":2.5,
    "terrain_type":"uniform",
    "terrain_probability":"fixed",
    "terrain_uniform_range":(0.02, 0.02),
    "perlin_params": None
}

EVAL_ENV_CONFIG_3 = {
    "enable_external_force":True,
    "force_range": (550,550),
    "force_frequency": (4,4),
    "target_distance":2.5,
    "terrain_type":"uniform",
    "terrain_probability":"fixed",
    "terrain_uniform_range":(0.02, 0.02),
    "perlin_params": None
}



# ------------- flat 0.04

EVAL_ENV_CONFIG_4 = {
    "enable_external_force":True,
    "force_range": (350,350),
    "force_frequency": (4,4),
    "target_distance":2.5,
    "terrain_type":"uniform",
    "terrain_probability":"fixed",
    "terrain_uniform_range":(0.04, 0.04),
    "perlin_params": None
}

EVAL_ENV_CONFIG_5 = {
    "enable_external_force":True,
    "force_range": (450,450),
    "force_frequency": (4,4),
    "target_distance":2.5,
    "terrain_type":"uniform",
    "terrain_probability":"fixed",
    "terrain_uniform_range":(0.04, 0.04),
    "perlin_params": None
}

EVAL_ENV_CONFIG_6 = {
    "enable_external_force":True,
    "force_range": (550,550),
    "force_frequency": (4,4),
    "target_distance":2.5,
    "terrain_type":"uniform",
    "terrain_probability":"fixed",
    "terrain_uniform_range":(0.04, 0.04),
    "perlin_params": None
}


# ------------- flat 0.06 {'scale':150,'octaves':4, 'persistence':0.2},

EVAL_ENV_CONFIG_7 = {
    "enable_external_force":True,
    "force_range": (350,350),
    "force_frequency": (4,4),
    "target_distance":2.5,
    "terrain_type":"uniform",
    "terrain_probability":"fixed",
    "terrain_uniform_range":(0.06, 0.06),
    "perlin_params": None
}

EVAL_ENV_CONFIG_8 = {
    "enable_external_force":True,
    "force_range": (450,450),
    "force_frequency": (4,4),
    "target_distance":2.5,
    "terrain_type":"uniform",
    "terrain_probability":"fixed",
    "terrain_uniform_range":(0.06, 0.06),
    "perlin_params": None
}

EVAL_ENV_CONFIG_9 = {
    "enable_external_force":True,
    "force_range": (550,550),
    "force_frequency": (4,4),
    "target_distance":2.5,
    "terrain_type":"uniform",
    "terrain_probability":"fixed",
    "terrain_uniform_range":(0.06, 0.06),
    "perlin_params": None
}

# ------------- perlin 1

EVAL_ENV_CONFIG_10 = {
    "enable_external_force":True,
    "force_range": (350,350),
    "force_frequency": (4,4),
    "target_distance":2.5,
    "terrain_type":"perlin",
    "terrain_probability":"fixed",
    "perlin_params": [
                {'scale':150,'octaves':4, 'persistence':0.2}
            ]
}

EVAL_ENV_CONFIG_11 = {
    "enable_external_force":True,
    "force_range": (450,450),
    "force_frequency": (4,4),
    "target_distance":2.5,
    "terrain_type":"perlin",
    "terrain_probability":"fixed",
    "perlin_params": [
                {'scale':150,'octaves':4, 'persistence':0.2}
            ]
}

EVAL_ENV_CONFIG_12 = {
    "enable_external_force":True,
    "force_range": (550,550),
    "force_frequency": (4,4),
    "target_distance":2.5,
    "terrain_type":"perlin",
    "terrain_probability":"fixed",
    "terrain_uniform_range":(0, 0),
    "perlin_params": [
                {'scale':200,'octaves':4, 'persistence':0.2}
            ]
}


# ------------- perlin 2

EVAL_ENV_CONFIG_13 = {
    "enable_external_force":True,
    "force_range": (350,350),
    "force_frequency": (4,4),
    "target_distance":2.5,
    "terrain_type":"perlin",
    "terrain_probability":"fixed",
    "perlin_params": [
                {'scale':170,'octaves':6, 'persistence':0.3}
            ]
}

EVAL_ENV_CONFIG_14 = {
    "enable_external_force":True,
    "force_range": (450,450),
    "force_frequency": (4,4),
    "target_distance":2.5,
    "terrain_type":"perlin",
    "terrain_probability":"fixed",
    "perlin_params": [
                {'scale':170,'octaves':6, 'persistence':0.3}
            ]
}

EVAL_ENV_CONFIG_15 = {
    "enable_external_force":True,
    "force_range": (550,550),
    "force_frequency": (4,4),
    "target_distance":2.5,
    "terrain_type":"perlin",
    "terrain_probability":"fixed",
    "terrain_uniform_range":(0, 0),
    "perlin_params": [
                {'scale':170,'octaves':6, 'persistence':0.3}
            ]
}



ENV_CONFIGS = {
    'train': namedtuple("EnvConfig",TRAINING_ENV_CONFIG.keys())(**TRAINING_ENV_CONFIG),
    'eval_1':namedtuple("EnvConfig",EVAL_ENV_CONFIG_1.keys())(**EVAL_ENV_CONFIG_1),
    'eval_2':namedtuple("EnvConfig",EVAL_ENV_CONFIG_2.keys())(**EVAL_ENV_CONFIG_2),
    'eval_3':namedtuple("EnvConfig",EVAL_ENV_CONFIG_3.keys())(**EVAL_ENV_CONFIG_3),
    'eval_4':namedtuple("EnvConfig",EVAL_ENV_CONFIG_4.keys())(**EVAL_ENV_CONFIG_4),
    'eval_5':namedtuple("EnvConfig",EVAL_ENV_CONFIG_5.keys())(**EVAL_ENV_CONFIG_5),
    'eval_6':namedtuple("EnvConfig",EVAL_ENV_CONFIG_6.keys())(**EVAL_ENV_CONFIG_6),
    'eval_7':namedtuple("EnvConfig",EVAL_ENV_CONFIG_7.keys())(**EVAL_ENV_CONFIG_7),
    'eval_8':namedtuple("EnvConfig",EVAL_ENV_CONFIG_8.keys())(**EVAL_ENV_CONFIG_8),
    'eval_9':namedtuple("EnvConfig",EVAL_ENV_CONFIG_9.keys())(**EVAL_ENV_CONFIG_9),
    'eval_10':namedtuple("EnvConfig",EVAL_ENV_CONFIG_10.keys())(**EVAL_ENV_CONFIG_10),
    'eval_11':namedtuple("EnvConfig",EVAL_ENV_CONFIG_11.keys())(**EVAL_ENV_CONFIG_11),
    'eval_12':namedtuple("EnvConfig",EVAL_ENV_CONFIG_12.keys())(**EVAL_ENV_CONFIG_12),
    'eval_13':namedtuple("EnvConfig",EVAL_ENV_CONFIG_13.keys())(**EVAL_ENV_CONFIG_13),
    'eval_14':namedtuple("EnvConfig",EVAL_ENV_CONFIG_14.keys())(**EVAL_ENV_CONFIG_14),
    'eval_15':namedtuple("EnvConfig",EVAL_ENV_CONFIG_15.keys())(**EVAL_ENV_CONFIG_15),
}