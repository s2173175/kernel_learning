from collections import namedtuple
from unicodedata import name

TRAINING_ENV_CONFIG = {
    "enable_external_force":True,
    "force_range": (250,500),
    "force_frequency": (4,10),
    "target_distance":2.5,
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

EVAL_ENV_CONFIG_1 = {}



ENV_CONFIGS = {
    'train': namedtuple("EnvConfig",TRAINING_ENV_CONFIG.keys())(**TRAINING_ENV_CONFIG)
}