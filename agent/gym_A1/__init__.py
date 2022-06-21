from gym.envs.registration import register


register(
    id='A1_all_terrains-v2',
    entry_point='gym_A1.envs:A1_env_v2_3',
)