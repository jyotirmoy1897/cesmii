from gym.envs.registration import register

register(
    id='ds-v1',
    entry_point='DSEnv.envs:DsEnv',
)
