from gymnasium.envs.registration import register

register(
    id="EPlusGym/Amphitheater-v0",
    entry_point="eplus_gym.envs.env:AmphitheaterEnv",
)
