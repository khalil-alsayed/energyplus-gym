import gymnasium as gym

def test_can_make_env():
    import eplus_gym   # this registers the env id
    env = gym.make("EPlusGym/Amphitheater-v0", env_config={"output": "runs/test"})
    assert env is not None
