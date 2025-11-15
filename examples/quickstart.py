import gymnasium as gym
import eplus_gym

env = gym.make("EPlusGym/Amphitheater-v0", env_config={"output": "runs/demo"})
print("Env created:", env)
