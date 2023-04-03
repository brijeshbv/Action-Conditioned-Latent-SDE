import gym
import numpy as np

from stable_baselines3 import SAC

env = gym.make("Hopper-v2")

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("sac_hopper")

del model # remove to demonstrate saving and loading

model = SAC.load("sac_hopper")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()