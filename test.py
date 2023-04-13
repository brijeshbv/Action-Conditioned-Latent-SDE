import numpy as np
import gym

def func1():
    env = gym.make('Hopper-v2')
    obs = env.reset()
    return obs


print(type(func1()))
