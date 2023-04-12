from copy import deepcopy

import gym
import numpy as np
from gym.envs.mujoco.hopper import HopperEnv


class PseudoGym(HopperEnv):
    def set_internal_state(self, obs):
        qpos = np.concatenate(([obs[0]],obs[:5]))
        self.set_state(qpos, obs[5:])

    def get_obs(self):
        return self._get_obs()