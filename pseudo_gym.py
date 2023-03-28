from copy import deepcopy

import gym
import numpy as np
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv


class PseudoGym(HalfCheetahEnv):
    def set_internal_state(self, obs):
        qpos = np.concatenate(([obs[0]],obs[:8]))
        self.set_state(qpos, obs[8:])

    def get_obs(self):
        return self._get_obs()