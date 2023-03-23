import gym 
from stable_baselines3 import SAC
import os
import torch
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_env_samples(env, model_file, batch_size, steps, device, t0=0., t1=2.):
    env = gym.make(env)
    model = SAC.load(model_file, device=device)
    data_buffer = np.array([], dtype=np.float32)

    for i in range(batch_size):
        obs = env.reset()
        observations = np.array([obs], dtype=np.float32)
        for j in range(steps-1):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            observations = np.vstack((observations, obs))
        if i == 0:
            data_buffer = np.array([observations])
        else:
            data_buffer = np.append(data_buffer, [observations], axis=0)
    ts = torch.linspace(t0, t1, steps=steps, device=device)
    data_buffer = np.transpose(data_buffer, (1,0,2))
    print(data_buffer.shape)
    return torch.tensor(data_buffer, dtype=torch.float32), ts
        

if __name__ == "__main__":
    data_buffer = get_env_samples('Pendulum-v1','sac_pendulum', 10, 50,device )
    print(data_buffer.shape)
